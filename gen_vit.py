# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from functools import partial
from itertools import repeat
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from pygcn.models import GCN, GIN
from pygcn.models_gru import GCN_gru
import numpy as np
# import cupy as cp
from models import *
from torch.autograd import Variable
import torch.nn.init as init
from torch import Tensor
from typing import Tuple
from utils import *
EOS = 1e-15


dtype = torch.cuda.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from torch._six import container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        att, adj = self.fn(x, *args, **kwargs)
        out = att + x, adj
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        out = self.fn(temp, *args, **kwargs)
        return out

# class PreForward(nn.Module):
#     def __init__(self, dim, hidden_dim, kernel_size, num_channels, dropout=0.):
#         super().__init__()
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim, dim),
#         #     nn.Dropout(dropout)
#         # )
#         self.tcn = TemporalConvNet(dim, num_channels, hidden_dim, kernel_size, dropout)
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         # )
#
#     def forward(self, x):
#         r = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
#         # r = self.net(r)
#         return r


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout=0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     # nn.Linear(hidden_dim, dim),
        #     # nn.Dropout(dropout)
        # )
        self.net = nn.Identity()
    def forward(self, x):
        return self.net(x)


# def inverse_gumbel_cdf(y, mu, beta):
#     return mu - beta * torch.log(-torch.log(y))
class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, topk=4, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.diff_routing = diff_routing
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, adj: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """

        attn_logit = adj  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index

class KGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, k:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_k = k.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_k = torch.gather(k.view(n, 1, p2, w2, c_k).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_k) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_k = r_weight.view(n, p2, topk, 1, 1) * topk_k # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_k

class Attention_3GIN2(nn.Module):
    def __init__(self,
                 depth,
                 i,
                 dim,
                 image_size,
                 patch_size,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 downsample=0.,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False
                 ):
        super().__init__()
        self.scale = dim ** -0.5
        # self.drop_ratio = 0.1

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.with_cls_token = with_cls_token

        self.length = int(image_size / patch_size) ** 2
        self.length2 = int(image_size / patch_size * downsample) ** 2

        self.layer_index = i
        self.depth = depth

        dim_in = dim
        dim_out = dim

        self.conv_proj_qkv = GIN(nfeat=dim_in,
                               head_num=heads,
                               nhid=3*dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        # self.proj_q.weight.requires_grad = False
        # self.proj_k.weight.requires_grad = False
        # self.proj_v.weight.requires_grad = False

        self.attn_drop = nn.Dropout(attn_drop)  #
        self.proj_drop = nn.Dropout(proj_drop)  #
        self.leakyrelu = nn.LeakyReLU()


        # sparse_D = torch.ones((self.num_heads, self.length), requires_grad=True).to(device)
        # self.sparse_D = torch.nn.Parameter(sparse_D)
        # self.register_parameter("sparse_D", self.sparse_D)
        #
        # sparse_D2 = torch.ones((self.num_heads, self.length2), requires_grad=True).to(device)
        # self.sparse_D2 = torch.nn.Parameter(sparse_D2)
        # self.register_parameter("sparse_D2", self.sparse_D2)
        #
        randomatrix = torch.randn((int(self.num_heads),
                                   int(self.num_heads)), requires_grad=True).to(device)
        self.randomatrix = torch.nn.Parameter(randomatrix)
        self.register_parameter("Ablah", self.randomatrix)

        # self.diff_routing = False
        # self.router = TopkRouting(topk=self.topk, diff_routing=self.diff_routing)
        # mul_weight = 'none'
        # self.k_gather = KGather(mul_weight=mul_weight)
        self.attn_act = nn.Softmax(dim=-1)

    def forward_conv_qkv(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)
        if self.conv_proj_qkv is not None:
            qkv, adj_copy = self.conv_proj_qkv(x, rep_adj)
            # q = F.dropout(F.relu(q), self.drop_ratio, training=self.training)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')
        q, k ,v = torch.chunk(qkv, chunks=3, dim=-1)

        return q, k, v, adj_copy


    def forward(self, x, adj, rep_adj_dis):
        x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
        current_length = adj.size(-1)
        b, head, L, _ = x.size()


        q, k, v, _ = self.forward_conv_qkv(x, adj)
        q_pix = q #rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)
        k_pix = rearrange(k, 'n h t c -> n h c t')
        v_pix = v #rearrange(v, 'n h t c -> (n t) h c').unsqueeze(2)

        attn_weight = (q_pix * self.scale) @ k_pix  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)

        attn_weight = self.attn_act(attn_weight)
        v = attn_weight @ v_pix

        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=head)))

        out = v

        return out, attn_weight

class Attention_global(nn.Module):
    def __init__(self,
                 depth,
                 i,
                 dim,
                 image_size,
                 patch_size,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 downsample=0.,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False
                 ):
        super().__init__()
        self.scale = dim ** -0.5
        # self.drop_ratio = 0.1

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.with_cls_token = with_cls_token

        self.length = int(image_size / patch_size) ** 2
        self.length2 = int(image_size / patch_size * downsample) ** 2

        self.layer_index = i
        self.depth = depth

        dim_in = dim
        dim_out = dim

        self.conv_proj_qk = GIN(nfeat=dim_in,
                               head_num=heads,
                               nhid=2*dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)

        self.conv_proj_v = GIN(nfeat=dim_in,  # GCN_gru
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        # self.proj_q.weight.requires_grad = False
        # self.proj_k.weight.requires_grad = False
        # self.proj_v.weight.requires_grad = False

        self.attn_drop = nn.Dropout(attn_drop)  #
        self.proj_drop = nn.Dropout(proj_drop)  #
        self.leakyrelu = nn.ReLU()


        # sparse_D = torch.ones((self.num_heads, self.length), requires_grad=True).to(device)
        # self.sparse_D = torch.nn.Parameter(sparse_D)
        # self.register_parameter("sparse_D", self.sparse_D)
        #
        # sparse_D2 = torch.ones((self.num_heads, self.length2), requires_grad=True).to(device)
        # self.sparse_D2 = torch.nn.Parameter(sparse_D2)
        # self.register_parameter("sparse_D2", self.sparse_D2)

        # randomatrix = torch.randn((int(self.num_heads),
        #                            int(self.num_heads)), requires_grad=True).to(device)
        # self.randomatrix = torch.nn.Parameter(randomatrix)
        # self.register_parameter("Ablah", self.randomatrix)

    def forward_conv_qk(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)
        if self.conv_proj_qk is not None:
            qk, _ = self.conv_proj_qk(x, rep_adj)
            # q = F.dropout(F.relu(q), self.drop_ratio, training=self.training)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')
        q, k = torch.chunk(qk, chunks=2, dim=-1)
        return q, k

    def forward_conv_v(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)

        if self.conv_proj_v is not None:
            v, _ = self.conv_proj_v(x, rep_adj)
            # v = F.dropout(v, self.drop_ratio, training=self.training)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return v

    def forward(self, x, adj, rep_adj_dis):
        x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
        current_length = adj.size(-1)
        b, head, L, _ = x.size()

        q, k = self.forward_conv_qk(x, adj)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        # attn_score = self.leakyrelu(attn_score)  # F.gelu

        ## ---1-----
        # attn_score = torch.matmul(torch.matmul(Random_RM, attn_score), Random_RM)
        ## ---2-----
        # Lambda = self.randomatrix
        # Lambda = Lambda.expand(b, -1, -1).to(device)
        # attn_score = rearrange(attn_score, 'b h l t -> b h (l t)')
        # attn_score = torch.einsum('blh,bhk->blk', [Lambda, attn_score])
        # attn_score = rearrange(attn_score, 'b h (l k) -> b h l k', l=current_length)

        norm_attn_score = torch.softmax(attn_score/2, dim=-1)

        # attn_score = (attn_score + attn_score.transpose(3, 2)) / 2
        # D = torch.diag_embed(torch.sum(attn_score, dim=-1) ** (-1 / 2))
        # D[D == float('inf')] = 0
        # norm_attn_score = torch.matmul(torch.matmul(D, attn_score), D)

        v = self.forward_conv_v(x, norm_attn_score)

        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=head)))

        out = self.proj_v(v)

        return out, norm_attn_score


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 image_size,
                 patch_size,
                 kernel_size,
                 batch_size,
                 in_chans,
                 embed_dim,
                 stride,
                 padding,
                 norm_layer=None):
        super().__init__()
        # kernel_size = to_2tuple(kernel_size)
        # self.patch_size = patch_size

        # self.proj = ResNet18(embed_dim).to(device)

        # self.proj = nn.Conv2d(
        #     in_chans, embed_dim,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding
        # )

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_chans, int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # groups=in_chans
            )),
            #### ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size), int(image_size / patch_size)))),
            ('bn', nn.BatchNorm2d(int(embed_dim))),
            ('relu', nn.GELU()),
            #('conv2', nn.Conv2d(
            #    int(embed_dim), int(embed_dim),
            #    kernel_size=kernel_size,
              #  stride=stride,
              #  padding=padding,
               # groups=int(embed_dim)
            #)),
            # ('pooling', nn.AdaptiveMaxPool2d((int(patch_size / 7), int(patch_size / 7)))),
            ('pooling', nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding,)),
            # ('bn', nn.BatchNorm2d(int(embed_dim))),
            # ('relu', nn.GELU()),
        ]))

        # self.proj = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(
        #         in_chans, int(embed_dim),
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         # groups=in_chans
        #     )),
        #     ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size / 4), int(image_size / patch_size / 4)))),
        #     ('bn', nn.BatchNorm2d(int(embed_dim))),
        #     ('relu', nn.GELU()),
        # ]))
        # self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        sp_features = self.proj(x).to(device)  # proj_conv  proj

        return sp_features


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample, batch_size, in_chans,
                 patch_stride, patch_padding, norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = ConvEmbed(
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            batch_size=batch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=dim//4,
            norm_layer=norm_layer
        )
        self.patch_dim = ((patch_size // 4) ** 2) * int(dim) // 4
        self.dim = dim
        #
        # channels = 3
        # self.patch_dim = channels * patch_size ** 2

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim).to(device)

        self.layers = nn.ModuleList([])
        self.depth = depth

        for i in range(self.depth):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, Attention(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                PreNorm(dim, Residual(Attention_global(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout),
            ]))

        self.dropout = nn.Dropout(dropout)

        # self.norm = nn.ModuleList([])
        # for _ in range(depth):
        #     self.norm.append(nn.LayerNorm(dim))

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.head_num = heads

        self.Upool_out = nn.Sequential(
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, img, adj, label):
        p = self.patch_size
        b, n, imgh, imgw = img.shape
        adj = adj.expand(b, self.head_num, -1, -1)

        # if label is True:
        #     adj_clone = adj.clone()
        #     pcent = float(0.1 + torch.rand(1) * 0.1)
        #     adj = self.replace_nonzero_with_zero(adj_clone, pcent)
        #     # pcent = float(0.2 + torch.rand(1) * 0.3)
        #     # adj_clone = adj.clone()
        #     # random_matrix = torch.rand(*adj.shape)
        #     # adj_clone[random_matrix < pcent] = 0
        #     # adj = adj_clone

        x = rearrange(img, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p, p2=p)
        conv_img = self.patch_embed(x)
        conv_img = rearrange(conv_img, '(b s) c p1 p2 -> b s (c p1 p2)', b=b)
        x = self.patch_to_embedding(conv_img)
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # x = self.patch_to_embedding(x)


        x = rearrange(x, 'b (h w) c -> b c h w', h=int(imgh / p), w=int(imgw / p))
        x = rearrange(x, 'b c h w -> b (h w) c')

        # adj = adj.unsqueeze(1)
        # rep_adj = adj.expand(-1, self.head_num, -1, -1).to(device)
        rep_adj = adj.to(device)
        rep_adj_dis = rep_adj

        for attn, ff in self.layers:
            # x = attn(x, self.rep_adj, 0)
            x, norm_attn_score = attn(x, rep_adj, rep_adj_dis)
            x = ff(x)
        # x_out = F.normalize(x, dim=-1)
        return norm_attn_score

    def replace_nonzero_with_zero(self, tensor, replace_ratio=0.2):
        # 步骤1: 找到所有非零元素的索引
        batch_size=tensor.size(0)
        tensor = rearrange(tensor, 'b h w v -> (b h) w v')
        non_zero_count = torch.count_nonzero(tensor)

        # 确保tensor是三维的
        assert tensor.dim() == 3, "Tensor must be 3-dimensional"

        # 找到所有非零元素的索引
        nonzero_indices = torch.nonzero(tensor, as_tuple=False)

        # 计算非零元素的数量
        num_nonzero = nonzero_indices.size(0)

        # 如果非零元素少于或等于需要替换的数量，则直接返回原tensor
        if num_nonzero <= 0 or num_nonzero * replace_ratio <= 1:
            return tensor

            # 计算需要替换的索引数量
        num_to_replace = int(num_nonzero * replace_ratio)

        # 随机选择需要替换的索引
        # 注意：这里我们随机选择的是索引的索引，即随机打乱所有非零索引的排列
        random_indices = torch.randperm(num_nonzero)[:num_to_replace]


        # 由于上述for循环的局限性，这里我们采用另一种方法：直接通过索引赋值
        tensor[tuple(nonzero_indices[random_indices].T)] = 0


        tensor = rearrange(tensor, ' (b h) w v ->b h w v', b=batch_size)
        non_zero_count = torch.count_nonzero(tensor)
        return tensor


class gen_vit(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, downsample, batch_size, num_classes, dim, depth, heads, mlp_dim, patch_stride, patch_pading, in_chans, dropout=0., emb_dropout = 0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        pantchesalow = image_size // patch_size
        num_patches = pantchesalow ** 2

        adj_matrix = [[0 for i in range(num_patches)] for i in range(num_patches)]
        adj_matrix = torch.as_tensor(adj_matrix).float().to(device)

        for j in range(num_patches):
            if (j - pantchesalow - 1) >= 0:
                adj_matrix[j][j - 1] = 1
                adj_matrix[j][j - pantchesalow] = 1
                adj_matrix[j][j - pantchesalow - 1] = 1
                adj_matrix[j][j - pantchesalow + 1] = 1
            if (j + pantchesalow + 1) < num_patches:
                adj_matrix[j][j + 1] = 1
                adj_matrix[j][j + pantchesalow] = 1
                adj_matrix[j][j + pantchesalow - 1] = 1
                adj_matrix[j][j + pantchesalow + 1] = 1
        # random_matrix = torch.rand(*adj_matrix.shape)
        # adj_matrix[random_matrix < 0.8] = 1   # leaner view has no random connections...

        self.adj_matrix = adj_matrix

        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample,
                                       batch_size, in_chans, patch_stride=patch_stride, patch_padding=patch_pading)

        self.to_cls_token = nn.Identity()
        self.heads = heads

    def forward(self, img, label):
        # p = float(0.1 + torch.rand(1) * 0.1)
        # revised_adj = self.set_random_nonzero_to_zero(self.adj_matrix, p)
        norm_attn_score = self.transformer(img, self.adj_matrix, label)

        # x = self.to_cls_token(x[:, -1])
        return norm_attn_score





