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
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        return self.fn(temp, *args, **kwargs)


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

class Attention_local(nn.Module):
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
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

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

        # self.proj_k_f = nn.Linear(2 * dim_in, self.num_heads, bias=qkv_bias)
        # self.proj_k_f2 = nn.Linear(self.num_heads, self.length, bias=qkv_bias)
        # self.proj_k_f3 = nn.Linear(self.num_heads, self.length2, bias=qkv_bias)
        # self.proj_k_f.weight.requires_grad = False
        # self.proj_k_f2.weight.requires_grad = False
        # self.proj_k_f3.weight.requires_grad = False

        self.topk = self.length2
        self.diff_routing = False
        self.router = TopkRouting(topk=self.topk, diff_routing=self.diff_routing)
        mul_weight = 'none'
        self.k_gather = KGather(mul_weight=mul_weight)
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
        b, head, L, _ = x.size()


        q, k, v, adj_copy = self.forward_conv_qkv(x, adj)

        adj = rearrange(adj_copy, '(b h) t d -> b h t d', h=head)


        kv = torch.cat((k, v), dim=-1)


        kv_pix = rearrange(kv, 'n h t c -> (n h) t c').unsqueeze(2)
        r_weight, r_idx = self.router(adj)
        r_weight, r_idx = rearrange(r_weight, 'n h t c -> (n h) t c'), rearrange(r_idx, 'n h t c -> (n h) t c')
        kv_pix_sel = self.k_gather(r_idx=r_idx, r_weight=r_weight, k=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)

        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.dim//head, self.dim//head], dim=-1)

        k_pix_sel = rearrange(k_pix_sel, '(n h) t k w2 c -> (n t) h  c (k w2)', h=head)
        v_pix_sel = rearrange(v_pix_sel, '(n h) t k w2 c -> (n t) h  (k w2) c', h=head)
        q_pix = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)


        attn_weight = (q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)

        # attn_score = self.leakyrelu(attn_score)  # F.gelu

        attn_weight = self.attn_act(attn_weight)
        v = attn_weight @ v_pix_sel

        v = rearrange(v.squeeze(), '(n t) h k -> n h t k', n=b)


        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=head)))

        out = self.proj_v(v)

        return out


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
        # self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

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

        randomatrix = torch.randn((int(self.num_heads), int(self.num_heads)), requires_grad=True).to(device)
        self.randomatrix = torch.nn.Parameter(randomatrix)
        self.register_parameter("Ablah", self.randomatrix)

        # from torch.nn.parameter import Parameter
        # self.randomatrix = Parameter(torch.FloatTensor(int(self.num_heads), int(self.num_heads))).to(device)

        # self.proj_k_f = nn.Linear(2 * dim_in, self.num_heads, bias=qkv_bias)
        # self.proj_k_f2 = nn.Linear(self.num_heads, self.length, bias=qkv_bias)
        # self.proj_k_f3 = nn.Linear(self.num_heads, self.length2, bias=qkv_bias)
        # self.proj_k_f.weight.requires_grad = False
        # self.proj_k_f2.weight.requires_grad = False
        # self.proj_k_f3.weight.requires_grad = False

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

        current_length = adj.size(-1)

        x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
        b, head, L, _ = x.size()
        q, k = self.forward_conv_qk(x, adj)
        # q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        # k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)



        # values, indices = adj.topk(8, dim=-1, largest=True, sorted=True)
        #
        # k_expanded = k.permute(0, 1, 3, 2).unsqueeze(-1).repeat(1, 1, 1, 1, L)
        # k_gather = torch.index_select(k_expanded, dim=-1, index=indices)
        #
        # k_gather = torch.gather(k_expanded, dim=-1, index=indices)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        # attn_score = self.leakyrelu(attn_score)

        ## ---1-----
        # attn_score = torch.matmul(torch.matmul(Random_RM, attn_score), Random_RM)
        ## ---2-----
        Lambda = self.randomatrix
        Lambda = Lambda.expand(b, -1, -1).to(device)
        attn_score = rearrange(attn_score, 'b h l t -> b h (l t)')
        attn_score = torch.einsum('blh,bhk->blk', [Lambda, attn_score])
        attn_score = rearrange(attn_score, 'b h (l k) -> b h l k', l=current_length)


        rep_adj = F.softmax(attn_score, dim=-1)

        # attn_score = (attn_score + attn_score.transpose(3, 2)) / 2
        # D = torch.diag_embed(torch.sum(attn_score, dim=-1) ** (-1 / 2))
        # D[D == float('inf')] = 0
        # rep_adj = torch.matmul(torch.matmul(D, attn_score), D)

        v = self.forward_conv_v(x, rep_adj)

        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=self.num_heads)))

        out = v

        return out


# class Attention_global(nn.Module):
#     def __init__(self,
#                  depth,
#                  i,
#                  dim,
#                  image_size,
#                  patch_size,
#                  heads=8,
#                  dropout=0,
#                  qkv_bias=False,
#                  attn_drop=0.,
#                  proj_drop=0.,
#                  downsample=0.,
#                  kernel_size=3,
#                  stride_kv=1,
#                  stride_q=1,
#                  padding_kv=1,
#                  padding_q=1,
#                  with_cls_token=False
#                  ):
#         super().__init__()
#         self.scale = dim ** -0.5
#         # self.drop_ratio = 0.1
#
#         # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
#         # self.to_out = nn.Sequential(
#         #     nn.Linear(dim, dim),
#         #     nn.Dropout(dropout)
#         # )
#
#         self.stride_kv = stride_kv
#         self.stride_q = stride_q
#         self.dim = dim
#         self.num_heads = heads
#         self.with_cls_token = with_cls_token
#
#         self.length = int(image_size / patch_size) ** 2
#         self.length2 = int(image_size / patch_size * downsample) ** 2
#
#         self.layer_index = i
#         self.depth = depth
#
#         dim_in = dim
#         dim_out = dim
#
#         self.conv_proj_qkv = GIN(nfeat=dim_in,
#                                head_num=heads,
#                                nhid=3*dim_out,
#                                image_size=image_size,
#                                patch_size=patch_size,
#                                stride=2,
#                                padding=1,  # using 2 when kernel_size = 4
#                                kernel_size=kernel_size,  # kernel_size of GCN
#                                nclass=None,
#                                dropout=dropout)
#
#         # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
#         # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
#
#         # self.proj_q.weight.requires_grad = False
#         # self.proj_k.weight.requires_grad = False
#         # self.proj_v.weight.requires_grad = False
#
#         self.attn_drop = nn.Dropout(attn_drop)  #
#         self.proj_drop = nn.Dropout(proj_drop)  #
#         self.leakyrelu = nn.LeakyReLU()
#
#
#         # sparse_D = torch.ones((self.num_heads, self.length), requires_grad=True).to(device)
#         # self.sparse_D = torch.nn.Parameter(sparse_D)
#         # self.register_parameter("sparse_D", self.sparse_D)
#         #
#         # sparse_D2 = torch.ones((self.num_heads, self.length2), requires_grad=True).to(device)
#         # self.sparse_D2 = torch.nn.Parameter(sparse_D2)
#         # self.register_parameter("sparse_D2", self.sparse_D2)
#         #
#         randomatrix = torch.randn((int(self.num_heads),
#                                    int(self.num_heads)), requires_grad=True).to(device)
#         self.randomatrix = torch.nn.Parameter(randomatrix)
#         self.register_parameter("Ablah", self.randomatrix)
#
#         # self.diff_routing = False
#         # self.router = TopkRouting(topk=self.topk, diff_routing=self.diff_routing)
#         # mul_weight = 'none'
#         # self.k_gather = KGather(mul_weight=mul_weight)
#         self.attn_act = nn.Softmax(dim=-1)
#
#     def forward_conv_qkv(self, x, rep_adj):
#         # if self.with_cls_token:
#         #     cls_token, x = torch.split(x, [1, h * w], 1)
#         if self.conv_proj_qkv is not None:
#             qkv, adj_copy = self.conv_proj_qkv(x, rep_adj)
#             # q = F.dropout(F.relu(q), self.drop_ratio, training=self.training)
#         else:
#             q = rearrange(x, 'b c h w -> b (h w) c')
#         q, k ,v = torch.chunk(qkv, chunks=3, dim=-1)
#
#         return q, k, v, adj_copy
#
#
#     def forward(self, x, adj, rep_adj_dis):
#         x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
#         current_length = adj.size(-1)
#         b, head, L, _ = x.size()
#
#
#         q, k, v, _ = self.forward_conv_qkv(x, adj)
#         q_pix = q #rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)
#         k_pix = rearrange(k, 'n h t c -> n h c t')
#         v_pix = v #rearrange(v, 'n h t c -> (n t) h c').unsqueeze(2)
#
#         attn_weight = (q_pix * self.scale) @ k_pix  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
#
#         attn_weight = self.attn_act(attn_weight)
#         v = attn_weight @ v_pix
#
#         v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=head)))
#
#         out = v
#
#         return out

class ConvEmbed(nn.Module):
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

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_chans, int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # groups=in_chans
            )),
            ('bn', nn.BatchNorm2d(int(embed_dim))),
            ('relu', nn.GELU()),
            ('pooling', nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding,)),
        ]))

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
        self.depth4pool = depth//4   #1.5

        self.depth4pool2 = int(depth // 1.5)  # 1.5
        for i in range(self.depth4pool):
            self.layers.append(nn.ModuleList([
                # Residual(PreNorm(dim, Attention_local(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                PreNorm(dim, Residual(Attention_local(depth, i, dim, image_size=image_size, patch_size=patch_size, heads=heads, dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout),
            ]))

        for i in range(depth-self.depth4pool):
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
        # UT = torch.randn((int(image_size/patch_size*downsample)**2, dim), requires_grad=True).to(device)
        # self.UT = torch.nn.Parameter(UT)
        # self.register_parameter("Ablah2", self.UT)

        # UT2 = torch.randn((int(image_size / patch_size * downsample * downsample) ** 2, dim), requires_grad=True).to(device)
        # self.UT2 = torch.nn.Parameter(UT2)
        # self.register_parameter("Ablah3", self.UT2)

        self.Upool = nn.Sequential(
            nn.Linear(dim, int(image_size/patch_size*downsample)**2, bias=True),
            # nn.Dropout(dropout)
        )
        self.Upool2 = nn.Sequential(
            nn.Linear(dim, int(image_size / patch_size * downsample * 0.5) ** 2, bias=True),
            # nn.Dropout(dropout)
        )

        # Upool_out = torch.randn((1, dim), requires_grad=True).to(device)
        # self.Upool_out = torch.nn.Parameter(Upool_out)
        # self.register_parameter("Ablah4", self.Upool_out)
        self.Upool_out = nn.Sequential(
            nn.Linear(dim, 1, bias=False),)
        # self.Upool_inter = nn.Sequential(
        #     nn.Linear(dim, 1, bias=False), )

    def forward(self, img, adj):
        p = self.patch_size
        b, n, imgh, imgw = img.shape

        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # x = self.patch_to_embedding(x)

        x = rearrange(img, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p, p2=p)
        conv_img = self.patch_embed(x)
        conv_img = rearrange(conv_img, '(b s) c p1 p2 -> b s (c p1 p2)', b=b)
        x = self.patch_to_embedding(conv_img)


        x = rearrange(x, 'b (h w) c -> b c h w', h=int(imgh / p), w=int(imgw / p))
        x = rearrange(x, 'b c h w -> b (h w) c')

        # adj = adj.unsqueeze(1)
        # rep_adj = adj.expand(-1, self.head_num, -1, -1).to(device)
        rep_adj = adj.to(device)
        rep_adj_dis = rep_adj

        index = 0
        for attn, ff in self.layers:
            # x = attn(x, self.rep_adj, 0)
            if index < self.depth4pool:
                x = attn(x, rep_adj, rep_adj_dis)
                x = ff(x)
            else:
                if index == self.depth4pool:
                    # temp = torch.matmul(self.UT, x.permute(0, 2, 1))
                    temp = self.Upool(x).permute(0, 2, 1)
                    C = F.gumbel_softmax(temp, dim=-1, tau=1.0)
                    # C = F.softmax(temp/2, dim=-1)
                    x = torch.matmul(C, x)
                    C = C.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
                    temp2 = torch.matmul(C, rep_adj)
                    rep_adj = torch.matmul(temp2, C.permute(0, 1, 3, 2))

                # if index == self.depth4pool2 - 1:
                #     temp = self.Upool_inter(x).permute(0, 2, 1)
                #     temp = F.softmax(temp / 2, -1)
                #     x_inter = torch.matmul(temp, x)


                if index == self.depth4pool2:
                    # temp = torch.matmul(self.UT2, x.permute(0, 2, 1))
                    temp = self.Upool2(x).permute(0, 2, 1)
                    C = F.gumbel_softmax(temp, dim=-1, tau=1.0)
                    # C = F.softmax(temp/2, dim=-1)
                    x = torch.matmul(C, x)
                    C = C.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
                    temp2 = torch.matmul(C, rep_adj)
                    rep_adj = torch.matmul(temp2, C.permute(0, 1, 3, 2))

                x = attn(x, rep_adj, rep_adj_dis)
                x = ff(x)

            index = index + 1
        # temp = torch.matmul(self.Upool_out, x.permute(0, 2, 1))
        temp = self.Upool_out(x).permute(0, 2, 1)
        # temp = F.gumbel_softmax(temp, dim=-1, tau=1.0)
        temp = F.softmax(temp/2, -1)
        x_out = torch.matmul(temp, x)
        # x_out = torch.cat((x_out, x_inter), dim=-1)
        # x_out = F.normalize(x, dim=-1)
        return x_out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, downsample, batch_size, num_classes, dim, depth, heads,
                 mlp_dim, patch_stride, patch_pading, in_chans, dropout=0., emb_dropout=0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'


        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample,
                                       batch_size, in_chans, patch_stride=patch_stride, patch_padding=patch_pading)

        self.to_cls_token = nn.Identity()  #MLP(dim)  #
        self.heads = heads
        # self.projection = MLP(dim, dim)
        self.predictor = Predictor(dim, num_classes)

    def forward(self, img, gen_adj, label, adj_matrix, interation):

        # if interation is not None and interation %3 ==0:
        #     alpha = 1 - torch.tanh(torch.tensor(interation/16))  # total epoch /2 +1
        #     alpha = int(alpha)
        #     adj_matrix = alpha * adj_matrix + (1-alpha)*gen_adj

        if label == 0:
            x = self.transformer(img, adj_matrix)
        else:
            x = self.transformer(img, gen_adj)
        # x = self.to_cls_token(x[:, -1])
        pred = self.to_cls_token(x.squeeze())
        class_result = self.predictor(x.squeeze())
        return pred, class_result, adj_matrix


# class MLP(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         hidden_size = dim
#         self.net = nn.Sequential(
#             nn.Linear(dim*2, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             # nn.ReLU(inplace=True),
#             # nn.Linear(hidden_size, projection_size)
#         )
#
#     def forward(self, x):
#         return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        hidden_size = dim
        self.mlp_head = nn.Sequential(
            # nn.Linear(dim, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, num_classes),
            nn.Linear(dim, num_classes)
        )

    # def init(self):
    #     init.xavier_uniform_(self.mlp_head[0].weight)  # 例如，使用 Kaiming Normal 初始化
    #     init.constant_(self.mlp_head[0].bias, 0)  # 例如，将所有偏置设置为 0
    #     init.xavier_uniform_(self.mlp_head[3].weight)  # 例如，使用 Kaiming Normal 初始化
    #     init.constant_(self.mlp_head[3].bias, 0)  # 例如，将所有偏置设置为 0

    def forward(self, x):
        return self.mlp_head(x)
