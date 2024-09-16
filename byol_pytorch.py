import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

# helper functions


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# loss fn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor


def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x, gen_adj, label, adj_matrix, interation):
        pred, classresult, adj_matrix_return = self.net(x, gen_adj, label, adj_matrix, interation)
        return pred, classresult, adj_matrix_return

        # if self.layer == -1:
        #     return self.net(x)
        #
        # if not self.hook_registered:
        #     self._register_hook()
        #
        # self.hidden.clear()
        # _ = self.net(x)
        # hidden = self.hidden[x.device]
        # self.hidden.clear()
        #
        # assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        # return hidden

    def forward(self, x, gen_adj, label, adj_matrix, interation, return_projection=True):
        representation, _, adj_matrix_return = self.get_representation(x, gen_adj, label, adj_matrix, interation)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation, adj_matrix_return


# main class
class BYOL(nn.Module):
    def __init__(
        self,
        net,
        gen_vit,
        image_size,
        hidden_layer,
        projection_size,
        projection_hidden_size,
        augment_fn=None,
        augment_fn2=None,
        moving_average_decay=0.99,
        use_momentum=True,
        sync_batchnorm=None
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            # T.Scale(int(1.2*image_size)),
            # RandomApply(
            #     T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            #     p = 0.3
            # ),
            # T.RandomGrayscale(p=0.2),
            # T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (1.0, 2.0)),
            #     p = 0.2
            # ),
            T.RandomRotation(degrees=90, fill=(255, 255, 255)),
            T.RandomResizedCrop(size=image_size, scale=(0.6, 0.9)),  #, ratio=(0.99, 1.0)
            # T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm
        )
        self.gen_vit = gen_vit

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        pantchesalow = image_size // self.net.patch_size
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
        # adj_matrix[random_matrix < 0.8] = 1

        self.adj_matrix2d = adj_matrix  # anker view


        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(2, 3, image_size, image_size, device=device), torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        img1, img2, interation, adj_matrix,
        return_embedding=False,
        return_projection=True
    ):
        assert not (self.training and img1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(img1, return_projection=return_projection)

        image_one, image_two = self.augment1(img1), self.augment2(img2)

        gen_adj = self.gen_vit(image_two, label=True)

        b, _, _, _ = image_one.shape
        if adj_matrix is None:
            adj_matrix = self.adj_matrix2d.expand(b, self.net.heads, -1, -1)

        online_proj_one, _, adj_matrix_return = self.online_encoder(image_one, gen_adj, 1, adj_matrix, interation)
        online_proj_two, _, adj_matrix_return = self.online_encoder(image_two, gen_adj, 1, adj_matrix, interation)
        # label ==0: adj_matrix; label==1: gen_adj
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_proj_one, _, _ = target_encoder(image_one, gen_adj, 0, adj_matrix, interation)
            target_proj_two, _, _ = target_encoder(image_two, gen_adj, 0, adj_matrix, interation)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean(), gen_adj, adj_matrix
