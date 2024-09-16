import torch
from byol_pytorch import BYOL
from torchvision import models
import argparse
from models import *
from models.vit import ViT
from models.gen_vit import gen_vit
from utils import progress_bar
from sklearn.metrics import roc_auc_score
import os
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from torch.utils.data.dataset import ConcatDataset
from utilsfile.mask_utils import create_subgraph_mask2coords, create_rectangle_mask, create_rectangle_mask2coords, create_bond_mask2coords
from utilsfile.public_utils import setup_device
from skimage.feature import corner_harris
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import corner_peaks
from utilsfile.harris import CornerDetection
import time
from warmup_scheduler import GradualWarmupScheduler
import copy


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default='16')  #64
parser.add_argument('--weight_decay', default=1e-6, type=float, help='SGD weight decay')
parser.add_argument('--data_address', default='../data/pretraining\pubchem-10m/bace/', type=str)
parser.add_argument('--n_epochs', type=int, default='0')
parser.add_argument('--n_epochs_tafter', type=int, default='1')
parser.add_argument('--dim', type=int, default='320')
parser.add_argument('--imagesize', type=int, default='224')  #288
parser.add_argument('--patch', default='14', type=int)  #24
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--tau', type=float, default=0.99)
parser.add_argument('--cos', default='True', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
size = int(args.imagesize)

vit = ViT(
        image_size=int(args.imagesize),
        patch_size=args.patch,
        kernel_size=5,
        downsample=0.5,
        batch_size=args.bs,
        num_classes=args.num_classes,
        dim=args.dim,
        depth=12,
        heads=4,
        mlp_dim=args.dim,
        patch_stride=2,
        patch_pading=1,
        in_chans=3,
        dropout=0.2,  # 0.1
        emb_dropout=0.2,  # 0.1
        expansion_factor=2
    ).to(device)

gen_vit = gen_vit(
    image_size=int(args.imagesize),
    patch_size=args.patch,
    kernel_size=5,
    downsample=0.5,####
    batch_size=args.bs,
    num_classes=args.num_classes,
    dim=args.dim,
    depth=3,
    heads=4,
    mlp_dim=args.dim,
    patch_stride=2,
    patch_pading=1,
    in_chans=3,####
    dropout=0.2,   # 0.1
    emb_dropout=0.2,   # 0.1
    expansion_factor=1
    )

learner = BYOL(
    vit,
    gen_vit,
    image_size=args.imagesize,
    hidden_layer='to_cls_token',
    projection_size=args.dim,
    projection_hidden_size=4096
)
# learner.to(device)
opt = torch.optim.Adam(learner.parameters(), lr=args.lr)  #, weight_decay=args.weight_decay, betas=(0.5, 0.999)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(args.n_epochs / 2) + 1)
scheduler = GradualWarmupScheduler(opt, multiplier=2, total_epoch=int(args.n_epochs / 2) + 1,
                                        after_scheduler=scheduler_cosine)


transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

##############kaishi

testset = torchvision.datasets.ImageFolder(root='../data/bace/test_scoffold', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=True, num_workers=0)
transf = transforms.ToTensor()
unloader = transforms.ToPILImage()
harris_detector = CornerDetection().to(device)


transform = transforms.Compose([transforms.Resize(int(size)),
                                # transforms.RandomCrop(size, padding=2),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomVerticalFlip(),
                                # transforms.RandomGrayscale(p=0.2),
                                # transforms.RandomRotation(degrees=0, fill=(255, 255, 255)),
                                # transforms.GaussianBlur(kernel_size=3, sigma=(2.0, 2.0)),
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])


import random
class RandomlyApplyCrop(object):
    def __init__(self, crop_transform, p=0.5):
        """
        p: 概率值，表示应用crop_transform的概率
        crop_transform: 需要随机应用的裁剪变换，例如 transforms.RandomCrop
        """
        self.crop_transform = crop_transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.crop_transform(img)
        else:
            return img
# 定义裁剪变换
random_number = random.uniform(0.7, 0.9)
crop_transform = transforms.RandomCrop(int(size*random_number))
rotation_transform = transforms.RandomRotation(degrees=45, fill=(255, 255, 255))
# 定义随机应用裁剪的变换
randomly_apply_crop = RandomlyApplyCrop(crop_transform, p=0.8)  # 50% 的概率应用裁剪
randomly_apply_rotation = RandomlyApplyCrop(rotation_transform, p=0.9)  # 50% 的概率应用裁剪
transformaug = transforms.Compose([transforms.Resize(int(size)),
                                # randomly_apply_crop,  # 随机应用裁剪
                                randomly_apply_rotation,
                                transforms.Resize(int(size)),
                                # transforms.RandomCrop(size, padding=2),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomVerticalFlip(),
                                # transforms.RandomGrayscale(p=0.2),
                                # transforms.RandomRotation(degrees=0, fill=(255, 255, 255)),
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])


train_after_dataset = torchvision.datasets.ImageFolder(root='../data/bace/train_scoffold/', transform=transform_test)
# for i in range(1):
#     temp = torchvision.datasets.ImageFolder(root='../data/hiv/train_scoffold/', transform=transform)
#     train_after_dataset = ConcatDataset([train_after_dataset, temp])

trainafterloader = torch.utils.data.DataLoader(train_after_dataset, batch_size=int(args.bs), shuffle=True, num_workers=0)
if args.cos:
    from warmup_scheduler import GradualWarmupScheduler


class ImagePairDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.transformaug = transformaug
        self.image_pairs = self.get_image_pairs()

    def get_image_pairs(self):
        # 假设您的图像对是按照一定的命名规则组织的，例如 "image1.jpg" 和 "image2.jpg" 是一对
        image_pairs = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):  # 或者其他图像格式
                base_name = filename.split('.')[0]
                pair1_path = os.path.join(self.image_dir, filename)
                pair2_path = os.path.join(self.image_dir, filename)  # 假设命名规则是这样的, base_name + '_pair.png'
                if os.path.exists(pair2_path):  # 确保第二个图像存在
                    image_pairs.append((pair1_path, pair2_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image1_path, image2_path = self.image_pairs[idx]
        image1 = Image.open(image1_path).convert('RGB')
        image2 = Image.open(image2_path).convert('RGB')

        label = random.randint(0, 0)
        if label == 0:
            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
        else:
            image0 = image1
            image1 = self.transform(image0)
            image2 = self.transformaug(image0)

        return image1, image1


# 定义文件datasetm Saving ..
# Tue Nov 14 00:27:36 2023 Epoch 35, test loss: 1.56787, acc: 88.67925, roc_auc_avg: 0.89025
# best acc=88
training_dir = args.data_address  # 训练集地址

pair_dataset = ImagePairDataset(image_dir=training_dir, transform=transform)

# 定义图像dataloader
train_dataloader = DataLoader(pair_dataset, shuffle=True, batch_size=int(args.bs), pin_memory=True, num_workers=0)

criterion_ce = nn.CrossEntropyLoss().to(device)


def train_after(epoch, vit4trainafter, net4trainafter, opt4net, opt4gen, scheduler2, scheduler3):
    print('\nEpoch: %d' % epoch)
    vit4trainafter.train()
    net4trainafter.train()
    train_loss = 0
    correct = 0
    total = 0
    accumulation = 4


    def hook_gradients(module, grad_in, grad_out):
        if grad_out[0] is not None:
            grad_map = grad_out[0].detach().cpu()  # 捕获梯度
            grad_map = rearrange(grad_map, '(b h w) (c) (p1) (p2) -> b c (h p1) (w p2)', b=args.bs,
                                 h=args.imagesize // args.patch, w=args.imagesize // args.patch)
            grad_map = grad_map.mean(dim=1, keepdim=True)  # 按通道平均
            grad_map_normalized = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min())  # 归一化

            # 绘图
            plt.imshow(grad_map_normalized[0, 0, :, :], cmap='coolwarm', interpolation='bilinear')
            plt.colorbar()
            plt.show()
    first_conv_layer = list(vit4trainafter.transformer.children())[0]
    # 注册梯度钩子
    first_conv_layer.register_backward_hook(hook_gradients)


    for batch_idx, (inputs, targets) in enumerate(trainafterloader):
        inputs, targets = inputs.to(device), targets.to(device)

        gen_adj = vit4trainafter(inputs, label=False)
        _, outputs, _ = net4trainafter(inputs, gen_adj, 1, None, None)

        loss = criterion_ce(outputs, targets)  #+ 0.1*loss0

        loss.backward()

        ##  exhibit the original image
        mean = [0.485, 0.456, 0.406]  # 这些是 ImageNet 的 RGB 通道的均值
        std = [0.229, 0.224, 0.225]  # 这些是 ImageNet 的 RGB 通道的标准差
        img = inputs.float()
        images_denorm = img * torch.tensor(std).view(-1, 1, 1).to(device) + torch.tensor(mean).view(-1, 1, 1).to(device)
        images_denorm = images_denorm.type_as(img)
        augmented_img1 = unloader(images_denorm[0, :, :, :]).convert('RGB')
        plt.imshow(augmented_img1, cmap="brg")
        plt.show()


        if ((batch_idx + 1) % accumulation) == 0:
            opt4net.step()
            opt4net.zero_grad()
            opt4gen.step()
            opt4gen.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    content = time.ctime() + 'opt4net' + f'Epoch {epoch}, lr: {opt4net.param_groups[0]["lr"]:.5f}'
    print(content)

    scheduler2.step(epoch)
    scheduler3.step(epoch)


def test(epoch, vit4trainafter, net4trainafter):

    vit4trainafter.eval()
    net4trainafter.eval()
    test_loss = 0
    correct = 0
    total = 0
    total4roc = 0
    roc_auc = 0



    first_conv_layer = list(vit4trainafter.transformer.children())[0]
    def hook_feature(module, input, output):
        # 保存特征图
        global feature_maps
        feature_maps = output.detach()
        feature_maps = rearrange(feature_maps, '(b h w) (c) (p1) (p2) -> b c (h p1) (w p2)', b=args.bs, h=args.imagesize//args.patch, w=args.imagesize//args.patch)

    hook_handle = first_conv_layer.register_forward_hook(hook_feature)



    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # _, outputs = learner.online_encoder.net(inputs, None, 0, None, None)

            gen_adj = vit4trainafter(inputs, label=False)



            import matplotlib.pyplot as plt
            plt.imshow(feature_maps[0, 0, :, :].cpu(), cmap='hot', interpolation='bilinear')  #bilinear bicubic  nearest
            plt.colorbar()
            plt.show()
            ##  exhibit the original image
            # mean = [0.485, 0.456, 0.406]  # 这些是 ImageNet 的 RGB 通道的均值
            # std = [0.229, 0.224, 0.225]  # 这些是 ImageNet 的 RGB 通道的标准差
            # img = inputs.float()
            # images_denorm = img * torch.tensor(std).view(-1, 1, 1).to(device) + torch.tensor(mean).view(-1, 1, 1).to(device)
            # images_denorm = images_denorm.type_as(img)
            # augmented_img1 = unloader(images_denorm[0, :, :, :]).convert('RGB')
            # plt.imshow(augmented_img1, cmap="brg")
            # plt.show()


            out_representation, outputs, _ = net4trainafter(inputs, gen_adj, 1, None, None)

            loss = criterion_ce(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

            outputssig = torch.sigmoid(outputs)
            tempp = outputssig[:, 1]

            try:
                roc_auc += roc_auc_score(targets.cpu(), tempp.cpu(), average='micro')
                total4roc += 1
            except ValueError:
                pass

    # Save checkpoint.
    acc = 100. * correct / total
    roc_auc_avg = roc_auc / total4roc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, test loss: {test_loss:.5f}, acc: {(acc):.5f}, roc_auc_avg: {(roc_auc_avg):.5f}'
    print(content)
    # with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
    #     appender.write(content + "\n")
    return test_loss, acc, roc_auc_avg


############ train using label $###################################
###################################################################
def trainandevl():
    best_roc = 0
    best_acc = 0
    import copy
    vit4trainafter = copy.deepcopy(learner.gen_vit)
    net4trainafter = copy.deepcopy(learner.online_encoder.net)
    opt4net = torch.optim.Adam(net4trainafter.parameters(),
                               lr=args.lr)  # , weight_decay=args.weight_decay, betas=(0.5, 0.999)
    opt4gen = torch.optim.Adam(vit4trainafter.parameters(),
                               lr=args.lr)  # , weight_decay=args.weight_decay, betas=(0.5, 0.999)

    scheduler_cosine2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4net, int(args.n_epochs_tafter / 2) + 1)

    scheduler_cosine3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt4gen, int(args.n_epochs_tafter / 2) + 1)
    scheduler2 = GradualWarmupScheduler(opt4net, multiplier=2, total_epoch=int(args.n_epochs_tafter / 2) + 1,
                                        after_scheduler=scheduler_cosine2)
    scheduler3 = GradualWarmupScheduler(opt4gen, multiplier=2, total_epoch=int(args.n_epochs_tafter / 2) + 1,
                                        after_scheduler=scheduler_cosine3)


    for epoch in range(0, args.n_epochs_tafter):
        train_after(epoch, vit4trainafter, net4trainafter, opt4net, opt4gen, scheduler2, scheduler3)
        if epoch%1==0:
            test_loss, acc, roc_auc = test(epoch, vit4trainafter, net4trainafter)

            if roc_auc > best_roc:
                best_roc = roc_auc
                best_acc = acc
                # best_model = copy.deepcopy(learner)

    del net4trainafter, vit4trainafter
    return best_acc, best_roc  #, best_model

# learner.load_state_dict(torch.load('improved-net.pth'), strict=False) #-224-0.8056

if args.n_epochs==0:
    # learner.load_state_dict(torch.load('improved-net.pth'), strict=False) #-224-0.8056
    trainandevl()

min_loss = 1e5
accumulation = 4   #4
best_acc_global = 0  # best test accuracy
best_roc_global = 0  # best test roc
biaozhi = 0
total_batch = 0

for interation in range(args.n_epochs):
    print('interation=%d'%interation)
    torch.cuda.synchronize()
    start = time.time()
    train_loss = 0
    learner.train()

    for i, data in enumerate(train_dataloader, 0):
    # for images1, images2 in train_dataloader:

        img1, img2 = data
        # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img1, img2 = img1.to(device), img2.to(device)  # 数据移至GPU

        if interation==0:
            loss, gen_adj, adj_temp = learner(img1, img2, interation, None)
            torch.save(adj_temp, f'./matrix/tensor_{i}.pth')
        else:
            adj_temp = torch.load(f'./matrix/tensor_{i}.pth')
            loss, gen_adj, _ = learner(img1, img2, interation, adj_temp.cuda())

        if interation % 1 == 0:
            adj_temp = adj_temp * args.tau + gen_adj.detach() * (1 - args.tau)
            torch.save(adj_temp, f'./matrix/tensor_{i}.pth')

        train_loss += loss.item()
        loss.backward()
        if ((i + 1) % accumulation) == 0:
            opt.step()
            opt.zero_grad()
        total_batch = i

    learner.update_moving_average()  # update moving average of target encoder
    train_loss = train_loss / (total_batch + 1)

    content = time.ctime() + ' ' + f'Epoch {interation}, Train loss: {train_loss:.4f}, lr: {opt.param_groups[0]["lr"]:.5f}'
    print(content)

    if train_loss < min_loss:
        min_loss = train_loss
        biaozhi = 1

    # if interation >= 50 and interation%10 == 0:
    if interation >= 100 and (interation%5 == 0 or biaozhi==1):
        biaozhi = 0
        best_acc, best_roc = trainandevl()
        if best_roc > best_roc_global:
            best_roc_global = best_roc
            best_acc_global = best_acc
            torch.save(learner.state_dict(), './improved-net.pth')

    # if train_loss < min_loss:
    #     min_loss = train_loss
    #     # save your improved network
    #     torch.save(learner.state_dict(), './improved-net.pth')
    #     torch.save(learner.online_encoder.net.state_dict(), './net.pth')

    scheduler.step(interation)

    torch.cuda.synchronize()
    end = time.time()

    # print("cost time", end-start, "s")
