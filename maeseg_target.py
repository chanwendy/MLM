import shutil
import time
from functools import partial
import cv2
import ipdb
# from monai.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.metrics import DiceMetric

from utils.draw import scatter_plot, scatter_plot_multi
from utils.utils import MySpatialTransform, compute_dtm, hd_loss, AAAI_sdf_loss, compute_sdf_forsdf
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
import torch
import torch.nn as nn
import torchvision
import argparse
import random
from tensorboardX import SummaryWriter
import os
import importlib
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from monai.networks.nets import UNETR,SwinUNETR
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from utils.utils import plot_slides, BaseDataset, NumpyLoader_Multi, NumpyLoader_Multi_merge, NiiLoader, image_resize, \
    CropResize, CopyField, ExtendSqueeze, Reshape, PadToSize, Clip, Binarize, CenterIntensities
from utils.evaluation import binarize, confident_binarize
import random
from utils.saver import Saver
import argparse

from utils.utils import boundary_loss, compute_sdf

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("prefix", help="prefix")
# parser.add_argument("--prefix", help="prefix", default="test")
# parser.add_argument('--model_name', default="featurefusion", type=str)
parser.add_argument("-P", "--target_phase", help="target_phase", default='arterial')
parser.add_argument("-G", "--GPU", help="GPU", default='0')
parser.add_argument("-b", "--batch_size", type=int, help="batch_size", default=8)
parser.add_argument("-E", "--max_epoch", type=int, help="max_epoch", default=1600)
parser.add_argument("--warmup_epochs", type=int, help="max_epoch", default=50)
parser.add_argument("--save_epoch", type=int, help="save_epoch", default=400)
parser.add_argument("--eval_epoch", type=int, help="eval_epoch", default=20)
parser.add_argument("--turn_epoch", type=int, help="turn_epoch", default=-1)
parser.add_argument("-S", "--softrelu", type=int, help="softrelu", default=0)
parser.add_argument("-M", "--method", help="method", default='boundary_train')
# data-root 传入的是multi_all.json文件中对应数据集的路径 如NIH： 每一个病人的data在/data1/2021_stu/MING/NIH/pancreas_xxxx 中， 则data root为：/data1/2021_stu/MING/NIH
parser.add_argument("-R", "--data_root", help="data_root", default='/data/MING/data/NIH')
parser.add_argument("-V", "--val_data_root", help="val_data_root", default='/data/MING/data/NIH')
parser.add_argument("-l", "--data_path", help="data_path", default='/data/MING/code/VAE_06/data/My_Multi_all.json')
parser.add_argument("-t", "--train_list", help="train_list", default='NIH_train')
parser.add_argument("-v", "--val_list", help="val_list", default='NIH_val')
parser.add_argument("--load_prefix", help="load_prefix", default=None)
parser.add_argument("--load_prefix_source", help="load_prefix", default=None)
parser.add_argument("--gan_mode", help="gan_mode", default='cyc')
parser.add_argument("--checkpoint_name", help="checkpoint_name", default="best_model.ckpt")
parser.add_argument("--load_prefix_vae", help="load_prefix_vae", default=None)
parser.add_argument("--load_prefix_mae", help="load_prefix_mae", default=None)
parser.add_argument("--load_prefix_seg", help="load_prefix_seg", default=None)
# parser.add_argument("--load_prefix_mae", help="load_prefix_mae", default=None)
parser.add_argument("--load_prefix_fix", help="load_prefix_fix", default=None)
parser.add_argument("--pan_index", help="pan_index", default='1')
parser.add_argument("--val_finetune", help="val_finetune", default=0)
parser.add_argument("--lambda_vae", type=float, help="lambda_vae", default=0.1)
parser.add_argument("--lambda_vae_warmup", type=int, help="save_epoch", default=0)
parser.add_argument("--lr_seg", type=float, help="lr_seg", default=1e-2)  # for seg 1e-1
parser.add_argument("--lr_vae", type=float, help="lr_vae", default=0)  # for vae 1e-1
parser.add_argument("--lr_mae", type=float, help="lr_mae", default=2.5e-4)  # for vae 1e-1
parser.add_argument("--test_only", help="test_only", action='store_true')
parser.add_argument("--domain_loss_type", type=int, help="domain_loss_type",default=0)
parser.add_argument("--kl", help="use kl loss in domain adaptation", action='store_true')
parser.add_argument("--resume", help="resume", action='store_true')
parser.add_argument("--save_more_reference", help="save_more_reference", action='store_true')
parser.add_argument("--save_eval_result", help="save_more_reference", action='store_true')
parser.add_argument("--no_aug", help="no_aug", action='store_true')
parser.add_argument("--adam", help="no_aug", action='store_true')
parser.add_argument("--mode", help="mode", type=int, default=0)
parser.add_argument("--shift", help="shift", type=int, default=0)
args = parser.parse_args()
kl = args.kl

domain_loss_type = args.domain_loss_type
data_root = args.data_root
val_data_root = args.val_data_root
lr1 = args.lr_seg  # for seg 1e-1
lr2 = args.lr_vae  # for vae 1e-1
lr3 = args.lr_mae # foe mae2.5e-4
train_list = args.train_list
softrelu = args.softrelu
val_list = args.val_list
torch.backends.cudnn.benchmark = True
weight_decay = 0.05
num_workers = 24
gan_mode = args.gan_mode
trainbatch = args.batch_size
shift = args.shift
valbatch = 1
load_prefix = args.load_prefix
checkpoint_name = args.checkpoint_name
load_prefix_vae = args.load_prefix_vae
load_prefix_mae = args.load_prefix_mae
load_prefix_seg = args.load_prefix_seg
load_prefix_fix = args.load_prefix_fix
val_finetune = args.val_finetune
use_confident_binarize = False
load_epoch_seg = 240
load_epoch = 60
prefix = args.prefix
data_path = os.path.join('lists', args.data_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
load_prefix_source = args.load_prefix_source
# NIH
save_root_path = '/data/MING/data/NIH/model'
save_path = '/data/MING/data/NIH/model/' + prefix
display_path = '/data/MING/data/NIH/tensorboard/' + prefix
middle_path = '/data/MING/data/NIH/domain_cache/' + prefix
result_path = '/data/MING/data/NIH/result/' + prefix
if not os.path.exists(save_path):
    # CHAOS
    # save_path = '/HDD_data/MING/VAE/CHAOS/Train_Sets/model/' + prefix
    # NIH
    save_path = '/data/MING/data/NIH/model/' + prefix
    os.mkdir(save_path)

# CHAOS
# save_root_path = '/HDD_data/MING/VAE/CHAOS/Train_Sets/model'
# save_path = '/HDD_data/MING/VAE/CHAOS/Train_Sets/model/' + prefix
# display_path = '/HDD_data/MING/VAE/CHAOS/Train_Sets/tensorboard/' + prefix
# middle_path = '/HDD_data/MING/VAE/CHAOS/Train_Sets/domain_cache/' + prefix
# result_path = '/HDD_data/MING/VAE/CHAOS/Train_Sets/result/' + prefix
max_epoch = args.max_epoch
save_epoch = args.save_epoch
eval_epoch = args.eval_epoch
turn_epoch = args.turn_epoch
pre_epochs = None

assert save_epoch % eval_epoch == 0
assert turn_epoch % eval_epoch == 0 or turn_epoch == -1
# ?
pan_index = args.pan_index
if pan_index != '10':
    # [[0, 0], [1, 1]]
    mask_index = [[0, 0]] + [[int(f), idx + 1] for idx, f in enumerate(pan_index.split(','))]
else:
    mask_index = [[0, 0], [[1, 2], 1]]
target_phase = args.target_phase
# 平衡参数
lambda_vae = args.lambda_vae
lambda_vae_warmup = args.lambda_vae_warmup
test_only = args.test_only
resume = args.resume
method = args.method
save_more_reference = args.save_more_reference
save_eval_result = args.save_eval_result
if save_eval_result and not os.path.exists(result_path):
    os.mkdir(result_path)
no_aug = args.no_aug
adam = args.adam

mode = args.mode
input_phases = ['venous']
output_keys = ['venous']

# ['venous', 'venous_mask', 'venous_origin', 'venous_lung', 'venous_pancreas']
input_phases_mask = input_phases + [f + '_mask' for f in input_phases] + [f + '_origin' for f in input_phases] + [
    f + '_lung' for f in input_phases] + [f + '_pancreas' for f in input_phases]

shift_L = 0
shift_W = 0
shift_H = 0

img_size = 128
patch = 16
channel = 2
input_size = [256, 256, 256]
patch_size = [img_size, img_size, img_size]

pad_size = [0, 0, 0]

## define trainer myself


def filedict_from_json(json_path, key, epoch=1):
    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    listdict = json_dict.get(key, [])
    output = []
    for i in range(epoch):
        output += listdict
    return output

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice(A, B):
    return 2.0 * torch.sum(A * B) / (torch.sum(A) + torch.sum(B) + 0.000001)


# binary cross entropy
def avg_ce(data_dict, source_key='align_lung', target_key='source_lung'):
    source_mask = data_dict[source_key]
    target_mask = data_dict[target_key]
    criterion = nn.BCELoss()

    if not isinstance(source_mask, list):
        source_mask = [source_mask]
    standard_loss_sum = 0
    for im in source_mask:
        standard_loss_sum += criterion(im, target_mask)
    return standard_loss_sum / len(source_mask)


def KLloss(data_dict, mean_key='mean', std_key='std'):
    Mean = data_dict[mean_key]
    Std = data_dict[std_key]
    return torch.mean(0.5 * (torch.sum(torch.pow(Std, 2), (1)) + torch.sum(torch.pow(Mean, 2), (1)) - 2 * torch.sum(
        torch.log(Std + 0.00001), (1))))



# average dice loss
def avg_dsc(data_dict, source_key='align_lung', target_key='source_lung', binary=False, topindex=2, botindex=0,
            pad=[0, 0, 0], return_mean=True, detach=False):
    source_mask = data_dict[source_key]
    target_mask = data_dict[target_key]
    if not detach:
        target_mask = target_mask.cuda()
    else:
        target_mask = target_mask.cuda().detach()

    standard_loss_sum = 0

    if binary:
        label = (torch.argmax(source_mask, dim=1, keepdim=True)).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0), source_mask.size(1), label.size(2), label.size(3),
                                         label.size(4)).zero_()
        source_mask = one_hot.scatter_(1, label.data, 1)
        label = (torch.argmax(target_mask, dim=1, keepdim=True)).type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(label.size(0), target_mask.size(1), label.size(2), label.size(3),
                                         label.size(4)).zero_()
        target_mask = one_hot.scatter_(1, label.data, 1)
    else:
        source_mask = source_mask.cuda()

    if source_mask.shape[1] > 1:
        # standard_loss_sum = standard_loss_sum + dice(source_mask[:,1:2,...],target_mask[:,1:2,...])
        # standard_loss_sum = standard_loss_sum + dice(source_mask[:,2:3,...],target_mask[:,2:3,...])
        if return_mean:
            standard_loss_sum += torch.mean((2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001))[:,
                                            botindex:topindex, ...])
        else:
            standard_loss_sum += torch.mean((2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001))[:,
                                            botindex:topindex, ...], 1)
    else:
        if return_mean:
            standard_loss_sum += torch.mean(2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001))
        else:
            standard_loss_sum += torch.mean(2 * torch.sum(source_mask * target_mask, (2, 3, 4)) / (
                        torch.sum(source_mask, (2, 3, 4)) + torch.sum(target_mask, (2, 3, 4)) + 0.0001), 1)
    return standard_loss_sum

set_threadnumbers = 2
cv2.setNumThreads(set_threadnumbers)
torch.set_num_threads(set_threadnumbers)
os.environ["OMP_NUM_THREADS"] = str(set_threadnumbers)  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = str(set_threadnumbers)  # 设置MKL-DNN CPU加速库的线程数。


if __name__ == "__main__":
    # origin 42 new 21   4  202  66
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # dataset return [] eval_epoch 次的NIH_train的内容
    train_data_list = filedict_from_json(data_path, train_list, eval_epoch)
    # print(train_data_list)

    transforms = {'train': []}
    # define training data pipeline   mask_index [[0, 0], [1, 1]]
    # input_phases = ["venous"]
    transforms['train'].append(
        NumpyLoader_Multi_merge(fields=input_phases, root_dir=data_root, load_mask=True, mask_index=mask_index))
    transforms['train'].append(CropResize(fields=input_phases, output_size=patch_size, shift=shift, shift_W=shift_W, shift_L=shift_L, shift_H=shift_H))
    # transforms['train'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))

    if not no_aug:
        transforms['train'].append(Reshape(fields=input_phases_mask))
        # TODO: do_rotation
        transforms['train'].append(
            MySpatialTransform(patch_size, [dis // 2 - 5 for dis in patch_size], random_crop=True,
                               scale=(0.85, 1.15),
                               do_elastic_deform=False, alpha=(0, 500),
                               do_rotation=True, sigma=(10, 30.),
                               angle_x=(-0.2, 0.2), angle_y=(-0.2, 0.2),
                               angle_z=(-0.2, 0.2),
                               border_mode_data="constant",
                               border_cval_data=-1024,
                               data_key="venous", p_el_per_sample=0, label_key="venous_pancreas",
                               p_scale_per_sample=1, p_rot_per_sample=1))

    # transforms['train'].append(PadToSize(fields=input_phases, size=[1,1]+patch_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))
    for phase in input_phases:
        transforms['train'].append(CopyField(fields=[phase], to_field=[phase + '_origin']))
    transforms['train'].append(Clip(fields=input_phases, new_min=-200, new_max=400))
    transforms['train'].append(CenterIntensities(fields=input_phases, subtrahend=100, divisor=300))
    transforms['train'].append(Reshape(fields=input_phases_mask, reshape_view=[-1] + patch_size))

    val_data_list = filedict_from_json(data_path, val_list)
    transforms['val'] = []
    ## define validation data pipeline

    transforms['val'].append(
        NumpyLoader_Multi_merge(fields=input_phases, root_dir=val_data_root, load_mask=True, mask_index=mask_index))
    transforms['val'].append(CropResize(fields=input_phases, output_size=patch_size, shift=shift, shift_W=shift_W, shift_L=shift_L, shift_H=shift_H))
    # transforms['val'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=False,load_mask=True))
    for phase in input_phases:
        transforms['val'].append(CopyField(fields=[phase], to_field=[phase + '_origin']))
    transforms['val'].append(Clip(fields=input_phases, new_min=-200, new_max=400))
    transforms['val'].append(CenterIntensities(fields=input_phases, subtrahend=100, divisor=300))
    transforms['val'].append(Reshape(fields=input_phases_mask, reshape_view=[-1] + patch_size))

    for k, v in transforms.items():
        transforms[k] = torchvision.transforms.Compose(v)

    ###############################################################################################
    ############################ Create Datasets ##################################################
    ###############################################################################################
    print("Loading data.")
    train_dataset = BaseDataset(train_data_list, transforms=transforms['train'])
    val_dataset = BaseDataset(val_data_list, transforms=transforms['val'])
    if method != "domain_adaptation" or method != "domain_adaptation_vit":
        train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=False, num_workers=num_workers,
                                  drop_last=True, pin_memory=True)
        print("domain!")
    val_loader = DataLoader(val_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)
    # if save_more_reference:
    #     train_loader_2 = DataLoader(train_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers,
    #                                 pin_memory=True)

    ## model build and load
    print("Building model.")
    models = importlib.import_module('joint_model')
    # vm_model = importlib.import_module('models.' + 'voxelmorph3D_joint')
    if method == 'vae_train':
        model = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128,
                           soft=softrelu == 1)
    elif method == 'seg_train' or method== "boundary_train" or method=="hd_train" or method=="sdf_train":
        model= models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
        if load_prefix_mae:
            # model_ref = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128)
            mae_model= models.Mae3D2MaskSeg(img_size=img_size, patch_size=patch, in_chans=2,
                                 embed_dim=768, depth=12, num_heads=12,
                                 decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
                                 mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            # mae_model = mae_model.cuda()

    elif method== "boundary_train_vit" or method=="hd_train_vit":
        model = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
        fix_segmodel =  SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
    elif method == "domain_adaptation":
        model= models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
        best_model = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
        fix_segmodel = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
        mae_model = models.Mae3D2MaskSeg(img_size=img_size, patch_size=patch, in_chans=2,
                                         embed_dim=768, depth=12, num_heads=12,
                                         decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
                                         mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if val_finetune != 0:
            finetune_model = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
            finetune_model = finetune_model.cuda()
    elif method == "domain_adaptation_vit":
        model = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )

        fix_segmodel =SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
        mae_model = models.Mae3D2MaskSeg(img_size=img_size, patch_size=patch, in_chans=2,
                                         embed_dim=768, depth=12, num_heads=12,
                                         decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
                                         mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if val_finetune != 0:
            finetune_model = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
            finetune_model = finetune_model.cuda()
    elif method == "domain_adaptation_vit_vae":
        model = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )

        fix_segmodel =SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
        vae_model = models.VAE(n_channels=len(mask_index), n_class=len(mask_index),norm_type=1,dim=128)

        if val_finetune != 0:
            finetune_model = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )
            finetune_model = finetune_model.cuda()



    model = model.cuda()
    # best_model = best_model.cuda()

    if load_prefix:
        register_model_path = save_root_path + '/' + load_prefix + '/best_model.ckpt'
        if method == "seg_train" or method == "domain_adaptation_vit_vae" or method == "boundary_train_vit" or method == "hd_train_vit" or method == "boundary_train" or method == "hd_train":
            try:
                model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            except Exception as e:
                new_state_dict = {}
                checkpoint = torch.load(register_model_path)['model_state_dict']
                for k, v in checkpoint.items():
                    new_state_dict[k[7:]] = v
                model.load_state_dict(new_state_dict)
            print("seg load finish")
        elif method == "domain_adaptation" or method == "domain_adaptation_vit":
            try:
                model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
                fix_segmodel.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            except Exception as e:
                new_state_dict = {}
                checkpoint = torch.load(register_model_path)['model_state_dict']
                for k, v in checkpoint.items():
                    new_state_dict[k[7:]] = v
                model.load_state_dict(new_state_dict)
                fix_segmodel.load_state_dict(new_state_dict)

            fix_segmodel = fix_segmodel.cuda()
            for param in fix_segmodel.parameters():
                param.requires_grad = False
            fix_segmodel.eval()
            print("both seg load finish")
        else:
            model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])

    # if load_prefix_source :
    #     register_model_path = save_root_path + '/' + load_prefix_source + '/best_model.ckpt'
    #     fix_model_path = save_root_path + '/' + load_prefix + '/best_model.ckpt'
    #     if method == "seg_train" or method == "domain_adaptation_vit_vae" or method == "boundary_train_vit" or method == "hd_train_vit" or method == "boundary_train" or method == "hd_train":
    #         try:
    #             model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    #         except Exception as e:
    #             new_state_dict = {}
    #             checkpoint = torch.load(register_model_path)['model_state_dict']
    #             for k, v in checkpoint.items():
    #                 new_state_dict[k[7:]] = v
    #             model.load_state_dict(new_state_dict)
    #         print("seg load finish")
    #     elif method == "domain_adaptation" or method == "domain_adaptation_vit":
    #         # model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    #         try:
    #             model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    #             fix_segmodel.load_state_dict(torch.load(fix_model_path)['model_state_dict'])
    #         except Exception as e:
    #             new_state_dict = {}
    #             checkpoint = torch.load(register_model_path)['model_state_dict']
    #             for k, v in checkpoint.items():
    #                 new_state_dict[k[7:]] = v
    #             model.load_state_dict(new_state_dict)
    #
    #             fnew_state_dict = {}
    #             fcheckpoint = torch.load(fix_model_path)['model_state_dict']
    #             for k, v in fcheckpoint.items():
    #                 fnew_state_dict[k[7:]] = v
    #             fix_segmodel.load_state_dict(fnew_state_dict)
    #
    #         fix_segmodel = fix_segmodel.cuda()
    #         for param in fix_segmodel.parameters():
    #             param.requires_grad = False
    #         fix_segmodel.eval()
    #         print("both seg load finish")
    #     else:
    #         model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])


    if load_prefix_seg:
        register_model_path = save_root_path + '/' + load_prefix_seg + '/best_model.ckpt'
        fix_model_path = save_root_path + '/' + load_prefix_fix + '/best_model.ckpt'
        if method == "seg_train":
            model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            print("seg load finish")
        elif method == "domain_adaptation" or method == "domain_adaptation_vit":
            new_state_dict = {}
            checkpoint = torch.load(register_model_path)['model_state_dict']
            for k, v in checkpoint.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
            pre_epochs = torch.load(register_model_path)['epoch']
            pre_best_result = torch.load(register_model_path)['best_result']
            fix_segmodel.load_state_dict(torch.load(fix_model_path)['model_state_dict'])
            fix_segmodel = fix_segmodel.cuda()
            for param in fix_segmodel.parameters():
                param.requires_grad = False
            fix_segmodel.eval()
            print("both seg load finish")
        else:
            model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    if method != 'domain_adaptation_vit_vae':
        # VAE
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=lr1, weight_decay=weight_decay, momentum=0.9)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr3, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr1,weight_decay = weight_decay,momentum=0.9)
        optimizer_finetune = torch.optim.SGD(finetune_model.parameters(),
                                             lr=lr1, weight_decay=weight_decay, momentum=0)

        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=lr1,weight_decay = weight_decay,momentum=0.9)
    #
    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epoch
    # )

    print("Loading prefix.")
    if load_prefix_vae:
        register_model_path = save_root_path+'/'+load_prefix_vae+'/best_model.ckpt'
        vae_model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        vae_model = vae_model.cuda()
        '''
        pretrained_dict = torch.load(register_model_path)['model_state_dict']
        model_dict = register_model.segmentation_model.state_dict()
        pretrained_dict = {k.split('.',1)[1]: v for k, v in pretrained_dict.items() if k.split('.',1)[0]=='segmentation_model' and k.split('.',1)[1] in model_dict}
        model_dict.update(pretrained_dict) 
        register_model.segmentation_model.load_state_dict(model_dict)
        register_model.segmentation_model.eval()
        '''
    if load_prefix_mae:
        # load best
        register_model_path = save_root_path + '/' + load_prefix_mae + '/best_model.ckpt'
        # load newest
        # register_model_path = save_root_path + '/' + load_prefix_mae + '/model_epoch4800.ckpt'
        if method == "mae_train" or method == "mae_train_input" or method == "mae768_train" or method == "mae768_1ctrain" or method == "maecyc_train" or method == "maecyc_train_input" or method == "maecyc1c_train":
            model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            pre_epochs = torch.load(register_model_path)['epoch']
            pre_best_result = torch.load(register_model_path)['best_result']
        else:
            pass
            if method == "seg_train" or method == "domain_adaptation" or method == "domain_adaptation_vit":
                print('load finish')
                mae_model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
                ref_model_parallel = nn.DataParallel(mae_model).cuda()
                for param in ref_model_parallel.parameters():
                    param.requires_grad = False
                mae_model.eval()
            else:
                model.MAE.load_state_dict(torch.load(register_model_path)['model_state_dict'])
    # fix mae
    # if method == 'joint_train' or method == 'sep_joint_train' or method == 'embed_train' or method == 'domain_adaptation':
    #     for param in model.MAE.parameters():
    #         param.requires_grad = False
    #     model.MAE.eval()
    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model = nn.DataParallel(model).cuda()

    label_key = 'venous_pancreas'
    img_key = 'venous'
    if method == "mae_train_input":
        best_result = 1
    else:
        best_result = 0
    train_dis = 0
    max_idx_in_epoch = 0
    if gan_mode == "cyc":
        criterionGAN = torch.nn.MSELoss()
    else:
        criterionGAN = torch.nn.BCELoss()
    saver = Saver(display_dir=display_path, display_freq=10)
    MSE_Loss = MSELoss()
    best_result_ct = 0
    if pre_epochs:
        epochs = pre_epochs // eval_epoch
        best_result = pre_best_result
    else:
        epochs = 0
    ## training loop
    soft = nn.Softmax(dim=1)
    print("Start training")
    pretrain_discriminator_epochs = 5
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    pretrain_epoch = 0
    alpha = 1.0
    beta = 0.996
    warm_up_epoch = 0
    flag = 1
    current_epoch=0
    for epoch in range(epochs, max_epoch // eval_epoch):
        if not test_only:

            for idx, batch in enumerate(train_loader):
                if idx > max_idx_in_epoch:
                    max_idx_in_epoch = idx
                # optimizer.param_groups[0]['lr'] = lr3/(10**(epoch//10))
                # for out_list in range(len(output_keys)):
                if epoch>= warm_up_epoch and flag :
                    # model = best_model
                    flag = 0


                optimizer.zero_grad()
                # forward + backward + optimize

                if method == 'seg_train':
                    start = time.time()
                    # if epoch == 0: continue
                    # label_key = "venous_pancreas"  img_key = "venous"
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[img_key] = batch[img_key].cuda()
                    # print("1")
                    # 获取输出结果 label_key + '_pred' shape [b,2,imgsize,imgsize,imgsize]
                    batch = model(batch, img_key, label_key + '_pred')
                    h = batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                    if load_prefix_mae:
                        with torch.no_grad():
                            maeloss, batch[label_key + '_pred_recon'], batch[
                                label_key + '_mae_mask'] = ref_model_parallel(
                                batch[label_key + '_pred'], unpatch=True, mask_ratio=0.75, method="ratio")  # 0.2
                            # batch, maeloss = model(batch, img_key, label_key + '_pred', label_key + '_pred_recon', label_key + '_mae_mask',
                            #                              unpatch=True, mask_ratio=2)

                            # 获得mae重建结果以及maeloss
                            mae_mask = batch[label_key + '_mae_mask'].detach()
                            # mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.MAE.patch_size[0]**3 * 2)  # todo
                            # mae_mask = model.MAE.unpatchify(mae_mask)
                            mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, mae_model.patch_size[0] ** 3 * 2)  # todo
                            mae_mask = mae_model.unpatchify(mae_mask)

                            # 在tensorboard中显示的图像从左到右有三幅图像，分别为输入图像， 以及label，最后为网络的预测
                            batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                                       batch[label_key + '_only'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       mae_mask[:1, :1, :, :, h // 2],
                                                                       batch[label_key + '_pred_recon'][0:1, 1:2, :, :,
                                                                       h // 2]),
                                                                      dim=0)
                    else:
                        batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                                   batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                                                   batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2]),
                                                                  dim=0)
                    dsc_loss = 1 - avg_dsc(batch, source_key=label_key + '_pred', target_key=label_key + '_only',
                                           botindex=1, topindex=len(mask_index))
                    if load_prefix_mae:
                        recon_loss = 1 - avg_dsc(batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_pred_recon', botindex=1,
                                                 topindex=len(mask_index), return_mean=True)
                        # maeloss = maeloss.mean()
                        final_loss = dsc_loss + 0.6 * recon_loss
                        # final_loss = dsc_loss + 0.6 * recon_loss
                    else:
                        final_loss = dsc_loss
                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    if load_prefix_mae:
                        loss.append(['mae_loss', maeloss.item()])
                        loss.append(['recon_diceloss', recon_loss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])

                    display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)


                    # print("cost time {}".format(end - start))

                if method == "boundary_train" or method=="hd_train" or method=="boundary_train_vit" or method=="hd_train_vit"or method=="sdf_train":
                    # if epoch == 0:continue
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[img_key] = batch[img_key].cuda()
                    batch[label_key] = batch[label_key].cuda()
                    if method == "boundary_train" or method == "hd_train" or method=="sdf_train":
                        # batch = model(data_dict=batch, in_key=img_key, out_key=label_key + '_pred', method=method)
                        batch = model(batch, img_key, label_key + '_pred',method=method)

                    else:
                        batch[label_key + '_pred'] = model(batch[img_key])
                        batch[label_key + '_pred'] = soft(batch[label_key + '_pred'])
                    h = batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]

                    if method == "boundary_train" or method == "boundary_train_vit":
                        loss_seg = F.cross_entropy(batch[label_key + '_pred'], batch[label_key][:, 0, ...].type(torch.cuda.LongTensor))
                        batch[label_key + '_pred'] = F.softmax(batch[label_key + '_pred'], dim=1)
                        dsc_loss = 1 - avg_dsc(batch, source_key=label_key + '_pred',
                                               target_key=label_key + '_only',
                                               botindex=1, topindex=len(mask_index))
                        with torch.no_grad():
                            gt_sdf_npy = compute_sdf(batch[label_key].cpu().numpy(), batch[label_key + '_pred'].shape)
                            gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(batch[label_key + '_pred'].device.index)

                        loss_boundary = boundary_loss(batch[label_key + '_pred'], gt_sdf)
                        final_loss = alpha * (loss_seg + dsc_loss) + (1 - alpha) * loss_boundary
                    elif method == "hd_train" or method == "hd_train_vit":
                        loss_seg = F.cross_entropy(batch[label_key + '_pred'], batch[label_key][:, 0, ...].type(torch.cuda.LongTensor))

                        batch[label_key + '_pred'] = F.softmax(batch[label_key + '_pred'], dim=1)
                        dsc_loss = 1 - avg_dsc(batch, source_key=label_key + '_pred',
                                               target_key=label_key + '_only',
                                               botindex=1, topindex=len(mask_index))
                        with torch.no_grad():
                            gt_dtm_npy = compute_dtm(batch[label_key].cpu().numpy(), batch[label_key + '_pred'].shape)
                            gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(batch[label_key + '_pred'].device.index)
                            seg_dtm_npy = compute_dtm(batch[label_key + '_pred'][:, 1, :, :, :].cpu().numpy() > 0.5, batch[label_key + '_pred'].shape)
                            seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(batch[label_key + '_pred'].device.index)

                        loss_hd = hd_loss(batch[label_key + '_pred'], batch[label_key][:,0,...], seg_dtm, gt_dtm)
                        final_loss = alpha * (loss_seg + dsc_loss) + (1 - alpha) * loss_hd
                    else:

                        with torch.no_grad():
                            gt_dis = compute_sdf_forsdf(batch[label_key][:,0,...].cpu().numpy(), batch[label_key + '_pred'][:,1:,...].shape)
                            # print('np.max(gt_dis), np.min(gt_dis): ', np.max(gt_dis), np.min(gt_dis))
                            gt_dis = torch.from_numpy(gt_dis).float().cuda()
                            gt_dis_prob = torch.sigmoid(-1500 * gt_dis)
                            # gt_dis_prob = 1.0 / (1.0 + torch.exp(1500*gt_dis))
                            gt_dis_dice = dice_loss(gt_dis_prob[:, 0, :, :, :], batch[label_key])
                            # gt_dis_dice loss should be <= 0.05 (Dice Score>0.95), which means the pre-computed SDF is right.
                            print('check gt_dis; dice score = ', 1 - gt_dis_dice.cpu().numpy())
                        loss_sdf_aaai = AAAI_sdf_loss(batch[label_key + '_pred'][:,1:,...], gt_dis)
                        # loss_l1 = torch.norm(batch[label_key + '_pred'][:, 0, :, :, :] - gt_dis, 1)/torch.numel(batch[label_key + '_pred'][:, 0, :, :, :])
                        # SDF Prediction -> heaviside function [0,1] -> Dice loss
                        outputs_soft = torch.sigmoid(-1500 * batch[label_key + '_pred'])
                        dsc_loss = dice_loss(outputs_soft[:, 0, :, :, :], batch[label_key + '_only'])

                        final_loss = dsc_loss + 10 * loss_sdf_aaai  # lambda=10 in this paper
                        # final_loss = dsc_loss + loss_l1  # lambda=10 in this paper

                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    if load_prefix_mae:
                        loss.append(['mae_loss', maeloss.item()])
                        loss.append(['recon_diceloss', recon_loss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])

                    # display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)


                    # print("cost time {}".format(end - start))

                if method == 'domain_adaptation' or method == 'domain_adaptation_vit':
                    # if epoch == 0:continue

                    if current_epoch != epoch:
                        print("Updating Network")
                        # model_fix.load_state_dict(model.state_dict())
                        sd_teacher = fix_segmodel.state_dict()
                        sd_student = model.state_dict()
                        for key in sd_student:
                            sd_teacher[key[7:]] = beta * sd_teacher[key[7:]] + (1 - beta) * sd_student[key]
                        fix_segmodel.load_state_dict(sd_teacher)
                            # if tag: lambda_vae /= 10

                    current_epoch = epoch

                    # label_key = "venous_pancreas"  img_key = "venous"
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[img_key] = batch[img_key].cuda()
                    # print("1")
                    # 获取输出结果 label_key + '_pred' shape [b,2,imgsize,imgsize,imgsize]
                    if method == 'domain_adaptation':
                        batch = model(batch, img_key, label_key + '_pred')
                    elif method == "domain_adaptation_vit":
                        batch[label_key + '_pred'] = model(batch[img_key])
                        batch[label_key + '_pred'] = soft(batch[label_key + '_pred'])

                    h = batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]

                    with torch.no_grad():
                        if method == 'domain_adaptation':
                            batch = fix_segmodel(batch, img_key, label_key + '_only_fake')
                        elif method == "domain_adaptation_vit":
                            batch[label_key + '_only_fake'] = fix_segmodel(batch[img_key])
                            batch[label_key + '_only_fake'] = soft(batch[label_key + '_only_fake'])

                        if load_prefix_mae:
                            maeloss, batch[label_key + '_pred_recon'], batch[
                                label_key + '_mae_mask'] = ref_model_parallel(
                                batch[label_key + '_pred'], unpatch=True, mask_ratio=0.75, method="ratio")  # 0.2
                            # batch, maeloss = model(batch, img_key, label_key + '_pred', label_key + '_pred_recon', label_key + '_mae_mask',
                            #                              unpatch=True, mask_ratio=2)

                            # 获得mae重建结果以及maeloss
                            mae_mask = batch[label_key + '_mae_mask'].detach()
                            # mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.MAE.patch_size[0]**3 * 2)  # todo
                            # mae_mask = model.MAE.unpatchify(mae_mask)
                            mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, mae_model.patch_size[0] ** 3 * 2)  # todo
                            mae_mask = mae_model.unpatchify(mae_mask)

                            # 在tensorboard中显示的图像从左到右有三幅图像，分别为输入图像， 以及label，最后为网络的预测
                            batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                                       batch[label_key + '_only'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_only_fake'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred_recon'][0:1, 1:2, :, :,
                                                                       h // 2]
                                                                       ),
                                                                      dim=0)
                        else:
                            batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                                       batch[label_key + '_only'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_only_fake'][0:1, 1:2, :, :,
                                                                       h // 2]),
                                                                      dim=0)

                    dsc_loss = 1 - abs(avg_dsc(batch, source_key=label_key + '_pred', target_key=label_key + '_only',
                                               botindex=1, topindex=len(mask_index)))
                    dsc_loss_pseudo = 1 - abs(avg_dsc(batch, source_key=label_key + '_only_fake',
                                                      target_key=label_key + '_only',
                                                      botindex=1, topindex=len(mask_index)))
                    dsc_loss_fake = 1 - abs(avg_dsc(batch, source_key=label_key + '_pred',
                                                    target_key=label_key + '_only_fake', botindex=1,
                                                    topindex=len(mask_index), return_mean=True))

                    if load_prefix_mae and epoch >=warm_up_epoch:
                        recon_loss = 1 - abs(avg_dsc(batch, source_key=label_key + '_pred',
                                                     target_key=label_key + '_pred_recon', botindex=1,
                                                     topindex=len(mask_index), return_mean=True))
                        if recon_loss < 0:
                            recon_loss = abs(recon_loss)

                        # final_loss = 1.0 * dsc_loss_fake

                        final_loss = 1.8 * dsc_loss_fake + 0.5 * recon_loss

                        # final_loss = dsc_loss + 0.6 * recon_loss

                    else:

                        final_loss = dsc_loss_fake

                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    loss.append(['dice_loss_fake', dsc_loss_fake.item()])
                    loss.append(['dsc_loss_pseudo', dsc_loss_pseudo.item()])
                    if load_prefix_mae and epoch>=warm_up_epoch:
                        loss.append(['mae_loss', maeloss.item()])
                        loss.append(['recon_diceloss', recon_loss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])

                    display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)
                    end = time.time()

                    # print("cost time {}".format(end - start))

                if method == 'domain_adaptation_vit_vae':

                    current_epoch = epoch
                    # if epoch == 0: continue
                    # label_key = "venous_pancreas"  img_key = "venous"
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[img_key] = batch[img_key].cuda()
                    # print("1")
                    # 获取输出结果 label_key + '_pred' shape [b,2,imgsize,imgsize,imgsize]
                    batch[label_key + '_pred'] = model(batch[img_key])
                    batch[label_key + '_pred'] = soft(batch[label_key + '_pred'])

                    h = batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]

                    with torch.no_grad():
                        batch[label_key + '_only_fake'] = fix_segmodel(batch[img_key])
                        batch[label_key + '_only_fake'] = soft(batch[label_key + '_only_fake'])


                        if load_prefix_vae:
                            batch[label_key + '_pred_recon'], batch['mean'], batch['std'] = vae_model(
                                batch[label_key + '_pred'], if_random=True, scale=0.35)  # 0.2
                            # batch, maeloss = model(batch, img_key, label_key + '_pred', label_key + '_pred_recon', label_key + '_mae_mask',
                            #                              unpatch=True, mask_ratio=2)


                            # 在tensorboard中显示的图像从左到右有三幅图像，分别为输入图像， 以及label，最后为网络的预测
                            batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                                       batch[label_key + '_only'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_only_fake'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred_recon'][0:1, 1:2, :, :,
                                                                       h // 2]
                                                                       ),
                                                                      dim=0)
                        else:
                            batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                                       batch[label_key + '_only'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_pred'][0:1, 1:2, :, :,
                                                                       h // 2],
                                                                       batch[label_key + '_only_fake'][0:1, 1:2, :, :,
                                                                       h // 2]),
                                                                      dim=0)

                    dsc_loss = 1 - abs(avg_dsc(batch, source_key=label_key + '_pred', target_key=label_key + '_only',
                                               botindex=1, topindex=len(mask_index)))
                    dsc_loss_pseudo = 1 - abs(avg_dsc(batch, source_key=label_key + '_only_fake',
                                                      target_key=label_key + '_only',
                                                      botindex=1, topindex=len(mask_index)))
                    dsc_loss_fake = 1 - abs(avg_dsc(batch, source_key=label_key + '_pred',
                                                    target_key=label_key + '_only_fake', botindex=1,
                                                    topindex=len(mask_index), return_mean=True))
                    klloss = KLloss(batch)

                    if load_prefix_vae:
                        recon_loss = 1 - avg_dsc(batch, source_key=label_key + '_pred',
                                                     target_key=label_key + '_pred_recon', botindex=1,
                                                     topindex=len(mask_index), return_mean=True)

                        # final_loss = 1.0 * dsc_loss_fake
                        if domain_loss_type == 8 or domain_loss_type == 15 or domain_loss_type == 16:
                            if recon_loss < 0.15:
                                cur_lambda = lambda_vae * 0.6
                            elif recon_loss < 0.225:
                                cur_lambda = lambda_vae * 1.2
                            elif recon_loss < 0.3:
                                cur_lambda = lambda_vae * 2.0
                            else:
                                cur_lambda = lambda_vae * 3.0
                            if cur_lambda > 1:
                                if kl:
                                    final_loss = recon_loss + klloss + 1 / cur_lambda * dsc_loss_fake
                                else:
                                    final_loss = recon_loss + 1 / cur_lambda * dsc_loss_fake
                            else:
                                if kl:
                                    final_loss = cur_lambda * (recon_loss + klloss) + dsc_loss_fake
                                else:
                                    final_loss = cur_lambda * recon_loss + dsc_loss_fake
                        elif domain_loss_type == 9:
                            if recon_loss < 0.15:
                                cur_lambda = lambda_vae * 0.6
                            elif recon_loss < 0.225:
                                cur_lambda = lambda_vae * 1.2
                            elif recon_loss < 0.3:
                                cur_lambda = lambda_vae * 2.0
                            else:
                                cur_lambda = lambda_vae * 3.0
                            final_loss = (cur_lambda * recon_loss + dsc_loss_fake) / (1 + cur_lambda)
                        elif domain_loss_type == 10:
                            loss_square = torch.mean(torch.square(val_batch[label_key + '_pred']))
                            final_loss = loss_square + recon_loss + dsc_loss_fake
                            print(final_loss)
                        elif domain_loss_type == 11:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake + recon_loss * dsc_loss_fake
                        elif domain_loss_type == 12:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake - recon_loss * dsc_loss_fake
                        elif domain_loss_type == 13:
                            recon_loss -= 0.15
                            recon_loss[recon_loss < 0] = 0
                            final_loss = lambda_vae * recon_loss
                        elif domain_loss_type == 14:
                            recon_loss -= 0.1
                            recon_loss[recon_loss < 0] = 0
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake
                        elif turn_epoch != -1:
                            if (epoch // turn_epoch) % 2 == 0:
                                final_loss = lambda_vae * recon_loss
                            else:
                                final_loss = lambda_vae * recon_loss + dsc_loss_fake
                        elif epoch >= lambda_vae_warmup:
                            final_loss = lambda_vae * recon_loss + dsc_loss_fake
                            if kl: final_loss += 0.00002 * lambda_vae * klloss
                        else:
                            final_loss = lambda_vae * epoch / lambda_vae_warmup * recon_loss + dsc_loss_fake

                        # final_loss = dsc_loss + 0.6 * recon_loss

                    else:
                        final_loss = dsc_loss_fake

                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    loss.append(['dice_loss_fake', dsc_loss_fake.item()])
                    loss.append(['dsc_loss_pseudo', dsc_loss_pseudo.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])

                    display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)
                    end = time.time()

                    # print("cost time {}".format(end - start))


                # long running
                # do something other
                # final_loss.backward(torch.ones(final_loss.shape).to("cuda:0"))
                final_loss.backward()
                optimizer.step()
                # scheduler.step()


                if method == 'seg_train':
                    print('[%3d, %3d] loss: %.4f, %.4f, %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), maeloss.item(), recon_loss.item()))
                if method == 'boundary_train' or method == "boundary_train_vit":
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), loss_boundary.item()))
                if method == 'hd_train' or method == "hd_train_vit":
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), loss_hd.item()))
                if method == 'sdf_train' or method == "sdf_train_vit":
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), loss_sdf_aaai.item()))
                          # ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), loss_l1.item()))

                if method == 'domain_adaptation' or method == 'domain_adaptation_vit':
                    if load_prefix_mae and epoch >=warm_up_epoch:
                        print('[%3d, %3d] loss: %.4f, %.4f, %.4f %.4f' %
                              ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), recon_loss.item(),
                               dsc_loss_pseudo.item(), dsc_loss_fake.item()))


                    else:
                        print('[%3d, %3d] loss: %.4f, %.4f, ' %
                              ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), dsc_loss_fake.item()))

                if method == 'domain_adaptation_vit_vae':
                    print('[%3d, %3d] loss: %.4f, %.4f, ' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), dsc_loss_fake.item()))
                end = time.time()
                # print("after seg time cost {}".format(end - start))

            end = time.time()
            if method == "boundary_train" or method == "boundary_train_vit":

                alpha -= 0.01
                if alpha <= 0.01:
                    alpha = 0.01
            else:
                alpha -= 0.001
                if alpha <= 0.001:
                    alpha = 0.001
            print("all spend time {}".format(end - start))

        print("Ready validation")
        # epoch 4 weird
        # validation
        if (epoch + 1) % 1 == 0 or test_only:
            print("Start evaluation")
            model.eval()
            score = {}
            score_visiable = {}
            score_figure = {}
            score_figure_gt = {}
            score_figure_pseudo = {}

            if method == 'vae_train':
                dsc_pancreas = 0.0
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        val_batch[label_key + '_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key + '_only'].size(0), len(mask_index),
                                                         val_batch[label_key + '_only'].size(2),
                                                         val_batch[label_key + '_only'].size(3),
                                                         val_batch[label_key + '_only'].size(4)).zero_()
                        val_batch[label_key + '_only'] = one_hot.scatter_(1, val_batch[label_key + '_only'].data, 1)
                        val_batch[label_key + '_recon'], _, _ = model(val_batch[label_key + '_only'], if_random=False)
                        if save_more_reference and val_idx == epoch % len(val_loader):
                            h = val_batch[label_key + '_only'][0:1, 0:1, :, :, :].shape[4]
                            val_batch[label_key + '_display'] = torch.cat(
                                (val_batch[label_key + '_only'][0:1, 0:1, :, :, h // 2], \
                                 val_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                 val_batch[label_key + '_recon'][0:1, 1:2, :, :, h // 2]), dim=0)
                        score[val_idx] = avg_dsc(val_batch, source_key=label_key + '_recon',
                                                 target_key=label_key + '_only', binary=True, botindex=1,
                                                 topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]

                    dsc_pancreas /= (val_idx + 1)

            if method == "boundary_train" or method=="hd_train" or method=="boundary_train_vit" or method=="hd_train_vit" or method=="sdf_train":
                analysis_figure_name = "None"
                dsc_pancreas = 0.0
                score_mr = {}
                dsc_pancreas_mr = 0.0
                loss_gt = 0.0
                loss_recon = 0.0
                loss_fake = 0.0
                display_image = {}

                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        val_batch[label_key + '_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key + '_only'].size(0), len(mask_index),
                                                         val_batch[label_key + '_only'].size(2),
                                                         val_batch[label_key + '_only'].size(3),
                                                         val_batch[label_key + '_only'].size(4)).zero_()
                        val_batch[label_key + '_only'] = one_hot.scatter_(1, val_batch[label_key + '_only'].data, 1)
                        val_batch[img_key] = val_batch[img_key].cuda()
                        if method == "boundary_train" or method == "hd_train" :
                            val_batch = model(val_batch, img_key, label_key + '_pred', method=method)
                            val_batch[label_key + '_pred'] = soft(val_batch[label_key + '_pred'])
                        elif method == "sdf_train":
                            val_batch = model(val_batch, img_key, label_key + '_pred', method=method)

                            val_batch[label_key + '_pred'] = torch.sigmoid(-1500 * val_batch[label_key + '_pred'])

                        else:
                            val_batch[label_key + '_pred'] = model(val_batch[img_key])
                            val_batch[label_key + '_pred'] = soft(val_batch[label_key + '_pred'])

                        score[val_idx] = avg_dsc(val_batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_only', binary=True, botindex=1,
                                                 topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]
                    dsc_pancreas /= (val_idx + 1)

            if method == 'seg_train' or method == 'joint_train' or method == 'domain_adaptation' or method == 'domain_adaptation_vit' or method == 'sep_joint_train' or method == 'embed_train' or method == 'refine_vae':
                dsc_pancreas = 0.0
                score_mr = {}
                dsc_pancreas_mr = 0.0
                loss_gt = 0.0
                loss_recon = 0.0
                loss_fake = 0.0
                display_image = {}
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        val_batch[label_key + '_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key + '_only'].size(0), len(mask_index),
                                                         val_batch[label_key + '_only'].size(2),
                                                         val_batch[label_key + '_only'].size(3),
                                                         val_batch[label_key + '_only'].size(4)).zero_()
                        val_batch[label_key + '_only'] = one_hot.scatter_(1, val_batch[label_key + '_only'].data, 1)
                        val_batch[img_key] = val_batch[img_key].cuda()

                        if method == 'joint_train' or method == 'sep_joint_train':
                            # val_batch = model(val_batch, img_key, label_key + '_pred', label_key + '_recon_pred')
                            val_batch, maeloss = model(val_batch, img_key, label_key + '_pred',
                                                       label_key + '_pred_recon', label_key + '_mae_mask',
                                                       unpatch=True, mask_ratio=0.75, method="ratio")

                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key + '_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key + '_only']).cpu().detach().numpy())
                            if save_more_reference and val_idx == epoch % len(val_loader):
                                mae_mask = val_batch[label_key + '_mae_mask'].detach()
                                mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.MAE.patch_size[0] ** 3 * 2)
                                mae_mask = model.MAE.unpatchify(mae_mask)

                                h = val_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                                val_batch[label_key + '_display'] = torch.cat(
                                    (val_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                     val_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                     val_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2], \
                                     mae_mask[:1, :1, :, :, h // 2],
                                     val_batch[label_key + '_pred_recon'][0:1, 1:2, :, :, h // 2]), dim=0)
                                display_image.update({label_key + '_display_val': val_batch[label_key + '_display']})

                        elif method == 'embed_train' or method == 'refine_vae':
                            val_batch = model(val_batch, img_key, label_key + '_pred', test_mode=True)
                        else:
                            if method == "domain_adaptation":
                                val_batch = model(val_batch, img_key, label_key + '_pred', method="domain_adaptation")
                            elif method == "domain_adaptation_vit":
                                val_batch[label_key + '_pred'] = model(val_batch[img_key])
                                val_batch[label_key + '_pred'] = soft(val_batch[label_key + '_pred'])
                            # if save_eval_result and epoch % 10 == 0:
                            #     # filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.npy')
                            #     # np.save(filename, binarize(val_batch[label_key+'_pred']).detach().cpu().numpy())
                            #     filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt.npy')
                            #     np.save(filename, binarize(val_batch[label_key+'_only']).detach().cpu().numpy())
                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key + '_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key + '_only']).cpu().detach().numpy())
                                if load_prefix_vae is not None:
                                    val_batch[label_key + '_only_recon'], _, _ = ref_model_parallel(
                                        val_batch[label_key + '_only'], if_random=False, scale=0)
                                    filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt_recon')
                                    np.save(filename,
                                            binarize(val_batch[label_key + '_only_recon']).cpu().detach().numpy())
                            if save_more_reference and val_idx == epoch % len(
                                    val_loader) and load_prefix_mae is not None:
                                maeloss, val_batch[label_key + '_pred_recon'], val_batch[
                                    label_key + '_mae_mask'] = ref_model_parallel(
                                    val_batch[label_key + '_pred'], unpatch=True, mask_ratio=0.75,
                                    method="ratio")  # 0.2
                                mae_mask = val_batch[label_key + '_mae_mask'].detach()
                                mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, mae_model.patch_size[0] ** 3 * 2)
                                mae_mask = mae_model.unpatchify(mae_mask)

                                h = val_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                                val_batch[label_key + '_display'] = torch.cat(
                                    (val_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                     val_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                     val_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2], \
                                     mae_mask[:1, :1, :, :, h // 2],
                                     val_batch[label_key + '_pred_recon'][0:1, 1:2, :, :, h // 2]), dim=0)

                                display_image.update({label_key + '_display_val': val_batch[label_key + '_display']})

                        score[val_idx] = avg_dsc(val_batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_only', binary=True, botindex=1,
                                                 topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]
                        if val_list == "AMOS_val_mr" or val_list == "AMOS_test_ctmr":
                            list_key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                        22,
                                        23, 24, 31, 32, 37, 38, 41, 51, 72]
                            if val_idx in list_key:
                                score_mr[val_idx] = score[val_idx]
                                dsc_pancreas_mr += score_mr[val_idx]

                        analysis_figure_name = prefix
                        if analysis_figure_name != None:
                            # label_key+'_only' ->label ,  label_key+'_only_recon' -> label经过VAE得到的重建图
                            # label_key+'_pred' ->预测的分割图 ，label_key+'_recon_pred' ->预测分割图的重建图
                            # label_key+'_only_fake'->pesudo label ，label_key+'_only_fake_recon' -> pesudo通过VAE的重建图
                            # update seg recon
                            maeloss, val_batch[label_key + '_pred_recon'], val_batch[
                                label_key + '_mae_mask'] = ref_model_parallel(
                                val_batch[label_key + '_pred'], unpatch=True, mask_ratio=0.75, method="ratio")  # 0.2
                            # label recon
                            _, val_batch[label_key + '_only_recon'], _ = ref_model_parallel(
                                val_batch[label_key + '_only'], unpatch=True, mask_ratio=0.75, method="ratio")  # 0.2
                            # pseudo recon


                            if method == "domain_adaptation":

                                val_batch = fix_segmodel(val_batch, img_key, label_key + '_only_fake',method="domain_adaptation")
                            elif method == "domain_adaptation_vit":
                                val_batch[label_key + '_only_fake'] = fix_segmodel(val_batch[img_key])
                                val_batch[label_key + '_only_fake'] = soft(val_batch[label_key + '_only_fake'])

                            _, val_batch[label_key + '_only_fake_recon'], _ = ref_model_parallel(
                                val_batch[label_key + '_only_fake'], unpatch=True, mask_ratio=0.75,
                                method="ratio")  # 0.2

                            gt_recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_only_recon',
                                                        target_key=label_key + '_only',
                                                        binary=True, botindex=1, topindex=len(mask_index)).item()
                            gt_dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
                                                           target_key=label_key + '_only', botindex=1,
                                                           topindex=len(mask_index),
                                                           return_mean=True).item()

                            recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_pred',
                                                     target_key=label_key + '_pred_recon',
                                                     binary=True, botindex=1, topindex=len(mask_index)).item()
                            dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_pred',
                                                        target_key=label_key + '_only_fake',
                                                        botindex=1, topindex=len(mask_index), return_mean=True).item()

                            pseudo_recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
                                                            target_key=label_key + '_only_fake_recon', botindex=1,
                                                            topindex=len(mask_index), return_mean=True).item()
                            pseudo_dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
                                                               target_key=label_key + '_only_fake', botindex=1,
                                                               topindex=len(mask_index), return_mean=True).item()
                            print(f"result: {val_idx}")
                            print(1 - score[val_idx])
                            print(gt_recon_loss)
                            print(recon_loss)
                            print(pseudo_recon_loss)
                            print(dsc_loss_fake)
                            print(gt_dsc_loss_fake)
                            print(pseudo_dsc_loss_fake)
                            loss_gt += gt_recon_loss
                            loss_recon += recon_loss
                            loss_fake += dsc_loss_fake
                            # 预测的分割图和预测分割图通过VAE重建后的重建损失 以及 预测的分割图和pseudo图的pseudo损失
                            score_figure[val_idx] = [dsc_loss_fake, recon_loss]
                            # pseudo图和label的dice    以及  label和 label经过VAE后的重建图计算dice
                            score_figure_gt[val_idx] = [gt_dsc_loss_fake, gt_recon_loss]
                            # pseudo和pseudo经过VAE后的重建图的dice 以及pesudo 和pesudo的dice
                            score_figure_pseudo[val_idx] = [pseudo_dsc_loss_fake, pseudo_recon_loss]
                    if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR":
                        dsc_pancreas_ct = (dsc_pancreas - dsc_pancreas_mr) / (val_idx + 1 - len(list_key))
                        dsc_pancreas_mr /= len(list_key)
                    dsc_pancreas /= (val_idx + 1)
                    if analysis_figure_name != None:
                        loss_gt /= (val_idx + 1)
                        loss_recon /= (val_idx + 1)
                        loss_fake /= (val_idx + 1)
                        print("gt_recon_loss")
                        print(loss_gt)
                        print("recon_loss")
                        print(loss_recon)
                        print("fake_loss")
                        print(loss_fake)
                        scatter_plot(score_figure, epoch, analysis_figure_name, "ours", "Pseudo_loss", "Recon_loss")
                        scatter_plot(score_figure_gt, epoch, analysis_figure_name, "GT", "Pseudo_loss", "Recon_loss")
                        scatter_plot(score_figure_pseudo, epoch, analysis_figure_name, "Pseudo", "Pseudo_loss",
                                     "Recon_loss")
                        scatter_plot_multi(score_figure, score_figure_gt, epoch, analysis_figure_name, 'analysis_MAE')




            output_score = os.path.join(display_path, f"score_{epoch}.json")
            if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR" or val_list == "AMOS_test_ctmr":
                output_score_mr = os.path.join(display_path, f"score_mr_{epoch}.json")
                with open(output_score_mr, "w") as f:
                    json.dump(score_mr, f, indent=4)
            with open(output_score, "w") as f:
                json.dump(score, f, indent=4)

            loss = []
            if method != "maecyc_train_input" and method != "mae_train_input":
                loss.append(['val_result', dsc_pancreas])
                if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR" or val_list == "AMOS_test_ctmr":
                    loss.append(['mr_val_result', dsc_pancreas_mr])
                    loss.append(['ct_val_result', dsc_pancreas_ct])
                    if dsc_pancreas_ct > best_result_ct:
                        best_result_ct = dsc_pancreas_ct
                        torch.save({
                            'epoch': (epoch + 1) * eval_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "best_result": best_result
                        }, os.path.join(save_path, 'bestct_model.ckpt'))

            else:
                pass

            # loss.append(['val_visiable_result', dsc_pancreas_visiable])
            saver.write_display((epoch + 1) * (max_idx_in_epoch + 1), loss, display_image, force_write=True)

            if dsc_pancreas > best_result:
                best_result = dsc_pancreas
                torch.save({
                    'epoch': (epoch + 1) * eval_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "best_result": best_result
                }, os.path.join(save_path, 'best_model.ckpt'))
                best_model = model
                # scatter_plot(score_figure, 999, analysis_figure_name, "best_ours", "Pseudo_loss", "Recon_loss")
                # scatter_plot(score_figure_gt, 999, analysis_figure_name, "GT", "Pseudo_loss", "Recon_loss")
                # scatter_plot(score_figure_pseudo, 999, analysis_figure_name, "Pseudo", "Pseudo_loss", "Recon_loss")
                # scatter_plot_multi(score_figure, score_figure_gt, 999, analysis_figure_name, 'analysis_MAE')
            if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR" or val_list == "AMOS_test_ctmr":
                print('epoch %d validation result: %f mr result:%f ct result:%f, best result %f.' % (
                    epoch + 1, dsc_pancreas, dsc_pancreas_mr, dsc_pancreas_ct, best_result))
            else:
                print('epoch %d validation result: %f, best result %f.' % (epoch + 1, dsc_pancreas, best_result))

            if test_only: break
            model.train()
            # if method == 'joint_train' or method == 'sep_joint_train' or method == 'domain_adaptation':
            #     model.Vae.eval()

            ## save model
        if (epoch + 1) % (save_epoch // eval_epoch) == 0:
            print('saving model')
            torch.save({
                'epoch': (epoch + 1) * eval_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_result": best_result
            }, os.path.join(save_path, 'model_epoch' + str((epoch + 1) * eval_epoch) + '{}.ckpt'.format(best_result)))

            '''
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': generator_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'generator_model_epoch'+str(epoch+1)+'.ckpt'))
            '''
        print('Finished Training')

