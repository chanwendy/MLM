import shutil
import time
from functools import partial
import ipdb
# from monai.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.utils import MySpatialTransform
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
import torch
import torch.nn as nn
import torchvision
import argparse
import random
from tensorboardX import SummaryWriter
from monai.losses import DiceCELoss
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
from utils.evaluation import binarize
import random
from utils.saver import Saver
import argparse
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("prefix", help="prefix")
# parser.add_argument("-prefix", help="prefix", default="test")
# parser.add_argument('--model_name', default="featurefusion", type=str)
parser.add_argument("-P", "--target_phase", help="target_phase", default='arterial')
parser.add_argument("-G", "--GPU", help="GPU", default='0')
parser.add_argument("-b", "--batch_size", type=int, help="batch_size", default=4)
parser.add_argument("-E", "--max_epoch", type=int, help="max_epoch", default=1600)
parser.add_argument("--warmup_epochs", type=int, help="max_epoch", default=50)
parser.add_argument("--save_epoch", type=int, help="save_epoch", default=50)
parser.add_argument("--eval_epoch", type=int, help="eval_epoch", default=50)
parser.add_argument("--turn_epoch", type=int, help="turn_epoch", default=-1)
parser.add_argument("-S", "--softrelu", type=int, help="softrelu", default=0)
parser.add_argument("-M", "--method", help="method", default='mae_train')
# data-root 传入的是multi_all.json文件中对应数据集的路径 如NIH： 每一个病人的data在/data1/2021_stu/MING/NIH/pancreas_xxxx 中， 则data root为：/data1/2021_stu/MING/NIH
parser.add_argument("-R", "--data_root", help="data_root", default='/data/MING/data/synapse/aorta_data')
parser.add_argument("-V", "--val_data_root", help="val_data_root", default='/data/MING/data/synapse/aorta_data')
parser.add_argument("-l", "--data_path", help="data_path", default='/data/MING/code/VAE_06/data/My_Multi_all.json')
parser.add_argument("-t", "--train_list", help="train_list", default='SYN_maetrain')
parser.add_argument("-v", "--val_list", help="val_list", default='SYN_maetest')
parser.add_argument("--load_prefix", help="load_prefix", default=None)
parser.add_argument("--gan_mode", help="gan_mode", default='cyc')
parser.add_argument("--checkpoint_name", help="checkpoint_name", default="best_model.ckpt")
parser.add_argument("--load_prefix_vae", help="load_prefix_vae", default=None)
parser.add_argument("--load_prefix_mae", help="load_prefix_mae", default=None)
parser.add_argument("--load_prefix_joint", help="load_prefix_joint", default=None)
parser.add_argument("--pan_index", help="pan_index", default='1')
parser.add_argument("--lambda_vae", type=float, help="lambda_vae", default=0.1)
parser.add_argument("--lambda_vae_warmup", type=int, help="save_epoch", default=0)
parser.add_argument("--lr_seg", type=float, help="lr_seg", default=1e-2)  # for seg 1e-1
parser.add_argument("--lr_vae", type=float, help="lr_vae", default=0)  # for vae 1e-1
parser.add_argument("--lr_mae", type=float, help="lr_mae", default=2.5e-4)  # for vae 1e-1
parser.add_argument("--test_only", help="test_only", action='store_true')
parser.add_argument("--resume", help="resume", action='store_true')
parser.add_argument("--save_more_reference", help="save_more_reference", action='store_true')
parser.add_argument("--save_eval_result", help="save_more_reference", action='store_true')
parser.add_argument("--no_aug", help="no_aug", action='store_true')
parser.add_argument("--adam", help="no_aug", action='store_true')
parser.add_argument("--mode", help="mode", type=int, default=0)
args = parser.parse_args()

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
num_workers = 4
gan_mode = args.gan_mode
trainbatch = args.batch_size
valbatch = 1
load_prefix = args.load_prefix
checkpoint_name = args.checkpoint_name
load_prefix_vae = args.load_prefix_vae
load_prefix_mae = args.load_prefix_mae
load_prefix_joint = args.load_prefix_joint
load_epoch_seg = 240
load_epoch = 60
prefix = args.prefix
data_path = os.path.join('lists', args.data_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
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


# img_size = 192
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

import cv2
set_threadnumbers = 2
cv2.setNumThreads(set_threadnumbers)
torch.set_num_threads(set_threadnumbers)
os.environ["OMP_NUM_THREADS"] = str(set_threadnumbers)  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = str(set_threadnumbers)  # 设置MKL-DNN CPU加速库的线程数。

if __name__ == "__main__":
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
    transforms['train'].append(CropResize(fields=input_phases, output_size=patch_size))
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
    transforms['val'].append(CropResize(fields=input_phases, output_size=patch_size))
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
    if method != "domain_adaptation":
        train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=False, num_workers=num_workers,
                                  drop_last=True, pin_memory=True)
        print("domain!")
    val_loader = DataLoader(val_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)
    if save_more_reference:
        train_loader_2 = DataLoader(train_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers,
                                    pin_memory=True)

    ## model build and load
    print("Building model.")
    models = importlib.import_module('joint_model')
    # vm_model = importlib.import_module('models.' + 'voxelmorph3D_joint')
    if method == 'vae_train':
        model = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128,
                           soft=softrelu == 1)
    elif method == 'mae_train':
        # todo
        # model = models.Mae3D(img_size=img_size, patch_size=patch, in_chans=2,
        #                      embed_dim=768, depth=12, num_heads=12,
        #                      decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
        #                      mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # mask pancreas with a little of patch
        # model = models.Mae3D2Mask(img_size=img_size, patch_size=patch, in_chans=2,
        #                      embed_dim=768, depth=12, num_heads=12,
        #                      decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
        #                      mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # mask ratio to mask pancreas

        # model = models.Mae3DMaskAL(img_size=img_size, patch_size=patch, in_chans=2,
        #                      embed_dim=768, depth=12, num_heads=12,
        #                      decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
        #                      mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model = models.Mae3D2MaskSeg(img_size=img_size, patch_size=patch, in_chans=2,
                                         embed_dim=768, depth=12, num_heads=12,
                                         decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
                                         mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # mask pancreas huge
        # model = models.Mae3DMask(img_size=img_size, patch_size=patch, in_chans=2,
        #                      embed_dim=1152, depth=12, num_heads=12,
        #                      decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=6,
        #                      mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # vidoe sincos
        # model = models.Mae3Dsin(img_size=img_size, patch_size=patch, in_chans=2,
        #                      embed_dim=768, depth=12, num_heads=12,
        #                      decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
        #                      mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # model = models.Mae3D(img_size=144, patch_size=12, in_chans=2,
        #          embed_dim=3456, depth=4, num_heads=6,
        #          decoder_embed_dim=1536, decoder_depth=2, decoder_num_heads=12,
        #          mlp_dim=13824, norm_layer=nn.LayerNorm)

    elif method == 'seg_train':
        model = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
        model_ref = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128)
    elif method == 'unter_seg_train':
        model = UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(img_size, img_size, img_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    elif method == 'swinunter_seg_train':
        model = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        )

    elif method == 'joint_train' or method == "domain_adaptation":
        model = []
        model.append(models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128))
        model = models.Joint(models=model)
    elif method == 'embed_train' or method == 'refine_vae':
        model = []
        model.append(models.Encoder(n_channels=1, dim=128, norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128))
        model.append(
            models.Fusion(n_channels_img=1, n_channels_mask=len(mask_index), n_class=len(mask_index), norm_type=1))
        model = models.Embed(models=model)
    elif method == 'sep_joint_train':
        model = []
        model.append(models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1))
        model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128))
        model = models.Joint(models=model)
        tea_model = []
        tea_model.append(models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1))
        tea_model.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128))
        tea_model = models.Joint(models=tea_model)
    else:
        raise ValueError("Try a valid method.")
    model = model.cuda()

    if method != 'joint_train':
        # VAE
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=lr1, weight_decay=weight_decay, momentum=0.9)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=lr3, weight_decay=weight_decay)
    else:
        '''
        optimizer = torch.optim.SGD([{'params':model.Seg.down3.parameters(),'lr':lr1,'model':'Seg'},
                                    {'params':model.Seg.down4.parameters(),'lr':lr1,'model':'Seg'},
                                        {'params':model.Vae.parameters(),'lr':lr2,'model':'Vae'} ],
                                    weight_decay = weight_decay)

        '''
        if not adam:
            optimizer = torch.optim.SGD([{'params': model.Seg.parameters(), 'lr': lr1, 'model': 'Seg'},
                                         {'params': model.Vae.parameters(), 'lr': lr2, 'model': 'Vae'}],
                                        weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.Adam([{'params': model.Seg.parameters(), 'lr': lr1, 'model': 'Seg'},
                                          {'params': model.Vae.parameters(), 'lr': lr2, 'model': 'Vae'}],
                                         betas=(0.9, 0.999), weight_decay=weight_decay)
    # scheduler = LinearWarmupCosineAnnealingLR(
    #     optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epoch
    # )

    print("Loading prefix.")
    if load_prefix:
        register_model_path = os.path.join(save_root_path, load_prefix, checkpoint_name)
        # register_model_path = "/data/MING/data/NIH/model/swinunter_seg_nih_word128_eval1_forlimited_data_delete_valid_e100/model_itreation36000.ckpt"
        if method == "seg_train" or method == "unter_seg_train" or method == "swinunter_seg_train":
            try:
                model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            except Exception as e:
                new_state_dict = {}
                checkpoint = torch.load(register_model_path)['model_state_dict']
                for k, v in checkpoint.items():
                    new_state_dict[k[7:]] = v
                model.load_state_dict(new_state_dict)

        else:
            model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])

    if load_prefix_vae:
        register_model_path = save_root_path + '/' + load_prefix_vae + '/best_model.ckpt'
        if method == "seg_train":
            model_ref.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            ref_model_parallel = nn.DataParallel(model_ref).cuda()
            for param in ref_model_parallel.parameters():
                param.requires_grad = False
            model_ref.eval()
        else:
            model.Vae.load_state_dict(torch.load(register_model_path)['model_state_dict'])

    if load_prefix_mae:
        # load best
        register_model_path = save_root_path + '/' + load_prefix_mae + '/best_model.ckpt'
        # load newest
        # register_model_path = save_root_path + '/' + load_prefix_mae + '/model_epoch4800.ckpt'
        if method == "mae_train" or method == "mae_train_input" or method == "mae768_train" or method =="mae768_1ctrain" or method =="maecyc_train" or method == "maecyc_train_input" or method == "maecyc1c_train":
            model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            pre_epochs = torch.load(register_model_path)['epoch']
            # todo:modify#
            # pre_best_result = torch.load(register_model_path)['best_result']
        else:
            pass

    if load_prefix_joint:
        register_model_path = save_root_path + '/' + load_prefix_joint + '/best_model.ckpt'
        model.load_state_dict(torch.load(register_model_path)['model_state_dict'])

    if method == 'sep_joint_train':
        if load_prefix_joint:
            register_model_path = save_root_path + '/' + load_prefix_joint + '/best_model.ckpt'
            tea_model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        else:
            register_model_path = save_root_path + '/' + load_prefix + '/best_model.ckpt'
            tea_model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])
            register_model_path = save_root_path + '/' + load_prefix_vae + '/best_model.ckpt'
            tea_model.Vae.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        tea_model_parallel = nn.DataParallel(tea_model).cuda()
        for param in tea_model.parameters():
            param.requires_grad = False

    if method == 'joint_train' or method == 'sep_joint_train' or method == 'embed_train' or method == 'domain_adaptation':
        for param in model.Vae.parameters():
            param.requires_grad = False
        model.Vae.eval()
    if method == 'refine_vae':
        Encoder_list = ['in_block', 'down1', 'down2', 'down3', 'down4', 'down5', 'fc_mean', 'fc_std']
        for param_name, param in model.Vae.named_parameters():
            if param_name.split('.')[0] in Encoder_list:
                param.requires_grad = False
            else:
                param.requires_grad = True
    final_model_parallel = nn.DataParallel(model).cuda()

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
    diceloss = DiceCELoss(to_onehot_y=True, softmax=True)
    soft = nn.Softmax(dim=1)

    if pre_epochs:
        epochs = pre_epochs // eval_epoch
        # todo:modify #
        # best_result = pre_best_result
    else:
        epochs = 0
    ## training loop

    print("Start training")
    best_result_ct = 0
    pretrain_discriminator_epochs = 5
    pretrain_epoch = 0
    iteration = 0
    if "limited" in train_list :
        limited_dataset = np.load("/data/MING/data/limited_data/dataset.npy")
        total_dice_file = {}
        dice_file = {}
        for i, name in enumerate(list(limited_dataset)):
            dice_file.update({
                f'{name.split(".")[0]}': {  # 每个样本的名字，我们根据这个来挑选数据
                    'idx': i,  # 每个样本对应的id号
                    'dice': np.zeros(0, dtype='float32'),  # 同上
                    'corr': np.zeros(0, dtype='float32'),  # 同上
                    'conf': 0,  # 置信度
                    'var': 0,  # 方差
                    'mean_corr': 0,
                    'nums': 0,  # 每个样本训练的次数
                    'dataset': name.split('_')[0]}  #
            }
            )
        for i, name in enumerate(list(limited_dataset)):
            total_dice_file.update({
                f'{name.split(".")[0]}': {  # 每个样本的名字，我们根据这个来挑选数据
                    'idx': i,  # 每个样本对应的id号
                    'dice': np.zeros(0, dtype='float32'),  # 同上
                    'corr': np.zeros(0, dtype='float32'),  # 同上
                    'conf': 0,  # 置信度
                    'var': 0,  # 方差
                    'mean_corr': 0,
                    'nums': 0,  # 每个样本训练的次数
                    'dataset': name.split('_')[0]}  #
            }
            )
    for epoch in range(epochs, max_epoch // eval_epoch):
        if not test_only:
            if epoch == 0 and method == "domain_adaptation":
                if not os.path.exists(middle_path):
                    os.mkdir(middle_path)
                for idx, batch in enumerate(train_loader):
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[img_key] = batch[img_key].cuda()
                    batch = final_model_parallel(batch, img_key, label_key + '_pred', label_key + '_recon_pred')
                    filename = os.path.join(middle_path, f'{idx}_pred.pt')
                    torch.save(batch[label_key + '_pred'], filename)
                    filename = os.path.join(middle_path, f'{idx}_recon.pt')
                    torch.save(batch[label_key + '_recon_pred'], filename)

            for idx, batch in enumerate(train_loader):
                if idx > max_idx_in_epoch:
                    max_idx_in_epoch = idx
                # optimizer.param_groups[0]['lr'] = lr3/(10**(epoch//10))
                # for out_list in range(len(output_keys)):

                optimizer.zero_grad()
                # forward + backward + optimize
                if method == 'vae_train':
                    # label_key = "venous_pancreas"  img_key = "venous"
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    # (batch, 2 , L, H, W)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[label_key + '_recon'], batch['mean'], batch['std'] = final_model_parallel(
                        batch[label_key + '_only'], if_random=True, scale=0.35)  # 0.2
                    h = batch[label_key + '_only'][0:1, 0:1, :, :, :].shape[4]
                    #
                    batch[label_key + '_display'] = torch.cat((batch[label_key + '_only'][0:1, 0:1, :, :, h // 2], \
                                                               batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                                               batch[label_key + '_recon'][0:1, 1:2, :, :, h // 2]),
                                                              dim=0)
                    # batch={"venous_pancreas_only":batch["venous_pancreas"],
                    # "venous_pancreas_recon" = model output
                    # "venous_pancreas_display" = ...
                    # }
                    # batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    klloss = KLloss(batch)
                    dsc_loss = 1 - avg_dsc(batch, source_key=label_key + '_recon', target_key=label_key + '_only',
                                           botindex=1, topindex=len(mask_index))
                    final_loss = dsc_loss + 0.00002 * klloss
                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    loss.append(['kl_loss', klloss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])
                    display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)

                if method == 'mae_train':
                    # label_key = "venous_pancreas"  img_key = "venous"
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    # batch[img_key] = batch[img_key].cuda().float()
                    # (batch, 2 , L, H, W)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    maeloss, batch[label_key + '_recon'], batch[label_key + '_mae_mask'] = final_model_parallel(
                        batch[label_key + '_only'], unpatch=True, mask_ratio=0.75, method="ratio")  # 0.2
                    # maeloss, batch[label_key + '_recon'], batch[label_key + '_mae_mask'] = final_model_parallel(
                    #     batch[label_key + '_only'], unpatch=True, mask_ratio=0.98)  # 0.2
                    mae_mask = batch[label_key + '_mae_mask'].detach()
                    mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.patch_size[0]**3 * 2)  # todo
                    mae_mask = model.unpatchify(mae_mask)
                    im_masked = batch[label_key + '_only'] * (1 - mae_mask)
                    im_paste = im_masked + batch[label_key + '_recon'] * mae_mask
                    h = batch[label_key + '_only'][0:1, 0:1, :, :, :].shape[4]
                    # h = batch[img_key][0:1, 0:1, :, :, :].shape[4]
                    # ipdb.set_trace()
                    # label
                    batch[label_key + "_recon_visiable"] = im_paste
                    batch[label_key + '_display'] = torch.cat((
                        mae_mask[:1, :1, :, :, h // 2],
                        batch[label_key + '_only'][0:1, 1:, :, :, h // 2],
                        batch[label_key + '_recon'][0:1, 1:, :, :, h // 2],
                        batch[label_key + "_recon_visiable"][0:1, 1:, :, :, h // 2]), dim=0)

                    # input image
                    # batch[img_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                    #                                            batch[img_key + '_recon'][0:1, 0:1, :, :, h // 2]),
                    #                                           dim=0)

                    # batch={"venous_pancreas_only":batch["venous_pancreas"],
                    # "venous_pancreas_recon" = model output
                    # "venous_pancreas_display" = ...
                    # }
                    # batch = register_model(batch)
                    '''
                    final_loss = standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='dummy_img', mask_label_key='source_mask')*(1-beta) + \
                                (standard_loss(batch, do_mask=True, source_label_key='target', target_label_key='dummy_img', mask_label_key='source_mask') + \
                                standard_loss(batch, do_mask=True, source_label_key='align_img', target_label_key='source', mask_label_key='source_mask'))*beta/2
                    '''
                    dsc_loss = 1 - avg_dsc(batch, source_key=label_key + '_recon', target_key=label_key + '_only',
                                           botindex=1, topindex=len(mask_index))

                    final_loss = maeloss
                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    loss.append(['mae_loss', maeloss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])
                    display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)


                if method == 'seg_train' or method == 'unter_seg_train' or method == 'swinunter_seg_train':

                    if epoch == 0: continue
                    # label_key = "venous_pancreas"  img_key = "venous"
                    iteration += 1
                    if iteration % 1000 == 0:
                        torch.save({
                            'epoch': (epoch + 1) * eval_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "best_result": best_result
                        }, os.path.join(save_path, 'model_itreation' + str(iteration) + '.ckpt'))
                    batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                                                     batch[label_key + '_only'].size(2),
                                                     batch[label_key + '_only'].size(3),
                                                     batch[label_key + '_only'].size(4)).zero_()
                    batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    batch[img_key] = batch[img_key].cuda()
                    # 获取输出结果
                    if method == "seg_train":
                        batch = final_model_parallel(batch, img_key, label_key + '_pred')
                        if load_prefix_vae is not None:
                            batch[label_key + '_recon_pred'], _, _ = ref_model_parallel(batch[label_key + '_pred'],
                                                                                        if_random=False, scale=0)
                    else:
                        batch[label_key + '_pred'] = final_model_parallel(batch[img_key])
                        batch[label_key + '_pred'] = soft(batch[label_key + '_pred'])


                    h = batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                    # 在tensorboard中显示的图像从左到右有三幅图像，分别为输入图像， 以及label，最后为网络的预测
                    batch[label_key + '_display'] = torch.cat((batch[img_key][0:1, 0:1, :, :, h // 2], \
                                                               batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                                               batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2]),
                                                              dim=0)

                    fdice = avg_dsc(batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_only', binary=True, botindex=1,
                                                 topindex=len(mask_index)).item()

                    dsc_loss = 1 - abs(avg_dsc(batch, source_key=label_key + '_pred', target_key=label_key + '_only',
                                       botindex=1, topindex=len(mask_index)))

                    final_loss = dsc_loss
                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])
                    if load_prefix_vae is not None:
                        recon_loss = 1 - avg_dsc(batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_recon_pred', botindex=1,
                                                 topindex=len(mask_index), return_mean=True)
                        loss.append(['recon_loss', recon_loss.item()])
                    display_image.update({label_key + '_display': batch[label_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)

                    # ipdb.set_trace()
                    if "limited" in train_list :
                        dice_file[batch["name"][0]]['dice'] = np.append(dice_file[batch["name"][0]]['dice'],
                                                                 np.asarray(fdice, dtype='float32'))
                        total_dice_file[batch["name"][0]]['dice'] = np.append(total_dice_file[batch["name"][0]]['dice'],
                                                                       np.asarray(fdice, dtype='float32'))
                        dice_file[batch["name"][0]]['nums'] += 1
                        total_dice_file[batch["name"][0]]['nums'] += 1


                if method == 'mae_train_input':
                    # label_key = "venous_pancreas"  img_key = "venous"
                    # batch[label_key + '_only'] = batch[label_key].type(torch.cuda.LongTensor)
                    batch[img_key] = batch[img_key].cuda().float()
                    # # (batch, 2 , L, H, W)
                    # one_hot = torch.cuda.FloatTensor(batch[label_key + '_only'].size(0), len(mask_index),
                    #                                  batch[label_key + '_only'].size(2),
                    #                                  batch[label_key + '_only'].size(3),
                    #                                  batch[label_key + '_only'].size(4)).zero_()
                    # batch[label_key + '_only'] = one_hot.scatter_(1, batch[label_key + '_only'].data, 1)
                    maeloss, batch[img_key + '_recon'], batch[img_key + '_mae_mask'] = final_model_parallel(
                        batch[img_key], unpatch=True, mask_ratio=0.75)  # 0.2
                    mae_mask = batch[img_key + '_mae_mask'].detach()
                    mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.patch_size[0] ** 3 * 1)
                    mae_mask = model.unpatchify(mae_mask)
                    im_masked = batch[img_key] * (1 - mae_mask)
                    im_paste = im_masked + batch[img_key + '_recon'] * mae_mask
                    # h = batch[label_key + '_only'][0:1, 0:1, :, :, :].shape[4]
                    h = batch[img_key][0:1, 0:1, :, :, :].shape[4]
                    # ipdb.set_trace()
                    # input image
                    batch[img_key + '_display'] = torch.cat((mae_mask[:1, :1, :, :, h // 2],
                                                             batch[img_key][0:1, 0:1, :, :, h // 2],
                                                             batch[img_key + '_recon'][0:1, 0:1, :, :, h // 2],
                                                             im_paste[0:1, 0:1, :, :, h // 2]), dim=0)
                    dsc_loss = 1 - avg_dsc(batch, source_key=img_key + '_recon', target_key=img_key,
                                           botindex=1, topindex=len(mask_index))
                    final_loss = maeloss
                    loss = []
                    display_image = {}
                    loss.append(['dice_loss', dsc_loss.item()])
                    loss.append(['mae_loss', maeloss.item()])
                    loss.append(['lr', optimizer.param_groups[0]['lr']])
                    display_image.update({img_key + '_display': batch[img_key + '_display']})
                    saver.write_display(idx + epoch * (max_idx_in_epoch + 1), loss, display_image)



                if method != "maecyc_train_input" and  method != "maecyc_train" and method != "maecyc1c_train":
                    final_loss.backward()
                    optimizer.step()
                    # scheduler.step()
                else:
                    pass
                # print statistics
                if method == 'vae_train':
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), klloss.item()))
                if method == 'mae_train':
                    print('[%3d, %3d] loss: %.4f, %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item(), maeloss.item()))

                if method == 'seg_train' or method == 'unter_seg_train' or method == 'swinunter_seg_train':
                    print('[%3d, %3d] loss: %.4f' %
                          ((epoch + 1) * eval_epoch, idx + 1, dsc_loss.item()))
            end = time.time()
            print("all spend time {}".format(end - start))

        if "limited" in train_list :
            if epoch % 10 == 0:
                dice_file_root = os.path.join("/data/MING/data/NIH/result", prefix)
                if not os.path.exists(dice_file_root):
                    os.mkdir(dice_file_root)
                dice_root = os.path.join(dice_file_root, "dice_files")
                if not os.path.exists(dice_root):
                    os.mkdir(dice_root)
                np.save( dice_root + f'/e{epoch}_patch_dice_train{train_list}.npy', dice_file,)
                dice_file = {}
                for i, name in enumerate(list(limited_dataset)):
                    dice_file.update({
                        f'{name.split(".")[0]}': {  # 每个样本的名字，我们根据这个来挑选数据
                            'idx': i,  # 每个样本对应的id号
                            'dice': np.zeros(0, dtype='float32'),  # 同上
                            'corr': np.zeros(0, dtype='float32'),  # 同上
                            'conf': 0,  # 置信度
                            'var': 0,  # 方差
                            'mean_corr': 0,
                            'nums': 0,  # 每个样本训练的次数
                            'dataset': name.split('_')[0]}  #
                    }
                    )


        print("Ready validation")
        # epoch 4 weird
        # validation
        if (epoch + 1) % 1 == 0 or test_only:
            if "limited" in train_list :
                dice_file_root = os.path.join("/data/MING/data/NIH/result", prefix)
                if not os.path.exists(dice_file_root):
                    os.mkdir(dice_file_root)
                dice_root = os.path.join(dice_file_root, "total_dice_files")
                if not os.path.exists(dice_root):
                    os.mkdir(dice_root)
                np.save(dice_root + f'/e{epoch}_patch_dice_train{train_list}.npy', total_dice_file, )
            print("Start evaluation")
            model.eval()
            score = {}
            score_visiable = {}
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

            if method == 'mae_train':
                dsc_pancreas = 0.0
                display_image = {}
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        val_batch[label_key + '_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
                        one_hot = torch.cuda.FloatTensor(val_batch[label_key + '_only'].size(0), len(mask_index),
                                                         val_batch[label_key + '_only'].size(2),
                                                         val_batch[label_key + '_only'].size(3),
                                                         val_batch[label_key + '_only'].size(4)).zero_()
                        val_batch[label_key + '_only'] = one_hot.scatter_(1, val_batch[label_key + '_only'].data, 1)
                        # val_batch[label_key + '_recon'], _, _ = model(val_batch[label_key + '_only'], if_random=False)
                        # maeloss, val_batch[label_key + '_recon'], val_batch[label_key + '_mae_mask'] = final_model_parallel(
                        #     val_batch[label_key + '_only'][:, :1, ...], unpatch=True, mask_ratio=0.9)  # 0.2
                        maeloss, val_batch[label_key + '_recon'], val_batch[label_key + '_mae_mask'] = final_model_parallel(
                            val_batch[label_key + '_only'], unpatch=True, mask_ratio=0.75,method="ratio")  # 0.2

                        if save_more_reference and val_idx == epoch % len(val_loader):
                            mae_mask = val_batch[label_key + '_mae_mask'].detach()
                            mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.patch_size[0] ** 3 * 2)
                            mae_mask = model.unpatchify(mae_mask)
                            im_masked = val_batch[label_key + '_only'] * (1 - mae_mask)
                            im_paste = im_masked + val_batch[label_key + '_recon'] * mae_mask
                            h = val_batch[label_key + '_only'][0:1, 0:1, :, :, :].shape[4]
                            val_batch[label_key + "_recon_visiable"] = im_paste
                            val_batch[label_key + '_display'] = torch.cat((
                                mae_mask[:1, :1, :, :, h // 2],
                                val_batch[label_key + '_only'][0:1, 1:, :, :, h // 2],
                                val_batch[label_key + '_recon'][0:1, 1:, :, :, h // 2],
                                val_batch[label_key + "_recon_visiable"][0:1, 1:, :, :, h // 2]), dim=0)

                            display_image.update({label_key + '_display_val': val_batch[label_key + '_display']})

                        score[val_idx] = avg_dsc(val_batch, source_key=label_key + '_recon',
                                                 target_key=label_key + '_only', binary=True, botindex=1,
                                                 topindex=len(mask_index)).item()
                        dsc_pancreas += score[val_idx]
                        # if val_idx >= 0:
                        #     mae_mask = val_batch[label_key + '_mae_mask'].detach()
                        #     mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, model.patch_size[0] ** 3 * 2)
                        #     mae_mask = model.unpatchify(mae_mask)
                        #     im_masked = val_batch[label_key + '_only'] * (1 - mae_mask)
                        #     im_paste = im_masked + val_batch[label_key + '_recon'] * mae_mask
                        #     z = val_batch[label_key + '_recon'].argmax(1).unsqueeze(1)
                        #     h = 100
                        #     # val_batch[label_key + "_show"] = logit_map.argmax(1).unsqueeze(1)
                        #     val_batch[label_key + '_display'] = torch.cat((mae_mask[0:1, 0:1, :, :, h // 2],
                        #                                                    val_batch[label_key + '_only'][0:1, 0:1, :,
                        #                                                    :, h // 2],
                        #                                                    val_batch[label_key + "_recon"][0:1, 0:1, :,
                        #                                                    :, h // 2], im_paste[:1, :1, :, :, h // 2],
                        #                                                    z[:1, :1, :, :, h // 2]), dim=0)
                        #     arrdisplay = np.array(val_batch[label_key + '_display'].detach().cpu())
                        #     savename = path.split(val_data_list[val_idx])[0]
                        #     save_root = "/HDD_data/MING/VAE/NIH/model/mae_nih_e4d2_mask0.9_lr1.6e-3/figure"
                        #     for i in range(5):
                        #         arrshow = arrdisplay[i, 0]
                        #         plt.imshow(arrshow, cmap="gray")
                        #         plt.savefig(save_root + "/{}_{}.png".format(savename, i))
                        #         plt.show()
                        #         plt.close()
                    dsc_pancreas /= (val_idx + 1)
                    # x = np.array(val_batch["venous_pancreas_recon"].detach().cpu())
                    # y = x[:1,:1,...,50].reshape(144,144)
                    # z = x.argmax(1)


            if method == 'seg_train' or method == 'unter_seg_train' or method == 'swinunter_seg_train' or method == 'joint_train' or method == 'domain_adaptation' or method == 'sep_joint_train' or method == 'embed_train' or method == 'refine_vae':
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
                        if method == 'joint_train' or method == 'domain_adaptation' or method == 'sep_joint_train':
                            val_batch = model(val_batch, img_key, label_key + '_pred', label_key + '_recon_pred')

                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key + '_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key + '_only']).cpu().detach().numpy())
                            if save_more_reference and val_idx == epoch % len(val_loader):
                                h = val_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                                # if method == 'domain_adaptation':
                                #     filename = os.path.join(middle_path, f'{val_idx}_pred.npy')
                                #     with open(filename, 'rb') as f:
                                #         dat = np.load(f)
                                #     val_batch[label_key+'_display']= torch.cat((val_batch[label_key+'_recon_pred'][0:1,1:2,:,:,h//2], \
                                #         val_batch[label_key+'_only'][0:1,1:2,:,:,h//2],val_batch[label_key+'_pred'][0:1,1:2,:,:,h//2], \
                                #         dat), dim=0)
                                # else:
                                val_batch[label_key + '_display'] = torch.cat(
                                    (val_batch[label_key + '_recon_pred'][0:1, 1:2, :, :, h // 2], \
                                     val_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                     val_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2]), dim=0)
                                display_image.update({label_key + '_display_val': val_batch[label_key + '_display']})
                        elif method == 'embed_train' or method == 'refine_vae':
                            val_batch = model(val_batch, img_key, label_key + '_pred', test_mode=True)
                        else:
                            if method == 'seg_train':
                                val_batch = model(val_batch, img_key, label_key + '_pred')
                            else:
                                val_batch[label_key + '_pred'] = model(val_batch[img_key])
                                val_batch[label_key + '_pred'] = soft(val_batch[label_key + '_pred'])

                            if save_eval_result and epoch % 10 == 0:
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pred.join')
                                np.save(filename, binarize(val_batch[label_key + '_pred']).cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_pic')
                                np.save(filename, val_batch[img_key].cpu().detach().numpy())
                                filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt')
                                np.save(filename, binarize(val_batch[label_key + '_only']).cpu().detach().numpy())
                                if load_prefix_vae is not None:
                                    val_batch[label_key + '_only_recon'], _, _ = model_ref(
                                        val_batch[label_key + '_only'], if_random=False, scale=0)
                                    filename = os.path.join(result_path, f'{epoch}_{val_idx}_gt_recon')
                                    np.save(filename,
                                            binarize(val_batch[label_key + '_only_recon']).cpu().detach().numpy())
                            if save_more_reference and val_idx == epoch % len(
                                    val_loader) and load_prefix_vae is not None:
                                val_batch[label_key + '_recon_pred'], _, _ = model_ref(val_batch[label_key + '_pred'],
                                                                                       if_random=False, scale=0)
                                h = val_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                                val_batch[label_key + '_display'] = torch.cat(
                                    (val_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                     val_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                     val_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2], \
                                     val_batch[label_key + '_recon_pred'][0:1, 1:2, :, :, h // 2]), dim=0)

                                display_image.update({label_key + '_display_val': val_batch[label_key + '_display']})

                        score[val_idx] = avg_dsc(val_batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_only', binary=True, botindex=1,
                                                 topindex=len(mask_index)).item()

                        dsc_pancreas += score[val_idx]
                        if val_list == "AMOS_val_mr":
                            list_key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                        22,
                                        23, 24, 31, 32, 37, 38, 41, 51, 72]
                            if val_idx in list_key:
                                score_mr[val_idx] = score[val_idx]
                                dsc_pancreas_mr += score_mr[val_idx]
                    if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR":
                        dsc_pancreas_ct = (dsc_pancreas - dsc_pancreas_mr) / (val_idx + 1 - len(list_key))
                        dsc_pancreas_mr /= len(list_key)
                    dsc_pancreas /= (val_idx + 1)


                    if not test_only and save_more_reference:
                        for tr_idx, tr_batch in enumerate(train_loader_2):
                            if tr_idx != epoch % len(train_loader_2): continue
                            tr_batch[label_key + '_only'] = tr_batch[label_key].type(torch.cuda.LongTensor)
                            one_hot = torch.cuda.FloatTensor(tr_batch[label_key + '_only'].size(0), len(mask_index),
                                                             tr_batch[label_key + '_only'].size(2),
                                                             tr_batch[label_key + '_only'].size(3),
                                                             tr_batch[label_key + '_only'].size(4)).zero_()
                            tr_batch[label_key + '_only'] = one_hot.scatter_(1, tr_batch[label_key + '_only'].data, 1)
                            tr_batch[img_key] = tr_batch[img_key].cuda()
                            if method == 'joint_train' or method == 'domain_adaptation' or method == 'sep_joint_train':
                                tr_batch = model(tr_batch, img_key, label_key + '_pred', label_key + '_recon_pred')
                                h = tr_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                                tr_batch[label_key + '_display'] = torch.cat(
                                    (tr_batch[label_key + '_recon_pred'][0:1, 1:2, :, :, h // 2], \
                                     tr_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                     tr_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2]), dim=0)
                            elif method == 'embed_train' or method == 'refine_vae':
                                tr_batch = model(tr_batch, img_key, label_key + '_pred', test_mode=True)
                            else:
                                tr_batch = model(tr_batch, img_key, label_key + '_pred')
                                h = tr_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                                if load_prefix_vae != None:
                                    tr_batch[label_key + '_recon_pred'], _, _ = model_ref(tr_batch[label_key + '_pred'],
                                                                                          if_random=False, scale=0)
                                    if method == 'domain_adaptation':
                                        filename = os.path.join(middle_path, f'{tr_idx}_pred.pt')
                                        dat = torch.load(filename)
                                        tr_batch[label_key + '_display'] = torch.cat(
                                            (tr_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                             tr_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                             tr_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2], \
                                             tr_batch[label_key + '_recon_pred'][0:1, 1:2, :, :, h // 2], \
                                             dat[0:1, 1:2, :, :, h // 2]), dim=0)
                                    else:
                                        tr_batch[label_key + '_display'] = torch.cat(
                                            (tr_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                             tr_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                             tr_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2], \
                                             tr_batch[label_key + '_recon_pred'][0:1, 1:2, :, :, h // 2]), dim=0)
                                else:
                                    tr_batch[label_key + '_display'] = torch.cat(
                                        (tr_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                         tr_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                         tr_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2]), dim=0)

                            display_image.update({label_key + '_display_train': tr_batch[label_key + '_display']})

            output_score = os.path.join(display_path, f"score_{epoch}.json")
            if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR":
                output_score_mr = os.path.join(display_path, f"score_mr_{epoch}.json")
                with open(output_score_mr, "w") as f:
                    json.dump(score_mr, f, indent=4)
            with open(output_score, "w") as f:
                json.dump(score, f, indent=4)

            loss = []
            if method != "maecyc_train_input" and method != "mae_train_input":
                loss.append(['val_result', dsc_pancreas])
                if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR":
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

            if val_list == "AMOS_val_mr" or val_list == "CHAOS_val_CTMR":
                print('epoch %d validation result: %f mr result:%f ct result:%f, best result %f.' % (
                    epoch + 1, dsc_pancreas, dsc_pancreas_mr, dsc_pancreas_ct, best_result))
            else:
                print('epoch %d validation result: %f, best result %f.' % (epoch + 1, dsc_pancreas, best_result))
            if test_only: break
            model.train()
            if method == 'joint_train' or method == 'sep_joint_train' or method == 'domain_adaptation':
                model.Vae.eval()

            ## save model

        if (epoch + 1) % (save_epoch // eval_epoch) == 0:
            print('saving model')
            torch.save({
                'epoch': (epoch + 1) * eval_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_result": best_result
            }, os.path.join(save_path, 'model_epoch' + str((epoch + 1) * eval_epoch) + '.ckpt'))
            '''
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': generator_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(save_path,'generator_model_epoch'+str(epoch+1)+'.ckpt'))
            '''
        print('Finished Training')
