
import math
import shutil
import time
from functools import partial
# import ipdb
# from monai.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.draw import scatter_plot, scatter_plot_multi
from utils.utils import MySpatialTransform, vis_image
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
from monai.networks.nets import SwinUNETR
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
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from utils.utils import plot_slides, BaseDataset, NumpyLoader_Multi, NumpyLoader_Multi_merge, NiiLoader, image_resize, \
    CropResize, CopyField, ExtendSqueeze, Reshape, PadToSize, Clip, Binarize, CenterIntensities
from utils.evaluation import binarize
import random
from utils.saver import Saver
import argparse
from lr_scheduler import LinearWarmupCosineAnnealingLR
from skimage.transform import resize
import nibabel as nib
# import cc3d
from monai.metrics import DiceMetric

parser = argparse.ArgumentParser()
# parser.add_argument("prefix", help="prefix")
parser.add_argument("--prefix", help="prefix", default="ttt")
# parser.add_argument('--model_name', default="featurefusion", type=str)
parser.add_argument("-P", "--target_phase", help="target_phase", default='arterial')
parser.add_argument("-G", "--GPU", help="GPU", default='2')
parser.add_argument("-b", "--batch_size", type=int, help="batch_size", default=8)
parser.add_argument("-E", "--max_epoch", type=int, help="max_epoch", default=1600)
parser.add_argument("--warmup_epochs", type=int, help="max_epoch", default=50)
parser.add_argument("--save_epoch", type=int, help="save_epoch", default=400)
parser.add_argument("--eval_epoch", type=int, help="eval_epoch", default=20)
parser.add_argument("--turn_epoch", type=int, help="turn_epoch", default=-1)
parser.add_argument("-S", "--softrelu", type=int, help="softrelu", default=0)
parser.add_argument("-M", "--method", help="method", default='domain_adaptation')
# data-root 传入的是multi_all.json文件中对应数据集的路径 如NIH： 每一个病人的data在/data1/2021_stu/MING/NIH/pancreas_xxxx 中， 则data root为：/data1/2021_stu/MING/NIH
parser.add_argument("-R", "--data_root", help="data_root", default='/data/MING/data/WORD/data')
parser.add_argument("-V", "--val_data_root", help="val_data_root", default='/data/MING/data/WORD/data')
parser.add_argument("-l", "--data_path", help="data_path", default='/data/MING/code/VAE_06/data/My_Multi_all.json')
parser.add_argument("-t", "--train_list", help="train_list", default='WORD_train')
parser.add_argument("-v", "--val_list", help="val_list", default='WORD_val')
parser.add_argument("--load_prefix", help="load_prefix", default="segmae2_msd_batch1_loadnih_0.6maedice_adamw")
parser.add_argument("--gan_mode", help="gan_mode", default='cyc')
parser.add_argument("--checkpoint_name", help="checkpoint_name", default="best_model.ckpt")
parser.add_argument("--load_prefix_vae", help="load_prefix_vae", default=None)
parser.add_argument("--load_prefix_mae", help="load_prefix_mae", default="mae_nih_pmask0.6_img128p16_batch1_all_rotation")
# parser.add_argument("--load_prefix_mae", help="load_prefix_mae", default=None)
parser.add_argument("--load_prefix_joint", help="load_prefix_joint", default=None)
parser.add_argument("--pan_index", help="pan_index", default='1')
parser.add_argument("--val_finetune", help="val_finetune", default=0)
parser.add_argument("--lambda_vae", type=float, help="lambda_vae", default=0.1)
parser.add_argument("--lambda_vae_warmup", type=int, help="save_epoch", default=0)
parser.add_argument("--lr_seg", type=float, help="lr_seg", default=1e-2)  # for seg 1e-1
parser.add_argument("--lr_vae", type=float, help="lr_vae", default=0)  # for vae 1e-1
parser.add_argument("--lr_mae", type=float, help="lr_mae", default=1e-4)  # for vae 1e-1
parser.add_argument("--test_only", help="test_only", action='store_true')
parser.add_argument("--resume", help="resume", action='store_true')
parser.add_argument("--save_more_reference", help="save_more_reference", action='store_true')
parser.add_argument("--save_eval_result", help="save_more_reference", action='store_true')
parser.add_argument("--no_aug", help="no_aug", action='store_true')
parser.add_argument("--adam", help="no_aug", action='store_true')
parser.add_argument("--mode", help="mode", type=int, default=0)
args = parser.parse_args([])
# data_root = args.data_root
# val_data_root = args.val_data_root
# train_list = args.train_list
# val_list = args.val_list
# todo:modify data path
# AMOS pancreas
# dataset = "amos"
# origin_data_path = "/data/MING/data/AMOS/images"
# origin_label_path = "/data/MING/data/AMOS/labels"
# organ="pancreas"
# train_list = "AMOS_train_mr"
# val_list = "AMOS_val_mr"
# data_root = "/data/MING/data/AMOS/data_mr"
# val_data_root = "/data/MING/data/AMOS/data_mr"

# MSD pancreas
dataset = "msd"
origin_data_path = "/data/MING/data/MSD/Task07_Pancreas/imagesTr"
origin_label_path = "/data/MING/data/MSD/Task07_Pancreas/labelsTr"
organ="pancreas"
train_list = "MSD_train"
val_list = "MSD_val"
data_root = "/data/MING/data/MSD/Task07_Pancreas/data"
val_data_root = "/data/MING/data/MSD/Task07_Pancreas/data"

# MSD Spleen
# dataset = "msd"
# origin_data_path = "/data/MING/data/MSD/Task09_Spleen/imagesTr"
# origin_label_path = "/data/MING/data/MSD/Task09_Spleen/labelsTr"
# organ="spleen"
# train_list = "MSD_spleen_train"
# val_list = "MSD_spleen_val"
# data_root = "/data/MING/data/MSD/Task09_Spleen/data"
# val_data_root = "/data/MING/data/MSD/Task09_Spleen/data"

#WORD pancreas
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "pancreas"
# data_root = "/data/MING/data/WORD/data"
# val_data_root = "/data/MING/data/WORD/data"
# train_list = "WORD_train"
# val_list = "WORD_val"


# WORD spleen
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "spleen"
# data_root = "/data/MING/data/WORD/spleen_data"
# val_data_root = "/data/MING/data/WORD/spleen_data"
# train_list = "WORD_train"
# val_list = "WORD_val"

# Word liver
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "liver"
# data_root = "/data/MING/data/WORD/liver_data"
# val_data_root = "/data/MING/data/WORD/liver_data"
# train_list = "WORD_train"
# val_list = "WORD_val"

# Word rkidney
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "rkidney"
# data_root = "/data/MING/data/WORD/rkidney_data"
# val_data_root = "/data/MING/data/WORD/rkidney_data"
# train_list = "WORD_train"
# val_list = "WORD_val"


# Word lkidney
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "lkidney"
# data_root = "/data/MING/data/WORD/lkidney_data"
# val_data_root = "/data/MING/data/WORD/lkidney_data"
# train_list = "WORD_lkidney_train"
# val_list = "WORD_lkidney_val"

# WORD  stomach
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "stomach"
# data_root = "/data/MING/data/WORD/stomach_data"
# val_data_root = "/data/MING/data/WORD/stomach_data"
# train_list = "WORD_train"
# val_list = "WORD_val"

# WORD  gall
# dataset = "word"
# origin_data_path = "/data/MING/data/WORD/imagesVal"
# origin_label_path = "/data/MING/data/WORD/labelsVal"
# organ = "gallblader"
# data_root = "/data/MING/data/WORD/gall_data"
# val_data_root = "/data/MING/data/WORD/gall_data"
# train_list = "WORD_train"
# val_list = "WORD_val"


lr1 = args.lr_seg  # for seg 1e-1
lr2 = args.lr_vae  # for vae 1e-1
lr3 = args.lr_mae  # foe mae2.5e-4

softrelu = args.softrelu
torch.backends.cudnn.benchmark = True
weight_decay = 0.05
num_workers = 8
gan_mode = args.gan_mode
trainbatch = args.batch_size
valbatch = 1
load_prefix = args.load_prefix
val_finetune = args.val_finetune
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

import cv2, copy
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

def post_processing(vol, p=0.5):
    """
    :param vol: Segmentation map of model's prediction.
    :param p: (0~1). A proportion of max region. The region which less than max region multiply by p
     will be remove.
    :return: Denoised segmentation map.
    """
    vol_ = vol.copy()
    vol_[vol_ > 0] = 1
    vol_ = vol_.astype(np.int64)
    vol_cc = cc3d.connected_components(vol_)
    cc_sum = [(i, vol_cc[vol_cc == i].shape[0]) for i in range(vol_cc.max() + 1)]
    cc_sum.sort(key=lambda x: x[1], reverse=True)
    cc_sum.pop(0)  # remove background

    reduce_cc = [cc_sum[i][0] for i in range(1, len(cc_sum)) if cc_sum[i][1] < cc_sum[0][1] * p]
    for i in reduce_cc:
        vol[vol_cc == i] = 0

    return vol

def color_map(N=256, normalized=False):
  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)
  dtype = 'float32' if normalized else 'uint8'
  cmap = np.zeros((N, 3), dtype=dtype)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3
    cmap[i] = np.array([r, g, b])
  cmap = cmap / 255 if normalized else cmap
  return cmap
def visualize(im, vote_map,MAE_vote, VAE_vote, label, n_class=9, ratio=1.0):
    """
    im: 原图的np
    vote_map:direct test label
    MAE_vote:MAE test label
    VAE_vote: VAE test label
    label: GT
    """
    im -= im.min()
    im = (im / im.max() * 255).astype(np.uint8)
    cmap = color_map()
    im = im[..., np.newaxis]
    im = im.repeat(3, axis=-1)
    pre_vis = copy.deepcopy(im)
    MAE_vis = copy.deepcopy(im)
    VAE_vis = copy.deepcopy(im)
    for c_idx in range(1, n_class):
        im[..., 0][label == c_idx] = cmap[c_idx, 0] * ratio + im[..., 0][label == c_idx] * (1. - ratio)
        im[..., 1][label == c_idx] = cmap[c_idx, 1] * ratio + im[..., 1][label == c_idx] * (1. - ratio)
        im[..., 2][label == c_idx] = cmap[c_idx, 2] * ratio + im[..., 2][label == c_idx] * (1. - ratio)
        pre_vis[..., 0][vote_map == c_idx] = cmap[c_idx, 0] * ratio + pre_vis[..., 0][vote_map == c_idx] * (1. - ratio)
        pre_vis[..., 1][vote_map == c_idx] = cmap[c_idx, 1] * ratio + pre_vis[..., 1][vote_map == c_idx] * (1. - ratio)
        pre_vis[..., 2][vote_map == c_idx] = cmap[c_idx, 2] * ratio + pre_vis[..., 2][vote_map == c_idx] * (1. - ratio)

        MAE_vis[..., 0][MAE_vote == c_idx] = cmap[c_idx, 0] * ratio + MAE_vis[..., 0][MAE_vote == c_idx] * (1. - ratio)
        MAE_vis[..., 1][MAE_vote == c_idx] = cmap[c_idx, 1] * ratio + MAE_vis[..., 1][MAE_vote == c_idx] * (1. - ratio)
        MAE_vis[..., 2][MAE_vote == c_idx] = cmap[c_idx, 2] * ratio + MAE_vis[..., 2][MAE_vote == c_idx] * (1. - ratio)

        VAE_vis[..., 0][VAE_vote == c_idx] = cmap[c_idx, 0] * ratio + VAE_vis[..., 0][VAE_vote == c_idx] * (1. - ratio)
        VAE_vis[..., 1][VAE_vote == c_idx] = cmap[c_idx, 1] * ratio + VAE_vis[..., 1][VAE_vote == c_idx] * (1. - ratio)
        VAE_vis[..., 2][VAE_vote == c_idx] = cmap[c_idx, 2] * ratio + VAE_vis[..., 2][VAE_vote == c_idx] * (1. - ratio)

    # 1.将对应vis resize 回上一个尺寸记得（preserve_range=True,clip=False）
    # 2.赋值到上一个crop种
    # 3.再赋值到全图种
    # 4.concatenate
    vis = np.concatenate((pre_vis, im), axis=2)
    vis = np.concatenate((vis, MAE_vis), axis=2)
    vis = np.concatenate((vis, VAE_vis), axis=2)
    return vis
def visualize_complete(im, vote_map,MAE_vote, VAE_vote, label, image, crop_img1, crop_img2, crop_index1, crop_index2,nnunet_out=None, n_class=9, ratio=1.0,):
    """
    im: 原图的np
    vote_map:direct test label
    MAE_vote:MAE test label
    VAE_vote: VAE test label
    label: GT
    image: 完整的图
    crop_img1:第一次crop之后的图
    crop_img2:第二次crop之后的图
    crop_index1:第一次crop的index
    crop_index2:第二次crop的index
    """
    im -= im.min()
    im = (im / im.max() * 255).astype(np.uint8)
    cmap = color_map()
    im = im[..., np.newaxis]
    im = im.repeat(3, axis=-1)
    pre_vis = copy.deepcopy(im)
    MAE_vis = copy.deepcopy(im)
    VAE_vis = copy.deepcopy(im)

    # for complete image
    image -= image.min()
    image = (image / image.max() * 255).astype(np.uint8)

    crop_img1 -= crop_img1.min()
    crop_img1 = (crop_img1 / crop_img1.max() * 255).astype(np.uint8)

    image = image[..., np.newaxis]
    image = image.repeat(3, axis=-1)
    image_vis = copy.deepcopy(image)
    preimage_vis = copy.deepcopy(image)
    MAEimg_vis = copy.deepcopy(image)
    VAEimg_vis = copy.deepcopy(image)

    crop_img1 = crop_img1[..., np.newaxis]
    crop_img1 = crop_img1.repeat(3, axis=-1)
    crop1_vis = copy.deepcopy(crop_img1)
    precrop1_vis = copy.deepcopy(crop_img1)
    MAEcrop1_vis = copy.deepcopy(crop_img1)
    VAEcrop1_vis = copy.deepcopy(crop_img1)
    try:
        if nnunet_out.any():
            print("1")
            nnunet_vis = copy.deepcopy(im)
            nnimg_vis = copy.deepcopy(image)
            nncrop1_vis = copy.deepcopy(crop_img1)

    except Exception as e:
        pass
    for c_idx in range(1, n_class):
        im[..., 0][label == c_idx] = cmap[c_idx, 0] * ratio + im[..., 0][label == c_idx] * (1. - ratio)
        im[..., 1][label == c_idx] = cmap[c_idx, 1] * ratio + im[..., 1][label == c_idx] * (1. - ratio)
        im[..., 2][label == c_idx] = cmap[c_idx, 2] * ratio + im[..., 2][label == c_idx] * (1. - ratio)
        pre_vis[..., 0][vote_map == c_idx] = cmap[c_idx, 0] * ratio + pre_vis[..., 0][vote_map == c_idx] * (1. - ratio)
        pre_vis[..., 1][vote_map == c_idx] = cmap[c_idx, 1] * ratio + pre_vis[..., 1][vote_map == c_idx] * (1. - ratio)
        pre_vis[..., 2][vote_map == c_idx] = cmap[c_idx, 2] * ratio + pre_vis[..., 2][vote_map == c_idx] * (1. - ratio)

        MAE_vis[..., 0][MAE_vote == c_idx] = cmap[c_idx, 0] * ratio + MAE_vis[..., 0][MAE_vote == c_idx] * (1. - ratio)
        MAE_vis[..., 1][MAE_vote == c_idx] = cmap[c_idx, 1] * ratio + MAE_vis[..., 1][MAE_vote == c_idx] * (1. - ratio)
        MAE_vis[..., 2][MAE_vote == c_idx] = cmap[c_idx, 2] * ratio + MAE_vis[..., 2][MAE_vote == c_idx] * (1. - ratio)

        VAE_vis[..., 0][VAE_vote == c_idx] = cmap[c_idx, 0] * ratio + VAE_vis[..., 0][VAE_vote == c_idx] * (1. - ratio)
        VAE_vis[..., 1][VAE_vote == c_idx] = cmap[c_idx, 1] * ratio + VAE_vis[..., 1][VAE_vote == c_idx] * (1. - ratio)
        VAE_vis[..., 2][VAE_vote == c_idx] = cmap[c_idx, 2] * ratio + VAE_vis[..., 2][VAE_vote == c_idx] * (1. - ratio)
        try:
            if nnunet_out.any():
                print("2")
                nnunet_vis[..., 0][nnunet_out == c_idx] = cmap[c_idx, 0] * ratio + nnunet_vis[..., 0][nnunet_out == c_idx] * (
                            1. - ratio)
                nnunet_vis[..., 1][nnunet_out == c_idx] = cmap[c_idx, 1] * ratio + nnunet_vis[..., 1][nnunet_out == c_idx] * (
                            1. - ratio)
                nnunet_vis[..., 2][nnunet_out == c_idx] = cmap[c_idx, 2] * ratio + nnunet_vis[..., 2][nnunet_out == c_idx] * (
                            1. - ratio)

        except Exception as e:
            pass

    # 1.将对应vis resize 回上一个尺寸记得（preserve_range=True,clip=False）
    size1 = list(im.shape)
    size1[0], size1[1], size1[2] = crop_img2.shape
    size1 = tuple(size1)
    im = resize(im, size1, preserve_range=True,clip=False)
    pre_vis = resize(pre_vis, size1, preserve_range=True, clip=False)
    MAE_vis = resize(MAE_vis, size1, preserve_range=True, clip=False)
    VAE_vis = resize(VAE_vis, size1, preserve_range=True, clip=False)

    # 2.赋值到上一个crop种
    crop1_vis[crop_index2[0][0]:crop_index2[0][1], crop_index2[1][0]:crop_index2[1][1], crop_index2[2][0]:crop_index2[2][1]] = im
    precrop1_vis[crop_index2[0][0]:crop_index2[0][1], crop_index2[1][0]:crop_index2[1][1], crop_index2[2][0]:crop_index2[2][1]] = pre_vis
    MAEcrop1_vis[crop_index2[0][0]:crop_index2[0][1], crop_index2[1][0]:crop_index2[1][1], crop_index2[2][0]:crop_index2[2][1]] = MAE_vis
    VAEcrop1_vis[crop_index2[0][0]:crop_index2[0][1], crop_index2[1][0]:crop_index2[1][1], crop_index2[2][0]:crop_index2[2][1]] = VAE_vis
    # 3.再赋值到全图种
    image_vis[crop_index1[0][0]:crop_index1[0][1], crop_index1[1][0]:crop_index1[1][1], crop_index1[2][0]:crop_index1[2][1]] = crop1_vis
    preimage_vis[crop_index1[0][0]:crop_index1[0][1], crop_index1[1][0]:crop_index1[1][1], crop_index1[2][0]:crop_index1[2][1]] = precrop1_vis
    MAEimg_vis[crop_index1[0][0]:crop_index1[0][1], crop_index1[1][0]:crop_index1[1][1], crop_index1[2][0]:crop_index1[2][1]] = MAEcrop1_vis
    VAEimg_vis[crop_index1[0][0]:crop_index1[0][1], crop_index1[1][0]:crop_index1[1][1], crop_index1[2][0]:crop_index1[2][1]] = VAEcrop1_vis

    # 4.concatenate
    image_vis = np.swapaxes(image_vis, 0, 2)
    preimage_vis = np.swapaxes(preimage_vis, 0, 2)
    MAEimg_vis = np.swapaxes(MAEimg_vis, 0, 2)
    VAEimg_vis = np.swapaxes(VAEimg_vis, 0, 2)
    image_vis = np.swapaxes(image_vis, 1, 2)
    preimage_vis = np.swapaxes(preimage_vis, 1, 2)
    MAEimg_vis = np.swapaxes(MAEimg_vis, 1, 2)
    VAEimg_vis = np.swapaxes(VAEimg_vis, 1, 2)

    vis = np.concatenate((preimage_vis, image_vis), axis=2)
    vis = np.concatenate((vis, MAEimg_vis), axis=2)
    vis = np.concatenate((vis, VAEimg_vis), axis=2)
    try:
        if nnunet_out.any():
            nnunet_vis = resize(nnunet_vis, size1, preserve_range=True, clip=False)
            nncrop1_vis[crop_index2[0][0]:crop_index2[0][1], crop_index2[1][0]:crop_index2[1][1], crop_index2[2][0]:crop_index2[2][1]] = nnunet_vis
            nnimg_vis[crop_index1[0][0]:crop_index1[0][1], crop_index1[1][0]:crop_index1[1][1], crop_index1[2][0]:crop_index1[2][1]] = nncrop1_vis
            nnimg_vis = np.swapaxes(nnimg_vis, 0, 2)
            nnimg_vis = np.swapaxes(nnimg_vis, 1, 2)
            vis = np.concatenate((vis, nnimg_vis), axis=2)
            print("3")

    except Exception as e:
        pass

    return vis

import copy,cv2

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        return root,files, dirs  # 当前路径下所有非目录子文件,列表

set_threadnumbers = 2
cv2.setNumThreads(set_threadnumbers)
torch.set_num_threads(set_threadnumbers)
os.environ["OMP_NUM_THREADS"] = str(set_threadnumbers)  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = str(set_threadnumbers)  # 设置MKL-DNN CPU加速库的线程数
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
val_finetune = 0
## model build and load
print("Building model.")
models = importlib.import_module('joint_model')
# vm_model = importlib.import_module('models.' + 'voxelmorph3D_joint')
if method == 'vae_train':
    model = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128,
                       soft=softrelu == 1)
elif method == 'seg_train':
    model = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
    if load_prefix_mae:
        # model_ref = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128)
        mae_model = models.Mae3D2MaskSeg(img_size=img_size, patch_size=patch, in_chans=2,
                                         embed_dim=768, depth=12, num_heads=12,
                                         decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
                                         mlp_dim=3072, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # mae_model = mae_model.cuda()
elif method == "domain_adaptation":
    model = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
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
model = model.cuda()
# for VAE
model_VAE = []
model_VAE.append(models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1))
model_VAE.append(models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128))
model_VAE = models.Joint(models=model_VAE, vae_forward_scale=0.0, vae_decoder_dropout=0.0)
model_VAE = model_VAE.cuda()
# #AMOS
# model_path = "/data/MING/data/NIH/model/new_domain_amos_mr_dh/best_model.ckpt"
# # MSD
model_path = "/data/MING/data/NIH/model/new_domain_msd_dh_ft1/best_model.ckpt"
# # word
# # model_path = "/data/MING/data/NIH/model/domain_word_formae/best_model.ckpt"
#
# # model_path = "/data/MING/data/NIH/model/new_domain_msd_dh_ft1/best_model.ckpt"
# model_VAE.load_state_dict(torch.load(model_path)["model_state_dict"])

# todo:modifi path
# for  cnn upper
# model_VAE = models.Segmentation(n_channels=1, n_class=len(mask_index), norm_type=1)
# for vit upper
# model_VAE = SwinUNETR(
#     img_size=(img_size, img_size, img_size),
#     in_channels=1,
#     out_channels=2,
#     feature_size=48,
#     use_checkpoint=True,
# )
# model_VAE = model_VAE.cuda()
# model_path = "/data/MING/data/NIH/model/unter_seg_word_gall_upper_formae128/best_model.ckpt"
# word gall
# model_path = "/data/MING/data/NIH/model/seg_word_gall_upper_formae/best_model.ckpt"
# word spleen
# model_path = "/data/MING/data/NIH/model/seg_word_spleen_upper_formae/best_model.ckpt"
# Word liver modify
# model_path = "/data/MING/data/NIH/model/seg_word_liver_upper_formae/best_model.ckpt"
# WORD gall balder
# model_path = "/data/MING/data/NIH/model/seg_word_gall_upper_formae/best_model.ckpt"
# Word rkidney
# model_path = "/data/MING/data/NIH/model/seg_word_rkidney_upper_formae/best_model.ckpt"
# WORD lkidney
# model_path = "/data/MING/data/NIH/model/seg_word_lkidney_upper_formae/best_model.ckpt"
# msd spleen
# model_path = "/data/MING/data/NIH/model/seg_msd_spleen_upper_formae/best_model.ckpt"
# MSD liver
# model_path = "/data/MING/data/NIH/model/seg_msdliver_upper_formae/best_model.ckpt"

model_VAE.load_state_dict(torch.load(model_path)["model_state_dict"])




load_prefix = None
# 1shot
# amos pancreas
# load_prefix_seg = "domain0.75_amosmr_lr5e-4_1.0pseudo_1.0recon"
# word pancreas
# load_prefix_seg = "domain0.75_word_lr5e-4_loadpseudo_1.2pseudo_1.0recon"
# word spleen
# load_prefix_seg ="domain0.75_word_spleen_lr5e-4_1.5pseudo_1.0recon_1shot"
# Word Liver
# load_prefix_seg ="domain0.75_word_liver_lr5e-4_1.7pseudo_1.0recon_1epoch_1shot"
# word rkidney
# load_prefix_seg ="domain0.75_word_rkidney_lr7.5e-4_1.7pseudo_1.0recon_1shot"
# Word lkidney
# load_prefix_seg ="domain0.75_word_lkidney_lr5e-4_1.0pseudo_1.0recon_1shot_newval_newMAE"
# Word gall blader
# load_prefix_seg = "domain0.75_word_gall_lr7.5e-4_0.6pseudo_1.2recon_1shot"
# msd spleen
# load_prefix_seg ="domain0.75_msd_spleen_lr5e-4_1.0pseudo_1.0recon_1shot"
# msd pancreas
load_prefix_seg = "domain0.75_msd_lr5e-4_loadpseudo_0.6pseudo_1.2recon"

# swinunter gall
# load_prefix_seg = "domain0.75_word_gall_lr5e-4_1.2pseudo_1.0recon_1epoch_1shot_vit"
# swinunter spleen
# load_prefix_seg = "domain0.75_word_spleen_lr7.5e-4_1.7pseudo_1.0recon_1epoch_1shot_vit"
# swinunter liver
# load_prefix_seg = "domain0.75_word_liver_lr5e-4_1.5pseudo_1.0recon_1epoch_1shot_vit"

# 5shot
# word liver
# load_prefix_seg ="domain0.75_word_liver_lr5e-4_1.0pseudo_1.0recon_5shot_1epoch"
# 4shot
# load_prefix_seg ="domain0.75_word_liver_lr5e-4_1.7pseudo_1.0recon_4shot_1epoch"
# 3shot
# load_prefix_seg = "domain0.75_word_liver_lr5e-4_1.7pseudo_1.0recon_3shot_1epoch"
# 2 shot
# load_prefix_seg = "domain0.75_word_liver_lr5e-4_1.0pseudo_1.0recon_2shot_2epoch"
# word stomach
# load_prefix_seg = "domain0.75_word_stomach_lr5e-4_0.6pseudo_1.2recon_2shot"

# 3D UNET
load_prefix_fix = "seg_nih_formae128"
# SWiN unter
# load_prefix_fix = "swinunter_seg_nih_formae128"
# load_prefix_fix = "domain0.75_word_spleen_lr5e-4_onlypseudo_1epoch_1shot"

# MAE model
load_prefix_mae = "mae_nih_pmask0.75_img128p16_batch1_all_rotation"
# todo:modify MAE
# 1shot
# spleen MAE
# load_prefix_mae ="1shot_spleen"
# Rkidney MAE
# load_prefix_mae ="1shot_rkidney"
# Word Lkidney MAE
# load_prefix_mae ="1shot_lkidney_new"
# LIVER MAE
# load_prefix_mae ="1shot"
# gall MAE
# load_prefix_mae ="1shot_gall"
# 5shot
# LIVER MAE
# load_prefix_mae ="5shot_liver_1epoch"
# 4shot liver MAE
# load_prefix_mae ="4shot_liver_1epoch"
# 3shot
# load_prefix_mae ="3shot_liver_1epoch"
# 2 shot
# load_prefix_mae ="2shot_liver_2epoch"

# stomach mae
# load_prefix_mae ="2shot_stomach_nepoch"

if load_prefix_seg:
    register_model_path = save_root_path + '/' + load_prefix_seg + '/best_model.ckpt'
    fix_model_path = save_root_path + '/' + load_prefix_fix + '/best_model.ckpt'
    if method == "seg_train":
        model.load_state_dict(torch.load(register_model_path)['model_state_dict'])
        print("seg load finish")
    elif method == "domain_adaptation" or method == "domain_adaptation_vit":
        try:
            new_state_dict = {}
            checkpoint = torch.load(register_model_path)['model_state_dict']
            for k, v in checkpoint.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
            pre_epochs = torch.load(register_model_path)['epoch']
            pre_best_result = torch.load(register_model_path)['best_result']
            fix_segmodel.load_state_dict(torch.load(fix_model_path)['model_state_dict'])
            # fixnew_state_dict = {}
            # fixcheckpoint = torch.load(fix_model_path)['model_state_dict']
            # for k, v in fixcheckpoint.items():
            #     fixnew_state_dict[k[7:]] = v
            # fix_segmodel.load_state_dict(fixnew_state_dict)
            fix_segmodel = fix_segmodel.cuda()
        except Exception as e:
            new_state_dict = {}
            checkpoint = torch.load(register_model_path)['model_state_dict']
            for k, v in checkpoint.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)
            pre_epochs = torch.load(register_model_path)['epoch']
            pre_best_result = torch.load(register_model_path)['best_result']

            fixnew_state_dict = {}
            fixcheckpoint = torch.load(fix_model_path)['model_state_dict']
            for k, v in fixcheckpoint.items():
                fixnew_state_dict[k[7:]] = v
            fix_segmodel.load_state_dict(fixnew_state_dict)
            # fix_segmodel.load_state_dict(torch.load(fix_model_path)['model_state_dict'])
            fix_segmodel = fix_segmodel.cuda()
        for param in fix_segmodel.parameters():
            param.requires_grad = False
        fix_segmodel.eval()
        print("both seg load finish")
    else:
        model.Seg.load_state_dict(torch.load(register_model_path)['model_state_dict'])
if method != 'seg_train':
    # VAE
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=lr1, weight_decay=weight_decay, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr3, weight_decay=weight_decay)
else:
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr3, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=lr1,weight_decay = weight_decay,momentum=0.9)
scheduler = LinearWarmupCosineAnnealingLR(
    optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epoch
)
print("Loading prefix.")
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
soft = nn.Softmax(dim=1)
if pre_epochs:
    epochs = pre_epochs // eval_epoch
    best_result = pre_best_result
else:
    epochs = 0
## training loop
dsc_pancreas = 0.0
dsc_pancreas_noft = 0.0
score_noft = {}
display_image = {}
epoch = 2
score = {}
score_visiable = {}
score_figure = {}
score_figure_gt = {}
score_figure_pseudo = {}
post_sum = 0
sum=0
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
if method == 'seg_train' or method == 'joint_train' or method == "domain_adaptation" or method == "domain_adaptation_vit" or method == 'sep_joint_train' or method == 'embed_train' or method == 'refine_vae':
    dsc_pancreas = 0.0
    if val_finetune != 0:
        dsc_pancreas_noft = 0.0
        score_noft = {}
    loss_gt = 0.0
    loss_recon = 0.0
    loss_fake = 0.0
    display_image = {}

    for val_idx, val_batch in enumerate(val_loader):
        val_batch[label_key + '_only'] = val_batch[label_key].type(torch.cuda.LongTensor)
        one_hot = torch.cuda.FloatTensor(val_batch[label_key + '_only'].size(0), len(mask_index),
                                         val_batch[label_key + '_only'].size(2),
                                         val_batch[label_key + '_only'].size(3),
                                         val_batch[label_key + '_only'].size(4)).zero_()
        val_batch[label_key + '_only'] = one_hot.scatter_(1, val_batch[label_key + '_only'].data, 1)
        val_batch[img_key] = val_batch[img_key].cuda()

        if val_finetune != 0:
            newmodel_state_dict = {}
            for k, v in model.state_dict().items():
                newmodel_state_dict[k[7:]] = v
            finetune_model.load_state_dict(newmodel_state_dict)
            optimizer_finetune = torch.optim.AdamW(finetune_model.parameters(), lr=lr3, weight_decay=weight_decay)
            if epoch != 0:
                for i in range(val_finetune):
                    val_batch = finetune_model(val_batch, img_key, label_key + '_pred')
                    with torch.no_grad():
                        val_batch = fix_segmodel(val_batch, img_key, label_key + '_only_fake')
                        # val_batch[label_key + '_only_fake'] = binarize(val_batch[label_key + '_only_fake'])
                        if load_prefix_mae:
                            maeloss, val_batch[label_key + '_pred_recon'], val_batch[
                                label_key + '_mae_mask'] = ref_model_parallel(
                                val_batch[label_key + '_pred'], unpatch=True, mask_ratio=1,
                                method="nums")  # 0.2
                            mae_mask = val_batch[label_key + '_mae_mask'].detach()
                            mae_mask = mae_mask.unsqueeze(-1).repeat(1, 1, mae_model.patch_size[
                                0] ** 3 * 2)
                            mae_mask = mae_model.unpatchify(mae_mask)
                            h = val_batch[label_key + '_pred'][0:1, 0:1, :, :, :].shape[4]
                            val_batch[label_key + '_display'] = torch.cat(
                                (val_batch[img_key][0:1, 0:1, :, :, h // 2], \
                                 val_batch[label_key + '_only'][0:1, 1:2, :, :, h // 2],
                                 val_batch[label_key + '_pred'][0:1, 1:2, :, :, h // 2], \
                                 mae_mask[:1, :1, :, :, h // 2],
                                 val_batch[label_key + '_pred_recon'][0:1, 1:2, :, :, h // 2]),
                                dim=0)
                    dsc_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_pred',
                                           target_key=label_key + '_only',
                                           botindex=1, topindex=len(mask_index))
                    dsc_loss_pseudo = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
                                                  target_key=label_key + '_only',
                                                  botindex=1, topindex=len(mask_index))
                    dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_pred',
                                                target_key=label_key + '_only_fake', botindex=1,
                                                topindex=len(mask_index), return_mean=True)

                    if load_prefix_mae:
                        recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_pred',
                                                 target_key=label_key + '_pred_recon', botindex=1,
                                                 topindex=len(mask_index), return_mean=True)
                        # maeloss = maeloss.mean()
                        if dsc_loss_fake <= 0.15:
                            final_loss = 1.0 * dsc_loss_fake + 1.0 * recon_loss
                        else:
                            final_loss = 0.5 * dsc_loss_fake + 2.0 * recon_loss
                        # final_loss = dsc_loss + 0.6 * recon_loss

                    else:
                        final_loss = dsc_loss_fake
                    optimizer_finetune.zero_grad()
                    final_loss.backward()
                    optimizer_finetune.step()
                    loss = []
                    display_image = {}
                    loss.append(['finetune_dice_loss', dsc_loss.item()])
                    loss.append(['finetune_dice_loss_fake', dsc_loss_fake.item()])
                    loss.append(['finetune_dsc_loss_pseudo', dsc_loss_pseudo.item()])
                    if load_prefix_mae:
                        loss.append(['finetune_mae_loss', maeloss.item()])
                        loss.append(['finetune_recon_loss', recon_loss.item()])
                    loss.append(['lr', optimizer_finetune.param_groups[0]['lr']])

                    display_image.update({label_key + '_display': val_batch[label_key + '_display']})
                    saver.write_display(i + val_idx * val_finetune + epoch * (max_idx_in_epoch + 1) * 2,
                                        loss, display_image, force_write=True)

        with torch.no_grad():
            if method == 'joint_train':

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

                if val_finetune != 0 and method == "domain_adaptation" or method == "domain_adaptation_vit" :
                    val_batch = model(val_batch, img_key, label_key + '_pred_noft')
                    val_batch = finetune_model(val_batch, img_key, label_key + '_pred')
                    if save_more_reference and val_idx == epoch % len(val_loader) and load_prefix_mae is not None:
                        maeloss, val_batch[label_key + '_pred_recon'], val_batch[
                            label_key + '_mae_mask'] = ref_model_parallel(
                            val_batch[label_key + '_pred'], unpatch=True, mask_ratio=1, method="nums")  # 0.2
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
                else:
                    # MAE out
                    if method == "domain_adaptation":
                        val_batch = model(val_batch, img_key, label_key + '_pred')
                    elif method == "domain_adaptation_vit":
                        val_batch[label_key + '_pred'] = model(val_batch[img_key])
                        val_batch[label_key + '_pred'] = soft(val_batch[label_key + '_pred'])

                    # #连通域
                    val_id = val_batch["id"][0]
                    print("val id is {}".format(val_id))
                    arr_out = np.squeeze(np.array(val_batch[label_key + '_pred'][:,1:,...].detach().squeeze(dim=1).cpu()),
                                            axis=0)
                    val_batch[label_key + '_post_pred'] = post_processing(arr_out)
                    label_out = np.squeeze(
                            np.array(val_batch[label_key + '_only'][:, 1:, ...].detach().squeeze(dim=1).cpu()), axis=0)
                    post_dice = dice_metric(y_pred=torch.tensor(val_batch[label_key + '_post_pred']).unsqueeze(0).unsqueeze(0), y=torch.tensor(label_out).unsqueeze(0).unsqueeze(0))
                    before_dice = dice_metric(y_pred=torch.tensor(arr_out).unsqueeze(0).unsqueeze(0),y=torch.tensor(label_out).unsqueeze(0).unsqueeze(0))
                    print("before dice {}".format(before_dice))
                    print("post dice {}".format(post_dice))
                    sum += before_dice
                    post_sum += post_dice

                    # VAE out
                    # pancreas
                    val_batch = model_VAE(val_batch, img_key, label_key + '_VAE_pred', label_key + '_VAE_recon_pred')
                    # other organ
                    # if method == "domain_adaptation":
                    #     val_batch = model_VAE(val_batch, img_key, label_key + '_VAE_pred')
                    # elif method == "domain_adaptation_vit":
                    #     val_batch[label_key + '_VAE_pred'] = model_VAE(val_batch[img_key])
                    #     val_batch[label_key + '_VAE_pred'] = soft(val_batch[label_key + '_VAE_pred'])

                    # direct out
                    if method == "domain_adaptation":
                        val_batch = fix_segmodel(val_batch, img_key, label_key + '_fix_pred')
                    elif method == "domain_adaptation_vit":
                        val_batch[label_key + '_fix_pred'] = fix_segmodel(val_batch[img_key])
                        val_batch[label_key + '_fix_pred'] = soft(val_batch[label_key + '_fix_pred'])

                    # val_batch = fix_segmodel(val_batch, img_key, label_key + '_fix_pred')

                    if val_idx >= 0:

                        val_id = val_batch["id"][0]
                        print("val id is {}".format(val_id))
                        valimage = np.squeeze(np.array(val_batch[img_key].detach().squeeze(dim=0).cpu()), axis=0)
                        direct_out = np.squeeze(np.array(val_batch[label_key + '_fix_pred'][:,1:,...].detach().squeeze(dim=1).cpu()),
                                            axis=0)
                        MAE_out = np.squeeze(np.array(val_batch[label_key + '_pred'][:,1:,...].detach().squeeze(dim=1).cpu()),
                                            axis=0)
                        # todo:VAE out may be need to update
                        # VAE_out = np.squeeze(np.array(val_batch[label_key + '_VAE_pred'][:,1:,...].detach().squeeze(dim=1).cpu()),
                        #                     axis=0)

                        VAE_out = np.array(np.squeeze(np.argmax(val_batch[label_key + '_VAE_pred'].detach().cpu(), axis=1), axis=0))

                        # direct_out = np.squeeze(np.array(val_batch[label_key + '_fix_pred'].detach().argmax(dim=1).cpu()),
                        #                     axis=0)
                        # MAE_out = np.squeeze(np.array(val_batch[label_key + '_pred'].detach().argmax(dim=1).cpu()),
                        #                     axis=0)

                        label_out = np.squeeze(
                            np.array(val_batch[label_key + '_only'][:, 1:, ...].detach().squeeze(dim=1).cpu()), axis=0)

                        re_img, crop_img1, crop_index1, crop_img2, crop_index2, image, origin_label = vis_image(dataset, origin_data_path, origin_label_path, val_id, organ)
                        if organ == "pancreas":
                            nnunet_data_path = "/data/MING/data/nnunet_data/useful_data/inferrenceTs"
                            if dataset == "msd":
                                fomatkeys = "pancreas_"
                                label_fomatkeys = fomatkeys
                            elif dataset == "syn":
                                fomatkeys = "syn_"
                                label_fomatkeys = "label"
                            elif dataset == "word":
                                fomatkeys = "word_"
                                label_fomatkeys = fomatkeys
                            elif dataset == "amos":
                                fomatkeys = "amos_"
                                label_fomatkeys = fomatkeys
                            data_files = fomatkeys + val_id
                            # 2. 加载文件和label
                            print("nnunet")

                            nnunet_out = nib.load(os.path.join(nnunet_data_path, data_files + ".nii.gz"))
                            nnunet_out = nnunet_out.get_data()
                            nnunet_out = np.swapaxes(nnunet_out,0 ,2)

                            boundary_model = models.Segmentation_array(n_channels=1, n_class=len(mask_index),
                                                                       norm_type=1)
                            hd_model = models.Segmentation_array(n_channels=1, n_class=len(mask_index), norm_type=1)
                            boundary_path = "/data/MING/data/NIH/model/seg_nih_formae_boundary/best_model.ckpt"
                            hd_path = "/data/MING/data/NIH/model/seg_nih_formae_hd_lr5e-4/best_model.ckpt"
                            fixnew_state_dict = {}
                            fixcheckpoint = torch.load(hd_path)['model_state_dict']
                            for k, v in fixcheckpoint.items():
                                fixnew_state_dict[k[7:]] = v
                            hd_model.load_state_dict(fixnew_state_dict)

                            fixnew_state_dict = {}
                            fixcheckpoint = torch.load(boundary_path)['model_state_dict']
                            for k, v in fixcheckpoint.items():
                                fixnew_state_dict[k[7:]] = v
                            boundary_model.load_state_dict(fixnew_state_dict)
                            boundary_model = boundary_model.cuda()
                            hd_model = hd_model.cuda()
                            boundary_out = boundary_model(val_batch[img_key])
                            hd_out = hd_model(val_batch[img_key])
                            save_boundary = copy.deepcopy(boundary_out.detach())
                            save_boundary = np.array(save_boundary.argmax(dim=1).detach().cpu()).astype(np.int32)
                            mask_boundary = save_boundary[0]

                            save_hd = copy.deepcopy(hd_out.detach())
                            save_hd = np.array(save_hd.argmax(dim=1).detach().cpu()).astype(np.int32)
                            mask_hd = save_hd[0]
                            vis = visualize_complete(re_img, direct_out, MAE_out, VAE_out, label_out, image,
                                                     crop_img1, crop_img2, crop_index1, crop_index2,nnunet_out=nnunet_out, n_class=2,ratio=0)
                        else:
                            # 原图 direct MAE VAE label
                            vis = visualize_complete(re_img, direct_out, MAE_out, VAE_out, label_out, image,
                                                              crop_img1, crop_img2, crop_index1, crop_index2, n_class=2)

                        # vis = visualize(valimage, direct_out, MAE_out, VAE_out, label_out, n_class=2)
                        # WORD PANCREAS
                        # src_path = '/data/MING/data/NIH/result/word_pancreas'
                        # src_path = '/data/MING/data/NIH/result/word_spleen'
                        # src_path = '/data/MING/data/NIH/result/word_rkidney'
                        # src_path = '/data/MING/data/NIH/result/word_lkidney'
                        # src_path = '/data/MING/data/NIH/result/word_liver_5shot'
                        # src_path = '/data/MING/data/NIH/result/word_liver_4shot'
                        # src_path = '/data/MING/data/NIH/result/word_liver_3shot'
                        # src_path = '/data/MING/data/NIH/result/word_liver_2shot'
                        # src_path = '/data/MING/data/NIH/result/word_stomach_2shot'
                        # src_path = '/data/MING/data/NIH/result/word_gall_2shot'
                        # src_path = '/data/MING/data/NIH/result/swin_word_spleen_1shot'
                        # src_path = '/data/MING/data/NIH/result/swin_word_liver_1shot'
                        # src_path = '/data/MING/data/NIH/result/swin_word_gall_1shot'
                        src_path = '/data/MING/data/NIH/result/msd_pancreas_2d'


                        # src_path = '/data/MING/data/NIH/result/amos_pancreas'

                        # MSD pancreas
                        # src_path = '/data/MING/data/NIH/result/msd_pancreas'
                        # src_path = '/data/MING/data/NIH/result/msd_spleen'

                        if not os.path.exists(src_path):
                            os.mkdir(src_path)
                        figurepath = os.path.join(src_path, val_batch["id"][0])
                        if not os.path.exists(figurepath):
                            os.mkdir(figurepath)

                        # todo:modify this path
                        for index, im in enumerate(vis):
                            cv2.imwrite(figurepath + "/{}.png".format(index), im)
                            directcolor = (0, 255, 255)
                            labelcolor = (0, 0, 128)
                            vaecolor = (0, 165, 255)
                            maecolor = (128, 0, 128)
                            nnunetcolor = (0, 255, 255)
                            bdcolor = (255, 0, 0)
                            hdcolor = (255, 255, 0)

                            if len(set(list(origin_label[...,index].flatten())))>1:
                                one = im.shape[1] // 5
                                img = im[:, :one, :]
                                mask = direct_out[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                # BGR

                                directedge = cv2.drawContours(img, contours, -1, directcolor, 1)

                                mask = label_out[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                labeledge = cv2.drawContours(directedge, contours, -1, labelcolor, 1)

                                mask = VAE_out[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                VAEedge = cv2.drawContours(labeledge, contours, -1, vaecolor, 1)

                                mask = MAE_out[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                MAEedge = cv2.drawContours(VAEedge, contours, -1, maecolor, 1)

                                mask = nnunet_out[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                nnunetedge = cv2.drawContours(MAEedge, contours, -1, nnunetcolor, 1)

                                mask = mask_boundary[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                bdedge = cv2.drawContours(nnunetedge, contours, -1, bdcolor, 1)

                                mask = mask_hd[..., index].astype(np.uint8)
                                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL	, cv2.CHAIN_APPROX_SIMPLE)
                                hdedge = cv2.drawContours(bdedge, contours, -1, hdcolor, 1)


                                cv2.imwrite(src_path + "/edge_{}.png".format(index), hdedge)


                        # 2.设置源路径与保存路径

                        sav_path = '/data/MING/data/NIH/result/msd_pancreas_2d_video'
                        # sav_path = '/data/MING/data/NIH/result/msd_pancreas_video'
                        # sav_path = '/data/MING/data/NIH/result/msd_spleen_video'

                        # sav_path = '/data/MING/data/NIH/result/word_spleen_video'
                        # sav_path = '/data/MING/data/NIH/result/word_rkidney_video'
                        # sav_path = '/data/MING/data/NIH/result/word_lkidney_video'
                        # sav_path = '/data/MING/data/NIH/result/word_liver_5shot_video'
                        # sav_path = '/data/MING/data/NIH/result/word_liver_4shot_video'
                        # sav_path = '/data/MING/data/NIH/result/word_liver_3shot_video'
                        # sav_path = '/data/MING/data/NIH/result/word_liver_2shot_video'
                        # sav_path = '/data/MING/data/NIH/result/word_stomach_2shot_video'
                        # sav_path = '/data/MING/data/NIH/result/word_gall_2shot_video'
                        # sav_path = '/data/MING/data/NIH/result/swin_word_spleen_1shot_video'
                        # sav_path = '/data/MING/data/NIH/result/swin_word_liver_1shot_video'
                        # sav_path = '/data/MING/data/NIH/result/swin_word_gall_1shot_video'

                        # sav_path = '/data/MING/data/NIH/result/amos_pancreas_video'

                        # sav_path = '/data/MING/data/NIH/result/word_pancreas_video'
                        if not os.path.exists(sav_path):
                            os.mkdir(sav_path)
                        # 3.获取图片总的个数
                        # 3.1获取图片的文件夹
                        # ipdb.set_trace()
                        root, files, dirs = file_name(src_path)
                        for i in range(len(dirs)):
                            if dirs[i] == val_id:
                                video_savepath = os.path.join(sav_path, dirs[i])
                                if not os.path.exists(video_savepath):
                                    os.mkdir(video_savepath)
                                video_savepath = video_savepath + "/video.avi"
                                video_path = os.path.join(root, dirs[i])
                                # 4.设置视频写入器
                                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MP4格式
                                # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
                                # videowrite = cv2.VideoWriter(sav_path, fourcc, 2, size)  # 2是每秒的帧数，size是图片尺寸
                                videowrite = cv2.VideoWriter(video_savepath, fourcc, 2, (vis.shape[2], vis.shape[1]))  # 2是每秒的帧数，size是图片尺寸
                                # 5.临时存放图片的数组
                                img_array = []
                                _, imgfiles, _ = file_name(video_path)
                                # 6.读取所有图片
                                for i in range(len(imgfiles)):
                                    img_path = video_path + "/{}.png".format(i)
                                    img = cv2.imread(img_path)
                                    img_array.append(img)
                                    videowrite.write(img_array[i])
                            else:
                                continue

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
                    if save_more_reference and val_idx == epoch % len(val_loader) and load_prefix_mae is not None:
                        maeloss, val_batch[label_key + '_pred_recon'], val_batch[
                            label_key + '_mae_mask'] = ref_model_parallel(
                            val_batch[label_key + '_pred'], unpatch=True, mask_ratio=1, method="nums")  # 0.2
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
        if val_finetune != 0:
            score_noft[val_idx] = avg_dsc(val_batch, source_key=label_key + '_pred_noft',
                                          target_key=label_key + '_only', binary=True, botindex=1,
                                          topindex=len(mask_index)).item()
            dsc_pancreas_noft += score_noft[val_idx]

        score[val_idx] = avg_dsc(val_batch, source_key=label_key + '_pred',
                                 target_key=label_key + '_only', binary=True, botindex=1,
                                 topindex=len(mask_index)).item()
        print("out dice {}".format(score[val_idx]))

        dsc_pancreas += score[val_idx]

        # for MAE
        analysis_figure_name = None
        if analysis_figure_name != None:
            # label_key+'_only' ->label ,  label_key+'_only_recon' -> label经过VAE得到的重建图
            # label_key+'_pred' ->预测的分割图 ，label_key+'_recon_pred' ->预测分割图的重建图
            # label_key+'_only_fake'->pesudo label ，label_key+'_only_fake_recon' -> pesudo通过VAE的重建图
            # update seg recon
            maeloss, val_batch[label_key + '_pred_recon'], val_batch[
                label_key + '_mae_mask'] = ref_model_parallel(
                val_batch[label_key + '_pred'], unpatch=True, mask_ratio=1, method="nums")  # 0.2
            # label recon
            _, val_batch[label_key + '_only_recon'], _ = ref_model_parallel(
                val_batch[label_key + '_only'], unpatch=True, mask_ratio=1, method="nums")  # 0.2
            # pseudo recon
            val_batch = fix_segmodel(val_batch, img_key, label_key + '_only_fake')
            _, val_batch[label_key + '_only_fake_recon'], _ = ref_model_parallel(
                val_batch[label_key + '_only_fake'], unpatch=True, mask_ratio=1, method="nums")  # 0.2

            gt_recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_only_recon', target_key=label_key + '_only',
                                        binary=True, botindex=1, topindex=len(mask_index)).item()
            gt_dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
                                           target_key=label_key + '_only', botindex=1, topindex=len(mask_index),
                                           return_mean=True).item()

            recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_pred', target_key=label_key + '_pred_recon',
                                     binary=True, botindex=1, topindex=len(mask_index)).item()
            dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_pred', target_key=label_key + '_only_fake',
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
        # for VAE

        # analysis_figure_name = "VAE"
        # if analysis_figure_name != None:
        #     # label_key+'_only' ->label ,  label_key+'_only_recon' -> label经过VAE得到的重建图
        #     # label_key+'_pred' ->预测的分割图 ，label_key+'_recon_pred' ->预测分割图的重建图
        #     # label_key+'_only_fake'->pesudo label ，label_key+'_only_fake_recon' -> pesudo通过VAE的重建图
        #     # fix_vae = models.VAE(n_channels=len(mask_index), n_class=len(mask_index), norm_type=1, dim=128,
        #     #                      soft=softrelu == 1)
        #     # fix_vae = fix_vae.cuda()
        #     # vae_model_path = "/data1/2021_stu/MING/NIH/model/vae_nih/best_model.ckpt"
        #     # fix_vae.load_state_dict(torch.load(vae_model_path)["model_state_dict"])
        #     val_batch[label_key + '_only_recon'], _, _ = model_VAE.Vae(val_batch[label_key + '_only'], if_random=False,
        #                                                            scale=0)
        #     val_batch = fix_segmodel(val_batch, img_key, label_key + '_only_fake')
        #     val_batch[label_key + '_only_fake_recon'], _, _ = model_VAE.Vae(val_batch[label_key + '_only_fake'], if_random=False,
        #                                                            scale=0)
        #
        #     gt_recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_only_recon', target_key=label_key + '_only',
        #                                 binary=True, botindex=1, topindex=len(mask_index)).item()
        #     gt_dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
        #                                    target_key=label_key + '_only', botindex=1, topindex=len(mask_index),
        #                                    return_mean=True).item()
        #     recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_VAE_pred', target_key=label_key + '_VAE_recon_pred',
        #                              binary=True, botindex=1, topindex=len(mask_index)).item()
        #     dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_pred', target_key=label_key + '_only_fake',
        #                                 botindex=1, topindex=len(mask_index), return_mean=True).item()
        #     pseudo_recon_loss = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
        #                                     target_key=label_key + '_only_fake_recon', botindex=1,
        #                                     topindex=len(mask_index), return_mean=True).item()
        #     pseudo_dsc_loss_fake = 1 - avg_dsc(val_batch, source_key=label_key + '_only_fake',
        #                                        target_key=label_key + '_only_fake', botindex=1,
        #                                        topindex=len(mask_index), return_mean=True).item()
        #     print(f"result: {val_idx}")
        #     print(1 - score[val_idx])
        #     print(gt_recon_loss)
        #     print(recon_loss)
        #     print(pseudo_recon_loss)
        #     print(dsc_loss_fake)
        #     print(gt_dsc_loss_fake)
        #     print(pseudo_dsc_loss_fake)
        #     loss_gt += gt_recon_loss
        #     loss_recon += recon_loss
        #     loss_fake += dsc_loss_fake
        #     # 预测的分割图和预测分割图通过VAE重建后的重建损失 以及 预测的分割图和pseudo图的pseudo损失
        #     score_figure[val_idx] = [dsc_loss_fake, recon_loss]
        #     # pseudo图和label的dice    以及  label和 label经过VAE后的重建图计算dice
        #     score_figure_gt[val_idx] = [gt_dsc_loss_fake, gt_recon_loss]
        #     # pseudo和pseudo经过VAE后的重建图的dice 以及pesudo 和pesudo的dice
        #     score_figure_pseudo[val_idx] = [pseudo_dsc_loss_fake, pseudo_recon_loss]

    dsc_pancreas /= (val_idx + 1)
    if val_finetune != 0:
        dsc_pancreas_noft /= (val_idx + 1)
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



# val_list = "WORD_val"
# val_data_root = "/data1/2021_stu/MING/WORD/data"
# np.random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # dataset return [] eval_epoch 次的NIH_train的内容
# train_data_list = filedict_from_json(data_path, train_list, eval_epoch)
# # print(train_data_list)
# transforms = {'train': []}
# # define training data pipeline   mask_index [[0, 0], [1, 1]]
# # input_phases = ["venous"]
# transforms['train'].append(
#     NumpyLoader_Multi_merge(fields=input_phases, root_dir=data_root, load_mask=True, mask_index=mask_index))
# transforms['train'].append(CropResize(fields=input_phases, output_size=patch_size))
# # transforms['train'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))
# if not no_aug:
#     transforms['train'].append(Reshape(fields=input_phases_mask))
#     # TODO: do_rotation
#     transforms['train'].append(
#         MySpatialTransform(patch_size, [dis // 2 - 5 for dis in patch_size], random_crop=True,
#                            scale=(0.85, 1.15),
#                            do_elastic_deform=False, alpha=(0, 500),
#                            do_rotation=True, sigma=(10, 30.),
#                            angle_x=(-0.2, 0.2), angle_y=(-0.2, 0.2),
#                            angle_z=(-0.2, 0.2),
#                            border_mode_data="constant",
#                            border_cval_data=-1024,
#                            data_key="venous", p_el_per_sample=0, label_key="venous_pancreas",
#                            p_scale_per_sample=1, p_rot_per_sample=1))
# # transforms['train'].append(PadToSize(fields=input_phases, size=[1,1]+patch_size, pad_val=-1024, seg_pad_val=0,random_subpadding=True,load_mask=method=='seg_train'))
# for phase in input_phases:
#     transforms['train'].append(CopyField(fields=[phase], to_field=[phase + '_origin']))
# transforms['train'].append(Clip(fields=input_phases, new_min=-200, new_max=400))
# transforms['train'].append(CenterIntensities(fields=input_phases, subtrahend=100, divisor=300))
# transforms['train'].append(Reshape(fields=input_phases_mask, reshape_view=[-1] + patch_size))
# val_data_list = filedict_from_json(data_path, val_list)
# transforms['val'] = []
# ## define validation data pipeline
# transforms['val'].append(
#     NumpyLoader_Multi_merge(fields=input_phases, root_dir=val_data_root, load_mask=True, mask_index=mask_index))
# transforms['val'].append(CropResize(fields=input_phases, output_size=patch_size))
# # transforms['val'].append(PadToSize(fields=input_phases, size=input_size, pad_val=-1024, seg_pad_val=0,random_subpadding=False,load_mask=True))
# for phase in input_phases:
#     transforms['val'].append(CopyField(fields=[phase], to_field=[phase + '_origin']))
# transforms['val'].append(Clip(fields=input_phases, new_min=-200, new_max=400))
# transforms['val'].append(CenterIntensities(fields=input_phases, subtrahend=100, divisor=300))
# transforms['val'].append(Reshape(fields=input_phases_mask, reshape_view=[-1] + patch_size))
# for k, v in transforms.items():
#     transforms[k] = torchvision.transforms.Compose(v)
# ###############################################################################################
# ############################ Create Datasets ##################################################
# ###############################################################################################
# print("Loading data.")
# train_dataset = BaseDataset(train_data_list, transforms=transforms['train'])
# val_dataset = BaseDataset(val_data_list, transforms=transforms['val'])
# if method != "domain_adaptation":
#     train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=True, num_workers=num_workers,
#                               drop_last=True, pin_memory=True)
# else:
#     train_loader = DataLoader(train_dataset, batch_size=trainbatch, shuffle=False, num_workers=num_workers,
#                               drop_last=True, pin_memory=True)
#     print("domain!")
# val_loader = DataLoader(val_dataset, batch_size=valbatch, shuffle=False, num_workers=num_workers, pin_memory=True)