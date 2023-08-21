import math
import ipdb
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from functools import partial
# from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from utils.pos_embed import get_3d_sincos_pos_embed, get_sinusoid_encoding_table
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock,PatchEmbed
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep, optional_import
from typing import Sequence, Type, Union
from torch.nn import LayerNorm
from monai.networks.layers import Conv, trunc_normal_
import torch
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def Normalization(norm_type, out_channels,num_group=1):
    if norm_type==1:
        return nn.InstanceNorm3d(out_channels)
    elif norm_type==2:
        return nn.BatchNorm3d(out_channels,momentum=0.1)
    elif norm_type==3:
        return GSNorm3d(out_channels,num_group=num_group)

class GSNorm3d(torch.nn.Module):
    def __init__(self, out_ch, num_group=1):
        super().__init__()
        self.out_ch = out_ch
        self.num_group=num_group
        #self.activation = nn.ReLU()
    def forward(self, x):
        interval = self.out_ch//self.num_group
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            #dominator = torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)
            #dominator = dominator + (dominator<0.001)*1
            tensors.append(x[:,start_index:start_index+interval,...]/(torch.sum(x[:,start_index:start_index+interval,...],dim=1,keepdim=True)+0.0001))
            start_index = start_index+interval
        
        return torch.cat(tuple(tensors),dim=1)

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            Normalization(norm_type,out_ch),
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),  
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),     
            activation,
            torch.nn.Conv3d(out_ch, out_ch, 3, padding=1), 
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class Up_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch,num_group=1,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='trilinear'),
            DoubleConv_GS(in_ch, out_ch, num_group,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Down_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv_GS(in_ch, out_ch, num_group,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class Conv_GS(torch.nn.Module):
    def __init__(self, in_ch, out_ch, num_group=1, activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2, num_group=1,activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch,norm_type=2,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )


    def forward(self, x):
        x = self.conv(x)
        return x

class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,kernal_size=(2,2,2),stride=(2,2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class GSConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, num_group=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, if_sub=None,trainable=True):
        super(GSConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
        self.num_group = num_group
        self.interval = self.in_channels//self.num_group
    def forward(self, x):
        
        weight = torch.abs(self.weight)
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            tensors.append(weight[:,start_index:start_index+self.interval,...]/torch.sum(weight[:,start_index:start_index+self.interval,...],1,keepdim=True))
            start_index += self.interval
        weight = torch.cat(tuple(tensors),1)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class GSConvTranspose3d(nn.ConvTranspose3d):

    def __init__(self, in_channels, out_channels, kernel_size, num_group=1, stride=1,
                 padding=0, dilation=1, output_padding=0,groups=1, bias=False, if_sub=None,trainable=True):
        super(GSConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups,bias, dilation )
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
        self.num_group = num_group
        self.interval = self.in_channels//self.num_group
    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        weight = torch.abs(self.weight)
        start_index = 0
        tensors = []
        for i in range(self.num_group):
            tensors.append(weight[:,start_index:start_index+self.interval,...]/torch.sum(weight[:,start_index:start_index+self.interval,...],1,keepdim=True))
            start_index += self.interval
        weight = torch.cat(tuple(tensors),1)
        return F.conv_transpose3d(x, weight, self.bias, self.stride,
                        self.padding,output_padding,  self.groups,self.dilation)
class SConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, if_sub=None,trainable=True):
        super(SConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #self.if_sub=torch.tensor(if_sub,dtype=torch.float).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if not trainable:
            self.weight.requires_grad=False
    def forward(self, x):
        
        weight = self.weight
        weight_mean = weight.mean([2,3,4], keepdim=True)
        weight = weight - weight_mean

        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
         
class VAE(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256],dim=1024,soft=False):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_class, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)
        self.down5 = Down(n_fmaps[4], n_fmaps[5],norm_type=norm_type,soft=False)
        self.fc_mean = torch.nn.Linear(16384,dim)
        self.fc_std = torch.nn.Linear(16384,dim)
        self.fc2 = torch.nn.Linear(dim,16384)
        self.up1 = Up(n_fmaps[5],n_fmaps[4],norm_type=norm_type,soft=False)
        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, x,if_random=False,scale=1,mid_input=False,dropout=0.0):
        #'pred_only','pred_recon',if_random=False
        #x = data_dict[in_key]
        # print(x.shape)
        
        if not mid_input:
            #input_res = data_dict.get(self.in_key2)
            #input_x = self.SConv3d(input_x)
            # ipdb.set_trace()
            # x shape ([4, 2, 128, 128, 128]) 经过in_block后 x shape torch.Size([4, 8, 128, 128, 128]) 后面同理，dim=1 相当于2D的Channel
            x = self.in_block(x)
            x = self.down1(x)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.down5(x)
            x = x.view(x.size(0),16384)
            x_mean = self.fc_mean(x)
            x_std = nn.ReLU()(self.fc_std(x))
            #data_dict['mean'] = x_mean
            #data_dict['std'] = x_std
            z = torch.randn(x_mean.size(0),x_mean.size(1)).type(torch.cuda.FloatTensor)
            if if_random:
                x = self.fc2(x_mean+z*x_std*scale)
            else:
                x = self.fc2(x_mean)
        else:
            x = self.fc2(x)
        x = x.view(x.size(0),256,4,4,4)
        
        x = self.up1(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up2(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up3(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up4(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up5(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.out_block(x)
        x = self.final(x)
       
        #data_dict[out_key] = x
        if not mid_input:
            return x,x_mean,x_std
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class myBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class myPatchEmbed(PatchEmbed):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.
    Example::
     from monai.networks.blocks import PatchEmbed
     PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
    """

    def __init__(
        self,
        patch_size: Union[Sequence[int], int] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """
        super().__init__()
        PatchEmbed.__init__(self, patch_size=patch_size, in_chans =in_chans,embed_dim=embed_dim, norm_layer=norm_layer,spatial_dims=spatial_dims)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            # todo
            # x = x.permute((0, 1, 4, 2, 3))
            x = x.flatten(2).transpose(1, 2)
            x  = self.norm(x)
        return x

class videoPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        # x = x.permute((0,1,4,2,3))
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Mae3D2Mask(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=96, patch_size=16, in_chans=2,
                 embed_dim=8192, depth=12, num_heads=8,
                 decoder_embed_dim=5464, decoder_depth=8, decoder_num_heads=16,
                 mlp_dim=32768, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics [4, 2, 128, 128, 128]
        self.in_chans = in_chans
        self.patch_embed = myPatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # self.patch_embed = PatchEmbeddingBlock(img_size=(img_size, img_size, img_size), patch_size=(patch_size, patch_size, patch_size), in_channels=2,
        #                                        hidden_size=embed_dim, num_heads=num_heads, pos_embed="conv")
        # num_patches = self.patch_embed.num_patches      # patch的个数 or 位置个数
        self.num_patches = (img_size // patch_size) ** 3  # patch的个数 or 位置个数
        self.patch_size = (patch_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 开头的token
        # 位置信息
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # mlp_ratio作用为对MLP的输入维度*4作为MLP输出维度
        # 24个block
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, embed_dim * 4, num_heads)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # decoder 的token

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        # 8个transformer block
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_embed_dim * 4, decoder_num_heads)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed ->197,768  +上token
        # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
        # embed_dim=8192 不能够被3整除,导致在position embed中出现了维度不一致 不能对position 进行初始化。
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.num_patches ** (1 / 3)),
                                            cls_token=True)

        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # video sin cos
        # pos_embed = get_sinusoid_encoding_table(self.num_patches, self.pos_embed.shape[-1])[0]
        # pos_embed = np.concatenate([np.zeros([1, self.pos_embed.shape[-1]]), pos_embed], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    math.ceil(self.num_patches ** (1 / 3)), cls_token=True)
        # video sincos
        # decoder_pos_embed = get_sinusoid_encoding_table(self.num_patches, self.decoder_pos_embed.shape[-1])[0]
        # decoder_pos_embed = np.concatenate([np.zeros([1, self.decoder_pos_embed.shape[-1]]), decoder_pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            # deit
            # trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 2, H, W， D)
        x: (N, L, patch_size**3 *2)
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p
        # (batch, 2 8,16,8,16,8,16) 2为channel数， 8为个数， 16为patch size大小
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p, d, p))
        # (batch, 8,8,8,16,16,16,2)
        x = torch.einsum('nchpwqdf->nhwdpqfc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *2)
        imgs: (N, 2, H, W， D)
        """
        p = self.patch_size[0]
        h = w = d = math.ceil(x.shape[1] ** (1 / 3))
        assert h * w * d == x.shape[1]
        # (batch, 8, 8, 8, 16, 16, 16, 2) 2为channel数， 8为个数， 16为patch size大小
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, self.in_chans))
        # (batch, 2, 8, 16, 8, 16, 8, 16)
        x = torch.einsum('nhwdpqfc->nchpwqdf', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # 生成（N,L） 再0-1的噪声
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 根据shuffle index 进行mask    (16, 49, 768)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def myrandom_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros(
            (N, self.in_chans, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        testout = self.patch_embed(zerotest).detach()
        templist = []
        templist2 = []
        for i in range(L):
            if i == L:
                print(i)
                break
            if self.in_chans == 1:
                if (testout[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
            else:
                if (x[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
        ones = torch.ones((N, L), device=x.device)
        len_keep = int(len(templist2) * (1 - mask_ratio))
        flag = len_keep // 3
        k = 0
        if flag < 1:
            for i in range(len(templist2)):
                ones[0][templist2[i]] = 0
        else:
            for i in range(len(templist2)):
                ones[0][templist2[i]] = 0.0001 * i
            ones[0][templist2[0]] = 0
            ones[0][templist2[len(templist2) // 2]] = 0
            ones[0][templist2[-2]] = 0

        # elif 1 <= flag < 2:
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.0001 * i
        #     ones[0][templist2[0]] = 0
        #     ones[0][templist2[len(templist2) // 2]] = 0
        #     ones[0][templist2[-2]] = 0
        # else:
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.0001*i
        #     ones[0][templist2[0]] = 0
        #     ones[0][templist2[len(templist2) // 5 ]] = 0
        #     ones[0][templist2[(len(templist2) // 5)  *2]] = 0
        #     ones[0][templist2[(len(templist2) // 5)  *3]] = 0
        #     ones[0][templist2[(len(templist2) // 5)  *4]] = 0
        #     ones[0][templist2[-2]] = 0

        new_ids_shuffle = torch.argsort(ones, dim=1)

        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 根据shuffle index 进行mask    (16, 49, 768)
        for i in range(len(templist2)):
            if ids_shuffle[0][i] in templist2:
                k += 1
            else:
                print("{} not match ".format(i))
        assert k == len(templist2)
        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def twomask(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        onestest = torch.ones((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        # 确保0channel 的pancreas像素表示为0，1channel  pancreas的像素表示为1，不然会出错。
        # test = torch.cat((zerotest, onestest), dim=1)
        firstblock = torch.cat((onestest, zerotest), dim=1)
        firstblockout = self.patch_embed(firstblock).detach()
        if self.in_chans==1:
            testout = self.patch_embed(zerotest).detach()

        templist = []
        templist2 = []
        for i in range(L):
            if i == L:
                print(i)
                break
            if self.in_chans == 1:
                if (testout[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
            else:
                if (firstblockout[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)

        ones = torch.zeros((N, L), device=x.device)

        len_keep = int(L - mask_ratio)

        flag = len_keep // 2
        k = 0
        # for i in range(len(templist)):
        #     ones[0][templist[i]] = 0.0001 * i
        # if flag < 1 :
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.00001 * i
        # else:
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.00001 * i
        factor = torch.rand(1)
        for i in range(len(templist2)):
            ones[0][templist2[i]] = 0.1 * factor
        ones[0][templist2[3]] = 1
            # ones[0][templist2[-5]] = 1
        new_ids_shuffle = torch.argsort(ones, dim=1)

        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # 根据shuffle index 进行mask    (16, 49, 768)

        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def myrandom_edgemasking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        onestest = torch.ones((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        # 确保0channel 的pancreas像素表示为0，1channel  pancreas的像素表示为1，不然会出错。
        test = torch.cat((zerotest, onestest), dim=1)
        firstblock = torch.cat((onestest, zerotest), dim=1)
        firstblockout = self.patch_embed(firstblock).detach()
        if self.in_chans == 1:
            zeroout = self.patch_embed(zerotest).detach()
            oneout = self.patch_embed(onestest).detach()
        else:
            testout = self.patch_embed(test).detach()
        templist = []
        templist2 = []
        onelist = []
        edgelist = []
        for i in range(L):
            if i == L:
                print(i)
                break
            if self.in_chans == 1:
                if (zeroout[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
                    if (oneout[0][0] == x[0][i]).all():
                        onelist.append(i)
                    else:
                        edgelist.append(i)
            else:
                if (x[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
                    if (testout[0][0] == x[0][i]).all():
                        onelist.append(i)
                    else:
                        edgelist.append(i)
        ones = torch.ones((N, L), device=x.device)
        len_keep = int(len(edgelist) * (1 - mask_ratio))
        flag = len_keep // 3
        k = 0
        if flag < 1:
            for i in range(len(edgelist)):
                ones[0][edgelist[i]] = 0
        else:
            for i in range(len(edgelist)):
                ones[0][edgelist[i]] = 0.0001 * i
            ones[0][edgelist[0]] = 0
            ones[0][edgelist[len(edgelist) // 2]] = 0
            ones[0][edgelist[-2]] = 0

        new_ids_shuffle = torch.argsort(ones, dim=1)
        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 根据shuffle index 进行mask    (16, 49, 768)
        for i in range(len(edgelist)):
            if ids_shuffle[0][i] in edgelist:
                k += 1
            else:
                print("{} not match ".format(i))
        assert k == len(edgelist)
        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def batchmyrandom_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros(
            (N, self.in_chans, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        testout = self.patch_embed(zerotest).detach()
        templist = [[] for i in range(N)]
        templist2 = [[] for i in range(N)]
        for j in range(N):
            for i in range(L):
                if i == L:
                    print(i)
                    break
                if self.in_chans == 1:
                    if (testout[0][0] == x[j][i]).all():
                        templist[j].append(i)
                    else:
                        templist2[j].append(i)
                else:
                    if (x[j][0] == x[j][i]).all():
                        templist[j].append(i)
                    else:
                        templist2[j].append(i)
        ones = torch.ones((N, L), device=x.device)

        len_keep = int(len(templist2) * (1 - mask_ratio))
        flag = len_keep // 3
        k = 0
        if flag < 1:
            for i in range(len(templist2)):
                ones[0][templist2[i]] = 0
        else:
            for i in range(len(templist2)):
                ones[0][templist2[i]] = 0.0001 * i
            ones[0][templist2[0]] = 0
            ones[0][templist2[len(templist2) // 2]] = 0
            ones[0][templist2[-2]] = 0

        new_ids_shuffle = torch.argsort(ones, dim=1)

        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 根据shuffle index 进行mask    (16, 49, 768)
        for i in range(len(templist2)):
            if ids_shuffle[0][i] in templist2:
                k += 1
            else:
                print("{} not match ".format(i))
        assert k == len(templist2)
        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches (B, N, embed_dim)
        x = self.patch_embed(x)
        # B,N,W,H,D = x.shape
        # x = x.reshape()
        # ipdb.set_trace()

        # add pos embed w/o cls token 对除了 cls token添加位置信息
        # x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio   # 将添加有位置信息的每一个patch块进行mask
        x, mask, ids_restore = self.twomask(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # expand 到batch size个cls token  ->(batch, 1, embedim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # 在位置的维度进行拼接
        x = torch.cat((cls_tokens, x), dim=1)
        """
        392
                        pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
                pos_embed_sampled = torch.gather(
                    pos_embed[:, s:, :],
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(
                        1, 1, self.pos_embed.shape[2]
                    ),
                )
                if self.cls_embed_on:
                    pos_embed_sampled = torch.cat(
                        [pos_embed[:, :s, :], pos_embed_sampled], 1
                    )
                x += pos_embed_sampled
        """

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # out (batch, nums_patch * mask_ratio + 1, embed_dim)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens    input x shape (16, 50, 768) ->output (16, 50, 512)
        x = self.decoder_embed(x)

        # append mask tokens to sequence self.mask_token(1,1,embed_dim)
        # ->mask_tokens (batch, num_patchs + 1 - nums_patch * mask_ratio + 1, 512) ->(16, 147, 512)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # 获取到无cls token 的信息
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle 回去重新排序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # 添加cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed (1, 197, 512)
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        # print(x.shape)
        x = self.decoder_norm(x)

        # predictor projection  还原回原来的pixel patch * patch * 3 ->(batch, num_patchs, pixel patch * patch * 3)
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]  (N, 2, H, W， D)
        pred: (N, L, patch_size**3 *2)
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        # ipdb.set_trace()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # all loss
        loss = loss.sum() / float(loss.shape[1] * loss.shape[0])
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches /只对出除了的patch计算loss
        return loss

    def forward(self, imgs, unpatch=True, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        if unpatch:
            pred = self.unpatchify(pred)
        return loss, pred, mask


class Mae3D2MaskSeg(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=96, patch_size=16, in_chans=2,
                 embed_dim=8192, depth=12, num_heads=8,
                 decoder_embed_dim=5464, decoder_depth=8, decoder_num_heads=16,
                 mlp_dim=32768, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics [4, 2, 128, 128, 128]
        self.in_chans = in_chans
        self.patch_embed = myPatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # self.patch_embed = PatchEmbeddingBlock(img_size=(img_size, img_size, img_size), patch_size=(patch_size, patch_size, patch_size), in_channels=2,
        #                                        hidden_size=embed_dim, num_heads=num_heads, pos_embed="conv")
        # num_patches = self.patch_embed.num_patches      # patch的个数 or 位置个数
        self.num_patches = (img_size // patch_size) ** 3  # patch的个数 or 位置个数
        self.patch_size = (patch_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 开头的token
        # 位置信息
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # mlp_ratio作用为对MLP的输入维度*4作为MLP输出维度
        # 24个block
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, embed_dim * 4, num_heads)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # decoder 的token

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        # 8个transformer block
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_embed_dim * 4, decoder_num_heads)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed ->197,768  +上token
        # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
        # embed_dim=8192 不能够被3整除,导致在position embed中出现了维度不一致 不能对position 进行初始化。
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.num_patches ** (1 / 3)),
                                            cls_token=True)

        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # video sin cos
        # pos_embed = get_sinusoid_encoding_table(self.num_patches, self.pos_embed.shape[-1])[0]
        # pos_embed = np.concatenate([np.zeros([1, self.pos_embed.shape[-1]]), pos_embed], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    math.ceil(self.num_patches ** (1 / 3)), cls_token=True)
        # video sincos
        # decoder_pos_embed = get_sinusoid_encoding_table(self.num_patches, self.decoder_pos_embed.shape[-1])[0]
        # decoder_pos_embed = np.concatenate([np.zeros([1, self.decoder_pos_embed.shape[-1]]), decoder_pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            # deit
            # trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 2, H, W， D)
        x: (N, L, patch_size**3 *2)
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p
        # (batch, 2 8,16,8,16,8,16) 2为channel数， 8为个数， 16为patch size大小
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p, d, p))
        # (batch, 8,8,8,16,16,16,2)
        x = torch.einsum('nchpwqdf->nhwdpqfc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *2)
        imgs: (N, 2, H, W， D)
        """
        p = self.patch_size[0]
        h = w = d = math.ceil(x.shape[1] ** (1 / 3))
        assert h * w * d == x.shape[1]
        # (batch, 8, 8, 8, 16, 16, 16, 2) 2为channel数， 8为个数， 16为patch size大小
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, self.in_chans))
        # (batch, 2, 8, 16, 8, 16, 8, 16)
        x = torch.einsum('nhwdpqfc->nchpwqdf', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p, d * p))
        return imgs

    def myrandom_masking(self, x, mask_ratio,argx):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        # onestest = torch.ones((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        patch1embed = myPatchEmbed(patch_size=16, in_chans=1, embed_dim=D).cuda().float()
        argout = patch1embed(argx)
        # 确保0channel 的pancreas像素表示为0，1channel  pancreas的像素表示为1，不然会出错。
        # test = torch.cat((zerotest, onestest), dim=1)
        # firstblock = torch.cat((onestest, zerotest), dim=1)
        firstblockout = patch1embed(zerotest).detach()
        if self.in_chans==1:
            testout = self.patch_embed(zerotest).detach()

        templist = []
        templist2 = []
        for i in range(L):
            if i == L:
                print(i)
                break
            if self.in_chans == 1:
                if (testout[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
            else:
                if (firstblockout[0][0] == argout[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)

        ones = torch.zeros((N, L), device=x.device)
        len_keep = int(L - (len(templist2) *  mask_ratio))
        flag = len_keep // 3
        k = 0
        if flag < 1 :
            for i in range(len(templist2)):
                ones[0][templist2[i]] = 0
        else:
            for i in range(len(templist2)):
                ones[0][templist2[i]] = 0.1 * torch.rand(1)
                if ones[0][templist2[i]] >= 1 :
                    raise ValueError("templist2 value too large")

            # ones[0][templist2[0]] = 0
            # ones[0][templist2[len(templist2) // 2]] = 0
            # ones[0][templist2[-2]] = 0

        new_ids_shuffle = torch.argsort(ones, dim=1)
        # ipdb.set_trace()

        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 根据shuffle index 进行mask    (16, 49, 768)
        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)


        return x_masked, mask, ids_restore

    def twomask(self, x, mask_ratio, argx):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        # onestest = torch.ones((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        patch1embed = myPatchEmbed(patch_size=16, in_chans=1, embed_dim=D).cuda().float()
        argout = patch1embed(argx)
        # 确保0channel 的pancreas像素表示为0，1channel  pancreas的像素表示为1，不然会出错。
        # test = torch.cat((zerotest, onestest), dim=1)
        # firstblock = torch.cat((onestest, zerotest), dim=1)
        firstblockout = patch1embed(zerotest).detach()
        if self.in_chans==1:
            testout = self.patch_embed(zerotest).detach()

        templist = []
        templist2 = []
        for i in range(L):
            if i == L:
                print(i)
                break
            if self.in_chans == 1:
                if (testout[0][0] == x[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)
            else:
                if (firstblockout[0][0] == argout[0][i]).all():
                    templist.append(i)
                else:
                    templist2.append(i)

        ones = torch.zeros((N, L), device=x.device)

        len_keep = int(L - mask_ratio)

        # for i in range(len(templist)):
        #     ones[0][templist[i]] = 0.0001 * i
        # if flag < 1 :
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.00001 * i
        # else:
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.00001 * i
        for i in range(len(templist2)):
            factor = torch.rand(1)
            ones[0][templist2[i]] = 0.1 * factor
        if len(templist2) >= 3:
            ones[0][templist2[2]] = 1
        else:
            pass
            # ones[0][templist2[-5]] = 1
        new_ids_shuffle = torch.argsort(ones, dim=1)

        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # 根据shuffle index 进行mask    (16, 49, 768)

        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def batchtwomask(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        zerotest = torch.zeros((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        onestest = torch.ones((N, 1, self.patch_size[0], self.patch_size[0], self.patch_size[0])).cuda().float()
        # 确保0channel 的pancreas像素表示为0，1channel  pancreas的像素表示为1，不然会出错。
        # test = torch.cat((zerotest, onestest), dim=1)
        firstblock = torch.cat((onestest, zerotest), dim=1)
        firstblockout = self.patch_embed(firstblock).detach()
        if self.in_chans==1:
            testout = self.patch_embed(zerotest).detach()

        templist = [[] for i in range(N)]
        templist2 = [[] for i in range(N)]

        for j in range(N):
            for i in range(L):
                if i == L:
                    print(i)
                    break
                if self.in_chans == 1:
                    if (testout[0][0] == x[j][i]).all():
                        templist[j].append(i)
                    else:
                        templist2[j].append(i)
                else:
                    if (firstblockout[j][0] == x[j][i]).all():
                        templist[j].append(i)
                    else:
                        templist2[j].append(i)

        ones = torch.zeros((N, L), device=x.device)

        len_keep = int(L - mask_ratio)

        flag = len_keep // 2
        k = 0
        # for i in range(len(templist)):
        #     ones[0][templist[i]] = 0.0001 * i
        # if flag < 1 :
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.00001 * i
        # else:
        #     for i in range(len(templist2)):
        #         ones[0][templist2[i]] = 0.00001 * i
        for j in range(N):
            factor = torch.rand(1)
            for i in range(len(templist2[j])):
                ones[j][templist2[j][i]] = 0.1 * factor
            ones[j][templist2[j][3]] = 1
            # ones[0][templist2[-5]] = 1

        new_ids_shuffle = torch.argsort(ones, dim=1)

        # 生成（N,L） 再0-1的噪声
        noise = ones  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # 根据shuffle index 进行mask    (16, 49, 768)

        # print(k)
        x = x + self.pos_embed[:, 1:, :]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, method):
        # embed patches (B, N, embed_dim)
        argx = x.detach().argmax(1).unsqueeze(1).float()
        x = self.patch_embed(x)
        # B,N,W,H,D = x.shape
        # x = x.reshape()
        # ipdb.set_trace()

        # add pos embed w/o cls token 对除了 cls token添加位置信息
        # x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio   # 将添加有位置信息的每一个patch块进行mask
        if method == "nums":
            x, mask, ids_restore = self.twomask(x, mask_ratio, argx)
        else:
            # print("ok")
            x, mask, ids_restore = self.myrandom_masking(x, mask_ratio, argx)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # expand 到batch size个cls token  ->(batch, 1, embedim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # 在位置的维度进行拼接
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # out (batch, nums_patch * mask_ratio + 1, embed_dim)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens    input x shape (16, 50, 768) ->output (16, 50, 512)
        x = self.decoder_embed(x)

        # append mask tokens to sequence self.mask_token(1,1,embed_dim)
        # ->mask_tokens (batch, num_patchs + 1 - nums_patch * mask_ratio + 1, 512) ->(16, 147, 512)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # 获取到无cls token 的信息
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle 回去重新排序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # 添加cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed (1, 197, 512)
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        # print(x.shape)
        x = self.decoder_norm(x)

        # predictor projection  还原回原来的pixel patch * patch * 3 ->(batch, num_patchs, pixel patch * patch * 3)
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        cls_token =x[:,:1,:]
        cls_token = cls_token.mean(-1)

        return x, cls_token

    def forward_loss(self, imgs, pred, mask,cls_token):
        """
        imgs: [N, 3, H, W]  (N, 2, H, W， D)
        pred: (N, L, patch_size**3 *2)
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        # ipdb.set_trace()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # all loss
        loss = loss.sum() / float(loss.shape[1] * loss.shape[0])
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches /只对出除了的patch计算loss
        return loss

    def forward(self, imgs, unpatch=True, mask_ratio=0.75, method ="nums"):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, method)
        pred, cls_token = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask,cls_token)
        if unpatch:
            pred = self.unpatchify(pred)
        return loss, pred, mask


class videoMae(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=96, patch_size=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=0, tubelet_size=2,
                 use_learnable_pos_emb=False, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics [4, 2, 128, 128, 128]
        self.in_chans = in_chans
        # self.patch_embed = myPatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = videoPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_frames=img_size, tubelet_size=patch_size)
        # num_patches = self.patch_embed.num_patches      # patch的个数 or 位置个数
        self.num_patches = self.patch_embed.num_patches   # patch的个数 or 位置个数
        self.patch_size = (patch_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 开头的token
        # 位置信息
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # mlp_ratio作用为对MLP的输入维度*4作为MLP输出维度
        # 24个block
        # self.blocks = nn.ModuleList([
        #     TransformerBlock(embed_dim, mlp_dim, num_heads)
        #     for i in range(depth)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            myBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        # self.blocks = nn.ModuleList([
        #     TransformerBlock(embed_dim, embed_dim * 4, num_heads)
        #     for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # decoder 的token

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        # 8个transformer block
        # self.decoder_blocks = nn.ModuleList([
        #     TransformerBlock(decoder_embed_dim, mlp_dim, decoder_num_heads)
        #     for i in range(decoder_depth)])
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
            myBlock(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(decoder_depth)])
        # self.decoder_blocks = nn.ModuleList([
        #     TransformerBlock(decoder_embed_dim, decoder_embed_dim * 4, decoder_num_heads)
        #     for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans)  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed ->197,768  +上token
        # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
        # embed_dim=8192 不能够被3整除,导致在position embed中出现了维度不一致 不能对position 进行初始化。
        # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.num_patches ** (1 / 3)),
        #                                     cls_token=True)

        pos_embed = get_sinusoid_encoding_table(self.num_patches, self.pos_embed.shape[-1])[0]
        pos_embed = np.concatenate([np.zeros([1, self.pos_embed.shape[-1]]), pos_embed], axis=0)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             math.ceil(self.num_patches ** (1 / 3)), cls_token=True)

        decoder_pos_embed = get_sinusoid_encoding_table(self.num_patches, self.decoder_pos_embed.shape[-1])[0]
        decoder_pos_embed = np.concatenate([np.zeros([1, self.decoder_pos_embed.shape[-1]]), decoder_pos_embed], axis=0)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 2, H, W， D)
        x: (N, L, patch_size**3 *2) L应该为patch 数量
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p
        # (batch, 2 8,16,8,16,8,16) 2为channel数， 8为个数， 16为patch size大小
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p, d, p))
        # (batch, 8,8,8,16,16,16,2)
        x = torch.einsum('nchpwqdf->nhwdpqfc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *2)
        imgs: (N, 2, H, W， D)
        """
        p = self.patch_size[0]
        h = w = d = math.ceil(x.shape[1] ** (1 / 3))
        assert h * w * d == x.shape[1]
        # (batch, 8, 8, 8, 16, 16, 16, 2) 2为channel数， 8为个数， 16为patch size大小
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, self.in_chans))
        # (batch, 2, 8, 16, 8, 16, 8, 16)
        x = torch.einsum('nhwdpqfc->nchpwqdf', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p, d * p))
        imgs = imgs.permute(0,1,3,4,2)
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # 生成（N,L） 再0-1的噪声
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # 根据shuffle index 进行mask    (16, 49, 768)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask  (16, 196)
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # 这个ids_REStore的作用 就是把mask当成noise 然后把mask按照#restore的位置排序  这样得到的mask就是一个  有mask的地方为1 没mask的地方为0的二维张量。

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # print("mask_ratio {}".format(mask_ratio))
        # embed patches (B, N, embed_dim)
        x = self.patch_embed(x)
        # B,N,W,H,D = x.shape
        # x = x.reshape()
        # ipdb.set_trace()

        # add pos embed w/o cls token 对除了 cls token添加位置信息
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio   # 将添加有位置信息的每一个patch块进行mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # expand 到batch size个cls token  ->(batch, 1, embedim)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # 在位置的维度进行拼接
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # out (batch, nums_patch * mask_ratio + 1, embed_dim)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens    input x shape (16, 50, 768) ->output (16, 50, 512)
        x = self.decoder_embed(x)

        # append mask tokens to sequence self.mask_token(1,1,embed_dim)
        # ->mask_tokens (batch, num_patchs + 1 - nums_patch * mask_ratio + 1, 512) ->(16, 147, 512)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # 获取到无cls token 的信息 将mask和visiable合并
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle 回去重新排序 获取原patch 顺序
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # 添加cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed (1, 197, 512)
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        # print(x.shape)
        x = self.decoder_norm(x)

        # predictor projection  还原回原来的pixel patch * patch * 3 ->(batch, num_patchs, pixel patch * patch * 3)
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]  (N, 2, H, W， D)
        pred: (N, L, patch_size**3 *2)
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        # ipdb.set_trace()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches /只对出除了的patch计算loss
        return loss

    def forward(self, imgs, unpatch, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        if unpatch:
            pred = self.unpatchify(pred)
        return loss, pred, mask

class Encoder(torch.nn.Module):
    # [16,32,64,128,256,512]
    def __init__(self, n_channels, dim, norm_type=2, n_fmaps=[8,16,32,64,128,256],soft=False):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)
        self.down5 = Down(n_fmaps[4], n_fmaps[5],norm_type=norm_type,soft=False)
        self.fc1 = torch.nn.Linear(16384, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc_mean = torch.nn.Linear(128, dim)
    def forward(self, x):
        #'pred_only','pred_recon',if_random=False
        #x = data_dict[in_key]
        # import pdb; pdb.set_trace()
        x = self.in_block(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = x.view(x.size(0),16384)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x_mean = self.fc_mean(x)
        x_mean = torch.sigmoid(x_mean)
        return x_mean

class Segmentation_GS(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv_GS(n_channels, n_fmaps[0],num_group=2,soft=False)
        self.down1 = Down_GS(n_fmaps[0], n_fmaps[1],num_group=2,soft=False)
        self.down2 = Down_GS(n_fmaps[1], n_fmaps[2],num_group=2,soft=False)
        self.down3 = Down_GS(n_fmaps[2], n_fmaps[3],num_group=4,soft=False)
        self.norm1 = GSNorm3d(n_fmaps[0],num_group=2)
        self.norm2 = GSNorm3d(n_fmaps[1],num_group=4)
        self.norm3 = GSNorm3d(n_fmaps[2],num_group=8)
        self.norm4 = GSNorm3d(n_fmaps[3],num_group=8)
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        self.up4 = torch.nn.Upsample(scale_factor=4, mode='trilinear')
        self.up8 = torch.nn.Upsample(scale_factor=8, mode='trilinear')

        self.out_block1 = Conv_GS(n_fmaps[0]+n_fmaps[1]+n_fmaps[2]+n_fmaps[3], 32 ,soft=False)
        self.out_block2 = torch.nn.Conv3d(32, n_class, 1, padding=0)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, data_dict,in_key,out_key):
        x = data_dict[in_key]
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1 = self.in_block(x)
        x1_norm = self.norm1(x1)
        x2 = self.down1(x1)
        x2_norm = self.up2(self.norm2(x2))
        x3 = self.down2(x2)
        x3_norm = self.up4(self.norm3(x3))
        x4 = self.down3(x3)
        x4_norm = self.up8(self.norm4(x4))
        x = torch.cat((x1_norm,x2_norm,x3_norm,x4_norm),dim=1)
        x = self.out_block1(x)
        x = self.out_block2(x)
        x = self.final(x)
        data_dict[out_key] = x
        return data_dict

class Segmentation(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)


        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.n_class = n_class
    def forward(self, data_dict,in_key="venous",out_key="venous_pancreas_pred", dropout=0.0,method="seg_train"):
        x = data_dict[in_key]
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up2(x5)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up3(x)+x3
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up4(x)+x2
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up5(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.out_block(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        if method == "seg_train" or method=="domain_adaptation" or method=="domain_adaptation_vit":
            x = self.final(x)
        elif method=="sdf_train":
            x = self.tanh(x)
        else:
            pass

        data_dict[out_key] = x
        return data_dict

class Segmentation_array(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)


        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, input, dropout=0.0):
        x = input
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up2(x5)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up3(x)+x3
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up4(x)+x2
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.up5(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.out_block(x)
        if dropout: x = torch.nn.functional.dropout(x, p=dropout, training=True)
        x = self.final(x)
        result = x
        return result
        
class Fusion(torch.nn.Module):

    # [16,32,64,128,256,512]
    def __init__(self, n_channels_img, n_channels_mask, n_class, norm_type=2, n_fmaps=[8,16,32,64,128,256]):
        super().__init__()
        #self.SConv3d = SConv3d(1,n_fmaps[0],3,padding=1,bias=True)
        self.in_block = Conv(n_channels_img, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1 = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)

        self.in_block_mask = Conv(n_channels_mask, n_fmaps[0],norm_type=norm_type,soft=False)
        self.down1_mask = Down(n_fmaps[0], n_fmaps[1],norm_type=norm_type,soft=False)
        self.merge = Conv(n_fmaps[1], n_fmaps[1],norm_type=norm_type,soft=False)
        self.down2 = Down(n_fmaps[1], n_fmaps[2],norm_type=norm_type,soft=False)
        self.down3 = Down(n_fmaps[2], n_fmaps[3],norm_type=norm_type,soft=False)
        self.down4 = Down(n_fmaps[3], n_fmaps[4],norm_type=norm_type,soft=False)

        self.up2 = Up(n_fmaps[4],n_fmaps[3],norm_type=norm_type,soft=False)
        self.up3 = Up(n_fmaps[3],n_fmaps[2],norm_type=norm_type,soft=False)
        self.up4 = Up(n_fmaps[2],n_fmaps[1],norm_type=norm_type,soft=False)
        self.up5 = Up(n_fmaps[1],n_fmaps[0],norm_type=norm_type,soft=False)
        self.out_block = torch.nn.Conv3d(n_fmaps[0], n_class, 3, padding=1)
        self.final = nn.Softmax(dim=1)
        self.n_class = n_class
    def forward(self, data_dict,in_key_img,in_key_mask,out_key):
        x_img = data_dict[in_key_img]
        x_mask = data_dict[in_key_mask]
        #input_res = data_dict.get(self.in_key2)
        #input_x = self.SConv3d(input_x)
        x1_img = self.in_block(x_img)
        x1_mask = self.in_block_mask(x_mask)
        x2_img = self.down1(x1_img)
        x2_mask = self.down1_mask(x1_mask)
        x2 = x2_img+x2_mask
        x2 = self.merge(x2)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up2(x5)
        x = self.up3(x)+x3
        x = self.up4(x)+x2
        x = self.up5(x)
        x = self.out_block(x)
        x = self.final(x)
        data_dict[out_key] = x
        return data_dict

class Joint(torch.nn.Module):
    def __init__(self, models, vae_forward_scale=0.0, vae_decoder_dropout=0.0, seg_dropout=0.0):
        super().__init__()
        self.Seg = models[0]
        self.Vae = models[1]
        self.vae_forward_scale = vae_forward_scale
        self.vae_decoder_dropout = vae_decoder_dropout
        self.seg_dropout = seg_dropout

    def forward(self, data_dict,in_key,out_key,out_key_recon,dropout=False):
        if dropout: data_dict = self.Seg(data_dict,in_key,out_key, dropout=self.seg_dropout)
        else: data_dict = self.Seg(data_dict,in_key,out_key)
        if dropout: data_dict[out_key_recon],_,_ = self.Vae(data_dict[out_key],if_random=False,scale=self.vae_forward_scale,dropout=self.vae_decoder_dropout)
        else:
            # data_dict[out_key] shape is torch.Size([1, 2, 128, 128, 128])
            # print("data_dict[out_key] shape is {}".format(data_dict[out_key].shape))
            data_dict[out_key_recon],data_dict["mean"],data_dict["std"] = self.Vae(data_dict[out_key],if_random=False,scale=self.vae_forward_scale)
        return data_dict


class Joint_MAE(torch.nn.Module):
    def __init__(self, models, vae_forward_scale=0.0, vae_decoder_dropout=0.0, seg_dropout=0.0):
        super().__init__()
        self.Seg = models[0]
        self.MAE = models[1]
        self.vae_forward_scale = vae_forward_scale
        self.vae_decoder_dropout = vae_decoder_dropout
        self.seg_dropout = seg_dropout

    def forward(self, data_dict,in_key,out_key,out_key_recon,dropout=False):
        if dropout: data_dict = self.Seg(data_dict,in_key,out_key, dropout=self.seg_dropout)
        else: data_dict = self.Seg(data_dict,in_key,out_key)
        if dropout: data_dict[out_key_recon],_,_ = self.Vae(data_dict[out_key],if_random=False,scale=self.vae_forward_scale,dropout=self.vae_decoder_dropout)
        else: data_dict[out_key_recon],data_dict["mean"],data_dict["std"] = self.Vae(data_dict[out_key],if_random=False,scale=self.vae_forward_scale)
        return data_dict


class Joint2(torch.nn.Module):
    def __init__(self, models, seg_dropout=0.0):
        super().__init__()
        self.Seg = models[0]
        self.Dis = models[1]
        self.seg_dropout = seg_dropout

    def forward(self, data_dict,in_key,out_key,score_key,dropout=False):
        if dropout: data_dict = self.Seg(data_dict,in_key,out_key, dropout=self.seg_dropout)
        else: data_dict = self.Seg(data_dict,in_key,out_key)
        data_dict[score_key] = self.Dis(data_dict[out_key][:,1:2,:,:,:])
        return data_dict


class Embed(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.Encoder = models[0]
        self.Vae = models[1]
        self.Fusion = models[2]
    def forward(self, data_dict,in_key,out_key,test_mode=False,loop_input=None,seg_input=None,latent_input=None):
        if latent_input:
            data_dict['latent_code'] = data_dict[latent_input]
        else:
            data_dict['latent_code'] = self.Encoder(data_dict[in_key])
        data_dict['gt_recon'],data_dict['latent_code_gt'],data_dict['latent_code_std'] = self.Vae(data_dict['venous_pancreas_only'],if_random=True,scale=0.5,mid_input=False)
        if loop_input:
            data_dict[loop_input],data_dict['latent_code_loop'],_ = self.Vae(data_dict[loop_input],if_random=False,scale=0,mid_input=False)
        if seg_input:
            data_dict['init_seg'] = data_dict[seg_input]
        else:
            data_dict['init_seg'] = self.Vae(data_dict['latent_code'],if_random=False,scale=0,mid_input=True)

        if loop_input:
            #T = 2*torch.sum(data_dict['venous_pancreas_only']*data_dict[loop_input],(2,3,4))/(torch.sum(data_dict[loop_input],(2,3,4))+torch.sum(data_dict['venous_pancreas_only'],(2,3,4)))
            #data_dict = self.Fusion(data_dict,in_key,loop_input,out_key,T[:,1].detach().view(-1,1,1,1,1))
            data_dict = self.Fusion(data_dict,in_key,loop_input,out_key)
        else:
            if test_mode:
                #T = 2*torch.sum(data_dict['venous_pancreas_only']*data_dict['init_seg'],(2,3,4))/(torch.sum(data_dict['init_seg'],(2,3,4))+torch.sum(data_dict['venous_pancreas_only'],(2,3,4)))
                data_dict = self.Fusion(data_dict,in_key,'init_seg',out_key)
            else:
                #T = 2*torch.sum(data_dict['venous_pancreas_only']*data_dict['gt_recon'],(2,3,4))/(torch.sum(data_dict['gt_recon'],(2,3,4))+torch.sum(data_dict['venous_pancreas_only'],(2,3,4)))
                data_dict = self.Fusion(data_dict,in_key,'gt_recon',out_key)
        data_dict['seg_recon'],_,_ = self.Vae(data_dict['init_seg'].detach(),if_random=False,scale=0,mid_input=False)
        #print(T[:,1].detach())
        return data_dict

# class Mae3D(nn.Module):
#     """ Masked Autoencoder with VisionTransformer backbone
#     """
#
#     def __init__(self, img_size=96, patch_size=16, in_chans=2,
#                  embed_dim=8192, depth=12, num_heads=8,
#                  decoder_embed_dim=5464, decoder_depth=8, decoder_num_heads=16,
#                  mlp_dim=32768, norm_layer=nn.LayerNorm, norm_pix_loss=False):
#         super().__init__()
#
#         # --------------------------------------------------------------------------
#         # MAE encoder specifics [4, 2, 128, 128, 128]
#         self.in_chans = in_chans
#         self.patch_embed = myPatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         # self.patch_embed = PatchEmbeddingBlock(img_size=(img_size, img_size, img_size), patch_size=(patch_size, patch_size, patch_size), in_channels=2,
#         #                                        hidden_size=embed_dim, num_heads=num_heads, pos_embed="conv")
#         # num_patches = self.patch_embed.num_patches      # patch的个数 or 位置个数
#         self.num_patches = (img_size // patch_size) ** 3   # patch的个数 or 位置个数
#         self.patch_size = (patch_size, patch_size, patch_size)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 开头的token
#         # 位置信息
#         # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
#                                       requires_grad=False)  # fixed sin-cos embedding
#         # mlp_ratio作用为对MLP的输入维度*4作为MLP输出维度
#         # 24个block
#         # self.blocks = nn.ModuleList([
#         #     TransformerBlock(embed_dim, mlp_dim, num_heads)
#         #     for i in range(depth)])
#
#         self.blocks = nn.ModuleList([
#             TransformerBlock(embed_dim, embed_dim * 4, num_heads)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#         # --------------------------------------------------------------------------
#
#         # --------------------------------------------------------------------------
#         # MAE decoder specifics
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
#
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # decoder 的token
#
#         # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
#                                               requires_grad=False)  # fixed sin-cos embedding
#         # 8个transformer block
#         # self.decoder_blocks = nn.ModuleList([
#         #     TransformerBlock(decoder_embed_dim, mlp_dim, decoder_num_heads)
#         #     for i in range(decoder_depth)])
#
#         self.decoder_blocks = nn.ModuleList([
#             TransformerBlock(decoder_embed_dim, decoder_embed_dim * 4, decoder_num_heads)
#             for i in range(decoder_depth)])
#         self.decoder_norm = norm_layer(decoder_embed_dim)
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 3 * in_chans, bias=True)  # decoder to patch
#         # --------------------------------------------------------------------------
#
#         self.norm_pix_loss = norm_pix_loss
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         # pos_embed ->197,768  +上token
#         # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
#         # embed_dim=8192 不能够被3整除,导致在position embed中出现了维度不一致 不能对position 进行初始化。
#         pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.num_patches ** (1 / 3)),
#                                             cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
#         decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
#                                                     math.ceil(self.num_patches ** (1 / 3)), cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def patchify(self, imgs):
#         """
#         imgs: (N, 2, H, W， D)
#         x: (N, L, patch_size**3 *2) L应该为patch 数量
#         """
#         p = self.patch_size[0]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#         h = w = d = imgs.shape[2] // p
#         # (batch, 2 8,16,8,16,8,16) 2为channel数， 8为个数， 16为patch size大小
#         x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p, d, p))
#         # (batch, 8,8,8,16,16,16,2)
#         x = torch.einsum('nchpwqdf->nhwdpqfc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * self.in_chans))
#         return x
#
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**3 *2)
#         imgs: (N, 2, H, W， D)
#         """
#         p = self.patch_size[0]
#         h = w = d = math.ceil(x.shape[1] ** (1 / 3))
#         assert h * w * d == x.shape[1]
#         # (batch, 8, 8, 8, 16, 16, 16, 2) 2为channel数， 8为个数， 16为patch size大小
#         x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, self.in_chans))
#         # (batch, 2, 8, 16, 8, 16, 8, 16)
#         x = torch.einsum('nhwdpqfc->nchpwqdf', x)
#         imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p, d * p))
#         return imgs
#
#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#
#         # 生成（N,L） 再0-1的噪声
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         # 对从小到大的index 再sort再获得index的index ->用于reshuffle 的index
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         # 根据shuffle index 进行mask    (16, 49, 768)
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask  (16, 196)
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#         # 这个ids_REStore的作用 就是把mask当成noise 然后把mask按照#restore的位置排序  这样得到的mask就是一个  有mask的地方为1 没mask的地方为0的二维张量。
#
#         return x_masked, mask, ids_restore
#
#     def forward_encoder(self, x, mask_ratio):
#         # print("mask_ratio {}".format(mask_ratio))
#         # embed patches (B, N, embed_dim)
#         x = self.patch_embed(x)
#         # B,N,W,H,D = x.shape
#         # x = x.reshape()
#         # ipdb.set_trace()
#
#         # add pos embed w/o cls token 对除了 cls token添加位置信息
#         x = x + self.pos_embed[:, 1:, :]
#
#         # masking: length -> length * mask_ratio   # 将添加有位置信息的每一个patch块进行mask
#         x, mask, ids_restore = self.random_masking(x, mask_ratio)
#
#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         # expand 到batch size个cls token  ->(batch, 1, embedim)
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         # 在位置的维度进行拼接
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)
#         # out (batch, nums_patch * mask_ratio + 1, embed_dim)
#         x = self.norm(x)
#
#         return x, mask, ids_restore
#
#     def forward_decoder(self, x, ids_restore):
#         # embed tokens    input x shape (16, 50, 768) ->output (16, 50, 512)
#         x = self.decoder_embed(x)
#
#         # append mask tokens to sequence self.mask_token(1,1,embed_dim)
#         # ->mask_tokens (batch, num_patchs + 1 - nums_patch * mask_ratio + 1, 512) ->(16, 147, 512)
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         # 获取到无cls token 的信息 将mask和visiable合并
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         # unshuffle 回去重新排序 获取原patch 顺序
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         # 添加cls token
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
#
#         # add pos embed (1, 197, 512)
#         x = x + self.decoder_pos_embed
#
#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         # print(x.shape)
#         x = self.decoder_norm(x)
#
#         # predictor projection  还原回原来的pixel patch * patch * 3 ->(batch, num_patchs, pixel patch * patch * 3)
#         x = self.decoder_pred(x)
#
#         # remove cls token
#         x = x[:, 1:, :]
#
#         return x
#
#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [N, 3, H, W]  (N, 2, H, W， D)
#         pred: (N, L, patch_size**3 *2)
#         mask: [N, L], 0 is keep, 1 is remove,
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6) ** .5
#         # ipdb.set_trace()
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#
#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches /只对出除了的patch计算loss
#         return loss
#
#     def forward(self, imgs, unpatch, mask_ratio=0.75):
#         latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
#         pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#         loss = self.forward_loss(imgs, pred, mask)
#         if unpatch:
#             pred = self.unpatchify(pred)
#         return loss, pred, mask