# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block


from utils.pos_embed import get_3d_sincos_pos_embed


from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import ensure_tuple_rep, optional_import
from typing import Sequence, Type, Union
from monai.networks.layers import Conv, trunc_normal_
from torch.nn import LayerNorm
class PatchEmbed(nn.Module):
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
        image_size: Union[Sequence[int], int] = 2,
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

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size[0] / patch_size[0]) * (image_size[1] / patch_size[1]) * (image_size[2] / patch_size[2])
        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

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
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
        return x


class Mae3D768(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=2,
                 embed_dim=768, depth=12, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_dim=8192, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics [4, 2, 128, 128, 128]
        # self.patch_embed = PatchEmbed(image_size=(img_size, img_size, img_size), patch_size=(patch_size, patch_size, patch_size), in_chans=2, embed_dim=768)
        self.patch_embed = PatchEmbeddingBlock(img_size=(128, 128, 128), patch_size=(16, 16, 16), in_channels=2, hidden_size=embed_dim, num_heads=num_heads, pos_embed="conv")
        # num_patches = self.patch_embed.num_patches      # patch的个数 or 位置个数
        self.num_patches = 512      # patch的个数 or 位置个数
        self.patch_size = (patch_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))     # 开头的token
        # 位置信息
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # mlp_ratio作用为对MLP的输入维度*4作为MLP输出维度
        # 24个block
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, mlp_dim, num_heads)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))    # decoder 的token

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # 8个transformer block
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, mlp_dim, num_heads)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        # self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed ->197,768  +上token
        # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
        # embed_dim=8192 不能够被3整除,导致在position embed中出现了维度不一致 不能对position 进行初始化。
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], math.ceil(self.num_patches**(1 / 3)), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], math.ceil(self.patch_embed.num_patches**(1 / 3)), cls_token=True)
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], math.ceil(self.num_patches**(1 / 3)), cls_token=True)
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p
        # (batch, 2 8,16,8,16,8,16) 2为channel数， 8为个数， 16为patch size大小
        x = imgs.reshape(shape=(imgs.shape[0], 2, h, p, w, p, d, p))
        # (batch, 8,8,8,16,16,16,2)
        x = torch.einsum('nchpwqdf->nhwdpqfc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * 2))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**3 *2)
        imgs: (N, 2, H, W, D)
        """
        p = self.patch_size[0]
        h = w = d = math.ceil(x.shape[1]**(1 / 3))
        assert h * w * d == x.shape[1]
        # (batch, 8, 8, 8, 16, 16, 16, 2) 2为channel数， 8为个数， 16为patch size大小
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 2))
        # (batch, 2, 8, 16, 8, 16, 8, 16)
        x = torch.einsum('nhwdpqfc->nchpwqdf', x)
        imgs = x.reshape(shape=(x.shape[0], 2, h * p, h * p, d * p))
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

    def forward_encoder(self, x, mask_ratio):
        # embed patches (B, N, embed_dim)
        x = self.patch_embed(x)

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
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

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


# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

#
# # set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

