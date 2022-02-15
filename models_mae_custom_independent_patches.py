# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.highres_patch_embed = PatchEmbed(patch_size*4, patch_size, in_chans, embed_dim)
        highres_num_patches = self.highres_patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # High-Res MAE decoder specifics
        self.highres_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.highres_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.highres_decoder_pos_embed = nn.Parameter(torch.zeros(1, highres_num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.highres_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.highres_decoder_norm = norm_layer(decoder_embed_dim)
        self.highres_decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        highres_decoder_pos_embed = get_2d_sincos_pos_embed(self.highres_decoder_pos_embed.shape[-1], int(self.highres_patch_embed.num_patches**.5), cls_token=True)
        self.highres_decoder_pos_embed.data.copy_(torch.from_numpy(highres_decoder_pos_embed).float().unsqueeze(0))

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
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def highres_patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.highres_patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
        

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def highres_unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.highres_patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # (128, 49)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        '''
        x.shape = (128, 3, 224, 224)
        mask_ratio = 0.75
        '''
        # embed patches
        x = self.patch_embed(x) # (128, 196, 768) (b, l, e)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] # (128, 196, 768)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # x_masked: shape = (128, 49, 768)
        # mask.shape = (128, 196) (0 is kept, 1 is removed)
        # ids_restore = (128, 196)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # (1, 1, 768)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) # (128, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1) # (128, 50, 768)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # (128, 50, 768)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        '''
        x.shape = (128, 50, 768)
        ids_restore.shape = (128, 196)
        '''
        # embed tokens
        x = self.decoder_embed(x) # (128, 50, 512)

        # append mask tokens to sequence
        # (1, 1, 512) --> (128, 147, 512)
        mask_tokens = self.mask_token.repeat(x.shape[0],
                                             ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token (128, 196, 512)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle (128, 196, 512)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token (128, 197, 512)

        # add pos embed
        x = x + self.decoder_pos_embed # (128, 197, 512)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x) # (128, 197, 512)

        # predictor projection
        x = self.decoder_pred(x) # (128, 197, 768)

        # remove cls token
        x = x[:, 1:, :] # (128, 196, 768)

        return x

    def forward_highres_decoder(self, x, ids_restore):
        # embed tokens
        x = self.highres_decoder_embed(x) # (128, 50, 512)

        # append mask tokens to sequence
        highres_mask_tokens = self.highres_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], highres_mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle (128, 196, 512)

        # x_ = torch.permute(x_, (0, 2, 1)) # (128, 512, 196)
        # x_ = torch.reshape(x_, (x_.shape[0], x_.shape[1], int(self.patch_embed.num_patches**.5), int(self.patch_embed.num_patches**.5))) # (128, 512, 14, 14)
        # x_ = torch.repeat_interleave(x_, 2, dim=2)
        # x_ = torch.repeat_interleave(x_, 2, dim=3)
        # x_ = torch.flatten(x_, 2) # (128, 512, 768)
        # x_ = torch.permute(x_, (0, 2, 1)) # (128, 768, 512)

        x_ = torch.reshape(x_, (x_.shape[0]*x_.shape[1], x_.shape[2])) # (128*196, 512)
        x_ = x_[:, None, :] # (128*196, 1, 512)
        x_ = torch.repeat_interleave(x_, self.highres_patch_embed.num_patches, dim=1) # (128*196, 196, 512)

        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x[:, :1, :] # (128, 1, 512)
        x = torch.squeeze(x) # (128, 512)

        x = torch.repeat_interleave(x, self.patch_embed.num_patches, dim=0) # (128*196, 512)
        x = x[:, None, :] # (128*196, 1, 512)

        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = torch.cat([x, x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.highres_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.highres_decoder_blocks:
            x = blk(x)
        x = self.highres_decoder_norm(x)

        # predictor projection
        x = self.highres_decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, highres_pred, mask, highres_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs) # (128, 196, 768)

        # highres_imgs = F.interpolate(imgs, size=self.img_size*2)
        # highres_target = self.highres_patchify(highres_imgs)
        highres_target = target.clone() # (128, 196, 768)
        highres_target = torch.reshape(highres_target, (target.shape[0]*target.shape[1], target.shape[2])) # (128*196, 768)
        highres_target = highres_target[:, None, :] # (128*196, 1, 768)
        highres_imgs = self.highres_unpatchify(highres_target) # (128*196, 3, 16, 16)
        highres_imgs = F.interpolate(highres_imgs, size=int(self.highres_patch_embed.num_patches**.5)*self.patch_size) # (128*196, 3, 32, 32)
        highres_target = self.highres_patchify(highres_imgs) # (128*196, 4, 768)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

            highres_mean = highres_target.mean(dim=-1, keepdim=True)
            highres_var = highres_target.var(dim=-1, keepdim=True)
            highres_target = (highres_target - highres_mean) / (highres_var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # loss = loss.sum() / torch.numel(loss)

        highres_loss = (highres_pred - highres_target) ** 2
        print('highres_loss.shape', highres_loss.shape)
        highres_loss = highres_loss.mean(dim=-1)
        print('highres_loss.shape', highres_loss.shape)
        highres_loss = (highres_loss * highres_mask).sum() / highres_mask.sum()
        print('highres_loss.shape', highres_loss.shape)
        # highres_loss = highres_loss.sum() / torch.numel(highres_loss)

        while True:
            continue

        return loss + highres_loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio) # (128, 196)
        # highres_mask = torch.repeat_interleave(mask, 4, dim=1)
        highres_mask = torch.reshape(mask, (mask.shape[0]*mask.shape[1], 1)) # (128*196, 1)
        highres_mask = torch.repeat_interleave(highres_mask, self.highres_patch_embed.num_patches, dim=1) # (128*196, 4)

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        highres_pred = self.forward_highres_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, highres_pred, mask, highres_mask)
        return loss, pred, mask, highres_pred, highres_mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
