# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        kv_downfactor: int = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.kv_downfactor = kv_downfactor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self,
        x: Tensor,
        pos=None,
        pH: int = None,
        pW: int = None,
        patch_start_idx: int = None,
        keyframe_indices: Tensor = None,
    ) -> Tensor:
        B, N, C = x.shape

        if self.kv_downfactor == 1:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            pos_k = pos
        elif self.kv_downfactor > 1:
            if pH is None or pW is None or patch_start_idx is None:
                raise ValueError("pH, pW, and patch_start_idx must be provided when kv_downfactor > 1")

            # 1. Slice weights for Q and KV
            q_weight = self.qkv.weight[:C, :]
            q_bias = self.qkv.bias[:C] if self.qkv.bias is not None else None
            
            kv_weight = self.qkv.weight[C:, :]
            kv_bias = self.qkv.bias[C:] if self.qkv.bias is not None else None
            
            # 2. Compute Q at full resolution
            q = F.linear(x, q_weight, q_bias)
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            # 3. Downsample x for K and V
            def downscale_tensor(tensor):
                B_t, N_seq, C_t = tensor.shape
                P = patch_start_idx + pH * pW
                S = N_seq // P
                
                tensor = tensor.view(B_t, S, P, C_t)
                prefix = tensor[:, :, :patch_start_idx, :]
                spatial = tensor[:, :, patch_start_idx:, :]
                
                spatial = spatial.reshape(B_t * S, pH, pW, C_t).permute(0, 3, 1, 2)
                
                new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor
                spatial = F.interpolate(
                    spatial.float(), size=(new_pH, new_pW), 
                    mode="bilinear", align_corners=False
                ).to(spatial.dtype)
                
                spatial = spatial.permute(0, 2, 3, 1).reshape(B_t, S, -1, C_t)
                out = torch.cat([prefix, spatial], dim=2).reshape(B_t, -1, C_t)
                if keyframe_indices is not None:
                    B_idx = torch.arange(B_t, device=tensor.device).unsqueeze(1)
                    reference = tensor[B_idx, keyframe_indices, patch_start_idx:, :]
                    reference = reference.reshape(B_t, -1, C_t)
                    out = torch.cat([reference, out], dim=1)
                return out
                
            x_kv = downscale_tensor(x)
            
            # 4. Compute K and V at lower resolution
            kv = F.linear(x_kv, kv_weight, kv_bias)
            N_kv = x_kv.shape[1]
            kv = kv.reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

            pos_k = pos
            if pos is not None:
                def downscale_pos(pos_tensor):
                    B_pos, N_seq, pos_dim = pos_tensor.shape
                    P = patch_start_idx + pH * pW
                    S = N_seq // P
                    
                    pos_tensor = pos_tensor.view(B_pos, S, P, pos_dim)
                    prefix_pos = pos_tensor[:, :, :patch_start_idx, :]
                    spatial_pos = pos_tensor[:, :, patch_start_idx:, :]
                    
                    spatial_pos = spatial_pos.reshape(B_pos * S, pH, pW, pos_dim).permute(0, 3, 1, 2)
                    
                    new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor
                    spatial_pos_float = spatial_pos.float()
                    spatial_pos_float = F.interpolate(spatial_pos_float, size=(new_pH, new_pW), mode='nearest')
                    spatial_pos = spatial_pos_float.to(spatial_pos.dtype)
                    
                    spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(B_pos, S, -1, pos_dim)
                    
                    out_pos = torch.cat([prefix_pos, spatial_pos], dim=2).reshape(B_pos, -1, pos_dim)
                    if keyframe_indices is not None:
                        B_idx = torch.arange(B_pos, device=pos_tensor.device).unsqueeze(1)
                        reference = pos_tensor[B_idx, keyframe_indices, patch_start_idx:, :]
                        reference = reference.reshape(B_pos, -1, pos_dim)
                        out_pos = torch.cat([reference, out_pos], dim=1)
                    return out_pos
                
                pos_k = downscale_pos(pos)
        else:
            raise ValueError("kv_downfactor must be >= 1")

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos_k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        pos=None,
        pH: int = None,
        pW: int = None,
        patch_start_idx: int = None,
        keyframe_indices: Tensor = None,
    ) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(
                x,
                pos=pos,
                pH=pH,
                pW=pW,
                patch_start_idx=patch_start_idx,
                keyframe_indices=keyframe_indices,
            )

        B, N, C = x.shape

        if self.kv_downfactor == 1:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = unbind(qkv, 2)
        elif self.kv_downfactor > 1:
            if pH is None or pW is None or patch_start_idx is None:
                raise ValueError("pH, pW, and patch_start_idx must be provided when kv_downfactor > 1")

            q_weight = self.qkv.weight[:C, :]
            q_bias = self.qkv.bias[:C] if self.qkv.bias is not None else None
            
            kv_weight = self.qkv.weight[C:, :]
            kv_bias = self.qkv.bias[C:] if self.qkv.bias is not None else None
            
            q = F.linear(x, q_weight, q_bias)
            q = q.reshape(B, N, self.num_heads, C // self.num_heads)
            
            def downscale_tensor(tensor):
                B_size, N_seq, C_t = tensor.shape
                P = patch_start_idx + pH * pW
                S = N_seq // P
                
                tensor = tensor.view(B_size, S, P, C_t)
                prefix = tensor[:, :, :patch_start_idx, :]
                spatial = tensor[:, :, patch_start_idx:, :]

                spatial = spatial.reshape(B_size * S, pH, pW, C_t).permute(0, 3, 1, 2)

                new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor
                spatial = F.interpolate(
                    spatial.float(), size=(new_pH, new_pW), 
                    mode="bilinear", align_corners=False
                ).to(spatial.dtype)

                spatial = spatial.permute(0, 2, 3, 1).reshape(B_size, S, -1, C_t)
                out = torch.cat([prefix, spatial], dim=2)
                out = out.reshape(B_size, -1, C_t)
                if keyframe_indices is not None:
                    B_idx = torch.arange(B_size, device=tensor.device).unsqueeze(1)
                    reference = tensor[B_idx, keyframe_indices, patch_start_idx:, :]
                    reference = reference.reshape(B_size, -1, C_t)
                    out = torch.cat([reference, out], dim=1)
                return out

            x_kv = downscale_tensor(x)
            
            kv = F.linear(x_kv, kv_weight, kv_bias)
            N_kv = x_kv.shape[1]
            kv = kv.reshape(B, N_kv, 2, self.num_heads, C // self.num_heads)
            k, v = unbind(kv, 2)
        else:
            raise ValueError("kv_downfactor must be >= 1")

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
