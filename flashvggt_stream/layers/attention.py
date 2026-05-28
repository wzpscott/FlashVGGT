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
        cache_kv: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.kv_downfactor = kv_downfactor
        self.cache_kv = cache_kv

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        if self.cache_kv:
            self.register_buffer("k_cache", None)
            self.register_buffer("v_cache", None)

    def clear_kv_cache(self):
        pass

    def forward(
        self,
        x: Tensor,
        pos=None,
        pH: int = None,
        pW: int = None,
        patch_start_idx: int = None,
        is_first_chunk: bool = False,
        memory_drop_rate: int = 1,
        kv_cache=None,
    ) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.kv_downfactor > 1:
            if pH is None or pW is None or patch_start_idx is None:
                raise ValueError("pH, pW, and patch_start_idx must be provided when kv_downfactor > 1")

            def downscale_kv_tensor(tensor):
                B_t, H_t, N_seq, D_t = tensor.shape
                P = patch_start_idx + pH * pW
                S = N_seq // P
                
                tensor = tensor.view(B_t, H_t, S, P, D_t)
                prefix = tensor[:, :, :, :patch_start_idx, :]
                spatial = tensor[:, :, :, patch_start_idx:, :]
                
                spatial = spatial.reshape(B_t * H_t * S, pH, pW, D_t).permute(0, 3, 1, 2)
                
                new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor
                spatial = F.interpolate(
                    spatial.float(), size=(new_pH, new_pW), 
                    mode="nearest"
                ).to(spatial.dtype)
                
                spatial = spatial.permute(0, 2, 3, 1).reshape(B_t, H_t, S, -1, D_t)
                out = torch.cat([prefix, spatial], dim=3)
                if memory_drop_rate > 1:
                    out = out[:, :, ::memory_drop_rate, :, :]
                out = out.reshape(B_t, H_t, -1, D_t)
                if is_first_chunk:
                    reference = tensor[:, :, 0, patch_start_idx:, :]
                    reference = reference.reshape(B_t, H_t, -1, D_t)
                    out = torch.cat([reference, out], dim=2)
                return out
                
            k = downscale_kv_tensor(k)
            v = downscale_kv_tensor(v)
        elif self.kv_downfactor < 1:
            raise ValueError("kv_downfactor must be >= 1")

        if self.cache_kv:
            k_cache, v_cache = kv_cache if kv_cache is not None else (None, None)
            if k_cache is None:
                new_k_cache = k
                new_v_cache = v
            else:
                new_k_cache = torch.cat([k_cache.detach(), k], dim=2)
                new_v_cache = torch.cat([v_cache.detach(), v], dim=2)
            k = new_k_cache
            v = new_v_cache
            new_kv_cache = (new_k_cache, new_v_cache)
        else:
            new_kv_cache = None

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
        return x, new_kv_cache

class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, **kwargs) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
