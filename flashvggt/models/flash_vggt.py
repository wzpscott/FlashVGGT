# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from flashvggt.models.aggregator import Aggregator
from flashvggt.heads.camera_head import CameraHead
from flashvggt.heads.dpt_head import DPTHead
from flashvggt.heads.track_head import TrackHead


class FlashVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        img_size=518, 
        patch_size=14, 
        embed_dim=1024,
        enable_camera=True, 
        enable_depth=True, 
        kv_downfactor: int = 4, 
        keyframe_every: int = 200
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            kv_downfactor=kv_downfactor, 
            keyframe_every=keyframe_every
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None

    def forward(self, images: torch.Tensor):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def load_ckpt(self, ckpt_path):
        def _checkpoint_to_state_dict(ckpt: object) -> dict:
            if not isinstance(ckpt, dict):
                return ckpt  # type: ignore[return-value]
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                return ckpt["model"]
            return ckpt

        ckpt = torch.load(ckpt_path)
        state_dict = _checkpoint_to_state_dict(ckpt)
        self.load_state_dict(state_dict, strict=True)