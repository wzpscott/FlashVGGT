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
from flashvggt.utils.geometry_cuda import unproject_depth_map_to_point_map
from flashvggt.utils.pose_enc import pose_encoding_to_extri_intri


class FlashVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        img_size=518, 
        patch_size=14, 
        embed_dim=1024,
        enable_camera=True, 
        enable_point=False, 
        enable_depth=True, 
        enable_track=False, 
        kv_downfactor: int = 3,  # we change the default kv_downfactor to 3 for better performance in wide-baseline scenarios.
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

    def forward(self, images: torch.Tensor, return_points: bool = True):
        """
        Forward pass of the FlashVGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            return_points (bool): Whether to return world points.
                Default: True
        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast('cuda', enabled=False):
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

        if return_points:
            if "pose_enc" in predictions and "depth" in predictions:
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
                
                B, S = images.shape[:2]
                extrinsic_flat = extrinsic.flatten(0, 1)
                intrinsic_flat = intrinsic.flatten(0, 1)
                depth_flat = predictions["depth"].flatten(0, 1)
                
                world_points_flat = unproject_depth_map_to_point_map(depth_flat, extrinsic_flat, intrinsic_flat)
                
                predictions["world_points"] = world_points_flat.unflatten(0, (B, S))
                predictions["world_points_conf"] = predictions["depth_conf"]

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
        self.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Running on {device}")
    print(f"Using dtype: {dtype}")
    
    model_df1 = FlashVGGT(img_size=518, patch_size=14, embed_dim=1024, kv_downfactor=1).to(device)
    model_df1.eval()
    
    model_df4 = FlashVGGT(img_size=518, patch_size=14, embed_dim=1024, kv_downfactor=4).to(device)
    model_df4.eval()
    
    B, S, C, H, W = 1, 10, 3, 392, 518
    images = torch.rand(B, S, C, H, W).to(device)
    
    with torch.cuda.amp.autocast(dtype=dtype):
        with torch.no_grad():
            # Warmup
            print("Warming up...")
            for _ in range(2):
                model_df4(images)
                    
            # Test without downfactor
            print("Testing without kv_downfactor (kv_downfactor=1)...")
            start = time.time()
            out = model_df1(images)
            end = time.time()
            print(f"Time taken (kv_downfactor=1): {end - start:.4f} seconds")
            
            # Test with downfactor
            print("Testing with kv_downfactor=4...")
            start = time.time()
            out_df4 = model_df4(images)
            end = time.time()
            print(f"Time taken (kv_downfactor=4): {end - start:.4f} seconds")
    
    print("Test passed!")

