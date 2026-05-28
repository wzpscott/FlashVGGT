# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
from tqdm import trange

from flashvggt_stream.models.aggregator import Aggregator
from flashvggt_stream.heads.camera_head import CameraHead
from flashvggt_stream.heads.dpt_head import DPTHead


class FlashVGGTStream(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, 
        img_size=518, 
        patch_size=14, 
        embed_dim=1024,
        enable_camera=True, 
        enable_depth=True, 
        kv_downfactor: int = 3,
        chunk_sizes: list = [2, 4, 6, 12, 24]
    ):
        super().__init__()
        self.chunk_sizes = chunk_sizes

        self.aggregator = Aggregator(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            kv_downfactor=kv_downfactor
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None

    def forward(self, images: torch.Tensor, chunk_size: int = 12):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 100.
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
            
        B, S, C_in, H, W = images.shape
        if self.training:
            import random
            chunk_size = random.choice(self.chunk_sizes)
            chunk_size = min(chunk_size, S)
        else:
            if chunk_size is None or chunk_size > S:
                chunk_size = S
            
        predictions = {}
        all_pose_tokens = []
        all_depth = []
        all_depth_conf = []

        kv_cache_list = None

        pbar = trange(0, images.shape[1], chunk_size) if not self.training else range(0, images.shape[1], chunk_size)
        for i in pbar:
            is_first_chunk = (i == 0)
            chunk_images = images[:, i:i+chunk_size]
            aggregated_tokens_list, patch_start_idx, kv_cache_list = self.aggregator(
                chunk_images, is_first_chunk=is_first_chunk, kv_cache_list=kv_cache_list
            )

            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    all_pose_tokens.append(aggregated_tokens_list[-1][:, :, 0:1, :]) # [B, S, 1, C]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=chunk_images, patch_start_idx=patch_start_idx
                    )
                    if not self.training:
                        depth = depth.cpu()
                        depth_conf = depth_conf.cpu()
                    all_depth.append(depth)
                    all_depth_conf.append(depth_conf)

            if not self.training:
                pbar.set_description(f"Memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        if self.camera_head is not None:
            with torch.amp.autocast("cuda", enabled=False):
                pose_tokens = torch.cat(all_pose_tokens, dim=1) # [B, S, 1, C]
                pose_enc_list = self.camera_head([pose_tokens])
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

        if self.depth_head is not None:
            predictions["depth"] = torch.cat(all_depth, dim=1)
            predictions["depth_conf"] = torch.cat(all_depth_conf, dim=1)

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


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Running on {device}")
    print(f"Using dtype: {dtype}")
    
    model = FlashVGGTStream(img_size=518, patch_size=14, embed_dim=1024, kv_downfactor=4).to(device)
    model.eval()
    
    B, S, C, H, W = 1, 10, 3, 392, 518
    images = torch.rand(B, S, C, H, W).to(device)
    
    with torch.cuda.amp.autocast(dtype=dtype):
        with torch.no_grad():            
            # Test with chunking
            print("Testing stream inference (chunk_size=5)...")
            start = time.time()
            out = model(images, chunk_size=5)
            end = time.time()
            print(f"Time taken: {end - start:.4f} seconds")
            
            print(f"Output pose_enc shape: {out['pose_enc'].shape}")
            if 'depth' in out:
                print(f"Output depth shape: {out['depth'].shape}")
    
    print("Test passed!")

