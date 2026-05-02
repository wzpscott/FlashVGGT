import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCompressor(nn.Module):
    def __init__(self, dim: int, kv_downfactor: int, compressor_type: str = "dwconv"):
        super().__init__()
        self.kv_downfactor = kv_downfactor
        self.compressor_type = compressor_type
        
        if self.kv_downfactor > 1 and self.compressor_type == "dwconv":
            self.compressor = nn.Conv2d(
                dim, dim, kernel_size=self.kv_downfactor, stride=self.kv_downfactor, groups=dim, bias=False
            )
            nn.init.constant_(self.compressor.weight, 1.0 / (self.kv_downfactor * self.kv_downfactor))
        else:
            self.compressor = None

    def forward(self, spatial: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        if self.compressor_type == "dwconv":
            return self.compressor(spatial)
        else:
            new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor
            align_corners = False if self.compressor_type == "bilinear" else None
            spatial = F.interpolate(
                spatial.float(), size=(new_pH, new_pW), 
                mode=self.compressor_type, align_corners=align_corners
            ).to(spatial.dtype)
            return spatial

    def compress_pos(self, spatial_pos: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        new_pH, new_pW = pH // self.kv_downfactor, pW // self.kv_downfactor
        spatial_pos_float = spatial_pos.float()
        spatial_pos_float = F.interpolate(spatial_pos_float, size=(new_pH, new_pW), mode='nearest')
        return spatial_pos_float.to(spatial_pos.dtype)

