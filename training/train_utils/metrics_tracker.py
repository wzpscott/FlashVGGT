import logging
import torch
import torchvision
from typing import Dict, List, Mapping, Any, Optional

from train_utils.general import AverageMeter, DurationMeter, ProgressMeter


class MetricsTracker:
    def __init__(self, logging_conf: Any, device: torch.device, logger: Optional[Any], max_iterations: int):
        self.logging_conf = logging_conf
        self.device = device
        self.logger = logger
        self.max_iterations = max_iterations
        
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")
        self.loss_meters: Dict[str, AverageMeter] = {}
        self.batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        self.data_time = AverageMeter("Data Time", self.device, ":.4f")
        self.mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        self.progress: Optional[ProgressMeter] = None

    def _get_scalar_log_keys(self) -> List[str]:
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log["train"].keys_to_log
        return []

    def _get_backend_log_freq(self) -> int:
        return max(1, int(getattr(self.logging_conf, "backend_log_freq", self.logging_conf.log_freq)))

    def _should_log_backend(self, step: int) -> bool:
        return step % self._get_backend_log_freq() == 0

    def build_loss_meters(self, gradient_clipper: Optional[Any] = None) -> Dict[str, AverageMeter]:
        loss_names = [f"Loss/train_{name}" for name in self._get_scalar_log_keys()]
        self.loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}
        
        if gradient_clipper is not None:
            for config in gradient_clipper.configs:
                param_names = ",".join(config["module_names"])
                self.loss_meters[f"Grad/{param_names}"] = AverageMeter(
                    f"Grad/{param_names}",
                    self.device,
                    ":.4f",
                )
        return self.loss_meters

    def setup_progress_meter(self):
        self.progress = ProgressMeter(
            num_batches=self.max_iterations,
            meters=[self.batch_time, self.data_time, self.mem, self.time_elapsed_meter, *self.loss_meters.values()],
            real_meters={},
            prefix="Train Iter:",
        )

    def update_and_log_scalars(self, data: Mapping, step: int):
        keys_to_log = self._get_scalar_log_keys()
        batch_size = data["extrinsics"].shape[0]
        backend_payload = {}

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                self.loss_meters[f"Loss/train_{key}"].update(value, batch_size)
                if self.logger is not None and self._should_log_backend(step):
                    backend_payload[f"Values/train/{key}"] = value
        if backend_payload:
            self.logger.log_dict(backend_payload, step)

    def log_train_summary(self, step: int) -> None:
        summary = {
            "batch_time_avg": self.batch_time.average,
            "data_time_avg": self.data_time.average,
            "time_elapsed_sec": self.time_elapsed_meter.val,
            "memory_gb_avg": self.mem.average,
        }
        for meter_name, meter in self.loss_meters.items():
            summary[f"{meter_name}_avg"] = meter.average
        logging.info("Iteration %s train summary: %s", step, summary)

        if self.logger is not None:
            iteration_payload = {f"Iteration/train/{name}": value for name, value in summary.items()}
            self.logger.log_dict(iteration_payload, step)

    def log_visuals(self, batch: Mapping, step: int) -> None:
        if self.logger is None:
            return
        if not (
            self.logging_conf.log_visuals
            and ("train" in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency["train"] > 0
            and (step % self.logging_conf.log_visual_frequency["train"] == 0)
        ):
            return

        if hasattr(self.logger, "run") and self.logger.run is not None:
            import wandb
            import numpy as np
            import torch.nn.functional as F
            
            payload = {}
            if "images" in batch:
                # batch["images"][0] is [S, 3, H, W] in [0, 1]
                imgs_tensor = batch["images"][0][:8]
                S, _, H, W = imgs_tensor.shape
                imgs = imgs_tensor.detach().cpu().float().numpy()
                imgs = np.transpose(imgs, (0, 2, 3, 1)) # [S, H, W, 3]
                imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
                
                # Draw keypoints on the images if norm_coords are available
                if "norm_coords" in batch:
                    coords = batch["norm_coords"][0].detach().cpu().float().numpy() # [S, K, 2]
                    for s in range(S):
                        xs = ((coords[s, :, 0] + 1) / 2.0 * (W - 1)).astype(int)
                        ys = ((coords[s, :, 1] + 1) / 2.0 * (H - 1)).astype(int)
                        for x, y in zip(xs, ys):
                            if 0 <= x < W and 0 <= y < H:
                                # Draw a 3x3 green square for each keypoint
                                y_min, y_max = max(0, y-1), min(H, y+2)
                                x_min, x_max = max(0, x-1), min(W, x+2)
                                imgs[s, y_min:y_max, x_min:x_max] = [0, 255, 0]
                                
                # Convert back to tensor [S, 3, H, W] for make_grid
                imgs_tensor_grid = torch.from_numpy(imgs).permute(0, 3, 1, 2)
                grid = torchvision.utils.make_grid(imgs_tensor_grid, nrow=4)
                grid_np = grid.permute(1, 2, 0).numpy()
                payload["Visuals/images"] = wandb.Image(grid_np)
                
            if "world_points" not in batch and ("depth" in batch or "depths" in batch) and "extrinsics" in batch and "intrinsics" in batch:
                from flashvggt.utils.geometry import unproject_depth_map_to_point_map
                depth_map = batch.get("depth", batch.get("depths"))[0]
                extrinsics_cam = batch["extrinsics"][0]
                intrinsics_cam = batch["intrinsics"][0]
                pts_np = unproject_depth_map_to_point_map(
                    depth_map.detach().cpu().numpy(),
                    extrinsics_cam.detach().cpu().numpy(),
                    intrinsics_cam.detach().cpu().numpy()
                )
                batch["world_points"] = torch.from_numpy(pts_np).to(depth_map.device).unsqueeze(0)

            if "world_points" in batch and "images" in batch and "point_masks" in batch:
                pts = batch["world_points"][0] # [S, H, W, 3]
                imgs_tensor = batch["images"][0] # [S, 3, H, W]
                mask = batch["point_masks"][0] # [S, H, W]
                
                # Flatten and filter by mask
                pts_flat = pts[mask] # [N, 3]
                colors = imgs_tensor.permute(0, 2, 3, 1)[mask] # [N, 3]
                colors = torch.clamp(colors * 255.0, 0, 255)
                
                pts_with_colors = torch.cat([pts_flat, colors], dim=-1) # [N, 6]
                
                # Subsample if too many points
                N = pts_with_colors.shape[0]
                max_points = 200_000
                if N > max_points:
                    indices = torch.randperm(N, device=pts_with_colors.device)[:max_points]
                    pts_with_colors = pts_with_colors[indices]
                    
                pts_with_colors = pts_with_colors.detach().cpu().float().numpy()
                
                payload["Visuals/dense_points"] = wandb.Object3D(pts_with_colors)
            
            if payload:
                self.logger.run.log(payload, step=step)
