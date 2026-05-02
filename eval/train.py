from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm
from accelerate.utils import DistributedDataParallelKwargs, set_seed

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import os.path as osp
import random
import numpy as np
import wandb
import time
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler

from flash_vggt.utils.geometry import unproject_depth_map_to_point_map
from flash_vggt.utils.pose_enc import pose_encoding_to_extri_intri

from data.dataloader_utils import InfiniteLoopingSampler

# # --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# # Enables asynchronous error handling for NCCL, which can prevent hangs.
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

@hydra.main(version_base=None, config_path="./train_configs", config_name="flash_v1.yaml")
def train(cfg: DictConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, gradient_as_bucket_view=True)
    accelerator = Accelerator(
        project_dir=cfg.log_dir,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        log_with=["wandb"]
    )
    set_seed(cfg.seed, device_specific=False)
    
    wandb_init_kwargs = {"name": cfg.exp_name, "dir": cfg.log_dir}
    if cfg.resume_iter:
        wandb_init_kwargs["resume"] = "must"
        wandb_init_kwargs["id"] = cfg.wandb_id
        
    accelerator.init_trackers(
        cfg.project, config=dict(cfg), init_kwargs={"wandb": wandb_init_kwargs})
    wandb_tracker = accelerator.get_tracker("wandb")
    
    # create logging directory
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(osp.join(cfg.checkpoint_dir, "weights"), exist_ok=True)
    os.makedirs(osp.join(cfg.checkpoint_dir, "states"), exist_ok=True)
    
    # prepare data
    train_dataset = ConcatDataset(
        [instantiate(dataset, **cfg.data.training.common) for dataset in cfg.data.training.datasets]
    )
    val_dataset = ConcatDataset(
        [instantiate(dataset, **cfg.data.evaluation.common) for dataset in cfg.data.evaluation.datasets]
    )
    
    accelerator.print(
        f"[Data Info] Use data from {len(train_dataset.datasets)} datasets with total length {len(train_dataset)}. "
        f"Select {len(val_dataset)} samples for validation."
    )

    sampler = InfiniteLoopingSampler(train_dataset, shuffle=True, seed=cfg.seed + accelerator.process_index)
    # train_loader = DataLoader(
    #     train_dataset, batch_size=1, sampler=sampler, num_workers=cfg.num_workers)
    train_loader = DataLoader(
        train_dataset, batch_size=1, sampler=sampler, num_workers=cfg.num_workers)

    # set up models
    model = instantiate(cfg.model)
    model.train()
    if cfg.ckpt_path and not cfg.resume_iter:
        model.load_ckpt(cfg.ckpt_path)

    if cfg.freeze_keys:
        for name, param in model.named_parameters():
            for freeze_key in cfg.freeze_keys:
                if freeze_key in name:
                    param.requires_grad = False

    accelerator.print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2e} M | "
                f"Params with grad: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2e} M")

    # set up optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.lr * accelerator.num_processes, weight_decay=cfg.optimizer.weight_decay)
    scheduler = ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=int(cfg.iterations * 0.05 * accelerator.num_processes)),
        CosineAnnealingLR(optimizer, T_max=int(cfg.iterations * 0.95 * accelerator.num_processes), eta_min=1e-8)
    ])

    # set up loss
    loss_fn = instantiate(cfg.loss)

    # prepare for training
    model, train_loader, optimizer, scheduler, loss_fn = accelerator.prepare(
        model, train_loader, optimizer, scheduler, loss_fn)
    accelerator.register_for_checkpointing(scheduler)
    current_iter = 1

    if cfg.resume_iter:
        current_iter = cfg.resume_iter
        resume_path = osp.join(cfg.checkpoint_dir, "states", f"{current_iter:06d}")
        accelerator.load_state(resume_path)
        # accelerator.skip_first_batches(train_loader, current_iter)
    
    accelerator.print(f"Start training...")
    train_loader_iter = iter(train_loader)
    start_time = time.time()
    for i_iter in range(current_iter, cfg.iterations + 1):
        batch_data = next(train_loader_iter)
        for key in batch_data:
            if batch_data[key].shape[0] == 1:
                batch_data[key] = batch_data[key].squeeze(0)

        model_preds = model(batch_data["images"])

        loss_dict = loss_fn(model_preds, batch_data)
        
        optimizer.zero_grad()
        accelerator.backward(loss_dict["objective"])
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        if i_iter % cfg.log_freq == 0:
            accelerator.print(
                f"[Iter {i_iter}/{cfg.iterations}] "
                f"Camera loss: {loss_dict['loss_camera'].item():.3e} | "
                f"Depth loss: {loss_dict['loss_reg_depth'].item() + loss_dict['loss_grad_depth'].item():.3e} | "
                # f"Point loss: {loss_dict['loss_reg_point'].item() + loss_dict['loss_grad_point'].item():.3e} | "
                f"Total loss: {loss_dict['objective'].item():.3e} | "
                f"LR: {scheduler.get_last_lr()[0]:.3e} | "
                f"Memory: {torch.cuda.max_memory_allocated(0)/1024/1024/1024:.1f} GB | "
                f"ETA: {((time.time() - start_time) / (i_iter + 1) * cfg.iterations) / 3600:.1f} h"
            )
            torch.cuda.reset_peak_memory_stats()

            accelerator.log({
                f"loss/{k}": v.item() for k, v in loss_dict.items()
            }, step=i_iter)
            accelerator.log({
                "optim/lr": scheduler.get_last_lr()[0],
                "optim/where": (i_iter + 1) / cfg.iterations
            }, step=i_iter)

        del batch_data, model_preds, loss_dict

        if i_iter % cfg.val_freq == 0:
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            model.eval()

            accelerator.print(f"[Iter {i_iter}/{cfg.iterations}] Start validation...")
            if accelerator.is_main_process:
                for i in range(len(val_dataset)):
                    batch_images = val_dataset[i]["images"].to(accelerator.device).squeeze(0)
                    
                    points, colors, conf = eval_sequence(model, batch_images)
                    wandb_tracker.log({
                        f"val_depth/sequence_{i:03d}": wandb.Object3D(np.concatenate([points, colors], axis=-1)),
                    }, step=i_iter)

                    # log the images for one time
                    if i_iter == cfg.val_freq:
                        batch_images = (batch_images * 255.0).cpu().numpy().astype(np.uint8)

                        wandb_tracker.log({
                            f"val_images/sequence_{i:03d}": [wandb.Image(batch_images[i].transpose(1, 2, 0)) for i in range(batch_images.shape[0])],
                        }, step=i_iter)

                del batch_images, points, colors, conf

            model.train()
            torch.cuda.empty_cache()

        if i_iter % cfg.save_ckpt_freq == 0:
            accelerator.save_model(model, osp.join(cfg.checkpoint_dir, "weights", f"{i_iter:06d}"))

        if i_iter % cfg.save_state_freq == 0:
            # accelerator.save_model(model, osp.join(cfg.checkpoint_dir, "checkpoint_latest"))
            accelerator.save_state(osp.join(cfg.checkpoint_dir, "states", f"{i_iter:06d}"))

    accelerator.end_training()

@torch.no_grad()
def eval_sequence(model, batch_images, max_log_points=250_000, conf_threshold=10.0):
    pred_dict = model(batch_images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred_dict["pose_enc"], batch_images.shape[-2:])
    pred_dict["extrinsic"] = extrinsic
    pred_dict["intrinsic"] = intrinsic

    for key in pred_dict.keys():
        if isinstance(pred_dict[key], torch.Tensor):
            pred_dict[key] = pred_dict[key].cpu().numpy().squeeze(0)

    images = pred_dict["images"]
    depth_map = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]
    extrinsic = pred_dict["extrinsic"]
    intrinsic = pred_dict["intrinsic"]

    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    conf = depth_conf

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # Filter points based on confidence
    threshold_val = np.percentile(conf_flat, conf_threshold)
    conf_mask = (conf_flat >= threshold_val) & (conf_flat > 0.1)
    points = points[conf_mask]
    colors_flat = colors_flat[conf_mask]

    if len(points) > max_log_points:
        indices = np.random.choice(len(points), max_log_points, replace=False)
        points = points[indices]
        colors_flat = colors_flat[indices]

    return points, colors_flat, conf_flat

if __name__ == "__main__":
    train()