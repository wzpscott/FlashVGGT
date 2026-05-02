# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import math
import os
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from hydra.utils import instantiate

from train_utils.checkpoint_manager import CheckpointManager
from train_utils.freeze import freeze_modules
from train_utils.general import (
    copy_data_to_device,
    get_resume_checkpoint,
    model_summary,
    safe_makedirs,
    set_seeds,
)
from train_utils.logging import setup_logging
from train_utils.metrics_tracker import MetricsTracker
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


class Trainer:
    """Accelerate-based trainer for Key3R fine-tuning."""

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_iterations: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        cuda: Optional[Dict[str, Any]] = None,
        limit_train_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accelerate: Optional[Dict[str, Any]] = None,
        resolved_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim
        self.accelerate_conf = accelerate or {}
        self.resolved_config = resolved_config

        self.max_iterations = max_iterations
        self.mode = mode
        self.limit_train_batches = limit_train_batches
        self.seed_value = seed_value
        self.where = 0.0
        self.device_type = device

        self.accelerator = Accelerator(
            cpu=device == "cpu",
            mixed_precision=self.accelerate_conf.get(
                "mixed_precision",
                self._infer_mixed_precision_mode(),
            ),
            project_dir=self.logging_conf.log_dir,
        )
        self.device = self.accelerator.device
        self.rank = self.accelerator.process_index
        self.distributed_rank = self.rank
        self.local_rank = self.accelerator.local_process_index

        self._setup_cuda_backend(cuda)

        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        __import__("logging").info(
            "Accelerate initialised on rank %s/%s using device %s",
            self.rank,
            self.accelerator.num_processes,
            self.device,
        )

        set_seeds(seed_value, self.max_iterations, self.distributed_rank)

        self._setup_components()
        self._setup_dataloaders()
        self._prepare_training_components()

        self.checkpoint_manager = CheckpointManager(
            checkpoint_conf=self.checkpoint_conf,
            accelerator=self.accelerator,
            model=self.model,
            optims=self.optims,
        )

        resume_path = self.checkpoint_conf.resume_checkpoint_path
        vggt_path = getattr(self.checkpoint_conf, "vggt_checkpoint_path", None)

        if resume_path is not None:
            # Full training resume: strict load; ignore VGGT path.
            state = self.checkpoint_manager.load_resuming_checkpoint(
                resume_path,
                strict=True,
                load_optimizer=True,
                reset_training_state=False,
            )
            self._apply_resumed_state(state)
        elif vggt_path is not None:
            state = self.checkpoint_manager.load_resuming_checkpoint(
                vggt_path,
                strict=False,
                load_optimizer=False,
                reset_training_state=True,
                log_prefix="Loading pretrained weights",
            )
            self._apply_resumed_state(state)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                state = self.checkpoint_manager.load_resuming_checkpoint(
                    ckpt_path,
                    strict=self.checkpoint_conf.strict,
                    load_optimizer=True,
                    reset_training_state=False,
                )
                self._apply_resumed_state(state)

        self.accelerator.wait_for_everyone()

    def _apply_resumed_state(self, state: Dict[str, Any]):
        self.step = state["step"]
        self.data_epoch = state["data_epoch"]
        self.ckpt_time_elapsed = state["time_elapsed"]

    def _setup_timers(self):
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_cuda_backend(self, cuda_conf: Optional[Dict[str, Any]]) -> None:
        if not torch.cuda.is_available() or cuda_conf is None:
            return
        torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
        torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
        torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

    def _infer_mixed_precision_mode(self) -> str:
        amp_conf = getattr(self.optim_conf, "amp", None)
        if amp_conf is None or not amp_conf.enabled:
            return "no"
        amp_type = str(amp_conf.amp_dtype).lower()
        if amp_type == "bfloat16":
            return "bf16"
        if amp_type == "float16":
            return "fp16"
        raise ValueError(f"Unsupported AMP dtype: {amp_type}")

    def _get_console_log_freq(self) -> int:
        return max(1, int(getattr(self.logging_conf, "console_log_freq", self.logging_conf.log_freq)))

    def _should_log_console(self, step: int) -> bool:
        return step % self._get_console_log_freq() == 0

    def _setup_components(self):
        logging.info("Setting up components: model, loss, logger, and clipping.")
        self.data_epoch = 0
        self.step = 0

        writer_conf = self.logging_conf.get("writer", None)
        if writer_conf is None:
            writer_conf = self.logging_conf.get("tensorboard_writer", None)
        if writer_conf is not None:
            writer_conf = dict(writer_conf)
        self.logger = instantiate(writer_conf, _recursive_=False) if writer_conf is not None else None

        self.metrics_tracker = MetricsTracker(
            logging_conf=self.logging_conf,
            device=self.device,
            logger=self.logger,
            max_iterations=self.max_iterations,
        )

        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)

        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                "[Start] Freezing modules: %s on rank %s",
                self.optim_conf.frozen_module_names,
                self.distributed_rank,
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                "[Done] Freezing modules: %s on rank %s",
                self.optim_conf.frozen_module_names,
                self.distributed_rank,
            )

        if self.accelerator.is_main_process:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info("Model summary saved to %s", model_summary_path)

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
        self.train_dataset.seed = self.seed_value

    def _scale_learning_rate_for_world_size(self) -> None:
        if self.optim_conf is None:
            return

        world_size = int(self.accelerator.num_processes)
        if world_size <= 1:
            return

        scale_factor = float(world_size)

        optimizer_conf = getattr(self.optim_conf, "optimizer", None)
        if optimizer_conf is not None and hasattr(optimizer_conf, "lr"):
            base_lr = optimizer_conf.lr
            if isinstance(base_lr, (int, float)):
                optimizer_conf.lr = float(base_lr) * scale_factor
                logging.info(
                    "Scaled optimizer lr for world size %s: %.8g -> %.8g",
                    world_size,
                    float(base_lr),
                    float(optimizer_conf.lr),
                )

        lr_schedule_conf = getattr(self.optim_conf, "lr_schedule", None)
        if lr_schedule_conf is not None:
            scaled_scheduler_fields = 0
            if hasattr(lr_schedule_conf, "min_lr") and isinstance(
                lr_schedule_conf.min_lr, (int, float)
            ):
                lr_schedule_conf.min_lr = float(lr_schedule_conf.min_lr) * scale_factor
                scaled_scheduler_fields += 1
            if hasattr(lr_schedule_conf, "warmup_start_lr") and isinstance(
                lr_schedule_conf.warmup_start_lr,
                (int, float),
            ):
                lr_schedule_conf.warmup_start_lr = (
                    float(lr_schedule_conf.warmup_start_lr) * scale_factor
                )
                scaled_scheduler_fields += 1
        else:
            scaled_scheduler_fields = 0

        if scaled_scheduler_fields:
            logging.info(
                "Scaled %s lr schedule value(s) by world size factor %.0f.",
                scaled_scheduler_fields,
                scale_factor,
            )

    def _prepare_training_components(self):
        self.model.to(self.device)
        self.loss.to(self.device)

        self._scale_learning_rate_for_world_size()
        self.optims = construct_optimizers(
            self.model,
            self.optim_conf,
            max_iterations=self.max_iterations,
        )

        if self.optims:
            optimizers = [optim.optimizer for optim in self.optims]
            prepared = self.accelerator.prepare(self.model, *optimizers)
            self.model = prepared[0]
            prepared_optimizers = prepared[1:]
            for optim_wrapper, prepared_optimizer in zip(self.optims, prepared_optimizers):
                optim_wrapper.optimizer = prepared_optimizer
        else:
            self.model = self.accelerator.prepare(self.model)

        if self.gradient_clipper is not None and self.mode == "train":
            self.gradient_clipper.setup_clipping(self.model)

    def run(self):
        assert self.mode == "train", f"Invalid mode: {self.mode}. Only 'train' is supported."
        self.run_train()

        if self.logger is not None:
            self.logger.close()

    def _get_infinite_batches(self):
        while True:
            set_seeds(
                self.seed_value + self.data_epoch * 100,
                self.max_iterations,
                self.distributed_rank,
            )
            train_loader = self.train_dataset.get_loader(epoch=int(self.data_epoch))
            for batch in train_loader:
                yield batch
            
            del train_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            self.data_epoch += 1

    def run_train(self):
        self.metrics_tracker.build_loss_meters(self.gradient_clipper)
        self.metrics_tracker.setup_progress_meter()

        self.model.train()
        batch_iterator = self._get_infinite_batches()
        end = time.time()

        while self.step < self.max_iterations:
            self.metrics_tracker.data_time.update(time.time() - end)
            batch = next(batch_iterator)

            batch = self._prepare_batch_for_model(batch)

            self._run_step_on_batch(batch)

            self.where = float(self.step) / max(1, self.max_iterations)
            self._step_schedulers()
            self._log_optimizer_state()

            current_iteration = self.step
            if (
                self.checkpoint_conf.save_freq > 0
                and current_iteration % self.checkpoint_conf.save_freq == 0
            ):
                self.checkpoint_manager.save_checkpoint(
                    iteration=current_iteration,
                    data_epoch=self.data_epoch,
                    time_elapsed=self.metrics_tracker.time_elapsed_meter.val,
                )

            self.metrics_tracker.batch_time.update(time.time() - end)
            end = time.time()
            self.metrics_tracker.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            if torch.cuda.is_available():
                self.metrics_tracker.mem.update(torch.cuda.max_memory_allocated() / 1e9)

            if self._should_log_console(current_iteration - 1):
                self.metrics_tracker.progress.display(current_iteration - 1)

        self.metrics_tracker.log_train_summary(step=self.step)
        self.checkpoint_manager.save_checkpoint(
            iteration=self.step,
            data_epoch=self.data_epoch,
            time_elapsed=self.metrics_tracker.time_elapsed_meter.val,
            checkpoint_names=["checkpoint"],
        )
        return True

    def _prepare_batch_for_model(self, batch: Mapping) -> Mapping:
        batch = self._process_batch(batch)
        return copy_data_to_device(batch, self.device, non_blocking=True)

    def _run_step_on_batch(
        self,
        batch: Mapping[str, Any],
    ):
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        with self.accelerator.autocast():
            loss_dict = self._step(batch, self.model)

        loss = loss_dict["objective"]
        batch_size = batch["images"].shape[0]
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Loss is {loss.item()}, stopping training")

        self.metrics_tracker.loss_meters["Loss/train_loss_objective"].update(loss.item(), batch_size)
        self.accelerator.backward(loss)

        if self.gradient_clipper is not None:
            grad_norm_dict = self.gradient_clipper(model=self.model)
            for key, grad_norm in grad_norm_dict.items():
                meter_key = f"Grad/{key}"
                if meter_key in self.metrics_tracker.loss_meters:
                    self.metrics_tracker.loss_meters[meter_key].update(grad_norm)

        for optim in self.optims:
            optim.optimizer.step()

    def _step_schedulers(self):
        for optim in self.optims:
            optim.step_schedulers(self.where)

    def _log_optimizer_state(self):
        if self.logger is None or not self.metrics_tracker._should_log_backend(self.step):
            return
        payload = {}
        for i, optim in enumerate(self.optims):
            for j, param_group in enumerate(optim.optimizer.param_groups):
                optim_prefix = (
                    f"{i}_"
                    if len(self.optims) > 1
                    else (f"{j}_" if len(optim.optimizer.param_groups) > 1 else "")
                )
                if "lr" in param_group:
                    payload[os.path.join("Optim", f"{optim_prefix}", "lr")] = param_group["lr"]
                if "weight_decay" in param_group:
                    payload[os.path.join("Optim", f"{optim_prefix}", "weight_decay")] = (
                        param_group["weight_decay"]
                    )
        payload[os.path.join("Optim", "where")] = self.where
        if payload:
            self.logger.log_dict(payload, self.step)

    def _process_batch(self, batch: Mapping):
        has_world_points = "world_points" in batch and batch["world_points"] is not None
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = (
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch.get("cam_points"),
                world_points=batch["world_points"] if has_world_points else None,
                depths=batch.get("depths"),
                scale_by_points=has_world_points,
                point_masks=batch["point_masks"],
            )
        )

        batch["extrinsics"] = normalized_extrinsics
        if normalized_cam_points is not None:
            batch["cam_points"] = normalized_cam_points
        if normalized_world_points is not None:
            batch["world_points"] = normalized_world_points
        if normalized_depths is not None:
            batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module):
        y_hat = model(images=batch["images"])
        loss_dict = self.loss(y_hat, batch)

        log_data = {**batch, **y_hat, **loss_dict}
        log_data["loss_objective"] = loss_dict["objective"]

        self.metrics_tracker.update_and_log_scalars(log_data, self.step)
        self.metrics_tracker.log_visuals(log_data, self.step)

        self.step += 1
        
        # Detach loss_dict to prevent memory leaks
        detached_loss_dict = {k: v.detach() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        detached_loss_dict["objective"] = loss_dict["objective"] # Keep objective attached for backward
        
        return detached_loss_dict
