# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Union

import hydra
import torch
import torch.nn as nn


class OptimizerWrapper:
    """Wrap a torch optimizer and optional native LR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step_schedulers(self, where: float = 0.0) -> None:
        """Kept for trainer compatibility; native scheduler ignores `where`."""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_schedule_conf: Any,
    max_iterations: int,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if lr_schedule_conf is None:
        return None

    schedule_type = str(getattr(lr_schedule_conf, "type", "constant")).lower()
    if schedule_type == "constant":
        return None
    if schedule_type != "cosine":
        raise ValueError(
            f"Unsupported lr_schedule.type '{schedule_type}'. "
            "Supported values: 'constant', 'cosine'."
        )

    warmup_ratio = float(getattr(lr_schedule_conf, "warmup_ratio", 0.0))
    if warmup_ratio < 0.0 or warmup_ratio >= 1.0:
        raise ValueError(
            f"lr_schedule.warmup_ratio must be in [0, 1), got {warmup_ratio}."
        )

    warmup_start_lr = float(getattr(lr_schedule_conf, "warmup_start_lr", 1e-8))
    min_lr = float(getattr(lr_schedule_conf, "min_lr", 0.0))
    if warmup_start_lr < 0.0:
        raise ValueError(
            f"lr_schedule.warmup_start_lr must be non-negative, got {warmup_start_lr}."
        )
    if min_lr < 0.0:
        raise ValueError(f"lr_schedule.min_lr must be non-negative, got {min_lr}.")

    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not base_lrs:
        return None
    reference_base_lr = base_lrs[0]
    if reference_base_lr <= 0.0:
        raise ValueError(
            f"optimizer lr must be positive to build lr schedule, got {reference_base_lr}."
        )

    warmup_steps = int(math.floor(max_iterations * warmup_ratio))
    if warmup_ratio > 0.0:
        warmup_steps = max(1, warmup_steps)

    cosine_steps = max(1, int(max_iterations) - warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=min_lr,
    )

    if warmup_steps == 0:
        return cosine

    start_factor = max(1e-12, warmup_start_lr / reference_base_lr)
    if start_factor > 1.0:
        raise ValueError(
            "lr_schedule.warmup_start_lr must be <= optimizer lr. "
            f"Got warmup_start_lr={warmup_start_lr} and lr={reference_base_lr}."
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def construct_optimizer(
    model: nn.Module,
    optimizer_conf: Any,
    lr_schedule_conf: Optional[Any] = None,
    max_iterations: int = 0,
) -> OptimizerWrapper:
    optimizer = hydra.utils.instantiate(
        optimizer_conf,
        model.parameters(),
    )
    lr_scheduler = _build_lr_scheduler(
        optimizer=optimizer,
        lr_schedule_conf=lr_schedule_conf,
        max_iterations=max(1, int(max_iterations)),
    )
    return OptimizerWrapper(optimizer=optimizer, lr_scheduler=lr_scheduler)


def construct_optimizers(
    model: nn.Module,
    optim_conf: Any,
    max_iterations: int,
) -> Union[list[OptimizerWrapper], None]:
    """Convenience wrapper producing a single OptimizerWrapper list."""
    if optim_conf is None:
        return None

    optimizer = construct_optimizer(
        model=model,
        optimizer_conf=optim_conf.optimizer,
        lr_schedule_conf=getattr(optim_conf, "lr_schedule", None),
        max_iterations=max_iterations,
    )
    return [optimizer]
