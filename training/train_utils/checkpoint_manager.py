import logging
import os
from typing import Any, Dict, List, Optional

import torch
from iopath.common.file_io import g_pathmgr

from train_utils.general import safe_makedirs


class CheckpointManager:
    def __init__(self, checkpoint_conf, accelerator, model, optims):
        self.checkpoint_conf = checkpoint_conf
        self.accelerator = accelerator
        self.model = model
        self.optims = optims

    def load_resuming_checkpoint(
        self,
        ckpt_path: str,
        *,
        strict: Optional[bool] = None,
        load_optimizer: bool = True,
        reset_training_state: bool = False,
        log_prefix: str = "Resuming training",
    ) -> Dict[str, Any]:
        """
        Loads a checkpoint and returns the restored state (steps, data_epoch, time_elapsed).

        Args:
            ckpt_path: Path to ``.pt`` file.
            strict: Passed to ``load_state_dict``. If ``None``, uses ``checkpoint_conf.strict``.
            load_optimizer: If False, optimizer state in the file is ignored (e.g. pretrained backbone).
            reset_training_state: If True, return step/data_epoch/time_elapsed as zeros after loading
                weights (e.g. init from VGGT then train from scratch).
            log_prefix: First words of the log line (e.g. "Loading pretrained weights").
        """
        if strict is None:
            strict = self.checkpoint_conf.strict
        logging.info("%s from %s (rank %s)", log_prefix, ckpt_path, self.accelerator.process_index)
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        model = self.accelerator.unwrap_model(self.model)
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(
            model_state_dict,
            strict=strict,
        )
        if self.accelerator.is_main_process:
            logging.info(
                "Model state loaded (strict=%s). Missing keys: %s. Unexpected keys: %s.",
                strict,
                missing or "None",
                unexpected or "None",
            )

        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None and self.optims:
                if not isinstance(optimizer_state, list):
                    optimizer_state = [optimizer_state]
                for optim_wrapper, state in zip(self.optims, optimizer_state):
                    optim_wrapper.optimizer.load_state_dict(state)

        if reset_training_state:
            return {"step": 0, "data_epoch": 0, "time_elapsed": 0.0}

        loaded_steps = checkpoint.get("steps")
        if isinstance(loaded_steps, dict) and "train" in loaded_steps:
            step = loaded_steps["train"]
        else:
            resumed_iteration = checkpoint.get(
                "iteration",
                checkpoint.get("epoch", checkpoint.get("prev_iteration", checkpoint.get("prev_epoch", 0))),
            )
            step = int(resumed_iteration)

        data_epoch = checkpoint.get("data_epoch", step)
        ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        return {
            "step": step,
            "data_epoch": data_epoch,
            "time_elapsed": ckpt_time_elapsed,
        }

    def save_checkpoint(
        self,
        iteration: int,
        data_epoch: int,
        time_elapsed: float,
        checkpoint_names: Optional[List[str]] = None,
    ):
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(iteration) % self.checkpoint_conf.save_freq == 0
                and (int(iteration) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(iteration)}")

        checkpoint_content = {
            "iteration": int(iteration),
            "prev_iteration": int(iteration),
            # Backward-compatible fields for older checkpoints/loaders.
            "epoch": int(iteration),
            "prev_epoch": int(iteration),
            "data_epoch": int(data_epoch),
            "steps": {"train": int(iteration)},
            "time_elapsed": time_elapsed,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
            "model": self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if len(checkpoint_content["optimizer"]) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]

        if self.accelerator.is_main_process:
            for ckpt_name in checkpoint_names:
                checkpoint_path = os.path.join(checkpoint_folder, f"{ckpt_name}.pt")
                logging.info("Saving checkpoint at iteration %s to %s", iteration, checkpoint_path)
                self.accelerator.save(checkpoint_content, checkpoint_path)
        self.accelerator.wait_for_everyone()
