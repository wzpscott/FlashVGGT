import atexit
import logging
import os
from typing import Any, Dict, Optional, Union

import torch


def _get_rank() -> int:
    """Return the global distributed rank, defaulting to 0 outside a launcher."""
    try:
        return int(os.environ.get("RANK", 0))
    except (TypeError, ValueError):
        return 0


class WandbLogger:
    """Weights & Biases logger with distributed-training support.

    Only rank-0 initialises a W&B run; all other ranks are no-ops, so this
    class is safe to instantiate on every process without creating duplicate
    runs.

    The interface is intentionally identical to TensorBoardLogger so the two
    can be swapped in the trainer config without touching trainer code.

    Args:
        project:   W&B project name.
        run_name:  Display name for this run (shown in the W&B UI).
        config:    Flat or nested dict of hyper-parameters to attach to the run.
                   Typically the full Hydra config converted to a plain dict via
                   ``OmegaConf.to_container(cfg, resolve=True)``.
        dir:       Local directory where W&B stores its offline artefacts.
                   Defaults to the current working directory.
        tags:      Optional list of string tags for the run.
        notes:     Free-text description of the run.
        resume:    W&B resume mode ("allow", "must", "never", "auto").
                   Pass "allow" to resume a run that crashed mid-epoch.
        run_id:    Explicit W&B run-id to resume.  Only used when
                   ``resume != "never"``.
        mode:      "online" (default), "offline", or "disabled".  Use
                   "disabled" for dry-run unit tests.
        **kwargs:  Any additional keyword arguments are forwarded verbatim to
                   ``wandb.init``.
    """

    def __init__(
        self,
        project: str = "key3r",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: str = "allow",
        run_id: Optional[str] = None,
        mode: str = "online",
        **kwargs: Any,
    ) -> None:
        self._run = None
        self._rank = _get_rank()

        if self._rank == 0:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "wandb is not installed. Run `pip install wandb` or add it to requirements.txt."
                ) from exc

            init_kwargs: Dict[str, Any] = dict(
                project=project,
                name=run_name,
                config=config or {},
                dir=dir or os.getcwd(),
                tags=tags,
                notes=notes,
                resume=resume,
                mode=mode,
                **kwargs,
            )
            if run_id is not None:
                init_kwargs["id"] = run_id

            self._run = wandb.init(**init_kwargs)
            logging.info(
                f"W&B run initialised: {self._run.name}  url={self._run.url}"
            )
        else:
            logging.debug(
                f"WandbLogger: skipping init on rank {self._rank} (not rank 0)."
            )

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run(self):
        """The underlying ``wandb.Run`` object, or ``None`` on non-rank-0."""
        return self._run

    @property
    def url(self) -> Optional[str]:
        """URL of the W&B run dashboard, or ``None`` on non-rank-0."""
        return self._run.url if self._run else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finish the W&B run and flush remaining logs.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._run is not None:
            self._run.finish()
            self._run = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, name: str, data: Any, step: int) -> None:
        """Log a single scalar value.

        Args:
            name: Metric name (slashes are preserved as W&B section separators).
            data: Scalar value — ``float``, ``int``, or a 0-d ``torch.Tensor``.
            step: Global training step.
        """
        if self._run is None:
            return
        if isinstance(data, torch.Tensor):
            data = data.item()
        self._run.log({name: data}, step=step)

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        """Log multiple scalars in a single W&B call (more efficient than
        calling ``log`` in a loop because it produces one timeline point).

        Args:
            payload: Mapping of metric-name → scalar value.
            step:    Global training step.
        """
        if self._run is None:
            return
        scalar_payload = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in payload.items()
        }
        self._run.log(scalar_payload, step=step)

    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, Any],
        step: int,
        fps: int = 4,
    ) -> None:
        """Log image or video data to W&B.

        Args:
            name: Metric name shown in the W&B media panel.
            data: A NumPy array or ``torch.Tensor``.
                  - 3-D ``(C, H, W)`` → logged as a single image.
                  - 5-D ``(1, T, C, H, W)`` → logged as a video clip.
            step: Global training step.
            fps:  Frames-per-second for video logging.

        Raises:
            ValueError: If ``data`` has an unsupported number of dimensions.
        """
        if self._run is None:
            return

        import wandb
        import numpy as np

        if isinstance(data, torch.Tensor):
            if data.dtype == torch.bfloat16:
                data = data.to(torch.float16)
            data = data.cpu().numpy()

        if data.ndim == 3:
            # (C, H, W) → W&B Image expects (H, W, C) or (H, W)
            img = np.transpose(data, (1, 2, 0))
            # Rescale [-1,1] or [0,1] floats to uint8
            if img.dtype != np.uint8:
                img = np.clip((img + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
            self._run.log({name: wandb.Image(img)}, step=step)

        elif data.ndim == 5:
            # (B, T, C, H, W) — take first batch element
            clip = data[0]  # (T, C, H, W)
            if clip.dtype != np.uint8:
                clip = np.clip((clip + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
            # wandb.Video expects (T, H, W, C)
            clip = np.transpose(clip, (0, 2, 3, 1))
            try:
                self._run.log({name: wandb.Video(clip, fps=fps, format="gif")}, step=step)
            except Exception as exc:
                logging.warning(
                    f"WandbLogger.log_visuals: video logging failed ({exc}). "
                    "Install moviepy via `pip install wandb[media]` to enable it."
                )

        else:
            raise ValueError(
                f"Unsupported data dimensions: {data.ndim}. "
                "Expected 3-D (C,H,W) for images or 5-D (B,T,C,H,W) for videos."
            )

    def watch(self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100) -> None:
        """Hook W&B gradient/parameter histograms onto a model.

        Args:
            model:    The ``nn.Module`` to watch.
            log:      What to log — "gradients", "parameters", or "all".
            log_freq: Log every N backward passes.
        """
        if self._run is not None:
            import wandb
            wandb.watch(model, log=log, log_freq=log_freq)
