"""Training diagnostic plugins for algorithm-specific observability."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from stochaflow.diffusion import DDPM
from stochaflow.sampling import save_image_grid
from stochaflow.training.diagnostic_context import DiagnosticBuildContext
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES


def _first_tensor_from_batch(batch: Any) -> torch.Tensor | None:
    if isinstance(batch, torch.Tensor):
        return batch
    if (
        isinstance(batch, (tuple, list))
        and batch
        and isinstance(batch[0], torch.Tensor)
    ):
        return batch[0]
    return None


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else device.index]


def _with_optional_ema(
    model: DDPM,
    ema: ExponentialMovingAverage | None,
    *,
    enabled: bool,
):
    class _EMAContext:
        def __enter__(self) -> None:
            if ema is not None and enabled:
                ema.store(model)
                ema.copy_to(model)

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, exc, traceback
            if ema is not None and enabled:
                ema.restore(model)

    return _EMAContext()


@REGISTRIES.diagnostics.register("ddpm")
class DDPMDiagnosticLogger:
    """DDPM-specific training diagnostics backed by the experiment logger."""

    @classmethod
    def context_parameters(
        cls,
        context: DiagnosticBuildContext,
    ) -> dict[str, Any]:
        """Request the configured sampling shape from the runtime context."""

        return {"sample_shape": context.sample_shape}

    def __init__(
        self,
        *,
        logger: ExperimentLogger,
        output_dir: str | Path,
        sample_shape: Sequence[int],
        interval: int = 100,
        timestep_buckets: int = 10,
        sample_every_epochs: int = 5,
        sample_num: int = 16,
        sample_seed: int = 123,
        sample_grid_size: int = 4,
        reconstruction_every_epochs: int = 5,
        reconstruction_timesteps: Sequence[int] = (50, 250, 500, 900),
        use_ema_for_artifacts: bool = True,
    ) -> None:
        if interval <= 0:
            raise ValueError("ddpm diagnostic interval must be positive")
        if timestep_buckets <= 0:
            raise ValueError("ddpm diagnostic timestep_buckets must be positive")
        if sample_every_epochs <= 0:
            raise ValueError("ddpm diagnostic sample_every_epochs must be positive")
        if sample_num <= 0:
            raise ValueError("ddpm diagnostic sample_num must be positive")
        if sample_grid_size <= 0:
            raise ValueError("ddpm diagnostic sample_grid_size must be positive")
        if len(sample_shape) != 3 or any(int(value) <= 0 for value in sample_shape):
            raise ValueError(
                "ddpm diagnostic sample_shape must contain positive C, H, W"
            )
        if reconstruction_every_epochs <= 0:
            raise ValueError(
                "ddpm diagnostic reconstruction_every_epochs must be positive"
            )

        self.logger = logger
        self.output_dir = Path(output_dir) / "diagnostics" / "ddpm"
        self.interval = interval
        self.timestep_buckets = timestep_buckets
        self.sample_every_epochs = sample_every_epochs
        self.sample_num = sample_num
        self.sample_seed = sample_seed
        self.sample_grid_size = sample_grid_size
        self.sample_shape = tuple(int(value) for value in sample_shape)
        self.reconstruction_every_epochs = reconstruction_every_epochs
        self.reconstruction_timesteps = [
            int(value) for value in reconstruction_timesteps
        ]
        self.use_ema_for_artifacts = use_ema_for_artifacts
        self._last_clean_batch: torch.Tensor | None = None

    def on_train_batch_end(
        self,
        *,
        trainer: Any,
        batch: Any,
        output: Any,
        loss: float,
        global_step: int,
        epoch_index: int | None,
    ) -> None:
        del loss, epoch_index
        clean_batch = _first_tensor_from_batch(batch)
        if clean_batch is not None:
            self._last_clean_batch = clean_batch.detach().cpu()
        if global_step % self.interval != 0:
            return

        diagnostics = getattr(output, "diagnostics", {})
        timesteps = diagnostics.get("timesteps")
        per_sample_loss = diagnostics.get("per_sample_loss")
        if not isinstance(timesteps, torch.Tensor) or not isinstance(
            per_sample_loss, torch.Tensor
        ):
            return
        if not isinstance(trainer.model, DDPM):
            return

        metrics = self._bucket_loss_metrics(
            timesteps.detach().cpu(),
            per_sample_loss.detach().cpu(),
            num_timesteps=trainer.model.num_timesteps,
        )
        pred_noise = diagnostics.get("predicted_noise")
        target_noise = diagnostics.get("target_noise")
        if isinstance(pred_noise, torch.Tensor):
            pred_noise = pred_noise.detach().float().cpu()
            metrics["ddpm/pred_noise_mean"] = float(pred_noise.mean())
            metrics["ddpm/pred_noise_std"] = float(pred_noise.std())
        if isinstance(target_noise, torch.Tensor):
            target_noise = target_noise.detach().float().cpu()
            metrics["ddpm/target_noise_mean"] = float(target_noise.mean())
            metrics["ddpm/target_noise_std"] = float(target_noise.std())
        if metrics:
            self.logger.log_metrics(metrics, step=global_step)

    def _bucket_loss_metrics(
        self,
        timesteps: torch.Tensor,
        per_sample_loss: torch.Tensor,
        *,
        num_timesteps: int,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        width = max(
            1,
            (num_timesteps + self.timestep_buckets - 1) // self.timestep_buckets,
        )
        digits = max(3, len(str(num_timesteps)))
        for bucket_index in range(self.timestep_buckets):
            start = 1 + bucket_index * width
            end = min(num_timesteps, start + width - 1)
            if start > end:
                break
            mask = (timesteps >= start) & (timesteps <= end)
            if not bool(mask.any()):
                continue
            metrics[f"ddpm/loss_t_{start:0{digits}d}_{end:0{digits}d}"] = float(
                per_sample_loss[mask].mean()
            )
        return metrics

    def on_train_epoch_end(
        self,
        *,
        trainer: Any,
        epoch_index: int,
        metrics: dict[str, float],
    ) -> None:
        del metrics
        if not isinstance(trainer.model, DDPM):
            return
        if epoch_index % self.sample_every_epochs == 0:
            self._save_samples(trainer, epoch_index)
        if epoch_index % self.reconstruction_every_epochs == 0:
            self._save_reconstructions(trainer, epoch_index)

    def _save_samples(self, trainer: Any, epoch_index: int) -> None:
        model: DDPM = trainer.model
        device = trainer.device
        sample_shape = torch.Size((self.sample_num, *self.sample_shape))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad(), torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(self.sample_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.sample_seed)
            with _with_optional_ema(
                model,
                trainer.ema,
                enabled=self.use_ema_for_artifacts,
            ):
                model.eval()
                samples = model.sample(sample_shape, device=device).detach().cpu()

        tensor_path = self.output_dir / f"epoch_{epoch_index:04d}_samples.pt"
        grid_path = self.output_dir / f"epoch_{epoch_index:04d}_samples.png"
        torch.save(samples, tensor_path)
        save_image_grid(
            samples,
            grid_path,
            nrow=self.sample_grid_size,
            denormalize=True,
        )
        self.logger.log_text(
            "ddpm/sample_artifacts",
            f"samples={grid_path}; raw_samples={tensor_path}",
            step=trainer.global_step,
        )

    def _save_reconstructions(self, trainer: Any, epoch_index: int) -> None:
        if self._last_clean_batch is None:
            return
        model: DDPM = trainer.model
        device = trainer.device
        x0 = self._last_clean_batch[: self.sample_num].to(device)
        if x0.ndim != 4:
            return

        recon_rows: list[torch.Tensor] = []
        with torch.no_grad(), torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(self.sample_seed + epoch_index)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.sample_seed + epoch_index)
            with _with_optional_ema(
                model,
                trainer.ema,
                enabled=self.use_ema_for_artifacts,
            ):
                model.eval()
                for timestep in self.reconstruction_timesteps:
                    if not 1 <= timestep <= model.num_timesteps:
                        continue
                    timesteps = torch.full(
                        (x0.shape[0],),
                        timestep,
                        dtype=torch.long,
                        device=device,
                    )
                    xt, _ = model.add_noise(x0, timesteps)
                    pred_noise = model._predict_noise(xt, timesteps)
                    x0_hat = model._estimate_x0_from_epsilon(
                        xt,
                        timesteps,
                        predicted_noise=pred_noise,
                        clip_denoised=model.clip_denoised,
                    )
                    recon_rows.extend(
                        [
                            x0.detach().cpu(),
                            xt.detach().cpu(),
                            x0_hat.detach().cpu(),
                        ]
                    )
        if not recon_rows:
            return
        recon_grid = torch.cat(recon_rows, dim=0)
        grid_path = self.output_dir / f"epoch_{epoch_index:04d}_recon.png"
        tensor_path = self.output_dir / f"epoch_{epoch_index:04d}_recon.pt"
        torch.save(recon_grid, tensor_path)
        save_image_grid(
            recon_grid,
            grid_path,
            nrow=x0.shape[0],
            denormalize=True,
        )
        self.logger.log_text(
            "ddpm/reconstruction_artifacts",
            f"recon={grid_path}; raw_recon={tensor_path}",
            step=trainer.global_step,
        )
