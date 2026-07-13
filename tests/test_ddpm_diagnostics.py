"""Tests for DDPM-specific training diagnostics."""

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion import DDPM, LinearDDPMScheduler
from stochaflow.training.diagnostics import DDPMDiagnosticLogger
from stochaflow.training.trainer import TrainStepOutput
from stochaflow.utils.logging import ExperimentLogger


class RecordingLogger(ExperimentLogger):
    def __init__(self) -> None:
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.text: list[tuple[str, str, int | None]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.metrics.append((step, metrics))

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        self.text.append((tag, text, step))

    def close(self) -> None:
        return None


class ZeroDenoiser(nn.Module):
    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return torch.zeros_like(xt)


def test_ddpm_diagnostic_logs_bucket_metrics(tmp_path) -> None:
    logger = RecordingLogger()
    diagnostic = DDPMDiagnosticLogger(
        logger=logger,
        output_dir=tmp_path,
        interval=1,
        timestep_buckets=2,
    )
    model = DDPM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ZeroDenoiser(),
    )
    output = TrainStepOutput(
        loss=torch.tensor(0.0),
        diagnostics={
            "timesteps": torch.tensor([1, 5, 6, 10]),
            "per_sample_loss": torch.tensor([1.0, 3.0, 5.0, 7.0]),
            "predicted_noise": torch.zeros(4, 1, 4, 4),
            "target_noise": torch.ones(4, 1, 4, 4),
        },
    )

    diagnostic.on_train_batch_end(
        trainer=SimpleNamespace(model=model),
        batch=torch.zeros(4, 1, 4, 4),
        output=output,
        loss=0.0,
        global_step=1,
        epoch_index=1,
    )

    assert logger.metrics
    _, metrics = logger.metrics[0]
    assert metrics["ddpm/loss_t_001_005"] == 2.0
    assert metrics["ddpm/loss_t_006_010"] == 6.0
    assert metrics["ddpm/pred_noise_std"] == 0.0


def test_ddpm_diagnostic_writes_sample_and_reconstruction_artifacts(tmp_path) -> None:
    logger = RecordingLogger()
    diagnostic = DDPMDiagnosticLogger(
        logger=logger,
        output_dir=tmp_path,
        interval=1,
        sample_every_epochs=1,
        sample_num=2,
        sample_grid_size=2,
        reconstruction_every_epochs=1,
        reconstruction_timesteps=[1, 2],
    )
    model = DDPM(
        scheduler=LinearDDPMScheduler(num_timesteps=4),
        model=ZeroDenoiser(),
    )
    output = TrainStepOutput(
        loss=torch.tensor(0.0),
        diagnostics={
            "timesteps": torch.tensor([1, 2]),
            "per_sample_loss": torch.tensor([1.0, 1.0]),
        },
    )
    diagnostic.on_train_batch_end(
        trainer=SimpleNamespace(model=model),
        batch=torch.zeros(2, 1, 4, 4),
        output=output,
        loss=0.0,
        global_step=1,
        epoch_index=1,
    )

    trainer = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        ema=None,
        global_step=1,
    )
    diagnostic.on_train_epoch_end(trainer=trainer, epoch_index=1, metrics={})

    diagnostic_dir = tmp_path / "diagnostics" / "ddpm"
    assert (diagnostic_dir / "epoch_0001_samples.pt").exists()
    assert (diagnostic_dir / "epoch_0001_samples.png").exists()
    assert (diagnostic_dir / "epoch_0001_recon.pt").exists()
    assert (diagnostic_dir / "epoch_0001_recon.png").exists()
