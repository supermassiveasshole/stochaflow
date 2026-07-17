"""Shared fixtures and test doubles for diagnostic tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion import GaussianDiffusion
from stochaflow.training import (
    FitStartEvent,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainStepOutput,
)
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES


class RecordingLogger(ExperimentLogger):
    def __init__(self) -> None:
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.text: list[tuple[str, str, int | None]] = []
        self.images: list[tuple[str, Path, int, str | None]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        self.metrics.append((step, {key: float(value) for key, value in metrics.items()}))

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        self.text.append((tag, text, step))

    def log_image(
        self,
        tag: str,
        path: str | Path,
        *,
        step: int,
        caption: str | None = None,
    ) -> None:
        self.images.append((tag, Path(path), step, caption))

    def close(self) -> None:
        return None


class ZeroDenoiser(nn.Module):
    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return torch.zeros_like(xt)


class TinyDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return self.projection(xt)


class RecordingSampler(GaussianDiffusion):
    records: dict[str, list[torch.Tensor]] = {}

    def __init__(self, noise_schedule, model: nn.Module, *, marker: str) -> None:
        super().__init__(noise_schedule=noise_schedule, model=model)
        self.marker = marker

    def sample_from_noise(self, initial_noise: torch.Tensor) -> torch.Tensor:
        self.records.setdefault(self.marker, []).append(
            initial_noise.detach().cpu().clone()
        )
        return initial_noise.clamp(-1.0, 1.0)


class FailingSampler(GaussianDiffusion):
    def sample_from_noise(self, initial_noise: torch.Tensor) -> torch.Tensor:
        del initial_noise
        raise RuntimeError("sampling failed")


REGISTRIES.diffusions.add("test_recording_diagnostic", RecordingSampler)
REGISTRIES.diffusions.add("test_failing_diagnostic", FailingSampler)


def profiles(*, trajectory: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "id": "ddpm_full",
            "name": "ddpm",
            "params": {},
            "trajectory": {
                "enabled": trajectory,
                "params": {"state_interval": 1} if trajectory else {},
                "gif_fps": 4,
            },
        },
        {
            "id": "ddim_2",
            "name": "ddim",
            "params": {"num_inference_steps": 2, "eta": 0.0},
            "trajectory": {
                "enabled": trajectory,
                "params": {"step_interval": 1} if trajectory else {},
                "gif_fps": 4,
            },
        },
    ]


def provider_config(
    *,
    step_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "step_metrics": (
            step_metrics
            if step_metrics is not None
            else [
                {"name": "timestep_bucket_loss", "params": {"buckets": 2}},
                {"name": "noise_alignment", "params": {}},
                {
                    "name": "x0_reconstruction",
                    "params": {"timesteps": [1, 2]},
                },
            ]
        ),
        "sampler_metrics": [
            {"name": "sample_statistics", "params": {}},
            {"name": "sampling_performance", "params": {}},
        ],
        "denoiser_artifacts": [
            {
                "name": "reconstruction_panel",
                "params": {"timesteps": [1, 2], "max_samples": 2},
            }
        ],
        "sampler_artifacts": [
            {"name": "sample_grid", "params": {"nrow": 2}},
            {"name": "trajectory", "params": {"nrow": 2}},
        ],
    }


def trainer(model: GaussianDiffusion, *, ema=None, global_step: int = 1):
    return SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        ema=ema,
        global_step=global_step,
    )


def train_output(batch_size: int = 2) -> TrainStepOutput:
    return TrainStepOutput(
        loss=torch.tensor(0.0),
        diagnostics={
            "timesteps": torch.tensor([1, 4])[:batch_size],
            "per_sample_loss": torch.tensor([1.0, 3.0])[:batch_size],
            "predicted_noise": torch.zeros(batch_size, 1, 4, 4),
            "target_noise": torch.ones(batch_size, 1, 4, 4),
            "clean_samples": torch.zeros(batch_size, 1, 4, 4),
        },
    )


def fit_event(runtime, validation=None) -> FitStartEvent:
    return FitStartEvent(
        trainer=runtime,
        train_dataloader=[],
        validation_dataloader=validation,
    )


def batch_event(runtime, *, global_step: int = 1) -> TrainBatchEndEvent:
    output = train_output()
    return TrainBatchEndEvent(
        trainer=runtime,
        batch=output.diagnostics["clean_samples"],
        output=output,
        loss=0.0,
        global_step=global_step,
        epoch_index=1,
    )


def epoch_event(runtime, epoch_index: int = 1) -> TrainEpochEndEvent:
    return TrainEpochEndEvent(
        trainer=runtime,
        epoch_index=epoch_index,
        metrics={},
    )


__all__ = [
    "FailingSampler",
    "RecordingLogger",
    "RecordingSampler",
    "TinyDenoiser",
    "ZeroDenoiser",
    "batch_event",
    "epoch_event",
    "fit_event",
    "profiles",
    "provider_config",
    "train_output",
    "trainer",
]
