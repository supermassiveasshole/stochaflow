"""Shared fixtures and test doubles for diagnostic tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import (
    GaussianPrediction,
    Sampler,
    SamplerResult,
    SamplingObservation,
)
from stochaflow.training import (
    DiagnosticBuildContext,
    FitStartEvent,
    GaussianDenoisingTrainingStrategy,
    GaussianStepObservation,
    ManagedTrainingModule,
    MSEObjective,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainStepOutput,
)
from stochaflow.training.diagnostics.runtime import TrainingDiagnosticModelAccess
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


class GaussianTestAssets(nn.Module):
    """Test-only asset holder for a Gaussian diagnostic runtime."""

    def __init__(
        self,
        inference_model: nn.Module,
        process: DiscreteGaussianProcess,
    ) -> None:
        super().__init__()
        self.inference_model = inference_model
        self.process = process
        self.objective = MSEObjective()
        self.strategy = GaussianDenoisingTrainingStrategy(
            inference_model,
            process,
            self.objective,
            prediction_type="epsilon",
        )

    @property
    def model(self) -> nn.Module:
        return self

    @property
    def prediction_type(self):
        return self.strategy.prediction_type

    def forward(self, state: torch.Tensor, model_time: torch.Tensor) -> torch.Tensor:
        return self.inference_model(state, model_time)


class RecordingSampler(Sampler):
    records: ClassVar[dict[str, list[torch.Tensor]]] = {}

    def __init__(self, *, marker: str) -> None:
        self.marker = marker

    def sample(self, dynamics, initial_state, **kwargs) -> SamplerResult:
        observer = kwargs.get("observer")
        self.records.setdefault(self.marker, []).append(
            initial_state.detach().cpu().clone()
        )
        final = initial_state.clamp(-1.0, 1.0)
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    0,
                    dynamics.process.terminal_time,
                    initial_state,
                    False,
                    {},
                )
            )
            observer.observe(
                SamplingObservation(
                    1,
                    dynamics.process.clean_time,
                    final,
                    True,
                    {},
                )
            )
        return SamplerResult(final, 1, {})


class FailingSampler(Sampler):
    def sample(self, dynamics, initial_state, **kwargs) -> SamplerResult:
        del dynamics, initial_state, kwargs
        raise RuntimeError("sampling failed")


REGISTRIES.samplers.add("test_recording_diagnostic", RecordingSampler)
REGISTRIES.samplers.add("test_failing_diagnostic", FailingSampler)


def profiles(*, trajectory: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "id": "ddpm_full",
            "name": "ddpm",
            "params": {},
            "trajectory": {
                "enabled": trajectory,
                "every_steps": 1,
                "gif_fps": 4,
            },
        },
        {
            "id": "ddim_2",
            "name": "ddim",
            "params": {"num_inference_steps": 2, "eta": 0.0},
            "trajectory": {
                "enabled": trajectory,
                "every_steps": 1,
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


def gaussian_system(
    model: nn.Module | None = None,
    *,
    num_timesteps: int = 4,
) -> GaussianTestAssets:
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": num_timesteps,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )
    return GaussianTestAssets(model or ZeroDenoiser(), process)


def trainer(model: GaussianTestAssets, *, ema=None, global_step: int = 1):
    return SimpleNamespace(
        model=model.inference_model,
        process=model.process,
        strategy=model.strategy,
        managed_modules={
            "primary_model": ManagedTrainingModule(model.inference_model),
            "process": ManagedTrainingModule(model.process),
            "objective": ManagedTrainingModule(model.objective),
        },
        ema_model=model.inference_model,
        device=torch.device("cpu"),
        ema=ema,
        global_step=global_step,
    )


def train_output(batch_size: int = 2) -> TrainStepOutput:
    clean = torch.zeros(batch_size, 1, 4, 4)
    predicted_noise = torch.zeros_like(clean)
    return TrainStepOutput(
        loss=torch.tensor(0.0),
        diagnostic_observation=GaussianStepObservation(
            state_times=torch.tensor([1, 4])[:batch_size],
            prediction=GaussianPrediction(
                clean=clean,
                epsilon=predicted_noise,
                model_output=predicted_noise,
            ),
            noise_target=torch.ones_like(clean),
            clean_samples=clean,
            per_sample_loss=torch.tensor([1.0, 3.0])[:batch_size],
        ),
    )


def fit_event(runtime, validation=None) -> FitStartEvent:
    return FitStartEvent(
        train_dataloader=[],
        validation_dataloader=validation,
        global_step=runtime.global_step,
    )


def batch_event(runtime, *, global_step: int = 1) -> TrainBatchEndEvent:
    output = train_output()
    return TrainBatchEndEvent(
        batch=torch.zeros(2, 1, 4, 4),
        loss=0.0,
        global_step=global_step,
        epoch_index=1,
        diagnostic_observation=output.diagnostic_observation,
    )


def epoch_event(runtime, epoch_index: int = 1) -> TrainEpochEndEvent:
    return TrainEpochEndEvent(
        epoch_index=epoch_index,
        global_step=runtime.global_step,
        metrics={},
    )


def diagnostic_context(
    runtime,
    logger: RecordingLogger,
    output_dir: str | Path,
    *,
    component_name: str = "diffusion_quality",
) -> DiagnosticBuildContext:
    """Build the explicit runtime context used by a Diagnostic constructor."""

    access = TrainingDiagnosticModelAccess(
        device=runtime.device,
        model=runtime.model,
        ema=runtime.ema,
        managed_modules=tuple(
            (name, asset.module)
            for name, asset in runtime.managed_modules.items()
            if name != "primary_model"
        ),
    )
    return DiagnosticBuildContext(
        component_name=component_name,
        logger=logger,
        output_dir=output_dir,
        model_access=access,
        _strategy=runtime.strategy,
        _process=runtime.process,
    )


__all__ = [
    "FailingSampler",
    "GaussianTestAssets",
    "RecordingLogger",
    "RecordingSampler",
    "TinyDenoiser",
    "ZeroDenoiser",
    "batch_event",
    "diagnostic_context",
    "epoch_event",
    "fit_event",
    "gaussian_system",
    "profiles",
    "provider_config",
    "train_output",
    "trainer",
]
