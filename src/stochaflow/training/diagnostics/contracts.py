"""Public contracts for training diagnostics and diagnostic providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from stochaflow.sampling import SamplingObservation
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES


@dataclass(frozen=True, slots=True)
class DiagnosticBuildContext:
    """Runtime values available while constructing a diagnostic."""

    logger: ExperimentLogger
    output_dir: str | Path
    sample_shape: tuple[int, ...] | None


class ContextAwareDiagnostic(Protocol):
    """Optional class-level hook for requesting runtime constructor values."""

    @classmethod
    def context_parameters(
        cls,
        context: DiagnosticBuildContext,
    ) -> Mapping[str, Any]:
        """Return constructor parameters derived from the runtime context."""

        ...


@dataclass(frozen=True, slots=True)
class FitStartEvent:
    """Values supplied to diagnostics immediately before the training loop."""

    trainer: Any
    train_dataloader: Iterable[Any]
    validation_dataloader: Iterable[Any] | None


@dataclass(frozen=True, slots=True)
class TrainBatchEndEvent:
    """Values supplied after a successful optimizer step."""

    trainer: Any
    batch: Any
    output: Any
    loss: float
    global_step: int
    epoch_index: int | None


@dataclass(frozen=True, slots=True)
class TrainEpochEndEvent:
    """Values supplied after a completed training epoch."""

    trainer: Any
    epoch_index: int
    metrics: Mapping[str, float]


class TrainingDiagnostic(ABC):
    """Typed lifecycle contract implemented by every training diagnostic."""

    def on_fit_start(self, event: FitStartEvent) -> None:
        """Initialize resources and validate runtime capabilities."""

        del event

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        """Process one completed training step."""

        del event

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        """Process one completed training epoch."""

        del event


REGISTRIES.diagnostics.require_base(TrainingDiagnostic)


@dataclass(frozen=True, slots=True)
class ProviderValidationContext:
    """Stable runtime capabilities available to provider validation."""

    process: Any
    sample_shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class ReconstructionFrame:
    """One fixed-timestep denoiser reconstruction result."""

    timestep: int
    clean: torch.Tensor
    noisy: torch.Tensor
    predicted_clean: torch.Tensor
    mse: float
    psnr: float


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """Ordered reconstruction frames produced for configured timesteps."""

    frames: tuple[ReconstructionFrame, ...]


class ReconstructionCallable(Protocol):
    """Callable service used by reconstruction providers."""

    def __call__(
        self,
        *,
        clean_samples: torch.Tensor,
        timesteps: Sequence[int],
        max_samples: int,
        use_ema: bool,
    ) -> ReconstructionResult:
        """Evaluate clean-sample reconstruction at fixed timesteps."""

        ...


@dataclass(frozen=True, slots=True)
class StepMetricContext:
    """Inputs shared by step-level metric providers."""

    process: Any
    diagnostics: Mapping[str, Any]
    clean_samples: torch.Tensor | None
    sample_num: int
    use_ema: bool
    reconstruct: ReconstructionCallable


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """One sampler profile result shared across downstream providers."""

    samples: torch.Tensor
    trajectory: tuple[SamplingObservation, ...] | None
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class SamplerMetricContext:
    """Inputs shared by sampler metric providers."""

    profile_id: str
    profile_name: str
    result: SamplingResult


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """A local artifact optionally eligible for image logger fan-out."""

    kind: str
    path: Path
    image_tag: str | None = None
    caption: str | None = None


class ArtifactStoreProtocol(Protocol):
    """Collision-safe path allocator used by artifact providers."""

    def reserve(self, relative_path: str | Path) -> Path:
        """Reserve and return one artifact path under the epoch directory."""

        ...


@dataclass(frozen=True, slots=True)
class DenoiserArtifactContext:
    """Inputs shared by denoiser artifact providers."""

    store: ArtifactStoreProtocol
    clean_samples: torch.Tensor | None
    reconstruct: ReconstructionCallable
    use_ema: bool


@dataclass(frozen=True, slots=True)
class SamplerArtifactContext:
    """Inputs shared by sampler artifact providers."""

    store: ArtifactStoreProtocol
    profile_id: str
    profile_name: str
    trajectory_enabled: bool
    trajectory_gif_fps: int
    result: SamplingResult


class DiagnosticProvider(ABC):
    """Common validation contract for all diagnostic providers."""

    def validate(self, context: ProviderValidationContext) -> None:
        """Validate provider requirements before the training loop starts."""

        del context


class StepMetricProvider(DiagnosticProvider):
    """Produce scalar metrics from one completed training step."""

    @abstractmethod
    def collect(self, context: StepMetricContext) -> Mapping[str, float]:
        """Return a flat metric mapping."""


class SamplerMetricProvider(DiagnosticProvider):
    """Produce scalar metrics from one shared sampler result."""

    @abstractmethod
    def collect(self, context: SamplerMetricContext) -> Mapping[str, float]:
        """Return a flat metric mapping."""


class DenoiserArtifactProvider(DiagnosticProvider):
    """Write artifacts derived from clean samples and denoiser predictions."""

    @abstractmethod
    def render(
        self,
        context: DenoiserArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        """Write and return artifact records."""


class SamplerArtifactProvider(DiagnosticProvider):
    """Write artifacts derived from a shared sampler result."""

    @abstractmethod
    def render(
        self,
        context: SamplerArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        """Write and return artifact records."""


class ReferenceMetricProvider(DiagnosticProvider):
    """Stateful reference-distribution metric adapter."""

    @abstractmethod
    def update(self, images: torch.Tensor, *, real: bool) -> None:
        """Update real or generated feature state."""

    @abstractmethod
    def compute(self) -> Mapping[str, float]:
        """Compute metric values using cached real and current fake features."""

    @abstractmethod
    def reset_fake(self) -> None:
        """Reset generated features while retaining cached real features."""


__all__ = [
    "ArtifactRecord",
    "ArtifactStoreProtocol",
    "ContextAwareDiagnostic",
    "DenoiserArtifactContext",
    "DenoiserArtifactProvider",
    "DiagnosticBuildContext",
    "DiagnosticProvider",
    "FitStartEvent",
    "ProviderValidationContext",
    "ReconstructionCallable",
    "ReconstructionFrame",
    "ReconstructionResult",
    "ReferenceMetricProvider",
    "SamplerArtifactContext",
    "SamplerArtifactProvider",
    "SamplerMetricContext",
    "SamplerMetricProvider",
    "SamplingResult",
    "StepMetricContext",
    "StepMetricProvider",
    "TrainBatchEndEvent",
    "TrainEpochEndEvent",
    "TrainingDiagnostic",
]
