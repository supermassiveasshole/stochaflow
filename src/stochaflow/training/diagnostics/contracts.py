"""Public contracts for training diagnostics and diagnostic providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import torch

from stochaflow.sampling.sampler import SamplingObservation
from stochaflow.utils.logging_contracts import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES

CapabilityT = TypeVar("CapabilityT")


@runtime_checkable
class DiagnosticModelAccess(Protocol):
    """Narrow protected access to the current raw or EMA model snapshot."""

    @property
    def device(self) -> torch.device:
        """Return the device used by managed training modules."""

        ...

    @property
    def ema_available(self) -> bool:
        """Report whether an EMA snapshot can be selected."""

        ...

    def evaluation(
        self,
        *,
        seed: int,
        prefer_ema: bool,
    ) -> AbstractContextManager[None]:
        """Protect RNG, inference mode, weights, and managed module modes."""

        ...


@dataclass(frozen=True, slots=True)
class DiagnosticBuildContext:
    """Runtime values available while constructing a diagnostic."""

    component_name: str
    logger: ExperimentLogger
    output_dir: str | Path
    model_access: DiagnosticModelAccess
    _strategy: object = field(repr=False)
    _process: object | None = field(repr=False)

    def require_strategy_capability(
        self,
        capability: type[CapabilityT],
    ) -> CapabilityT:
        """Return a required narrow Strategy capability or fail construction."""

        if not _supports_capability(self._strategy, capability):
            raise TypeError(
                f"diagnostic '{self.component_name}' requires Strategy capability "
                f"{_capability_name(capability)}"
            )
        return cast(CapabilityT, self._strategy)

    def optional_strategy_capability(
        self,
        capability: type[CapabilityT],
    ) -> CapabilityT | None:
        """Return an optional narrow Strategy capability when present."""

        if not _supports_capability(self._strategy, capability):
            return None
        return cast(CapabilityT, self._strategy)

    def require_process_capability(
        self,
        capability: type[CapabilityT],
    ) -> CapabilityT:
        """Return a required Process capability or fail construction."""

        if not _supports_capability(self._process, capability):
            raise TypeError(
                f"diagnostic '{self.component_name}' requires Process capability "
                f"{_capability_name(capability)}"
            )
        return cast(CapabilityT, self._process)


def _supports_capability(value: object, capability: type[object]) -> bool:
    try:
        return isinstance(value, capability)
    except TypeError as exc:
        raise TypeError(
            f"diagnostic capability {_capability_name(capability)} must support "
            "runtime isinstance checks"
        ) from exc


def _capability_name(capability: type[object]) -> str:
    return f"{capability.__module__}.{capability.__qualname__}"


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

    train_dataloader: Iterable[Any]
    validation_dataloader: Iterable[Any] | None
    global_step: int


@dataclass(frozen=True, slots=True)
class TrainBatchEndEvent:
    """Values supplied after a successful optimizer step."""

    batch: Any
    loss: float
    global_step: int
    epoch_index: int | None
    diagnostic_observation: object | None


@dataclass(frozen=True, slots=True)
class TrainEpochEndEvent:
    """Values supplied after a completed training epoch."""

    epoch_index: int
    global_step: int
    metrics: Mapping[str, float]


class TrainingDiagnostic:
    """Observation-only lifecycle contract for training diagnostics.

    Implementations may emit metrics and artifacts, but must not mutate managed
    training state or own checkpoint-restored state. Each training invocation
    constructs fresh diagnostic instances; caches and counters are ephemeral.
    """

    def on_fit_start(self, event: FitStartEvent) -> None:
        """Initialize resources and validate runtime capabilities."""

        del event

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        """Process one completed training step."""

        del event

    def on_train_epoch_end(
        self,
        event: TrainEpochEndEvent,
    ) -> None:
        """Observe one completed epoch without feeding values into training."""

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
    diagnostic_observation: object | None
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


class DiagnosticProvider:
    """Common validation contract for all diagnostic providers."""

    def validate(self, context: ProviderValidationContext) -> None:
        """Validate provider requirements before the training loop starts."""

        del context


class StepMetricProvider(DiagnosticProvider, ABC):
    """Produce scalar metrics from one completed training step."""

    @abstractmethod
    def collect(self, context: StepMetricContext) -> Mapping[str, float]:
        """Return a flat metric mapping."""


class SamplerMetricProvider(DiagnosticProvider, ABC):
    """Produce scalar metrics from one shared sampler result."""

    @abstractmethod
    def collect(self, context: SamplerMetricContext) -> Mapping[str, float]:
        """Return a flat metric mapping."""


class DenoiserArtifactProvider(DiagnosticProvider, ABC):
    """Write artifacts derived from clean samples and denoiser predictions."""

    @abstractmethod
    def render(
        self,
        context: DenoiserArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        """Write and return artifact records."""


class SamplerArtifactProvider(DiagnosticProvider, ABC):
    """Write artifacts derived from a shared sampler result."""

    @abstractmethod
    def render(
        self,
        context: SamplerArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        """Write and return artifact records."""


class ReferenceMetricProvider(DiagnosticProvider, ABC):
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
    "DiagnosticModelAccess",
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
