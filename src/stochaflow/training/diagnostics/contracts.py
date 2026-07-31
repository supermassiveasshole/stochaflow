"""Public contracts for training diagnostics and diagnostic providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import torch

from stochaflow.metrics import MetricDataRole, MetricSource
from stochaflow.sampling import SamplingObservation
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES

_DIAGNOSTIC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DIAGNOSTIC_METRIC_KEY_PATTERN = re.compile(
    r"^diagnostics/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)+$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DiagnosticBuildContext:
    """Runtime values available while constructing a diagnostic."""

    logger: ExperimentLogger
    output_dir: str | Path


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


@dataclass(frozen=True, slots=True)
class DiagnosticSourceRequest:
    """One source role and protocol descriptor requested by a diagnostic."""

    id: str
    data_role: MetricDataRole
    protocol: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.id), str)
            or _DIAGNOSTIC_ID_PATTERN.fullmatch(self.id) is None
        ):
            raise ValueError(
                "diagnostic source request.id must match "
                f"{_DIAGNOSTIC_ID_PATTERN.pattern!r}"
            )
        if self.data_role not in {"train", "validation", "test", "external"}:
            raise ValueError(
                "diagnostic source request.data_role must be train, "
                "validation, test, or external"
            )
        if not isinstance(cast(object, self.protocol), Mapping):
            raise TypeError("diagnostic source request.protocol must be a mapping")
        if any(
            not isinstance(cast(object, key), str) or not key
            for key in self.protocol
        ):
            raise ValueError(
                "diagnostic source request.protocol keys must be non-empty strings"
            )
        object.__setattr__(
            self,
            "protocol",
            MappingProxyType(dict(self.protocol)),
        )


@dataclass(frozen=True, slots=True)
class VerifiedMetricSource:
    """Composition-verified source metadata bound to one diagnostic result."""

    id: str
    metadata: MetricSource
    protocol_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.id), str)
            or _DIAGNOSTIC_ID_PATTERN.fullmatch(self.id) is None
        ):
            raise ValueError(
                "verified metric source.id must match "
                f"{_DIAGNOSTIC_ID_PATTERN.pattern!r}"
            )
        if not isinstance(cast(object, self.metadata), MetricSource):
            raise TypeError(
                "verified metric source.metadata must be a MetricSource"
            )
        if self.metadata.origin != "diagnostic":
            raise ValueError(
                "verified metric source.metadata must use origin='diagnostic'"
            )
        if (
            not isinstance(cast(object, self.protocol_digest), str)
            or _SHA256_PATTERN.fullmatch(self.protocol_digest) is None
        ):
            raise ValueError(
                "verified metric source.protocol_digest must be a lowercase "
                "SHA-256 digest"
            )
        expected_protocol_id = f"sha256:{self.protocol_digest}"
        if self.metadata.protocol_id != expected_protocol_id:
            raise ValueError(
                "verified metric source metadata.protocol_id must equal "
                f"{expected_protocol_id!r}"
            )
        if self.metadata.selection_eligible and (
            self.metadata.data_role != "validation"
        ):
            raise ValueError(
                "selection-eligible diagnostic sources require "
                "data_role='validation'"
            )


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Metrics emitted for one bound source during a due diagnostic epoch."""

    source_id: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.source_id), str)
            or _DIAGNOSTIC_ID_PATTERN.fullmatch(self.source_id) is None
        ):
            raise ValueError(
                "diagnostic result.source_id must match "
                f"{_DIAGNOSTIC_ID_PATTERN.pattern!r}"
            )
        if not isinstance(cast(object, self.metrics), Mapping):
            raise TypeError("diagnostic result.metrics must be a mapping")
        normalized: dict[str, float] = {}
        for key, value in self.metrics.items():
            if (
                not isinstance(cast(object, key), str)
                or _DIAGNOSTIC_METRIC_KEY_PATTERN.fullmatch(key) is None
            ):
                raise ValueError(
                    "diagnostic result metric keys must use "
                    "'diagnostics/<diagnostic-id>/...' canonical paths"
                )
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"diagnostic result metric {key!r} must be numeric"
                )
            normalized[key] = float(value)
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(normalized),
        )


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
    ) -> tuple[DiagnosticResult, ...] | None:
        """Return due source results, or ``None`` when no source is due."""

        del event
        return None


REGISTRIES.diagnostics.require_base(TrainingDiagnostic)


@runtime_checkable
class DiagnosticSourceProvider(Protocol):
    """Optional capability declaring source roles and protocol descriptors."""

    @property
    def metric_source_requests(self) -> tuple[DiagnosticSourceRequest, ...]:
        """Return sources that composition must verify before training."""

        ...


@dataclass(frozen=True, slots=True)
class BoundTrainingDiagnostic:
    """One diagnostic together with composition-verified result sources."""

    id: str
    diagnostic: TrainingDiagnostic
    sources: Mapping[str, VerifiedMetricSource]
    source_iterables: Mapping[str, Iterable[Any]]

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.id), str)
            or _DIAGNOSTIC_ID_PATTERN.fullmatch(self.id) is None
        ):
            raise ValueError(
                "bound training diagnostic.id must match "
                f"{_DIAGNOSTIC_ID_PATTERN.pattern!r}"
            )
        if not isinstance(cast(object, self.diagnostic), TrainingDiagnostic):
            raise TypeError(
                "bound training diagnostic.diagnostic must be a "
                "TrainingDiagnostic"
            )
        if not isinstance(cast(object, self.sources), Mapping):
            raise TypeError("bound training diagnostic.sources must be a mapping")
        normalized: dict[str, VerifiedMetricSource] = {}
        for source_id, source in self.sources.items():
            if (
                not isinstance(cast(object, source_id), str)
                or not source_id
            ):
                raise ValueError(
                    "bound training diagnostic source ids must be non-empty strings"
                )
            if not isinstance(cast(object, source), VerifiedMetricSource):
                raise TypeError(
                    "bound training diagnostic sources must contain "
                    "VerifiedMetricSource values"
                )
            if source.id != source_id:
                raise ValueError(
                    "bound training diagnostic source mapping key must match "
                    "VerifiedMetricSource.id"
                )
            normalized[source_id] = source
        eligible = [
            source
            for source in normalized.values()
            if source.metadata.selection_eligible
        ]
        if len(eligible) > 1:
            raise ValueError(
                "a diagnostic may expose at most one selection-eligible source"
            )
        if not isinstance(cast(object, self.source_iterables), Mapping):
            raise TypeError(
                "bound training diagnostic.source_iterables must be a mapping"
            )
        normalized_iterables: dict[str, Iterable[Any]] = {}
        for source_id, source_iterable in self.source_iterables.items():
            if source_id not in normalized:
                raise ValueError(
                    "bound training diagnostic iterable source ids must exist "
                    "in sources"
                )
            source = normalized[source_id]
            if source.metadata.data_role not in {"train", "validation"}:
                raise ValueError(
                    "only train and validation diagnostic sources may bind "
                    "fit iterables"
                )
            if not isinstance(cast(object, source_iterable), Iterable):
                raise TypeError(
                    "bound training diagnostic source iterables must be iterable"
                )
            if isinstance(cast(object, source_iterable), Iterator):
                raise TypeError(
                    "bound training diagnostic source iterables must be "
                    "re-iterable"
                )
            normalized_iterables[source_id] = source_iterable
        for source_id, source in normalized.items():
            if (
                source.metadata.data_role in {"train", "validation"}
                and source_id not in normalized_iterables
            ):
                raise ValueError(
                    f"diagnostic source {source_id!r} with "
                    f"data_role={source.metadata.data_role!r} requires an "
                    "actual fit iterable binding"
                )
        object.__setattr__(
            self,
            "sources",
            MappingProxyType(normalized),
        )
        object.__setattr__(
            self,
            "source_iterables",
            MappingProxyType(normalized_iterables),
        )


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
    "BoundTrainingDiagnostic",
    "ContextAwareDiagnostic",
    "DenoiserArtifactContext",
    "DenoiserArtifactProvider",
    "DiagnosticBuildContext",
    "DiagnosticProvider",
    "DiagnosticResult",
    "DiagnosticSourceProvider",
    "DiagnosticSourceRequest",
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
    "VerifiedMetricSource",
]
