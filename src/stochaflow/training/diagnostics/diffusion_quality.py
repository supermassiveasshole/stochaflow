"""Unconditional Gaussian diffusion quality diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from stochaflow.training.diagnostics.config import SamplerProfileConfig
from stochaflow.training.diagnostics.contracts import (
    DiagnosticResult,
    DiagnosticSourceRequest,
    FitStartEvent,
    ReconstructionCallable,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainingDiagnostic,
)
from stochaflow.training.diagnostics.gaussian_quality_engine import (
    GaussianQualityEngine,
    GaussianQualitySamplerPool,
    GaussianQualitySamplerRunner,
)
from stochaflow.training.diagnostics.providers.reference import (
    ReferenceMetricSuite,
)
from stochaflow.training.diagnostics.runtime import (
    BoundSampler,
    GaussianTrainingRuntime,
    ReconstructionEvaluator,
    SamplerPool,
    SamplerRunner,
    SeedPolicy,
    gaussian_training_runtime,
)
from stochaflow.training.strategy import ReferenceImageBatchSemantics
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES


class UnconditionalGaussianQualityFamily:
    """Supply unconditional Gaussian collaborators to the shared engine."""

    def training_runtime(self, trainer: Any) -> GaussianTrainingRuntime:
        """Resolve unconditional Gaussian diagnostic semantics."""

        return gaussian_training_runtime(trainer)

    def build_sampling(
        self,
        runtime: GaussianTrainingRuntime,
        profiles: Sequence[SamplerProfileConfig],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        GaussianQualitySamplerPool[BoundSampler],
        GaussianQualitySamplerRunner[BoundSampler],
    ]:
        """Build an unconditional sampler pool and runner."""

        return (
            SamplerPool(runtime, profiles, device=device),
            SamplerRunner(batch_size),
        )

    def capture_batch(self, event: TrainBatchEndEvent) -> None:
        """Accept any batch supported by the unconditional diagnostic."""

        del event

    def reconstruction_evaluator(
        self,
        trainer: Any,
        seed_policy: SeedPolicy,
    ) -> ReconstructionCallable:
        """Build unconditional reconstruction semantics."""

        return ReconstructionEvaluator(trainer, seed_policy)

    def reference_sampler(self, sampler: BoundSampler) -> BoundSampler:
        """Return the sampler expected by the reference metric suite."""

        return sampler

    def reference_image_extractor(
        self,
        trainer: Any,
    ) -> Callable[[Any], torch.Tensor]:
        """Resolve explicit strategy-owned reference image extraction."""

        strategy = getattr(trainer, "strategy", None)
        if not isinstance(strategy, ReferenceImageBatchSemantics):
            raise TypeError(
                "diffusion_quality reference metrics require a "
                "ReferenceImageBatchSemantics strategy"
            )
        return strategy.extract_reference_images

    def profile_manifest_metadata(
        self,
        profile: SamplerProfileConfig,
    ) -> Mapping[str, Any]:
        """Return no unconditional profile-specific metadata."""

        del profile
        return {}

    def manifest_metadata(
        self,
        event: TrainEpochEndEvent,
    ) -> Mapping[str, Any]:
        """Return no unconditional epoch-specific metadata."""

        del event
        return {}


@REGISTRIES.diagnostics.register("diffusion_quality")
class DiffusionQualityDiagnostic(TrainingDiagnostic):
    """Coordinate unconditional Gaussian quality providers."""

    def __init__(
        self,
        *,
        logger: ExperimentLogger,
        output_dir: str | Path,
        samplers: Sequence[Mapping[str, Any]],
        modules: Sequence[str] = (),
        cadence: Mapping[str, Any] | None = None,
        sampling: Mapping[str, Any] | None = None,
        providers: Mapping[str, Any] | None = None,
        reference: Mapping[str, Any] | None = None,
        use_ema: bool = True,
        failure_policy: str = "raise",
    ) -> None:
        self._engine = GaussianQualityEngine(
            logger=logger,
            output_dir=output_dir,
            samplers=samplers,
            modules=modules,
            cadence=cadence,
            sampling=sampling,
            providers=providers,
            reference=reference,
            use_ema=use_ema,
            failure_policy=failure_policy,
            diagnostic_name="diffusion_quality",
            family=UnconditionalGaussianQualityFamily(),
        )
        self.config = self._engine.config
        self.step_metrics = self._engine.step_metrics
        self.sampler_metrics = self._engine.sampler_metrics
        self.denoiser_artifacts = self._engine.denoiser_artifacts
        self.sampler_artifacts = self._engine.sampler_artifacts

    @property
    def metric_source_requests(self) -> tuple[DiagnosticSourceRequest, ...]:
        """Expose source requests for composition-time verification."""

        return self._engine.metric_source_requests

    @property
    def _reference_suite(self) -> ReferenceMetricSuite | None:
        return self._engine._reference_suite

    def on_fit_start(self, event: FitStartEvent) -> None:
        """Initialize the shared quality engine."""

        self._engine.on_fit_start(event)

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        """Dispatch one successful training batch to the shared engine."""

        self._engine.on_train_batch_end(event)

    def on_train_epoch_end(
        self,
        event: TrainEpochEndEvent,
    ) -> tuple[DiagnosticResult, ...] | None:
        """Return due epoch results from the shared engine."""

        return self._engine.on_train_epoch_end(event)


__all__ = ["DiffusionQualityDiagnostic"]
