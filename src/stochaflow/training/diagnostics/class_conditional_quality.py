"""Class-conditional Gaussian quality diagnostics with explicit labels."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from stochaflow.training.diagnostics.config import SamplerProfileConfig
from stochaflow.training.diagnostics.contracts import (
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
from stochaflow.training.diagnostics.runtime import (
    BoundSampler,
    ClassConditionalBoundSampler,
    ClassConditionalGaussianTrainingRuntime,
    ClassConditionalReconstructionEvaluator,
    ClassConditionalSamplerPool,
    ClassConditionalSamplerRunner,
    SeedPolicy,
    class_conditional_gaussian_training_runtime,
)
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES


@dataclass(frozen=True, slots=True)
class DiagnosticClassAllocation:
    """One ordered class allocation for diagnostic sampling."""

    class_label: int
    count: int


def _class_allocations(
    value: object,
) -> tuple[DiagnosticClassAllocation, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            "class_conditional_diffusion_quality conditions must be a sequence"
        )
    allocations: list[DiagnosticClassAllocation] = []
    for index, item in enumerate(value):
        path = f"class_conditional_diffusion_quality.conditions[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{path} must be a mapping")
        if set(item) != {"class_label", "count"}:
            raise ValueError(
                f"{path} must contain only class_label and count"
            )
        label = cast(object, item["class_label"])
        count = cast(object, item["count"])
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError(f"{path}.class_label must be non-negative")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{path}.count must be positive")
        allocations.append(DiagnosticClassAllocation(label, count))
    if not allocations:
        raise ValueError(
            "class_conditional_diffusion_quality conditions must not be empty"
        )
    return tuple(allocations)


def _guidance_scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "class_conditional_diffusion_quality guidance_scale must be numeric"
        )
    scale = float(value)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "class_conditional_diffusion_quality guidance_scale must be "
            "finite and non-negative"
        )
    return scale


class ClassConditionalGaussianQualityFamily:
    """Supply label-aware Gaussian collaborators to the shared engine."""

    def __init__(
        self,
        allocations: Sequence[DiagnosticClassAllocation],
        guidance_scale: float,
    ) -> None:
        self.allocations = tuple(allocations)
        self.guidance_scale = guidance_scale
        self._last_class_labels: torch.Tensor | None = None
        self._sampler_runner: ClassConditionalSamplerRunner | None = None

    def training_runtime(
        self,
        trainer: Any,
    ) -> ClassConditionalGaussianTrainingRuntime:
        """Resolve class-conditional Gaussian diagnostic semantics."""

        return class_conditional_gaussian_training_runtime(trainer)

    def build_sampling(
        self,
        runtime: ClassConditionalGaussianTrainingRuntime,
        profiles: Sequence[SamplerProfileConfig],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        GaussianQualitySamplerPool[ClassConditionalBoundSampler],
        GaussianQualitySamplerRunner[ClassConditionalBoundSampler],
    ]:
        """Build a label-aligned sampler pool and CFG runner."""

        labels = torch.tensor(
            [
                allocation.class_label
                for allocation in self.allocations
                for _ in range(allocation.count)
            ],
            dtype=torch.long,
            device=device,
        )
        if bool(torch.any(labels >= runtime.num_classes)):
            raise ValueError(
                "class_conditional_diffusion_quality class labels exceed "
                "the model's num_classes"
            )
        runner = ClassConditionalSamplerRunner(
            batch_size,
            runtime=runtime,
            class_labels=labels,
            guidance_scale=self.guidance_scale,
        )
        self._sampler_runner = runner
        return (
            ClassConditionalSamplerPool(profiles, device=device),
            runner,
        )

    def capture_batch(self, event: TrainBatchEndEvent) -> None:
        """Capture labels aligned with the retained clean samples."""

        diagnostics = getattr(event.output, "diagnostics", None)
        labels = (
            diagnostics.get("class_labels")
            if isinstance(diagnostics, Mapping)
            else None
        )
        clean = (
            diagnostics.get("clean_samples")
            if isinstance(diagnostics, Mapping)
            else None
        )
        if (
            not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or labels.dtype != torch.long
            or not isinstance(clean, torch.Tensor)
            or clean.ndim == 0
            or labels.shape[0] != clean.shape[0]
        ):
            raise ValueError(
                "conditional diagnostic requires aligned clean_samples "
                "and 1D long class_labels diagnostics"
            )
        self._last_class_labels = labels.detach().cpu()

    def reconstruction_evaluator(
        self,
        trainer: Any,
        seed_policy: SeedPolicy,
    ) -> ReconstructionCallable:
        """Build reconstruction semantics for the latest labeled batch."""

        if self._last_class_labels is None:
            raise RuntimeError(
                "conditional diagnostic has not captured a labeled batch"
            )
        return ClassConditionalReconstructionEvaluator(
            trainer,
            seed_policy,
            self._last_class_labels,
        )

    def reference_sampler(
        self,
        sampler: ClassConditionalBoundSampler,
    ) -> BoundSampler:
        """Reject the unsupported unconditional reference protocol."""

        del sampler
        raise RuntimeError(
            "class-conditional reference metrics require a class-aware evaluator"
        )

    def reference_image_extractor(
        self,
        trainer: Any,
    ) -> Callable[[Any], torch.Tensor]:
        """Reject reference extraction for this unsupported evaluator."""

        del trainer
        raise RuntimeError(
            "class-conditional reference metrics require a class-aware evaluator"
        )

    def profile_manifest_metadata(
        self,
        profile: SamplerProfileConfig,
    ) -> Mapping[str, Any]:
        """Record model evaluation counts from the latest CFG profile."""

        if self._sampler_runner is None:
            return {}
        try:
            counts = self._sampler_runner.counts_for(profile.id)
        except RuntimeError:
            return {}
        return {
            "model_evaluations": {
                "forward_calls": counts.forward_calls,
                "conditional_branches": counts.conditional_branches,
                "unconditional_branches": counts.unconditional_branches,
            }
        }

    def manifest_metadata(
        self,
        event: TrainEpochEndEvent,
    ) -> Mapping[str, Any]:
        """Record fixed class allocation and CFG policy."""

        del event
        return {
            "conditioning": {
                "guidance_scale": self.guidance_scale,
                "allocations": [
                    {
                        "class_label": allocation.class_label,
                        "count": allocation.count,
                    }
                    for allocation in self.allocations
                ],
            }
        }


@REGISTRIES.diagnostics.register("class_conditional_diffusion_quality")
class ClassConditionalDiffusionQualityDiagnostic(TrainingDiagnostic):
    """Run label-aligned reconstruction and CFG sampler diagnostics."""

    def __init__(
        self,
        *,
        logger: ExperimentLogger,
        output_dir: str | Path,
        conditions: Sequence[Mapping[str, Any]],
        guidance_scale: float,
        samplers: Sequence[Mapping[str, Any]],
        modules: Sequence[str] = (),
        cadence: Mapping[str, Any] | None = None,
        sampling: Mapping[str, Any] | None = None,
        providers: Mapping[str, Any] | None = None,
        reference: Mapping[str, Any] | None = None,
        use_ema: bool = True,
        failure_policy: str = "raise",
    ) -> None:
        self.allocations = _class_allocations(conditions)
        self.guidance_scale = _guidance_scale(guidance_scale)
        family = ClassConditionalGaussianQualityFamily(
            self.allocations,
            self.guidance_scale,
        )
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
            diagnostic_name="class_conditional_diffusion_quality",
            family=family,
        )
        self.config = self._engine.config
        if self.config.reference.enabled:
            raise ValueError(
                "class_conditional_diffusion_quality reference metrics are "
                "not supported; use a class-aware post-training evaluator"
            )
        allocated = sum(item.count for item in self.allocations)
        if allocated != self.config.sampling.sample_num:
            raise ValueError(
                "class_conditional_diffusion_quality condition counts must "
                "sum to sampling.sample_num"
            )
        self.step_metrics = self._engine.step_metrics
        self.sampler_metrics = self._engine.sampler_metrics
        self.denoiser_artifacts = self._engine.denoiser_artifacts
        self.sampler_artifacts = self._engine.sampler_artifacts

    def on_fit_start(self, event: FitStartEvent) -> None:
        """Initialize label-aware runtime and sampler resources."""

        self._engine.on_fit_start(event)

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        """Dispatch one successful labeled training batch."""

        self._engine.on_train_batch_end(event)

    def on_train_epoch_end(
        self,
        event: TrainEpochEndEvent,
    ) -> None:
        """Emit due epoch observations and artifacts."""

        self._engine.on_train_epoch_end(event)


__all__ = [
    "ClassConditionalDiffusionQualityDiagnostic",
    "DiagnosticClassAllocation",
]
