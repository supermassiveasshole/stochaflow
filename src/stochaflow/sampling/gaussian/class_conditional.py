"""Class-conditional Gaussian denoising with classifier-free guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch

from stochaflow.models.conditioning import (
    ClassConditionalDenoiser,
    predict_prevalidated_class_conditioned,
)
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
    LearnedRangeGaussianVarianceProcess,
)
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES

from ..builder import (
    SamplingBuilder,
    SamplingOutput,
    WeightSelection,
)
from ..sampler import Sampler, SamplerResult
from ..writers import SamplingBatch
from .builder import (
    DenoisingTrajectoryConfig,
    StandardDenoisingObserver,
    _variance_mode_from_declaration,
)
from .dynamics import (
    GaussianModelDynamics,
    PredictionType,
    VarianceMode,
)


@dataclass(frozen=True, slots=True)
class ClassConditionAllocation:
    """One ordered class label and requested sample count."""

    class_label: int
    count: int


@dataclass(frozen=True, slots=True)
class ClassConditionalDenoisingConfig:
    """Validated private configuration for conditional denoising."""

    weights: WeightSelection
    prediction_type: PredictionType
    variance_mode: VarianceMode
    clip_denoised: bool
    guidance_scale: float
    conditions: tuple[ClassConditionAllocation, ...]
    sampler: ComponentConfig
    trajectory: DenoisingTrajectoryConfig


@dataclass(slots=True)
class ClassConditionalEvaluationCounts:
    """Mutable model-evaluation accounting shared by sampling batches."""

    forward_calls: int = 0
    conditional_branches: int = 0
    unconditional_branches: int = 0


@REGISTRIES.sampling_builders.register("class_conditional_denoising")
class ClassConditionalDenoisingBuilder(SamplingBuilder):
    """Generate ordered class allocations with DDPM/DDIM-compatible CFG."""

    def run(self) -> SamplingOutput:
        """Compose conditional model adaptation with existing Gaussian solvers."""

        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "class_conditional_denoising requires DiscreteGaussianDenoisingProcess"
            )
        if self.context.shape is None:
            raise ValueError("class_conditional_denoising requires sampling.shape")
        config = self._parse_params(self.context.params)
        if config.variance_mode == "learned_range" and not isinstance(
            process,
            LearnedRangeGaussianVarianceProcess,
        ):
            raise TypeError(
                "class_conditional_denoising learned_range variance requires "
                "LearnedRangeGaussianVarianceProcess"
            )
        model_value, resolved_weights = self.context.model_provider.resolve(
            config.weights
        )
        model = _validate_denoiser(
            model_value,
            role=(
                "resolved "
                f"{resolved_weights} class_conditional_denoising inference model"
            ),
        )
        labels = self._expand_conditions(
            config.conditions,
            num_classes=model.num_classes,
            num_samples=self.context.num_samples,
            device=self.context.device,
        )
        sampler_value = REGISTRIES.samplers.create(
            config.sampler.name,
            **config.sampler.params,
        )
        if not isinstance(sampler_value, Sampler):
            raise TypeError("class_conditional_denoising sampler must satisfy Sampler")
        sampler = sampler_value
        generator = torch.Generator(device=self.context.device)
        generator.manual_seed(self.context.seed)
        counts = ClassConditionalEvaluationCounts()
        batches: list[SamplingBatch] = []
        solver_diagnostics: list[dict[str, Any]] = []
        offset = 0
        for count in self._batch_counts(
            self.context.num_samples,
            self.context.batch_size,
        ):
            class_labels = labels[offset : offset + count]
            offset += count
            initial = process.sample_terminal_prior(
                (count, *self.context.shape),
                device=self.context.device,
                generator=generator,
            )
            predict = ClassifierFreeGuidancePredictor(
                model,
                class_labels,
                guidance_scale=config.guidance_scale,
                variance_mode=config.variance_mode,
                counts=counts,
            )
            dynamics = GaussianModelDynamics(
                process,
                predict,
                prediction_type=config.prediction_type,
                variance_mode=config.variance_mode,
                clip_denoised=config.clip_denoised,
            )
            lifecycle = StandardDenoisingObserver(
                process=process,
                expected_shape=initial.shape,
                trajectory=config.trajectory,
            )
            result_value = cast(
                object,
                sampler.sample(
                    dynamics,
                    initial,
                    generator=generator,
                    observer=lifecycle,
                ),
            )
            if not isinstance(result_value, SamplerResult):
                raise TypeError("Sampler.sample() must return SamplerResult")
            lifecycle.validate_complete(result_value)
            final_state = result_value.final_state
            if not isinstance(final_state, torch.Tensor):
                raise TypeError(
                    "class_conditional_denoising sampler must return Tensor state"
                )
            _validate_shape(
                final_state,
                initial.shape,
                label="sampler final state",
            )
            batches.append(
                SamplingBatch(
                    samples=final_state.detach().to(device="cpu", copy=True),
                    trajectory=lifecycle.observations,
                )
            )
            solver_diagnostics.append(dict(result_value.diagnostics))
        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "weights": resolved_weights,
                "prediction_type": config.prediction_type,
                "variance": {"mode": config.variance_mode},
                "clip_denoised": config.clip_denoised,
                "guidance_scale": config.guidance_scale,
                "conditions": [
                    {
                        "class_label": allocation.class_label,
                        "count": allocation.count,
                    }
                    for allocation in config.conditions
                ],
                "sampler": {
                    "name": config.sampler.name,
                    "params": dict(config.sampler.params),
                },
                "trajectory": {
                    "enabled": config.trajectory.enabled,
                    "every_steps": config.trajectory.every_steps,
                },
                "forward_call_count": counts.forward_calls,
                "conditional_branch_evaluation_count": counts.conditional_branches,
                "unconditional_branch_evaluation_count": (
                    counts.unconditional_branches
                ),
                "solver_diagnostics": solver_diagnostics,
            },
        )

    @staticmethod
    def _parse_params(params: dict[str, Any]) -> ClassConditionalDenoisingConfig:
        allowed = {
            "weights",
            "prediction_type",
            "variance",
            "clip_denoised",
            "guidance_scale",
            "conditions",
            "sampler",
            "trajectory",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                "unknown class_conditional_denoising parameter(s): "
                + ", ".join(unknown)
            )
        weights = params.get("weights", "auto")
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError(
                "class_conditional_denoising weights must be auto, raw, or ema"
            )
        prediction_type = params.get("prediction_type", "epsilon")
        if prediction_type not in {"epsilon", "x0", "v", "score"}:
            raise ValueError(
                "class_conditional_denoising prediction_type must be "
                "epsilon, x0, v, or score"
            )
        variance_mode = _variance_mode_from_declaration(
            params.get("variance"),
            path="class_conditional_denoising.variance",
        )
        clip_denoised = params.get("clip_denoised", True)
        if not isinstance(clip_denoised, bool):
            raise TypeError("class_conditional_denoising clip_denoised must be boolean")
        guidance_scale = _guidance_scale(params.get("guidance_scale", 1.0))
        conditions = ClassConditionalDenoisingBuilder._condition_allocations(
            params.get("conditions")
        )
        sampler = ClassConditionalDenoisingBuilder._component(
            params.get("sampler"),
            "class_conditional_denoising.sampler",
        )
        trajectory = ClassConditionalDenoisingBuilder._trajectory(
            params.get("trajectory", {})
        )
        return ClassConditionalDenoisingConfig(
            weights=cast(WeightSelection, weights),
            prediction_type=cast(PredictionType, prediction_type),
            variance_mode=variance_mode,
            clip_denoised=clip_denoised,
            guidance_scale=guidance_scale,
            conditions=conditions,
            sampler=sampler,
            trajectory=trajectory,
        )

    @staticmethod
    def _condition_allocations(
        raw: object,
    ) -> tuple[ClassConditionAllocation, ...]:
        if not isinstance(raw, list) or not raw:
            raise ValueError(
                "class_conditional_denoising conditions must be a non-empty list"
            )
        allocations: list[ClassConditionAllocation] = []
        for index, item in enumerate(raw):
            path = f"class_conditional_denoising.conditions[{index}]"
            if not isinstance(item, dict):
                raise TypeError(f"{path} must be a mapping")
            unknown = sorted(set(item) - {"class_label", "count"})
            if unknown:
                raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
            if set(item) != {"class_label", "count"}:
                raise ValueError(f"{path} must contain class_label and count")
            class_label = item["class_label"]
            count = item["count"]
            if isinstance(class_label, bool) or not isinstance(class_label, int):
                raise TypeError(f"{path}.class_label must be an integer")
            if class_label < 0:
                raise ValueError(f"{path}.class_label must be non-negative")
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{path}.count must be an integer")
            if count <= 0:
                raise ValueError(f"{path}.count must be positive")
            allocations.append(ClassConditionAllocation(class_label, count))
        return tuple(allocations)

    @staticmethod
    def _component(raw: object, path: str) -> ComponentConfig:
        if not isinstance(raw, dict):
            raise TypeError(f"{path} must be a component mapping")
        unknown = sorted(set(raw) - {"name", "params"})
        if unknown:
            raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
        name = raw.get("name")
        params = raw.get("params", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        if not isinstance(params, dict):
            raise TypeError(f"{path}.params must be a mapping")
        return ComponentConfig(name=name, params=dict(params))

    @staticmethod
    def _trajectory(raw: object) -> DenoisingTrajectoryConfig:
        if not isinstance(raw, dict):
            raise TypeError("class_conditional_denoising.trajectory must be a mapping")
        unknown = sorted(set(raw) - {"enabled", "every_steps"})
        if unknown:
            raise ValueError(
                "unknown class_conditional_denoising.trajectory parameter(s): "
                + ", ".join(unknown)
            )
        enabled = raw.get("enabled", False)
        every_steps = raw.get("every_steps", 1)
        if not isinstance(enabled, bool):
            raise TypeError("trajectory.enabled must be boolean")
        if (
            isinstance(every_steps, bool)
            or not isinstance(every_steps, int)
            or every_steps <= 0
        ):
            raise ValueError("trajectory.every_steps must be a positive integer")
        return DenoisingTrajectoryConfig(enabled, every_steps)

    @staticmethod
    def _expand_conditions(
        allocations: tuple[ClassConditionAllocation, ...],
        *,
        num_classes: int,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        invalid = next(
            (
                allocation.class_label
                for allocation in allocations
                if allocation.class_label >= num_classes
            ),
            None,
        )
        if invalid is not None:
            raise ValueError(
                "class_conditional_denoising class_label values must lie in "
                f"[0, {num_classes})"
            )
        allocated = sum(allocation.count for allocation in allocations)
        if allocated != num_samples:
            raise ValueError(
                "class_conditional_denoising condition counts must sum to "
                f"sampling.num_samples ({num_samples}), got {allocated}"
            )
        return torch.tensor(
            [
                allocation.class_label
                for allocation in allocations
                for _ in range(allocation.count)
            ],
            device=device,
            dtype=torch.long,
        )

    @staticmethod
    def _batch_counts(num_samples: int, batch_size: int) -> tuple[int, ...]:
        return tuple(
            min(batch_size, num_samples - offset)
            for offset in range(0, num_samples, batch_size)
        )


class ClassifierFreeGuidancePredictor:
    """Adapt one conditional capability call to optimized CFG prediction."""

    def __init__(
        self,
        model: ClassConditionalDenoiser,
        class_labels: torch.Tensor,
        *,
        guidance_scale: float,
        variance_mode: VarianceMode = "fixed",
        counts: ClassConditionalEvaluationCounts,
    ) -> None:
        if variance_mode not in ("fixed", "learned_range"):
            raise ValueError(
                "classifier-free guidance variance_mode must be fixed or "
                "learned_range"
            )
        self.model = model
        self.class_labels = class_labels
        self.guidance_scale = guidance_scale
        self.variance_mode = variance_mode
        self.counts = counts

    def __call__(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        if state.shape[0] != self.class_labels.shape[0]:
            raise ValueError(
                "class_conditional_denoising model state must match its label batch"
            )
        if state.device != self.class_labels.device:
            raise ValueError(
                "class_conditional_denoising model state and labels must share a device"
            )
        null_labels = torch.full_like(
            self.class_labels,
            self.model.null_class_id,
        )
        if self.guidance_scale == 0.0:
            self.counts.unconditional_branches += 1
            return self._predict(state, model_time, null_labels)
        if self.guidance_scale == 1.0:
            self.counts.conditional_branches += 1
            return self._predict(state, model_time, self.class_labels)
        self.counts.conditional_branches += 1
        self.counts.unconditional_branches += 1
        doubled_state = torch.cat((state, state), dim=0)
        doubled_time = torch.cat((model_time, model_time), dim=0)
        doubled_labels = torch.cat((self.class_labels, null_labels), dim=0)
        doubled_prediction = self._predict(
            doubled_state,
            doubled_time,
            doubled_labels,
        )
        conditional, unconditional = doubled_prediction.chunk(2, dim=0)
        if self.variance_mode == "fixed":
            return unconditional + self.guidance_scale * (
                conditional - unconditional
            )
        conditional_mean, conditional_variance = conditional.chunk(2, dim=1)
        unconditional_mean, _ = unconditional.chunk(2, dim=1)
        guided_mean = unconditional_mean + self.guidance_scale * (
            conditional_mean - unconditional_mean
        )
        return torch.cat((guided_mean, conditional_variance), dim=1)

    def _predict(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        self.counts.forward_calls += 1
        output = cast(
            object,
            predict_prevalidated_class_conditioned(
                self.model,
                state,
                model_time,
                class_labels,
            ),
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError("class_conditional_denoising model must return a Tensor")
        expected_shape = state.shape
        if self.variance_mode == "learned_range":
            if state.ndim < 2:
                raise ValueError(
                    "learned_range class-conditional state must include a "
                    "channel dimension"
                )
            expected_shape = torch.Size(
                (state.shape[0], state.shape[1] * 2, *state.shape[2:])
            )
        if output.shape != expected_shape:
            raise ValueError(
                "class_conditional_denoising model output has shape "
                f"{tuple(output.shape)}, expected {tuple(expected_shape)}"
            )
        if output.device != state.device:
            raise ValueError(
                "class_conditional_denoising model output must share the state device"
            )
        if not torch.is_floating_point(output):
            raise TypeError(
                "class_conditional_denoising model output must be floating-point"
            )
        return output


def _validate_denoiser(
    value: object,
    *,
    role: str,
) -> ClassConditionalDenoiser:
    if not isinstance(value, ClassConditionalDenoiser):
        raise TypeError(f"{role} must satisfy ClassConditionalDenoiser")
    num_classes = cast(object, value.num_classes)
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(f"{role} num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError(f"{role} num_classes must be positive")
    null_class_id = cast(object, value.null_class_id)
    if isinstance(null_class_id, bool) or not isinstance(null_class_id, int):
        raise TypeError(f"{role} null_class_id must be an integer")
    if null_class_id != num_classes:
        raise ValueError(f"{role} null_class_id must equal num_classes")
    return value


def _guidance_scale(value: object) -> float:
    path = "class_conditional_denoising guidance_scale"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if result < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _validate_shape(
    value: torch.Tensor,
    expected: torch.Size,
    *,
    label: str,
) -> None:
    if value.shape != expected:
        raise ValueError(
            f"class_conditional_denoising {label} has shape {tuple(value.shape)}, "
            f"expected {tuple(expected)}"
        )


__all__ = [
    "ClassConditionalDenoisingBuilder",
]
