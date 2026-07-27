"""Sampling-builder contracts and the standard denoising recipe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianDenoisingProcess, Process
from stochaflow.sampling.gaussian import GaussianModelDynamics, PredictionType
from stochaflow.sampling.sampler import (
    Sampler,
    SamplerResult,
    SamplingObservation,
)
from stochaflow.sampling.writers import SamplingBatch
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.device import move_module_to_device
from stochaflow.utils.registry import REGISTRIES

WeightSelection = Literal["auto", "raw", "ema"]


class InferenceModelProvider:
    """Construct raw or EMA inference models from portable checkpoint state."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        raw_state_dict: Mapping[str, torch.Tensor],
        ema_state_dict: Mapping[str, torch.Tensor] | None,
        device: torch.device,
        prefer_ema: bool,
    ) -> None:
        self._model_factory = model_factory
        self._raw_state_dict = raw_state_dict
        self._ema_state_dict = ema_state_dict
        self.device = device
        self.prefer_ema = prefer_ema

    def resolve(self, weights: WeightSelection) -> tuple[nn.Module, str]:
        """Build the selected model and return its resolved weight label."""

        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("sampling weights must be auto, raw, or ema")
        resolved = (
            "ema"
            if weights == "ema" or (weights == "auto" and self.prefer_ema)
            else "raw"
        )
        state = self._ema_state_dict if resolved == "ema" else self._raw_state_dict
        if state is None:
            raise ValueError("EMA weights were requested but are unavailable")
        model_value = cast(object, self._model_factory())
        if not isinstance(model_value, nn.Module):
            raise TypeError("inference model factory must return nn.Module")
        model_value.load_state_dict(state)
        move_module_to_device(
            model_value,
            self.device,
            role="inference model",
        )
        model_value.eval()
        return model_value, resolved

    def get(self, weights: WeightSelection = "auto") -> nn.Module:
        """Build and return a selected inference model."""

        return self.resolve(weights)[0]


@dataclass(frozen=True, slots=True)
class SamplingBuilderContext:
    """Runtime services and user parameters supplied to a sampling builder."""

    params: dict[str, Any]
    process: Process | None
    model_provider: InferenceModelProvider
    device: torch.device
    seed: int
    shape: tuple[int, ...] | None
    num_samples: int
    batch_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", deepcopy(self.params))


@dataclass(frozen=True, slots=True)
class SamplingOutput:
    """Generated batches and recipe-resolved metadata."""

    batches: tuple[SamplingBatch, ...]
    metadata: Mapping[str, Any]


class SamplingBuilder(ABC):
    """Assemble and execute one complete task-specific sampling workflow."""

    def __init__(self, context: SamplingBuilderContext) -> None:
        self.context = context

    @abstractmethod
    def run(self) -> SamplingOutput:
        """Execute the workflow once and return writer-ready batches."""


REGISTRIES.sampling_builders.require_base(SamplingBuilder)


@dataclass(frozen=True, slots=True)
class DenoisingTrajectoryConfig:
    enabled: bool
    every_steps: int


@dataclass(frozen=True, slots=True)
class StandardDenoisingConfig:
    weights: WeightSelection
    prediction_type: PredictionType
    clip_denoised: bool
    sampler: ComponentConfig
    trajectory: DenoisingTrajectoryConfig


class StandardDenoisingObserver:
    """Validate one standard denoising lifecycle and optionally retain it."""

    def __init__(
        self,
        *,
        process: DiscreteGaussianDenoisingProcess,
        expected_shape: torch.Size,
        trajectory: DenoisingTrajectoryConfig,
    ) -> None:
        self._process = process
        self._expected_shape = expected_shape
        self._trajectory = trajectory
        self._previous_step: int | None = None
        self._final_seen = False
        self._retained: list[SamplingObservation] = []

    @property
    def observations(self) -> tuple[SamplingObservation, ...] | None:
        if not self._trajectory.enabled:
            return None
        return tuple(self._retained)

    def observe(self, observation: object) -> None:
        """Validate and optionally retain one accepted solver state."""

        if not isinstance(observation, SamplingObservation):
            raise TypeError("sampler observer events must be SamplingObservation")
        if self._final_seen:
            raise ValueError("sampler emitted an observation after its final state")
        step_index = cast(object, observation.step_index)
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise TypeError("sampling observation step_index must be an integer")
        if self._previous_step is None:
            if step_index != 0:
                raise ValueError("sampler initial observation must use step index 0")
            if observation.coordinate != self._process.terminal_time:
                raise ValueError(
                    "standard_denoising requires its initial observation at "
                    f"terminal time {self._process.terminal_time}"
                )
        elif step_index <= self._previous_step:
            raise ValueError("sampling observation step indices must increase")
        coordinate = cast(object, observation.coordinate)
        if isinstance(coordinate, bool) or not isinstance(
            coordinate, (int, float)
        ):
            raise TypeError("sampling observation coordinate must be numeric")
        is_final = cast(object, observation.is_final)
        if not isinstance(is_final, bool):
            raise TypeError("sampling observation is_final must be boolean")
        observation_diagnostics = cast(object, observation.diagnostics)
        if not isinstance(observation_diagnostics, Mapping):
            raise TypeError("sampling observation diagnostics must be a mapping")
        state = observation.state
        if not isinstance(state, torch.Tensor):
            raise TypeError("standard denoising observations must contain Tensors")
        _validate_shape(state, self._expected_shape, label=f"trajectory step {step_index}")
        if observation.is_final:
            if observation.coordinate != self._process.clean_time:
                raise ValueError(
                    "standard_denoising requires its final observation at clean "
                    f"time {self._process.clean_time}"
                )
            self._final_seen = True
        self._previous_step = step_index
        if self._should_retain(observation):
            self._retained.append(
                SamplingObservation(
                    step_index=step_index,
                    coordinate=observation.coordinate,
                    state=state.detach().to(device="cpu", copy=True),
                    is_final=observation.is_final,
                    diagnostics=dict(observation.diagnostics),
                )
            )

    def validate_complete(self, result: SamplerResult) -> None:
        """Require a complete initial-to-final lifecycle matching the result."""

        if self._previous_step is None:
            raise ValueError("sampler did not emit an initial observation")
        if not self._final_seen:
            raise ValueError("sampler did not emit a final observation")
        if self._previous_step != result.num_steps:
            raise ValueError(
                "sampler final observation step index must equal result.num_steps"
            )

    def _should_retain(self, observation: SamplingObservation) -> bool:
        return self._trajectory.enabled and (
            observation.step_index == 0
            or observation.is_final
            or observation.step_index % self._trajectory.every_steps == 0
        )


@REGISTRIES.sampling_builders.register("standard_denoising")
class StandardDenoisingBuilder(SamplingBuilder):
    """Fixed-shape unconditional denoising with a registered sampler."""

    def run(self) -> SamplingOutput:
        """Build the prior, dynamics, sampler, observer, and generated batches."""

        config = self._parse_params(self.context.params)
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "standard_denoising requires "
                "DiscreteGaussianDenoisingProcess capability"
            )
        if self.context.shape is None:
            raise ValueError("standard_denoising requires sampling.shape")
        model, resolved_weights = self.context.model_provider.resolve(config.weights)
        sampler = cast(
            Sampler,
            REGISTRIES.samplers.create(
                config.sampler.name,
                **config.sampler.params,
            ),
        )
        dynamics = GaussianModelDynamics(
            process,
            model,
            prediction_type=config.prediction_type,
            clip_denoised=config.clip_denoised,
        )
        generator = torch.Generator(device=self.context.device)
        generator.manual_seed(self.context.seed)
        batches: list[SamplingBatch] = []
        diagnostics: list[Mapping[str, Any]] = []
        for count in self._batch_counts(
            self.context.num_samples, self.context.batch_size
        ):
            initial = process.sample_terminal_prior(
                (count, *self.context.shape),
                device=self.context.device,
                generator=generator,
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
                raise TypeError("standard denoising sampler must return Tensor state")
            _validate_shape(final_state, initial.shape, label="sampler final state")
            batches.append(
                SamplingBatch(
                    samples=final_state.detach().to(device="cpu", copy=True),
                    trajectory=lifecycle.observations,
                )
            )
            diagnostics.append(dict(result_value.diagnostics))
        return SamplingOutput(
            tuple(batches),
            {
                "weights": resolved_weights,
                "prediction_type": config.prediction_type,
                "clip_denoised": config.clip_denoised,
                "sampler": {
                    "name": config.sampler.name,
                    "params": deepcopy(config.sampler.params),
                },
                "trajectory": {
                    "enabled": config.trajectory.enabled,
                    "every_steps": config.trajectory.every_steps,
                },
                "solver_diagnostics": diagnostics,
            },
        )

    @staticmethod
    def _parse_params(params: dict[str, Any]) -> StandardDenoisingConfig:
        allowed = {
            "weights",
            "prediction_type",
            "clip_denoised",
            "sampler",
            "trajectory",
        }
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ValueError(
                "unknown standard_denoising parameter(s): " + ", ".join(unknown)
            )
        weights = params.get("weights", "auto")
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("standard_denoising weights must be auto, raw, or ema")
        prediction_type = params.get("prediction_type", "epsilon")
        if prediction_type not in {"epsilon", "x0", "v", "score"}:
            raise ValueError(
                "standard_denoising prediction_type must be epsilon, x0, v, or score"
            )
        clip_denoised = params.get("clip_denoised", True)
        if not isinstance(clip_denoised, bool):
            raise TypeError("standard_denoising clip_denoised must be boolean")
        sampler = StandardDenoisingBuilder._component(
            params.get("sampler"), "standard_denoising.sampler"
        )
        trajectory_raw = params.get("trajectory", {})
        if not isinstance(trajectory_raw, dict):
            raise TypeError("standard_denoising.trajectory must be a mapping")
        trajectory_unknown = sorted(
            set(trajectory_raw) - {"enabled", "every_steps"}
        )
        if trajectory_unknown:
            raise ValueError(
                "unknown standard_denoising.trajectory parameter(s): "
                + ", ".join(trajectory_unknown)
            )
        enabled = trajectory_raw.get("enabled", False)
        every_steps = trajectory_raw.get("every_steps", 1)
        if not isinstance(enabled, bool):
            raise TypeError("trajectory.enabled must be boolean")
        if (
            isinstance(every_steps, bool)
            or not isinstance(every_steps, int)
            or every_steps <= 0
        ):
            raise ValueError("trajectory.every_steps must be a positive integer")
        return StandardDenoisingConfig(
            weights=cast(WeightSelection, weights),
            prediction_type=cast(PredictionType, prediction_type),
            clip_denoised=clip_denoised,
            sampler=sampler,
            trajectory=DenoisingTrajectoryConfig(enabled, every_steps),
        )

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
        return ComponentConfig(name=name, params=deepcopy(params))

    @staticmethod
    def _batch_counts(num_samples: int, batch_size: int) -> tuple[int, ...]:
        return tuple(
            min(batch_size, num_samples - offset)
            for offset in range(0, num_samples, batch_size)
        )


def _validate_shape(
    value: torch.Tensor,
    expected: torch.Size,
    *,
    label: str,
) -> None:
    if value.shape != expected:
        raise ValueError(
            f"standard_denoising {label} has shape {tuple(value.shape)}, "
            f"expected {tuple(expected)}"
        )


__all__ = [
    "InferenceModelProvider",
    "SamplingBuilder",
    "SamplingBuilderContext",
    "SamplingOutput",
    "StandardDenoisingBuilder",
]
