"""Conditional reconstruction dynamics, solver composition, and builder."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import torch

from stochaflow.extensions import (
    REGISTRIES,
    ComponentConfig,
    DDIMSampler,
    DiscreteGaussianDenoisingProcess,
    GaussianDenoisingDynamics,
    GaussianPrediction,
    GenerativeDynamics,
    PredictionType,
    Sampler,
    SamplerResult,
    SamplingBatch,
    SamplingBuilder,
    SamplingObservation,
    SamplingObserver,
    SamplingOutput,
    normalize_gaussian_prediction,
)

from ._alignment import load_alignment
from ._config import (
    copied_mapping,
    optional_mapping,
    pop_bool,
    pop_float,
    pop_int,
    pop_optional_range,
    pop_path,
    pop_string,
    reject_unknown,
    required_mapping,
)
from .data import TrajectoryTripletDataset
from .model import ConditionalDenoiser
from .physics import (
    conditioning_gradient,
    correction_gradient,
    correction_residual_loss,
)

WeightSelection = Literal["auto", "raw", "ema"]


class PhysicsCorrectionDynamics(GaussianDenoisingDynamics, ABC):
    """Project-private Gaussian capability required by guided DDIM."""

    @abstractmethod
    def evaluate(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> tuple[GaussianPrediction, torch.Tensor, Mapping[str, float]]:
        """Atomically return prediction and source-state correction."""


class PhysicsGaussianDynamics(PhysicsCorrectionDynamics):
    """Combine the Process, conditional model, and two physics directions."""

    def __init__(
        self,
        process: DiscreteGaussianDenoisingProcess,
        model: ConditionalDenoiser,
        *,
        prediction_type: PredictionType,
        clip_denoised: bool,
        conditioning_strength: float,
        correction_strength: float,
    ) -> None:
        self._process = process
        self.model = model
        self.prediction_type: PredictionType = prediction_type
        self.clip_denoised = clip_denoised
        self.conditioning_strength = conditioning_strength
        self.correction_strength = correction_strength

    @property
    def process(self) -> DiscreteGaussianDenoisingProcess:
        return self._process

    def predict(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> GaussianPrediction:
        state_times = self.process.validate_noisy_state_times(state_times)
        condition, _ = conditioning_gradient(
            state,
            self.model,
            strength=self.conditioning_strength,
        )
        return self._prediction(state, state_times, condition)

    def evaluate(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> tuple[GaussianPrediction, torch.Tensor, Mapping[str, float]]:
        state_times = self.process.validate_noisy_state_times(state_times)
        condition, condition_residual = conditioning_gradient(
            state,
            self.model,
            strength=self.conditioning_strength,
        )
        correction, correction_residual = correction_gradient(
            state,
            self.model,
            strength=self.correction_strength,
        )
        prediction = self._prediction(state, state_times, condition)
        return prediction, correction, {
            "conditioning_residual": float(condition_residual.cpu()),
            "correction_residual": float(correction_residual.cpu()),
        }

    def _prediction(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
        condition: torch.Tensor,
    ) -> GaussianPrediction:
        model_times = state_times - self.process.clean_time - 1
        output = self.model(state, model_times, condition)
        return normalize_gaussian_prediction(
            self.process,
            state,
            state_times,
            output,
            prediction_type=self.prediction_type,
            clip_denoised=self.clip_denoised,
        )

@REGISTRIES.samplers.register("physics-reconstruction.guided-ddim")
class GuidedDDIMSampler(Sampler):
    """Compose public DDIM primitives with a post-transition physics correction."""

    def __init__(
        self,
        *,
        num_inference_steps: object = None,
        schedule: object = None,
        eta: object = 0.0,
    ) -> None:
        self._ddim = DDIMSampler(
            num_inference_steps=num_inference_steps,
            schedule=schedule,
            eta=eta,
        )

    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        if not isinstance(dynamics, PhysicsCorrectionDynamics):
            raise TypeError(
                "physics-reconstruction.guided-ddim requires "
                "PhysicsCorrectionDynamics"
            )
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("guided DDIM initial_state must be a Tensor")
        process = dynamics.process
        states = self._ddim.resolve_schedule(process, device=initial_state.device)
        num_steps = int(states.numel() - 1)
        current = initial_state
        if observer is not None:
            observer.observe(
                SamplingObservation(0, int(states[0]), current, False, {})
            )
        residuals: list[float] = []
        for step_index, (source, target) in enumerate(
            pairwise(states),
            start=1,
        ):
            source_state = current
            source_times = source.expand(source_state.shape[0])
            target_times = target.expand(source_state.shape[0])
            evaluation = cast(object, dynamics.evaluate(source_state, source_times))
            if not isinstance(evaluation, tuple) or len(evaluation) != 3:
                raise TypeError(
                    "physics correction dynamics evaluate() must return "
                    "(prediction, correction, diagnostics)"
                )
            prediction, correction_value, correction_diagnostics_value = evaluation
            if not isinstance(correction_value, torch.Tensor):
                raise TypeError("physics correction must be a Tensor")
            correction = correction_value
            if not torch.is_floating_point(correction):
                raise TypeError("physics correction must be floating-point")
            if correction.shape != source_state.shape:
                raise ValueError("physics correction must match the source state shape")
            if correction.device != source_state.device:
                raise ValueError("physics correction must share the source state device")
            if correction.dtype != source_state.dtype:
                raise ValueError("physics correction must share the source state dtype")
            if not isinstance(correction_diagnostics_value, Mapping):
                raise TypeError("physics correction diagnostics must be a mapping")
            correction_diagnostics = correction_diagnostics_value
            residual_value = correction_diagnostics.get("correction_residual")
            if (
                isinstance(residual_value, bool)
                or not isinstance(residual_value, (int, float))
                or not math.isfinite(float(residual_value))
            ):
                raise ValueError(
                    "physics correction diagnostics require finite numeric "
                    "correction_residual"
                )
            transitioned = self._ddim.transition(
                process,
                source_state,
                source_times,
                target_times,
                prediction,
            ).sample(generator=generator)
            current = transitioned - correction
            residuals.append(float(residual_value))
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index,
                        int(target),
                        current,
                        step_index == num_steps,
                        {
                            "num_dynamics_evaluations": step_index,
                            "num_correction_evaluations": step_index,
                            **correction_diagnostics,
                        },
                    )
                )
        return SamplerResult(
            current,
            num_steps,
            {
                "num_dynamics_evaluations": num_steps,
                "num_correction_evaluations": num_steps,
                "mean_correction_residual": (
                    sum(residuals) / len(residuals) if residuals else 0.0
                ),
            },
        )


class ReconstructionObserver:
    def __init__(
        self,
        *,
        start_time: int,
        end_time: int,
        expected_shape: torch.Size,
        enabled: bool,
        every_steps: int,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.expected_shape = expected_shape
        self.enabled = enabled
        self.every_steps = every_steps
        self._last_step = -1
        self._final_seen = False
        self._observations: list[SamplingObservation] = []

    @property
    def observations(self) -> tuple[SamplingObservation, ...] | None:
        return tuple(self._observations) if self.enabled else None

    def observe(self, observation: SamplingObservation) -> None:
        if observation.step_index <= self._last_step:
            raise ValueError("reconstruction sampler step indices must increase")
        if self._last_step < 0 and (
            observation.step_index != 0
            or observation.coordinate != self.start_time
        ):
            raise ValueError(
                "reconstruction sampler must start at the configured partial time"
            )
        if self._final_seen:
            raise ValueError("reconstruction sampler emitted after its final state")
        if not isinstance(observation.state, torch.Tensor):
            raise TypeError("reconstruction observations must contain Tensors")
        if observation.state.shape != self.expected_shape:
            raise ValueError("reconstruction observation shape changed")
        if observation.is_final:
            if observation.coordinate != self.end_time:
                raise ValueError(
                    "reconstruction sampler must end at the process clean time"
                )
            self._final_seen = True
        self._last_step = observation.step_index
        if self.enabled and (
            observation.step_index == 0
            or observation.is_final
            or observation.step_index % self.every_steps == 0
        ):
            self._observations.append(
                SamplingObservation(
                    observation.step_index,
                    observation.coordinate,
                    observation.state.detach().to(device="cpu", copy=True),
                    observation.is_final,
                    dict(observation.diagnostics),
                )
            )

    def validate_complete(self, result: SamplerResult) -> None:
        if not self._final_seen:
            raise ValueError("reconstruction sampler did not emit a final state")
        if self._last_step != result.num_steps:
            raise ValueError(
                "final observation step must equal SamplerResult.num_steps"
            )


@dataclass(frozen=True, slots=True)
class ReconstructionSourceConfig:
    path: Path
    trajectory_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ReconstructionTrajectoryConfig:
    enabled: bool
    every_steps: int


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    weights: WeightSelection
    prediction_type: PredictionType
    clip_denoised: bool
    partial_noise_time: int
    conditioning_strength: float
    correction_strength: float
    source: ReconstructionSourceConfig
    reference: ReconstructionSourceConfig | None
    alignment: Path | None
    sampler: ComponentConfig
    trajectory: ReconstructionTrajectoryConfig


@REGISTRIES.sampling_builders.register("physics-reconstruction.reconstruction")
class ReconstructionSamplingBuilder(SamplingBuilder):
    """Run time-aligned partial noising and one registered Gaussian sampler."""

    def run(self) -> SamplingOutput:
        config = self._parse(self.context.params)
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "physics reconstruction requires DiscreteGaussianDenoisingProcess"
            )
        if not process.clean_time < config.partial_noise_time <= process.terminal_time:
            raise ValueError("partial_noise_time must lie in (clean_time, terminal_time]")
        model_value, resolved_weights = self.context.model_provider.resolve(
            config.weights
        )
        if not isinstance(model_value, ConditionalDenoiser):
            raise TypeError("physics reconstruction requires ConditionalDenoiser")
        model = model_value
        source = TrajectoryTripletDataset(
            config.source.path,
            config.source.trajectory_range,
        )
        reference = (
            TrajectoryTripletDataset(
                config.reference.path,
                config.reference.trajectory_range,
            )
            if config.reference is not None
            else None
        )
        if config.alignment is not None:
            self._validate_alignment(config.alignment, source, reference)
        if self.context.num_samples > len(source):
            raise ValueError(
                f"sampling requests {self.context.num_samples} items but source "
                f"contains {len(source)}"
            )
        if reference is not None and self.context.num_samples > len(reference):
            raise ValueError("reference source has fewer items than sampling requests")
        if reference is not None and (
            len(reference) != len(source)
            or reference.sample_shape != source.sample_shape
        ):
            raise ValueError(
                "reference and observation sources must align positionally"
            )
        if self.context.shape is not None and tuple(self.context.shape) != source.sample_shape:
            raise ValueError(
                f"sample.shape {self.context.shape} does not match source "
                f"{source.sample_shape}"
            )
        sampler = cast(
            Sampler,
            REGISTRIES.samplers.create(
                config.sampler.name,
                **config.sampler.params,
            ),
        )
        dynamics = PhysicsGaussianDynamics(
            process,
            model,
            prediction_type=config.prediction_type,
            clip_denoised=config.clip_denoised,
            conditioning_strength=config.conditioning_strength,
            correction_strength=config.correction_strength,
        )
        generator = torch.Generator(device=self.context.device).manual_seed(
            self.context.seed
        )
        batches: list[SamplingBatch] = []
        solver_diagnostics: list[dict[str, Any]] = []
        observation_error = 0.0
        reference_error = 0.0
        physics_error = 0.0
        for offset in range(0, self.context.num_samples, self.context.batch_size):
            count = min(
                self.context.batch_size,
                self.context.num_samples - offset,
            )
            observed_physical = torch.stack(
                [source[offset + index] for index in range(count)]
            ).to(self.context.device)
            normalized = model.normalize(observed_physical)
            state_times = torch.full(
                (count,),
                config.partial_noise_time,
                device=self.context.device,
                dtype=torch.long,
            )
            initial, _ = process.sample_marginal(
                normalized,
                state_times,
                generator=generator,
            )
            lifecycle = ReconstructionObserver(
                start_time=config.partial_noise_time,
                end_time=process.clean_time,
                expected_shape=initial.shape,
                enabled=config.trajectory.enabled,
                every_steps=config.trajectory.every_steps,
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
                raise TypeError("reconstruction sampler must return SamplerResult")
            lifecycle.validate_complete(result_value)
            final = result_value.final_state
            if not isinstance(final, torch.Tensor) or final.shape != initial.shape:
                raise TypeError(
                    "reconstruction sampler final state must be a shape-preserving Tensor"
                )
            reconstructed = model.denormalize(final)
            observation_error += float(
                (reconstructed - observed_physical)
                .square()
                .flatten(1)
                .mean(dim=1)
                .sum()
                .cpu()
            )
            physics_error += float(
                correction_residual_loss(reconstructed, model).cpu()
            ) * count
            if reference is not None:
                reference_physical = torch.stack(
                    [reference[offset + index] for index in range(count)]
                ).to(self.context.device)
                reference_error += float(
                    (reconstructed - reference_physical)
                    .square()
                    .flatten(1)
                    .mean(dim=1)
                    .sum()
                    .cpu()
                )
            trajectory = self._physical_trajectory(
                lifecycle.observations,
                model,
            )
            batches.append(
                SamplingBatch(
                    samples=reconstructed.detach().to(device="cpu", copy=True),
                    num_samples=count,
                    trajectory=trajectory,
                )
            )
            solver_diagnostics.append(dict(result_value.diagnostics))
        metrics: dict[str, float | int] = {
            "num_samples": self.context.num_samples,
            "mean_mse_to_observation": observation_error / self.context.num_samples,
            "mean_physics_residual": physics_error / self.context.num_samples,
        }
        if reference is not None:
            metrics["mean_mse_to_reference"] = (
                reference_error / self.context.num_samples
            )
        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "workflow": "physics-reconstruction",
                "weights": resolved_weights,
                "prediction_type": config.prediction_type,
                "clip_denoised": config.clip_denoised,
                "partial_noise_time": config.partial_noise_time,
                "conditioning_strength": config.conditioning_strength,
                "correction_strength": config.correction_strength,
                "sampler": {
                    "name": config.sampler.name,
                    "params": deepcopy(config.sampler.params),
                },
                "trajectory": {
                    "enabled": config.trajectory.enabled,
                    "every_steps": config.trajectory.every_steps,
                },
                "metrics": metrics,
                "solver_diagnostics": solver_diagnostics,
                "alignment": (
                    str(config.alignment) if config.alignment is not None else None
                ),
            },
        )

    @staticmethod
    def _physical_trajectory(
        observations: tuple[SamplingObservation, ...] | None,
        model: ConditionalDenoiser,
    ) -> tuple[SamplingObservation, ...] | None:
        if observations is None:
            return None
        mean = float(model.normalization_mean.detach().cpu())
        scale = float(model.normalization_scale.detach().cpu())
        return tuple(
            SamplingObservation(
                observation.step_index,
                observation.coordinate,
                observation.state.detach().cpu() * scale + mean,
                observation.is_final,
                dict(observation.diagnostics),
            )
            for observation in observations
        )

    @classmethod
    def _parse(cls, raw: dict[str, Any]) -> ReconstructionConfig:
        params = copied_mapping(raw, path="resolved sampling recipe")
        weights = pop_string(
            params,
            "weights",
            path="sample.options",
            default="auto",
        )
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("sampling weights must be auto, raw, or ema")
        prediction_type = pop_string(
            params,
            "prediction_type",
            path="checkpoint.inference_recipe.contract",
            default="epsilon",
        )
        if prediction_type not in {"epsilon", "x0", "v", "score"}:
            raise ValueError("sampling prediction_type must be epsilon, x0, v, or score")
        clip_denoised = pop_bool(
            params,
            "clip_denoised",
            path="sample.options",
            default=False,
        )
        if clip_denoised:
            raise ValueError(
                "physics reconstruction forbids clip_denoised because normalized "
                "vorticity is not bounded to [-1, 1]"
            )
        partial_noise_time = pop_int(
            params,
            "partial_noise_time",
            path="sample.options",
        )
        conditioning_strength = pop_float(
            params,
            "conditioning_strength",
            path="sample.options",
            default=1.0,
            minimum=0.0,
        )
        correction_strength = pop_float(
            params,
            "correction_strength",
            path="sample.options",
            default=1.0,
            minimum=0.0,
        )
        source = cls._source(
            required_mapping(params, "source", path="sample.options"),
            path="sample.options.source",
        )
        reference_raw = optional_mapping(
            params,
            "reference",
            path="sample.options",
        )
        reference = (
            cls._source(reference_raw, path="sample.options.reference")
            if reference_raw is not None
            else None
        )
        alignment_value = params.pop("alignment", None)
        if alignment_value is not None and (
            not isinstance(alignment_value, str) or not alignment_value.strip()
        ):
            raise ValueError("sample.options.alignment must be a path string")
        alignment = Path(alignment_value) if alignment_value is not None else None
        sampler = cls._component(
            required_mapping(params, "sampler", path="sample"),
            path="sample.sampler",
        )
        trajectory_raw = optional_mapping(
            params,
            "trajectory",
            path="sample.options",
        ) or {}
        enabled = pop_bool(
            trajectory_raw,
            "enabled",
            path="sample.options.trajectory",
            default=False,
        )
        every_steps = pop_int(
            trajectory_raw,
            "every_steps",
            path="sample.options.trajectory",
            default=1,
        )
        reject_unknown(
            trajectory_raw,
            path="sample.options.trajectory",
        )
        reject_unknown(params, path="resolved sampling recipe")
        return ReconstructionConfig(
            cast(WeightSelection, weights),
            cast(PredictionType, prediction_type),
            clip_denoised,
            partial_noise_time,
            conditioning_strength,
            correction_strength,
            source,
            reference,
            alignment,
            sampler,
            ReconstructionTrajectoryConfig(enabled, every_steps),
        )

    @staticmethod
    def _source(raw: dict[str, Any], *, path: str) -> ReconstructionSourceConfig:
        source_path = pop_path(raw, "path", path=path)
        trajectory_range = pop_optional_range(raw, "trajectories", path=path)
        reject_unknown(raw, path=path)
        if trajectory_range is None:
            raise ValueError(f"{path}.trajectories is required")
        return ReconstructionSourceConfig(source_path, trajectory_range)

    @staticmethod
    def _component(raw: dict[str, Any], *, path: str) -> ComponentConfig:
        name = pop_string(raw, "name", path=path)
        component_params = copied_mapping(
            raw.pop("params", {}),
            path=f"{path}.params",
        )
        reject_unknown(raw, path=path)
        return ComponentConfig(name=name, params=component_params)

    @staticmethod
    def _validate_alignment(
        path: Path,
        source: TrajectoryTripletDataset,
        reference: TrajectoryTripletDataset | None,
    ) -> None:
        if reference is None:
            raise ValueError("alignment sidecar requires a reference source")
        alignment = load_alignment(path)
        checks = (
            (
                "observation",
                alignment.observation,
                source,
            ),
            (
                "reference",
                alignment.reference,
                reference,
            ),
        )
        for label, declared, dataset in checks:
            if declared.path != dataset.path.resolve():
                raise ValueError(f"alignment {label} path does not match config")
            if declared.trajectory_range != dataset.trajectory_range:
                raise ValueError(
                    f"alignment {label} trajectory range does not match config"
                )
            if declared.shape != dataset.source_shape:
                raise ValueError(f"alignment {label} shape does not match data")
        if alignment.sample_count != len(source) or len(reference) != len(source):
            raise ValueError("alignment sample_count does not match paired data")


__all__ = [
    "GuidedDDIMSampler",
    "PhysicsCorrectionDynamics",
    "PhysicsGaussianDynamics",
    "ReconstructionSamplingBuilder",
]
