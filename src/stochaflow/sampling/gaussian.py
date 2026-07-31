"""Gaussian denoising dynamics contracts and model adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import torch

from stochaflow.processes import (
    DiscreteGaussianDenoisingProcess,
    LearnedRangeGaussianVarianceProcess,
)

from .dynamics import GenerativeDynamics

PredictionType = Literal["epsilon", "x0", "v", "score"]
VarianceMode = Literal["fixed", "learned_range"]
PredictFn = Callable[[torch.Tensor, torch.Tensor], object]


@dataclass(frozen=True, slots=True)
class GaussianPrediction:
    """Equivalent clean-state and noise predictions at one noisy state."""

    clean: torch.Tensor
    epsilon: torch.Tensor
    model_output: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("clean", "epsilon", "model_output"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"GaussianPrediction.{name} must be a Tensor")
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"GaussianPrediction.{name} must be floating-point"
                )


@dataclass(frozen=True, slots=True)
class LearnedVarianceGaussianPrediction(GaussianPrediction):
    """Gaussian prediction carrying target-aware learned log variance."""

    log_variance: torch.Tensor

    def __post_init__(self) -> None:
        GaussianPrediction.__post_init__(self)
        log_variance_value = cast(object, self.log_variance)
        if not isinstance(log_variance_value, torch.Tensor):
            raise TypeError(
                "LearnedVarianceGaussianPrediction.log_variance must be a Tensor"
            )
        if not torch.is_floating_point(log_variance_value):
            raise TypeError(
                "LearnedVarianceGaussianPrediction.log_variance must be "
                "floating-point"
            )


@dataclass(frozen=True, slots=True)
class GaussianTransition:
    """One discrete Gaussian transition distribution."""

    mean: torch.Tensor
    standard_deviation: torch.Tensor

    def __post_init__(self) -> None:
        mean_value = cast(object, self.mean)
        if not isinstance(mean_value, torch.Tensor):
            raise TypeError("GaussianTransition.mean must be a Tensor")
        standard_deviation_value = cast(object, self.standard_deviation)
        if not isinstance(standard_deviation_value, torch.Tensor):
            raise TypeError(
                "GaussianTransition.standard_deviation must be a Tensor"
            )
        if not torch.is_floating_point(mean_value):
            raise TypeError("GaussianTransition.mean must be floating-point")
        if not torch.is_floating_point(standard_deviation_value):
            raise TypeError(
                "GaussianTransition.standard_deviation must be floating-point"
            )
        if mean_value.device != standard_deviation_value.device:
            raise ValueError("GaussianTransition tensors must share a device")
        if mean_value.dtype != standard_deviation_value.dtype:
            raise ValueError("GaussianTransition tensors must share a dtype")
        try:
            broadcast_shape = torch.broadcast_shapes(
                mean_value.shape,
                standard_deviation_value.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "GaussianTransition standard deviation must broadcast to its mean"
            ) from exc
        if tuple(broadcast_shape) != tuple(mean_value.shape):
            raise ValueError(
                "GaussianTransition standard deviation must broadcast to its mean"
            )
        if bool(torch.any(standard_deviation_value < 0)):
            raise ValueError(
                "GaussianTransition standard deviation must be non-negative"
            )

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the transition without consuming RNG for zero variance."""

        if not bool(torch.any(self.standard_deviation != 0)):
            return self.mean
        noise = torch.randn(
            self.mean.shape,
            device=self.mean.device,
            dtype=self.mean.dtype,
            generator=generator,
        )
        return self.mean + self.standard_deviation * noise


class GaussianDenoisingDynamics(GenerativeDynamics, ABC):
    """Family capability consumed by Gaussian denoising samplers."""

    @property
    @abstractmethod
    def process(self) -> DiscreteGaussianDenoisingProcess:
        """Return the model-free Gaussian process used by this direction."""

    @abstractmethod
    def predict(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> GaussianPrediction:
        """Return normalized Gaussian prediction semantics for one state."""


@runtime_checkable
class TargetAwareGaussianDenoisingDynamics(Protocol):
    """Optional dynamics capability for target-dependent Gaussian predictions."""

    def predict_transition(
        self,
        state: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> GaussianPrediction:
        """Return prediction semantics for one selected reverse pair."""

        ...


@runtime_checkable
class CleanTargetVarianceReferenceGaussianDenoisingDynamics(Protocol):
    """Dynamics accepting the preceding pair for final variance clipping."""

    def predict_transition_with_variance_reference(
        self,
        state: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        *,
        clean_target_reference_times: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> GaussianPrediction:
        """Return a target-aware prediction with respaced clipping context."""

        ...


class GaussianModelDynamics(GaussianDenoisingDynamics):
    """Adapt a model prediction callable to Gaussian denoising dynamics."""

    predict_fn: PredictFn
    prediction_type: PredictionType
    variance_mode: VarianceMode
    clip_denoised: bool

    def __init__(
        self,
        process: DiscreteGaussianDenoisingProcess,
        predict_fn: PredictFn,
        *,
        prediction_type: PredictionType = "epsilon",
        variance_mode: VarianceMode = "fixed",
        clip_denoised: bool = True,
    ) -> None:
        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "Gaussian dynamics require DiscreteGaussianDenoisingProcess"
            )
        predict_fn_value = cast(object, predict_fn)
        if not callable(predict_fn_value):
            raise TypeError("Gaussian predict_fn must be callable")
        prediction_type_value = cast(object, prediction_type)
        if not isinstance(prediction_type_value, str) or prediction_type_value not in (
            "epsilon",
            "x0",
            "v",
            "score",
        ):
            raise ValueError(
                "Gaussian prediction_type must be epsilon, x0, v, or score"
            )
        clip_denoised_value = cast(object, clip_denoised)
        if not isinstance(clip_denoised_value, bool):
            raise TypeError("Gaussian clip_denoised must be boolean")
        variance_mode_value = cast(object, variance_mode)
        if (
            not isinstance(variance_mode_value, str)
            or variance_mode_value not in ("fixed", "learned_range")
        ):
            raise ValueError(
                "Gaussian variance_mode must be fixed or learned_range"
            )
        if variance_mode_value == "learned_range" and not isinstance(
            process_value,
            LearnedRangeGaussianVarianceProcess,
        ):
            raise TypeError(
                "learned_range Gaussian dynamics require "
                "LearnedRangeGaussianVarianceProcess"
            )
        self._process = process_value
        self.predict_fn = cast(PredictFn, predict_fn_value)
        self.prediction_type = cast(PredictionType, prediction_type_value)
        self.variance_mode = cast(VarianceMode, variance_mode_value)
        self.clip_denoised = clip_denoised_value

    @property
    def process(self) -> DiscreteGaussianDenoisingProcess:
        """Return the model-free Gaussian process used by this adapter."""

        return self._process

    def predict(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> GaussianPrediction:
        """Evaluate the model and normalize its prediction semantics."""

        state_times = self.process.validate_noisy_state_times(state_times)
        model_output, _ = self._model_output_heads(state, state_times)
        return normalize_gaussian_prediction(
            self.process,
            state,
            state_times,
            model_output,
            prediction_type=self.prediction_type,
            clip_denoised=self.clip_denoised,
        )

    def predict_transition(
        self,
        state: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> GaussianPrediction:
        """Evaluate one target-aware fixed or learned-range prediction."""

        return self.predict_transition_with_variance_reference(
            state,
            source_times,
            target_times,
        )

    def predict_transition_with_variance_reference(
        self,
        state: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        *,
        clean_target_reference_times: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> GaussianPrediction:
        """Evaluate one target-aware fixed or learned-range prediction."""

        source_times = self.process.validate_noisy_state_times(source_times)
        target_times = _validate_target_times(
            self.process,
            source_times,
            target_times,
        )
        model_output, variance_values = self._model_output_heads(
            state,
            source_times,
        )
        prediction = normalize_gaussian_prediction(
            self.process,
            state,
            source_times,
            model_output,
            prediction_type=self.prediction_type,
            clip_denoised=self.clip_denoised,
        )
        if self.variance_mode == "fixed":
            return prediction
        if variance_values is None:
            raise RuntimeError(
                "learned_range Gaussian dynamics produced no variance values"
            )
        process = cast(LearnedRangeGaussianVarianceProcess, self.process)
        if clean_target_reference_times is None:
            bounds = process.reverse_log_variance_bounds(
                source_times,
                target_times,
                state.size(),
            )
        else:
            bounds = process.reverse_log_variance_bounds(
                source_times,
                target_times,
                state.size(),
                clean_target_reference_times=clean_target_reference_times,
            )
        fraction = (variance_values + 1.0) / 2.0
        upper = bounds.upper.to(
            device=variance_values.device,
            dtype=variance_values.dtype,
        )
        lower = bounds.lower.to(
            device=variance_values.device,
            dtype=variance_values.dtype,
        )
        log_variance = (
            fraction * upper + (1.0 - fraction) * lower
        ).to(dtype=state.dtype)
        return LearnedVarianceGaussianPrediction(
            clean=prediction.clean,
            epsilon=prediction.epsilon,
            model_output=prediction.model_output,
            log_variance=log_variance,
        )

    def _model_output_heads(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        model_times = state_times - self.process.clean_time - 1
        raw_output = self.predict_fn(state, model_times)
        return _split_gaussian_model_output(
            raw_output,
            state=state,
            variance_mode=self.variance_mode,
        )


def normalize_gaussian_prediction(
    process: DiscreteGaussianDenoisingProcess,
    state: torch.Tensor,
    state_times: torch.Tensor,
    model_output: object,
    *,
    prediction_type: PredictionType = "epsilon",
    clip_denoised: bool = True,
) -> GaussianPrediction:
    """Normalize one raw model output into clean and epsilon predictions."""

    process_value = cast(object, process)
    if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "Gaussian prediction normalization requires "
            "DiscreteGaussianDenoisingProcess"
        )
    process = process_value
    state_value = cast(object, state)
    if not isinstance(state_value, torch.Tensor):
        raise TypeError("Gaussian prediction state must be a Tensor")
    state = state_value
    if state.ndim == 0:
        raise ValueError("Gaussian prediction state must include a batch dimension")
    if not torch.is_floating_point(state):
        raise TypeError("Gaussian prediction state must be floating-point")
    state_times = process.validate_noisy_state_times(state_times)
    if state_times.shape[0] != state.shape[0]:
        raise ValueError("Gaussian state times must match the state batch")
    if state_times.device != state.device:
        raise ValueError("Gaussian state times must share the state device")
    if not isinstance(model_output, torch.Tensor):
        raise TypeError("Gaussian predict_fn must return a Tensor")
    if model_output.shape != state.shape:
        raise ValueError("Gaussian predict_fn output must match the state shape")
    if model_output.device != state.device:
        raise ValueError("Gaussian model output must share the state device")
    if not torch.is_floating_point(model_output):
        raise TypeError("Gaussian model output must be floating-point")
    prediction_type_value = cast(object, prediction_type)
    if not isinstance(prediction_type_value, str) or prediction_type_value not in (
        "epsilon",
        "x0",
        "v",
        "score",
    ):
        raise ValueError(
            "Gaussian prediction_type must be epsilon, x0, v, or score"
        )
    prediction_type = cast(PredictionType, prediction_type_value)
    clip_denoised_value = cast(object, clip_denoised)
    if not isinstance(clip_denoised_value, bool):
        raise TypeError("Gaussian clip_denoised must be boolean")
    clip_denoised = clip_denoised_value
    scales = process.marginal_scales(state_times, state.size())
    signal = scales.signal
    noise = scales.noise
    if prediction_type == "epsilon":
        epsilon = model_output
        clean = (state - noise * epsilon) / signal
    elif prediction_type == "x0":
        clean = model_output
        epsilon = (state - signal * clean) / noise
    elif prediction_type == "v":
        scale_energy = signal.square() + noise.square()
        clean = (signal * state - noise * model_output) / scale_energy
        epsilon = (noise * state + signal * model_output) / scale_energy
    else:
        epsilon = -noise * model_output
        clean = (state - noise * epsilon) / signal
    if clip_denoised:
        clean = clean.clamp(-1.0, 1.0)
        epsilon = (state - signal * clean) / noise
    return GaussianPrediction(clean, epsilon, model_output)


def _validate_gaussian_prediction(
    value: object,
    *,
    state: torch.Tensor,
) -> GaussianPrediction:
    if not isinstance(value, GaussianPrediction):
        raise TypeError("Gaussian dynamics must return GaussianPrediction")
    for name in ("clean", "epsilon", "model_output"):
        tensor = getattr(value, name)
        if tensor.shape != state.shape:
            raise ValueError(f"Gaussian prediction {name} must match the state shape")
        if tensor.device != state.device:
            raise ValueError(f"Gaussian prediction {name} must share the state device")
    if isinstance(value, LearnedVarianceGaussianPrediction):
        log_variance = value.log_variance
        if log_variance.device != state.device:
            raise ValueError(
                "Gaussian prediction log_variance must share the state device"
            )
        if log_variance.dtype != state.dtype:
            raise ValueError(
                "Gaussian prediction log_variance must share the state dtype"
            )
        try:
            broadcast_shape = torch.broadcast_shapes(
                state.shape,
                log_variance.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Gaussian prediction log_variance must broadcast to the state"
            ) from exc
        if tuple(broadcast_shape) != tuple(state.shape):
            raise ValueError(
                "Gaussian prediction log_variance must broadcast to the state"
            )
    return value


def _split_gaussian_model_output(
    value: object,
    *,
    state: torch.Tensor,
    variance_mode: VarianceMode,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian predict_fn must return a Tensor")
    if value.device != state.device:
        raise ValueError("Gaussian model output must share the state device")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian model output must be floating-point")
    if variance_mode == "fixed":
        if value.shape != state.shape:
            raise ValueError("Gaussian predict_fn output must match the state shape")
        return value, None
    if state.ndim < 2:
        raise ValueError(
            "learned_range Gaussian state must include a channel dimension"
        )
    expected_shape = (state.shape[0], state.shape[1] * 2, *state.shape[2:])
    if value.shape != expected_shape:
        raise ValueError(
            "learned_range Gaussian predict_fn output must have shape "
            f"{expected_shape}, got {tuple(value.shape)}"
        )
    model_output, variance_values = value.chunk(2, dim=1)
    return model_output, variance_values


def _validate_target_times(
    process: DiscreteGaussianDenoisingProcess,
    source_times: torch.Tensor,
    target_times: object,
) -> torch.Tensor:
    if not isinstance(target_times, torch.Tensor):
        raise TypeError("Gaussian target times must be a Tensor")
    if target_times.ndim != 1:
        raise ValueError("Gaussian target times must be a 1D tensor")
    if (
        target_times.dtype == torch.bool
        or torch.is_floating_point(target_times)
        or torch.is_complex(target_times)
    ):
        raise TypeError("Gaussian target times must contain integer states")
    normalized = target_times.to(dtype=torch.long)
    if normalized.shape != source_times.shape:
        raise ValueError("Gaussian target times must match source times")
    if normalized.device != source_times.device:
        raise ValueError("Gaussian target times must share the source-time device")
    if torch.any(normalized < process.clean_time) or torch.any(
        normalized > process.terminal_time
    ):
        raise ValueError("Gaussian target times must lie in the process time range")
    if torch.any(normalized >= source_times):
        raise ValueError("Gaussian target times must be smaller than source times")
    return normalized


def _variance_mode_from_declaration(
    value: object,
    *,
    path: str,
) -> VarianceMode:
    if value is None:
        return "fixed"
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    unknown = sorted(set(value) - {"mode"})
    if unknown:
        raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
    mode = value.get("mode", "fixed")
    if not isinstance(mode, str) or mode not in ("fixed", "learned_range"):
        raise ValueError(f"{path}.mode must be fixed or learned_range")
    return cast(VarianceMode, mode)


__all__ = [
    "CleanTargetVarianceReferenceGaussianDenoisingDynamics",
    "GaussianDenoisingDynamics",
    "GaussianModelDynamics",
    "GaussianPrediction",
    "GaussianTransition",
    "LearnedVarianceGaussianPrediction",
    "PredictionType",
    "TargetAwareGaussianDenoisingDynamics",
    "VarianceMode",
    "normalize_gaussian_prediction",
]
