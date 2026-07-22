"""Gaussian denoising dynamics contracts and model adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch

from stochaflow.processes import DiscreteGaussianDenoisingProcess

from .dynamics import GenerativeDynamics

PredictionType = Literal["epsilon", "x0", "v", "score"]
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


class GaussianModelDynamics(GaussianDenoisingDynamics):
    """Adapt a model prediction callable to Gaussian denoising dynamics."""

    predict_fn: PredictFn
    prediction_type: PredictionType
    clip_denoised: bool

    def __init__(
        self,
        process: DiscreteGaussianDenoisingProcess,
        predict_fn: PredictFn,
        *,
        prediction_type: PredictionType = "epsilon",
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
        self._process = process_value
        self.predict_fn = cast(PredictFn, predict_fn_value)
        self.prediction_type = cast(PredictionType, prediction_type_value)
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
        model_times = state_times - self.process.clean_time - 1
        model_output = self.predict_fn(state, model_times)
        return normalize_gaussian_prediction(
            self.process,
            state,
            state_times,
            model_output,
            prediction_type=self.prediction_type,
            clip_denoised=self.clip_denoised,
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
        clean = signal * state - noise * model_output
        epsilon = noise * state + signal * model_output
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
    return value


__all__ = [
    "GaussianDenoisingDynamics",
    "GaussianModelDynamics",
    "GaussianPrediction",
    "GaussianTransition",
    "PredictionType",
    "normalize_gaussian_prediction",
]
