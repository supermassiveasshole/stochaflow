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
        if not isinstance(model_output, torch.Tensor):
            raise TypeError("Gaussian predict_fn must return a Tensor")
        if model_output.shape != state.shape:
            raise ValueError("Gaussian predict_fn output must match the state shape")
        scales = self.process.marginal_scales(state_times, state.size())
        signal = scales.signal
        noise = scales.noise
        if self.prediction_type == "epsilon":
            epsilon = model_output
            clean = (state - noise * epsilon) / signal
        elif self.prediction_type == "x0":
            clean = model_output
            epsilon = (state - signal * clean) / noise
        elif self.prediction_type == "v":
            clean = signal * state - noise * model_output
            epsilon = noise * state + signal * model_output
        else:
            epsilon = -noise * model_output
            clean = (state - noise * epsilon) / signal
        if self.clip_denoised:
            clean = clean.clamp(-1.0, 1.0)
            epsilon = (state - signal * clean) / noise
        return GaussianPrediction(clean, epsilon, model_output)


__all__ = [
    "GaussianDenoisingDynamics",
    "GaussianModelDynamics",
    "GaussianPrediction",
    "PredictionType",
]
