"""Pure Gaussian prediction-parameterization mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

PredictionType = Literal["epsilon", "x0", "v", "score"]


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


def normalize_gaussian_prediction(
    state: torch.Tensor,
    model_output: object,
    *,
    signal_scale: torch.Tensor,
    noise_scale: torch.Tensor,
    prediction_type: PredictionType = "epsilon",
) -> GaussianPrediction:
    """Normalize a model output using explicit Gaussian marginal scales."""

    state = _validate_state(state)
    model_output = _validate_model_output(model_output, state=state)
    signal_scale, noise_scale = _validate_scales(
        signal_scale,
        noise_scale,
        reference=state,
    )
    prediction_type = _validate_prediction_type(prediction_type)
    if prediction_type == "epsilon":
        epsilon = model_output
        clean = (state - noise_scale * epsilon) / signal_scale
    elif prediction_type == "x0":
        clean = model_output
        epsilon = (state - signal_scale * clean) / noise_scale
    elif prediction_type == "v":
        scale_energy = signal_scale.square() + noise_scale.square()
        clean = (
            signal_scale * state - noise_scale * model_output
        ) / scale_energy
        epsilon = (
            noise_scale * state + signal_scale * model_output
        ) / scale_energy
    else:
        epsilon = -noise_scale * model_output
        clean = (state - noise_scale * epsilon) / signal_scale
    return GaussianPrediction(clean, epsilon, model_output)


def _validate_state(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian prediction state must be a Tensor")
    if value.ndim == 0:
        raise ValueError("Gaussian prediction state must include a batch dimension")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian prediction state must be floating-point")
    return value


def _validate_model_output(
    value: object,
    *,
    state: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian predict_fn must return a Tensor")
    if value.shape != state.shape:
        raise ValueError("Gaussian predict_fn output must match the state shape")
    if value.device != state.device:
        raise ValueError("Gaussian model output must share the state device")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian model output must be floating-point")
    return value


def _validate_scales(
    signal_scale: object,
    noise_scale: object,
    *,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(signal_scale, torch.Tensor) or not isinstance(
        noise_scale, torch.Tensor
    ):
        raise TypeError("Gaussian signal and noise scales must be Tensors")
    if signal_scale.shape != noise_scale.shape:
        raise ValueError("Gaussian signal and noise scales must share a shape")
    try:
        broadcast_shape = torch.broadcast_shapes(
            reference.shape,
            signal_scale.shape,
        )
    except RuntimeError as exc:
        raise ValueError("Gaussian scales must broadcast to the state") from exc
    if tuple(broadcast_shape) != tuple(reference.shape):
        raise ValueError("Gaussian scales must broadcast to the state")
    return signal_scale, noise_scale


def _validate_prediction_type(value: object) -> PredictionType:
    if not isinstance(value, str) or value not in ("epsilon", "x0", "v", "score"):
        raise ValueError(
            "Gaussian prediction_type must be epsilon, x0, v, or score"
        )
    return cast(PredictionType, value)


__all__ = [
    "GaussianPrediction",
    "PredictionType",
    "normalize_gaussian_prediction",
]
