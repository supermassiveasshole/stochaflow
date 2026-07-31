"""Pure prediction mathematics shared by the Gaussian algorithm family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

PredictionType = Literal["epsilon", "x0", "v", "score"]
VarianceMode = Literal["fixed", "learned_range"]


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


def normalize_gaussian_prediction(
    state: torch.Tensor,
    model_output: object,
    *,
    signal_scale: torch.Tensor,
    noise_scale: torch.Tensor,
    prediction_type: PredictionType = "epsilon",
) -> GaussianPrediction:
    """Normalize a model output using explicit Gaussian marginal scales."""

    state = _validate_state(state, path="Gaussian prediction state")
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


def split_gaussian_model_output(
    value: object,
    *,
    state: torch.Tensor,
    variance_mode: VarianceMode,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split fixed or learned-range Gaussian model-output heads."""

    state = _validate_state(state, path="Gaussian prediction state")
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian predict_fn must return a Tensor")
    if value.device != state.device:
        raise ValueError("Gaussian model output must share the state device")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian model output must be floating-point")
    variance_mode = _validate_variance_mode(variance_mode)
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


def gaussian_signal_to_noise_ratio(
    *,
    signal_scale: torch.Tensor,
    noise_scale: torch.Tensor,
) -> torch.Tensor:
    """Return element-aligned signal-to-noise ratios from marginal scales."""

    signal_scale, noise_scale = _validate_scales(signal_scale, noise_scale)
    return signal_scale.square() / noise_scale.square()


def gaussian_training_target(
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    signal_scale: torch.Tensor,
    noise_scale: torch.Tensor,
    prediction_type: PredictionType,
) -> torch.Tensor:
    """Build a Gaussian training target from samples and marginal scales."""

    clean = _validate_state(clean, path="Gaussian training clean state")
    noise_value = cast(object, noise)
    if not isinstance(noise_value, torch.Tensor):
        raise TypeError("Gaussian training noise must be a Tensor")
    noise = noise_value
    if not torch.is_floating_point(noise):
        raise TypeError("Gaussian training noise must be floating-point")
    if noise.shape != clean.shape:
        raise ValueError("Gaussian training noise must match the clean state shape")
    if noise.device != clean.device:
        raise ValueError("Gaussian training noise must share the clean state device")
    if noise.dtype != clean.dtype:
        raise ValueError("Gaussian training noise must share the clean state dtype")
    signal_scale, noise_scale = _validate_scales(
        signal_scale,
        noise_scale,
        reference=clean,
    )
    prediction_type = _validate_prediction_type(prediction_type)
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "x0":
        return clean
    if prediction_type == "v":
        return signal_scale * noise - noise_scale * clean
    return -noise / noise_scale


def _validate_state(value: object, *, path: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must be a Tensor")
    if value.ndim == 0:
        raise ValueError(f"{path} must include a batch dimension")
    if not torch.is_floating_point(value):
        raise TypeError(f"{path} must be floating-point")
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
    reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(signal_scale, torch.Tensor) or not isinstance(
        noise_scale, torch.Tensor
    ):
        raise TypeError("Gaussian signal and noise scales must be Tensors")
    if signal_scale.shape != noise_scale.shape:
        raise ValueError("Gaussian signal and noise scales must share a shape")
    if reference is not None:
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


def _validate_variance_mode(value: object) -> VarianceMode:
    if not isinstance(value, str) or value not in ("fixed", "learned_range"):
        raise ValueError("Gaussian variance_mode must be fixed or learned_range")
    return cast(VarianceMode, value)


__all__ = [
    "GaussianPrediction",
    "LearnedVarianceGaussianPrediction",
    "PredictionType",
    "VarianceMode",
    "gaussian_signal_to_noise_ratio",
    "gaussian_training_target",
    "normalize_gaussian_prediction",
    "split_gaussian_model_output",
]
