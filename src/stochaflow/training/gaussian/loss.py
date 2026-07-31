"""Gaussian training loss values and layer-local tensor interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from stochaflow.families.gaussian import GaussianPrediction, PredictionType
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
)


@dataclass(frozen=True, slots=True)
class GaussianLossComputation:
    """One normalized prediction and its final training loss."""

    loss: torch.Tensor
    prediction: GaussianPrediction
    target: torch.Tensor
    per_sample_loss: torch.Tensor | None


def gaussian_training_target(
    process: DiscreteGaussianDenoisingProcess,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    prediction_type: PredictionType,
) -> torch.Tensor:
    """Build a target from one Process marginal and prediction convention."""

    process = _validate_process(process)
    clean_value = cast(object, clean)
    noise_value = cast(object, noise)
    if not isinstance(clean_value, torch.Tensor) or not isinstance(
        noise_value, torch.Tensor
    ):
        raise TypeError("Gaussian training clean state and noise must be Tensors")
    clean = clean_value
    noise = noise_value
    if clean.ndim == 0:
        raise ValueError("Gaussian training clean state must have a batch dimension")
    if not torch.is_floating_point(clean) or not torch.is_floating_point(noise):
        raise TypeError(
            "Gaussian training clean state and noise must be floating-point"
        )
    if noise.shape != clean.shape:
        raise ValueError("Gaussian training noise must match the clean state shape")
    if noise.device != clean.device:
        raise ValueError("Gaussian training noise must share the clean state device")
    if noise.dtype != clean.dtype:
        raise ValueError("Gaussian training noise must share the clean state dtype")
    if prediction_type not in ("epsilon", "x0", "v", "score"):
        raise ValueError(
            "Gaussian prediction_type must be epsilon, x0, v, or score"
        )
    state_times = process.validate_noisy_state_times(state_times)
    if state_times.shape[0] != clean.shape[0]:
        raise ValueError("Gaussian training state times must match the batch")
    if state_times.device != clean.device:
        raise ValueError(
            "Gaussian training state times must share the clean state device"
        )
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "x0":
        return clean
    scales = process.marginal_scales(state_times, clean.size())
    if prediction_type == "v":
        return scales.signal * noise - scales.noise * clean
    return -noise / scales.noise


def gaussian_loss_diagnostics(
    computation: GaussianLossComputation,
) -> dict[str, torch.Tensor]:
    """Detach the per-sample loss consumed by Gaussian diagnostics."""

    if computation.per_sample_loss is None:
        return {}
    return {"per_sample_loss": computation.per_sample_loss.detach()}


def validate_scalar_objective_loss(
    value: object,
    *,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Validate the scalar returned by a standard Gaussian Objective."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("training objective must return a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError("training objective must return a floating-point Tensor")
    if value.ndim != 0:
        raise ValueError("training objective must return a scalar Tensor")
    if value.device != prediction.device:
        raise ValueError("training objective loss must be on the prediction device")
    return value


def _validate_process(value: object) -> DiscreteGaussianDenoisingProcess:
    if not isinstance(value, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "Gaussian training requires DiscreteGaussianDenoisingProcess"
        )
    return cast(DiscreteGaussianDenoisingProcess, value)


__all__ = [
    "gaussian_training_target",
]
