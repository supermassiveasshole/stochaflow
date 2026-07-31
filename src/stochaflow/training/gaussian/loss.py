"""Gaussian training loss values and layer-local tensor interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from stochaflow.families.gaussian import GaussianPrediction, PredictionType
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
)

from .contracts import VarianceMode


@dataclass(frozen=True, slots=True)
class GaussianLossComputation:
    """One normalized prediction and its batch-aligned loss components."""

    loss: torch.Tensor
    prediction: GaussianPrediction
    target: torch.Tensor
    snr: torch.Tensor
    timestep_loss_weight: torch.Tensor
    per_sample_simple_loss: torch.Tensor | None
    per_sample_weighted_simple_loss: torch.Tensor | None
    per_sample_variational_bound: torch.Tensor | None
    per_sample_loss: torch.Tensor | None


def split_gaussian_training_output(
    value: object,
    *,
    state: torch.Tensor,
    variance_mode: VarianceMode,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Interpret the fixed or learned-range heads of a training model."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian model must return a Tensor")
    if value.device != state.device:
        raise ValueError("Gaussian model output must share the noisy state device")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian model output must be floating-point")
    if variance_mode == "fixed":
        if value.shape != state.shape:
            raise ValueError(
                "fixed-variance Gaussian model output must match the state shape"
            )
        return value, None
    if state.ndim < 2:
        raise ValueError(
            "learned_range Gaussian states must include a channel dimension"
        )
    expected = (state.shape[0], state.shape[1] * 2, *state.shape[2:])
    if value.shape != expected:
        raise ValueError(
            "learned_range Gaussian model output must have shape "
            f"{expected}, got {tuple(value.shape)}"
        )
    return cast(tuple[torch.Tensor, torch.Tensor], value.chunk(2, dim=1))


def gaussian_signal_to_noise_ratio(
    process: DiscreteGaussianDenoisingProcess,
    state_times: torch.Tensor,
) -> torch.Tensor:
    """Return cumulative VP signal-to-noise ratios at public state times."""

    process = _validate_process(process)
    state_times = process.validate_noisy_state_times(state_times)
    scales = process.marginal_scales(state_times, state_times.size())
    return scales.signal.square() / scales.noise.square()


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


def validate_gaussian_timestep_weights(
    value: object,
    *,
    snr: torch.Tensor,
) -> torch.Tensor:
    """Validate one concrete Strategy's batch-aligned loss weights."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian timestep loss weights must be a Tensor")
    if value.ndim != 1 or value.shape != snr.shape:
        raise ValueError("Gaussian timestep loss weights must have shape [B]")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian timestep loss weights must be floating-point")
    if value.device != snr.device:
        raise ValueError("Gaussian timestep loss weights must share the SNR device")
    if value.dtype != snr.dtype:
        raise ValueError("Gaussian timestep loss weights must share the SNR dtype")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Gaussian timestep loss weights must be finite")
    if bool(torch.any(value < 0)):
        raise ValueError("Gaussian timestep loss weights must be non-negative")
    return value


def gaussian_loss_diagnostics(
    computation: GaussianLossComputation,
) -> dict[str, torch.Tensor]:
    """Detach standardized Gaussian loss diagnostics."""

    diagnostics = {
        "snr": computation.snr.detach(),
        "timestep_loss_weight": computation.timestep_loss_weight.detach(),
    }
    for name in (
        "per_sample_simple_loss",
        "per_sample_weighted_simple_loss",
        "per_sample_variational_bound",
        "per_sample_loss",
    ):
        value = getattr(computation, name)
        if value is not None:
            diagnostics[name] = value.detach()
    return diagnostics


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
