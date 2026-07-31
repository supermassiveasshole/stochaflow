"""Process-free Gaussian model-output layout mathematics."""

from __future__ import annotations

from typing import Literal, cast

import torch

VarianceMode = Literal["fixed", "learned_range"]


def split_gaussian_model_output(
    value: object,
    *,
    state: torch.Tensor,
    variance_mode: VarianceMode,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a fixed ``C`` or learned-range ``2C`` model output."""

    if state.ndim == 0:
        raise ValueError("Gaussian prediction state must include a batch dimension")
    if not torch.is_floating_point(state):
        raise TypeError("Gaussian prediction state must be floating-point")
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian model callable must return a Tensor")
    if value.device != state.device:
        raise ValueError("Gaussian model output must share the state device")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian model output must be floating-point")
    if variance_mode not in ("fixed", "learned_range"):
        raise ValueError("Gaussian variance_mode must be fixed or learned_range")
    variance_mode = cast(VarianceMode, variance_mode)
    if variance_mode == "fixed":
        if value.shape != state.shape:
            raise ValueError("Gaussian model output must match the state shape")
        return value, None
    if state.ndim < 2:
        raise ValueError(
            "learned_range Gaussian state must include a channel dimension"
        )
    expected_shape = (state.shape[0], state.shape[1] * 2, *state.shape[2:])
    if value.shape != expected_shape:
        raise ValueError(
            "learned_range Gaussian model output must have shape "
            f"{expected_shape}, got {tuple(value.shape)}"
        )
    mean_output, variance_values = value.chunk(2, dim=1)
    return mean_output, variance_values


def interpolate_gaussian_log_variance(
    variance_values: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Interpolate learned-range log-variance bounds from model values."""

    if not torch.is_floating_point(variance_values):
        raise TypeError("Gaussian variance values must be floating-point")
    if not torch.is_floating_point(lower) or not torch.is_floating_point(upper):
        raise TypeError("Gaussian log-variance bounds must be floating-point")
    lower = lower.to(
        device=variance_values.device,
        dtype=variance_values.dtype,
    )
    upper = upper.to(
        device=variance_values.device,
        dtype=variance_values.dtype,
    )
    try:
        lower_shape = torch.broadcast_shapes(variance_values.shape, lower.shape)
        upper_shape = torch.broadcast_shapes(variance_values.shape, upper.shape)
    except RuntimeError as exc:
        raise ValueError(
            "Gaussian log-variance bounds must broadcast to variance values"
        ) from exc
    if (
        tuple(lower_shape) != tuple(variance_values.shape)
        or tuple(upper_shape) != tuple(variance_values.shape)
    ):
        raise ValueError(
            "Gaussian log-variance bounds must broadcast to variance values"
        )
    fraction = (variance_values + 1.0) / 2.0
    return fraction * upper + (1.0 - fraction) * lower


__all__ = [
    "VarianceMode",
    "interpolate_gaussian_log_variance",
    "split_gaussian_model_output",
]
