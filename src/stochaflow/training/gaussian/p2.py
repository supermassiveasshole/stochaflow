"""Private P2 mathematics shared by the concrete Gaussian strategies."""

from __future__ import annotations

import math

import torch


def validate_p2_parameters(k: object, gamma: object) -> tuple[float, float]:
    """Return finite P2 parameters satisfying the paper's domain."""

    validated_k = _finite_number(k, path="P2 k")
    validated_gamma = _finite_number(gamma, path="P2 gamma")
    if validated_k <= 0.0:
        raise ValueError("P2 k must be greater than zero")
    if validated_gamma < 0.0:
        raise ValueError("P2 gamma must be non-negative")
    return validated_k, validated_gamma


def p2_timestep_loss_weights(
    signal_to_noise_ratio: torch.Tensor,
    *,
    k: float,
    gamma: float,
) -> torch.Tensor:
    """Return exact unnormalized ``(k + SNR) ** (-gamma)`` weights."""

    return (k + signal_to_noise_ratio).pow(-gamma)


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


__all__: list[str] = []
