"""Generic schedule base classes and utilities for stochastic flows."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from stochaflow.utils.registry import register_scheduler


def _extract_into_tensor(
    values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
) -> torch.Tensor:
    """Gather per-timestep values and reshape them for batch-first broadcasting.

    Args:
        values: A 1D tensor of shape ``(T,)`` containing schedule coefficients.
        timesteps: A 1D integer tensor of shape ``(batch,)`` with timestep indices.
        broadcast_shape: The shape of the tensor the gathered values will be applied to.
            The first dimension is assumed to be batch-aligned with ``timesteps``;
            all remaining dimensions are treated as broadcast dimensions.

    Returns:
        A tensor of shape ``(batch, 1, ..., 1)`` with the same rank as
        ``broadcast_shape``, suitable for elementwise operations against a batch-first
        tensor of shape ``broadcast_shape``.
    """

    gathered = values.gather(0, timesteps.to(values.device))
    reshape = (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1)
    return gathered.reshape(reshape)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a linearly spaced beta schedule."""

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError("beta_start and beta_end must be positive")
    if beta_start >= 1 or beta_end >= 1:
        raise ValueError("beta_start and beta_end must be smaller than 1")
    if beta_start >= beta_end:
        raise ValueError("beta_start must be smaller than beta_end")
    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)


def cosine_beta_schedule(
    timesteps: int,
    *,
    s: float = 0.008,
    max_beta: float = 0.999,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the cosine schedule introduced by Nichol and Dhariwal."""

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if not 0.0 < max_beta < 1.0:
        raise ValueError("max_beta must be in (0, 1)")

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=max_beta).to(dtype=dtype)


class DiffusionScheduler(nn.Module, ABC):
    """Abstract base class for discrete-time coefficient schedules."""

    def __init__(self, num_timesteps: int) -> None:
        super().__init__()
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        self.num_timesteps = int(num_timesteps)

    def validate_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Validate and normalize a batch of timestep indices."""

        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a 1D tensor")
        timesteps = timesteps.to(dtype=torch.long)
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.num_timesteps):
            raise ValueError("timesteps out of range")
        return timesteps

    @property
    @abstractmethod
    def coefficient_names(self) -> tuple[str, ...]:
        """Names of timestep-indexed coefficient tables exposed by this scheduler."""

    def validate_broadcast_shape(
        self, timesteps: torch.Tensor, broadcast_shape: torch.Size
    ) -> torch.Size:
        """Validate the shape a coefficient tensor will be broadcast against."""

        if len(broadcast_shape) == 0:
            raise ValueError("broadcast_shape must have at least one dimension")
        if broadcast_shape[0] != timesteps.shape[0]:
            raise ValueError("broadcast_shape batch dimension must match timesteps")
        return broadcast_shape

    def register_coefficient(self, name: str, values: torch.Tensor) -> None:
        """Register a named 1D coefficient table as a persistent buffer."""

        if name not in self.coefficient_names:
            raise ValueError(f"Unsupported coefficient name: {name}")
        values = torch.as_tensor(values)
        if values.ndim != 1:
            raise ValueError("coefficient values must be a 1D tensor")
        if values.shape[0] != self.num_timesteps:
            raise ValueError("coefficient table length must equal num_timesteps")
        self.register_buffer(name, values)

    def has_coefficient(self, name: str) -> bool:
        """Return whether a named coefficient is supported and registered."""

        return name in self.coefficient_names and hasattr(self, name)

    def extract(
        self, values: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size
    ) -> torch.Tensor:
        """Broadcast 1D schedule values to match the rank of a target tensor."""

        timesteps = self.validate_timesteps(timesteps)
        broadcast_shape = self.validate_broadcast_shape(timesteps, broadcast_shape)
        return _extract_into_tensor(values, timesteps, broadcast_shape)

    def coefficients_at(
        self, name: str, timesteps: torch.Tensor, broadcast_shape: torch.Size
    ) -> torch.Tensor:
        """Return named schedule coefficients for a batch of timesteps."""

        if name not in self.coefficient_names:
            raise ValueError(f"Unsupported coefficient name: {name}")
        if not hasattr(self, name):
            raise RuntimeError(f"Coefficient {name!r} has not been registered")
        values = getattr(self, name)
        return self.extract(values, timesteps, broadcast_shape)


@register_scheduler("linear_ddpm")
class LinearDDPMScheduler(DiffusionScheduler):
    """DDPM scheduler with linear beta interpolation."""

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return (
            "beta_t",
            "alpha_t",
            "alpha_bar_t",
            "sqrt_alpha_bar_t",
            "sqrt_one_minus_alpha_bar_t",
            "sqrt_recip_alpha_t",
            "beta_over_sqrt_one_minus_alpha_bar_t",
            "sqrt_posterior_variance_t",
        )

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(num_timesteps)

        beta_t = linear_beta_schedule(num_timesteps, beta_start, beta_end, dtype=dtype)
        alpha_t = 1.0 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        alpha_bar_t_minus_one = torch.cat(
            (torch.tensor([1.0], dtype=dtype), alpha_bar_t[:-1]), dim=0
        )

        self.register_coefficient("beta_t", beta_t)
        self.register_coefficient("alpha_t", alpha_t)
        self.register_coefficient("alpha_bar_t", alpha_bar_t)
        self.register_coefficient("sqrt_alpha_bar_t", torch.sqrt(alpha_bar_t))
        self.register_coefficient(
            "sqrt_one_minus_alpha_bar_t",
            torch.sqrt(1.0 - alpha_bar_t),
        )
        self.register_coefficient("sqrt_recip_alpha_t", 1 / torch.sqrt(alpha_t))
        self.register_coefficient(
            "beta_over_sqrt_one_minus_alpha_bar_t", beta_t / torch.sqrt(1 - alpha_bar_t)
        )
        self.register_coefficient(
            "sqrt_posterior_variance_t",
            torch.sqrt(beta_t * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)),
        )
