"""Discrete variance-preserving noise paths."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import cast

import torch

from .base import GaussianNoiseSchedule, GaussianScales


@dataclass(frozen=True, slots=True)
class DiscreteVPCoefficients:
    """Adjacent-transition coefficients for requested noisy state times."""

    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    previous_alpha_bar: torch.Tensor

    def __post_init__(self) -> None:
        values = (self.beta, self.alpha, self.alpha_bar, self.previous_alpha_bar)
        raw_values = cast(tuple[object, ...], values)
        if any(not isinstance(value, torch.Tensor) for value in raw_values):
            raise TypeError("discrete VP coefficients must be Tensors")
        if any(value.shape != self.beta.shape for value in values[1:]):
            raise ValueError("discrete VP coefficients must share a shape")


class DiscreteVPSchedule(GaussianNoiseSchedule):
    """Mathematical capability for an immutable discrete VP schedule."""

    @property
    @abstractmethod
    def num_timesteps(self) -> int:
        """Return the number of discrete forward transitions."""

    @property
    def terminal_time(self) -> int:
        """Return terminal mathematical state time ``T``."""

        return self.num_timesteps

    def validate_state_times(self, state_times: torch.Tensor) -> torch.Tensor:
        """Validate integer public state times in ``[0, T]``."""

        if state_times.ndim != 1:
            raise ValueError("state_times must be a 1D tensor")
        if state_times.dtype == torch.bool or torch.is_floating_point(state_times):
            raise TypeError("state_times must contain integer mathematical states")
        normalized = state_times.to(dtype=torch.long)
        if torch.any(normalized < 0) or torch.any(normalized > self.num_timesteps):
            raise ValueError("state_times must lie in [0, T]")
        return normalized

    @abstractmethod
    def transition_coefficients(
        self,
        state_times: torch.Tensor,
    ) -> DiscreteVPCoefficients:
        """Return adjacent-transition coefficients for states in ``[1, T]``."""


class TabulatedDiscreteVPSchedule(DiscreteVPSchedule):
    """Implement a fixed discrete VP schedule with coefficient tables."""

    beta_t: torch.Tensor
    alpha_t: torch.Tensor
    alpha_bar_t: torch.Tensor
    sqrt_alpha_bar_t: torch.Tensor
    sqrt_one_minus_alpha_bar_t: torch.Tensor

    def __init__(self, betas: torch.Tensor) -> None:
        super().__init__()
        betas = self._validate_betas(betas)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._num_timesteps = int(betas.shape[0])
        self.register_buffer("beta_t", betas)
        self.register_buffer("alpha_t", alphas)
        self.register_buffer("alpha_bar_t", alpha_bars)
        self.register_buffer("sqrt_alpha_bar_t", alpha_bars.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alpha_bar_t", (1.0 - alpha_bars).sqrt()
        )

    @property
    def num_timesteps(self) -> int:
        """Return the number of tabulated transitions."""

        return self._num_timesteps

    def marginal_scales(
        self,
        state_times: torch.Tensor,
    ) -> GaussianScales:
        """Return VP marginal scales for public states ``0..T``."""

        state_times = self.validate_state_times(state_times)
        indices = (state_times - 1).clamp_min(0)
        signal = self._gather(self.sqrt_alpha_bar_t, indices)
        noise = self._gather(self.sqrt_one_minus_alpha_bar_t, indices)
        mask = (state_times == 0).to(signal.device)
        return GaussianScales(
            torch.where(mask, torch.ones_like(signal), signal),
            torch.where(mask, torch.zeros_like(noise), noise),
        )

    def transition_coefficients(
        self,
        state_times: torch.Tensor,
    ) -> DiscreteVPCoefficients:
        """Gather fixed adjacent-transition coefficients for states ``1..T``."""

        state_times = self.validate_state_times(state_times)
        if torch.any(state_times == 0):
            raise ValueError("transition state times must lie in [1, T]")
        indices = state_times - 1
        alpha_bar = self._gather(self.alpha_bar_t, indices)
        previous = self._gather(self.alpha_bar_t, (indices - 1).clamp_min(0))
        previous = torch.where(
            (state_times == 1).to(previous.device),
            torch.ones_like(previous),
            previous,
        )
        return DiscreteVPCoefficients(
            beta=self._gather(self.beta_t, indices),
            alpha=self._gather(self.alpha_t, indices),
            alpha_bar=alpha_bar,
            previous_alpha_bar=previous,
        )

    @staticmethod
    def _validate_betas(betas: torch.Tensor) -> torch.Tensor:
        betas = torch.as_tensor(betas)
        if betas.ndim != 1 or betas.numel() == 0:
            raise ValueError("betas must be a non-empty 1D tensor")
        if not torch.is_floating_point(betas):
            betas = betas.to(dtype=torch.float32)
        if not torch.all(torch.isfinite(betas)):
            raise ValueError("betas must contain only finite values")
        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("every beta must lie in (0, 1)")
        return betas

    @staticmethod
    def _validate_num_timesteps(num_timesteps: object) -> None:
        if isinstance(num_timesteps, bool) or not isinstance(num_timesteps, int):
            raise TypeError("num_timesteps must be an integer")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

    @staticmethod
    def _validate_dtype(dtype: object) -> None:
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point torch dtype")

    @staticmethod
    def _gather(
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return values.gather(0, indices.to(values.device))


__all__ = [
    "DiscreteVPCoefficients",
    "DiscreteVPSchedule",
    "TabulatedDiscreteVPSchedule",
]
