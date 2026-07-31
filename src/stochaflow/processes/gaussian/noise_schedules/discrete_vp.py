"""Gaussian-family discrete variance-preserving noise paths."""

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


@dataclass(frozen=True, slots=True)
class DiscreteVPScheduleSnapshot:
    """One coherent schedule snapshot used to derive process coefficients."""

    marginal_scales: GaussianScales
    transition_coefficients: DiscreteVPCoefficients
    storage_dtype: torch.dtype

    def __post_init__(self) -> None:
        storage_dtype_value = cast(object, self.storage_dtype)
        if (
            not isinstance(storage_dtype_value, torch.dtype)
            or not storage_dtype_value.is_floating_point
        ):
            raise TypeError("schedule snapshot storage dtype must be floating-point")


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
        if (
            state_times.dtype == torch.bool
            or torch.is_floating_point(state_times)
            or torch.is_complex(state_times)
        ):
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

    def coefficient_snapshot(
        self,
        state_times: torch.Tensor,
        transition_times: torch.Tensor,
    ) -> DiscreteVPScheduleSnapshot:
        """Return coherent marginal and transition coefficient tables."""

        scales = self.marginal_scales(state_times)
        coefficients = self.transition_coefficients(transition_times)
        return DiscreteVPScheduleSnapshot(
            marginal_scales=scales,
            transition_coefficients=coefficients,
            storage_dtype=scales.signal.dtype,
        )


class TabulatedDiscreteVPSchedule(DiscreteVPSchedule):
    """Implement a fixed discrete VP schedule with coefficient tables."""

    reference_beta_t: torch.Tensor
    reference_alpha_t: torch.Tensor
    reference_alpha_bar_t: torch.Tensor
    beta_t: torch.Tensor
    alpha_t: torch.Tensor
    alpha_bar_t: torch.Tensor
    sqrt_alpha_bar_t: torch.Tensor
    sqrt_one_minus_alpha_bar_t: torch.Tensor

    def __init__(
        self,
        betas: torch.Tensor,
        *,
        storage_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        betas = self._validate_betas(betas)
        if storage_dtype is None:
            storage_dtype = betas.dtype
        self._validate_dtype(storage_dtype)
        # Match the reference implementations: derive cumulative schedule
        # coefficients in float64, then expose ordinary runtime buffers.
        precise_betas = betas.to(dtype=torch.float64)
        precise_alphas = 1.0 - precise_betas
        precise_alpha_bars = torch.cumprod(precise_alphas, dim=0)
        self.register_buffer(
            "reference_beta_t",
            precise_betas,
            persistent=False,
        )
        self.register_buffer(
            "reference_alpha_t",
            precise_alphas,
            persistent=False,
        )
        self.register_buffer(
            "reference_alpha_bar_t",
            precise_alpha_bars,
            persistent=False,
        )
        betas = precise_betas.to(dtype=storage_dtype)
        alphas = precise_alphas.to(dtype=storage_dtype)
        alpha_bars = precise_alpha_bars.to(dtype=storage_dtype)
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

    def coefficient_snapshot(
        self,
        state_times: torch.Tensor,
        transition_times: torch.Tensor,
    ) -> DiscreteVPScheduleSnapshot:
        """Return float64 reference coefficients with the runtime storage dtype."""

        state_times = self.validate_state_times(state_times)
        transition_times = self.validate_state_times(transition_times)
        if torch.any(transition_times == 0):
            raise ValueError("transition state times must lie in [1, T]")

        state_indices = (state_times - 1).clamp_min(0)
        reference_alpha_bars = self.reference_alpha_bar_t
        signal = self._gather(reference_alpha_bars.sqrt(), state_indices)
        noise = self._gather(
            (1.0 - reference_alpha_bars).sqrt(),
            state_indices,
        )
        clean_mask = (state_times == 0).to(signal.device)
        scales = GaussianScales(
            torch.where(clean_mask, torch.ones_like(signal), signal),
            torch.where(clean_mask, torch.zeros_like(noise), noise),
        )

        indices = transition_times - 1
        alpha_bar = self._gather(reference_alpha_bars, indices)
        previous = self._gather(
            reference_alpha_bars,
            (indices - 1).clamp_min(0),
        )
        previous = torch.where(
            (transition_times == 1).to(previous.device),
            torch.ones_like(previous),
            previous,
        )
        coefficients = DiscreteVPCoefficients(
            beta=self._gather(self.reference_beta_t, indices),
            alpha=self._gather(self.reference_alpha_t, indices),
            alpha_bar=alpha_bar,
            previous_alpha_bar=previous,
        )
        return DiscreteVPScheduleSnapshot(
            marginal_scales=scales,
            transition_coefficients=coefficients,
            storage_dtype=self.beta_t.dtype,
        )

    @staticmethod
    def _validate_betas(betas: torch.Tensor) -> torch.Tensor:
        betas = torch.as_tensor(betas)
        if betas.ndim != 1 or betas.numel() == 0:
            raise ValueError("betas must be a non-empty 1D tensor")
        if torch.is_complex(betas):
            raise TypeError("betas must be a real-valued tensor")
        if not torch.is_floating_point(betas):
            betas = betas.to(dtype=torch.float32)
        if not torch.all(torch.isfinite(betas)):
            raise ValueError("betas must contain only finite values")
        if torch.any(betas <= 0) or torch.any(betas >= 1):
            raise ValueError("every beta must lie in (0, 1)")
        return betas.detach().clone()

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
    "DiscreteVPScheduleSnapshot",
    "TabulatedDiscreteVPSchedule",
]
