"""Model-free Gaussian denoising process contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import torch

from ..base import Process
from .noise_schedules.base import GaussianScales


@dataclass(frozen=True, slots=True)
class GaussianMarginalCoefficientSnapshot:
    """Broadcast cumulative VP coefficients for one selected reverse pair."""

    source_alpha_bar: torch.Tensor
    target_alpha_bar: torch.Tensor
    transition_alpha: torch.Tensor

    def __post_init__(self) -> None:
        raw_values = cast(
            tuple[object, ...],
            (
                self.source_alpha_bar,
                self.target_alpha_bar,
                self.transition_alpha,
            ),
        )
        if any(not isinstance(value, torch.Tensor) for value in raw_values):
            raise TypeError("Gaussian marginal coefficients must be Tensors")
        values = cast(tuple[torch.Tensor, ...], raw_values)
        if any(not torch.is_floating_point(value) for value in values):
            raise TypeError("Gaussian marginal coefficients must be floating-point")
        if any(value.shape != self.source_alpha_bar.shape for value in values[1:]):
            raise ValueError("Gaussian marginal coefficients must share a shape")
        if any(value.device != self.source_alpha_bar.device for value in values[1:]):
            raise ValueError("Gaussian marginal coefficients must share a device")
        if any(value.dtype != self.source_alpha_bar.dtype for value in values[1:]):
            raise ValueError("Gaussian marginal coefficients must share a dtype")
        if any(not bool(torch.all(torch.isfinite(value))) for value in values):
            raise ValueError("Gaussian marginal coefficients must be finite")
        if bool(torch.any(self.source_alpha_bar <= 0)) or bool(
            torch.any(self.target_alpha_bar <= 0)
        ):
            raise ValueError("Gaussian cumulative alpha values must be positive")
        if bool(torch.any(self.source_alpha_bar > self.target_alpha_bar)):
            raise ValueError(
                "Gaussian source alpha_bar must not exceed target alpha_bar"
            )
        if bool(torch.any(self.transition_alpha <= 0)) or bool(
            torch.any(self.transition_alpha > 1)
        ):
            raise ValueError("Gaussian transition alpha must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class GaussianLogVarianceBounds:
    """Broadcast lower and upper log-variance bounds for learned range."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        lower_value = cast(object, self.lower)
        upper_value = cast(object, self.upper)
        if not isinstance(lower_value, torch.Tensor) or not isinstance(
            upper_value, torch.Tensor
        ):
            raise TypeError("Gaussian log-variance bounds must be Tensors")
        lower = lower_value
        upper = upper_value
        if not torch.is_floating_point(lower) or not torch.is_floating_point(upper):
            raise TypeError("Gaussian log-variance bounds must be floating-point")
        if lower.shape != upper.shape:
            raise ValueError("Gaussian log-variance bounds must share a shape")
        if lower.device != upper.device:
            raise ValueError("Gaussian log-variance bounds must share a device")
        if lower.dtype != upper.dtype:
            raise ValueError("Gaussian log-variance bounds must share a dtype")
        if not bool(torch.all(torch.isfinite(lower))) or not bool(
            torch.all(torch.isfinite(upper))
        ):
            raise ValueError("Gaussian log-variance bounds must be finite")
        if bool(torch.any(lower > upper)):
            raise ValueError(
                "Gaussian lower log variance must not exceed its upper bound"
            )


@runtime_checkable
class SelectedPairGaussianProcess(Protocol):
    """Gaussian process capability exposing arbitrary reverse-time marginals."""

    def marginal_coefficient_snapshot(
        self,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> GaussianMarginalCoefficientSnapshot:
        """Return cumulative coefficients for pairs with ``target < source``."""

        ...


@runtime_checkable
class LearnedRangeGaussianVarianceProcess(Protocol):
    """Gaussian process capability supplying learned-range variance bounds."""

    def reverse_log_variance_bounds(
        self,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        broadcast_shape: torch.Size,
        *,
        clean_target_reference_times: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> GaussianLogVarianceBounds:
        """Return lower posterior and upper transition log variances.

        ``clean_target_reference_times`` supplies the preceding selected pair
        whose posterior variance clips a zero final posterior. This preserves
        learned-range semantics for a respaced ancestral chain.
        """

        ...


class DiscreteGaussianDenoisingProcess(Process, ABC):
    """Discrete Gaussian path used by denoising training and samplers."""

    @property
    @abstractmethod
    def clean_time(self) -> int:
        """Return the public state time of clean data."""

    @property
    @abstractmethod
    def terminal_time(self) -> int:
        """Return the terminal public state time."""

    @abstractmethod
    def sample_terminal_prior(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the terminal prior."""

    @abstractmethod
    def sample_marginal(
        self,
        clean: torch.Tensor,
        state_times: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a noisy marginal and return the realized noise."""

    @abstractmethod
    def marginal_scales(
        self,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> GaussianScales:
        """Return broadcast marginal signal and noise scales."""

    @abstractmethod
    def validate_noisy_state_times(
        self, state_times: torch.Tensor
    ) -> torch.Tensor:
        """Validate source state times accepted by reverse transitions."""

    @abstractmethod
    def posterior_mean(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
        clean_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Return the adjacent reverse posterior mean."""

    @abstractmethod
    def posterior_standard_deviation(
        self,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        """Return the adjacent reverse posterior standard deviation."""


__all__ = [
    "DiscreteGaussianDenoisingProcess",
    "GaussianLogVarianceBounds",
    "GaussianMarginalCoefficientSnapshot",
    "LearnedRangeGaussianVarianceProcess",
    "SelectedPairGaussianProcess",
]
