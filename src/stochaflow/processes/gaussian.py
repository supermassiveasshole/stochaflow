"""Model-free Gaussian denoising process contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .base import Process
from .noise_schedules import GaussianScales


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


__all__ = ["DiscreteGaussianDenoisingProcess"]
