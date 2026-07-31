"""Gaussian-family contracts for forward marginal schedules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from stochaflow.utils.registry import REGISTRIES


@dataclass(frozen=True, slots=True)
class GaussianScales:
    """Signal and noise coefficients for Gaussian marginal states."""

    signal: torch.Tensor
    noise: torch.Tensor

    def __post_init__(self) -> None:
        signal = cast(object, self.signal)
        noise = cast(object, self.noise)
        if not isinstance(signal, torch.Tensor) or not isinstance(noise, torch.Tensor):
            raise TypeError("Gaussian scales must be Tensors")
        if self.signal.shape != self.noise.shape:
            raise ValueError("Gaussian signal and noise scales must share a shape")


class GaussianNoiseSchedule(nn.Module, ABC):
    r"""Define a Gaussian forward marginal through signal and noise scales."""

    @property
    def clean_time(self) -> int | float:
        """Return the public clean-state time."""

        return 0

    @property
    @abstractmethod
    def terminal_time(self) -> int | float:
        """Return the terminal public state time."""

    @abstractmethod
    def validate_state_times(self, state_times: torch.Tensor) -> torch.Tensor:
        """Validate and normalize public mathematical state times."""

    @abstractmethod
    def marginal_scales(
        self,
        state_times: torch.Tensor,
    ) -> GaussianScales:
        """Return one signal/noise coefficient pair per public state time."""

    def signal_to_noise_ratio(
        self,
        state_times: torch.Tensor,
    ) -> torch.Tensor:
        """Return the marginal signal-to-noise ratio."""

        scales = self.marginal_scales(state_times)
        return scales.signal.square() / scales.noise.square()


REGISTRIES.noise_schedules.require_base(GaussianNoiseSchedule)


__all__ = ["GaussianNoiseSchedule", "GaussianScales"]
