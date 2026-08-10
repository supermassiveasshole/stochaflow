"""Gaussian-family linear-beta variance-preserving schedules."""

import torch

from .discrete_vp import TabulatedDiscreteVPSchedule


class LinearBetaSchedule(TabulatedDiscreteVPSchedule):
    """Build a discrete VP path from linearly interpolated betas."""

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._validate_num_timesteps(num_timesteps)
        self._validate_dtype(dtype)
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("expected 0 < beta_start < beta_end < 1")
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        super().__init__(
            torch.linspace(
                beta_start,
                beta_end,
                num_timesteps,
                dtype=torch.float64,
            ),
            storage_dtype=dtype,
        )


__all__ = ["LinearBetaSchedule"]
