"""Cosine-alpha-bar variance-preserving schedules."""

import math

import torch

from stochaflow.utils.registry import REGISTRIES

from .discrete_vp import TabulatedDiscreteVPSchedule


@REGISTRIES.noise_schedules.register("cosine_alpha_bar")
class CosineAlphaBarSchedule(TabulatedDiscreteVPSchedule):
    """Build a discrete VP path from a cosine cumulative-signal curve."""

    def __init__(
        self,
        num_timesteps: int,
        s: float = 0.008,
        max_beta: float = 0.999,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._validate_num_timesteps(num_timesteps)
        self._validate_dtype(dtype)
        if not math.isfinite(s) or s < 0:
            raise ValueError("s must be finite and non-negative")
        if not 0 < max_beta < 1:
            raise ValueError("max_beta must be in (0, 1)")
        self.s = float(s)
        self.max_beta = float(max_beta)
        states = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(
            ((states / num_timesteps) + s) / (1 + s) * math.pi * 0.5
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=max_beta)
        super().__init__(betas, storage_dtype=dtype)


__all__ = ["CosineAlphaBarSchedule"]
