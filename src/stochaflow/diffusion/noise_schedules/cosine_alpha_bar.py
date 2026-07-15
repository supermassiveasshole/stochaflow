"""Cosine-alpha-bar variance-preserving schedules."""

import math

import torch

from stochaflow.utils.registry import REGISTRIES

from .discrete_vp import DiscreteVPSchedule


@REGISTRIES.noise_schedules.register("cosine_alpha_bar")
class CosineAlphaBarSchedule(DiscreteVPSchedule):
    r"""Build a discrete VP path from a cosine cumulative-signal curve.

    This is an alpha-bar-native parameterization. It first evaluates a
    normalized cosine curve at mathematical states ``0..T`` and then derives
    transition betas from adjacent ratios. ``max_beta`` bounds the final
    discretized transitions without changing the public time convention.

    Args:
        num_timesteps: Number ``T`` of forward transitions.
        s: Non-negative offset used by the cosine alpha-bar curve.
        max_beta: Upper bound applied to every derived transition beta.
        dtype: Floating-point dtype used for the stored coefficient tables.
    """

    def __init__(
        self,
        num_timesteps: int,
        s: float = 0.008,
        max_beta: float = 0.999,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._validate_num_timesteps(num_timesteps)
        self._validate_dtype(dtype)
        self._validate_parameters(s, max_beta)

        self.s = float(s)
        self.max_beta = float(max_beta)
        alpha_bar = self._build_alpha_bar(num_timesteps, self.s)
        betas = self._discretize_alpha_bar(alpha_bar, self.max_beta)
        super().__init__(betas.to(dtype=dtype))

    @staticmethod
    def _validate_parameters(s: float, max_beta: float) -> None:
        """Validate the cosine offset and beta cap."""

        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise TypeError("s must be numeric")
        if isinstance(max_beta, bool) or not isinstance(max_beta, (int, float)):
            raise TypeError("max_beta must be numeric")
        if not math.isfinite(s) or s < 0:
            raise ValueError("s must be finite and non-negative")
        if not 0 < max_beta < 1:
            raise ValueError("max_beta must be in (0, 1)")

    @staticmethod
    def _build_alpha_bar(num_timesteps: int, s: float) -> torch.Tensor:
        """Evaluate the normalized cosine ``ᾱ`` curve."""

        state_times = torch.linspace(
            0,
            num_timesteps,
            num_timesteps + 1,
            dtype=torch.float64,
        )
        alpha_bar = torch.cos(
            ((state_times / num_timesteps) + s) / (1 + s) * math.pi * 0.5
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        alpha_bar[0] = 1.0
        return alpha_bar

    @staticmethod
    def _discretize_alpha_bar(
        alpha_bar: torch.Tensor,
        max_beta: float,
    ) -> torch.Tensor:
        """Convert state-indexed alpha-bar values into transition betas."""

        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=max_beta)
