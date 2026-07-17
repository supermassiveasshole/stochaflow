"""Denoising Diffusion Probabilistic Model process and sampler."""

from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion.gaussian import (
    DiffusionForwardOutput,
    GaussianDiffusion,
)
from stochaflow.diffusion.noise_schedules import DiscreteVPSchedule
from stochaflow.sampling.sampler import SamplingTrace, TrajectoryFrame
from stochaflow.utils.registry import REGISTRIES

# Compatibility alias for callers using the original algorithm-specific name.
DDPMForwardOutput = DiffusionForwardOutput


@REGISTRIES.diffusions.register("ddpm")
class DDPM(GaussianDiffusion):
    r"""DDPM reverse process over a shared discrete Gaussian noise path.

    The injected :class:`DiscreteVPSchedule` owns only the forward-path
    quantities shared by compatible diffusion samplers. This class derives and
    stores the DDPM-specific posterior variance and posterior-mean coefficients
    required for ``q(x_{t-1} | x_t, x_0)``.

    Every public process method uses mathematical state times ``0..T``. The
    complete reverse path is ``x_T -> ... -> x_0`` and performs exactly ``T``
    transitions. A source state ``t >= 1`` maps internally to noise-schedule
    and model-conditioning index ``t-1``; no public negative sentinel exists.
    """

    sqrt_posterior_variance_t: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    def __init__(
        self,
        noise_schedule: DiscreteVPSchedule,
        model: nn.Module,
        *,
        clip_denoised: bool = True,
    ) -> None:
        """Initialize DDPM and derive its algorithm-specific coefficients.

        Args:
            noise_schedule: Shared discrete forward noise path containing beta,
                alpha, and cumulative-alpha tables.
            model: Denoiser predicting epsilon at zero-based model timesteps.
            clip_denoised: Whether to clip reconstructed clean samples to
                ``[-1, 1]`` before computing the posterior mean.
        """

        super().__init__(
            noise_schedule=noise_schedule,
            model=model,
            clip_denoised=clip_denoised,
        )
        self._register_posterior_coefficients()

    def _register_posterior_coefficients(self) -> None:
        """Materialize the DDPM posterior tables from the shared noise path."""

        beta_t = self.noise_schedule.beta_t
        alpha_t = self.noise_schedule.alpha_t
        alpha_bar_t = self.noise_schedule.alpha_bar_t
        alpha_bar_t_minus_one = torch.cat(
            (torch.ones_like(alpha_bar_t[:1]), alpha_bar_t[:-1]),
        )
        posterior_variance_t = (
            beta_t
            * (1.0 - alpha_bar_t_minus_one)
            / (1.0 - alpha_bar_t)
        )
        self.register_buffer(
            "sqrt_posterior_variance_t",
            torch.sqrt(posterior_variance_t),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            beta_t
            * torch.sqrt(alpha_bar_t_minus_one)
            / (1.0 - alpha_bar_t),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            torch.sqrt(alpha_t)
            * (1.0 - alpha_bar_t_minus_one)
            / (1.0 - alpha_bar_t),
        )

    def reverse(
        self,
        x_from: torch.Tensor,
        timestep_from: int,
        timestep_to: int = 0,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Run adjacent DDPM transitions between public mathematical states.

        Args:
            x_from: Batch at mathematical state time ``timestep_from``.
            timestep_from: Current state time in ``[0, T]``.
            timestep_to: Target state time in ``[0, timestep_from]``. The
                default zero traverses to the clean endpoint.
            clip_denoised: Optional clipping override applied at every step.

        Returns:
            The batch at mathematical state time ``timestep_to``.

        Raises:
            ValueError: If the state times are outside the process horizon or
                ordered in the forward direction.
        """

        if not 0 <= timestep_to <= timestep_from <= self.num_timesteps:
            raise ValueError(
                "expected 0 <= timestep_to <= timestep_from <= num_timesteps"
            )

        x_t = x_from
        for state_timestep in range(timestep_from, timestep_to, -1):
            timesteps = torch.full(
                (x_t.shape[0],),
                state_timestep,
                dtype=torch.long,
                device=x_t.device,
            )
            x_t = self.reverse_step(
                x_t,
                timesteps,
                clip_denoised=clip_denoised,
            )
        return x_t

    def sample(
        self,
        sample_shape: torch.Size,
        device: torch.device | None = None,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Generate samples with the complete ``x_T -> ... -> x_0`` path."""

        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        initial_noise = torch.randn(sample_shape, device=device)
        return self.sample_from_noise(
            initial_noise,
            clip_denoised=clip_denoised,
        )

    def sample_from_noise(
        self,
        initial_noise: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Generate samples from a caller-provided terminal noise batch."""

        return self.reverse(
            initial_noise,
            self.num_timesteps,
            0,
            clip_denoised=clip_denoised,
        )

    def sample_trajectory(
        self,
        sample_shape: torch.Size,
        device: torch.device | None = None,
        *,
        state_interval: int = 100,
    ) -> SamplingTrace:
        """Generate a debug trace at fixed mathematical-state intervals."""

        if state_interval <= 0:
            raise ValueError("DDPM trajectory state_interval must be positive")
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        initial_noise = torch.randn(sample_shape, device=device)
        return self.sample_trajectory_from_noise(
            initial_noise,
            state_interval=state_interval,
        )

    def sample_trajectory_from_noise(
        self,
        initial_noise: torch.Tensor,
        *,
        state_interval: int = 100,
    ) -> SamplingTrace:
        """Trace sampling from a caller-provided terminal noise batch."""

        if state_interval <= 0:
            raise ValueError("DDPM trajectory state_interval must be positive")
        state_time = self.num_timesteps
        current = initial_noise
        frames = [TrajectoryFrame(state_time, current.detach().cpu())]
        while state_time > 0:
            target_time = max(0, state_time - state_interval)
            current = self.reverse(current, state_time, target_time)
            state_time = target_time
            frames.append(TrajectoryFrame(state_time, current.detach().cpu()))
        return SamplingTrace(samples=current, frames=frames)

    def reverse_step(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Run one adjacent transition ``x_t -> x_{t-1}``.

        ``timesteps`` contains batch-aligned public source states in ``[1, T]``.
        State time one performs the final transition to the clean state.
        """

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        predicted_noise = self._predict_noise(xt, timesteps)
        x0 = self._estimate_x0_from_epsilon(
            xt,
            timesteps,
            predicted_noise=predicted_noise,
            clip_denoised=self._resolve_clip_denoised(clip_denoised),
        )
        posterior_mean = (
            self._posterior_coefficients_at(
                self.posterior_mean_coef1,
                timesteps,
                xt.size(),
            )
            * x0
            + self._posterior_coefficients_at(
                self.posterior_mean_coef2,
                timesteps,
                xt.size(),
            )
            * xt
        )
        return posterior_mean + self._nonzero_timestep_mask(xt, timesteps) * (
            self._posterior_noise(xt, timesteps)
        )

    def _posterior_coefficients_at(
        self,
        values: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        """Gather one DDPM-owned posterior table at public source states."""

        schedule_timesteps = self._state_to_schedule_timesteps(timesteps)
        if len(broadcast_shape) == 0:
            raise ValueError("broadcast_shape must have at least one dimension")
        if broadcast_shape[0] != schedule_timesteps.shape[0]:
            raise ValueError(
                "broadcast_shape batch dimension must match timesteps"
            )

        gathered = values.gather(0, schedule_timesteps.to(values.device))
        return gathered.reshape(
            (schedule_timesteps.shape[0],)
            + (1,) * (len(broadcast_shape) - 1)
        )

    def _posterior_noise(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the DDPM posterior standard-deviation term."""

        return self._posterior_coefficients_at(
            self.sqrt_posterior_variance_t,
            timesteps,
            xt.size(),
        ) * self._sample_noise(xt)

    def _nonzero_timestep_mask(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Disable stochastic noise for the final transition ``x_1 -> x_0``."""

        return (
            (timesteps > 1)
            .to(dtype=xt.dtype)
            .reshape((xt.shape[0],) + (1,) * (xt.ndim - 1))
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Migrate legacy scheduler-owned DDPM posterior buffers."""

        for name in (
            "sqrt_posterior_variance_t",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ):
            legacy_key = f"{prefix}scheduler.{name}"
            current_key = f"{prefix}{name}"
            if legacy_key in state_dict and current_key not in state_dict:
                state_dict[current_key] = state_dict[legacy_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
