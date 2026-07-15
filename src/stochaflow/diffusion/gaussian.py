"""Shared training contract for discrete Gaussian diffusion processes."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion.noise_schedules import DiscreteVPSchedule


@dataclass(slots=True)
class DiffusionForwardOutput:
    """Structured tensors produced by an epsilon-prediction training pass.

    ``timesteps`` contains public mathematical state times in ``[1, T]``. The
    denoiser receives the corresponding zero-based conditioning indices
    ``timesteps - 1``.
    """

    timesteps: torch.Tensor
    xt: torch.Tensor
    noise: torch.Tensor
    predicted_noise: torch.Tensor


class GaussianDiffusion(nn.Module):
    r"""Common forward process and epsilon-prediction model contract.

    This base class owns only behavior shared by samplers that use the same
    discrete variance-preserving forward path. It does not define a reverse
    equation or sampling loop. DDPM and DDIM therefore share training logic
    without either sampler inheriting the other's reverse-process parameters.

    Public methods use mathematical state times ``0..T``. State zero is clean
    and state ``T`` is terminal noise. The underlying
    :class:`DiscreteVPSchedule` stores coefficient indices ``0..T-1``, so a
    noisy source state ``t >= 1`` maps to coefficient and model index ``t-1``.
    """

    def __init__(
        self,
        noise_schedule: DiscreteVPSchedule,
        model: nn.Module,
        *,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(noise_schedule, DiscreteVPSchedule):
            raise TypeError("GaussianDiffusion requires a DiscreteVPSchedule")
        self.noise_schedule = noise_schedule
        self.model = model
        self.clip_denoised = clip_denoised

    @property
    def num_timesteps(self) -> int:
        """Return the number of forward transitions in the noise path."""

        return self.noise_schedule.num_timesteps

    def forward(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> DiffusionForwardOutput:
        """Run one epsilon-prediction training forward pass.

        Args:
            x0: Batch of clean samples at mathematical state time zero.
            timesteps: Optional batch-aligned mathematical noisy-state times in
                ``[1, T]``. When omitted, they are sampled uniformly.

        Returns:
            The selected public state times, noisy samples, target noise, and
            model-predicted noise.
        """

        if timesteps is None:
            timesteps = self._sample_timesteps(x0.size(0), x0.device)
        xt, noise = self.add_noise(x0, timesteps)
        predicted_noise = self._predict_noise(xt, timesteps)
        return DiffusionForwardOutput(
            timesteps=timesteps,
            xt=xt,
            noise=noise,
            predicted_noise=predicted_noise,
        )

    def add_noise(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the forward marginal at public state times ``0..T``.

        State time zero is represented analytically with ``ᾱ_0 = 1`` and
        returns ``x0`` exactly. No clean-state entry
        is inserted into the schedule's internal coefficient tables.
        """

        if noise is None:
            noise = self._sample_noise(x0)
        timesteps = self._validate_state_timesteps(timesteps, allow_clean=True)
        signal_scale, noise_scale = self.noise_schedule.marginal_scales(
            timesteps,
            x0.size(),
        )
        return signal_scale * x0 + noise_scale * noise, noise

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample public noisy-state times uniformly from ``[1, T]``."""

        return torch.randint(
            1,
            self.num_timesteps + 1,
            torch.Size((batch_size,)),
            device=device,
        )

    def _sample_noise(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample standard Gaussian noise shaped like ``reference``."""

        return torch.randn_like(reference)

    def _predict_noise(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise at public source states using model indices ``t-1``."""

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        return self.model(xt, timesteps - 1)

    def _estimate_x0_from_epsilon(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        predicted_noise: torch.Tensor,
        clip_denoised: bool,
    ) -> torch.Tensor:
        """Estimate the clean sample from an epsilon-parameterized prediction."""

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        signal_scale, noise_scale = self.noise_schedule.marginal_scales(
            timesteps,
            xt.size(),
        )
        x0 = (xt - noise_scale * predicted_noise) / signal_scale
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def _validate_state_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        allow_clean: bool,
    ) -> torch.Tensor:
        """Validate public mathematical state times without changing meaning."""

        timesteps = self.noise_schedule.validate_state_times(timesteps)
        if not allow_clean and torch.any(timesteps == 0):
            raise ValueError("source timesteps must be mathematical states in [1, T]")
        return timesteps

    def _state_to_schedule_timesteps(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Map public source state times ``1..T`` to table indices ``0..T-1``."""

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        return timesteps - 1

    def _resolve_clip_denoised(self, override: bool | None) -> bool:
        """Resolve a per-call clipping override against the process default."""

        return self.clip_denoised if override is None else override

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
        """Migrate legacy ``scheduler.*`` forward buffers while loading."""

        legacy_prefix = f"{prefix}scheduler."
        schedule_prefix = f"{prefix}noise_schedule."
        for name in self.noise_schedule.state_dict():
            legacy_key = f"{legacy_prefix}{name}"
            current_key = f"{schedule_prefix}{name}"
            if legacy_key in state_dict and current_key not in state_dict:
                state_dict[current_key] = state_dict[legacy_key]
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                del state_dict[key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
