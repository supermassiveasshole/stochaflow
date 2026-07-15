"""Discrete variance-preserving noise paths."""

import torch

from .base import NoiseSchedule


class DiscreteVPSchedule(NoiseSchedule):
    r"""Store the canonical tables of a discrete VP forward process.

    The schedule is initialized from one beta per forward transition and owns
    only quantities shared by variance-preserving processes: ``β_t``,
    ``α_t = 1 - β_t``, and ``ᾱ_t = ∏_{s=1}^t α_s``, together with the two
    marginal scales derived from ``ᾱ_t``.

    Public state times follow the mathematical convention ``0..T``: state
    ``0`` is clean and state ``T`` is terminal. Stored tables have length
    ``T`` and use zero-based array indices ``0..T-1``; array index ``i``
    stores the coefficient for mathematical state ``i + 1``. The clean
    endpoint is represented analytically and is not inserted into the tables.

    Reverse-process quantities, including DDPM posterior coefficients, are
    intentionally outside this class. They belong to the process or sampler
    whose reverse equation defines them.

    Args:
        betas: One-dimensional tensor containing one beta for each forward
            transition. Values must be finite and lie strictly in ``(0, 1)``.
    """

    beta_t: torch.Tensor
    alpha_t: torch.Tensor
    alpha_bar_t: torch.Tensor
    sqrt_alpha_bar_t: torch.Tensor
    sqrt_one_minus_alpha_bar_t: torch.Tensor

    def __init__(self, betas: torch.Tensor) -> None:
        super().__init__()
        betas = self._validate_betas(betas)

        alpha_t = 1.0 - betas
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer("beta_t", betas)
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("alpha_bar_t", alpha_bar_t)
        self.register_buffer("sqrt_alpha_bar_t", torch.sqrt(alpha_bar_t))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar_t",
            torch.sqrt(1.0 - alpha_bar_t),
        )

    @property
    def terminal_time(self) -> int:
        """Return terminal mathematical state time ``T``."""

        return self.num_timesteps

    def validate_state_times(self, state_times: torch.Tensor) -> torch.Tensor:
        """Validate integer public state times in the closed interval ``[0, T]``."""

        if state_times.ndim != 1:
            raise ValueError("state_times must be a 1D tensor")
        if state_times.dtype == torch.bool or torch.is_floating_point(state_times):
            raise TypeError("state_times must contain integer mathematical states")

        state_times = state_times.to(dtype=torch.long)
        if torch.any(state_times < 0) or torch.any(
            state_times > self.num_timesteps
        ):
            raise ValueError("state_times must lie in [0, T]")
        return state_times

    def marginal_scales(
        self,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return VP marginal scales at public state times ``0..T``.

        State zero is returned exactly as ``(1, 0)``. A noisy state ``t >= 1``
        reads internal table index ``t - 1`` and returns
        ``(sqrt(ᾱ_t), sqrt(1 - ᾱ_t))``.
        """

        state_times = self.validate_state_times(state_times)
        self._validate_broadcast_shape(state_times, broadcast_shape)

        table_indices = (state_times - 1).clamp_min(0)
        signal_scale = self._gather_table(
            self.sqrt_alpha_bar_t,
            table_indices,
            broadcast_shape,
        )
        noise_scale = self._gather_table(
            self.sqrt_one_minus_alpha_bar_t,
            table_indices,
            broadcast_shape,
        )
        clean_mask = (state_times == self.clean_time).reshape(
            self._broadcastable_shape(state_times, broadcast_shape)
        )
        return (
            torch.where(clean_mask, torch.ones_like(signal_scale), signal_scale),
            torch.where(clean_mask, torch.zeros_like(noise_scale), noise_scale),
        )

    @staticmethod
    def _validate_betas(betas: torch.Tensor) -> torch.Tensor:
        """Normalize and validate the defining VP transition table."""

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
    def _validate_num_timesteps(num_timesteps: int) -> None:
        """Require a positive, non-boolean number of discrete transitions."""

        if isinstance(num_timesteps, bool) or not isinstance(num_timesteps, int):
            raise TypeError("num_timesteps must be an integer")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

    @staticmethod
    def _validate_dtype(dtype: torch.dtype) -> None:
        """Require a real floating-point dtype for coefficient tables."""

        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point torch dtype")

    @staticmethod
    def _validate_broadcast_shape(
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> None:
        """Require a batch-first target shape aligned with ``state_times``."""

        if len(broadcast_shape) == 0:
            raise ValueError("broadcast_shape must have at least one dimension")
        if broadcast_shape[0] != state_times.shape[0]:
            raise ValueError("broadcast_shape batch dimension must match state_times")

    @staticmethod
    def _broadcastable_shape(
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> tuple[int, ...]:
        """Return the singleton-expanded shape for per-sample coefficients."""

        return (state_times.shape[0],) + (1,) * (len(broadcast_shape) - 1)

    def _gather_table(
        self,
        values: torch.Tensor,
        table_indices: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        """Gather one schedule-owned table for batch broadcasting."""

        gathered = values.gather(0, table_indices.to(values.device))
        return gathered.reshape(
            self._broadcastable_shape(table_indices, broadcast_shape)
        )
