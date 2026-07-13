"""DDPM process skeleton."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from stochaflow.diffusion.schedules import DiffusionScheduler
from stochaflow.utils.registry import register_diffusion


@dataclass(slots=True)
class DDPMForwardOutput:
    """Structured tensors produced by one DDPM training forward pass.

    ``timesteps`` contains mathematical state times in ``[1, T]``. The model
    receives the corresponding zero-based conditioning indices ``t - 1``.
    """

    timesteps: torch.Tensor
    xt: torch.Tensor
    noise: torch.Tensor
    predicted_noise: torch.Tensor


@register_diffusion("ddpm")
class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model process wrapper.

    This class owns the DDPM-specific training and sampling logic. The
    scheduler is responsible only for providing timestep-dependent weighting
    terms; this class is responsible for combining those terms into forward
    diffusion, denoising targets, and reverse-time sampling.

    Public methods should stay narrow:
    - DDPM training-time forward logic on clean samples
    - explicit forward diffusion as a reusable process primitive
    - reverse-process traversal
    - end-to-end generation

    Optimizer orchestration, checkpointing, logging, and data-loop management
    are intentionally outside this class. Stepwise reverse-process helpers and
    target-construction utilities are kept private because they are
    implementation details of the DDPM parameterization.

    Time convention:
        Every public process method uses mathematical state times ``0`` through
        ``T``. State time zero is the clean sample ``x_0`` and state time ``T``
        is the terminal noisy state ``x_T``. The scheduler stores only the
        ``T`` noisy-state coefficient entries: table index ``i`` contains the
        coefficients for state time ``i + 1``. Before invoking the denoiser, a
        source state time ``t >= 1`` is converted to model timestep ``t - 1``.
        No public API uses a negative timestep.
    """

    def __init__(
        self,
        scheduler: DiffusionScheduler,
        model: nn.Module,
        *,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.model = model
        self.clip_denoised = clip_denoised

    @property
    def num_timesteps(self) -> int:
        """Return the number of forward transitions and coefficient entries.

        A value of ``N`` denotes ``N`` betas and the mathematical state path
        ``x_0 -> ... -> x_N``. It does not mean that the clean state is stored
        at scheduler index zero.
        """

        return self.scheduler.num_timesteps

    def forward(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> DDPMForwardOutput:
        """Run one DDPM epsilon-prediction training forward pass.

        This method is a higher-level orchestration entry point for training. It
        selects timesteps when they are not provided, applies the forward
        noising primitive, predicts the noise component with the denoiser, and
        returns the structured tensors needed by a training objective.

        - sample or accept timesteps for the batch
        - sample Gaussian noise
        - construct noisy samples ``x_t``
        - predict ``epsilon_theta(x_t, t)``

        ``timesteps`` are mathematical noisy-state times in ``[1, T]``. A
        state time ``t`` selects scheduler index ``t - 1`` and is passed to the
        denoiser as model timestep ``t - 1``.

        Returns:
            A ``DDPMForwardOutput`` containing:
            - ``timesteps`` with shape ``(batch,)``
            - ``xt`` with the same shape as ``x0``
            - ``noise`` with the same shape as ``x0``
            - ``predicted_noise`` with the same shape as ``x0``
        """

        if timesteps is None:
            timesteps = self._sample_timesteps(x0.size(0), x0.device)
        xt, noise = self.add_noise(x0, timesteps)
        predicted_noise = self._predict_noise(xt, timesteps)

        return DDPMForwardOutput(
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
        """Add noise to clean samples according to the forward process.

        This is the reusable forward diffusion primitive. If ``noise`` is not
        provided, the method samples Gaussian noise internally. The return value
        is a tuple ``(xt, noise)`` so callers can reuse the exact perturbation
        used to construct ``x_t``.

        ``timesteps`` contains mathematical state times in ``[0, T]``. State
        time zero returns the clean input unchanged; state time ``t >= 1`` uses
        scheduler table index ``t - 1``.
        """

        if noise is None:
            noise = self._sample_noise(x0)

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=True)
        sqrt_alpha_bar_t = self._coefficients_at_state_timesteps(
            "sqrt_alpha_bar_t",
            timesteps,
            x0.size(),
            clean_value=1.0,
        )
        sqrt_one_minus_alpha_bar_t = self._coefficients_at_state_timesteps(
            "sqrt_one_minus_alpha_bar_t",
            timesteps,
            x0.size(),
            clean_value=0.0,
        )

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def reverse(
        self,
        x_from: torch.Tensor,
        timestep_from: int,
        timestep_to: int = 0,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Run the reverse process between mathematical state times.

        This is the higher-level reverse traversal API. The full batch is
        assumed to start at the shared discrete timestep ``timestep_from`` and
        is iteratively stepped backwards until it reaches ``timestep_to``.

        State times satisfy ``0 <= timestep_to <= timestep_from <= T``. Each
        transition uses its source state time ``t >= 1`` and internally maps it
        to scheduler/model index ``t - 1``. State time zero is a real public
        endpoint and no negative sentinel is used.

        Args:
            x_from: Batch at mathematical state time ``timestep_from``.
            timestep_from: Current state time in ``[0, T]``.
            timestep_to: Target state time in ``[0, timestep_from]``. The
                default zero traverses to the clean state ``x_0``.
            clip_denoised: Optional clipping override for each reverse step.

        Returns:
            The batch at mathematical state time ``timestep_to``.

        Raises:
            ValueError: If the requested state times are outside the horizon or
                ordered in the forward direction.
        """

        if not 0 <= timestep_to <= timestep_from <= self.num_timesteps:
            raise ValueError(
                "expected 0 <= timestep_to <= timestep_from <= num_timesteps"
            )

        x_t = x_from
        for t in range(timestep_from, timestep_to, -1):
            timesteps = torch.full(
                (x_t.shape[0],),
                t,
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
        """Generate samples by traversing mathematical states ``T`` to ``0``.

        Args:
            sample_shape: Full batch-first shape of the desired samples.
            device: Optional target device for the initial Gaussian noise.
        """

        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        x_t = torch.randn(sample_shape, device=device)
        x_0 = self.reverse(
            x_t,
            self.num_timesteps,
            0,
            clip_denoised=clip_denoised,
        )

        return x_0

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample training timesteps for an input batch.

        This is intentionally private: timestep sampling policy is a training
        detail owned by higher-level training code. Returned values are
        mathematical noisy-state times in ``[1, T]``.
        """

        timesteps = torch.randint(
            1, self.num_timesteps + 1, torch.Size((batch_size,)), device=device
        )

        return timesteps

    def _sample_noise(self, reference: torch.Tensor) -> torch.Tensor:
        """Sample Gaussian noise shaped like ``reference``."""

        noise = torch.randn_like(reference)

        return noise

    def _predict_noise(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise at mathematical source-state times ``t >= 1``.

        The underlying denoiser receives zero-based model conditioning indices
        ``t - 1``.
        """

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        return self.model(xt, timesteps - 1)

    def reverse_step(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Run one reverse DDPM transition ``x_t -> x_{t-1}``.

        This is the batch-wise single-step reverse transition. ``timesteps`` is
        expected to be a tensor of shape ``(batch,)`` so each sample may carry
        its own mathematical source-state time in ``[1, T]``. State time one
        performs the final transition ``x_1 -> x_0``. Scheduler and model
        indices are derived internally by subtracting one.
        """

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        return self._reverse_step_from_epsilon(
            xt,
            timesteps,
            clip_denoised=self._resolve_clip_denoised(clip_denoised),
        )

    def _estimate_x0_from_epsilon(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        predicted_noise: torch.Tensor,
        clip_denoised: bool,
    ) -> torch.Tensor:
        """Estimate ``x_0`` from the epsilon parameterization."""

        scheduler_timesteps = self._state_to_scheduler_timesteps(timesteps)
        sqrt_alpha_bar_t = self.scheduler.coefficients_at(
            "sqrt_alpha_bar_t", scheduler_timesteps, xt.size()
        )
        sqrt_one_minus_alpha_bar_t = self.scheduler.coefficients_at(
            "sqrt_one_minus_alpha_bar_t", scheduler_timesteps, xt.size()
        )

        x0 = (xt - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def _reverse_step_from_epsilon(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        clip_denoised: bool,
    ) -> torch.Tensor:
        """Reverse one step from epsilon prediction and posterior coefficients.

        The denoiser predicts epsilon, from which this method reconstructs
        ``x_0``. When enabled, clipping is applied to that reconstructed clean
        prediction before it is substituted into the posterior mean, matching
        OpenAI's epsilon-parameterized DDPM sampler.
        """

        predicted_noise = self._predict_noise(xt, timesteps)
        x0 = self._estimate_x0_from_epsilon(
            xt,
            timesteps,
            predicted_noise=predicted_noise,
            clip_denoised=clip_denoised,
        )
        scheduler_timesteps = self._state_to_scheduler_timesteps(timesteps)
        posterior_mean_coef1 = self.scheduler.coefficients_at(
            "posterior_mean_coef1", scheduler_timesteps, xt.size()
        )
        posterior_mean_coef2 = self.scheduler.coefficients_at(
            "posterior_mean_coef2", scheduler_timesteps, xt.size()
        )
        posterior_mean = posterior_mean_coef1 * x0 + posterior_mean_coef2 * xt
        return posterior_mean + self._nonzero_timestep_mask(xt, timesteps) * (
            self._posterior_noise(xt, timesteps)
        )

    def _posterior_noise(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the stochastic reverse-process noise term for the timestep batch.

        This returns ``sqrt(posterior_variance_t) * z`` rather than the raw
        variance itself, so it is ready to be added directly inside
        ``reverse_step``.
        """

        z = self._sample_noise(xt)

        scheduler_timesteps = self._state_to_scheduler_timesteps(timesteps)
        sqrt_posterior_variance_t = self.scheduler.coefficients_at(
            "sqrt_posterior_variance_t", scheduler_timesteps, xt.size()
        )

        return sqrt_posterior_variance_t * z

    def _nonzero_timestep_mask(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Build the mask that disables noise for ``x_1 -> x_0``."""

        return (
            (timesteps > 1)
            .to(dtype=xt.dtype)
            .reshape((xt.shape[0],) + (1,) * (xt.ndim - 1))
        )

    def _validate_state_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        allow_clean: bool,
    ) -> torch.Tensor:
        """Validate public mathematical state times without changing meaning."""

        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a 1D tensor")
        if timesteps.dtype == torch.bool or torch.is_floating_point(timesteps):
            raise TypeError("timesteps must contain integer state times")
        timesteps = timesteps.to(dtype=torch.long)
        minimum = 0 if allow_clean else 1
        if torch.any(timesteps < minimum) or torch.any(timesteps > self.num_timesteps):
            interval = "[0, T]" if allow_clean else "[1, T]"
            raise ValueError(
                f"timesteps must be mathematical state times in {interval}"
            )
        return timesteps

    def _state_to_scheduler_timesteps(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Map source state times ``1..T`` to scheduler indices ``0..T-1``."""

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        return timesteps - 1

    def _coefficients_at_state_timesteps(
        self,
        name: str,
        timesteps: torch.Tensor,
        broadcast_shape: torch.Size,
        *,
        clean_value: float,
    ) -> torch.Tensor:
        """Gather a coefficient table with an explicit mathematical ``t=0``."""

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=True)
        scheduler_timesteps = (timesteps - 1).clamp_min(0)
        values = self.scheduler.coefficients_at(
            name,
            scheduler_timesteps,
            broadcast_shape,
        )
        clean_mask = (timesteps == 0).reshape(
            (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1)
        )
        return torch.where(clean_mask, torch.full_like(values, clean_value), values)

    def _resolve_clip_denoised(self, override: bool | None) -> bool:
        return self.clip_denoised if override is None else override
