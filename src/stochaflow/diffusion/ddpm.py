"""DDPM process skeleton."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from stochaflow.diffusion.schedules import DiffusionScheduler
from stochaflow.utils.registry import register_diffusion


@dataclass(slots=True)
class DDPMForwardOutput:
    """Structured tensors produced by one DDPM training forward pass."""

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
    """

    def __init__(self, scheduler: DiffusionScheduler, model: nn.Module) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.model = model

    @property
    def num_timesteps(self) -> int:
        """Return the length of the scheduler's discrete time horizon."""

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
        """

        if noise is None:
            noise = self._sample_noise(x0)

        sqrt_alpha_bar_t = self.scheduler.coefficients_at(
            "sqrt_alpha_bar_t", timesteps, x0.size()
        )
        sqrt_one_minus_alpha_bar_t = self.scheduler.coefficients_at(
            "sqrt_one_minus_alpha_bar_t", timesteps, x0.size()
        )

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise, noise

    def reverse(
        self,
        x_from: torch.Tensor,
        timestep_from: int,
        timestep_to: int = 0,
    ) -> torch.Tensor:
        """Run the reverse process from one discrete timestep to another.

        This is the higher-level reverse traversal API. The full batch is
        assumed to start at the shared discrete timestep ``timestep_from`` and
        is iteratively stepped back until the batch reaches ``x_{timestep_to}``.

        Timestep indices are 0-based and must satisfy:
        ``0 <= timestep_to <= timestep_from < num_timesteps``.
        """

        if not 0 <= timestep_to <= timestep_from < self.num_timesteps:
            raise ValueError(
                "expected 0 <= timestep_to <= timestep_from < num_timesteps"
            )

        x_t = x_from
        for t in range(timestep_from, timestep_to, -1):
            timesteps = torch.full(
                (x_t.shape[0],),
                t,
                dtype=torch.long,
                device=x_t.device,
            )
            x_t = self.reverse_step(x_t, timesteps)

        return x_t

    def sample(
        self,
        sample_shape: torch.Size,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Generate samples by reversing from Gaussian noise at the final step.

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
        x_0 = self.reverse(x_t, self.num_timesteps - 1, 0)

        return x_0

    def sample_trajectory(
        self,
        sample_shape: torch.Size,
        *,
        device: torch.device | None = None,
        capture_every: int = 200,
    ) -> dict[int, torch.Tensor]:
        """Generate samples and capture intermediate reverse-process states.

        The returned dictionary is keyed by the 0-based DDPM timestep index.
        It always includes the initial Gaussian state at ``num_timesteps - 1``
        and the final generated state at ``0``. Intermediate states are captured
        every ``capture_every`` reverse steps.
        """

        if capture_every <= 0:
            raise ValueError("capture_every must be positive")
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        x_t = torch.randn(sample_shape, device=device)
        trajectory: dict[int, torch.Tensor] = {
            self.num_timesteps - 1: x_t.detach().cpu()
        }

        steps_since_capture = 0
        for t in range(self.num_timesteps - 1, 0, -1):
            timesteps = torch.full(
                (x_t.shape[0],),
                t,
                dtype=torch.long,
                device=x_t.device,
            )
            x_t = self.reverse_step(x_t, timesteps)
            current_timestep = t - 1
            steps_since_capture += 1
            if steps_since_capture >= capture_every or current_timestep == 0:
                trajectory[current_timestep] = x_t.detach().cpu()
                steps_since_capture = 0

        return trajectory

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample training timesteps for an input batch.

        This is intentionally private: timestep sampling policy is a training
        detail owned by higher-level training code.
        """

        timesteps = torch.randint(
            0, self.num_timesteps, torch.Size((batch_size,)), device=device
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
        """Predict the noise component for ``x_t`` at the given timesteps.

        This helper should usually delegate to the underlying denoiser model.
        """

        return self.model(xt, timesteps)

    def reverse_step(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Run one reverse DDPM step from ``x_t`` to ``x_{t-1}``.

        This is the batch-wise single-step reverse transition. ``timesteps`` is
        expected to be a tensor of shape ``(batch,)`` so each sample may carry
        its own current discrete timestep. The full reverse traversal exposed by
        ``reverse()`` currently uses a shared scalar timestep for the whole
        batch and expands it to this batch form.
        """

        mask = (
            (timesteps > 0)
            .to(dtype=xt.dtype)
            .reshape((xt.shape[0],) + (1,) * (xt.ndim - 1))
        )

        mu = self._p_mean(xt, timesteps)
        noise_term = self._p_noise(xt, timesteps)

        x_prev = mu + mask * noise_term

        return x_prev

    def _p_mean(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Construct the reverse-process mean for the current timestep batch."""

        sqrt_recip_alpha_t = self.scheduler.coefficients_at(
            "sqrt_recip_alpha_t", timesteps, xt.size()
        )
        beta_over_sqrt_one_minus_alpha_bar_t = self.scheduler.coefficients_at(
            "beta_over_sqrt_one_minus_alpha_bar_t", timesteps, xt.size()
        )

        p_mean = sqrt_recip_alpha_t * (
            xt
            - beta_over_sqrt_one_minus_alpha_bar_t * self._predict_noise(xt, timesteps)
        )

        return p_mean

    def _p_noise(
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

        sqrt_posterior_variance_t = self.scheduler.coefficients_at(
            "sqrt_posterior_variance_t", timesteps, xt.size()
        )

        return sqrt_posterior_variance_t * z
