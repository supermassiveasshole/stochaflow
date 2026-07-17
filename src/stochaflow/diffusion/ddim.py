"""DDIM sampling interfaces built on the shared epsilon-training contract."""

from collections.abc import Sequence

import torch
import torch.nn as nn

from stochaflow.diffusion.gaussian import GaussianDiffusion
from stochaflow.diffusion.noise_schedules import DiscreteVPSchedule
from stochaflow.sampling.sampler import SamplingTrace, TrajectoryFrame
from stochaflow.utils.registry import REGISTRIES

SamplingTimesteps = Sequence[int] | torch.Tensor


@REGISTRIES.diffusions.register("ddim")
class DDIM(GaussianDiffusion):
    """Denoising Diffusion Implicit Model with an explicit inference schedule.

    DDIM uses the same forward noising process and epsilon-prediction training
    contract as DDPM through their common :class:`GaussianDiffusion` base. It
    does not inherit or allocate DDPM-specific posterior coefficients.
    Its sampling interface is deliberately state-schedule based. Public
    schedules contain mathematical state times in ``[0, T]`` including both
    endpoints: ``T`` is the terminal noisy state and ``0`` is the clean state.
    A schedule with ``K + 1`` state points defines exactly ``K`` transitions.
    Every transition invokes the denoiser only at its source state ``t >= 1``;
    the model receives zero-based timestep ``t - 1``. Scheduler indices follow
    the same internal conversion. This supports uniformly spaced inference as
    well as arbitrary, non-uniform state-time subsequences without assigning
    special semantics to a public ``stride`` argument.

    :meth:`reverse_step` implements one selected-pair DDIM transition.
    :meth:`reverse` and :meth:`sample` resolve a complete schedule and apply
    each selected transition in descending state-time order.
    """

    def __init__(
        self,
        noise_schedule: DiscreteVPSchedule,
        model: nn.Module,
        *,
        num_inference_steps: int | None = None,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> None:
        """Initialize DDIM's training process and default inference settings.

        Args:
            noise_schedule: Training-time noise path whose cumulative alpha
                coefficients are reused for DDIM sampling.
            model: Denoiser that predicts epsilon for a batch of noisy samples.
            num_inference_steps: Default number of reverse transitions and
                denoiser evaluations during sampling. The corresponding
                schedule contains one additional state point. A uniformly
                spaced descending state-time sequence is derived when a call
                does not provide explicit ``timesteps``. The value must lie in
                ``[1, noise_schedule.num_timesteps]``.
            eta: Default DDIM stochasticity in ``[0, 1]``. ``0`` selects
                deterministic DDIM and ``1`` uses the DDPM-style posterior
                variance. This parameter is independent of the selected
                inference timesteps.
            clip_denoised: Whether predicted clean samples should be clipped by
                default during future reverse sampling.

        Raises:
            TypeError: If ``num_inference_steps`` is not an integer or ``eta``
                is not numeric.
            ValueError: If ``num_inference_steps`` or ``eta`` is out of range.
        """

        if num_inference_steps is None:
            num_inference_steps = noise_schedule.num_timesteps
        self._validate_num_inference_steps(
            num_inference_steps,
            num_train_timesteps=noise_schedule.num_timesteps,
        )
        if isinstance(eta, bool) or not isinstance(eta, (int, float)):
            raise TypeError("eta must be numeric")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")

        super().__init__(
            noise_schedule=noise_schedule,
            model=model,
            clip_denoised=clip_denoised,
        )
        self.num_inference_steps = num_inference_steps
        self.eta = float(eta)

    def sampling_timesteps(
        self,
        *,
        num_inference_steps: int | None = None,
        timesteps: SamplingTimesteps | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Resolve a complete descending DDIM state-time schedule.

        An explicitly supplied sequence is authoritative and allows schedules
        with non-uniform gaps. Otherwise, this method constructs a uniformly
        spaced sequence from ``num_inference_steps`` or, when it is omitted,
        from :attr:`num_inference_steps` configured at construction time. The
        returned sequence contains ``K + 1`` mathematical states for ``K``
        transitions. It always begins at terminal state time ``T`` and ends at
        clean state time ``0``. Adjacent entries define the selected reverse
        transitions; only the first ``K`` entries are denoiser source states.

        Args:
            num_inference_steps: Number ``K`` of reverse transitions and
                denoiser evaluations in a generated, uniformly spaced schedule.
                Mutually exclusive with ``timesteps``.
            timesteps: Explicit mathematical state times ordered strictly from
                larger to smaller values. Every value must be an integer in
                ``[0, T]``. A complete sampling schedule must begin at ``T``
                and end at ``0``.
            device: Device on which to return the resolved indices.

        Returns:
            A one-dimensional ``torch.long`` tensor of strictly descending
            mathematical state times. Its length is ``num_inference_steps + 1``
            for a generated schedule, or the number of explicit entries.

        Raises:
            ValueError: If both schedule inputs are provided, an explicit
            sequence is empty, out of range, or is not strictly descending.
            TypeError: If a requested number of inference steps is not an
                integer, or if explicit timesteps are not integral values.
        """

        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("num_inference_steps and timesteps are mutually exclusive")

        if timesteps is not None:
            return self._validate_sampling_timesteps(timesteps, device=device)

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        self._validate_num_inference_steps(
            num_inference_steps,
            num_train_timesteps=self.num_timesteps,
        )

        return (
            torch.linspace(
                self.num_timesteps,
                0,
                steps=num_inference_steps + 1,
                dtype=torch.float32,
                device=device,
            )
            .round()
            .to(dtype=torch.long)
        )

    @staticmethod
    def _validate_num_inference_steps(
        num_inference_steps: int,
        *,
        num_train_timesteps: int,
    ) -> None:
        """Validate the cardinality of a generated inference schedule."""

        if isinstance(num_inference_steps, bool) or not isinstance(
            num_inference_steps, int
        ):
            raise TypeError("num_inference_steps must be an integer")
        if not 1 <= num_inference_steps <= num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be in [1, noise_schedule.num_timesteps]"
            )

    def _validate_sampling_timesteps(
        self,
        timesteps: SamplingTimesteps,
        *,
        device: torch.device | None,
    ) -> torch.Tensor:
        """Normalize and validate an explicit DDIM inference schedule."""

        try:
            resolved = torch.as_tensor(timesteps, device=device)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "timesteps must be a one-dimensional integer sequence"
            ) from exc

        if resolved.ndim != 1:
            raise ValueError("timesteps must be a one-dimensional sequence")
        if resolved.numel() < 2:
            raise ValueError("timesteps must contain at least two state times")
        if resolved.dtype == torch.bool or torch.is_floating_point(resolved):
            raise TypeError("timesteps must contain integer values")

        resolved = resolved.to(dtype=torch.long)
        if torch.any(resolved < 0) or torch.any(resolved > self.num_timesteps):
            raise ValueError("timesteps must be mathematical state times in [0, T]")
        if resolved.numel() > 1 and not torch.all(resolved[:-1] > resolved[1:]):
            raise ValueError("timesteps must be strictly descending and unique")
        if int(resolved[0]) != self.num_timesteps or int(resolved[-1]) != 0:
            raise ValueError(
                "a complete sampling schedule must start at T and end at 0"
            )
        return resolved

    def reverse_step(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        previous_timesteps: torch.Tensor,
        eta: float | None = None,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        r"""Apply one DDIM transition between selected mathematical states.

        ``previous_timesteps`` is deliberately required: unlike DDPM, DDIM does
        not intrinsically mean the adjacent state ``t - 1``. This enables a
        batch to transition from a source state to any earlier selected state.
        For each source-target pair ``t -> s``, this method evaluates

        .. math::

            x_s = \sqrt{\bar{\alpha}_s}\,\hat{x}_0
                + \sqrt{1 - \bar{\alpha}_s - \sigma_{t \to s}^2}\,
                  \hat{\epsilon}_{\mathrm{direction}}
                + \sigma_{t \to s} z,
            \qquad
            \sigma_{t \to s}^2 = \eta^2\widetilde{\beta}_{t \to s}.

        If clipping changes :math:`\hat{x}_0`, the residual used in the
        direction term is recomputed from the clipped value. This preserves
        the identity

        .. math::

            x_t = \sqrt{\bar{\alpha}_t}\,\hat{x}_0
                + \sqrt{1 - \bar{\alpha}_t}\,\hat{\epsilon}

        and keeps the adjacent :math:`\eta = 1` update consistent with DDPM's
        clipped posterior mean. With :math:`\eta = 0`, no transition noise is
        sampled.

        Args:
            xt: Noisy samples at the batch-aligned source state times in
                ``timesteps``.
            timesteps: One-dimensional integer tensor containing mathematical
                source state times in ``[1, T]``. The denoiser will receive
                ``timesteps - 1``.
            previous_timesteps: One-dimensional integer tensor, with the same
                shape as ``timesteps``, naming mathematical target state times
                in ``[0, T - 1]``. Each target must be smaller than its
                corresponding source. State time zero is the clean endpoint.
            eta: Optional per-call stochasticity override. ``None`` uses
                :attr:`eta` configured at construction time.
            clip_denoised: Optional per-call clipping override.

        Returns:
            The samples after their selected DDIM transition.

        Raises:
            TypeError: If state times are non-integral or ``eta`` is not a
                real numeric value.
            ValueError: If state times are out of range or unordered, or if
                ``eta`` lies outside ``[0, 1]``.
        """

        timesteps = self._validate_state_timesteps(timesteps, allow_clean=False)
        previous_timesteps = self._validate_state_timesteps(
            previous_timesteps, allow_clean=True
        )
        if torch.any(previous_timesteps >= timesteps):
            raise ValueError("previous timesteps must be smaller than timesteps")

        eps = self._predict_noise(xt, timesteps)
        x0_hat = self._estimate_x0_from_epsilon(
            xt,
            timesteps,
            predicted_noise=eps,
            clip_denoised=clip_denoised
            if clip_denoised is not None
            else self.clip_denoised,
        )
        signal_scale_t, noise_scale_t = self.noise_schedule.marginal_scales(
            timesteps, xt.size()
        )
        signal_scale_s, noise_scale_s = self.noise_schedule.marginal_scales(
            previous_timesteps, xt.size()
        )
        beta_tild = (
            noise_scale_s.square()
            / noise_scale_t.square()
            * (1 - signal_scale_t.square() / signal_scale_s.square())
        )
        eps_for_direction = (xt - signal_scale_t * x0_hat) / noise_scale_t
        if eta is None:
            eta = self.eta
        if isinstance(eta, bool) or not isinstance(eta, (int, float)):
            raise TypeError("eta must be numeric")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        noise = torch.randn_like(xt) if eta > 0.0 else 0.0
        xs = (
            signal_scale_s * x0_hat
            + torch.sqrt((noise_scale_s.square() - (eta**2) * beta_tild).clamp_min(0.0))
            * eps_for_direction
            + eta * beta_tild.clamp_min(0.0).sqrt() * noise
        )
        return xs

    def reverse(
        self,
        x_from: torch.Tensor,
        *,
        num_inference_steps: int | None = None,
        timesteps: SamplingTimesteps | None = None,
        eta: float | None = None,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Traverse a complete DDIM mathematical state-time schedule.

        The first state time is ``T`` and identifies the terminal noise level
        represented by ``x_from``. The method applies one transition for every
        adjacent pair and finishes at the explicit clean state time ``0``.
        Thus a schedule with ``K + 1`` entries performs exactly ``K`` denoiser
        evaluations and transitions.

        Args:
            x_from: Batch of samples at terminal mathematical state time ``T``.
            num_inference_steps: Number of uniformly spaced reverse transitions
                and denoiser evaluations.
                Mutually exclusive with ``timesteps`` and defaults to
                :attr:`num_inference_steps`.
            timesteps: Explicit, strictly descending mathematical state-time
                schedule beginning at ``T`` and ending at ``0``.
            eta: Optional per-call stochasticity override. ``None`` uses
                :attr:`eta`.
            clip_denoised: Optional per-call clipping override.

        Returns:
            The predicted clean-sample batch after all DDIM transitions.
        """

        timesteps = self.sampling_timesteps(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=x_from.device,
        )

        xt = x_from
        for timestep, previous_timestep in zip(timesteps[:-1], timesteps[1:]):
            timestep = timestep.broadcast_to((xt.size(0),))
            previous_timestep = previous_timestep.broadcast_to((xt.size(0),))
            xt = self.reverse_step(
                xt,
                timestep,
                previous_timesteps=previous_timestep,
                eta=eta,
                clip_denoised=clip_denoised,
            )
        return xt

    def sample(
        self,
        sample_shape: torch.Size,
        device: torch.device | None = None,
        *,
        num_inference_steps: int | None = None,
        timesteps: SamplingTimesteps | None = None,
        eta: float | None = None,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Generate samples by reversing Gaussian noise along a DDIM schedule.

        ``sample`` initializes a batch at mathematical terminal state time
        ``T``. Every generated or explicit schedule includes ``T`` as its first
        point and clean state time ``0`` as its last point.

        Args:
            sample_shape: Full batch-first shape of the generated samples.
            device: Device for the initial Gaussian noise. If omitted, the
                denoiser's parameter device will be used.
            num_inference_steps: Number ``K`` of uniformly spaced reverse
                transitions and denoiser evaluations. The resolved schedule
                contains ``K + 1`` state points.
                Mutually exclusive with ``timesteps`` and defaults to
                :attr:`num_inference_steps`.
            timesteps: Explicit, strictly descending mathematical state-time
                schedule starting at ``T`` and ending at ``0``.
            eta: Optional per-call stochasticity override. ``None`` uses
                :attr:`eta`.
            clip_denoised: Optional per-call clipping override.

        Returns:
            A batch of generated clean samples.
        """

        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        initial_noise = torch.randn(sample_shape, device=device)
        return self.sample_from_noise(
            initial_noise,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            eta=eta,
            clip_denoised=clip_denoised,
        )

    def sample_from_noise(
        self,
        initial_noise: torch.Tensor,
        *,
        num_inference_steps: int | None = None,
        timesteps: SamplingTimesteps | None = None,
        eta: float | None = None,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        """Generate samples from a caller-provided terminal noise batch."""

        return self.reverse(
            initial_noise,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            eta=eta,
            clip_denoised=clip_denoised,
        )

    def sample_trajectory(
        self,
        sample_shape: torch.Size,
        device: torch.device | None = None,
        *,
        step_interval: int = 1,
    ) -> SamplingTrace:
        """Generate a debug trace along the configured inference schedule."""

        if step_interval <= 0:
            raise ValueError("DDIM trajectory step_interval must be positive")
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        initial_noise = torch.randn(sample_shape, device=device)
        return self.sample_trajectory_from_noise(
            initial_noise,
            step_interval=step_interval,
        )

    def sample_trajectory_from_noise(
        self,
        initial_noise: torch.Tensor,
        *,
        step_interval: int = 1,
    ) -> SamplingTrace:
        """Trace sampling from a caller-provided terminal noise batch."""

        if step_interval <= 0:
            raise ValueError("DDIM trajectory step_interval must be positive")
        schedule = self.sampling_timesteps(device=initial_noise.device)
        current = initial_noise
        frames = [TrajectoryFrame(int(schedule[0]), current.detach().cpu())]
        transitions = zip(schedule[:-1], schedule[1:])
        for transition_index, (state_time, target_time) in enumerate(
            transitions,
            start=1,
        ):
            current = self.reverse_step(
                current,
                state_time.broadcast_to((current.size(0),)),
                previous_timesteps=target_time.broadcast_to((current.size(0),)),
            )
            if transition_index % step_interval == 0 or int(target_time) == 0:
                frames.append(
                    TrajectoryFrame(int(target_time), current.detach().cpu())
                )
        return SamplingTrace(samples=current, frames=frames)
