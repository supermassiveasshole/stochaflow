"""Denoising Diffusion Implicit Model sampler."""

from collections.abc import Sequence
from typing import Any

import torch

from stochaflow.utils.registry import REGISTRIES

from .dynamics import GenerativeDynamics
from .gaussian import GaussianDenoisingDynamics
from .sampler import (
    Sampler,
    SamplerResult,
    SamplingObservation,
    SamplingObserver,
)


@REGISTRIES.samplers.register("ddim")
class DDIMSampler(Sampler):
    """Run arbitrary descending discrete Gaussian state schedules."""

    def __init__(
        self,
        *,
        num_inference_steps: object = None,
        schedule: object = None,
        eta: object = 0.0,
    ) -> None:
        if num_inference_steps is not None and schedule is not None:
            raise ValueError("num_inference_steps and schedule are mutually exclusive")
        if num_inference_steps is not None and (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, int)
            or num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        if isinstance(eta, bool) or not isinstance(eta, (int, float)):
            raise TypeError("DDIM eta must be numeric")
        if not 0 <= eta <= 1:
            raise ValueError("DDIM eta must be in [0, 1]")
        if schedule is not None:
            if isinstance(schedule, (str, bytes)) or not isinstance(
                schedule, Sequence
            ):
                raise TypeError("DDIM schedule must be a sequence of integers")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in schedule
            ):
                raise TypeError("DDIM schedule must contain integer states")
        self.num_inference_steps: int | None = num_inference_steps
        self.explicit_schedule: tuple[int, ...] | None = (
            tuple(schedule) if schedule is not None else None
        )
        self.eta = float(eta)

    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        """Execute one complete DDIM schedule."""

        if not isinstance(dynamics, GaussianDenoisingDynamics):
            raise TypeError("ddim sampler requires GaussianDenoisingDynamics")
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("ddim initial_state must be a Tensor")
        process = dynamics.process
        states = self._schedule(
            process.clean_time, process.terminal_time, initial_state.device
        )
        num_steps = states.numel() - 1
        current = initial_state
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    step_index=0,
                    coordinate=int(states[0]),
                    state=current,
                    is_final=False,
                    diagnostics={},
                )
            )
        evaluations = 0
        for step_index, (source, target) in enumerate(
            zip(states[:-1], states[1:]), start=1
        ):
            source_times = source.expand(current.shape[0])
            target_times = target.expand(current.shape[0])
            prediction = dynamics.predict(current, source_times)
            evaluations += 1
            if self.eta == 1.0 and int(target) == int(source) - 1:
                current = dynamics.process.posterior_mean(
                    current, source_times, prediction.clean
                )
                if int(target) > process.clean_time:
                    stochastic_scale = (
                        dynamics.process.posterior_standard_deviation(
                            source_times, current.size()
                        )
                    )
                    if bool(torch.any(stochastic_scale != 0)):
                        transition_noise = torch.randn(
                            current.shape,
                            device=current.device,
                            dtype=current.dtype,
                            generator=generator,
                        )
                        current = current + stochastic_scale * transition_noise
            else:
                scales_t = dynamics.process.marginal_scales(
                    source_times, current.size()
                )
                scales_s = dynamics.process.marginal_scales(
                    target_times, current.size()
                )
                signal_t, noise_t = scales_t.signal, scales_t.noise
                signal_s, noise_s = scales_s.signal, scales_s.noise
                posterior_variance = (
                    noise_s.square()
                    / noise_t.square()
                    * (1.0 - signal_t.square() / signal_s.square())
                ).clamp_min(0.0)
                direction_scale = (
                    noise_s.square() - self.eta**2 * posterior_variance
                ).clamp_min(0.0).sqrt()
                current = (
                    signal_s * prediction.clean
                    + direction_scale * prediction.epsilon
                )
                stochastic_scale = self.eta * posterior_variance.sqrt()
                if (
                    int(target) > process.clean_time
                    and bool(torch.any(stochastic_scale != 0))
                ):
                    transition_noise = torch.randn(
                        current.shape,
                        device=current.device,
                        dtype=current.dtype,
                        generator=generator,
                    )
                    current = current + stochastic_scale * transition_noise
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=int(target),
                        state=current,
                        is_final=step_index == num_steps,
                        diagnostics={"num_model_evaluations": evaluations},
                    )
                )
        return SamplerResult(
            current,
            num_steps,
            {"num_model_evaluations": evaluations},
        )

    def _schedule(
        self, clean_time: int, terminal_time: int, device: torch.device
    ) -> torch.Tensor:
        if self.explicit_schedule is not None:
            states = torch.as_tensor(self.explicit_schedule, device=device)
            if states.ndim != 1 or states.numel() < 2:
                raise ValueError("DDIM schedule must contain at least two states")
            if states.dtype == torch.bool or torch.is_floating_point(states):
                raise TypeError("DDIM schedule must contain integer states")
            states = states.to(dtype=torch.long)
        else:
            transitions = terminal_time - clean_time
            steps = self.num_inference_steps or transitions
            if steps > transitions:
                raise ValueError(
                    "num_inference_steps must not exceed process transitions"
                )
            states = torch.linspace(
                terminal_time,
                clean_time,
                steps=steps + 1,
                device=device,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
        if int(states[0]) > terminal_time or int(states[-1]) < clean_time:
            raise ValueError(
                "DDIM schedule states must lie in [clean_time, terminal_time]"
            )
        if not torch.all(states[:-1] > states[1:]):
            raise ValueError("DDIM schedule must be strictly descending and unique")
        return states


__all__ = ["DDIMSampler"]
