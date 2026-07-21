"""Ancestral DDPM sampler."""

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


@REGISTRIES.samplers.register("ddpm")
class DDPMAncestralSampler(Sampler):
    """Run adjacent ancestral reverse transitions for Gaussian dynamics."""

    def __init__(
        self,
        *,
        start_time: object = None,
        end_time: object = 0,
    ) -> None:
        if start_time is not None and (
            isinstance(start_time, bool) or not isinstance(start_time, int)
        ):
            raise TypeError("DDPM start_time must be an integer or null")
        if isinstance(end_time, bool) or not isinstance(end_time, int):
            raise TypeError("DDPM end_time must be an integer")
        self.start_time: int | None = start_time
        self.end_time: int = end_time

    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        """Execute the configured adjacent reverse path."""

        if not isinstance(dynamics, GaussianDenoisingDynamics):
            raise TypeError("ddpm sampler requires GaussianDenoisingDynamics")
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("ddpm initial_state must be a Tensor")
        process = dynamics.process
        start = process.terminal_time if self.start_time is None else self.start_time
        end = self.end_time
        if not process.clean_time <= end <= start <= process.terminal_time:
            raise ValueError(
                "DDPM requires clean_time <= end_time <= start_time <= terminal_time"
            )

        state = initial_state
        num_steps = start - end
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    step_index=0,
                    coordinate=start,
                    state=state,
                    is_final=num_steps == 0,
                    diagnostics={},
                )
            )
        evaluations = 0
        for step_index, state_time in enumerate(range(start, end, -1), start=1):
            times = torch.full(
                (state.shape[0],), state_time, device=state.device, dtype=torch.long
            )
            prediction = dynamics.predict(state, times)
            evaluations += 1
            state = process.posterior_mean(state, times, prediction.clean)
            if state_time - 1 > process.clean_time:
                noise = torch.randn(
                    state.shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
                state = state + process.posterior_standard_deviation(
                    times, state.size()
                ) * noise
            coordinate = state_time - 1
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=coordinate,
                        state=state,
                        is_final=step_index == num_steps,
                        diagnostics={"num_model_evaluations": evaluations},
                    )
                )
        return SamplerResult(
            state,
            num_steps,
            {"num_model_evaluations": evaluations},
        )


__all__ = ["DDPMAncestralSampler"]
