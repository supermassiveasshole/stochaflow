"""Ancestral DDPM sampler."""

from typing import Any, cast

import torch

from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.utils.registry import REGISTRIES

from .dynamics import GenerativeDynamics
from .gaussian import (
    GaussianDenoisingDynamics,
    GaussianPrediction,
    GaussianTransition,
    _validate_gaussian_prediction,
)
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

    def transition(
        self,
        process: DiscreteGaussianDenoisingProcess,
        state: torch.Tensor,
        state_times: torch.Tensor,
        prediction: GaussianPrediction,
    ) -> GaussianTransition:
        """Build one adjacent ``x_t -> x_{t-1}`` transition distribution."""

        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "DDPM transition requires DiscreteGaussianDenoisingProcess"
            )
        process = process_value
        state_value = cast(object, state)
        if not isinstance(state_value, torch.Tensor):
            raise TypeError("DDPM transition state must be a Tensor")
        state = state_value
        if state.ndim == 0:
            raise ValueError("DDPM transition state must include a batch dimension")
        if not torch.is_floating_point(state):
            raise TypeError("DDPM transition state must be floating-point")
        state_times = process.validate_noisy_state_times(state_times)
        if state_times.shape[0] != state.shape[0]:
            raise ValueError("DDPM state times must match the state batch")
        if state_times.device != state.device:
            raise ValueError("DDPM state times must share the state device")
        prediction = _validate_gaussian_prediction(prediction, state=state)
        mean = process.posterior_mean(state, state_times, prediction.clean)
        standard_deviation = process.posterior_standard_deviation(
            state_times,
            state.size(),
        )
        target_times = state_times - 1
        stochastic_mask = (target_times > process.clean_time).reshape(
            (state.shape[0],) + (1,) * (state.ndim - 1)
        )
        standard_deviation = standard_deviation * stochastic_mask
        standard_deviation = standard_deviation.to(
            device=mean.device,
            dtype=mean.dtype,
        )
        return GaussianTransition(mean, standard_deviation)

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
            transition = self.transition(
                process,
                state,
                times,
                prediction,
            )
            coordinate = state_time - 1
            state = (
                transition.mean
                if coordinate == process.clean_time
                else transition.sample(generator=generator)
            )
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=coordinate,
                        state=state,
                        is_final=step_index == num_steps,
                        diagnostics={"num_dynamics_evaluations": evaluations},
                    )
                )
        return SamplerResult(
            state,
            num_steps,
            {"num_dynamics_evaluations": evaluations},
        )


__all__ = ["DDPMAncestralSampler"]
