"""Denoising Diffusion Implicit Model sampler."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Any, cast

import torch

from stochaflow.processes import (
    DiscreteGaussianDenoisingProcess,
    SelectedPairGaussianProcess,
)
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

    def transition(
        self,
        process: DiscreteGaussianDenoisingProcess,
        state: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        prediction: GaussianPrediction,
    ) -> GaussianTransition:
        """Build one selected-pair ``x_t -> x_s`` transition distribution."""

        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "DDIM transition requires DiscreteGaussianDenoisingProcess"
            )
        process = process_value
        state_value = cast(object, state)
        if not isinstance(state_value, torch.Tensor):
            raise TypeError("DDIM transition state must be a Tensor")
        state = state_value
        if state.ndim == 0:
            raise ValueError("DDIM transition state must include a batch dimension")
        if not torch.is_floating_point(state):
            raise TypeError("DDIM transition state must be floating-point")
        source_times = process.validate_noisy_state_times(source_times)
        if source_times.shape[0] != state.shape[0]:
            raise ValueError("DDIM source times must match the state batch")
        if source_times.device != state.device:
            raise ValueError("DDIM source times must share the state device")
        target_times_value = cast(object, target_times)
        if not isinstance(target_times_value, torch.Tensor):
            raise TypeError("DDIM target times must be a Tensor")
        target_times = target_times_value
        if target_times.ndim != 1:
            raise ValueError("DDIM target times must be a 1D tensor")
        if (
            target_times.dtype == torch.bool
            or torch.is_floating_point(target_times)
            or torch.is_complex(target_times)
        ):
            raise TypeError("DDIM target times must contain integer states")
        target_times = target_times.to(dtype=torch.long)
        if target_times.shape != source_times.shape:
            raise ValueError("DDIM target times must match source times")
        if target_times.device != state.device:
            raise ValueError("DDIM target times must share the state device")
        if torch.any(target_times < process.clean_time) or torch.any(
            target_times > process.terminal_time
        ):
            raise ValueError("DDIM target times must lie in the process time range")
        if torch.any(target_times >= source_times):
            raise ValueError("DDIM target times must be smaller than source times")
        prediction = _validate_gaussian_prediction(prediction, state=state)

        if isinstance(process, SelectedPairGaussianProcess):
            snapshot = process.marginal_coefficient_snapshot(
                source_times,
                target_times,
                state.size(),
            )
            target_scales = process.marginal_scales(
                target_times,
                state.size(),
            )
            source_alpha_bar = snapshot.source_alpha_bar.to(
                device=state.device,
                dtype=target_scales.signal.dtype,
            )
            target_alpha_bar = snapshot.target_alpha_bar.to(
                device=state.device,
                dtype=target_scales.signal.dtype,
            )
            transition_alpha = source_alpha_bar / target_alpha_bar
            signal_s = target_alpha_bar.sqrt()
            noise_s_squared = 1.0 - target_alpha_bar
            posterior_variance = (
                noise_s_squared
                / (1.0 - source_alpha_bar)
                * (1.0 - transition_alpha)
            ).clamp_min(0.0)
        else:
            source_scales = process.marginal_scales(source_times, state.size())
            target_scales = process.marginal_scales(target_times, state.size())
            signal_t, noise_t = source_scales.signal, source_scales.noise
            signal_s, noise_s = target_scales.signal, target_scales.noise
            noise_s_squared = noise_s.square()
            posterior_variance = (
                noise_s_squared
                / noise_t.square()
                * (1.0 - signal_t.square() / signal_s.square())
            ).clamp_min(0.0)
        direction_scale = (
            noise_s_squared - self.eta**2 * posterior_variance
        ).clamp_min(0.0).sqrt()
        mean = signal_s * prediction.clean + direction_scale * prediction.epsilon
        standard_deviation = self.eta * posterior_variance.sqrt()

        if self.eta == 1.0:
            adjacent = target_times == source_times - 1
            if bool(torch.any(adjacent)):
                adjacent_mask = adjacent.reshape(
                    (state.shape[0],) + (1,) * (state.ndim - 1)
                )
                posterior_mean = process.posterior_mean(
                    state,
                    source_times,
                    prediction.clean,
                )
                posterior_standard_deviation = (
                    process.posterior_standard_deviation(
                        source_times,
                        state.size(),
                    )
                )
                mean = torch.where(adjacent_mask, posterior_mean, mean)
                standard_deviation = torch.where(
                    adjacent_mask,
                    posterior_standard_deviation,
                    standard_deviation,
                )

        stochastic_mask = (target_times > process.clean_time).reshape(
            (state.shape[0],) + (1,) * (state.ndim - 1)
        )
        standard_deviation = standard_deviation * stochastic_mask
        standard_deviation = standard_deviation.to(
            device=mean.device,
            dtype=mean.dtype,
        )
        return GaussianTransition(mean, standard_deviation)

    def resolve_schedule(
        self,
        process: DiscreteGaussianDenoisingProcess,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Resolve the configured descending mathematical state schedule."""

        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "DDIM schedule requires DiscreteGaussianDenoisingProcess"
            )
        process = process_value
        if device is None:
            device = torch.device("cpu")
        schedule_device = torch.device("cpu")
        clean_time = process.clean_time
        terminal_time = process.terminal_time
        if self.explicit_schedule is not None:
            states = torch.as_tensor(
                self.explicit_schedule,
                device=schedule_device,
            )
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
                device=schedule_device,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
        if int(states[0]) > terminal_time or int(states[-1]) < clean_time:
            raise ValueError(
                "DDIM schedule states must lie in [clean_time, terminal_time]"
            )
        if not torch.all(states[:-1] > states[1:]):
            raise ValueError("DDIM schedule must be strictly descending and unique")
        return states.to(device=device)

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
        states = self.resolve_schedule(process)
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
        for step_index, (source, target) in enumerate(pairwise(states), start=1):
            source_coordinate = int(source)
            target_coordinate = int(target)
            source_times = torch.full(
                (current.shape[0],),
                source_coordinate,
                device=current.device,
                dtype=torch.long,
            )
            target_times = torch.full(
                (current.shape[0],),
                target_coordinate,
                device=current.device,
                dtype=torch.long,
            )
            prediction = dynamics.predict(current, source_times)
            evaluations += 1
            transition = self.transition(
                process,
                current,
                source_times,
                target_times,
                prediction,
            )
            current = (
                transition.mean
                if self.eta == 0.0 or target_coordinate == process.clean_time
                else transition.sample(generator=generator)
            )
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=target_coordinate,
                        state=current,
                        is_final=step_index == num_steps,
                        diagnostics={"num_dynamics_evaluations": evaluations},
                    )
                )
        return SamplerResult(
            current,
            num_steps,
            {"num_dynamics_evaluations": evaluations},
        )


__all__ = ["DDIMSampler"]
