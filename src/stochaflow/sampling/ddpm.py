"""Ancestral DDPM sampler."""

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
    CleanTargetVarianceReferenceGaussianDenoisingDynamics,
    GaussianDenoisingDynamics,
    GaussianPrediction,
    GaussianTransition,
    LearnedVarianceGaussianPrediction,
    TargetAwareGaussianDenoisingDynamics,
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
    """Run full or uniformly respaced ancestral Gaussian transitions."""

    def __init__(
        self,
        *,
        start_time: object = None,
        end_time: object = 0,
        num_inference_steps: object = None,
        schedule: object = None,
    ) -> None:
        if start_time is not None and (
            isinstance(start_time, bool) or not isinstance(start_time, int)
        ):
            raise TypeError("DDPM start_time must be an integer or null")
        if isinstance(end_time, bool) or not isinstance(end_time, int):
            raise TypeError("DDPM end_time must be an integer")
        if num_inference_steps is not None and schedule is not None:
            raise ValueError("num_inference_steps and schedule are mutually exclusive")
        if num_inference_steps is not None and (
            isinstance(num_inference_steps, bool)
            or not isinstance(num_inference_steps, int)
            or num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        if schedule is not None:
            if isinstance(schedule, (str, bytes)) or not isinstance(
                schedule, Sequence
            ):
                raise TypeError("DDPM schedule must be a sequence of integers")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in schedule
            ):
                raise TypeError("DDPM schedule must contain integer states")
        if (num_inference_steps is not None or schedule is not None) and (
            start_time is not None or end_time != 0
        ):
            raise ValueError(
                "DDPM start_time/end_time cannot be combined with a respaced schedule"
            )
        self.start_time: int | None = start_time
        self.end_time: int = end_time
        self.num_inference_steps: int | None = num_inference_steps
        self.explicit_schedule: tuple[int, ...] | None = (
            tuple(schedule) if schedule is not None else None
        )

    def transition(
        self,
        process: DiscreteGaussianDenoisingProcess,
        state: torch.Tensor,
        source_times: torch.Tensor,
        prediction: GaussianPrediction,
        *,
        target_times: torch.Tensor | None = None,
    ) -> GaussianTransition:
        """Build one selected-pair ancestral transition distribution."""

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
        source_times = process.validate_noisy_state_times(source_times)
        if source_times.shape[0] != state.shape[0]:
            raise ValueError("DDPM source times must match the state batch")
        if source_times.device != state.device:
            raise ValueError("DDPM source times must share the state device")
        if target_times is None:
            target_times = source_times - 1
        target_times = self._validate_target_times(
            process,
            source_times,
            target_times,
            state=state,
        )
        prediction = _validate_gaussian_prediction(prediction, state=state)
        adjacent = target_times == source_times - 1
        if isinstance(process, SelectedPairGaussianProcess):
            snapshot = process.marginal_coefficient_snapshot(
                source_times,
                target_times,
                state.size(),
            )
            denominator = 1.0 - snapshot.source_alpha_bar
            transition_variance = 1.0 - snapshot.transition_alpha
            posterior_variance = (
                transition_variance
                * (1.0 - snapshot.target_alpha_bar)
                / denominator
            ).clamp_min(0.0)
            runtime_dtype = process.marginal_scales(
                source_times,
                state.size(),
            ).signal.dtype
            clean_coefficient = (
                snapshot.target_alpha_bar.sqrt()
                * transition_variance
                / denominator
            ).to(device=state.device, dtype=runtime_dtype)
            state_coefficient = (
                snapshot.transition_alpha.sqrt()
                * (1.0 - snapshot.target_alpha_bar)
                / denominator
            ).to(device=state.device, dtype=runtime_dtype)
            mean = (
                clean_coefficient * prediction.clean
                + state_coefficient * state
            )
            standard_deviation = posterior_variance.sqrt().to(
                device=state.device,
                dtype=runtime_dtype,
            )
            if bool(torch.any(adjacent)):
                adjacent_mask = adjacent.reshape(
                    (state.shape[0],) + (1,) * (state.ndim - 1)
                )
                adjacent_mean = process.posterior_mean(
                    state,
                    source_times,
                    prediction.clean,
                )
                adjacent_standard_deviation = (
                    process.posterior_standard_deviation(
                        source_times,
                        state.size(),
                    )
                )
                mean = torch.where(adjacent_mask, adjacent_mean, mean)
                standard_deviation = torch.where(
                    adjacent_mask,
                    adjacent_standard_deviation,
                    standard_deviation,
                )
        else:
            if not bool(torch.all(adjacent)):
                raise TypeError(
                    "respaced DDPM transitions require SelectedPairGaussianProcess"
                )
            mean = process.posterior_mean(
                state,
                source_times,
                prediction.clean,
            )
            standard_deviation = process.posterior_standard_deviation(
                source_times,
                state.size(),
            )
        if isinstance(prediction, LearnedVarianceGaussianPrediction):
            standard_deviation = (0.5 * prediction.log_variance).exp()
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
        """Resolve a descending public-state ancestral schedule."""

        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "DDPM schedule requires DiscreteGaussianDenoisingProcess"
            )
        process = process_value
        if device is None:
            device = torch.device("cpu")
        if self.explicit_schedule is not None:
            states = torch.as_tensor(
                self.explicit_schedule,
                device=torch.device("cpu"),
            )
            if states.ndim != 1 or states.numel() < 2:
                raise ValueError("DDPM schedule must contain at least two states")
            states = states.to(dtype=torch.long)
        elif self.num_inference_steps is not None:
            transitions = process.terminal_time - process.clean_time
            if self.num_inference_steps > transitions:
                raise ValueError(
                    "num_inference_steps must not exceed process transitions"
                )
            model_times = self._uniform_section_model_times(
                transitions,
                self.num_inference_steps,
            )
            states = torch.tensor(
                [
                    process.clean_time + time + 1
                    for time in reversed(model_times)
                ]
                + [process.clean_time],
                dtype=torch.long,
            )
        else:
            start = (
                process.terminal_time
                if self.start_time is None
                else self.start_time
            )
            end = self.end_time
            if not process.clean_time <= end <= start <= process.terminal_time:
                raise ValueError(
                    "DDPM requires clean_time <= end_time <= start_time "
                    "<= terminal_time"
                )
            states = torch.arange(start, end - 1, -1, dtype=torch.long)
        if int(states[0]) > process.terminal_time or int(
            states[-1]
        ) < process.clean_time:
            raise ValueError(
                "DDPM schedule states must lie in [clean_time, terminal_time]"
            )
        if states.numel() > 1 and not torch.all(states[:-1] > states[1:]):
            raise ValueError("DDPM schedule must be strictly descending and unique")
        return states.to(device=device)

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
        states = self.resolve_schedule(process)
        state = initial_state
        num_steps = states.numel() - 1
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    step_index=0,
                    coordinate=int(states[0]),
                    state=state,
                    is_final=num_steps == 0,
                    diagnostics={},
                )
            )
        evaluations = 0
        for step_index, (source, target) in enumerate(
            pairwise(states),
            start=1,
        ):
            source_coordinate = int(source)
            target_coordinate = int(target)
            source_times = torch.full(
                (state.shape[0],),
                source_coordinate,
                device=state.device,
                dtype=torch.long,
            )
            target_times = torch.full(
                (state.shape[0],),
                target_coordinate,
                device=state.device,
                dtype=torch.long,
            )
            if isinstance(
                dynamics,
                CleanTargetVarianceReferenceGaussianDenoisingDynamics,
            ):
                clean_target_reference_times = None
                if (
                    target_coordinate == process.clean_time
                    and step_index > 1
                ):
                    reference_source = int(states[step_index - 2])
                    reference_target = int(states[step_index - 1])
                    clean_target_reference_times = (
                        torch.full(
                            (state.shape[0],),
                            reference_source,
                            device=state.device,
                            dtype=torch.long,
                        ),
                        torch.full(
                            (state.shape[0],),
                            reference_target,
                            device=state.device,
                            dtype=torch.long,
                        ),
                    )
                prediction = dynamics.predict_transition_with_variance_reference(
                    state,
                    source_times,
                    target_times,
                    clean_target_reference_times=(
                        clean_target_reference_times
                    ),
                )
            elif isinstance(dynamics, TargetAwareGaussianDenoisingDynamics):
                prediction = dynamics.predict_transition(
                    state,
                    source_times,
                    target_times,
                )
            else:
                prediction = dynamics.predict(state, source_times)
            evaluations += 1
            if target_coordinate == source_coordinate - 1:
                transition = self.transition(
                    process,
                    state,
                    source_times,
                    prediction,
                )
            else:
                transition = self.transition(
                    process,
                    state,
                    source_times,
                    prediction,
                    target_times=target_times,
                )
            state = (
                transition.mean
                if target_coordinate == process.clean_time
                else transition.sample(generator=generator)
            )
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=target_coordinate,
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

    @staticmethod
    def _validate_target_times(
        process: DiscreteGaussianDenoisingProcess,
        source_times: torch.Tensor,
        target_times: object,
        *,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(target_times, torch.Tensor):
            raise TypeError("DDPM target times must be a Tensor")
        if target_times.ndim != 1:
            raise ValueError("DDPM target times must be a 1D tensor")
        if (
            target_times.dtype == torch.bool
            or torch.is_floating_point(target_times)
            or torch.is_complex(target_times)
        ):
            raise TypeError("DDPM target times must contain integer states")
        normalized = target_times.to(dtype=torch.long)
        if normalized.shape != source_times.shape:
            raise ValueError("DDPM target times must match source times")
        if normalized.device != state.device:
            raise ValueError("DDPM target times must share the state device")
        if torch.any(normalized < process.clean_time) or torch.any(
            normalized > process.terminal_time
        ):
            raise ValueError("DDPM target times must lie in the process time range")
        if torch.any(normalized >= source_times):
            raise ValueError("DDPM target times must be smaller than source times")
        return normalized

    @staticmethod
    def _uniform_section_model_times(
        num_timesteps: int,
        count: int,
    ) -> tuple[int, ...]:
        if count == 1:
            return (0,)
        fractional_stride = (num_timesteps - 1) / (count - 1)
        current = 0.0
        selected: list[int] = []
        for _ in range(count):
            selected.append(round(current))
            current += fractional_stride
        return tuple(selected)


__all__ = ["DDPMAncestralSampler"]
