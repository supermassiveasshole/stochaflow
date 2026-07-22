"""Tests for DDIM on the unified complete-sampler interface."""

from typing import Any

import pytest
import torch

from stochaflow.processes import (
    DiscreteGaussianDenoisingProcess,
    DiscreteGaussianProcess,
    GaussianScales,
)
from stochaflow.sampling import (
    DDIMSampler,
    DDPMAncestralSampler,
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
    GaussianPrediction,
    GaussianTransition,
    GenerativeDynamics,
    Sampler,
    SamplerResult,
    SamplingObservation,
    SamplingObserver,
    TrajectoryObserver,
    normalize_gaussian_prediction,
)


def _process(steps: int = 10) -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": steps,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
            },
        }
    )


def test_ddim_supports_nonuniform_explicit_schedule() -> None:
    process = _process(10)
    model_times: list[torch.Tensor] = []

    def predict(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        model_times.append(time.detach().clone())
        return torch.zeros_like(state)

    result = DDIMSampler(schedule=[10, 8, 3, 0]).sample(
        GaussianModelDynamics(process, predict, clip_denoised=False),
        torch.randn(2, 4),
    )

    assert result.num_steps == 3
    assert [int(time[0]) for time in model_times] == [9, 7, 2]


def test_ddim_public_schedule_resolver_supports_uniform_and_partial_paths() -> None:
    process = _process(10)

    assert DDIMSampler(num_inference_steps=4).resolve_schedule(process).tolist() == [
        10,
        8,
        5,
        2,
        0,
    ]
    assert DDIMSampler(schedule=[7, 4, 2]).resolve_schedule(process).tolist() == [
        7,
        4,
        2,
    ]


def test_ddim_public_transition_supports_batch_selected_pairs() -> None:
    process = _process(10)
    state = torch.randn(2, 3)
    source_times = torch.tensor([10, 7])
    target_times = torch.tensor([3, 0])
    prediction = normalize_gaussian_prediction(
        process,
        state,
        source_times,
        torch.randn_like(state),
        clip_denoised=False,
    )

    deterministic = DDIMSampler(eta=0).transition(
        process,
        state,
        source_times,
        target_times,
        prediction,
    )
    target_scales = process.marginal_scales(target_times, state.size())
    expected = (
        target_scales.signal * prediction.clean
        + target_scales.noise * prediction.epsilon
    )
    stochastic = DDIMSampler(eta=0.5).transition(
        process,
        state,
        source_times,
        target_times,
        prediction,
    )

    assert isinstance(deterministic, GaussianTransition)
    assert torch.allclose(deterministic.mean, expected)
    assert torch.count_nonzero(deterministic.standard_deviation) == 0
    assert torch.count_nonzero(stochastic.standard_deviation[0]) > 0
    assert torch.count_nonzero(stochastic.standard_deviation[1]) == 0


def test_adjacent_eta_one_public_transition_matches_ddpm_distribution() -> None:
    process = _process(5)
    state = torch.randn(2, 3)
    source_times = torch.tensor([5, 2])
    target_times = source_times - 1
    prediction = normalize_gaussian_prediction(
        process,
        state,
        source_times,
        torch.zeros_like(state),
        clip_denoised=False,
    )

    ddpm = DDPMAncestralSampler().transition(
        process,
        state,
        source_times,
        prediction,
    )
    ddim = DDIMSampler(eta=1).transition(
        process,
        state,
        source_times,
        target_times,
        prediction,
    )

    assert torch.allclose(ddim.mean, ddpm.mean)
    assert torch.allclose(ddim.standard_deviation, ddpm.standard_deviation)


def test_gaussian_transitions_materialize_statistics_in_state_precision() -> None:
    process = _process(5)
    state = torch.randn(2, 3, dtype=torch.float64)
    source_times = torch.tensor([5, 3])
    prediction = normalize_gaussian_prediction(
        process,
        state,
        source_times,
        torch.zeros_like(state),
        clip_denoised=False,
    )

    ddpm = DDPMAncestralSampler().transition(
        process,
        state,
        source_times,
        prediction,
    )
    ddim = DDIMSampler(eta=0.5).transition(
        process,
        state,
        source_times,
        torch.tensor([2, 0]),
        prediction,
    )

    assert ddpm.mean.dtype == torch.float64
    assert ddpm.standard_deviation.dtype == torch.float64
    assert ddpm.sample(generator=torch.Generator().manual_seed(3)).dtype == torch.float64
    assert ddim.mean.dtype == torch.float64
    assert ddim.standard_deviation.dtype == torch.float64
    assert ddim.sample(generator=torch.Generator().manual_seed(3)).dtype == torch.float64


def test_ddim_complete_sample_delegates_to_public_primitives() -> None:
    process = _process(4)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        clip_denoised=False,
    )

    class RecordingSampler(DDIMSampler):
        def __init__(self) -> None:
            super().__init__(schedule=[4, 2, 0])
            self.schedule_calls = 0
            self.transitions: list[tuple[int, int]] = []

        def resolve_schedule(
            self,
            process: DiscreteGaussianDenoisingProcess,
            *,
            device: torch.device | None = None,
        ) -> torch.Tensor:
            self.schedule_calls += 1
            return super().resolve_schedule(process, device=device)

        def transition(
            self,
            process: DiscreteGaussianDenoisingProcess,
            state: torch.Tensor,
            source_times: torch.Tensor,
            target_times: torch.Tensor,
            prediction: GaussianPrediction,
        ) -> GaussianTransition:
            self.transitions.append(
                (int(source_times[0]), int(target_times[0]))
            )
            return super().transition(
                process,
                state,
                source_times,
                target_times,
                prediction,
            )

    sampler = RecordingSampler()
    sampler.sample(dynamics, torch.randn(2, 3))

    assert sampler.schedule_calls == 1
    assert sampler.transitions == [(4, 2), (2, 0)]


def test_ddim_explicit_schedule_supports_partial_denoising() -> None:
    process = _process(10)
    observer = TrajectoryObserver()

    result = DDIMSampler(schedule=[7, 4, 2]).sample(
        GaussianModelDynamics(
            process,
            lambda state, time: torch.zeros_like(state),
            clip_denoised=False,
        ),
        torch.randn(2, 4),
        observer=observer,
    )

    assert result.num_steps == 2
    assert [item.coordinate for item in observer.observations] == [7, 4, 2]


@pytest.mark.parametrize(
    "schedule",
    [[10, 4, 4, 0], [10, 11, 0], [7, 4, -1], [3, 4, 0]],
)
def test_ddim_rejects_invalid_explicit_schedules(schedule: list[int]) -> None:
    process = _process(10)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state)
    )
    with pytest.raises(ValueError):
        DDIMSampler(schedule=schedule).sample(dynamics, torch.randn(1, 2))


def test_eta_zero_does_not_consume_transition_generator() -> None:
    process = _process(6)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state), clip_denoised=False
    )
    generator = torch.Generator().manual_seed(27)
    expected = torch.Generator().manual_seed(27)

    DDIMSampler(num_inference_steps=3, eta=0).sample(
        dynamics, torch.randn(2, 3), generator=generator
    )

    assert torch.equal(torch.randn(4, generator=generator), torch.randn(4, generator=expected))


def test_eta_positive_clean_transition_does_not_consume_generator() -> None:
    process = _process(6)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        clip_denoised=False,
    )
    generator = torch.Generator().manual_seed(31)
    expected = torch.Generator().manual_seed(31)

    DDIMSampler(schedule=[6, 0], eta=0.5).sample(
        dynamics,
        torch.randn(2, 3),
        generator=generator,
    )

    assert torch.equal(
        torch.randn(4, generator=generator),
        torch.randn(4, generator=expected),
    )


def test_eta_positive_zero_variance_transition_does_not_consume_generator() -> None:
    class PlateauProcess(DiscreteGaussianProcess):
        def marginal_scales(
            self,
            state_times: torch.Tensor,
            broadcast_shape: torch.Size,
        ) -> GaussianScales:
            shape = (state_times.shape[0],) + (1,) * (len(broadcast_shape) - 1)
            return GaussianScales(
                torch.full(shape, 0.8, device=state_times.device),
                torch.full(shape, 0.6, device=state_times.device),
            )

    process = PlateauProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 3,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
            },
        }
    )
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        clip_denoised=False,
    )
    generator = torch.Generator().manual_seed(37)
    expected = torch.Generator().manual_seed(37)

    DDIMSampler(schedule=[3, 2], eta=0.5).sample(
        dynamics,
        torch.randn(2, 3),
        generator=generator,
    )

    assert torch.equal(
        torch.randn(4, generator=generator),
        torch.randn(4, generator=expected),
    )


def test_adjacent_eta_one_matches_ddpm_with_shared_generator() -> None:
    process = _process(5)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state), clip_denoised=False
    )
    initial = torch.randn(2, 3)

    ddpm = DDPMAncestralSampler().sample(
        dynamics, initial, generator=torch.Generator().manual_seed(3)
    )
    ddim = DDIMSampler(num_inference_steps=5, eta=1).sample(
        dynamics, initial, generator=torch.Generator().manual_seed(3)
    )

    assert torch.allclose(ddpm.final_state, ddim.final_state, atol=1e-5)


def test_condition_is_captured_by_dynamics_not_sampler_interface() -> None:
    process = _process(4)
    condition = torch.full((2, 3), 0.25)
    calls = 0

    def conditioned(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return condition.to(state) + time[:, None].to(state) * 0

    result = DDIMSampler(num_inference_steps=2).sample(
        GaussianModelDynamics(process, conditioned, clip_denoised=False),
        torch.randn(2, 3),
    )

    assert result.final_state.shape == condition.shape
    assert calls == 2


def test_trajectory_observer_does_not_change_sampling_math() -> None:
    process = _process(8)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state), clip_denoised=False
    )
    initial = torch.randn(2, 3)
    observer = TrajectoryObserver(every_steps=2)
    sampler = DDIMSampler(num_inference_steps=4, eta=0)

    with_observer = sampler.sample(dynamics, initial, observer=observer)
    without_observer = sampler.sample(dynamics, initial)

    assert torch.equal(with_observer.final_state, without_observer.final_state)
    assert [item.step_index for item in observer.observations] == [0, 2, 4]
    assert observer.observations[-1].coordinate == 0


class _PhysicsCorrectionDynamics(GaussianDenoisingDynamics):
    def __init__(
        self,
        process: DiscreteGaussianDenoisingProcess,
        *,
        correction: float,
    ) -> None:
        self._process = process
        self.correction = correction

    @property
    def process(self) -> DiscreteGaussianDenoisingProcess:
        return self._process

    def predict_with_correction(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> tuple[GaussianPrediction, torch.Tensor]:
        prediction = normalize_gaussian_prediction(
            self.process,
            state,
            state_times,
            torch.zeros_like(state),
            clip_denoised=False,
        )
        correction = torch.full_like(state, self.correction)
        return prediction, correction

    def predict(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
    ) -> GaussianPrediction:
        prediction, _ = self.predict_with_correction(state, state_times)
        return prediction


class _PhysicsCorrectedDDIMSampler(Sampler):
    def __init__(self, base: DDIMSampler) -> None:
        self.base = base

    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        if not isinstance(dynamics, _PhysicsCorrectionDynamics):
            raise TypeError("physics-corrected DDIM requires its task dynamics")
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("physics-corrected DDIM requires a Tensor state")
        states = self.base.resolve_schedule(
            dynamics.process,
            device=initial_state.device,
        )
        current = initial_state
        num_steps = states.numel() - 1
        if observer is not None:
            observer.observe(
                SamplingObservation(0, int(states[0]), current, False, {})
            )
        for step_index, (source, target) in enumerate(
            zip(states[:-1], states[1:]),
            start=1,
        ):
            source_times = source.expand(current.shape[0])
            target_times = target.expand(current.shape[0])
            prediction, correction = dynamics.predict_with_correction(
                current,
                source_times,
            )
            current = self.base.transition(
                dynamics.process,
                current,
                source_times,
                target_times,
                prediction,
            ).sample(generator=generator)
            current = current - correction
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index,
                        int(target),
                        current,
                        step_index == num_steps,
                        {"num_dynamics_evaluations": step_index},
                    )
                )
        return SamplerResult(
            current,
            num_steps,
            {"num_dynamics_evaluations": num_steps},
        )


def test_zero_correction_sampler_reuses_ddim_primitives_exactly() -> None:
    process = _process(4)
    initial = torch.randn(2, 3)
    standard = DDIMSampler(schedule=[4, 3, 1, 0], eta=0.5).sample(
        GaussianModelDynamics(
            process,
            lambda state, time: torch.zeros_like(state),
            clip_denoised=False,
        ),
        initial,
        generator=torch.Generator().manual_seed(23),
    )
    guided = _PhysicsCorrectedDDIMSampler(
        DDIMSampler(schedule=[4, 3, 1, 0], eta=0.5)
    ).sample(
        _PhysicsCorrectionDynamics(process, correction=0.0),
        initial,
        generator=torch.Generator().manual_seed(23),
    )

    assert torch.equal(guided.final_state, standard.final_state)
    assert guided.num_steps == standard.num_steps


def test_physics_correction_is_applied_after_transition_before_observation() -> None:
    process = _process(4)
    initial = torch.randn(2, 3)
    dynamics = _PhysicsCorrectionDynamics(process, correction=0.125)
    base = DDIMSampler(schedule=[4, 0], eta=0)
    source_times = torch.full((2,), 4, dtype=torch.long)
    target_times = torch.zeros(2, dtype=torch.long)
    prediction = dynamics.predict(initial, source_times)
    expected = base.transition(
        process,
        initial,
        source_times,
        target_times,
        prediction,
    ).mean - 0.125
    observer = TrajectoryObserver()

    result = _PhysicsCorrectedDDIMSampler(base).sample(
        dynamics,
        initial,
        observer=observer,
    )

    assert torch.equal(result.final_state, expected)
    assert torch.equal(observer.observations[-1].state, expected)
    assert observer.observations[-1].is_final
