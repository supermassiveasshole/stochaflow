"""Tests for the discrete Gaussian process and ancestral sampler."""

from typing import Any, cast

import pytest
import torch

from stochaflow.processes import (
    DiscreteGaussianDenoisingProcess,
    DiscreteGaussianProcess,
    GaussianScales,
)
from stochaflow.sampling import (
    GenerativeDynamics,
    Sampler,
    SamplerResult,
    SamplingObservation,
    TrajectoryObserver,
)
from stochaflow.sampling.gaussian import (
    DDIMSampler,
    DDPMAncestralSampler,
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
    GaussianPrediction,
    GaussianTransition,
    PredictionType,
    normalize_gaussian_prediction,
)


def _process(steps: int = 8) -> DiscreteGaussianProcess:
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


def test_process_marginal_and_posterior_use_public_state_time() -> None:
    process = _process(4)
    clean = torch.randn(3, 2)
    noise = torch.randn_like(clean)
    times = torch.tensor([0, 1, 4])

    noisy, returned_noise = process.sample_marginal(clean, times, noise=noise)

    assert torch.equal(returned_noise, noise)
    assert torch.equal(noisy[0], clean[0])
    scales = process.marginal_scales(times, clean.size())
    assert torch.allclose(noisy, scales.signal * clean + scales.noise * noise)
    assert not hasattr(process, "denoising_dynamics")


def test_gaussian_model_dynamics_owns_prediction_configuration() -> None:
    process = _process()
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        prediction_type="v",
        clip_denoised=False,
    )

    assert dynamics.process is process
    assert dynamics.prediction_type == "v"
    assert dynamics.clip_denoised is False
    assert not hasattr(process, "convert_prediction")


def test_gaussian_model_dynamics_validates_constructor_contract() -> None:
    process = _process()

    def predict(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(state)

    with pytest.raises(ValueError, match="prediction_type"):
        GaussianModelDynamics(
            process,
            predict,
            prediction_type=cast(Any, "invalid"),
        )
    with pytest.raises(TypeError, match="clip_denoised"):
        GaussianModelDynamics(
            process,
            predict,
            clip_denoised=cast(Any, "true"),
        )
    with pytest.raises(TypeError, match="predict_fn"):
        GaussianModelDynamics(process, cast(Any, object()))
    with pytest.raises(TypeError, match="DiscreteGaussianDenoisingProcess"):
        GaussianModelDynamics(cast(Any, object()), predict)


@pytest.mark.parametrize(
    ("model_output", "error", "message"),
    [
        (object(), TypeError, "must return a Tensor"),
        (torch.zeros(1), ValueError, "must match the state shape"),
    ],
)
def test_gaussian_model_dynamics_validates_untrusted_model_output(
    model_output: object,
    error: type[Exception],
    message: str,
) -> None:
    process = _process()
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: model_output,
    )

    with pytest.raises(error, match=message):
        dynamics.predict(torch.randn(2, 3), torch.tensor([1, 2]))


def test_prediction_guided_dynamics_wrapper_reuses_builtin_samplers() -> None:
    process = _process(2)

    class PhysicsGuidedDynamics(GaussianDenoisingDynamics):
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = GaussianModelDynamics(
                process,
                lambda state, time: torch.zeros_like(state),
                clip_denoised=False,
            )

        @property
        def process(self) -> DiscreteGaussianProcess:
            return process

        def predict(
            self,
            state: torch.Tensor,
            state_times: torch.Tensor,
        ) -> GaussianPrediction:
            self.calls += 1
            prediction = self.delegate.predict(state, state_times)
            corrected_clean = prediction.clean - 0.01 * state.tanh()
            scales = process.marginal_scales(state_times, state.size())
            corrected_epsilon = (
                state - scales.signal * corrected_clean
            ) / scales.noise
            return GaussianPrediction(
                corrected_clean,
                corrected_epsilon,
                prediction.model_output,
            )

    initial = torch.randn(1, 3)
    ddpm_dynamics = PhysicsGuidedDynamics()
    ddim_dynamics = PhysicsGuidedDynamics()
    ddpm = DDPMAncestralSampler().sample(ddpm_dynamics, initial)
    ddim = DDIMSampler(num_inference_steps=2).sample(ddim_dynamics, initial)

    assert ddpm.final_state.shape == initial.shape
    assert ddim.final_state.shape == initial.shape
    assert ddpm.num_steps == 2
    assert ddim.num_steps == 2
    assert ddpm_dynamics.calls == 2
    assert ddim_dynamics.calls == 2


def test_one_custom_gaussian_process_interface_reuses_builtin_samplers() -> None:
    class DelegatingGaussianProcess(DiscreteGaussianDenoisingProcess):
        def __init__(self) -> None:
            super().__init__()
            self.delegate = _process(4)

        @property
        def clean_time(self) -> int:
            return self.delegate.clean_time

        @property
        def terminal_time(self) -> int:
            return self.delegate.terminal_time

        def sample_terminal_prior(self, shape, *, device, dtype=torch.float32, generator=None):
            return self.delegate.sample_terminal_prior(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )

        def sample_marginal(self, clean, state_times, *, noise=None, generator=None):
            return self.delegate.sample_marginal(
                clean,
                state_times,
                noise=noise,
                generator=generator,
            )

        def marginal_scales(self, state_times, broadcast_shape):
            return self.delegate.marginal_scales(state_times, broadcast_shape)

        def validate_noisy_state_times(self, state_times):
            return self.delegate.validate_noisy_state_times(state_times)

        def posterior_mean(self, state, state_times, clean_prediction):
            return self.delegate.posterior_mean(
                state,
                state_times,
                clean_prediction,
            )

        def posterior_standard_deviation(self, state_times, broadcast_shape):
            return self.delegate.posterior_standard_deviation(
                state_times,
                broadcast_shape,
            )

    process = DelegatingGaussianProcess()
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        clip_denoised=False,
    )
    initial = torch.randn(2, 3)

    ddpm = DDPMAncestralSampler().sample(
        dynamics,
        initial,
        generator=torch.Generator().manual_seed(5),
    )
    ddim = DDIMSampler(num_inference_steps=2).sample(dynamics, initial)

    assert ddpm.final_state.shape == initial.shape
    assert ddim.final_state.shape == initial.shape


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_prediction_parameterizations_convert_to_same_clean_and_epsilon(
    prediction_type: PredictionType,
) -> None:
    process = _process()
    clean = torch.empty(2, 3).uniform_(-0.8, 0.8)
    epsilon = torch.randn_like(clean)
    times = torch.tensor([2, 7])
    scales = process.marginal_scales(times, clean.size())
    state = scales.signal * clean + scales.noise * epsilon
    outputs = {
        "epsilon": epsilon,
        "x0": clean,
        "v": scales.signal * epsilon - scales.noise * clean,
        "score": -epsilon / scales.noise,
    }

    prediction = GaussianModelDynamics(
        process,
        lambda _state, _time: outputs[prediction_type],
        prediction_type=prediction_type,
        clip_denoised=False,
    ).predict(state, times)

    assert torch.allclose(prediction.clean, clean, atol=1e-5)
    assert torch.allclose(prediction.epsilon, epsilon, atol=1e-5)


def test_v_prediction_conversion_does_not_assume_unit_normalized_scales() -> None:
    class NonNormalizedGaussianProcess(DiscreteGaussianDenoisingProcess):
        @property
        def clean_time(self) -> int:
            return 0

        @property
        def terminal_time(self) -> int:
            return 1

        def sample_terminal_prior(
            self,
            shape,
            *,
            device,
            dtype=torch.float32,
            generator=None,
        ):
            return torch.randn(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )

        def sample_marginal(
            self,
            clean,
            state_times,
            *,
            noise=None,
            generator=None,
        ):
            if noise is None:
                noise = torch.randn(
                    clean.shape,
                    device=clean.device,
                    dtype=clean.dtype,
                    generator=generator,
                )
            scales = self.marginal_scales(state_times, clean.size())
            return scales.signal * clean + scales.noise * noise, noise

        def marginal_scales(self, state_times, broadcast_shape):
            self.validate_noisy_state_times(state_times)
            shape = (state_times.shape[0],) + (1,) * (len(broadcast_shape) - 1)
            return GaussianScales(
                torch.full(shape, 2.0, device=state_times.device),
                torch.full(shape, 3.0, device=state_times.device),
            )

        def validate_noisy_state_times(self, state_times):
            if (
                state_times.ndim != 1
                or state_times.dtype != torch.long
                or bool(torch.any(state_times != 1))
            ):
                raise ValueError("state_times must contain only terminal time 1")
            return state_times

        def posterior_mean(
            self,
            state,
            state_times,
            clean_prediction,
        ):
            self.validate_noisy_state_times(state_times)
            return clean_prediction

        def posterior_standard_deviation(self, state_times, broadcast_shape):
            self.validate_noisy_state_times(state_times)
            return torch.zeros(
                (state_times.shape[0],) + (1,) * (len(broadcast_shape) - 1),
                device=state_times.device,
            )

    process = NonNormalizedGaussianProcess()
    clean = torch.randn(2, 3)
    epsilon = torch.randn_like(clean)
    times = torch.ones(2, dtype=torch.long)
    scales = process.marginal_scales(times, clean.size())
    state = scales.signal * clean + scales.noise * epsilon
    velocity = scales.signal * epsilon - scales.noise * clean

    prediction = normalize_gaussian_prediction(
        process,
        state,
        times,
        velocity,
        prediction_type="v",
        clip_denoised=False,
    )

    assert torch.allclose(prediction.clean, clean)
    assert torch.allclose(prediction.epsilon, epsilon)


def test_clipping_recomputes_epsilon_from_clipped_clean_prediction() -> None:
    process = _process()
    state = torch.randn(2, 4)
    times = torch.tensor([2, 5])
    prediction = GaussianModelDynamics(
        process,
        lambda _state, _time: torch.full_like(state, 4.0),
        prediction_type="x0",
        clip_denoised=True,
    ).predict(state, times)
    scales = process.marginal_scales(times, state.size())

    assert prediction.clean.max() == 1
    torch.testing.assert_close(
        state,
        scales.signal * prediction.clean + scales.noise * prediction.epsilon,
        rtol=1e-5,
        atol=5e-7,
    )


def test_prediction_normalization_is_reusable_without_model_adapter() -> None:
    process = _process(4)
    state = torch.randn(2, 3)
    times = torch.tensor([1, 4])
    raw_prediction = torch.randn_like(state)

    prediction = normalize_gaussian_prediction(
        process,
        state,
        times,
        raw_prediction,
        prediction_type="epsilon",
        clip_denoised=False,
    )

    scales = process.marginal_scales(times, state.size())
    assert torch.equal(prediction.epsilon, raw_prediction)
    torch.testing.assert_close(
        state,
        scales.signal * prediction.clean + scales.noise * prediction.epsilon,
        rtol=1e-5,
        atol=5e-7,
    )


def test_gaussian_prediction_rejects_non_floating_values() -> None:
    integer = torch.ones(1, dtype=torch.long)

    with pytest.raises(TypeError, match="floating-point"):
        GaussianPrediction(integer, integer, integer)


def test_prediction_normalization_rejects_non_floating_model_output() -> None:
    process = _process(2)

    with pytest.raises(TypeError, match="model output must be floating-point"):
        normalize_gaussian_prediction(
            process,
            torch.zeros(1, 2),
            torch.tensor([1]),
            torch.ones(1, 2, dtype=torch.long),
        )


def test_ddpm_public_transition_exposes_adjacent_distribution() -> None:
    process = _process(4)
    state = torch.randn(2, 3)
    times = torch.tensor([4, 1])
    prediction = normalize_gaussian_prediction(
        process,
        state,
        times,
        torch.zeros_like(state),
        clip_denoised=False,
    )

    result = DDPMAncestralSampler().transition(
        process,
        state,
        times,
        prediction,
    )

    assert isinstance(result, GaussianTransition)
    assert torch.equal(
        result.mean,
        process.posterior_mean(state, times, prediction.clean),
    )
    assert torch.count_nonzero(result.standard_deviation[0]) > 0
    assert torch.count_nonzero(result.standard_deviation[1]) == 0


def test_ddpm_complete_sample_delegates_to_public_transition() -> None:
    process = _process(3)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        clip_denoised=False,
    )

    class RecordingSampler(DDPMAncestralSampler):
        def __init__(self) -> None:
            super().__init__()
            self.source_times: list[int] = []

        def transition(
            self,
            process: DiscreteGaussianDenoisingProcess,
            state: torch.Tensor,
            state_times: torch.Tensor,
            prediction: GaussianPrediction,
        ) -> GaussianTransition:
            self.source_times.append(int(state_times[0]))
            return super().transition(process, state, state_times, prediction)

    sampler = RecordingSampler()
    sampler.sample(dynamics, torch.randn(2, 3))

    assert sampler.source_times == [3, 2, 1]


def test_ddpm_unified_sample_emits_initial_accepted_and_unique_final() -> None:
    process = _process(5)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
        clip_denoised=False,
    )
    observer = TrajectoryObserver(every_steps=1)

    initial = torch.randn(2, 3)

    result = DDPMAncestralSampler().sample(
        dynamics,
        initial,
        generator=torch.Generator().manual_seed(4),
        observer=observer,
    )

    assert result.final_state.shape == initial.shape
    assert result.num_steps == 5
    assert result.diagnostics == {"num_dynamics_evaluations": 5}
    assert [item.step_index for item in observer.observations] == list(range(6))
    assert [item.coordinate for item in observer.observations] == [5, 4, 3, 2, 1, 0]
    assert [item.is_final for item in observer.observations].count(True) == 1
    assert observer.observations[-1].is_final


def test_trajectory_observer_copies_state_at_observation_time() -> None:
    observer = TrajectoryObserver()
    state = torch.zeros(1, requires_grad=True)

    observer.observe(SamplingObservation(0, 1, state, False, {}))
    with torch.no_grad():
        state.add_(1)
    observer.observe(SamplingObservation(1, 0, state, True, {}))

    assert [item.state.item() for item in observer.observations] == [0.0, 1.0]
    assert observer.observations[0].state is not state
    assert not observer.observations[0].state.requires_grad


def test_trajectory_observer_copies_structured_nonleaf_tensors() -> None:
    observer = TrajectoryObserver()
    state = {"primary": torch.ones(1, requires_grad=True) * 2, "history": []}

    observer.observe(SamplingObservation(0, 1, state, True, {}))
    with torch.no_grad():
        state["primary"].add_(3)

    retained = observer.observations[0].state
    assert retained["primary"].item() == 2
    assert retained["primary"].is_leaf
    assert not retained["primary"].requires_grad


def test_observer_exception_propagates_from_sampler() -> None:
    process = _process(2)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state),
    )

    class FailingObserver:
        def observe(self, observation: SamplingObservation) -> None:
            del observation
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        DDPMAncestralSampler().sample(
            dynamics,
            torch.randn(1, 2),
            observer=FailingObserver(),
        )


def test_ddpm_final_clean_transition_does_not_consume_rng() -> None:
    process = _process(5)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state), clip_denoised=False
    )
    initial = torch.randn(2, 3)
    actual_generator = torch.Generator().manual_seed(13)
    expected_generator = torch.Generator().manual_seed(13)

    DDPMAncestralSampler().sample(
        dynamics,
        initial,
        generator=actual_generator,
    )
    for _ in range(process.num_timesteps - 1):
        torch.randn(initial.shape, generator=expected_generator)

    assert torch.equal(
        torch.randn(4, generator=actual_generator),
        torch.randn(4, generator=expected_generator),
    )


def test_ddpm_partial_path_and_generator_are_deterministic() -> None:
    process = _process(8)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, time: torch.zeros_like(state), clip_denoised=False
    )
    initial = torch.randn(2, 3)
    sampler = DDPMAncestralSampler(start_time=6, end_time=2)

    first = sampler.sample(
        dynamics, initial, generator=torch.Generator().manual_seed(19)
    )
    second = sampler.sample(
        dynamics, initial, generator=torch.Generator().manual_seed(19)
    )

    assert first.num_steps == 4
    assert torch.equal(first.final_state, second.final_state)


def test_sampler_rejects_incompatible_dynamics() -> None:
    with pytest.raises(TypeError, match="GaussianDenoisingDynamics"):
        DDPMAncestralSampler().sample(GenerativeDynamics(), torch.randn(1, 2))


def test_unified_sampler_allows_multiple_dynamics_evaluations_per_outer_step() -> None:
    class CountingDynamics(GenerativeDynamics):
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, state: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return state * 0

    class HeunLikeSampler(Sampler):
        def sample(self, dynamics, initial_state, **kwargs) -> SamplerResult:
            del kwargs
            assert isinstance(dynamics, CountingDynamics)
            state = initial_state
            for _ in range(3):
                first = dynamics.evaluate(state)
                second = dynamics.evaluate(state + first)
                state = state + (first + second) * 0.5
            return SamplerResult(
                state,
                3,
                {"num_dynamics_evaluations": dynamics.calls},
            )

    dynamics = CountingDynamics()
    sampler = HeunLikeSampler()
    result = sampler.sample(dynamics, torch.ones(2))

    assert result.num_steps == 3
    assert result.diagnostics["num_dynamics_evaluations"] == 6
    assert not hasattr(sampler, "step")


def test_unified_sampler_can_own_history_and_rejected_internal_attempts() -> None:
    class HistorySampler(Sampler):
        def sample(self, dynamics, initial_state, **kwargs) -> SamplerResult:
            del dynamics, kwargs
            history = [initial_state]
            rejected = 0
            for attempt in range(3):
                candidate = history[-1] + 1
                if attempt == 1:
                    rejected += 1
                    continue
                history.append(candidate)
            return SamplerResult(
                history[-1],
                len(history) - 1,
                {
                    "history_length": len(history),
                    "num_rejected_steps": rejected,
                },
            )

    sampler = HistorySampler()
    result = sampler.sample(GenerativeDynamics(), torch.zeros(1))

    assert result.num_steps == 2
    assert result.diagnostics == {
        "history_length": 3,
        "num_rejected_steps": 1,
    }
    assert torch.equal(result.final_state, torch.tensor([2.0]))
    assert not hasattr(sampler, "step")
