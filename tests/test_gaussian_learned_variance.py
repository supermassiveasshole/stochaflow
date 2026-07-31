"""Selected-pair and learned-range Gaussian sampling tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from stochaflow.processes import (
    DiscreteGaussianProcess,
    GaussianLogVarianceBounds,
    GaussianMarginalCoefficientSnapshot,
    LearnedRangeGaussianVarianceProcess,
    SelectedPairGaussianProcess,
)
from stochaflow.sampling import (
    DDIMSampler,
    DDPMAncestralSampler,
    GaussianModelDynamics,
    GaussianPrediction,
    LearnedVarianceGaussianPrediction,
    SamplerResult,
    SamplingObservation,
    StandardDenoisingBuilder,
    TrajectoryObserver,
)

_REFERENCE = (
    Path(__file__).parent / "fixtures" / "gaussian" / "p2_reference.json"
)


def gaussian_process(steps: int = 10) -> DiscreteGaussianProcess:
    """Build one compact linear-beta process."""

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


def test_process_exposes_selected_pair_coefficients_and_learned_bounds() -> None:
    process = gaussian_process(10)
    assert isinstance(process, SelectedPairGaussianProcess)
    assert isinstance(process, LearnedRangeGaussianVarianceProcess)
    source_times = torch.tensor([10, 1])
    target_times = torch.tensor([3, 0])
    shape = torch.Size((2, 3, 4, 4))

    snapshot = process.marginal_coefficient_snapshot(
        source_times,
        target_times,
        shape,
    )
    betas = torch.linspace(1e-4, 2e-2, 10, dtype=torch.float64)
    alpha_bars = torch.cat(
        (
            torch.ones(1, dtype=torch.float64),
            torch.cumprod(1.0 - betas, dim=0),
        )
    )
    precise_source = alpha_bars[source_times].reshape(2, 1, 1, 1)
    precise_target = alpha_bars[target_times].reshape(2, 1, 1, 1)

    assert isinstance(snapshot, GaussianMarginalCoefficientSnapshot)
    assert snapshot.source_alpha_bar.shape == (2, 1, 1, 1)
    assert snapshot.source_alpha_bar.dtype == torch.float64
    assert torch.equal(snapshot.source_alpha_bar, precise_source)
    assert torch.equal(snapshot.target_alpha_bar, precise_target)
    assert torch.allclose(
        snapshot.transition_alpha,
        snapshot.source_alpha_bar / snapshot.target_alpha_bar,
    )

    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        shape,
    )
    precise_transition = precise_source / precise_target
    transition_variance = 1.0 - precise_transition
    posterior_variance = transition_variance * (
        1.0 - precise_target
    ) / (1.0 - precise_source)
    clipped_source = alpha_bars[2]
    clipped_target = alpha_bars[1]
    clipped_transition = clipped_source / clipped_target
    clipped_lower = (1.0 - clipped_transition) * (
        1.0 - clipped_target
    ) / (1.0 - clipped_source)
    posterior_variance[1] = clipped_lower

    assert isinstance(bounds, GaussianLogVarianceBounds)
    assert torch.equal(bounds.upper, transition_variance.log().float())
    assert torch.equal(bounds.lower, posterior_variance.log().float())


def test_target_aware_dynamics_maps_learned_range_endpoints() -> None:
    process = gaussian_process(4)
    state = torch.randn(2, 3, 4, 4)
    source_times = torch.tensor([4, 3])
    target_times = torch.tensor([2, 1])
    variance_values = torch.stack(
        (
            torch.full_like(state[0], -1.0),
            torch.full_like(state[1], 1.0),
        )
    )
    seen_model_times: list[torch.Tensor] = []

    def predict(current: torch.Tensor, model_times: torch.Tensor) -> torch.Tensor:
        seen_model_times.append(model_times.detach().clone())
        return torch.cat((torch.zeros_like(current), variance_values), dim=1)

    dynamics = GaussianModelDynamics(
        process,
        predict,
        variance_mode="learned_range",
        clip_denoised=False,
    )
    prediction = dynamics.predict_transition(
        state,
        source_times,
        target_times,
    )
    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        state.size(),
    )

    assert isinstance(prediction, LearnedVarianceGaussianPrediction)
    assert torch.equal(seen_model_times[0], source_times - 1)
    assert torch.allclose(
        prediction.log_variance[0],
        bounds.lower[0].expand_as(prediction.log_variance[0]),
    )
    assert torch.allclose(
        prediction.log_variance[1],
        bounds.upper[1].expand_as(prediction.log_variance[1]),
    )


def test_default_process_final_posterior_matches_float64_reference() -> None:
    fixture = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    reference = fixture["respaced_sampling"]
    process = gaussian_process(reference["num_timesteps"])
    state = torch.randn(2, 3)
    clean = torch.randn_like(state)
    prediction = GaussianPrediction(
        clean=clean,
        epsilon=torch.zeros_like(state),
        model_output=torch.zeros_like(state),
    )

    transition = DDPMAncestralSampler().transition(
        process,
        state,
        torch.ones(2, dtype=torch.long),
        prediction,
    )

    assert process.posterior_mean_coef1[0].item() == pytest.approx(
        reference["runtime_final_posterior_mean_coefficient_1"],
        abs=0.0,
    )
    assert process.posterior_mean_coef2[0].item() == pytest.approx(
        reference["runtime_final_posterior_mean_coefficient_2"],
        abs=0.0,
    )
    assert torch.equal(transition.mean, clean)


def test_respaced_clean_target_bound_uses_preceding_selected_pair() -> None:
    fixture = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    reference = fixture["respaced_sampling"]
    process = gaussian_process(reference["num_timesteps"])
    final_source, final_target = reference["final_public_pair"]
    clipping_source, clipping_target = reference[
        "clipping_reference_public_pair"
    ]

    bounds = process.reverse_log_variance_bounds(
        torch.tensor([final_source]),
        torch.tensor([final_target]),
        torch.Size((1, 3, 4, 4)),
        clean_target_reference_times=(
            torch.tensor([clipping_source]),
            torch.tensor([clipping_target]),
        ),
    )

    assert bounds.lower.item() == pytest.approx(
        reference["runtime_final_lower_log_variance"],
        rel=2e-6,
        abs=2e-6,
    )


def test_respaced_sampler_supplies_clean_target_clipping_reference() -> None:
    class RecordingVarianceProcess(DiscreteGaussianProcess):
        """Record final learned-variance clipping context from a sampler."""

        clean_target_reference: tuple[torch.Tensor, torch.Tensor] | None

        def __init__(self) -> None:
            super().__init__(
                {
                    "name": "linear_beta",
                    "params": {"num_timesteps": 10},
                }
            )
            self.clean_target_reference = None

        def reverse_log_variance_bounds(
            self,
            source_times: torch.Tensor,
            target_times: torch.Tensor,
            broadcast_shape: torch.Size,
            *,
            clean_target_reference_times: (
                tuple[torch.Tensor, torch.Tensor] | None
            ) = None,
        ) -> GaussianLogVarianceBounds:
            if bool(torch.any(target_times == self.clean_time)):
                self.clean_target_reference = clean_target_reference_times
            return super().reverse_log_variance_bounds(
                source_times,
                target_times,
                broadcast_shape,
                clean_target_reference_times=clean_target_reference_times,
            )

    process = RecordingVarianceProcess()
    dynamics = GaussianModelDynamics(
        process,
        lambda state, model_times: torch.cat(
            (torch.zeros_like(state), torch.zeros_like(state)),
            dim=1,
        ),
        variance_mode="learned_range",
        clip_denoised=False,
    )

    DDPMAncestralSampler(num_inference_steps=4).sample(
        dynamics,
        torch.zeros(1, 2),
        generator=torch.Generator().manual_seed(3),
    )

    assert process.clean_target_reference is not None
    reference_source, reference_target = process.clean_target_reference
    assert reference_source.tolist() == [4]
    assert reference_target.tolist() == [1]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_learned_range_sampling_accepts_mixed_process_and_state_dtype(
    dtype: torch.dtype,
) -> None:
    process = gaussian_process(2)
    initial = torch.randn(2, 3, dtype=dtype)
    fixed = GaussianModelDynamics(
        process,
        lambda state, model_times: torch.zeros_like(state),
        clip_denoised=False,
    )
    learned = GaussianModelDynamics(
        process,
        lambda state, model_times: torch.cat(
            (torch.zeros_like(state), torch.zeros_like(state)),
            dim=1,
        ),
        variance_mode="learned_range",
        clip_denoised=False,
    )
    sampler = DDPMAncestralSampler(schedule=[1, 0])

    fixed_result = sampler.sample(fixed, initial)
    learned_result = sampler.sample(learned, initial)

    assert learned_result.final_state.dtype == fixed_result.final_state.dtype
    assert torch.all(torch.isfinite(learned_result.final_state))


def test_learned_range_dynamics_rejects_wrong_output_layout() -> None:
    process = gaussian_process(2)
    dynamics = GaussianModelDynamics(
        process,
        lambda state, model_times: torch.zeros_like(state),
        variance_mode="learned_range",
    )

    with pytest.raises(ValueError, match="must have shape"):
        dynamics.predict_transition(
            torch.randn(2, 3, 4, 4),
            torch.tensor([2, 2]),
            torch.tensor([1, 1]),
        )


def test_ddpm_resolves_improved_diffusion_uniform_section_schedule() -> None:
    process = gaussian_process(10)

    assert DDPMAncestralSampler().resolve_schedule(process).tolist() == list(
        range(10, -1, -1)
    )
    assert DDPMAncestralSampler(
        num_inference_steps=4
    ).resolve_schedule(process).tolist() == [10, 7, 4, 1, 0]
    assert DDPMAncestralSampler(
        num_inference_steps=1
    ).resolve_schedule(process).tolist() == [1, 0]
    assert DDPMAncestralSampler(
        schedule=[10, 8, 3, 0]
    ).resolve_schedule(process).tolist() == [10, 8, 3, 0]

    thousand_step_process = gaussian_process(1000)
    schedule = DDPMAncestralSampler(
        num_inference_steps=250
    ).resolve_schedule(thousand_step_process)
    assert schedule.numel() == 251
    assert schedule[:8].tolist() == [1000, 996, 992, 988, 984, 980, 976, 972]
    assert schedule[-8:].tolist() == [25, 21, 17, 13, 9, 5, 1, 0]

    with pytest.raises(ValueError, match="mutually exclusive"):
        DDPMAncestralSampler(
            num_inference_steps=4,
            schedule=[10, 7, 4, 1, 0],
        )
    with pytest.raises(ValueError, match="strictly descending"):
        DDPMAncestralSampler(schedule=[10, 7, 7, 0]).resolve_schedule(process)


def test_ddpm_uniform_schedule_honors_nonzero_clean_time() -> None:
    class OffsetScheduleGaussianProcess(DiscreteGaussianProcess):
        """Expose a shifted public time domain for schedule resolution."""

        @property
        def clean_time(self) -> int:
            return 5

        @property
        def terminal_time(self) -> int:
            return 15

    process = OffsetScheduleGaussianProcess(
        {
            "name": "linear_beta",
            "params": {"num_timesteps": 10},
        }
    )

    assert DDPMAncestralSampler(
        num_inference_steps=4
    ).resolve_schedule(process).tolist() == [15, 12, 9, 6, 5]


def test_respaced_ddpm_preserves_model_times_observers_and_nfe() -> None:
    process = gaussian_process(10)
    model_times: list[int] = []

    def predict(state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        model_times.append(int(times[0]))
        return torch.zeros_like(state)

    observer = TrajectoryObserver()
    result = DDPMAncestralSampler(num_inference_steps=4).sample(
        GaussianModelDynamics(process, predict, clip_denoised=False),
        torch.randn(2, 3),
        generator=torch.Generator().manual_seed(7),
        observer=observer,
    )

    assert result.num_steps == 4
    assert result.diagnostics == {"num_dynamics_evaluations": 4}
    assert model_times == [9, 6, 3, 0]
    assert [item.coordinate for item in observer.observations] == [10, 7, 4, 1, 0]


def test_ddpm_selected_pair_posterior_matches_closed_form() -> None:
    process = gaussian_process(10)
    state = torch.randn(2, 3, 4, 4)
    clean = torch.randn_like(state)
    source_times = torch.tensor([10, 7])
    target_times = torch.tensor([3, 0])
    prediction = GaussianPrediction(
        clean=clean,
        epsilon=torch.zeros_like(state),
        model_output=torch.zeros_like(state),
    )

    transition = DDPMAncestralSampler().transition(
        process,
        state,
        source_times,
        prediction,
        target_times=target_times,
    )
    snapshot = process.marginal_coefficient_snapshot(
        source_times,
        target_times,
        state.size(),
    )
    denominator = 1.0 - snapshot.source_alpha_bar
    pair_variance = 1.0 - snapshot.transition_alpha
    expected_mean = (
        (
            snapshot.target_alpha_bar.sqrt()
            * pair_variance
            / denominator
        ).float()
        * clean
        + (
            snapshot.transition_alpha.sqrt()
            * (1.0 - snapshot.target_alpha_bar)
            / denominator
        ).float()
        * state
    )
    expected_variance = pair_variance * (
        1.0 - snapshot.target_alpha_bar
    ) / denominator

    assert torch.allclose(transition.mean, expected_mean)
    assert torch.allclose(
        transition.standard_deviation[:1],
        expected_variance[:1].sqrt().float(),
    )
    assert torch.count_nonzero(transition.standard_deviation[1]) == 0


def test_respaced_ddpm_non_adjacent_transition_matches_pinned_upstream() -> None:
    fixture = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    reference = fixture["respaced_sampling"]
    source, target = reference["non_adjacent_public_pair"]
    process = gaussian_process(reference["num_timesteps"])
    state = torch.tensor([[0.0], [1.0]])
    clean = torch.tensor([[1.0], [0.0]])
    prediction = GaussianPrediction(
        clean=clean,
        epsilon=torch.zeros_like(state),
        model_output=torch.zeros_like(state),
    )

    transition = DDPMAncestralSampler().transition(
        process,
        state,
        torch.tensor([source, source]),
        prediction,
        target_times=torch.tensor([target, target]),
    )

    expected_mean = torch.tensor(
        [
            [reference["runtime_non_adjacent_posterior_mean_coefficient_1"]],
            [reference["runtime_non_adjacent_posterior_mean_coefficient_2"]],
        ]
    )
    expected_standard_deviation = math.sqrt(
        reference["runtime_non_adjacent_posterior_variance"]
    )
    assert torch.equal(transition.mean, expected_mean)
    assert torch.allclose(
        transition.standard_deviation,
        torch.full_like(state, expected_standard_deviation),
        rtol=1e-7,
        atol=0.0,
    )


def test_respaced_ddpm_uses_target_aware_learned_variance() -> None:
    process = gaussian_process(10)
    state = torch.randn(2, 3, 4, 4)
    source_times = torch.tensor([10, 10])
    target_times = torch.tensor([3, 3])

    dynamics = GaussianModelDynamics(
        process,
        lambda current, model_times: torch.cat(
            (torch.zeros_like(current), torch.ones_like(current)),
            dim=1,
        ),
        variance_mode="learned_range",
        clip_denoised=False,
    )
    prediction = dynamics.predict_transition(
        state,
        source_times,
        target_times,
    )
    transition = DDPMAncestralSampler().transition(
        process,
        state,
        source_times,
        prediction,
        target_times=target_times,
    )
    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        state.size(),
    )

    assert isinstance(prediction, LearnedVarianceGaussianPrediction)
    assert torch.allclose(
        transition.standard_deviation,
        (0.5 * bounds.upper).exp().expand_as(state),
    )


def test_adjacent_stochastic_ddpm_transition_samples_learned_variance() -> None:
    process = gaussian_process(10)
    state = torch.tensor([[0.25, -0.5]])
    source_times = torch.tensor([6])
    target_times = torch.tensor([5])
    dynamics = GaussianModelDynamics(
        process,
        lambda current, model_times: torch.cat(
            (torch.zeros_like(current), torch.ones_like(current)),
            dim=1,
        ),
        variance_mode="learned_range",
        clip_denoised=False,
    )
    prediction = dynamics.predict_transition(
        state,
        source_times,
        target_times,
    )
    transition = DDPMAncestralSampler().transition(
        process,
        state,
        source_times,
        prediction,
    )
    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        state.size(),
    )
    expected_standard_deviation = (0.5 * bounds.upper).exp().expand_as(state)
    actual_generator = torch.Generator().manual_seed(29)
    expected_generator = torch.Generator().manual_seed(29)
    expected_noise = torch.randn(
        state.shape,
        dtype=state.dtype,
        generator=expected_generator,
    )

    assert isinstance(prediction, LearnedVarianceGaussianPrediction)
    assert torch.allclose(
        transition.standard_deviation,
        expected_standard_deviation,
    )
    assert torch.equal(
        transition.sample(generator=actual_generator),
        transition.mean + expected_standard_deviation * expected_noise,
    )


@pytest.mark.parametrize(
    ("num_inference_steps", "expected_steps"),
    [(250, 250), (None, 1000)],
)
def test_ddpm_production_length_chains_preserve_seed_observer_and_nfe(
    num_inference_steps: int | None,
    expected_steps: int,
) -> None:
    process = gaussian_process(1000)
    sampler = DDPMAncestralSampler(
        num_inference_steps=num_inference_steps,
    )
    schedule = sampler.resolve_schedule(process)
    initial = torch.zeros(1, 1)

    def run_once() -> tuple[
        SamplerResult,
        tuple[SamplingObservation, ...],
        list[int],
    ]:
        model_times: list[int] = []

        def predict(
            state: torch.Tensor,
            times: torch.Tensor,
        ) -> torch.Tensor:
            model_times.append(int(times[0]))
            return torch.zeros_like(state)

        observer = TrajectoryObserver()
        result = sampler.sample(
            GaussianModelDynamics(process, predict),
            initial.clone(),
            generator=torch.Generator().manual_seed(41),
            observer=observer,
        )
        return result, observer.observations, model_times

    first_result, first_trajectory, first_model_times = run_once()
    second_result, second_trajectory, second_model_times = run_once()
    expected_coordinates = schedule.tolist()
    expected_model_times = (schedule[:-1] - 1).tolist()

    assert first_result.num_steps == second_result.num_steps == expected_steps
    assert first_result.diagnostics == second_result.diagnostics == {
        "num_dynamics_evaluations": expected_steps
    }
    assert torch.equal(first_result.final_state, second_result.final_state)
    assert first_model_times == second_model_times == expected_model_times
    assert [item.coordinate for item in first_trajectory] == expected_coordinates
    assert [item.coordinate for item in second_trajectory] == expected_coordinates
    assert len(first_trajectory) == len(second_trajectory) == expected_steps + 1
    for step_index, (first, second) in enumerate(
        zip(first_trajectory, second_trajectory, strict=True)
    ):
        assert first.step_index == second.step_index == step_index
        assert first.diagnostics == second.diagnostics
        assert torch.equal(first.state, second.state)
        assert first.is_final == second.is_final == (step_index == expected_steps)


def test_ddim_ignores_learned_variance_but_consumes_prediction_head() -> None:
    process = gaussian_process(4)
    initial = torch.randn(2, 3, 4, 4)

    fixed = GaussianModelDynamics(
        process,
        lambda state, model_times: torch.zeros_like(state),
        clip_denoised=False,
    )
    learned = GaussianModelDynamics(
        process,
        lambda state, model_times: torch.cat(
            (torch.zeros_like(state), torch.full_like(state, 100.0)),
            dim=1,
        ),
        variance_mode="learned_range",
        clip_denoised=False,
    )
    sampler = DDIMSampler(schedule=[4, 2, 0], eta=0)

    fixed_result = sampler.sample(fixed, initial)
    learned_result = sampler.sample(learned, initial)

    assert torch.equal(fixed_result.final_state, learned_result.final_state)


def test_ddim_consumes_process_selected_pair_snapshot() -> None:
    class RecordingSelectedPairProcess(DiscreteGaussianProcess):
        """Count selected-pair snapshot use by one DDIM transition."""

        def __init__(self) -> None:
            super().__init__(
                {
                    "name": "linear_beta",
                    "params": {"num_timesteps": 4},
                }
            )
            self.snapshot_calls = 0

        def marginal_coefficient_snapshot(
            self,
            source_times: torch.Tensor,
            target_times: torch.Tensor,
            broadcast_shape: torch.Size,
        ) -> GaussianMarginalCoefficientSnapshot:
            self.snapshot_calls += 1
            return super().marginal_coefficient_snapshot(
                source_times,
                target_times,
                broadcast_shape,
            )

    process = RecordingSelectedPairProcess()
    state = torch.randn(2, 3)
    source_times = torch.tensor([4, 4])
    target_times = torch.tensor([2, 2])
    prediction = GaussianPrediction(
        clean=torch.zeros_like(state),
        epsilon=torch.zeros_like(state),
        model_output=torch.zeros_like(state),
    )

    DDIMSampler(eta=0.5).transition(
        process,
        state,
        source_times,
        target_times,
        prediction,
    )

    assert process.snapshot_calls == 1


def test_sampling_recipe_parser_validates_variance_mode() -> None:
    config = StandardDenoisingBuilder._parse_params(
        {
            "variance": {"mode": "learned_range"},
            "sampler": {"name": "ddpm", "params": {}},
        }
    )
    assert config.variance_mode == "learned_range"

    with pytest.raises(ValueError, match="fixed or learned_range"):
        StandardDenoisingBuilder._parse_params(
            {
                "variance": {"mode": "unknown"},
                "sampler": {"name": "ddpm", "params": {}},
            }
        )
