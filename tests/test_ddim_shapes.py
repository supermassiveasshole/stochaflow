"""Tests for DDIM on the unified complete-sampler interface."""

import pytest
import torch

from stochaflow.processes import DiscreteGaussianProcess, GaussianScales
from stochaflow.sampling import (
    DDIMSampler,
    DDPMAncestralSampler,
    GaussianModelDynamics,
    TrajectoryObserver,
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
