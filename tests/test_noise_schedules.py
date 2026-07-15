"""Tests for the object-oriented forward noise-schedule package."""

import inspect

import pytest
import torch

from stochaflow.diffusion.noise_schedules import (
    CosineAlphaBarSchedule,
    DiscreteVPSchedule,
    LinearBetaSchedule,
    NoiseSchedule,
)


def test_noise_schedule_defines_process_level_abstract_contract() -> None:
    assert inspect.isabstract(NoiseSchedule)
    assert NoiseSchedule.__abstractmethods__ == {
        "terminal_time",
        "validate_state_times",
        "marginal_scales",
    }


def test_discrete_vp_schedule_accepts_a_precomputed_beta_parameterization() -> None:
    schedule = DiscreteVPSchedule(torch.linspace(0.01, 0.1, 8))

    assert schedule.num_timesteps == 8
    assert torch.allclose(schedule.alpha_t, 1 - schedule.beta_t)
    assert torch.allclose(schedule.alpha_bar_t, torch.cumprod(schedule.alpha_t, 0))


def test_discrete_vp_schedule_rejects_invalid_transition_tables() -> None:
    with pytest.raises(ValueError, match="non-empty 1D"):
        DiscreteVPSchedule(torch.empty(0))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        DiscreteVPSchedule(torch.tensor([0.0, 0.1]))
    with pytest.raises(ValueError, match="finite"):
        DiscreteVPSchedule(torch.tensor([0.1, float("nan")]))


def test_linear_beta_schedule_owns_linear_construction_policy() -> None:
    schedule = LinearBetaSchedule(
        num_timesteps=10,
        beta_start=0.001,
        beta_end=0.01,
        dtype=torch.float64,
    )

    assert schedule.beta_t.shape == (10,)
    assert schedule.beta_t.dtype == torch.float64
    assert schedule.beta_t[0].item() == pytest.approx(0.001)
    assert schedule.beta_t[-1].item() == pytest.approx(0.01)
    assert torch.all(schedule.beta_t[:-1] < schedule.beta_t[1:])


def test_linear_beta_schedule_validates_its_own_parameters() -> None:
    with pytest.raises(TypeError, match="num_timesteps"):
        LinearBetaSchedule(num_timesteps=True)
    with pytest.raises(ValueError, match="beta_start"):
        LinearBetaSchedule(num_timesteps=10, beta_start=0.02, beta_end=0.01)
    with pytest.raises(TypeError, match="floating-point"):
        LinearBetaSchedule(num_timesteps=10, dtype=torch.int64)


def test_cosine_alpha_bar_schedule_derives_a_valid_discrete_vp_path() -> None:
    schedule = CosineAlphaBarSchedule(num_timesteps=32, max_beta=0.9)

    assert schedule.beta_t.shape == (32,)
    assert torch.all(schedule.beta_t > 0)
    assert torch.all(schedule.beta_t <= 0.9)
    assert schedule.beta_t[-1].item() == pytest.approx(0.9)
    assert torch.all(schedule.alpha_bar_t[:-1] > schedule.alpha_bar_t[1:])


def test_cosine_alpha_bar_schedule_validates_its_own_parameters() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CosineAlphaBarSchedule(num_timesteps=10, s=-0.1)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        CosineAlphaBarSchedule(num_timesteps=10, max_beta=1.0)


def test_discrete_vp_schedule_exposes_public_state_time_domain() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)

    assert schedule.clean_time == 0
    assert schedule.terminal_time == 8
    assert torch.equal(
        schedule.validate_state_times(torch.tensor([0, 1, 8])),
        torch.tensor([0, 1, 8]),
    )


def test_discrete_vp_schedule_returns_exact_clean_marginal_scales() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)
    state_times = torch.tensor([0, 1, 8])

    signal_scale, noise_scale = schedule.marginal_scales(
        state_times,
        torch.Size([3, 2, 4, 4]),
    )

    assert signal_scale.shape == (3, 1, 1, 1)
    assert noise_scale.shape == (3, 1, 1, 1)
    assert signal_scale[0].item() == 1.0
    assert noise_scale[0].item() == 0.0
    assert torch.allclose(signal_scale[1, 0, 0, 0], schedule.sqrt_alpha_bar_t[0])
    assert torch.allclose(signal_scale[2, 0, 0, 0], schedule.sqrt_alpha_bar_t[7])


def test_noise_schedule_computes_snr_from_marginal_scales() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)
    state_times = torch.tensor([1, 8])

    snr = schedule.signal_to_noise_ratio(state_times, torch.Size([2, 3]))

    expected = schedule.alpha_bar_t[[0, 7]] / (1 - schedule.alpha_bar_t[[0, 7]])
    assert torch.allclose(snr.flatten(), expected)


def test_discrete_vp_schedule_validates_query_shape_and_domain() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)

    with pytest.raises(TypeError, match="integer mathematical states"):
        schedule.marginal_scales(
            torch.tensor([0.0, 1.0]),
            torch.Size([2, 3]),
        )
    with pytest.raises(ValueError, match=r"\[0, T\]"):
        schedule.marginal_scales(torch.tensor([0, 9]), torch.Size([2, 3]))
    with pytest.raises(ValueError, match="batch dimension"):
        schedule.marginal_scales(torch.tensor([0, 1]), torch.Size([3, 3]))


def test_schedule_does_not_extract_tables_owned_by_other_processes() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)

    assert not hasattr(schedule, "extract")
    assert set(schedule.state_dict()) == {
        "beta_t",
        "alpha_t",
        "alpha_bar_t",
        "sqrt_alpha_bar_t",
        "sqrt_one_minus_alpha_bar_t",
    }
