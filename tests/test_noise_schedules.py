"""Tests for the object-oriented forward noise-schedule package."""

import inspect
from typing import ClassVar

import pytest
import torch

from stochaflow.processes.noise_schedules import (
    CosineAlphaBarSchedule,
    DiscreteVPCoefficients,
    DiscreteVPSchedule,
    GaussianNoiseSchedule,
    GaussianScales,
    LinearBetaSchedule,
    TabulatedDiscreteVPSchedule,
)
from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.utils.registry import REGISTRIES


class AnalyticDiscreteVPSchedule(DiscreteVPSchedule):
    """Table-free constant-alpha schedule used to verify the VP capability."""

    last_instance: ClassVar["AnalyticDiscreteVPSchedule | None"] = None
    rate: torch.Tensor

    def __init__(self, num_timesteps: int = 4) -> None:
        super().__init__()
        self._num_timesteps = num_timesteps
        self.register_buffer("rate", torch.tensor(0.9))
        AnalyticDiscreteVPSchedule.last_instance = self

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    def marginal_scales(self, state_times: torch.Tensor) -> GaussianScales:
        state_times = self.validate_state_times(state_times)
        alpha_bar = torch.pow(self.rate.to(state_times.device), state_times)
        return GaussianScales(alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt())

    def transition_coefficients(
        self,
        state_times: torch.Tensor,
    ) -> DiscreteVPCoefficients:
        state_times = self.validate_state_times(state_times)
        if torch.any(state_times == 0):
            raise ValueError("transition state times must lie in [1, T]")
        alpha = self.rate.to(state_times.device).expand(state_times.shape)
        alpha_bar = torch.pow(alpha, state_times)
        return DiscreteVPCoefficients(
            beta=1.0 - alpha,
            alpha=alpha,
            alpha_bar=alpha_bar,
            previous_alpha_bar=torch.pow(alpha, state_times - 1),
        )


class LearnableAnalyticSchedule(AnalyticDiscreteVPSchedule):
    def __init__(self) -> None:
        super().__init__()
        self.learnable_rate = torch.nn.Parameter(torch.tensor(0.9))


class InvalidSnapshotSchedule(AnalyticDiscreteVPSchedule):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def marginal_scales(self, state_times: torch.Tensor) -> GaussianScales:
        scales = super().marginal_scales(state_times)
        signal = scales.signal.clone()
        if self.mode == "shape":
            signal = signal[:-1]
        elif self.mode == "nonfinite":
            signal[-1] = torch.inf
        elif self.mode == "gradient":
            signal.requires_grad_()
        elif self.mode == "integer":
            signal = signal.to(dtype=torch.long)
        return GaussianScales(signal, scales.noise[: signal.shape[0]])


REGISTRIES.noise_schedules.add("test_analytic_vp", AnalyticDiscreteVPSchedule)
REGISTRIES.noise_schedules.add(
    "test_learnable_analytic_vp", LearnableAnalyticSchedule
)
REGISTRIES.noise_schedules.add("test_invalid_snapshot", InvalidSnapshotSchedule)


def test_noise_schedule_defines_process_level_abstract_contract() -> None:
    assert inspect.isabstract(GaussianNoiseSchedule)
    assert GaussianNoiseSchedule.__abstractmethods__ == {
        "terminal_time",
        "validate_state_times",
        "marginal_scales",
    }
    assert inspect.isabstract(DiscreteVPSchedule)
    assert DiscreteVPSchedule.__abstractmethods__ == {
        "marginal_scales",
        "num_timesteps",
        "transition_coefficients",
    }


def test_discrete_vp_schedule_accepts_a_precomputed_beta_parameterization() -> None:
    schedule = TabulatedDiscreteVPSchedule(torch.linspace(0.01, 0.1, 8))

    assert schedule.num_timesteps == 8
    assert torch.allclose(schedule.alpha_t, 1 - schedule.beta_t)
    assert torch.allclose(schedule.alpha_bar_t, torch.cumprod(schedule.alpha_t, 0))


def test_discrete_vp_schedule_rejects_invalid_transition_tables() -> None:
    with pytest.raises(ValueError, match="non-empty 1D"):
        TabulatedDiscreteVPSchedule(torch.empty(0))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        TabulatedDiscreteVPSchedule(torch.tensor([0.0, 0.1]))
    with pytest.raises(ValueError, match="finite"):
        TabulatedDiscreteVPSchedule(torch.tensor([0.1, float("nan")]))


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

    scales = schedule.marginal_scales(state_times)

    assert scales.signal.shape == state_times.shape
    assert scales.noise.shape == state_times.shape
    assert scales.signal[0].item() == 1.0
    assert scales.noise[0].item() == 0.0
    assert torch.allclose(scales.signal[1], schedule.sqrt_alpha_bar_t[0])
    assert torch.allclose(scales.signal[2], schedule.sqrt_alpha_bar_t[7])


def test_noise_schedule_computes_snr_from_marginal_scales() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)
    state_times = torch.tensor([1, 8])

    snr = schedule.signal_to_noise_ratio(state_times)

    expected = schedule.alpha_bar_t[[0, 7]] / (1 - schedule.alpha_bar_t[[0, 7]])
    assert torch.allclose(snr, expected)


def test_discrete_vp_schedule_validates_query_shape_and_domain() -> None:
    schedule = LinearBetaSchedule(num_timesteps=8)

    with pytest.raises(TypeError, match="integer mathematical states"):
        schedule.marginal_scales(torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\[0, T\]"):
        schedule.marginal_scales(torch.tensor([0, 9]))

    process = DiscreteGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 8}}
    )
    with pytest.raises(ValueError, match="batch dimension"):
        process.marginal_scales(torch.tensor([0, 1]), torch.Size([3, 3]))


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


def test_process_depends_on_vp_capability_not_table_storage() -> None:
    process = DiscreteGaussianProcess(
        {"name": "test_analytic_vp", "params": {"num_timesteps": 6}}
    )
    state_times = torch.tensor([0, 3, 6])

    scales = process.marginal_scales(state_times, torch.Size([3, 2, 4]))
    schedule = AnalyticDiscreteVPSchedule.last_instance

    assert schedule is not None
    assert not hasattr(process, "schedule")
    assert scales.signal.shape == (3, 1, 1)
    assert scales.noise.shape == (3, 1, 1)
    assert process.posterior_mean_coef1.shape == (6,)
    assert not any(name.startswith("schedule.") for name in process.state_dict())

    schedule.rate.fill_(0.5)
    repeated = process.marginal_scales(state_times, torch.Size([3, 2, 4]))
    assert torch.equal(repeated.signal, scales.signal)
    assert torch.equal(repeated.noise, scales.noise)


def test_process_snapshot_moves_as_the_single_authoritative_state() -> None:
    process = DiscreteGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    ).to(dtype=torch.float64)

    assert process.marginal_signal_t.dtype == torch.float64
    assert process.marginal_noise_t.dtype == torch.float64
    assert process.posterior_mean_coef1.dtype == torch.float64
    assert set(process.state_dict()) == {
        "marginal_signal_t",
        "marginal_noise_t",
        "sqrt_posterior_variance_t",
        "posterior_mean_coef1",
        "posterior_mean_coef2",
    }


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("shape", ValueError, "shape"),
        ("nonfinite", ValueError, "finite"),
        ("gradient", TypeError, "gradients"),
        ("integer", TypeError, "floating-point"),
    ],
)
def test_process_rejects_invalid_schedule_snapshots(
    mode: str,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        DiscreteGaussianProcess(
            {"name": "test_invalid_snapshot", "params": {"mode": mode}}
        )


def test_process_rejects_learnable_schedule_until_coefficients_are_dynamic() -> None:
    with pytest.raises(TypeError, match="immutable schedule without Parameters"):
        DiscreteGaussianProcess(
            {"name": "test_learnable_analytic_vp", "params": {}}
        )
