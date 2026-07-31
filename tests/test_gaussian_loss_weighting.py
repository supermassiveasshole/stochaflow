"""Exact P2 and learned-range loss tests for concrete Strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training import (
    GaussianDenoisingTrainingStrategy,
    MSEObjective,
    P2GaussianDenoisingTrainingStrategy,
)
from stochaflow.training.gaussian import GaussianVarianceConfig
from stochaflow.training.gaussian.loss import GaussianLossComputation
from stochaflow.training.gaussian.variance import learned_range_log_variance

REFERENCE = Path(__file__).parent / "fixtures" / "gaussian" / "p2_reference.json"


class PlaceholderDenoiser(nn.Module):
    """Model placeholder for directly characterized Strategy loss methods."""

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        return torch.zeros_like(state)


class InspectableGaussianStrategy(GaussianDenoisingTrainingStrategy):
    """Expose the inherited loss template for deterministic tensor fixtures."""

    def compute_for_test(self, **kwargs: Any) -> GaussianLossComputation:
        return self._compute_loss(**kwargs)


class InspectableP2Strategy(P2GaussianDenoisingTrainingStrategy):
    """Expose the concrete P2 loss template for deterministic fixtures."""

    def compute_for_test(self, **kwargs: Any) -> GaussianLossComputation:
        return self._compute_loss(**kwargs)


class CountingMSEObjective(MSEObjective):
    """Record the per-sample call made by a P2 Strategy."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.per_sample_calls = 0

    def per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        self.per_sample_calls += 1
        return super().per_sample_loss(prediction, target)

class IntegerPerSampleMSEObjective(MSEObjective):
    """Deliberately violate the concrete MSE per-sample contract."""

    def per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        del prediction, target
        return torch.ones(2, dtype=torch.int64)


def gaussian_process(
    steps: int = 8,
    *,
    dtype: torch.dtype = torch.float32,
) -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": steps,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "dtype": dtype,
            },
        }
    )


def signal_to_noise_ratio(
    process: DiscreteGaussianProcess,
    state_times: torch.Tensor,
) -> torch.Tensor:
    scales = process.marginal_scales(state_times, state_times.size())
    return scales.signal.square() / scales.noise.square()


def standard_strategy(
    process: DiscreteGaussianProcess,
    *,
    reduction: str = "mean",
    variance: GaussianVarianceConfig | None = None,
    objective: MSEObjective | None = None,
) -> InspectableGaussianStrategy:
    return InspectableGaussianStrategy(
        PlaceholderDenoiser(),
        process,
        MSEObjective(reduction) if objective is None else objective,
        prediction_type="epsilon",
        variance=variance,
    )


def p2_strategy(
    process: DiscreteGaussianProcess,
    *,
    reduction: str = "mean",
    variance: GaussianVarianceConfig | None = None,
    objective: MSEObjective | None = None,
    k: float = 1.0,
    gamma: float = 1.0,
) -> InspectableP2Strategy:
    return InspectableP2Strategy(
        PlaceholderDenoiser(),
        process,
        MSEObjective(reduction) if objective is None else objective,
        variance=variance,
        k=k,
        gamma=gamma,
    )


def compute_loss(
    strategy: InspectableGaussianStrategy | InspectableP2Strategy,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    raw_model_output: torch.Tensor,
) -> GaussianLossComputation:
    noisy, _ = strategy.process.sample_marginal(
        clean,
        state_times,
        noise=noise,
    )
    return strategy.compute_for_test(
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw_model_output,
    )


def test_p2_matches_t1000_linear_schedule_fixture() -> None:
    process = gaussian_process(1000, dtype=torch.float64)
    state_times = torch.tensor([1, 500, 1000])
    snr = signal_to_noise_ratio(process, state_times)
    result = compute_loss(
        p2_strategy(process),
        clean=torch.zeros(3, 1, dtype=torch.float64),
        noise=torch.zeros(3, 1, dtype=torch.float64),
        state_times=state_times,
        raw_model_output=torch.ones(3, 1, dtype=torch.float64),
    )

    torch.testing.assert_close(
        snr,
        torch.tensor(
            [9999.0, 0.08528994446263685, 4.03599265116842e-5],
            dtype=torch.float64,
        ),
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        result.per_sample_loss,
        torch.tensor(
            [1.0e-4, 0.9214127571182217, 0.9999596417023462],
            dtype=torch.float64,
        ),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_gamma_zero_is_exact_standard_loss_identity(reduction: str) -> None:
    process = gaussian_process(dtype=torch.float64)
    clean = torch.tensor([[[[0.25]]], [[[0.75]]]], dtype=torch.float64)
    noise = torch.tensor([[[[0.1]]], [[[0.2]]]], dtype=torch.float64)
    output = torch.tensor([[[[0.4]]], [[[0.8]]]], dtype=torch.float64)
    state_times = torch.tensor([2, 7])

    standard = compute_loss(
        standard_strategy(process, reduction=reduction),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=output,
    )
    p2 = compute_loss(
        p2_strategy(process, reduction=reduction, k=7.0, gamma=0.0),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=output,
    )

    assert p2.per_sample_loss is not None
    assert standard.per_sample_loss is not None
    assert torch.equal(p2.per_sample_loss, standard.per_sample_loss)
    assert torch.equal(p2.loss, standard.loss)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_weighted_reduction_matches_manual_mse_semantics(reduction: str) -> None:
    process = gaussian_process(dtype=torch.float64)
    clean = torch.zeros(2, 1, 1, 1, dtype=torch.float64)
    noise = torch.tensor([[[[0.1]]], [[[0.2]]]], dtype=torch.float64)
    output = torch.tensor([[[[0.4]]], [[[0.8]]]], dtype=torch.float64)
    state_times = torch.tensor([1, 8])
    strategy = p2_strategy(process, reduction=reduction, k=1.0, gamma=1.0)

    result = compute_loss(
        strategy,
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=output,
    )

    simple = (output - noise).square().flatten(1)
    simple = simple.mean(dim=1) if reduction == "mean" else simple.sum(dim=1)
    snr = signal_to_noise_ratio(process, state_times)
    expected_per_sample = (1.0 + snr).reciprocal() * simple
    expected = (
        expected_per_sample.mean()
        if reduction == "mean"
        else expected_per_sample.sum()
    )
    torch.testing.assert_close(result.per_sample_loss, expected_per_sample)
    torch.testing.assert_close(result.loss, expected)


def test_hybrid_loss_matches_pinned_upstream_fixture_and_leaves_vb_unweighted() -> None:
    fixture = json.loads(REFERENCE.read_text(encoding="utf-8"))
    values = fixture["hybrid_loss"]
    process = gaussian_process(values["num_timesteps"], dtype=torch.float64)
    clean = torch.tensor(values["clean"], dtype=torch.float64).reshape(2, 1, 2, 2)
    noise = torch.tensor(values["noise"], dtype=torch.float64).reshape_as(clean)
    mean_head = torch.tensor(values["mean_head"], dtype=torch.float64).reshape_as(clean)
    variance_head = torch.tensor(
        values["variance_head"], dtype=torch.float64
    ).reshape_as(clean)
    state_times = torch.tensor(values["model_timesteps"]) + 1
    raw_output = torch.cat((mean_head, variance_head), dim=1)
    variance = GaussianVarianceConfig(mode="learned_range")

    constant = compute_loss(
        standard_strategy(process, variance=variance),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw_output,
    )
    p2 = compute_loss(
        p2_strategy(process, variance=variance),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw_output,
    )

    expected_constant = values["constant"]
    expected_p2 = values["p2_k1_gamma1"]
    torch.testing.assert_close(
        constant.per_sample_loss,
        torch.tensor(expected_constant["loss"], dtype=torch.float64),
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        p2.per_sample_loss,
        torch.tensor(expected_p2["loss"], dtype=torch.float64),
        rtol=2e-6,
        atol=2e-8,
    )
    simple = (mean_head - noise).square().flatten(1).mean(dim=1)
    weights = (1.0 + signal_to_noise_ratio(process, state_times)).reciprocal()
    assert constant.per_sample_loss is not None
    assert p2.per_sample_loss is not None
    constant_vb = constant.per_sample_loss - simple
    p2_vb = p2.per_sample_loss - weights * simple
    torch.testing.assert_close(p2_vb, constant_vb)


def test_learned_range_vb_detaches_mean_prediction_but_trains_variance() -> None:
    process = gaussian_process(dtype=torch.float64)
    variance = GaussianVarianceConfig(mode="learned_range")
    clean = torch.zeros(2, 1, 2, 2, dtype=torch.float64)
    noise = torch.full_like(clean, 0.25)
    state_times = torch.tensor([2, 7])
    mean = noise.clone().requires_grad_(True)
    variance_head = torch.zeros_like(clean, requires_grad=True)

    result = compute_loss(
        p2_strategy(process, variance=variance),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.cat((mean, variance_head), dim=1),
    )
    result.loss.backward()

    assert mean.grad is None or not torch.any(mean.grad)
    assert variance_head.grad is not None
    assert torch.any(variance_head.grad != 0)


def test_p2_calls_mse_per_sample_once() -> None:
    process = gaussian_process()
    objective = CountingMSEObjective()
    clean = torch.zeros(2, 1, 1, 1)
    noise = torch.ones_like(clean)
    output = torch.zeros_like(clean)

    compute_loss(
        p2_strategy(process, objective=objective),
        clean=clean,
        noise=noise,
        state_times=torch.tensor([1, 8]),
        raw_model_output=output,
    )

    assert objective.per_sample_calls == 1


@pytest.mark.parametrize(
    ("objective", "error_type", "message"),
    [
        (
            IntegerPerSampleMSEObjective(),
            TypeError,
            "per-sample objective capability must return a floating-point Tensor",
        ),
    ],
)
def test_p2_rejects_invalid_concrete_mse_outputs(
    objective: MSEObjective,
    error_type: type[Exception],
    message: str,
) -> None:
    process = gaussian_process()
    with pytest.raises(error_type, match=message):
        compute_loss(
            p2_strategy(process, objective=objective),
            clean=torch.zeros(2, 1, 1, 1),
            noise=torch.ones(2, 1, 1, 1),
            state_times=torch.tensor([1, 8]),
            raw_model_output=torch.zeros(2, 1, 1, 1),
        )


def test_learned_range_log_variance_uses_process_capability() -> None:
    process = gaussian_process()
    source_times = torch.tensor([2, 8])
    target_times = source_times - 1
    values = torch.tensor([[[[-1.0]]], [[[1.0]]]])
    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        values.size(),
    )

    interpolated = learned_range_log_variance(
        process,
        source_times,
        target_times,
        values,
    )

    assert torch.equal(interpolated[0], bounds.lower[0])
    assert torch.equal(interpolated[1], bounds.upper[1])
