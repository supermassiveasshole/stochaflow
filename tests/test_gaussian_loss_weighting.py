"""Exact learned-range Gaussian loss tests for concrete Strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training import GaussianDenoisingTrainingStrategy, MSEObjective
from stochaflow.training.gaussian import GaussianVarianceConfig
from stochaflow.training.gaussian.loss import GaussianLossComputation
from stochaflow.training.gaussian.variance import (
    _discretized_gaussian_log_likelihood,
    learned_range_log_variance,
)

REFERENCE = (
    Path(__file__).parent
    / "fixtures"
    / "gaussian"
    / "learned_range_reference.json"
)


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


class CountingMSEObjective(MSEObjective):
    """Record the per-sample call made by learned-range training."""

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


def compute_loss(
    strategy: InspectableGaussianStrategy,
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


def test_hybrid_loss_matches_pinned_iddpm_fixture() -> None:
    fixture = json.loads(REFERENCE.read_text(encoding="utf-8"))
    values = fixture["hybrid_loss"]
    process = gaussian_process(values["num_timesteps"], dtype=torch.float64)
    clean = torch.tensor(values["clean"], dtype=torch.float64).reshape(2, 1, 2, 2)
    noise = torch.tensor(values["noise"], dtype=torch.float64).reshape_as(clean)
    mean_head = torch.tensor(values["mean_head"], dtype=torch.float64).reshape_as(
        clean
    )
    variance_head = torch.tensor(
        values["variance_head"], dtype=torch.float64
    ).reshape_as(clean)
    state_times = torch.tensor(values["model_timesteps"]) + 1
    raw_output = torch.cat((mean_head, variance_head), dim=1)
    variance = GaussianVarianceConfig(mode="learned_range")

    result = compute_loss(
        standard_strategy(process, variance=variance),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw_output,
    )

    expected = values["constant"]
    expected_simple = torch.tensor(expected["simple"], dtype=torch.float64)
    expected_variational_bound = torch.tensor(
        expected["variational_bound"], dtype=torch.float64
    )
    expected_loss = torch.tensor(expected["loss"], dtype=torch.float64)
    simple = (mean_head - noise).square().flatten(1).mean(dim=1)

    assert result.per_sample_loss is not None
    assert result.per_sample_simple_loss is not None
    assert result.per_sample_variational_bound is not None
    torch.testing.assert_close(simple, expected_simple, rtol=2e-6, atol=2e-8)
    torch.testing.assert_close(
        result.per_sample_loss - simple,
        expected_variational_bound,
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        result.per_sample_simple_loss,
        expected_simple,
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        result.per_sample_variational_bound,
        expected_variational_bound,
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        result.per_sample_loss,
        expected_loss,
        rtol=2e-6,
        atol=2e-8,
    )


def test_learned_range_vb_detaches_mean_prediction_but_trains_variance() -> None:
    process = gaussian_process(dtype=torch.float64)
    variance = GaussianVarianceConfig(mode="learned_range")
    clean = torch.zeros(2, 1, 2, 2, dtype=torch.float64)
    noise = torch.full_like(clean, 0.25)
    state_times = torch.tensor([2, 7])
    mean = noise.clone().requires_grad_(True)
    variance_head = torch.zeros_like(clean, requires_grad=True)

    result = compute_loss(
        standard_strategy(process, variance=variance),
        clean=clean,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.cat((mean, variance_head), dim=1),
    )
    result.loss.backward()

    assert mean.grad is None or not torch.any(mean.grad)
    assert variance_head.grad is not None
    assert torch.any(variance_head.grad != 0)


def _production_v_cosine_hybrid(
    dtype: torch.dtype,
) -> tuple[GaussianLossComputation, torch.Tensor, torch.Tensor]:
    process = DiscreteGaussianProcess(
        {
            "name": "cosine_alpha_bar",
            "params": {
                "num_timesteps": 1000,
                "s": 0.008,
                "max_beta": 0.999,
            },
        }
    )
    strategy = InspectableGaussianStrategy(
        PlaceholderDenoiser(),
        process,
        MSEObjective("mean"),
        prediction_type="v",
        variance=GaussianVarianceConfig(mode="learned_range"),
    )
    clean = torch.tensor(
        [[-0.75, 0.25], [0.5, -0.125], [0.875, -0.625]],
        dtype=dtype,
    ).reshape(3, 1, 1, 2)
    noise = torch.tensor(
        [[0.5, -0.25], [-0.75, 0.125], [0.375, 0.875]],
        dtype=dtype,
    ).reshape_as(clean)
    state_times = torch.tensor([1, 500, 1000])
    mean_head = torch.tensor(
        [[0.125, -0.375], [0.25, 0.625], [-0.5, 0.25]],
        dtype=dtype,
    ).reshape_as(clean)
    mean_head.requires_grad_(True)
    variance_head = torch.tensor(
        [[-0.75, 0.5], [0.0, 0.75], [-0.5, 0.25]],
        dtype=dtype,
    ).reshape_as(clean)
    variance_head.requires_grad_(True)

    return (
        compute_loss(
            strategy,
            clean=clean,
            noise=noise,
            state_times=state_times,
            raw_model_output=torch.cat((mean_head, variance_head), dim=1),
        ),
        mean_head,
        variance_head,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_production_v_cosine_learned_range_hybrid_is_finite(
    dtype: torch.dtype,
) -> None:
    result, _, _ = _production_v_cosine_hybrid(dtype)

    assert result.per_sample_loss is not None
    assert result.per_sample_variational_bound is not None
    assert torch.all(torch.isfinite(result.per_sample_loss))
    assert torch.all(torch.isfinite(result.per_sample_variational_bound))


def test_production_v_cosine_learned_range_hybrid_has_split_gradients() -> None:
    result, mean_head, variance_head = _production_v_cosine_hybrid(torch.float32)
    simple_mean = mean_head.detach().clone().requires_grad_(True)
    simple_loss = (
        (simple_mean - result.target.detach()).square().flatten(1).mean(dim=1).mean()
    )
    (expected_mean_grad,) = torch.autograd.grad(simple_loss, simple_mean)

    result.loss.backward()

    assert mean_head.grad is not None
    assert variance_head.grad is not None
    assert torch.all(torch.isfinite(mean_head.grad))
    assert torch.all(torch.isfinite(variance_head.grad))
    assert torch.any(mean_head.grad != 0)
    assert torch.any(variance_head.grad != 0)
    torch.testing.assert_close(
        mean_head.grad,
        expected_mean_grad,
        rtol=1e-6,
        atol=1e-7,
    )


def test_decoder_likelihood_handles_exact_pixel_tails() -> None:
    values = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float64)
    means = torch.zeros_like(values)
    log_scales = torch.zeros_like(values)

    actual = _discretized_gaussian_log_likelihood(
        values,
        means=means,
        log_scales=log_scales,
    )
    half_bin = 1.0 / 255.0
    exact_cdf = torch.special.ndtr
    expected = torch.stack(
        (
            torch.log(exact_cdf(values[0, 0] + half_bin)),
            torch.log(
                exact_cdf(values[0, 1] + half_bin)
                - exact_cdf(values[0, 1] - half_bin)
            ),
            torch.log1p(-exact_cdf(values[0, 2] - half_bin)),
        )
    )

    assert torch.all(torch.isfinite(actual))
    torch.testing.assert_close(
        actual[0, 0],
        actual[0, 2],
        rtol=0.0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual[0],
        expected,
        rtol=0.0,
        atol=1.2e-3,
    )


def test_learned_range_calls_mse_per_sample_once() -> None:
    process = gaussian_process()
    objective = CountingMSEObjective()
    clean = torch.zeros(2, 1, 1, 1)
    noise = torch.ones_like(clean)
    variance = GaussianVarianceConfig(mode="learned_range")

    compute_loss(
        standard_strategy(process, variance=variance, objective=objective),
        clean=clean,
        noise=noise,
        state_times=torch.tensor([1, 8]),
        raw_model_output=torch.zeros(2, 2, 1, 1),
    )

    assert objective.per_sample_calls == 1


def test_learned_range_rejects_sum_reduction() -> None:
    with pytest.raises(ValueError, match="requires MSEObjective reduction='mean'"):
        standard_strategy(
            gaussian_process(),
            reduction="sum",
            variance=GaussianVarianceConfig(mode="learned_range"),
        )


def test_learned_range_rejects_invalid_concrete_mse_output() -> None:
    process = gaussian_process()
    variance = GaussianVarianceConfig(mode="learned_range")
    objective = IntegerPerSampleMSEObjective()

    with pytest.raises(
        TypeError,
        match="per-sample objective capability must return a floating-point Tensor",
    ):
        compute_loss(
            standard_strategy(process, variance=variance, objective=objective),
            clean=torch.zeros(2, 1, 1, 1),
            noise=torch.ones(2, 1, 1, 1),
            state_times=torch.tensor([1, 8]),
            raw_model_output=torch.zeros(2, 2, 1, 1),
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
