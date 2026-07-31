"""Exact P2 weighting and Gaussian hybrid-loss regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training.gaussian import GaussianDenoisingTrainingStrategy
from stochaflow.training.gaussian_loss import (
    GaussianLossWeightingConfig,
    GaussianVarianceConfig,
    compute_gaussian_training_loss,
    gaussian_loss_diagnostics,
    gaussian_signal_to_noise_ratio,
    gaussian_timestep_loss_weights,
    learned_range_log_variance,
    parse_gaussian_loss_weighting,
    parse_gaussian_variance,
)
from stochaflow.training.objectives import MSEObjective

_REFERENCE = (
    Path(__file__).parent / "fixtures" / "gaussian" / "p2_reference.json"
)


def gaussian_process(
    steps: int = 8,
    *,
    dtype: torch.dtype = torch.float32,
) -> DiscreteGaussianProcess:
    """Build a small fixed schedule for exact loss tests."""

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


def test_p2_snr_uses_cumulative_marginal_scales() -> None:
    process = gaussian_process()
    state_times = torch.tensor([1, 3, 8])
    scales = process.marginal_scales(state_times, state_times.size())

    snr = gaussian_signal_to_noise_ratio(process, state_times)

    assert torch.allclose(snr, scales.signal.square() / scales.noise.square())


def test_p2_matches_pinned_upstream_numeric_fixture() -> None:
    process = gaussian_process(dtype=torch.float64)
    state_times = torch.tensor([1, 4, 8])
    snr, weights = gaussian_timestep_loss_weights(
        process,
        state_times,
        GaussianLossWeightingConfig(name="p2", k=1.0, gamma=1.0),
    )

    assert torch.allclose(
        snr,
        torch.tensor(
            [9999.0, 56.59299861, 11.85548694],
            dtype=torch.float64,
        ),
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.allclose(
        weights,
        torch.tensor(
            [1.0e-4, 1.73632216e-2, 7.77877963e-2],
            dtype=torch.float64,
        ),
        rtol=2e-5,
        atol=2e-7,
    )


def test_hybrid_loss_matches_pinned_upstream_decoder_kl_and_rescaling() -> None:
    fixture = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    assert fixture["source"]["commit"] == (
        "3da0947ac350072e457c211401218175bc94e137"
    )
    values = fixture["hybrid_loss"]
    process = gaussian_process(
        values["num_timesteps"],
        dtype=torch.float64,
    )
    clean = torch.tensor(values["clean"], dtype=torch.float64).reshape(2, 1, 2, 2)
    noise = torch.tensor(values["noise"], dtype=torch.float64).reshape_as(clean)
    mean_head = torch.tensor(
        values["mean_head"],
        dtype=torch.float64,
    ).reshape_as(clean)
    variance_head = torch.tensor(
        values["variance_head"],
        dtype=torch.float64,
    ).reshape_as(clean)
    state_times = torch.tensor(values["model_timesteps"]) + 1
    noisy, _ = process.sample_marginal(clean, state_times, noise=noise)
    variance = GaussianVarianceConfig(
        mode="learned_range",
        loss="rescaled_variational_bound",
    )

    constant = compute_gaussian_training_loss(
        objective=MSEObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.cat((mean_head, variance_head), dim=1),
        prediction_type="epsilon",
        variance=variance,
        loss_weighting=GaussianLossWeightingConfig(),
    )
    p2 = compute_gaussian_training_loss(
        objective=MSEObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.cat((mean_head, variance_head), dim=1),
        prediction_type="epsilon",
        variance=variance,
        loss_weighting=GaussianLossWeightingConfig(
            name="p2",
            k=1.0,
            gamma=1.0,
        ),
    )

    expected_constant = values["constant"]
    expected_p2 = values["p2_k1_gamma1"]
    assert torch.equal(constant.target, noise)
    assert torch.equal(p2.target, noise)
    assert constant.per_sample_simple_loss is not None
    assert constant.per_sample_variational_bound is not None
    assert constant.per_sample_loss is not None
    assert p2.per_sample_weighted_simple_loss is not None
    assert p2.per_sample_variational_bound is not None
    assert p2.per_sample_loss is not None
    torch.testing.assert_close(
        constant.per_sample_simple_loss,
        torch.tensor(expected_constant["simple"], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        constant.per_sample_variational_bound,
        torch.tensor(
            expected_constant["variational_bound"],
            dtype=torch.float64,
        ),
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        constant.per_sample_loss,
        torch.tensor(expected_constant["loss"], dtype=torch.float64),
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        p2.per_sample_weighted_simple_loss,
        torch.tensor(expected_p2["weighted_simple"], dtype=torch.float64),
        rtol=2e-6,
        atol=2e-10,
    )
    torch.testing.assert_close(
        p2.per_sample_variational_bound,
        torch.tensor(expected_p2["variational_bound"], dtype=torch.float64),
        rtol=2e-6,
        atol=2e-8,
    )
    torch.testing.assert_close(
        p2.per_sample_loss,
        torch.tensor(expected_p2["loss"], dtype=torch.float64),
        rtol=2e-6,
        atol=2e-8,
    )


def test_p2_gamma_zero_and_closed_form_match_reference_identities() -> None:
    process = gaussian_process()
    state_times = torch.tensor([1, 4, 8])
    scales = process.marginal_scales(state_times, state_times.size())

    _, gamma_zero = gaussian_timestep_loss_weights(
        process,
        state_times,
        GaussianLossWeightingConfig(name="p2", k=7.0, gamma=0.0),
    )
    _, paper_weight = gaussian_timestep_loss_weights(
        process,
        state_times,
        GaussianLossWeightingConfig(name="p2", k=1.0, gamma=1.0),
    )

    assert torch.equal(gamma_zero, torch.ones_like(gamma_zero))
    assert torch.allclose(paper_weight, scales.noise.square(), atol=1e-7)


def test_p2_loss_uses_raw_weights_without_batch_renormalization() -> None:
    process = gaussian_process()
    clean = torch.zeros(2, 1, 2, 2)
    noise = torch.ones_like(clean)
    state_times = torch.tensor([1, 8])
    noisy, _ = process.sample_marginal(clean, state_times, noise=noise)
    weighting = GaussianLossWeightingConfig(name="p2", k=1.0, gamma=1.0)
    _, expected_weights = gaussian_timestep_loss_weights(
        process,
        state_times,
        weighting,
    )

    result = compute_gaussian_training_loss(
        objective=MSEObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.zeros_like(clean),
        prediction_type="epsilon",
        variance=GaussianVarianceConfig(),
        loss_weighting=weighting,
    )

    assert result.per_sample_simple_loss is not None
    assert torch.equal(result.per_sample_simple_loss, torch.ones(2))
    assert result.per_sample_weighted_simple_loss is not None
    assert result.per_sample_variational_bound is None
    assert (
        "per_sample_variational_bound"
        not in gaussian_loss_diagnostics(result)
    )
    assert torch.allclose(
        result.per_sample_weighted_simple_loss,
        expected_weights,
    )
    assert result.loss.item() == pytest.approx(expected_weights.mean().item())
    assert result.loss.item() != pytest.approx(1.0)


def test_fixed_scalar_objective_computation_carries_training_target() -> None:
    process = gaussian_process(2)
    clean = torch.zeros(2, 1)
    noise = torch.tensor([[0.25], [0.75]])
    state_times = torch.tensor([1, 2])
    noisy, _ = process.sample_marginal(clean, state_times, noise=noise)

    result = compute_gaussian_training_loss(
        objective=nn.MSELoss(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.zeros_like(clean),
        prediction_type="epsilon",
        variance=GaussianVarianceConfig(),
        loss_weighting=GaussianLossWeightingConfig(),
    )

    assert result.per_sample_simple_loss is None
    assert torch.equal(result.target, noise)


class AbsolutePerSampleObjective(nn.Module):
    """Independent per-sample Objective used to prove capability reuse."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (prediction - target).abs().mean()

    def per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (prediction - target).abs().reshape(prediction.shape[0], -1).mean(1)


def test_p2_reuses_an_independent_per_sample_objective() -> None:
    process = gaussian_process(2)
    clean = torch.zeros(2, 1)
    noise = torch.tensor([[1.0], [2.0]])
    state_times = torch.tensor([1, 2])
    noisy, _ = process.sample_marginal(clean, state_times, noise=noise)

    result = compute_gaussian_training_loss(
        objective=AbsolutePerSampleObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.zeros_like(clean),
        prediction_type="epsilon",
        variance=GaussianVarianceConfig(),
        loss_weighting=GaussianLossWeightingConfig(
            name="p2",
            k=1.0,
            gamma=1.0,
        ),
    )

    assert result.per_sample_simple_loss is not None
    assert torch.equal(result.per_sample_simple_loss, torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("prediction_type", ["x0", "v", "score"])
def test_p2_is_rejected_for_non_epsilon_prediction(
    prediction_type: str,
) -> None:
    process = gaussian_process()

    with pytest.raises(ValueError, match="requires epsilon"):
        GaussianDenoisingTrainingStrategy(
            nn.Linear(1, 1),
            process,
            MSEObjective(),
            prediction_type=cast(Any, prediction_type),
            loss_weighting=GaussianLossWeightingConfig(
                name="p2",
                k=1.0,
                gamma=1.0,
            ),
        )


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ({"name": "p2", "k": 0.0}, ValueError, "greater than zero"),
        ({"name": "p2", "k": float("nan")}, ValueError, "finite"),
        ({"name": "p2", "gamma": -1.0}, ValueError, "non-negative"),
        ({"name": "p2", "gamma": float("inf")}, ValueError, "finite"),
        ({"name": "constant", "gamma": 1.0}, ValueError, "unknown"),
        ({"name": "other"}, ValueError, "constant or p2"),
    ],
)
def test_loss_weighting_parser_fails_closed(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        parse_gaussian_loss_weighting(value, path="training.params.loss_weighting")


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ({"mode": "other"}, ValueError, "fixed or learned_range"),
        ({"mode": "fixed", "loss": "vb"}, ValueError, "unknown"),
        (
            {"mode": "learned_range"},
            ValueError,
            "rescaled_variational_bound",
        ),
    ],
)
def test_variance_parser_fails_closed(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        parse_gaussian_variance(value, path="training.params.variance")


def test_learned_range_endpoints_and_extrapolation_use_selected_pair_bounds() -> None:
    process = gaussian_process(4)
    source_times = torch.tensor([2, 4])
    target_times = torch.tensor([1, 2])
    shape = torch.Size((2, 1, 2, 2))
    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        shape,
    )

    lower = learned_range_log_variance(
        process,
        source_times,
        target_times,
        -torch.ones(shape),
    )
    upper = learned_range_log_variance(
        process,
        source_times,
        target_times,
        torch.ones(shape),
    )
    extrapolated = learned_range_log_variance(
        process,
        source_times,
        target_times,
        torch.full(shape, 3.0),
    )

    assert torch.equal(lower, bounds.lower.expand(shape))
    assert torch.equal(upper, bounds.upper.expand(shape))
    assert torch.all(extrapolated > upper)


def test_variational_bound_does_not_backpropagate_into_mean_head() -> None:
    process = gaussian_process(4)
    clean = torch.full((2, 1, 2, 2), 0.25)
    noise = torch.full_like(clean, 0.5)
    state_times = torch.tensor([2, 4])
    noisy, _ = process.sample_marginal(clean, state_times, noise=noise)
    mean_head = torch.zeros_like(clean, requires_grad=True)
    variance_head = torch.zeros_like(clean, requires_grad=True)

    result = compute_gaussian_training_loss(
        objective=MSEObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=torch.cat((mean_head, variance_head), dim=1),
        prediction_type="epsilon",
        variance=GaussianVarianceConfig(
            mode="learned_range",
            loss="rescaled_variational_bound",
        ),
        loss_weighting=GaussianLossWeightingConfig(),
    )

    assert result.per_sample_variational_bound is not None
    result.per_sample_variational_bound.sum().backward()
    assert mean_head.grad is None or not torch.any(mean_head.grad != 0)
    assert variance_head.grad is not None
    assert torch.any(variance_head.grad != 0)


def test_p2_changes_only_the_simple_term_not_variational_bound() -> None:
    process = gaussian_process(4)
    clean = torch.zeros(2, 1, 2, 2)
    noise = torch.ones_like(clean)
    state_times = torch.tensor([1, 4])
    noisy, _ = process.sample_marginal(clean, state_times, noise=noise)
    raw = torch.zeros(2, 2, 2, 2)
    variance = GaussianVarianceConfig(
        mode="learned_range",
        loss="rescaled_variational_bound",
    )

    baseline = compute_gaussian_training_loss(
        objective=MSEObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw,
        prediction_type="epsilon",
        variance=variance,
        loss_weighting=GaussianLossWeightingConfig(),
    )
    weighted = compute_gaussian_training_loss(
        objective=MSEObjective(),
        process=process,
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw,
        prediction_type="epsilon",
        variance=variance,
        loss_weighting=GaussianLossWeightingConfig(
            name="p2",
            k=1.0,
            gamma=1.0,
        ),
    )

    assert baseline.per_sample_variational_bound is not None
    assert weighted.per_sample_variational_bound is not None
    assert torch.equal(
        baseline.per_sample_variational_bound,
        weighted.per_sample_variational_bound,
    )
    assert baseline.per_sample_weighted_simple_loss is not None
    assert weighted.per_sample_weighted_simple_loss is not None
    assert not torch.equal(
        baseline.per_sample_weighted_simple_loss,
        weighted.per_sample_weighted_simple_loss,
    )


class RecordingGaussianModel(nn.Module):
    """Record model timesteps while predicting epsilon."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.seen_times: torch.Tensor | None = None

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_times = model_time.detach().clone()
        return torch.zeros_like(state) + self.offset


def test_training_maps_public_state_one_to_model_timestep_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = gaussian_process(4)
    model = RecordingGaussianModel()
    strategy = GaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
    )

    def fixed_state_times(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return torch.tensor([1, 4])

    monkeypatch.setattr(torch, "randint", fixed_state_times)
    strategy.training_step(torch.zeros(2, 1, 2, 2))

    assert model.seen_times is not None
    assert torch.equal(model.seen_times, torch.tensor([0, 3]))
