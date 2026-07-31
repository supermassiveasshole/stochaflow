"""Tests for reusable Objective capabilities and validation."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from stochaflow.training.objectives import (
    MSEObjective,
    compute_objective,
    validate_per_sample_loss,
)


class CountingPerSampleObjective(nn.Module):
    """Objective that records scalar and optional diagnostic evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.per_sample_calls = 0

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_calls += 1
        return (prediction - target).abs().mean()

    def per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        self.per_sample_calls += 1
        return (prediction - target).abs().flatten(1).mean(dim=1)

class ScalarObjective(nn.Module):
    """Objective exposing only the ordinary scalar contract."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (prediction - target).abs().sum()


@pytest.mark.parametrize(
    ("reduction", "expected_per_sample", "expected_scalar"),
    [
        ("mean", torch.tensor([2.0, 10.0]), 6.0),
        ("sum", torch.tensor([4.0, 20.0]), 24.0),
    ],
)
def test_mse_objective_preserves_scalar_and_per_sample_reduction_semantics(
    reduction: str,
    expected_per_sample: torch.Tensor,
    expected_scalar: float,
) -> None:
    objective = MSEObjective(reduction=reduction)
    prediction = torch.tensor([[0.0, 2.0], [1.0, 5.0]])
    target = torch.tensor([[0.0, 0.0], [3.0, 1.0]])

    per_sample = objective.per_sample_loss(prediction, target)

    assert torch.equal(per_sample, expected_per_sample)
    assert objective(prediction, target).item() == pytest.approx(expected_scalar)


def test_compute_objective_preserves_forward_as_the_generic_scalar_authority() -> None:
    objective = CountingPerSampleObjective()
    prediction = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    target = torch.zeros_like(prediction)

    loss, per_sample = compute_objective(objective, prediction, target)

    assert loss.item() == pytest.approx(3.0)
    assert per_sample is not None
    assert torch.equal(per_sample, torch.tensor([2.0, 4.0]))
    assert objective.forward_calls == 1
    assert objective.per_sample_calls == 1


def test_compute_objective_preserves_scalar_only_objective_support() -> None:
    prediction = torch.tensor([[1.0], [3.0]])
    target = torch.zeros_like(prediction)

    loss, per_sample = compute_objective(ScalarObjective(), prediction, target)

    assert loss.item() == pytest.approx(4.0)
    assert per_sample is None


@pytest.mark.parametrize(
    ("value_factory", "error_type", "match"),
    [
        (lambda: [1.0, 2.0], TypeError, "must return a Tensor"),
        (
            lambda: torch.tensor([1, 2]),
            TypeError,
            "must return a floating-point Tensor",
        ),
        (
            lambda: torch.tensor(1.0),
            ValueError,
            "must match the prediction batch",
        ),
        (
            lambda: torch.ones(2, 1),
            ValueError,
            "must match the prediction batch",
        ),
        (
            lambda: torch.ones(3),
            ValueError,
            "must match the prediction batch",
        ),
        (
            lambda: torch.ones(2, device="meta"),
            ValueError,
            "must be on the prediction device",
        ),
    ],
)
def test_validate_per_sample_loss_rejects_invalid_outputs(
    value_factory: Callable[[], object],
    error_type: type[Exception],
    match: str,
) -> None:
    prediction = torch.zeros(2, 3)

    with pytest.raises(error_type, match=match):
        validate_per_sample_loss(value_factory(), prediction=prediction)


def test_validate_per_sample_loss_allows_autocast_promoted_dtype() -> None:
    prediction = torch.zeros(2, 3, dtype=torch.bfloat16)
    per_sample_loss = torch.ones(2, dtype=torch.float32)

    result = validate_per_sample_loss(
        per_sample_loss,
        prediction=prediction,
    )

    assert result is per_sample_loss
