"""Tests for step output and built-in strategy semantics."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import PredictionType
from stochaflow.training import (
    GaussianDenoisingTrainingStrategy,
    MSEObjective,
    TrainStepOutput,
    gaussian_training_target,
    validate_train_step_output,
)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (object(), "must return TrainStepOutput"),
        (TrainStepOutput(torch.ones(2)), "scalar Tensor"),
        (TrainStepOutput(torch.tensor(1)), "floating point"),
        (
            TrainStepOutput(torch.tensor(1.0), metrics={"bad": torch.ones(2)}),
            "scalar Tensor",
        ),
        (
            TrainStepOutput(
                torch.tensor(1.0),
                metrics={"bad": torch.tensor(True)},
            ),
            "real numeric",
        ),
        (
            TrainStepOutput(
                torch.tensor(1.0),
                metrics={"bad": torch.tensor(1.0 + 2.0j)},
            ),
            "real numeric",
        ),
    ],
)
def test_train_step_output_validation(value: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_train_step_output(value)


class DeterministicGaussianProcess(DiscreteGaussianProcess):
    def sample_marginal(self, clean, state_times, **kwargs):
        noise = torch.full_like(clean, 0.5)
        return super().sample_marginal(clean, state_times, noise=noise)


class PerfectTargetModel(nn.Module):
    def __init__(
        self,
        process: DeterministicGaussianProcess,
        prediction_type: PredictionType,
        clean_value: float,
    ) -> None:
        super().__init__()
        self.process = process
        self.prediction_type = prediction_type
        self.clean_value = clean_value
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor, model_time: torch.Tensor) -> torch.Tensor:
        state_times = model_time + self.process.clean_time + 1
        clean = torch.full_like(state, self.clean_value)
        noise = torch.full_like(state, 0.5)
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = clean
        else:
            scales = self.process.marginal_scales(state_times, state.size())
            if self.prediction_type == "v":
                target = scales.signal * noise - scales.noise * clean
            else:
                target = -noise / scales.noise
        return target + self.offset * 0.0


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_gaussian_strategy_supports_all_prediction_targets(
    prediction_type: PredictionType,
) -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = PerfectTargetModel(process, prediction_type, clean_value=0.25)
    strategy = GaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
        prediction_type=prediction_type,
    )

    output = strategy.training_step(torch.full((2, 1, 2, 2), 0.25))

    assert output.loss.item() == pytest.approx(0.0)
    assert torch.allclose(
        output.diagnostics["predicted_noise"],
        output.diagnostics["target_noise"],
    )
    assert torch.allclose(
        output.diagnostics["predicted_clean"],
        output.diagnostics["clean_samples"],
    )


def test_gaussian_strategy_rejects_unhandled_conditions() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )
    strategy = GaussianDenoisingTrainingStrategy(
        PerfectTargetModel(process, "epsilon", 0.0),
        process,
        MSEObjective(),
    )

    with pytest.raises(TypeError, match="custom TrainingBuilder"):
        strategy.training_step(
            (torch.zeros(1, 1, 2, 2), {"low_res": torch.zeros(1, 1, 1, 1)})
        )


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_gaussian_training_target_is_reusable_by_custom_strategies(
    prediction_type: PredictionType,
) -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    clean = torch.full((2, 3), 0.25)
    noise = torch.full_like(clean, 0.5)
    state_times = torch.tensor([1, 4])
    scales = process.marginal_scales(state_times, clean.size())
    expected = {
        "epsilon": noise,
        "x0": clean,
        "v": scales.signal * noise - scales.noise * clean,
        "score": -noise / scales.noise,
    }[prediction_type]

    target = gaussian_training_target(
        process,
        clean=clean,
        noise=noise,
        state_times=state_times,
        prediction_type=prediction_type,
    )

    assert torch.equal(target, expected)


def test_gaussian_training_target_rejects_clean_state_time() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )

    with pytest.raises(ValueError, match="source state times"):
        gaussian_training_target(
            process,
            clean=torch.zeros(1, 2),
            noise=torch.ones(1, 2),
            state_times=torch.tensor([0]),
            prediction_type="epsilon",
        )


@pytest.mark.parametrize(
    ("clean", "noise", "message"),
    [
        (
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.long),
            "floating-point",
        ),
        (
            torch.zeros(1, 2, dtype=torch.float32),
            torch.ones(1, 2, dtype=torch.float64),
            "share the clean state dtype",
        ),
    ],
)
def test_gaussian_training_target_rejects_invalid_numeric_contract(
    clean: torch.Tensor,
    noise: torch.Tensor,
    message: str,
) -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )

    with pytest.raises((TypeError, ValueError), match=message):
        gaussian_training_target(
            process,
            clean=clean,
            noise=noise,
            state_times=torch.tensor([1]),
            prediction_type="epsilon",
        )
