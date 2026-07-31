"""Validate P2 as a concrete Gaussian TrainingStrategy and Builder."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training import (
    MSEObjective,
    P2GaussianDenoisingTrainingBuilder,
    P2GaussianDenoisingTrainingStrategy,
)
from stochaflow.training.builder import TrainingBuilderContext
from stochaflow.utils.config import ComponentConfig


class ZeroDenoiser(nn.Module):
    """Return an epsilon prediction with one trainable scalar dependency."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        return torch.zeros_like(state) + self.offset


def gaussian_process() -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"k": True}, TypeError, "P2 k must be numeric"),
        ({"k": 0.0}, ValueError, "P2 k must be greater than zero"),
        ({"k": float("inf")}, ValueError, "P2 k must be finite"),
        ({"gamma": False}, TypeError, "P2 gamma must be numeric"),
        ({"gamma": -1.0}, ValueError, "P2 gamma must be non-negative"),
        ({"gamma": float("nan")}, ValueError, "P2 gamma must be finite"),
    ],
)
def test_p2_strategy_validates_its_private_parameters(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        P2GaussianDenoisingTrainingStrategy(
            ZeroDenoiser(),
            gaussian_process(),
            MSEObjective(),
            **kwargs,  # type: ignore[arg-type]
        )


def test_p2_strategy_fixes_epsilon_prediction() -> None:
    strategy = P2GaussianDenoisingTrainingStrategy(
        ZeroDenoiser(),
        gaussian_process(),
        MSEObjective(),
    )

    assert strategy.prediction_type == "epsilon"


def test_p2_builder_rejects_old_loss_weighting_declaration() -> None:
    process = gaussian_process()
    context = TrainingBuilderContext(
        params={
            "loss_weighting": {
                "name": "p2",
                "params": {"k": 1.0, "gamma": 1.0},
            }
        },
        primary_model=ZeroDenoiser(),
        process=process,
        objective=MSEObjective(),
        model_factory=_unexpected_factory,
        objective_factory=_unexpected_factory,
    )

    with pytest.raises(ValueError, match=r"unknown .* loss_weighting"):
        P2GaussianDenoisingTrainingBuilder(context).build()


def test_p2_builder_records_concrete_strategy_identity_in_config() -> None:
    process = gaussian_process()
    context = TrainingBuilderContext(
        params={"k": 2.0, "gamma": 0.5},
        primary_model=ZeroDenoiser(),
        process=process,
        objective=MSEObjective(reduction="sum"),
        model_factory=_unexpected_factory,
        objective_factory=_unexpected_factory,
    )

    plan = P2GaussianDenoisingTrainingBuilder(context).build()

    assert isinstance(plan.strategy, P2GaussianDenoisingTrainingStrategy)
    assert plan.strategy.k == 2.0
    assert plan.strategy.gamma == 0.5
    assert plan.inference_recipe is not None
    assert plan.inference_recipe.contract["prediction_type"] == "epsilon"


def _unexpected_factory(config: ComponentConfig) -> nn.Module:
    raise AssertionError(f"unexpected factory call: {config}")
