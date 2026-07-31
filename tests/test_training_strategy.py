"""Tests for step output and built-in strategy semantics."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import PredictionType
from stochaflow.training import (
    ClassConditionalGaussianDenoisingTrainingStrategy,
    GaussianDenoisingTrainingBuilder,
    GaussianDenoisingTrainingStrategy,
    MetricChannelProvider,
    MetricUpdate,
    MSEObjective,
    SupervisedTrainingStrategy,
    TrainStepOutput,
    gaussian_training_target,
    loss_aggregation_weight_to_float,
    validate_train_step_output,
)
from stochaflow.training.builder import TrainingBuilderContext
from stochaflow.training.gaussian_loss import (
    GaussianLossComposer,
    build_gaussian_loss_composer,
)
from stochaflow.training.gaussian_variance import parse_gaussian_variance
from stochaflow.training.gaussian_weighting import (
    build_gaussian_simple_loss_weighting,
)
from stochaflow.utils.config import ComponentConfig


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
        (
            TrainStepOutput(
                torch.tensor(1.0),
                metric_updates={"": MetricUpdate()},
            ),
            "metric update channels",
        ),
        (
            TrainStepOutput(
                torch.tensor(1.0),
                metric_updates={"valid.channel": cast(MetricUpdate, object())},
            ),
            "must be MetricUpdate",
        ),
        (
            TrainStepOutput(torch.tensor(1.0), loss_aggregation_weight=True),
            "real numeric scalar",
        ),
        (
            TrainStepOutput(torch.tensor(1.0), loss_aggregation_weight=-1),
            "non-negative",
        ),
        (
            TrainStepOutput(torch.tensor(1.0), loss_aggregation_weight=float("inf")),
            "finite",
        ),
        (
            TrainStepOutput(
                torch.tensor(1.0),
                loss_aggregation_weight=torch.ones(2),
            ),
            "scalar Tensor",
        ),
        (
            TrainStepOutput(
                torch.tensor(1.0),
                loss_aggregation_weight=torch.tensor(1.0 + 2.0j),
            ),
            "real numeric",
        ),
    ],
)
def test_train_step_output_validation(value: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_train_step_output(value)


def test_train_step_output_accepts_zero_and_detached_tensor_weights() -> None:
    tensor_weight = torch.tensor(3.0, requires_grad=True)
    output = TrainStepOutput(
        loss=torch.tensor(2.0, requires_grad=True),
        metric_updates={"valid.channel": MetricUpdate(args=(torch.tensor(1.0),))},
        loss_aggregation_weight=tensor_weight,
    )

    validated = validate_train_step_output(output)

    assert validated is output
    assert loss_aggregation_weight_to_float(tensor_weight) == 3.0
    assert loss_aggregation_weight_to_float(0) == 0.0
    assert validated.loss.item() == 2.0
    assert validated.loss.requires_grad
    validated.loss.backward()
    assert validated.loss.grad is not None
    assert validated.loss.grad.item() == 1.0


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


class LearnedVarianceGaussianModel(nn.Module):
    """Return epsilon and learned-range interpolation heads."""

    def __init__(
        self,
        *,
        mean_value: float = 0.0,
        variance_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.mean_value = mean_value
        self.variance_value = variance_value
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        mean = torch.full_like(state, self.mean_value) + self.offset
        variance = torch.full_like(state, self.variance_value)
        return torch.cat((mean, variance), dim=1)


class DeclaredLayoutGaussianModel(nn.Module):
    """Declare static channel counts for builder-boundary validation."""

    def __init__(self, *, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        return state + self.offset


class PerfectConditionalTargetModel(PerfectTargetModel):
    @property
    def num_classes(self) -> int:
        return 3

    @property
    def null_class_id(self) -> int:
        return self.num_classes

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        assert class_labels.shape == model_time.shape
        return self.forward(state, model_time)


def _gaussian_loss_composer(
    process: DiscreteGaussianProcess,
    objective: nn.Module,
    *,
    prediction_type: PredictionType = "epsilon",
    variance: object = None,
    loss_weighting: object = None,
) -> GaussianLossComposer:
    return build_gaussian_loss_composer(
        objective=objective,
        process=process,
        prediction_type=prediction_type,
        variance=parse_gaussian_variance(
            variance,
            path="test Gaussian variance",
        ),
        loss_weighting=build_gaussian_simple_loss_weighting(
            loss_weighting,
            path="test Gaussian loss weighting",
        ),
        path="test Gaussian training policy",
    )


def test_supervised_strategy_declares_and_emits_metric_channel() -> None:
    model = nn.Linear(2, 1, bias=False)
    objective = MSEObjective()
    strategy = SupervisedTrainingStrategy(model, objective)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    targets = torch.tensor([[0.5], [1.0], [1.5]])

    output = strategy.training_step((inputs, targets))

    assert isinstance(strategy, MetricChannelProvider)
    assert strategy.metric_channels == frozenset(
        ("supervised.prediction_target",)
    )
    prediction, metric_target = output.metric_updates[
        "supervised.prediction_target"
    ].args
    assert isinstance(prediction, torch.Tensor)
    assert metric_target is targets
    assert output.loss_aggregation_weight == 3
    assert torch.equal(output.loss, (prediction - targets).square().mean())


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_gaussian_strategy_supports_all_prediction_targets(
    prediction_type: PredictionType,
) -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = PerfectTargetModel(process, prediction_type, clean_value=0.25)
    objective = MSEObjective()
    strategy = GaussianDenoisingTrainingStrategy(
        model,
        process,
        _gaussian_loss_composer(
            process,
            objective,
            prediction_type=prediction_type,
        ),
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
    assert isinstance(strategy, MetricChannelProvider)
    assert strategy.metric_channels == frozenset(
        (
            "gaussian.prediction_target",
            "gaussian.clean_reconstruction",
        )
    )
    model_output, target = output.metric_updates[
        "gaussian.prediction_target"
    ].args
    predicted_clean, clean = output.metric_updates[
        "gaussian.clean_reconstruction"
    ].args
    assert torch.allclose(model_output, target)
    assert torch.allclose(predicted_clean, clean)
    assert output.loss_aggregation_weight == 2


def test_class_conditional_gaussian_strategy_emits_shared_gaussian_channels() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = PerfectConditionalTargetModel(process, "epsilon", clean_value=0.25)
    objective = MSEObjective()
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        _gaussian_loss_composer(process, objective),
    )
    clean = torch.full((2, 1, 2, 2), 0.25)
    labels = torch.tensor([0, 2])

    output = strategy.evaluation_step((clean, {"class_label": labels}))

    assert isinstance(strategy, MetricChannelProvider)
    assert strategy.metric_channels == frozenset(
        (
            "gaussian.prediction_target",
            "gaussian.clean_reconstruction",
        )
    )
    model_output, target = output.metric_updates[
        "gaussian.prediction_target"
    ].args
    predicted_clean, metric_clean = output.metric_updates[
        "gaussian.clean_reconstruction"
    ].args
    assert torch.allclose(model_output, target)
    assert torch.allclose(predicted_clean, metric_clean)
    assert metric_clean is clean
    assert output.loss_aggregation_weight == 2


def test_gaussian_strategy_rejects_unhandled_conditions() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )
    objective = MSEObjective()
    strategy = GaussianDenoisingTrainingStrategy(
        PerfectTargetModel(process, "epsilon", 0.0),
        process,
        _gaussian_loss_composer(process, objective),
    )

    with pytest.raises(TypeError, match="custom TrainingBuilder"):
        strategy.training_step(
            (torch.zeros(1, 1, 2, 2), {"low_res": torch.zeros(1, 1, 1, 1)})
        )


def test_unconditional_builder_composes_p2_learned_range_recipe() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = LearnedVarianceGaussianModel(
        mean_value=0.25,
        variance_value=-0.75,
    )
    objective = MSEObjective()

    def model_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected model")

    def objective_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected objective")

    builder = GaussianDenoisingTrainingBuilder(
        TrainingBuilderContext(
            params={
                "prediction_type": "epsilon",
                "variance": {
                    "mode": "learned_range",
                    "loss": "rescaled_variational_bound",
                },
                "loss_weighting": {
                    "name": "p2",
                    "params": {"k": 1.0, "gamma": 1.0},
                },
            },
            primary_model=model,
            process=process,
            objective=objective,
            model_factory=model_factory,
            objective_factory=objective_factory,
        )
    )

    plan = builder.build()
    output = plan.strategy.training_step(torch.zeros(2, 1, 2, 2))

    assert plan.inference_recipe is not None
    assert plan.inference_recipe.contract == {
        "prediction_type": "epsilon",
        "variance": {"mode": "learned_range"},
    }
    assert torch.isfinite(output.loss)
    assert output.diagnostics["per_sample_loss"].shape == (2,)
    metric_prediction, metric_target = output.metric_updates[
        "gaussian.prediction_target"
    ].args
    metric_clean, clean = output.metric_updates[
        "gaussian.clean_reconstruction"
    ].args
    assert isinstance(metric_prediction, torch.Tensor)
    assert isinstance(metric_target, torch.Tensor)
    assert metric_prediction.shape == metric_target.shape == (2, 1, 2, 2)
    assert metric_prediction.is_contiguous()
    assert metric_target.is_contiguous()
    assert torch.allclose(
        metric_prediction,
        torch.full_like(metric_prediction, 0.25),
    )
    assert torch.allclose(
        metric_target,
        torch.full_like(metric_target, 0.5),
    )
    assert isinstance(metric_clean, torch.Tensor)
    assert isinstance(clean, torch.Tensor)
    assert metric_clean.shape == clean.shape == (2, 1, 2, 2)
    assert output.loss_aggregation_weight == 2
    assert output.diagnostics["timestep_loss_weight"].sum().item() < 2.0


@pytest.mark.parametrize(
    ("params", "out_channels", "message"),
    [
        ({}, 2, "fixed Gaussian variance requires 1 output channels"),
        (
            {
                "variance": {
                    "mode": "learned_range",
                    "loss": "rescaled_variational_bound",
                }
            },
            1,
            "learned_range Gaussian variance requires 2 output channels",
        ),
    ],
)
def test_unconditional_builder_preflights_declared_model_output_layout(
    params: dict[str, object],
    out_channels: int,
    message: str,
) -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = DeclaredLayoutGaussianModel(
        in_channels=1,
        out_channels=out_channels,
    )
    objective = MSEObjective()

    def model_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected model")

    def objective_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected objective")

    builder = GaussianDenoisingTrainingBuilder(
        TrainingBuilderContext(
            params=params,
            primary_model=model,
            process=process,
            objective=objective,
            model_factory=model_factory,
            objective_factory=objective_factory,
        )
    )

    with pytest.raises(ValueError, match=message):
        builder.build()


def test_unconditional_builder_accepts_declared_learned_range_layout() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = DeclaredLayoutGaussianModel(in_channels=1, out_channels=2)
    objective = MSEObjective()

    def model_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected model")

    def objective_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected objective")

    plan = GaussianDenoisingTrainingBuilder(
        TrainingBuilderContext(
            params={
                "variance": {
                    "mode": "learned_range",
                    "loss": "rescaled_variational_bound",
                }
            },
            primary_model=model,
            process=process,
            objective=objective,
            model_factory=model_factory,
            objective_factory=objective_factory,
        )
    ).build()

    assert plan.primary_model is model


def test_unconditional_builder_retains_runtime_layout_check_for_opaque_model() -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    model = LearnedVarianceGaussianModel()
    objective = MSEObjective()

    def model_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected model")

    def objective_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("Gaussian builder must preserve the injected objective")

    plan = GaussianDenoisingTrainingBuilder(
        TrainingBuilderContext(
            params={},
            primary_model=model,
            process=process,
            objective=objective,
            model_factory=model_factory,
            objective_factory=objective_factory,
        )
    ).build()

    with pytest.raises(ValueError, match="output must match the state shape"):
        plan.strategy.training_step(torch.zeros(2, 1, 2, 2))


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
            [0.0, 0.0],
            torch.ones(1, 2),
            "must be Tensors",
        ),
        (
            torch.tensor(0.0),
            torch.tensor(1.0),
            "batch dimension",
        ),
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
    clean: object,
    noise: object,
    message: str,
) -> None:
    process = DeterministicGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )

    with pytest.raises((TypeError, ValueError), match=message):
        gaussian_training_target(
            process,
            clean=clean,  # type: ignore[arg-type]
            noise=noise,  # type: ignore[arg-type]
            state_times=torch.tensor([1]),
            prediction_type="epsilon",
        )
