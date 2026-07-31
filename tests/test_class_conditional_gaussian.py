"""Tests for built-in class-conditional Gaussian training."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import PredictionType
from stochaflow.training.builder import TrainingBuilderContext
from stochaflow.training.class_conditional_gaussian import (
    ClassConditionalGaussianDenoisingTrainingBuilder,
    ClassConditionalGaussianDenoisingTrainingStrategy,
    ClassConditionalGaussianDiagnosticSemantics,
)
from stochaflow.training.gaussian_loss import (
    GaussianLossWeightingConfig,
    GaussianVarianceConfig,
)
from stochaflow.training.objectives import MSEObjective
from stochaflow.utils.config import ComponentConfig


class DeterministicConditionalGaussianProcess(DiscreteGaussianProcess):
    """Use fixed noise so exact target predictions are testable."""

    def sample_marginal(
        self,
        clean: torch.Tensor,
        state_times: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del noise, generator
        fixed_noise = torch.full_like(clean, 0.5)
        return super().sample_marginal(
            clean,
            state_times,
            noise=fixed_noise,
        )


class ToyClassConditionalDenoiser(nn.Module):
    """Independent structural implementation of the conditional capability."""

    def __init__(
        self,
        process: DeterministicConditionalGaussianProcess,
        prediction_type: PredictionType,
        *,
        num_classes: int = 3,
        null_class_id: int | None = None,
    ) -> None:
        super().__init__()
        self.process = process
        self.prediction_type = prediction_type
        self._num_classes = num_classes
        self._null_class_id = num_classes if null_class_id is None else null_class_id
        self.offset = nn.Parameter(torch.zeros(()))
        self.seen_labels: list[torch.Tensor] = []

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def null_class_id(self) -> int:
        return self._null_class_id

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        self.seen_labels.append(class_labels.detach().clone())
        state_times = model_time + self.process.clean_time + 1
        clean = torch.full_like(state, 0.25)
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


class PrevalidatedToyClassConditionalDenoiser(ToyClassConditionalDenoiser):
    """Expose and record the optional certified-value prediction path."""

    def __init__(
        self,
        process: DeterministicConditionalGaussianProcess,
        prediction_type: PredictionType,
    ) -> None:
        super().__init__(process, prediction_type)
        self.prevalidated_calls = 0

    def predict_prevalidated_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        self.prevalidated_calls += 1
        return self.predict_class_conditioned(
            state,
            model_time,
            class_labels,
        )


class LearnedVarianceToyClassConditionalDenoiser(ToyClassConditionalDenoiser):
    """Return a prediction head plus learned-range interpolation values."""

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        mean = super().predict_class_conditioned(
            state,
            model_time,
            class_labels,
        )
        return torch.cat((mean, torch.zeros_like(mean)), dim=1)


class DeclaredLayoutToyClassConditionalDenoiser(
    ToyClassConditionalDenoiser,
):
    """Add a static output-layout declaration to the independent denoiser."""

    def __init__(
        self,
        process: DeterministicConditionalGaussianProcess,
        *,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(process, "epsilon")
        self.in_channels = in_channels
        self.out_channels = out_channels


class NonConditionalModel(nn.Module):
    """Parameter-bearing model that intentionally lacks the capability."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.weight


def _process() -> DeterministicConditionalGaussianProcess:
    return DeterministicConditionalGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )


def _batch(
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if labels is None:
        labels = torch.tensor([0, 2])
    return torch.full((2, 1, 2, 2), 0.25), {"class_label": labels}


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_conditional_strategy_supports_all_gaussian_targets(
    prediction_type: PredictionType,
) -> None:
    process = _process()
    model = ToyClassConditionalDenoiser(process, prediction_type)
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
        prediction_type=prediction_type,
    )

    output = strategy.training_step(_batch())

    assert output.loss.item() == pytest.approx(0.0)
    assert torch.equal(model.seen_labels[-1], torch.tensor([0, 2]))
    assert torch.allclose(
        output.diagnostics["predicted_noise"],
        output.diagnostics["target_noise"],
    )
    assert torch.allclose(
        output.diagnostics["predicted_clean"],
        output.diagnostics["clean_samples"],
    )
    assert isinstance(strategy, ClassConditionalGaussianDiagnosticSemantics)


def test_conditional_strategy_uses_optional_prevalidated_model_path() -> None:
    process = _process()
    model = PrevalidatedToyClassConditionalDenoiser(process, "epsilon")
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
    )

    strategy.training_step(_batch())

    assert model.prevalidated_calls == 1


def test_training_applies_dropout_but_evaluation_never_drops_conditions() -> None:
    process = _process()
    model = ToyClassConditionalDenoiser(process, "epsilon")
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
        condition_dropout=1.0,
    )

    training = strategy.training_step(_batch())
    evaluation = strategy.evaluation_step(_batch())

    assert torch.equal(model.seen_labels[-2], torch.tensor([3, 3]))
    assert torch.equal(model.seen_labels[-1], torch.tensor([0, 2]))
    assert torch.all(training.diagnostics["condition_dropout_mask"])
    assert not torch.any(evaluation.diagnostics["condition_dropout_mask"])


def test_learned_range_p2_metrics_use_prediction_head_and_batch_weight() -> None:
    process = _process()
    model = LearnedVarianceToyClassConditionalDenoiser(process, "epsilon")
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
        condition_dropout=1.0,
        variance=GaussianVarianceConfig(
            mode="learned_range",
            loss="rescaled_variational_bound",
        ),
        loss_weighting=GaussianLossWeightingConfig(
            name="p2",
            k=1.0,
            gamma=1.0,
        ),
    )

    training = strategy.training_step(_batch())
    evaluation = strategy.evaluation_step(_batch())

    assert torch.equal(model.seen_labels[-2], torch.tensor([3, 3]))
    assert torch.equal(model.seen_labels[-1], torch.tensor([0, 2]))
    for output in (training, evaluation):
        prediction, target = output.metric_updates[
            "gaussian.prediction_target"
        ].args
        reconstructed, clean = output.metric_updates[
            "gaussian.clean_reconstruction"
        ].args
        assert isinstance(prediction, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert prediction.shape == target.shape == (2, 1, 2, 2)
        assert prediction.is_contiguous()
        assert target.is_contiguous()
        assert torch.allclose(prediction, target)
        assert isinstance(reconstructed, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert reconstructed.shape == clean.shape == (2, 1, 2, 2)
        assert output.loss_aggregation_weight == 2
        assert output.diagnostics["timestep_loss_weight"].sum().item() < 2.0


def test_training_condition_dropout_is_sample_aligned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _process()
    model = ToyClassConditionalDenoiser(process, "epsilon")
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        MSEObjective(),
        condition_dropout=0.5,
    )

    def fixed_dropout_draw(
        shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        assert tuple(shape) == (2,)
        assert device == torch.device("cpu")
        return torch.tensor([0.25, 0.75], device=device)

    monkeypatch.setattr(torch, "rand", fixed_dropout_draw)

    output = strategy.training_step(_batch())

    assert torch.equal(model.seen_labels[-1], torch.tensor([3, 2]))
    assert torch.equal(
        output.diagnostics["condition_dropout_mask"],
        torch.tensor([True, False]),
    )


@pytest.mark.parametrize(
    ("batch", "error", "message"),
    [
        (
            torch.zeros(2, 1),
            TypeError,
            "expects",
        ),
        (
            (torch.zeros(2, 1), {}),
            ValueError,
            "only 'class_label'",
        ),
        (
            (torch.zeros(2, 1), {"class_label": [0, 1]}),
            TypeError,
            "must be a Tensor",
        ),
        (
            _batch(torch.tensor([[0], [1]])),
            ValueError,
            "1D Tensor",
        ),
        (
            _batch(torch.tensor([0.0, 1.0])),
            TypeError,
            "contain integers",
        ),
        (
            _batch(torch.tensor([0])),
            ValueError,
            "match the batch",
        ),
        (
            _batch(torch.tensor([-1, 1])),
            ValueError,
            r"\[0, 3\)",
        ),
        (
            _batch(torch.tensor([0, 3])),
            ValueError,
            r"\[0, 3\)",
        ),
    ],
)
def test_conditional_strategy_strictly_validates_labels(
    batch: Any,
    error: type[Exception],
    message: str,
) -> None:
    process = _process()
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        ToyClassConditionalDenoiser(process, "epsilon"),
        process,
        MSEObjective(),
    )

    with pytest.raises(error, match=message):
        strategy.training_step(batch)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "numeric"),
        (float("nan"), ValueError, "finite"),
        (-0.1, ValueError, r"\[0, 1\]"),
        (1.1, ValueError, r"\[0, 1\]"),
    ],
)
def test_condition_dropout_is_strictly_validated(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    process = _process()

    with pytest.raises(error, match=message):
        ClassConditionalGaussianDenoisingTrainingStrategy(
            ToyClassConditionalDenoiser(process, "epsilon"),
            process,
            MSEObjective(),
            condition_dropout=cast(float, value),
        )


def test_builder_validates_structural_model_capability_and_null_class() -> None:
    process = _process()
    objective = MSEObjective()

    with pytest.raises(TypeError, match="ClassConditionalDenoiser"):
        ClassConditionalGaussianDenoisingTrainingBuilder(
            _builder_context(NonConditionalModel(), process, objective)
        ).build()

    with pytest.raises(ValueError, match="null_class_id must equal num_classes"):
        ClassConditionalGaussianDenoisingTrainingBuilder(
            _builder_context(
                ToyClassConditionalDenoiser(
                    process,
                    "epsilon",
                    null_class_id=4,
                ),
                process,
                objective,
            )
        ).build()


def test_builder_preserves_injected_assets_and_private_configuration() -> None:
    process = _process()
    model = ToyClassConditionalDenoiser(process, "v")
    objective = MSEObjective()
    context = _builder_context(model, process, objective)
    context.params.update({"prediction_type": "v", "condition_dropout": 0.25})

    plan = ClassConditionalGaussianDenoisingTrainingBuilder(context).build()

    assert plan.primary_model is model
    assert plan.process is process
    assert plan.objective is objective
    assert isinstance(
        plan.strategy,
        ClassConditionalGaussianDenoisingTrainingStrategy,
    )
    assert plan.strategy.prediction_type == "v"
    assert plan.strategy.condition_dropout == pytest.approx(0.25)


def test_builder_composes_p2_learned_range_and_freezes_variance_recipe() -> None:
    process = _process()
    model = LearnedVarianceToyClassConditionalDenoiser(process, "epsilon")
    objective = MSEObjective()
    context = _builder_context(model, process, objective)
    context.params.update(
        {
            "prediction_type": "epsilon",
            "condition_dropout": 0.1,
            "variance": {
                "mode": "learned_range",
                "loss": "rescaled_variational_bound",
            },
            "loss_weighting": {"name": "p2", "k": 1.0, "gamma": 1.0},
        }
    )

    plan = ClassConditionalGaussianDenoisingTrainingBuilder(context).build()
    output = plan.strategy.training_step(_batch())

    assert plan.inference_recipe is not None
    assert plan.inference_recipe.contract == {
        "prediction_type": "epsilon",
        "variance": {"mode": "learned_range"},
    }
    assert torch.isfinite(output.loss)
    assert output.diagnostics["per_sample_variational_bound"].shape == (2,)
    assert output.diagnostics["timestep_loss_weight"].shape == (2,)


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
def test_conditional_builder_preflights_declared_model_output_layout(
    params: dict[str, object],
    out_channels: int,
    message: str,
) -> None:
    process = _process()
    model = DeclaredLayoutToyClassConditionalDenoiser(
        process,
        in_channels=1,
        out_channels=out_channels,
    )
    context = _builder_context(model, process, MSEObjective())
    context.params.update(params)

    with pytest.raises(ValueError, match=message):
        ClassConditionalGaussianDenoisingTrainingBuilder(context).build()


def test_conditional_builder_accepts_declared_learned_range_layout() -> None:
    process = _process()
    model = DeclaredLayoutToyClassConditionalDenoiser(
        process,
        in_channels=1,
        out_channels=2,
    )
    context = _builder_context(model, process, MSEObjective())
    context.params.update(
        {
            "variance": {
                "mode": "learned_range",
                "loss": "rescaled_variational_bound",
            }
        }
    )

    plan = ClassConditionalGaussianDenoisingTrainingBuilder(context).build()

    assert plan.primary_model is model


def test_conditional_builder_retains_runtime_layout_check_for_opaque_model() -> None:
    process = _process()
    model = LearnedVarianceToyClassConditionalDenoiser(process, "epsilon")
    context = _builder_context(model, process, MSEObjective())

    plan = ClassConditionalGaussianDenoisingTrainingBuilder(context).build()

    with pytest.raises(ValueError, match="fixed-variance Gaussian model output"):
        plan.strategy.training_step(_batch())


def _builder_context(
    model: nn.Module,
    process: DeterministicConditionalGaussianProcess,
    objective: nn.Module,
) -> TrainingBuilderContext:
    def model_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("conditional builder must not construct a model")

    def objective_factory(config: ComponentConfig) -> nn.Module:
        del config
        raise AssertionError("conditional builder must not construct an objective")

    return TrainingBuilderContext(
        params={},
        primary_model=model,
        process=process,
        objective=objective,
        model_factory=model_factory,
        objective_factory=objective_factory,
    )
