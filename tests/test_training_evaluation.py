"""Tests for Evaluation-backed checkpoint selection during training."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn
from torchmetrics.aggregation import MeanMetric

from stochaflow.evaluation import (
    EvaluationBuilder,
    EvaluationPlan,
    EvaluationSamplingCapability,
    EvaluationSamplingRequest,
    EvaluationStepOutput,
)
from stochaflow.metrics import MetricSpec, MetricUpdate
from stochaflow.sampling import (
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)
from stochaflow.scripts.epoch_validation import (
    EvaluationBackedEpochValidator,
    LiveEpochEvaluationSubject,
)
from stochaflow.training import (
    ExponentialMovingAverage,
    SupervisedTrainingStrategy,
    Trainer,
    TrainingPlan,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ComponentConfig,
    ValidationEvaluationConfig,
    ValidationEvaluationProtocolConfig,
)
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog
from stochaflow.utils.sampling_recipe import SamplingRecipe

LIVE_MEAN_METRIC = "tests.live-epoch-mean"
LIVE_SAMPLING_BUILDER = "tests.live-epoch-sampling"


class LiveEpochMeanMetric(MeanMetric):
    """Mean metric registered only for the live Evaluation tests."""


REGISTRIES.metrics.add(LIVE_MEAN_METRIC, LiveEpochMeanMetric)


class ScalarEvaluationModel(nn.Module):
    """One-parameter model that exposes raw/EMA identity in metric values."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class LiveModelSamplingBuilder(SamplingBuilder):
    """Generate scalar samples through the injected pinned model provider."""

    observed_weights: str | None = None

    def run(self) -> SamplingOutput:
        """Run the pinned model and expose which weight authority was used."""

        model, weights = self.context.model_provider.resolve("auto")
        type(self).observed_weights = weights
        inputs = torch.ones(
            (self.context.num_samples, *cast(tuple[int, ...], self.context.shape)),
            device=self.context.device,
        )
        samples = model(inputs)
        return SamplingOutput(
            batches=(
                SamplingBatch(
                    samples=samples,
                    num_samples=self.context.num_samples,
                ),
            ),
            metadata={"weights": weights},
        )


REGISTRIES.sampling_builders.add(
    LIVE_SAMPLING_BUILDER,
    LiveModelSamplingBuilder,
)


class LiveModelValueEvaluator:
    """Convert one opaque validation tensor into model-backed metric updates."""

    metric_channels = frozenset({"tests.live-values"})

    def __init__(self, model: nn.Module, *, fail: bool = False) -> None:
        self.model = model
        self.fail = fail
        self.position = 0

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        if self.fail:
            raise RuntimeError("injected live Evaluation failure")
        if not isinstance(batch, torch.Tensor):
            raise TypeError("live Evaluation test batch must be a Tensor")
        values = cast(torch.Tensor, self.model(batch))
        count = int(values.shape[0])
        sample_ids = tuple(
            f"live-sample-{index}"
            for index in range(self.position, self.position + count)
        )
        self.position += count
        return EvaluationStepOutput(
            num_examples=count,
            sample_ids=sample_ids,
            metric_update_groups=(
                {
                    "tests.live-values": MetricUpdate(
                        args=(values,),
                    )
                },
            ),
        )


class LiveModelEvaluationBuilder(EvaluationBuilder):
    """Compose the tiny live model evaluator from injected dependencies."""

    last_subject: LiveEpochEvaluationSubject | None = None

    def build(self) -> EvaluationPlan:
        model = cast(object, self.context.inference)
        if not isinstance(model, nn.Module):
            raise TypeError("live Evaluation requires an injected model")
        subject = cast(object, self.context.subject)
        if not isinstance(subject, LiveEpochEvaluationSubject):
            raise TypeError("live Evaluation subject has the wrong identity")
        type(self).last_subject = subject
        return EvaluationPlan(
            evaluator=LiveModelValueEvaluator(
                model,
                fail=bool(self.context.params.get("fail", False)),
            ),
            data=self.context.data,
            metric_specs=self.context.metric_specs,
            protocol=self.context.protocol,
            subject=self.context.subject,
            data_identity=self.context.data_identity,
            modules={"primary": model},
        )


class SamplePairEvaluator:
    """Pair validation references with samples from the injected capability."""

    metric_channels = frozenset({"tests.live-sample-pairs"})

    def __init__(self, sampling: EvaluationSamplingCapability) -> None:
        self.sampling = sampling
        self.position = 0

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("sample-pair validation batch must be a Tensor")
        count = int(batch.shape[0])
        output = self.sampling.execute(
            EvaluationSamplingRequest(
                options={},
                sampler=None,
                shape=tuple(batch.shape[1:]),
                num_samples=count,
                batch_size=count,
                seed=7,
                expected_recipe_name=LIVE_SAMPLING_BUILDER,
                expected_recipe_contract={},
            )
        )
        fake = cast(torch.Tensor, output.batches[0].samples)
        sample_ids = tuple(
            f"sample-pair-{index}"
            for index in range(self.position, self.position + count)
        )
        self.position += count
        return EvaluationStepOutput(
            num_examples=count,
            sample_ids=sample_ids,
            metric_update_groups=(
                {
                    "tests.live-sample-pairs": MetricUpdate(
                        args=(fake, batch),
                    )
                },
            ),
        )


class SamplingEvaluationBuilder(EvaluationBuilder):
    """Compose an evaluator that must invoke the live SamplingBuilder seam."""

    def build(self) -> EvaluationPlan:
        sampling = cast(object, self.context.sampling)
        if not isinstance(sampling, EvaluationSamplingCapability):
            raise TypeError("sampling Evaluation requires a sampling capability")
        model = cast(object, self.context.inference)
        if not isinstance(model, nn.Module):
            raise TypeError("sampling Evaluation requires an injected model")
        return EvaluationPlan(
            evaluator=SamplePairEvaluator(sampling),
            data=self.context.data,
            metric_specs=self.context.metric_specs,
            protocol=self.context.protocol,
            subject=self.context.subject,
            data_identity=self.context.data_identity,
            modules={"primary": model},
        )


def _registries() -> RegistryCatalog:
    catalog = RegistryCatalog()
    catalog.evaluation_builders.add("tests.live-model", LiveModelEvaluationBuilder)
    catalog.evaluation_builders.add(
        "tests.live-sampling",
        SamplingEvaluationBuilder,
    )
    return catalog


def _trainer(
    *,
    sampling_recipe_name: str = "tests.unused-sampling",
) -> Trainer:
    model = ScalarEvaluationModel(2.0)
    objective = nn.MSELoss()
    plan = TrainingPlan(
        strategy=SupervisedTrainingStrategy(model, objective),
        primary_model=model,
        objective=objective,
        inference_recipe=SamplingRecipe(name=sampling_recipe_name),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ema = ExponentialMovingAverage(model)
    for shadow in ema.shadow_params.values():
        shadow.fill_(5.0)
    return Trainer(
        plan,
        optimizer,
        device="cpu",
        ema=ema,
    )


def _config(*, weights: str = "ema", fail: bool = False) -> ValidationEvaluationConfig:
    return ValidationEvaluationConfig(
        enabled=True,
        start_epoch=2,
        every_epochs=2,
        include_final=True,
        weights=weights,
        evaluation=ComponentConfig(
            name="tests.live-model",
            params={"fail": fail},
        ),
        metrics=[
            MetricSpec(
                id="value",
                name=LIVE_MEAN_METRIC,
                channel="tests.live-values",
            )
        ],
        metric_keys=["valid/metrics/value"],
        protocol=ValidationEvaluationProtocolConfig(
            id="tests-live-validation-v1",
            expected_examples=2,
            strict_complete=True,
        ),
    )


def _data() -> list[torch.Tensor]:
    return [torch.tensor([2.0]), torch.tensor([4.0])]


def test_live_evaluation_uses_ema_and_restores_training_state() -> None:
    trainer = _trainer()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=_config(),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )
    model = cast(ScalarEvaluationModel, trainer.model)
    raw_weight = model.weight.detach().clone()
    assert model.training

    result = validator.evaluate(epoch=2, global_step=7)

    assert result.metrics == {"valid/metrics/value": 15.0}
    assert torch.equal(model.weight, raw_weight)
    assert model.training
    subject = LiveModelEvaluationBuilder.last_subject
    assert subject is not None
    assert subject.epoch == 2
    assert subject.global_step == 7
    assert subject.weights == "ema"
    assert subject.profile_digest == validator.identity.profile_digest


def test_live_evaluation_raw_profile_uses_current_model() -> None:
    validator = EvaluationBackedEpochValidator(
        trainer=_trainer(),
        config=_config(weights="raw"),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )

    result = validator.evaluate(epoch=2, global_step=7)

    assert result.metrics == {"valid/metrics/value": 6.0}


def test_live_evaluation_restores_ema_weights_after_failure() -> None:
    trainer = _trainer()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=_config(fail=True),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )
    model = cast(ScalarEvaluationModel, trainer.model)
    raw_weight = model.weight.detach().clone()

    with pytest.raises(RuntimeError, match="injected live Evaluation failure"):
        validator.evaluate(epoch=2, global_step=7)

    assert torch.equal(model.weight, raw_weight)
    assert model.training


def test_live_evaluation_profile_digest_covers_weight_authority() -> None:
    trainer = _trainer()
    common = {
        "trainer": trainer,
        "validation_data": _data(),
        "data_identity": {"source": "training", "split": "validation"},
    }

    raw = EvaluationBackedEpochValidator(
        config=_config(weights="raw"),
        registries=_registries(),
        **common,
    )
    ema = EvaluationBackedEpochValidator(
        config=_config(weights="ema"),
        registries=_registries(),
        **common,
    )

    assert raw.identity.profile_digest != ema.identity.profile_digest


def test_training_evaluation_samples_scores_and_saves_best_checkpoint(
    tmp_path,
) -> None:
    trainer = _trainer(sampling_recipe_name=LIVE_SAMPLING_BUILDER)
    trainer.checkpoint_manager = CheckpointManager(
        model=trainer.model,
        objective=trainer.objective,
        optimizer=trainer.optimizer,
        ema=trainer.ema,
    )
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_every = 1
    config = ValidationEvaluationConfig(
        enabled=True,
        start_epoch=1,
        every_epochs=1,
        include_final=True,
        weights="ema",
        evaluation=ComponentConfig(
            name="tests.live-sampling",
            params={},
        ),
        metrics=[
            MetricSpec(
                id="quality",
                name="mse",
                channel="tests.live-sample-pairs",
            )
        ],
        metric_keys=["valid/metrics/quality"],
        protocol=ValidationEvaluationProtocolConfig(
            id="tests-live-sampling-validation-v1",
            expected_examples=2,
            strict_complete=True,
        ),
    )
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=config,
        validation_data=[torch.zeros((2, 1))],
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )

    history = trainer.fit(
        [(torch.ones((1, 1)), torch.zeros((1, 1)))],
        num_epochs=1,
        show_progress=False,
        epoch_validation_evaluator=validator,
        early_stopping_monitor="valid/metrics/quality",
        early_stopping_mode="min",
    )

    assert trainer.ema is not None
    ema_weight = next(iter(trainer.ema.shadow_params.values())).item()
    expected_quality = ema_weight**2
    assert history[0]["valid/metrics/quality"] == pytest.approx(
        expected_quality
    )
    assert LiveModelSamplingBuilder.observed_weights == "ema"
    assert trainer.best_epoch == 1
    payload = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "best.pt"
    )
    assert payload.get("epoch") == 1
    metrics = payload.get("metrics")
    assert isinstance(metrics, dict)
    assert metrics["valid/metrics/quality"] == pytest.approx(
        expected_quality
    )
