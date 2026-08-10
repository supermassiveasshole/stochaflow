"""Tests for Evaluation-backed checkpoint selection during training."""

from __future__ import annotations

import random
from typing import Any, ClassVar, cast

import numpy as np
import pytest
import torch
from torch import nn
from torchmetrics.aggregation import MeanMetric
from torchmetrics.regression import MeanSquaredError

from stochaflow.evaluation import (
    EvaluationArtifactSink,
    EvaluationBuilder,
    EvaluationPlan,
    EvaluationProtocolIdentity,
    EvaluationSamplingCapability,
    EvaluationSamplingRequest,
    EvaluationStepOutput,
    PredictionArtifactDraft,
)
from stochaflow.metrics import MetricSpec, MetricUpdate
from stochaflow.sampling import (
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)
from stochaflow.training import (
    ExponentialMovingAverage,
    SupervisedTrainingStrategy,
    Trainer,
    TrainingPlan,
)
from stochaflow.training.epoch_evaluation import (
    EvaluationBackedEpochValidator,
    LiveEpochEvaluationSubject,
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
LIVE_MSE_METRIC = "tests.live-epoch-mse"
LIVE_SAMPLING_BUILDER = "tests.live-epoch-sampling"


class LiveEpochMeanMetric(MeanMetric):
    """Mean metric registered only for the live Evaluation tests."""


class LiveEpochMSEMetric(MeanSquaredError):
    """Pair metric registered only in an isolated test catalog."""


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
    provider_revision: ClassVar[int] = 1

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
            protocol_identity=EvaluationProtocolIdentity(
                providers={
                    "mean": {
                        "name": LIVE_MEAN_METRIC,
                        "revision": type(self).provider_revision,
                    }
                },
                preprocessing={"kind": "identity"},
            ),
            modules={"primary": model},
        )


class AlternateLiveModelEvaluationBuilder(LiveModelEvaluationBuilder):
    """Behavior-compatible provider with a distinct implementation identity."""


class LiveCleanupSink(EvaluationArtifactSink):
    """Observe cleanup of a temporary identity-preflight sink."""

    def __init__(self) -> None:
        self.aborted = False

    def consume(self, output: EvaluationStepOutput) -> None:
        raise AssertionError("identity preflight must not consume its sink")

    def finalize(self) -> PredictionArtifactDraft:
        raise AssertionError("identity preflight must not finalize its sink")

    def abort(self) -> None:
        self.aborted = True


class SinkDeclaringLiveEvaluationBuilder(LiveModelEvaluationBuilder):
    """Declare a temporary sink to exercise identity-preflight cleanup."""

    last_sink: ClassVar[LiveCleanupSink | None] = None

    def build(self) -> EvaluationPlan:
        plan = super().build()
        sink = LiveCleanupSink()
        type(self).last_sink = sink
        return EvaluationPlan(
            evaluator=plan.evaluator,
            data=plan.data,
            metric_specs=plan.metric_specs,
            protocol=plan.protocol,
            subject=plan.subject,
            data_identity=plan.data_identity,
            protocol_identity=plan.protocol_identity,
            modules=plan.modules,
            artifact_sink=sink,
        )


class RNGConsumingEvaluationBuilder(LiveModelEvaluationBuilder):
    """Exercise task-owned construction without perturbing training RNG."""

    def build(self) -> EvaluationPlan:
        random.random()
        np.random.random()
        torch.rand(())
        return super().build()


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
            protocol_identity=EvaluationProtocolIdentity(
                providers={"mse": {"name": LIVE_MSE_METRIC}},
                preprocessing={"kind": "paired-samples"},
            ),
            modules={"primary": model},
        )


def _registries() -> RegistryCatalog:
    catalog = RegistryCatalog()
    catalog.metrics.add(LIVE_MEAN_METRIC, LiveEpochMeanMetric)
    catalog.metrics.add(LIVE_MSE_METRIC, LiveEpochMSEMetric)
    catalog.sampling_builders.add(
        LIVE_SAMPLING_BUILDER,
        LiveModelSamplingBuilder,
    )
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


def _config(
    *,
    weights: str = "ema",
    fail: bool = False,
) -> ValidationEvaluationConfig:
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


def test_live_evaluation_is_reusable_across_epochs() -> None:
    trainer = _trainer()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=_config(),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )
    model = cast(ScalarEvaluationModel, trainer.model)

    first = validator.evaluate(epoch=2, global_step=7)
    torch.testing.assert_close(model.weight, torch.tensor(2.0))
    assert model.training

    with torch.no_grad():
        model.weight.fill_(3.0)
        assert trainer.ema is not None
        for shadow in trainer.ema.shadow_params.values():
            shadow.fill_(7.0)

    second = validator.evaluate(epoch=4, global_step=14)

    assert (first.epoch, first.global_step) == (2, 7)
    assert first.metrics == {"valid/metrics/value": 15.0}
    assert (second.epoch, second.global_step) == (4, 14)
    assert second.metrics == {"valid/metrics/value": 21.0}
    torch.testing.assert_close(model.weight, torch.tensor(3.0))
    assert model.training


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


def test_live_evaluation_restores_raw_weights_when_ema_copy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    ema = trainer.ema
    assert ema is not None
    original_copy_to = ema.copy_to
    copy_error = RuntimeError("injected EMA copy failure")

    def copy_to_and_fail(module: nn.Module) -> None:
        original_copy_to(module)
        raise copy_error

    monkeypatch.setattr(ema, "copy_to", copy_to_and_fail)

    with pytest.raises(RuntimeError) as caught:
        validator.evaluate(epoch=2, global_step=7)

    assert caught.value is copy_error
    assert torch.equal(model.weight, raw_weight)
    assert model.training


def test_live_mode_entry_failure_restores_raw_weights_and_changed_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=_config(),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )
    model = cast(ScalarEvaluationModel, trainer.model)
    objective = trainer.objective
    assert objective is not None
    raw_weight = model.weight.detach().clone()
    original_train = objective.train
    entry_error = RuntimeError("injected live mode entry failure")

    def enter_eval_and_fail(mode: bool = True) -> nn.Module:
        result = original_train(mode)
        if not mode:
            raise entry_error
        return result

    monkeypatch.setattr(objective, "train", enter_eval_and_fail)

    with pytest.raises(RuntimeError) as caught:
        validator.evaluate(epoch=2, global_step=7)

    assert caught.value is entry_error
    assert torch.equal(model.weight, raw_weight)
    assert model.training
    assert objective.training


def test_live_evaluation_attempts_every_module_restore_after_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=_config(fail=True),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )
    model = cast(ScalarEvaluationModel, trainer.model)
    objective = trainer.objective
    assert objective is not None
    raw_weight = model.weight.detach().clone()
    model_train = model.train
    objective_train = objective.train

    def failing_model_train(mode: bool = True) -> nn.Module:
        result = model_train(mode)
        if mode:
            raise RuntimeError("model mode restore failed")
        return result

    def failing_objective_train(mode: bool = True) -> nn.Module:
        result = objective_train(mode)
        if mode:
            raise RuntimeError("objective mode restore failed")
        return result

    monkeypatch.setattr(model, "train", failing_model_train)
    monkeypatch.setattr(objective, "train", failing_objective_train)

    with pytest.raises(
        RuntimeError,
        match="injected live Evaluation failure",
    ) as caught:
        validator.evaluate(epoch=2, global_step=7)

    assert torch.equal(model.weight, raw_weight)
    assert model.training
    assert objective.training
    notes = "\n".join(caught.value.__notes__)
    assert "model mode restore failed" in notes
    assert "objective mode restore failed" in notes


def test_live_ema_restore_failure_does_not_block_module_mode_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=_config(fail=True),
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=_registries(),
    )
    model = cast(ScalarEvaluationModel, trainer.model)
    objective = trainer.objective
    ema = trainer.ema
    assert objective is not None
    assert ema is not None
    raw_weight = model.weight.detach().clone()
    restored_modes: list[str] = []
    model_train = model.train
    objective_train = objective.train
    original_restore = ema.restore

    def recording_model_train(mode: bool = True) -> nn.Module:
        result = model_train(mode)
        if mode:
            restored_modes.append("model")
        return result

    def recording_objective_train(mode: bool = True) -> nn.Module:
        result = objective_train(mode)
        if mode:
            restored_modes.append("objective")
        return result

    def restore_and_fail(module: nn.Module) -> None:
        original_restore(module)
        raise RuntimeError("EMA restore failed")

    monkeypatch.setattr(model, "train", recording_model_train)
    monkeypatch.setattr(objective, "train", recording_objective_train)
    monkeypatch.setattr(ema, "restore", restore_and_fail)

    with pytest.raises(
        RuntimeError,
        match="injected live Evaluation failure",
    ) as caught:
        validator.evaluate(epoch=2, global_step=7)

    assert torch.equal(model.weight, raw_weight)
    assert model.training
    assert objective.training
    assert restored_modes == ["objective", "model"]
    assert "EMA restore failed" in "\n".join(caught.value.__notes__)


def test_live_identity_preflight_restore_failure_is_reported_after_restoring_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    model = cast(ScalarEvaluationModel, trainer.model)
    original_train = model.train
    restore_error = RuntimeError("identity mode restore failed")

    def failing_train(mode: bool = True) -> nn.Module:
        result = original_train(mode)
        if mode:
            raise restore_error
        return result

    monkeypatch.setattr(model, "train", failing_train)

    with pytest.raises(RuntimeError) as caught:
        EvaluationBackedEpochValidator(
            trainer=trainer,
            config=_config(weights="raw"),
            validation_data=_data(),
            data_identity={"source": "training", "split": "validation"},
            registries=_registries(),
        )

    assert caught.value is restore_error
    assert model.training


def test_live_identity_preflight_eval_failure_restores_changed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _trainer()
    model = cast(ScalarEvaluationModel, trainer.model)
    original_train = model.train
    eval_error = RuntimeError("identity eval failed")

    def failing_eval(mode: bool = True) -> nn.Module:
        result = original_train(mode)
        if not mode:
            raise eval_error
        return result

    monkeypatch.setattr(model, "train", failing_eval)

    with pytest.raises(RuntimeError) as caught:
        EvaluationBackedEpochValidator(
            trainer=trainer,
            config=_config(weights="raw"),
            validation_data=_data(),
            data_identity={"source": "training", "split": "validation"},
            registries=_registries(),
        )

    assert caught.value is eval_error
    assert model.training


def test_live_identity_preflight_aborts_temporary_artifact_sink() -> None:
    registries = _registries()
    registries.evaluation_builders.add(
        "tests.live-sink",
        SinkDeclaringLiveEvaluationBuilder,
    )
    config = _config(weights="raw")
    config.evaluation = ComponentConfig(name="tests.live-sink", params={})
    trainer = _trainer()

    EvaluationBackedEpochValidator(
        trainer=trainer,
        config=config,
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=registries,
    )

    sink = SinkDeclaringLiveEvaluationBuilder.last_sink
    assert sink is not None
    assert sink.aborted
    assert trainer.model.training


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


def test_live_profile_digest_covers_registered_provider_implementation() -> None:
    standard_registries = _registries()
    alternate_registries = RegistryCatalog()
    alternate_registries.metrics.add(LIVE_MEAN_METRIC, LiveEpochMeanMetric)
    alternate_registries.metrics.add(LIVE_MSE_METRIC, LiveEpochMSEMetric)
    alternate_registries.sampling_builders.add(
        LIVE_SAMPLING_BUILDER,
        LiveModelSamplingBuilder,
    )
    alternate_registries.evaluation_builders.add(
        "tests.live-model",
        AlternateLiveModelEvaluationBuilder,
    )
    common = {
        "trainer": _trainer(),
        "config": _config(weights="raw"),
        "validation_data": _data(),
        "data_identity": {"source": "training", "split": "validation"},
    }

    standard = EvaluationBackedEpochValidator(
        registries=standard_registries,
        **common,
    )
    alternate = EvaluationBackedEpochValidator(
        registries=alternate_registries,
        **common,
    )

    assert standard.identity.profile_digest != alternate.identity.profile_digest


def test_live_profile_digest_covers_builder_declared_provider_identity() -> None:
    common = {
        "trainer": _trainer(),
        "validation_data": _data(),
        "data_identity": {"source": "training", "split": "validation"},
    }

    try:
        LiveModelEvaluationBuilder.provider_revision = 1
        first = EvaluationBackedEpochValidator(
            config=_config(weights="raw"),
            registries=_registries(),
            **common,
        )
        LiveModelEvaluationBuilder.provider_revision = 2
        second = EvaluationBackedEpochValidator(
            config=_config(weights="raw"),
            registries=_registries(),
            **common,
        )
    finally:
        LiveModelEvaluationBuilder.provider_revision = 1

    assert first.identity.profile_digest != second.identity.profile_digest


def test_live_evaluation_identity_preflight_preserves_global_rng_state() -> None:
    registries = _registries()
    registries.evaluation_builders.add(
        "tests.rng-consuming",
        RNGConsumingEvaluationBuilder,
    )
    config = _config(weights="raw")
    config.evaluation = ComponentConfig(
        name="tests.rng-consuming",
        params={},
    )
    trainer = _trainer()

    random.seed(881)
    np.random.seed(881)
    torch.manual_seed(881)
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )
    random.seed(881)
    np.random.seed(881)
    torch.manual_seed(881)

    EvaluationBackedEpochValidator(
        trainer=trainer,
        config=config,
        validation_data=_data(),
        data_identity={"source": "training", "split": "validation"},
        registries=registries,
    )
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    assert actual == expected


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
                name=LIVE_MSE_METRIC,
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
    registries = _registries()
    assert LIVE_MSE_METRIC not in REGISTRIES.metrics.names()
    assert LIVE_SAMPLING_BUILDER not in REGISTRIES.sampling_builders.names()
    validator = EvaluationBackedEpochValidator(
        trainer=trainer,
        config=config,
        validation_data=[torch.zeros((2, 1))],
        data_identity={"source": "training", "split": "validation"},
        registries=registries,
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
