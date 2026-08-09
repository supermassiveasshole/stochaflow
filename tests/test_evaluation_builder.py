"""Tests for task-neutral evaluation composition contracts."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

import stochaflow.evaluation as evaluation_api
from stochaflow.evaluation import (
    EvaluationBuilder,
    EvaluationBuilderContext,
    EvaluationPlan,
    EvaluationProtocol,
    EvaluationProtocolIdentity,
    EvaluationResult,
    EvaluationRunOutcome,
    EvaluationStepOutput,
    Evaluator,
    build_evaluation_plan,
    validate_evaluation_plan,
)
from stochaflow.metrics import MetricSpec, MetricUpdate
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import RegistryCatalog, RegistryError


@dataclass(frozen=True, slots=True)
class OpaqueEvaluationBatch:
    """Task-owned batch whose fields core must not interpret."""

    payload: object


class OpaqueEvaluationData:
    """Re-iterable task-owned selected split."""

    def __iter__(self) -> Iterator[OpaqueEvaluationBatch]:
        yield OpaqueEvaluationBatch(payload={"task_state": [1, 2]})


class RecordingEvaluator:
    """Small structural Evaluator used to prove opaque batch support."""

    metric_channels = frozenset({"opaque.values"})

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        assert isinstance(batch, OpaqueEvaluationBatch)
        values = torch.tensor(cast(dict[str, list[int]], batch.payload)["task_state"])
        return EvaluationStepOutput(
            num_examples=2,
            sample_ids=("sample-1", "sample-2"),
            metric_update_groups=({"opaque.values": MetricUpdate(args=(values,))},),
            records=batch,
            measurements={"latency_ms": 1.25},
        )


class RecordingEvaluationBuilder(EvaluationBuilder):
    """Builder that records its injected context for contract assertions."""

    last_context: EvaluationBuilderContext | None = None

    def build(self) -> EvaluationPlan:
        type(self).last_context = self.context
        return EvaluationPlan(
            evaluator=RecordingEvaluator(),
            data=self.context.data,
            modules={"projection": nn.Linear(2, 2)},
            metric_specs=self.context.metric_specs,
            protocol=self.context.protocol,
            subject=self.context.subject,
            data_identity=self.context.data_identity,
            protocol_identity=EvaluationProtocolIdentity(
                providers={"opaque": {"name": "recording"}},
                preprocessing={"axes": [0, 1]},
            ),
        )


class WrongReturnEvaluationBuilder(EvaluationBuilder):
    """Builder fixture that violates the result contract."""

    def build(self) -> EvaluationPlan:
        return cast(EvaluationPlan, object())


def metric_specs() -> tuple[MetricSpec, ...]:
    """Return a nested declaration used by builder tests."""

    return (
        MetricSpec(
            id="opaque_mean",
            name="single_output_mean_absolute_error",
            channel="opaque.values",
            params={"preprocess": {"axes": [0, 1]}},
        ),
    )


def protocol() -> EvaluationProtocol:
    """Return a strict example protocol."""

    return EvaluationProtocol(
        id="opaque-v1",
        expected_examples=2,
        strict_complete=True,
    )


def test_evaluation_public_api_exports_protocol_identity() -> None:
    assert "EvaluationProtocolIdentity" in evaluation_api.__all__
    assert evaluation_api.EvaluationProtocolIdentity is EvaluationProtocolIdentity


def build_recording_plan() -> tuple[
    EvaluationPlan,
    object,
    OpaqueEvaluationData,
    object,
]:
    """Build a plan through a private registry catalog."""

    catalog = RegistryCatalog()
    catalog.evaluation_builders.add("recording", RecordingEvaluationBuilder)
    subject = object()
    data = OpaqueEvaluationData()
    inference = object()
    plan = build_evaluation_plan(
        ComponentConfig(
            name="recording",
            params={"nested": {"values": [1, 2]}},
        ),
        subject=subject,
        data=data,
        data_identity={"dataset": {"digest": "a" * 64, "ids": ["a", "b"]}},
        inference=inference,
        metric_specs=metric_specs(),
        protocol=protocol(),
        registries=catalog,
    )
    return plan, subject, data, inference


def test_registered_builder_composes_opaque_evaluation_plan() -> None:
    plan, subject, data, inference = build_recording_plan()
    context = RecordingEvaluationBuilder.last_context

    assert context is not None
    assert plan.subject is subject
    assert plan.data is data
    assert context.inference is inference
    assert isinstance(plan.evaluator, Evaluator)
    output = plan.evaluator.evaluate_batch(next(iter(plan.data)))
    assert output.sample_ids == ("sample-1", "sample-2")
    assert output.records == OpaqueEvaluationBatch(payload={"task_state": [1, 2]})


def test_builder_context_and_plan_mappings_are_deeply_read_only() -> None:
    specs = metric_specs()
    declaration = ComponentConfig(
        name="recording",
        params={"nested": {"values": [1, 2]}},
    )
    identity = {"dataset": {"ids": ["a", "b"]}}
    catalog = RegistryCatalog()
    catalog.evaluation_builders.add("recording", RecordingEvaluationBuilder)

    plan = build_evaluation_plan(
        declaration,
        subject=object(),
        data=OpaqueEvaluationData(),
        data_identity=identity,
        inference=None,
        metric_specs=specs,
        protocol=protocol(),
        registries=catalog,
    )
    context = RecordingEvaluationBuilder.last_context
    assert context is not None
    cast(dict[str, Any], declaration.params["nested"])["values"].append(3)
    cast(dict[str, Any], identity["dataset"])["ids"].append("c")
    cast(dict[str, Any], specs[0].params["preprocess"])["axes"].append(2)

    nested_params = cast(dict[str, Any], context.params["nested"])
    nested_identity = cast(dict[str, Any], plan.data_identity["dataset"])
    nested_metric = cast(dict[str, Any], plan.metric_specs[0].params["preprocess"])
    assert nested_params["values"] == (1, 2)
    assert nested_identity["ids"] == ("a", "b")
    assert nested_metric["axes"] == (0, 1)

    with pytest.raises(TypeError):
        cast(dict[str, Any], context.params)["other"] = 1
    with pytest.raises(TypeError):
        nested_params["other"] = 1
    with pytest.raises(TypeError):
        cast(dict[str, Any], plan.data_identity)["other"] = "value"
    with pytest.raises(TypeError):
        nested_identity["other"] = "value"
    with pytest.raises(TypeError):
        plan.metric_specs[0].params["other"] = 1
    with pytest.raises(TypeError):
        nested_metric["other"] = 1
    with pytest.raises(TypeError):
        cast(dict[str, nn.Module], plan.modules)["other"] = nn.Linear(1, 1)
    with pytest.raises(TypeError):
        cast(dict[str, Any], plan.protocol_identity.providers)["other"] = {}
    with pytest.raises(TypeError):
        cast(dict[str, Any], plan.protocol_identity.preprocessing)["other"] = 1


def test_protocol_identity_requires_explicit_nonempty_facts() -> None:
    with pytest.raises(ValueError, match="providers must be non-empty"):
        EvaluationProtocolIdentity(providers={}, preprocessing={"kind": "identity"})
    with pytest.raises(ValueError, match="preprocessing must be non-empty"):
        EvaluationProtocolIdentity(providers={"test": {}}, preprocessing={})
    with pytest.raises(ValueError, match="duplicate metric provider"):
        EvaluationProtocolIdentity(
            providers={"test": {}},
            preprocessing={"kind": "identity"},
            metric_providers=("mean", "mean"),
        )

    identity = EvaluationProtocolIdentity(
        providers={"test": {"revision": 1}},
        preprocessing={"kind": "identity"},
        dependencies=("Torch-Fidelity",),
    )

    assert identity.dependencies == ("torch-fidelity",)


def test_builder_registry_and_return_contract_are_checked() -> None:
    catalog = RegistryCatalog()
    catalog.evaluation_builders.require_base(EvaluationBuilder)
    with pytest.raises(RegistryError, match=r"must inherit .*EvaluationBuilder"):
        catalog.evaluation_builders.add("wrong_base", cast(Any, object))

    catalog.evaluation_builders.add("wrong_return", WrongReturnEvaluationBuilder)
    with pytest.raises(TypeError, match="must return EvaluationPlan"):
        build_evaluation_plan(
            ComponentConfig("wrong_return"),
            subject=object(),
            data=OpaqueEvaluationData(),
            data_identity={"dataset": "opaque"},
            inference=None,
            metric_specs=metric_specs(),
            protocol=protocol(),
            registries=catalog,
        )


def test_builder_must_preserve_injected_composition_boundaries() -> None:
    class ReplacingDataEvaluationBuilder(EvaluationBuilder):
        """Builder fixture that tries to replace the selected split."""

        def build(self) -> EvaluationPlan:
            return EvaluationPlan(
                evaluator=RecordingEvaluator(),
                data=OpaqueEvaluationData(),
                metric_specs=self.context.metric_specs,
                protocol=self.context.protocol,
                subject=self.context.subject,
                data_identity=self.context.data_identity,
                protocol_identity=EvaluationProtocolIdentity(
                    providers={"test": {"kind": "wrong-data"}},
                    preprocessing={"kind": "identity"},
                ),
            )

    catalog = RegistryCatalog()
    catalog.evaluation_builders.add("replacing", ReplacingDataEvaluationBuilder)
    with pytest.raises(ValueError, match="data must preserve"):
        build_evaluation_plan(
            ComponentConfig("replacing"),
            subject=object(),
            data=OpaqueEvaluationData(),
            data_identity={"dataset": "opaque"},
            inference=None,
            metric_specs=metric_specs(),
            protocol=protocol(),
            registries=catalog,
        )


def test_plan_validation_checks_evaluator_channels_and_reiterable_data() -> None:
    with pytest.raises(ValueError, match="missing metric channel"):
        validate_evaluation_plan(
            EvaluationPlan(
                evaluator=RecordingEvaluator(),
                data=OpaqueEvaluationData(),
                metric_specs=(MetricSpec("missing", "metric", "other.channel"),),
                protocol=protocol(),
                subject=object(),
                data_identity={"dataset": "opaque"},
                protocol_identity=EvaluationProtocolIdentity(
                    providers={"test": {"kind": "missing-channel"}},
                    preprocessing={"kind": "identity"},
                ),
            )
        )

    with pytest.raises(TypeError, match="re-iterable"):
        validate_evaluation_plan(
            EvaluationPlan(
                evaluator=RecordingEvaluator(),
                data=iter([OpaqueEvaluationBatch(object())]),
                metric_specs=metric_specs(),
                protocol=protocol(),
                subject=object(),
                data_identity={"dataset": "opaque"},
                protocol_identity=EvaluationProtocolIdentity(
                    providers={"test": {"kind": "one-shot"}},
                    preprocessing={"kind": "identity"},
                ),
            )
        )


def test_step_output_validates_and_freezes_metric_groups() -> None:
    update = MetricUpdate(args=(torch.tensor([1.0]),))
    output = EvaluationStepOutput(
        num_examples=1,
        sample_ids=("sample-1",),
        metric_update_groups=({"opaque.values": update},),
        measurements={"latency_ms": 1.0},
    )

    assert output.metric_update_groups[0]["opaque.values"] is update
    with pytest.raises(TypeError):
        cast(dict[str, MetricUpdate], output.metric_update_groups[0])["other"] = update
    with pytest.raises(TypeError):
        cast(dict[str, float], output.measurements)["other"] = 1.0

    with pytest.raises(ValueError, match="sample_ids length"):
        EvaluationStepOutput(2, ("only-one",), ())
    with pytest.raises(ValueError, match="duplicate sample id"):
        EvaluationStepOutput(2, ("same", "same"), ())
    with pytest.raises(ValueError, match="finite"):
        EvaluationStepOutput(1, ("sample",), (), measurements={"latency": float("nan")})


def test_result_and_outcome_snapshot_portable_mappings(tmp_path: Path) -> None:
    subject = {"kind": "checkpoint", "weights": {"requested": "raw"}}
    result = EvaluationResult(
        schema_version=1,
        evaluation_id="opaque-evaluation",
        protocol_id="opaque-v1",
        protocol_digest="a" * 64,
        status="complete",
        subject=subject,
        data={"split": "validation", "sample_ids": ["sample-1"]},
        metrics={"eval/metrics/score": 1.0},
        measurements={"eval/measurements/latency_ms": 2.0},
        artifacts={"predictions": {"path": "predictions.jsonl", "sha256": "b" * 64}},
        completeness={"expected": 1, "observed": 1},
        provenance={"builder": {"name": "opaque"}},
    )
    outcome = EvaluationRunOutcome(
        evaluation_id="opaque-evaluation",
        protocol_id="opaque-v1",
        status="complete",
        output_dir=tmp_path,
        subject=subject,
        split="validation",
        metrics=result.metrics,
        measurements=result.measurements,
        artifacts={"predictions": tmp_path / "predictions.jsonl"},
        manifest_path=tmp_path / "manifest.yaml",
        result_path=tmp_path / "result.json",
    )
    subject["weights"]["requested"] = "ema"

    assert result.subject["weights"]["requested"] == "raw"
    assert outcome.subject["weights"]["requested"] == "raw"
    with pytest.raises(TypeError):
        cast(dict[str, Any], result.subject["weights"])["requested"] = "ema"
    with pytest.raises(TypeError):
        cast(dict[str, float], result.metrics)["other"] = 2.0
    with pytest.raises(TypeError):
        cast(dict[str, Path], outcome.artifacts)["other"] = tmp_path / "other"
