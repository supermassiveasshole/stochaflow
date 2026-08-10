"""Failure-injection coverage for core-owned Evaluation cleanup."""

from __future__ import annotations

import hashlib
import traceback
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
import torch
from torch import nn
from torchmetrics import Metric

import stochaflow.evaluation.runtime as evaluation_runtime
from stochaflow.evaluation import (
    PREDICTION_JSONL_MEDIA_TYPE,
    PREDICTION_RECORD_FORMAT,
    EvaluationPlan,
    EvaluationProtocol,
    EvaluationProtocolIdentity,
    EvaluationStepOutput,
    PredictionArtifactDraft,
    PredictionSampleIdentity,
    PredictionShard,
    execute_evaluation_plan,
)
from stochaflow.metrics import (
    MetricEngine,
    MetricRuntimeError,
    MetricSpec,
    MetricUpdate,
)
from stochaflow.utils.registry import RegistryCatalog


class CleanupEvaluator:
    """Return one result or raise one injected body exception."""

    metric_channels = frozenset()

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.cause: BaseException | None = None

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        if self.error is not None:
            raise self.error from self.cause
        return EvaluationStepOutput(
            num_examples=1,
            sample_ids=("sample-a",),
            metric_update_groups=(),
        )


class CleanupProbeModule(nn.Module):
    """Record mode transitions and optionally fail after restoring state."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        eval_error: BaseException | None = None,
        restore_error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.eval_error = eval_error
        self.restore_error = restore_error

    def train(self, mode: bool = True) -> CleanupProbeModule:
        self.events.append(f"{self.name}:{'train' if mode else 'eval'}")
        super().train(mode)
        if not mode and self.eval_error is not None:
            raise self.eval_error
        if mode and self.restore_error is not None:
            raise self.restore_error
        return self


class CleanupMetricEngine:
    """Minimal MetricEngine substitute with injectable lifecycle failures."""

    def __init__(
        self,
        events: list[str],
        *,
        to_error: BaseException | None = None,
        reset_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.to_error = to_error
        self.reset_error = reset_error

    def to(self, device: torch.device) -> CleanupMetricEngine:
        self.events.append("metric:to")
        if self.to_error is not None:
            raise self.to_error
        return self

    def update(self, updates: Mapping[str, MetricUpdate]) -> None:
        self.events.append("metric:update")

    def compute(self, *, reset: bool = False) -> dict[str, float]:
        self.events.append(f"metric:compute:{reset}")
        return {}

    def reset(self) -> None:
        self.events.append("metric:reset")
        if self.reset_error is not None:
            raise self.reset_error


class CleanupSink:
    """Record sink lifecycle and inject finalize or abort failures."""

    def __init__(
        self,
        events: list[str],
        *,
        finalize_result: object | None = None,
        finalize_error: BaseException | None = None,
        abort_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.finalize_result = finalize_result
        self.finalize_error = finalize_error
        self.abort_error = abort_error

    def consume(self, output: EvaluationStepOutput) -> None:
        self.events.append("sink:consume")

    def finalize(self) -> PredictionArtifactDraft:
        self.events.append("sink:finalize")
        if self.finalize_error is not None:
            raise self.finalize_error
        return cast(PredictionArtifactDraft, self.finalize_result)

    def abort(self) -> None:
        self.events.append("sink:abort")
        if self.abort_error is not None:
            raise self.abort_error


class UpdateAndResetFailureMetric(Metric):
    """Fail an update and its immediate cleanup for precedence testing."""

    def update(self, value: torch.Tensor) -> None:
        raise ValueError("provider update failed")

    def compute(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def reset(self) -> None:
        raise RuntimeError("provider reset failed")


class OverridingAddNoteError(RuntimeError):
    """Reject dynamic note dispatch to exercise BaseException-safe cleanup."""

    def add_note(self, note: str) -> None:
        raise RuntimeError("overridden add_note must not be called")


def _plan(
    *,
    evaluator: CleanupEvaluator | None = None,
    modules: Mapping[str, nn.Module] | None = None,
    sink: CleanupSink | None = None,
) -> EvaluationPlan:
    return EvaluationPlan(
        evaluator=evaluator or CleanupEvaluator(),
        data=(object(),),
        metric_specs=(),
        protocol=EvaluationProtocol(
            id="cleanup-test",
            expected_examples=1,
            strict_complete=True,
        ),
        subject=object(),
        data_identity={},
        protocol_identity=EvaluationProtocolIdentity(
            providers={"test": {"name": "cleanup"}},
            preprocessing={"kind": "identity"},
        ),
        modules=modules or {},
        artifact_sink=sink,
    )


def _install_metric_engine(
    monkeypatch: pytest.MonkeyPatch,
    engine: CleanupMetricEngine,
) -> None:
    def build_engine(
        specs: Sequence[MetricSpec],
        *,
        registry: object,
    ) -> CleanupMetricEngine:
        engine.events.append("metric:init")
        return engine

    monkeypatch.setattr(evaluation_runtime, "MetricEngine", build_engine)


def _draft(sample_id: str) -> PredictionArtifactDraft:
    return PredictionArtifactDraft(
        samples=(PredictionSampleIdentity(sample_id, "input", 0),),
        shards=(
            PredictionShard(
                path="predictions.jsonl",
                media_type=PREDICTION_JSONL_MEDIA_TYPE,
                format=PREDICTION_RECORD_FORMAT,
                sha256=hashlib.sha256(b"record\n").hexdigest(),
                size_bytes=7,
                record_count=1,
            ),
        ),
    )


def test_success_restores_modules_and_metric_before_sink_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    engine = CleanupMetricEngine(events)
    _install_metric_engine(monkeypatch, engine)
    first = CleanupProbeModule("first", events)
    second = CleanupProbeModule("second", events)
    sink = CleanupSink(events, finalize_result=_draft("sample-a"))

    facts = execute_evaluation_plan(
        _plan(modules={"first": first, "second": second}, sink=sink),
        device=torch.device("cpu"),
    )

    assert facts.prediction_artifact is not None
    assert events.count("metric:reset") == 1
    assert "metric:compute:False" in events
    assert events.index("second:train") < events.index("first:train")
    assert events.index("first:train") < events.index("metric:reset")
    assert events.index("metric:reset") < events.index("sink:finalize")
    assert first.training
    assert second.training


@pytest.mark.parametrize("failure_point", ["construct", "to"])
def test_metric_setup_failure_aborts_unpublished_sink(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    events: list[str] = []
    setup_error = RuntimeError(f"metric {failure_point} failed")
    sink = CleanupSink(events)
    if failure_point == "construct":

        def build_engine(
            specs: Sequence[MetricSpec],
            *,
            registry: object,
        ) -> CleanupMetricEngine:
            raise setup_error

        monkeypatch.setattr(evaluation_runtime, "MetricEngine", build_engine)
    else:
        _install_metric_engine(
            monkeypatch,
            CleanupMetricEngine(events, to_error=setup_error),
        )

    with pytest.raises(RuntimeError) as caught:
        execute_evaluation_plan(
            _plan(sink=sink),
            device=torch.device("cpu"),
        )

    assert caught.value is setup_error
    assert events[-1] == "sink:abort"


def test_body_base_exception_keeps_identity_and_collects_all_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    body_error = KeyboardInterrupt("body interrupted")
    body_cause = RuntimeError("body cause")
    evaluator = CleanupEvaluator(body_error)
    evaluator.cause = body_cause
    first_error = RuntimeError("first restore failed")
    second_error = RuntimeError("second restore failed")
    reset_error = RuntimeError("metric reset failed")
    abort_error = RuntimeError("sink abort failed")
    first = CleanupProbeModule("first", events, restore_error=first_error)
    second = CleanupProbeModule("second", events, restore_error=second_error)
    sink = CleanupSink(events, abort_error=abort_error)
    _install_metric_engine(
        monkeypatch,
        CleanupMetricEngine(events, reset_error=reset_error),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        execute_evaluation_plan(
            _plan(
                evaluator=evaluator,
                modules={"first": first, "second": second},
                sink=sink,
            ),
            device=torch.device("cpu"),
        )

    assert caught.value is body_error
    assert caught.value.__cause__ is body_cause
    traceback_names = [
        frame.name for frame in traceback.extract_tb(caught.value.__traceback__)
    ]
    assert "evaluate_batch" in traceback_names
    assert first.training
    assert second.training
    assert events.index("second:train") < events.index("first:train")
    notes = "\n".join(caught.value.__notes__)
    assert "second restore failed" in notes
    assert "first restore failed" in notes
    assert "metric reset failed" in notes
    assert "sink abort failed" in notes


def test_runtime_error_keeps_explicit_cause_and_origin_during_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    body_error = RuntimeError("body failed")
    body_cause = ValueError("body cause")
    evaluator = CleanupEvaluator(body_error)
    evaluator.cause = body_cause
    module = CleanupProbeModule(
        "module",
        events,
        restore_error=RuntimeError("restore failed"),
    )
    sink = CleanupSink(
        events,
        abort_error=RuntimeError("abort failed"),
    )
    _install_metric_engine(
        monkeypatch,
        CleanupMetricEngine(
            events,
            reset_error=RuntimeError("reset failed"),
        ),
    )

    with pytest.raises(RuntimeError) as caught:
        execute_evaluation_plan(
            _plan(evaluator=evaluator, modules={"module": module}, sink=sink),
            device=torch.device("cpu"),
        )

    assert caught.value is body_error
    assert caught.value.__cause__ is body_cause
    traceback_names = [
        frame.name for frame in traceback.extract_tb(caught.value.__traceback__)
    ]
    assert "evaluate_batch" in traceback_names
    notes = "\n".join(caught.value.__notes__)
    assert "restore failed" in notes
    assert "reset failed" in notes
    assert "abort failed" in notes


def test_overridden_add_note_cannot_replace_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    body_error = OverridingAddNoteError("body failed")
    module = CleanupProbeModule(
        "module",
        events,
        restore_error=RuntimeError("restore failed"),
    )
    _install_metric_engine(monkeypatch, CleanupMetricEngine(events))

    with pytest.raises(OverridingAddNoteError) as caught:
        execute_evaluation_plan(
            _plan(
                evaluator=CleanupEvaluator(body_error),
                modules={"module": module},
            ),
            device=torch.device("cpu"),
        )

    assert caught.value is body_error
    assert "restore failed" in "\n".join(caught.value.__notes__)


def test_overridden_add_note_cannot_replace_first_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    later_error = RuntimeError("later restore failed")
    primary_error = OverridingAddNoteError("primary restore failed")
    first = CleanupProbeModule(
        "first",
        events,
        restore_error=later_error,
    )
    second = CleanupProbeModule(
        "second",
        events,
        restore_error=primary_error,
    )
    _install_metric_engine(monkeypatch, CleanupMetricEngine(events))

    with pytest.raises(OverridingAddNoteError) as caught:
        execute_evaluation_plan(
            _plan(modules={"first": first, "second": second}),
            device=torch.device("cpu"),
        )

    assert caught.value is primary_error
    notes = "\n".join(caught.value.__notes__)
    assert "primary cleanup action" in notes
    assert "later restore failed" in notes


def test_module_eval_failure_after_mode_change_restores_every_entered_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    eval_error = RuntimeError("second eval failed")
    first = CleanupProbeModule("first", events)
    second = CleanupProbeModule(
        "second",
        events,
        eval_error=eval_error,
    )
    _install_metric_engine(monkeypatch, CleanupMetricEngine(events))

    with pytest.raises(RuntimeError) as caught:
        execute_evaluation_plan(
            _plan(modules={"first": first, "second": second}),
            device=torch.device("cpu"),
        )

    assert caught.value is eval_error
    assert first.training
    assert second.training
    assert events.index("second:train") < events.index("first:train")
    assert events[-1] == "metric:reset"


def test_first_cleanup_failure_is_primary_and_other_cleanup_still_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    first_error = RuntimeError("first restore failed")
    second_error = RuntimeError("second restore failed")
    reset_error = RuntimeError("metric reset failed")
    first = CleanupProbeModule("first", events, restore_error=first_error)
    second = CleanupProbeModule("second", events, restore_error=second_error)
    sink = CleanupSink(events)
    _install_metric_engine(
        monkeypatch,
        CleanupMetricEngine(events, reset_error=reset_error),
    )

    with pytest.raises(RuntimeError) as caught:
        execute_evaluation_plan(
            _plan(modules={"first": first, "second": second}, sink=sink),
            device=torch.device("cpu"),
        )

    assert caught.value is second_error
    assert first.training
    assert second.training
    assert events[-1] == "sink:abort"
    notes = "\n".join(caught.value.__notes__)
    assert "first restore failed" in notes
    assert "metric reset failed" in notes


@pytest.mark.parametrize("failure_point", ["finalize", "draft"])
def test_finalize_or_draft_failure_aborts_without_masking_primary(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    events: list[str] = []
    abort_error = RuntimeError("abort after finalize failed")
    finalize_error = (
        RuntimeError("finalize failed") if failure_point == "finalize" else None
    )
    sink = CleanupSink(
        events,
        finalize_result=(
            _draft("wrong-sample") if failure_point == "draft" else None
        ),
        finalize_error=finalize_error,
        abort_error=abort_error,
    )
    _install_metric_engine(monkeypatch, CleanupMetricEngine(events))

    with pytest.raises((RuntimeError, ValueError)) as caught:
        execute_evaluation_plan(
            _plan(sink=sink),
            device=torch.device("cpu"),
        )

    if finalize_error is not None:
        assert caught.value is finalize_error
    else:
        assert "sample plan must match" in str(caught.value)
    assert events[-2:] == ["sink:finalize", "sink:abort"]
    assert "abort after finalize failed" in "\n".join(
        caught.value.__notes__
    )


def test_metric_update_failure_keeps_provider_error_as_cause_when_reset_fails(
) -> None:
    registries = RegistryCatalog()
    registries.metrics.add(
        "tests.update-and-reset-failure",
        UpdateAndResetFailureMetric,
    )
    engine = MetricEngine(
        (
            MetricSpec(
                id="failure",
                name="tests.update-and-reset-failure",
                channel="values",
            ),
        ),
        registry=registries.metrics,
    )

    with pytest.raises(MetricRuntimeError) as caught:
        engine.update(
            {"values": MetricUpdate(args=(torch.tensor([1.0]),))}
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "provider update failed"
    assert "provider reset failed" in "\n".join(caught.value.__notes__)
