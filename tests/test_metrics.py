"""Contract tests for task-neutral metric construction and runtime state."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
from torchmetrics import Metric

from stochaflow.metrics import (
    EpochMetricSnapshot,
    MetricConfig,
    MetricEngine,
    MetricRuntimeError,
    MetricSource,
    MetricSpec,
    MetricUpdate,
    build_metric,
    detach_metric_update,
    detach_metric_updates,
    validate_metric_configs,
    validate_metric_updates,
)
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog, RegistryError


class RecordingSumMetric(Metric):
    """Record update payload identity while summing scalar tensors."""

    observed: ClassVar[list[torch.Tensor]] = []
    grad_enabled: ClassVar[list[bool]] = []
    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, value: torch.Tensor) -> None:
        type(self).observed.append(value)
        type(self).grad_enabled.append(torch.is_grad_enabled())
        self.total += value.sum()

    def compute(self) -> torch.Tensor:
        return self.total


class ConfiguredResultMetric(Metric):
    """Return one selected result shape for normalization contract tests."""

    total: torch.Tensor

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.add_state(
            "total",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, value: torch.Tensor) -> None:
        self.total += value.sum()

    def compute(self) -> Any:
        if self.kind == "flat":
            return {"sum": self.total, "twice": self.total * 2}
        if self.kind == "bool":
            return True
        if self.kind == "vector":
            return torch.stack((self.total, self.total))
        if self.kind == "nested":
            return {"outer": {"inner": self.total}}
        if self.kind == "empty":
            return {}
        if self.kind == "invalid_key":
            return {"bad/key": self.total}
        if self.kind == "list":
            return [self.total]
        raise AssertionError(f"unknown configured result kind {self.kind!r}")


class FailingComputeMetric(Metric):
    """Fail in compute and expose whether reset still ran."""

    reset_calls: ClassVar[int] = 0
    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, value: torch.Tensor) -> None:
        self.total += value.sum()

    def compute(self) -> torch.Tensor:
        raise RuntimeError("intentional compute failure")

    def reset(self) -> None:
        type(self).reset_calls += 1
        super().reset()


class FailingUpdateMetric(Metric):
    """Reject every update so incomplete dispatches cannot count as success."""

    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, value: torch.Tensor) -> None:
        raise RuntimeError(f"intentional update failure for {value.item()}")

    def compute(self) -> torch.Tensor:
        return self.total


REGISTRIES.metrics.add("test.recording_sum", RecordingSumMetric)
REGISTRIES.metrics.add("test.configured_result", ConfiguredResultMetric)
REGISTRIES.metrics.add("test.failing_compute", FailingComputeMetric)
REGISTRIES.metrics.add("test.failing_update", FailingUpdateMetric)


def test_metric_update_validates_and_freezes_kwargs() -> None:
    update = MetricUpdate(args=(1,), kwargs={"weight": 2})

    assert update.args == (1,)
    assert dict(update.kwargs) == {"weight": 2}
    with pytest.raises(TypeError):
        cast(dict[str, Any], update.kwargs)["other"] = 3
    with pytest.raises(TypeError, match="args must be a tuple"):
        MetricUpdate(args=cast(Any, [1]))
    with pytest.raises(TypeError, match="kwargs must be a mapping"):
        MetricUpdate(kwargs=cast(Any, []))
    with pytest.raises(ValueError, match="non-empty"):
        MetricUpdate(kwargs={"": 1})


def test_metric_update_mapping_validates_channels_and_values() -> None:
    update = MetricUpdate(args=(torch.tensor(1.0),))

    validated = validate_metric_updates({"prediction": update})

    assert validated["prediction"] is update
    with pytest.raises(TypeError):
        cast(dict[str, MetricUpdate], validated)["other"] = update
    with pytest.raises(ValueError, match="non-empty"):
        validate_metric_updates({" ": update})
    with pytest.raises(TypeError, match="must be a MetricUpdate"):
        validate_metric_updates({"prediction": cast(Any, object())})


def test_recursive_detach_handles_nested_payloads_without_mutating_input() -> None:
    value = torch.tensor(2.0, requires_grad=True)
    update = MetricUpdate(
        args=({"nested": [value]},),
        kwargs={"metadata": (value,)},
    )

    detached = detach_metric_update(update)
    detached_mapping = detach_metric_updates({"channel": update})

    args_value = cast(dict[str, list[torch.Tensor]], detached.args[0])
    kwargs_value = cast(tuple[torch.Tensor, ...], detached.kwargs["metadata"])
    mapped_value = cast(
        dict[str, list[torch.Tensor]],
        detached_mapping["channel"].args[0],
    )
    assert value.requires_grad
    assert not args_value["nested"][0].requires_grad
    assert not kwargs_value[0].requires_grad
    assert not mapped_value["nested"][0].requires_grad


@pytest.mark.parametrize(
    ("source_factory", "message"),
    [
        (
            lambda: MetricSource(
                cast(Any, "unknown"),
                "validation",
                None,
                False,
            ),
            "origin",
        ),
        (
            lambda: MetricSource("system", "validation", None, False),
            "system metric",
        ),
        (
            lambda: MetricSource("phase", None, None, False),
            "require a data_role",
        ),
        (
            lambda: MetricSource("phase", "external", None, False),
            "phase metric",
        ),
        (
            lambda: MetricSource("phase", "test", None, True),
            "cannot be selection eligible",
        ),
    ],
)
def test_metric_source_enforces_selection_and_role_invariants(
    source_factory: Callable[[], MetricSource],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        source_factory()


def test_metric_source_accepts_non_selectable_test_role() -> None:
    source = MetricSource("phase", "test", None, False)

    assert source.data_role == "test"


def test_metric_source_and_snapshot_round_trip_as_read_only_metadata() -> None:
    source = MetricSource(
        origin="phase",
        data_role="validation",
        protocol_id="validation-v1",
        selection_eligible=True,
    )
    snapshot = EpochMetricSnapshot(
        values={"valid/loss": 1.25, "valid/metrics/mse": math.nan},
        sources={
            "valid/loss": source,
            "valid/metrics/mse": source,
        },
    )

    restored = EpochMetricSnapshot.from_dict(snapshot.to_dict())

    assert restored.sources["valid/loss"] == source
    assert restored.values["valid/loss"] == pytest.approx(1.25)
    assert math.isnan(restored.values["valid/metrics/mse"])
    with pytest.raises(TypeError):
        cast(dict[str, float], restored.values)["new"] = 1.0
    with pytest.raises(TypeError):
        cast(dict[str, MetricSource], restored.sources)["new"] = source


def test_snapshot_rejects_mismatched_keys_and_invalid_serialized_source() -> None:
    source = MetricSource("phase", "validation", None, True)

    with pytest.raises(ValueError, match="exactly match"):
        EpochMetricSnapshot(
            values={"valid/loss": 1.0},
            sources={"valid/other": source},
        )
    with pytest.raises(TypeError, match="numeric"):
        EpochMetricSnapshot(
            values={"valid/loss": cast(Any, True)},
            sources={"valid/loss": source},
        )
    with pytest.raises(ValueError, match="unknown extra"):
        MetricSource.from_dict(
            {
                **source.to_dict(),
                "extra": "value",
            }
        )


def test_snapshot_rejects_phase_key_and_source_role_conflicts() -> None:
    train_source = MetricSource("phase", "train", None, True)

    with pytest.raises(ValueError, match="conflicts with data role"):
        EpochMetricSnapshot(
            values={"valid/loss": 1.0},
            sources={"valid/loss": train_source},
        )


def test_metric_registry_reserves_native_torchmetrics_namespace() -> None:
    with pytest.raises(ValueError, match="reserved namespace"):
        REGISTRIES.metrics.add(
            "torchmetrics.classification.MulticlassAccuracy",
            RecordingSumMetric,
        )


def test_metric_config_validation_is_local_strict_and_detects_duplicates() -> None:
    configs = [
        MetricConfig(
            id="reconstruction_mse",
            name="mse",
            channel="gaussian.clean_reconstruction",
            phases=["train", "validation"],
        ),
        MetricConfig(
            id="prediction_mae",
            name="mae",
            channel="gaussian.prediction_target",
            phases=["test"],
        ),
    ]

    validate_metric_configs(configs)

    with pytest.raises(ValueError, match="duplicate metric id"):
        validate_metric_configs([configs[0], configs[0]])
    with pytest.raises(ValueError, match="duplicate phase"):
        validate_metric_configs(
            [
                MetricConfig(
                    id="mse",
                    name="mse",
                    channel="prediction",
                    phases=["train", "train"],
                )
            ]
        )
    with pytest.raises(ValueError, match="must match"):
        validate_metric_configs(
            [
                MetricConfig(
                    id="bad/id",
                    name="mse",
                    channel="prediction",
                )
            ]
        )
    with pytest.raises(ValueError, match="train, validation, or test"):
        validate_metric_configs(
            [
                MetricConfig(
                    id="mse",
                    name="mse",
                    channel="prediction",
                    phases=["predict"],
                )
            ]
        )
    with pytest.raises(TypeError, match="params must be a mapping"):
        validate_metric_configs(
            [
                MetricConfig(
                    id="mse",
                    name="mse",
                    channel="prediction",
                    params=cast(Any, []),
                )
            ]
        )


def test_metric_registry_requires_torchmetrics_base_and_factory_is_allowlisted() -> None:
    build_metric(MetricSpec("mean", "mean", "value"))

    with pytest.raises(RegistryError, match="must inherit"):
        RegistryCatalog().metrics.add("wrong", cast(Any, object))
    with pytest.raises(RegistryError, match="unknown metric"):
        build_metric(MetricSpec("missing", "not.registered", "channel"))


def test_builtin_mean_uses_explicit_weights_and_resets_between_scopes() -> None:
    engine = MetricEngine([MetricSpec("average", "mean", "scalar")])

    engine.update(
        {
            "scalar": MetricUpdate(
                args=(torch.tensor([1.0, 3.0]),),
                kwargs={"weight": torch.tensor([1.0, 3.0])},
            )
        }
    )
    assert engine.compute(reset=True) == {"average": pytest.approx(2.5)}

    engine.update({"scalar": MetricUpdate(args=(torch.tensor([10.0]),))})
    assert engine.compute() == {"average": pytest.approx(10.0)}
    with pytest.raises(ValueError, match="fixes nan_strategy"):
        build_metric(
            MetricSpec(
                "average",
                "mean",
                "scalar",
                {"nan_strategy": "ignore"},
            )
        )


def test_builtin_mse_and_mae_share_one_opaque_channel() -> None:
    engine = MetricEngine(
        [
            MetricSpec("mse", "mse", "prediction"),
            MetricSpec("mae", "mae", "prediction"),
        ]
    )

    engine.update(
        {
            "prediction": MetricUpdate(
                args=(
                    torch.tensor([1.0, 3.0]),
                    torch.tensor([0.0, 1.0]),
                )
            )
        }
    )

    assert engine.compute() == {
        "mse": pytest.approx(2.5),
        "mae": pytest.approx(1.5),
    }


def test_engine_detaches_each_channel_once_and_updates_under_no_grad() -> None:
    RecordingSumMetric.observed.clear()
    RecordingSumMetric.grad_enabled.clear()
    engine = MetricEngine(
        [
            MetricSpec("first", "test.recording_sum", "shared"),
            MetricSpec("second", "test.recording_sum", "shared"),
        ]
    )
    value = torch.tensor(3.0, requires_grad=True)

    engine.update({"shared": MetricUpdate(args=(value,))})

    assert len(RecordingSumMetric.observed) == 2
    assert RecordingSumMetric.observed[0] is RecordingSumMetric.observed[1]
    assert not RecordingSumMetric.observed[0].requires_grad
    assert RecordingSumMetric.grad_enabled == [False, False]
    assert value.requires_grad


def test_engine_requires_bound_channels_but_ignores_valid_unbound_channels() -> None:
    engine = MetricEngine([MetricSpec("sum", "test.recording_sum", "bound")])

    assert engine.required_channels == frozenset({"bound"})
    with pytest.raises(MetricRuntimeError, match="missing bound"):
        engine.update({"other": MetricUpdate(args=(torch.tensor(1.0),))})

    engine.update(
        {
            "bound": MetricUpdate(args=(torch.tensor(2.0),)),
            "other": MetricUpdate(args=(torch.tensor(100.0),)),
        }
    )
    assert engine.compute() == {"sum": pytest.approx(2.0)}


def test_engine_flattens_mapping_results_and_moves_metric_state() -> None:
    engine = MetricEngine(
        [
            MetricSpec(
                "statistics",
                "test.configured_result",
                "value",
                {"kind": "flat"},
            )
        ]
    )

    assert engine.to("cpu") is engine
    engine.update({"value": MetricUpdate(args=(torch.tensor(2.0),))})

    assert engine.compute() == {
        "statistics/sum": pytest.approx(2.0),
        "statistics/twice": pytest.approx(4.0),
    }


@pytest.mark.parametrize(
    ("kind", "error_type", "message"),
    [
        ("bool", TypeError, "numeric scalar"),
        ("vector", MetricRuntimeError, "scalar tensor"),
        ("nested", TypeError, "numeric scalar"),
        ("empty", MetricRuntimeError, "empty mapping"),
        ("invalid_key", MetricRuntimeError, "result key"),
        ("list", TypeError, "numeric scalar"),
    ],
)
def test_engine_rejects_unsupported_result_shapes(
    kind: str,
    error_type: type[Exception],
    message: str,
) -> None:
    engine = MetricEngine(
        [
            MetricSpec(
                "invalid",
                "test.configured_result",
                "value",
                {"kind": kind},
            )
        ]
    )
    engine.update({"value": MetricUpdate(args=(torch.tensor(1.0),))})

    with pytest.raises(error_type, match=message):
        engine.compute()


def test_engine_rejects_duplicate_ids_before_constructing_ambiguous_results() -> None:
    with pytest.raises(ValueError, match="duplicate id"):
        MetricEngine(
            [
                MetricSpec("same", "mean", "first"),
                MetricSpec("same", "mean", "second"),
            ]
        )


def test_compute_reset_true_resets_in_finally_after_failure() -> None:
    engine = MetricEngine(
        [MetricSpec("failure", "test.failing_compute", "value")]
    )
    engine.update({"value": MetricUpdate(args=(torch.tensor(1.0),))})
    FailingComputeMetric.reset_calls = 0

    with pytest.raises(RuntimeError, match="intentional"):
        engine.compute(reset=True)

    assert FailingComputeMetric.reset_calls == 1


def test_engine_rejects_compute_without_a_complete_successful_update() -> None:
    engine = MetricEngine(
        [
            MetricSpec("partial", "test.recording_sum", "value"),
            MetricSpec("failure", "test.failing_update", "value"),
        ]
    )

    with pytest.raises(RuntimeError, match="intentional update failure"):
        engine.update({"value": MetricUpdate(args=(torch.tensor(1.0),))})
    with pytest.raises(
        MetricRuntimeError,
        match=r"metrics=partial, failure; channels=value",
    ):
        engine.compute()


def test_engine_reset_clears_successful_update_guard() -> None:
    engine = MetricEngine([MetricSpec("average", "mean", "value")])
    engine.update({"value": MetricUpdate(args=(torch.tensor(1.0),))})
    assert engine.compute() == {"average": pytest.approx(1.0)}

    engine.reset()

    with pytest.raises(MetricRuntimeError, match="before a successful update"):
        engine.compute()


def test_mean_rejects_nan_while_snapshot_preserves_non_finite_facts() -> None:
    engine = MetricEngine([MetricSpec("average", "mean", "value")])

    with pytest.raises(RuntimeError, match=r"[Nn][Aa][Nn]"):
        engine.update(
            {"value": MetricUpdate(args=(torch.tensor([math.nan]),))}
        )

    source = MetricSource("phase", "validation", None, True)
    snapshot = EpochMetricSnapshot(
        {"valid/metrics/external": math.inf},
        {"valid/metrics/external": source},
    )
    assert math.isinf(snapshot.values["valid/metrics/external"])


def test_config_import_does_not_load_torchmetrics_or_register_builtins() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_path = str(repository_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import stochaflow.utils.config",
                    "assert not any(",
                    "    name == 'torchmetrics' or name.startswith('torchmetrics.')",
                    "    for name in sys.modules",
                    ")",
                    "assert 'stochaflow.metrics.builtin' not in sys.modules",
                )
            ),
        ],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
