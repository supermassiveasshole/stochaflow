"""Contract tests for task-neutral metric construction and runtime state."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
from torchmetrics import Metric

from stochaflow.metrics import (
    MetricEngine,
    MetricRuntimeError,
    MetricSpec,
    MetricUpdate,
    build_metric,
)
from stochaflow.metrics.contracts import (
    prepare_metric_updates,
    validate_metric_updates,
)
from stochaflow.utils.config import (
    TrainingMetricConfig,
    validate_training_metric_configs,
    validate_training_monitor_key,
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


class MutatingFailingUpdateMetric(Metric):
    """Mutate state before rejecting negative inputs."""

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
        if torch.any(value < 0):
            raise RuntimeError("intentional failure after mutation")

    def compute(self) -> torch.Tensor:
        return self.total


REGISTRIES.metrics.add("test.recording_sum", RecordingSumMetric)
REGISTRIES.metrics.add("test.configured_result", ConfiguredResultMetric)
REGISTRIES.metrics.add("test.failing_compute", FailingComputeMetric)
REGISTRIES.metrics.add("test.failing_update", FailingUpdateMetric)
REGISTRIES.metrics.add(
    "test.mutating_failing_update",
    MutatingFailingUpdateMetric,
)


def test_metric_update_validates_and_freezes_kwargs() -> None:
    update = MetricUpdate(args=(1,), kwargs={"weight": 2})

    assert update.args == (1,)
    assert dict(update.kwargs) == {"weight": 2}
    with pytest.raises(TypeError):
        cast(dict[str, Any], update.kwargs)["other"] = 3
    with pytest.raises(TypeError, match="args must be an exact tuple"):
        MetricUpdate(args=cast(Any, [1]))
    with pytest.raises(TypeError, match="kwargs must be an exact dictionary"):
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

    prepared = prepare_metric_updates({"channel": update})
    detached = prepared.values["channel"]

    args_value = cast(dict[str, list[torch.Tensor]], detached.args[0])
    kwargs_value = cast(tuple[torch.Tensor, ...], detached.kwargs["metadata"])
    assert value.requires_grad
    assert not args_value["nested"][0].requires_grad
    assert not kwargs_value[0].requires_grad


def test_metric_registry_reserves_native_torchmetrics_namespace() -> None:
    with pytest.raises(ValueError, match="reserved namespace"):
        REGISTRIES.metrics.add(
            "torchmetrics.classification.MulticlassAccuracy",
            RecordingSumMetric,
        )


def test_metric_config_validation_is_local_strict_and_detects_duplicates() -> None:
    configs = [
        TrainingMetricConfig(
            id="reconstruction_mse",
            name="mse",
            channel="gaussian.clean_reconstruction",
            phases=["train", "validation"],
        ),
        TrainingMetricConfig(
            id="prediction_mae",
            name="mae",
            channel="gaussian.prediction_target",
            phases=["test"],
        ),
    ]

    validate_training_metric_configs(configs)
    first_spec = configs[0].to_metric_spec()
    assert first_spec == MetricSpec(
        id="reconstruction_mse",
        name="mse",
        channel="gaussian.clean_reconstruction",
    )
    assert first_spec.params is not configs[0].params

    with pytest.raises(ValueError, match="duplicate metric id"):
        validate_training_metric_configs([configs[0], configs[0]])
    with pytest.raises(ValueError, match="duplicate phase"):
        validate_training_metric_configs(
            [
                TrainingMetricConfig(
                    id="mse",
                    name="mse",
                    channel="prediction",
                    phases=["train", "train"],
                )
            ]
        )
    with pytest.raises(ValueError, match="must match"):
        validate_training_metric_configs(
            [
                TrainingMetricConfig(
                    id="bad/id",
                    name="mse",
                    channel="prediction",
                )
            ]
        )
    with pytest.raises(ValueError, match="train, validation, or test"):
        validate_training_metric_configs(
            [
                TrainingMetricConfig(
                    id="mse",
                    name="mse",
                    channel="prediction",
                    phases=["predict"],
                )
            ]
        )
    with pytest.raises(TypeError, match="params must be a mapping"):
        validate_training_metric_configs(
            [
                TrainingMetricConfig(
                    id="mse",
                    name="mse",
                    channel="prediction",
                    params=cast(Any, []),
                )
            ]
        )


@pytest.mark.parametrize(
    "monitor",
    [
        "valid/loss",
        "valid/metrics/prediction_mae",
        "valid/metrics/scorecard/mae",
    ],
)
def test_training_monitor_accepts_only_validation_results(monitor: str) -> None:
    assert validate_training_monitor_key(monitor) == monitor


@pytest.mark.parametrize(
    "monitor",
    [
        "train/loss",
        "test/loss",
        "diagnostics/quality/fid",
        "valid/metrics/id/nested/subkey",
    ],
)
def test_training_monitor_rejects_non_validation_results(monitor: str) -> None:
    with pytest.raises(ValueError, match="valid/loss"):
        validate_training_monitor_key(monitor)


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


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("mse", {"squared": False}, "fixes squared=True"),
        ("mse", {"num_outputs": 2}, "fixes num_outputs=1"),
        ("mse", {"num_outputs": 1.0}, "fixes num_outputs=1"),
        ("mae", {"num_outputs": 2}, "fixes num_outputs=1"),
        ("mae", {"num_outputs": True}, "fixes num_outputs=1"),
    ],
)
def test_builtin_error_metrics_reject_non_scalar_semantics(
    name: str,
    params: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_metric(MetricSpec("error", name, "prediction", params))


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

    with pytest.raises(
        MetricRuntimeError,
        match=r"metric 'failure'.*channel 'value'",
    ) as error:
        engine.update({"value": MetricUpdate(args=(torch.tensor(1.0),))})
    assert isinstance(error.value.__cause__, RuntimeError)
    with pytest.raises(
        MetricRuntimeError,
        match=r"metrics=partial, failure; channels=value",
    ):
        engine.compute()


def test_failed_metric_update_resets_all_accumulated_and_partial_state() -> None:
    engine = MetricEngine(
        [
            MetricSpec("sum", "test.recording_sum", "value"),
            MetricSpec(
                "guarded",
                "test.mutating_failing_update",
                "value",
            ),
        ]
    )
    engine.update({"value": MetricUpdate(args=(torch.tensor(2.0),))})

    with pytest.raises(
        MetricRuntimeError,
        match=r"metric 'guarded'.*channel 'value'",
    ) as error:
        engine.update({"value": MetricUpdate(args=(torch.tensor(-1.0),))})

    assert isinstance(error.value.__cause__, RuntimeError)
    engine.update({"value": MetricUpdate(args=(torch.tensor(3.0),))})
    assert engine.compute() == {
        "sum": pytest.approx(3.0),
        "guarded": pytest.approx(3.0),
    }


def test_engine_reset_clears_successful_update_guard() -> None:
    engine = MetricEngine([MetricSpec("average", "mean", "value")])
    engine.update({"value": MetricUpdate(args=(torch.tensor(1.0),))})
    assert engine.compute() == {"average": pytest.approx(1.0)}

    engine.reset()

    with pytest.raises(MetricRuntimeError, match="before a successful update"):
        engine.compute()


def test_mean_rejects_nan() -> None:
    engine = MetricEngine([MetricSpec("average", "mean", "value")])

    with pytest.raises(
        MetricRuntimeError,
        match=r"metric 'average'.*channel 'value'",
    ) as error:
        engine.update(
            {"value": MetricUpdate(args=(torch.tensor([math.nan]),))}
        )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "nan" in str(error.value.__cause__).lower()



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
