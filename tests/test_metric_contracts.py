"""Focused tests for the task-neutral metric payload boundary."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import pytest
import torch

from stochaflow import metrics
from stochaflow.metrics import MetricUpdate
from stochaflow.metrics.contracts import prepare_metric_updates


@dataclass
class StatefulPayload:
    """A custom state container that the metric boundary must reject."""

    value: torch.Tensor


class TensorPair(NamedTuple):
    """Named tuple payload that must not bypass the exact-tree boundary."""

    prediction: torch.Tensor
    target: torch.Tensor


def test_public_metrics_surface_excludes_training_and_provenance_contracts() -> None:
    assert set(metrics.__all__) == {
        "ErrorOnNanMeanMetric",
        "FrechetInceptionDistanceMetric",
        "KernelInceptionDistanceMetric",
        "MetricEngine",
        "MetricRuntimeError",
        "MetricSpec",
        "MetricUpdate",
        "SingleOutputMeanAbsoluteError",
        "SingleOutputMeanSquaredError",
        "build_metric",
    }
    for name in (
        "EpochMetricSnapshot",
        "MetricConfig",
        "MetricDataRole",
        "MetricOrigin",
        "MetricPayloadDetachable",
        "MetricSource",
        "detach_metric_update",
        "detach_metric_updates",
        "detach_metric_value",
        "validate_metric_configs",
        "validate_metric_updates",
        "validate_training_monitor_key",
    ):
        assert not hasattr(metrics, name)


def test_payload_tree_accepts_exact_builtin_containers_and_scalar_leaves() -> None:
    value = torch.tensor(1.0, requires_grad=True)
    update = MetricUpdate(
        args=(
            {
                "list": [value, None, True, 3, 2.5],
                "tuple": ("label", b"bytes", 1 + 2j),
            },
        ),
        kwargs={"weight": value},
    )

    prepared = prepare_metric_updates({"channel": update})
    detached = prepared.values["channel"]
    payload = cast(dict[str, Any], detached.args[0])

    assert value.requires_grad
    assert not cast(torch.Tensor, payload["list"][0]).requires_grad
    assert not cast(torch.Tensor, detached.kwargs["weight"]).requires_grad
    with pytest.raises(TypeError):
        cast(dict[str, MetricUpdate], prepared.values)["other"] = update


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(StatefulPayload(torch.tensor(1.0)), id="dataclass"),
        pytest.param(
            TensorPair(torch.tensor(1.0), torch.tensor(2.0)),
            id="named-tuple",
        ),
        pytest.param(OrderedDict(value=torch.tensor(1.0)), id="ordered-dict"),
        pytest.param(torch.Size((2, 3)), id="torch-size"),
        pytest.param(object(), id="opaque-object"),
    ],
)
def test_payload_tree_rejects_custom_and_container_subclass_values(
    payload: object,
) -> None:
    with pytest.raises(TypeError, match="unsupported value type"):
        MetricUpdate(args=(payload,))


def test_payload_tree_rejects_stateful_nested_mapping_keys() -> None:
    with pytest.raises(TypeError, match="unsupported key type"):
        MetricUpdate(args=({cast(Any, object()): torch.tensor(1.0)},))
