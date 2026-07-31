"""Focused tests for canonical metric keys and detached payload contracts."""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import NamedTuple, cast

import pytest
import torch

from stochaflow.metrics import (
    EpochMetricSnapshot,
    MetricPayloadDetachable,
    MetricSource,
    MetricUpdate,
    detach_metric_update,
    detach_metric_value,
    validate_training_monitor_key,
)


class TensorPair(NamedTuple):
    """Named tuple containing two metric payload values."""

    prediction: torch.Tensor
    target: torch.Tensor


class ExplicitlyDetachablePayload:
    """Custom payload that opts into the public detach contract."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def detach_metric_payload(self) -> ExplicitlyDetachablePayload:
        """Return a detached payload of the same public type."""

        return ExplicitlyDetachablePayload(self.value.detach())


class UnsupportedStatefulPayload:
    """Custom stateful payload without the explicit detach capability."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value


@pytest.mark.parametrize(
    "key",
    [
        "train/loss",
        "valid/loss",
        "test/loss",
        "train/metrics/mse",
        "valid/metrics/class_f1/macro",
        "test/metrics/accuracy/top-5",
        "diagnostics/quality/fid",
        "diagnostics/quality/reference/fid",
    ],
)
def test_training_monitor_accepts_canonical_phase_and_diagnostic_keys(
    key: str,
) -> None:
    assert validate_training_monitor_key(key) == key


@pytest.mark.parametrize(
    "key",
    [
        "train/legacy",
        "valid/metrics",
        "test/metrics/id/nested/subkey",
        "diagnostics/quality",
        "diagnostics//fid",
        "system/train/loss",
        " train/loss",
    ],
)
def test_training_monitor_rejects_noncanonical_or_system_keys(key: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"canonical epoch metric key|whitespace",
    ):
        validate_training_monitor_key(key)


def test_snapshot_accepts_each_canonical_metric_namespace() -> None:
    train_source = MetricSource("phase", "train", None, True)
    test_source = MetricSource("phase", "test", None, False)
    diagnostic_source = MetricSource(
        "diagnostic",
        "validation",
        "sha256:" + ("a" * 64),
        True,
    )
    system_source = MetricSource("system", None, None, False)
    values = {
        "train/loss": 1.0,
        "train/metrics/mse/batch": 2.0,
        "test/metrics/accuracy": 3.0,
        "diagnostics/quality/reference/fid": 4.0,
        "system/train/skipped_optimizer_steps": 5.0,
    }

    snapshot = EpochMetricSnapshot(
        values=values,
        sources={
            "train/loss": train_source,
            "train/metrics/mse/batch": train_source,
            "test/metrics/accuracy": test_source,
            "diagnostics/quality/reference/fid": diagnostic_source,
            "system/train/skipped_optimizer_steps": system_source,
        },
    )

    assert dict(snapshot.values) == values


@pytest.mark.parametrize(
    ("key", "source"),
    [
        ("train/legacy", MetricSource("phase", "train", None, True)),
        (
            "valid/metrics/id/nested/subkey",
            MetricSource("phase", "validation", None, True),
        ),
        (
            "diagnostics/quality",
            MetricSource("diagnostic", "validation", "sha256:" + ("b" * 64), True),
        ),
        ("system/train", MetricSource("system", None, None, False)),
    ],
)
def test_snapshot_rejects_loose_namespace_prefixes(
    key: str,
    source: MetricSource,
) -> None:
    with pytest.raises(ValueError, match="canonical epoch metric key"):
        EpochMetricSnapshot(values={key: 1.0}, sources={key: source})


def test_snapshot_rejects_canonical_key_from_the_wrong_source() -> None:
    source = MetricSource("diagnostic", "validation", "sha256:" + ("c" * 64), True)

    with pytest.raises(ValueError, match="conflicts with canonical phase"):
        EpochMetricSnapshot(
            values={"valid/metrics/fid": 1.0},
            sources={"valid/metrics/fid": source},
        )


def test_detach_preserves_supported_container_types() -> None:
    value = torch.tensor(1.0, requires_grad=True)
    named = TensorPair(value, value)
    ordered = OrderedDict([("named", named)])

    detached = detach_metric_value(
        {
            "list": [value],
            "tuple": (value,),
            "named": named,
            "ordered": ordered,
        }
    )

    assert type(detached) is dict
    assert type(detached["list"]) is list
    assert type(detached["tuple"]) is tuple
    assert type(detached["named"]) is TensorPair
    assert type(detached["ordered"]) is OrderedDict
    assert not detached["named"].prediction.requires_grad
    assert not detached["ordered"]["named"].target.requires_grad


def test_detach_preserves_mapping_proxy_as_read_only() -> None:
    value = torch.tensor(1.0, requires_grad=True)

    detached = detach_metric_value(MappingProxyType({"value": value}))

    assert type(detached) is MappingProxyType
    assert not detached["value"].requires_grad
    with pytest.raises(TypeError):
        cast(dict[str, torch.Tensor], detached)["other"] = value


def test_custom_payload_requires_explicit_runtime_detach_capability() -> None:
    value = torch.tensor(1.0, requires_grad=True)
    supported = ExplicitlyDetachablePayload(value)

    assert isinstance(supported, MetricPayloadDetachable)
    detached = detach_metric_value(supported)
    assert isinstance(detached, ExplicitlyDetachablePayload)
    assert detached is not supported
    assert not detached.value.requires_grad

    with pytest.raises(TypeError, match="MetricPayloadDetachable"):
        detach_metric_update(MetricUpdate(args=(UnsupportedStatefulPayload(value),)))
