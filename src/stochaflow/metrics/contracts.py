"""Task-neutral payload contracts shared by metric producers and consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import torch

_METRIC_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes)
_METRIC_MAPPING_KEY_TYPES = (type(None), bool, int, float, complex, str, bytes)


def _non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return value


def _validate_mapping_key(value: object, *, path: str) -> None:
    if type(value) not in _METRIC_MAPPING_KEY_TYPES:
        raise TypeError(
            f"{path} has unsupported key type {type(value).__name__!r}; "
            "metric payload mapping keys must be immutable scalar values"
        )


def _validate_metric_value(value: object, *, path: str) -> None:
    if isinstance(value, torch.Tensor) or type(value) in _METRIC_LEAF_TYPES:
        return
    if type(value) is tuple or type(value) is list:
        for index, item in enumerate(cast(tuple[Any, ...] | list[Any], value)):
            _validate_metric_value(item, path=f"{path}[{index}]")
        return
    if type(value) is dict or type(value) is MappingProxyType:
        for key, item in cast(Mapping[Any, Any], value).items():
            _validate_mapping_key(key, path=path)
            _validate_metric_value(item, path=f"{path}[{key!r}]")
        return
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__name__!r}; "
        "metric payloads support tensors, scalar leaves, and exact "
        "dict/list/tuple trees only"
    )


def _detach_metric_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if type(value) is tuple:
        return tuple(_detach_metric_value(item) for item in value)
    if type(value) is list:
        return [_detach_metric_value(item) for item in value]
    if type(value) is dict:
        return {
            key: _detach_metric_value(item)
            for key, item in value.items()
        }
    if type(value) is MappingProxyType:
        return MappingProxyType(
            {
                key: _detach_metric_value(item)
                for key, item in value.items()
            }
        )
    return value


@dataclass(frozen=True, slots=True)
class MetricUpdate:
    """Opaque positional and keyword arguments for one documented channel."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.args) is not tuple:
            raise TypeError("metric update.args must be an exact tuple")
        if type(self.kwargs) not in {dict, MappingProxyType}:
            raise TypeError("metric update.kwargs must be an exact dictionary")
        normalized: dict[str, Any] = {}
        for key, value in self.kwargs.items():
            _non_empty_string(key, path="metric update.kwargs key")
            _validate_metric_value(value, path=f"metric update.kwargs[{key!r}]")
            normalized[key] = value
        for index, value in enumerate(self.args):
            _validate_metric_value(value, path=f"metric update.args[{index}]")
        object.__setattr__(self, "kwargs", MappingProxyType(normalized))


def validate_metric_updates(
    updates: object,
    *,
    path: str = "metric updates",
) -> Mapping[str, MetricUpdate]:
    """Validate and freeze a channel-to-update mapping."""

    if not isinstance(updates, Mapping):
        raise TypeError(f"{path} must be a mapping")
    normalized: dict[str, MetricUpdate] = {}
    for channel, update in updates.items():
        _non_empty_string(channel, path=f"{path} channel")
        if not isinstance(update, MetricUpdate):
            raise TypeError(
                f"{path}[{channel!r}] must be a MetricUpdate, "
                f"got {type(update).__name__}"
            )
        normalized[channel] = update
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class PreparedMetricUpdates:
    """Internal detached channel payloads awaiting a metric-state commit."""

    values: Mapping[str, MetricUpdate]

    def __post_init__(self) -> None:
        if type(self.values) is not MappingProxyType:
            raise TypeError("prepared metric updates must use a read-only mapping")


def prepare_metric_updates(
    updates: object,
    *,
    path: str = "metric updates",
) -> PreparedMetricUpdates:
    """Validate and detach a channel mapping once before deferred commit."""

    validated = validate_metric_updates(updates, path=path)
    detached = {
        channel: MetricUpdate(
            args=cast(tuple[Any, ...], _detach_metric_value(update.args)),
            kwargs=cast(Mapping[str, Any], _detach_metric_value(update.kwargs)),
        )
        for channel, update in validated.items()
    }
    return PreparedMetricUpdates(MappingProxyType(detached))


__all__ = ["MetricUpdate"]
