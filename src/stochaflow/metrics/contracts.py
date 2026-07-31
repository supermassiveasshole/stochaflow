"""Task-neutral data contracts shared by metric producers and consumers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Number, Real
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, runtime_checkable

import torch

from stochaflow.metrics.config import classify_epoch_metric_key

type MetricOrigin = Literal["phase", "diagnostic", "system"]
type MetricDataRole = Literal["train", "validation", "test", "external"]

METRIC_SOURCE_FIELDS = frozenset(
    {
        "origin",
        "data_role",
        "protocol_id",
        "selection_eligible",
    }
)
EPOCH_METRIC_SNAPSHOT_FIELDS = frozenset({"values", "sources"})


@runtime_checkable
class MetricPayloadDetachable(Protocol):
    """Explicit detach capability for a custom stateful metric payload."""

    def detach_metric_payload(self) -> Any:
        """Return an equivalent payload with every contained tensor detached."""


def _non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return value


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} field names must be strings")
    names = cast(set[str], set(value))
    missing = sorted(fields - names)
    unknown = sorted(names - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")
    return cast(Mapping[str, Any], value)


@dataclass(frozen=True, slots=True)
class MetricUpdate:
    """Opaque positional and keyword arguments for one documented channel."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.args), tuple):
            raise TypeError("metric update.args must be a tuple")
        if not isinstance(cast(object, self.kwargs), Mapping):
            raise TypeError("metric update.kwargs must be a mapping")
        normalized: dict[str, Any] = {}
        for key, value in self.kwargs.items():
            _non_empty_string(key, path="metric update.kwargs key")
            normalized[key] = value
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


def _is_named_tuple_instance(value: object) -> bool:
    fields = getattr(type(value), "_fields", None)
    return not (
        not isinstance(value, tuple)
        or not isinstance(fields, tuple)
        or any(not isinstance(field_name, str) for field_name in fields)
    )


def _unsupported_metric_payload(value: object) -> TypeError:
    return TypeError(
        "metric payload contains unsupported value type "
        f"{type(value).__name__!r}; custom stateful payload containers must "
        "implement MetricPayloadDetachable.detach_metric_payload()"
    )


def detach_metric_value(value: Any) -> Any:
    """Recursively detach tensors in a supported metric payload tree."""

    if isinstance(value, torch.Tensor):
        return value.detach()
    if type(value) is dict:
        return {
            key: detach_metric_value(item)
            for key, item in value.items()
        }
    if type(value) is OrderedDict:
        return OrderedDict(
            (key, detach_metric_value(item))
            for key, item in value.items()
        )
    if type(value) is MappingProxyType:
        return MappingProxyType(
            {
                key: detach_metric_value(item)
                for key, item in value.items()
            }
        )
    if type(value) is tuple:
        return tuple(detach_metric_value(item) for item in value)
    if _is_named_tuple_instance(value):
        constructor = cast(Any, type(value))
        return constructor(*(detach_metric_value(item) for item in value))
    if type(value) is list:
        return [detach_metric_value(item) for item in value]
    if type(value) is torch.Size:
        return torch.Size(detach_metric_value(item) for item in value)
    if isinstance(value, MetricPayloadDetachable):
        return value.detach_metric_payload()
    if value is None or isinstance(
        value,
        (
            str,
            bytes,
            Number,
            Enum,
            torch.device,
            torch.dtype,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    raise _unsupported_metric_payload(value)


def detach_metric_update(update: MetricUpdate) -> MetricUpdate:
    """Return one update whose nested tensor values are detached."""

    if not isinstance(cast(object, update), MetricUpdate):
        raise TypeError("metric update must be a MetricUpdate")
    return MetricUpdate(
        args=cast(tuple[Any, ...], detach_metric_value(update.args)),
        kwargs=cast(Mapping[str, Any], detach_metric_value(update.kwargs)),
    )


def detach_metric_updates(
    updates: object,
    *,
    path: str = "metric updates",
) -> Mapping[str, MetricUpdate]:
    """Validate and detach every channel payload exactly once."""

    validated = validate_metric_updates(updates, path=path)
    return MappingProxyType(
        {
            channel: detach_metric_update(update)
            for channel, update in validated.items()
        }
    )


@dataclass(frozen=True, slots=True)
class MetricSource:
    """Minimal provenance and selection metadata for one epoch metric."""

    origin: MetricOrigin
    data_role: MetricDataRole | None
    protocol_id: str | None
    selection_eligible: bool

    def __post_init__(self) -> None:
        if self.origin not in {"phase", "diagnostic", "system"}:
            raise ValueError(
                "metric source.origin must be phase, diagnostic, or system"
            )
        if self.data_role not in {
            None,
            "train",
            "validation",
            "test",
            "external",
        }:
            raise ValueError(
                "metric source.data_role must be train, validation, test, "
                "external, or null"
            )
        if not isinstance(cast(object, self.selection_eligible), bool):
            raise TypeError("metric source.selection_eligible must be a bool")
        if self.protocol_id is not None:
            _non_empty_string(
                self.protocol_id,
                path="metric source.protocol_id",
            )
        if self.origin == "system":
            if self.data_role is not None or self.selection_eligible:
                raise ValueError(
                    "system metric sources require data_role=null and "
                    "selection_eligible=false"
                )
        elif self.data_role is None:
            raise ValueError(
                "phase and diagnostic metric sources require a data_role"
            )
        if self.origin == "phase" and self.data_role == "external":
            raise ValueError(
                "phase metric sources require train, validation, or test data"
            )
        if self.data_role == "test" and self.selection_eligible:
            raise ValueError(
                "test-role metric sources cannot be selection eligible"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this source using the stable checkpoint representation."""

        return {
            "origin": self.origin,
            "data_role": self.data_role,
            "protocol_id": self.protocol_id,
            "selection_eligible": self.selection_eligible,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "metric source",
    ) -> MetricSource:
        """Parse strict source metadata from a persisted mapping."""

        raw = _strict_mapping(value, fields=METRIC_SOURCE_FIELDS, path=path)
        return cls(
            origin=raw["origin"],
            data_role=raw["data_role"],
            protocol_id=raw["protocol_id"],
            selection_eligible=raw["selection_eligible"],
        )


def _validate_source_key_consistency(
    key: str,
    source: MetricSource,
) -> None:
    origin, data_role = classify_epoch_metric_key(
        key,
        path="epoch metric snapshot key",
    )
    if origin != source.origin:
        raise ValueError(
            f"{source.origin} metric source conflicts with canonical "
            f"{origin} metric key {key!r}"
        )
    if origin != "phase":
        return
    if source.data_role != data_role:
        raise ValueError(
            f"phase metric key {key!r} conflicts with data role "
            f"{source.data_role!r}; canonical key has data role {data_role!r}"
        )


@dataclass(frozen=True, slots=True)
class EpochMetricSnapshot:
    """Read-only canonical metric values and their matching sources."""

    values: Mapping[str, float]
    sources: Mapping[str, MetricSource]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.values), Mapping):
            raise TypeError("epoch metric snapshot.values must be a mapping")
        if not isinstance(cast(object, self.sources), Mapping):
            raise TypeError("epoch metric snapshot.sources must be a mapping")
        normalized_values: dict[str, float] = {}
        for key, value in self.values.items():
            _non_empty_string(key, path="epoch metric snapshot value key")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"epoch metric snapshot.values[{key!r}] must be numeric"
                )
            normalized_values[key] = float(value)
        normalized_sources: dict[str, MetricSource] = {}
        for key, source in self.sources.items():
            _non_empty_string(key, path="epoch metric snapshot source key")
            if not isinstance(cast(object, source), MetricSource):
                raise TypeError(
                    f"epoch metric snapshot.sources[{key!r}] must be a "
                    "MetricSource"
                )
            normalized_sources[key] = source
        value_keys = set(normalized_values)
        source_keys = set(normalized_sources)
        if value_keys != source_keys:
            missing_sources = sorted(value_keys - source_keys)
            unknown_sources = sorted(source_keys - value_keys)
            raise ValueError(
                "epoch metric snapshot sources must exactly match values: "
                f"missing={missing_sources or '<none>'}, "
                f"unknown={unknown_sources or '<none>'}"
            )
        for key, source in normalized_sources.items():
            _validate_source_key_consistency(key, source)
        object.__setattr__(
            self,
            "values",
            MappingProxyType(normalized_values),
        )
        object.__setattr__(
            self,
            "sources",
            MappingProxyType(normalized_sources),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this snapshot without exposing its internal mappings."""

        return {
            "values": dict(self.values),
            "sources": {
                key: source.to_dict()
                for key, source in self.sources.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "epoch metric snapshot",
    ) -> EpochMetricSnapshot:
        """Parse a strict serialized snapshot."""

        raw = _strict_mapping(
            value,
            fields=EPOCH_METRIC_SNAPSHOT_FIELDS,
            path=path,
        )
        values = raw["values"]
        sources = raw["sources"]
        if not isinstance(values, Mapping):
            raise TypeError(f"{path}.values must be a mapping")
        if not isinstance(sources, Mapping):
            raise TypeError(f"{path}.sources must be a mapping")
        parsed_sources: dict[str, MetricSource] = {}
        for key, source in sources.items():
            _non_empty_string(key, path=f"{path}.sources key")
            parsed_sources[key] = MetricSource.from_dict(
                source,
                path=f"{path}.sources[{key!r}]",
            )
        return cls(values=cast(Mapping[str, float], values), sources=parsed_sources)


__all__ = [
    "EpochMetricSnapshot",
    "MetricDataRole",
    "MetricOrigin",
    "MetricPayloadDetachable",
    "MetricSource",
    "MetricUpdate",
    "detach_metric_update",
    "detach_metric_updates",
    "detach_metric_value",
    "validate_metric_updates",
]
