"""Strict task-neutral and training-facing metric declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

METRIC_TAG_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRAINING_MONITOR_PATTERN = re.compile(
    r"^(?:train|valid)/(?:loss|metrics/"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?)$"
)
TRAINING_METRIC_PHASES = frozenset({"train", "validation", "test"})


@dataclass(slots=True)
class MetricSpec:
    """Task-neutral metric construction and update-channel declaration."""

    id: str
    name: str
    channel: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricConfig(MetricSpec):
    """Metric declaration bound to one or more training phases."""

    phases: list[str] = field(default_factory=lambda: ["validation"])


def _non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return value


def validate_metric_spec(
    spec: object,
    *,
    path: str = "metric",
) -> None:
    """Validate fields shared by every metric runtime."""

    if not isinstance(spec, MetricSpec):
        raise TypeError(f"{path} must be a MetricSpec")
    metric_id = cast(object, spec.id)
    if (
        not isinstance(metric_id, str)
        or METRIC_TAG_SEGMENT_PATTERN.fullmatch(metric_id) is None
    ):
        raise ValueError(
            f"{path}.id must match {METRIC_TAG_SEGMENT_PATTERN.pattern!r}"
        )
    _non_empty_string(spec.name, path=f"{path}.name")
    _non_empty_string(spec.channel, path=f"{path}.channel")
    if not isinstance(cast(object, spec.params), dict):
        raise TypeError(f"{path}.params must be a mapping")
    for key in spec.params:
        _non_empty_string(key, path=f"{path}.params key")


def validate_metric_configs(
    configs: object,
    *,
    path: str = "metrics",
) -> list[MetricConfig]:
    """Validate and return local bindings without importing implementations."""

    if not isinstance(configs, list):
        raise TypeError(f"{path} must be a list")
    seen_ids: set[str] = set()
    for index, config in enumerate(configs):
        item_path = f"{path}[{index}]"
        if not isinstance(config, MetricConfig):
            raise TypeError(f"{item_path} must be a MetricConfig")
        validate_metric_spec(config, path=item_path)
        phases = cast(object, config.phases)
        if not isinstance(phases, list):
            raise TypeError(f"{item_path}.phases must be a list")
        if not phases:
            raise ValueError(f"{item_path}.phases must not be empty")
        seen_phases: set[str] = set()
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, str) or phase not in TRAINING_METRIC_PHASES:
                raise ValueError(
                    f"{item_path}.phases[{phase_index}] must be train, "
                    "validation, or test"
                )
            if phase in seen_phases:
                raise ValueError(
                    f"{item_path}.phases contains duplicate phase {phase!r}"
                )
            seen_phases.add(phase)
        if config.id in seen_ids:
            raise ValueError(f"{path} contains duplicate metric id {config.id!r}")
        seen_ids.add(config.id)
    return configs


def validate_training_monitor_key(
    value: object,
    *,
    path: str = "training monitor",
) -> str:
    """Validate a monitor supported by the M0-M1 epoch snapshot."""

    monitor = _non_empty_string(value, path=path)
    if _TRAINING_MONITOR_PATTERN.fullmatch(monitor) is None:
        raise ValueError(
            f"{path} must use a canonical epoch metric key such as "
            "'train/loss', 'valid/loss', or "
            "'valid/metrics/prediction_mae'"
        )
    return monitor


__all__ = [
    "TRAINING_METRIC_PHASES",
    "MetricConfig",
    "MetricSpec",
    "validate_metric_configs",
    "validate_metric_spec",
    "validate_training_monitor_key",
]
