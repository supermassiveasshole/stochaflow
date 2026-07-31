"""Strict task-neutral and training-facing metric declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

METRIC_TAG_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_METRIC_TAG_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
_PHASE_EPOCH_METRIC_PATTERN = re.compile(
    rf"^(?P<phase>train|valid|test)/(?:loss|metrics/"
    rf"{_METRIC_TAG_SEGMENT}(?:/{_METRIC_TAG_SEGMENT})?)$"
)
_DIAGNOSTIC_EPOCH_METRIC_PATTERN = re.compile(
    rf"^diagnostics/{_METRIC_TAG_SEGMENT}/"
    rf"{_METRIC_TAG_SEGMENT}(?:/{_METRIC_TAG_SEGMENT})*$"
)
_SYSTEM_EPOCH_METRIC_PATTERN = re.compile(
    rf"^system/{_METRIC_TAG_SEGMENT}/"
    rf"{_METRIC_TAG_SEGMENT}(?:/{_METRIC_TAG_SEGMENT})*$"
)
TRAINING_METRIC_PHASES = frozenset({"train", "validation", "test"})
type EpochMetricKeyOrigin = Literal["phase", "diagnostic", "system"]
type EpochMetricKeyDataRole = Literal["train", "validation", "test"]


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
    """Validate a selectable phase or diagnostic epoch metric key."""

    monitor = _non_empty_string(value, path=path)
    message = (
        f"{path} must use a canonical epoch metric key such as "
        "'train/loss', 'valid/loss', or "
        "'diagnostics/quality/fid'"
    )
    try:
        origin, _ = classify_epoch_metric_key(monitor, path=path)
    except ValueError as error:
        raise ValueError(message) from error
    if origin == "system":
        raise ValueError(message)
    return monitor


def classify_epoch_metric_key(
    value: object,
    *,
    path: str = "epoch metric key",
) -> tuple[EpochMetricKeyOrigin, EpochMetricKeyDataRole | None]:
    """Classify one strict canonical epoch metric key."""

    key = _non_empty_string(value, path=path)
    phase_match = _PHASE_EPOCH_METRIC_PATTERN.fullmatch(key)
    if phase_match is not None:
        phase = phase_match.group("phase")
        data_role: EpochMetricKeyDataRole = (
            "validation" if phase == "valid" else cast(EpochMetricKeyDataRole, phase)
        )
        return "phase", data_role
    if _DIAGNOSTIC_EPOCH_METRIC_PATTERN.fullmatch(key) is not None:
        return "diagnostic", None
    if _SYSTEM_EPOCH_METRIC_PATTERN.fullmatch(key) is not None:
        return "system", None
    raise ValueError(
        f"{path} must use a canonical epoch metric key such as "
        "'train/loss', 'valid/metrics/prediction_mae', "
        "'diagnostics/quality/fid', or "
        "'system/train/skipped_optimizer_steps'"
    )


__all__ = [
    "TRAINING_METRIC_PHASES",
    "MetricConfig",
    "MetricSpec",
    "validate_metric_configs",
    "validate_metric_spec",
    "validate_training_monitor_key",
]
