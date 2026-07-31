"""Task-neutral metric construction declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

METRIC_TAG_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(slots=True)
class MetricSpec:
    """Task-neutral metric construction and update-channel declaration."""

    id: str
    name: str
    channel: str
    params: dict[str, Any] = field(default_factory=dict)


def validate_metric_spec(
    spec: object,
    *,
    path: str = "metric",
) -> MetricSpec:
    """Validate and return one task-neutral metric declaration."""

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
    for name, value in (("name", spec.name), ("channel", spec.channel)):
        declared_value = cast(object, value)
        if not isinstance(declared_value, str) or not declared_value.strip():
            raise ValueError(f"{path}.{name} must be a non-empty string")
        if declared_value != declared_value.strip():
            raise ValueError(f"{path}.{name} must not contain surrounding whitespace")
    if type(cast(object, spec.params)) is not dict:
        raise TypeError(f"{path}.params must be a mapping")
    for key in spec.params:
        declared_key = cast(object, key)
        if not isinstance(declared_key, str) or not declared_key.strip():
            raise ValueError(f"{path}.params keys must be non-empty strings")
        if declared_key != declared_key.strip():
            raise ValueError(
                f"{path}.params keys must not contain surrounding whitespace"
            )
    return spec


__all__ = ["MetricSpec"]
