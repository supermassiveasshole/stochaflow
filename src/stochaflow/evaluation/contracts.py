"""Task-neutral runtime value contracts for formal evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, runtime_checkable

from stochaflow.evaluation.config import (
    EVALUATION_SPLITS,
    EvaluationSplit,
    _freeze_evaluation_mapping,
)
from stochaflow.metrics.config import METRIC_TAG_SEGMENT_PATTERN
from stochaflow.metrics.contracts import MetricUpdate, validate_metric_updates

EvaluationStatus = Literal["complete", "incomplete"]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _non_empty_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result:
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


def _validate_status(value: object, *, path: str) -> EvaluationStatus:
    if value not in {"complete", "incomplete"}:
        raise ValueError(f"{path} must be 'complete' or 'incomplete'")
    return cast(EvaluationStatus, value)


def _snapshot_numeric_mapping(
    value: object,
    *,
    path: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    normalized: dict[str, float] = {}
    for declared_key, declared_value in value.items():
        key = _non_empty_string(declared_key, path=f"{path} key")
        if isinstance(declared_value, bool) or not isinstance(declared_value, Real):
            raise TypeError(f"{path}[{key!r}] must be a real scalar")
        numeric_value = float(declared_value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{path}[{key!r}] must be finite")
        normalized[key] = numeric_value
    return MappingProxyType(normalized)


def _validate_result_keys(
    values: Mapping[str, float],
    *,
    namespace: str,
    path: str,
) -> None:
    for key in values:
        segments = key.split("/")
        if (
            len(segments) not in {3, 4}
            or segments[:2] != ["eval", namespace]
            or any(
                METRIC_TAG_SEGMENT_PATTERN.fullmatch(segment) is None
                for segment in segments[2:]
            )
        ):
            raise ValueError(
                f"{path} keys must use canonical "
                f"'eval/{namespace}/<id>[/<subkey>]' names"
            )


def _validate_artifact_references(value: Any, *, path: str) -> None:
    """Require portable relative paths for serialized artifact references."""

    if isinstance(value, Mapping):
        for declared_key, item in value.items():
            key = _non_empty_string(declared_key, path=f"{path} key")
            item_path = f"{path}[{key!r}]"
            if key == "path":
                reference = _non_empty_string(item, path=item_path)
                posix_path = PurePosixPath(reference)
                windows_path = PureWindowsPath(reference)
                if (
                    posix_path.is_absolute()
                    or windows_path.is_absolute()
                    or ".." in posix_path.parts
                    or ".." in windows_path.parts
                ):
                    raise ValueError(
                        f"{item_path} must be a portable relative path"
                    )
            else:
                _validate_artifact_references(item, path=item_path)
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(cast(Collection[Any], value)):
            _validate_artifact_references(item, path=f"{path}[{index}]")


@runtime_checkable
class Evaluator(Protocol):
    """Interpret one task-owned batch through an injected inference capability."""

    @property
    def metric_channels(self) -> Collection[str]:
        """Return channels this evaluator can emit for metric updates."""

        ...

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        """Evaluate one opaque batch without owning core lifecycle concerns."""

        ...


@dataclass(frozen=True, slots=True)
class EvaluationStepOutput:
    """Validated task output for one evaluation work batch."""

    num_examples: int
    sample_ids: tuple[str, ...]
    metric_update_groups: tuple[Mapping[str, MetricUpdate], ...]
    records: Any | None = None
    measurements: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = cast(object, self.num_examples)
        if type(count) is not int or cast(int, count) <= 0:
            raise ValueError("evaluation step num_examples must be a positive integer")
        if type(cast(object, self.sample_ids)) is not tuple:
            raise TypeError("evaluation step sample_ids must be an exact tuple")
        if len(self.sample_ids) != self.num_examples:
            raise ValueError(
                "evaluation step sample_ids length must equal num_examples"
            )
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for index, declared_id in enumerate(self.sample_ids):
            sample_id = _non_empty_string(
                declared_id,
                path=f"evaluation step sample_ids[{index}]",
            )
            if sample_id in seen:
                raise ValueError(
                    f"evaluation step contains duplicate sample id {sample_id!r}"
                )
            seen.add(sample_id)
            normalized_ids.append(sample_id)
        if type(cast(object, self.metric_update_groups)) is not tuple:
            raise TypeError(
                "evaluation step metric_update_groups must be an exact tuple"
            )
        groups = tuple(
            validate_metric_updates(
                group,
                path=f"evaluation step metric_update_groups[{index}]",
            )
            for index, group in enumerate(self.metric_update_groups)
        )
        measurements = _snapshot_numeric_mapping(
            self.measurements,
            path="evaluation step measurements",
        )
        for measurement_id in measurements:
            if METRIC_TAG_SEGMENT_PATTERN.fullmatch(measurement_id) is None:
                raise ValueError(
                    "evaluation step measurement ids must match "
                    f"{METRIC_TAG_SEGMENT_PATTERN.pattern!r}"
                )
        object.__setattr__(self, "sample_ids", tuple(normalized_ids))
        object.__setattr__(self, "metric_update_groups", groups)
        object.__setattr__(self, "measurements", measurements)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Portable immutable facts published by one evaluation run."""

    schema_version: int
    evaluation_id: str
    protocol_id: str
    protocol_digest: str
    status: EvaluationStatus
    subject: Mapping[str, Any]
    data: Mapping[str, Any]
    metrics: Mapping[str, float]
    measurements: Mapping[str, float]
    artifacts: Mapping[str, Any]
    completeness: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            type(cast(object, self.schema_version)) is not int
            or self.schema_version != 1
        ):
            raise ValueError("evaluation result schema_version must be integer 1")
        _non_empty_string(self.evaluation_id, path="evaluation result evaluation_id")
        _non_empty_string(self.protocol_id, path="evaluation result protocol_id")
        digest = _non_empty_string(
            self.protocol_digest,
            path="evaluation result protocol_digest",
        )
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(
                "evaluation result protocol_digest must be a lowercase SHA-256 digest"
            )
        object.__setattr__(
            self,
            "status",
            _validate_status(self.status, path="evaluation result status"),
        )
        object.__setattr__(
            self,
            "subject",
            _freeze_evaluation_mapping(
                self.subject,
                path="evaluation result subject",
            ),
        )
        object.__setattr__(
            self,
            "data",
            _freeze_evaluation_mapping(self.data, path="evaluation result data"),
        )
        metrics = _snapshot_numeric_mapping(
            self.metrics,
            path="evaluation result metrics",
        )
        _validate_result_keys(
            metrics,
            namespace="metrics",
            path="evaluation result metrics",
        )
        object.__setattr__(self, "metrics", metrics)
        measurements = _snapshot_numeric_mapping(
            self.measurements,
            path="evaluation result measurements",
        )
        _validate_result_keys(
            measurements,
            namespace="measurements",
            path="evaluation result measurements",
        )
        object.__setattr__(self, "measurements", measurements)
        _validate_artifact_references(
            self.artifacts,
            path="evaluation result artifacts",
        )
        object.__setattr__(
            self,
            "artifacts",
            _freeze_evaluation_mapping(
                self.artifacts,
                path="evaluation result artifacts",
            ),
        )
        object.__setattr__(
            self,
            "completeness",
            _freeze_evaluation_mapping(
                self.completeness,
                path="evaluation result completeness",
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_evaluation_mapping(
                self.provenance,
                path="evaluation result provenance",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationRunOutcome:
    """Local paths and convenient scalar views from a completed evaluation run."""

    evaluation_id: str
    protocol_id: str
    status: EvaluationStatus
    output_dir: Path
    subject: Mapping[str, Any]
    split: EvaluationSplit
    metrics: Mapping[str, float]
    measurements: Mapping[str, float]
    artifacts: Mapping[str, Path]
    manifest_path: Path
    result_path: Path

    def __post_init__(self) -> None:
        _non_empty_string(self.evaluation_id, path="evaluation outcome evaluation_id")
        _non_empty_string(self.protocol_id, path="evaluation outcome protocol_id")
        object.__setattr__(
            self,
            "status",
            _validate_status(self.status, path="evaluation outcome status"),
        )
        if self.split not in EVALUATION_SPLITS:
            raise ValueError("evaluation outcome split must be 'validation' or 'test'")
        for path_name in (
            "output_dir",
            "manifest_path",
            "result_path",
        ):
            path_value = cast(object, getattr(self, path_name))
            if not isinstance(path_value, Path):
                raise TypeError(f"evaluation outcome {path_name} must be a Path")
        object.__setattr__(
            self,
            "subject",
            _freeze_evaluation_mapping(
                self.subject,
                path="evaluation outcome subject",
            ),
        )
        metrics = _snapshot_numeric_mapping(
            self.metrics,
            path="evaluation outcome metrics",
        )
        _validate_result_keys(
            metrics,
            namespace="metrics",
            path="evaluation outcome metrics",
        )
        object.__setattr__(self, "metrics", metrics)
        measurements = _snapshot_numeric_mapping(
            self.measurements,
            path="evaluation outcome measurements",
        )
        _validate_result_keys(
            measurements,
            namespace="measurements",
            path="evaluation outcome measurements",
        )
        object.__setattr__(self, "measurements", measurements)
        declared_artifacts = cast(object, self.artifacts)
        if not isinstance(declared_artifacts, Mapping):
            raise TypeError("evaluation outcome artifacts must be a mapping")
        artifacts: dict[str, Path] = {}
        for declared_name, declared_path in cast(
            Mapping[object, object],
            declared_artifacts,
        ).items():
            name = _non_empty_string(
                declared_name,
                path="evaluation outcome artifact name",
            )
            if not isinstance(cast(object, declared_path), Path):
                raise TypeError(f"evaluation outcome artifact {name!r} must be a Path")
            artifacts[name] = cast(Path, declared_path)
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))


__all__ = [
    "EvaluationResult",
    "EvaluationRunOutcome",
    "EvaluationStatus",
    "EvaluationStepOutput",
    "Evaluator",
]
