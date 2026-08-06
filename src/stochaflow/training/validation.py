"""Narrow contracts for validation evaluation during training."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from stochaflow.utils.config import validate_training_monitor_key

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EPOCH_VALIDATION_STATE_SCHEMA_VERSION = 1


def _positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{path} must be a positive int")
    return value


def _non_negative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{path} must be a non-negative int")
    return value


def _validation_metrics(
    value: object,
    *,
    path: str,
    allow_empty: bool,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    normalized: dict[str, float] = {}
    for key, raw_metric in value.items():
        validate_training_monitor_key(key, path=f"{path} key")
        if not cast(str, key).startswith("valid/metrics/"):
            raise ValueError(
                f"{path} keys must start with 'valid/metrics/'"
            )
        if isinstance(raw_metric, bool) or not isinstance(
            raw_metric,
            (int, float),
        ):
            raise TypeError(f"{path}[{key!r}] must be numeric")
        metric = float(raw_metric)
        if not math.isfinite(metric):
            raise ValueError(f"{path}[{key!r}] must be finite")
        if key in normalized:
            raise ValueError(f"{path} contains duplicate key {key!r}")
        normalized[cast(str, key)] = metric
    if not normalized and not allow_empty:
        raise ValueError(f"{path} must not be empty")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class EpochValidationCadence:
    """Deterministic epoch cadence for one validation evaluation profile."""

    first_epoch: int
    every_n_epochs: int
    include_final: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "first_epoch",
            _positive_int(
                cast(object, self.first_epoch),
                path="epoch validation cadence.first_epoch",
            ),
        )
        object.__setattr__(
            self,
            "every_n_epochs",
            _positive_int(
                cast(object, self.every_n_epochs),
                path="epoch validation cadence.every_n_epochs",
            ),
        )
        include_final = cast(object, self.include_final)
        if not isinstance(include_final, bool):
            raise TypeError(
                "epoch validation cadence.include_final must be a bool"
            )

    def is_due(self, epoch: int, *, final_epoch: int | None = None) -> bool:
        """Return whether this cadence requires evaluation at ``epoch``."""

        epoch_value = _positive_int(
            cast(object, epoch),
            path="epoch validation epoch",
        )
        if final_epoch is not None:
            final_value = _positive_int(
                cast(object, final_epoch),
                path="epoch validation final_epoch",
            )
            if epoch_value > final_value:
                raise ValueError(
                    "epoch validation epoch must not exceed final_epoch"
                )
            if self.include_final and epoch_value == final_value:
                return True
        return self.is_interval_due(epoch_value)

    def is_interval_due(self, epoch: int) -> bool:
        """Return whether ``epoch`` belongs to the absolute interval cadence."""

        epoch_value = _positive_int(
            cast(object, epoch),
            path="epoch validation epoch",
        )
        return (
            epoch_value >= self.first_epoch
            and (epoch_value - self.first_epoch) % self.every_n_epochs == 0
        )

    def interval_count_through(self, completed_epoch: int) -> int:
        """Return the number of interval observations due through one epoch."""

        completed = _non_negative_int(
            cast(object, completed_epoch),
            path="completed_epoch",
        )
        if completed < self.first_epoch:
            return 0
        return (completed - self.first_epoch) // self.every_n_epochs + 1

    def latest_scheduled_epoch(self, completed_epoch: int) -> int | None:
        """Return the latest interval-scheduled epoch already completed."""

        completed = _non_negative_int(
            cast(object, completed_epoch),
            path="completed_epoch",
        )
        if completed < self.first_epoch:
            return None
        intervals = (completed - self.first_epoch) // self.every_n_epochs
        return self.first_epoch + intervals * self.every_n_epochs

    @classmethod
    def from_mapping(cls, value: object) -> EpochValidationCadence:
        """Parse one strict persisted cadence mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("epoch_validation.identity.cadence must be a mapping")
        required = {"first_epoch", "every_n_epochs", "include_final"}
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required, key=str)
            raise ValueError(
                "epoch_validation.identity.cadence has invalid fields: "
                f"missing={missing or '<none>'}, "
                f"unknown={unknown or '<none>'}"
            )
        return cls(
            first_epoch=value["first_epoch"],
            every_n_epochs=value["every_n_epochs"],
            include_final=value["include_final"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this cadence for checkpoint metadata."""

        return {
            "first_epoch": self.first_epoch,
            "every_n_epochs": self.every_n_epochs,
            "include_final": self.include_final,
        }


@dataclass(frozen=True, slots=True)
class EpochValidationIdentity:
    """Stable evaluation profile, metric surface, and cadence identity."""

    profile_digest: str
    metric_keys: tuple[str, ...]
    cadence: EpochValidationCadence

    def __post_init__(self) -> None:
        digest = cast(object, self.profile_digest)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(
                "epoch validation profile_digest must be a lowercase SHA-256 "
                "digest"
            )
        keys_value = cast(object, self.metric_keys)
        if not isinstance(keys_value, tuple) or not keys_value:
            raise TypeError(
                "epoch validation metric_keys must be a non-empty tuple"
            )
        normalized_keys: list[str] = []
        seen: set[str] = set()
        for index, key in enumerate(keys_value):
            validate_training_monitor_key(
                key,
                path=f"epoch validation metric_keys[{index}]",
            )
            if not cast(str, key).startswith("valid/metrics/"):
                raise ValueError(
                    "epoch validation metric_keys must start with "
                    "'valid/metrics/'"
                )
            if key in seen:
                raise ValueError(
                    f"epoch validation metric_keys contains duplicate {key!r}"
                )
            seen.add(cast(str, key))
            normalized_keys.append(cast(str, key))
        cadence_value = cast(object, self.cadence)
        if not isinstance(cadence_value, EpochValidationCadence):
            raise TypeError(
                "epoch validation cadence must be EpochValidationCadence"
            )
        object.__setattr__(self, "metric_keys", tuple(sorted(normalized_keys)))

    @classmethod
    def from_mapping(cls, value: object) -> EpochValidationIdentity:
        """Parse one strict persisted evaluator identity."""

        if not isinstance(value, Mapping):
            raise TypeError("epoch_validation.identity must be a mapping")
        required = {"profile_digest", "metric_keys", "cadence"}
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required, key=str)
            raise ValueError(
                "epoch_validation.identity has invalid fields: "
                f"missing={missing or '<none>'}, "
                f"unknown={unknown or '<none>'}"
            )
        metric_keys = value["metric_keys"]
        if not isinstance(metric_keys, list):
            raise TypeError("epoch_validation.identity.metric_keys must be a list")
        return cls(
            profile_digest=value["profile_digest"],
            metric_keys=tuple(metric_keys),
            cadence=EpochValidationCadence.from_mapping(value["cadence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this identity for checkpoint metadata."""

        return {
            "profile_digest": self.profile_digest,
            "metric_keys": list(self.metric_keys),
            "cadence": self.cadence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EpochValidationResult:
    """Validated metric observations from one due epoch evaluation."""

    epoch: int
    global_step: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epoch",
            _positive_int(
                cast(object, self.epoch),
                path="epoch validation result.epoch",
            ),
        )
        object.__setattr__(
            self,
            "global_step",
            _non_negative_int(
                cast(object, self.global_step),
                path="epoch validation result.global_step",
            ),
        )
        object.__setattr__(
            self,
            "metrics",
            _validation_metrics(
                cast(object, self.metrics),
                path="epoch validation result.metrics",
                allow_empty=False,
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> EpochValidationResult:
        """Parse one strict persisted epoch validation result."""

        if not isinstance(value, Mapping):
            raise TypeError("epoch validation result must be a mapping")
        required = {"epoch", "global_step", "metrics"}
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required, key=str)
            raise ValueError(
                "epoch validation result has invalid fields: "
                f"missing={missing or '<none>'}, "
                f"unknown={unknown or '<none>'}"
            )
        return cls(
            epoch=value["epoch"],
            global_step=value["global_step"],
            metrics=value["metrics"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this completed result for checkpoint metadata."""

        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class EpochValidationState:
    """Strict-resume state for a configured epoch validation evaluator."""

    identity: EpochValidationIdentity
    results: tuple[EpochValidationResult, ...] = ()

    def __post_init__(self) -> None:
        identity = cast(object, self.identity)
        if not isinstance(identity, EpochValidationIdentity):
            raise TypeError(
                "epoch validation state.identity must be "
                "EpochValidationIdentity"
            )
        raw_results = cast(object, self.results)
        if not isinstance(raw_results, tuple):
            raise TypeError("epoch validation state.results must be a tuple")
        results: list[EpochValidationResult] = []
        previous_epoch = 0
        previous_global_step = 0
        interval_epochs: list[int] = []
        cadence = self.identity.cadence
        for index, raw_result in enumerate(raw_results):
            if not isinstance(raw_result, EpochValidationResult):
                raise TypeError(
                    "epoch validation state.results"
                    f"[{index}] must be EpochValidationResult"
                )
            result = raw_result
            if result.epoch <= previous_epoch:
                raise ValueError(
                    "epoch validation state result epochs must be strictly "
                    "increasing"
                )
            if results and result.global_step < previous_global_step:
                raise ValueError(
                    "epoch validation state result global_step values must be "
                    "non-decreasing"
                )
            if set(result.metrics) != set(self.identity.metric_keys):
                raise ValueError(
                    "epoch validation state result metrics must exactly match "
                    "the declared metric keys"
                )
            if cadence.is_interval_due(result.epoch):
                interval_epochs.append(result.epoch)
            elif not cadence.include_final:
                raise ValueError(
                    "epoch validation state contains an off-cadence result "
                    "while include_final=false"
                )
            results.append(result)
            previous_epoch = result.epoch
            previous_global_step = result.global_step
        if results:
            expected_interval_count = cadence.interval_count_through(
                results[-1].epoch
            )
            expected_interval_epochs = [
                cadence.first_epoch + index * cadence.every_n_epochs
                for index in range(expected_interval_count)
            ]
            if interval_epochs != expected_interval_epochs:
                raise ValueError(
                    "epoch validation state is missing or reorders an interval "
                    "result"
                )
        object.__setattr__(self, "results", tuple(results))

    @property
    def last_result(self) -> EpochValidationResult | None:
        """Return the last completed result in the persisted history."""

        return self.results[-1] if self.results else None

    @property
    def last_evaluated_epoch(self) -> int | None:
        """Return the epoch of the last completed result, if any."""

        result = self.last_result
        return None if result is None else result.epoch

    @property
    def last_metrics(self) -> Mapping[str, float]:
        """Return the metric surface of the last completed result."""

        result = self.last_result
        return MappingProxyType({}) if result is None else result.metrics

    @property
    def off_cadence_final_epochs(self) -> tuple[int, ...]:
        """Return recorded final observations outside the interval cadence."""

        cadence = self.identity.cadence
        return tuple(
            result.epoch
            for result in self.results
            if not cadence.is_interval_due(result.epoch)
        )

    def with_result(
        self,
        result: EpochValidationResult,
        *,
        final_epoch: int | None = None,
    ) -> EpochValidationState:
        """Return state advanced by one evaluator result."""

        if set(result.metrics) != set(self.identity.metric_keys):
            raise ValueError(
                "epoch validation result metrics must exactly match the "
                "declared metric keys"
            )
        previous = self.last_result
        if previous is not None and result.epoch <= previous.epoch:
            raise ValueError(
                "epoch validation result epoch must follow the previous "
                "evaluated epoch"
            )
        if previous is not None and result.global_step < previous.global_step:
            raise ValueError(
                "epoch validation result global_step must not precede the "
                "previous result"
            )
        cadence = self.identity.cadence
        if not cadence.is_interval_due(result.epoch) and (
            not cadence.include_final
            or final_epoch is None
            or final_epoch != result.epoch
        ):
            raise ValueError(
                "off-cadence epoch validation result must be the declared "
                "final epoch"
            )
        return EpochValidationState(
            identity=self.identity,
            results=(*self.results, result),
        )

    def observation_count_through(self, completed_epoch: int) -> int:
        """Return interval plus recorded off-cadence observations through epoch."""

        completed = _non_negative_int(
            cast(object, completed_epoch),
            path="completed_epoch",
        )
        return sum(result.epoch <= completed for result in self.results)

    def is_observation_epoch(self, epoch: int) -> bool:
        """Return whether one epoch is in the persisted observation history."""

        epoch_value = _positive_int(
            cast(object, epoch),
            path="epoch validation observation epoch",
        )
        return any(result.epoch == epoch_value for result in self.results)

    def observations_after(self, epoch: int, *, through: int) -> int:
        """Return the persisted observation count after ``epoch`` through a bound."""

        epoch_value = _positive_int(
            cast(object, epoch),
            path="epoch validation observation epoch",
        )
        completed = _non_negative_int(
            cast(object, through),
            path="completed_epoch",
        )
        if epoch_value > completed:
            raise ValueError("observation epoch must not exceed completed_epoch")
        return self.observation_count_through(
            completed
        ) - self.observation_count_through(epoch_value)

    def latest_observation_through(self, completed_epoch: int) -> int | None:
        """Return the latest required interval or recorded final observation."""

        completed = _non_negative_int(
            cast(object, completed_epoch),
            path="completed_epoch",
        )
        latest_interval = self.identity.cadence.latest_scheduled_epoch(completed)
        latest_final = next(
            (
                epoch
                for epoch in reversed(self.off_cadence_final_epochs)
                if epoch <= completed
            ),
            None,
        )
        values = tuple(
            value
            for value in (latest_interval, latest_final)
            if value is not None
        )
        return max(values) if values else None

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> EpochValidationState:
        """Parse one strict persisted evaluator state."""

        if not isinstance(value, Mapping):
            raise TypeError("training_loop.epoch_validation must be a mapping")
        required = {"schema_version", "identity", "results"}
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required, key=str)
            raise ValueError(
                "training_loop.epoch_validation has invalid fields: "
                f"missing={missing or '<none>'}, "
                f"unknown={unknown or '<none>'}"
            )
        version = value["schema_version"]
        if type(version) is not int:
            raise TypeError(
                "training_loop.epoch_validation.schema_version must be an "
                "exact int"
            )
        if version != _EPOCH_VALIDATION_STATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported training_loop.epoch_validation schema_version "
                f"{version!r}; expected "
                f"{_EPOCH_VALIDATION_STATE_SCHEMA_VERSION}"
            )
        identity = EpochValidationIdentity.from_mapping(value["identity"])
        raw_results = value["results"]
        if not isinstance(raw_results, list):
            raise TypeError(
                "training_loop.epoch_validation.results must be a list"
            )
        return cls(
            identity=identity,
            results=tuple(
                EpochValidationResult.from_mapping(result)
                for result in raw_results
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this state for checkpoint metadata."""

        return {
            "schema_version": _EPOCH_VALIDATION_STATE_SCHEMA_VERSION,
            "identity": self.identity.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


class EpochValidationEvaluator(ABC):
    """Evaluate a live training snapshot under one immutable profile."""

    @property
    @abstractmethod
    def identity(self) -> EpochValidationIdentity:
        """Return the immutable profile, metric, and cadence identity."""

    @abstractmethod
    def evaluate(
        self,
        *,
        epoch: int,
        global_step: int,
    ) -> EpochValidationResult:
        """Evaluate one due live snapshot and return canonical metrics."""


__all__ = [
    "EpochValidationCadence",
    "EpochValidationEvaluator",
    "EpochValidationIdentity",
    "EpochValidationResult",
    "EpochValidationState",
]
