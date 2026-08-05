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
        return (
            epoch_value >= self.first_epoch
            and (epoch_value - self.first_epoch) % self.every_n_epochs == 0
        )

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


@dataclass(frozen=True, slots=True)
class EpochValidationState:
    """Strict-resume state for a configured epoch validation evaluator."""

    identity: EpochValidationIdentity
    last_evaluated_epoch: int | None
    last_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        identity = cast(object, self.identity)
        if not isinstance(identity, EpochValidationIdentity):
            raise TypeError(
                "epoch validation state.identity must be "
                "EpochValidationIdentity"
            )
        last_epoch = cast(object, self.last_evaluated_epoch)
        if last_epoch is not None:
            last_epoch = _positive_int(
                last_epoch,
                path="epoch validation state.last_evaluated_epoch",
            )
        metrics = _validation_metrics(
            cast(object, self.last_metrics),
            path="epoch validation state.last_metrics",
            allow_empty=last_epoch is None,
        )
        if last_epoch is None and metrics:
            raise ValueError(
                "epoch validation state without a last epoch requires empty "
                "last_metrics"
            )
        if last_epoch is not None and set(metrics) != set(self.identity.metric_keys):
            raise ValueError(
                "epoch validation state.last_metrics must exactly match the "
                "declared metric keys"
            )
        if (
            last_epoch is not None
            and not self.identity.cadence.include_final
            and not self.identity.cadence.is_due(last_epoch)
        ):
            raise ValueError(
                "epoch validation state.last_evaluated_epoch is outside the "
                "declared cadence"
            )
        object.__setattr__(self, "last_evaluated_epoch", last_epoch)
        object.__setattr__(self, "last_metrics", metrics)

    def with_result(self, result: EpochValidationResult) -> EpochValidationState:
        """Return state advanced by one evaluator result."""

        if set(result.metrics) != set(self.identity.metric_keys):
            raise ValueError(
                "epoch validation result metrics must exactly match the "
                "declared metric keys"
            )
        previous_epoch = self.last_evaluated_epoch
        if previous_epoch is not None and result.epoch <= previous_epoch:
            raise ValueError(
                "epoch validation result epoch must follow the previous "
                "evaluated epoch"
            )
        return EpochValidationState(
            identity=self.identity,
            last_evaluated_epoch=result.epoch,
            last_metrics=result.metrics,
        )

    @classmethod
    def from_mapping(cls, value: object) -> EpochValidationState:
        """Parse one strict persisted evaluator state."""

        if not isinstance(value, Mapping):
            raise TypeError("training_loop.epoch_validation must be a mapping")
        required = {"identity", "last_evaluated_epoch", "last_metrics"}
        if set(value) != required:
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required, key=str)
            raise ValueError(
                "training_loop.epoch_validation has invalid fields: "
                f"missing={missing or '<none>'}, "
                f"unknown={unknown or '<none>'}"
            )
        return cls(
            identity=EpochValidationIdentity.from_mapping(value["identity"]),
            last_evaluated_epoch=value["last_evaluated_epoch"],
            last_metrics=_validation_metrics(
                value["last_metrics"],
                path="training_loop.epoch_validation.last_metrics",
                allow_empty=value["last_evaluated_epoch"] is None,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this state for checkpoint metadata."""

        return {
            "identity": self.identity.to_dict(),
            "last_evaluated_epoch": self.last_evaluated_epoch,
            "last_metrics": dict(self.last_metrics),
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
