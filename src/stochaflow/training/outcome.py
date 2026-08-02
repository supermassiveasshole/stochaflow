"""Immutable structured outcome for one completed training run."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from stochaflow.utils.config import validate_training_monitor_key

CheckpointSelectionKind = Literal["best", "final"]


def _freeze_metrics(
    value: Mapping[str, float],
    *,
    path: str,
    require_non_empty: bool,
) -> Mapping[str, float]:
    value_object = cast(object, value)
    if not isinstance(value_object, Mapping):
        raise TypeError(f"{path} must be a mapping")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in cast(
        Mapping[object, object], value_object
    ).items():
        if not isinstance(raw_name, str) or not raw_name:
            raise TypeError(f"{path} keys must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{path}[{raw_name!r}] must be numeric")
        metric = float(raw_value)
        if not math.isfinite(metric):
            raise ValueError(f"{path}[{raw_name!r}] must be finite")
        normalized[raw_name] = metric
    if require_non_empty and not normalized:
        raise ValueError(f"{path} must not be empty")
    return cast(Mapping[str, float], MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class TrainingRunOutcome:
    """Portable, immutable result of one successfully completed training run."""

    output_dir: Path
    final_epoch: int
    final_metrics: Mapping[str, float]
    latest_checkpoint: Path | None
    best_epoch: int | None
    best_metric_name: str | None
    best_metric_value: float | None
    best_checkpoint: Path | None
    selected_checkpoint: Path | None
    selected_checkpoint_kind: CheckpointSelectionKind | None
    stopped_early: bool
    phase_test_metrics: Mapping[str, float]
    manifest_path: Path
    metrics_path: Path | None
    log_path: Path | None

    def __post_init__(self) -> None:
        for name in ("output_dir", "manifest_path"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        for name in (
            "latest_checkpoint",
            "best_checkpoint",
            "selected_checkpoint",
            "metrics_path",
            "log_path",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path or null")
        final_epoch = cast(object, self.final_epoch)
        if type(final_epoch) is not int or final_epoch <= 0:
            raise ValueError("final_epoch must be a positive integer")
        object.__setattr__(
            self,
            "final_metrics",
            _freeze_metrics(
                self.final_metrics,
                path="final_metrics",
                require_non_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "phase_test_metrics",
            _freeze_metrics(
                self.phase_test_metrics,
                path="phase_test_metrics",
                require_non_empty=False,
            ),
        )
        best_identity = (
            self.best_epoch,
            self.best_metric_name,
            self.best_metric_value,
        )
        if any(value is None for value in best_identity) != all(
            value is None for value in best_identity
        ):
            raise ValueError(
                "best epoch, metric name, and metric value must all be set "
                "or all be null"
            )
        if self.best_checkpoint is not None and self.best_epoch is None:
            raise ValueError(
                "best_checkpoint requires recorded best metric identity"
            )
        best_epoch = cast(object, self.best_epoch)
        if best_epoch is not None and (
            type(best_epoch) is not int or best_epoch <= 0
        ):
            raise ValueError("best_epoch must be a positive integer or null")
        best_metric_name = cast(object, self.best_metric_name)
        if best_metric_name is not None and (
            not isinstance(best_metric_name, str) or not best_metric_name
        ):
            raise ValueError("best_metric_name must be non-empty or null")
        if best_metric_name is not None:
            validate_training_monitor_key(
                best_metric_name,
                path="best_metric_name",
            )
        best_metric_value = cast(object, self.best_metric_value)
        if best_metric_value is not None:
            if isinstance(best_metric_value, bool) or not isinstance(
                best_metric_value, (int, float)
            ):
                raise TypeError("best_metric_value must be numeric or null")
            if not math.isfinite(float(best_metric_value)):
                raise ValueError("best_metric_value must be finite or null")
        selection_kind = cast(object, self.selected_checkpoint_kind)
        if selection_kind not in {None, "best", "final"}:
            raise ValueError(
                "selected_checkpoint_kind must be best, final, or null"
            )
        if (self.selected_checkpoint is None) != (
            self.selected_checkpoint_kind is None
        ):
            raise ValueError(
                "selected_checkpoint and selected_checkpoint_kind must both "
                "be set or both be null"
            )
        if self.selected_checkpoint_kind == "best" and (
            self.best_checkpoint != self.selected_checkpoint
        ):
            raise ValueError(
                "best selection must reference the recorded best checkpoint"
            )
        if self.selected_checkpoint_kind == "final" and (
            self.latest_checkpoint != self.selected_checkpoint
        ):
            raise ValueError(
                "final selection must reference the recorded latest checkpoint"
            )
        if self.best_epoch is not None and self.best_epoch > self.final_epoch:
            raise ValueError("best_epoch must not exceed final_epoch")
        if type(self.stopped_early) is not bool:
            raise TypeError("stopped_early must be a bool")
        if (self.metrics_path is None) != (self.log_path is None):
            raise ValueError(
                "metrics_path and log_path must both be set or both be null"
            )

    def to_manifest(self) -> dict[str, Any]:
        """Serialize the portable outcome for ``run_manifest.yaml``."""

        return {
            "output_dir": str(self.output_dir),
            "final_epoch": self.final_epoch,
            "final_metrics": dict(self.final_metrics),
            "latest_checkpoint": (
                str(self.latest_checkpoint)
                if self.latest_checkpoint is not None
                else None
            ),
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "best_checkpoint": (
                str(self.best_checkpoint)
                if self.best_checkpoint is not None
                else None
            ),
            "selected_checkpoint": (
                str(self.selected_checkpoint)
                if self.selected_checkpoint is not None
                else None
            ),
            "selected_checkpoint_kind": self.selected_checkpoint_kind,
            "stopped_early": self.stopped_early,
            "phase_test_metrics": dict(self.phase_test_metrics),
            "manifest_path": str(self.manifest_path),
            "metrics_path": (
                str(self.metrics_path) if self.metrics_path is not None else None
            ),
            "log_path": str(self.log_path) if self.log_path is not None else None,
        }


__all__ = ["CheckpointSelectionKind", "TrainingRunOutcome"]
