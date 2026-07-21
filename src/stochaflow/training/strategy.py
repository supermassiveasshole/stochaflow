"""Training-computation contracts independent from loop orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

import torch

Batch: TypeAlias = Any
ScalarMetric: TypeAlias = float | int | torch.Tensor


@dataclass(frozen=True, slots=True)
class TrainStepOutput:
    """One strategy computation consumed by the automatic training loop."""

    loss: torch.Tensor
    metrics: Mapping[str, ScalarMetric] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class TrainingStrategy(ABC):
    """Define batch interpretation and loss computation only."""

    @abstractmethod
    def training_step(self, batch: Batch) -> TrainStepOutput:
        """Compute one training loss and optional reporting values."""

    def evaluation_step(self, batch: Batch) -> TrainStepOutput:
        """Compute one evaluation loss using the same semantics by default."""

        return self.training_step(batch)


def validate_train_step_output(value: object) -> TrainStepOutput:
    """Validate an extension-produced step result at the runtime boundary."""

    if not isinstance(value, TrainStepOutput):
        raise TypeError("TrainingStrategy step must return TrainStepOutput")
    loss_value = cast(object, value.loss)
    if not isinstance(loss_value, torch.Tensor):
        raise TypeError("TrainStepOutput.loss must be a Tensor")
    if loss_value.ndim != 0:
        raise ValueError("TrainStepOutput.loss must be a scalar Tensor")
    if not loss_value.is_floating_point():
        raise TypeError("TrainStepOutput.loss must be floating point")
    metrics = cast(object, value.metrics)
    if not isinstance(metrics, Mapping):
        raise TypeError("TrainStepOutput.metrics must be a mapping")
    for declared_name, declared_metric in metrics.items():
        name = cast(object, declared_name)
        if not isinstance(name, str) or not name:
            raise ValueError("TrainStepOutput metric names must be non-empty strings")
        metric = cast(object, declared_metric)
        if isinstance(metric, bool):
            raise TypeError(f"TrainStepOutput metric '{name}' must be numeric")
        if isinstance(metric, torch.Tensor):
            if metric.ndim != 0:
                raise ValueError(
                    f"TrainStepOutput metric '{name}' must be a scalar Tensor"
                )
        elif not isinstance(metric, (int, float)):
            raise TypeError(f"TrainStepOutput metric '{name}' must be numeric")
    diagnostics = cast(object, value.diagnostics)
    if not isinstance(diagnostics, Mapping):
        raise TypeError("TrainStepOutput.diagnostics must be a mapping")
    if any(not isinstance(key, str) or not key for key in diagnostics):
        raise ValueError(
            "TrainStepOutput diagnostic names must be non-empty strings"
        )
    return value


__all__ = [
    "Batch",
    "ScalarMetric",
    "TrainStepOutput",
    "TrainingStrategy",
    "validate_train_step_output",
]
