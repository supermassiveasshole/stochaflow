"""Training-computation contracts independent from loop orchestration."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, cast, runtime_checkable

import torch

from stochaflow.metrics.contracts import MetricUpdate

type Batch = Any
type ScalarMetric = float | int | torch.Tensor


@runtime_checkable
class DeviceTransferableBatch(Protocol):
    """Optional device-transfer capability for a domain-specific batch."""

    def to_device(self, device: torch.device) -> Self:
        """Return this batch with its tensor state on ``device``."""

        ...


@runtime_checkable
class ReferenceImageBatchSemantics(Protocol):
    """Optional strategy capability for extracting reference metric images."""

    def extract_reference_images(self, batch: Batch) -> torch.Tensor:
        """Return the clean image tensor represented by a validation batch."""

        ...


@runtime_checkable
class MetricChannelProvider(Protocol):
    """Optional strategy capability declaring emitted metric update channels."""

    @property
    def metric_channels(self) -> frozenset[str]:
        """Return the metric update channels emitted by strategy steps."""

        ...


@dataclass(frozen=True, slots=True)
class TrainStepOutput:
    """One strategy computation consumed by the automatic training loop.

    ``loss`` is the scalar optimization loss for one logical micro-batch.
    Automatic accumulation gives each micro-batch scalar equal weight inside
    an optimizer window. Strategies that require ordinary effective-batch
    semantics should therefore return a mean over logical samples.

    ``metric_updates`` contains task-owned channel payloads for configured
    metrics. ``diagnostic_observation`` is an opaque task-owned value forwarded
    unchanged to training Diagnostics after a successful optimizer step. Core
    never inspects its type or fields. ``loss_aggregation_weight`` affects only
    epoch reporting; it never scales the loss used for backward propagation.
    """

    loss: torch.Tensor
    metrics: Mapping[str, ScalarMetric] = field(default_factory=dict)
    diagnostic_observation: object | None = None
    metric_updates: Mapping[str, MetricUpdate] = field(default_factory=dict)
    loss_aggregation_weight: float | int | torch.Tensor = 1.0


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
            if metric.dtype == torch.bool or torch.is_complex(metric):
                raise TypeError(
                    f"TrainStepOutput metric '{name}' must be real numeric"
                )
        elif not isinstance(metric, (int, float)):
            raise TypeError(f"TrainStepOutput metric '{name}' must be numeric")
    metric_updates = cast(object, value.metric_updates)
    if not isinstance(metric_updates, Mapping):
        raise TypeError("TrainStepOutput.metric_updates must be a mapping")
    for declared_channel, declared_update in metric_updates.items():
        channel = cast(object, declared_channel)
        if (
            not isinstance(channel, str)
            or not channel
            or channel != channel.strip()
        ):
            raise ValueError(
                "TrainStepOutput metric update channels must be non-empty "
                "strings without surrounding whitespace"
            )
        update = cast(object, declared_update)
        if not isinstance(update, MetricUpdate):
            raise TypeError(
                f"TrainStepOutput metric update '{channel}' must be MetricUpdate"
            )
        update_args = cast(object, update.args)
        if not isinstance(update_args, tuple):
            raise TypeError(
                f"TrainStepOutput metric update '{channel}' args must be a tuple"
            )
        update_kwargs = cast(object, update.kwargs)
        if not isinstance(update_kwargs, Mapping):
            raise TypeError(
                f"TrainStepOutput metric update '{channel}' kwargs must be a mapping"
            )
        if any(not isinstance(key, str) or not key for key in update_kwargs):
            raise ValueError(
                f"TrainStepOutput metric update '{channel}' keyword names "
                "must be non-empty strings"
            )
    loss_aggregation_weight_to_float(value.loss_aggregation_weight)
    return value


def loss_aggregation_weight_to_float(value: object) -> float:
    """Normalize one finite non-negative detached epoch aggregation weight."""

    if isinstance(value, bool):
        raise TypeError(
            "TrainStepOutput.loss_aggregation_weight must be a real numeric scalar"
        )
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(
                "TrainStepOutput.loss_aggregation_weight must be a scalar Tensor"
            )
        if value.dtype == torch.bool or torch.is_complex(value):
            raise TypeError(
                "TrainStepOutput.loss_aggregation_weight must be real numeric"
            )
        weight = float(value.detach().item())
    elif isinstance(value, (int, float)):
        weight = float(value)
    else:
        raise TypeError(
            "TrainStepOutput.loss_aggregation_weight must be a real numeric scalar"
        )
    if not math.isfinite(weight):
        raise ValueError(
            "TrainStepOutput.loss_aggregation_weight must be finite"
        )
    if weight < 0.0:
        raise ValueError(
            "TrainStepOutput.loss_aggregation_weight must be non-negative"
        )
    return weight


__all__ = [
    "Batch",
    "DeviceTransferableBatch",
    "MetricChannelProvider",
    "MetricUpdate",
    "ReferenceImageBatchSemantics",
    "ScalarMetric",
    "TrainStepOutput",
    "TrainingStrategy",
    "loss_aggregation_weight_to_float",
    "validate_train_step_output",
]
