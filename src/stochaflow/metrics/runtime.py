"""Single-scope state lifecycle for task-neutral epoch metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import cast

import torch
from torchmetrics import Metric

from stochaflow.metrics.config import (
    METRIC_TAG_SEGMENT_PATTERN,
    MetricSpec,
    validate_metric_spec,
)
from stochaflow.metrics.contracts import (
    MetricUpdate,
    PreparedMetricUpdates,
    prepare_metric_updates,
)
from stochaflow.metrics.factory import build_metric
from stochaflow.utils.registry import REGISTRIES, Registry


class MetricRuntimeError(ValueError):
    """Raised when metric channels or computed results violate their contract."""


def _scalar_metric_value(value: object, *, path: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise MetricRuntimeError(
                f"{path} must be a scalar tensor, got shape {tuple(value.shape)}"
            )
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{path} must be a numeric scalar, got {type(value).__name__}"
        )
    return float(value)


def _normalize_metric_result(
    metric_id: str,
    result: object,
) -> dict[str, float]:
    if not isinstance(result, Mapping):
        return {
            metric_id: _scalar_metric_value(
                result,
                path=f"metric {metric_id!r} result",
            )
        }
    if not result:
        raise MetricRuntimeError(
            f"metric {metric_id!r} returned an empty mapping"
        )
    normalized: dict[str, float] = {}
    for subkey, value in result.items():
        if (
            not isinstance(subkey, str)
            or METRIC_TAG_SEGMENT_PATTERN.fullmatch(subkey) is None
        ):
            raise MetricRuntimeError(
                f"metric {metric_id!r} result key must match "
                f"{METRIC_TAG_SEGMENT_PATTERN.pattern!r}, got {subkey!r}"
            )
        key = f"{metric_id}/{subkey}"
        normalized[key] = _scalar_metric_value(
            value,
            path=f"metric {metric_id!r} result[{subkey!r}]",
        )
    return normalized


class MetricEngine:
    """Own metric instances and update bindings for one isolated scope."""

    def __init__(
        self,
        specs: Sequence[MetricSpec],
        *,
        registry: Registry[type[Metric]] = REGISTRIES.metrics,
    ) -> None:
        specs_value = cast(object, specs)
        if isinstance(specs_value, (str, bytes)) or not isinstance(
            specs_value,
            Sequence,
        ):
            raise TypeError("metric engine specs must be a sequence")
        if not isinstance(cast(object, registry), Registry):
            raise TypeError("metric engine registry must be a Registry")
        metrics: dict[str, Metric] = {}
        channels: dict[str, list[str]] = {}
        for index, spec in enumerate(specs):
            validate_metric_spec(spec, path=f"metric engine specs[{index}]")
            if spec.id in metrics:
                raise ValueError(
                    f"metric engine specs contain duplicate id {spec.id!r}"
                )
            metrics[spec.id] = build_metric(spec, registry=registry)
            channels.setdefault(spec.channel, []).append(spec.id)
        self._metrics = metrics
        self._channels = {
            channel: tuple(metric_ids)
            for channel, metric_ids in channels.items()
        }
        self._successful_updates = 0

    @property
    def required_channels(self) -> frozenset[str]:
        """Return channels that every update call must provide."""

        return frozenset(self._channels)

    def reset(self) -> None:
        """Clear every metric state in this scope."""

        try:
            with torch.no_grad():
                for metric in self._metrics.values():
                    metric.reset()
        finally:
            self._successful_updates = 0

    def update(self, updates: Mapping[str, MetricUpdate]) -> None:
        """Prepare and immediately commit one channel payload mapping."""

        self._update_prepared(prepare_metric_updates(updates))

    def _update_prepared(self, updates: PreparedMetricUpdates) -> None:
        """Commit payloads already detached for deferred optimizer semantics."""

        updates_value = cast(object, updates)
        if not isinstance(updates_value, PreparedMetricUpdates):
            raise TypeError("prepared metric updates must be PreparedMetricUpdates")
        values = updates_value.values
        missing = sorted(self.required_channels - set(values))
        if missing:
            raise MetricRuntimeError(
                "metric updates are missing bound channel(s): "
                + ", ".join(missing)
            )
        with torch.no_grad():
            for channel, metric_ids in self._channels.items():
                update = values[channel]
                for metric_id in metric_ids:
                    try:
                        self._metrics[metric_id].update(
                            *update.args,
                            **update.kwargs,
                        )
                    except Exception as error:
                        self.reset()
                        raise MetricRuntimeError(
                            f"metric {metric_id!r} update failed for "
                            f"channel {channel!r}"
                        ) from error
        self._successful_updates += 1

    def compute(self, *, reset: bool = False) -> dict[str, float]:
        """Compute normalized scalar results, optionally resetting in ``finally``."""

        if not isinstance(cast(object, reset), bool):
            raise TypeError("metric engine compute reset must be a bool")
        try:
            if self._metrics and self._successful_updates == 0:
                metric_ids = ", ".join(self._metrics)
                channels = ", ".join(sorted(self.required_channels))
                raise MetricRuntimeError(
                    "metric engine cannot compute before a successful update: "
                    f"metrics={metric_ids}; channels={channels}"
                )
            normalized: dict[str, float] = {}
            with torch.no_grad():
                for metric_id, metric in self._metrics.items():
                    current = _normalize_metric_result(
                        metric_id,
                        metric.compute(),
                    )
                    collisions = sorted(set(normalized).intersection(current))
                    if collisions:
                        raise MetricRuntimeError(
                            "metric result key collision(s): "
                            + ", ".join(collisions)
                        )
                    normalized.update(current)
            return normalized
        finally:
            if reset:
                self.reset()

    def to(self, device: torch.device | str) -> MetricEngine:
        """Move every metric state to a device and return this engine."""

        for metric in self._metrics.values():
            metric.to(device)
        return self


def _commit_prepared_metric_updates(
    engine: MetricEngine,
    updates: PreparedMetricUpdates,
) -> None:
    """Commit prepared payloads through the internal trusted runtime path."""

    engine._update_prepared(updates)


__all__ = ["MetricEngine", "MetricRuntimeError"]
