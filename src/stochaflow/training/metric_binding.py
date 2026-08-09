"""Training-phase composition for task-neutral metric engines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import torch

from stochaflow.metrics import (
    MetricEngine,
    MetricUpdate,
)
from stochaflow.metrics._prepared_updates import commit_prepared_metric_updates
from stochaflow.metrics.contracts import (
    PreparedMetricUpdates,
    prepare_metric_updates,
)
from stochaflow.training.strategy import (
    MetricChannelProvider,
    TrainingStrategy,
)
from stochaflow.utils.config import (
    TrainingMetricConfig,
    validate_training_metric_configs,
)

type TrainingMetricPhase = Literal["train", "validation", "test"]

_PHASE_PREFIXES: dict[TrainingMetricPhase, str] = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


class TrainingMetricRuntime:
    """Own one isolated metric engine per configured training phase."""

    def __init__(
        self,
        configs: list[TrainingMetricConfig],
        strategy: TrainingStrategy,
        *,
        device: torch.device | str,
    ) -> None:
        declarations = validate_training_metric_configs(configs)
        if declarations:
            strategy_value = cast(object, strategy)
            if not isinstance(strategy_value, MetricChannelProvider):
                raise TypeError(
                    "configured metrics require the TrainingStrategy to satisfy "
                    "MetricChannelProvider"
                )
            channels_value = cast(object, strategy_value.metric_channels)
            if not isinstance(channels_value, frozenset) or any(
                not isinstance(channel, str) or not channel
                for channel in channels_value
            ):
                raise TypeError(
                    "MetricChannelProvider.metric_channels must be a frozenset "
                    "of non-empty strings"
                )
            channels = cast(frozenset[str], channels_value)
            missing = sorted(
                {
                    declaration.channel
                    for declaration in declarations
                    if declaration.channel not in channels
                }
            )
            if missing:
                raise ValueError(
                    "configured metric channel(s) are not provided by "
                    f"{type(strategy).__name__}: {', '.join(missing)}"
                )

        engines: dict[TrainingMetricPhase, MetricEngine] = {}
        metric_ids: dict[TrainingMetricPhase, frozenset[str]] = {}
        for phase in _PHASE_PREFIXES:
            specs = tuple(
                declaration.to_metric_spec()
                for declaration in declarations
                if phase in declaration.phases
            )
            if specs:
                engines[phase] = MetricEngine(specs)
                metric_ids[phase] = frozenset(spec.id for spec in specs)
        self._engines = engines
        self._metric_ids = metric_ids
        self.to(device)

    @property
    def configured(self) -> bool:
        """Return whether at least one phase has configured metrics."""

        return bool(self._engines)

    def has_phase(self, phase: TrainingMetricPhase) -> bool:
        """Return whether the phase owns a metric engine."""

        return phase in self._engines

    def has_metric(
        self,
        phase: TrainingMetricPhase,
        metric_id: str,
    ) -> bool:
        """Return whether one metric id is configured for a phase."""

        return metric_id in self._metric_ids.get(phase, frozenset())

    def reset_phase(self, phase: TrainingMetricPhase) -> None:
        """Reset the isolated state for one phase when it is configured."""

        engine = self._engines.get(phase)
        if engine is not None:
            engine.reset()

    def update_phase(
        self,
        phase: TrainingMetricPhase,
        updates: Mapping[str, MetricUpdate],
    ) -> None:
        """Dispatch one strategy update mapping to a configured phase."""

        engine = self._engines.get(phase)
        if engine is not None:
            engine.update(updates)

    def prepare_updates(
        self,
        updates: Mapping[str, MetricUpdate],
    ) -> PreparedMetricUpdates:
        """Detach one training-step payload before its optimizer commit point."""

        return prepare_metric_updates(updates)

    def commit_phase(
        self,
        phase: TrainingMetricPhase,
        updates: PreparedMetricUpdates,
    ) -> None:
        """Commit a prepared payload after its optimizer window succeeds."""

        engine = self._engines.get(phase)
        if engine is not None:
            commit_prepared_metric_updates(engine, updates)

    def compute_phase(
        self,
        phase: TrainingMetricPhase,
        *,
        reset: bool = True,
    ) -> dict[str, float]:
        """Compute canonical phase keys and optionally reset metric state."""

        engine = self._engines.get(phase)
        if engine is None:
            return {}
        prefix = _PHASE_PREFIXES[phase]
        return {
            f"{prefix}/metrics/{name}": value
            for name, value in engine.compute(reset=reset).items()
        }

    def to(
        self,
        device: torch.device | str,
    ) -> TrainingMetricRuntime:
        """Move every configured phase engine to a device and return this runtime."""

        for engine in self._engines.values():
            engine.to(device)
        return self


__all__ = ["TrainingMetricPhase", "TrainingMetricRuntime"]
