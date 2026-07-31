"""Training-phase composition for task-neutral metric engines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import torch

from stochaflow.metrics import (
    MetricConfig,
    MetricEngine,
    MetricSpec,
    MetricUpdate,
    validate_metric_configs,
)
from stochaflow.training.strategy import (
    MetricChannelProvider,
    TrainingStrategy,
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
        configs: list[MetricConfig],
        strategy: TrainingStrategy,
        *,
        device: torch.device | str,
    ) -> None:
        declarations = validate_metric_configs(configs)
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
        for phase in _PHASE_PREFIXES:
            specs = tuple(
                MetricSpec(
                    id=declaration.id,
                    name=declaration.name,
                    channel=declaration.channel,
                    params=dict(declaration.params),
                )
                for declaration in declarations
                if phase in declaration.phases
            )
            if specs:
                engines[phase] = MetricEngine(specs).to(device)
        self._engines = engines

    @property
    def configured(self) -> bool:
        """Return whether at least one phase has configured metrics."""

        return bool(self._engines)

    def has_phase(self, phase: TrainingMetricPhase) -> bool:
        """Return whether the phase owns a metric engine."""

        return phase in self._engines

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


__all__ = ["TrainingMetricPhase", "TrainingMetricRuntime"]
