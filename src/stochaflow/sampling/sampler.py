"""Unified sampler, observer, and trajectory contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from stochaflow.sampling.dynamics import GenerativeDynamics
from stochaflow.utils.registry import REGISTRIES


@dataclass(frozen=True, slots=True)
class SamplerResult:
    """Result of one complete solver execution."""

    final_state: Any
    num_steps: int
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        num_steps = cast(object, self.num_steps)
        diagnostics = cast(object, self.diagnostics)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TypeError("SamplerResult.num_steps must be an integer")
        if num_steps < 0:
            raise ValueError("SamplerResult.num_steps must be non-negative")
        if not isinstance(diagnostics, Mapping):
            raise TypeError("SamplerResult.diagnostics must be a mapping")


@dataclass(frozen=True, slots=True)
class SamplingObservation:
    """One outwardly visible state in a sampler lifecycle."""

    step_index: int
    coordinate: int | float
    state: Any
    is_final: bool
    diagnostics: Mapping[str, Any]


class SamplingObserver(Protocol):
    """Consume accepted, outwardly visible solver states."""

    def observe(self, observation: SamplingObservation) -> None:
        """Record one observation."""


class TrajectoryObserver:
    """Retain the initial, periodic, and final accepted solver states."""

    def __init__(
        self,
        every_steps: int = 1,
        *,
        copy_state: Callable[[Any], Any] | None = None,
    ) -> None:
        raw_every_steps = cast(object, every_steps)
        raw_copy_state = cast(object, copy_state)
        if isinstance(raw_every_steps, bool) or not isinstance(raw_every_steps, int):
            raise TypeError("trajectory every_steps must be an integer")
        if raw_every_steps <= 0:
            raise ValueError("trajectory every_steps must be positive")
        if raw_copy_state is not None and not callable(raw_copy_state):
            raise TypeError("trajectory copy_state must be callable")
        self.every_steps = every_steps
        self._copy_state: Callable[[Any], Any] = copy_state or _copy_observed_state
        self._observations: list[SamplingObservation] = []

    @property
    def observations(self) -> tuple[SamplingObservation, ...]:
        """Return retained observations in delivery order."""

        return tuple(self._observations)

    def observe(self, observation: SamplingObservation) -> None:
        """Retain an observation according to the configured interval."""

        keep = (
            observation.step_index == 0
            or observation.is_final
            or observation.step_index % self.every_steps == 0
        )
        if not keep:
            return
        retained = SamplingObservation(
            observation.step_index,
            observation.coordinate,
            self._copy_state(observation.state),
            observation.is_final,
            dict(observation.diagnostics),
        )
        self._observations.append(retained)


class Sampler(ABC):
    """Execute a complete solver over already assembled dynamics."""

    @abstractmethod
    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        """Run the solver from ``initial_state`` to its generation endpoint."""


REGISTRIES.samplers.require_base(Sampler)


def _copy_observed_state(state: Any) -> Any:
    """Copy an observed state so later solver mutations cannot change history."""

    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        copied = state.copy()
        for key, value in state.items():
            copied[key] = _copy_observed_state(value)
        return copied
    if isinstance(state, list):
        return [_copy_observed_state(value) for value in state]
    if isinstance(state, tuple):
        values = tuple(_copy_observed_state(value) for value in state)
        if hasattr(state, "_fields"):
            return type(state)(*values)
        return values
    return deepcopy(state)


__all__ = [
    "Sampler",
    "SamplerResult",
    "SamplingObservation",
    "SamplingObserver",
    "TrajectoryObserver",
]
