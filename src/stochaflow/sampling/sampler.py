"""Shared result contracts for sampler-specific debug trajectories."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TrajectoryFrame:
    """One ordered reverse-process snapshot at a mathematical state time."""

    state_time: int
    samples: torch.Tensor


@dataclass(frozen=True, slots=True)
class SamplingTrace:
    """Final samples plus ordered sampler-specific debug snapshots."""

    samples: torch.Tensor
    frames: list[TrajectoryFrame]


__all__ = ["SamplingTrace", "TrajectoryFrame"]
