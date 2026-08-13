"""Narrow control-plane contracts for fixed-topology distributed training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DistributedTopology:
    """One fixed, single-node rank layout supplied by ``torchrun``."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    def __post_init__(self) -> None:
        for name in ("rank", "local_rank", "world_size", "local_world_size"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"distributed topology {name} must be an integer")
        if self.world_size <= 0:
            raise ValueError("distributed topology world_size must be positive")
        if self.local_world_size <= 0:
            raise ValueError(
                "distributed topology local_world_size must be positive"
            )
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                "distributed topology rank must be in [0, world_size)"
            )
        if not 0 <= self.local_rank < self.local_world_size:
            raise ValueError(
                "distributed topology local_rank must be in "
                "[0, local_world_size)"
            )
        if self.world_size != self.local_world_size:
            raise ValueError(
                "fixed single-node distributed training requires world_size "
                "to equal local_world_size"
            )
        if self.rank != self.local_rank:
            raise ValueError(
                "fixed single-node distributed training requires rank to "
                "equal local_rank"
            )

    @property
    def is_primary(self) -> bool:
        """Return whether this process owns rank-zero control-plane work."""

        return self.rank == 0


class DistributedCollectives(Protocol):
    """Small control-plane collective surface consumed by ``DDPTrainer``."""

    def broadcast_from_primary(self, value: object) -> object:
        """Broadcast one small control value from rank zero."""

        ...

    def gather_to_primary(self, value: object) -> tuple[object, ...] | None:
        """Gather one small value per rank, returning it only on rank zero."""

        ...

    def all_true(self, value: bool) -> bool:
        """Return true only when every rank supplied true."""

        ...

    def all_equal(self, value: object) -> bool:
        """Compare deterministic, data-only control values across all ranks."""

        ...

    def sum_int(self, value: int) -> int:
        """Return the exact integer sum across ranks."""

        ...

    def sum_float(self, value: float) -> float:
        """Return the floating-point sum across ranks."""

        ...

    def min_int(self, value: int) -> int:
        """Return the smallest exact integer across ranks."""

        ...

    def max_int(self, value: int) -> int:
        """Return the largest exact integer across ranks."""

        ...


__all__ = ["DistributedCollectives", "DistributedTopology"]
