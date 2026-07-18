"""Minimal data-builder extension contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sized
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog


@dataclass(frozen=True, slots=True)
class DataBuilderContext:
    """Copied component parameters and deterministic experiment seed."""

    params: dict[str, Any]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.params, dict):
            raise TypeError("data builder params must be a mapping")
        object.__setattr__(self, "params", deepcopy(self.params))


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Ready-to-use iterables for one independent training run."""

    train: Iterable[Any]
    validation: Iterable[Any] | None = None
    test: Iterable[Any] | None = None
    steps_per_epoch: int | None = None

    def __post_init__(self) -> None:
        for role in ("train", "validation", "test"):
            loader = getattr(self, role)
            if loader is not None and not isinstance(loader, Iterable):
                raise TypeError(f"{role} loader must be iterable")
            if loader is not None and isinstance(loader, Iterator):
                raise TypeError(
                    f"{role} loader must be re-iterable, not a one-shot iterator"
                )
        if self.steps_per_epoch is not None:
            if (
                not isinstance(self.steps_per_epoch, int)
                or isinstance(self.steps_per_epoch, bool)
                or self.steps_per_epoch <= 0
            ):
                raise ValueError("steps_per_epoch must be a positive integer")
        elif not isinstance(self.train, Sized):
            raise ValueError(
                "train loader must expose len() or set steps_per_epoch"
            )
        if isinstance(self.train, Sized):
            train_length = len(self.train)
            if train_length <= 0:
                raise ValueError("train loader must yield at least one batch")
            if (
                self.steps_per_epoch is not None
                and self.steps_per_epoch > train_length
            ):
                raise ValueError(
                    "steps_per_epoch must not exceed the sized train loader length"
                )


class DataBuilder(ABC):
    """Extension point that assembles one run's complete data loading stack."""

    def __init__(self, context: DataBuilderContext) -> None:
        self.context = context

    @abstractmethod
    def build(self) -> DataLoaders:
        """Return ready train, validation, and test iterables."""


REGISTRIES.data_builders.require_base(DataBuilder)


def build_data_loaders(
    config: ComponentConfig,
    *,
    seed: int,
    registries: RegistryCatalog = REGISTRIES,
) -> DataLoaders:
    """Construct and validate one registered data builder."""

    builder = registries.data_builders.create(
        config.name,
        DataBuilderContext(params=config.params, seed=seed),
    )
    if not isinstance(builder, DataBuilder):
        raise TypeError(
            f"registered data builder '{config.name}' did not produce DataBuilder"
        )
    loaders = builder.build()
    if not isinstance(loaders, DataLoaders):
        raise TypeError(
            f"data builder '{config.name}' must return DataLoaders"
        )
    return loaders


__all__ = [
    "DataBuilder",
    "DataBuilderContext",
    "DataLoaders",
    "build_data_loaders",
]
