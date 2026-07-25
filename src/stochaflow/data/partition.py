"""Private index partition helpers for built-in map-style recipes."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.utils.data import Dataset, Subset

from stochaflow.data.recipe_config import PartitionRecipeConfig
from stochaflow.data.sources import SourceDatasets
from stochaflow.utils.config import ConfigError


@dataclass(frozen=True, slots=True)
class PartitionedDatasets:
    train: Dataset[Any]
    validation: Dataset[Any] | None = None
    test: Dataset[Any] | None = None


def validate_partition(config: PartitionRecipeConfig, *, path: str) -> None:
    if not isinstance(cast(object, config.mode), str):
        raise ConfigError(f"{path}.mode must be a string")
    if config.mode not in {"none", "official", "holdout", "kfold"}:
        raise ConfigError(f"{path}.mode must be none, official, holdout, or kfold")
    requested = cast(object, config.validation_size)
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, (int, float)):
            raise ConfigError(f"{path}.validation_size must be numeric or null")
        if isinstance(requested, float) and not 0.0 < requested < 1.0:
            raise ConfigError(f"{path}.validation_size must be between 0 and 1")
        if isinstance(requested, int) and requested <= 0:
            raise ConfigError(f"{path}.validation_size must be positive")
    if config.mode == "holdout" and requested is None:
        raise ConfigError(f"{path}.validation_size is required for holdout")
    if config.mode != "holdout" and requested is not None:
        raise ConfigError(
            f"{path}.validation_size is only valid for holdout"
        )
    if config.mode == "kfold":
        if (
            not isinstance(config.num_folds, int)
            or isinstance(config.num_folds, bool)
            or config.num_folds < 2
        ):
            raise ConfigError(f"{path}.num_folds must be at least 2 for kfold")
        if not isinstance(config.fold_index, int) or isinstance(
            config.fold_index, bool
        ):
            raise ConfigError(f"{path}.fold_index is required for kfold")
        if not 0 <= config.fold_index < config.num_folds:
            raise ConfigError(f"{path}.fold_index must be in [0, num_folds)")
    elif config.num_folds is not None or config.fold_index is not None:
        raise ConfigError(
            f"{path}.num_folds and fold_index are only valid for kfold"
        )


def _permutation(size: int, *, seed: int) -> list[int]:
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()


def _validation_count(size: int, requested: float | None) -> int:
    assert requested is not None
    count = int(size * requested) if isinstance(requested, float) else requested
    if count <= 0 or count >= size:
        raise ValueError(
            "validation_size must leave at least one training and validation sample"
        )
    return count


def partition_datasets(
    source: SourceDatasets,
    config: PartitionRecipeConfig,
    *,
    seed: int,
) -> PartitionedDatasets:
    """Apply one recipe-private partition policy to a finite source."""

    validate_partition(config, path="data.params.partition")
    size = len(cast(Sized, source.train))
    if size <= 0:
        raise ValueError("training dataset must contain at least one sample")
    if config.mode == "official":
        return PartitionedDatasets(
            train=source.train,
            validation=source.validation,
            test=source.test,
        )
    if config.mode == "none":
        return PartitionedDatasets(train=source.train, test=source.test)

    indices = _permutation(size, seed=seed)
    if config.mode == "holdout":
        validation_count = _validation_count(size, config.validation_size)
        return PartitionedDatasets(
            train=Subset(source.train, indices[validation_count:]),
            validation=Subset(source.train, indices[:validation_count]),
            test=source.test,
        )

    assert config.num_folds is not None
    assert config.fold_index is not None
    if config.num_folds > size:
        raise ValueError("num_folds must not exceed the training dataset size")
    fold_sizes = [size // config.num_folds] * config.num_folds
    for index in range(size % config.num_folds):
        fold_sizes[index] += 1
    folds: list[list[int]] = []
    offset = 0
    for fold_size in fold_sizes:
        folds.append(indices[offset : offset + fold_size])
        offset += fold_size
    validation_indices = folds[config.fold_index]
    training_indices = [
        index
        for fold_index, fold in enumerate(folds)
        if fold_index != config.fold_index
        for index in fold
    ]
    return PartitionedDatasets(
        train=Subset(source.train, training_indices),
        validation=Subset(source.train, validation_indices),
        test=source.test,
    )


__all__ = ["PartitionedDatasets", "partition_datasets", "validate_partition"]
