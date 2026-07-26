"""Private index partition helpers for built-in map-style recipes."""

from __future__ import annotations

from collections.abc import Sized
from typing import cast

import torch
from torch.utils.data import Subset

from stochaflow.data.datasets import ImageDatasetPartitions
from stochaflow.data.recipe_config import PartitionRecipeConfig


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
    source: ImageDatasetPartitions,
    config: PartitionRecipeConfig,
    *,
    seed: int,
) -> ImageDatasetPartitions:
    """Apply one recipe-private partition policy to a finite source."""

    size = len(cast(Sized, source.train))
    if size <= 0:
        raise ValueError("training dataset must contain at least one sample")
    if config.mode == "official":
        return ImageDatasetPartitions(
            train=source.train,
            validation=source.validation,
            test=source.test,
        )
    if config.mode == "none":
        return ImageDatasetPartitions(train=source.train, test=source.test)

    indices = _permutation(size, seed=seed)
    if config.mode == "holdout":
        validation_count = _validation_count(size, config.validation_size)
        return ImageDatasetPartitions(
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
    return ImageDatasetPartitions(
        train=Subset(source.train, training_indices),
        validation=Subset(source.train, validation_indices),
        test=source.test,
    )


__all__ = ["partition_datasets"]
