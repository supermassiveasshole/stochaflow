"""Private index partition helpers for built-in map-style recipes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence, Sized
from typing import cast

import torch
from torch.utils.data import Subset

from stochaflow.data.datasets import ImageDatasetPartitions
from stochaflow.data.image_contracts import ClassLabeledImageFileRecord
from stochaflow.data.recipe_config import (
    ClassStratifiedPartitionRecipeConfig,
    PartitionRecipeConfig,
)

_CLASS_PARTITION_DOMAIN = b"stochaflow.class-stratified-partition.v1"


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


def _record_identity(
    record: ClassLabeledImageFileRecord,
) -> tuple[str, str, str]:
    image = record.image
    return image.tree, image.path, image.sha256


def _seed_bytes(value: int | str) -> bytes:
    if isinstance(value, int):
        return f"integer:{value}".encode("ascii")
    return b"string:" + value.encode("utf-8")


def _record_rank(
    record: ClassLabeledImageFileRecord,
    *,
    partition_seed: int | str,
) -> bytes:
    digest = hashlib.sha256()
    for value in (
        _CLASS_PARTITION_DOMAIN,
        _seed_bytes(partition_seed),
        record.image.tree.encode("utf-8"),
        record.image.path.encode("utf-8"),
        record.image.sha256.encode("ascii"),
    ):
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)
    return digest.digest()


def partition_class_labeled_records(
    train_records: Sequence[ClassLabeledImageFileRecord],
    config: ClassStratifiedPartitionRecipeConfig,
    *,
    seed: int,
) -> tuple[
    tuple[ClassLabeledImageFileRecord, ...],
    tuple[ClassLabeledImageFileRecord, ...],
]:
    """Reserve a stable, exact validation count from every represented class."""

    config.validate(path="data.params.partition")
    seed_value = cast(object, seed)
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        raise TypeError("class-labeled partition run seed must be an integer")
    records_value = cast(object, train_records)
    if isinstance(records_value, (str, bytes)) or not isinstance(
        records_value,
        Sequence,
    ):
        raise TypeError("class-labeled train inventory must be a sequence")
    records = tuple(train_records)
    if not records:
        raise ValueError("class-labeled train inventory must not be empty")
    if any(
        not isinstance(cast(object, record), ClassLabeledImageFileRecord)
        for record in records
    ):
        raise TypeError(
            "class-labeled train inventory must contain "
            "ClassLabeledImageFileRecord"
        )

    identities = tuple(_record_identity(record) for record in records)
    if len(identities) != len(set(identities)):
        raise ValueError(
            "class-labeled train inventory contains duplicate image identities"
        )
    by_class: dict[int, list[ClassLabeledImageFileRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_label].append(record)

    partition_seed = config.seed if config.seed is not None else seed
    selected: set[tuple[str, str, str]] = set()
    for class_label, candidates in sorted(by_class.items()):
        if len(candidates) <= config.validation_per_class:
            raise ValueError(
                f"class label {class_label} has {len(candidates)} records; "
                f"validation_per_class={config.validation_per_class} must "
                "leave at least one training record"
            )
        ranked = sorted(
            candidates,
            key=lambda record: (
                _record_rank(record, partition_seed=partition_seed),
                _record_identity(record),
            ),
        )
        selected.update(
            _record_identity(record)
            for record in ranked[: config.validation_per_class]
        )

    train = tuple(
        record
        for record in records
        if _record_identity(record) not in selected
    )
    validation = tuple(
        record
        for record in records
        if _record_identity(record) in selected
    )
    return train, validation


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


__all__ = ["partition_class_labeled_records", "partition_datasets"]
