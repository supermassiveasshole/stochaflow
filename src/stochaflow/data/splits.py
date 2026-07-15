"""Object-oriented global split strategies for dataset mixtures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch

from stochaflow.data.contracts import (
    DatasetBuildRequest,
    DatasetFactory,
    DatasetMixture,
    DatasetSelection,
)
from stochaflow.utils.config import DataSplitConfig, DatasetConfig
from stochaflow.utils.registry import REGISTRIES

LogicalSplit = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class ConfiguredDatasetFactory:
    """Pair one source declaration with its instantiated factory."""

    config: DatasetConfig
    factory: DatasetFactory


class DatasetMaterializer:
    """Build aligned source views for logical experiment partitions."""

    def __init__(
        self,
        sources: tuple[ConfiguredDatasetFactory, ...],
        *,
        seed: int,
    ) -> None:
        if not sources:
            raise ValueError("dataset materializer requires at least one source")
        self.sources = sources
        self.seed = seed

    def build(
        self,
        logical_split: LogicalSplit,
        *,
        role: Literal["train", "eval"],
    ) -> DatasetMixture | None:
        native_splits = [
            getattr(source.config.splits, logical_split) for source in self.sources
        ]
        if any(split is None for split in native_splits) and any(
            split is not None for split in native_splits
        ):
            raise ValueError(
                f"logical split '{logical_split}' must be declared by every "
                "configured source or omitted by every source"
            )
        views = []
        for source in self.sources:
            native_split = getattr(source.config.splits, logical_split)
            if native_split is None:
                continue
            view = source.factory.build(
                DatasetBuildRequest(
                    native_split=native_split,
                    role=role,
                    seed=self.seed,
                )
            )
            if view.source_id != source.config.id:
                raise ValueError(
                    f"dataset factory '{source.config.factory}' returned source id "
                    f"'{view.source_id}', expected '{source.config.id}'"
                )
            views.append(view)
        return DatasetMixture(views) if views else None


@dataclass(frozen=True, slots=True)
class SplitContext:
    """Inputs shared by all split strategy implementations."""

    config: DataSplitConfig
    datasets: DatasetMaterializer
    seed: int


@dataclass(frozen=True, slots=True)
class DataPartitions:
    """Datasets selected for one independent training run."""

    train: DatasetMixture | DatasetSelection
    valid: DatasetMixture | DatasetSelection | None = None
    test: DatasetMixture | DatasetSelection | None = None
    fold_index: int | None = None
    num_folds: int | None = None


class SplitStrategy(ABC):
    """Polymorphic policy for global train/validation/test selection."""

    @abstractmethod
    def split(self, context: SplitContext) -> list[DataPartitions]:
        """Build one or more independent sets of data partitions."""

    @staticmethod
    def _required(
        dataset: DatasetMixture | None,
        *,
        logical_split: str,
    ) -> DatasetMixture:
        if dataset is None:
            raise ValueError(f"no dataset declares logical split '{logical_split}'")
        return dataset

    @staticmethod
    def _validate_aligned(
        train: DatasetMixture,
        valid: DatasetMixture,
    ) -> None:
        if train.sample_keys != valid.sample_keys:
            raise ValueError(
                "training and evaluation views of the logical train split must "
                "have identical stable sample_keys"
            )


REGISTRIES.split_strategies.require_base(SplitStrategy)


@REGISTRIES.split_strategies.register("none")
class TrainOnlySplitStrategy(SplitStrategy):
    """Use complete native training partitions without validation."""

    def split(self, context: SplitContext) -> list[DataPartitions]:
        train = self._required(
            context.datasets.build("train", role="train"),
            logical_split="train",
        )
        test = context.datasets.build("test", role="eval")
        return [DataPartitions(train=train, test=test)]


@REGISTRIES.split_strategies.register("official")
class OfficialSplitStrategy(SplitStrategy):
    """Use each source's configured native train/validation/test mapping."""

    def split(self, context: SplitContext) -> list[DataPartitions]:
        train = self._required(
            context.datasets.build("train", role="train"),
            logical_split="train",
        )
        return [
            DataPartitions(
                train=train,
                valid=context.datasets.build("validation", role="eval"),
                test=context.datasets.build("test", role="eval"),
            )
        ]


def _permuted_indices(size: int, *, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(size, generator=generator).tolist()


def _validation_count(size: int, requested: int | float | None) -> int:
    if requested is None:
        raise ValueError("random_holdout requires validation_size")
    count = int(size * requested) if isinstance(requested, float) else requested
    if count <= 0 or count >= size:
        raise ValueError(
            "validation_size must leave at least one training and validation sample"
        )
    return count


@REGISTRIES.split_strategies.register("random_holdout")
class RandomHoldoutSplitStrategy(SplitStrategy):
    """Generate one deterministic global holdout after source union."""

    def split(self, context: SplitContext) -> list[DataPartitions]:
        train_view = self._required(
            context.datasets.build("train", role="train"),
            logical_split="train",
        )
        valid_view = self._required(
            context.datasets.build("train", role="eval"),
            logical_split="train",
        )
        self._validate_aligned(train_view, valid_view)
        validation_count = _validation_count(
            len(train_view),
            context.config.validation_size,
        )
        indices = _permuted_indices(len(train_view), seed=context.seed)
        valid_indices = indices[:validation_count]
        train_indices = indices[validation_count:]
        return [
            DataPartitions(
                train=DatasetSelection(train_view, train_indices),
                valid=DatasetSelection(valid_view, valid_indices),
                test=context.datasets.build("test", role="eval"),
            )
        ]


def _folds(indices: list[int], *, num_folds: int) -> list[list[int]]:
    fold_sizes = [len(indices) // num_folds] * num_folds
    for fold_index in range(len(indices) % num_folds):
        fold_sizes[fold_index] += 1
    result: list[list[int]] = []
    offset = 0
    for fold_size in fold_sizes:
        result.append(indices[offset : offset + fold_size])
        offset += fold_size
    return result


@REGISTRIES.split_strategies.register("kfold")
class KFoldSplitStrategy(SplitStrategy):
    """Generate deterministic balanced folds over the global source union."""

    def split(self, context: SplitContext) -> list[DataPartitions]:
        num_folds = context.config.num_folds
        if num_folds is None or num_folds < 2:
            raise ValueError("kfold requires num_folds >= 2")
        train_view = self._required(
            context.datasets.build("train", role="train"),
            logical_split="train",
        )
        valid_view = self._required(
            context.datasets.build("train", role="eval"),
            logical_split="train",
        )
        self._validate_aligned(train_view, valid_view)
        if num_folds > len(train_view):
            raise ValueError("num_folds must not exceed the combined dataset size")

        fold_indices = _folds(
            _permuted_indices(len(train_view), seed=context.seed),
            num_folds=num_folds,
        )
        requested = context.config.fold_index
        selected = range(num_folds) if requested is None else [requested]
        test = context.datasets.build("test", role="eval")
        bundles: list[DataPartitions] = []
        for fold_index in selected:
            valid_indices = fold_indices[fold_index]
            train_indices = [
                index
                for other_index, fold in enumerate(fold_indices)
                if other_index != fold_index
                for index in fold
            ]
            bundles.append(
                DataPartitions(
                    train=DatasetSelection(train_view, train_indices),
                    valid=DatasetSelection(valid_view, valid_indices),
                    test=test,
                    fold_index=fold_index,
                    num_folds=num_folds,
                )
            )
        return bundles


__all__ = [
    "ConfiguredDatasetFactory",
    "DataPartitions",
    "DatasetMaterializer",
    "KFoldSplitStrategy",
    "OfficialSplitStrategy",
    "RandomHoldoutSplitStrategy",
    "SplitContext",
    "SplitStrategy",
    "TrainOnlySplitStrategy",
]
