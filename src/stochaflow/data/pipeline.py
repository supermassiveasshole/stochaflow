"""Data split and dataloader pipeline construction."""

from collections.abc import Sized
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from stochaflow.utils.config import ComponentConfig, DataConfig, DataloaderConfig


@dataclass(slots=True)
class SplitData:
    """Dataset and dataloader for one named data split."""

    name: str
    dataset: Dataset[Any]
    dataloader: DataLoader[Any]


@dataclass(slots=True)
class DataBundle:
    """Train plus optional validation/test data for one training run."""

    train: SplitData
    valid: SplitData | None = None
    test: SplitData | None = None
    fold_index: int | None = None
    num_folds: int | None = None


def _dataset_length(dataset: Dataset[Any]) -> int:
    if not isinstance(dataset, Sized):
        raise TypeError("data splitting requires sized map-style datasets")
    return len(dataset)


def _dataset_config(
    base_config: ComponentConfig,
    *,
    split: str,
    role: str,
) -> ComponentConfig:
    config = deepcopy(base_config)
    config.params["split"] = split
    config.params["role"] = role
    return config


def _eval_dataloader_config(config: DataloaderConfig) -> DataloaderConfig:
    return DataloaderConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )


def _resolve_validation_size(total_size: int, configured_size: int | float) -> int:
    if isinstance(configured_size, float):
        valid_size = int(round(total_size * configured_size))
    else:
        valid_size = configured_size
    if valid_size <= 0 or valid_size >= total_size:
        raise ValueError(
            "data.splits.validation_size must produce a non-empty validation "
            "split and leave at least one training sample"
        )
    return valid_size


def _build_dataset(component: ComponentConfig) -> Dataset[Any]:
    from stochaflow.utils.factory import build_dataset

    return build_dataset(component)


def _build_dataloader(
    dataset: Dataset[Any],
    config: DataloaderConfig,
    *,
    seed: int | None,
) -> DataLoader[Any]:
    from stochaflow.utils.factory import build_dataloader

    return build_dataloader(dataset, config, seed=seed)


def _split_data(
    name: str,
    dataset: Dataset[Any],
    dataloader_config: DataloaderConfig,
    *,
    seed: int | None,
) -> SplitData:
    return SplitData(
        name=name,
        dataset=dataset,
        dataloader=_build_dataloader(dataset, dataloader_config, seed=seed),
    )


def _build_source_dataset(
    data_config: DataConfig,
    *,
    split: str,
    role: str,
) -> Dataset[Any]:
    return _build_dataset(
        _dataset_config(data_config.dataset, split=split, role=role)
    )


def _build_optional_test_split(
    data_config: DataConfig,
    *,
    eval_dataloader_config: DataloaderConfig,
    seed: int | None,
) -> SplitData | None:
    if data_config.splits.test_split is None:
        return None
    test_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.test_split,
        role="eval",
    )
    return _split_data("test", test_dataset, eval_dataloader_config, seed=seed)


def _build_random_holdout_bundle(
    data_config: DataConfig,
    *,
    seed: int | None,
) -> DataBundle:
    if data_config.splits.validation_size is None:
        raise ValueError("random_holdout requires data.splits.validation_size")

    full_train_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.train_split,
        role="train",
    )
    full_valid_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.train_split,
        role="eval",
    )
    full_train_size = _dataset_length(full_train_dataset)
    full_valid_size = _dataset_length(full_valid_dataset)
    if full_train_size != full_valid_size:
        raise ValueError("train and validation source datasets must have equal length")

    valid_size = _resolve_validation_size(
        full_train_size,
        data_config.splits.validation_size,
    )
    train_size = full_train_size - valid_size
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    indices = torch.randperm(full_train_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    eval_config = _eval_dataloader_config(data_config.dataloader)
    return DataBundle(
        train=_split_data(
            "train",
            Subset(full_train_dataset, train_indices),
            data_config.dataloader,
            seed=seed,
        ),
        valid=_split_data(
            "valid",
            Subset(full_valid_dataset, valid_indices),
            eval_config,
            seed=seed,
        ),
        test=_build_optional_test_split(
            data_config,
            eval_dataloader_config=eval_config,
            seed=seed,
        ),
    )


def _build_official_bundle(
    data_config: DataConfig,
    *,
    seed: int | None,
) -> DataBundle:
    eval_config = _eval_dataloader_config(data_config.dataloader)
    valid = None
    if data_config.splits.validation_split is not None:
        valid_dataset = _build_source_dataset(
            data_config,
            split=data_config.splits.validation_split,
            role="eval",
        )
        valid = _split_data("valid", valid_dataset, eval_config, seed=seed)

    train_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.train_split,
        role="train",
    )
    return DataBundle(
        train=_split_data("train", train_dataset, data_config.dataloader, seed=seed),
        valid=valid,
        test=_build_optional_test_split(
            data_config,
            eval_dataloader_config=eval_config,
            seed=seed,
        ),
    )


def _build_all_bundle(
    data_config: DataConfig,
    *,
    seed: int | None,
) -> DataBundle:
    if not data_config.splits.train_splits:
        raise ValueError("all mode requires data.splits.train_splits")
    datasets = [
        _build_source_dataset(data_config, split=split, role="train")
        for split in data_config.splits.train_splits
    ]
    train_dataset: Dataset[Any]
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        train_dataset = ConcatDataset(datasets)
    return DataBundle(
        train=_split_data("train", train_dataset, data_config.dataloader, seed=seed),
    )


def _build_none_bundle(
    data_config: DataConfig,
    *,
    seed: int | None,
) -> DataBundle:
    train_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.train_split,
        role="train",
    )
    return DataBundle(
        train=_split_data("train", train_dataset, data_config.dataloader, seed=seed),
    )


def _kfold_indices(
    total_size: int,
    *,
    num_folds: int,
    seed: int | None,
) -> list[list[int]]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)
    return [fold.tolist() for fold in torch.tensor_split(indices, num_folds)]


def _build_kfold_bundles(
    data_config: DataConfig,
    *,
    seed: int | None,
) -> list[DataBundle]:
    num_folds = data_config.splits.num_folds
    if num_folds is None or num_folds < 2:
        raise ValueError("kfold mode requires data.splits.num_folds >= 2")

    full_train_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.train_split,
        role="train",
    )
    full_valid_dataset = _build_source_dataset(
        data_config,
        split=data_config.splits.train_split,
        role="eval",
    )
    total_size = _dataset_length(full_train_dataset)
    if total_size != _dataset_length(full_valid_dataset):
        raise ValueError("train and validation source datasets must have equal length")

    folds = _kfold_indices(total_size, num_folds=num_folds, seed=seed)
    selected_fold_indices = (
        [data_config.splits.fold_index]
        if data_config.splits.fold_index is not None
        else list(range(num_folds))
    )
    eval_config = _eval_dataloader_config(data_config.dataloader)
    test = _build_optional_test_split(
        data_config,
        eval_dataloader_config=eval_config,
        seed=seed,
    )

    bundles: list[DataBundle] = []
    for fold_index in selected_fold_indices:
        if fold_index is None:
            raise ValueError("fold_index cannot be null when selected")
        valid_indices = folds[fold_index]
        train_indices = [
            index
            for current_fold, fold in enumerate(folds)
            if current_fold != fold_index
            for index in fold
        ]
        bundles.append(
            DataBundle(
                train=_split_data(
                    "train",
                    Subset(full_train_dataset, train_indices),
                    data_config.dataloader,
                    seed=seed,
                ),
                valid=_split_data(
                    "valid",
                    Subset(full_valid_dataset, valid_indices),
                    eval_config,
                    seed=seed,
                ),
                test=test,
                fold_index=fold_index,
                num_folds=num_folds,
            )
        )
    return bundles


def build_data_bundles(
    data_config: DataConfig,
    *,
    seed: int | None = None,
) -> list[DataBundle]:
    """Build one or more training data bundles from configuration."""

    mode = data_config.splits.mode
    if mode == "random_holdout":
        return [_build_random_holdout_bundle(data_config, seed=seed)]
    if mode == "official":
        return [_build_official_bundle(data_config, seed=seed)]
    if mode == "all":
        return [_build_all_bundle(data_config, seed=seed)]
    if mode == "none":
        return [_build_none_bundle(data_config, seed=seed)]
    if mode == "kfold":
        return _build_kfold_bundles(data_config, seed=seed)
    raise ValueError(f"unsupported data split mode '{mode}'")


def build_data_bundle(
    data_config: DataConfig,
    *,
    seed: int | None = None,
) -> DataBundle:
    """Build a single data bundle from configuration."""

    bundles = build_data_bundles(data_config, seed=seed)
    if len(bundles) != 1:
        raise ValueError(
            "data configuration produced multiple bundles; use build_data_bundles"
        )
    return bundles[0]
