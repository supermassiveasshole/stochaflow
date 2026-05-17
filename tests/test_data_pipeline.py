"""Tests for unified data bundle construction."""

from collections.abc import Sized

import torch
from torch.utils.data import Subset, TensorDataset

from stochaflow.data.pipeline import build_data_bundle, build_data_bundles
from stochaflow.utils.config import (
    ComponentConfig,
    DataConfig,
    DataSplitConfig,
    DataloaderConfig,
)
from stochaflow.utils.registry import register_dataset


def _length(value: object) -> int:
    assert isinstance(value, Sized)
    return len(value)


@register_dataset("split_aware_tensor_dataset")
def build_split_aware_tensor_dataset(
    *,
    split: str = "train",
    role: str = "train",
    sizes: dict[str, int] | None = None,
) -> TensorDataset:
    sizes = sizes or {"train": 9, "val": 3, "test": 2}
    size = sizes[split]
    role_offset = 100 if role == "eval" else 0
    split_offset = {"train": 0, "val": 1000, "test": 2000}.get(split, 3000)
    values = torch.arange(size).unsqueeze(1).float() + role_offset + split_offset
    return TensorDataset(values)


def _data_config(splits: DataSplitConfig) -> DataConfig:
    return DataConfig(
        dataset=ComponentConfig(
            name="split_aware_tensor_dataset",
            params={"sizes": {"train": 9, "val": 3, "test": 2}},
        ),
        dataloader=DataloaderConfig(
            batch_size=2,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=False,
        ),
        splits=splits,
    )


def test_random_holdout_is_deterministic_and_builds_optional_test() -> None:
    config = _data_config(
        DataSplitConfig(
            mode="random_holdout",
            train_split="train",
            validation_size=3,
            test_split="test",
        )
    )

    first = build_data_bundle(config, seed=7)
    second = build_data_bundle(config, seed=7)

    assert first.valid is not None
    assert second.valid is not None
    assert first.test is not None
    assert isinstance(first.train.dataset, Subset)
    assert isinstance(first.valid.dataset, Subset)
    assert isinstance(second.train.dataset, Subset)
    assert isinstance(second.valid.dataset, Subset)
    assert first.train.dataset.indices == second.train.dataset.indices
    assert first.valid.dataset.indices == second.valid.dataset.indices
    assert _length(first.train.dataset) == 6
    assert _length(first.valid.dataset) == 3
    assert _length(first.test.dataset) == 2


def test_official_mode_uses_named_splits_without_subsetting() -> None:
    bundle = build_data_bundle(
        _data_config(
            DataSplitConfig(
                mode="official",
                train_split="train",
                validation_split="val",
                test_split="test",
            )
        ),
        seed=3,
    )

    assert _length(bundle.train.dataset) == 9
    assert bundle.valid is not None
    assert _length(bundle.valid.dataset) == 3
    assert bundle.test is not None
    assert _length(bundle.test.dataset) == 2


def test_all_mode_concatenates_requested_training_splits() -> None:
    bundle = build_data_bundle(
        _data_config(
            DataSplitConfig(
                mode="all",
                train_splits=["train", "val", "test"],
                validation_size=None,
                test_split=None,
            )
        ),
        seed=5,
    )

    assert _length(bundle.train.dataset) == 14
    assert bundle.valid is None
    assert bundle.test is None


def test_none_mode_builds_train_only() -> None:
    bundle = build_data_bundle(
        _data_config(DataSplitConfig(mode="none", train_split="train")),
        seed=11,
    )

    assert _length(bundle.train.dataset) == 9
    assert bundle.valid is None
    assert bundle.test is None


def test_kfold_validation_indices_cover_each_sample_once() -> None:
    bundles = build_data_bundles(
        _data_config(
            DataSplitConfig(
                mode="kfold",
                train_split="train",
                num_folds=3,
                test_split=None,
            )
        ),
        seed=13,
    )

    validation_indices: list[int] = []
    for fold_index, bundle in enumerate(bundles):
        assert bundle.fold_index == fold_index
        assert bundle.num_folds == 3
        assert bundle.valid is not None
        assert isinstance(bundle.valid.dataset, Subset)
        validation_indices.extend(bundle.valid.dataset.indices)

    assert sorted(validation_indices) == list(range(9))
    assert len(set(validation_indices)) == 9
