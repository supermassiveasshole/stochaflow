"""Tests for built-in image-oriented data recipes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, cast

from PIL import Image
import pytest
from torch.utils.data import Dataset

from stochaflow.data import build_data_loaders
from stochaflow.data.sources import SourceDatasets, build_image_source
from stochaflow.utils.config import ComponentConfig, ConfigError


def _write_image(
    path: Path,
    *,
    size: tuple[int, int] = (16, 16),
    color: tuple[int, int, int] = (40, 80, 120),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _folder(path: Path, count: int, *, size: tuple[int, int] = (16, 16)) -> None:
    for index in range(count):
        _write_image(
            path / f"sample_{index:03d}.png",
            size=size,
            color=(index % 255, 40, 80),
        )


def _loader_params(**overrides: Any) -> dict[str, Any]:
    result = {
        "batch_size": 2,
        "num_workers": 0,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": False,
        "persistent_workers": False,
        "steps_per_epoch": "auto",
    }
    result.update(overrides)
    return result


def _image_component(
    path: Path,
    *,
    partition: dict[str, Any] | None = None,
) -> ComponentConfig:
    return ComponentConfig(
        name="image",
        params={
            "source": {"kind": "image_folder", "path": str(path)},
            "partition": partition or {"mode": "none"},
            "image": {
                "size": [8, 8],
                "channels": 3,
                "normalize": False,
                "random_horizontal_flip": False,
            },
            "loader": _loader_params(),
        },
    )


def test_image_folder_is_stable_and_emits_standard_batch(tmp_path: Path) -> None:
    _write_image(tmp_path / "z.png")
    _write_image(tmp_path / "nested" / "a.jpg")

    loaders = build_data_loaders(_image_component(tmp_path), seed=3)
    images, condition = next(iter(loaders.train))

    assert images.shape == (2, 3, 8, 8)
    assert condition == {}
    paths = cast(Any, loaders.train).dataset.dataset.paths
    assert list(paths) == sorted(paths)


def test_image_holdout_and_single_kfold_are_deterministic(tmp_path: Path) -> None:
    _folder(tmp_path, 12)
    holdout = build_data_loaders(
        _image_component(
            tmp_path,
            partition={"mode": "holdout", "validation_size": 3},
        ),
        seed=11,
    )
    first_fold = build_data_loaders(
        _image_component(
            tmp_path,
            partition={"mode": "kfold", "num_folds": 3, "fold_index": 1},
        ),
        seed=11,
    )
    repeated_fold = build_data_loaders(
        _image_component(
            tmp_path,
            partition={"mode": "kfold", "num_folds": 3, "fold_index": 1},
        ),
        seed=11,
    )

    holdout_train = cast(Any, holdout.train)
    holdout_validation = cast(Any, holdout.validation)
    first_train = cast(Any, first_fold.train)
    first_validation = cast(Any, first_fold.validation)
    repeated_validation = cast(Any, repeated_fold.validation)
    assert len(holdout_train.dataset) == 9
    assert holdout.validation is not None
    assert len(holdout_validation.dataset) == 3
    assert first_fold.validation is not None
    assert len(first_train.dataset) == 8
    assert len(first_validation.dataset) == 4
    first_indices = first_validation.dataset.dataset.indices
    repeated_indices = repeated_validation.dataset.dataset.indices
    assert first_indices == repeated_indices


@pytest.mark.parametrize("recipe", ["image", "super_resolution"])
def test_shuffled_recipe_index_order_is_rebuilt_from_seed_and_epoch(
    tmp_path: Path,
    recipe: str,
) -> None:
    _folder(tmp_path, 12, size=(20, 16))
    if recipe == "image":
        component = _image_component(tmp_path)
    else:
        component = ComponentConfig(
            name="super_resolution",
            params={
                "source": {"kind": "image_folder", "path": str(tmp_path)},
                "partition": {"mode": "none"},
                "image": {
                    "high_resolution": [12, 12],
                    "low_resolution": [4, 4],
                    "channels": 3,
                    "normalize": False,
                    "random_horizontal_flip": False,
                },
                "low_resolution": {"kind": "bicubic"},
                "loader": _loader_params(shuffle=True),
            },
        )
    component.params["loader"]["shuffle"] = True

    uninterrupted = cast(Any, build_data_loaders(component, seed=17).train)
    uninterrupted.sampler.set_epoch(1)
    epoch_one = list(uninterrupted.sampler)
    uninterrupted.sampler.set_epoch(2)
    epoch_two = list(uninterrupted.sampler)

    rebuilt = cast(Any, build_data_loaders(component, seed=17).train)
    rebuilt.sampler.set_epoch(2)

    assert epoch_one != epoch_two
    assert list(rebuilt.sampler) == epoch_two


def test_kfold_requires_one_explicit_fold(tmp_path: Path) -> None:
    _folder(tmp_path, 6)
    with pytest.raises(ConfigError, match="fold_index is required"):
        build_data_loaders(
            _image_component(
                tmp_path,
                partition={"mode": "kfold", "num_folds": 3},
            ),
            seed=1,
        )


def test_image_recipe_rejects_unknown_and_mistyped_params(tmp_path: Path) -> None:
    _folder(tmp_path, 2)
    unknown = _image_component(tmp_path)
    unknown.params["unexpected"] = True
    with pytest.raises(ConfigError, match="unexpected"):
        build_data_loaders(unknown, seed=1)

    mistyped = _image_component(tmp_path)
    mistyped.params["loader"]["batch_size"] = True
    with pytest.raises(ConfigError, match="batch_size"):
        build_data_loaders(mistyped, seed=1)


def test_official_local_image_folders(tmp_path: Path) -> None:
    _folder(tmp_path / "train", 4)
    _folder(tmp_path / "validation", 2)
    _folder(tmp_path / "test", 2)
    loaders = build_data_loaders(
        _image_component(tmp_path, partition={"mode": "official"}),
        seed=5,
    )

    assert len(cast(Any, loaders.train).dataset) == 4
    assert loaders.validation is not None
    assert len(cast(Any, loaders.validation).dataset) == 2
    assert loaders.test is not None
    assert len(cast(Any, loaders.test).dataset) == 2


class _TinyVisionDataset(Dataset[Any]):
    def __init__(self, size: int = 3) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        return Image.new("RGB", (12, 10), (index, 20, 30)), index


@pytest.mark.parametrize("dataset_name", ["MNIST", "CIFAR10", "Flowers102"])
def test_torchvision_sources_use_curated_adapters(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
) -> None:
    from stochaflow.data import sources

    if dataset_name == "Flowers102":
        monkeypatch.setattr(
            sources.datasets,
            "Flowers102",
            lambda *args, split, **kwargs: _TinyVisionDataset(
                2 if split == "val" else 3
            ),
        )
    else:
        monkeypatch.setattr(
            sources.datasets,
            dataset_name,
            lambda *args, train, **kwargs: _TinyVisionDataset(3 if train else 2),
        )

    result = build_image_source(
        {
            "kind": "torchvision",
            "dataset": dataset_name,
            "download": False,
        },
        partition_mode="official",
    )

    assert len(cast(Any, result.train)) == 3
    assert result.test is not None
    expected_test_size = 3 if dataset_name == "Flowers102" else 2
    assert len(cast(Any, result.test)) == expected_test_size


def test_generated_super_resolution_batch(tmp_path: Path) -> None:
    _folder(tmp_path, 4, size=(20, 16))
    component = ComponentConfig(
        name="super_resolution",
        params={
            "source": {"kind": "image_folder", "path": str(tmp_path)},
            "partition": {"mode": "none"},
            "image": {
                "high_resolution": [12, 12],
                "low_resolution": [4, 4],
                "channels": 3,
                "normalize": False,
                "random_horizontal_flip": False,
            },
            "low_resolution": {"kind": "bicubic"},
            "loader": _loader_params(),
        },
    )

    loaders = build_data_loaders(component, seed=7)
    high, condition = next(iter(loaders.train))

    assert high.shape == (2, 3, 12, 12)
    assert condition["low_res"].shape == (2, 3, 4, 4)


def test_paired_super_resolution_aligns_and_reports_missing_pairs(
    tmp_path: Path,
) -> None:
    high_root = tmp_path / "high"
    low_root = tmp_path / "low"
    _folder(high_root, 3, size=(16, 16))
    _folder(low_root, 3, size=(8, 8))
    component = ComponentConfig(
        name="super_resolution",
        params={
            "source": {
                "kind": "paired_folders",
                "high_resolution_path": str(high_root),
                "low_resolution_path": str(low_root),
            },
            "partition": {"mode": "none"},
            "image": {
                "high_resolution": [8, 8],
                "low_resolution": [4, 4],
                "channels": 3,
                "normalize": True,
                "random_horizontal_flip": True,
            },
            "low_resolution": {"kind": "paired"},
            "loader": _loader_params(),
        },
    )

    high, condition = next(iter(build_data_loaders(component, seed=9).train))
    assert high.shape == (2, 3, 8, 8)
    assert condition["low_res"].shape == (2, 3, 4, 4)
    assert high.min() >= -1 and high.max() <= 1

    (low_root / "sample_000.png").unlink()
    with pytest.raises(ValueError, match="missing LR"):
        build_data_loaders(component, seed=9)


def test_paired_super_resolution_rejects_duplicate_stems_and_bad_scale(
    tmp_path: Path,
) -> None:
    duplicate_high = tmp_path / "duplicate_high"
    duplicate_low = tmp_path / "duplicate_low"
    _write_image(duplicate_high / "sample.png", size=(16, 16))
    _write_image(duplicate_high / "sample.jpg", size=(16, 16))
    _write_image(duplicate_low / "sample.png", size=(8, 8))

    def component(high: Path, low: Path) -> ComponentConfig:
        return ComponentConfig(
            name="super_resolution",
            params={
                "source": {
                    "kind": "paired_folders",
                    "high_resolution_path": str(high),
                    "low_resolution_path": str(low),
                },
                "partition": {"mode": "none"},
                "image": {
                    "high_resolution": [8, 8],
                    "low_resolution": [4, 4],
                    "channels": 3,
                    "normalize": False,
                },
                "low_resolution": {"kind": "paired"},
                "loader": _loader_params(),
            },
        )

    with pytest.raises(ValueError, match="duplicate relative image stem"):
        build_data_loaders(component(duplicate_high, duplicate_low), seed=2)

    bad_high = tmp_path / "bad_high"
    bad_low = tmp_path / "bad_low"
    _write_image(bad_high / "sample.png", size=(15, 16))
    _write_image(bad_low / "sample.png", size=(8, 8))
    loaders = build_data_loaders(component(bad_high, bad_low), seed=2)
    with pytest.raises(ValueError, match="configured scale"):
        next(iter(loaders.train))


def test_multi_resolution_batches_are_homogeneous_and_weighted(
    tmp_path: Path,
) -> None:
    square = tmp_path / "square"
    landscape = tmp_path / "landscape"
    _folder(square, 6, size=(16, 16))
    _folder(landscape, 6, size=(32, 16))
    component = ComponentConfig(
        name="multi_resolution_image",
        params={
            "sources": [
                {
                    "id": "square",
                    "source": {"kind": "image_folder", "path": str(square)},
                    "sampling_weight": 0.2,
                },
                {
                    "id": "landscape",
                    "source": {
                        "kind": "image_folder",
                        "path": str(landscape),
                    },
                    "sampling_weight": 0.8,
                },
            ],
            "partition": {"mode": "none"},
            "image": {
                "channels": 3,
                "normalize": False,
                "random_horizontal_flip": False,
            },
            "batching": {
                "buckets": [
                    {"name": "square", "height": 16, "width": 16},
                    {"name": "landscape", "height": 16, "width": 32},
                ],
                "base_bucket": "square",
                "dynamic_batch_size": True,
            },
            "loader": _loader_params(batch_size=4, steps_per_epoch=200),
        },
    )

    loaders = build_data_loaders(component, seed=13)
    train_loader = cast(Any, loaders.train)
    sampler = train_loader.batch_sampler
    dataset = train_loader.dataset
    counts: Counter[str] = Counter()
    for indices in sampler:
        bucket_ids = {dataset.bucket_ids[index] for index in indices}
        source_ids = {dataset.source_ids[index] for index in indices}
        assert len(bucket_ids) == 1
        assert len(source_ids) == 1
        counts.update(source_ids)

    assert counts["landscape"] / sum(counts.values()) == pytest.approx(0.8, abs=0.1)
    assert len(next(iter(loaders.train))[0]) in {2, 4}


def test_multi_resolution_holdout_ignores_native_validation_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stochaflow.data import builtin

    def source(raw, **kwargs):
        del kwargs
        validation = (
            _TinyVisionDataset(2)
            if raw["dataset"] == "Flowers102"
            else None
        )
        return SourceDatasets(
            train=_TinyVisionDataset(6),
            validation=validation,
            test=_TinyVisionDataset(2),
        )

    monkeypatch.setattr(builtin, "build_image_source", source)
    component = ComponentConfig(
        name="multi_resolution_image",
        params={
            "sources": [
                {
                    "id": "digits",
                    "source": {"kind": "torchvision", "dataset": "MNIST"},
                },
                {
                    "id": "flowers",
                    "source": {
                        "kind": "torchvision",
                        "dataset": "Flowers102",
                    },
                },
            ],
            "partition": {"mode": "holdout", "validation_size": 2},
            "image": {"channels": 3, "normalize": False},
            "batching": {
                "buckets": [{"name": "square", "height": 8, "width": 8}],
                "base_bucket": "square",
            },
            "loader": _loader_params(),
        },
    )

    loaders = build_data_loaders(component, seed=3)

    assert len(cast(Any, loaders.train).dataset) == 10
    assert loaders.validation is not None
    assert len(cast(Any, loaders.validation).dataset) == 2
