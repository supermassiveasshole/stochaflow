"""Regression tests for built-in image-oriented DataBuilders."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image
from torch.utils.data import Dataset

from stochaflow.data import build_data_loaders, sources
from stochaflow.data import datasets as dataset_module
from stochaflow.data.artifacts import DataSourceContext
from stochaflow.data.datasets import ImageDatasetFactory
from stochaflow.data.sources import TorchvisionImageDataSource
from stochaflow.utils.config import ComponentConfig, ConfigError


def write_image(
    path: Path,
    *,
    size: tuple[int, int] = (16, 16),
    color: tuple[int, int, int] = (40, 80, 120),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def write_folder(
    path: Path,
    count: int,
    *,
    size: tuple[int, int] = (16, 16),
) -> None:
    for index in range(count):
        write_image(
            path / f"sample_{index:03d}.png",
            size=size,
            color=(index % 255, 40, 80),
        )


def loader_params(**overrides: Any) -> dict[str, Any]:
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


def image_source(
    root: Path,
    *,
    layout: str = "flat",
    cache_name: str = "cache",
) -> dict[str, Any]:
    return {
        "name": "image_folder",
        "params": {"root": str(root), "layout": layout},
        "materialization": {
            "cache_root": str(root.parent / cache_name),
            "policy": "ensure",
            "verification": "full",
        },
    }


def image_component(
    path: Path,
    *,
    partition: dict[str, Any] | None = None,
) -> ComponentConfig:
    selected_partition = partition or {"mode": "none"}
    layout = (
        "split"
        if selected_partition.get("mode") == "official"
        else "flat"
    )
    return ComponentConfig(
        name="image",
        params={
            "source": image_source(path, layout=layout),
            "partition": selected_partition,
            "image": {
                "size": [8, 8],
                "channels": 3,
                "normalize": False,
                "random_horizontal_flip": False,
            },
            "loader": loader_params(),
        },
    )


def test_image_folder_is_stable_and_emits_standard_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "z.png")
    write_image(root / "nested" / "a.jpg")

    loaders = build_data_loaders(image_component(root), seed=3)
    images, condition = next(iter(loaders.train))
    records = cast(Any, loaders.train).dataset.dataset.records

    assert images.shape == (2, 3, 8, 8)
    assert condition == {}
    assert [record.path for record in records] == ["nested/a.jpg", "z.png"]
    assert loaders.artifact_bindings is not None
    assert loaders.artifact_bindings.ids == ("source",)


def test_image_holdout_and_kfold_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "images"
    write_folder(root, 12)
    holdout = build_data_loaders(
        image_component(
            root,
            partition={"mode": "holdout", "validation_size": 3},
        ),
        seed=11,
    )
    first_fold = build_data_loaders(
        image_component(
            root,
            partition={"mode": "kfold", "num_folds": 3, "fold_index": 1},
        ),
        seed=11,
    )
    repeated_fold = build_data_loaders(
        image_component(
            root,
            partition={"mode": "kfold", "num_folds": 3, "fold_index": 1},
        ),
        seed=11,
    )

    assert len(cast(Any, holdout.train).dataset) == 9
    assert len(cast(Any, holdout.validation).dataset) == 3
    assert len(cast(Any, first_fold.train).dataset) == 8
    first_validation = cast(Any, first_fold.validation).dataset.dataset
    repeated_validation = cast(Any, repeated_fold.validation).dataset.dataset
    assert first_validation.indices == repeated_validation.indices


def test_official_local_image_folders(tmp_path: Path) -> None:
    root = tmp_path / "images"
    write_folder(root / "train", 4)
    write_folder(root / "validation", 2)
    write_folder(root / "test", 2)

    loaders = build_data_loaders(
        image_component(root, partition={"mode": "official"}),
        seed=5,
    )

    assert len(cast(Any, loaders.train).dataset) == 4
    assert len(cast(Any, loaders.validation).dataset) == 2
    assert len(cast(Any, loaders.test).dataset) == 2


def test_recipe_rejects_old_source_schema_and_invalid_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_folder(root, 2)
    old_schema = image_component(root)
    old_schema.params["source"] = {
        "kind": "image_folder",
        "path": str(root),
    }
    with pytest.raises(ConfigError, match=r"source\.kind"):
        build_data_loaders(old_schema, seed=1)

    missing_materialization = image_component(root)
    del missing_materialization.params["source"]["materialization"]
    with pytest.raises(ConfigError, match=r"source\.materialization"):
        build_data_loaders(missing_materialization, seed=1)

    old_download = image_component(root)
    old_download.params["source"] = {
        "name": "torchvision",
        "params": {"dataset": "MNIST", "download": True},
        "materialization": {
            "cache_root": str(tmp_path / "torchvision-cache"),
            "policy": "ensure",
            "verification": "full",
        },
    }
    with pytest.raises(ConfigError, match=r"source\.params\.download"):
        build_data_loaders(old_download, seed=1)

    mistyped = image_component(root)
    mistyped.params["loader"]["batch_size"] = True
    with pytest.raises(ConfigError, match="batch_size"):
        build_data_loaders(mistyped, seed=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", []),
        ("policy", {}),
        ("verification", []),
        ("verification", {}),
    ],
)
def test_recipe_rejects_non_string_materialization_enums(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "images"
    write_folder(root, 2)
    component = image_component(root)
    component.params["source"]["materialization"][field] = value

    with pytest.raises(ConfigError, match=rf"source\.materialization\.{field}"):
        build_data_loaders(component, seed=1)


@pytest.mark.parametrize("value", [[], {}])
def test_recipe_rejects_non_string_image_source_layout(
    tmp_path: Path,
    value: object,
) -> None:
    root = tmp_path / "images"
    write_folder(root, 2)
    component = image_component(root)
    component.params["source"]["params"]["layout"] = value

    with pytest.raises(ConfigError, match=r"source\.params\.layout"):
        build_data_loaders(component, seed=1)


@pytest.mark.parametrize("value", [[], {}])
def test_recipe_rejects_non_string_low_resolution_kind(
    tmp_path: Path,
    value: object,
) -> None:
    root = tmp_path / "images"
    write_folder(root, 2)
    component = super_resolution_component(root)
    component.params["low_resolution"]["kind"] = value

    with pytest.raises(ConfigError, match=r"low_resolution\.kind"):
        build_data_loaders(component, seed=1)


class FixtureTinyVisionDataset(Dataset[Any]):
    """Tiny acquisition-compatible torchvision test double."""

    def __init__(
        self,
        root: str,
        *,
        train: bool | None = None,
        split: str | None = None,
        download: bool,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if download:
            data_root = Path(root)
            data_root.mkdir(parents=True, exist_ok=True)
            marker = split or ("train" if train else "test")
            (data_root / f"{marker}.bin").write_bytes(marker.encode("ascii"))
        if split == "train":
            self.size = 3
        elif split == "val":
            self.size = 2
        elif split == "test":
            self.size = 1
        elif train is False:
            self.size = 2
        else:
            self.size = 3

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        return Image.new("RGB", (12, 10), (index, 20, 30)), index


@pytest.mark.parametrize("dataset_name", ["MNIST", "CIFAR10", "Flowers102"])
def test_torchvision_sources_preserve_official_splits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_name: str,
) -> None:
    monkeypatch.setattr(
        sources.datasets,
        dataset_name,
        FixtureTinyVisionDataset,
    )
    monkeypatch.setattr(
        dataset_module.datasets,
        dataset_name,
        FixtureTinyVisionDataset,
    )
    source = TorchvisionImageDataSource(
        {"dataset": dataset_name},
        config_path="data.params.source",
    )
    artifact = source.materialize(
        DataSourceContext(
            cache_root=tmp_path / "cache",
            policy="ensure",
            verification="full",
        )
    )
    result = ImageDatasetFactory().build(artifact)

    assert len(cast(Sized, result.train)) == 3
    assert result.test is not None
    expected_test_size = 1 if dataset_name == "Flowers102" else 2
    assert len(cast(Sized, result.test)) == expected_test_size
    if dataset_name == "Flowers102":
        assert result.validation is not None
        assert len(cast(Sized, result.validation)) == 2


def super_resolution_component(
    root: Path,
    *,
    paired_root: Path | None = None,
    layout: str = "flat",
    partition: dict[str, Any] | None = None,
) -> ComponentConfig:
    paired = paired_root is not None
    source = (
        {
            "name": "paired_image_folders",
            "params": {
                "high_resolution_root": str(root),
                "low_resolution_root": str(paired_root),
                "layout": layout,
            },
            "materialization": {
                "cache_root": str(root.parent / "paired-cache"),
                "policy": "ensure",
                "verification": "full",
            },
        }
        if paired
        else image_source(root, layout=layout, cache_name="sr-cache")
    )
    return ComponentConfig(
        name="super_resolution",
        params={
            "source": source,
            "partition": partition or {"mode": "none"},
            "image": {
                "high_resolution": [12 if not paired else 8, 12 if not paired else 8],
                "low_resolution": [4, 4],
                "channels": 3,
                "normalize": paired,
                "random_horizontal_flip": paired,
            },
            "low_resolution": {"kind": "paired" if paired else "bicubic"},
            "loader": loader_params(),
        },
    )


def test_generated_and_paired_super_resolution_batches(
    tmp_path: Path,
) -> None:
    generated_root = tmp_path / "generated"
    write_folder(generated_root, 4, size=(20, 16))
    generated = build_data_loaders(
        super_resolution_component(generated_root),
        seed=7,
    )
    high, condition = next(iter(generated.train))
    assert high.shape == (2, 3, 12, 12)
    assert condition["low_res"].shape == (2, 3, 4, 4)

    high_root = tmp_path / "high"
    low_root = tmp_path / "low"
    write_folder(high_root, 3, size=(16, 16))
    write_folder(low_root, 3, size=(8, 8))
    paired = build_data_loaders(
        super_resolution_component(high_root, paired_root=low_root),
        seed=9,
    )
    high, condition = next(iter(paired.train))
    assert high.shape == (2, 3, 8, 8)
    assert condition["low_res"].shape == (2, 3, 4, 4)
    assert high.min() >= -1
    assert high.max() <= 1


def test_paired_super_resolution_preserves_official_splits(
    tmp_path: Path,
) -> None:
    high_root = tmp_path / "high"
    low_root = tmp_path / "low"
    for role, count in (("train", 4), ("validation", 2), ("test", 2)):
        write_folder(high_root / role, count, size=(16, 16))
        write_folder(low_root / role, count, size=(8, 8))

    loaders = build_data_loaders(
        super_resolution_component(
            high_root,
            paired_root=low_root,
            layout="split",
            partition={"mode": "official"},
        ),
        seed=9,
    )

    assert len(cast(Any, loaders.train).dataset) == 4
    assert len(cast(Any, loaders.validation).dataset) == 2
    assert len(cast(Any, loaders.test).dataset) == 2


def test_paired_super_resolution_rejects_non_integer_scale(
    tmp_path: Path,
) -> None:
    component = super_resolution_component(
        tmp_path / "high",
        paired_root=tmp_path / "low",
    )
    component.params["image"]["high_resolution"] = [10, 8]

    with pytest.raises(ConfigError, match="integer multiple"):
        build_data_loaders(component, seed=9)


def test_paired_source_reports_missing_and_duplicate_pairs(
    tmp_path: Path,
) -> None:
    high_root = tmp_path / "high"
    low_root = tmp_path / "low"
    write_image(high_root / "sample.png", size=(16, 16))
    write_image(low_root / "other.png", size=(8, 8))
    with pytest.raises(ValueError, match="missing LR"):
        build_data_loaders(
            super_resolution_component(high_root, paired_root=low_root),
            seed=2,
        )

    (low_root / "other.png").unlink()
    write_image(low_root / "sample.png", size=(8, 8))
    write_image(high_root / "sample.jpg", size=(16, 16))
    with pytest.raises(ValueError, match=r"duplicate.*stem"):
        build_data_loaders(
            super_resolution_component(high_root, paired_root=low_root),
            seed=2,
        )


def multi_resolution_component(
    sources_config: list[dict[str, Any]],
    *,
    partition: dict[str, Any] | None = None,
    steps_per_epoch: int | str = 200,
) -> ComponentConfig:
    return ComponentConfig(
        name="multi_resolution_image",
        params={
            "sources": sources_config,
            "partition": partition or {"mode": "none"},
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
            "loader": loader_params(
                batch_size=4,
                steps_per_epoch=steps_per_epoch,
            ),
        },
    )


def test_multi_resolution_batches_are_homogeneous_and_weighted(
    tmp_path: Path,
) -> None:
    square = tmp_path / "square"
    landscape = tmp_path / "landscape"
    write_folder(square, 6, size=(16, 16))
    write_folder(landscape, 6, size=(32, 16))
    component = multi_resolution_component(
        [
            {
                "id": "square",
                "source": image_source(square, cache_name="mix-cache"),
                "sampling_weight": 0.2,
            },
            {
                "id": "landscape",
                "source": image_source(landscape, cache_name="mix-cache"),
                "sampling_weight": 0.8,
            },
        ]
    )

    loaders = build_data_loaders(component, seed=13)
    train_loader = cast(Any, loaders.train)
    sampler = train_loader.batch_sampler
    dataset = train_loader.dataset
    counts: Counter[str] = Counter()
    for indices in sampler:
        assert len({dataset.bucket_ids[index] for index in indices}) == 1
        source_ids = {dataset.source_ids[index] for index in indices}
        assert len(source_ids) == 1
        counts.update(source_ids)

    assert counts["landscape"] / sum(counts.values()) == pytest.approx(
        0.8,
        abs=0.1,
    )
    assert loaders.artifact_bindings is not None
    assert loaders.artifact_bindings.ids == ("landscape", "square")


def test_multi_resolution_holdout_ignores_native_validation_mismatch(
    tmp_path: Path,
) -> None:
    flat = tmp_path / "flat"
    split = tmp_path / "split"
    write_folder(flat, 6)
    write_folder(split / "train", 6)
    write_folder(split / "validation", 2)
    component = multi_resolution_component(
        [
            {
                "id": "flat",
                "source": image_source(flat, cache_name="holdout-cache"),
            },
            {
                "id": "split",
                "source": image_source(
                    split,
                    layout="split",
                    cache_name="holdout-cache",
                ),
            },
        ],
        partition={"mode": "holdout", "validation_size": 2},
        steps_per_epoch="auto",
    )

    loaders = build_data_loaders(component, seed=3)

    assert len(cast(Any, loaders.train).dataset) == 10
    assert loaders.validation is not None
    assert len(cast(Any, loaders.validation).dataset) == 2


def test_multi_resolution_rejects_duplicate_source_ids() -> None:
    component = multi_resolution_component(
        [
            {
                "id": "duplicate",
                "source": image_source(Path("first")),
            },
            {
                "id": "duplicate",
                "source": image_source(Path("second")),
            },
        ]
    )

    with pytest.raises(ConfigError, match=r"sources\[1\]\.id must be unique"):
        build_data_loaders(component, seed=3)


@pytest.mark.parametrize(
    ("first_weight", "second_weight", "message"),
    [
        (0.5, None, "must be set for every source or none"),
        (0.5, 0.0, "must be positive"),
        (0.5, True, "must be positive"),
    ],
)
def test_multi_resolution_rejects_invalid_weight_sets(
    first_weight: float,
    second_weight: float | bool | None,
    message: str,
) -> None:
    component = multi_resolution_component(
        [
            {
                "id": "first",
                "source": image_source(Path("first")),
                "sampling_weight": first_weight,
            },
            {
                "id": "second",
                "source": image_source(Path("second")),
                "sampling_weight": second_weight,
            },
        ]
    )

    with pytest.raises(ConfigError, match=message):
        build_data_loaders(component, seed=3)


def test_multi_resolution_official_partition_rejects_role_mismatch(
    tmp_path: Path,
) -> None:
    flat = tmp_path / "flat"
    split = tmp_path / "split"
    write_folder(flat, 4)
    write_folder(split / "train", 4)
    write_folder(split / "validation", 2)
    component = multi_resolution_component(
        [
            {
                "id": "flat",
                "source": image_source(flat, cache_name="official-cache"),
            },
            {
                "id": "split",
                "source": image_source(
                    split,
                    layout="split",
                    cache_name="official-cache",
                ),
            },
        ],
        partition={"mode": "official"},
        steps_per_epoch="auto",
    )

    with pytest.raises(
        ValueError,
        match=r"validation.*every source or none",
    ):
        build_data_loaders(component, seed=3)
