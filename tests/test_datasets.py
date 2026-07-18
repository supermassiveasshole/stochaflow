"""Tests for bucket-aware built-in dataset factories."""

from PIL import Image
import torch
from torch.utils.data import Dataset

from stochaflow.data import (
    DatasetBuildRequest,
    DatasetFactoryContext,
    Flowers102DatasetFactory,
    ResolutionBucket,
    ResolutionBucketPolicy,
)
from stochaflow.data import datasets as dataset_module
from stochaflow.data.datasets import BucketImageTransform
from stochaflow.utils.config import ResolutionBucketConfig
from stochaflow.utils.registry import REGISTRIES


def test_bucket_image_transform_normalizes_and_converts_channels() -> None:
    transform = BucketImageTransform(
        ResolutionBucket("square", 2, 2),
        role="eval",
        channels=3,
        normalize=True,
        random_horizontal_flip=False,
    )

    black = transform(Image.new("L", (4, 2), color=0))
    white = transform(Image.new("L", (4, 2), color=255))

    assert black.shape == (3, 2, 2)
    assert white.shape == (3, 2, 2)
    assert torch.allclose(black, torch.full((3, 2, 2), -1.0))
    assert torch.allclose(white, torch.full((3, 2, 2), 1.0))


def test_bucket_image_transform_resize_cover_preserves_rectangular_target() -> None:
    transform = BucketImageTransform(
        ResolutionBucket("landscape", 32, 64),
        role="eval",
        channels=3,
        normalize=False,
        random_horizontal_flip=False,
    )

    tensor = transform(Image.new("RGB", (80, 100), color=(255, 255, 255)))

    assert tensor.shape == (3, 32, 64)
    assert torch.allclose(tensor, torch.ones_like(tensor))


def test_flowers_factory_returns_raw_images_and_size_metadata(monkeypatch) -> None:
    seen_splits: list[str] = []

    class FakeFlowers102(Dataset):
        def __init__(
            self,
            *,
            root: str,
            split: str,
            transform,
            download: bool,
        ) -> None:
            del root, download
            assert transform is None
            seen_splits.append(split)
            self.images = [
                Image.new("RGB", (40, 40)),
                Image.new("RGB", (120, 80)),
            ]

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int):
            return self.images[index], index

    monkeypatch.setattr(dataset_module.datasets, "Flowers102", FakeFlowers102)
    policy = ResolutionBucketPolicy(
        [
            ResolutionBucketConfig("square", 32, 32),
            ResolutionBucketConfig("landscape", 64, 96),
        ],
        base_bucket="landscape",
        dynamic_batch_size=True,
    )
    factory = Flowers102DatasetFactory(
        DatasetFactoryContext(
            source_id="flowers",
            params={"root": "./data", "download": False},
        )
    )

    view = factory.build(
        DatasetBuildRequest(native_split="validation", role="eval", seed=1)
    )

    assert seen_splits == ["val"]
    assert view.batch_metadata is not None
    assert [
        policy.select(item.width, item.height).name
        for item in view.batch_metadata
    ] == ["square", "landscape"]
    first_image, _ = view[0]
    second_image, _ = view[1]
    assert first_image.size == (40, 40)
    assert second_image.size == (120, 80)
    assert tuple(view.sample_keys) == ("validation:0", "validation:1")


def test_builtin_dataset_registry_contains_factory_classes() -> None:
    assert REGISTRIES.dataset_factories.resolve("flowers102") is (
        Flowers102DatasetFactory
    )
