"""Tests for dataset preprocessing utilities."""

from typing import cast

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from stochaflow.data import datasets as dataset_module
from stochaflow.data.datasets import (
    _build_flowers102_transform,
    _build_image_transform,
    build_flowers102_dataset,
)


def test_image_transform_normalizes_to_minus_one_to_one() -> None:
    transform = _build_image_transform(
        image_size=2,
        channels=1,
        normalize=True,
        random_horizontal_flip=False,
        grayscale_output_channels=1,
    )

    black = Image.new("L", (2, 2), color=0)
    white = Image.new("L", (2, 2), color=255)

    black_tensor = cast(torch.Tensor, transform(black))
    white_tensor = cast(torch.Tensor, transform(white))

    assert torch.allclose(black_tensor, torch.full((1, 2, 2), -1.0))
    assert torch.allclose(white_tensor, torch.full((1, 2, 2), 1.0))


def test_image_transform_can_leave_tensor_in_zero_to_one_range() -> None:
    transform = _build_image_transform(
        image_size=2,
        channels=1,
        normalize=False,
        random_horizontal_flip=False,
        grayscale_output_channels=1,
    )

    white = Image.new("L", (2, 2), color=255)

    white_tensor = cast(torch.Tensor, transform(white))

    assert torch.allclose(white_tensor, torch.ones((1, 2, 2)))


def test_flowers102_transform_outputs_fixed_square_rgb_tensors() -> None:
    transform = _build_flowers102_transform(
        split="val",
        image_size=64,
        channels=3,
        normalize=True,
        random_horizontal_flip=False,
    )

    white = Image.new("RGB", (96, 128), color=(255, 255, 255))
    tensor = cast(torch.Tensor, transform(white))

    assert tensor.shape == (3, 64, 64)
    assert torch.all(tensor >= -1.0)
    assert torch.all(tensor <= 1.0)
    assert torch.allclose(tensor, torch.ones((3, 64, 64)))


def test_flowers102_center_crop_train_transform_uses_stable_debug_baseline() -> None:
    transform = _build_flowers102_transform(
        split="train",
        image_size=64,
        channels=3,
        normalize=True,
        random_horizontal_flip=True,
        preprocess_mode="center_crop",
        resize_size=96,
    )

    assert isinstance(transform.transforms[0], transforms.Resize)
    assert transform.transforms[0].size == 96
    assert isinstance(transform.transforms[1], transforms.CenterCrop)
    assert transform.transforms[1].size == (64, 64)
    assert isinstance(transform.transforms[2], transforms.RandomHorizontalFlip)


def test_flowers102_eval_transform_disables_random_flip() -> None:
    transform = _build_flowers102_transform(
        split="eval",
        image_size=64,
        channels=3,
        normalize=True,
        random_horizontal_flip=True,
        preprocess_mode="center_crop",
        resize_size=96,
    )

    assert not any(
        isinstance(step, transforms.RandomHorizontalFlip)
        for step in transform.transforms
    )


def test_flowers102_random_crop_and_random_resized_crop_modes_build() -> None:
    random_crop = _build_flowers102_transform(
        split="train",
        image_size=64,
        channels=3,
        normalize=True,
        random_horizontal_flip=False,
        preprocess_mode="random_crop",
        resize_size=96,
    )
    random_resized_crop = _build_flowers102_transform(
        split="train",
        image_size=64,
        channels=3,
        normalize=True,
        random_horizontal_flip=False,
        preprocess_mode="random_resized_crop",
    )

    assert any(isinstance(step, transforms.RandomCrop) for step in random_crop.transforms)
    assert any(
        isinstance(step, transforms.RandomResizedCrop)
        for step in random_resized_crop.transforms
    )


def test_flowers102_builder_accepts_official_splits(monkeypatch) -> None:
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
            del root, transform, download
            seen_splits.append(split)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            del index
            return torch.zeros(3, 64, 64), 0

    monkeypatch.setattr(dataset_module.datasets, "Flowers102", FakeFlowers102)

    for split in ("train", "val", "test"):
        build_flowers102_dataset(root="./data", split=split, download=False)

    assert seen_splits == ["train", "val", "test"]
