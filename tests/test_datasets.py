"""Tests for dataset preprocessing utilities."""

from typing import cast

from PIL import Image
import torch

from stochaflow.data.datasets import _build_image_transform


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
