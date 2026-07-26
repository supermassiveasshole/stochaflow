"""Image transformations for built-in data recipes."""

from __future__ import annotations

import math
import random
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomCrop
from torchvision.transforms import functional as vision_functional


def extract_image(sample: Any) -> Any:
    """Extract the image value from common torchvision samples."""

    if isinstance(sample, (tuple, list)) and sample:
        return sample[0]
    return sample


def image_size(image: Any) -> tuple[int, int]:
    """Return width and height for a PIL image or image tensor."""

    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, torch.Tensor) and image.ndim >= 2:
        return int(image.shape[-1]), int(image.shape[-2])
    raise TypeError("built-in image recipes require PIL images or image tensors")


def _convert_channels(image: Any, channels: int) -> Any:
    if channels not in {1, 3}:
        raise ValueError("built-in image recipes support 1 or 3 channels")
    if isinstance(image, Image.Image):
        return image.convert("L" if channels == 1 else "RGB")
    if not isinstance(image, torch.Tensor):
        raise TypeError("image transforms require PIL images or tensors")
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[-3] == channels:
        return image
    if channels == 1:
        return vision_functional.rgb_to_grayscale(image, num_output_channels=1)
    if image.shape[-3] == 1:
        return image.repeat_interleave(3, dim=-3)
    raise ValueError(f"cannot convert {image.shape[-3]} image channels to 3")


def _to_float_tensor(image: Any, *, normalize: bool) -> torch.Tensor:
    if isinstance(image, Image.Image):
        tensor = vision_functional.pil_to_tensor(image)
        tensor = vision_functional.convert_image_dtype(tensor, torch.float32)
    elif isinstance(image, torch.Tensor):
        if image.is_floating_point():
            tensor = image.to(dtype=torch.float32)
        else:
            tensor = vision_functional.convert_image_dtype(image, torch.float32)
    else:
        raise TypeError("image transforms require PIL images or tensors")
    if normalize:
        tensor = tensor.mul(2.0).sub(1.0)
    return tensor


class ImageTransform:
    """Resize-cover, crop, augment, convert, and normalize one image."""

    def __init__(
        self,
        size: tuple[int, int],
        *,
        role: str,
        channels: int,
        normalize: bool,
        random_horizontal_flip: bool,
    ) -> None:
        self.height, self.width = size
        self.role = role
        self.channels = channels
        self.normalize = normalize
        self.random_horizontal_flip = random_horizontal_flip

    def __call__(self, image: Any) -> torch.Tensor:
        image = _convert_channels(image, self.channels)
        width, height = image_size(image)
        scale = max(self.width / width, self.height / height)
        resized_width = max(self.width, round(width * scale))
        resized_height = max(self.height, round(height * scale))
        image = vision_functional.resize(
            image,
            [resized_height, resized_width],
            antialias=True,
        )
        crop_size = (self.height, self.width)
        if self.role == "train":
            top, left, crop_height, crop_width = RandomCrop.get_params(
                image,
                output_size=crop_size,
            )
            image = vision_functional.crop(
                image, top, left, crop_height, crop_width
            )
            if self.random_horizontal_flip and random.random() < 0.5:
                image = vision_functional.hflip(image)
        else:
            image = vision_functional.center_crop(image, list(crop_size))
        return _to_float_tensor(image, normalize=self.normalize)


class GeneratedSuperResolutionTransform:
    """Generate an aligned low-resolution observation from one HR image."""

    def __init__(
        self,
        high_resolution: tuple[int, int],
        low_resolution: tuple[int, int],
        *,
        role: str,
        channels: int,
        normalize: bool,
        random_horizontal_flip: bool,
    ) -> None:
        self.high_transform = ImageTransform(
            high_resolution,
            role=role,
            channels=channels,
            normalize=False,
            random_horizontal_flip=random_horizontal_flip,
        )
        self.low_resolution = low_resolution
        self.normalize = normalize

    def __call__(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        high = self.high_transform(image)
        low = vision_functional.resize(
            high,
            list(self.low_resolution),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        low = low.clamp(0.0, 1.0)
        if self.normalize:
            high = high.mul(2.0).sub(1.0)
            low = low.mul(2.0).sub(1.0)
        return high, low


class PairedSuperResolutionTransform:
    """Apply scale-aligned geometry to an existing HR/LR image pair."""

    def __init__(
        self,
        high_resolution: tuple[int, int],
        low_resolution: tuple[int, int],
        *,
        role: str,
        channels: int,
        normalize: bool,
        random_horizontal_flip: bool,
    ) -> None:
        high_height, high_width = high_resolution
        low_height, low_width = low_resolution
        self.scale_y = high_height // low_height
        self.scale_x = high_width // low_width
        self.high_resolution = high_resolution
        self.low_resolution = low_resolution
        self.role = role
        self.channels = channels
        self.normalize = normalize
        self.random_horizontal_flip = random_horizontal_flip

    def __call__(self, pair: tuple[Any, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        high, low = pair
        high = _convert_channels(high, self.channels)
        low = _convert_channels(low, self.channels)
        high_width, high_height = image_size(high)
        low_width, low_height = image_size(low)
        if high_width != low_width * self.scale_x or high_height != low_height * self.scale_y:
            raise ValueError(
                "paired LR/HR dimensions do not match the configured scale"
            )

        target_low_height, target_low_width = self.low_resolution
        resize_factor = max(
            target_low_width / low_width,
            target_low_height / low_height,
            1.0,
        )
        resized_low_width = max(target_low_width, math.ceil(low_width * resize_factor))
        resized_low_height = max(
            target_low_height, math.ceil(low_height * resize_factor)
        )
        resized_high_width = resized_low_width * self.scale_x
        resized_high_height = resized_low_height * self.scale_y
        low = vision_functional.resize(
            low,
            [resized_low_height, resized_low_width],
            antialias=True,
        )
        high = vision_functional.resize(
            high,
            [resized_high_height, resized_high_width],
            antialias=True,
        )

        if self.role == "train":
            top, left, _, _ = RandomCrop.get_params(
                low,
                output_size=self.low_resolution,
            )
        else:
            top = (resized_low_height - target_low_height) // 2
            left = (resized_low_width - target_low_width) // 2
        low = vision_functional.crop(
            low,
            top,
            left,
            target_low_height,
            target_low_width,
        )
        high = vision_functional.crop(
            high,
            top * self.scale_y,
            left * self.scale_x,
            self.high_resolution[0],
            self.high_resolution[1],
        )
        if (
            self.role == "train"
            and self.random_horizontal_flip
            and random.random() < 0.5
        ):
            high = vision_functional.hflip(high)
            low = vision_functional.hflip(low)
        return (
            _to_float_tensor(high, normalize=self.normalize),
            _to_float_tensor(low, normalize=self.normalize),
        )


__all__ = [
    "GeneratedSuperResolutionTransform",
    "ImageTransform",
    "PairedSuperResolutionTransform",
    "extract_image",
    "image_size",
]
