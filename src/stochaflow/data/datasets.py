"""Dataset builders and transform utilities."""

from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from stochaflow.utils.registry import register_dataset


def _validate_image_channels(channels: int) -> None:
    if channels <= 0:
        raise ValueError("channels must be positive")


def _resolve_train_flag(split: str) -> bool:
    normalized = split.lower()
    if normalized == "train":
        return True
    if normalized in {"test", "eval", "validation", "val"}:
        return False
    raise ValueError(f"unsupported dataset split '{split}'")


def _build_image_transform(
    *,
    image_size: int,
    channels: int,
    normalize: bool,
    random_horizontal_flip: bool,
    grayscale_output_channels: int | None = None,
) -> transforms.Compose:
    """Build image preprocessing for diffusion training.

    When ``normalize`` is true, images are mapped from ``[0, 1]`` to ``[-1, 1]``
    after ``ToTensor`` using ``Normalize(mean=0.5, std=0.5)`` per channel.
    """

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    _validate_image_channels(channels)

    transform_steps: list[Any] = [
        transforms.Resize(image_size),
    ]
    if random_horizontal_flip:
        transform_steps.append(transforms.RandomHorizontalFlip())
    if grayscale_output_channels is not None:
        transform_steps.append(
            transforms.Grayscale(num_output_channels=grayscale_output_channels)
        )
    transform_steps.append(transforms.ToTensor())
    if normalize:
        mean: Sequence[float] = [0.5] * channels
        std: Sequence[float] = [0.5] * channels
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)


@register_dataset("mnist")
def build_mnist_dataset(
    *,
    root: str = "./data",
    split: str = "train",
    image_size: int = 32,
    channels: int = 1,
    download: bool = True,
    normalize: bool = True,
) -> Dataset:
    """Build an MNIST dataset with DDPM-friendly image preprocessing."""

    if channels not in {1, 3}:
        raise ValueError("mnist supports channels set to 1 or 3")
    train = _resolve_train_flag(split)
    transform = _build_image_transform(
        image_size=image_size,
        channels=channels,
        normalize=normalize,
        random_horizontal_flip=False,
        grayscale_output_channels=channels,
    )
    return datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )


@register_dataset("cifar10")
def build_cifar10_dataset(
    *,
    root: str = "./data",
    split: str = "train",
    image_size: int = 32,
    channels: int = 3,
    download: bool = True,
    normalize: bool = True,
    random_horizontal_flip: bool = True,
) -> Dataset:
    """Build a CIFAR-10 dataset with configurable image preprocessing."""

    if channels not in {1, 3}:
        raise ValueError("cifar10 supports channels set to 1 or 3")
    train = _resolve_train_flag(split)
    transform = _build_image_transform(
        image_size=image_size,
        channels=channels,
        normalize=normalize,
        random_horizontal_flip=train and random_horizontal_flip,
        grayscale_output_channels=1 if channels == 1 else None,
    )
    return datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )
