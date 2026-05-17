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


def _resolve_flowers102_split(split: str) -> str:
    normalized = split.lower()
    if normalized in {"train", "val", "test"}:
        return normalized
    if normalized == "validation":
        return "val"
    raise ValueError(f"unsupported Flowers102 split '{split}'")


def _resolve_dataset_role(role: str | None, *, split: str) -> str:
    if role is None:
        return "train" if split.lower() == "train" else "eval"
    normalized = role.lower()
    if normalized in {"train", "eval"}:
        return normalized
    raise ValueError("role must be 'train' or 'eval'")


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


def _build_flowers102_transform(
    *,
    split: str,
    image_size: int,
    channels: int,
    normalize: bool,
    random_horizontal_flip: bool,
    preprocess_mode: str = "center_crop",
    resize_size: int | None = None,
) -> transforms.Compose:
    """Build fixed-square preprocessing for Oxford Flowers 102 images."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if image_size < 8:
        raise ValueError("image_size must be at least 8 for Flowers102")
    if channels not in {1, 3}:
        raise ValueError("flowers102 supports channels set to 1 or 3")

    resolved_split = split.lower()
    if resolved_split == "validation":
        resolved_split = "val"
    if resolved_split not in {"train", "eval", "val", "test"}:
        raise ValueError("Flowers102 transform split must be train, eval, val, or test")

    resolved_mode = preprocess_mode.lower()
    if resolved_mode not in {"center_crop", "random_crop", "random_resized_crop"}:
        raise ValueError(
            "Flowers102 preprocess_mode must be center_crop, random_crop, "
            "or random_resized_crop"
        )
    if resize_size is None:
        resize_size = int(round(image_size * 1.5))
    if resize_size < image_size:
        raise ValueError("Flowers102 resize_size must be at least image_size")

    transform_steps: list[Any] = []
    if resolved_split == "train" and resolved_mode == "random_resized_crop":
        transform_steps.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            )
        )
        if random_horizontal_flip:
            transform_steps.append(transforms.RandomHorizontalFlip())
    else:
        transform_steps.append(transforms.Resize(resize_size))
        if resolved_split == "train" and resolved_mode == "random_crop":
            transform_steps.append(transforms.RandomCrop(image_size))
        else:
            transform_steps.append(transforms.CenterCrop(image_size))
        if resolved_split == "train" and random_horizontal_flip:
            transform_steps.append(transforms.RandomHorizontalFlip())
    if channels == 1:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
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
    role: str | None = None,
    image_size: int = 32,
    channels: int = 1,
    download: bool = True,
    normalize: bool = True,
) -> Dataset:
    """Build an MNIST dataset with DDPM-friendly image preprocessing."""

    if channels not in {1, 3}:
        raise ValueError("mnist supports channels set to 1 or 3")
    del role
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
    role: str | None = None,
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
    resolved_role = _resolve_dataset_role(role, split=split)
    transform = _build_image_transform(
        image_size=image_size,
        channels=channels,
        normalize=normalize,
        random_horizontal_flip=resolved_role == "train" and random_horizontal_flip,
        grayscale_output_channels=1 if channels == 1 else None,
    )
    return datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download,
    )


@register_dataset("flowers102")
def build_flowers102_dataset(
    *,
    root: str = "./data",
    split: str = "train",
    role: str | None = None,
    transform_split: str | None = None,
    image_size: int = 64,
    channels: int = 3,
    download: bool = True,
    normalize: bool = True,
    random_horizontal_flip: bool = True,
    preprocess_mode: str = "center_crop",
    resize_size: int | None = None,
) -> Dataset:
    """Build an Oxford Flowers 102 dataset with fixed-square preprocessing."""

    resolved_split = _resolve_flowers102_split(split)
    if transform_split is not None:
        resolved_transform_split = _resolve_flowers102_split(transform_split)
    else:
        resolved_transform_split = _resolve_dataset_role(role, split=resolved_split)
    transform = _build_flowers102_transform(
        split=resolved_transform_split,
        image_size=image_size,
        channels=channels,
        normalize=normalize,
        random_horizontal_flip=random_horizontal_flip,
        preprocess_mode=preprocess_mode,
        resize_size=resize_size,
    )
    return datasets.Flowers102(
        root=root,
        split=resolved_split,
        transform=transform,
        download=download,
    )
