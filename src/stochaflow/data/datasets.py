"""Built-in class-based image dataset factories."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sized
import random
from typing import Any, cast

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as vision_functional

from stochaflow.data.contracts import (
    DatasetBuildRequest,
    DatasetFactory,
    DatasetView,
    ResolutionBucket,
)
from stochaflow.utils.registry import REGISTRIES

REGISTRIES.dataset_factories.require_base(DatasetFactory)


def _image_size(image: Any) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return image.size
    if isinstance(image, torch.Tensor) and image.ndim >= 2:
        return int(image.shape[-1]), int(image.shape[-2])
    raise TypeError(
        "image datasets must return a PIL image or a tensor as the first value"
    )


def _first_value(sample: Any) -> Any:
    if isinstance(sample, (tuple, list)) and sample:
        return sample[0]
    return sample


def _replace_first_value(sample: Any, image: torch.Tensor) -> Any:
    if isinstance(sample, tuple):
        return (image, *sample[1:])
    if isinstance(sample, list):
        return [image, *sample[1:]]
    return image


class BucketImageTransform:
    """Resize-cover and crop an image into one explicit output bucket."""

    def __init__(
        self,
        bucket: ResolutionBucket,
        *,
        role: str,
        channels: int,
        normalize: bool,
        random_horizontal_flip: bool,
    ) -> None:
        if channels not in {1, 3}:
            raise ValueError("built-in image factories support 1 or 3 channels")
        self.bucket = bucket
        self.role = role
        self.channels = channels
        self.normalize = normalize
        self.random_horizontal_flip = random_horizontal_flip

    def _convert_channels(self, image: Any) -> Any:
        if isinstance(image, Image.Image):
            return image.convert("L" if self.channels == 1 else "RGB")
        if not isinstance(image, torch.Tensor):
            raise TypeError("bucket transforms require PIL images or tensors")
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[-3] == self.channels:
            return image
        if self.channels == 1:
            return vision_functional.rgb_to_grayscale(image, num_output_channels=1)
        if image.shape[-3] == 1:
            return image.repeat_interleave(3, dim=-3)
        raise ValueError(f"cannot convert {image.shape[-3]} image channels to 3")

    def __call__(self, image: Any) -> torch.Tensor:
        image = self._convert_channels(image)
        width, height = _image_size(image)
        scale = max(self.bucket.width / width, self.bucket.height / height)
        resized_width = max(self.bucket.width, round(width * scale))
        resized_height = max(self.bucket.height, round(height * scale))
        image = vision_functional.resize(
            image,
            [resized_height, resized_width],
            antialias=True,
        )
        crop_size = (self.bucket.height, self.bucket.width)
        if self.role == "train":
            top, left, height, width = transforms.RandomCrop.get_params(
                image,
                output_size=crop_size,
            )
            image = vision_functional.crop(image, top, left, height, width)
            if self.random_horizontal_flip and random.random() < 0.5:
                image = vision_functional.hflip(image)
        else:
            image = vision_functional.center_crop(image, list(crop_size))

        if isinstance(image, Image.Image):
            tensor = vision_functional.pil_to_tensor(image)
            tensor = vision_functional.convert_image_dtype(tensor, torch.float32)
        else:
            tensor = image
            if not tensor.is_floating_point():
                tensor = vision_functional.convert_image_dtype(tensor, torch.float32)
            else:
                tensor = tensor.to(dtype=torch.float32)
        if self.normalize:
            tensor = tensor.mul(2.0).sub(1.0)
        return tensor


class BucketedVisionDataset(Dataset[Any]):
    """Apply the transform corresponding to each sample's assigned bucket."""

    def __init__(
        self,
        dataset: Dataset[Any],
        bucket_ids: tuple[str, ...],
        transforms_by_bucket: dict[str, BucketImageTransform],
    ) -> None:
        if len(cast(Sized, dataset)) != len(bucket_ids):
            raise ValueError("bucket metadata length must match dataset length")
        self.dataset = dataset
        self.bucket_ids = bucket_ids
        self.transforms_by_bucket = transforms_by_bucket

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        image = _first_value(sample)
        transform = self.transforms_by_bucket[self.bucket_ids[index]]
        return _replace_first_value(sample, transform(image))


class TorchvisionDatasetFactory(DatasetFactory):
    """Base class for map-style torchvision datasets with bucket metadata."""

    config_parameters: frozenset[str] = frozenset()
    random_horizontal_flip = False

    @abstractmethod
    def _build_raw_dataset(self, request: DatasetBuildRequest) -> Dataset[Any]:
        """Build the native dataset without image transforms."""

    def _fixed_image_size(self) -> tuple[int, int] | None:
        return None

    def _bucket_ids(self, dataset: Dataset[Any]) -> tuple[str, ...]:
        fixed_size = self._fixed_image_size()
        if fixed_size is not None:
            bucket = self.context.buckets.select(*fixed_size)
            return (bucket.name,) * len(cast(Sized, dataset))
        bucket_ids: list[str] = []
        for index in range(len(cast(Sized, dataset))):
            width, height = _image_size(_first_value(dataset[index]))
            bucket_ids.append(self.context.buckets.select(width, height).name)
        return tuple(bucket_ids)

    def build(self, request: DatasetBuildRequest) -> DatasetView:
        raw_dataset = self._build_raw_dataset(request)
        bucket_ids = self._bucket_ids(raw_dataset)
        transforms_by_bucket = {
            bucket.name: BucketImageTransform(
                bucket,
                role=request.role,
                channels=self.context.image.channels,
                normalize=self.context.image.normalize,
                random_horizontal_flip=self.random_horizontal_flip,
            )
            for bucket in self.context.buckets.buckets
        }
        dataset = BucketedVisionDataset(
            raw_dataset,
            bucket_ids,
            transforms_by_bucket,
        )
        sample_keys = tuple(
            f"{request.native_split}:{index}" for index in range(len(dataset))
        )
        return DatasetView(
            source_id=self.context.source_id,
            dataset=dataset,
            sample_keys=sample_keys,
            bucket_ids=bucket_ids,
        )

    def _params(self) -> dict[str, Any]:
        params = dict(self.context.params)
        unknown = sorted(set(params) - self.config_parameters)
        if unknown:
            raise TypeError(
                f"dataset factory '{type(self).__name__}' received unknown "
                f"parameter(s): {', '.join(unknown)}"
            )
        return params


def _resolve_train_flag(split: str) -> bool:
    normalized = split.lower()
    if normalized == "train":
        return True
    if normalized in {"test", "eval", "validation", "val"}:
        return False
    raise ValueError(f"unsupported dataset split '{split}'")


@REGISTRIES.dataset_factories.register("mnist")
class MNISTDatasetFactory(TorchvisionDatasetFactory):
    """Build bucket-aware MNIST views."""

    config_parameters = frozenset({"root", "download"})

    def _fixed_image_size(self) -> tuple[int, int]:
        return 28, 28

    def _build_raw_dataset(self, request: DatasetBuildRequest) -> Dataset[Any]:
        params = self._params()
        return datasets.MNIST(
            root=str(params.get("root", "./data")),
            train=_resolve_train_flag(request.native_split),
            transform=None,
            download=bool(params.get("download", True)),
        )


@REGISTRIES.dataset_factories.register("cifar10")
class CIFAR10DatasetFactory(TorchvisionDatasetFactory):
    """Build bucket-aware CIFAR-10 views."""

    config_parameters = frozenset(
        {"root", "download", "random_horizontal_flip"}
    )
    random_horizontal_flip = True

    def _fixed_image_size(self) -> tuple[int, int]:
        return 32, 32

    def _build_raw_dataset(self, request: DatasetBuildRequest) -> Dataset[Any]:
        params = self._params()
        self.random_horizontal_flip = bool(
            params.get("random_horizontal_flip", True)
        )
        return datasets.CIFAR10(
            root=str(params.get("root", "./data")),
            train=_resolve_train_flag(request.native_split),
            transform=None,
            download=bool(params.get("download", True)),
        )


def _resolve_flowers102_split(split: str) -> str:
    normalized = split.lower()
    if normalized in {"train", "val", "test"}:
        return normalized
    if normalized == "validation":
        return "val"
    raise ValueError(f"unsupported Flowers102 split '{split}'")


@REGISTRIES.dataset_factories.register("flowers102")
class Flowers102DatasetFactory(TorchvisionDatasetFactory):
    """Build per-image bucketed Oxford Flowers 102 views."""

    config_parameters = frozenset(
        {"root", "download", "random_horizontal_flip"}
    )
    random_horizontal_flip = True

    def _build_raw_dataset(self, request: DatasetBuildRequest) -> Dataset[Any]:
        params = self._params()
        self.random_horizontal_flip = bool(
            params.get("random_horizontal_flip", True)
        )
        return datasets.Flowers102(
            root=str(params.get("root", "./data")),
            split=_resolve_flowers102_split(request.native_split),
            transform=None,
            download=bool(params.get("download", True)),
        )


__all__ = [
    "BucketImageTransform",
    "BucketedVisionDataset",
    "CIFAR10DatasetFactory",
    "Flowers102DatasetFactory",
    "MNISTDatasetFactory",
    "TorchvisionDatasetFactory",
]
