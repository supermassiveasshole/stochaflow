"""Private configuration types for built-in data recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LoaderRecipeConfig:
    batch_size: int = 128
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = None
    steps_per_epoch: int | str = "auto"


@dataclass(slots=True)
class PartitionRecipeConfig:
    mode: str = "none"
    validation_size: int | float | None = None
    num_folds: int | None = None
    fold_index: int | None = None


@dataclass(slots=True)
class ImageRecipeConfig:
    size: list[int]
    channels: int = 3
    normalize: bool = True
    random_horizontal_flip: bool = False


@dataclass(slots=True)
class ImageDataBuilderConfig:
    source: dict[str, Any]
    image: ImageRecipeConfig
    partition: PartitionRecipeConfig = field(
        default_factory=PartitionRecipeConfig
    )
    loader: LoaderRecipeConfig = field(default_factory=LoaderRecipeConfig)


@dataclass(slots=True)
class SuperResolutionImageRecipeConfig:
    high_resolution: list[int]
    low_resolution: list[int]
    channels: int = 3
    normalize: bool = True
    random_horizontal_flip: bool = False


@dataclass(slots=True)
class LowResolutionRecipeConfig:
    kind: str = "bicubic"


@dataclass(slots=True)
class SuperResolutionDataBuilderConfig:
    source: dict[str, Any]
    image: SuperResolutionImageRecipeConfig
    low_resolution: LowResolutionRecipeConfig = field(
        default_factory=LowResolutionRecipeConfig
    )
    partition: PartitionRecipeConfig = field(
        default_factory=PartitionRecipeConfig
    )
    loader: LoaderRecipeConfig = field(default_factory=LoaderRecipeConfig)


@dataclass(slots=True)
class MultiResolutionSourceConfig:
    id: str
    source: dict[str, Any]
    sampling_weight: float | None = None


@dataclass(slots=True)
class MultiResolutionImageRecipeConfig:
    channels: int = 3
    normalize: bool = True
    random_horizontal_flip: bool = False


@dataclass(slots=True)
class ResolutionBucketRecipeConfig:
    name: str
    height: int
    width: int


@dataclass(slots=True)
class ResolutionBatchingRecipeConfig:
    buckets: list[ResolutionBucketRecipeConfig]
    base_bucket: str
    dynamic_batch_size: bool = True


@dataclass(slots=True)
class MultiResolutionDataBuilderConfig:
    sources: list[MultiResolutionSourceConfig]
    image: MultiResolutionImageRecipeConfig
    batching: ResolutionBatchingRecipeConfig
    partition: PartitionRecipeConfig = field(
        default_factory=PartitionRecipeConfig
    )
    loader: LoaderRecipeConfig = field(default_factory=LoaderRecipeConfig)


__all__ = [
    "ImageDataBuilderConfig",
    "ImageRecipeConfig",
    "LoaderRecipeConfig",
    "LowResolutionRecipeConfig",
    "MultiResolutionDataBuilderConfig",
    "MultiResolutionImageRecipeConfig",
    "MultiResolutionSourceConfig",
    "PartitionRecipeConfig",
    "ResolutionBatchingRecipeConfig",
    "ResolutionBucketRecipeConfig",
    "SuperResolutionDataBuilderConfig",
    "SuperResolutionImageRecipeConfig",
]
