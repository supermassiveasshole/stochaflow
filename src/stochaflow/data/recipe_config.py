"""Typed configuration for built-in data recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from stochaflow.utils.config import ConfigError


def _resolved_size(value: object, *, path: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{path} must contain [height, width]")
    if any(
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
        for dimension in value
    ):
        raise ConfigError(f"{path} dimensions must be positive integers")
    return int(value[0]), int(value[1])


@dataclass(slots=True)
class LoaderRecipeConfig:
    """DataLoader construction policy for one built-in recipe."""

    batch_size: int = 128
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int | None = None
    steps_per_epoch: int | str = "auto"

    def validate(self, *, path: str) -> None:
        """Validate DataLoader parameters at the recipe boundary."""

        batch_size = cast(object, self.batch_size)
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ConfigError(f"{path}.batch_size must be positive")
        num_workers = cast(object, self.num_workers)
        if (
            not isinstance(num_workers, int)
            or isinstance(num_workers, bool)
            or num_workers < 0
        ):
            raise ConfigError(f"{path}.num_workers must be non-negative")
        for name in (
            "shuffle",
            "drop_last",
            "pin_memory",
            "persistent_workers",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{path}.{name} must be boolean")
        if self.persistent_workers and self.num_workers == 0:
            raise ConfigError(
                f"{path}.persistent_workers requires num_workers > 0"
            )
        prefetch_factor = cast(object, self.prefetch_factor)
        if prefetch_factor is not None:
            if (
                not isinstance(prefetch_factor, int)
                or isinstance(prefetch_factor, bool)
                or prefetch_factor <= 0
            ):
                raise ConfigError(f"{path}.prefetch_factor must be positive")
            if self.num_workers == 0:
                raise ConfigError(
                    f"{path}.prefetch_factor requires num_workers > 0"
                )
        steps_per_epoch = cast(object, self.steps_per_epoch)
        if steps_per_epoch != "auto" and (
            not isinstance(steps_per_epoch, int)
            or isinstance(steps_per_epoch, bool)
            or steps_per_epoch <= 0
        ):
            raise ConfigError(
                f"{path}.steps_per_epoch must be positive or 'auto'"
            )


@dataclass(slots=True)
class PartitionRecipeConfig:
    """Finite-dataset partition policy."""

    mode: str = "none"
    validation_size: int | float | None = None
    num_folds: int | None = None
    fold_index: int | None = None

    def validate(self, *, path: str) -> None:
        """Validate mutually exclusive partition modes and parameters."""

        mode = cast(object, self.mode)
        if not isinstance(mode, str):
            raise ConfigError(f"{path}.mode must be a string")
        if self.mode not in {"none", "official", "holdout", "kfold"}:
            raise ConfigError(
                f"{path}.mode must be none, official, holdout, or kfold"
            )
        requested = cast(object, self.validation_size)
        if requested is not None:
            if isinstance(requested, bool) or not isinstance(
                requested, (int, float)
            ):
                raise ConfigError(
                    f"{path}.validation_size must be numeric or null"
                )
            if isinstance(requested, float) and not 0.0 < requested < 1.0:
                raise ConfigError(
                    f"{path}.validation_size must be between 0 and 1"
                )
            if isinstance(requested, int) and requested <= 0:
                raise ConfigError(
                    f"{path}.validation_size must be positive"
                )
        if self.mode == "holdout" and requested is None:
            raise ConfigError(
                f"{path}.validation_size is required for holdout"
            )
        if self.mode != "holdout" and requested is not None:
            raise ConfigError(
                f"{path}.validation_size is only valid for holdout"
            )
        if self.mode == "kfold":
            num_folds = cast(object, self.num_folds)
            if (
                not isinstance(num_folds, int)
                or isinstance(num_folds, bool)
                or num_folds < 2
            ):
                raise ConfigError(
                    f"{path}.num_folds must be at least 2 for kfold"
                )
            fold_index = cast(object, self.fold_index)
            if not isinstance(fold_index, int) or isinstance(fold_index, bool):
                raise ConfigError(
                    f"{path}.fold_index is required for kfold"
                )
            if not 0 <= fold_index < num_folds:
                raise ConfigError(
                    f"{path}.fold_index must be in [0, num_folds)"
                )
        elif self.num_folds is not None or self.fold_index is not None:
            raise ConfigError(
                f"{path}.num_folds and fold_index are only valid for kfold"
            )


@dataclass(slots=True)
class DataSourceMaterializationConfig:
    """Cache, acquisition, and verification policy for one data source."""

    cache_root: str = "./data"
    policy: str = "ensure"
    verification: str = "full"

    def validate(self, *, path: str) -> None:
        """Validate the source lifecycle policy."""

        cache_root = cast(object, self.cache_root)
        if not isinstance(cache_root, str) or not cache_root.strip():
            raise ConfigError(f"{path}.cache_root must be a non-empty string")
        policy = cast(object, self.policy)
        if not isinstance(policy, str) or policy not in {"require", "ensure"}:
            raise ConfigError(f"{path}.policy must be require or ensure")
        verification = cast(object, self.verification)
        if not isinstance(verification, str) or verification not in {
            "manifest",
            "full",
        }:
            raise ConfigError(f"{path}.verification must be manifest or full")


@dataclass(slots=True)
class ImageSourceConfig:
    """Canonical registered image-source selection."""

    name: str
    materialization: DataSourceMaterializationConfig
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, path: str) -> None:
        """Validate the source envelope without interpreting provider params."""

        name = cast(object, self.name)
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{path}.name must be a non-empty string")
        params = cast(object, self.params)
        if not isinstance(params, dict):
            raise ConfigError(f"{path}.params must be a mapping")
        self.materialization.validate(path=f"{path}.materialization")


@dataclass(slots=True)
class ImageRecipeConfig:
    """Image geometry and augmentation policy."""

    size: list[int]
    channels: int = 3
    normalize: bool = True
    random_horizontal_flip: bool = False

    @property
    def resolved_size(self) -> tuple[int, int]:
        """Return the already validated ``(height, width)`` pair."""

        return int(self.size[0]), int(self.size[1])

    def validate(self, *, path: str) -> None:
        """Validate image dimensions and shared transform options."""

        _resolved_size(self.size, path=f"{path}.size")
        channels = cast(object, self.channels)
        if (
            not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels not in {1, 3}
        ):
            raise ConfigError(f"{path}.channels must be 1 or 3")
        for name in ("normalize", "random_horizontal_flip"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{path}.{name} must be boolean")


@dataclass(slots=True)
class ImageDataBuilderConfig:
    """Complete single-source image recipe."""

    source: ImageSourceConfig
    image: ImageRecipeConfig
    partition: PartitionRecipeConfig = field(
        default_factory=PartitionRecipeConfig
    )
    loader: LoaderRecipeConfig = field(default_factory=LoaderRecipeConfig)

    def validate(self, *, path: str = "data.params") -> None:
        """Validate the complete recipe once before materialization."""

        self.source.validate(path=f"{path}.source")
        self.image.validate(path=f"{path}.image")
        self.partition.validate(path=f"{path}.partition")
        self.loader.validate(path=f"{path}.loader")


@dataclass(slots=True)
class SuperResolutionImageRecipeConfig:
    """Image geometry for generated or paired super-resolution."""

    high_resolution: list[int]
    low_resolution: list[int]
    channels: int = 3
    normalize: bool = True
    random_horizontal_flip: bool = False

    @property
    def resolved_high_resolution(self) -> tuple[int, int]:
        """Return validated high-resolution geometry."""

        return int(self.high_resolution[0]), int(self.high_resolution[1])

    @property
    def resolved_low_resolution(self) -> tuple[int, int]:
        """Return validated low-resolution geometry."""

        return int(self.low_resolution[0]), int(self.low_resolution[1])

    def validate(self, *, path: str) -> None:
        """Validate both resolutions and shared transform options."""

        _resolved_size(
            self.high_resolution,
            path=f"{path}.high_resolution",
        )
        _resolved_size(
            self.low_resolution,
            path=f"{path}.low_resolution",
        )
        channels = cast(object, self.channels)
        if (
            not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels not in {1, 3}
        ):
            raise ConfigError(f"{path}.channels must be 1 or 3")
        for name in ("normalize", "random_horizontal_flip"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{path}.{name} must be boolean")


@dataclass(slots=True)
class LowResolutionRecipeConfig:
    """Select generated or paired low-resolution observations."""

    kind: str = "bicubic"

    def validate(self, *, path: str) -> None:
        """Validate the super-resolution observation mode."""

        kind = cast(object, self.kind)
        if not isinstance(kind, str) or kind not in {"bicubic", "paired"}:
            raise ConfigError(f"{path}.kind must be bicubic or paired")


@dataclass(slots=True)
class SuperResolutionDataBuilderConfig:
    """Complete single-source super-resolution recipe."""

    source: ImageSourceConfig
    image: SuperResolutionImageRecipeConfig
    low_resolution: LowResolutionRecipeConfig = field(
        default_factory=LowResolutionRecipeConfig
    )
    partition: PartitionRecipeConfig = field(
        default_factory=PartitionRecipeConfig
    )
    loader: LoaderRecipeConfig = field(default_factory=LoaderRecipeConfig)

    def validate(self, *, path: str = "data.params") -> None:
        """Validate the complete recipe once before materialization."""

        self.source.validate(path=f"{path}.source")
        self.image.validate(path=f"{path}.image")
        self.low_resolution.validate(path=f"{path}.low_resolution")
        if self.low_resolution.kind == "paired":
            high_resolution = self.image.resolved_high_resolution
            low_resolution = self.image.resolved_low_resolution
            if any(
                high % low != 0
                for high, low in zip(
                    high_resolution,
                    low_resolution,
                    strict=True,
                )
            ):
                raise ConfigError(
                    f"{path}.image paired high_resolution must be an integer "
                    "multiple of low_resolution on each axis"
                )
        self.partition.validate(path=f"{path}.partition")
        self.loader.validate(path=f"{path}.loader")


@dataclass(slots=True)
class MultiResolutionSourceConfig:
    """One named source in a multi-resolution mixture."""

    id: str
    source: ImageSourceConfig
    sampling_weight: float | None = None

    def validate(self, *, path: str) -> None:
        """Validate one mixture source."""

        source_id = cast(object, self.id)
        if not isinstance(source_id, str) or not source_id.strip():
            raise ConfigError(f"{path}.id must be a non-empty string")
        sampling_weight = cast(object, self.sampling_weight)
        if sampling_weight is not None and (
            not isinstance(sampling_weight, (int, float))
            or isinstance(sampling_weight, bool)
            or sampling_weight <= 0
        ):
            raise ConfigError(f"{path}.sampling_weight must be positive")
        self.source.validate(path=f"{path}.source")


@dataclass(slots=True)
class MultiResolutionImageRecipeConfig:
    """Shared image options for a multi-resolution mixture."""

    channels: int = 3
    normalize: bool = True
    random_horizontal_flip: bool = False

    def validate(self, *, path: str) -> None:
        """Validate shared image transform options."""

        channels = cast(object, self.channels)
        if (
            not isinstance(channels, int)
            or isinstance(channels, bool)
            or channels not in {1, 3}
        ):
            raise ConfigError(f"{path}.channels must be 1 or 3")
        for name in ("normalize", "random_horizontal_flip"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{path}.{name} must be boolean")


@dataclass(slots=True)
class ResolutionBucketRecipeConfig:
    """One target resolution in a dynamic batching policy."""

    name: str
    height: int
    width: int

    def validate(self, *, path: str) -> None:
        """Validate one named positive resolution."""

        name = cast(object, self.name)
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{path}.name must be a non-empty string")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in cast(
                tuple[object, object],
                (self.height, self.width),
            )
        ):
            raise ConfigError(f"{path} dimensions must be positive")


@dataclass(slots=True)
class ResolutionBatchingRecipeConfig:
    """Bucket set and pixel-budget batching policy."""

    buckets: list[ResolutionBucketRecipeConfig]
    base_bucket: str
    dynamic_batch_size: bool = True

    def validate(self, *, path: str) -> None:
        """Validate bucket uniqueness and the selected base bucket."""

        if not self.buckets:
            raise ConfigError(f"{path}.buckets must not be empty")
        names: set[str] = set()
        for index, bucket in enumerate(self.buckets):
            bucket.validate(path=f"{path}.buckets[{index}]")
            if bucket.name in names:
                raise ConfigError(
                    f"{path}.buckets[{index}].name must be unique"
                )
            names.add(bucket.name)
        base_bucket = cast(object, self.base_bucket)
        if not isinstance(base_bucket, str) or base_bucket not in names:
            raise ConfigError(
                f"{path}.base_bucket must name a configured bucket"
            )
        dynamic_batch_size = cast(object, self.dynamic_batch_size)
        if not isinstance(dynamic_batch_size, bool):
            raise ConfigError(f"{path}.dynamic_batch_size must be boolean")


@dataclass(slots=True)
class MultiResolutionDataBuilderConfig:
    """Complete multi-source, multi-resolution image recipe."""

    sources: list[MultiResolutionSourceConfig]
    image: MultiResolutionImageRecipeConfig
    batching: ResolutionBatchingRecipeConfig
    partition: PartitionRecipeConfig = field(
        default_factory=PartitionRecipeConfig
    )
    loader: LoaderRecipeConfig = field(default_factory=LoaderRecipeConfig)

    def validate(self, *, path: str = "data.params") -> None:
        """Validate the complete recipe once before materialization."""

        if not self.sources:
            raise ConfigError(f"{path}.sources must not be empty")
        source_ids: set[str] = set()
        weights: list[float | None] = []
        for index, source in enumerate(self.sources):
            source.validate(path=f"{path}.sources[{index}]")
            if source.id in source_ids:
                raise ConfigError(
                    f"{path}.sources[{index}].id must be unique"
                )
            source_ids.add(source.id)
            weights.append(source.sampling_weight)
        if any(weight is None for weight in weights) and any(
            weight is not None for weight in weights
        ):
            raise ConfigError(
                f"{path}.sources[].sampling_weight must be set for every "
                "source or none"
            )
        self.image.validate(path=f"{path}.image")
        self.batching.validate(path=f"{path}.batching")
        self.partition.validate(path=f"{path}.partition")
        self.loader.validate(path=f"{path}.loader")


__all__ = [
    "DataSourceMaterializationConfig",
    "ImageDataBuilderConfig",
    "ImageRecipeConfig",
    "ImageSourceConfig",
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
