"""Built-in image-oriented data builder recipes."""

from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence, Sized
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from stochaflow.data.builder import DataBuilder, DataBuilderContext, DataLoaders
from stochaflow.data.partition import partition_datasets, validate_partition
from stochaflow.data.recipe_config import (
    ImageDataBuilderConfig,
    LoaderRecipeConfig,
    MultiResolutionDataBuilderConfig,
    SuperResolutionDataBuilderConfig,
)
from stochaflow.data.samplers import (
    MixtureBatchSampler,
    ResolutionBucketPolicy,
)
from stochaflow.data.sources import SourceDatasets, build_image_source
from stochaflow.data.transforms import (
    GeneratedSuperResolutionTransform,
    ImageRecipeDataset,
    ImageTransform,
    PairedSuperResolutionTransform,
    SuperResolutionRecipeDataset,
    collate_image_batch,
    collate_super_resolution_batch,
    extract_image,
    image_size,
    validate_size,
)
from stochaflow.utils.config import ConfigError, coerce_config_section
from stochaflow.utils.registry import REGISTRIES


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _validate_loader(config: LoaderRecipeConfig, *, path: str) -> None:
    batch_size = cast(object, config.batch_size)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ConfigError(f"{path}.batch_size must be positive")
    num_workers = cast(object, config.num_workers)
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
        if not isinstance(getattr(config, name), bool):
            raise ConfigError(f"{path}.{name} must be boolean")
    if config.persistent_workers and config.num_workers == 0:
        raise ConfigError(f"{path}.persistent_workers requires num_workers > 0")
    if config.prefetch_factor is not None:
        prefetch_factor = cast(object, config.prefetch_factor)
        if (
            not isinstance(prefetch_factor, int)
            or isinstance(prefetch_factor, bool)
            or prefetch_factor <= 0
        ):
            raise ConfigError(f"{path}.prefetch_factor must be positive")
        if config.num_workers == 0:
            raise ConfigError(f"{path}.prefetch_factor requires num_workers > 0")
    steps = config.steps_per_epoch
    if steps != "auto" and (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps <= 0
    ):
        raise ConfigError(f"{path}.steps_per_epoch must be positive or 'auto'")


def _validate_image_options(config: Any, *, path: str) -> None:
    if (
        not isinstance(config.channels, int)
        or isinstance(config.channels, bool)
        or config.channels not in {1, 3}
    ):
        raise ConfigError(f"{path}.channels must be 1 or 3")
    for name in ("normalize", "random_horizontal_flip"):
        if not isinstance(getattr(config, name), bool):
            raise ConfigError(f"{path}.{name} must be boolean")


def _loader_kwargs(config: LoaderRecipeConfig, *, seed: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "generator": torch.Generator().manual_seed(seed),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["worker_init_fn"] = _seed_worker
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


class _EpochRandomSampler(Sampler[int]):
    """Rebuild one deterministic shuffled index stream from seed and epoch."""

    def __init__(self, dataset: Dataset[Any], *, seed: int) -> None:
        self.size = len(cast(Sized, dataset))
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Select the shuffled index stream for one training epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch


def _build_loader(
    dataset: Dataset[Any] | None,
    config: LoaderRecipeConfig,
    *,
    training: bool,
    seed: int,
    collate_fn: Callable[[list[Any]], Any],
) -> DataLoader[Any] | None:
    if dataset is None:
        return None
    sampler = (
        _EpochRandomSampler(dataset, seed=seed)
        if training and config.shuffle
        else None
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=config.drop_last if training else False,
        collate_fn=collate_fn,
        **_loader_kwargs(config, seed=seed),
    )


def _steps_per_epoch(config: LoaderRecipeConfig) -> int | None:
    steps = config.steps_per_epoch
    return None if steps == "auto" else cast(int, steps)


def _image_transform(
    config: ImageDataBuilderConfig,
    *,
    role: str,
) -> ImageTransform:
    size = validate_size(config.image.size, path="data.params.image.size")
    return ImageTransform(
        size,
        role=role,
        channels=config.image.channels,
        normalize=config.image.normalize,
        random_horizontal_flip=config.image.random_horizontal_flip,
    )


def _wrap_image_partitions(
    partitions: Any,
    config: ImageDataBuilderConfig,
) -> tuple[Dataset[Any], Dataset[Any] | None, Dataset[Any] | None]:
    train = ImageRecipeDataset(
        partitions.train,
        _image_transform(config, role="train"),
    )
    validation = (
        ImageRecipeDataset(
            partitions.validation,
            _image_transform(config, role="eval"),
        )
        if partitions.validation is not None
        else None
    )
    test = (
        ImageRecipeDataset(
            partitions.test,
            _image_transform(config, role="eval"),
        )
        if partitions.test is not None
        else None
    )
    return train, validation, test


@REGISTRIES.data_builders.register("image")
class ImageDataBuilder(DataBuilder):
    """Standard single-source image training recipe."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            ImageDataBuilderConfig,
            coerce_config_section(
                ImageDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        _validate_loader(self.config.loader, path="data.params.loader")
        validate_partition(self.config.partition, path="data.params.partition")
        validate_size(self.config.image.size, path="data.params.image.size")
        _validate_image_options(self.config.image, path="data.params.image")

    def build(self) -> DataLoaders:
        source = build_image_source(
            self.config.source,
            partition_mode=self.config.partition.mode,
        )
        partitions = partition_datasets(
            source,
            self.config.partition,
            seed=self.context.seed,
        )
        train, validation, test = _wrap_image_partitions(partitions, self.config)
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                _build_loader(
                    train,
                    self.config.loader,
                    training=True,
                    seed=self.context.seed,
                    collate_fn=collate_image_batch,
                ),
            ),
            validation=_build_loader(
                validation,
                self.config.loader,
                training=False,
                seed=self.context.seed + 1,
                collate_fn=collate_image_batch,
            ),
            test=_build_loader(
                test,
                self.config.loader,
                training=False,
                seed=self.context.seed + 2,
                collate_fn=collate_image_batch,
            ),
            steps_per_epoch=_steps_per_epoch(self.config.loader),
        )


@REGISTRIES.data_builders.register("super_resolution")
class SuperResolutionDataBuilder(DataBuilder):
    """Paired or on-the-fly bicubic image super-resolution recipe."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            SuperResolutionDataBuilderConfig,
            coerce_config_section(
                SuperResolutionDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        _validate_loader(self.config.loader, path="data.params.loader")
        validate_partition(self.config.partition, path="data.params.partition")
        self.high_resolution = validate_size(
            self.config.image.high_resolution,
            path="data.params.image.high_resolution",
        )
        self.low_resolution = validate_size(
            self.config.image.low_resolution,
            path="data.params.image.low_resolution",
        )
        _validate_image_options(self.config.image, path="data.params.image")
        if self.config.low_resolution.kind not in {"bicubic", "paired"}:
            raise ConfigError(
                "data.params.low_resolution.kind must be bicubic or paired"
            )
        source_kind = self.config.source.get("kind")
        expected = (
            "paired_folders"
            if self.config.low_resolution.kind == "paired"
            else {"torchvision", "image_folder"}
        )
        if isinstance(expected, str):
            valid_source = source_kind == expected
        else:
            valid_source = source_kind in expected
        if not valid_source:
            raise ConfigError(
                "data.params.source.kind does not match low_resolution.kind"
            )

    def _transform(self, *, role: str) -> Any:
        common = {
            "role": role,
            "channels": self.config.image.channels,
            "normalize": self.config.image.normalize,
            "random_horizontal_flip": (
                self.config.image.random_horizontal_flip
            ),
        }
        if self.config.low_resolution.kind == "paired":
            return PairedSuperResolutionTransform(
                self.high_resolution,
                self.low_resolution,
                **common,
            )
        return GeneratedSuperResolutionTransform(
            self.high_resolution,
            self.low_resolution,
            **common,
        )

    def _wrap(self, dataset: Dataset[Any] | None, *, role: str) -> Dataset[Any] | None:
        if dataset is None:
            return None
        return SuperResolutionRecipeDataset(
            dataset,
            self._transform(role=role),
            paired=self.config.low_resolution.kind == "paired",
        )

    def build(self) -> DataLoaders:
        paired = self.config.low_resolution.kind == "paired"
        source = build_image_source(
            self.config.source,
            partition_mode=self.config.partition.mode,
            paired=paired,
        )
        partitions = partition_datasets(
            source,
            self.config.partition,
            seed=self.context.seed,
        )
        train = cast(Dataset[Any], self._wrap(partitions.train, role="train"))
        validation = self._wrap(partitions.validation, role="eval")
        test = self._wrap(partitions.test, role="eval")
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                _build_loader(
                    train,
                    self.config.loader,
                    training=True,
                    seed=self.context.seed,
                    collate_fn=collate_super_resolution_batch,
                ),
            ),
            validation=_build_loader(
                validation,
                self.config.loader,
                training=False,
                seed=self.context.seed + 1,
                collate_fn=collate_super_resolution_batch,
            ),
            test=_build_loader(
                test,
                self.config.loader,
                training=False,
                seed=self.context.seed + 2,
                collate_fn=collate_super_resolution_batch,
            ),
            steps_per_epoch=_steps_per_epoch(self.config.loader),
        )


class _SourceConcatDataset(Dataset[Any]):
    def __init__(self, datasets: Sequence[tuple[str, Dataset[Any]]]) -> None:
        if not datasets:
            raise ValueError("multi-resolution sources must not be empty")
        self.datasets = tuple(datasets)
        self.ends: list[int] = []
        self.source_ids: list[str] = []
        total = 0
        for source_id, dataset in self.datasets:
            dataset_size = len(cast(Sized, dataset))
            total += dataset_size
            self.ends.append(total)
            self.source_ids.extend([source_id] * dataset_size)
        if total <= 0:
            raise ValueError("multi-resolution sources contain no samples")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        source_index = bisect_right(self.ends, index)
        start = 0 if source_index == 0 else self.ends[source_index - 1]
        return self.datasets[source_index][1][index - start]


def _source_id(dataset: Dataset[Any], index: int) -> str:
    if isinstance(dataset, Subset):
        return _source_id(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, _SourceConcatDataset):
        return dataset.source_ids[index]
    raise TypeError("multi-resolution dataset lost private source metadata")


class _MultiResolutionDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        dataset: Dataset[Any],
        policy: ResolutionBucketPolicy,
        *,
        role: str,
        channels: int,
        normalize: bool,
        random_horizontal_flip: bool,
    ) -> None:
        self.dataset = dataset
        dataset_size = len(cast(Sized, dataset))
        self.source_ids = tuple(
            _source_id(dataset, index) for index in range(dataset_size)
        )
        self.bucket_ids: list[str] = []
        for index in range(dataset_size):
            width, height = image_size(extract_image(dataset[index]))
            self.bucket_ids.append(policy.select(width, height).name)
        self.transforms = {
            bucket.name: ImageTransform(
                (bucket.height, bucket.width),
                role=role,
                channels=channels,
                normalize=normalize,
                random_horizontal_flip=random_horizontal_flip,
            )
            for bucket in policy.buckets
        }

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image = extract_image(self.dataset[index])
        return self.transforms[self.bucket_ids[index]](image), {}


def _combine_sources(
    sources: Sequence[tuple[str, SourceDatasets]],
    role: str,
) -> Dataset[Any] | None:
    selected = [(source_id, getattr(source, role)) for source_id, source in sources]
    present = [(source_id, dataset) for source_id, dataset in selected if dataset is not None]
    if not present:
        return None
    if len(present) != len(selected):
        raise ValueError(
            f"multi-resolution source role '{role}' must be present for every source or none"
        )
    return _SourceConcatDataset(cast(list[tuple[str, Dataset[Any]]], present))


@REGISTRIES.data_builders.register("multi_resolution_image")
class MultiResolutionImageDataBuilder(DataBuilder):
    """Multi-source image recipe with bucket-homogeneous dynamic batches."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            MultiResolutionDataBuilderConfig,
            coerce_config_section(
                MultiResolutionDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        _validate_loader(self.config.loader, path="data.params.loader")
        validate_partition(self.config.partition, path="data.params.partition")
        if not self.config.sources:
            raise ConfigError("data.params.sources must not be empty")
        _validate_image_options(self.config.image, path="data.params.image")
        source_ids: set[str] = set()
        weights: list[float | None] = []
        for index, source in enumerate(self.config.sources):
            source_id = cast(object, source.id)
            if (
                not isinstance(source_id, str)
                or not source_id.strip()
                or source_id in source_ids
            ):
                raise ConfigError(
                    f"data.params.sources[{index}].id must be non-empty and unique"
                )
            source_ids.add(source.id)
            if source.sampling_weight is not None:
                sampling_weight = cast(object, source.sampling_weight)
                if (
                    not isinstance(sampling_weight, (int, float))
                    or isinstance(sampling_weight, bool)
                    or sampling_weight <= 0
                ):
                    raise ConfigError(
                        f"data.params.sources[{index}].sampling_weight must be positive"
                    )
            weights.append(source.sampling_weight)
        if any(weight is None for weight in weights) and any(
            weight is not None for weight in weights
        ):
            raise ConfigError(
                "sampling_weight must be set for every source or none"
            )
        if not self.config.batching.buckets:
            raise ConfigError("data.params.batching.buckets must not be empty")
        bucket_names: set[str] = set()
        for index, bucket in enumerate(self.config.batching.buckets):
            bucket_name = cast(object, bucket.name)
            if (
                not isinstance(bucket_name, str)
                or not bucket_name.strip()
                or bucket_name in bucket_names
            ):
                raise ConfigError(
                    f"data.params.batching.buckets[{index}].name must be unique"
                )
            bucket_names.add(bucket.name)
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in cast(
                    tuple[object, object], (bucket.height, bucket.width)
                )
            ):
                raise ConfigError(
                    f"data.params.batching.buckets[{index}] dimensions must be positive"
                )
        base_bucket = cast(object, self.config.batching.base_bucket)
        if not isinstance(base_bucket, str) or base_bucket not in bucket_names:
            raise ConfigError(
                "data.params.batching.base_bucket must name a configured bucket"
            )
        if not isinstance(
            cast(object, self.config.batching.dynamic_batch_size), bool
        ):
            raise ConfigError(
                "data.params.batching.dynamic_batch_size must be boolean"
            )

    def _weights(self) -> dict[str, float] | None:
        if all(source.sampling_weight is None for source in self.config.sources):
            return None
        return {
            source.id: cast(float, source.sampling_weight)
            for source in self.config.sources
        }

    def _wrap(
        self,
        dataset: Dataset[Any] | None,
        policy: ResolutionBucketPolicy,
        *,
        role: str,
    ) -> _MultiResolutionDataset | None:
        if dataset is None:
            return None
        return _MultiResolutionDataset(
            dataset,
            policy,
            role=role,
            channels=self.config.image.channels,
            normalize=self.config.image.normalize,
            random_horizontal_flip=(
                self.config.image.random_horizontal_flip
            ),
        )

    def _loader(
        self,
        dataset: _MultiResolutionDataset | None,
        policy: ResolutionBucketPolicy,
        *,
        training: bool,
        seed: int,
    ) -> DataLoader[Any] | None:
        if dataset is None:
            return None
        sampler = MixtureBatchSampler(
            dataset,
            policy,
            base_batch_size=self.config.loader.batch_size,
            drop_last=self.config.loader.drop_last if training else False,
            shuffle=self.config.loader.shuffle if training else False,
            seed=seed,
            source_weights=self._weights() if training else None,
            steps_per_epoch=(
                self.config.loader.steps_per_epoch if training else "auto"
            ),
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_image_batch,
            **_loader_kwargs(self.config.loader, seed=seed),
        )

    def build(self) -> DataLoaders:
        sources = [
            (
                source.id,
                build_image_source(
                    source.source,
                    partition_mode=self.config.partition.mode,
                    path=f"data.params.sources[{index}].source",
                ),
            )
            for index, source in enumerate(self.config.sources)
        ]
        combined = SourceDatasets(
            train=cast(Dataset[Any], _combine_sources(sources, "train")),
            validation=(
                _combine_sources(sources, "validation")
                if self.config.partition.mode == "official"
                else None
            ),
            test=_combine_sources(sources, "test"),
        )
        partitions = partition_datasets(
            combined,
            self.config.partition,
            seed=self.context.seed,
        )
        policy = ResolutionBucketPolicy(
            self.config.batching.buckets,
            base_bucket=self.config.batching.base_bucket,
            dynamic_batch_size=self.config.batching.dynamic_batch_size,
        )
        train = self._wrap(partitions.train, policy, role="train")
        assert train is not None
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                self._loader(
                    train,
                    policy,
                    training=True,
                    seed=self.context.seed,
                ),
            ),
            validation=self._loader(
                self._wrap(partitions.validation, policy, role="eval"),
                policy,
                training=False,
                seed=self.context.seed + 1,
            ),
            test=self._loader(
                self._wrap(partitions.test, policy, role="eval"),
                policy,
                training=False,
                seed=self.context.seed + 2,
            ),
            steps_per_epoch=_steps_per_epoch(self.config.loader),
        )


__all__ = [
    "ImageDataBuilder",
    "MultiResolutionImageDataBuilder",
    "SuperResolutionDataBuilder",
]
