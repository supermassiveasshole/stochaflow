"""Built-in map-style and multi-resolution image data pipelines."""

from __future__ import annotations

import random
from collections.abc import Sized
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stochaflow.data.contracts import DatasetFactory, DatasetFactoryContext
from stochaflow.data.datasets import BucketedVisionDataset
from stochaflow.data.pipeline import (
    DataBundle,
    DataPipeline,
    DataPipelineContext,
    SplitData,
)
from stochaflow.data.samplers import FixedBatchSampler, MixtureBatchSampler
from stochaflow.data.splits import (
    ConfiguredDatasetFactory,
    DataPartitions,
    DatasetMaterializer,
    SplitContext,
    build_data_partitions,
)
from stochaflow.utils.config import (
    ConfigError,
    DataSplitConfig,
    DataloaderConfig,
    DatasetConfig,
    MapDataPipelineConfig,
    MultiResolutionImageDataPipelineConfig,
    coerce_config_section,
)
from stochaflow.utils.registry import REGISTRIES


def _seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _validate_dataloader(config: DataloaderConfig, *, path: str) -> None:
    if config.batch_size <= 0:
        raise ConfigError(f"{path}.batch_size must be positive")
    if config.num_workers < 0:
        raise ConfigError(f"{path}.num_workers must be non-negative")
    if config.persistent_workers and config.num_workers == 0:
        raise ConfigError(f"{path}.persistent_workers requires num_workers > 0")
    if config.prefetch_factor is not None:
        if config.prefetch_factor <= 0:
            raise ConfigError(f"{path}.prefetch_factor must be positive")
        if config.num_workers == 0:
            raise ConfigError(f"{path}.prefetch_factor requires num_workers > 0")
    steps = config.steps_per_epoch
    if steps != "auto" and (
        not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0
    ):
        raise ConfigError(f"{path}.steps_per_epoch must be positive or 'auto'")


def _validate_splits(config: DataSplitConfig, *, path: str) -> None:
    if config.mode not in {"none", "official", "random_holdout", "kfold"}:
        raise ConfigError(
            f"{path}.mode must be none, official, random_holdout, or kfold"
        )
    requested = config.validation_size
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, (int, float)):
            raise ConfigError(f"{path}.validation_size must be numeric or null")
        if isinstance(requested, float) and not 0.0 < requested < 1.0:
            raise ConfigError(
                f"{path}.validation_size must be between 0 and 1 when a float"
            )
        if isinstance(requested, int) and requested <= 0:
            raise ConfigError(f"{path}.validation_size must be positive")
    if config.mode == "random_holdout" and requested is None:
        raise ConfigError(f"{path}.validation_size is required for random_holdout")
    if config.mode == "kfold":
        if config.num_folds is None or config.num_folds < 2:
            raise ConfigError(f"{path}.num_folds must be at least 2 for kfold")
        if config.fold_index is not None and not (
            0 <= config.fold_index < config.num_folds
        ):
            raise ConfigError(f"{path}.fold_index must be in [0, num_folds)")


def _validate_datasets(
    datasets: list[DatasetConfig],
    *,
    path: str,
    allow_sampling_weight: bool,
) -> None:
    if not datasets:
        raise ConfigError(f"{path} must declare at least one dataset")
    source_ids: set[str] = set()
    weights: list[float | None] = []
    for index, dataset in enumerate(datasets):
        item_path = f"{path}[{index}]"
        if not dataset.id:
            raise ConfigError(f"{item_path}.id must be non-empty")
        if dataset.id in source_ids:
            raise ConfigError(f"duplicate data source id '{dataset.id}'")
        source_ids.add(dataset.id)
        if not dataset.factory:
            raise ConfigError(f"{item_path}.factory must be non-empty")
        if not dataset.splits.train:
            raise ConfigError(f"{item_path}.splits.train must be non-empty")
        weight = dataset.sampling_weight
        if weight is not None:
            if not allow_sampling_weight:
                raise ConfigError(f"{item_path}.sampling_weight is not supported")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ConfigError(f"{item_path}.sampling_weight must be positive")
            if float(weight) <= 0:
                raise ConfigError(f"{item_path}.sampling_weight must be positive")
        weights.append(weight)
    if any(weight is None for weight in weights) and any(
        weight is not None for weight in weights
    ):
        raise ConfigError(
            f"{path} sampling_weight must be set for every source or none"
        )


def _loader_kwargs(config: DataloaderConfig, *, seed: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "generator": torch.Generator().manual_seed(seed),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["worker_init_fn"] = _seed_dataloader_worker
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def _configured_factories(
    datasets: list[DatasetConfig],
) -> tuple[ConfiguredDatasetFactory, ...]:
    result: list[ConfiguredDatasetFactory] = []
    for source in datasets:
        factory = REGISTRIES.dataset_factories.create(
            source.factory,
            DatasetFactoryContext(
                source_id=source.id,
                params=dict(source.params),
            ),
        )
        if not isinstance(factory, DatasetFactory):
            raise TypeError(
                f"registered dataset factory '{source.factory}' did not produce "
                "DatasetFactory"
            )
        result.append(ConfiguredDatasetFactory(config=source, factory=factory))
    return tuple(result)


def _partitions(
    datasets: list[DatasetConfig],
    splits: DataSplitConfig,
    *,
    seed: int,
) -> list[DataPartitions]:
    materializer = DatasetMaterializer(_configured_factories(datasets), seed=seed)
    return build_data_partitions(
        SplitContext(config=splits, datasets=materializer, seed=seed)
    )


class _MapLoaderFactory:
    def __init__(self, config: DataloaderConfig, *, seed: int) -> None:
        self.config = config
        self.seed = seed

    def build(
        self,
        name: str,
        dataset: Dataset[Any] | None,
        *,
        training: bool,
        seed_offset: int,
    ) -> SplitData | None:
        if dataset is None:
            return None
        seed = self.seed + seed_offset
        sampler = FixedBatchSampler(
            cast(Sized, dataset),
            batch_size=self.config.batch_size,
            drop_last=self.config.drop_last if training else False,
            shuffle=self.config.shuffle if training else False,
            seed=seed,
            steps_per_epoch=self.config.steps_per_epoch if training else "auto",
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            **_loader_kwargs(self.config, seed=seed),
        )
        return SplitData(
            name=name,
            dataset=dataset,
            dataloader=loader,
            num_samples=len(cast(Sized, dataset)),
            num_batches=len(loader),
        )


@REGISTRIES.data_pipelines.register("map")
class MapDataPipeline(DataPipeline):
    """One map-style dataset with fixed batches and common split policies."""

    def __init__(self, context: DataPipelineContext) -> None:
        super().__init__(context)
        self.config = cast(
            MapDataPipelineConfig,
            coerce_config_section(
                MapDataPipelineConfig,
                context.params,
                "data.params",
            ),
        )
        _validate_datasets(
            [self.config.dataset],
            path="data.params.dataset",
            allow_sampling_weight=False,
        )
        _validate_dataloader(
            self.config.dataloader,
            path="data.params.dataloader",
        )
        _validate_splits(self.config.splits, path="data.params.splits")

    def build(self) -> list[DataBundle]:
        """Build one bundle or the requested deterministic folds."""

        loader_factory = _MapLoaderFactory(
            self.config.dataloader,
            seed=self.context.seed,
        )
        bundles: list[DataBundle] = []
        for partition in _partitions(
            [self.config.dataset],
            self.config.splits,
            seed=self.context.seed,
        ):
            seed_offset = partition.fold_index or 0
            train = loader_factory.build(
                "train",
                partition.train,
                training=True,
                seed_offset=seed_offset,
            )
            assert train is not None
            bundles.append(
                DataBundle(
                    train=train,
                    valid=loader_factory.build(
                        "valid",
                        partition.valid,
                        training=False,
                        seed_offset=seed_offset,
                    ),
                    test=loader_factory.build(
                        "test",
                        partition.test,
                        training=False,
                        seed_offset=seed_offset,
                    ),
                    fold_index=partition.fold_index,
                    num_folds=partition.num_folds,
                )
            )
        return bundles


class _ImageLoaderFactory:
    def __init__(
        self,
        config: MultiResolutionImageDataPipelineConfig,
        *,
        seed: int,
    ) -> None:
        from stochaflow.data.contracts import ResolutionBucketPolicy

        self.config = config
        self.seed = seed
        self.bucket_policy = ResolutionBucketPolicy(
            config.batching.buckets,
            base_bucket=config.batching.base_bucket,
            dynamic_batch_size=config.batching.dynamic_batch_size,
        )

    def _source_weights(self) -> dict[str, float] | None:
        if all(source.sampling_weight is None for source in self.config.datasets):
            return None
        return {
            source.id: float(source.sampling_weight)
            for source in self.config.datasets
            if source.sampling_weight is not None
        }

    def build(
        self,
        name: str,
        dataset: Dataset[Any] | None,
        *,
        training: bool,
        seed_offset: int,
    ) -> SplitData | None:
        if dataset is None:
            return None
        prepared = BucketedVisionDataset(
            dataset,
            self.bucket_policy,
            self.config.image,
            role="train" if training else "eval",
        )
        seed = self.seed + seed_offset
        sampler = MixtureBatchSampler(
            prepared,
            self.bucket_policy,
            base_batch_size=self.config.dataloader.batch_size,
            drop_last=self.config.dataloader.drop_last if training else False,
            shuffle=self.config.dataloader.shuffle if training else False,
            seed=seed,
            source_weights=self._source_weights() if training else None,
            steps_per_epoch=(
                self.config.dataloader.steps_per_epoch if training else "auto"
            ),
        )
        loader = DataLoader(
            prepared,
            batch_sampler=sampler,
            **_loader_kwargs(self.config.dataloader, seed=seed),
        )
        return SplitData(
            name=name,
            dataset=prepared,
            dataloader=loader,
            num_samples=len(prepared),
            num_batches=len(loader),
        )


@REGISTRIES.data_pipelines.register("multi_resolution_image")
class MultiResolutionImageDataPipeline(DataPipeline):
    """Multi-source image mixture with resolution-homogeneous batches."""

    def __init__(self, context: DataPipelineContext) -> None:
        super().__init__(context)
        self.config = cast(
            MultiResolutionImageDataPipelineConfig,
            coerce_config_section(
                MultiResolutionImageDataPipelineConfig,
                context.params,
                "data.params",
            ),
        )
        _validate_datasets(
            self.config.datasets,
            path="data.params.datasets",
            allow_sampling_weight=True,
        )
        _validate_dataloader(
            self.config.dataloader,
            path="data.params.dataloader",
        )
        _validate_splits(self.config.splits, path="data.params.splits")
        if self.config.image.channels not in {1, 3}:
            raise ConfigError("data.params.image.channels must be 1 or 3")
        if not self.config.batching.buckets:
            raise ConfigError("data.params.batching.buckets must not be empty")
        bucket_names: set[str] = set()
        for index, bucket in enumerate(self.config.batching.buckets):
            if not bucket.name or bucket.name in bucket_names:
                raise ConfigError(
                    f"data.params.batching.buckets[{index}].name must be unique"
                )
            bucket_names.add(bucket.name)
            if bucket.height <= 0 or bucket.width <= 0:
                raise ConfigError(
                    f"data.params.batching.buckets[{index}] dimensions must be positive"
                )
        if self.config.batching.base_bucket not in bucket_names:
            raise ConfigError(
                "data.params.batching.base_bucket must name a configured bucket"
            )

    def build(self) -> list[DataBundle]:
        """Build bucket-aware bundles after materialization and splitting."""

        loader_factory = _ImageLoaderFactory(
            self.config,
            seed=self.context.seed,
        )
        bundles: list[DataBundle] = []
        for partition in _partitions(
            self.config.datasets,
            self.config.splits,
            seed=self.context.seed,
        ):
            seed_offset = partition.fold_index or 0
            train = loader_factory.build(
                "train",
                partition.train,
                training=True,
                seed_offset=seed_offset,
            )
            assert train is not None
            bundles.append(
                DataBundle(
                    train=train,
                    valid=loader_factory.build(
                        "valid",
                        partition.valid,
                        training=False,
                        seed_offset=seed_offset,
                    ),
                    test=loader_factory.build(
                        "test",
                        partition.test,
                        training=False,
                        seed_offset=seed_offset,
                    ),
                    fold_index=partition.fold_index,
                    num_folds=partition.num_folds,
                )
            )
        return bundles


__all__ = ["MapDataPipeline", "MultiResolutionImageDataPipeline"]
