"""Class-based dataset construction, splitting, mixing, and loading."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stochaflow.data.collation import ImageBatchCollator
from stochaflow.data.contracts import (
    DatasetFactory,
    DatasetFactoryContext,
    ResolutionBucketPolicy,
)
from stochaflow.data.samplers import BucketedDataset, MixtureBatchSampler
from stochaflow.data.splits import (
    ConfiguredDatasetFactory,
    DataPartitions,
    DatasetMaterializer,
    SplitContext,
    SplitStrategy,
)
from stochaflow.utils.config import DataConfig, DataloaderConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog


@dataclass(frozen=True, slots=True)
class SplitData:
    """One named dataset partition paired with its bucket-aware loader."""

    name: str
    dataset: Dataset[Any]
    dataloader: DataLoader[Any]


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Datasets and loaders required for one independent training run."""

    train: SplitData
    valid: SplitData | None = None
    test: SplitData | None = None
    fold_index: int | None = None
    num_folds: int | None = None


def _seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class DataLoaderFactory:
    """Build reproducible DataLoaders around bucket-aware batch samplers."""

    def __init__(
        self,
        config: DataloaderConfig,
        bucket_policy: ResolutionBucketPolicy,
        *,
        seed: int,
        steps_per_epoch: int | str,
        source_weights: dict[str, float] | None,
    ) -> None:
        self.config = config
        self.bucket_policy = bucket_policy
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.source_weights = source_weights

    def _build_loader(
        self,
        dataset: Dataset[Any],
        *,
        training: bool,
        seed: int,
    ) -> DataLoader[Any]:
        if not hasattr(dataset, "bucket_ids") or not hasattr(dataset, "source_ids"):
            raise TypeError(
                "bucket-aware dataloaders require dataset bucket_ids and source_ids"
            )
        batch_sampler = MixtureBatchSampler(
            cast(BucketedDataset, dataset),
            self.bucket_policy,
            base_batch_size=self.config.batch_size,
            drop_last=self.config.drop_last if training else False,
            shuffle=self.config.shuffle if training else False,
            seed=seed,
            source_weights=self.source_weights if training else None,
            steps_per_epoch=self.steps_per_epoch if training else "auto",
        )
        kwargs: dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "collate_fn": ImageBatchCollator(),
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "generator": torch.Generator().manual_seed(seed),
        }
        if self.config.num_workers > 0:
            kwargs["persistent_workers"] = self.config.persistent_workers
            kwargs["worker_init_fn"] = _seed_dataloader_worker
            if self.config.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def build_train(self, dataset: Dataset[Any], *, seed_offset: int = 0) -> SplitData:
        """Build a shuffled, optionally weighted training loader."""

        return SplitData(
            name="train",
            dataset=dataset,
            dataloader=self._build_loader(
                dataset,
                training=True,
                seed=self.seed + seed_offset,
            ),
        )

    def build_evaluation(
        self,
        name: str,
        dataset: Dataset[Any] | None,
        *,
        seed_offset: int = 0,
    ) -> SplitData | None:
        """Build a deterministic, non-dropping evaluation loader."""

        if dataset is None:
            return None
        return SplitData(
            name=name,
            dataset=dataset,
            dataloader=self._build_loader(
                dataset,
                training=False,
                seed=self.seed + seed_offset,
            ),
        )


class DataPipeline:
    """Orchestrate registered factories, global splits, and data loaders."""

    def __init__(
        self,
        config: DataConfig,
        *,
        seed: int,
        registries: RegistryCatalog = REGISTRIES,
    ) -> None:
        self.config = config
        self.seed = seed
        self.registries = registries
        if not config.datasets:
            raise ValueError("data pipeline requires at least one dataset")
        source_ids = [source.id for source in config.datasets]
        if any(not source_id for source_id in source_ids):
            raise ValueError("data source ids must be non-empty")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("data source ids must be unique")
        self.bucket_policy = ResolutionBucketPolicy(
            config.batching.buckets,
            sample_bucket=config.batching.sample_bucket,
            dynamic_batch_size=config.batching.dynamic_batch_size,
        )

    def _factories(self) -> tuple[ConfiguredDatasetFactory, ...]:
        self.registries.load_modules(self.config.modules)
        sources: list[ConfiguredDatasetFactory] = []
        for source_config in self.config.datasets:
            context = DatasetFactoryContext(
                source_id=source_config.id,
                params=dict(source_config.params),
                image=self.config.image,
                buckets=self.bucket_policy,
            )
            factory = self.registries.dataset_factories.create(
                source_config.factory,
                context,
            )
            if not isinstance(factory, DatasetFactory):
                raise TypeError(
                    f"registered dataset factory '{source_config.factory}' did "
                    "not produce DatasetFactory"
                )
            sources.append(
                ConfiguredDatasetFactory(
                    config=source_config,
                    factory=factory,
                )
            )
        return tuple(sources)

    def _source_weights(self) -> dict[str, float] | None:
        weights = [source.sampling_weight for source in self.config.datasets]
        if all(weight is None for weight in weights):
            return None
        if any(weight is None for weight in weights):
            raise ValueError(
                "sampling_weight must be specified for every source or omitted "
                "for every source"
            )
        return {
            source.id: float(source.sampling_weight)
            for source in self.config.datasets
            if source.sampling_weight is not None
        }

    def _partitions(self) -> list[DataPartitions]:
        materializer = DatasetMaterializer(self._factories(), seed=self.seed)
        strategy = self.registries.split_strategies.create(
            self.config.splits.mode
        )
        if not isinstance(strategy, SplitStrategy):
            raise TypeError(
                f"registered split strategy '{self.config.splits.mode}' did not "
                "produce SplitStrategy"
            )
        return strategy.split(
            SplitContext(
                config=self.config.splits,
                datasets=materializer,
                seed=self.seed,
            )
        )

    def build(self) -> list[DataBundle]:
        """Build one data bundle, or one per requested K-fold."""

        loader_factory = DataLoaderFactory(
            self.config.dataloader,
            self.bucket_policy,
            seed=self.seed,
            steps_per_epoch=self.config.batching.steps_per_epoch,
            source_weights=self._source_weights(),
        )
        bundles: list[DataBundle] = []
        for partitions in self._partitions():
            seed_offset = partitions.fold_index or 0
            bundles.append(
                DataBundle(
                    train=loader_factory.build_train(
                        partitions.train,
                        seed_offset=seed_offset,
                    ),
                    valid=loader_factory.build_evaluation(
                        "valid",
                        partitions.valid,
                        seed_offset=seed_offset,
                    ),
                    test=loader_factory.build_evaluation(
                        "test",
                        partitions.test,
                        seed_offset=seed_offset,
                    ),
                    fold_index=partitions.fold_index,
                    num_folds=partitions.num_folds,
                )
            )
        return bundles


__all__ = ["DataBundle", "DataLoaderFactory", "DataPipeline", "SplitData"]
