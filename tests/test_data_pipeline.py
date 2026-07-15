"""Tests for class-based multi-source data orchestration."""

from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

from stochaflow.data import (
    DataPipeline,
    DatasetBuildRequest,
    DatasetFactory,
    DatasetSelection,
    DatasetView,
    MixtureBatchSampler,
    ResolutionBucketPolicy,
)
from stochaflow.diffusion import DDPM, DDPMEpsilonObjective, LinearBetaSchedule
from stochaflow.training import Trainer
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.config import (
    DataBatchingConfig,
    DataConfig,
    DataSplitConfig,
    DataloaderConfig,
    DatasetConfig,
    DatasetSplitMapConfig,
    ImageDataConfig,
    ResolutionBucketConfig,
)
from stochaflow.utils.registry import REGISTRIES


def _length(value: object) -> int:
    return len(cast(Sized, value))


class SyntheticImageDataset(Dataset):
    def __init__(
        self,
        bucket_ids: Sequence[str],
        policy: ResolutionBucketPolicy,
        *,
        offset: int,
        retain_payload: bool,
    ) -> None:
        self.bucket_ids = tuple(bucket_ids)
        self.policy = policy
        self.offset = offset
        self.retain_payload = retain_payload

    def __len__(self) -> int:
        return len(self.bucket_ids)

    def __getitem__(self, index: int):
        bucket = self.policy.resolve(self.bucket_ids[index])
        image = torch.full(
            (3, bucket.height, bucket.width),
            float(index + self.offset),
        )
        return (image, index) if self.retain_payload else image


@REGISTRIES.dataset_factories.register("synthetic_images")
class SyntheticDatasetFactory(DatasetFactory):
    def build(self, request: DatasetBuildRequest) -> DatasetView:
        sizes = self.context.params["sizes"]
        size = int(sizes[request.native_split])
        dimensions = tuple(self.context.params.get("dimensions", (32, 32)))
        bucket = self.context.buckets.select(*dimensions)
        bucket_ids = (bucket.name,) * size
        role_offset = 100 if request.role == "eval" else 0
        split_offset = {"train": 0, "val": 1000, "test": 2000}.get(
            request.native_split,
            3000,
        )
        return DatasetView(
            source_id=self.context.source_id,
            dataset=SyntheticImageDataset(
                bucket_ids,
                self.context.buckets,
                offset=role_offset + split_offset,
                retain_payload=bool(
                    self.context.params.get("retain_payload", True)
                ),
            ),
            sample_keys=tuple(
                f"{request.native_split}:{index}" for index in range(size)
            ),
            bucket_ids=bucket_ids,
        )


def _source(
    source_id: str,
    *,
    train_size: int,
    dimensions: tuple[int, int],
    sampling_weight: float | None = None,
    retain_payload: bool = True,
) -> DatasetConfig:
    return DatasetConfig(
        id=source_id,
        factory="synthetic_images",
        sampling_weight=sampling_weight,
        params={
            "sizes": {"train": train_size, "val": 2, "test": 1},
            "dimensions": dimensions,
            "retain_payload": retain_payload,
        },
        splits=DatasetSplitMapConfig(
            train="train",
            validation="val",
            test="test",
        ),
    )


def _data_config(
    splits: DataSplitConfig,
    *,
    sources: list[DatasetConfig] | None = None,
    steps_per_epoch: int | str = "auto",
    drop_last: bool = False,
) -> DataConfig:
    return DataConfig(
        datasets=sources
        or [
            _source("small", train_size=6, dimensions=(32, 32)),
            _source("large", train_size=4, dimensions=(64, 64)),
        ],
        image=ImageDataConfig(channels=3, normalize=True),
        batching=DataBatchingConfig(
            buckets=[
                ResolutionBucketConfig("square_32", 32, 32),
                ResolutionBucketConfig("square_64", 64, 64),
            ],
            sample_bucket="square_64",
            dynamic_batch_size=True,
            steps_per_epoch=steps_per_epoch,
        ),
        dataloader=DataloaderConfig(
            batch_size=2,
            num_workers=0,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=False,
            persistent_workers=False,
        ),
        splits=splits,
    )


def test_random_holdout_is_global_deterministic_and_builds_test_union() -> None:
    config = _data_config(
        DataSplitConfig(mode="random_holdout", validation_size=3)
    )

    first = DataPipeline(config, seed=7).build()[0]
    second = DataPipeline(config, seed=7).build()[0]

    assert first.valid is not None
    assert second.valid is not None
    assert first.test is not None
    assert isinstance(first.train.dataset, DatasetSelection)
    assert isinstance(first.valid.dataset, DatasetSelection)
    assert isinstance(second.train.dataset, DatasetSelection)
    assert isinstance(second.valid.dataset, DatasetSelection)
    assert first.train.dataset.indices == second.train.dataset.indices
    assert first.valid.dataset.indices == second.valid.dataset.indices
    assert len(first.train.dataset) == 7
    assert len(first.valid.dataset) == 3
    assert _length(first.test.dataset) == 2
    assert {source for source, _ in first.train.dataset.sample_keys} <= {
        "small",
        "large",
    }


def test_official_mode_uses_each_sources_native_split_mapping() -> None:
    bundle = DataPipeline(
        _data_config(DataSplitConfig(mode="official")),
        seed=3,
    ).build()[0]

    assert _length(bundle.train.dataset) == 10
    assert bundle.valid is not None
    assert _length(bundle.valid.dataset) == 4
    assert bundle.test is not None
    assert _length(bundle.test.dataset) == 2


def test_train_only_mode_keeps_optional_test_union() -> None:
    bundle = DataPipeline(
        _data_config(DataSplitConfig(mode="none")),
        seed=11,
    ).build()[0]

    assert _length(bundle.train.dataset) == 10
    assert bundle.valid is None
    assert bundle.test is not None
    assert _length(bundle.test.dataset) == 2


def test_kfold_validation_indices_cover_global_union_once() -> None:
    bundles = DataPipeline(
        _data_config(DataSplitConfig(mode="kfold", num_folds=3)),
        seed=13,
    ).build()

    validation_indices: list[int] = []
    fold_sizes: list[int] = []
    for fold_index, bundle in enumerate(bundles):
        assert bundle.fold_index == fold_index
        assert bundle.num_folds == 3
        assert bundle.valid is not None
        assert isinstance(bundle.valid.dataset, DatasetSelection)
        validation_indices.extend(bundle.valid.dataset.indices)
        fold_sizes.append(len(bundle.valid.dataset))

    assert sorted(validation_indices) == list(range(10))
    assert len(set(validation_indices)) == 10
    assert max(fold_sizes) - min(fold_sizes) <= 1


def test_bucket_sampler_keeps_shapes_homogeneous_and_scales_batch_size() -> None:
    bundle = DataPipeline(
        _data_config(DataSplitConfig(mode="none")),
        seed=17,
    ).build()[0]
    sampler = bundle.train.dataloader.batch_sampler
    assert isinstance(sampler, MixtureBatchSampler)
    train_dataset = cast(Any, bundle.train.dataset)

    seen_sizes: dict[str, set[int]] = {"square_32": set(), "square_64": set()}
    for indices in sampler:
        bucket_ids = {train_dataset.bucket_ids[index] for index in indices}
        assert len(bucket_ids) == 1
        bucket_id = next(iter(bucket_ids))
        seen_sizes[bucket_id].add(len(indices))

    assert seen_sizes["square_32"] == {6}
    assert seen_sizes["square_64"] == {2}
    batches = list(bundle.train.dataloader)
    assert {tuple(batch[0].shape[-2:]) for batch in batches} == {(32, 32), (64, 64)}


def test_loader_collates_tensor_and_tuple_sources_into_image_batch() -> None:
    sources = [
        _source(
            "with_payload",
            train_size=1,
            dimensions=(64, 64),
            retain_payload=True,
        ),
        _source(
            "image_only",
            train_size=1,
            dimensions=(64, 64),
            retain_payload=False,
        ),
    ]
    bundle = DataPipeline(
        _data_config(DataSplitConfig(mode="none"), sources=sources),
        seed=19,
    ).build()[0]

    batch = next(iter(bundle.train.dataloader))

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (2, 3, 64, 64)


def test_weighted_sampler_uses_source_homogeneous_step_probabilities() -> None:
    sources = [
        _source(
            "small",
            train_size=3,
            dimensions=(32, 32),
            sampling_weight=0.2,
        ),
        _source(
            "large",
            train_size=3,
            dimensions=(64, 64),
            sampling_weight=0.8,
        ),
    ]
    bundle = DataPipeline(
        _data_config(
            DataSplitConfig(mode="none"),
            sources=sources,
            steps_per_epoch=500,
        ),
        seed=23,
    ).build()[0]
    sampler = bundle.train.dataloader.batch_sampler
    assert isinstance(sampler, MixtureBatchSampler)
    train_dataset = cast(Any, bundle.train.dataset)

    counts = {"small": 0, "large": 0}
    for indices in sampler:
        source_ids = {train_dataset.source_ids[index] for index in indices}
        assert len(source_ids) == 1
        counts[next(iter(source_ids))] += 1

    assert counts["small"] / 500 == pytest.approx(0.2, abs=0.05)
    assert counts["large"] / 500 == pytest.approx(0.8, abs=0.05)


def test_weighted_explicit_steps_cycle_sources_smaller_than_one_batch() -> None:
    sources = [
        _source(
            "tiny",
            train_size=2,
            dimensions=(32, 32),
            sampling_weight=1.0,
        )
    ]
    bundle = DataPipeline(
        _data_config(
            DataSplitConfig(mode="none"),
            sources=sources,
            steps_per_epoch=3,
            drop_last=True,
        ),
        seed=27,
    ).build()[0]
    sampler = bundle.train.dataloader.batch_sampler
    assert isinstance(sampler, MixtureBatchSampler)

    batches = list(sampler)

    assert len(batches) == 3
    assert {len(batch) for batch in batches} == {8}
    assert all(set(batch) <= {0, 1} for batch in batches)


def test_sampler_set_epoch_is_reproducible_and_changes_order() -> None:
    bundle = DataPipeline(
        _data_config(DataSplitConfig(mode="none")),
        seed=29,
    ).build()[0]
    sampler = bundle.train.dataloader.batch_sampler
    assert isinstance(sampler, MixtureBatchSampler)

    sampler.set_epoch(1)
    first = list(sampler)
    sampler.set_epoch(1)
    repeated = list(sampler)
    sampler.set_epoch(2)
    second = list(sampler)

    assert first == repeated
    assert first != second


def test_data_modules_auto_import_and_inject_factory_context(
    monkeypatch,
    tmp_path,
) -> None:
    module_path = tmp_path / "auto_dataset_plugin.py"
    module_path.write_text(
        """
import torch
from torch.utils.data import TensorDataset

from stochaflow.data import DatasetFactory, DatasetView
from stochaflow.utils.registry import REGISTRIES


@REGISTRIES.dataset_factories.register("auto_test_factory")
class AutoTestFactory(DatasetFactory):
    def build(self, request):
        bucket = self.context.buckets.sample_bucket
        dataset = TensorDataset(
            torch.zeros(2, self.context.image.channels, bucket.height, bucket.width)
        )
        return DatasetView(
            source_id=self.context.source_id,
            dataset=dataset,
            sample_keys=("first", "second"),
            bucket_ids=(bucket.name, bucket.name),
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _data_config(
        DataSplitConfig(mode="none"),
        sources=[
            DatasetConfig(
                id="automatic",
                factory="auto_test_factory",
                splits=DatasetSplitMapConfig(train="train"),
            )
        ],
    )
    config.modules = ["auto_dataset_plugin"]

    bundle = DataPipeline(config, seed=31).build()[0]

    assert _length(bundle.train.dataset) == 2
    assert tuple(cast(Any, bundle.train.dataset).source_ids) == (
        "automatic",
        "automatic",
    )


def test_multi_source_bucket_loader_runs_ddpm_training_smoke() -> None:
    class TinyDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(3, 3, kernel_size=1)

        def forward(
            self,
            images: torch.Tensor,
            timesteps: torch.Tensor,
        ) -> torch.Tensor:
            del timesteps
            return self.projection(images)

    bundle = DataPipeline(
        _data_config(DataSplitConfig(mode="none")),
        seed=37,
    ).build()[0]
    diffusion = DDPM(
        model=TinyDenoiser(),
        noise_schedule=LinearBetaSchedule(num_timesteps=4),
    )
    trainer = Trainer(
        model=diffusion,
        optimizer=Adam(diffusion.parameters(), lr=1.0e-3),
        criterion=DDPMEpsilonObjective(),
        device="cpu",
        train_step_fn=ddpm_epsilon_train_step,
    )

    metrics = trainer.train_epoch(
        bundle.train.dataloader,
        epoch_index=1,
        show_progress=False,
        max_batches=2,
    )

    assert metrics["num_batches"] == 2.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
