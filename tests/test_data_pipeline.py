"""Tests for registered modality-neutral data pipelines."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sized
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset

from stochaflow.data import (
    DataBundle,
    DataPipeline,
    DataPipelineContext,
    DatasetBuildRequest,
    DatasetFactory,
    DatasetSelection,
    DatasetView,
    ImageSampleMetadata,
    MixtureBatchSampler,
    SplitData,
    build_data_pipeline,
)
from stochaflow.diffusion import DDPM, DDPMEpsilonObjective, LinearBetaSchedule
from stochaflow.training import Trainer
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryError


class StructuredDataset(Dataset[Any]):
    def __init__(self, size: int, *, kind: str, offset: int = 0) -> None:
        self.size = size
        self.kind = kind
        self.offset = offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        value = torch.tensor([float(index + self.offset)])
        if self.kind == "tensor":
            return value
        if self.kind == "mapping":
            return {"state": value, "condition": {"index": index}}
        if self.kind == "tuple":
            return value, index
        if self.kind == "list":
            return [value, index]
        raise ValueError(self.kind)


@REGISTRIES.dataset_factories.register("stage2_structured")
class StructuredDatasetFactory(DatasetFactory):
    def build(self, request: DatasetBuildRequest) -> DatasetView:
        sizes = cast(Mapping[str, int], self.context.params["sizes"])
        size = sizes[request.native_split]
        offset = 100 if request.role == "eval" else 0
        dataset = StructuredDataset(
            size,
            kind=str(self.context.params.get("kind", "tensor")),
            offset=offset,
        )
        return DatasetView(
            source_id=self.context.source_id,
            dataset=dataset,
            sample_keys=tuple(
                f"{request.native_split}:{index}" for index in range(size)
            ),
        )


class RawImageDataset(Dataset[Any]):
    def __init__(self, size: int, dimensions: tuple[int, int], offset: int) -> None:
        self.size = size
        self.width, self.height = dimensions
        self.offset = offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, int]]:
        image = torch.full(
            (3, self.height, self.width),
            float(index + self.offset) / 255.0,
        )
        return image, {"label": index}


@REGISTRIES.dataset_factories.register("stage2_images")
class RawImageDatasetFactory(DatasetFactory):
    def build(self, request: DatasetBuildRequest) -> DatasetView:
        sizes = cast(Mapping[str, int], self.context.params["sizes"])
        size = sizes[request.native_split]
        dimensions = cast(tuple[int, int], self.context.params["dimensions"])
        dataset = RawImageDataset(
            size,
            dimensions,
            100 if request.role == "eval" else 0,
        )
        metadata = ImageSampleMetadata(*dimensions)
        return DatasetView(
            source_id=self.context.source_id,
            dataset=dataset,
            sample_keys=tuple(
                f"{request.native_split}:{index}" for index in range(size)
            ),
            batch_metadata=(metadata,) * size,
        )


def _map_component(
    *,
    mode: str = "none",
    kind: str = "tensor",
    validation_size: int | float | None = None,
    num_folds: int | None = None,
) -> ComponentConfig:
    return ComponentConfig(
        name="map",
        params={
            "dataset": {
                "id": "physics",
                "factory": "stage2_structured",
                "params": {
                    "sizes": {"train": 10, "val": 3, "test": 2},
                    "kind": kind,
                },
                "splits": {
                    "train": "train",
                    "validation": "val",
                    "test": "test",
                },
            },
            "splits": {
                "mode": mode,
                "validation_size": validation_size,
                "num_folds": num_folds,
            },
            "dataloader": {
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": True,
                "drop_last": False,
                "pin_memory": False,
                "persistent_workers": False,
                "steps_per_epoch": "auto",
            },
        },
    )


def _image_source(
    source_id: str,
    *,
    size: int,
    dimensions: tuple[int, int],
    weight: float | None = None,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "factory": "stage2_images",
        "sampling_weight": weight,
        "params": {
            "sizes": {"train": size, "val": 2, "test": 1},
            "dimensions": list(dimensions),
        },
        "splits": {"train": "train", "validation": "val", "test": "test"},
    }


def _image_component(
    *,
    sources: list[dict[str, Any]] | None = None,
    mode: str = "none",
    validation_size: int | float | None = None,
    steps_per_epoch: int | str = "auto",
) -> ComponentConfig:
    return ComponentConfig(
        name="multi_resolution_image",
        params={
            "datasets": sources
            or [
                _image_source("small", size=6, dimensions=(32, 32)),
                _image_source("large", size=4, dimensions=(64, 64)),
            ],
            "image": {"channels": 3, "normalize": True},
            "batching": {
                "buckets": [
                    {"name": "square_32", "height": 32, "width": 32},
                    {"name": "square_64", "height": 64, "width": 64},
                ],
                "base_bucket": "square_64",
                "dynamic_batch_size": True,
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": True,
                "drop_last": False,
                "pin_memory": False,
                "persistent_workers": False,
                "steps_per_epoch": steps_per_epoch,
            },
            "splits": {
                "mode": mode,
                "validation_size": validation_size,
            },
        },
    )


@pytest.mark.parametrize("kind", ["tensor", "mapping", "tuple", "list"])
def test_map_pipeline_preserves_default_collation_structures(kind: str) -> None:
    bundle = build_data_pipeline(_map_component(kind=kind), seed=7)[0]

    batch = next(iter(bundle.train.dataloader))

    if kind == "tensor":
        assert isinstance(batch, torch.Tensor)
    elif kind == "mapping":
        assert isinstance(batch, Mapping)
        assert set(batch) == {"state", "condition"}
    else:
        assert isinstance(batch, list)
        assert len(batch) == 2


def test_map_pipeline_supports_official_random_holdout_and_kfold() -> None:
    official = build_data_pipeline(_map_component(mode="official"), seed=11)[0]
    holdout = build_data_pipeline(
        _map_component(mode="random_holdout", validation_size=3),
        seed=11,
    )[0]
    folds = build_data_pipeline(
        _map_component(mode="kfold", num_folds=3),
        seed=11,
    )

    assert official.valid is not None and official.valid.num_samples == 3
    assert holdout.valid is not None and holdout.valid.num_samples == 3
    assert isinstance(holdout.valid.dataset, DatasetSelection)
    assert len(folds) == 3
    assert [bundle.fold_index for bundle in folds] == [0, 1, 2]


def test_map_pipeline_rejects_sampling_weight() -> None:
    component = _map_component()
    cast(dict[str, Any], component.params["dataset"])["sampling_weight"] = 1.0

    with pytest.raises(ValueError, match="sampling_weight is not supported"):
        build_data_pipeline(component, seed=1)


def test_image_pipeline_keeps_bucket_batches_and_labels() -> None:
    bundle = build_data_pipeline(_image_component(), seed=17)[0]
    sampler = cast(Any, bundle.train.dataloader).batch_sampler
    dataset = cast(Any, bundle.train.dataset)

    assert isinstance(sampler, MixtureBatchSampler)
    for indices in sampler:
        assert len({dataset.bucket_ids[index] for index in indices}) == 1
    batches = list(bundle.train.dataloader)
    assert {tuple(batch[0].shape[-2:]) for batch in batches} == {
        (32, 32),
        (64, 64),
    }
    assert all(isinstance(batch[1], Mapping) and "label" in batch[1] for batch in batches)


def test_image_pipeline_weighting_and_set_epoch_are_deterministic() -> None:
    component = _image_component(
        sources=[
            _image_source("small", size=3, dimensions=(32, 32), weight=0.2),
            _image_source("large", size=3, dimensions=(64, 64), weight=0.8),
        ],
        steps_per_epoch=500,
    )
    bundle = build_data_pipeline(component, seed=23)[0]
    sampler = cast(
        MixtureBatchSampler,
        cast(Any, bundle.train.dataloader).batch_sampler,
    )
    dataset = cast(Any, bundle.train.dataset)
    counts = {"small": 0, "large": 0}
    for indices in sampler:
        sources = {dataset.source_ids[index] for index in indices}
        assert len(sources) == 1
        counts[next(iter(sources))] += 1
    assert counts["small"] / 500 == pytest.approx(0.2, abs=0.05)

    sampler.set_epoch(2)
    first = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == first
    sampler.set_epoch(3)
    assert list(sampler) != first


class FiniteStream:
    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            yield torch.zeros(2, 1)


@REGISTRIES.data_pipelines.register("stage2_stream")
class StreamingPipeline(DataPipeline):
    def build(self) -> list[DataBundle]:
        return [
            DataBundle(
                train=SplitData(
                    name="train",
                    dataloader=FiniteStream(),
                    num_batches=int(self.context.params["num_batches"]),
                )
            )
        ]


def test_custom_iterable_pipeline_needs_no_dataset_or_sample_keys() -> None:
    bundles = build_data_pipeline(
        ComponentConfig(name="stage2_stream", params={"num_batches": 5}),
        seed=31,
    )

    assert bundles[0].train.dataset is None
    assert bundles[0].train.resolved_num_batches() == 5


class EmptyPipeline(DataPipeline):
    def build(self) -> list[DataBundle]:
        return []


def test_pipeline_registry_and_build_contract_fail_clearly() -> None:
    with pytest.raises(RegistryError, match="must inherit"):
        REGISTRIES.data_pipelines.add("stage2_wrong_base", object)
    REGISTRIES.data_pipelines.add("stage2_empty", EmptyPipeline)
    with pytest.raises(ValueError, match="non-empty"):
        build_data_pipeline(ComponentConfig(name="stage2_empty"), seed=1)
    with pytest.raises(RegistryError, match="unknown data pipeline"):
        build_data_pipeline(ComponentConfig(name="missing_stage2_pipeline"), seed=1)


def test_custom_pipeline_context_receives_a_params_copy() -> None:
    params = {"nested": {"value": 1}}
    context = DataPipelineContext(params=params, seed=41)

    cast(dict[str, Any], params["nested"])["value"] = 9

    assert cast(dict[str, Any], context.params["nested"])["value"] == 1


def test_multi_resolution_batches_run_tensor_diffusion_training() -> None:
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

    bundle = build_data_pipeline(_image_component(), seed=37)[0]
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


def test_dataset_view_validates_optional_batch_metadata_length() -> None:
    dataset = StructuredDataset(2, kind="tensor")
    with pytest.raises(ValueError, match="batch_metadata length"):
        DatasetView(
            source_id="bad",
            dataset=dataset,
            sample_keys=("a", "b"),
            batch_metadata=({},),
        )


def _length(value: object) -> int:
    return len(cast(Sized, value))
