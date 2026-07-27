"""DataLoader contracts, collation, and deterministic construction."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.data.datasets import (
    ClassLabeledImageDataset,
    MultiResolutionDataset,
)
from stochaflow.data.recipe_config import LoaderRecipeConfig
from stochaflow.data.samplers import (
    EpochRandomSampler,
    EpochTaggedIndexSampler,
    MixtureBatchSampler,
    ResolutionBucketPolicy,
)
from stochaflow.utils.iterables import try_length


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Ready-to-use iterables for one independent training run."""

    train: Iterable[Any]
    validation: Iterable[Any] | None = None
    test: Iterable[Any] | None = None
    steps_per_epoch: int | None = None
    artifact_bindings: DataArtifactBindings | None = None

    def __post_init__(self) -> None:
        for role in ("train", "validation", "test"):
            loader = getattr(self, role)
            if loader is not None and not isinstance(loader, Iterable):
                raise TypeError(f"{role} loader must be iterable")
            if loader is not None and isinstance(loader, Iterator):
                raise TypeError(
                    f"{role} loader must be re-iterable, not a one-shot iterator"
                )
        if self.steps_per_epoch is not None:
            steps_per_epoch = cast(object, self.steps_per_epoch)
            if (
                isinstance(steps_per_epoch, bool)
                or not isinstance(steps_per_epoch, int)
                or steps_per_epoch <= 0
            ):
                raise ValueError("steps_per_epoch must be a positive integer")
        train_length = try_length(self.train)
        if self.steps_per_epoch is None and train_length is None:
            raise ValueError(
                "train loader must expose len() or set steps_per_epoch"
            )
        if train_length is not None:
            if train_length <= 0:
                raise ValueError("train loader must yield at least one batch")
            if (
                self.steps_per_epoch is not None
                and self.steps_per_epoch > train_length
            ):
                raise ValueError(
                    "steps_per_epoch must not exceed the sized train loader length"
                )
        artifact_bindings = cast(object, self.artifact_bindings)
        if artifact_bindings is not None and not isinstance(
            artifact_bindings, DataArtifactBindings
        ):
            raise TypeError(
                "artifact_bindings must be DataArtifactBindings or None"
            )


def collate_image_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Stack a standard image-recipe batch."""

    return torch.stack([image for image, _ in batch]), {}


def collate_super_resolution_batch(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack targets and aligned low-resolution conditions."""

    return (
        torch.stack([high for high, _ in batch]),
        {"low_res": torch.stack([condition["low_res"] for _, condition in batch])},
    )


def collate_class_labeled_image_batch(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack images and expose class labels through the condition mapping."""

    if not batch:
        raise ValueError("class-labeled image batch must not be empty")
    return (
        torch.stack([image for image, _ in batch]),
        {
            "class_label": torch.tensor(
                [class_label for _, class_label in batch],
                dtype=torch.long,
            )
        },
    )


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def data_loader_kwargs(
    config: LoaderRecipeConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Build common deterministic DataLoader keyword arguments."""

    kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "generator": torch.Generator().manual_seed(seed),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["worker_init_fn"] = seed_data_loader_worker
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def build_map_data_loader(
    dataset: Dataset[Any] | None,
    config: LoaderRecipeConfig,
    *,
    training: bool,
    seed: int,
    collate_fn: Callable[[list[Any]], Any],
) -> DataLoader[Any] | None:
    """Build one finite map-style DataLoader."""

    if dataset is None:
        return None
    sampler = (
        EpochRandomSampler(dataset, seed=seed)
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
        **data_loader_kwargs(config, seed=seed),
    )


def build_class_labeled_image_data_loader(
    dataset: ClassLabeledImageDataset | None,
    config: LoaderRecipeConfig,
    *,
    training: bool,
    seed: int,
) -> DataLoader[Any] | None:
    """Build a class-labeled loader with epoch-aware training indices."""

    if dataset is None:
        return None
    sampler = (
        EpochTaggedIndexSampler(
            dataset,
            seed=seed,
            shuffle=config.shuffle,
        )
        if training
        else None
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=config.drop_last if training else False,
        collate_fn=collate_class_labeled_image_batch,
        **data_loader_kwargs(config, seed=seed),
    )


def build_multi_resolution_data_loader(
    dataset: MultiResolutionDataset | None,
    policy: ResolutionBucketPolicy,
    config: LoaderRecipeConfig,
    *,
    training: bool,
    seed: int,
    source_weights: Mapping[str, float] | None,
    collate_fn: Callable[[list[Any]], Any],
) -> DataLoader[Any] | None:
    """Build a bucket-homogeneous DataLoader for a source mixture."""

    if dataset is None:
        return None
    sampler = MixtureBatchSampler(
        dataset,
        policy,
        base_batch_size=config.batch_size,
        drop_last=config.drop_last if training else False,
        shuffle=config.shuffle if training else False,
        seed=seed,
        source_weights=source_weights if training else None,
        steps_per_epoch=config.steps_per_epoch if training else "auto",
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        **data_loader_kwargs(config, seed=seed),
    )


def configured_steps_per_epoch(
    config: LoaderRecipeConfig,
) -> int | None:
    """Return an explicit epoch length or ``None`` for natural length."""

    return (
        None
        if config.steps_per_epoch == "auto"
        else cast(int, config.steps_per_epoch)
    )


__all__ = [
    "DataLoaders",
    "build_class_labeled_image_data_loader",
    "build_map_data_loader",
    "build_multi_resolution_data_loader",
    "collate_class_labeled_image_batch",
    "collate_image_batch",
    "collate_super_resolution_batch",
    "configured_steps_per_epoch",
    "data_loader_kwargs",
    "seed_data_loader_worker",
]
