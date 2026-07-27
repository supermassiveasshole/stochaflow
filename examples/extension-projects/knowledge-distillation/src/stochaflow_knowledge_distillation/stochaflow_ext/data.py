"""Classification data assembly for the distillation reference project."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sized
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset

from stochaflow.extensions import REGISTRIES, DataBuilder, DataLoaders

_PREFIX = "stochaflow-knowledge-distillation"
_MAX_SEED = (1 << 63) - 1


class EpochShuffleSampler(Sampler[int]):
    """Derive each shuffle solely from the experiment seed and epoch."""

    def __init__(self, data_source: object, *, seed: int) -> None:
        if not isinstance(data_source, Sized):
            raise TypeError("epoch shuffle requires a sized data source")
        self._data_source = data_source
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic permutation for one one-based epoch."""

        runtime_epoch = cast(object, epoch)
        if (
            isinstance(runtime_epoch, bool)
            or not isinstance(runtime_epoch, int)
            or runtime_epoch < 0
        ):
            raise ValueError("sampler epoch must be a non-negative integer")
        self._epoch = runtime_epoch

    def __iter__(self) -> Iterator[int]:
        epoch_seed = (self._seed + self._epoch) % _MAX_SEED
        generator = torch.Generator().manual_seed(epoch_seed)
        yield from torch.randperm(len(self), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self._data_source)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return dict(cast(Mapping[str, Any], value))


def _unknown(params: Mapping[str, Any], scope: str) -> None:
    if params:
        raise ValueError(f"unknown {scope} params: {', '.join(sorted(params))}")


def _synthetic_splits(
    params: dict[str, Any],
    *,
    seed: int,
) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
    input_features = _positive_int(params.pop("input_features", 8), "input_features")
    num_classes = _positive_int(params.pop("num_classes", 4), "num_classes")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    train_samples = _positive_int(params.pop("train_samples", 64), "train_samples")
    validation_samples = _positive_int(
        params.pop("validation_samples", 16),
        "validation_samples",
    )
    test_samples = _positive_int(params.pop("test_samples", 16), "test_samples")
    noise_std = params.pop("noise_std", 0.25)
    if (
        isinstance(noise_std, bool)
        or not isinstance(noise_std, (int, float))
        or not math.isfinite(float(noise_std))
        or noise_std < 0
    ):
        raise ValueError("noise_std must be a finite non-negative number")
    _unknown(params, "synthetic data recipe")

    prototype_generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(
        num_classes,
        input_features,
        generator=prototype_generator,
    )

    def make_split(num_samples: int, split_seed: int) -> TensorDataset:
        generator = torch.Generator().manual_seed(split_seed)
        labels = torch.randint(
            num_classes,
            (num_samples,),
            generator=generator,
        )
        inputs = prototypes[labels].clone()
        if noise_std:
            inputs.add_(
                float(noise_std)
                * torch.randn(inputs.shape, generator=generator),
            )
        return TensorDataset(inputs, labels)

    return (
        make_split(train_samples, seed + 1),
        make_split(validation_samples, seed + 2),
        make_split(test_samples, seed + 3),
    )


def _loaders(
    params: dict[str, Any],
    *,
    datasets: tuple[Dataset[Any], Dataset[Any], Dataset[Any]],
    seed: int,
) -> DataLoaders:
    batch_size = _positive_int(params.pop("batch_size", 8), "batch_size")
    num_workers = _nonnegative_int(params.pop("num_workers", 0), "num_workers")
    shuffle = params.pop("shuffle", True)
    drop_last = params.pop("drop_last", False)
    pin_memory = params.pop("pin_memory", False)
    persistent_workers = params.pop("persistent_workers", False)
    for name, value in (
        ("shuffle", shuffle),
        ("drop_last", drop_last),
        ("pin_memory", pin_memory),
        ("persistent_workers", persistent_workers),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    _unknown(params, "loader")

    train, validation, test = datasets
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    return DataLoaders(
        train=DataLoader(
            train,
            shuffle=False,
            sampler=(
                EpochShuffleSampler(train, seed=seed + 100)
                if shuffle
                else None
            ),
            drop_last=drop_last,
            **common,
        ),
        validation=DataLoader(validation, shuffle=False, drop_last=False, **common),
        test=DataLoader(test, shuffle=False, drop_last=False, **common),
        artifact_bindings=None,
    )


@REGISTRIES.data_builders.register(f"{_PREFIX}.classification")
class ClassificationDataBuilder(DataBuilder):
    """Build a deterministic in-memory classification fixture.

    This recipe has no external input artifact. Real dataset acquisition belongs
    in an extension-owned DataSource/DataArtifact pair or a compatible framework
    data recipe.
    """

    def build(self) -> DataLoaders:
        """Assemble one independent train/validation/test loader set."""

        params = dict(self.context.params)
        synthetic = _mapping(params.pop("synthetic", {}), "synthetic")
        loader = _mapping(params.pop("loader", {}), "loader")
        _unknown(params, "data builder")
        datasets = _synthetic_splits(synthetic, seed=self.context.seed)
        return _loaders(loader, datasets=datasets, seed=self.context.seed)


__all__ = ["ClassificationDataBuilder"]
