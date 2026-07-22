"""Classification data assembly for the distillation reference project."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sized
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset, random_split

from stochaflow.extensions import DataBuilder, DataLoaders, REGISTRIES

_PREFIX = "stochaflow-knowledge-distillation"
_MAX_SEED = (1 << 63) - 1


class _EpochShuffleSampler(Sampler[int]):
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
    _unknown(params, "synthetic source")

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


def _validation_count(value: object, *, total: int) -> int:
    if isinstance(value, bool):
        raise TypeError("validation_size must be an integer or fraction")
    if isinstance(value, int):
        count = value
    elif isinstance(value, float):
        if not 0.0 < value < 1.0:
            raise ValueError("fractional validation_size must satisfy 0 < value < 1")
        count = round(total * value)
    else:
        raise TypeError("validation_size must be an integer or fraction")
    if count <= 0 or count >= total:
        raise ValueError(
            "validation_size must leave non-empty train and validation sets"
        )
    return count


def _torchvision_splits(
    params: dict[str, Any],
    *,
    seed: int,
) -> tuple[Dataset[Any], Dataset[Any], Dataset[Any]]:
    dataset_name = params.pop("dataset", "MNIST")
    if dataset_name not in {"MNIST", "FashionMNIST", "CIFAR10"}:
        raise ValueError(
            "torchvision dataset must be MNIST, FashionMNIST, or CIFAR10"
        )
    root = params.pop("root", "data/torchvision")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("torchvision root must be a non-empty path string")
    download = params.pop("download", False)
    if not isinstance(download, bool):
        raise TypeError("torchvision download must be boolean")
    validation_size = params.pop("validation_size", 0.1)
    _unknown(params, "torchvision source")

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "torchvision source requires the project's 'vision' extra"
        ) from exc

    dataset_type = getattr(datasets, dataset_name)
    transform = transforms.ToTensor()
    full_train = dataset_type(
        root=str(Path(root)),
        train=True,
        download=download,
        transform=transform,
    )
    test = dataset_type(
        root=str(Path(root)),
        train=False,
        download=download,
        transform=transform,
    )
    validation_count = _validation_count(validation_size, total=len(full_train))
    train_count = len(full_train) - validation_count
    train, validation = random_split(
        full_train,
        (train_count, validation_count),
        generator=torch.Generator().manual_seed(seed),
    )
    return train, validation, test


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
                _EpochShuffleSampler(train, seed=seed + 100)
                if shuffle
                else None
            ),
            drop_last=drop_last,
            **common,
        ),
        validation=DataLoader(validation, shuffle=False, drop_last=False, **common),
        test=DataLoader(test, shuffle=False, drop_last=False, **common),
    )


@REGISTRIES.data_builders.register(f"{_PREFIX}.classification")
class ClassificationDataBuilder(DataBuilder):
    """Build deterministic synthetic or optional torchvision classification data."""

    def build(self) -> DataLoaders:
        """Assemble one independent train/validation/test loader set."""

        params = dict(self.context.params)
        source = _mapping(params.pop("source", {"kind": "synthetic"}), "source")
        loader = _mapping(params.pop("loader", {}), "loader")
        _unknown(params, "data builder")
        kind = source.pop("kind", "synthetic")
        if kind == "synthetic":
            datasets = _synthetic_splits(source, seed=self.context.seed)
        elif kind == "torchvision":
            datasets = _torchvision_splits(source, seed=self.context.seed)
        else:
            raise ValueError("source.kind must be synthetic or torchvision")
        return _loaders(loader, datasets=datasets, seed=self.context.seed)


__all__ = ["ClassificationDataBuilder"]
