"""Memory-mapped trajectory triplets for Kolmogorov-flow experiments."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from stochaflow.extensions import REGISTRIES, DataBuilder, DataLoaders

from ._config import (
    copied_mapping,
    pop_bool,
    pop_int,
    pop_optional_range,
    pop_path,
    reject_unknown,
    required_mapping,
)


def _open_trajectories(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"trajectory array does not exist: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"trajectory source must be a NumPy array: {path}")
    if value.ndim != 4:
        raise ValueError(
            "trajectory array must have shape [trajectory, time, height, width]"
        )
    if value.shape[1] < 3:
        raise ValueError("trajectory array must contain at least three time frames")
    if value.shape[2] < 2 or value.shape[3] < 2:
        raise ValueError("trajectory spatial dimensions must be at least 2x2")
    if value.shape[2] != value.shape[3] or value.shape[2] % 2:
        raise ValueError("trajectory fields must use an even, square spectral grid")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("trajectory array must contain floating-point values")
    return value


class TrajectoryTripletDataset(Dataset[torch.Tensor]):
    """Expose consecutive three-frame windows without loading the full array."""

    def __init__(self, path: Path, trajectory_range: tuple[int, int]) -> None:
        self.path = path
        self.trajectory_range = trajectory_range
        array = _open_trajectories(path)
        _, stop = trajectory_range
        if stop > array.shape[0]:
            raise ValueError(
                f"trajectory range {trajectory_range} exceeds {array.shape[0]} rows"
            )
        self._num_frames = int(array.shape[1])
        self._height = int(array.shape[2])
        self._width = int(array.shape[3])
        self._num_trajectories = int(array.shape[0])
        self._array: np.ndarray | None = None

    @property
    def sample_shape(self) -> tuple[int, int, int]:
        """Return the raw `(time, height, width)` sample shape."""

        return 3, self._height, self._width

    @property
    def source_shape(self) -> tuple[int, int, int, int]:
        """Return the complete memory-mapped source shape."""

        return (
            self._num_trajectories,
            self._num_frames,
            self._height,
            self._width,
        )

    def __len__(self) -> int:
        start, stop = self.trajectory_range
        return (stop - start) * (self._num_frames - 2)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        windows_per_trajectory = self._num_frames - 2
        relative_trajectory, frame = divmod(index, windows_per_trajectory)
        trajectory = self.trajectory_range[0] + relative_trajectory
        triplet = np.array(
            self._data()[trajectory, frame : frame + 3],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(triplet)

    def _data(self) -> np.ndarray:
        if self._array is None:
            self._array = _open_trajectories(self.path)
        return self._array

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_array"] = None
        return state


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class EpochShuffleSampler(Sampler[int]):
    """Derive each permutation from `(seed, epoch)` for strict resume."""

    def __init__(self, size: int, *, seed: int) -> None:
        if isinstance(size, bool) or size <= 0:
            raise ValueError("sampler size must be a positive integer")
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        normalized_seed = (self.seed + self.epoch) % (2**63 - 1)
        generator = torch.Generator().manual_seed(normalized_seed)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _loader(
    dataset: Dataset[torch.Tensor],
    *,
    loader_params: Mapping[str, Any],
    seed: int,
    training: bool,
) -> DataLoader[torch.Tensor]:
    params = copied_mapping(loader_params, path="data.params.loader")
    batch_size = pop_int(params, "batch_size", path="data.params.loader")
    num_workers = pop_int(
        params,
        "num_workers",
        path="data.params.loader",
        default=0,
        minimum=0,
    )
    declared_shuffle = pop_bool(
        params,
        "shuffle",
        path="data.params.loader",
        default=training,
    )
    declared_drop_last = pop_bool(
        params,
        "drop_last",
        path="data.params.loader",
        default=training,
    )
    shuffle = declared_shuffle if training else False
    drop_last = declared_drop_last if training else False
    pin_memory = pop_bool(
        params,
        "pin_memory",
        path="data.params.loader",
        default=False,
    )
    persistent_workers = pop_bool(
        params,
        "persistent_workers",
        path="data.params.loader",
        default=False,
    )
    prefetch_factor = params.pop("prefetch_factor", None)
    if prefetch_factor is not None and (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor <= 0
    ):
        raise ValueError("data.params.loader.prefetch_factor must be null or positive")
    params.pop("steps_per_epoch", None)
    reject_unknown(params, path="data.params.loader")
    if num_workers == 0 and persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")
    if num_workers == 0 and prefetch_factor is not None:
        raise ValueError("prefetch_factor requires num_workers > 0")
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        EpochShuffleSampler(len(cast(Any, dataset)), seed=seed)
        if shuffle
        else None
    )
    kwargs: dict[str, Any] = {}
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        **kwargs,
    )


def _steps_per_epoch(loader_params: Mapping[str, Any]) -> int | None:
    raw = loader_params.get("steps_per_epoch", "auto")
    if raw == "auto":
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError("data.params.loader.steps_per_epoch must be auto or positive")
    return raw


def _validate_disjoint_ranges(
    ranges: Mapping[str, tuple[int, int] | None],
) -> None:
    declared = [(name, value) for name, value in ranges.items() if value is not None]
    for index, (left_name, left) in enumerate(declared):
        for right_name, right in declared[index + 1 :]:
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError(
                    f"trajectory ranges '{left_name}' and '{right_name}' overlap"
                )


@REGISTRIES.data_builders.register(
    "physics-reconstruction.kolmogorov-trajectories"
)
class KolmogorovDataBuilder(DataBuilder):
    """Build raw triplet loaders from explicit trajectory ranges."""

    def build(self) -> DataLoaders:
        params = copied_mapping(self.context.params, path="data.params")
        path = pop_path(params, "path", path="data.params")
        train_range = pop_optional_range(
            params, "train_trajectories", path="data.params"
        )
        validation_range = pop_optional_range(
            params, "validation_trajectories", path="data.params"
        )
        test_range = pop_optional_range(
            params, "test_trajectories", path="data.params"
        )
        loader_params = required_mapping(params, "loader", path="data.params")
        reject_unknown(params, path="data.params")
        if train_range is None:
            raise ValueError("data.params.train_trajectories is required")
        _validate_disjoint_ranges(
            {
                "train": train_range,
                "validation": validation_range,
                "test": test_range,
            }
        )
        train_dataset = TrajectoryTripletDataset(path, train_range)
        validation_dataset = (
            TrajectoryTripletDataset(path, validation_range)
            if validation_range is not None
            else None
        )
        test_dataset = (
            TrajectoryTripletDataset(path, test_range)
            if test_range is not None
            else None
        )
        train = _loader(
            train_dataset,
            loader_params=loader_params,
            seed=self.context.seed,
            training=True,
        )
        validation = (
            _loader(
                validation_dataset,
                loader_params=loader_params,
                seed=self.context.seed + 1,
                training=False,
            )
            if validation_dataset is not None
            else None
        )
        test = (
            _loader(
                test_dataset,
                loader_params=loader_params,
                seed=self.context.seed + 2,
                training=False,
            )
            if test_dataset is not None
            else None
        )
        return DataLoaders(
            train=train,
            validation=validation,
            test=test,
            steps_per_epoch=_steps_per_epoch(loader_params),
        )


__all__ = [
    "EpochShuffleSampler",
    "KolmogorovDataBuilder",
    "TrajectoryTripletDataset",
]
