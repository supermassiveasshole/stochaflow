"""Bucket-homogeneous batch sampling for dataset mixtures."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from torch.utils.data import Sampler

from stochaflow.data.recipe_config import ResolutionBucketRecipeConfig


@dataclass(frozen=True, slots=True)
class ResolutionBucket:
    """One private multi-resolution recipe bucket."""

    name: str
    height: int
    width: int

    @property
    def pixels(self) -> int:
        return self.height * self.width


class ResolutionBucketPolicy:
    """Assign images to buckets and derive pixel-budget batch sizes."""

    def __init__(
        self,
        buckets: Sequence[ResolutionBucketRecipeConfig],
        *,
        base_bucket: str,
        dynamic_batch_size: bool,
    ) -> None:
        self.buckets = tuple(
            ResolutionBucket(bucket.name, bucket.height, bucket.width)
            for bucket in buckets
        )
        if not self.buckets:
            raise ValueError("resolution buckets must not be empty")
        self._by_name = {bucket.name: bucket for bucket in self.buckets}
        if len(self._by_name) != len(self.buckets):
            raise ValueError("resolution bucket names must be unique")
        try:
            self.base_bucket = self._by_name[base_bucket]
        except KeyError as exc:
            raise ValueError(f"unknown base bucket '{base_bucket}'") from exc
        self.dynamic_batch_size = dynamic_batch_size

    def resolve(self, name: str) -> ResolutionBucket:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise ValueError(f"unknown resolution bucket '{name}'") from exc

    def select(self, width: int, height: int) -> ResolutionBucket:
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        image_ratio = width / height
        image_area = width * height

        def distance(bucket: ResolutionBucket) -> tuple[float, float]:
            ratio = bucket.width / bucket.height
            return (
                abs(math.log(image_ratio / ratio)),
                abs(math.log(image_area / bucket.pixels)),
            )

        return min(self.buckets, key=distance)

    def batch_size(self, bucket_name: str, *, base_batch_size: int) -> int:
        if base_batch_size <= 0:
            raise ValueError("base batch size must be positive")
        if not self.dynamic_batch_size:
            return base_batch_size
        bucket = self.resolve(bucket_name)
        return max(
            1,
            math.floor(
                base_batch_size * self.base_bucket.pixels / bucket.pixels
            ),
        )


class BucketedDataset(Protocol):
    """Structural metadata required by :class:`MixtureBatchSampler`."""

    @property
    def bucket_ids(self) -> Sequence[str]: ...

    @property
    def source_ids(self) -> Sequence[str]: ...

    def __len__(self) -> int: ...


class CyclingIndexPool:
    """Draw shuffled indices indefinitely while retaining every sample."""

    def __init__(self, indices: Sequence[int], rng: random.Random) -> None:
        if not indices:
            raise ValueError("cycling index pools must not be empty")
        self._source = list(indices)
        self._rng = rng
        self._current: list[int] = []
        self._offset = 0

    def _refill(self) -> None:
        self._current = list(self._source)
        self._rng.shuffle(self._current)
        self._offset = 0

    def draw(self, count: int) -> list[int]:
        values: list[int] = []
        while len(values) < count:
            if self._offset >= len(self._current):
                self._refill()
            available = min(count - len(values), len(self._current) - self._offset)
            values.extend(self._current[self._offset : self._offset + available])
            self._offset += available
        return values


class MixtureBatchSampler(Sampler[list[int]]):
    """Emit same-resolution batches with optional step-level source weighting."""

    def __init__(
        self,
        dataset: BucketedDataset,
        bucket_policy: ResolutionBucketPolicy,
        *,
        base_batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed: int,
        source_weights: Mapping[str, float] | None = None,
        steps_per_epoch: int | str = "auto",
    ) -> None:
        if len(dataset.bucket_ids) != len(dataset):
            raise ValueError("bucket metadata length must match dataset length")
        if len(dataset.source_ids) != len(dataset):
            raise ValueError("source metadata length must match dataset length")
        if base_batch_size <= 0:
            raise ValueError("base batch size must be positive")
        if steps_per_epoch != "auto" and (
            not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0
        ):
            raise ValueError("steps_per_epoch must be positive or 'auto'")

        self.dataset = dataset
        self.bucket_policy = bucket_policy
        self.base_batch_size = base_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.source_weights = (
            dict(source_weights) if source_weights is not None else None
        )
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0

        self._bucket_indices: dict[str, list[int]] = defaultdict(list)
        self._source_bucket_indices: dict[tuple[str, str], list[int]] = defaultdict(
            list
        )
        for index, (source_id, bucket_id) in enumerate(
            zip(dataset.source_ids, dataset.bucket_ids, strict=True)
        ):
            bucket_policy.resolve(bucket_id)
            self._bucket_indices[bucket_id].append(index)
            self._source_bucket_indices[(source_id, bucket_id)].append(index)

        if self.source_weights is not None:
            if any(weight <= 0 for weight in self.source_weights.values()):
                raise ValueError("source weights must be positive")
            present_sources = set(dataset.source_ids)
            missing = sorted(set(self.source_weights) - present_sources)
            if missing:
                raise ValueError(
                    "weighted source(s) have no samples after splitting: "
                    + ", ".join(missing)
                )

        natural_steps = sum(
            self._batch_count(bucket_id, len(indices), drop_last=drop_last)
            for bucket_id, indices in self._bucket_indices.items()
        )
        can_cycle_weighted_explicit_steps = (
            self.source_weights is not None
            and isinstance(self.steps_per_epoch, int)
        )
        if natural_steps <= 0 and not can_cycle_weighted_explicit_steps:
            raise ValueError("batch sampler would yield no batches")
        self._natural_steps = natural_steps

    def _batch_size(self, bucket_id: str) -> int:
        return self.bucket_policy.batch_size(
            bucket_id,
            base_batch_size=self.base_batch_size,
        )

    def _batch_count(self, bucket_id: str, size: int, *, drop_last: bool) -> int:
        batch_size = self._batch_size(bucket_id)
        if drop_last:
            return size // batch_size
        return math.ceil(size / batch_size)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle stream for an epoch."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        if isinstance(self.steps_per_epoch, int):
            return self.steps_per_epoch
        return self._natural_steps

    def _natural_batches(self, rng: random.Random) -> list[list[int]]:
        batches: list[list[int]] = []
        for bucket in self.bucket_policy.buckets:
            indices = list(self._bucket_indices.get(bucket.name, ()))
            if not indices:
                continue
            if self.shuffle:
                rng.shuffle(indices)
            batch_size = self._batch_size(bucket.name)
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset : offset + batch_size]
                if len(batch) < batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _unweighted_batches(self, rng: random.Random) -> Iterator[list[int]]:
        target_steps = len(self)
        yielded = 0
        while yielded < target_steps:
            batches = self._natural_batches(rng)
            if not batches:
                raise RuntimeError("batch sampler produced an empty epoch")
            for batch in batches:
                if yielded >= target_steps:
                    return
                yielded += 1
                yield batch

    def _weighted_batches(self, rng: random.Random) -> Iterator[list[int]]:
        assert self.source_weights is not None
        sources = list(self.source_weights)
        weights = [self.source_weights[source] for source in sources]
        pools = {
            key: CyclingIndexPool(indices, rng)
            for key, indices in self._source_bucket_indices.items()
        }
        source_buckets: dict[str, list[str]] = {}
        source_bucket_weights: dict[str, list[int]] = {}
        for source in sources:
            bucket_names: list[str] = []
            batch_counts: list[int] = []
            for bucket in self.bucket_policy.buckets:
                indices = self._source_bucket_indices.get((source, bucket.name))
                if not indices:
                    continue
                bucket_names.append(bucket.name)
                batch_counts.append(
                    max(
                        1,
                        math.ceil(len(indices) / self._batch_size(bucket.name)),
                    )
                )
            source_buckets[source] = bucket_names
            source_bucket_weights[source] = batch_counts

        for _ in range(len(self)):
            source = rng.choices(sources, weights=weights, k=1)[0]
            bucket_id = rng.choices(
                source_buckets[source],
                weights=source_bucket_weights[source],
                k=1,
            )[0]
            yield pools[(source, bucket_id)].draw(self._batch_size(bucket_id))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self.source_weights is None:
            yield from self._unweighted_batches(rng)
        else:
            yield from self._weighted_batches(rng)


__all__ = [
    "BucketedDataset",
    "MixtureBatchSampler",
    "ResolutionBucket",
    "ResolutionBucketPolicy",
]
