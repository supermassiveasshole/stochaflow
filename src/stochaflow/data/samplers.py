"""Bucket-homogeneous batch sampling for dataset mixtures."""

from __future__ import annotations

import hashlib
import math
import random
from array import array
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from torch.utils.data import Dataset, Sampler

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
    def bucket_codes(self) -> Sequence[int]: ...

    @property
    def source_codes(self) -> Sequence[int]: ...

    @property
    def bucket_names(self) -> Sequence[str]: ...

    @property
    def source_names(self) -> Sequence[str]: ...

    def __len__(self) -> int: ...


class EpochRandomSampler(Sampler[int]):
    """Rebuild a deterministic shuffled index stream from seed and epoch."""

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


class EpochTaggedIndexSampler(Sampler[tuple[int, int]]):
    """Emit deterministic indices tagged with their current training epoch."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.size = len(cast(Sized, dataset))
        if self.size <= 0:
            raise ValueError("epoch-tagged sampler dataset must not be empty")
        seed_value = cast(object, seed)
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError("epoch-tagged sampler seed must be an integer")
        shuffle_value = cast(object, shuffle)
        if not isinstance(shuffle_value, bool):
            raise TypeError("epoch-tagged sampler shuffle must be boolean")
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        epoch = self.epoch
        if self.shuffle:
            identity = (
                f"stochaflow.epoch-tagged-index.shuffle.v1\0"
                f"{self.seed}\0{epoch}"
            ).encode()
            epoch_seed = int.from_bytes(
                hashlib.sha256(identity).digest()[:8],
                byteorder="little",
            )
            generator = torch.Generator().manual_seed(epoch_seed)
            indices = cast(
                list[int],
                torch.randperm(self.size, generator=generator).tolist(),
            )
        else:
            indices = range(self.size)
        yield from ((epoch, index) for index in indices)

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation and sample-randomness epoch."""

        epoch_value = cast(object, epoch)
        if (
            isinstance(epoch_value, bool)
            or not isinstance(epoch_value, int)
            or epoch_value < 0
        ):
            raise ValueError("sampler epoch must be a non-negative integer")
        self.epoch = epoch


class CyclingIndexPool:
    """Draw shuffled indices indefinitely while retaining every sample."""

    def __init__(
        self,
        indices: CompactSamplerIndex,
        rng: random.Random,
    ) -> None:
        if not indices:
            raise ValueError("cycling index pools must not be empty")
        self._source = indices
        self._rng = rng
        self._current = array(indices.typecode)
        self._offset = 0

    def _refill(self) -> None:
        self._current = self._source.mutable_copy()
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


class CompactSamplerIndex(Sequence[int]):
    """Packed sampler indices with controlled mutable copies for shuffling."""

    __slots__ = ("_values",)

    def __init__(self, *, maximum: int) -> None:
        if maximum < 0:
            raise ValueError("sampler index maximum must be non-negative")
        if maximum < 2**32:
            typecode = "I"
        elif maximum < 2**64:
            typecode = "Q"
        else:
            raise ValueError("sampler index maximum exceeds 64-bit storage")
        self._values = array(typecode)

    @property
    def typecode(self) -> str:
        """Return the selected packed-array type code."""

        return self._values.typecode

    @property
    def storage_bytes(self) -> int:
        """Return packed payload bytes, excluding constant object overhead."""

        return len(self._values) * self._values.itemsize

    def append(self, value: int) -> None:
        """Append one validated non-negative dataset index."""

        if value < 0:
            raise ValueError("sampler indices must be non-negative")
        try:
            self._values.append(value)
        except OverflowError as exc:
            raise ValueError("sampler index exceeds selected storage") from exc

    def mutable_copy(self) -> array[int]:
        """Return an isolated packed copy for one ephemeral shuffle."""

        return array(self._values.typecode, self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int | slice) -> int | array[int]:
        return self._values[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)


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
        if len(dataset.bucket_codes) != len(dataset):
            raise ValueError("bucket metadata length must match dataset length")
        if len(dataset.source_codes) != len(dataset):
            raise ValueError("source metadata length must match dataset length")
        if not dataset.bucket_names or not dataset.source_names:
            raise ValueError("bucket and source codebooks must not be empty")
        if len(set(dataset.bucket_names)) != len(dataset.bucket_names):
            raise ValueError("bucket metadata names must be unique")
        if len(set(dataset.source_names)) != len(dataset.source_names):
            raise ValueError("source metadata names must be unique")
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

        bucket_count = len(dataset.bucket_names)
        source_count = len(dataset.source_names)
        for name in dataset.bucket_names:
            bucket_policy.resolve(name)
        counts = [0] * bucket_count
        maximum_index = len(dataset) - 1
        self._bucket_indices: dict[int, CompactSamplerIndex] | None = None
        self._source_bucket_indices: (
            dict[tuple[int, int], CompactSamplerIndex] | None
        ) = None
        if self.source_weights is None:
            self._bucket_indices = {}
        else:
            self._source_bucket_indices = {}
        for index, (source_code, bucket_code) in enumerate(
            zip(dataset.source_codes, dataset.bucket_codes, strict=True)
        ):
            if source_code < 0 or source_code >= source_count:
                raise ValueError("source metadata code is out of range")
            if bucket_code < 0 or bucket_code >= bucket_count:
                raise ValueError("bucket metadata code is out of range")
            counts[bucket_code] += 1
            if self._bucket_indices is not None:
                selected = self._bucket_indices.setdefault(
                    bucket_code,
                    CompactSamplerIndex(maximum=maximum_index),
                )
            else:
                assert self._source_bucket_indices is not None
                selected = self._source_bucket_indices.setdefault(
                    (source_code, bucket_code),
                    CompactSamplerIndex(maximum=maximum_index),
                )
            selected.append(index)

        if self.source_weights is not None:
            for weight in self.source_weights.values():
                weight_value = cast(object, weight)
                if isinstance(weight_value, bool) or not isinstance(
                    weight_value,
                    (int, float),
                ):
                    raise TypeError(
                        "source weights must be finite positive numbers"
                    )
                try:
                    is_finite = math.isfinite(weight_value)
                except OverflowError:
                    is_finite = False
                if not is_finite or weight_value <= 0:
                    raise ValueError(
                        "source weights must be finite positive numbers"
                    )
            known_sources = set(dataset.source_names)
            unknown = sorted(set(self.source_weights) - known_sources)
            if unknown:
                raise ValueError(
                    "source weights contain unknown source(s): "
                    + ", ".join(unknown)
                )
            present_codes = set(dataset.source_codes)
            present_sources = {
                dataset.source_names[code]
                for code in present_codes
            }
            missing = sorted(set(self.source_weights) - present_sources)
            if missing:
                raise ValueError(
                    "weighted source(s) have no samples after splitting: "
                    + ", ".join(missing)
                )

        natural_steps = sum(
            self._batch_count(
                dataset.bucket_names[bucket_code],
                count,
                drop_last=drop_last,
            )
            for bucket_code, count in enumerate(counts)
            if count
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

    def _natural_batches(
        self,
        rng: random.Random,
    ) -> Iterator[list[int]]:
        assert self._bucket_indices is not None
        bucket_codes = {
            name: code
            for code, name in enumerate(self.dataset.bucket_names)
        }
        batch_ends: list[int] = []
        bucket_batches: list[tuple[Sequence[int], int]] = []
        batch_total = 0
        for bucket in self.bucket_policy.buckets:
            packed = self._bucket_indices.get(bucket_codes[bucket.name])
            if packed is None:
                continue
            if self.shuffle:
                indices: Sequence[int] = packed.mutable_copy()
                rng.shuffle(indices)
            else:
                indices = packed
            batch_size = self._batch_size(bucket.name)
            batch_count = self._batch_count(
                bucket.name,
                len(indices),
                drop_last=self.drop_last,
            )
            if batch_count <= 0:
                continue
            batch_total += batch_count
            batch_ends.append(batch_total)
            bucket_batches.append((indices, batch_size))

        if batch_total <= 0:
            return
        if self.shuffle:
            typecode = "I" if batch_total <= 2**32 else "Q"
            batch_order: Sequence[int] = array(
                typecode,
                range(batch_total),
            )
            rng.shuffle(batch_order)
        else:
            batch_order = range(batch_total)

        for batch_code in batch_order:
            bucket_position = bisect_right(batch_ends, batch_code)
            previous_end = (
                0 if bucket_position == 0 else batch_ends[bucket_position - 1]
            )
            local_batch = batch_code - previous_end
            indices, batch_size = bucket_batches[bucket_position]
            offset = local_batch * batch_size
            yield list(indices[offset : offset + batch_size])

    def _unweighted_batches(self, rng: random.Random) -> Iterator[list[int]]:
        target_steps = len(self)
        yielded = 0
        while yielded < target_steps:
            cycle_start = yielded
            for batch in self._natural_batches(rng):
                if yielded >= target_steps:
                    return
                yielded += 1
                yield batch
            if yielded == cycle_start:
                raise RuntimeError("batch sampler produced an empty epoch")

    def _weighted_batches(self, rng: random.Random) -> Iterator[list[int]]:
        assert self.source_weights is not None
        assert self._source_bucket_indices is not None
        source_codes = {
            name: code
            for code, name in enumerate(self.dataset.source_names)
        }
        bucket_codes = {
            name: code
            for code, name in enumerate(self.dataset.bucket_names)
        }
        sources = [source_codes[name] for name in self.source_weights]
        weights = [
            self.source_weights[self.dataset.source_names[source_code]]
            for source_code in sources
        ]
        pools = {
            key: CyclingIndexPool(indices, rng)
            for key, indices in self._source_bucket_indices.items()
        }
        source_buckets: dict[int, list[int]] = {}
        source_bucket_weights: dict[int, list[int]] = {}
        for source_code in sources:
            selected_bucket_codes: list[int] = []
            batch_counts: list[int] = []
            for bucket in self.bucket_policy.buckets:
                bucket_code = bucket_codes[bucket.name]
                indices = self._source_bucket_indices.get(
                    (source_code, bucket_code)
                )
                if not indices:
                    continue
                selected_bucket_codes.append(bucket_code)
                batch_counts.append(
                    max(
                        1,
                        math.ceil(len(indices) / self._batch_size(bucket.name)),
                    )
                )
            source_buckets[source_code] = selected_bucket_codes
            source_bucket_weights[source_code] = batch_counts

        for _ in range(len(self)):
            source_code = rng.choices(sources, weights=weights, k=1)[0]
            bucket_code = rng.choices(
                source_buckets[source_code],
                weights=source_bucket_weights[source_code],
                k=1,
            )[0]
            bucket_name = self.dataset.bucket_names[bucket_code]
            yield pools[(source_code, bucket_code)].draw(
                self._batch_size(bucket_name)
            )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        if self.source_weights is None:
            yield from self._unweighted_batches(rng)
        else:
            yield from self._weighted_batches(rng)


__all__ = [
    "BucketedDataset",
    "CompactSamplerIndex",
    "CyclingIndexPool",
    "EpochRandomSampler",
    "EpochTaggedIndexSampler",
    "MixtureBatchSampler",
    "ResolutionBucket",
    "ResolutionBucketPolicy",
]
