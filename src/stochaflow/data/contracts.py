"""Public dataset-factory contracts and image-pipeline helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Hashable, Sequence, Sized
from dataclasses import dataclass
import math
from typing import Any, Literal, cast

from torch.utils.data import Dataset

from stochaflow.utils.config import ResolutionBucketConfig

DatasetRole = Literal["train", "eval"]


@dataclass(frozen=True, slots=True)
class ResolutionBucket:
    """Resolved spatial bucket used by datasets and batch samplers."""

    name: str
    height: int
    width: int

    @property
    def pixels(self) -> int:
        """Return the number of spatial pixels in one sample."""

        return self.height * self.width


class ResolutionBucketPolicy:
    """Assign images to explicit buckets and derive pixel-budget batch sizes."""

    def __init__(
        self,
        buckets: Sequence[ResolutionBucketConfig | ResolutionBucket],
        *,
        base_bucket: str,
        dynamic_batch_size: bool,
    ) -> None:
        self._buckets = tuple(
            bucket
            if isinstance(bucket, ResolutionBucket)
            else ResolutionBucket(bucket.name, bucket.height, bucket.width)
            for bucket in buckets
        )
        if not self._buckets:
            raise ValueError("resolution bucket policy requires at least one bucket")
        self._by_name = {bucket.name: bucket for bucket in self._buckets}
        if len(self._by_name) != len(self._buckets):
            raise ValueError("resolution bucket names must be unique")
        try:
            self._base_bucket = self._by_name[base_bucket]
        except KeyError as exc:
            raise ValueError(
                f"unknown base bucket '{base_bucket}'"
            ) from exc
        self.dynamic_batch_size = dynamic_batch_size

    @property
    def buckets(self) -> tuple[ResolutionBucket, ...]:
        """Return configured buckets in declaration order."""

        return self._buckets

    @property
    def base_bucket(self) -> ResolutionBucket:
        """Return the bucket defining the dynamic batch pixel budget."""

        return self._base_bucket

    def resolve(self, name: str) -> ResolutionBucket:
        """Resolve one bucket by name."""

        try:
            return self._by_name[name]
        except KeyError as exc:
            available = ", ".join(self._by_name)
            raise ValueError(
                f"unknown resolution bucket '{name}'. Available: {available}"
            ) from exc

    def select(self, width: int, height: int) -> ResolutionBucket:
        """Choose the nearest aspect ratio, then nearest area, then config order."""

        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        image_ratio = width / height
        image_area = width * height

        def distance(bucket: ResolutionBucket) -> tuple[float, float]:
            bucket_ratio = bucket.width / bucket.height
            ratio_distance = abs(math.log(image_ratio / bucket_ratio))
            area_distance = abs(math.log(image_area / bucket.pixels))
            return ratio_distance, area_distance

        return min(self._buckets, key=distance)

    def batch_size(self, bucket_name: str, *, base_batch_size: int) -> int:
        """Return a fixed or pixel-budget-scaled batch size for a bucket."""

        if base_batch_size <= 0:
            raise ValueError("base batch size must be positive")
        if not self.dynamic_batch_size:
            return base_batch_size
        bucket = self.resolve(bucket_name)
        return max(
            1,
            math.floor(
                base_batch_size
                * self.base_bucket.pixels
                / bucket.pixels
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetFactoryContext:
    """Dependencies injected into one configured dataset factory."""

    source_id: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    """Request one native dataset split with role-specific preprocessing."""

    native_split: str
    role: DatasetRole
    seed: int

    def __post_init__(self) -> None:
        if not self.native_split:
            raise ValueError("native dataset split must not be empty")
        if self.role not in {"train", "eval"}:
            raise ValueError("dataset role must be 'train' or 'eval'")


@dataclass(frozen=True, slots=True)
class DatasetView:
    """A materialized dataset plus stable keys and optional batch metadata."""

    source_id: str
    dataset: Dataset[Any]
    sample_keys: Sequence[Hashable]
    batch_metadata: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        size = len(cast(Sized, self.dataset))
        if len(self.sample_keys) != size:
            raise ValueError("dataset sample_keys length must match dataset length")
        if self.batch_metadata is not None and len(self.batch_metadata) != size:
            raise ValueError(
                "dataset batch_metadata length must match dataset length"
            )
        if len(set(self.sample_keys)) != size:
            raise ValueError(
                f"dataset view '{self.source_id}' sample_keys must be unique"
            )

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


class DatasetFactory(ABC):
    """Class-only extension point for building dataset views."""

    def __init__(self, context: DatasetFactoryContext) -> None:
        self.context = context

    @abstractmethod
    def build(self, request: DatasetBuildRequest) -> DatasetView:
        """Build a role-specific view for one native split."""


class DatasetMixture(Dataset[Any]):
    """Indexable union preserving source, key, and optional batch metadata."""

    def __init__(self, views: Sequence[DatasetView]) -> None:
        if not views:
            raise ValueError("dataset mixture requires at least one view")
        self.views = tuple(views)
        self._ends: list[int] = []
        total = 0
        source_ids: list[str] = []
        sample_keys: list[tuple[str, Hashable]] = []
        batch_metadata: list[Any] = []
        has_batch_metadata = all(
            view.batch_metadata is not None for view in self.views
        )
        for view in self.views:
            total += len(view)
            self._ends.append(total)
            source_ids.extend([view.source_id] * len(view))
            sample_keys.extend(
                (view.source_id, sample_key) for sample_key in view.sample_keys
            )
            if view.batch_metadata is not None:
                batch_metadata.extend(view.batch_metadata)
        self.source_ids = tuple(source_ids)
        self.sample_keys = tuple(sample_keys)
        self.batch_metadata = tuple(batch_metadata) if has_batch_metadata else None

    def __len__(self) -> int:
        return self._ends[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        view_index = bisect_right(self._ends, index)
        start = 0 if view_index == 0 else self._ends[view_index - 1]
        return view_index, index - start

    def __getitem__(self, index: int) -> Any:
        view_index, local_index = self._locate(index)
        return self.views[view_index][local_index]

    def source_for(self, index: int) -> str:
        """Return the source id assigned to a global mixture index."""

        return self.source_ids[index]


class DatasetSelection(Dataset[Any]):
    """Metadata-preserving indexed selection from a dataset mixture."""

    def __init__(self, dataset: DatasetMixture, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        if any(index < 0 or index >= len(dataset) for index in self.indices):
            raise IndexError("dataset selection contains an out-of-range index")
        self.source_ids = tuple(dataset.source_ids[index] for index in self.indices)
        self.sample_keys = tuple(dataset.sample_keys[index] for index in self.indices)
        self.batch_metadata = (
            tuple(dataset.batch_metadata[index] for index in self.indices)
            if dataset.batch_metadata is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def source_for(self, index: int) -> str:
        """Return the source assigned to a selection-local index."""

        return self.source_ids[index]


__all__ = [
    "DatasetBuildRequest",
    "DatasetFactory",
    "DatasetFactoryContext",
    "DatasetMixture",
    "DatasetRole",
    "DatasetSelection",
    "DatasetView",
    "ResolutionBucket",
    "ResolutionBucketPolicy",
]
