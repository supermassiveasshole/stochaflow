"""Dataset implementations and artifact-to-Dataset adaptation."""

from __future__ import annotations

import hashlib
import io
from bisect import bisect_right
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from stochaflow.data.artifact_io import read_regular_file
from stochaflow.data.artifacts import DataArtifact
from stochaflow.data.samplers import ResolutionBucketPolicy
from stochaflow.data.sources import (
    ImageFilePair,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    PairedImageFolderArtifactPayload,
    TorchvisionImageArtifactPayload,
)
from stochaflow.data.transforms import (
    GeneratedSuperResolutionTransform,
    ImageTransform,
    PairedSuperResolutionTransform,
    extract_image,
    image_size,
)


@dataclass(frozen=True, slots=True)
class ImageDatasetPartitions:
    """Native Dataset partitions before recipe-level repartitioning."""

    train: Dataset[Any]
    validation: Dataset[Any] | None = None
    test: Dataset[Any] | None = None


def _verified_image(root: Path, record: ImageFileRecord) -> Image.Image:
    path = root / record.path
    try:
        encoded, metadata = read_regular_file(
            root,
            record.path,
            label="artifact image",
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read artifact image: {path}") from exc
    if metadata.st_size != record.size_bytes or len(encoded) != record.size_bytes:
        raise ValueError(f"artifact image size changed: {path}")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != record.sha256:
        raise ValueError(f"artifact image content changed: {path}")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            return image.copy()
    except Exception as exc:
        raise ValueError(f"cannot decode artifact image: {path}") from exc


class ImageFolderDataset(Dataset[Image.Image]):
    """Manifest-ordered image dataset with hash-on-read verification."""

    def __init__(
        self,
        roots: Mapping[str, Path],
        records: Sequence[ImageFileRecord],
    ) -> None:
        self.roots = dict(roots)
        self.records = tuple(records)
        if not self.records:
            raise ValueError("image dataset records must not be empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Image.Image:
        record = self.records[index]
        return _verified_image(self.roots[record.tree], record)


class PairedImageFolderDataset(
    Dataset[tuple[Image.Image, Image.Image]]
):
    """Manifest-ordered HR/LR dataset with hash-on-read verification."""

    def __init__(
        self,
        roots: Mapping[str, Path],
        pairs: Sequence[ImageFilePair],
    ) -> None:
        self.roots = dict(roots)
        self.pairs = tuple(pairs)
        if not self.pairs:
            raise ValueError("paired image dataset records must not be empty")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image]:
        pair = self.pairs[index]
        return (
            _verified_image(
                self.roots[pair.high_resolution.tree],
                pair.high_resolution,
            ),
            _verified_image(
                self.roots[pair.low_resolution.tree],
                pair.low_resolution,
            ),
        )


class ImageRecipeDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """Apply an image recipe transform and emit the standard condition dict."""

    def __init__(self, dataset: Dataset[Any], transform: ImageTransform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.transform(extract_image(self.dataset[index])), {}


class GeneratedSuperResolutionDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]
):
    """Generate an aligned low-resolution condition from each source image."""

    def __init__(
        self,
        dataset: Dataset[Any],
        transform: GeneratedSuperResolutionTransform,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        high, low = self.transform(extract_image(self.dataset[index]))
        return high, {"low_res": low}


class PairedSuperResolutionDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]
):
    """Transform an explicit high/low-resolution source pair."""

    def __init__(
        self,
        dataset: Dataset[Any],
        transform: PairedSuperResolutionTransform,
    ) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise TypeError("paired SR sources must return (high_res, low_res)")
        high, low = self.transform((sample[0], sample[1]))
        return high, {"low_res": low}


class SourceConcatDataset(Dataset[Any]):
    """Concatenate named datasets while preserving source identity."""

    def __init__(self, datasets: Sequence[tuple[str, Dataset[Any]]]) -> None:
        if not datasets:
            raise ValueError("multi-resolution sources must not be empty")
        self.datasets = tuple(datasets)
        self.ends: list[int] = []
        self.source_ids: list[str] = []
        total = 0
        for source_id, dataset in self.datasets:
            dataset_size = len(cast(Sized, dataset))
            total += dataset_size
            self.ends.append(total)
            self.source_ids.extend([source_id] * dataset_size)
        if total <= 0:
            raise ValueError("multi-resolution sources contain no samples")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        source_index = bisect_right(self.ends, index)
        start = 0 if source_index == 0 else self.ends[source_index - 1]
        return self.datasets[source_index][1][index - start]


def _source_id(dataset: Dataset[Any], index: int) -> str:
    if isinstance(dataset, Subset):
        return _source_id(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, SourceConcatDataset):
        return dataset.source_ids[index]
    raise TypeError("multi-resolution dataset lost source metadata")


class MultiResolutionDataset(
    Dataset[tuple[torch.Tensor, dict[str, Any]]]
):
    """Apply deterministic resolution buckets to a named source mixture."""

    def __init__(
        self,
        dataset: Dataset[Any],
        policy: ResolutionBucketPolicy,
        *,
        role: str,
        channels: int,
        normalize: bool,
        random_horizontal_flip: bool,
    ) -> None:
        self.dataset = dataset
        dataset_size = len(cast(Sized, dataset))
        self.source_ids = tuple(
            _source_id(dataset, index) for index in range(dataset_size)
        )
        self.bucket_ids: list[str] = []
        for index in range(dataset_size):
            width, height = image_size(extract_image(dataset[index]))
            self.bucket_ids.append(policy.select(width, height).name)
        self.transforms = {
            bucket.name: ImageTransform(
                (bucket.height, bucket.width),
                role=role,
                channels=channels,
                normalize=normalize,
                random_horizontal_flip=random_horizontal_flip,
            )
            for bucket in policy.buckets
        }

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image = extract_image(self.dataset[index])
        return self.transforms[self.bucket_ids[index]](image), {}


def combine_image_datasets(
    sources: Sequence[tuple[str, ImageDatasetPartitions]],
    role: str,
) -> Dataset[Any] | None:
    """Combine one partition across sources while enforcing role parity."""

    selected = [
        (source_id, getattr(source, role)) for source_id, source in sources
    ]
    present = [
        (source_id, dataset)
        for source_id, dataset in selected
        if dataset is not None
    ]
    if not present:
        return None
    if len(present) != len(selected):
        raise ValueError(
            f"multi-resolution source role '{role}' must be present "
            "for every source or none"
        )
    return SourceConcatDataset(
        cast(list[tuple[str, Dataset[Any]]], present)
    )


class ImageDatasetFactory:
    """Construct Dataset partitions from public image payload contracts."""

    def build(
        self,
        artifact: DataArtifact[Any],
    ) -> ImageDatasetPartitions:
        """Build native partitions without dispatching on a source name."""

        payload = artifact.payload
        if isinstance(payload, TorchvisionImageArtifactPayload):
            return self._torchvision(payload)
        if isinstance(payload, ImageFolderArtifactPayload):
            return ImageDatasetPartitions(
                train=ImageFolderDataset(payload.roots, payload.train),
                validation=(
                    ImageFolderDataset(payload.roots, payload.validation)
                    if payload.validation is not None
                    else None
                ),
                test=(
                    ImageFolderDataset(payload.roots, payload.test)
                    if payload.test is not None
                    else None
                ),
            )
        if isinstance(payload, PairedImageFolderArtifactPayload):
            return ImageDatasetPartitions(
                train=PairedImageFolderDataset(payload.roots, payload.train),
                validation=(
                    PairedImageFolderDataset(payload.roots, payload.validation)
                    if payload.validation is not None
                    else None
                ),
                test=(
                    PairedImageFolderDataset(payload.roots, payload.test)
                    if payload.test is not None
                    else None
                ),
            )
        raise TypeError(
            "image dataset factory requires a public image artifact payload"
        )

    def _torchvision(
        self,
        payload: TorchvisionImageArtifactPayload,
    ) -> ImageDatasetPartitions:
        root = str(payload.root)
        if payload.dataset == "mnist":
            return ImageDatasetPartitions(
                train=datasets.MNIST(
                    root,
                    train=True,
                    transform=None,
                    download=False,
                ),
                test=datasets.MNIST(
                    root,
                    train=False,
                    transform=None,
                    download=False,
                ),
            )
        if payload.dataset == "cifar10":
            return ImageDatasetPartitions(
                train=datasets.CIFAR10(
                    root,
                    train=True,
                    transform=None,
                    download=False,
                ),
                test=datasets.CIFAR10(
                    root,
                    train=False,
                    transform=None,
                    download=False,
                ),
            )
        return ImageDatasetPartitions(
            train=datasets.Flowers102(
                root,
                split="train",
                transform=None,
                download=False,
            ),
            validation=datasets.Flowers102(
                root,
                split="val",
                transform=None,
                download=False,
            ),
            test=datasets.Flowers102(
                root,
                split="test",
                transform=None,
                download=False,
            ),
        )


__all__ = [
    "GeneratedSuperResolutionDataset",
    "ImageDatasetFactory",
    "ImageDatasetPartitions",
    "ImageFolderDataset",
    "ImageRecipeDataset",
    "MultiResolutionDataset",
    "PairedImageFolderDataset",
    "PairedSuperResolutionDataset",
    "SourceConcatDataset",
    "combine_image_datasets",
]
