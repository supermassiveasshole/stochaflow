"""Dataset implementations and artifact-to-Dataset adaptation."""

from __future__ import annotations

import hashlib
import io
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, overload, runtime_checkable

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from stochaflow.data.artifact_io import read_regular_file
from stochaflow.data.artifacts import DataArtifact
from stochaflow.data.image_contracts import (
    ClassLabeledImageFileRecord,
    ImageDimensions,
    ImageDimensionTable,
    ImageFilePair,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    PairedImageFolderArtifactPayload,
    TorchvisionImageArtifactPayload,
)
from stochaflow.data.samplers import ResolutionBucketPolicy
from stochaflow.data.transforms import (
    GeneratedSuperResolutionTransform,
    ImageTransform,
    PairedSuperResolutionTransform,
    extract_image,
)


@dataclass(frozen=True, slots=True)
class ImageDatasetPartitions:
    """Native Dataset partitions before recipe-level repartitioning."""

    train: Dataset[Any]
    validation: Dataset[Any] | None = None
    test: Dataset[Any] | None = None


@runtime_checkable
class ImageDimensionProvider(Protocol):
    """Narrow metadata capability used by the multi-resolution recipe."""

    def image_dimensions(self, index: int) -> ImageDimensions:
        """Return trusted dimensions without reading or decoding a sample."""
        ...


def dataset_image_dimensions(
    dataset: Dataset[Any],
    index: int,
) -> ImageDimensions:
    """Resolve dimensions through metadata-preserving dataset wrappers."""

    if isinstance(dataset, Subset):
        return dataset_image_dimensions(
            dataset.dataset,
            int(dataset.indices[index]),
        )
    if isinstance(dataset, ImageDimensionProvider):
        return dataset.image_dimensions(index)
    raise TypeError(
        "multi-resolution datasets require authenticated image dimensions"
    )


class TorchvisionMetadataDataset(Dataset[Any]):
    """Attach artifact-authenticated dimensions to a torchvision Dataset."""

    def __init__(
        self,
        dataset: Dataset[Any],
        dimensions: ImageDimensionTable,
    ) -> None:
        if len(cast(Sized, dataset)) != len(dimensions):
            raise ValueError(
                "torchvision dimension metadata length does not match dataset"
            )
        self.dataset = dataset
        self.dimensions = dimensions

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]

    def image_dimensions(self, index: int) -> ImageDimensions:
        """Return dimensions loaded from the managed artifact sidecar."""

        return self.dimensions[index]


def _verified_image(root: Path, record: ImageFileRecord) -> Image.Image:
    path = root / record.path
    try:
        encoded, metadata = read_regular_file(
            root,
            record.path,
            label="artifact image",
        )
    except ValueError as exc:
        raise ValueError(f"cannot read artifact image: {path}: {exc}") from exc
    except OSError as exc:
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

    def image_dimensions(self, index: int) -> ImageDimensions:
        """Return dimensions authenticated by the reference inventory."""

        return self.records[index].dimensions


class ClassLabeledImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Transform authenticated images and emit their integer class labels."""

    def __init__(
        self,
        *,
        roots: Mapping[str, Path],
        records: Sequence[ClassLabeledImageFileRecord],
        transform: ImageTransform,
        seed: int,
    ) -> None:
        seed_value = cast(object, seed)
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError("class-labeled image dataset seed must be an integer")
        self.records = tuple(records)
        if any(
            not isinstance(record, ClassLabeledImageFileRecord)
            for record in cast(Sequence[object], self.records)
        ):
            raise TypeError(
                "class-labeled image dataset records must contain "
                "ClassLabeledImageFileRecord"
            )
        self.images = ImageFolderDataset(
            roots,
            tuple(record.image for record in self.records),
        )
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> tuple[torch.Tensor, int]:
        epoch = 0
        sample_index: object = index
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(
                    "epoch-tagged image index must contain two values"
                )
            epoch, sample_index = index
        epoch_value = cast(object, epoch)
        if (
            isinstance(epoch_value, bool)
            or not isinstance(epoch_value, int)
            or epoch_value < 0
        ):
            raise ValueError("sample epoch must be a non-negative integer")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("sample index must be an integer")
        if sample_index < 0 or sample_index >= len(self.records):
            raise IndexError(sample_index)

        record = self.records[sample_index]
        image_record = record.image
        identity = (
            f"stochaflow.class-labeled-image.sample.v1\0{self.seed}\0"
            f"{epoch}\0{image_record.tree}\0{image_record.path}\0"
            f"{image_record.sha256}"
        ).encode()
        random_seed = int.from_bytes(
            hashlib.sha256(identity).digest()[:8],
            byteorder="little",
        )
        return (
            self.transform(
                self.images[sample_index],
                random_seed=random_seed,
            ),
            record.class_label,
        )


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

    def image_dimensions(self, index: int) -> ImageDimensions:
        """Return authenticated high-resolution dimensions for this pair."""

        return self.pairs[index].high_resolution.dimensions


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
        ends: list[int] = []
        source_names: list[str] = []
        total = 0
        for source_id, dataset in self.datasets:
            dataset_size = len(cast(Sized, dataset))
            total += dataset_size
            ends.append(total)
            source_names.append(source_id)
        if total <= 0:
            raise ValueError("multi-resolution sources contain no samples")
        if len(source_names) != len(set(source_names)):
            raise ValueError("multi-resolution source names must be unique")
        self.ends = tuple(ends)
        self.source_names = tuple(source_names)

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        source_index = bisect_right(self.ends, index)
        start = 0 if source_index == 0 else self.ends[source_index - 1]
        return self.datasets[source_index][1][index - start]

    def source_code(self, index: int) -> int:
        """Return the compact source code for one concatenated sample."""

        if index < 0:
            index += len(self)
        return bisect_right(self.ends, index)

    def image_dimensions(self, index: int) -> ImageDimensions:
        """Resolve dimensions without reading the concatenated sample."""

        source_index = self.source_code(index)
        start = 0 if source_index == 0 else self.ends[source_index - 1]
        return dataset_image_dimensions(
            self.datasets[source_index][1],
            index - start,
        )


def dataset_source_names(dataset: Dataset[Any]) -> tuple[str, ...]:
    """Resolve the stable source codebook through partition wrappers."""

    if isinstance(dataset, Subset):
        return dataset_source_names(dataset.dataset)
    if isinstance(dataset, SourceConcatDataset):
        return dataset.source_names
    raise TypeError("multi-resolution dataset lost source metadata")


def dataset_source_code(dataset: Dataset[Any], index: int) -> int:
    """Resolve one source code through partition wrappers."""

    if isinstance(dataset, Subset):
        return dataset_source_code(
            dataset.dataset,
            int(dataset.indices[index]),
        )
    if isinstance(dataset, SourceConcatDataset):
        return dataset.source_code(index)
    raise TypeError("multi-resolution dataset lost source metadata")


class CompactIndexSequence(Sequence[int]):
    """Read-only view over a packed unsigned integer array."""

    __slots__ = ("_values",)

    def __init__(self, values: Iterable[int], *, maximum: int) -> None:
        if maximum < 0:
            raise ValueError("compact index maximum must be non-negative")
        if maximum < 2**8:
            typecode = "B"
        elif maximum < 2**16:
            typecode = "H"
        elif maximum < 2**32:
            typecode = "I"
        elif maximum < 2**64:
            typecode = "Q"
        else:
            raise ValueError("compact index maximum exceeds 64-bit storage")
        try:
            self._values = array(typecode, values)
        except OverflowError as exc:
            raise ValueError(
                "compact index value exceeds selected storage"
            ) from exc

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self._values[index])
        return self._values[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)

    @property
    def storage_bytes(self) -> int:
        """Return packed payload bytes, excluding constant object overhead."""

        return len(self._values) * self._values.itemsize


def compact_unsigned(
    values: Iterable[int],
    *,
    maximum: int,
) -> CompactIndexSequence:
    """Store non-negative codes in the narrowest portable unsigned array."""

    return CompactIndexSequence(values, maximum=maximum)


class CodedLabelSequence(Sequence[str]):
    """Read-only string labels backed by compact integer codes."""

    __slots__ = ("_codes", "_labels")

    def __init__(
        self,
        codes: Sequence[int],
        labels: Sequence[str],
    ) -> None:
        self._codes = codes
        self._labels = tuple(labels)

    def __len__(self) -> int:
        return len(self._codes)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            return tuple(
                self._labels[self._codes[position]]
                for position in range(*index.indices(len(self)))
            )
        return self._labels[self._codes[index]]

    def __iter__(self) -> Iterator[str]:
        return (
            self._labels[code]
            for code in self._codes
        )


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
        self.source_names = dataset_source_names(dataset)
        self.source_codes = compact_unsigned(
            (
                dataset_source_code(dataset, index)
                for index in range(dataset_size)
            ),
            maximum=len(self.source_names) - 1,
        )
        self.bucket_names = tuple(bucket.name for bucket in policy.buckets)
        bucket_codes = {
            bucket.name: index
            for index, bucket in enumerate(policy.buckets)
        }
        self.bucket_codes = compact_unsigned(
            (
                bucket_codes[
                    policy.select(
                        dimensions.width,
                        dimensions.height,
                    ).name
                ]
                for index in range(dataset_size)
                for dimensions in (
                    dataset_image_dimensions(dataset, index),
                )
            ),
            maximum=len(self.bucket_names) - 1,
        )
        self.source_ids = CodedLabelSequence(
            self.source_codes,
            self.source_names,
        )
        self.bucket_ids = CodedLabelSequence(
            self.bucket_codes,
            self.bucket_names,
        )
        self.transforms = tuple(
            ImageTransform(
                (bucket.height, bucket.width),
                role=role,
                channels=channels,
                normalize=normalize,
                random_horizontal_flip=random_horizontal_flip,
            )
            for bucket in policy.buckets
        )

    def __len__(self) -> int:
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image = extract_image(self.dataset[index])
        return self.transforms[self.bucket_codes[index]](image), {}


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
                train=TorchvisionMetadataDataset(
                    datasets.MNIST(
                        root,
                        train=True,
                        transform=None,
                        download=False,
                    ),
                    payload.train_dimensions,
                ),
                test=TorchvisionMetadataDataset(
                    datasets.MNIST(
                        root,
                        train=False,
                        transform=None,
                        download=False,
                    ),
                    cast(
                        ImageDimensionTable,
                        payload.test_dimensions,
                    ),
                ),
            )
        if payload.dataset == "cifar10":
            return ImageDatasetPartitions(
                train=TorchvisionMetadataDataset(
                    datasets.CIFAR10(
                        root,
                        train=True,
                        transform=None,
                        download=False,
                    ),
                    payload.train_dimensions,
                ),
                test=TorchvisionMetadataDataset(
                    datasets.CIFAR10(
                        root,
                        train=False,
                        transform=None,
                        download=False,
                    ),
                    cast(
                        ImageDimensionTable,
                        payload.test_dimensions,
                    ),
                ),
            )
        return ImageDatasetPartitions(
            train=TorchvisionMetadataDataset(
                datasets.Flowers102(
                    root,
                    split="train",
                    transform=None,
                    download=False,
                ),
                payload.train_dimensions,
            ),
            validation=TorchvisionMetadataDataset(
                datasets.Flowers102(
                    root,
                    split="val",
                    transform=None,
                    download=False,
                ),
                cast(
                    ImageDimensionTable,
                    payload.validation_dimensions,
                ),
            ),
            test=TorchvisionMetadataDataset(
                datasets.Flowers102(
                    root,
                    split="test",
                    transform=None,
                    download=False,
                ),
                cast(
                    ImageDimensionTable,
                    payload.test_dimensions,
                ),
            ),
        )


__all__ = [
    "ClassLabeledImageDataset",
    "CodedLabelSequence",
    "CompactIndexSequence",
    "GeneratedSuperResolutionDataset",
    "ImageDatasetFactory",
    "ImageDatasetPartitions",
    "ImageDimensionProvider",
    "ImageFolderDataset",
    "ImageRecipeDataset",
    "MultiResolutionDataset",
    "PairedImageFolderDataset",
    "PairedSuperResolutionDataset",
    "SourceConcatDataset",
    "TorchvisionMetadataDataset",
    "combine_image_datasets",
    "compact_unsigned",
    "dataset_image_dimensions",
    "dataset_source_code",
    "dataset_source_names",
]
