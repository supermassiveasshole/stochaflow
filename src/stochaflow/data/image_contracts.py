"""Formal payload and registry contracts for built-in image recipes."""

from __future__ import annotations

import unicodedata
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from struct import iter_unpack, unpack_from
from types import MappingProxyType
from typing import Any, Literal, cast, overload

from stochaflow.data.artifact_io import canonical_directory
from stochaflow.data.artifacts import DataSource
from stochaflow.utils.registry import Registry

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
IMAGE_FILE_RECORD_FIELDS = frozenset(
    {"tree", "path", "size_bytes", "sha256", "width", "height"}
)


def validate_relative_image_path(value: str, *, path: str) -> str:
    """Validate one portable, normalized inventory path."""

    if not value or value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{path} must be a non-empty NFC path")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{path} contains an invalid character")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise ValueError(f"{path} must be a relative POSIX path")
    for part in pure.parts:
        if part in {".", ".."} or part.endswith((" ", ".")) or ":" in part:
            raise ValueError(f"{path} contains an unsafe path component")
        stem = part.split(".", 1)[0].upper()
        if stem in WINDOWS_RESERVED_NAMES:
            raise ValueError(f"{path} contains a Windows-reserved name")
    if pure.as_posix() != value:
        raise ValueError(f"{path} must be a normalized POSIX path")
    return value


def strict_record_mapping(
    value: object,
    *,
    path: str,
) -> dict[str, Any]:
    """Parse the exact canonical fields of one image inventory record."""

    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{path} field names must be strings")
    names = cast(set[str], set(raw))
    missing = sorted(IMAGE_FILE_RECORD_FIELDS - names)
    unknown = sorted(names - IMAGE_FILE_RECORD_FIELDS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    """Positive image dimensions in width/height order."""

    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = cast(object, getattr(self, name))
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"image {name} must be a positive integer")


class ImageDimensionTable(Sequence[ImageDimensions]):
    """Compact immutable width/height metadata for one dataset partition."""

    __slots__ = ("_heights", "_widths")

    def __init__(
        self,
        dimensions: Iterable[ImageDimensions | tuple[int, int]],
    ) -> None:
        widths = array("I")
        heights = array("I")
        for value in dimensions:
            dimension = (
                value
                if isinstance(value, ImageDimensions)
                else ImageDimensions(*value)
            )
            try:
                widths.append(dimension.width)
                heights.append(dimension.height)
            except OverflowError as exc:
                raise ValueError(
                    "image dimensions exceed the supported 32-bit range"
                ) from exc
        self._widths = widths.tobytes()
        self._heights = heights.tobytes()

    def __len__(self) -> int:
        return len(self._widths) // 4

    @overload
    def __getitem__(self, index: int) -> ImageDimensions: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ImageDimensions, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> ImageDimensions | tuple[ImageDimensions, ...]:
        if isinstance(index, slice):
            return tuple(
                ImageDimensions(
                    unpack_from("=I", self._widths, position * 4)[0],
                    unpack_from("=I", self._heights, position * 4)[0],
                )
                for position in range(*index.indices(len(self)))
            )
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        return ImageDimensions(
            unpack_from("=I", self._widths, normalized * 4)[0],
            unpack_from("=I", self._heights, normalized * 4)[0],
        )

    def __iter__(self) -> Iterator[ImageDimensions]:
        for (width,), (height,) in zip(
            iter_unpack("=I", self._widths),
            iter_unpack("=I", self._heights),
            strict=True,
        ):
            yield ImageDimensions(width, height)

    def to_pairs(self) -> list[list[int]]:
        """Serialize the table without exposing its mutable arrays."""

        return [[value.width, value.height] for value in self]


@dataclass(frozen=True, slots=True)
class ImageFileRecord:
    """One immutable image record from a canonical artifact inventory."""

    tree: str
    path: str
    size_bytes: int
    sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        tree = cast(object, self.tree)
        if not isinstance(tree, str) or not tree:
            raise ValueError("image file record tree must be non-empty")
        validate_relative_image_path(
            self.path,
            path="image file record.path",
        )
        size_bytes = cast(object, self.size_bytes)
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError(
                "image file record size_bytes must be a non-negative integer"
            )
        digest = cast(object, self.sha256)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "image file record sha256 must be a lowercase SHA-256 digest"
            )
        ImageDimensions(self.width, self.height)

    @property
    def dimensions(self) -> ImageDimensions:
        """Return dimensions authenticated by this inventory record."""

        return ImageDimensions(self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record in canonical inventory form."""

        return {
            "tree": self.tree,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str) -> ImageFileRecord:
        """Parse one strict inventory record."""

        raw = strict_record_mapping(value, path=path)
        return cls(
            tree=raw["tree"],
            path=raw["path"],
            size_bytes=raw["size_bytes"],
            sha256=raw["sha256"],
            width=raw["width"],
            height=raw["height"],
        )


@dataclass(frozen=True, slots=True)
class ClassLabeledImageFileRecord:
    """One authenticated image record with a non-negative class label."""

    image: ImageFileRecord
    class_label: int

    def __post_init__(self) -> None:
        image = cast(object, self.image)
        if not isinstance(image, ImageFileRecord):
            raise TypeError(
                "class-labeled image record image must be ImageFileRecord"
            )
        class_label = cast(object, self.class_label)
        if (
            not isinstance(class_label, int)
            or isinstance(class_label, bool)
            or class_label < 0
        ):
            raise ValueError(
                "class-labeled image record class_label must be "
                "a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class ImageFilePair:
    """One high/low-resolution pair matched by relative stem."""

    high_resolution: ImageFileRecord
    low_resolution: ImageFileRecord

    def __post_init__(self) -> None:
        high_resolution = cast(object, self.high_resolution)
        if not isinstance(high_resolution, ImageFileRecord):
            raise TypeError(
                "image file pair high_resolution must be ImageFileRecord"
            )
        low_resolution = cast(object, self.low_resolution)
        if not isinstance(low_resolution, ImageFileRecord):
            raise TypeError(
                "image file pair low_resolution must be ImageFileRecord"
            )


def immutable_roots(
    roots: Mapping[str, Path],
    *,
    label: str,
) -> Mapping[str, Path]:
    """Validate and freeze a payload root mapping."""

    roots_value = cast(object, roots)
    if not isinstance(roots_value, Mapping):
        raise TypeError(f"{label} roots must be a mapping")
    normalized: dict[str, Path] = {}
    for name, root in roots.items():
        name_value = cast(object, name)
        if not isinstance(name_value, str) or not name_value:
            raise TypeError(f"{label} root names must be non-empty strings")
        normalized[name] = canonical_directory(
            Path(root),
            label=f"{label} root {name}",
        )
    if not normalized:
        raise FileNotFoundError(f"{label} roots must exist")
    return MappingProxyType(normalized)


def immutable_records(
    value: Sequence[ImageFileRecord],
    *,
    label: str,
) -> tuple[ImageFileRecord, ...]:
    """Copy and validate a public record sequence."""

    value_object = cast(object, value)
    if isinstance(value_object, (str, bytes)) or not isinstance(
        value_object,
        Sequence,
    ):
        raise TypeError(f"{label} inventory must be a sequence")
    records = tuple(value)
    if any(
        not isinstance(cast(object, record), ImageFileRecord)
        for record in records
    ):
        raise TypeError(f"{label} inventory must contain ImageFileRecord")
    return records


def immutable_pairs(
    value: Sequence[ImageFilePair],
    *,
    label: str,
) -> tuple[ImageFilePair, ...]:
    """Copy and validate a public pair sequence."""

    value_object = cast(object, value)
    if isinstance(value_object, (str, bytes)) or not isinstance(
        value_object,
        Sequence,
    ):
        raise TypeError(f"{label} inventory must be a sequence")
    pairs = tuple(value)
    if any(
        not isinstance(cast(object, pair), ImageFilePair)
        for pair in pairs
    ):
        raise TypeError(f"{label} inventory must contain ImageFilePair")
    return pairs


def immutable_class_mapping(
    value: Mapping[str, int],
    *,
    label: str,
) -> Mapping[str, int]:
    """Validate and freeze one contiguous class-name-to-label mapping."""

    value_object = cast(object, value)
    if not isinstance(value_object, Mapping):
        raise TypeError(f"{label} class_mapping must be a mapping")
    normalized: dict[str, int] = {}
    for class_name, class_label in value.items():
        class_name_value = cast(object, class_name)
        if not isinstance(class_name_value, str) or not class_name_value:
            raise TypeError(
                f"{label} class_mapping names must be non-empty strings"
            )
        class_label_value = cast(object, class_label)
        if (
            not isinstance(class_label_value, int)
            or isinstance(class_label_value, bool)
            or class_label_value < 0
        ):
            raise ValueError(
                f"{label} class_mapping labels must be non-negative integers"
            )
        normalized[class_name] = class_label
    if not normalized:
        raise ValueError(f"{label} class_mapping must not be empty")
    labels = tuple(normalized.values())
    if len(labels) != len(set(labels)):
        raise ValueError(f"{label} class_mapping labels must be unique")
    if set(labels) != set(range(len(labels))):
        raise ValueError(
            f"{label} class_mapping labels must be contiguous from zero"
        )
    return MappingProxyType(normalized)


def immutable_class_labeled_records(
    value: Sequence[ClassLabeledImageFileRecord],
    *,
    label: str,
) -> tuple[ClassLabeledImageFileRecord, ...]:
    """Copy and validate a public class-labeled record sequence."""

    value_object = cast(object, value)
    if isinstance(value_object, (str, bytes)) or not isinstance(
        value_object,
        Sequence,
    ):
        raise TypeError(f"{label} inventory must be a sequence")
    records = tuple(value)
    if any(
        not isinstance(cast(object, record), ClassLabeledImageFileRecord)
        for record in records
    ):
        raise TypeError(
            f"{label} inventory must contain ClassLabeledImageFileRecord"
        )
    return records


@dataclass(frozen=True, slots=True)
class TorchvisionImageArtifactPayload:
    """Locator and authenticated dimensions for one torchvision dataset."""

    dataset: Literal["mnist", "cifar10", "flowers102"]
    root: Path
    train_dimensions: ImageDimensionTable
    validation_dimensions: ImageDimensionTable | None = None
    test_dimensions: ImageDimensionTable | None = None

    def __post_init__(self) -> None:
        dataset = cast(object, self.dataset)
        if not isinstance(dataset, str) or dataset not in {
            "mnist",
            "cifar10",
            "flowers102",
        }:
            raise ValueError("unsupported torchvision image dataset")
        root = canonical_directory(
            Path(self.root),
            label="torchvision artifact root",
        )
        object.__setattr__(self, "root", root)
        train_dimensions = cast(object, self.train_dimensions)
        if not isinstance(train_dimensions, ImageDimensionTable):
            raise TypeError(
                "torchvision train_dimensions must be ImageDimensionTable"
            )
        if not self.train_dimensions:
            raise ValueError(
                "torchvision train dimensions must not be empty"
            )
        for role in ("validation", "test"):
            dimensions = getattr(self, f"{role}_dimensions")
            if dimensions is not None and not isinstance(
                dimensions,
                ImageDimensionTable,
            ):
                raise TypeError(
                    f"torchvision {role}_dimensions must be "
                    "ImageDimensionTable or None"
                )
        if self.test_dimensions is None:
            raise ValueError("torchvision test dimensions must not be empty")
        if self.dataset == "flowers102":
            if self.validation_dimensions is None:
                raise ValueError(
                    "Flowers102 validation dimensions must not be empty"
                )
        elif self.validation_dimensions is not None:
            raise ValueError(
                "MNIST and CIFAR10 do not have native validation dimensions"
            )


@dataclass(frozen=True, slots=True)
class ImageFolderArtifactPayload:
    """Native image partitions and their canonical file inventories."""

    roots: Mapping[str, Path]
    train: tuple[ImageFileRecord, ...]
    validation: tuple[ImageFileRecord, ...] | None = None
    test: tuple[ImageFileRecord, ...] | None = None

    def __post_init__(self) -> None:
        roots = immutable_roots(self.roots, label="image folder payload")
        object.__setattr__(self, "roots", roots)
        for role in ("train", "validation", "test"):
            records_value = getattr(self, role)
            if records_value is None:
                continue
            records = immutable_records(
                records_value,
                label=f"image folder payload {role}",
            )
            object.__setattr__(self, role, records)
            if any(record.tree not in roots for record in records):
                raise ValueError(
                    f"image folder payload {role} inventory uses an unknown tree"
                )
        if not self.train:
            raise ValueError(
                "image folder payload train inventory must not be empty"
            )


@dataclass(frozen=True, slots=True)
class ClassLabeledImageFolderArtifactPayload:
    """Class-labeled native partitions and canonical image inventories.

    The payload may carry native validation records. Individual DataBuilders
    define a narrower accepted contract when their partition recipe requires
    different native partitions.
    """

    roots: Mapping[str, Path]
    class_mapping: Mapping[str, int]
    train: tuple[ClassLabeledImageFileRecord, ...]
    validation: tuple[ClassLabeledImageFileRecord, ...] | None = None
    test: tuple[ClassLabeledImageFileRecord, ...] | None = None

    def __post_init__(self) -> None:
        label = "class-labeled image folder payload"
        roots = immutable_roots(self.roots, label=label)
        class_mapping = immutable_class_mapping(
            self.class_mapping,
            label=label,
        )
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "class_mapping", class_mapping)
        known_labels = frozenset(class_mapping.values())
        for role in ("train", "validation", "test"):
            records_value = getattr(self, role)
            if records_value is None:
                continue
            records = immutable_class_labeled_records(
                records_value,
                label=f"{label} {role}",
            )
            object.__setattr__(self, role, records)
            if any(record.image.tree not in roots for record in records):
                raise ValueError(
                    f"{label} {role} inventory uses an unknown tree"
                )
            if any(
                record.class_label not in known_labels
                for record in records
            ):
                raise ValueError(
                    f"{label} {role} inventory uses an unknown class label"
                )
        if not self.train:
            raise ValueError(f"{label} train inventory must not be empty")
        train_labels = {record.class_label for record in self.train}
        missing_labels = sorted(known_labels - train_labels)
        if missing_labels:
            raise ValueError(
                f"{label} train inventory is missing class labels: "
                + ", ".join(str(value) for value in missing_labels)
            )


@dataclass(frozen=True, slots=True)
class PairedImageFolderArtifactPayload:
    """Native paired-image partitions and immutable pair inventories."""

    roots: Mapping[str, Path]
    train: tuple[ImageFilePair, ...]
    validation: tuple[ImageFilePair, ...] | None = None
    test: tuple[ImageFilePair, ...] | None = None

    def __post_init__(self) -> None:
        roots = immutable_roots(
            self.roots,
            label="paired image payload",
        )
        object.__setattr__(self, "roots", roots)
        for role in ("train", "validation", "test"):
            pairs_value = getattr(self, role)
            if pairs_value is None:
                continue
            pairs = immutable_pairs(
                pairs_value,
                label=f"paired image payload {role}",
            )
            object.__setattr__(self, role, pairs)
            records = (
                record
                for pair in pairs
                for record in (pair.high_resolution, pair.low_resolution)
            )
            if any(record.tree not in roots for record in records):
                raise ValueError(
                    f"paired image payload {role} inventory uses an unknown tree"
                )
        if not self.train:
            raise ValueError(
                "paired image payload train inventory must not be empty"
            )


type ImageArtifactPayload = (
    ClassLabeledImageFolderArtifactPayload
    | TorchvisionImageArtifactPayload
    | ImageFolderArtifactPayload
    | PairedImageFolderArtifactPayload
)


class ImageDataSource(DataSource[ImageArtifactPayload]):
    """Registered image-family artifact source."""

    def __init__(self, params: dict[str, Any], *, config_path: str) -> None:
        params_value = cast(object, params)
        if not isinstance(params_value, dict):
            raise TypeError("image data source params must be a mapping")
        self.params = deepcopy(params)
        self.config_path = config_path


IMAGE_DATA_SOURCES: Registry[type[ImageDataSource]] = Registry(
    "image data source",
    expected_type=ImageDataSource,
)


__all__ = [
    "IMAGE_DATA_SOURCES",
    "IMAGE_SUFFIXES",
    "ClassLabeledImageFileRecord",
    "ClassLabeledImageFolderArtifactPayload",
    "ImageArtifactPayload",
    "ImageDataSource",
    "ImageDimensionTable",
    "ImageDimensions",
    "ImageFilePair",
    "ImageFileRecord",
    "ImageFolderArtifactPayload",
    "PairedImageFolderArtifactPayload",
    "TorchvisionImageArtifactPayload",
    "immutable_class_labeled_records",
    "immutable_class_mapping",
    "immutable_pairs",
    "immutable_records",
    "immutable_roots",
    "strict_record_mapping",
    "validate_relative_image_path",
]
