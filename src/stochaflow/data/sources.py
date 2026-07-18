"""Private raw-image sources used by built-in recipes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from stochaflow.utils.config import ConfigError

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _unknown_fields(raw: dict[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        rendered = ", ".join(f"{path}.{name}" for name in unknown)
        raise ConfigError(f"unknown config field(s): {rendered}")


def _required_string(raw: dict[str, Any], name: str, *, path: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{name} must be a non-empty string")
    return value


def _image_paths(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    paths = tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"image directory contains no supported images: {root}")
    return paths


class ImageFolderDataset(Dataset[Image.Image]):
    """Stable recursive local image directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.paths = _image_paths(self.root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as image:
            return image.copy()


def _relative_stems(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in _image_paths(root):
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in result:
            raise ValueError(
                f"duplicate relative image stem '{key}' under: {root}"
            )
        result[key] = path
    return result


class PairedImageFolderDataset(Dataset[tuple[Image.Image, Image.Image]]):
    """Stable LR/HR pairs matched by relative path without extensions."""

    def __init__(
        self,
        high_resolution_root: str | Path,
        low_resolution_root: str | Path,
    ) -> None:
        self.high_resolution_root = Path(high_resolution_root)
        self.low_resolution_root = Path(low_resolution_root)
        high = _relative_stems(self.high_resolution_root)
        low = _relative_stems(self.low_resolution_root)
        missing_low = sorted(set(high) - set(low))
        missing_high = sorted(set(low) - set(high))
        if missing_low or missing_high:
            details = []
            if missing_low:
                details.append("missing LR: " + ", ".join(missing_low))
            if missing_high:
                details.append("missing HR: " + ", ".join(missing_high))
            raise ValueError("paired image folders do not align; " + "; ".join(details))
        self.keys = tuple(sorted(high))
        self.pairs = tuple((high[key], low[key]) for key in self.keys)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image]:
        high_path, low_path = self.pairs[index]
        with Image.open(high_path) as high_image, Image.open(low_path) as low_image:
            return high_image.copy(), low_image.copy()


@dataclass(frozen=True, slots=True)
class SourceDatasets:
    """Native source partitions before recipe-specific partitioning."""

    train: Dataset[Any]
    validation: Dataset[Any] | None = None
    test: Dataset[Any] | None = None


def _torchvision_source(raw: dict[str, Any], *, path: str) -> SourceDatasets:
    _unknown_fields(raw, {"kind", "dataset", "root", "download"}, path=path)
    dataset_name = _required_string(raw, "dataset", path=path).lower()
    root = raw.get("root", "./data")
    if not isinstance(root, str) or not root.strip():
        raise ConfigError(f"{path}.root must be a non-empty string")
    download = raw.get("download", True)
    if not isinstance(download, bool):
        raise ConfigError(f"{path}.download must be boolean")
    if dataset_name == "mnist":
        return SourceDatasets(
            train=datasets.MNIST(root, train=True, transform=None, download=download),
            test=datasets.MNIST(root, train=False, transform=None, download=download),
        )
    if dataset_name == "cifar10":
        return SourceDatasets(
            train=datasets.CIFAR10(
                root, train=True, transform=None, download=download
            ),
            test=datasets.CIFAR10(
                root, train=False, transform=None, download=download
            ),
        )
    if dataset_name == "flowers102":
        return SourceDatasets(
            train=datasets.Flowers102(
                root, split="train", transform=None, download=download
            ),
            validation=datasets.Flowers102(
                root, split="val", transform=None, download=download
            ),
            test=datasets.Flowers102(
                root, split="test", transform=None, download=download
            ),
        )
    raise ConfigError(
        f"{path}.dataset must be MNIST, CIFAR10, or Flowers102"
    )


def _optional_folder(root: Path, names: tuple[str, ...]) -> Dataset[Any] | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return ImageFolderDataset(candidate)
    return None


def _image_folder_source(
    raw: dict[str, Any],
    *,
    path: str,
    official: bool,
) -> SourceDatasets:
    _unknown_fields(raw, {"kind", "path"}, path=path)
    root = Path(_required_string(raw, "path", path=path))
    if not official:
        return SourceDatasets(train=ImageFolderDataset(root))
    train = _optional_folder(root, ("train",))
    if train is None:
        raise ValueError(f"official image folder requires: {root / 'train'}")
    return SourceDatasets(
        train=train,
        validation=_optional_folder(root, ("validation", "val")),
        test=_optional_folder(root, ("test",)),
    )


def _paired_folder_source(
    raw: dict[str, Any],
    *,
    path: str,
    official: bool,
) -> SourceDatasets:
    _unknown_fields(
        raw,
        {"kind", "high_resolution_path", "low_resolution_path"},
        path=path,
    )
    high = Path(_required_string(raw, "high_resolution_path", path=path))
    low = Path(_required_string(raw, "low_resolution_path", path=path))
    if not official:
        return SourceDatasets(train=PairedImageFolderDataset(high, low))

    def optional(names: tuple[str, ...]) -> Dataset[Any] | None:
        for name in names:
            high_candidate = high / name
            low_candidate = low / name
            if high_candidate.is_dir() or low_candidate.is_dir():
                if not high_candidate.is_dir() or not low_candidate.is_dir():
                    raise ValueError(
                        f"paired official split '{name}' must exist in both roots"
                    )
                return PairedImageFolderDataset(high_candidate, low_candidate)
        return None

    train = optional(("train",))
    if train is None:
        raise ValueError("official paired folders require train in both roots")
    return SourceDatasets(
        train=train,
        validation=optional(("validation", "val")),
        test=optional(("test",)),
    )


def build_image_source(
    raw: dict[str, Any],
    *,
    partition_mode: str,
    path: str = "data.params.source",
    paired: bool = False,
) -> SourceDatasets:
    """Build one private built-in image source declaration."""

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping")
    kind = _required_string(raw, "kind", path=path).lower()
    official = partition_mode == "official"
    if kind == "torchvision" and not paired:
        return _torchvision_source(raw, path=path)
    if kind == "image_folder" and not paired:
        return _image_folder_source(raw, path=path, official=official)
    if kind == "paired_folders" and paired:
        return _paired_folder_source(raw, path=path, official=official)
    allowed = "paired_folders" if paired else "torchvision or image_folder"
    raise ConfigError(f"{path}.kind must be {allowed}")


__all__ = [
    "ImageFolderDataset",
    "PairedImageFolderDataset",
    "SourceDatasets",
    "build_image_source",
]
