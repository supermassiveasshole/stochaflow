"""Domain contracts shared by AFHQ-v2 preparation components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class PreparationError(RuntimeError):
    """Raised when AFHQ-v2 cannot be prepared without violating its contract."""


class SourceIntegrityError(PreparationError):
    """Raised when downloaded source bytes do not match the pinned source."""

    def __init__(
        self,
        message: str,
        *,
        actual_sha256: str | None = None,
        actual_bytes: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_sha256 = actual_sha256
        self.actual_bytes = actual_bytes


@dataclass(frozen=True)
class DatasetContract:
    """Expected identity and layout of the complete AFHQ-v2 source."""

    classes: tuple[str, ...]
    class_mapping: Mapping[str, int]
    train_count: int
    test_count: int
    total_count: int
    input_resolution: int
    image_mode: str
    image_format: str
    source_class_counts: Mapping[str, Mapping[str, int]] | None = None


@dataclass(frozen=True)
class SourceLock:
    """Pinned official AFHQ-v2 source and its dataset contract."""

    dataset: str
    url: str
    archive_name: str
    expected_bytes: int
    expected_sha256: str | None
    license_name: str
    license_url: str
    homepage: str
    citation: str
    contract: DatasetContract


@dataclass(frozen=True)
class SourceArchive:
    """Locally cached, content-addressed source archive."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SourceImage:
    """One canonical AFHQ-v2 source image inside the official archive."""

    member_name: str
    relative_path: str
    source_split: str
    class_name: str
    filename: str
    file_size: int
    crc32: int
