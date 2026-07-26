"""Download, audit, and deterministically prepare the complete AFHQ-v2 dataset."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Self, cast
from uuid import uuid4
from zipfile import BadZipFile, ZipFile, ZipInfo

import yaml
from PIL import Image, PngImagePlugin, UnidentifiedImageError, __version__

from stochaflow.data.artifact_io import (
    cache_entry_exists,
    canonical_directory,
    create_cache_directory,
    create_cache_file_exclusive,
    ensure_cache_directory,
    lexical_absolute_path,
    open_anchored_directory,
    open_cache_file,
    publish_cache_directory,
    publish_cache_file,
    quarantine_cache_entry,
    remove_cache_directory,
    remove_cache_file,
    write_cache_file,
)

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_LOCK_PATH = _SCRIPT_DIR / "resources" / "afhq-v2.lock.yaml"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_DOWNLOAD_PROGRESS_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 20_000
_MAX_ARCHIVE_BYTES = 12 * 1024 * 1024 * 1024
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
_PNG_COMPRESS_LEVEL = 6
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
_INVENTORY_LINE_PATTERN = re.compile(
    r"^([0-9a-f]{64})  ([1-9][0-9]*)  ([^\r\n]+)$"
)
_EXIF_ORIENTATION_TAG = 274
_RECIPE_ID = "stochaflow.afhq-v2.rgb-lanczos-png"
_RECIPE_VERSION = 1
_PATH_NORMALIZATION_VERSION = 1
_VALIDATION_ALGORITHM = "sha256-ranked-relative-path"
_VALIDATION_ALGORITHM_VERSION = 1
_DEFAULT_VALIDATION_SEED = "stochaflow-afhq-v2-validation-v1"
_VALIDATION_DOMAIN = b"stochaflow.afhq-v2.validation.v1\0"
_WINDOWS_LOCK_OFFSET = 4096
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


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


@dataclass(frozen=True)
class PreparedImageRecord:
    """One verified prepared image described by the artifact inventory."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PreparedArtifact:
    """Published prepared artifact and its stable identity."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    artifact_digest: str
    preparation_key: str
    file_count: int
    image_records: tuple[PreparedImageRecord, ...]
    cache_hit: bool


@dataclass(frozen=True)
class PreparationPlan:
    """Source-locked identity of one requested prepared artifact."""

    recipe: Mapping[str, object]
    recipe_sha256: str
    preparation_key: str
    counts: Mapping[str, Mapping[str, int]]


class SourceArchiveSession(AbstractContextManager["SourceArchiveSession"]):
    """Keep one pinned source descriptor across audit and preparation."""

    def __init__(
        self,
        source: SourceArchive,
        *,
        lock: SourceLock,
    ) -> None:
        self.source = source
        self.lock = lock
        self.stream: BinaryIO | None = None
        self.initial_state: tuple[int, int, int, int] | None = None

    def __enter__(self) -> Self:
        descriptor, opened = _open_regular_file_without_links(
            self.source.path,
            label="AFHQ-v2 source archive",
        )
        self.stream = os.fdopen(descriptor, "rb")
        self.initial_state = _regular_file_state(opened)
        try:
            self._verify_digest()
        except BaseException:
            self.stream.close()
            self.stream = None
            raise
        return self

    def _verify_digest(self) -> None:
        stream = self.stream
        initial_state = self.initial_state
        if stream is None or initial_state is None:
            raise RuntimeError("source archive session is not open")
        before = os.fstat(stream.fileno())
        if _regular_file_state(before) != initial_state:
            raise SourceIntegrityError(
                "source archive changed before it could be verified"
            )
        if before.st_size != self.source.size_bytes:
            raise SourceIntegrityError(
                "source archive byte count changed",
                actual_bytes=before.st_size,
            )
        stream.seek(0)
        digest = _sha256_stream(stream)
        after = os.fstat(stream.fileno())
        if _regular_file_state(after) != initial_state:
            raise SourceIntegrityError(
                "source archive changed while it was being verified"
            )
        _require_open_file_path_identity(
            self.source.path,
            after,
            label="AFHQ-v2 source archive",
        )
        expected_digest = self.lock.expected_sha256
        if (
            digest != self.source.sha256
            or expected_digest is None
            or digest != expected_digest
        ):
            raise SourceIntegrityError(
                "source archive SHA-256 changed",
                actual_sha256=digest,
                actual_bytes=after.st_size,
            )
        stream.seek(0)

    def verify_unchanged(self) -> None:
        """Rehash the held descriptor before publishing prepared content."""

        self._verify_digest()

    def __exit__(self, *exc_info: object) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreparationError(f"{label} must be a positive integer")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise PreparationError(f"{label} has unknown fields: {unknown}")
    if missing:
        raise PreparationError(f"{label} is missing fields: {missing}")


def load_source_lock(path: Path = _DEFAULT_LOCK_PATH) -> SourceLock:
    """Load and strictly validate the checked-in AFHQ-v2 source lock."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PreparationError(f"source lock does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise PreparationError(f"source lock is invalid YAML: {path}") from error

    root = _require_mapping(raw, label="source lock")
    _require_exact_keys(
        root,
        expected={
            "schema_version",
            "dataset",
            "source",
            "license",
            "homepage",
            "citation",
            "dataset_contract",
        },
        label="source lock",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise PreparationError("source lock schema_version must be 1")
    dataset = _require_string(root["dataset"], label="dataset")
    if dataset != "afhq-v2":
        raise PreparationError("source lock dataset must be 'afhq-v2'")

    source = _require_mapping(root["source"], label="source")
    _require_exact_keys(
        source,
        expected={"type", "url", "archive_name", "bytes", "sha256"},
        label="source",
    )
    if source["type"] != "official_archive":
        raise PreparationError("source.type must be 'official_archive'")
    expected_sha256 = source["sha256"]
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise PreparationError("source.sha256 must be null or lowercase SHA-256")

    license_value = _require_mapping(root["license"], label="license")
    _require_exact_keys(
        license_value,
        expected={"name", "url"},
        label="license",
    )
    contract_value = _require_mapping(
        root["dataset_contract"],
        label="dataset_contract",
    )
    _require_exact_keys(
        contract_value,
        expected={
            "classes",
            "class_mapping",
            "source_splits",
            "source_class_counts",
            "total_count",
            "input_resolution",
            "image_mode",
            "image_format",
        },
        label="dataset_contract",
    )

    raw_classes = contract_value["classes"]
    if not isinstance(raw_classes, Sequence) or isinstance(raw_classes, str):
        raise PreparationError("dataset_contract.classes must be a list")
    classes = tuple(
        _require_string(value, label="dataset_contract.classes item")
        for value in raw_classes
    )
    if classes != ("cat", "dog", "wild"):
        raise PreparationError(
            "dataset_contract.classes must be ['cat', 'dog', 'wild']"
        )

    raw_mapping = _require_mapping(
        contract_value["class_mapping"],
        label="dataset_contract.class_mapping",
    )
    class_mapping: dict[str, int] = {}
    for class_name in classes:
        value = raw_mapping.get(class_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise PreparationError(
                f"dataset_contract.class_mapping.{class_name} must be an integer"
            )
        class_mapping[class_name] = value
    if set(raw_mapping) != set(classes) or class_mapping != {
        "cat": 0,
        "dog": 1,
        "wild": 2,
    }:
        raise PreparationError(
            "dataset_contract.class_mapping must be cat: 0, dog: 1, wild: 2"
        )

    source_splits = _require_mapping(
        contract_value["source_splits"],
        label="dataset_contract.source_splits",
    )
    _require_exact_keys(
        source_splits,
        expected={"train", "test"},
        label="dataset_contract.source_splits",
    )
    train_count = _require_positive_int(
        source_splits["train"],
        label="dataset_contract.source_splits.train",
    )
    test_count = _require_positive_int(
        source_splits["test"],
        label="dataset_contract.source_splits.test",
    )
    raw_source_class_counts = _require_mapping(
        contract_value["source_class_counts"],
        label="dataset_contract.source_class_counts",
    )
    _require_exact_keys(
        raw_source_class_counts,
        expected={"train", "test"},
        label="dataset_contract.source_class_counts",
    )
    source_class_counts: dict[str, dict[str, int]] = {}
    for split, expected_split_count in (
        ("train", train_count),
        ("test", test_count),
    ):
        raw_split_counts = _require_mapping(
            raw_source_class_counts[split],
            label=f"dataset_contract.source_class_counts.{split}",
        )
        _require_exact_keys(
            raw_split_counts,
            expected=set(classes),
            label=f"dataset_contract.source_class_counts.{split}",
        )
        split_counts = {
            class_name: _require_positive_int(
                raw_split_counts[class_name],
                label=(
                    f"dataset_contract.source_class_counts."
                    f"{split}.{class_name}"
                ),
            )
            for class_name in classes
        }
        if sum(split_counts.values()) != expected_split_count:
            raise PreparationError(
                f"dataset_contract source class counts for {split} do not "
                "sum to its split count"
            )
        source_class_counts[split] = split_counts
    total_count = _require_positive_int(
        contract_value["total_count"],
        label="dataset_contract.total_count",
    )
    if train_count + test_count != total_count:
        raise PreparationError("source split counts do not sum to total_count")

    archive_name = _require_string(
        source["archive_name"],
        label="source.archive_name",
    )
    if archive_name != "afhq_v2.zip":
        raise PreparationError("source.archive_name must be exactly 'afhq_v2.zip'")
    input_resolution = _require_positive_int(
        contract_value["input_resolution"],
        label="dataset_contract.input_resolution",
    )
    if input_resolution != 512:
        raise PreparationError("dataset_contract.input_resolution must be 512")
    image_mode = _require_string(
        contract_value["image_mode"],
        label="dataset_contract.image_mode",
    )
    if image_mode != "RGB":
        raise PreparationError("dataset_contract.image_mode must be 'RGB'")
    image_format = _require_string(
        contract_value["image_format"],
        label="dataset_contract.image_format",
    )
    if image_format != "PNG":
        raise PreparationError("dataset_contract.image_format must be 'PNG'")

    return SourceLock(
        dataset=dataset,
        url=_require_string(source["url"], label="source.url"),
        archive_name=archive_name,
        expected_bytes=_require_positive_int(source["bytes"], label="source.bytes"),
        expected_sha256=expected_sha256,
        license_name=_require_string(license_value["name"], label="license.name"),
        license_url=_require_string(license_value["url"], label="license.url"),
        homepage=_require_string(root["homepage"], label="homepage"),
        citation=_require_string(root["citation"], label="citation"),
        contract=DatasetContract(
            classes=classes,
            class_mapping=class_mapping,
            train_count=train_count,
            test_count=test_count,
            total_count=total_count,
            input_resolution=input_resolution,
            image_mode=image_mode,
            image_format=image_format,
            source_class_counts=source_class_counts,
        ),
    )


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(_HASH_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file using bounded memory."""

    descriptor, opened = _open_regular_file_without_links(
        path,
        label="file to hash",
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _sha256_stream(stream)
        after = os.fstat(descriptor)
        if _regular_file_state(after) != _regular_file_state(opened):
            raise PreparationError(f"file changed while it was hashed: {path}")
        _require_open_file_path_identity(path, after, label="file to hash")
        return digest
    finally:
        os.close(descriptor)


def _regular_file_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _regular_file_state(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _open_regular_file_without_links(
    path: Path,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    canonical_path = lexical_absolute_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if os.name != "nt":
        flags |= getattr(os, "O_NONBLOCK", 0)
    if os.name != "nt":
        try:
            parent_descriptor = open_anchored_directory(
                canonical_path.parent,
                label=f"{label} parent",
            )
        except NotImplementedError:
            pass
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot open {label}: {canonical_path}"
            ) from error
        else:
            try:
                descriptor = os.open(
                    canonical_path.name,
                    flags,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise PreparationError(
                    f"cannot open {label}: {canonical_path}"
                ) from error
            finally:
                os.close(parent_descriptor)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise PreparationError(
                        f"{label} must be a regular file: {canonical_path}"
                    )
                return descriptor, opened
            except BaseException:
                os.close(descriptor)
                raise

    try:
        parent = canonical_directory(
            canonical_path.parent,
            label=f"{label} parent",
        )
        parent_before = parent.lstat()
        before = canonical_path.lstat()
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        raise PreparationError(f"cannot inspect {label}: {canonical_path}") from error
    if _is_reparse_or_symlink(before) or not stat.S_ISREG(before.st_mode):
        raise PreparationError(
            f"{label} must be a regular file without linked components: "
            f"{canonical_path}"
        )
    try:
        descriptor = os.open(canonical_path, flags)
    except OSError as error:
        raise PreparationError(f"cannot open {label}: {canonical_path}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _regular_file_identity(opened) != _regular_file_identity(before)
        ):
            raise PreparationError(
                f"{label} changed while it was opened: {canonical_path}"
            )
        parent_after = parent.lstat()
        if _regular_file_identity(parent_after) != _regular_file_identity(
            parent_before
        ):
            raise PreparationError(
                f"{label} parent changed while it was opened: {parent}"
            )
        _require_open_file_path_identity(
            canonical_path,
            opened,
            label=label,
        )
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _require_open_file_path_identity(
    path: Path,
    opened: os.stat_result,
    *,
    label: str,
) -> None:
    canonical_path = lexical_absolute_path(path)
    if os.name != "nt":
        try:
            current_descriptor, current = _open_regular_file_without_links(
                canonical_path,
                label=f"{label} identity",
            )
        except NotImplementedError:
            pass
        else:
            try:
                if _regular_file_identity(current) != _regular_file_identity(
                    opened
                ):
                    raise PreparationError(
                        f"{label} path no longer names the opened file: "
                        f"{canonical_path}"
                    )
                return
            finally:
                os.close(current_descriptor)
    try:
        parent = canonical_directory(
            canonical_path.parent,
            label=f"{label} parent",
        )
        parent_before = parent.lstat()
        observed = canonical_path.lstat()
        parent_after = parent.lstat()
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        raise PreparationError(
            f"{label} disappeared or became unsafe: {canonical_path}"
        ) from error
    if (
        _is_reparse_or_symlink(observed)
        or not stat.S_ISREG(observed.st_mode)
        or _regular_file_identity(observed) != _regular_file_identity(opened)
        or _regular_file_identity(parent_after)
        != _regular_file_identity(parent_before)
    ):
        raise PreparationError(f"{label} changed while it was in use: {canonical_path}")


def _write_descriptor(descriptor: int, payload: bytes | memoryview) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write the complete cache payload")
        remaining = remaining[written:]


def _is_reparse_or_symlink(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _require_real_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PreparationError(f"cannot inspect {label}: {path}") from error
    if _is_reparse_or_symlink(metadata):
        raise PreparationError(f"{label} must not be a symlink or reparse point: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise PreparationError(f"{label} is not a directory: {path}")
    return metadata


def _validate_relative_path(relative_path: PurePosixPath, *, label: str) -> None:
    raw = relative_path.as_posix()
    parts = relative_path.parts
    if (
        not raw
        or relative_path.is_absolute()
        or "\\" in raw
        or raw != unicodedata.normalize("NFC", raw)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise PreparationError(f"{label} has an unsafe path: {raw!r}")


def _read_regular_file_posix(
    root: Path,
    relative_path: PurePosixPath,
    *,
    label: str,
    read_content: bool,
) -> tuple[bytes | None, os.stat_result]:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0:
        raise NotImplementedError
    descriptors: list[int] = []
    try:
        try:
            root_descriptor = open_anchored_directory(
                root,
                label="prepared artifact root",
            )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"prepared artifact root is missing or unsafe: {root}"
            ) from error
        descriptors.append(root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise PreparationError(f"prepared artifact root is not a directory: {root}")
        parent_descriptor = root_descriptor
        for part in relative_path.parts[:-1]:
            try:
                descriptor = os.open(
                    part,
                    directory_flags | no_follow,
                    dir_fd=parent_descriptor,
                )
            except OSError as error:
                raise PreparationError(
                    f"{label} has an unsafe or missing ancestor: "
                    f"{relative_path.as_posix()!r}"
                ) from error
            descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise PreparationError(
                    f"{label} ancestor is not a directory: "
                    f"{relative_path.as_posix()!r}"
                )
            parent_descriptor = descriptor
        try:
            file_descriptor = os.open(
                relative_path.parts[-1],
                os.O_RDONLY
                | no_follow
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            raise PreparationError(
                f"{label} is missing or became a symlink: "
                f"{relative_path.as_posix()!r}"
            ) from error
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PreparationError(
                f"{label} is not a regular file: {relative_path.as_posix()!r}"
            )
        payload: bytes | None = None
        if read_content:
            with os.fdopen(file_descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
        after = os.fstat(file_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PreparationError(
                f"{label} changed while it was being read: "
                f"{relative_path.as_posix()!r}"
            )
        if payload is not None and len(payload) != after.st_size:
            raise PreparationError(
                f"{label} size changed while it was being read: "
                f"{relative_path.as_posix()!r}"
            )
        return payload, after
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_regular_file_portable(
    root: Path,
    relative_path: PurePosixPath,
    *,
    label: str,
    read_content: bool,
) -> tuple[bytes | None, os.stat_result]:
    root_before = _require_real_directory(root, label="prepared artifact root")
    current = root
    ancestors: list[tuple[Path, os.stat_result]] = [(root, root_before)]
    for part in relative_path.parts[:-1]:
        current /= part
        ancestors.append(
            (current, _require_real_directory(current, label=f"{label} ancestor"))
        )
    path = current / relative_path.parts[-1]
    try:
        before = path.lstat()
    except OSError as error:
        raise PreparationError(f"cannot read {label}: {path}") from error
    if _is_reparse_or_symlink(before):
        raise PreparationError(f"{label} must not be a symlink or reparse point: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise PreparationError(f"{label} is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
        ) != (
            before.st_dev,
            before.st_ino,
        ):
            raise PreparationError(f"{label} changed while it was opened: {path}")
        payload: bytes | None = None
        if read_content:
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PreparationError(f"{label} changed while it was being read: {path}")
        if payload is not None and len(payload) != after.st_size:
            raise PreparationError(f"{label} size changed while it was read: {path}")
    finally:
        os.close(descriptor)
    for ancestor, expected in ancestors:
        actual = _require_real_directory(ancestor, label=f"{label} ancestor")
        if (
            actual.st_dev,
            actual.st_ino,
            actual.st_mtime_ns,
        ) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_mtime_ns,
        ):
            raise PreparationError(
                f"{label} ancestor changed while the file was read: {ancestor}"
            )
    try:
        final_metadata = path.lstat()
    except OSError as error:
        raise PreparationError(f"{label} disappeared while it was read: {path}") from error
    if _is_reparse_or_symlink(final_metadata) or (
        final_metadata.st_dev,
        final_metadata.st_ino,
    ) != (
        after.st_dev,
        after.st_ino,
    ):
        raise PreparationError(f"{label} changed while it was read: {path}")
    return payload, after


def _read_regular_file_without_links(
    root: Path,
    relative_path: PurePosixPath,
    *,
    label: str,
    read_content: bool = True,
) -> tuple[bytes | None, os.stat_result]:
    _validate_relative_path(relative_path, label=label)
    if os.name != "nt":
        try:
            return _read_regular_file_posix(
                root,
                relative_path,
                label=label,
                read_content=read_content,
            )
        except NotImplementedError:
            pass
    return _read_regular_file_portable(
        root,
        relative_path,
        label=label,
        read_content=read_content,
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _enumerate_regular_files_posix(
    root: Path,
    relative_root: PurePosixPath,
) -> set[str]:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0 or os.scandir not in os.supports_fd:
        raise NotImplementedError
    descriptors: list[int] = []
    discovered: set[str] = set()

    def open_directory(
        name: str | Path,
        *,
        parent_descriptor: int | None = None,
    ) -> int:
        try:
            if parent_descriptor is None:
                descriptor = open_anchored_directory(
                    Path(name),
                    label="prepared artifact root",
                )
            else:
                descriptor = os.open(
                    name,
                    directory_flags | no_follow,
                    dir_fd=parent_descriptor,
                )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"prepared directory is missing or unsafe: {name}"
            ) from error
        descriptors.append(descriptor)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PreparationError(f"prepared path is not a directory: {name}")
        return descriptor

    def scan_directory(
        descriptor: int,
        relative_parts: tuple[str, ...],
    ) -> None:
        try:
            with os.scandir(descriptor) as entries:
                snapshot = list(entries)
        except OSError as error:
            raise PreparationError(
                "cannot enumerate prepared directory: "
                f"{PurePosixPath(*relative_parts)}"
            ) from error
        for entry in snapshot:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect prepared path: {entry.name}"
                ) from error
            if _is_reparse_or_symlink(metadata):
                raise PreparationError(
                    "prepared path must not be a symlink or reparse point: "
                    f"{entry.name}"
                )
            child_parts = (*relative_parts, entry.name)
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor = open_directory(
                    entry.name,
                    parent_descriptor=descriptor,
                )
                opened = os.fstat(child_descriptor)
                if _directory_identity(opened) != _directory_identity(metadata):
                    raise PreparationError(
                        "prepared directory changed while it was opened: "
                        f"{PurePosixPath(*child_parts)}"
                    )
                scan_directory(child_descriptor, child_parts)
            elif stat.S_ISREG(metadata.st_mode):
                discovered.add(PurePosixPath(*child_parts).as_posix())
            else:
                raise PreparationError(
                    "prepared path is neither a directory nor a regular file: "
                    f"{PurePosixPath(*child_parts)}"
                )

    try:
        descriptor = open_directory(root)
        relative_parts: tuple[str, ...] = ()
        for part in relative_root.parts:
            descriptor = open_directory(
                part,
                parent_descriptor=descriptor,
            )
            relative_parts = (*relative_parts, part)
        scan_directory(descriptor, relative_parts)
        return discovered
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _enumerate_regular_files_portable(
    root: Path,
    relative_root: PurePosixPath,
) -> set[str]:
    root_metadata = _require_real_directory(
        root,
        label="prepared artifact root",
    )
    ancestor_snapshots: list[tuple[Path, os.stat_result]] = [
        (root, root_metadata)
    ]
    split_root = root
    for part in relative_root.parts:
        split_root /= part
        ancestor_snapshots.append(
            (
                split_root,
                _require_real_directory(
                    split_root,
                    label=f"prepared split {relative_root}",
                ),
            )
        )
    discovered: set[str] = set()

    def recheck_ancestors(
        snapshots: Sequence[tuple[Path, os.stat_result]],
    ) -> None:
        for path, expected in snapshots:
            actual = _require_real_directory(path, label="prepared ancestor")
            if _directory_identity(actual) != _directory_identity(expected):
                raise PreparationError(
                    f"prepared ancestor changed while scanning: {path}"
                )

    def scan_directory(
        directory: Path,
        snapshots: list[tuple[Path, os.stat_result]],
    ) -> None:
        before = _require_real_directory(
            directory,
            label="prepared directory",
        )
        if _directory_identity(before) != _directory_identity(snapshots[-1][1]):
            raise PreparationError(
                f"prepared directory changed before scanning: {directory}"
            )
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PreparationError(
                f"cannot enumerate prepared directory: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect prepared path: {path}"
                ) from error
            if _is_reparse_or_symlink(metadata):
                raise PreparationError(
                    f"prepared path must not be a symlink or reparse point: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child_metadata = _require_real_directory(
                    path,
                    label="prepared directory",
                )
                if child_metadata.st_ino != entry.inode():
                    raise PreparationError(
                        f"prepared directory changed while it was inspected: {path}"
                    )
                scan_directory(
                    path,
                    [*snapshots, (path, child_metadata)],
                )
            elif stat.S_ISREG(metadata.st_mode):
                discovered.add(path.relative_to(root).as_posix())
            else:
                raise PreparationError(
                    f"prepared path is neither a directory nor a regular file: {path}"
                )
        recheck_ancestors(snapshots)

    scan_directory(split_root, ancestor_snapshots)
    return discovered


def _enumerate_regular_files_without_links(
    root: Path,
    relative_root: PurePosixPath,
) -> set[str]:
    _validate_relative_path(relative_root, label="prepared split")
    if os.name != "nt":
        try:
            return _enumerate_regular_files_posix(root, relative_root)
        except NotImplementedError:
            pass
    return _enumerate_regular_files_portable(root, relative_root)


def _validate_prepared_root_snapshot(
    entries: Mapping[str, os.stat_result],
) -> None:
    expected_files = {"dataset_manifest.yaml", "files.sha256"}
    expected_directories = {"train", "validation", "test"}
    expected = expected_files | expected_directories
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        raise PreparationError(
            "prepared artifact root has an invalid layout; "
            f"missing={missing or '<none>'}, "
            f"unexpected={unexpected or '<none>'}"
        )
    for name, metadata in entries.items():
        if _is_reparse_or_symlink(metadata):
            raise PreparationError(
                f"prepared artifact root entry must not be linked: {name}"
            )
        expected_mode = stat.S_ISREG if name in expected_files else stat.S_ISDIR
        if not expected_mode(metadata.st_mode):
            expected_kind = "regular file" if name in expected_files else "directory"
            raise PreparationError(
                f"prepared artifact root entry must be a {expected_kind}: {name}"
            )


def _validate_prepared_root_layout(root: Path) -> None:
    if os.name != "nt":
        try:
            descriptor = open_anchored_directory(
                root,
                label="prepared artifact root",
            )
        except NotImplementedError:
            pass
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot inspect prepared artifact root: {root}"
            ) from error
        else:
            try:
                before = os.fstat(descriptor)
                with os.scandir(descriptor) as iterator:
                    snapshot = {
                        entry.name: entry.stat(follow_symlinks=False)
                        for entry in iterator
                    }
                _validate_prepared_root_snapshot(snapshot)
                if _regular_file_state(os.fstat(descriptor)) != (
                    _regular_file_state(before)
                ):
                    raise PreparationError(
                        "prepared artifact root changed while it was inspected"
                    )
                return
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect prepared artifact root: {root}"
                ) from error
            finally:
                os.close(descriptor)

    before = _require_real_directory(root, label="prepared artifact root")
    try:
        with os.scandir(root) as iterator:
            snapshot = {
                entry.name: entry.stat(follow_symlinks=False)
                for entry in iterator
            }
    except OSError as error:
        raise PreparationError(
            f"cannot inspect prepared artifact root: {root}"
        ) from error
    _validate_prepared_root_snapshot(snapshot)
    after = _require_real_directory(root, label="prepared artifact root")
    if _regular_file_state(after) != _regular_file_state(before):
        raise PreparationError(
            "prepared artifact root changed while it was inspected"
        )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


class ArtifactPreparationLock(AbstractContextManager["ArtifactPreparationLock"]):
    """Bounded OS advisory lock retained as a persistent cache entry."""

    def __init__(
        self,
        path: Path,
        *,
        cache_root: Path | None = None,
        timeout_seconds: float = 120.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if timeout_seconds < 0:
            raise ValueError("lock timeout_seconds must not be negative")
        if poll_seconds <= 0:
            raise ValueError("lock poll_seconds must be positive")
        self._path = lexical_absolute_path(path)
        self._cache_root = lexical_absolute_path(
            cache_root if cache_root is not None else self._path.parent
        )
        self._timeout_seconds = timeout_seconds
        self._poll_seconds = poll_seconds
        self._nonce = uuid4().hex
        self._descriptor: int | None = None

    def __enter__(self) -> Self:
        try:
            descriptor = open_cache_file(
                self._cache_root,
                self._path,
                label="AFHQ-v2 preparation lock",
            )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot open preparation lock: {self._path}"
            ) from error
        if os.fstat(descriptor).st_size == 0:
            _write_descriptor(descriptor, b"\0")
            os.fsync(descriptor)
        deadline = time.monotonic() + self._timeout_seconds
        metadata = {
            "created_unix": time.time(),
            "hostname": socket.gethostname(),
            "nonce": self._nonce,
            "pid": os.getpid(),
        }
        payload = _canonical_json_bytes(metadata) + b"\n"
        while True:
            try:
                self._acquire_advisory_lock(descriptor)
                break
            except (BlockingIOError, OSError) as error:
                if not self._is_lock_contention(error):
                    os.close(descriptor)
                    raise PreparationError(
                        f"cannot acquire preparation lock: {self._path}"
                    ) from error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    owner = self._describe_owner()
                    os.close(descriptor)
                    raise PreparationError(
                        f"timed out waiting for preparation lock "
                        f"({self._path}); owner={owner}. Do not remove the "
                        "lock until the owner is confirmed inactive."
                    ) from error
                time.sleep(min(self._poll_seconds, remaining))
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            _write_descriptor(descriptor, payload)
            os.fsync(descriptor)
        except BaseException:
            self._release_advisory_lock(descriptor)
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    @staticmethod
    def _acquire_advisory_lock(descriptor: int) -> None:
        if os.name == "nt":
            os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _release_advisory_lock(descriptor: int) -> None:
        if os.name == "nt":
            os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)

    @staticmethod
    def _is_lock_contention(error: OSError) -> bool:
        if isinstance(error, BlockingIOError):
            return True
        return error.errno in {
            getattr(os, "EACCES", 13),
            getattr(os, "EAGAIN", 11),
            getattr(os, "EDEADLK", 36),
        }

    def _describe_owner(self) -> str:
        descriptor: int | None = None
        try:
            descriptor = open_cache_file(
                self._cache_root,
                self._path,
                label="AFHQ-v2 preparation lock owner",
            )
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = os.read(descriptor, 4096)
            value = json.loads(payload)
            if not isinstance(value, dict):
                return "unparseable lock metadata"
            return json.dumps(value, sort_keys=True)
        except (OSError, ValueError, json.JSONDecodeError):
            return "unreadable lock metadata"
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def __exit__(self, *exc_info: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            try:
                self._release_advisory_lock(descriptor)
            finally:
                os.close(descriptor)


def _build_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy is not None:
        normalized = proxy.strip()
        if not normalized:
            raise PreparationError("proxy must not be empty")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        handlers.append(
            urllib.request.ProxyHandler(
                {
                    "http": normalized,
                    "https": normalized,
                }
            )
        )
    return urllib.request.build_opener(*handlers)


def _parse_content_range(value: str | None, *, expected_start: int) -> int:
    if value is None:
        raise SourceIntegrityError("resumed response has no Content-Range header")
    match = _CONTENT_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise SourceIntegrityError(f"invalid Content-Range header: {value!r}")
    start, end, total = (int(group) for group in match.groups())
    if start != expected_start or end < start or total <= end:
        raise SourceIntegrityError(
            f"unexpected Content-Range for offset {expected_start}: {value!r}"
        )
    return total


def _download_once(
    *,
    opener: urllib.request.OpenerDirector,
    url: str,
    cache_root: Path,
    partial_path: Path,
    expected_bytes: int,
) -> None:
    existing_bytes = _cache_file_size(
        cache_root,
        partial_path,
        label="AFHQ-v2 partial download",
    )
    if existing_bytes is None:
        existing_bytes = 0
    if existing_bytes > expected_bytes:
        raise SourceIntegrityError(
            f"partial download is larger than expected: {partial_path}"
        )
    if existing_bytes == expected_bytes:
        return

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "Stochaflow-AFHQ-v2-preparer/1",
    }
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=90) as response:
        status = response.getcode()
        append = existing_bytes > 0 and status == 206
        if existing_bytes > 0 and status == 200:
            existing_bytes = 0
        elif existing_bytes > 0 and status != 206:
            raise SourceIntegrityError(
                f"server returned HTTP {status} for a resumed download"
            )
        elif existing_bytes == 0 and status != 200:
            raise SourceIntegrityError(
                f"server returned unexpected HTTP {status} for download"
            )

        if append:
            response_total = _parse_content_range(
                response.headers.get("Content-Range"),
                expected_start=existing_bytes,
            )
            if response_total != expected_bytes:
                raise SourceIntegrityError(
                    "resumed response length does not match the source lock"
                )
        else:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise SourceIntegrityError(
                    "response Content-Length does not match the source lock: "
                    f"{content_length} != {expected_bytes}"
                )

        mode = "ab" if append else "wb"
        downloaded = existing_bytes
        next_progress = (
            (downloaded // _DOWNLOAD_PROGRESS_BYTES) + 1
        ) * _DOWNLOAD_PROGRESS_BYTES
        started = time.monotonic()
        try:
            descriptor = open_cache_file(
                cache_root,
                partial_path,
                label="AFHQ-v2 partial download",
            )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot open AFHQ-v2 partial download: {partial_path}"
            ) from error
        try:
            if append:
                observed_size = os.fstat(descriptor).st_size
                if observed_size != existing_bytes:
                    raise SourceIntegrityError(
                        "partial download changed before resume"
                    )
                os.lseek(descriptor, 0, os.SEEK_END)
            else:
                os.ftruncate(descriptor, 0)
                os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(descriptor, mode, closefd=False) as destination:
                while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                    destination.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > expected_bytes:
                        raise SourceIntegrityError(
                            "download exceeded the byte count in the source lock"
                        )
                    if downloaded >= next_progress:
                        elapsed = max(time.monotonic() - started, 0.001)
                        transferred = downloaded - existing_bytes
                        rate_mib = transferred / elapsed / (1024 * 1024)
                        percent = downloaded * 100 / expected_bytes
                        print(
                            f"Downloaded {downloaded:,}/{expected_bytes:,} bytes "
                            f"({percent:.1f}%, {rate_mib:.1f} MiB/s)",
                            flush=True,
                        )
                        next_progress += _DOWNLOAD_PROGRESS_BYTES
                destination.flush()
                os.fsync(destination.fileno())
        finally:
            os.close(descriptor)


def _download_with_curl(
    *,
    lock: SourceLock,
    cache_root: Path,
    destination: Path,
    proxy: str | None,
) -> Path:
    executable = shutil.which("curl")
    if executable is None:
        raise PreparationError(
            "curl downloader was requested but curl is not on PATH"
        )
    staging = destination.with_name(
        f".{destination.name}.curl-{os.getpid()}-{uuid4().hex}.part"
    )
    command = [
        executable,
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--header",
        "Accept-Encoding: identity",
        "--user-agent",
        "Stochaflow-AFHQ-v2-preparer/1",
        "--silent",
        "--show-error",
    ]
    if proxy is not None:
        normalized_proxy = proxy.strip()
        if not normalized_proxy:
            raise PreparationError("proxy must not be empty")
        if "://" not in normalized_proxy:
            normalized_proxy = f"http://{normalized_proxy}"
        command.extend(("--proxy", normalized_proxy))
    command.append(lock.url)
    print("Downloading with curl into a private cache staging file", flush=True)
    descriptor: int | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        descriptor = create_cache_file_exclusive(
            cache_root,
            staging,
            label="AFHQ-v2 curl download staging",
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
        )
        if process.stdout is None:
            raise PreparationError("curl did not expose its response stream")
        downloaded = 0
        started = time.monotonic()
        next_progress = _DOWNLOAD_PROGRESS_BYTES
        while chunk := process.stdout.read(_DOWNLOAD_CHUNK_BYTES):
            _write_descriptor(descriptor, chunk)
            downloaded += len(chunk)
            if downloaded > lock.expected_bytes:
                process.kill()
                raise SourceIntegrityError(
                    "curl download exceeded the byte count in the source lock"
                )
            if downloaded >= next_progress:
                elapsed = max(time.monotonic() - started, 0.001)
                rate_mib = downloaded / elapsed / (1024 * 1024)
                percent = downloaded * 100 / lock.expected_bytes
                print(
                    f"Downloaded {downloaded:,}/{lock.expected_bytes:,} bytes "
                    f"({percent:.1f}%, {rate_mib:.1f} MiB/s)",
                    flush=True,
                )
                next_progress += _DOWNLOAD_PROGRESS_BYTES
        return_code = process.wait()
        process = None
        if return_code != 0:
            raise PreparationError(
                f"curl download failed with exit code {return_code}"
            )
        if downloaded != lock.expected_bytes:
            raise SourceIntegrityError(
                "curl download byte count does not match the source lock: "
                f"{downloaded} != {lock.expected_bytes}"
            )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        publish_cache_file(
            cache_root,
            staging,
            destination,
            label="AFHQ-v2 completed curl download",
        )
        return destination
    except OSError as error:
        raise PreparationError(f"could not launch curl: {error}") from error
    finally:
        if process is not None:
            process.kill()
            process.wait()
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError, OSError, ValueError):
            remove_cache_file(
                cache_root,
                staging,
                label="AFHQ-v2 curl download staging cleanup",
            )


def _cache_file_size(
    cache_root: Path,
    path: Path,
    *,
    label: str,
) -> int | None:
    try:
        exists = cache_entry_exists(cache_root, path, label=label)
    except (OSError, ValueError) as error:
        raise PreparationError(f"cannot inspect {label}: {path}") from error
    if not exists:
        return None
    try:
        descriptor = open_cache_file(cache_root, path, label=label)
    except (OSError, ValueError) as error:
        raise PreparationError(f"cannot open {label}: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PreparationError(f"{label} is not a regular file: {path}")
        return metadata.st_size
    finally:
        os.close(descriptor)


def _quarantine_invalid_download(
    path: Path,
    *,
    cache_root: Path,
    identity: str,
) -> Path:
    safe_identity = re.sub(r"[^0-9A-Za-z._-]", "-", identity)[:96]
    try:
        target = quarantine_cache_entry(
            cache_root,
            path,
            suffix=f"{safe_identity}.invalid",
            label="invalid AFHQ-v2 download",
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise PreparationError(
            f"cannot quarantine invalid AFHQ-v2 download: {path}"
        ) from error
    print(
        f"Quarantined invalid download at {target}; restarting from byte 0",
        file=sys.stderr,
        flush=True,
    )
    return target


def download_official_archive(
    *,
    lock: SourceLock,
    cache_root: Path,
    destination: Path,
    proxy: str | None,
    downloader: str = "auto",
    attempts: int = 6,
) -> Path:
    """Download the official archive atomically, resuming a partial transfer."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if downloader not in {"auto", "curl", "python"}:
        raise ValueError("downloader must be 'auto', 'curl', or 'python'")
    try:
        ensure_cache_directory(
            cache_root,
            destination.parent,
            label="AFHQ-v2 download directory",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot create AFHQ-v2 download directory: {destination.parent}"
        ) from error
    partial_path = destination.with_name(f"{destination.name}.part")
    existing_bytes = _cache_file_size(
        cache_root,
        partial_path,
        label="AFHQ-v2 partial download",
    )
    if existing_bytes is None:
        existing_bytes = 0
    if existing_bytes > lock.expected_bytes:
        _quarantine_invalid_download(
            partial_path,
            cache_root=cache_root,
            identity=f"bytes-{existing_bytes}",
        )
        existing_bytes = 0
    if existing_bytes == lock.expected_bytes:
        try:
            publish_cache_file(
                cache_root,
                partial_path,
                destination,
                label="AFHQ-v2 completed partial download",
            )
        except FileExistsError:
            remove_cache_file(
                cache_root,
                partial_path,
                label="redundant AFHQ-v2 partial download",
            )
        return destination
    required_bytes = max(lock.expected_bytes - existing_bytes, 0)
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < required_bytes + 1024 * 1024 * 1024:
        raise PreparationError(
            "insufficient free space for the AFHQ-v2 archive plus a 1 GiB "
            f"safety margin: need {required_bytes + 1024 * 1024 * 1024:,}, "
            f"have {free_bytes:,}"
        )
    use_curl = downloader == "curl" or (
        downloader == "auto" and shutil.which("curl") is not None
    )
    if use_curl:
        return _download_with_curl(
            lock=lock,
            cache_root=cache_root,
            destination=destination,
            proxy=proxy,
        )

    opener = _build_opener(proxy)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            _download_once(
                opener=opener,
                url=lock.url,
                cache_root=cache_root,
                partial_path=partial_path,
                expected_bytes=lock.expected_bytes,
            )
            completed_size = _cache_file_size(
                cache_root,
                partial_path,
                label="AFHQ-v2 partial download",
            )
            if completed_size != lock.expected_bytes:
                raise SourceIntegrityError(
                    "download ended before the source lock byte count"
                )
            try:
                publish_cache_file(
                    cache_root,
                    partial_path,
                    destination,
                    label="AFHQ-v2 completed download",
                )
            except FileExistsError:
                remove_cache_file(
                    cache_root,
                    partial_path,
                    label="redundant AFHQ-v2 partial download",
                )
            return destination
        except (
            OSError,
            SourceIntegrityError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            delay = min(2**attempt, 16)
            print(
                f"Download attempt {attempt + 1}/{attempts} failed: {error}. "
                f"Retrying in {delay}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise PreparationError(
        f"failed to download AFHQ-v2 after {attempts} attempts"
    ) from last_error


def _verify_source_file(path: Path, lock: SourceLock) -> SourceArchive:
    descriptor, opened = _open_regular_file_without_links(
        path,
        label="AFHQ-v2 source archive",
    )
    try:
        size_bytes = opened.st_size
        if size_bytes != lock.expected_bytes:
            raise SourceIntegrityError(
                f"archive byte count mismatch: {size_bytes} != {lock.expected_bytes}",
                actual_bytes=size_bytes,
            )
        print(f"Hashing source archive: {path}", flush=True)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            archive_sha256 = _sha256_stream(stream)
        after = os.fstat(descriptor)
        if _regular_file_state(after) != _regular_file_state(opened):
            raise SourceIntegrityError(
                "archive changed while it was being hashed",
                actual_sha256=archive_sha256,
                actual_bytes=after.st_size,
            )
        _require_open_file_path_identity(
            path,
            after,
            label="AFHQ-v2 source archive",
        )
        if (
            lock.expected_sha256 is not None
            and archive_sha256 != lock.expected_sha256
        ):
            raise SourceIntegrityError(
                "archive SHA-256 mismatch: "
                f"{archive_sha256} != {lock.expected_sha256}",
                actual_sha256=archive_sha256,
                actual_bytes=size_bytes,
            )
        return SourceArchive(
            path=lexical_absolute_path(path),
            sha256=archive_sha256,
            size_bytes=size_bytes,
        )
    finally:
        os.close(descriptor)


def _stage_verified_source_file(
    source_path: Path,
    *,
    lock: SourceLock,
    cache_root: Path,
    staging: Path,
) -> SourceArchive:
    source_descriptor, source_before = _open_regular_file_without_links(
        source_path,
        label="AFHQ-v2 source archive candidate",
    )
    staging_descriptor: int | None = None
    try:
        if source_before.st_size != lock.expected_bytes:
            raise SourceIntegrityError(
                "archive byte count mismatch: "
                f"{source_before.st_size} != {lock.expected_bytes}",
                actual_bytes=source_before.st_size,
            )
        staging_descriptor = create_cache_file_exclusive(
            cache_root,
            staging,
            label="AFHQ-v2 source archive staging",
        )
        digest = hashlib.sha256()
        copied = 0
        with os.fdopen(source_descriptor, "rb", closefd=False) as source_stream:
            while chunk := source_stream.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
                _write_descriptor(staging_descriptor, chunk)
                copied += len(chunk)
        source_after = os.fstat(source_descriptor)
        if (
            copied != source_before.st_size
            or _regular_file_state(source_after)
            != _regular_file_state(source_before)
        ):
            raise SourceIntegrityError(
                "source archive changed while it was copied",
                actual_bytes=copied,
            )
        _require_open_file_path_identity(
            source_path,
            source_after,
            label="AFHQ-v2 source archive candidate",
        )
        archive_sha256 = digest.hexdigest()
        if (
            lock.expected_sha256 is None
            or archive_sha256 != lock.expected_sha256
        ):
            raise SourceIntegrityError(
                "archive SHA-256 mismatch: "
                f"{archive_sha256} != {lock.expected_sha256}",
                actual_sha256=archive_sha256,
                actual_bytes=copied,
            )
        os.fsync(staging_descriptor)
        staging_after = os.fstat(staging_descriptor)
        if staging_after.st_size != copied:
            raise SourceIntegrityError(
                "source archive staging size changed",
                actual_bytes=staging_after.st_size,
            )
        return SourceArchive(
            path=staging,
            sha256=archive_sha256,
            size_bytes=copied,
        )
    finally:
        os.close(source_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)


def acquire_official_archive(
    *,
    lock: SourceLock,
    cache_root: Path,
    proxy: str | None,
    archive_override: Path | None = None,
    downloader: str = "auto",
) -> SourceArchive:
    """Acquire and content-address the official source under the raw cache."""

    try:
        cache_root = ensure_cache_directory(
            cache_root,
            cache_root,
            label="AFHQ-v2 cache root",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot create AFHQ-v2 cache root: {cache_root}"
        ) from error
    raw_root = cache_root / "raw" / "afhq-v2"
    internal_download = raw_root / ".downloads" / lock.archive_name
    locks_root = cache_root / ".locks"
    with ArtifactPreparationLock(
        locks_root / "afhq-v2-download.lock",
        cache_root=cache_root,
    ):
        if archive_override is not None:
            candidate = lexical_absolute_path(archive_override)
            candidate_is_internal = False
        elif lock.expected_sha256 is not None:
            pinned = (
                raw_root
                / lock.expected_sha256
                / lock.archive_name
            )
            try:
                pinned_exists = cache_entry_exists(
                    cache_root,
                    pinned,
                    label="pinned AFHQ-v2 source archive",
                )
            except (OSError, ValueError) as error:
                raise PreparationError(
                    f"cannot inspect pinned AFHQ-v2 archive: {pinned}"
                ) from error
            if pinned_exists:
                return _verify_source_file(pinned, lock)
            candidate = internal_download
            candidate_is_internal = True
        else:
            candidate = internal_download
            candidate_is_internal = True

        if candidate_is_internal:
            invalid_attempts = 0
            while True:
                try:
                    candidate_exists = cache_entry_exists(
                        cache_root,
                        candidate,
                        label="AFHQ-v2 completed download",
                    )
                except (OSError, ValueError) as error:
                    raise PreparationError(
                        f"cannot inspect AFHQ-v2 completed download: {candidate}"
                    ) from error
                if not candidate_exists:
                    candidate = download_official_archive(
                        lock=lock,
                        cache_root=cache_root,
                        destination=candidate,
                        proxy=proxy,
                        downloader=downloader,
                    )
                try:
                    source = _verify_source_file(candidate, lock)
                    break
                except SourceIntegrityError as error:
                    invalid_attempts += 1
                    identity = (
                        error.actual_sha256
                        or f"bytes-{error.actual_bytes or 'unknown'}"
                    )
                    _quarantine_invalid_download(
                        candidate,
                        cache_root=cache_root,
                        identity=identity,
                    )
                    if invalid_attempts >= 2:
                        raise SourceIntegrityError(
                            "two complete downloads failed source integrity "
                            "validation; inspect the quarantined files before "
                            "trying again"
                        ) from error
        else:
            source = _verify_source_file(candidate, lock)
        if lock.expected_sha256 is None:
            raise SourceIntegrityError(
                "the checked-in source lock is not pinned yet; audit the archive "
                f"and set source.sha256 to {source.sha256}. "
                f"The complete download remains at {candidate}"
            )

        destination = raw_root / source.sha256 / lock.archive_name
        try:
            ensure_cache_directory(
                cache_root,
                destination.parent,
                label="AFHQ-v2 pinned source directory",
            )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot create pinned AFHQ-v2 source directory: "
                f"{destination.parent}"
            ) from error
        if lexical_absolute_path(candidate) == lexical_absolute_path(destination):
            return _verify_source_file(destination, lock)

        staging = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}"
        )
        try:
            staged = _stage_verified_source_file(
                candidate,
                lock=lock,
                cache_root=cache_root,
                staging=staging,
            )
            try:
                publish_cache_file(
                    cache_root,
                    staging,
                    destination,
                    label="AFHQ-v2 pinned source archive",
                )
            except FileExistsError:
                return _verify_source_file(destination, lock)
            published = _verify_source_file(destination, lock)
            if (
                published.sha256 != staged.sha256
                or published.size_bytes != staged.size_bytes
            ):
                raise SourceIntegrityError(
                    "published AFHQ-v2 source archive changed identity"
                )
            return published
        finally:
            with suppress(FileNotFoundError, OSError, ValueError):
                remove_cache_file(
                    cache_root,
                    staging,
                    label="AFHQ-v2 source archive staging cleanup",
                )


def _zip_member_mode(info: ZipInfo) -> int:
    if info.create_system != 3:
        return 0
    return info.external_attr >> 16


def _validate_member_name(info: ZipInfo) -> tuple[str, tuple[str, ...]]:
    name = info.filename
    if not name or "\x00" in name:
        raise PreparationError("archive contains an empty or NUL-containing path")
    if "\\" in name:
        raise PreparationError(f"archive member uses a Windows separator: {name!r}")
    normalized = unicodedata.normalize("NFC", name)
    if normalized.startswith(("/", "//")) or _WINDOWS_DRIVE_PATTERN.match(
        normalized
    ):
        raise PreparationError(f"archive member has an absolute path: {name!r}")
    component_text = normalized
    if info.is_dir() and component_text.endswith("/"):
        component_text = component_text[:-1]
    raw_parts = component_text.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise PreparationError(f"archive member has an unsafe path: {name!r}")
    parts = tuple(raw_parts)
    for part in parts:
        if (
            any(ord(character) < 32 or ord(character) == 127 for character in part)
            or any(
                character in _WINDOWS_INVALID_FILENAME_CHARACTERS
                for character in part
            )
            or part.endswith((" ", "."))
        ):
            raise PreparationError(
                f"archive member is not portable to Windows: {name!r}"
            )
        stem = part.split(".", maxsplit=1)[0].rstrip(" .").upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise PreparationError(
                f"archive member uses a Windows reserved name: {name!r}"
            )

    mode = _zip_member_mode(info)
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise PreparationError(f"archive member is a symbolic link: {name!r}")
    if info.is_dir():
        if file_type not in {0, stat.S_IFDIR}:
            raise PreparationError(
                f"archive directory has a non-directory type: {name!r}"
            )
    elif file_type not in {0, stat.S_IFREG}:
        raise PreparationError(f"archive member is not a regular file: {name!r}")
    if info.flag_bits & 0x1:
        raise PreparationError(f"archive member is encrypted: {name!r}")
    return normalized, parts


def _source_image_from_member(
    info: ZipInfo,
    *,
    normalized_name: str,
    parts: tuple[str, ...],
    contract: DatasetContract,
) -> SourceImage:
    if len(parts) == 4:
        root, split, class_name, filename = parts
        if root != "afhq_v2":
            raise PreparationError(
                f"unexpected archive root for image: {normalized_name!r}"
            )
    elif len(parts) == 3:
        split, class_name, filename = parts
    else:
        raise PreparationError(
            f"image must use [afhq_v2/]split/class/file.png: {normalized_name!r}"
        )
    if split not in {"train", "test"}:
        raise PreparationError(
            f"unexpected source split {split!r}: {normalized_name!r}"
        )
    if class_name not in contract.classes:
        raise PreparationError(
            f"unexpected AFHQ class {class_name!r}: {normalized_name!r}"
        )
    if not filename or PurePosixPath(filename).suffix != ".png":
        raise PreparationError(f"source image must be lowercase .png: {normalized_name}")
    if info.file_size <= 0 or info.file_size > _MAX_IMAGE_BYTES:
        raise PreparationError(
            f"source image has an invalid size ({info.file_size}): "
            f"{normalized_name!r}"
        )
    if info.compress_size == 0:
        raise PreparationError(
            f"source image has an invalid compressed size: {normalized_name!r}"
        )
    if info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
        raise PreparationError(
            f"source image exceeds the compression-ratio limit: {normalized_name!r}"
        )
    relative_path = f"{split}/{class_name}/{filename}"
    return SourceImage(
        member_name=info.filename,
        relative_path=relative_path,
        source_split=split,
        class_name=class_name,
        filename=filename,
        file_size=info.file_size,
        crc32=info.CRC,
    )


def _inspect_archive_stream(
    archive_stream: BinaryIO,
    *,
    archive_label: str,
    contract: DatasetContract,
) -> tuple[SourceImage, ...]:
    archive_stream.seek(0)
    try:
        with ZipFile(archive_stream) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise PreparationError(
                    f"archive has too many members: {len(members)}"
                )
            if sum(info.file_size for info in members) > _MAX_ARCHIVE_BYTES:
                raise PreparationError("archive exceeds the uncompressed size limit")
            compressed_bytes = sum(info.compress_size for info in members)
            uncompressed_bytes = sum(info.file_size for info in members)
            if (
                compressed_bytes == 0
                or uncompressed_bytes / compressed_bytes > _MAX_COMPRESSION_RATIO
            ):
                raise PreparationError(
                    "archive exceeds the aggregate compression-ratio limit"
                )

            seen_names: set[str] = set()
            seen_casefolded: dict[str, str] = {}
            seen_relative: set[str] = set()
            image_layout_lengths: set[int] = set()
            images: list[SourceImage] = []
            for info in members:
                normalized_name, parts = _validate_member_name(info)
                if normalized_name in seen_names:
                    raise PreparationError(
                        f"archive contains a duplicate member: {normalized_name!r}"
                    )
                seen_names.add(normalized_name)
                casefolded = normalized_name.casefold()
                prior = seen_casefolded.get(casefolded)
                if prior is not None and prior != normalized_name:
                    raise PreparationError(
                        "archive has a case-insensitive path collision: "
                        f"{prior!r} and {normalized_name!r}"
                    )
                seen_casefolded[casefolded] = normalized_name
                if info.is_dir():
                    continue
                image_layout_lengths.add(len(parts))
                image = _source_image_from_member(
                    info,
                    normalized_name=normalized_name,
                    parts=parts,
                    contract=contract,
                )
                if image.relative_path in seen_relative:
                    raise PreparationError(
                        "archive maps multiple members to the canonical path "
                        f"{image.relative_path!r}"
                    )
                seen_relative.add(image.relative_path)
                images.append(image)
            if len(image_layout_lengths) != 1:
                raise PreparationError(
                    "archive mixes rooted and rootless image layouts"
                )
    except BadZipFile as error:
        raise PreparationError(f"invalid ZIP archive: {archive_label}") from error

    images.sort(key=lambda item: item.relative_path)
    split_counts = Counter(image.source_split for image in images)
    if split_counts != Counter(
        {
            "train": contract.train_count,
            "test": contract.test_count,
        }
    ):
        raise PreparationError(
            "source split counts do not match the lock: "
            f"{dict(sorted(split_counts.items()))}"
        )
    if len(images) != contract.total_count:
        raise PreparationError(
            f"source image count mismatch: {len(images)} != {contract.total_count}"
        )
    class_counts = Counter(image.class_name for image in images)
    if set(class_counts) != set(contract.classes):
        raise PreparationError(
            f"source classes do not match the lock: {sorted(class_counts)}"
        )
    if contract.source_class_counts is not None:
        actual_source_class_counts = {
            split: {
                class_name: sum(
                    image.source_split == split
                    and image.class_name == class_name
                    for image in images
                )
                for class_name in contract.classes
            }
            for split in ("train", "test")
        }
        expected_source_class_counts = {
            split: dict(counts)
            for split, counts in contract.source_class_counts.items()
        }
        if actual_source_class_counts != expected_source_class_counts:
            raise PreparationError(
                "source split/class counts do not match the lock: "
                f"{actual_source_class_counts}"
            )
    return tuple(images)


def inspect_archive(
    archive_path: Path,
    *,
    contract: DatasetContract,
) -> tuple[SourceImage, ...]:
    """Audit one stable ZIP descriptor and return its source inventory."""

    descriptor, opened = _open_regular_file_without_links(
        archive_path,
        label="AFHQ-v2 archive inspection source",
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            images = _inspect_archive_stream(
                stream,
                archive_label=str(archive_path),
                contract=contract,
            )
        after = os.fstat(descriptor)
        if _regular_file_state(after) != _regular_file_state(opened):
            raise PreparationError(
                f"archive changed while it was inspected: {archive_path}"
            )
        _require_open_file_path_identity(
            archive_path,
            after,
            label="AFHQ-v2 archive inspection source",
        )
        return images
    finally:
        os.close(descriptor)


def select_validation_members(
    images: Sequence[SourceImage],
    *,
    classes: Sequence[str],
    per_class: int,
    seed: str,
) -> frozenset[str]:
    """Select a stable validation subset from official train by ranked SHA-256."""

    if per_class <= 0:
        raise ValueError("validation per-class count must be positive")
    if not seed:
        raise ValueError("validation seed must be non-empty")
    selected: set[str] = set()
    for class_name in classes:
        candidates = [
            image
            for image in images
            if image.source_split == "train" and image.class_name == class_name
        ]
        if len(candidates) <= per_class:
            raise PreparationError(
                f"class {class_name!r} has only {len(candidates)} train images; "
                f"cannot reserve {per_class} for validation"
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                hashlib.sha256(
                    _VALIDATION_DOMAIN
                    + seed.encode("utf-8")
                    + b"\0"
                    + item.relative_path.encode("utf-8")
                ).digest(),
                item.relative_path,
            ),
        )
        selected.update(image.relative_path for image in ranked[:per_class])
    return frozenset(selected)


def _transform_recipe(
    *,
    input_resolution: int,
    output_resolution: int,
    validation_per_class: int,
    validation_seed: str,
) -> dict[str, object]:
    return {
        "id": _RECIPE_ID,
        "version": _RECIPE_VERSION,
        "path_normalization": {
            "form": "Unicode NFC",
            "sort": "canonical POSIX relative path",
            "version": _PATH_NORMALIZATION_VERSION,
        },
        "decoder": {
            "library": "Pillow",
            "version": __version__,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "open_sequence": "open-verify-reopen-convert",
            "accepted_format": "PNG",
            "accepted_mode": "RGB",
            "accepted_size": [input_resolution, input_resolution],
            "exif_orientation": "identity-only",
            "metadata_policy": "discard",
        },
        "resize": {
            "input_size": [input_resolution, input_resolution],
            "output_size": [output_resolution, output_resolution],
            "crop": False,
            "resample": "PIL.Image.Resampling.LANCZOS",
            "reducing_gap": None,
        },
        "encoding": {
            "format": "PNG",
            "mode": "RGB",
            "bit_depth": 8,
            "optimize": False,
            "compress_level": _PNG_COMPRESS_LEVEL,
            "pnginfo": "empty",
        },
        "validation_split": {
            "source_split": "train",
            "algorithm": _VALIDATION_ALGORITHM,
            "version": _VALIDATION_ALGORITHM_VERSION,
            "seed": validation_seed,
            "per_class": validation_per_class,
        },
    }


def build_preparation_plan(
    *,
    lock: SourceLock,
    resolution: int = 128,
    validation_per_class: int = 300,
    validation_seed: str = _DEFAULT_VALIDATION_SEED,
) -> PreparationPlan:
    """Derive the prepared cache identity without requiring the raw archive."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")
    if lock.expected_sha256 is None:
        raise PreparationError(
            "materialization requires a source lock with a pinned SHA-256"
        )
    source_counts = lock.contract.source_class_counts
    if source_counts is None:
        raise PreparationError(
            "materialization requires pinned per-class source counts"
        )
    train_counts = source_counts.get("train")
    test_counts = source_counts.get("test")
    if train_counts is None or test_counts is None:
        raise PreparationError(
            "materialization requires pinned train and test class counts"
        )
    prepared_counts: dict[str, dict[str, int]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    for class_name in lock.contract.classes:
        train_count = train_counts.get(class_name)
        test_count = test_counts.get(class_name)
        if (
            isinstance(train_count, bool)
            or not isinstance(train_count, int)
            or isinstance(test_count, bool)
            or not isinstance(test_count, int)
            or train_count <= validation_per_class
            or test_count < 0
        ):
            raise PreparationError(
                "source lock class counts are incompatible with validation split"
            )
        prepared_counts["train"][class_name] = (
            train_count - validation_per_class
        )
        prepared_counts["validation"][class_name] = validation_per_class
        prepared_counts["test"][class_name] = test_count
    recipe = _transform_recipe(
        input_resolution=lock.contract.input_resolution,
        output_resolution=resolution,
        validation_per_class=validation_per_class,
        validation_seed=validation_seed,
    )
    recipe_sha256 = _canonical_digest(recipe)
    preparation_key = _canonical_digest(
        {
            "recipe_sha256": recipe_sha256,
            "source": {
                "type": "official_archive",
                "sha256": lock.expected_sha256,
            },
        }
    )
    return PreparationPlan(
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        preparation_key=preparation_key,
        counts=prepared_counts,
    )


def require_prepared_artifact(
    *,
    lock: SourceLock,
    cache_root: Path,
    resolution: int = 128,
    validation_per_class: int = 300,
    validation_seed: str = _DEFAULT_VALIDATION_SEED,
    full: bool = True,
) -> PreparedArtifact:
    """Verify a prepared cache hit without requiring raw archive bytes."""

    try:
        canonical_cache_root = canonical_directory(
            cache_root,
            label="AFHQ-v2 artifact cache root",
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise PreparationError(
            f"required AFHQ-v2 artifact cache does not exist: {cache_root}"
        ) from error
    plan = build_preparation_plan(
        lock=lock,
        resolution=resolution,
        validation_per_class=validation_per_class,
        validation_seed=validation_seed,
    )
    assert lock.expected_sha256 is not None
    source = SourceArchive(
        path=(
            canonical_cache_root
            / "raw"
            / "afhq-v2"
            / lock.expected_sha256
            / lock.archive_name
        ),
        sha256=lock.expected_sha256,
        size_bytes=lock.expected_bytes,
    )
    root = (
        canonical_cache_root
        / "prepared"
        / "afhq-v2"
        / str(resolution)
        / plan.preparation_key
    )
    if not cache_entry_exists(
        canonical_cache_root,
        root,
        label="required AFHQ-v2 prepared artifact",
    ):
        raise PreparationError(
            f"required AFHQ-v2 prepared artifact does not exist: {root}"
        )
    return verify_prepared_artifact(
        root,
        expected_preparation_key=plan.preparation_key,
        expected_recipe=plan.recipe,
        source_archive=source,
        source_lock=lock,
        expected_counts=plan.counts,
        full=full,
    )


def _decode_and_resize(
    payload: bytes,
    *,
    member_name: str,
    input_resolution: int,
    output_resolution: int,
) -> tuple[Image.Image, str]:
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            if probe.format != "PNG":
                raise PreparationError(
                    f"source image is not PNG: {member_name!r}"
                )
            probe.verify()
        with Image.open(io.BytesIO(payload)) as source:
            if source.format != "PNG":
                raise PreparationError(
                    f"source image is not PNG: {member_name!r}"
                )
            if source.size != (input_resolution, input_resolution):
                raise PreparationError(
                    f"source image has size {source.size}, expected "
                    f"{input_resolution}x{input_resolution}: {member_name!r}"
                )
            if source.mode != "RGB":
                raise PreparationError(
                    f"source image has mode {source.mode!r}, expected RGB: "
                    f"{member_name!r}"
                )
            orientation = source.getexif().get(_EXIF_ORIENTATION_TAG, 1)
            if orientation != 1:
                raise PreparationError(
                    f"source image has non-identity EXIF orientation "
                    f"{orientation!r}: {member_name!r}"
                )
            rgb = source.convert("RGB")
            rgb.load()
            pixel_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            resized = rgb.resize(
                (output_resolution, output_resolution),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=None,
            )
            resized.info.clear()
            return resized, pixel_digest
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise PreparationError(
            f"failed to decode source image: {member_name!r}"
        ) from error


def _save_prepared_png(
    image: Image.Image,
    destination: Path,
    *,
    cache_root: Path,
) -> tuple[str, int]:
    encoded_stream = io.BytesIO()
    image.save(
        encoded_stream,
        format="PNG",
        optimize=False,
        compress_level=_PNG_COMPRESS_LEVEL,
        pnginfo=PngImagePlugin.PngInfo(),
    )
    encoded = encoded_stream.getvalue()
    write_cache_file(
        cache_root,
        destination,
        encoded,
        label="prepared AFHQ-v2 image",
    )
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _write_text_atomic(
    path: Path,
    content: str,
    *,
    cache_root: Path,
    label: str,
) -> None:
    write_cache_file(
        cache_root,
        path,
        content.encode("utf-8"),
        label=label,
    )


def _manifest_text(manifest: Mapping[str, object]) -> str:
    return yaml.safe_dump(
        dict(manifest),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )


def _prepared_counts(
    images: Sequence[SourceImage],
    *,
    validation_members: frozenset[str],
    classes: Sequence[str],
) -> dict[str, dict[str, int]]:
    counts = {
        split: dict.fromkeys(classes, 0)
        for split in ("train", "validation", "test")
    }
    for image in images:
        if image.source_split == "test":
            output_split = "test"
        elif image.relative_path in validation_members:
            output_split = "validation"
        else:
            output_split = "train"
        counts[output_split][image.class_name] += 1
    return counts


def _load_manifest_bytes(payload: bytes, *, path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PreparationError(f"cannot read prepared manifest: {path}") from error
    return _require_mapping(value, label="prepared manifest")


def _parse_prepared_image_records(
    payload: bytes,
    *,
    path: Path,
) -> tuple[PreparedImageRecord, ...]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PreparationError(f"cannot read prepared inventory: {path}") from error
    records: list[PreparedImageRecord] = []
    for line_number, line in enumerate(lines, start=1):
        match = _INVENTORY_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise PreparationError(
                f"invalid files.sha256 line {line_number}: {line!r}"
            )
        records.append(
            PreparedImageRecord(
                relative_path=match.group(3),
                size_bytes=int(match.group(2)),
                sha256=match.group(1),
            )
        )
    paths = [record.relative_path for record in records]
    if paths != sorted(paths):
        raise PreparationError("files.sha256 paths are not sorted")
    if len(set(paths)) != len(records):
        raise PreparationError("files.sha256 contains duplicate paths")
    return tuple(records)


def load_prepared_image_records(path: Path) -> tuple[PreparedImageRecord, ...]:
    """Load the strict, sorted prepared-image inventory without following links."""

    payload, _ = _read_regular_file_without_links(
        path.parent,
        PurePosixPath(path.name),
        label="prepared inventory",
    )
    assert payload is not None
    return _parse_prepared_image_records(payload, path=path)


def _manifest_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, object]]:
    return {
        split: {
            "classes": dict(class_counts),
            "total": sum(class_counts.values()),
        }
        for split, class_counts in counts.items()
    }


def verify_prepared_artifact(
    root: Path,
    *,
    expected_preparation_key: str,
    expected_recipe: Mapping[str, object],
    source_archive: SourceArchive,
    source_lock: SourceLock,
    expected_counts: Mapping[str, Mapping[str, int]],
    full: bool = True,
) -> PreparedArtifact:
    """Verify manifest/layout identity and optionally every prepared file."""

    if type(full) is not bool:
        raise TypeError("full verification flag must be boolean")

    try:
        root = canonical_directory(root, label="prepared artifact root")
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise PreparationError(
            "prepared artifact root contains a symlink, reparse point, "
            f"invalid directory, or missing path: {root}"
        ) from error
    if full:
        _validate_prepared_root_layout(root)
    manifest_path = root / "dataset_manifest.yaml"
    inventory_path = root / "files.sha256"
    manifest_bytes, _ = _read_regular_file_without_links(
        root,
        PurePosixPath("dataset_manifest.yaml"),
        label="prepared manifest",
    )
    assert manifest_bytes is not None
    manifest = _load_manifest_bytes(manifest_bytes, path=manifest_path)
    _require_exact_keys(
        manifest,
        expected={
            "schema_version",
            "dataset",
            "source",
            "preparation",
            "counts",
            "inventory",
            "artifact_digest",
        },
        label="prepared manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise PreparationError("prepared manifest schema_version must be 1")
    dataset = _require_mapping(
        manifest["dataset"],
        label="prepared manifest dataset",
    )
    preparation = _require_mapping(
        manifest["preparation"],
        label="prepared manifest preparation",
    )
    source = _require_mapping(
        manifest["source"],
        label="prepared manifest source",
    )
    inventory = _require_mapping(
        manifest["inventory"],
        label="prepared manifest inventory",
    )
    manifest_counts = _require_mapping(
        manifest["counts"],
        label="prepared manifest counts",
    )
    expected_dataset = {
        "name": "AFHQ-v2",
        "version": 2,
        "homepage": source_lock.homepage,
        "license": {
            "name": source_lock.license_name,
            "url": source_lock.license_url,
        },
        "citation": source_lock.citation,
        "class_mapping": dict(source_lock.contract.class_mapping),
    }
    if dict(dataset) != expected_dataset:
        raise PreparationError("prepared manifest dataset metadata was modified")
    _require_exact_keys(
        preparation,
        expected={"key", "recipe", "recipe_sha256"},
        label="prepared manifest preparation",
    )
    if preparation.get("key") != expected_preparation_key:
        raise PreparationError("prepared manifest has the wrong preparation key")
    manifest_recipe = _require_mapping(
        preparation["recipe"],
        label="prepared manifest preparation.recipe",
    )
    expected_recipe_hash = _canonical_digest(expected_recipe)
    actual_recipe_hash = _canonical_digest(manifest_recipe)
    if (
        dict(manifest_recipe) != dict(expected_recipe)
        or actual_recipe_hash != expected_recipe_hash
        or preparation.get("recipe_sha256") != expected_recipe_hash
    ):
        raise PreparationError("prepared manifest has the wrong recipe digest")
    _require_exact_keys(
        source,
        expected={
            "type",
            "url",
            "archive",
            "source_splits",
            "source_class_counts",
            "total_count",
            "canonical_rgb_inventory_sha256",
        },
        label="prepared manifest source",
    )
    archive = _require_mapping(
        source["archive"],
        label="prepared manifest source.archive",
    )
    expected_source = {
        "type": "official_archive",
        "url": source_lock.url,
        "archive": {
            "name": source_lock.archive_name,
            "sha256": source_archive.sha256,
            "bytes": source_archive.size_bytes,
        },
        "source_splits": {
            "train": source_lock.contract.train_count,
            "test": source_lock.contract.test_count,
        },
        "source_class_counts": {
            split: dict(counts)
            for split, counts in (
                source_lock.contract.source_class_counts or {}
            ).items()
        },
        "total_count": source_lock.contract.total_count,
    }
    actual_source = dict(source)
    source_inventory_digest = actual_source.pop(
        "canonical_rgb_inventory_sha256",
        None,
    )
    if (
        not isinstance(source_inventory_digest, str)
        or _SHA256_PATTERN.fullmatch(source_inventory_digest) is None
    ):
        raise PreparationError(
            "prepared manifest has an invalid source inventory digest"
        )
    if actual_source != expected_source:
        raise PreparationError("prepared manifest source metadata was modified")
    _require_exact_keys(
        archive,
        expected={"name", "sha256", "bytes"},
        label="prepared manifest source.archive",
    )

    expected_manifest_counts = _manifest_counts(expected_counts)
    if dict(manifest_counts) != expected_manifest_counts:
        raise PreparationError("prepared manifest counts do not match the recipe")
    _require_exact_keys(
        inventory,
        expected={"path", "file_count", "sha256"},
        label="prepared manifest inventory",
    )
    if inventory["path"] != "files.sha256":
        raise PreparationError("prepared manifest has the wrong inventory path")
    inventory_bytes, _ = _read_regular_file_without_links(
        root,
        PurePosixPath("files.sha256"),
        label="prepared inventory",
    )
    assert inventory_bytes is not None
    inventory_digest = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory.get("sha256") != inventory_digest:
        raise PreparationError("files.sha256 digest does not match the manifest")
    records = _parse_prepared_image_records(
        inventory_bytes,
        path=inventory_path,
    )
    expected_count = source_lock.contract.total_count
    if len(records) != expected_count:
        raise PreparationError(
            f"prepared file count mismatch: {len(records)} != {expected_count}"
        )
    if inventory["file_count"] != expected_count:
        raise PreparationError("prepared manifest has the wrong inventory file count")

    actual_counts = {
        split: dict.fromkeys(source_lock.contract.classes, 0)
        for split in ("train", "validation", "test")
    }
    for record in records:
        _validate_relative_path(
            PurePosixPath(record.relative_path),
            label="prepared inventory record",
        )
        parts = PurePosixPath(record.relative_path).parts
        if (
            len(parts) != 3
            or parts[0] not in actual_counts
            or parts[1] not in source_lock.contract.classes
            or not parts[2].endswith(".png")
        ):
            raise PreparationError(
                f"invalid prepared inventory path: {record.relative_path!r}"
            )
        actual_counts[parts[0]][parts[1]] += 1
    if actual_counts != {
        split: dict(class_counts)
        for split, class_counts in expected_counts.items()
    }:
        raise PreparationError(
            "prepared inventory split/class counts do not match the recipe"
        )

    listed_paths = {record.relative_path for record in records}
    actual_paths: set[str] = set()
    for split in ("train", "validation", "test"):
        actual_paths.update(
            _enumerate_regular_files_without_links(
                root,
                PurePosixPath(split),
            )
        )
    if actual_paths != listed_paths:
        missing = sorted(listed_paths - actual_paths)
        unexpected = sorted(actual_paths - listed_paths)
        raise PreparationError(
            "prepared files do not match files.sha256; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for record in records:
        payload, metadata = _read_regular_file_without_links(
            root,
            PurePosixPath(record.relative_path),
            label="prepared image",
            read_content=full,
        )
        if metadata.st_size != record.size_bytes:
            raise PreparationError(
                f"prepared file size mismatch: {record.relative_path!r}"
            )
        if full:
            assert payload is not None
            actual_digest = hashlib.sha256(payload).hexdigest()
            if actual_digest != record.sha256:
                raise PreparationError(
                    f"prepared file digest mismatch: {record.relative_path!r}"
                )

    artifact_digest = _canonical_digest(
        {
            "inventory_sha256": inventory_digest,
            "recipe_sha256": expected_recipe_hash,
        }
    )
    if manifest.get("artifact_digest") != artifact_digest:
        raise PreparationError("prepared artifact digest does not match its content")
    return PreparedArtifact(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        artifact_digest=artifact_digest,
        preparation_key=expected_preparation_key,
        file_count=len(records),
        image_records=records,
        cache_hit=True,
    )


def _process_images(
    *,
    archive_stream: BinaryIO,
    archive_label: str,
    images: Sequence[SourceImage],
    validation_members: frozenset[str],
    staging_root: Path,
    cache_root: Path,
    input_resolution: int,
    output_resolution: int,
) -> tuple[list[PreparedImageRecord], str]:
    output_records: list[PreparedImageRecord] = []
    source_pixel_records: list[tuple[str, str]] = []
    try:
        archive_stream.seek(0)
        with ZipFile(archive_stream) as archive:
            for index, image in enumerate(images, start=1):
                try:
                    payload = archive.read(image.member_name)
                except (BadZipFile, KeyError, RuntimeError) as error:
                    raise PreparationError(
                        f"failed ZIP integrity check for {image.member_name!r}"
                    ) from error
                if len(payload) != image.file_size:
                    raise PreparationError(
                        f"source member size changed: {image.member_name!r}"
                    )
                prepared, pixel_digest = _decode_and_resize(
                    payload,
                    member_name=image.member_name,
                    input_resolution=input_resolution,
                    output_resolution=output_resolution,
                )
                if image.source_split == "test":
                    output_split = "test"
                elif image.relative_path in validation_members:
                    output_split = "validation"
                else:
                    output_split = "train"
                relative = f"{output_split}/{image.class_name}/{image.filename}"
                destination = staging_root / PurePosixPath(relative)
                output_digest, output_size = _save_prepared_png(
                    prepared,
                    destination,
                    cache_root=cache_root,
                )
                output_records.append(
                    PreparedImageRecord(
                        relative_path=relative,
                        size_bytes=output_size,
                        sha256=output_digest,
                    )
                )
                source_pixel_records.append((pixel_digest, image.relative_path))
                if index % 250 == 0 or index == len(images):
                    print(
                        f"Prepared {index:,}/{len(images):,} images",
                        flush=True,
                    )
    except BadZipFile as error:
        raise PreparationError(
            f"archive became invalid while preparing: {archive_label}"
        ) from error

    output_records.sort(key=lambda item: item.relative_path)
    source_pixel_records.sort(key=lambda item: item[1])
    source_inventory_digest = _canonical_digest(
        [
            {"path": relative, "rgb_sha256": digest}
            for digest, relative in source_pixel_records
        ]
    )
    return output_records, source_inventory_digest


def _quarantine_invalid_prepared_artifact(
    root: Path,
    *,
    cache_root: Path,
) -> Path:
    try:
        return quarantine_cache_entry(
            cache_root,
            root,
            suffix=f"{time.time_ns()}.invalid",
            label="invalid AFHQ-v2 prepared artifact",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot quarantine invalid prepared artifact: {root}"
        ) from error


def prepare_archive(
    *,
    source: SourceArchive,
    lock: SourceLock,
    cache_root: Path,
    resolution: int = 128,
    validation_per_class: int = 300,
    validation_seed: str = _DEFAULT_VALIDATION_SEED,
    repair_invalid: bool = False,
) -> PreparedArtifact:
    """Validate and deterministically turn AFHQ-v2 into a training artifact."""

    if type(repair_invalid) is not bool:
        raise TypeError("repair_invalid must be boolean")
    plan = build_preparation_plan(
        lock=lock,
        resolution=resolution,
        validation_per_class=validation_per_class,
        validation_seed=validation_seed,
    )
    if (
        source.sha256 != lock.expected_sha256
        or source.size_bytes != lock.expected_bytes
    ):
        raise SourceIntegrityError(
            "source archive identity does not match the source lock",
            actual_sha256=source.sha256,
            actual_bytes=source.size_bytes,
        )
    recipe = plan.recipe
    recipe_hash = plan.recipe_sha256
    preparation_key = plan.preparation_key
    try:
        canonical_cache_root = ensure_cache_directory(
            cache_root,
            cache_root,
            label="AFHQ-v2 artifact cache root",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot create AFHQ-v2 artifact cache: {cache_root}"
        ) from error
    prepared_base = (
        canonical_cache_root / "prepared" / "afhq-v2" / str(resolution)
    )
    final_root = prepared_base / preparation_key
    lock_path = (
        canonical_cache_root
        / ".locks"
        / f"afhq-v2-prepare-{preparation_key}.lock"
    )

    with SourceArchiveSession(source, lock=lock) as source_session:
        archive_stream = source_session.stream
        assert archive_stream is not None
        images = _inspect_archive_stream(
            archive_stream,
            archive_label=str(source.path),
            contract=lock.contract,
        )
        validation_members = select_validation_members(
            images,
            classes=lock.contract.classes,
            per_class=validation_per_class,
            seed=validation_seed,
        )
        counts = _prepared_counts(
            images,
            validation_members=validation_members,
            classes=lock.contract.classes,
        )
        if counts != plan.counts:
            raise PreparationError(
                "source archive class counts do not match the pinned "
                "preparation plan"
            )

        with ArtifactPreparationLock(
            lock_path,
            cache_root=canonical_cache_root,
        ):
            try:
                final_exists = cache_entry_exists(
                    canonical_cache_root,
                    final_root,
                    label="AFHQ-v2 prepared artifact",
                )
            except (OSError, ValueError) as error:
                raise PreparationError(
                    "cannot inspect AFHQ-v2 prepared artifact directory: "
                    f"{final_root}"
                ) from error
            if final_exists:
                try:
                    cached = verify_prepared_artifact(
                        final_root,
                        expected_preparation_key=preparation_key,
                        expected_recipe=recipe,
                        source_archive=source,
                        source_lock=lock,
                        expected_counts=counts,
                    )
                except PreparationError:
                    if not repair_invalid:
                        raise
                    _quarantine_invalid_prepared_artifact(
                        final_root,
                        cache_root=canonical_cache_root,
                    )
                else:
                    source_session.verify_unchanged()
                    return cached

            try:
                ensure_cache_directory(
                    canonical_cache_root,
                    prepared_base,
                    label="AFHQ-v2 prepared artifact directory",
                )
            except (OSError, ValueError) as error:
                raise PreparationError(
                    f"cannot create AFHQ-v2 prepared artifact directory: "
                    f"{prepared_base}"
                ) from error
            staging_root = prepared_base / (
                f".{preparation_key}.tmp-{os.getpid()}-{uuid4().hex}"
            )
            staging_created = False
            try:
                create_cache_directory(
                    canonical_cache_root,
                    staging_root,
                    label="AFHQ-v2 prepared artifact staging",
                )
                staging_created = True
                output_records, source_inventory_digest = _process_images(
                    archive_stream=archive_stream,
                    archive_label=str(source.path),
                    images=images,
                    validation_members=validation_members,
                    staging_root=staging_root,
                    cache_root=canonical_cache_root,
                    input_resolution=lock.contract.input_resolution,
                    output_resolution=resolution,
                )
                inventory_text = "".join(
                    f"{record.sha256}  {record.size_bytes}  "
                    f"{record.relative_path}\n"
                    for record in output_records
                )
                inventory_path = staging_root / "files.sha256"
                _write_text_atomic(
                    inventory_path,
                    inventory_text,
                    cache_root=canonical_cache_root,
                    label="AFHQ-v2 prepared inventory",
                )
                inventory_digest = hashlib.sha256(
                    inventory_text.encode("utf-8")
                ).hexdigest()
                artifact_digest = _canonical_digest(
                    {
                        "inventory_sha256": inventory_digest,
                        "recipe_sha256": recipe_hash,
                    },
                )
                manifest: dict[str, object] = {
                    "schema_version": 1,
                    "dataset": {
                        "name": "AFHQ-v2",
                        "version": 2,
                        "homepage": lock.homepage,
                        "license": {
                            "name": lock.license_name,
                            "url": lock.license_url,
                        },
                        "citation": lock.citation,
                        "class_mapping": dict(lock.contract.class_mapping),
                    },
                    "source": {
                        "type": "official_archive",
                        "url": lock.url,
                        "archive": {
                            "name": lock.archive_name,
                            "sha256": source.sha256,
                            "bytes": source.size_bytes,
                        },
                        "source_splits": {
                            "train": lock.contract.train_count,
                            "test": lock.contract.test_count,
                        },
                        "source_class_counts": {
                            split: dict(source_counts)
                            for split, source_counts in (
                                lock.contract.source_class_counts or {}
                            ).items()
                        },
                        "total_count": lock.contract.total_count,
                        "canonical_rgb_inventory_sha256": (
                            source_inventory_digest
                        ),
                    },
                    "preparation": {
                        "key": preparation_key,
                        "recipe": recipe,
                        "recipe_sha256": recipe_hash,
                    },
                    "counts": _manifest_counts(counts),
                    "inventory": {
                        "path": "files.sha256",
                        "file_count": len(output_records),
                        "sha256": inventory_digest,
                    },
                    "artifact_digest": artifact_digest,
                }
                manifest_path = staging_root / "dataset_manifest.yaml"
                _write_text_atomic(
                    manifest_path,
                    _manifest_text(manifest),
                    cache_root=canonical_cache_root,
                    label="AFHQ-v2 prepared manifest",
                )
                source_session.verify_unchanged()
                try:
                    publish_cache_directory(
                        canonical_cache_root,
                        staging_root,
                        final_root,
                        label="AFHQ-v2 prepared artifact",
                    )
                except FileExistsError:
                    remove_cache_directory(
                        canonical_cache_root,
                        staging_root,
                        label="AFHQ-v2 losing prepared artifact staging",
                    )
                    return verify_prepared_artifact(
                        final_root,
                        expected_preparation_key=preparation_key,
                        expected_recipe=recipe,
                        source_archive=source,
                        source_lock=lock,
                        expected_counts=counts,
                    )
                published = verify_prepared_artifact(
                    final_root,
                    expected_preparation_key=preparation_key,
                    expected_recipe=recipe,
                    source_archive=source,
                    source_lock=lock,
                    expected_counts=counts,
                )
                return PreparedArtifact(
                    root=published.root,
                    manifest_path=published.manifest_path,
                    manifest_sha256=published.manifest_sha256,
                    artifact_digest=published.artifact_digest,
                    preparation_key=published.preparation_key,
                    file_count=published.file_count,
                    image_records=published.image_records,
                    cache_hit=False,
                )
            except BaseException:
                if staging_created and cache_entry_exists(
                    canonical_cache_root,
                    staging_root,
                    label="AFHQ-v2 prepared artifact staging cleanup",
                ):
                    remove_cache_directory(
                        canonical_cache_root,
                        staging_root,
                        label="AFHQ-v2 prepared artifact staging cleanup",
                    )
                raise
