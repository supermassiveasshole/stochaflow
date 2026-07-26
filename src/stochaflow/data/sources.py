"""Artifact-backed image data sources and their payload contracts."""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import socket
import stat
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from uuid import uuid4

from torchvision import datasets

from stochaflow.data.artifact_io import (
    ArtifactFileSnapshot,
    cache_entry_exists,
    canonical_directory,
    create_cache_directory,
    lexical_absolute_path,
    open_cache_file,
    publish_cache_directory,
    quarantine_cache_entry,
    read_regular_file,
    remove_cache_directory,
    scan_regular_files,
    write_cache_file,
)
from stochaflow.data.artifacts import (
    DataArtifact,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataSource,
    DataSourceContext,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
    ReferencedDataArtifact,
    ReferencedDataArtifactIdentity,
)
from stochaflow.data.recipe_config import ImageSourceConfig
from stochaflow.utils.config import ConfigError, coerce_config_section
from stochaflow.utils.registry import Registry

if TYPE_CHECKING:
    from stochaflow.data.builder import DataBuilderContext

_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_INVENTORY_RECORD_LIMIT = 100_000
_LOCK_WAIT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.05
_WINDOWS_LOCK_BYTE_OFFSET = 4096
_ADVISORY_LOCK_API: Any = importlib.import_module(
    "msvcrt" if os.name == "nt" else "fcntl"
)
_REFERENCE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "artifact_type",
        "source_name",
        "source_digest",
        "materializer_name",
        "materialization_digest",
        "layout",
        "inventory",
        "artifact_digest",
    }
)
_INVENTORY_FIELDS = frozenset(
    {"record_limit", "record_count", "shards"}
)
_SHARD_FIELDS = frozenset({"path", "record_count", "sha256"})
_RECORD_FIELDS = frozenset({"tree", "path", "size_bytes", "sha256"})
_MANAGED_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_MANAGED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "artifact_type",
        "source_name",
        "source_digest",
        "materializer_name",
        "materialization_digest",
        "artifact_digest",
        "files",
    }
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{path} field names must be strings")
    names = cast(set[str], set(raw))
    missing = sorted(fields - names)
    unknown = sorted(names - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")
    return cast(dict[str, Any], value)


def _load_canonical_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        encoded, _ = read_regular_file(
            path.parent,
            path.name,
            label=label,
        )
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    if encoded != _canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], value), encoded


def _unknown_fields(
    raw: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        rendered = ", ".join(f"{path}.{name}" for name in unknown)
        raise ConfigError(f"unknown config field(s): {rendered}")


def _absolute_directory(value: object, *, path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty path string")
    try:
        return canonical_directory(Path(value), label=path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{path} does not exist: {value}") from exc


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def _validate_relative_path(value: str, *, path: str) -> str:
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
        if stem in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"{path} contains a Windows-reserved name")
    if pure.as_posix() != value:
        raise ValueError(f"{path} must be a normalized POSIX path")
    return value


@dataclass(frozen=True, slots=True)
class ImageFileRecord:
    """One immutable image record from a canonical artifact inventory."""

    tree: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        tree = cast(object, self.tree)
        if not isinstance(tree, str) or not tree:
            raise ValueError("image file record tree must be non-empty")
        _validate_relative_path(self.path, path="image file record.path")
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record in canonical inventory form."""

        return {
            "tree": self.tree,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str) -> ImageFileRecord:
        """Parse one strict inventory record."""

        raw = _strict_mapping(value, fields=_RECORD_FIELDS, path=path)
        return cls(
            tree=raw["tree"],
            path=raw["path"],
            size_bytes=raw["size_bytes"],
            sha256=raw["sha256"],
        )


@dataclass(frozen=True, slots=True)
class ImageFilePair:
    """One high/low-resolution pair matched by relative stem."""

    high_resolution: ImageFileRecord
    low_resolution: ImageFileRecord


@dataclass(frozen=True, slots=True)
class TorchvisionImageArtifactPayload:
    """Locator for one fully acquired torchvision image dataset."""

    dataset: Literal["mnist", "cifar10", "flowers102"]
    root: Path

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


@dataclass(frozen=True, slots=True)
class ImageFolderArtifactPayload:
    """Native image partitions and their canonical file inventories."""

    roots: Mapping[str, Path]
    train: tuple[ImageFileRecord, ...]
    validation: tuple[ImageFileRecord, ...] | None = None
    test: tuple[ImageFileRecord, ...] | None = None

    def __post_init__(self) -> None:
        roots = {
            name: canonical_directory(
                Path(root),
                label=f"image folder payload root {name}",
            )
            for name, root in self.roots.items()
        }
        if not roots:
            raise FileNotFoundError("image folder payload roots must exist")
        object.__setattr__(self, "roots", roots)
        if not self.train:
            raise ValueError("image folder payload train inventory must not be empty")
        for role in ("train", "validation", "test"):
            records = getattr(self, role)
            if records is None:
                continue
            if any(record.tree not in roots for record in records):
                raise ValueError(
                    f"image folder payload {role} inventory uses an unknown tree"
                )


@dataclass(frozen=True, slots=True)
class PairedImageFolderArtifactPayload:
    """Native paired-image partitions and immutable pair inventories."""

    roots: Mapping[str, Path]
    train: tuple[ImageFilePair, ...]
    validation: tuple[ImageFilePair, ...] | None = None
    test: tuple[ImageFilePair, ...] | None = None

    def __post_init__(self) -> None:
        roots = {
            name: canonical_directory(
                Path(root),
                label=f"paired image payload root {name}",
            )
            for name, root in self.roots.items()
        }
        if not roots:
            raise FileNotFoundError("paired image payload roots must exist")
        object.__setattr__(self, "roots", roots)
        if not self.train:
            raise ValueError("paired image payload train inventory must not be empty")
        for role in ("train", "validation", "test"):
            pairs = getattr(self, role)
            if pairs is None:
                continue
            records = (
                record
                for pair in pairs
                for record in (pair.high_resolution, pair.low_resolution)
            )
            if any(record.tree not in roots for record in records):
                raise ValueError(
                    f"paired image payload {role} inventory uses an unknown tree"
                )


type ImageArtifactPayload = (
    TorchvisionImageArtifactPayload
    | ImageFolderArtifactPayload
    | PairedImageFolderArtifactPayload
)


class ArtifactMaterializationLock(AbstractContextManager["ArtifactMaterializationLock"]):
    """Cooperative lock with bounded waiting and diagnosable ownership."""

    def __init__(
        self,
        path: Path,
        *,
        cache_root: Path | None = None,
        wait_seconds: float = _LOCK_WAIT_SECONDS,
        poll_seconds: float = _LOCK_POLL_SECONDS,
    ) -> None:
        if wait_seconds < 0:
            raise ValueError("artifact lock wait_seconds must be non-negative")
        if poll_seconds <= 0:
            raise ValueError("artifact lock poll_seconds must be positive")
        self.path = lexical_absolute_path(path)
        self.cache_root = lexical_absolute_path(
            cache_root if cache_root is not None else self.path.parent
        )
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self.handle: int | None = None
        self.owner_bytes: bytes | None = None

    def __enter__(self) -> Self:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            handle = open_cache_file(
                self.cache_root,
                self.path,
                label="artifact materialization lock",
            )
            if _try_acquire_advisory_lock(handle):
                self.handle = handle
                break
            os.close(handle)
            if time.monotonic() >= deadline:
                owner = _lock_owner_diagnostic(self.path)
                raise RuntimeError(
                    "timed out waiting for data artifact materialization "
                    f"lock: {self.path}; observed owner: {owner}"
                )
            time.sleep(
                min(self.poll_seconds, max(0.0, deadline - time.monotonic()))
            )
        owner = {
            "schema_version": 1,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at_ns": time.time_ns(),
            "nonce": uuid4().hex,
        }
        self.owner_bytes = _canonical_json_bytes(owner)
        try:
            os.lseek(self.handle, 0, os.SEEK_SET)
            _write_descriptor(self.handle, self.owner_bytes)
            os.ftruncate(self.handle, len(self.owner_bytes))
            os.fsync(self.handle)
        except BaseException:
            _release_advisory_lock(self.handle)
            os.close(self.handle)
            self.handle = None
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.handle is None:
            return
        _release_advisory_lock(self.handle)
        os.close(self.handle)
        self.handle = None


def _write_descriptor(descriptor: int, encoded: bytes) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write artifact lock metadata")
        remaining = remaining[written:]


def _try_acquire_advisory_lock(descriptor: int) -> bool:
    if os.name == "nt":
        os.lseek(descriptor, _WINDOWS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        try:
            _ADVISORY_LOCK_API.locking(
                descriptor,
                _ADVISORY_LOCK_API.LK_NBLCK,
                1,
            )
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    try:
        _ADVISORY_LOCK_API.flock(
            descriptor,
            _ADVISORY_LOCK_API.LOCK_EX | _ADVISORY_LOCK_API.LOCK_NB,
        )
    except BlockingIOError:
        return False
    return True


def _release_advisory_lock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, _WINDOWS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        _ADVISORY_LOCK_API.locking(
            descriptor,
            _ADVISORY_LOCK_API.LK_UNLCK,
            1,
        )
        return

    _ADVISORY_LOCK_API.flock(descriptor, _ADVISORY_LOCK_API.LOCK_UN)


def _lock_owner_diagnostic(path: Path) -> str:
    try:
        encoded, _ = read_regular_file(
            path.parent,
            path.name,
            label="artifact materialization lock",
        )
        value = json.loads(encoded.decode("utf-8"))
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return "unreadable lock metadata"
    if not isinstance(value, dict):
        return "non-mapping lock metadata"
    fields = ("hostname", "pid", "created_at_ns", "nonce")
    return ", ".join(f"{name}={value.get(name)!r}" for name in fields)


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


def _scan_regular_file_snapshots(
    root: Path,
    *,
    hash_contents: bool,
    label: str,
    path_filter: Callable[[str], bool] | None = None,
) -> tuple[ArtifactFileSnapshot, ...]:
    return scan_regular_files(
        root,
        hash_contents=hash_contents,
        label=label,
        path_filter=path_filter,
    )


def _scan_image_tree(
    root: Path,
    *,
    tree: str,
    hash_contents: bool,
) -> tuple[ImageFileRecord, ...]:
    records: list[ImageFileRecord] = []
    collision_keys: dict[str, str] = {}
    snapshots = _scan_regular_file_snapshots(
        root,
        hash_contents=hash_contents,
        label="referenced data",
        path_filter=(
            lambda relative: (
                PurePosixPath(relative).suffix.lower() in _IMAGE_SUFFIXES
            )
        ),
    )
    for snapshot in snapshots:
        relative = snapshot.relative_path
        if relative != unicodedata.normalize("NFC", relative):
            raise ValueError(
                f"referenced file path must use NFC normalization: {relative}"
            )
        _validate_relative_path(relative, path=f"referenced file {relative}")
        collision_key = relative.casefold()
        previous = collision_keys.get(collision_key)
        if previous is not None:
            raise ValueError(
                "referenced data contains a case/NFC path collision: "
                f"{previous!r}, {relative!r}"
            )
        collision_keys[collision_key] = relative
        digest = snapshot.sha256 if hash_contents else "0" * 64
        if digest is None:
            raise RuntimeError("hashed reference scan did not return a digest")
        records.append(
            ImageFileRecord(
                tree=tree,
                path=relative,
                size_bytes=snapshot.size_bytes,
                sha256=digest,
            )
        )
    records.sort(key=lambda record: (record.tree, record.path))
    if not records:
        raise ValueError(f"image directory contains no supported images: {root}")
    return tuple(records)


def _records_without_hash(
    records: Sequence[ImageFileRecord],
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (record.tree, record.path, record.size_bytes) for record in records
    )


def _assert_unique_records(records: Sequence[ImageFileRecord]) -> None:
    keys = [(record.tree, record.path.casefold()) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("reference inventory contains duplicate paths")


def _write_inventory(
    cache_root: Path,
    index_root: Path,
    records: Sequence[ImageFileRecord],
) -> dict[str, Any]:
    inventory_root = index_root / "inventory"
    create_cache_directory(
        cache_root,
        inventory_root,
        label="reference inventory directory",
    )
    shards: list[dict[str, Any]] = []
    for shard_index, offset in enumerate(
        range(0, len(records), _INVENTORY_RECORD_LIMIT)
    ):
        selected = records[offset : offset + _INVENTORY_RECORD_LIMIT]
        relative_path = f"inventory/{shard_index:06d}.jsonl"
        encoded = b"".join(
            _canonical_json_bytes(record.to_dict()) for record in selected
        )
        write_cache_file(
            cache_root,
            index_root / relative_path,
            encoded,
            label="reference inventory shard",
        )
        shards.append(
            {
                "path": relative_path,
                "record_count": len(selected),
                "sha256": _sha256_bytes(encoded),
            }
        )
    return {
        "record_limit": _INVENTORY_RECORD_LIMIT,
        "record_count": len(records),
        "shards": shards,
    }


def _read_inventory(
    index_root: Path,
    value: object,
) -> tuple[ImageFileRecord, ...]:
    inventory = _strict_mapping(
        value,
        fields=_INVENTORY_FIELDS,
        path="reference manifest.inventory",
    )
    if (
        type(inventory["record_limit"]) is not int
        or inventory["record_limit"] != _INVENTORY_RECORD_LIMIT
    ):
        raise ValueError("reference inventory record_limit is invalid")
    if (
        type(inventory["record_count"]) is not int
        or inventory["record_count"] <= 0
    ):
        raise ValueError("reference inventory record_count must be positive")
    record_count = cast(int, inventory["record_count"])
    serialized_shards = inventory["shards"]
    if not isinstance(serialized_shards, list) or not serialized_shards:
        raise ValueError("reference inventory shards must be a non-empty list")
    expected_shard_count = (
        record_count + _INVENTORY_RECORD_LIMIT - 1
    ) // _INVENTORY_RECORD_LIMIT
    if len(serialized_shards) != expected_shard_count:
        raise ValueError("reference inventory shard count is not canonical")
    records: list[ImageFileRecord] = []
    for shard_index, value in enumerate(serialized_shards):
        shard = _strict_mapping(
            value,
            fields=_SHARD_FIELDS,
            path=f"reference manifest.inventory.shards[{shard_index}]",
        )
        expected_path = f"inventory/{shard_index:06d}.jsonl"
        if shard["path"] != expected_path:
            raise ValueError("reference inventory shard paths are not canonical")
        expected_records = min(
            _INVENTORY_RECORD_LIMIT,
            record_count - shard_index * _INVENTORY_RECORD_LIMIT,
        )
        if (
            type(shard["record_count"]) is not int
            or shard["record_count"] != expected_records
        ):
            raise ValueError(
                "reference inventory shard record count is not canonical"
            )
        shard_digest = shard["sha256"]
        if (
            not isinstance(shard_digest, str)
            or len(shard_digest) != 64
            or shard_digest != shard_digest.lower()
            or any(
                character not in "0123456789abcdef"
                for character in shard_digest
            )
        ):
            raise ValueError("reference inventory shard digest is invalid")
        encoded, _ = read_regular_file(
            index_root,
            expected_path,
            label="reference inventory shard",
        )
        if _sha256_bytes(encoded) != shard_digest:
            raise ValueError("reference inventory shard digest mismatch")
        lines = encoded.splitlines(keepends=True)
        if len(lines) != shard["record_count"]:
            raise ValueError("reference inventory shard record count mismatch")
        for line_index, line in enumerate(lines):
            try:
                raw_record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("reference inventory contains invalid JSON") from exc
            if line != _canonical_json_bytes(raw_record):
                raise ValueError("reference inventory record is not canonical JSON")
            records.append(
                ImageFileRecord.from_dict(
                    raw_record,
                    path=(
                        "reference manifest.inventory."
                        f"shards[{shard_index}][{line_index}]"
                    ),
                )
            )
    if len(records) != record_count:
        raise ValueError("reference inventory total record count mismatch")
    if tuple(records) != tuple(
        sorted(records, key=lambda record: (record.tree, record.path))
    ):
        raise ValueError("reference inventory records must be sorted")
    _assert_unique_records(records)
    return tuple(records)


def _reference_locator_path(
    cache_root: Path,
    source_name: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
) -> Path:
    locator_digest = _canonical_digest(
        {
            "source_name": source_name,
            "roots": {
                name: str(root)
                for name, root in sorted(roots.items())
            },
            "layout": layout,
        }
    )
    return (
        cache_root
        / "references"
        / "locators"
        / _canonical_digest(source_name)[:16]
        / f"{locator_digest}.json"
    )


def _reference_index_path(
    cache_root: Path,
    source_name: str,
    artifact_digest: str,
) -> Path:
    return (
        cache_root
        / "references"
        / _canonical_digest(source_name)[:16]
        / artifact_digest
    )


def _reference_lock_path(
    cache_root: Path,
    source_name: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
) -> Path:
    lock_digest = _canonical_digest(
        {
            "source": source_name,
            "roots": {
                name: str(root) for name, root in sorted(roots.items())
            },
            "layout": layout,
        }
    )
    return cache_root / "references" / "locks" / f"{lock_digest}.lock"


def _read_locator(path: Path) -> str:
    raw, _ = _load_canonical_json(path, label="reference locator")
    if set(raw) != {"artifact_digest"}:
        raise ValueError("reference locator has invalid fields")
    digest = raw["artifact_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("reference locator artifact_digest is invalid")
    return digest


def _write_locator(
    cache_root: Path,
    path: Path,
    artifact_digest: str,
) -> None:
    write_cache_file(
        cache_root,
        path,
        _canonical_json_bytes({"artifact_digest": artifact_digest}),
        label="data artifact locator",
    )


def _quarantine_path(cache_root: Path, path: Path) -> Path:
    return quarantine_cache_entry(
        cache_root,
        path,
        suffix="corrupt",
        label="corrupt data artifact cache entry",
    )


def _path_exists_without_following(cache_root: Path, path: Path) -> bool:
    return cache_entry_exists(
        cache_root,
        path,
        label="data artifact cache entry",
    )


def _read_locator_for_policy(
    cache_root: Path,
    path: Path,
    *,
    policy: Literal["require", "ensure"],
    quarantine_on_error: bool = True,
) -> str | None:
    if not _path_exists_without_following(cache_root, path):
        return None
    try:
        return _read_locator(path)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        if policy == "require":
            raise
        if quarantine_on_error and _path_exists_without_following(
            cache_root,
            path,
        ):
            with suppress(FileNotFoundError):
                _quarantine_path(cache_root, path)
        return None


def _reference_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> ReferencedDataArtifactIdentity:
    return ReferencedDataArtifactIdentity(
        artifact_type=manifest["artifact_type"],
        source_name=manifest["source_name"],
        source_digest=manifest["source_digest"],
        materializer_name=manifest["materializer_name"],
        materialization_digest=manifest["materialization_digest"],
        artifact_digest=manifest["artifact_digest"],
        manifest_sha256=manifest_sha256,
    )


def _load_reference_index(
    index_root: Path,
    *,
    source_name: str,
    artifact_type: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
    verification: Literal["manifest", "full"],
) -> tuple[ReferencedDataArtifactIdentity, tuple[ImageFileRecord, ...]]:
    manifest_path = index_root / "manifest.json"
    raw, manifest_bytes = _load_canonical_json(
        manifest_path,
        label="reference manifest",
    )
    manifest = _strict_mapping(
        raw,
        fields=_REFERENCE_MANIFEST_FIELDS,
        path="reference manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["kind"] != "referenced"
    ):
        raise ValueError("reference manifest has an unsupported schema")
    if (
        manifest["source_name"] != source_name
        or manifest["artifact_type"] != artifact_type
        or manifest["layout"] != dict(layout)
    ):
        raise ValueError("reference manifest does not match the selected source")
    identity = _reference_manifest_identity(
        manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )
    if index_root.name != identity.artifact_digest:
        raise ValueError(
            "reference index directory does not match its artifact digest"
        )
    records = _read_inventory(index_root, manifest["inventory"])
    live_records = tuple(
        record
        for tree, root in sorted(roots.items())
        for record in _scan_image_tree(
            root,
            tree=tree,
            hash_contents=verification == "full",
        )
    )
    live_records = tuple(
        sorted(live_records, key=lambda record: (record.tree, record.path))
    )
    if _records_without_hash(live_records) != _records_without_hash(records):
        raise ValueError("referenced data paths or sizes changed")
    if verification == "full" and live_records != records:
        raise ValueError("referenced data content digest changed")
    if identity.source_digest != _canonical_digest(
        [record.to_dict() for record in records]
    ):
        raise ValueError("reference manifest source digest is invalid")
    expected_materialization_digest = _canonical_digest(
        {
            "name": identity.materializer_name,
            "version": 1,
            "layout": layout,
        }
    )
    if identity.materialization_digest != expected_materialization_digest:
        raise ValueError(
            "reference manifest materialization digest is invalid"
        )
    expected_artifact_digest = _canonical_digest(
        {
            "kind": "referenced",
            "artifact_type": artifact_type,
            "source_name": source_name,
            "source_digest": identity.source_digest,
            "materializer_name": identity.materializer_name,
            "materialization_digest": identity.materialization_digest,
            "inventory_digest": _canonical_digest(
                [record.to_dict() for record in records]
            ),
        }
    )
    if identity.artifact_digest != expected_artifact_digest:
        raise ValueError("reference manifest artifact digest is invalid")
    return identity, records


def _build_reference_index(
    *,
    cache_root: Path,
    source_name: str,
    artifact_type: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
    expected_identity: ReferencedDataArtifactIdentity | None,
) -> tuple[Path, ReferencedDataArtifactIdentity, tuple[ImageFileRecord, ...]]:
    records = tuple(
        record
        for tree, root in sorted(roots.items())
        for record in _scan_image_tree(root, tree=tree, hash_contents=True)
    )
    records = tuple(
        sorted(records, key=lambda record: (record.tree, record.path))
    )
    _assert_unique_records(records)
    source_digest = _canonical_digest(
        [record.to_dict() for record in records]
    )
    materializer_name = "stochaflow.reference-image-inventory"
    materialization_digest = _canonical_digest(
        {
            "name": materializer_name,
            "version": 1,
            "layout": layout,
        }
    )
    artifact_digest = _canonical_digest(
        {
            "kind": "referenced",
            "artifact_type": artifact_type,
            "source_name": source_name,
            "source_digest": source_digest,
            "materializer_name": materializer_name,
            "materialization_digest": materialization_digest,
            "inventory_digest": _canonical_digest(
                [record.to_dict() for record in records]
            ),
        }
    )
    final_root = _reference_index_path(
        cache_root,
        source_name,
        artifact_digest,
    )
    staging_root = final_root.parent / f".{artifact_digest}.{uuid4().hex}.tmp"
    create_cache_directory(
        cache_root,
        staging_root,
        label="reference index staging directory",
    )
    try:
        inventory = _write_inventory(cache_root, staging_root, records)
        manifest = {
            "schema_version": 1,
            "kind": "referenced",
            "artifact_type": artifact_type,
            "source_name": source_name,
            "source_digest": source_digest,
            "materializer_name": materializer_name,
            "materialization_digest": materialization_digest,
            "layout": dict(layout),
            "inventory": inventory,
            "artifact_digest": artifact_digest,
        }
        manifest_path = staging_root / "manifest.json"
        manifest_bytes = _canonical_json_bytes(manifest)
        write_cache_file(
            cache_root,
            manifest_path,
            manifest_bytes,
            label="reference manifest",
        )
        identity = _reference_manifest_identity(
            manifest,
            manifest_sha256=_sha256_bytes(manifest_bytes),
        )
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(
                "strict resume referenced data identity does not match"
            )
        if _path_exists_without_following(cache_root, final_root):
            try:
                winner_identity, winner_records = _load_reference_index(
                    final_root,
                    source_name=source_name,
                    artifact_type=artifact_type,
                    roots=roots,
                    layout=layout,
                    verification="full",
                )
            except (FileNotFoundError, OSError, TypeError, ValueError):
                _quarantine_path(cache_root, final_root)
            else:
                remove_cache_directory(
                    cache_root,
                    staging_root,
                    label="reference index staging directory",
                )
                return final_root, winner_identity, winner_records
        try:
            publish_cache_directory(
                cache_root,
                staging_root,
                final_root,
                label="reference index",
            )
        except FileExistsError:
            winner_identity, winner_records = _load_reference_index(
                final_root,
                source_name=source_name,
                artifact_type=artifact_type,
                roots=roots,
                layout=layout,
                verification="full",
            )
            remove_cache_directory(
                cache_root,
                staging_root,
                label="reference index staging directory",
            )
            return final_root, winner_identity, winner_records
        return final_root, identity, records
    except BaseException:
        if _path_exists_without_following(cache_root, staging_root):
            remove_cache_directory(
                cache_root,
                staging_root,
                label="reference index staging directory",
            )
        raise


def _materialize_reference(
    context: DataSourceContext,
    *,
    source_name: str,
    artifact_type: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
) -> tuple[Path, ReferencedDataArtifactIdentity, tuple[ImageFileRecord, ...]]:
    cache_root = context.cache_root
    for root in roots.values():
        if _is_relative_to(cache_root, root) or _is_relative_to(
            root,
            cache_root,
        ):
            raise ValueError(
                "data source cache_root and referenced roots must not overlap"
            )
    expected = context.expected_identity
    locator = _reference_locator_path(
        cache_root,
        source_name,
        roots,
        layout,
    )
    if expected is not None:
        if not isinstance(expected, ReferencedDataArtifactIdentity):
            raise ValueError(
                "strict resume expected a different data artifact kind"
            )
        if (
            expected.source_name != source_name
            or expected.artifact_type != artifact_type
        ):
            raise ValueError(
                "strict resume expected a different referenced data source"
            )
        artifact_digest: str | None = expected.artifact_digest
    else:
        artifact_digest = _read_locator_for_policy(
            cache_root,
            locator,
            policy=context.policy,
            quarantine_on_error=False,
        )
    if artifact_digest is not None:
        index_root = _reference_index_path(
            cache_root,
            source_name,
            artifact_digest,
        )
        try:
            identity, records = _load_reference_index(
                index_root,
                source_name=source_name,
                artifact_type=artifact_type,
                roots=roots,
                layout=layout,
                verification=context.verification,
            )
            if expected is not None and identity != expected:
                raise ValueError(
                    "strict resume referenced data identity does not match"
                )
            return index_root, identity, records
        except (FileNotFoundError, OSError, TypeError, ValueError):
            if context.policy == "require":
                raise
    if context.policy == "require":
        raise FileNotFoundError(
            f"required reference artifact is not indexed for '{source_name}'"
        )
    lock_path = _reference_lock_path(
        cache_root,
        source_name,
        roots,
        layout,
    )
    with ArtifactMaterializationLock(lock_path, cache_root=cache_root):
        winner_digest = (
            expected.artifact_digest
            if expected is not None
            else _read_locator_for_policy(
                cache_root,
                locator,
                policy="ensure",
            )
        )
        if winner_digest is not None:
            winner_root = _reference_index_path(
                cache_root,
                source_name,
                winner_digest,
            )
            try:
                winner_identity, winner_records = _load_reference_index(
                    winner_root,
                    source_name=source_name,
                    artifact_type=artifact_type,
                    roots=roots,
                    layout=layout,
                    verification=context.verification,
                )
                if expected is not None and winner_identity != expected:
                    raise ValueError(
                        "strict resume referenced data identity does not match"
                    )
                return winner_root, winner_identity, winner_records
            except (FileNotFoundError, OSError, TypeError, ValueError):
                if _path_exists_without_following(cache_root, winner_root):
                    _quarantine_path(cache_root, winner_root)
        index_root, identity, records = _build_reference_index(
            cache_root=cache_root,
            source_name=source_name,
            artifact_type=artifact_type,
            roots=roots,
            layout=layout,
            expected_identity=expected,
        )
        if expected is not None and identity != expected:
            raise ValueError(
                "strict resume referenced data identity does not match"
            )
        if expected is None:
            _write_locator(cache_root, locator, identity.artifact_digest)
    return index_root, identity, records


def _partition_roots(root: Path, layout: str) -> dict[str, Path]:
    if layout == "flat":
        return {"train": root}
    if layout != "split":
        raise ConfigError("image source params.layout must be flat or split")
    train = root / "train"
    try:
        canonical_train = canonical_directory(
            train,
            label="split image folder train directory",
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"split image folder requires a train directory: {train}"
        ) from None
    roots = {"train": canonical_train}
    for candidate in (root / "validation", root / "val", root / "test"):
        if _is_link_or_reparse(candidate):
            raise ValueError(
                "split image folder must not use linked split directories: "
                f"{candidate}"
            )
    validation = [
        candidate
        for candidate in (root / "validation", root / "val")
        if candidate.is_dir()
    ]
    if len(validation) > 1:
        raise ValueError(
            "split image folder cannot contain both validation and val"
    )
    if validation:
        roots["validation"] = canonical_directory(
            validation[0],
            label="split image folder validation directory",
        )
    if (root / "test").is_dir():
        roots["test"] = canonical_directory(
            root / "test",
            label="split image folder test directory",
        )
    return roots


def _records_by_tree(
    records: Sequence[ImageFileRecord],
) -> dict[str, tuple[ImageFileRecord, ...]]:
    grouped: dict[str, list[ImageFileRecord]] = {}
    for record in records:
        grouped.setdefault(record.tree, []).append(record)
    return {
        tree: tuple(selected)
        for tree, selected in grouped.items()
    }


@dataclass(slots=True)
class TorchvisionImageSourceConfig:
    """Provider parameters for the built-in torchvision source."""

    dataset: str

    def validate(self, *, path: str) -> None:
        """Validate the allowlisted torchvision dataset."""

        dataset = cast(object, self.dataset)
        if not isinstance(dataset, str):
            raise ConfigError(f"{path}.dataset must be a string")
        self.dataset = dataset.lower()
        if self.dataset not in {"mnist", "cifar10", "flowers102"}:
            raise ConfigError(
                f"{path}.dataset must be MNIST, CIFAR10, or Flowers102"
            )


@dataclass(slots=True)
class ImageFolderSourceConfig:
    """Provider parameters for an external image folder."""

    root: str
    layout: str = "flat"

    def validate(self, *, path: str) -> None:
        """Validate folder locator and native split layout."""

        root = cast(object, self.root)
        if not isinstance(root, str) or not root.strip():
            raise ConfigError(f"{path}.root must be a non-empty string")
        layout = cast(object, self.layout)
        if not isinstance(layout, str) or layout not in {"flat", "split"}:
            raise ConfigError(f"{path}.layout must be flat or split")


@dataclass(slots=True)
class PairedImageFolderSourceConfig:
    """Provider parameters for external aligned HR/LR image folders."""

    high_resolution_root: str
    low_resolution_root: str
    layout: str = "flat"

    def validate(self, *, path: str) -> None:
        """Validate paired folder locators and native split layout."""

        for name in ("high_resolution_root", "low_resolution_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{path}.{name} must be a non-empty string")
        layout = cast(object, self.layout)
        if not isinstance(layout, str) or layout not in {"flat", "split"}:
            raise ConfigError(f"{path}.layout must be flat or split")


def _acquire_torchvision(
    dataset: str,
    root: Path,
    *,
    download: bool,
) -> None:
    root_value = str(root)
    if dataset == "mnist":
        datasets.MNIST(
            root_value,
            train=True,
            transform=None,
            download=download,
        )
        datasets.MNIST(
            root_value,
            train=False,
            transform=None,
            download=download,
        )
        return
    if dataset == "cifar10":
        datasets.CIFAR10(
            root_value,
            train=True,
            transform=None,
            download=download,
        )
        datasets.CIFAR10(
            root_value,
            train=False,
            transform=None,
            download=download,
        )
        return
    for split in ("train", "val", "test"):
        datasets.Flowers102(
            root_value,
            split=split,
            transform=None,
            download=download,
        )


def _managed_files(
    root: Path,
    *,
    hash_contents: bool,
) -> tuple[dict[str, Any], ...]:
    files: list[dict[str, Any]] = []
    snapshots = _scan_regular_file_snapshots(
        root,
        hash_contents=hash_contents,
        label="managed artifact",
        path_filter=lambda relative: relative != "manifest.json",
    )
    for snapshot in snapshots:
        relative = snapshot.relative_path
        digest = snapshot.sha256 if hash_contents else "0" * 64
        if digest is None:
            raise RuntimeError("hashed managed scan did not return a digest")
        files.append(
            {
                "path": relative,
                "size_bytes": snapshot.size_bytes,
                "sha256": digest,
            }
        )
    files.sort(key=lambda record: cast(str, record["path"]))
    if not files:
        raise ValueError("managed torchvision artifact contains no files")
    return tuple(files)


def _managed_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> ManagedDataArtifactIdentity:
    return ManagedDataArtifactIdentity(
        artifact_type=manifest["artifact_type"],
        source_name=manifest["source_name"],
        source_digest=manifest["source_digest"],
        materializer_name=manifest["materializer_name"],
        materialization_digest=manifest["materialization_digest"],
        artifact_digest=manifest["artifact_digest"],
        manifest_sha256=manifest_sha256,
    )


def _load_managed_torchvision(
    artifact_root: Path,
    *,
    dataset: str,
    verification: Literal["manifest", "full"],
) -> ManagedDataArtifact[TorchvisionImageArtifactPayload]:
    manifest_path = artifact_root / "manifest.json"
    raw, manifest_bytes = _load_canonical_json(
        manifest_path,
        label="managed manifest",
    )
    manifest = _strict_mapping(
        raw,
        fields=_MANAGED_MANIFEST_FIELDS,
        path="managed manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["kind"] != "managed"
        or manifest["source_name"] != "torchvision"
        or manifest["artifact_type"] != "stochaflow.torchvision-image.v1"
    ):
        raise ValueError("managed torchvision manifest is incompatible")
    identity = _managed_identity(
        manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )
    if artifact_root.name != identity.artifact_digest:
        raise ValueError(
            "managed artifact directory does not match its artifact digest"
        )
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("managed manifest files must be a non-empty list")
    normalized_files: list[dict[str, Any]] = []
    for index, value in enumerate(files):
        record = _strict_mapping(
            value,
            fields=_MANAGED_FILE_FIELDS,
            path=f"managed manifest.files[{index}]",
        )
        relative_path = record["path"]
        if not isinstance(relative_path, str):
            raise TypeError("managed manifest file path must be a string")
        _validate_relative_path(
            relative_path,
            path=f"managed manifest.files[{index}].path",
        )
        size_bytes = record["size_bytes"]
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError(
                "managed manifest file size_bytes must be non-negative"
            )
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("managed manifest file sha256 is invalid")
        normalized_files.append(record)
    recorded_paths = [record["path"] for record in normalized_files]
    if recorded_paths != sorted(recorded_paths) or len(recorded_paths) != len(
        set(recorded_paths)
    ):
        raise ValueError("managed manifest files must be sorted and unique")
    live = _managed_files(
        artifact_root,
        hash_contents=verification == "full",
    )
    expected_without_hash = tuple(
        (record["path"], record["size_bytes"]) for record in normalized_files
    )
    live_without_hash = tuple(
        (record["path"], record["size_bytes"]) for record in live
    )
    if live_without_hash != expected_without_hash:
        raise ValueError("managed artifact paths or sizes changed")
    if verification == "full" and tuple(normalized_files) != live:
        raise ValueError("managed artifact content digest changed")
    if identity.source_digest != _canonical_digest(normalized_files):
        raise ValueError("managed manifest source digest is invalid")
    materializer_name = "stochaflow.torchvision-download"
    expected_materialization_digest = _canonical_digest(
        {
            "name": materializer_name,
            "version": 1,
            "dataset": dataset,
        }
    )
    if (
        identity.materializer_name != materializer_name
        or identity.materialization_digest != expected_materialization_digest
    ):
        raise ValueError(
            "managed manifest materialization identity is invalid"
        )
    expected_artifact_digest = _canonical_digest(
        {
            "kind": "managed",
            "artifact_type": identity.artifact_type,
            "source_name": identity.source_name,
            "source_digest": identity.source_digest,
            "materializer_name": identity.materializer_name,
            "materialization_digest": identity.materialization_digest,
        }
    )
    if identity.artifact_digest != expected_artifact_digest:
        raise ValueError("managed manifest artifact digest is invalid")
    return ManagedDataArtifact(
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        identity=identity,
        payload=TorchvisionImageArtifactPayload(
            dataset=cast(Any, dataset),
            root=artifact_root / "data",
        ),
    )


@IMAGE_DATA_SOURCES.register("torchvision")
class TorchvisionImageDataSource(ImageDataSource):
    """Acquire and content-address one allowlisted torchvision dataset."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ManagedDataArtifact[TorchvisionImageArtifactPayload]:
        config = cast(
            TorchvisionImageSourceConfig,
            coerce_config_section(
                TorchvisionImageSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        expected = context.expected_identity
        pointer = (
            context.cache_root
            / "managed"
            / "torchvision"
            / config.dataset
            / "current.json"
        )
        if expected is not None:
            if not isinstance(expected, ManagedDataArtifactIdentity):
                raise ValueError(
                    "strict resume expected a different data artifact kind"
                )
            if expected.source_name != "torchvision":
                raise ValueError(
                    "strict resume expected a different managed data source"
                )
            artifact_digest: str | None = expected.artifact_digest
        else:
            artifact_digest = _read_locator_for_policy(
                context.cache_root,
                pointer,
                policy=context.policy,
                quarantine_on_error=False,
            )
        artifact_parent = (
            context.cache_root
            / "managed"
            / "torchvision"
            / config.dataset
            / "artifacts"
        )
        if artifact_digest is not None:
            try:
                artifact = _load_managed_torchvision(
                    artifact_parent / artifact_digest,
                    dataset=config.dataset,
                    verification=context.verification,
                )
                if expected is not None and artifact.identity != expected:
                    raise ValueError(
                        "strict resume managed data identity does not match"
                    )
                return artifact
            except (FileNotFoundError, OSError, TypeError, ValueError):
                if context.policy == "require":
                    raise
        if context.policy == "require":
            raise FileNotFoundError(
                f"required torchvision artifact is missing: {config.dataset}"
            )
        lock = (
            context.cache_root
            / "managed"
            / "torchvision"
            / config.dataset
            / "materialize.lock"
        )
        with ArtifactMaterializationLock(
            lock,
            cache_root=context.cache_root,
        ):
            winner_digest = (
                expected.artifact_digest
                if expected is not None
                else _read_locator_for_policy(
                    context.cache_root,
                    pointer,
                    policy="ensure",
                )
            )
            if winner_digest is not None:
                winner_root = artifact_parent / winner_digest
                try:
                    winner = _load_managed_torchvision(
                        winner_root,
                        dataset=config.dataset,
                        verification=context.verification,
                    )
                    if expected is not None and winner.identity != expected:
                        raise ValueError(
                            "strict resume managed data identity does not match"
                        )
                    return winner
                except (FileNotFoundError, OSError, TypeError, ValueError):
                    if _path_exists_without_following(
                        context.cache_root,
                        winner_root,
                    ):
                        _quarantine_path(context.cache_root, winner_root)
            staging = artifact_parent / f".staging.{uuid4().hex}"
            data_root = staging / "data"
            create_cache_directory(
                context.cache_root,
                staging,
                label="managed artifact staging directory",
            )
            create_cache_directory(
                context.cache_root,
                data_root,
                label="managed artifact data directory",
            )
            try:
                _acquire_torchvision(config.dataset, data_root, download=True)
                files = _managed_files(staging, hash_contents=True)
                source_digest = _canonical_digest(files)
                materializer_name = "stochaflow.torchvision-download"
                materialization_digest = _canonical_digest(
                    {
                        "name": materializer_name,
                        "version": 1,
                        "dataset": config.dataset,
                    }
                )
                artifact_digest = _canonical_digest(
                    {
                        "kind": "managed",
                        "artifact_type": "stochaflow.torchvision-image.v1",
                        "source_name": "torchvision",
                        "source_digest": source_digest,
                        "materializer_name": materializer_name,
                        "materialization_digest": materialization_digest,
                    }
                )
                manifest = {
                    "schema_version": 1,
                    "kind": "managed",
                    "artifact_type": "stochaflow.torchvision-image.v1",
                    "source_name": "torchvision",
                    "source_digest": source_digest,
                    "materializer_name": materializer_name,
                    "materialization_digest": materialization_digest,
                    "artifact_digest": artifact_digest,
                    "files": list(files),
                }
                manifest_bytes = _canonical_json_bytes(manifest)
                write_cache_file(
                    context.cache_root,
                    staging / "manifest.json",
                    manifest_bytes,
                    label="managed artifact manifest",
                )
                staging_identity = _managed_identity(
                    manifest,
                    manifest_sha256=_sha256_bytes(manifest_bytes),
                )
                if expected is not None and staging_identity != expected:
                    raise ValueError(
                        "strict resume managed data identity does not match"
                    )
                final = artifact_parent / artifact_digest
                if _path_exists_without_following(context.cache_root, final):
                    try:
                        winner = _load_managed_torchvision(
                            final,
                            dataset=config.dataset,
                            verification="full",
                        )
                    except (FileNotFoundError, OSError, TypeError, ValueError):
                        _quarantine_path(context.cache_root, final)
                    else:
                        remove_cache_directory(
                            context.cache_root,
                            staging,
                            label="managed artifact staging directory",
                        )
                        if expected is not None and winner.identity != expected:
                            raise ValueError(
                                "strict resume managed data identity "
                                "does not match"
                            )
                        if expected is None:
                            _write_locator(
                                context.cache_root,
                                context.cache_root
                                / "managed"
                                / "torchvision"
                                / config.dataset
                                / "current.json",
                                winner.identity.artifact_digest,
                            )
                        return winner
                publish_cache_directory(
                    context.cache_root,
                    staging,
                    final,
                    label="managed artifact",
                )
                artifact = _load_managed_torchvision(
                    final,
                    dataset=config.dataset,
                    verification=context.verification,
                )
                if expected is not None and artifact.identity != expected:
                    raise ValueError(
                        "strict resume managed data identity does not match"
                    )
                if expected is None:
                    _write_locator(
                        context.cache_root,
                        context.cache_root
                        / "managed"
                        / "torchvision"
                        / config.dataset
                        / "current.json",
                        artifact.identity.artifact_digest,
                    )
                return artifact
            except BaseException:
                if _path_exists_without_following(
                    context.cache_root,
                    staging,
                ):
                    remove_cache_directory(
                        context.cache_root,
                        staging,
                        label="managed artifact staging directory",
                    )
                raise


@IMAGE_DATA_SOURCES.register("image_folder")
class ImageFolderDataSource(ImageDataSource):
    """Index an external image directory without copying its content."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ReferencedDataArtifact[ImageFolderArtifactPayload]:
        config = cast(
            ImageFolderSourceConfig,
            coerce_config_section(
                ImageFolderSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        root = _absolute_directory(
            config.root,
            path=f"{self.config_path}.params.root",
        )
        roots = _partition_roots(root, config.layout)
        layout = {
            "type": "image_folder",
            "mode": config.layout,
            "trees": sorted(roots),
        }
        index_root, identity, records = _materialize_reference(
            context,
            source_name="image_folder",
            artifact_type="stochaflow.image-folder-reference.v1",
            roots=roots,
            layout=layout,
        )
        grouped = _records_by_tree(records)
        return ReferencedDataArtifact(
            index_root=index_root,
            manifest_path=index_root / "manifest.json",
            identity=identity,
            payload=ImageFolderArtifactPayload(
                roots=roots,
                train=grouped["train"],
                validation=grouped.get("validation"),
                test=grouped.get("test"),
            ),
        )


def _paired_partition_roots(
    high_root: Path,
    low_root: Path,
    layout: str,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    high = _partition_roots(high_root, layout)
    low = _partition_roots(low_root, layout)
    if set(high) != set(low):
        raise ValueError(
            "paired image folders must expose the same native splits"
        )
    roots = {
        f"{role}.high_resolution": high[role]
        for role in sorted(high)
    }
    roots.update(
        {
            f"{role}.low_resolution": low[role]
            for role in sorted(low)
        }
    )
    return roots, tuple(sorted(high))


def _pair_records(
    high: Sequence[ImageFileRecord],
    low: Sequence[ImageFileRecord],
    *,
    role: str,
) -> tuple[ImageFilePair, ...]:
    def by_stem(
        records: Sequence[ImageFileRecord],
        label: str,
    ) -> dict[str, ImageFileRecord]:
        result: dict[str, ImageFileRecord] = {}
        for record in records:
            stem = PurePosixPath(record.path).with_suffix("").as_posix()
            if stem in result:
                raise ValueError(
                    f"duplicate {label} relative image stem '{stem}' in {role}"
                )
            result[stem] = record
        return result

    high_by_stem = by_stem(high, "high-resolution")
    low_by_stem = by_stem(low, "low-resolution")
    missing_low = sorted(set(high_by_stem) - set(low_by_stem))
    missing_high = sorted(set(low_by_stem) - set(high_by_stem))
    if missing_low or missing_high:
        details: list[str] = []
        if missing_low:
            details.append("missing LR: " + ", ".join(missing_low))
        if missing_high:
            details.append("missing HR: " + ", ".join(missing_high))
        raise ValueError("paired image folders do not align; " + "; ".join(details))
    return tuple(
        ImageFilePair(high_by_stem[key], low_by_stem[key])
        for key in sorted(high_by_stem)
    )


@IMAGE_DATA_SOURCES.register("paired_image_folders")
class PairedImageFolderDataSource(ImageDataSource):
    """Index aligned external HR/LR directories without copying content."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ReferencedDataArtifact[PairedImageFolderArtifactPayload]:
        config = cast(
            PairedImageFolderSourceConfig,
            coerce_config_section(
                PairedImageFolderSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        high_root = _absolute_directory(
            config.high_resolution_root,
            path=f"{self.config_path}.params.high_resolution_root",
        )
        low_root = _absolute_directory(
            config.low_resolution_root,
            path=f"{self.config_path}.params.low_resolution_root",
        )
        if high_root == low_root:
            raise ValueError("paired image roots must be distinct")
        roots, roles = _paired_partition_roots(
            high_root,
            low_root,
            config.layout,
        )
        layout = {
            "type": "paired_image_folders",
            "mode": config.layout,
            "roles": list(roles),
        }
        index_root, identity, records = _materialize_reference(
            context,
            source_name="paired_image_folders",
            artifact_type="stochaflow.paired-image-folder-reference.v1",
            roots=roots,
            layout=layout,
        )
        grouped = _records_by_tree(records)
        pairs = {
            role: _pair_records(
                grouped[f"{role}.high_resolution"],
                grouped[f"{role}.low_resolution"],
                role=role,
            )
            for role in roles
        }
        return ReferencedDataArtifact(
            index_root=index_root,
            manifest_path=index_root / "manifest.json",
            identity=identity,
            payload=PairedImageFolderArtifactPayload(
                roots=roots,
                train=pairs["train"],
                validation=pairs.get("validation"),
                test=pairs.get("test"),
            ),
        )


class ImageSourceFactory:
    """Create and materialize registered image sources from canonical config."""

    def __init__(
        self,
        registry: Registry[type[ImageDataSource]] = IMAGE_DATA_SOURCES,
    ) -> None:
        self.registry = registry

    def materialize(
        self,
        config: ImageSourceConfig,
        *,
        binding_id: str,
        builder_context: DataBuilderContext,
        path: str,
    ) -> DataArtifact[ImageArtifactPayload]:
        """Materialize one source and enforce any strict-resume identity."""

        config.validate(path=path)
        expected: DataArtifactIdentity | None = None
        if builder_context.strict_resume:
            if builder_context.expected_artifacts is None:
                raise ValueError(
                    "strict resume checkpoint is missing data artifact identities"
                )
            expected = builder_context.expected_artifacts.identity_for(binding_id)
            if expected.source_name != config.name:
                raise ValueError(
                    "strict resume expected a different registered data source"
                )
        source = self.registry.create(
            config.name,
            config.params,
            config_path=path,
        )
        context = DataSourceContext(
            cache_root=Path(config.materialization.cache_root),
            policy=cast(Any, config.materialization.policy),
            verification=cast(Any, config.materialization.verification),
            expected_identity=expected,
        )
        artifact = source.materialize(context)
        if not isinstance(
            artifact,
            (ManagedDataArtifact, ReferencedDataArtifact),
        ):
            raise TypeError(
                f"image data source '{config.name}' must return DataArtifact"
            )
        if not isinstance(
            artifact.payload,
            (
                TorchvisionImageArtifactPayload,
                ImageFolderArtifactPayload,
                PairedImageFolderArtifactPayload,
            ),
        ):
            raise TypeError(
                f"image data source '{config.name}' returned an incompatible payload"
            )
        if artifact.identity.source_name != config.name:
            raise ValueError(
                f"image source '{config.name}' returned identity for "
                f"'{artifact.identity.source_name}'"
            )
        if expected is not None and artifact.identity != expected:
            raise ValueError(
                "strict resume data artifact identity does not match"
            )
        return artifact


def artifact_bindings(
    artifacts: Sequence[tuple[str, DataArtifact[Any]]],
) -> DataArtifactBindings:
    """Build the canonical identities selected by an image recipe."""

    return DataArtifactBindings(
        tuple(
            DataArtifactBinding(id=binding_id, identity=artifact.identity)
            for binding_id, artifact in artifacts
        )
    )


__all__ = [
    "IMAGE_DATA_SOURCES",
    "ArtifactMaterializationLock",
    "ImageArtifactPayload",
    "ImageDataSource",
    "ImageFilePair",
    "ImageFileRecord",
    "ImageFolderArtifactPayload",
    "ImageFolderDataSource",
    "ImageSourceFactory",
    "PairedImageFolderArtifactPayload",
    "PairedImageFolderDataSource",
    "TorchvisionImageArtifactPayload",
    "TorchvisionImageDataSource",
    "artifact_bindings",
]
