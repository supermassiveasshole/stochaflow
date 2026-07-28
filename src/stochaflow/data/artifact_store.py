"""Unified schema-v2 data artifact producer lifecycle."""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import re
import socket
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast
from uuid import uuid4

from stochaflow.data.artifact_io import (
    ArtifactFileSnapshot,
    cache_entry_exists,
    canonical_directory,
    create_cache_directory,
    ensure_cache_directory,
    lexical_absolute_path,
    open_cache_file,
    publish_cache_directory,
    publish_cache_file,
    read_regular_file,
    remove_cache_directory,
    scan_regular_files,
    write_cache_file,
)
from stochaflow.data.artifacts import (
    ArtifactVerificationEvent,
    ArtifactVerificationPhase,
    DataArtifact,
    DataArtifactIdentity,
    DataSourceContext,
)

_RECORD_LIMIT = 100_000
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "artifact_type",
        "source",
        "materializer",
        "content_digest",
        "stored_files",
        "domain_digest",
        "domain",
        "artifact_digest",
    }
)
_NAMED_DIGEST_FIELDS = frozenset({"name", "digest"})
_STORED_FILES_FIELDS = frozenset(
    {"digest", "record_limit", "record_count", "shards"}
)
_SHARD_FIELDS = frozenset({"path", "record_count", "sha256"})
_LOCATOR_FIELDS = frozenset({"schema_version", "artifact_digest"})
_SHA256_LENGTH = 64
_SENSITIVE_LOCATOR_TERMS = frozenset(
    {
        "auth",
        "authorization",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
        "token",
    }
)
_SENSITIVE_LOCATOR_COMPOUNDS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "bearertoken",
        "clientsecret",
        "privatekey",
        "refreshtoken",
        "secretkey",
    }
)

LOCK_WAIT_SECONDS = 30.0
LOCK_POLL_SECONDS = 0.05
WINDOWS_LOCK_BYTE_OFFSET = 4096
ADVISORY_LOCK_API: Any = importlib.import_module(
    "msvcrt" if os.name == "nt" else "fcntl"
)


class DataArtifactValidationError(ValueError):
    """Persisted candidate or represented content violates its contract."""


class DataArtifactLocatorMismatch(DataArtifactValidationError):
    """A valid locator target belongs to a different producer contract."""


class ArtifactVerificationObserverFailure(Exception):
    """Internal wrapper keeping observer errors outside corruption handling."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True, slots=True)
class ManagedDataArtifactBuild:
    """Producer facts returned after writing managed content."""

    source_digest: str
    materialization_digest: str
    domain: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ReferencedDataArtifactBuild:
    """Producer facts returned after indexing externally owned content."""

    source_digest: str
    materialization_digest: str
    content_digest: str
    domain: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DataArtifactLoadContext:
    """Verified final or staging paths supplied to a producer payload loader."""

    data_root: Path
    identity: DataArtifactIdentity
    domain: Mapping[str, object]
    verification: Literal["manifest", "full"]


@dataclass(frozen=True, slots=True)
class ValidatedArtifactObject:
    """Identity, domain, and exact evidence from one verified object scan."""

    identity: DataArtifactIdentity
    domain: Mapping[str, object]
    snapshot: tuple[ArtifactFileSnapshot, ...]


def _assert_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not (cast(float, value) == cast(float, value)) or value in {
            float("inf"),
            float("-inf"),
        }:
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        raw = cast(dict[object, object], value)
        for key, item in raw.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} field names must be strings")
            _assert_json_value(item, path=f"{path}.{key}")
        return
    raise TypeError(
        f"{path} must contain only JSON-safe dict, list, and scalar values"
    )


def canonical_artifact_json_bytes(value: object) -> bytes:
    """Encode a strict JSON-safe value as canonical newline-terminated UTF-8."""

    _assert_json_value(value, path="artifact JSON")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_artifact_digest(value: object) -> str:
    """Hash the canonical artifact JSON representation of one value."""

    return hashlib.sha256(canonical_artifact_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


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
        raise TypeError(f"{path} field names must be strings")
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
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    encoded, _ = read_regular_file(root, relative_path, label=label)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    if encoded != canonical_artifact_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], value), encoded


class ArtifactMaterializationLock(
    AbstractContextManager["ArtifactMaterializationLock"]
):
    """Framework-private cooperative lock with bounded waiting."""

    def __init__(
        self,
        path: Path,
        *,
        cache_root: Path,
        wait_seconds: float = LOCK_WAIT_SECONDS,
        poll_seconds: float = LOCK_POLL_SECONDS,
    ) -> None:
        if wait_seconds < 0:
            raise ValueError("artifact lock wait_seconds must be non-negative")
        if poll_seconds <= 0:
            raise ValueError("artifact lock poll_seconds must be positive")
        self.path = lexical_absolute_path(path)
        self.cache_root = lexical_absolute_path(cache_root)
        self.wait_seconds = wait_seconds
        self.poll_seconds = poll_seconds
        self.handle: int | None = None

    def __enter__(self) -> Self:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            handle = open_cache_file(
                self.cache_root,
                self.path,
                label="data artifact lock",
            )
            if _try_acquire_lock(handle):
                self.handle = handle
                break
            os.close(handle)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for data artifact lock: {self.path}"
                )
            time.sleep(
                min(
                    self.poll_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )
        owner = canonical_artifact_json_bytes(
            {
                "schema_version": 2,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "created_at_ns": time.time_ns(),
                "nonce": uuid4().hex,
            }
        )
        try:
            os.lseek(self.handle, 0, os.SEEK_SET)
            _write_descriptor(self.handle, owner)
            os.ftruncate(self.handle, len(owner))
            os.fsync(self.handle)
        except BaseException:
            _release_lock(self.handle)
            os.close(self.handle)
            self.handle = None
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.handle is None:
            return
        _release_lock(self.handle)
        os.close(self.handle)
        self.handle = None


def _write_descriptor(descriptor: int, encoded: bytes) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write artifact lock metadata")
        remaining = remaining[written:]


def _try_acquire_lock(descriptor: int) -> bool:
    if os.name == "nt":
        os.lseek(descriptor, WINDOWS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        try:
            ADVISORY_LOCK_API.locking(
                descriptor,
                ADVISORY_LOCK_API.LK_NBLCK,
                1,
            )
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True
    try:
        ADVISORY_LOCK_API.flock(
            descriptor,
            ADVISORY_LOCK_API.LOCK_EX | ADVISORY_LOCK_API.LOCK_NB,
        )
    except BlockingIOError:
        return False
    return True


def _release_lock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, WINDOWS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        ADVISORY_LOCK_API.locking(
            descriptor,
            ADVISORY_LOCK_API.LK_UNLCK,
            1,
        )
        return
    ADVISORY_LOCK_API.flock(descriptor, ADVISORY_LOCK_API.LOCK_UN)


@dataclass(frozen=True, slots=True)
class ArtifactStorePaths:
    cache_root: Path
    namespace_root: Path
    objects: Path
    staging: Path
    locators: Path
    locator_locks: Path
    object_locks: Path
    quarantine_objects: Path
    quarantine_locators: Path


def _store_paths(
    cache_root: Path,
    *,
    kind: Literal["managed", "referenced"],
    artifact_type: str,
) -> ArtifactStorePaths:
    namespace = (
        cache_root
        / "data-artifacts"
        / "v2"
        / kind
        / canonical_artifact_digest(artifact_type)
    )
    return ArtifactStorePaths(
        cache_root=cache_root,
        namespace_root=namespace,
        objects=namespace / "objects",
        staging=namespace / "staging",
        locators=namespace / "locators",
        locator_locks=namespace / "locks" / "locators",
        object_locks=namespace / "locks" / "objects",
        quarantine_objects=namespace / "quarantine" / "objects",
        quarantine_locators=namespace / "quarantine" / "locators",
    )


def _mapping_copy(value: Mapping[str, object], *, path: str) -> dict[str, object]:
    if not isinstance(cast(object, value), Mapping):
        raise TypeError(f"{path} must be a mapping")
    copied = dict(value)
    _assert_json_value(copied, path=path)
    return copied


def _reject_sensitive_locator_keys(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in cast(dict[str, object], value).items():
            expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            pieces = {
                piece
                for piece in re.split(r"[^a-z0-9]+", expanded.lower())
                if piece
            }
            compact = "".join(
                character for character in key.lower() if character.isalnum()
            )
            if pieces & _SENSITIVE_LOCATOR_TERMS or any(
                compact.startswith(term) or compact.endswith(term)
                for term in _SENSITIVE_LOCATOR_COMPOUNDS
            ):
                raise ValueError(f"{path} must not contain credentials: {key}")
            _reject_sensitive_locator_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_locator_keys(item, path=f"{path}[{index}]")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_referenced_roots(
    cache_root: Path,
    roots: Mapping[str, Path],
) -> dict[str, Path]:
    if not isinstance(cast(object, roots), Mapping) or not roots:
        raise ValueError("referenced_roots must be a non-empty mapping")
    normalized: dict[str, Path] = {}
    for name, value in roots.items():
        _non_empty_string(name, path="referenced_roots name")
        if not isinstance(cast(object, value), Path):
            raise TypeError(f"referenced_roots.{name} must be a Path")
        root = lexical_absolute_path(value)
        parent = canonical_directory(
            root.parent,
            label=f"referenced root {name} parent",
        )
        metadata = root.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0) & reparse_flag
        ):
            raise ValueError(
                f"referenced root {name} must not be a symlink or reparse point"
            )
        if stat.S_ISDIR(metadata.st_mode):
            root = canonical_directory(root, label=f"referenced root {name}")
        elif stat.S_ISREG(metadata.st_mode):
            root = parent / root.name
        else:
            raise ValueError(
                f"referenced root {name} must be a directory or regular file"
            )
        normalized[name] = root
    resolved_cache = cache_root.resolve(strict=False)
    for name, root in normalized.items():
        resolved_root = root.resolve(strict=True)
        if _is_relative_to(resolved_cache, resolved_root) or _is_relative_to(
            resolved_root, resolved_cache
        ):
            raise ValueError(
                f"referenced root {name} overlaps the data artifact cache root"
            )
    return normalized


def _locator_digest(
    *,
    kind: Literal["managed", "referenced"],
    artifact_type: str,
    source_name: str,
    materializer_name: str,
    locator_key: Mapping[str, object],
    referenced_roots: Mapping[str, Path],
) -> str:
    return canonical_artifact_digest(
        {
            "schema_version": 2,
            "kind": kind,
            "artifact_type": artifact_type,
            "source_name": source_name,
            "materializer_name": materializer_name,
            "locator_key": dict(locator_key),
            "referenced_roots": {
                name: str(path)
                for name, path in sorted(referenced_roots.items())
            },
        }
    )


def _path_exists(cache_root: Path, path: Path) -> bool:
    return cache_entry_exists(
        cache_root,
        path,
        label="data artifact cache entry",
    )


def _read_locator(path: Path) -> str:
    raw, _ = _load_canonical_json(
        path.parent,
        path.name,
        label="data artifact locator",
    )
    locator = _strict_mapping(
        raw,
        fields=_LOCATOR_FIELDS,
        path="data artifact locator",
    )
    if type(locator["schema_version"]) is not int or locator["schema_version"] != 2:
        raise ValueError("data artifact locator.schema_version must be 2")
    return _sha256(
        locator["artifact_digest"],
        path="data artifact locator.artifact_digest",
    )


def _write_locator(cache_root: Path, path: Path, artifact_digest: str) -> None:
    write_cache_file(
        cache_root,
        path,
        canonical_artifact_json_bytes(
            {"schema_version": 2, "artifact_digest": artifact_digest}
        ),
        label="data artifact locator",
    )


def _ensure_store_directories(paths: ArtifactStorePaths) -> None:
    for directory, label in (
        (paths.objects, "data artifact objects"),
        (paths.staging, "data artifact staging"),
        (paths.locators, "data artifact locators"),
        (paths.locator_locks, "data artifact locator locks"),
        (paths.object_locks, "data artifact object locks"),
        (paths.quarantine_objects, "data artifact object quarantine"),
        (paths.quarantine_locators, "data artifact locator quarantine"),
    ):
        ensure_cache_directory(paths.cache_root, directory, label=label)


def _quarantine_entry(
    paths: ArtifactStorePaths,
    path: Path,
    *,
    destination_root: Path,
    is_directory: bool,
) -> None:
    if not _path_exists(paths.cache_root, path):
        return
    destination = destination_root / f"{path.name}.{uuid4().hex}.corrupt"
    if is_directory:
        publish_cache_directory(
            paths.cache_root,
            path,
            destination,
            label="corrupt data artifact object",
        )
    else:
        publish_cache_file(
            paths.cache_root,
            path,
            destination,
            label="corrupt data artifact locator",
        )


def _inventory_records(
    snapshots: Sequence[ArtifactFileSnapshot],
) -> list[dict[str, object]]:
    return [
        {
            "path": snapshot.relative_path,
            "size_bytes": snapshot.size_bytes,
            "sha256": cast(str, snapshot.sha256),
        }
        for snapshot in snapshots
    ]


def _assert_build_scope(
    staging: Path,
    data_root: Path,
    *,
    original_data_root: os.stat_result,
) -> None:
    entries = tuple(staging.iterdir())
    if entries != (data_root,):
        raise RuntimeError(
            "data artifact build wrote outside its supplied data root"
        )
    metadata = data_root.lstat()
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise RuntimeError("data artifact build replaced its supplied data root")
    original_identity = (
        original_data_root.st_dev,
        original_data_root.st_ino,
    )
    current_identity = (metadata.st_dev, metadata.st_ino)
    if (
        original_identity == (0, 0)
        or current_identity == (0, 0)
        or original_identity != current_identity
    ):
        raise RuntimeError("data artifact build replaced its supplied data root")


def _write_inventory(
    paths: ArtifactStorePaths,
    object_root: Path,
    *,
    verification_workers: int | None,
) -> tuple[dict[str, object], str]:
    snapshots = scan_regular_files(
        object_root / "data",
        hash_contents=True,
        label="data artifact stored files",
        workers=verification_workers,
    )
    records = _inventory_records(snapshots)
    digest = canonical_artifact_digest(records)
    inventory_root = object_root / "inventory"
    create_cache_directory(
        paths.cache_root,
        inventory_root,
        label="data artifact inventory directory",
    )
    shards: list[dict[str, object]] = []
    for shard_index, offset in enumerate(range(0, len(records), _RECORD_LIMIT)):
        selected = records[offset : offset + _RECORD_LIMIT]
        relative_path = f"inventory/{shard_index:06d}.jsonl"
        encoded = b"".join(
            canonical_artifact_json_bytes(record) for record in selected
        )
        write_cache_file(
            paths.cache_root,
            object_root / relative_path,
            encoded,
            label="data artifact inventory shard",
        )
        shards.append(
            {
                "path": relative_path,
                "record_count": len(selected),
                "sha256": _sha256_bytes(encoded),
            }
        )
    return (
        {
            "digest": digest,
            "record_limit": _RECORD_LIMIT,
            "record_count": len(records),
            "shards": shards,
        },
        digest,
    )


def _read_inventory(
    root: Path,
    value: object,
) -> tuple[
    str,
    tuple[dict[str, object], ...],
    tuple[tuple[str, str], ...],
]:
    stored = _strict_mapping(
        value,
        fields=_STORED_FILES_FIELDS,
        path="data artifact manifest.stored_files",
    )
    expected_digest = _sha256(
        stored["digest"],
        path="data artifact manifest.stored_files.digest",
    )
    if (
        type(stored["record_limit"]) is not int
        or stored["record_limit"] != _RECORD_LIMIT
    ):
        raise ValueError("data artifact stored_files.record_limit is invalid")
    if type(stored["record_count"]) is not int or stored["record_count"] < 0:
        raise ValueError("data artifact stored_files.record_count is invalid")
    record_count = cast(int, stored["record_count"])
    serialized_shards = stored["shards"]
    if not isinstance(serialized_shards, list):
        raise TypeError("data artifact stored_files.shards must be a list")
    expected_shards = (record_count + _RECORD_LIMIT - 1) // _RECORD_LIMIT
    if len(serialized_shards) != expected_shards:
        raise ValueError("data artifact inventory shard count is not canonical")
    records: list[dict[str, object]] = []
    shard_digests: list[tuple[str, str]] = []
    for shard_index, raw_shard in enumerate(serialized_shards):
        shard = _strict_mapping(
            raw_shard,
            fields=_SHARD_FIELDS,
            path=f"data artifact stored_files.shards[{shard_index}]",
        )
        expected_path = f"inventory/{shard_index:06d}.jsonl"
        if shard["path"] != expected_path:
            raise ValueError("data artifact inventory shard path is not canonical")
        expected_count = min(
            _RECORD_LIMIT,
            record_count - shard_index * _RECORD_LIMIT,
        )
        if (
            type(shard["record_count"]) is not int
            or shard["record_count"] != expected_count
        ):
            raise ValueError("data artifact inventory shard count is invalid")
        shard_digest = _sha256(
            shard["sha256"],
            path=f"data artifact inventory shard {shard_index}.sha256",
        )
        encoded, _ = read_regular_file(
            root,
            expected_path,
            label="data artifact inventory shard",
        )
        if _sha256_bytes(encoded) != shard_digest:
            raise ValueError("data artifact inventory shard digest mismatch")
        shard_digests.append((expected_path, shard_digest))
        lines = encoded.splitlines(keepends=True)
        if len(lines) != expected_count:
            raise ValueError("data artifact inventory shard record count mismatch")
        for line in lines:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("data artifact inventory contains invalid JSON") from exc
            if line != canonical_artifact_json_bytes(record):
                raise ValueError("data artifact inventory record is not canonical")
            parsed = _strict_mapping(
                record,
                fields=frozenset({"path", "size_bytes", "sha256"}),
                path="data artifact inventory record",
            )
            _non_empty_string(parsed["path"], path="inventory record.path")
            if (
                type(parsed["size_bytes"]) is not int
                or parsed["size_bytes"] < 0
            ):
                raise ValueError("inventory record.size_bytes is invalid")
            _sha256(parsed["sha256"], path="inventory record.sha256")
            records.append(cast(dict[str, object], parsed))
    if records != sorted(records, key=lambda item: cast(str, item["path"])):
        raise ValueError("data artifact inventory records must be sorted")
    paths = [cast(str, record["path"]) for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("data artifact inventory contains duplicate paths")
    if canonical_artifact_digest(records) != expected_digest:
        raise ValueError("data artifact stored-files digest mismatch")
    return expected_digest, tuple(records), tuple(shard_digests)


def _validate_inventory_snapshot(
    snapshot: Sequence[ArtifactFileSnapshot],
    *,
    records: Sequence[Mapping[str, object]],
    shard_digests: Sequence[tuple[str, str]],
    manifest_sha256: str,
    verification: Literal["manifest", "full"],
) -> None:
    data_snapshots = tuple(
        item
        for item in snapshot
        if item.relative_path.startswith("data/")
    )
    if len(data_snapshots) != len(records):
        raise ValueError("data artifact stored-file count mismatch")
    for item, record in zip(data_snapshots, records, strict=True):
        relative_path = item.relative_path.removeprefix("data/")
        if (
            relative_path != record["path"]
            or item.size_bytes != record["size_bytes"]
            or (
                verification == "full"
                and item.sha256 != record["sha256"]
            )
        ):
            raise ValueError(
                "data artifact stored file does not match inventory: "
                f"{relative_path}"
            )
    if verification != "full":
        return
    by_path = {item.relative_path: item for item in snapshot}
    if by_path["manifest.json"].sha256 != manifest_sha256:
        raise ValueError("data artifact manifest changed during validation")
    for path, digest in shard_digests:
        if by_path[path].sha256 != digest:
            raise ValueError(
                "data artifact inventory shard changed during validation"
            )


def _validate_object_layout(
    root: Path,
    *,
    stored_paths: Sequence[str],
    shard_paths: Sequence[str],
    snapshot: Sequence[ArtifactFileSnapshot],
) -> None:
    expected_root_entries = {"data", "inventory", "manifest.json"}
    observed_root_entries = {entry.name for entry in root.iterdir()}
    if observed_root_entries != expected_root_entries:
        raise ValueError(
            "data artifact object root must contain exactly "
            "data, inventory, and manifest.json"
        )
    for name in ("data", "inventory"):
        directory = root / name
        metadata = directory.lstat()
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise ValueError(
                f"data artifact object {name} entry must be a real directory"
            )
    expected_inventory_entries = {
        path.removeprefix("inventory/") for path in shard_paths
    }
    observed_inventory_entries = {
        entry.name for entry in (root / "inventory").iterdir()
    }
    if observed_inventory_entries != expected_inventory_entries:
        raise ValueError(
            "data artifact inventory directory contains unexpected entries"
        )
    expected_files = {
        "manifest.json",
        *shard_paths,
        *(f"data/{path}" for path in stored_paths),
    }
    observed_files = {
        snapshot.relative_path
        for snapshot in snapshot
    }
    if observed_files != expected_files:
        raise ValueError("data artifact object contains unexpected stored files")
    expected_data_directories: set[str] = set()
    for stored_path in stored_paths:
        parts = stored_path.split("/")[:-1]
        expected_data_directories.update(
            "/".join(parts[:index])
            for index in range(1, len(parts) + 1)
        )
    data_root = root / "data"
    observed_data_directories = {
        (Path(current) / name).relative_to(data_root).as_posix()
        for current, names, _ in os.walk(data_root, followlinks=False)
        for name in names
    }
    if observed_data_directories != expected_data_directories:
        raise ValueError(
            "data artifact data directory contains unexpected directories"
        )


def _artifact_digest(
    *,
    kind: Literal["managed", "referenced"],
    artifact_type: str,
    source: Mapping[str, object],
    materializer: Mapping[str, object],
    content_digest: str,
    stored_files_digest: str,
    domain_digest: str,
) -> str:
    return canonical_artifact_digest(
        {
            "schema_version": 2,
            "kind": kind,
            "artifact_type": artifact_type,
            "source": dict(source),
            "materializer": dict(materializer),
            "content_digest": content_digest,
            "stored_files_digest": stored_files_digest,
            "domain_digest": domain_digest,
        }
    )


class DataArtifactStore:
    """Framework-owned producer lifecycle for all data artifact kinds."""

    def __init__(self, context: DataSourceContext) -> None:
        if not isinstance(cast(object, context), DataSourceContext):
            raise TypeError("context must be DataSourceContext")
        self.context = context

    def materialize_managed[PayloadT](
        self,
        *,
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        locator_key: Mapping[str, object],
        build: Callable[[Path], ManagedDataArtifactBuild],
        load: Callable[[DataArtifactLoadContext], PayloadT],
    ) -> DataArtifact[PayloadT]:
        """Load or build one cache-owned artifact."""

        try:
            return self._materialize(
                kind="managed",
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                locator_key=locator_key,
                referenced_roots={},
                build=build,
                load=load,
            )
        except ArtifactVerificationObserverFailure as exc:
            raise exc.error from None

    def materialize_referenced[PayloadT](
        self,
        *,
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        locator_key: Mapping[str, object],
        referenced_roots: Mapping[str, Path],
        build: Callable[[Path], ReferencedDataArtifactBuild],
        load: Callable[[DataArtifactLoadContext], PayloadT],
    ) -> DataArtifact[PayloadT]:
        """Load or build an index whose represented content remains external."""

        try:
            return self._materialize(
                kind="referenced",
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                locator_key=locator_key,
                referenced_roots=referenced_roots,
                build=build,
                load=load,
            )
        except ArtifactVerificationObserverFailure as exc:
            raise exc.error from None

    def _materialize[PayloadT](
        self,
        *,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        locator_key: Mapping[str, object],
        referenced_roots: Mapping[str, Path],
        build: Callable[
            [Path],
            ManagedDataArtifactBuild | ReferencedDataArtifactBuild,
        ],
        load: Callable[[DataArtifactLoadContext], PayloadT],
    ) -> DataArtifact[PayloadT]:
        artifact_type = _non_empty_string(
            artifact_type, path="artifact_type"
        )
        source_name = _non_empty_string(source_name, path="source_name")
        materializer_name = _non_empty_string(
            materializer_name, path="materializer_name"
        )
        locator = _mapping_copy(locator_key, path="locator_key")
        _reject_sensitive_locator_keys(locator, path="locator_key")
        cache_root = lexical_absolute_path(self.context.cache_root)
        roots = (
            _validate_referenced_roots(cache_root, referenced_roots)
            if kind == "referenced"
            else {}
        )
        if kind == "managed" and referenced_roots:
            raise ValueError("managed artifacts must not declare referenced_roots")
        expected = self.context.expected_identity
        if expected is not None and (
            expected.kind != kind
            or expected.artifact_type != artifact_type
            or expected.source_name != source_name
            or expected.materializer_name != materializer_name
        ):
            raise ValueError(
                "expected data artifact identity is incompatible with "
                "the requested producer"
            )
        paths = _store_paths(
            cache_root,
            kind=kind,
            artifact_type=artifact_type,
        )
        locator_digest = _locator_digest(
            kind=kind,
            artifact_type=artifact_type,
            source_name=source_name,
            materializer_name=materializer_name,
            locator_key=locator,
            referenced_roots=roots,
        )
        locator_path = paths.locators / f"{locator_digest}.json"

        if expected is not None:
            return self._materialize_expected(
                paths=paths,
                expected=expected,
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                build=build,
                load=load,
                kind=kind,
            )

        artifact_digest = self._locator_candidate(
            paths,
            locator_path,
            repair=False,
        )
        if artifact_digest is not None:
            try:
                return self._load_compatible_object(
                    paths.objects / artifact_digest,
                    kind=kind,
                    artifact_type=artifact_type,
                    source_name=source_name,
                    materializer_name=materializer_name,
                    verification=self.context.verification,
                    load=load,
                )
            except DataArtifactValidationError:
                if self.context.policy == "require":
                    raise
        elif self.context.policy == "require":
            raise FileNotFoundError(
                f"required data artifact locator does not exist: {locator_path}"
            )

        _ensure_store_directories(paths)
        with ArtifactMaterializationLock(
            paths.locator_locks / f"{locator_digest}.lock",
            cache_root=paths.cache_root,
        ):
            artifact_digest = self._locator_candidate(
                paths,
                locator_path,
                repair=True,
            )
            if artifact_digest is not None:
                object_root = paths.objects / artifact_digest
                try:
                    return self._load_compatible_object(
                        object_root,
                        kind=kind,
                        artifact_type=artifact_type,
                        source_name=source_name,
                        materializer_name=materializer_name,
                        verification=self.context.verification,
                        load=load,
                    )
                except DataArtifactLocatorMismatch:
                    _quarantine_entry(
                        paths,
                        locator_path,
                        destination_root=paths.quarantine_locators,
                        is_directory=False,
                    )
                except DataArtifactValidationError:
                    with ArtifactMaterializationLock(
                        paths.object_locks / f"{artifact_digest}.lock",
                        cache_root=paths.cache_root,
                    ):
                        try:
                            return self._load_compatible_object(
                                object_root,
                                kind=kind,
                                artifact_type=artifact_type,
                                source_name=source_name,
                                materializer_name=materializer_name,
                                verification=self.context.verification,
                                load=load,
                            )
                        except DataArtifactLocatorMismatch:
                            _quarantine_entry(
                                paths,
                                locator_path,
                                destination_root=paths.quarantine_locators,
                                is_directory=False,
                            )
                        except DataArtifactValidationError:
                            try:
                                candidate_identity = (
                                    self._load_framework_identity(
                                        object_root,
                                        kind=kind,
                                        artifact_type=artifact_type,
                                        verification="full",
                                    )
                                )
                            except (OSError, DataArtifactValidationError):
                                _quarantine_entry(
                                    paths,
                                    object_root,
                                    destination_root=paths.quarantine_objects,
                                    is_directory=True,
                                )
                            else:
                                if (
                                    candidate_identity.source_name
                                    != source_name
                                    or candidate_identity.materializer_name
                                    != materializer_name
                                ):
                                    _quarantine_entry(
                                        paths,
                                        locator_path,
                                        destination_root=(
                                            paths.quarantine_locators
                                        ),
                                        is_directory=False,
                                    )
                                elif kind == "managed":
                                    _quarantine_entry(
                                        paths,
                                        object_root,
                                        destination_root=(
                                            paths.quarantine_objects
                                        ),
                                        is_directory=True,
                                    )
            artifact = self._build_and_publish(
                paths=paths,
                kind=kind,
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                expected=None,
                build=build,
                load=load,
            )
            _write_locator(
                paths.cache_root,
                locator_path,
                artifact.identity.artifact_digest,
            )
            return artifact

    def _materialize_expected[PayloadT](
        self,
        *,
        paths: ArtifactStorePaths,
        expected: DataArtifactIdentity,
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        build: Callable[
            [Path],
            ManagedDataArtifactBuild | ReferencedDataArtifactBuild,
        ],
        load: Callable[[DataArtifactLoadContext], PayloadT],
        kind: Literal["managed", "referenced"],
    ) -> DataArtifact[PayloadT]:
        object_root = paths.objects / expected.artifact_digest
        try:
            validated = self._validate_object(
                object_root,
                kind=kind,
                artifact_type=artifact_type,
                expected=None,
                verification="full",
            )
        except (OSError, DataArtifactValidationError):
            if self.context.policy == "require":
                raise
        else:
            if validated.identity != expected:
                raise DataArtifactValidationError(
                    "data artifact identity does not match expected identity"
                )
            try:
                return self._load_validated_object(
                    object_root,
                    validated=validated,
                    verification="full",
                    load=load,
                )
            except DataArtifactValidationError:
                if self.context.policy == "require":
                    raise
        _ensure_store_directories(paths)
        with ArtifactMaterializationLock(
            paths.object_locks / f"{expected.artifact_digest}.lock",
            cache_root=paths.cache_root,
        ):
            try:
                validated = self._validate_object(
                    object_root,
                    kind=kind,
                    artifact_type=artifact_type,
                    expected=None,
                    verification="full",
                )
            except FileNotFoundError:
                pass
            except (OSError, DataArtifactValidationError):
                _quarantine_entry(
                    paths,
                    object_root,
                    destination_root=paths.quarantine_objects,
                    is_directory=True,
                )
            else:
                if validated.identity != expected:
                    raise DataArtifactValidationError(
                        "data artifact identity does not match expected identity"
                    )
                try:
                    return self._load_validated_object(
                        object_root,
                        validated=validated,
                        verification="full",
                        load=load,
                    )
                except DataArtifactValidationError:
                    try:
                        refreshed_identity = self._load_framework_identity(
                            object_root,
                            kind=kind,
                            artifact_type=artifact_type,
                            verification="full",
                        )
                    except (OSError, DataArtifactValidationError):
                        _quarantine_entry(
                            paths,
                            object_root,
                            destination_root=paths.quarantine_objects,
                            is_directory=True,
                        )
                    else:
                        if refreshed_identity != expected:
                            raise DataArtifactValidationError(
                                "data artifact identity does not match "
                                "expected identity"
                            )
                        if kind == "managed":
                            _quarantine_entry(
                                paths,
                                object_root,
                                destination_root=paths.quarantine_objects,
                                is_directory=True,
                            )
            return self._build_and_publish(
                paths=paths,
                kind=kind,
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                expected=expected,
                build=build,
                load=load,
                object_lock_held=True,
            )

    def _locator_candidate(
        self,
        paths: ArtifactStorePaths,
        locator_path: Path,
        *,
        repair: bool,
    ) -> str | None:
        if not _path_exists(paths.cache_root, locator_path):
            return None
        try:
            return _read_locator(locator_path)
        except (OSError, TypeError, ValueError) as exc:
            if self.context.policy == "require" or not repair:
                if self.context.policy == "require":
                    raise DataArtifactValidationError(str(exc)) from exc
                return None
            _quarantine_entry(
                paths,
                locator_path,
                destination_root=paths.quarantine_locators,
                is_directory=False,
            )
            return None

    def _build_and_publish[PayloadT](
        self,
        *,
        paths: ArtifactStorePaths,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        expected: DataArtifactIdentity | None,
        build: Callable[
            [Path],
            ManagedDataArtifactBuild | ReferencedDataArtifactBuild,
        ],
        load: Callable[[DataArtifactLoadContext], PayloadT],
        object_lock_held: bool = False,
    ) -> DataArtifact[PayloadT]:
        staging = paths.staging / uuid4().hex
        create_cache_directory(
            paths.cache_root,
            staging,
            label="data artifact staging directory",
        )
        try:
            data_root = staging / "data"
            create_cache_directory(
                paths.cache_root,
                data_root,
                label="data artifact staging data directory",
            )
            original_data_root = data_root.lstat()
            result = build(data_root)
            _assert_build_scope(
                staging,
                data_root,
                original_data_root=original_data_root,
            )
            if kind == "managed":
                if not isinstance(result, ManagedDataArtifactBuild):
                    raise TypeError(
                        "managed build must return ManagedDataArtifactBuild"
                    )
            elif not isinstance(result, ReferencedDataArtifactBuild):
                raise TypeError(
                    "referenced build must return ReferencedDataArtifactBuild"
                )
            source_digest = _sha256(
                result.source_digest, path="build.source_digest"
            )
            materialization_digest = _sha256(
                result.materialization_digest,
                path="build.materialization_digest",
            )
            domain = _mapping_copy(result.domain, path="build.domain")
            stored_files, stored_files_digest = _write_inventory(
                paths,
                staging,
                verification_workers=self.context.verification_workers,
            )
            if kind == "managed":
                content_digest = stored_files_digest
            else:
                content_digest = _sha256(
                    cast(ReferencedDataArtifactBuild, result).content_digest,
                    path="build.content_digest",
                )
            source = {"name": source_name, "digest": source_digest}
            materializer = {
                "name": materializer_name,
                "digest": materialization_digest,
            }
            domain_digest = canonical_artifact_digest(domain)
            artifact_digest = _artifact_digest(
                kind=kind,
                artifact_type=artifact_type,
                source=source,
                materializer=materializer,
                content_digest=content_digest,
                stored_files_digest=stored_files_digest,
                domain_digest=domain_digest,
            )
            manifest = {
                "schema_version": 2,
                "kind": kind,
                "artifact_type": artifact_type,
                "source": source,
                "materializer": materializer,
                "content_digest": content_digest,
                "stored_files": stored_files,
                "domain_digest": domain_digest,
                "domain": domain,
                "artifact_digest": artifact_digest,
            }
            manifest_bytes = canonical_artifact_json_bytes(manifest)
            write_cache_file(
                paths.cache_root,
                staging / "manifest.json",
                manifest_bytes,
                label="data artifact manifest",
            )
            identity = DataArtifactIdentity(
                kind=kind,
                artifact_type=artifact_type,
                source_name=source_name,
                source_digest=source_digest,
                materializer_name=materializer_name,
                materialization_digest=materialization_digest,
                content_digest=content_digest,
                artifact_digest=artifact_digest,
                manifest_sha256=_sha256_bytes(manifest_bytes),
            )
            if expected is not None and identity != expected:
                raise DataArtifactValidationError(
                    "built data artifact identity does not match expected identity"
                )
            self._load_object(
                staging,
                kind=kind,
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                expected=identity,
                verification="full",
                load=load,
            )
            destination = paths.objects / artifact_digest
            if object_lock_held:
                self._publish_or_select_winner(
                    paths=paths,
                    staging=staging,
                    destination=destination,
                    identity=identity,
                    kind=kind,
                    artifact_type=artifact_type,
                )
            else:
                with ArtifactMaterializationLock(
                    paths.object_locks / f"{artifact_digest}.lock",
                    cache_root=paths.cache_root,
                ):
                    self._publish_or_select_winner(
                        paths=paths,
                        staging=staging,
                        destination=destination,
                        identity=identity,
                        kind=kind,
                        artifact_type=artifact_type,
                    )
            return self._load_object(
                destination,
                kind=kind,
                artifact_type=artifact_type,
                source_name=source_name,
                materializer_name=materializer_name,
                expected=identity,
                verification="full",
                load=load,
            )
        finally:
            if _path_exists(paths.cache_root, staging):
                with suppress(FileNotFoundError):
                    remove_cache_directory(
                        paths.cache_root,
                        staging,
                        label="data artifact staging cleanup",
                    )

    def _load_framework_identity(
        self,
        root: Path,
        *,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
        verification: Literal["manifest", "full"],
    ) -> DataArtifactIdentity:
        return self._validate_object(
            root,
            kind=kind,
            artifact_type=artifact_type,
            expected=None,
            verification=verification,
        ).identity

    def _load_compatible_object[PayloadT](
        self,
        root: Path,
        *,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        verification: Literal["manifest", "full"],
        load: Callable[[DataArtifactLoadContext], PayloadT],
    ) -> DataArtifact[PayloadT]:
        try:
            validated = self._validate_object(
                root,
                kind=kind,
                artifact_type=artifact_type,
                expected=None,
                verification=verification,
            )
        except DataArtifactValidationError:
            raise
        except OSError as exc:
            raise DataArtifactValidationError(
                f"data artifact object could not be read: {root}"
            ) from exc
        if (
            validated.identity.source_name != source_name
            or validated.identity.materializer_name != materializer_name
        ):
            raise DataArtifactLocatorMismatch(
                "data artifact locator targets an incompatible producer"
            )
        return self._load_validated_object(
            root,
            validated=validated,
            verification=verification,
            load=load,
        )

    def _publish_or_select_winner(
        self,
        *,
        paths: ArtifactStorePaths,
        staging: Path,
        destination: Path,
        identity: DataArtifactIdentity,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
    ) -> None:
        if _path_exists(paths.cache_root, destination):
            try:
                winner_identity = self._load_framework_identity(
                    destination,
                    kind=kind,
                    artifact_type=artifact_type,
                    verification="full",
                )
            except (OSError, DataArtifactValidationError):
                _quarantine_entry(
                    paths,
                    destination,
                    destination_root=paths.quarantine_objects,
                    is_directory=True,
                )
            else:
                if winner_identity != identity:
                    raise RuntimeError(
                        "data artifact digest collision or producer contract bug"
                    )
                return
        try:
            publish_cache_directory(
                paths.cache_root,
                staging,
                destination,
                label="data artifact object publication",
            )
        except FileExistsError:
            winner_identity = self._load_framework_identity(
                destination,
                kind=kind,
                artifact_type=artifact_type,
                verification="full",
            )
            if winner_identity != identity:
                raise RuntimeError(
                    "data artifact digest collision or producer contract bug"
                ) from None

    def _load_object[PayloadT](
        self,
        root: Path,
        *,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        expected: DataArtifactIdentity | None,
        verification: Literal["manifest", "full"],
        load: Callable[[DataArtifactLoadContext], PayloadT],
    ) -> DataArtifact[PayloadT]:
        validated = self._validate_object(
            root,
            kind=kind,
            artifact_type=artifact_type,
            expected=expected,
            verification=verification,
        )
        if validated.identity.source_name != source_name:
            raise DataArtifactValidationError(
                "data artifact source name does not match"
            )
        if validated.identity.materializer_name != materializer_name:
            raise DataArtifactValidationError(
                "data artifact materializer name does not match"
            )
        return self._load_validated_object(
            root,
            validated=validated,
            verification=verification,
            load=load,
        )

    def _validate_object(
        self,
        root: Path,
        *,
        kind: Literal["managed", "referenced"],
        artifact_type: str,
        expected: DataArtifactIdentity | None,
        verification: Literal["manifest", "full"],
    ) -> ValidatedArtifactObject:
        try:
            root = canonical_directory(root, label="data artifact object")
            raw, manifest_bytes = _load_canonical_json(
                root,
                "manifest.json",
                label="data artifact manifest",
            )
            manifest = _strict_mapping(
                raw,
                fields=_MANIFEST_FIELDS,
                path="data artifact manifest",
            )
            if (
                type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != 2
            ):
                raise ValueError("data artifact manifest.schema_version must be 2")
            if manifest["kind"] != kind:
                raise ValueError("data artifact manifest kind does not match")
            if manifest["artifact_type"] != artifact_type:
                raise ValueError(
                    "data artifact manifest artifact_type does not match"
                )
            source = _strict_mapping(
                manifest["source"],
                fields=_NAMED_DIGEST_FIELDS,
                path="data artifact manifest.source",
            )
            materializer = _strict_mapping(
                manifest["materializer"],
                fields=_NAMED_DIGEST_FIELDS,
                path="data artifact manifest.materializer",
            )
            source_name = _non_empty_string(
                source["name"],
                path="data artifact manifest.source.name",
            )
            materializer_name = _non_empty_string(
                materializer["name"],
                path="data artifact manifest.materializer.name",
            )
            source_digest = _sha256(
                source["digest"], path="data artifact source.digest"
            )
            materialization_digest = _sha256(
                materializer["digest"],
                path="data artifact materializer.digest",
            )
            content_digest = _sha256(
                manifest["content_digest"],
                path="data artifact manifest.content_digest",
            )
            stored_digest, records, shard_digests = _read_inventory(
                root,
                manifest["stored_files"],
            )
            manifest_sha256 = _sha256_bytes(manifest_bytes)
            snapshot = scan_regular_files(
                root,
                hash_contents=verification == "full",
                label="data artifact validation input",
                on_progress=self._verification_progress(
                    artifact_type=artifact_type,
                    source_name=source_name,
                    materializer_name=materializer_name,
                    phase="validate",
                    verification=verification,
                ),
                workers=self.context.verification_workers,
            )
            stored_paths = tuple(
                cast(str, record["path"]) for record in records
            )
            shard_paths = tuple(path for path, _ in shard_digests)
            _validate_object_layout(
                root,
                stored_paths=stored_paths,
                shard_paths=shard_paths,
                snapshot=snapshot,
            )
            _validate_inventory_snapshot(
                snapshot,
                records=records,
                shard_digests=shard_digests,
                manifest_sha256=manifest_sha256,
                verification=verification,
            )
            if kind == "managed" and content_digest != stored_digest:
                raise ValueError(
                    "managed data artifact content digest does not match "
                    "stored files"
                )
            domain = _strict_mapping(
                manifest["domain"],
                fields=frozenset(cast(dict[str, Any], manifest["domain"])),
                path="data artifact manifest.domain",
            )
            domain_digest = _sha256(
                manifest["domain_digest"],
                path="data artifact manifest.domain_digest",
            )
            if canonical_artifact_digest(domain) != domain_digest:
                raise ValueError("data artifact domain digest mismatch")
            artifact_digest = _sha256(
                manifest["artifact_digest"],
                path="data artifact manifest.artifact_digest",
            )
            computed_artifact_digest = _artifact_digest(
                kind=kind,
                artifact_type=artifact_type,
                source=source,
                materializer=materializer,
                content_digest=content_digest,
                stored_files_digest=stored_digest,
                domain_digest=domain_digest,
            )
            if artifact_digest != computed_artifact_digest:
                raise ValueError("data artifact digest mismatch")
            if root.name != artifact_digest and root.parent.name == "objects":
                raise ValueError("data artifact object path digest does not match")
            identity = DataArtifactIdentity(
                kind=kind,
                artifact_type=artifact_type,
                source_name=source_name,
                source_digest=source_digest,
                materializer_name=materializer_name,
                materialization_digest=materialization_digest,
                content_digest=content_digest,
                artifact_digest=artifact_digest,
                manifest_sha256=manifest_sha256,
            )
            if expected is not None and identity != expected:
                raise ValueError(
                    "data artifact identity does not match expected identity"
                )
            return ValidatedArtifactObject(
                identity=identity,
                domain=domain,
                snapshot=snapshot,
            )
        except ArtifactVerificationObserverFailure:
            raise
        except FileNotFoundError:
            raise
        except DataArtifactValidationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise DataArtifactValidationError(str(exc)) from exc

    def _load_validated_object[PayloadT](
        self,
        root: Path,
        *,
        validated: ValidatedArtifactObject,
        verification: Literal["manifest", "full"],
        load: Callable[[DataArtifactLoadContext], PayloadT],
    ) -> DataArtifact[PayloadT]:
        try:
            artifact_without_payload: DataArtifact[None] = DataArtifact(
                root=root,
                identity=validated.identity,
                payload=None,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise DataArtifactValidationError(str(exc)) from exc
        payload = load(
            DataArtifactLoadContext(
                data_root=artifact_without_payload.root / "data",
                identity=validated.identity,
                domain=validated.domain,
                verification=verification,
            )
        )
        try:
            after_load = scan_regular_files(
                artifact_without_payload.root,
                hash_contents=verification == "full",
                label="data artifact load input",
                on_progress=self._verification_progress(
                    artifact_type=validated.identity.artifact_type,
                    source_name=validated.identity.source_name,
                    materializer_name=validated.identity.materializer_name,
                    phase="post_load",
                    verification=verification,
                ),
                workers=self.context.verification_workers,
            )
        except ArtifactVerificationObserverFailure:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "data artifact load callback mutated its artifact"
            ) from exc
        if validated.snapshot != after_load:
            raise RuntimeError("data artifact load callback mutated its artifact")
        try:
            return DataArtifact(
                root=artifact_without_payload.root,
                identity=validated.identity,
                payload=payload,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "data artifact load callback mutated its artifact"
            ) from exc

    def _verification_progress(
        self,
        *,
        artifact_type: str,
        source_name: str,
        materializer_name: str,
        phase: ArtifactVerificationPhase,
        verification: Literal["manifest", "full"],
    ) -> Callable[[int, int], None] | None:
        observer = self.context.verification_observer
        if observer is None or verification != "full":
            return None

        def notify(completed: int, total: int) -> None:
            try:
                observer(
                    ArtifactVerificationEvent(
                        artifact_type=artifact_type,
                        source_name=source_name,
                        materializer_name=materializer_name,
                        phase=phase,
                        completed=completed,
                        total=total,
                    )
                )
            except Exception as exc:
                raise ArtifactVerificationObserverFailure(exc) from exc

        return notify


__all__ = [
    "DataArtifactLoadContext",
    "DataArtifactStore",
    "DataArtifactValidationError",
    "ManagedDataArtifactBuild",
    "ReferencedDataArtifactBuild",
    "canonical_artifact_digest",
    "canonical_artifact_json_bytes",
]
