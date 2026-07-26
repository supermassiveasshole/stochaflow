"""Shared serialization, locking, and cache locator primitives."""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import socket
import time
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import Any, Literal, Self, cast
from uuid import uuid4

from stochaflow.data.artifact_io import (
    cache_entry_exists,
    lexical_absolute_path,
    open_cache_file,
    quarantine_cache_entry,
    read_regular_file,
    write_cache_file,
)

LOCK_WAIT_SECONDS = 30.0
LOCK_POLL_SECONDS = 0.05
WINDOWS_LOCK_BYTE_OFFSET = 4096
ADVISORY_LOCK_API: Any = importlib.import_module(
    "msvcrt" if os.name == "nt" else "fcntl"
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode one value as newline-terminated canonical JSON."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    """Hash the canonical JSON representation of one value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash an in-memory byte sequence."""

    return hashlib.sha256(value).hexdigest()


def strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> dict[str, Any]:
    """Require one mapping with exactly the declared string fields."""

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


def load_canonical_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """Load one canonical JSON mapping through link-safe artifact I/O."""

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
    if encoded != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return cast(dict[str, Any], value), encoded


class ArtifactMaterializationLock(
    AbstractContextManager["ArtifactMaterializationLock"]
):
    """Cooperative lock with bounded waiting and diagnosable ownership."""

    def __init__(
        self,
        path: Path,
        *,
        cache_root: Path | None = None,
        wait_seconds: float = LOCK_WAIT_SECONDS,
        poll_seconds: float = LOCK_POLL_SECONDS,
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
            if try_acquire_advisory_lock(handle):
                self.handle = handle
                break
            os.close(handle)
            if time.monotonic() >= deadline:
                owner = lock_owner_diagnostic(self.path)
                raise RuntimeError(
                    "timed out waiting for data artifact materialization "
                    f"lock: {self.path}; observed owner: {owner}"
                )
            time.sleep(
                min(
                    self.poll_seconds,
                    max(0.0, deadline - time.monotonic()),
                )
            )
        owner = {
            "schema_version": 1,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at_ns": time.time_ns(),
            "nonce": uuid4().hex,
        }
        self.owner_bytes = canonical_json_bytes(owner)
        try:
            os.lseek(self.handle, 0, os.SEEK_SET)
            write_descriptor(self.handle, self.owner_bytes)
            os.ftruncate(self.handle, len(self.owner_bytes))
            os.fsync(self.handle)
        except BaseException:
            release_advisory_lock(self.handle)
            os.close(self.handle)
            self.handle = None
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.handle is None:
            return
        release_advisory_lock(self.handle)
        os.close(self.handle)
        self.handle = None


def write_descriptor(descriptor: int, encoded: bytes) -> None:
    """Write all bytes to an open descriptor."""

    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write artifact lock metadata")
        remaining = remaining[written:]


def try_acquire_advisory_lock(descriptor: int) -> bool:
    """Attempt one platform-native non-blocking advisory lock."""

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


def release_advisory_lock(descriptor: int) -> None:
    """Release a lock acquired by :func:`try_acquire_advisory_lock`."""

    if os.name == "nt":
        os.lseek(descriptor, WINDOWS_LOCK_BYTE_OFFSET, os.SEEK_SET)
        ADVISORY_LOCK_API.locking(
            descriptor,
            ADVISORY_LOCK_API.LK_UNLCK,
            1,
        )
        return
    ADVISORY_LOCK_API.flock(descriptor, ADVISORY_LOCK_API.LOCK_UN)


def lock_owner_diagnostic(path: Path) -> str:
    """Render best-effort ownership metadata from a persistent lock file."""

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


def read_locator(path: Path) -> str:
    """Load one strict content-addressed artifact locator."""

    raw, _ = load_canonical_json(path, label="data artifact locator")
    if set(raw) != {"artifact_digest"}:
        raise ValueError("data artifact locator has invalid fields")
    digest = raw["artifact_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("data artifact locator artifact_digest is invalid")
    return digest


def write_locator(
    cache_root: Path,
    path: Path,
    artifact_digest: str,
) -> None:
    """Atomically write one content-addressed artifact locator."""

    write_cache_file(
        cache_root,
        path,
        canonical_json_bytes({"artifact_digest": artifact_digest}),
        label="data artifact locator",
    )


def quarantine_path(cache_root: Path, path: Path) -> Path:
    """Quarantine one corrupt cache entry without following links."""

    return quarantine_cache_entry(
        cache_root,
        path,
        suffix="corrupt",
        label="corrupt data artifact cache entry",
    )


def path_exists_without_following(cache_root: Path, path: Path) -> bool:
    """Check one cache entry without following its final component."""

    return cache_entry_exists(
        cache_root,
        path,
        label="data artifact cache entry",
    )


def read_locator_for_policy(
    cache_root: Path,
    path: Path,
    *,
    policy: Literal["require", "ensure"],
    quarantine_on_error: bool = True,
) -> str | None:
    """Read a locator under require/repair policy."""

    if not path_exists_without_following(cache_root, path):
        return None
    try:
        return read_locator(path)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        if policy == "require":
            raise
        if quarantine_on_error and path_exists_without_following(
            cache_root,
            path,
        ):
            with suppress(FileNotFoundError):
                quarantine_path(cache_root, path)
        return None


__all__ = [
    "ArtifactMaterializationLock",
    "canonical_digest",
    "canonical_json_bytes",
    "load_canonical_json",
    "lock_owner_diagnostic",
    "path_exists_without_following",
    "quarantine_path",
    "read_locator",
    "read_locator_for_policy",
    "release_advisory_lock",
    "sha256_bytes",
    "strict_mapping",
    "try_acquire_advisory_lock",
    "write_descriptor",
    "write_locator",
]
