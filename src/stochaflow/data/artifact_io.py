"""Link-safe filesystem primitives for data artifact I/O."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from uuid import uuid4

MAX_ARTIFACT_VERIFICATION_WORKERS = 8
_HASH_TASKS_PER_WORKER = 2


@dataclass(frozen=True, slots=True)
class ArtifactFileSnapshot:
    """Metadata and optional content digest read from one file descriptor."""

    relative_path: str
    size_bytes: int
    sha256: str | None
    device: int
    inode: int
    mode: int
    modified_ns: int
    changed_ns: int


def canonical_directory(path: Path, *, label: str) -> Path:
    """Return a stable lexical absolute path after rejecting linked components."""

    canonical = lexical_absolute_path(path)
    if os.name == "nt":
        _validate_windows_directory_chain(canonical, label=label)
        return canonical
    descriptor = _open_posix_directory_chain(canonical, label=label)
    os.close(descriptor)
    return canonical


def open_anchored_directory(path: Path, *, label: str) -> int:
    """Open a POSIX directory through a no-follow chain from its filesystem root.

    The caller owns the returned descriptor. Platforms without descriptor-relative
    no-follow traversal must use their portable identity-validation path instead.
    """

    if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError(
            "descriptor-anchored directory traversal is unavailable"
        )
    canonical = lexical_absolute_path(path)
    return _open_posix_directory_chain(canonical, label=label)


def lexical_absolute_path(path: Path) -> Path:
    """Normalize dot segments without resolving links or changing path identity."""

    absolute = path if path.is_absolute() else Path.cwd() / path
    normalized: list[str] = []
    for component in absolute.parts[1:]:
        if component in {"", "."}:
            continue
        if component == "..":
            if normalized:
                normalized.pop()
            continue
        normalized.append(component)
    return Path(absolute.anchor, *normalized)


def _cache_paths(
    cache_root: Path,
    path: Path,
    *,
    label: str,
    allow_root: bool = False,
) -> tuple[Path, Path]:
    canonical_root = lexical_absolute_path(cache_root)
    canonical = lexical_absolute_path(path)
    try:
        relative = canonical.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(f"{label} escaped its cache root") from exc
    if not relative.parts:
        if not allow_root:
            raise ValueError(f"{label} must be below its cache root")
        return canonical_root, canonical
    _relative_parts(relative.as_posix(), label=label)
    return canonical_root, canonical


def _ensure_directory_chain(path: Path, *, label: str) -> None:
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        flags = _posix_directory_flags()
        descriptor = os.open(path.anchor or os.sep, flags)
        try:
            for component in path.parts[1:]:
                try:
                    next_descriptor = os.open(
                        component,
                        flags,
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    with suppress(FileExistsError):
                        os.mkdir(component, mode=0o755, dir_fd=descriptor)
                    next_descriptor = os.open(
                        component,
                        flags,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError(
                            f"{label} contains a symlink or invalid directory: "
                            f"{path}"
                        ) from exc
                    raise
                metadata = os.fstat(next_descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    os.close(next_descriptor)
                    raise NotADirectoryError(
                        f"{label} is not a directory: {path}"
                    )
                os.close(descriptor)
                descriptor = next_descriptor
        finally:
            os.close(descriptor)
        return

    current = Path(path.anchor)
    snapshots: list[tuple[Path, os.stat_result]] = []
    anchor_metadata = current.lstat()
    if _is_link_or_reparse(anchor_metadata) or not stat.S_ISDIR(
        anchor_metadata.st_mode
    ):
        raise ValueError(f"{label} has an invalid filesystem anchor: {current}")
    snapshots.append((current, anchor_metadata))
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            with suppress(FileExistsError):
                current.mkdir()
            metadata = current.lstat()
        if _is_link_or_reparse(metadata):
            raise ValueError(
                f"{label} contains a symlink or reparse point: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(f"{label} is not a directory: {current}")
        snapshots.append((current, metadata))
    _validate_windows_mutation_snapshots(tuple(snapshots), label=label)


def _write_descriptor(descriptor: int, encoded: bytes) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write cache file")
        remaining = remaining[written:]


def ensure_cache_directory(
    cache_root: Path,
    directory: Path,
    *,
    label: str,
) -> Path:
    """Create one cache directory tree without following linked components."""

    canonical_root, canonical_directory_path = _cache_paths(
        cache_root,
        directory,
        label=label,
        allow_root=True,
    )
    _ensure_directory_chain(canonical_root, label=f"{label} root")
    _ensure_directory_chain(canonical_directory_path, label=label)
    return canonical_directory_path


def cache_entry_exists(
    cache_root: Path,
    path: Path,
    *,
    label: str,
) -> bool:
    """Return whether a cache entry exists after validating its parent chain."""

    _, canonical = _cache_paths(cache_root, path, label=label)
    try:
        canonical_directory(canonical.parent, label=f"{label} parent")
        canonical.lstat()
    except FileNotFoundError:
        return False
    return True


def create_cache_directory(
    cache_root: Path,
    directory: Path,
    *,
    label: str,
) -> Path:
    """Create one new directory through a link-safe cache parent."""

    canonical_root, canonical = _cache_paths(
        cache_root,
        directory,
        label=label,
    )
    parent = ensure_cache_directory(
        canonical_root,
        canonical.parent,
        label=f"{label} parent",
    )
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        parent_descriptor = _open_posix_directory_chain(
            parent,
            label=f"{label} parent",
        )
        try:
            os.mkdir(canonical.name, dir_fd=parent_descriptor)
            child_descriptor = _open_posix_relative_directory(
                parent_descriptor,
                canonical.name,
                label=label,
            )
            os.close(child_descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return canonical

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    canonical.mkdir()
    metadata = canonical.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a real directory: {canonical}")
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    return canonical


def write_cache_file(
    cache_root: Path,
    path: Path,
    encoded: bytes,
    *,
    label: str,
) -> Path:
    """Atomically write one cache file through a validated parent directory."""

    canonical_root, canonical = _cache_paths(cache_root, path, label=label)
    parent = ensure_cache_directory(
        canonical_root,
        canonical.parent,
        label=f"{label} parent",
    )
    staging_name = f".{canonical.name}.{uuid4().hex}.tmp"
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        parent_descriptor = _open_posix_directory_chain(
            parent,
            label=f"{label} parent",
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                staging_name,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            _write_descriptor(descriptor, encoded)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(
                staging_name,
                canonical.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(staging_name, dir_fd=parent_descriptor)
            os.close(parent_descriptor)
        return canonical

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    staging = parent / staging_name
    try:
        descriptor = os.open(
            staging,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
        )
        try:
            _write_descriptor(descriptor, encoded)
            os.fsync(descriptor)
            staged_metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        observed_staging = staging.lstat()
        if (
            _is_link_or_reparse(observed_staging)
            or not stat.S_ISREG(observed_staging.st_mode)
            or not _same_stable_file(staged_metadata, observed_staging)
        ):
            raise ValueError(f"{label} staging file changed: {staging}")
        _validate_windows_mutation_snapshots(parent_snapshots, label=label)
        staging.replace(canonical)
        published = canonical.lstat()
        if _is_link_or_reparse(published) or not _same_stable_file(
            staged_metadata,
            published,
        ):
            raise ValueError(f"{label} changed while it was published: {canonical}")
        _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    finally:
        try:
            current_parent = canonical_directory(
                parent,
                label=f"{label} cleanup parent",
            )
        except (FileNotFoundError, NotADirectoryError, ValueError):
            pass
        else:
            candidate = current_parent / staging_name
            try:
                candidate_metadata = candidate.lstat()
            except FileNotFoundError:
                pass
            else:
                if (
                    not _is_link_or_reparse(candidate_metadata)
                    and stat.S_ISREG(candidate_metadata.st_mode)
                ):
                    candidate.unlink()
    return canonical


def publish_cache_directory(
    cache_root: Path,
    staging: Path,
    destination: Path,
    *,
    label: str,
) -> Path:
    """Publish one staging directory without following either parent path."""

    canonical_root, canonical_staging = _cache_paths(
        cache_root,
        staging,
        label=f"{label} staging",
    )
    _, canonical_destination = _cache_paths(
        canonical_root,
        destination,
        label=f"{label} destination",
    )
    source_parent = canonical_directory(
        canonical_staging.parent,
        label=f"{label} staging parent",
    )
    destination_parent = ensure_cache_directory(
        canonical_root,
        canonical_destination.parent,
        label=f"{label} destination parent",
    )
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        source_parent_descriptor = _open_posix_directory_chain(
            source_parent,
            label=f"{label} staging parent",
        )
        try:
            destination_parent_descriptor = _open_posix_directory_chain(
                destination_parent,
                label=f"{label} destination parent",
            )
            try:
                source_metadata = os.stat(
                    canonical_staging.name,
                    dir_fd=source_parent_descriptor,
                    follow_symlinks=False,
                )
                if _is_link_or_reparse(source_metadata) or not stat.S_ISDIR(
                    source_metadata.st_mode
                ):
                    raise ValueError(f"{label} staging is not a real directory")
                try:
                    os.stat(
                        canonical_destination.name,
                        dir_fd=destination_parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(canonical_destination)
                _rename_posix_no_replace_between(
                    source_parent_descriptor,
                    canonical_staging.name,
                    destination_parent_descriptor,
                    canonical_destination.name,
                )
                published = os.stat(
                    canonical_destination.name,
                    dir_fd=destination_parent_descriptor,
                    follow_symlinks=False,
                )
                if not _same_stable_file(source_metadata, published):
                    raise ValueError(
                        f"{label} destination changed during publication"
                    )
                os.fsync(source_parent_descriptor)
                if destination_parent_descriptor != source_parent_descriptor:
                    os.fsync(destination_parent_descriptor)
            finally:
                os.close(destination_parent_descriptor)
        finally:
            os.close(source_parent_descriptor)
        return canonical_destination

    parent_snapshots = _windows_directory_snapshots(
        source_parent,
        label=f"{label} staging parent",
    )
    destination_snapshots = _windows_directory_snapshots(
        destination_parent,
        label=f"{label} destination parent",
    )
    source_metadata = canonical_staging.lstat()
    if _is_link_or_reparse(source_metadata) or not stat.S_ISDIR(
        source_metadata.st_mode
    ):
        raise ValueError(f"{label} staging is not a real directory")
    try:
        canonical_destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(canonical_destination)
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    canonical_staging.rename(canonical_destination)
    published = canonical_destination.lstat()
    if (
        _is_link_or_reparse(published)
        or not _same_stable_file(source_metadata, published)
    ):
        raise ValueError(f"{label} destination changed during publication")
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    _validate_windows_mutation_snapshots(destination_snapshots, label=label)
    return canonical_destination


def publish_cache_file(
    cache_root: Path,
    staging: Path,
    destination: Path,
    *,
    label: str,
) -> Path:
    """Publish one staging file without replacing an existing destination."""

    canonical_root, canonical_staging = _cache_paths(
        cache_root,
        staging,
        label=f"{label} staging",
    )
    _, canonical_destination = _cache_paths(
        canonical_root,
        destination,
        label=f"{label} destination",
    )
    source_parent = canonical_directory(
        canonical_staging.parent,
        label=f"{label} staging parent",
    )
    destination_parent = ensure_cache_directory(
        canonical_root,
        canonical_destination.parent,
        label=f"{label} destination parent",
    )
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        source_parent_descriptor = _open_posix_directory_chain(
            source_parent,
            label=f"{label} staging parent",
        )
        try:
            destination_parent_descriptor = _open_posix_directory_chain(
                destination_parent,
                label=f"{label} destination parent",
            )
            try:
                source_metadata = os.stat(
                    canonical_staging.name,
                    dir_fd=source_parent_descriptor,
                    follow_symlinks=False,
                )
                if _is_link_or_reparse(source_metadata) or not stat.S_ISREG(
                    source_metadata.st_mode
                ):
                    raise ValueError(f"{label} staging is not a regular file")
                try:
                    os.stat(
                        canonical_destination.name,
                        dir_fd=destination_parent_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(canonical_destination)
                _rename_posix_no_replace_between(
                    source_parent_descriptor,
                    canonical_staging.name,
                    destination_parent_descriptor,
                    canonical_destination.name,
                )
                published = os.stat(
                    canonical_destination.name,
                    dir_fd=destination_parent_descriptor,
                    follow_symlinks=False,
                )
                if not _same_stable_file(source_metadata, published):
                    raise ValueError(
                        f"{label} destination changed during publication"
                    )
                os.fsync(source_parent_descriptor)
                if destination_parent_descriptor != source_parent_descriptor:
                    os.fsync(destination_parent_descriptor)
            finally:
                os.close(destination_parent_descriptor)
        finally:
            os.close(source_parent_descriptor)
        return canonical_destination

    parent_snapshots = _windows_directory_snapshots(
        source_parent,
        label=f"{label} staging parent",
    )
    destination_snapshots = _windows_directory_snapshots(
        destination_parent,
        label=f"{label} destination parent",
    )
    source_metadata = canonical_staging.lstat()
    if _is_link_or_reparse(source_metadata) or not stat.S_ISREG(
        source_metadata.st_mode
    ):
        raise ValueError(f"{label} staging is not a regular file")
    try:
        canonical_destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(canonical_destination)
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    canonical_staging.rename(canonical_destination)
    published = canonical_destination.lstat()
    if (
        _is_link_or_reparse(published)
        or not _same_stable_file(source_metadata, published)
    ):
        raise ValueError(f"{label} destination changed during publication")
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    _validate_windows_mutation_snapshots(destination_snapshots, label=label)
    return canonical_destination


def quarantine_cache_entry(
    cache_root: Path,
    path: Path,
    *,
    suffix: str,
    label: str,
) -> Path:
    """Move one corrupt cache entry aside without following the entry itself."""

    canonical_root, canonical = _cache_paths(cache_root, path, label=label)
    parent = canonical_directory(canonical.parent, label=f"{label} parent")
    quarantine = canonical.with_name(f"{canonical.name}.{uuid4().hex}.{suffix}")
    _, quarantine = _cache_paths(
        canonical_root,
        quarantine,
        label=f"{label} quarantine",
    )
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        parent_descriptor = _open_posix_directory_chain(
            parent,
            label=f"{label} parent",
        )
        try:
            _rename_posix_no_replace(
                parent_descriptor,
                canonical.name,
                quarantine.name,
            )
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return quarantine

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    canonical.rename(quarantine)
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    return quarantine


def remove_cache_directory(
    cache_root: Path,
    directory: Path,
    *,
    label: str,
) -> None:
    """Remove one cache-owned directory without traversing a linked parent."""

    _, canonical = _cache_paths(cache_root, directory, label=label)
    parent = canonical_directory(canonical.parent, label=f"{label} parent")
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        parent_descriptor = _open_posix_directory_chain(
            parent,
            label=f"{label} parent",
        )
        try:
            metadata = os.stat(
                canonical.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"{label} is not a real directory")
            shutil.rmtree(canonical.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    canonical_directory(canonical, label=label)
    shutil.rmtree(canonical)
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)


def remove_cache_file(
    cache_root: Path,
    path: Path,
    *,
    label: str,
) -> None:
    """Remove one regular cache file through a link-safe parent."""

    _, canonical = _cache_paths(cache_root, path, label=label)
    parent = canonical_directory(canonical.parent, label=f"{label} parent")
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        parent_descriptor = _open_posix_directory_chain(
            parent,
            label=f"{label} parent",
        )
        try:
            metadata = os.stat(
                canonical.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{label} is not a regular file")
            os.unlink(canonical.name, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    metadata = canonical.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file")
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)
    canonical.unlink()
    _validate_windows_mutation_snapshots(parent_snapshots, label=label)


def create_cache_file_exclusive(
    cache_root: Path,
    path: Path,
    *,
    label: str,
) -> int:
    """Create one exclusive read/write cache file and return its descriptor."""

    canonical_root, canonical = _cache_paths(cache_root, path, label=label)
    parent = ensure_cache_directory(
        canonical_root,
        canonical.parent,
        label=f"{label} parent",
    )
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_RDWR
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        parent_descriptor = _open_posix_directory_chain(
            parent,
            label=f"{label} parent",
        )
        try:
            return os.open(
                canonical.name,
                flags | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
        finally:
            os.close(parent_descriptor)

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    descriptor = os.open(canonical, flags)
    try:
        opened = os.fstat(descriptor)
        observed = canonical.lstat()
        if (
            _is_link_or_reparse(observed)
            or not stat.S_ISREG(observed.st_mode)
            or not _same_stable_file(opened, observed)
        ):
            raise ValueError(f"{label} changed while it was created")
        _validate_windows_mutation_snapshots(parent_snapshots, label=label)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_cache_file(
    cache_root: Path,
    path: Path,
    *,
    label: str,
) -> int:
    """Open or create one persistent regular cache file without following links."""

    canonical_root, canonical = _cache_paths(cache_root, path, label=label)
    parent = ensure_cache_directory(
        canonical_root,
        canonical.parent,
        label=f"{label} parent",
    )
    flags = (
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        descriptor: int | None = None
        for attempt in range(2):
            parent_descriptor = _open_posix_directory_chain(
                parent,
                label=f"{label} parent",
            )
            try:
                descriptor = os.open(
                    canonical.name,
                    flags | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                break
            except FileNotFoundError:
                # macOS may transiently report ENOENT when two threads create
                # the same O_NOFOLLOW file. Reopening the no-follow directory
                # chain keeps the retry anchored and safe.
                if attempt > 0:
                    raise
            finally:
                os.close(parent_descriptor)
        if descriptor is None:
            raise RuntimeError(f"{label} could not be opened")
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(descriptor)
            raise ValueError(f"{label} is not a regular file")
        return descriptor

    parent_snapshots = _windows_directory_snapshots(
        parent,
        label=f"{label} parent",
    )
    try:
        observed_before = canonical.lstat()
    except FileNotFoundError:
        observed_before = None
    else:
        if _is_link_or_reparse(observed_before) or not stat.S_ISREG(
            observed_before.st_mode
        ):
            raise ValueError(f"{label} is not a regular file")
    descriptor = os.open(canonical, flags)
    try:
        opened = os.fstat(descriptor)
        observed_after = canonical.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_link_or_reparse(observed_after)
            or not _same_stable_file(opened, observed_after)
            or (
                observed_before is not None
                and not _same_stable_file(observed_before, opened)
            )
        ):
            raise ValueError(f"{label} changed while it was opened")
        _validate_windows_mutation_snapshots(parent_snapshots, label=label)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_regular_file(
    root: Path,
    relative_path: str,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    """Read one regular file without following any path component."""

    canonical_root = canonical_directory(root, label=f"{label} root")
    parts = _relative_parts(relative_path, label=label)
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        return _read_posix_regular_file(
            canonical_root,
            parts,
            label=label,
        )
    return _read_windows_regular_file(
        canonical_root,
        parts,
        label=label,
    )


def scan_regular_files(
    root: Path,
    *,
    hash_contents: bool,
    label: str,
    path_filter: Callable[[str], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    workers: int | None = None,
) -> tuple[ArtifactFileSnapshot, ...]:
    """Enumerate regular files and hash independent contents concurrently.

    Progress callbacks run on the calling thread in deterministic path order.
    """

    canonical_root = canonical_directory(root, label=label)
    candidates = _enumerate_regular_files(
        canonical_root,
        label=label,
        path_filter=path_filter,
    )
    if not hash_contents:
        return candidates
    total = len(candidates)
    if on_progress is not None:
        on_progress(0, total)
    if not candidates:
        return ()

    def report_hash_progress(completed: int, progress_total: int) -> None:
        if on_progress is not None and completed < progress_total:
            on_progress(completed, progress_total)

    hash_progress = report_hash_progress if on_progress is not None else None
    worker_count = _artifact_hash_worker_count(workers)
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        root_descriptor = _open_posix_directory_chain(
            canonical_root,
            label=label,
        )
        try:
            hashed = _hash_regular_file_candidates(
                candidates,
                worker_count=worker_count,
                hash_candidate=lambda candidate: (
                    _snapshot_posix_regular_path(
                        root_descriptor,
                        candidate,
                        label=label,
                    )
                ),
                on_progress=hash_progress,
            )
        finally:
            os.close(root_descriptor)
    else:
        hashed = _hash_regular_file_candidates(
            candidates,
            worker_count=worker_count,
            hash_candidate=lambda candidate: (
                _snapshot_windows_regular_path(
                    canonical_root,
                    candidate,
                    label=label,
                )
            ),
            on_progress=hash_progress,
        )
    after = _enumerate_regular_files(
        canonical_root,
        label=label,
        path_filter=path_filter,
    )
    if not _same_snapshot_metadata(candidates, after):
        raise ValueError(f"{label} changed during content hashing")
    if on_progress is not None:
        on_progress(total, total)
    return hashed


def _enumerate_regular_files(
    canonical_root: Path,
    *,
    label: str,
    path_filter: Callable[[str], bool] | None,
) -> tuple[ArtifactFileSnapshot, ...]:
    if os.name != "nt" and hasattr(os, "O_NOFOLLOW"):
        descriptor = _open_posix_directory_chain(
            canonical_root,
            label=label,
        )
        try:
            snapshots = _scan_posix_directory(
                descriptor,
                (),
                label=label,
                path_filter=path_filter,
            )
        finally:
            os.close(descriptor)
    else:
        snapshots = _scan_windows_directory(
            canonical_root,
            canonical_root,
            label=label,
            path_filter=path_filter,
        )
    return tuple(sorted(snapshots, key=lambda item: item.relative_path))


def _artifact_hash_worker_count(requested_workers: int | None = None) -> int:
    requested = cast(object, requested_workers)
    if requested is not None:
        if not isinstance(requested, int) or isinstance(requested, bool):
            raise TypeError("artifact hash workers must be an integer or None")
        if requested <= 0:
            raise ValueError("artifact hash workers must be positive")
        if requested > MAX_ARTIFACT_VERIFICATION_WORKERS:
            raise ValueError(
                "artifact hash workers must not exceed "
                f"{MAX_ARTIFACT_VERIFICATION_WORKERS}"
            )
        return requested
    return min(
        MAX_ARTIFACT_VERIFICATION_WORKERS,
        max(1, os.cpu_count() or 1),
    )


def _hash_regular_file_candidates(
    candidates: tuple[ArtifactFileSnapshot, ...],
    *,
    worker_count: int,
    hash_candidate: Callable[[ArtifactFileSnapshot], ArtifactFileSnapshot],
    on_progress: Callable[[int, int], None] | None,
) -> tuple[ArtifactFileSnapshot, ...]:
    total = len(candidates)
    if worker_count <= 1:
        snapshots: list[ArtifactFileSnapshot] = []
        for completed, candidate in enumerate(candidates, start=1):
            snapshots.append(hash_candidate(candidate))
            if on_progress is not None:
                on_progress(completed, total)
        return tuple(snapshots)

    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="stochaflow-artifact-hash",
    )
    pending: deque[Future[ArtifactFileSnapshot]] = deque()
    iterator: Iterator[ArtifactFileSnapshot] = iter(candidates)
    snapshots = []

    def submit_next() -> bool:
        try:
            candidate = next(iterator)
        except StopIteration:
            return False
        pending.append(executor.submit(hash_candidate, candidate))
        return True

    try:
        for _ in range(worker_count * _HASH_TASKS_PER_WORKER):
            if not submit_next():
                break
        while pending:
            snapshot = pending.popleft().result()
            snapshots.append(snapshot)
            if on_progress is not None:
                on_progress(len(snapshots), total)
            submit_next()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    return tuple(snapshots)


def _same_snapshot_metadata(
    before: tuple[ArtifactFileSnapshot, ...],
    after: tuple[ArtifactFileSnapshot, ...],
) -> bool:
    return len(before) == len(after) and all(
        first.relative_path == second.relative_path
        and _same_artifact_snapshot_state(first, second)
        for first, second in zip(before, after, strict=True)
    )


def _same_complete_snapshot_metadata(
    before: tuple[ArtifactFileSnapshot, ...],
    after: tuple[ArtifactFileSnapshot, ...],
) -> bool:
    """Compare complete file metadata while deliberately ignoring digests."""

    return len(before) == len(after) and all(
        first.relative_path == second.relative_path
        and first.size_bytes == second.size_bytes
        and first.device == second.device
        and first.inode == second.inode
        and first.mode == second.mode
        and first.modified_ns == second.modified_ns
        and first.changed_ns == second.changed_ns
        for first, second in zip(before, after, strict=True)
    )


def _same_artifact_snapshot_state(
    first: ArtifactFileSnapshot,
    second: ArtifactFileSnapshot,
) -> bool:
    first_identity = (first.device, first.inode)
    second_identity = (second.device, second.inode)
    same_file = (
        first_identity == second_identity
        if first_identity != (0, 0) and second_identity != (0, 0)
        else (
            first.mode == second.mode
            and first.size_bytes == second.size_bytes
            and first.modified_ns == second.modified_ns
        )
    )
    return (
        same_file
        and first.size_bytes == second.size_bytes
        and first.modified_ns == second.modified_ns
    )


def _snapshot_matches_metadata(
    snapshot: ArtifactFileSnapshot,
    metadata: os.stat_result,
) -> bool:
    return _same_artifact_snapshot_state(
        snapshot,
        ArtifactFileSnapshot(
            relative_path=snapshot.relative_path,
            size_bytes=metadata.st_size,
            sha256=None,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            modified_ns=metadata.st_mtime_ns,
            changed_ns=metadata.st_ctime_ns,
        ),
    )


def _relative_parts(relative_path: str, *, label: str) -> tuple[str, ...]:
    pure = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or any(ord(character) < 32 for character in relative_path)
        or pure.is_absolute()
        or not pure.parts
        or any(
            part in {"", ".", ".."}
            or ":" in part
            or part.endswith((" ", "."))
            for part in pure.parts
        )
        or pure.as_posix() != relative_path
    ):
        raise ValueError(f"{label} path must be a normalized relative POSIX path")
    return pure.parts


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _same_file(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    first_identity = (first.st_dev, first.st_ino)
    second_identity = (second.st_dev, second.st_ino)
    if first_identity != (0, 0) and second_identity != (0, 0):
        return first_identity == second_identity
    return (
        first.st_mode == second.st_mode
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
    )


def _same_stable_file(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    first_identity = (first.st_dev, first.st_ino)
    second_identity = (second.st_dev, second.st_ino)
    return (
        first_identity != (0, 0)
        and second_identity != (0, 0)
        and first_identity == second_identity
    )


def _validate_windows_mutation_snapshots(
    ancestors: tuple[tuple[Path, os.stat_result], ...],
    *,
    label: str,
) -> None:
    for path, observed in ancestors:
        current = path.lstat()
        if (
            _is_link_or_reparse(current)
            or not _same_stable_file(observed, current)
        ):
            raise ValueError(
                f"{label} ancestor changed, became linked, "
                f"or has no stable identity: {path}"
            )


def _same_file_state(
    first: os.stat_result,
    second: os.stat_result,
) -> bool:
    return (
        _same_file(first, second)
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
    )


def _digest_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _posix_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _rename_posix_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication is unavailable",
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            1,
        )
    elif sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameatx_np", None)
        if rename is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication is unavailable",
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source,
            parent_descriptor,
            destination,
            0x00000004,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace publication is unsupported on this platform",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


def _rename_posix_no_replace_between(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
) -> None:
    """Atomically rename between two directories without replacement."""

    source = os.fsencode(source_name)
    destination = os.fsencode(destination_name)
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication is unavailable",
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_descriptor,
            source,
            destination_parent_descriptor,
            destination,
            1,
        )
    elif sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameatx_np", None)
        if rename is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication is unavailable",
            )
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_descriptor,
            source,
            destination_parent_descriptor,
            destination,
            0x00000004,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace publication is unsupported on this platform",
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination_name,
    )


def _open_posix_directory_chain(path: Path, *, label: str) -> int:
    flags = _posix_directory_flags()
    anchor = path.anchor or os.sep
    descriptor = os.open(anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        f"{label} contains a symlink or reparse point, "
                        f"or an invalid directory: {path}"
                    ) from exc
                raise
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                raise NotADirectoryError(f"{label} is not a directory: {path}")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_posix_relative_directory(
    parent_descriptor: int,
    component: str,
    *,
    label: str,
) -> int:
    try:
        descriptor = os.open(
            component,
            _posix_directory_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"{label} contains a symlink or reparse point, "
                f"or an invalid directory: {component}"
            ) from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise NotADirectoryError(
            f"{label} path component is not a directory: {component}"
        )
    return descriptor


def _open_posix_regular_file(
    parent_descriptor: int,
    name: str,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"{label} contains a symlink or reparse point: {name}"
            ) from exc
        raise
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"{label} is not a regular file: {name}")
    return descriptor, metadata


def _read_posix_regular_file(
    root: Path,
    parts: tuple[str, ...],
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    directory_descriptor = _open_posix_directory_chain(root, label=f"{label} root")
    opened_directories = [directory_descriptor]
    descriptor: int | None = None
    try:
        for component in parts[:-1]:
            directory_descriptor = _open_posix_relative_directory(
                directory_descriptor,
                component,
                label=label,
            )
            opened_directories.append(directory_descriptor)
        descriptor, metadata = _open_posix_regular_file(
            directory_descriptor,
            parts[-1],
            label=label,
        )
        encoded = _read_descriptor(descriptor)
        if not _same_file_state(metadata, os.fstat(descriptor)):
            raise ValueError(f"{label} changed while it was being read")
        return encoded, metadata
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for opened in reversed(opened_directories):
            os.close(opened)


def _scan_posix_directory(
    directory_descriptor: int,
    relative_parts: tuple[str, ...],
    *,
    label: str,
    path_filter: Callable[[str], bool] | None,
) -> list[ArtifactFileSnapshot]:
    directory_before = os.fstat(directory_descriptor)
    try:
        with os.scandir(directory_descriptor) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        rendered = PurePosixPath(*relative_parts).as_posix()
        raise ValueError(f"cannot enumerate {label}: {rendered}") from exc
    snapshots: list[ArtifactFileSnapshot] = []
    for entry in entries:
        relative = (*relative_parts, entry.name)
        rendered = PurePosixPath(*relative).as_posix()
        try:
            observed = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"cannot inspect {label} entry: {rendered}"
            ) from exc
        if _is_link_or_reparse(observed):
            raise ValueError(
                f"{label} must not contain links or reparse points: {rendered}"
            )
        if stat.S_ISDIR(observed.st_mode):
            child_descriptor = _open_posix_relative_directory(
                directory_descriptor,
                entry.name,
                label=label,
            )
            try:
                opened = os.fstat(child_descriptor)
                if not _same_file(observed, opened):
                    raise ValueError(
                        f"{label} directory changed during enumeration: "
                        f"{rendered}"
                    )
                snapshots.extend(
                    _scan_posix_directory(
                        child_descriptor,
                        relative,
                        label=label,
                        path_filter=path_filter,
                    )
                )
            finally:
                os.close(child_descriptor)
        elif stat.S_ISREG(observed.st_mode):
            selected = path_filter is None or path_filter(rendered)
            snapshot = _snapshot_posix_regular_file(
                directory_descriptor,
                entry.name,
                observed=observed,
                relative_path=rendered,
                hash_contents=False,
                label=label,
            )
            if selected:
                snapshots.append(snapshot)
        else:
            raise ValueError(
                f"{label} contains an unsupported filesystem entry: {rendered}"
            )
    if not _same_file_state(
        directory_before,
        os.fstat(directory_descriptor),
    ):
        rendered = PurePosixPath(*relative_parts).as_posix()
        raise ValueError(
            f"{label} directory changed during enumeration: {rendered}"
        )
    return snapshots


def _snapshot_posix_regular_file(
    directory_descriptor: int,
    name: str,
    *,
    observed: os.stat_result,
    relative_path: str,
    hash_contents: bool,
    label: str,
) -> ArtifactFileSnapshot:
    descriptor, opened = _open_posix_regular_file(
        directory_descriptor,
        name,
        label=label,
    )
    try:
        if not _same_file(observed, opened):
            raise ValueError(
                f"{label} file changed during enumeration: {relative_path}"
            )
        digest = _digest_descriptor(descriptor) if hash_contents else None
        if not _same_file_state(opened, os.fstat(descriptor)):
            raise ValueError(
                f"{label} file changed during enumeration: {relative_path}"
            )
        return ArtifactFileSnapshot(
            relative_path=relative_path,
            size_bytes=opened.st_size,
            sha256=digest,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=opened.st_mode,
            modified_ns=opened.st_mtime_ns,
            changed_ns=opened.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _snapshot_posix_regular_path(
    root_descriptor: int,
    candidate: ArtifactFileSnapshot,
    *,
    label: str,
) -> ArtifactFileSnapshot:
    parts = _relative_parts(candidate.relative_path, label=label)
    directory_descriptor = root_descriptor
    opened_directories: list[int] = []
    try:
        for component in parts[:-1]:
            directory_descriptor = _open_posix_relative_directory(
                directory_descriptor,
                component,
                label=label,
            )
            opened_directories.append(directory_descriptor)
        observed = os.stat(
            parts[-1],
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            _is_link_or_reparse(observed)
            or not stat.S_ISREG(observed.st_mode)
            or not _snapshot_matches_metadata(candidate, observed)
        ):
            raise ValueError(
                f"{label} file changed during enumeration: "
                f"{candidate.relative_path}"
            )
        return _snapshot_posix_regular_file(
            directory_descriptor,
            parts[-1],
            observed=observed,
            relative_path=candidate.relative_path,
            hash_contents=True,
            label=label,
        )
    finally:
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


def _validate_windows_directory_chain(path: Path, *, label: str) -> None:
    _windows_directory_snapshots(path, label=label)


def _windows_directory_snapshots(
    path: Path,
    *,
    label: str,
) -> tuple[tuple[Path, os.stat_result], ...]:
    current = Path(path.anchor)
    snapshots: list[tuple[Path, os.stat_result]] = []
    try:
        anchor_metadata = current.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if _is_link_or_reparse(anchor_metadata) or not stat.S_ISDIR(
        anchor_metadata.st_mode
    ):
        raise ValueError(
            f"{label} contains a symlink, reparse point, "
            f"or invalid directory: {current}"
        )
    snapshots.append((current, anchor_metadata))
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label} does not exist: {path}") from exc
        if _is_link_or_reparse(metadata):
            raise ValueError(
                f"{label} contains a symlink or reparse point: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(f"{label} is not a directory: {current}")
        snapshots.append((current, metadata))
    return tuple(snapshots)


def _read_windows_regular_file(
    root: Path,
    parts: tuple[str, ...],
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    descriptor, metadata, ancestors = _open_windows_regular_file(
        root,
        parts,
        label=label,
    )
    try:
        encoded = _read_descriptor(descriptor)
        if not _same_file_state(metadata, os.fstat(descriptor)):
            raise ValueError(f"{label} changed while it was being read")
        _validate_windows_ancestor_snapshots(ancestors, label=label)
        return encoded, metadata
    finally:
        os.close(descriptor)


def _open_windows_regular_file(
    root: Path,
    parts: tuple[str, ...],
    *,
    label: str,
) -> tuple[int, os.stat_result, tuple[tuple[Path, os.stat_result], ...]]:
    path = root.joinpath(*parts)
    current = root
    ancestors = list(_windows_directory_snapshots(root, label=label))
    for component in parts[:-1]:
        current /= component
        metadata = current.lstat()
        if _is_link_or_reparse(metadata):
            raise ValueError(
                f"{label} contains a symlink or reparse point: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise NotADirectoryError(
                f"{label} path component is not a directory: {current}"
            )
        ancestors.append((current, metadata))
    observed = path.lstat()
    if _is_link_or_reparse(observed):
        raise ValueError(f"{label} contains a symlink or reparse point: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        after = path.lstat()
        if (
            not _same_file(observed, opened)
            or not _same_file(opened, after)
        ):
            raise ValueError(f"{label} changed while it was being opened: {path}")
        _validate_windows_ancestor_snapshots(
            tuple(ancestors),
            label=label,
        )
        return descriptor, opened, tuple(ancestors)
    except BaseException:
        os.close(descriptor)
        raise


def _validate_windows_ancestor_snapshots(
    ancestors: tuple[tuple[Path, os.stat_result], ...],
    *,
    label: str,
) -> None:
    for path, observed in ancestors:
        current = path.lstat()
        if _is_link_or_reparse(current) or not _same_file(observed, current):
            raise ValueError(
                f"{label} ancestor changed or became linked: {path}"
            )


def _scan_windows_directory(
    root: Path,
    directory: Path,
    *,
    label: str,
    path_filter: Callable[[str], bool] | None,
) -> list[ArtifactFileSnapshot]:
    before = directory.lstat()
    if _is_link_or_reparse(before) or not stat.S_ISDIR(before.st_mode):
        raise ValueError(
            f"{label} contains a symlink, reparse point, "
            f"or non-directory: {directory}"
        )
    try:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as exc:
        raise ValueError(f"cannot enumerate {label}: {directory}") from exc
    snapshots: list[ArtifactFileSnapshot] = []
    for entry in entries:
        path = Path(entry.path)
        relative = path.relative_to(root).as_posix()
        observed = path.lstat()
        if _is_link_or_reparse(observed):
            raise ValueError(
                f"{label} must not contain links or reparse points: {path}"
            )
        if stat.S_ISDIR(observed.st_mode):
            snapshots.extend(
                _scan_windows_directory(
                    root,
                    path,
                    label=label,
                    path_filter=path_filter,
                )
            )
        elif stat.S_ISREG(observed.st_mode):
            selected = path_filter is None or path_filter(relative)
            snapshot = _snapshot_windows_regular_file(
                root,
                relative,
                observed=observed,
                hash_contents=False,
                label=label,
            )
            if selected:
                snapshots.append(snapshot)
        else:
            raise ValueError(
                f"{label} contains an unsupported filesystem entry: {path}"
            )
    after = directory.lstat()
    if not _same_file_state(before, after):
        raise ValueError(f"{label} directory changed during enumeration: {directory}")
    return snapshots


def _snapshot_windows_regular_file(
    root: Path,
    relative_path: str,
    *,
    observed: os.stat_result,
    hash_contents: bool,
    label: str,
) -> ArtifactFileSnapshot:
    parts = PurePosixPath(relative_path).parts
    path = root.joinpath(*parts)
    descriptor, opened, ancestors = _open_windows_regular_file(
        root,
        parts,
        label=label,
    )
    try:
        if not _same_file(observed, opened):
            raise ValueError(
                f"{label} file changed during enumeration: {path}"
            )
        digest = _digest_descriptor(descriptor) if hash_contents else None
        if not _same_file_state(opened, os.fstat(descriptor)):
            raise ValueError(
                f"{label} file changed during enumeration: {path}"
            )
        _validate_windows_ancestor_snapshots(
            ancestors,
            label=label,
        )
        return ArtifactFileSnapshot(
            relative_path=relative_path,
            size_bytes=opened.st_size,
            sha256=digest,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=opened.st_mode,
            modified_ns=opened.st_mtime_ns,
            changed_ns=opened.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _snapshot_windows_regular_path(
    root: Path,
    candidate: ArtifactFileSnapshot,
    *,
    label: str,
) -> ArtifactFileSnapshot:
    parts = _relative_parts(candidate.relative_path, label=label)
    path = root.joinpath(*parts)
    observed = path.lstat()
    if (
        _is_link_or_reparse(observed)
        or not stat.S_ISREG(observed.st_mode)
        or not _snapshot_matches_metadata(candidate, observed)
    ):
        raise ValueError(
            f"{label} file changed during enumeration: "
            f"{candidate.relative_path}"
        )
    return _snapshot_windows_regular_file(
        root,
        candidate.relative_path,
        observed=observed,
        hash_contents=True,
        label=label,
    )


__all__ = [
    "ArtifactFileSnapshot",
    "cache_entry_exists",
    "canonical_directory",
    "create_cache_directory",
    "create_cache_file_exclusive",
    "ensure_cache_directory",
    "lexical_absolute_path",
    "open_anchored_directory",
    "open_cache_file",
    "publish_cache_directory",
    "publish_cache_file",
    "quarantine_cache_entry",
    "read_regular_file",
    "remove_cache_directory",
    "remove_cache_file",
    "scan_regular_files",
    "write_cache_file",
]
