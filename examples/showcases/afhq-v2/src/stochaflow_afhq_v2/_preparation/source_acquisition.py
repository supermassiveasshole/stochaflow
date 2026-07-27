"""Authenticate and content-address AFHQ-v2 source archives."""

from __future__ import annotations

import hashlib
import os
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from stochaflow.data.artifact_io import (
    cache_entry_exists,
    create_cache_file_exclusive,
    ensure_cache_directory,
    lexical_absolute_path,
    publish_cache_file,
    remove_cache_file,
)

from .contracts import (
    PreparationError,
    SourceArchive,
    SourceIntegrityError,
    SourceLock,
)
from .downloading import (
    _quarantine_invalid_download,
    download_official_archive,
)
from .locking import SourceAcquisitionLock
from .safe_file import (
    _open_regular_file_without_links,
    _regular_file_state,
    _require_open_file_path_identity,
    _sha256_stream,
    _write_descriptor,
)

_HASH_CHUNK_BYTES = 8 * 1024 * 1024

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
    with SourceAcquisitionLock(
        locks_root / "afhq-v2-download.lock",
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
