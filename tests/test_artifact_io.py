"""Failure-path tests for data artifact filesystem primitives."""

from __future__ import annotations

import errno
import hashlib
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from stochaflow.data import artifact_io


def test_scan_regular_files_hashes_independent_files_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        f"{index:02d}.bin": bytes([index]) * 8192
        for index in reversed(range(8))
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)

    original_digest = artifact_io._digest_descriptor
    lock = threading.Lock()
    active = 0
    max_active = 0

    def tracked_digest(descriptor: int) -> str:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return original_digest(descriptor)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        artifact_io,
        "_artifact_hash_worker_count",
        lambda _: 4,
    )
    monkeypatch.setattr(artifact_io, "_digest_descriptor", tracked_digest)

    snapshots = artifact_io.scan_regular_files(
        tmp_path,
        hash_contents=True,
        label="parallel hashing test",
    )

    assert max_active > 1
    assert [snapshot.relative_path for snapshot in snapshots] == sorted(payloads)
    assert {
        snapshot.relative_path: snapshot.sha256 for snapshot in snapshots
    } == {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in payloads.items()
    }


def test_scan_regular_files_propagates_parallel_hash_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(4):
        (tmp_path / f"{index}.bin").write_bytes(bytes([index]) * 8192)

    original_digest = artifact_io._digest_descriptor
    lock = threading.Lock()
    calls = 0

    def failing_digest(descriptor: int) -> str:
        nonlocal calls
        with lock:
            calls += 1
            should_fail = calls == 2
        if should_fail:
            raise OSError("parallel digest failed")
        return original_digest(descriptor)

    monkeypatch.setattr(
        artifact_io,
        "_artifact_hash_worker_count",
        lambda _: 4,
    )
    monkeypatch.setattr(artifact_io, "_digest_descriptor", failing_digest)

    with pytest.raises(OSError, match="parallel digest failed"):
        artifact_io.scan_regular_files(
            tmp_path,
            hash_contents=True,
            label="parallel hashing failure test",
        )


def test_scan_regular_files_bounds_outstanding_hash_tasks_and_reports_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(20):
        (tmp_path / f"{index:02d}.bin").write_bytes(bytes([index]) * 8192)

    original_executor = ThreadPoolExecutor
    original_digest = artifact_io._digest_descriptor
    lock = threading.Lock()
    outstanding = 0
    max_outstanding = 0
    configured_workers: int | None = None

    class TrackingExecutor:
        def __init__(
            self,
            max_workers: int | None = None,
            thread_name_prefix: str = "",
        ) -> None:
            nonlocal configured_workers
            configured_workers = max_workers
            self.executor = original_executor(
                max_workers=max_workers,
                thread_name_prefix=thread_name_prefix,
            )

        def submit(
            self,
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
        ) -> Future[Any]:
            nonlocal outstanding, max_outstanding
            with lock:
                outstanding += 1
                max_outstanding = max(max_outstanding, outstanding)
            future = self.executor.submit(function, *args, **kwargs)

            def completed(_: object) -> None:
                nonlocal outstanding
                with lock:
                    outstanding -= 1

            future.add_done_callback(completed)
            return future

        def shutdown(
            self,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            self.executor.shutdown(
                wait=wait,
                cancel_futures=cancel_futures,
            )

    def slow_digest(descriptor: int) -> str:
        time.sleep(0.02)
        return original_digest(descriptor)

    events: list[tuple[int, int, int]] = []
    caller_thread = threading.get_ident()
    monkeypatch.setattr(artifact_io, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(artifact_io, "_digest_descriptor", slow_digest)

    artifact_io.scan_regular_files(
        tmp_path,
        hash_contents=True,
        label="bounded hashing test",
        on_progress=lambda completed, total: events.append(
            (completed, total, threading.get_ident())
        ),
        workers=3,
    )

    assert 3 < max_outstanding <= 6
    assert configured_workers == 3
    assert [completed for completed, _, _ in events] == list(range(21))
    assert {total for _, total, _ in events} == {20}
    assert {thread_id for _, _, thread_id in events} == {caller_thread}


def test_parallel_hash_failure_is_reported_in_path_order() -> None:
    candidates = tuple(
        artifact_io.ArtifactFileSnapshot(
            relative_path=f"{index}.bin",
            size_bytes=1,
            sha256=None,
            device=1,
            inode=index + 1,
            mode=0,
            modified_ns=0,
            changed_ns=0,
        )
        for index in range(3)
    )

    def fail(candidate: artifact_io.ArtifactFileSnapshot):
        if candidate.relative_path == "0.bin":
            time.sleep(0.05)
            raise ValueError("failure at 0.bin")
        if candidate.relative_path == "1.bin":
            raise ValueError("failure at 1.bin")
        return candidate

    with pytest.raises(ValueError, match=r"failure at 0\.bin"):
        artifact_io._hash_regular_file_candidates(
            candidates,
            worker_count=3,
            hash_candidate=fail,
            on_progress=None,
        )


def test_complete_snapshot_metadata_ignores_only_content_digest() -> None:
    snapshot = artifact_io.ArtifactFileSnapshot(
        relative_path="data/payload.bin",
        size_bytes=16,
        sha256="a" * 64,
        device=1,
        inode=2,
        mode=0o100644,
        modified_ns=3,
        changed_ns=4,
    )

    assert artifact_io._same_complete_snapshot_metadata(
        (snapshot,),
        (replace(snapshot, sha256="b" * 64),),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", "data/replaced.bin"),
        ("size_bytes", 17),
        ("device", 5),
        ("inode", 6),
        ("mode", 0o100600),
        ("modified_ns", 7),
        ("changed_ns", 8),
    ],
)
def test_complete_snapshot_metadata_detects_every_metadata_change(
    field: str,
    value: object,
) -> None:
    snapshot = artifact_io.ArtifactFileSnapshot(
        relative_path="data/payload.bin",
        size_bytes=16,
        sha256="a" * 64,
        device=1,
        inode=2,
        mode=0o100644,
        modified_ns=3,
        changed_ns=4,
    )

    assert not artifact_io._same_complete_snapshot_metadata(
        (snapshot,),
        (replace(snapshot, **{field: value}),),  # type: ignore[arg-type]
    )


def test_complete_snapshot_metadata_detects_file_count_change() -> None:
    snapshot = artifact_io.ArtifactFileSnapshot(
        relative_path="data/payload.bin",
        size_bytes=16,
        sha256="a" * 64,
        device=1,
        inode=2,
        mode=0o100644,
        modified_ns=3,
        changed_ns=4,
    )

    assert not artifact_io._same_complete_snapshot_metadata((snapshot,), ())


def test_hash_progress_completes_after_consistency_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(2):
        (tmp_path / f"{index}.bin").write_bytes(bytes([index]))

    original_enumerate = artifact_io._enumerate_regular_files
    enumeration_count = 0
    events: list[tuple[int, int]] = []

    def tracked_enumerate(
        canonical_root: Path,
        *,
        label: str,
        path_filter: Callable[[str], bool] | None,
    ) -> tuple[artifact_io.ArtifactFileSnapshot, ...]:
        nonlocal enumeration_count
        enumeration_count += 1
        return original_enumerate(
            canonical_root,
            label=label,
            path_filter=path_filter,
        )

    monkeypatch.setattr(
        artifact_io,
        "_enumerate_regular_files",
        tracked_enumerate,
    )

    artifact_io.scan_regular_files(
        tmp_path,
        hash_contents=True,
        label="terminal progress test",
        on_progress=lambda completed, _: events.append(
            (completed, enumeration_count)
        ),
    )

    assert events == [(0, 1), (1, 1), (2, 2)]


@pytest.mark.parametrize(
    ("cpu_count", "expected"),
    [(None, 1), (1, 1), (4, 4), (64, 8)],
)
def test_artifact_hash_worker_count_is_bounded(
    cpu_count: int | None,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_io.os, "cpu_count", lambda: cpu_count)

    assert artifact_io._artifact_hash_worker_count() == expected


@pytest.mark.parametrize("requested_workers", [1, 3, 8])
def test_artifact_hash_worker_count_honors_explicit_value(
    requested_workers: int,
) -> None:
    assert (
        artifact_io._artifact_hash_worker_count(requested_workers)
        == requested_workers
    )


@pytest.mark.parametrize("requested_workers", [0, -1, True, 9, 10_000])
def test_artifact_hash_worker_count_rejects_invalid_explicit_value(
    requested_workers: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="workers"):
        artifact_io._artifact_hash_worker_count(requested_workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("entry_kind", ["directory", "file"])
def test_publication_closes_source_parent_when_destination_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    if os.name == "nt" or not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("descriptor-relative publication is unavailable")

    cache_root = tmp_path / "cache"
    staging_parent = cache_root / "staging"
    staging_parent.mkdir(parents=True)
    destination = cache_root / "objects" / "published"
    staging = staging_parent / "candidate"
    if entry_kind == "directory":
        staging.mkdir()
        publisher = artifact_io.publish_cache_directory
    else:
        staging.write_bytes(b"candidate")
        publisher = artifact_io.publish_cache_file

    original_open = artifact_io._open_posix_directory_chain
    source_descriptors: list[int] = []

    def failing_destination_open(path: Path, *, label: str) -> int:
        if label.endswith("destination parent"):
            raise PermissionError("destination parent is unavailable")
        descriptor = original_open(path, label=label)
        if label.endswith("staging parent"):
            source_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(
        artifact_io,
        "_open_posix_directory_chain",
        failing_destination_open,
    )

    with pytest.raises(PermissionError, match="destination parent"):
        publisher(
            cache_root,
            staging,
            destination,
            label="test publication",
        )

    descriptor = source_descriptors[-1]
    with pytest.raises(OSError, match="Bad file descriptor") as exc_info:
        os.fstat(descriptor)
    assert exc_info.value.errno == errno.EBADF
