"""Failure-path tests for data artifact filesystem primitives."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from stochaflow.data import artifact_io


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
