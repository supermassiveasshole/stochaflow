"""Hold one authenticated AFHQ-v2 source descriptor during preparation."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import BinaryIO, Self

from .contracts import SourceArchive, SourceIntegrityError, SourceLock
from .safe_file import (
    _open_regular_file_without_links,
    _regular_file_state,
    _require_open_file_path_identity,
    _sha256_stream,
)


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
