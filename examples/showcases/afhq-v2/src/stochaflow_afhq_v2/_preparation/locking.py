"""Narrow advisory lock for resumable AFHQ-v2 source acquisition."""

from __future__ import annotations

import errno
import importlib
import os
import time
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import Any, Self

from .contracts import PreparationError

_LOCK_API: Any = importlib.import_module("msvcrt" if os.name == "nt" else "fcntl")
_WINDOWS_LOCK_OFFSET = 4096


class SourceAcquisitionLock(AbstractContextManager["SourceAcquisitionLock"]):
    """Serialize downloads without participating in artifact publication."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 120.0,
        poll_seconds: float = 0.1,
    ) -> None:
        if timeout_seconds < 0:
            raise ValueError("source acquisition lock timeout must be non-negative")
        if poll_seconds <= 0:
            raise ValueError("source acquisition lock poll interval must be positive")
        self.path = path.absolute()
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.descriptor: int | None = None

    def __enter__(self) -> Self:
        descriptor: int | None = None
        acquired = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self.path, flags, 0o600)
            deadline = time.monotonic() + self.timeout_seconds
            while not _try_lock(descriptor):
                if time.monotonic() >= deadline:
                    raise PreparationError(
                        "timed out waiting for AFHQ-v2 source acquisition lock: "
                        f"{self.path}"
                    )
                time.sleep(
                    min(
                        self.poll_seconds,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
            acquired = True
            owner = f"pid={os.getpid()}\n".encode()
            os.lseek(descriptor, 0, os.SEEK_SET)
            _write_all(descriptor, owner)
            os.ftruncate(descriptor, len(owner))
            os.fsync(descriptor)
            self.descriptor = descriptor
        except BaseException as error:
            if descriptor is not None:
                if acquired:
                    with suppress(OSError):
                        _unlock(descriptor)
                os.close(descriptor)
            if isinstance(error, PreparationError):
                raise
            if isinstance(error, OSError):
                raise PreparationError(
                    f"cannot acquire AFHQ-v2 source download lock: {self.path}"
                ) from error
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        descriptor = self.descriptor
        if descriptor is None:
            return
        try:
            _unlock(descriptor)
        finally:
            os.close(descriptor)
            self.descriptor = None


def _try_lock(descriptor: int) -> bool:
    if os.name == "nt":
        os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            _LOCK_API.locking(descriptor, _LOCK_API.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True
    try:
        _LOCK_API.flock(descriptor, _LOCK_API.LOCK_EX | _LOCK_API.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _write_all(descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("cannot write source acquisition lock owner")
        remaining = remaining[written:]


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        _LOCK_API.locking(descriptor, _LOCK_API.LK_UNLCK, 1)
        return
    _LOCK_API.flock(descriptor, _LOCK_API.LOCK_UN)


__all__ = ["SourceAcquisitionLock"]
