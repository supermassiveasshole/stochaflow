"""Race-resistant access to individual AFHQ-v2 files."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from stochaflow.data.artifact_io import (
    canonical_directory,
    lexical_absolute_path,
    open_anchored_directory,
)

from .contracts import PreparationError

_HASH_CHUNK_BYTES = 8 * 1024 * 1024

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


def load_verified_prepared_image(
    root: Path,
    relative_path: str,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_width: int,
    expected_height: int,
) -> Image.Image:
    """Load one prepared RGB PNG while rechecking its authenticated record."""

    path = PurePosixPath(relative_path)
    payload, metadata = _read_regular_file_without_links(
        root,
        path,
        label="prepared AFHQ-v2 image",
    )
    assert payload is not None
    if metadata.st_size != expected_size_bytes or len(payload) != expected_size_bytes:
        raise PreparationError(
            f"prepared image size changed: {relative_path!r}"
        )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PreparationError(
            f"prepared image content changed: {relative_path!r}"
        )
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if source.format != "PNG":
                raise PreparationError(
                    f"prepared image is not PNG: {relative_path!r}"
                )
            if source.mode != "RGB":
                raise PreparationError(
                    f"prepared image is not RGB: {relative_path!r}"
                )
            if source.size != (expected_width, expected_height):
                raise PreparationError(
                    f"prepared image dimensions changed: {relative_path!r}"
                )
            return source.copy()
    except PreparationError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise PreparationError(
            f"cannot decode prepared image: {relative_path!r}"
        ) from error
