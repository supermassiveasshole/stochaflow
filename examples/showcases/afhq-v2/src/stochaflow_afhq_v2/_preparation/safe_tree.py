"""Race-resistant enumeration and validation of AFHQ-v2 artifact trees."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from stochaflow.data.artifact_io import open_anchored_directory

from .contracts import PreparationError
from .safe_file import (
    _is_reparse_or_symlink,
    _regular_file_state,
    _require_real_directory,
    _validate_relative_path,
)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _enumerate_regular_files_posix(
    root: Path,
    relative_root: PurePosixPath,
) -> set[str]:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if no_follow == 0 or os.scandir not in os.supports_fd:
        raise NotImplementedError
    descriptors: list[int] = []
    discovered: set[str] = set()

    def open_directory(
        name: str | Path,
        *,
        parent_descriptor: int | None = None,
    ) -> int:
        try:
            if parent_descriptor is None:
                descriptor = open_anchored_directory(
                    Path(name),
                    label="prepared artifact root",
                )
            else:
                descriptor = os.open(
                    name,
                    directory_flags | no_follow,
                    dir_fd=parent_descriptor,
                )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"prepared directory is missing or unsafe: {name}"
            ) from error
        descriptors.append(descriptor)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PreparationError(f"prepared path is not a directory: {name}")
        return descriptor

    def scan_directory(
        descriptor: int,
        relative_parts: tuple[str, ...],
    ) -> None:
        try:
            with os.scandir(descriptor) as entries:
                snapshot = list(entries)
        except OSError as error:
            raise PreparationError(
                "cannot enumerate prepared directory: "
                f"{PurePosixPath(*relative_parts)}"
            ) from error
        for entry in snapshot:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect prepared path: {entry.name}"
                ) from error
            if _is_reparse_or_symlink(metadata):
                raise PreparationError(
                    "prepared path must not be a symlink or reparse point: "
                    f"{entry.name}"
                )
            child_parts = (*relative_parts, entry.name)
            if stat.S_ISDIR(metadata.st_mode):
                child_descriptor = open_directory(
                    entry.name,
                    parent_descriptor=descriptor,
                )
                opened = os.fstat(child_descriptor)
                if _directory_identity(opened) != _directory_identity(metadata):
                    raise PreparationError(
                        "prepared directory changed while it was opened: "
                        f"{PurePosixPath(*child_parts)}"
                    )
                scan_directory(child_descriptor, child_parts)
            elif stat.S_ISREG(metadata.st_mode):
                discovered.add(PurePosixPath(*child_parts).as_posix())
            else:
                raise PreparationError(
                    "prepared path is neither a directory nor a regular file: "
                    f"{PurePosixPath(*child_parts)}"
                )

    try:
        descriptor = open_directory(root)
        relative_parts: tuple[str, ...] = ()
        for part in relative_root.parts:
            descriptor = open_directory(
                part,
                parent_descriptor=descriptor,
            )
            relative_parts = (*relative_parts, part)
        scan_directory(descriptor, relative_parts)
        return discovered
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _enumerate_regular_files_portable(
    root: Path,
    relative_root: PurePosixPath,
) -> set[str]:
    root_metadata = _require_real_directory(
        root,
        label="prepared artifact root",
    )
    ancestor_snapshots: list[tuple[Path, os.stat_result]] = [
        (root, root_metadata)
    ]
    split_root = root
    for part in relative_root.parts:
        split_root /= part
        ancestor_snapshots.append(
            (
                split_root,
                _require_real_directory(
                    split_root,
                    label=f"prepared split {relative_root}",
                ),
            )
        )
    discovered: set[str] = set()

    def recheck_ancestors(
        snapshots: Sequence[tuple[Path, os.stat_result]],
    ) -> None:
        for path, expected in snapshots:
            actual = _require_real_directory(path, label="prepared ancestor")
            if _directory_identity(actual) != _directory_identity(expected):
                raise PreparationError(
                    f"prepared ancestor changed while scanning: {path}"
                )

    def scan_directory(
        directory: Path,
        snapshots: list[tuple[Path, os.stat_result]],
    ) -> None:
        before = _require_real_directory(
            directory,
            label="prepared directory",
        )
        if _directory_identity(before) != _directory_identity(snapshots[-1][1]):
            raise PreparationError(
                f"prepared directory changed before scanning: {directory}"
            )
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise PreparationError(
                f"cannot enumerate prepared directory: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect prepared path: {path}"
                ) from error
            if _is_reparse_or_symlink(metadata):
                raise PreparationError(
                    f"prepared path must not be a symlink or reparse point: {path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                child_metadata = _require_real_directory(
                    path,
                    label="prepared directory",
                )
                if child_metadata.st_ino != entry.inode():
                    raise PreparationError(
                        f"prepared directory changed while it was inspected: {path}"
                    )
                scan_directory(
                    path,
                    [*snapshots, (path, child_metadata)],
                )
            elif stat.S_ISREG(metadata.st_mode):
                discovered.add(path.relative_to(root).as_posix())
            else:
                raise PreparationError(
                    f"prepared path is neither a directory nor a regular file: {path}"
                )
        recheck_ancestors(snapshots)

    scan_directory(split_root, ancestor_snapshots)
    return discovered


def _enumerate_regular_files_without_links(
    root: Path,
    relative_root: PurePosixPath,
) -> set[str]:
    _validate_relative_path(relative_root, label="prepared split")
    if os.name != "nt":
        try:
            return _enumerate_regular_files_posix(root, relative_root)
        except NotImplementedError:
            pass
    return _enumerate_regular_files_portable(root, relative_root)


def _validate_prepared_root_snapshot(
    entries: Mapping[str, os.stat_result],
) -> None:
    expected_files = {"dataset_manifest.yaml", "files.sha256"}
    expected_directories = {"train", "test"}
    expected = expected_files | expected_directories
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        raise PreparationError(
            "prepared artifact root has an unsafe or invalid layout; "
            f"missing={missing or '<none>'}, "
            f"unexpected={unexpected or '<none>'}"
        )
    for name, metadata in entries.items():
        if _is_reparse_or_symlink(metadata):
            raise PreparationError(
                "prepared artifact root entry must not be a symlink or "
                f"reparse point: {name}"
            )
        expected_mode = stat.S_ISREG if name in expected_files else stat.S_ISDIR
        if not expected_mode(metadata.st_mode):
            expected_kind = "regular file" if name in expected_files else "directory"
            raise PreparationError(
                f"prepared artifact root entry must be a {expected_kind}: {name}"
            )


def _validate_prepared_root_layout(root: Path) -> None:
    if os.name != "nt":
        try:
            descriptor = open_anchored_directory(
                root,
                label="prepared artifact root",
            )
        except NotImplementedError:
            pass
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot inspect prepared artifact root: {root}"
            ) from error
        else:
            try:
                before = os.fstat(descriptor)
                with os.scandir(descriptor) as iterator:
                    snapshot = {
                        entry.name: entry.stat(follow_symlinks=False)
                        for entry in iterator
                    }
                _validate_prepared_root_snapshot(snapshot)
                if _regular_file_state(os.fstat(descriptor)) != (
                    _regular_file_state(before)
                ):
                    raise PreparationError(
                        "prepared artifact root changed while it was inspected"
                    )
                return
            except OSError as error:
                raise PreparationError(
                    f"cannot inspect prepared artifact root: {root}"
                ) from error
            finally:
                os.close(descriptor)

    before = _require_real_directory(root, label="prepared artifact root")
    try:
        with os.scandir(root) as iterator:
            snapshot = {
                entry.name: entry.stat(follow_symlinks=False)
                for entry in iterator
            }
    except OSError as error:
        raise PreparationError(
            f"cannot inspect prepared artifact root: {root}"
        ) from error
    _validate_prepared_root_snapshot(snapshot)
    after = _require_real_directory(root, label="prepared artifact root")
    if _regular_file_state(after) != _regular_file_state(before):
        raise PreparationError(
            "prepared artifact root changed while it was inspected"
        )
