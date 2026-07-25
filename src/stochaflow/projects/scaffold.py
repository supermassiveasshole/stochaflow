"""Deterministic, package-manager-neutral extension project scaffolding."""

from __future__ import annotations

import ctypes
import keyword
import os
import re
import stat
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata, resources
from pathlib import Path, PurePosixPath
from typing import Final, cast

from packaging.version import InvalidVersion, Version


class ProjectScaffoldError(ValueError):
    """Raised when an extension project cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class _TemplateFile:
    resource_name: str
    output_path: str


@dataclass(frozen=True, slots=True)
class _RenderedFile:
    relative_path: PurePosixPath
    content: str


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _AnchoredEntry:
    parent: tuple[str, ...]
    name: str
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _AnchoredFile:
    entry: _AnchoredEntry
    identity_descriptor: int


_PROJECT_NAME_PATTERN: Final = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
    flags=re.ASCII,
)
_WINDOWS_RESERVED_NAMES: Final = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_RESERVED_PROJECT_NAMES: Final = _WINDOWS_RESERVED_NAMES | {"stochaflow"}

_PROJECT_SLUG_SENTINEL: Final = "__STOCHAFLOW_PROJECT_SLUG__"
_PACKAGE_NAME_SENTINEL: Final = "__STOCHAFLOW_PACKAGE_NAME__"
_VERSION_SENTINEL: Final = "__STOCHAFLOW_VERSION__"
_SENTINEL_PREFIX: Final = "__STOCHAFLOW_"

# This manifest is the sole mapping from package resources to generated paths.
# Dotfiles are outputs, never package-data resource names.
_TEMPLATE_MANIFEST: Final = (
    _TemplateFile("pyproject.toml.tmpl", "pyproject.toml"),
    _TemplateFile("gitignore.tmpl", ".gitignore"),
    _TemplateFile("README.md.tmpl", "README.md"),
    _TemplateFile("data.gitkeep.tmpl", "data/.gitkeep"),
    _TemplateFile("notebooks.gitkeep.tmpl", "notebooks/.gitkeep"),
    _TemplateFile("train.yaml.tmpl", "experiments/example/train.yaml"),
    _TemplateFile(
        "package-init.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/__init__.py",
    ),
    _TemplateFile(
        "extension-init.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/stochaflow_ext/__init__.py",
    ),
    _TemplateFile(
        "data.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/stochaflow_ext/data.py",
    ),
    _TemplateFile(
        "model.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/stochaflow_ext/model.py",
    ),
    _TemplateFile(
        "diagnostics.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/stochaflow_ext/diagnostics.py",
    ),
    _TemplateFile(
        "training.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/stochaflow_ext/training.py",
    ),
    _TemplateFile(
        "sampling.py.tmpl",
        f"src/{_PACKAGE_NAME_SENTINEL}/stochaflow_ext/sampling.py",
    ),
    _TemplateFile("test-extensions.py.tmpl", "tests/test_extensions.py"),
)


def validate_project_name(name: str) -> str:
    """Validate and return an already-canonical extension project slug."""

    name_value = cast(object, name)
    if not isinstance(name_value, str):
        raise ProjectScaffoldError("project name must be a string")
    name = name_value
    if len(name) > 64:
        raise ProjectScaffoldError("project name must contain at most 64 characters")
    if not _PROJECT_NAME_PATTERN.fullmatch(name):
        raise ProjectScaffoldError(
            "project name must match "
            "[a-z][a-z0-9]*(?:-[a-z0-9]+)* using ASCII characters"
        )
    package_name = name.replace("-", "_")
    if keyword.iskeyword(package_name) or keyword.issoftkeyword(package_name):
        raise ProjectScaffoldError(
            f"project name '{name}' maps to a reserved Python keyword"
        )
    if name in _RESERVED_PROJECT_NAMES:
        raise ProjectScaffoldError(f"project name '{name}' is reserved")
    return name


def _current_stochaflow_version() -> str:
    try:
        raw_version = metadata.version("stochaflow")
    except metadata.PackageNotFoundError as exc:
        raise ProjectScaffoldError(
            "cannot determine the installed Stochaflow version"
        ) from exc
    try:
        return str(Version(raw_version))
    except InvalidVersion as exc:
        raise ProjectScaffoldError(
            f"installed Stochaflow version is invalid: {raw_version!r}"
        ) from exc


def _render_template(content: str, replacements: dict[str, str]) -> str:
    rendered = content
    for sentinel, value in replacements.items():
        rendered = rendered.replace(sentinel, value)
    if _SENTINEL_PREFIX in rendered:
        raise ProjectScaffoldError("template contains an unresolved sentinel")
    return rendered


def _validated_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ProjectScaffoldError(f"invalid template output path: {value!r}")
    return path


def _render_project(name: str) -> tuple[_RenderedFile, ...]:
    package_name = name.replace("-", "_")
    replacements = {
        _PROJECT_SLUG_SENTINEL: name,
        _PACKAGE_NAME_SENTINEL: package_name,
        _VERSION_SENTINEL: _current_stochaflow_version(),
    }
    template_root = resources.files("stochaflow.projects").joinpath("templates")
    rendered: list[_RenderedFile] = []
    seen_paths: set[PurePosixPath] = set()
    declared_resources: set[str] = set()
    for template in _TEMPLATE_MANIFEST:
        if template.resource_name in declared_resources:
            raise ProjectScaffoldError(
                f"duplicate template resource: {template.resource_name}"
            )
        declared_resources.add(template.resource_name)
        relative_path = _validated_relative_path(
            _render_template(template.output_path, replacements)
        )
        if relative_path in seen_paths:
            raise ProjectScaffoldError(
                f"duplicate template output path: {relative_path.as_posix()}"
            )
        seen_paths.add(relative_path)
        try:
            content = template_root.joinpath(template.resource_name).read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, UnicodeDecodeError) as exc:
            raise ProjectScaffoldError(
                f"cannot read project template resource '{template.resource_name}'"
            ) from exc
        rendered.append(
            _RenderedFile(
                relative_path=relative_path,
                content=_render_template(content, replacements),
            )
        )
    _validate_output_topology(rendered)
    return tuple(rendered)


def _validate_output_topology(files: list[_RenderedFile]) -> None:
    file_paths = {item.relative_path for item in files}
    for file_path in file_paths:
        for parent in file_path.parents:
            if parent == PurePosixPath("."):
                break
            if parent in file_paths:
                raise ProjectScaffoldError(
                    "template output path is both a file and directory: "
                    f"{parent.as_posix()}"
                )


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _exclusive_write_text(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_new_tree(root: Path, files: tuple[_RenderedFile, ...]) -> None:
    for rendered in files:
        destination = root.joinpath(*rendered.relative_path.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _exclusive_write_text(destination, rendered.content)


def _raise_rename_error(source: Path, destination: Path) -> None:
    error_number = ctypes.get_errno()
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source} -> {destination}",
    )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing an entry created by another actor."""

    if os.name == "nt":
        # Windows rename is already no-clobber and raises FileExistsError
        # whenever the destination exists.
        source.rename(destination)
        return
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        if rename_exclusive(source_bytes, destination_bytes, 0x00000004) != 0:
            _raise_rename_error(source, destination)
        return
    if sys.platform.startswith("linux"):
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as exc:
            raise ProjectScaffoldError(
                "this Linux runtime does not provide atomic no-clobber rename"
            ) from exc
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        if rename_no_replace(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        ) != 0:
            _raise_rename_error(source, destination)
        return
    raise ProjectScaffoldError(
        f"atomic no-clobber project publication is unsupported on {sys.platform}"
    )


def _ordinary_directory_mode(staging: Path) -> int:
    """Observe normal mkdir/umask permissions without mutating process umask."""

    probe = staging / ".stochaflow-mode-probe"
    probe.mkdir()
    try:
        return stat.S_IMODE(probe.stat(follow_symlinks=False).st_mode)
    finally:
        probe.rmdir()


def _new_staging_directory(
    target: Path,
) -> tuple[Path, int, _DirectoryIdentity]:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.stochaflow-init-",
            dir=target.parent,
        )
    )
    identity: _DirectoryIdentity | None = None
    try:
        identity = _real_directory_identity(staging)
        return staging, _ordinary_directory_mode(staging), identity
    except BaseException:
        if identity is not None:
            _remove_owned_staging(staging, identity)
        raise


def _real_directory_identity(path: Path) -> _DirectoryIdentity:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ProjectScaffoldError(f"target directory disappeared: {path}") from exc
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ProjectScaffoldError(
            f"target must be an empty real directory or not exist: {path}"
        )
    return _DirectoryIdentity(
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
    )


def _same_directory(path: Path, identity: _DirectoryIdentity) -> bool:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(path_stat.st_mode) and (
        path_stat.st_dev,
        path_stat.st_ino,
    ) == (identity.device, identity.inode)


def _supports_anchored_publication() -> bool:
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(
            operation in os.supports_dir_fd
            for operation in (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
        )
    )


def _open_existing_directory(
    target: Path,
    expected: _DirectoryIdentity,
) -> int:
    if not _supports_anchored_publication():
        raise ProjectScaffoldError(
            "secure publication into a pre-existing empty directory is not "
            "supported on this platform; remove the directory and retry"
        )
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise ProjectScaffoldError(
            "target changed while the project was being created"
        ) from exc
    opened_stat = os.fstat(descriptor)
    if (opened_stat.st_dev, opened_stat.st_ino) != (
        expected.device,
        expected.inode,
    ):
        os.close(descriptor)
        raise ProjectScaffoldError(
            "target changed while the project was being created"
        )
    with os.scandir(descriptor) as entries:
        if next(entries, None) is not None:
            os.close(descriptor)
            raise ProjectScaffoldError(f"target directory is not empty: {target}")
    return descriptor


def _exclusive_write_at(
    parent_descriptor: int,
    name: str,
    content: str,
    *,
    parent: tuple[str, ...],
) -> _AnchoredFile:
    identity_descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o666,
        dir_fd=parent_descriptor,
    )
    entry_stat = os.fstat(identity_descriptor)
    entry = _AnchoredEntry(
        parent=parent,
        name=name,
        device=entry_stat.st_dev,
        inode=entry_stat.st_ino,
    )
    anchored_file = _AnchoredFile(
        entry=entry,
        identity_descriptor=identity_descriptor,
    )
    write_descriptor: int | None = None
    try:
        write_descriptor = os.dup(identity_descriptor)
        with os.fdopen(
            write_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(content)
        write_descriptor = None
    except BaseException:
        if write_descriptor is not None:
            with suppress(OSError):
                os.close(write_descriptor)
        try:
            if _entry_matches(parent_descriptor, entry):
                os.unlink(name, dir_fd=parent_descriptor)
        except OSError:
            pass
        os.close(identity_descriptor)
        raise
    return anchored_file


def _entry_matches(
    parent_descriptor: int,
    entry: _AnchoredEntry,
) -> bool:
    try:
        entry_stat = os.stat(
            entry.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return (entry_stat.st_dev, entry_stat.st_ino) == (
        entry.device,
        entry.inode,
    )


def _clear_directory_descriptor(descriptor: int) -> None:
    """Remove entries through one pinned directory without following replacements."""

    with os.scandir(descriptor) as entries:
        names = tuple(entry.name for entry in entries)
    for name in names:
        try:
            entry_stat = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        entry = _AnchoredEntry(
            parent=(),
            name=name,
            device=entry_stat.st_dev,
            inode=entry_stat.st_ino,
        )
        if stat.S_ISDIR(entry_stat.st_mode):
            try:
                child_descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except OSError:
                continue
            try:
                child_stat = os.fstat(child_descriptor)
                if (child_stat.st_dev, child_stat.st_ino) != (
                    entry.device,
                    entry.inode,
                ):
                    continue
                _clear_directory_descriptor(child_descriptor)
            finally:
                os.close(child_descriptor)
            if _entry_matches(descriptor, entry):
                with suppress(OSError):
                    os.rmdir(name, dir_fd=descriptor)
            continue
        if _entry_matches(descriptor, entry):
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=descriptor)


def _remove_owned_staging(
    staging: Path,
    identity: _DirectoryIdentity,
) -> None:
    """Best-effort cleanup without recursively traversing a reused pathname."""

    if not _supports_anchored_publication():
        # Without descriptor-relative primitives there is no safe way to recurse
        # after another actor can reuse the staging name. Never risk deleting the
        # replacement; an exceptional failure may leave the private staging tree.
        if _same_directory(staging, identity):
            with suppress(OSError):
                staging.rmdir()
        return
    try:
        descriptor = os.open(
            staging,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError:
        return
    try:
        opened_stat = os.fstat(descriptor)
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            identity.device,
            identity.inode,
        ):
            return
        _clear_directory_descriptor(descriptor)
    finally:
        os.close(descriptor)
    if _same_directory(staging, identity):
        with suppress(OSError):
            staging.rmdir()


def _expected_children(
    files: tuple[_RenderedFile, ...],
) -> dict[tuple[str, ...], set[str]]:
    result: dict[tuple[str, ...], set[str]] = {}
    for rendered in files:
        parts = rendered.relative_path.parts
        for index, name in enumerate(parts):
            result.setdefault(parts[:index], set()).add(name)
    return result


def _publish_to_existing_empty_directory(
    target: Path,
    files: tuple[_RenderedFile, ...],
) -> None:
    expected = _real_directory_identity(target)
    root_descriptor = _open_existing_directory(target, expected)
    directory_descriptors: dict[tuple[str, ...], int] = {(): root_descriptor}
    created_directories: list[_AnchoredEntry] = []
    created_files: list[_AnchoredFile] = []
    try:
        for rendered in files:
            parts = rendered.relative_path.parts
            for index, directory_name in enumerate(parts[:-1]):
                directory_parts = parts[: index + 1]
                if directory_parts in directory_descriptors:
                    continue
                parent = parts[:index]
                parent_descriptor = directory_descriptors[parent]
                os.mkdir(directory_name, dir_fd=parent_descriptor)
                created_stat = os.stat(
                    directory_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                created_entry = _AnchoredEntry(
                    parent=parent,
                    name=directory_name,
                    device=created_stat.st_dev,
                    inode=created_stat.st_ino,
                )
                created_directories.append(created_entry)
                descriptor = os.open(
                    directory_name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_descriptor,
                )
                directory_stat = os.fstat(descriptor)
                if (directory_stat.st_dev, directory_stat.st_ino) != (
                    created_entry.device,
                    created_entry.inode,
                ):
                    os.close(descriptor)
                    raise ProjectScaffoldError(
                        "target directory changed while the project was being created"
                    )
                directory_descriptors[directory_parts] = descriptor
            parent = parts[:-1]
            created_files.append(
                _exclusive_write_at(
                    directory_descriptors[parent],
                    parts[-1],
                    rendered.content,
                    parent=parent,
                )
            )

        for anchored_file in created_files:
            entry = anchored_file.entry
            if not _entry_matches(
                directory_descriptors[entry.parent],
                entry,
            ):
                raise ProjectScaffoldError(
                    "target directory changed while the project was being created"
                )
        for entry in created_directories:
            if not _entry_matches(
                directory_descriptors[entry.parent],
                entry,
            ):
                raise ProjectScaffoldError(
                    "target directory changed while the project was being created"
                )
        for parent, expected_names in _expected_children(files).items():
            with os.scandir(directory_descriptors[parent]) as entries:
                if {entry.name for entry in entries} != expected_names:
                    raise ProjectScaffoldError(
                        "target directory changed while the project was being created"
                    )
        if not _same_directory(target, expected):
            raise ProjectScaffoldError(
                "target changed while the project was being created"
            )
    except BaseException:
        for anchored_file in reversed(created_files):
            entry = anchored_file.entry
            parent_descriptor = directory_descriptors[entry.parent]
            if _entry_matches(parent_descriptor, entry):
                os.unlink(entry.name, dir_fd=parent_descriptor)
        for entry in reversed(created_directories):
            parent_descriptor = directory_descriptors[entry.parent]
            if _entry_matches(parent_descriptor, entry):
                with suppress(OSError):
                    os.rmdir(entry.name, dir_fd=parent_descriptor)
        raise
    finally:
        for anchored_file in created_files:
            os.close(anchored_file.identity_descriptor)
        for parts, descriptor in reversed(tuple(directory_descriptors.items())):
            del parts
            os.close(descriptor)


def _publish_to_absent_target(
    target: Path,
    files: tuple[_RenderedFile, ...],
) -> None:
    staging, publish_mode, staging_identity = _new_staging_directory(target)
    published = False
    try:
        _write_new_tree(staging, files)
        staging.chmod(publish_mode)
        try:
            _rename_no_replace(staging, target)
        except FileExistsError as exc:
            raise ProjectScaffoldError(
                f"target appeared during creation: {target}"
            ) from exc
        except OSError as exc:
            raise ProjectScaffoldError(
                f"could not publish project at {target}: {exc}"
            ) from exc
        published = True
    finally:
        if not published:
            _remove_owned_staging(staging, staging_identity)


def create_project(
    name: str,
    *,
    cwd: str | Path | None = None,
) -> Path:
    """Create one extension project below ``cwd`` without changing process state."""

    project_name = validate_project_name(name)
    root = Path.cwd() if cwd is None else Path(cwd)
    if not root.exists() or not root.is_dir():
        raise ProjectScaffoldError(
            f"project parent must be an existing directory: {root}"
        )
    root = root.resolve()
    target = root / project_name

    files = _render_project(project_name)
    try:
        if _path_exists(target):
            if target.is_symlink() or not target.is_dir():
                raise ProjectScaffoldError(
                    f"target must be an empty real directory or not exist: {target}"
                )
            if any(target.iterdir()):
                raise ProjectScaffoldError(
                    f"target directory is not empty: {target}"
                )
            _publish_to_existing_empty_directory(target, files)
        else:
            _publish_to_absent_target(target, files)
    except ProjectScaffoldError:
        raise
    except OSError as exc:
        raise ProjectScaffoldError(
            f"could not create project at {target}: {exc}"
        ) from exc
    return target


__all__ = [
    "ProjectScaffoldError",
    "create_project",
    "validate_project_name",
]
