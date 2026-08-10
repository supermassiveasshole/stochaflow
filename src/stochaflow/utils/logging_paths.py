"""Path validation shared without importing logging backends."""

from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_local_log_path(
    output_dir: str | Path,
    filename: object,
    *,
    field: str,
) -> Path:
    """Resolve one local logger filename without permitting path escape."""

    if not isinstance(filename, str) or not filename:
        raise ValueError(f"local logger {field} must be a non-empty string")
    posix = PurePosixPath(filename)
    windows = PureWindowsPath(filename)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or len(posix.parts) != 1
        or len(windows.parts) != 1
        or filename in {".", ".."}
    ):
        raise ValueError(
            f"local logger {field} must be a filename within the run directory"
        )
    return Path(output_dir) / filename


__all__ = ["resolve_local_log_path"]
