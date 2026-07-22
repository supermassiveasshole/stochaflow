"""Strict private parameter parsing for the reference extension."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")


def copied_mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    result: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"{path} keys must be non-empty strings")
        result[raw_key] = deepcopy(raw_value)
    return result


def reject_unknown(params: Mapping[str, Any], *, path: str) -> None:
    if params:
        raise ValueError(f"unknown {path} parameter(s): {', '.join(sorted(params))}")


def pop_string(
    params: dict[str, Any],
    name: str,
    *,
    path: str,
    default: str | None = None,
) -> str:
    value = params.pop(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{name} must be a non-empty string")
    return value


def pop_path(params: dict[str, Any], name: str, *, path: str) -> Path:
    return Path(pop_string(params, name, path=path))


def pop_bool(
    params: dict[str, Any],
    name: str,
    *,
    path: str,
    default: bool,
) -> bool:
    value = params.pop(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{path}.{name} must be boolean")
    return value


def pop_int(
    params: dict[str, Any],
    name: str,
    *,
    path: str,
    default: int | None = None,
    minimum: int = 1,
) -> int:
    value = params.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}.{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{path}.{name} must be at least {minimum}")
    return value


def pop_float(
    params: dict[str, Any],
    name: str,
    *,
    path: str,
    default: float,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    value = params.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}.{name} must be numeric")
    result = float(value)
    if strictly_positive and result <= 0:
        raise ValueError(f"{path}.{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path}.{name} must be at least {minimum}")
    return result


def pop_optional_range(
    params: dict[str, Any],
    name: str,
    *,
    path: str,
) -> tuple[int, int] | None:
    value = params.pop(name, None)
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}.{name} must be null or [start, stop]")
    if len(value) != 2:
        raise ValueError(f"{path}.{name} must contain exactly start and stop")
    start, stop = cast(Sequence[object], value)
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
    ):
        raise TypeError(f"{path}.{name} bounds must be integers")
    if start < 0 or stop <= start:
        raise ValueError(f"{path}.{name} must satisfy 0 <= start < stop")
    return start, stop


def required_mapping(
    params: dict[str, Any], name: str, *, path: str
) -> dict[str, Any]:
    if name not in params:
        raise ValueError(f"{path}.{name} is required")
    return copied_mapping(params.pop(name), path=f"{path}.{name}")


def optional_mapping(
    params: dict[str, Any], name: str, *, path: str
) -> dict[str, Any] | None:
    value = params.pop(name, None)
    if value is None:
        return None
    return copied_mapping(value, path=f"{path}.{name}")


__all__ = [
    "copied_mapping",
    "optional_mapping",
    "pop_bool",
    "pop_float",
    "pop_int",
    "pop_optional_range",
    "pop_path",
    "pop_string",
    "reject_unknown",
    "required_mapping",
]
