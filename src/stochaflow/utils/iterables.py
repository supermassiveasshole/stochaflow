"""Shared iterable capability helpers."""

from __future__ import annotations

from collections.abc import Sized
from typing import cast


def try_length(value: object) -> int | None:
    """Return an object's usable length, or ``None`` when it has no length."""

    try:
        return len(cast(Sized, value))
    except TypeError:
        return None


__all__ = ["try_length"]
