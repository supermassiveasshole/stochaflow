"""Strict positional-alignment sidecars for observation/reference pairs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AlignedSource:
    path: Path
    trajectory_range: tuple[int, int]
    shape: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Alignment:
    observation: AlignedSource
    reference: AlignedSource
    sample_count: int


def _relative_path(path: Path, *, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def write_alignment(
    path: Path,
    *,
    observation_path: Path,
    observation_range: tuple[int, int],
    observation_shape: tuple[int, int, int, int],
    reference_path: Path,
    reference_range: tuple[int, int],
    reference_shape: tuple[int, int, int, int],
    sample_count: int,
) -> None:
    """Atomically write the positional pair contract used at sampling time."""

    payload = {
        "format_version": 1,
        "pairing": "trajectory-major-consecutive-triplets",
        "sample_count": sample_count,
        "observation": {
            "path": _relative_path(observation_path, parent=path.parent),
            "trajectory_range": list(observation_range),
            "shape": list(observation_shape),
        },
        "reference": {
            "path": _relative_path(reference_path, parent=path.parent),
            "trajectory_range": list(reference_range),
            "shape": list(reference_shape),
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source(raw: object, *, sidecar: Path, label: str) -> AlignedSource:
    if not isinstance(raw, dict):
        raise TypeError(f"alignment {label} must be a mapping")
    if set(raw) != {"path", "trajectory_range", "shape"}:
        raise ValueError(f"alignment {label} fields are invalid")
    raw_path = raw["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"alignment {label}.path must be a non-empty string")
    raw_range = raw["trajectory_range"]
    if (
        not isinstance(raw_range, list)
        or len(raw_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_range)
    ):
        raise ValueError(f"alignment {label}.trajectory_range is invalid")
    start, stop = raw_range
    if start < 0 or stop <= start:
        raise ValueError(f"alignment {label}.trajectory_range is invalid")
    raw_shape = raw["shape"]
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw_shape)
    ):
        raise ValueError(f"alignment {label}.shape is invalid")
    return AlignedSource(
        (sidecar.parent / raw_path).resolve(),
        (start, stop),
        tuple(raw_shape),
    )


def load_alignment(path: Path) -> Alignment:
    """Load and strictly validate one positional-alignment sidecar."""

    if not path.is_file():
        raise FileNotFoundError(f"alignment sidecar does not exist: {path}")
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("alignment sidecar root must be a mapping")
    if set(raw) != {
        "format_version",
        "pairing",
        "sample_count",
        "observation",
        "reference",
    }:
        raise ValueError("alignment sidecar fields are invalid")
    if raw["format_version"] != 1:
        raise ValueError("alignment format_version must be 1")
    if raw["pairing"] != "trajectory-major-consecutive-triplets":
        raise ValueError("alignment pairing policy is unsupported")
    sample_count = raw["sample_count"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("alignment sample_count must be positive")
    return Alignment(
        observation=_source(raw["observation"], sidecar=path, label="observation"),
        reference=_source(raw["reference"], sidecar=path, label="reference"),
        sample_count=sample_count,
    )


__all__ = ["AlignedSource", "Alignment", "load_alignment", "write_alignment"]
