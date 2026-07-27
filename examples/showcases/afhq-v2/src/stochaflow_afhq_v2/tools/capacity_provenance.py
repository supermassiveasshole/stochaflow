"""Environment and code provenance for AFHQ-v2 capacity reports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

import stochaflow
from stochaflow.utils.config import StochaflowConfig


def canonical_json_bytes(value: object) -> bytes:
    """Encode a deterministic JSON representation for identity hashes."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def distribution_version(name: str) -> str | None:
    """Return an installed distribution version when available."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def code_tree_sha256(root: Path) -> str:
    """Hash the paths and contents of every Python source under a package."""

    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    ]
    return canonical_sha256(records)


def code_identity() -> dict[str, Any]:
    """Describe the exact core and extension Python trees used by a report."""

    core_root = Path(stochaflow.__file__).resolve().parent
    extension_root = Path(__file__).resolve().parents[1]
    return {
        "core": {
            "distribution": "stochaflow",
            "version": distribution_version("stochaflow"),
            "package_root": str(core_root),
            "python_tree_sha256": code_tree_sha256(core_root),
        },
        "extension": {
            "plugin_name": "stochaflow-afhq-v2",
            "distribution": "stochaflow-afhq-v2",
            "version": distribution_version("stochaflow-afhq-v2"),
            "entry_point": "stochaflow_afhq_v2.stochaflow_ext",
            "package_root": str(extension_root),
            "python_tree_sha256": code_tree_sha256(extension_root),
        },
    }


def trial_config_identity(config: StochaflowConfig) -> dict[str, Any]:
    """Capture one resolved trial configuration and its canonical identity."""

    snapshot = config.to_dict()
    return {
        "seed": config.experiment.seed,
        "output_dir": str(Path(config.experiment.output_dir).resolve()),
        "resolved_config": snapshot,
        "resolved_config_sha256": canonical_sha256(snapshot),
    }


def environment_report(device: torch.device) -> dict[str, Any]:
    """Describe the execution environment relevant to capacity measurements."""

    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["cuda_device"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        }
    else:
        report["cuda_device"] = None
    return report


def normalize_non_finite_floats(value: Any) -> Any:
    """Replace non-finite floats recursively so reports remain strict JSON."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            key: normalize_non_finite_floats(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_non_finite_floats(item) for item in value]
    return value


__all__ = [
    "canonical_json_bytes",
    "canonical_sha256",
    "code_identity",
    "code_tree_sha256",
    "distribution_version",
    "environment_report",
    "normalize_non_finite_floats",
    "trial_config_identity",
]
