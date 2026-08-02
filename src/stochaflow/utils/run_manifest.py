"""Shared serialization for extension provenance and invocation manifests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from stochaflow.utils.config import SampleConfig, StochaflowConfig
from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    ResolvedExtensions,
    extension_plugin_provenance_to_dicts,
)


def selected_training_component_identities(
    config: StochaflowConfig,
    *,
    inference_recipe: str | None = None,
) -> dict[str, str | list[str] | None]:
    """Summarize training-owned components without sample invocation values."""

    return {
        "data_builder": config.data.name,
        "model": config.model.name,
        "training_builder": config.training.name,
        "objective": config.objective.name if config.objective is not None else None,
        "process": config.process.name if config.process is not None else None,
        "optimizer": config.optimizer.name,
        "lr_scheduler": (
            config.lr_scheduler.name if config.lr_scheduler is not None else None
        ),
        "inference_recipe": inference_recipe,
        "loggers": [backend.name for backend in config.logging.backends],
        "diagnostics": [diagnostic.name for diagnostic in config.diagnostics],
        "metrics": [metric.name for metric in config.metrics],
    }


def selected_sampling_component_identities(
    config: StochaflowConfig,
    sample: SampleConfig,
    *,
    inference_recipe: str,
) -> dict[str, str | list[str] | None]:
    """Summarize checkpoint and sample authorities for one invocation."""

    return {
        "model": config.model.name,
        "process": config.process.name if config.process is not None else None,
        "inference_recipe": inference_recipe,
        "sampler": sample.sampler.name if sample.sampler is not None else None,
        "artifact_writers": [writer.name for writer in sample.writers],
    }


def _provenance_dict(value: ExtensionPluginProvenance) -> dict[str, str]:
    return extension_plugin_provenance_to_dicts((value,))[0]


def extension_runtime_metadata(
    extensions: ResolvedExtensions,
) -> dict[str, Any]:
    """Return the checkpoint-safe extension audit shared by all runtimes."""

    return {
        "extension_plugins": extension_plugin_provenance_to_dicts(
            extensions.provenance
        ),
        "extension_version_acceptance": [
            {
                "expected": _provenance_dict(item.expected),
                "current": _provenance_dict(item.current),
                "method": item.method,
            }
            for item in extensions.acceptance_audit
        ],
    }


def write_yaml_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically write one deterministic UTF-8 YAML manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


__all__ = [
    "extension_runtime_metadata",
    "selected_sampling_component_identities",
    "selected_training_component_identities",
    "write_yaml_manifest",
]
