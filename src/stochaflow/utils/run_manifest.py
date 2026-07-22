"""Shared serialization for extension provenance and invocation manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    ResolvedExtensions,
    extension_plugin_provenance_to_dicts,
)


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
    """Write one deterministic UTF-8 YAML manifest and return its path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return destination


__all__ = ["extension_runtime_metadata", "write_yaml_manifest"]
