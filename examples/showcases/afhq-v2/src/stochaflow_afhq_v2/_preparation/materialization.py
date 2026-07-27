"""Define the semantic AFHQ-v2 image materialization recipe."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

from PIL import __version__

from stochaflow.extensions import canonical_artifact_digest

from .contracts import PreparationError, SourceLock

AFHQV2_MATERIALIZER_NAME = "stochaflow.afhq-v2.rgb-lanczos-png"
_RECIPE_VERSION = 3
_PATH_NORMALIZATION_VERSION = 1
_PNG_COMPRESS_LEVEL = 6


@dataclass(frozen=True, slots=True)
class AFHQV2MaterializationSpec:
    """Semantic recipe and digest for one managed AFHQ-v2 artifact."""

    recipe: dict[str, Any]
    digest: str


def build_materialization_spec(
    *,
    lock: SourceLock,
    resolution: int,
) -> AFHQV2MaterializationSpec:
    """Build the source-independent preparation recipe."""

    if type(resolution) is not int:
        raise TypeError("resolution must be an integer")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if lock.expected_sha256 is None:
        raise PreparationError(
            "materialization requires a source lock with a pinned SHA-256"
        )
    if lock.contract.source_class_counts is None:
        raise PreparationError(
            "materialization requires pinned per-class source counts"
        )
    recipe: dict[str, Any] = {
        "schema_version": 1,
        "recipe": {
            "name": AFHQV2_MATERIALIZER_NAME,
            "version": _RECIPE_VERSION,
        },
        "partitions": {
            "source": ["train", "test"],
            "output": ["train", "test"],
            "path": "<partition>/<class>/<filename>",
            "classes": list(lock.contract.classes),
            "class_mapping": dict(lock.contract.class_mapping),
            "source_counts": {
                split: dict(counts)
                for split, counts in lock.contract.source_class_counts.items()
            },
        },
        "path_normalization": {
            "form": "Unicode NFC",
            "sort": "canonical POSIX relative path",
            "version": _PATH_NORMALIZATION_VERSION,
        },
        "decoder": {
            "library": "Pillow",
            "version": __version__,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "accepted_format": "PNG",
            "accepted_mode": "RGB",
            "accepted_size": [
                lock.contract.input_resolution,
                lock.contract.input_resolution,
            ],
            "exif_orientation": "identity-only",
            "metadata_policy": "discard",
        },
        "resize": {
            "input_size": [
                lock.contract.input_resolution,
                lock.contract.input_resolution,
            ],
            "output_size": [resolution, resolution],
            "crop": False,
            "resample": "PIL.Image.Resampling.LANCZOS",
            "reducing_gap": None,
        },
        "encoding": {
            "format": "PNG",
            "mode": "RGB",
            "bit_depth": 8,
            "optimize": False,
            "compress_level": _PNG_COMPRESS_LEVEL,
            "pnginfo": "empty",
        },
        "index": {
            "schema_version": 1,
            "path": "_index/images.json",
        },
    }
    return AFHQV2MaterializationSpec(
        recipe=recipe,
        digest=canonical_artifact_digest(recipe),
    )


__all__ = [
    "AFHQV2_MATERIALIZER_NAME",
    "AFHQV2MaterializationSpec",
    "build_materialization_spec",
]
