"""Define the pinned AFHQ-v2 Dog benchmark materialization recipe."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import __version__ as pillow_version

from stochaflow.extensions import canonical_artifact_digest

from .contracts import PreparationError, SourceLock

AFHQV2_DOG_MATERIALIZER_NAME = (
    "stochaflow.afhq-v2.dog.guided-diffusion-center-crop-png"
)
GUIDED_DIFFUSION_COMMIT = "8fb3ad9197f16bbc40620447b2742e13458d2831"
AFHQV2_DOG_RESOLUTION = 256
_RECIPE_VERSION = 1
_PATH_NORMALIZATION_VERSION = 1
_PNG_COMPRESS_LEVEL = 6


@dataclass(frozen=True, slots=True)
class AFHQV2DogMaterializationSpec:
    """Semantic recipe and digest for the AFHQ-v2 Dog benchmark artifact."""

    recipe: dict[str, Any]
    digest: str


def build_dog_materialization_spec(
    *,
    lock: SourceLock,
    resolution: int = AFHQV2_DOG_RESOLUTION,
) -> AFHQV2DogMaterializationSpec:
    """Build the pinned train/dog-only 256px preparation recipe."""

    if type(resolution) is not int:
        raise TypeError("resolution must be an integer")
    if resolution != AFHQV2_DOG_RESOLUTION:
        raise ValueError(
            f"AFHQ-v2 Dog resolution must be {AFHQV2_DOG_RESOLUTION}"
        )
    if lock.expected_sha256 is None:
        raise PreparationError(
            "materialization requires a source lock with a pinned SHA-256"
        )
    source_counts = lock.contract.source_class_counts
    if source_counts is None:
        raise PreparationError(
            "materialization requires pinned per-class source counts"
        )
    try:
        dog_count = source_counts["train"]["dog"]
    except KeyError as error:
        raise PreparationError(
            "materialization requires a pinned train/dog source count"
        ) from error
    recipe: dict[str, Any] = {
        "schema_version": 1,
        "recipe": {
            "name": AFHQV2_DOG_MATERIALIZER_NAME,
            "version": _RECIPE_VERSION,
        },
        "source_subset": {
            "partition": "train",
            "class": "dog",
            "count": dog_count,
        },
        "output": {
            "partition": "train",
            "path": "train/dog/<filename>",
            "class_labels": False,
        },
        "path_normalization": {
            "form": "Unicode NFC",
            "sort": "canonical POSIX relative path",
            "version": _PATH_NORMALIZATION_VERSION,
        },
        "decoder": {
            "library": "Pillow",
            "version": pillow_version,
            "numpy_version": np.__version__,
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
        "transform": {
            "upstream": {
                "repository": "openai/guided-diffusion",
                "commit": GUIDED_DIFFUSION_COMMIT,
                "file": "guided_diffusion/image_datasets.py",
                "function": "center_crop_arr",
            },
            "output_size": [resolution, resolution],
            "steps": [
                {
                    "condition": "min(width, height) >= 2 * output_resolution",
                    "size": "[width // 2, height // 2]",
                    "resample": "PIL.Image.Resampling.BOX",
                    "reducing_gap": None,
                },
                {
                    "scale": "output_resolution / min(width, height)",
                    "size": "[round(width * scale), round(height * scale)]",
                    "resample": "PIL.Image.Resampling.BICUBIC",
                    "reducing_gap": None,
                },
                {
                    "crop": "integer-centered",
                    "offset": [
                        "(height - output_resolution) // 2",
                        "(width - output_resolution) // 2",
                    ],
                    "size": [resolution, resolution],
                },
            ],
        },
        "augmentation": {
            "horizontal_flip": "runtime DataBuilder policy",
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
    return AFHQV2DogMaterializationSpec(
        recipe=recipe,
        digest=canonical_artifact_digest(recipe),
    )


__all__ = [
    "AFHQV2_DOG_MATERIALIZER_NAME",
    "AFHQV2_DOG_RESOLUTION",
    "GUIDED_DIFFUSION_COMMIT",
    "AFHQV2DogMaterializationSpec",
    "build_dog_materialization_spec",
]
