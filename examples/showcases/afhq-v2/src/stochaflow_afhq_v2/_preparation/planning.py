"""Derive deterministic AFHQ-v2 preparation identities and expected counts."""

from __future__ import annotations

import zlib

from PIL import __version__

from .contracts import PreparationError, PreparationPlan, SourceLock
from .identity import _canonical_digest

_RECIPE_ID = "stochaflow.afhq-v2.rgb-lanczos-png"
_RECIPE_VERSION = 2
_PATH_NORMALIZATION_VERSION = 1
_PNG_COMPRESS_LEVEL = 6


def _transform_recipe(
    *,
    input_resolution: int,
    output_resolution: int,
) -> dict[str, object]:
    return {
        "id": _RECIPE_ID,
        "version": _RECIPE_VERSION,
        "path_normalization": {
            "form": "Unicode NFC",
            "sort": "canonical POSIX relative path",
            "version": _PATH_NORMALIZATION_VERSION,
        },
        "decoder": {
            "library": "Pillow",
            "version": __version__,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "open_sequence": "open-verify-reopen-convert",
            "accepted_format": "PNG",
            "accepted_mode": "RGB",
            "accepted_size": [input_resolution, input_resolution],
            "exif_orientation": "identity-only",
            "metadata_policy": "discard",
        },
        "resize": {
            "input_size": [input_resolution, input_resolution],
            "output_size": [output_resolution, output_resolution],
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
    }


def build_preparation_plan(
    *,
    lock: SourceLock,
    resolution: int = 128,
) -> PreparationPlan:
    """Derive the prepared cache identity without requiring the raw archive."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if lock.expected_sha256 is None:
        raise PreparationError(
            "materialization requires a source lock with a pinned SHA-256"
        )
    source_counts = lock.contract.source_class_counts
    if source_counts is None:
        raise PreparationError(
            "materialization requires pinned per-class source counts"
        )
    train_counts = source_counts.get("train")
    test_counts = source_counts.get("test")
    if train_counts is None or test_counts is None:
        raise PreparationError(
            "materialization requires pinned train and test class counts"
        )
    prepared_counts: dict[str, dict[str, int]] = {
        "train": {},
        "test": {},
    }
    for class_name in lock.contract.classes:
        train_count = train_counts.get(class_name)
        test_count = test_counts.get(class_name)
        if (
            isinstance(train_count, bool)
            or not isinstance(train_count, int)
            or isinstance(test_count, bool)
            or not isinstance(test_count, int)
            or train_count <= 0
            or test_count < 0
        ):
            raise PreparationError(
                "source lock contains invalid train or test class counts"
            )
        prepared_counts["train"][class_name] = train_count
        prepared_counts["test"][class_name] = test_count
    recipe = _transform_recipe(
        input_resolution=lock.contract.input_resolution,
        output_resolution=resolution,
    )
    recipe_sha256 = _canonical_digest(recipe)
    preparation_key = _canonical_digest(
        {
            "recipe_sha256": recipe_sha256,
            "source": {
                "type": "official_archive",
                "sha256": lock.expected_sha256,
            },
        }
    )
    return PreparationPlan(
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        preparation_key=preparation_key,
        counts=prepared_counts,
    )
