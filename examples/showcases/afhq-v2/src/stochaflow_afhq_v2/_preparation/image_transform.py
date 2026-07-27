"""Decode, resize, encode, and describe prepared AFHQ-v2 images."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from PIL import Image, PngImagePlugin, UnidentifiedImageError

from stochaflow.data.artifact_io import write_cache_file

from .contracts import PreparationError, SourceImage

_EXIF_ORIENTATION_TAG = 274
_PNG_COMPRESS_LEVEL = 6

def _decode_and_resize(
    payload: bytes,
    *,
    member_name: str,
    input_resolution: int,
    output_resolution: int,
) -> tuple[Image.Image, str]:
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            if probe.format != "PNG":
                raise PreparationError(
                    f"source image is not PNG: {member_name!r}"
                )
            probe.verify()
        with Image.open(io.BytesIO(payload)) as source:
            if source.format != "PNG":
                raise PreparationError(
                    f"source image is not PNG: {member_name!r}"
                )
            if source.size != (input_resolution, input_resolution):
                raise PreparationError(
                    f"source image has size {source.size}, expected "
                    f"{input_resolution}x{input_resolution}: {member_name!r}"
                )
            if source.mode != "RGB":
                raise PreparationError(
                    f"source image has mode {source.mode!r}, expected RGB: "
                    f"{member_name!r}"
                )
            orientation = source.getexif().get(_EXIF_ORIENTATION_TAG, 1)
            if orientation != 1:
                raise PreparationError(
                    f"source image has non-identity EXIF orientation "
                    f"{orientation!r}: {member_name!r}"
                )
            rgb = source.convert("RGB")
            rgb.load()
            pixel_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            resized = rgb.resize(
                (output_resolution, output_resolution),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=None,
            )
            resized.info.clear()
            return resized, pixel_digest
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise PreparationError(
            f"failed to decode source image: {member_name!r}"
        ) from error


def _save_prepared_png(
    image: Image.Image,
    destination: Path,
    *,
    cache_root: Path,
) -> tuple[str, int]:
    encoded_stream = io.BytesIO()
    image.save(
        encoded_stream,
        format="PNG",
        optimize=False,
        compress_level=_PNG_COMPRESS_LEVEL,
        pnginfo=PngImagePlugin.PngInfo(),
    )
    encoded = encoded_stream.getvalue()
    write_cache_file(
        cache_root,
        destination,
        encoded,
        label="prepared AFHQ-v2 image",
    )
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _write_text_atomic(
    path: Path,
    content: str,
    *,
    cache_root: Path,
    label: str,
) -> None:
    write_cache_file(
        cache_root,
        path,
        content.encode("utf-8"),
        label=label,
    )


def _manifest_text(manifest: Mapping[str, object]) -> str:
    return yaml.safe_dump(
        dict(manifest),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )


def _prepared_counts(
    images: Sequence[SourceImage],
    *,
    classes: Sequence[str],
) -> dict[str, dict[str, int]]:
    counts = {
        split: dict.fromkeys(classes, 0)
        for split in ("train", "test")
    }
    for image in images:
        counts[image.source_split][image.class_name] += 1
    return counts
