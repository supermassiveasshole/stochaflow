"""Prepare AFHQ-v2 Dog images with the pinned guided-diffusion transform."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin, UnidentifiedImageError

from .contracts import PreparationError

_EXIF_ORIENTATION_TAG = 274
_PNG_COMPRESS_LEVEL = 6


def decode_and_center_crop(
    payload: bytes,
    *,
    member_name: str,
    expected_input_size: tuple[int, int],
    output_resolution: int,
) -> Image.Image:
    """Decode and apply guided-diffusion's iterative center-crop transform."""

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
            if source.size != expected_input_size:
                expected = "x".join(str(value) for value in expected_input_size)
                raise PreparationError(
                    f"source image has size {source.size}, expected "
                    f"{expected}: {member_name!r}"
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
            prepared = source.convert("RGB")
            prepared.load()
            while min(prepared.size) >= 2 * output_resolution:
                prepared = prepared.resize(
                    (prepared.size[0] // 2, prepared.size[1] // 2),
                    resample=Image.Resampling.BOX,
                    reducing_gap=None,
                )
            scale = output_resolution / min(prepared.size)
            prepared = prepared.resize(
                (
                    round(prepared.size[0] * scale),
                    round(prepared.size[1] * scale),
                ),
                resample=Image.Resampling.BICUBIC,
                reducing_gap=None,
            )
            pixels = np.array(prepared)
            crop_y = (pixels.shape[0] - output_resolution) // 2
            crop_x = (pixels.shape[1] - output_resolution) // 2
            cropped = pixels[
                crop_y : crop_y + output_resolution,
                crop_x : crop_x + output_resolution,
            ]
            if cropped.shape != (
                output_resolution,
                output_resolution,
                3,
            ):
                raise PreparationError(
                    f"source image cannot produce a {output_resolution}x"
                    f"{output_resolution} RGB crop: {member_name!r}"
                )
            result = Image.fromarray(np.ascontiguousarray(cropped))
            result.info.clear()
            return result
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise PreparationError(
            f"failed to decode source image: {member_name!r}"
        ) from error


def save_dog_png(
    image: Image.Image,
    destination: Path,
) -> tuple[str, int]:
    """Encode one transformed Dog image with the pinned PNG policy."""

    encoded_stream = io.BytesIO()
    image.save(
        encoded_stream,
        format="PNG",
        optimize=False,
        compress_level=_PNG_COMPRESS_LEVEL,
        pnginfo=PngImagePlugin.PngInfo(),
    )
    encoded = encoded_stream.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


__all__ = ["decode_and_center_crop", "save_dog_png"]
