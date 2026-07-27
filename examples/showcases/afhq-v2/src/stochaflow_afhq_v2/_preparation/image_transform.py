"""Decode, resize, encode, and describe prepared AFHQ-v2 images."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, PngImagePlugin, UnidentifiedImageError

from .contracts import PreparationError

_EXIF_ORIENTATION_TAG = 274
_PNG_COMPRESS_LEVEL = 6

def decode_and_resize(
    payload: bytes,
    *,
    member_name: str,
    input_resolution: int,
    output_resolution: int,
) -> Image.Image:
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
            resized = rgb.resize(
                (output_resolution, output_resolution),
                resample=Image.Resampling.LANCZOS,
                reducing_gap=None,
            )
            resized.info.clear()
            return resized
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise PreparationError(
            f"failed to decode source image: {member_name!r}"
        ) from error


def save_prepared_png(
    image: Image.Image,
    destination: Path,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


__all__ = ["decode_and_resize", "save_prepared_png"]
