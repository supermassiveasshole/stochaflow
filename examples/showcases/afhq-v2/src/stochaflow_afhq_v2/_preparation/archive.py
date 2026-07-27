"""Audit the structure and identity of an official AFHQ-v2 ZIP archive."""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile, ZipInfo

from .contracts import DatasetContract, PreparationError, SourceImage
from .safe_file import (
    _open_regular_file_without_links,
    _regular_file_state,
    _require_open_file_path_identity,
)

_MAX_ARCHIVE_MEMBERS = 20_000
_MAX_ARCHIVE_BYTES = 12 * 1024 * 1024 * 1024
_MAX_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_WINDOWS_INVALID_FILENAME_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
def _zip_member_mode(info: ZipInfo) -> int:
    if info.create_system != 3:
        return 0
    return info.external_attr >> 16


def _validate_member_name(info: ZipInfo) -> tuple[str, tuple[str, ...]]:
    name = info.filename
    if not name or "\x00" in name:
        raise PreparationError("archive contains an empty or NUL-containing path")
    if "\\" in name:
        raise PreparationError(f"archive member uses a Windows separator: {name!r}")
    normalized = unicodedata.normalize("NFC", name)
    if normalized.startswith(("/", "//")) or _WINDOWS_DRIVE_PATTERN.match(
        normalized
    ):
        raise PreparationError(f"archive member has an absolute path: {name!r}")
    component_text = normalized
    if info.is_dir() and component_text.endswith("/"):
        component_text = component_text[:-1]
    raw_parts = component_text.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise PreparationError(f"archive member has an unsafe path: {name!r}")
    parts = tuple(raw_parts)
    for part in parts:
        if (
            any(ord(character) < 32 or ord(character) == 127 for character in part)
            or any(
                character in _WINDOWS_INVALID_FILENAME_CHARACTERS
                for character in part
            )
            or part.endswith((" ", "."))
        ):
            raise PreparationError(
                f"archive member is not portable to Windows: {name!r}"
            )
        stem = part.split(".", maxsplit=1)[0].rstrip(" .").upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise PreparationError(
                f"archive member uses a Windows reserved name: {name!r}"
            )

    mode = _zip_member_mode(info)
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise PreparationError(f"archive member is a symbolic link: {name!r}")
    if info.is_dir():
        if file_type not in {0, stat.S_IFDIR}:
            raise PreparationError(
                f"archive directory has a non-directory type: {name!r}"
            )
    elif file_type not in {0, stat.S_IFREG}:
        raise PreparationError(f"archive member is not a regular file: {name!r}")
    if info.flag_bits & 0x1:
        raise PreparationError(f"archive member is encrypted: {name!r}")
    return normalized, parts


def _source_image_from_member(
    info: ZipInfo,
    *,
    normalized_name: str,
    parts: tuple[str, ...],
    contract: DatasetContract,
) -> SourceImage:
    if len(parts) == 4:
        root, split, class_name, filename = parts
        if root != "afhq_v2":
            raise PreparationError(
                f"unexpected archive root for image: {normalized_name!r}"
            )
    elif len(parts) == 3:
        split, class_name, filename = parts
    else:
        raise PreparationError(
            f"image must use [afhq_v2/]split/class/file.png: {normalized_name!r}"
        )
    if split not in {"train", "test"}:
        raise PreparationError(
            f"unexpected source split {split!r}: {normalized_name!r}"
        )
    if class_name not in contract.classes:
        raise PreparationError(
            f"unexpected AFHQ class {class_name!r}: {normalized_name!r}"
        )
    if not filename or PurePosixPath(filename).suffix != ".png":
        raise PreparationError(f"source image must be lowercase .png: {normalized_name}")
    if info.file_size <= 0 or info.file_size > _MAX_IMAGE_BYTES:
        raise PreparationError(
            f"source image has an invalid size ({info.file_size}): "
            f"{normalized_name!r}"
        )
    if info.compress_size == 0:
        raise PreparationError(
            f"source image has an invalid compressed size: {normalized_name!r}"
        )
    if info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
        raise PreparationError(
            f"source image exceeds the compression-ratio limit: {normalized_name!r}"
        )
    relative_path = f"{split}/{class_name}/{filename}"
    return SourceImage(
        member_name=info.filename,
        relative_path=relative_path,
        source_split=split,
        class_name=class_name,
        filename=filename,
        file_size=info.file_size,
        crc32=info.CRC,
    )


def _inspect_archive_stream(
    archive_stream: BinaryIO,
    *,
    archive_label: str,
    contract: DatasetContract,
) -> tuple[SourceImage, ...]:
    archive_stream.seek(0)
    try:
        with ZipFile(archive_stream) as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise PreparationError(
                    f"archive has too many members: {len(members)}"
                )
            if sum(info.file_size for info in members) > _MAX_ARCHIVE_BYTES:
                raise PreparationError("archive exceeds the uncompressed size limit")
            compressed_bytes = sum(info.compress_size for info in members)
            uncompressed_bytes = sum(info.file_size for info in members)
            if (
                compressed_bytes == 0
                or uncompressed_bytes / compressed_bytes > _MAX_COMPRESSION_RATIO
            ):
                raise PreparationError(
                    "archive exceeds the aggregate compression-ratio limit"
                )

            seen_names: set[str] = set()
            seen_casefolded: dict[str, str] = {}
            seen_relative: set[str] = set()
            image_layout_lengths: set[int] = set()
            images: list[SourceImage] = []
            for info in members:
                normalized_name, parts = _validate_member_name(info)
                if normalized_name in seen_names:
                    raise PreparationError(
                        f"archive contains a duplicate member: {normalized_name!r}"
                    )
                seen_names.add(normalized_name)
                casefolded = normalized_name.casefold()
                prior = seen_casefolded.get(casefolded)
                if prior is not None and prior != normalized_name:
                    raise PreparationError(
                        "archive has a case-insensitive path collision: "
                        f"{prior!r} and {normalized_name!r}"
                    )
                seen_casefolded[casefolded] = normalized_name
                if info.is_dir():
                    continue
                image_layout_lengths.add(len(parts))
                image = _source_image_from_member(
                    info,
                    normalized_name=normalized_name,
                    parts=parts,
                    contract=contract,
                )
                if image.relative_path in seen_relative:
                    raise PreparationError(
                        "archive maps multiple members to the canonical path "
                        f"{image.relative_path!r}"
                    )
                seen_relative.add(image.relative_path)
                images.append(image)
            if len(image_layout_lengths) != 1:
                raise PreparationError(
                    "archive mixes rooted and rootless image layouts"
                )
    except BadZipFile as error:
        raise PreparationError(f"invalid ZIP archive: {archive_label}") from error

    images.sort(key=lambda item: item.relative_path)
    split_counts = Counter(image.source_split for image in images)
    if split_counts != Counter(
        {
            "train": contract.train_count,
            "test": contract.test_count,
        }
    ):
        raise PreparationError(
            "source split counts do not match the lock: "
            f"{dict(sorted(split_counts.items()))}"
        )
    if len(images) != contract.total_count:
        raise PreparationError(
            f"source image count mismatch: {len(images)} != {contract.total_count}"
        )
    class_counts = Counter(image.class_name for image in images)
    if set(class_counts) != set(contract.classes):
        raise PreparationError(
            f"source classes do not match the lock: {sorted(class_counts)}"
        )
    if contract.source_class_counts is not None:
        actual_source_class_counts = {
            split: {
                class_name: sum(
                    image.source_split == split
                    and image.class_name == class_name
                    for image in images
                )
                for class_name in contract.classes
            }
            for split in ("train", "test")
        }
        expected_source_class_counts = {
            split: dict(counts)
            for split, counts in contract.source_class_counts.items()
        }
        if actual_source_class_counts != expected_source_class_counts:
            raise PreparationError(
                "source split/class counts do not match the lock: "
                f"{actual_source_class_counts}"
            )
    return tuple(images)


def inspect_archive(
    archive_path: Path,
    *,
    contract: DatasetContract,
) -> tuple[SourceImage, ...]:
    """Audit one stable ZIP descriptor and return its source inventory."""

    descriptor, opened = _open_regular_file_without_links(
        archive_path,
        label="AFHQ-v2 archive inspection source",
    )
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            images = _inspect_archive_stream(
                stream,
                archive_label=str(archive_path),
                contract=contract,
            )
        after = os.fstat(descriptor)
        if _regular_file_state(after) != _regular_file_state(opened):
            raise PreparationError(
                f"archive changed while it was inspected: {archive_path}"
            )
        _require_open_file_path_identity(
            archive_path,
            after,
            label="AFHQ-v2 archive inspection source",
        )
        return images
    finally:
        os.close(descriptor)
