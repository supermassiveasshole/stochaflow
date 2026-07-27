"""Image-specific producer callbacks for externally owned folders."""

from __future__ import annotations

import hashlib
import io
import json
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from PIL import Image

from stochaflow.data.artifact_io import (
    ArtifactFileSnapshot,
    read_regular_file,
    scan_regular_files,
)
from stochaflow.data.artifact_store import (
    DataArtifactLoadContext,
    DataArtifactValidationError,
    ReferencedDataArtifactBuild,
    canonical_artifact_digest,
    canonical_artifact_json_bytes,
)
from stochaflow.data.image_contracts import (
    IMAGE_SUFFIXES,
    ImageFileRecord,
    validate_relative_image_path,
)

IMAGE_REFERENCE_MATERIALIZER = "stochaflow.reference-image-inventory"
_INVENTORY_RECORD_LIMIT = 100_000
_DOMAIN_FIELDS = frozenset({"schema_version", "layout", "inventory"})
_INVENTORY_FIELDS = frozenset(
    {"record_limit", "record_count", "shards"}
)
_SHARD_FIELDS = frozenset({"path", "record_count", "sha256"})


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{path} field names must be strings")
    if set(raw) != fields:
        raise ValueError(f"{path} must contain exactly {sorted(fields)}")
    return cast(dict[str, Any], value)


def _scan_regular_file_snapshots(
    root: Path,
    *,
    hash_contents: bool,
    label: str,
) -> tuple[ArtifactFileSnapshot, ...]:
    return scan_regular_files(
        root,
        hash_contents=hash_contents,
        label=label,
        path_filter=(
            lambda relative: (
                PurePosixPath(relative).suffix.lower() in IMAGE_SUFFIXES
            )
        ),
    )


def _scan_image_tree_paths(
    root: Path,
    *,
    tree: str,
) -> tuple[tuple[str, str, int], ...]:
    collision_keys: dict[str, str] = {}
    snapshots = _scan_regular_file_snapshots(
        root,
        hash_contents=False,
        label="referenced image data",
    )
    for snapshot in snapshots:
        relative = snapshot.relative_path
        if relative != unicodedata.normalize("NFC", relative):
            raise ValueError(
                f"referenced file path must use NFC normalization: {relative}"
            )
        validate_relative_image_path(
            relative,
            path=f"referenced file {relative}",
        )
        collision_key = relative.casefold()
        previous = collision_keys.get(collision_key)
        if previous is not None:
            raise ValueError(
                "referenced data contains a case/NFC path collision: "
                f"{previous!r}, {relative!r}"
            )
        collision_keys[collision_key] = relative
    if not snapshots:
        raise ValueError(f"image directory contains no supported images: {root}")
    return tuple(
        (tree, snapshot.relative_path, snapshot.size_bytes)
        for snapshot in snapshots
    )


def _scan_image_tree(
    root: Path,
    *,
    tree: str,
) -> tuple[ImageFileRecord, ...]:
    records: list[ImageFileRecord] = []
    for _, relative, observed_size in _scan_image_tree_paths(root, tree=tree):
        encoded, metadata = read_regular_file(
            root,
            relative,
            label="referenced image",
        )
        if metadata.st_size != observed_size or len(encoded) != observed_size:
            raise ValueError(
                f"referenced image size changed while indexing: {relative}"
            )
        try:
            with Image.open(io.BytesIO(encoded)) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(
                f"cannot decode referenced image metadata: {root / relative}"
            ) from exc
        records.append(
            ImageFileRecord(
                tree=tree,
                path=relative,
                size_bytes=observed_size,
                sha256=hashlib.sha256(encoded).hexdigest(),
                width=width,
                height=height,
            )
        )
    records.sort(key=lambda record: (record.tree, record.path))
    return tuple(records)


def _records_without_hash(
    records: Sequence[ImageFileRecord],
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (record.tree, record.path, record.size_bytes) for record in records
    )


def _write_inventory(
    data_root: Path,
    records: Sequence[ImageFileRecord],
) -> dict[str, object]:
    inventory_root = data_root / "image-inventory"
    inventory_root.mkdir(parents=True)
    shards: list[dict[str, object]] = []
    for shard_index, offset in enumerate(
        range(0, len(records), _INVENTORY_RECORD_LIMIT)
    ):
        selected = records[offset : offset + _INVENTORY_RECORD_LIMIT]
        relative_path = f"image-inventory/{shard_index:06d}.jsonl"
        encoded = b"".join(
            canonical_artifact_json_bytes(record.to_dict())
            for record in selected
        )
        (data_root / relative_path).write_bytes(encoded)
        shards.append(
            {
                "path": relative_path,
                "record_count": len(selected),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return {
        "record_limit": _INVENTORY_RECORD_LIMIT,
        "record_count": len(records),
        "shards": shards,
    }


def _read_inventory(
    data_root: Path,
    value: object,
) -> tuple[ImageFileRecord, ...]:
    inventory = _strict_mapping(
        value,
        fields=_INVENTORY_FIELDS,
        path="image reference domain.inventory",
    )
    if (
        type(inventory["record_limit"]) is not int
        or inventory["record_limit"] != _INVENTORY_RECORD_LIMIT
    ):
        raise ValueError("image reference inventory record_limit is invalid")
    if (
        type(inventory["record_count"]) is not int
        or inventory["record_count"] <= 0
    ):
        raise ValueError(
            "image reference inventory record_count must be positive"
        )
    record_count = cast(int, inventory["record_count"])
    raw_shards = inventory["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("image reference inventory shards must be non-empty")
    expected_shards = (
        record_count + _INVENTORY_RECORD_LIMIT - 1
    ) // _INVENTORY_RECORD_LIMIT
    if len(raw_shards) != expected_shards:
        raise ValueError("image reference inventory shard count is invalid")
    records: list[ImageFileRecord] = []
    for shard_index, raw_shard in enumerate(raw_shards):
        shard = _strict_mapping(
            raw_shard,
            fields=_SHARD_FIELDS,
            path=f"image reference inventory.shards[{shard_index}]",
        )
        expected_path = f"image-inventory/{shard_index:06d}.jsonl"
        if shard["path"] != expected_path:
            raise ValueError(
                "image reference inventory shard paths are not canonical"
            )
        expected_count = min(
            _INVENTORY_RECORD_LIMIT,
            record_count - shard_index * _INVENTORY_RECORD_LIMIT,
        )
        if (
            type(shard["record_count"]) is not int
            or shard["record_count"] != expected_count
        ):
            raise ValueError(
                "image reference inventory shard record count is invalid"
            )
        digest = shard["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "image reference inventory shard digest is invalid"
            )
        encoded, _ = read_regular_file(
            data_root,
            expected_path,
            label="image reference inventory shard",
        )
        if hashlib.sha256(encoded).hexdigest() != digest:
            raise ValueError("image reference inventory shard digest mismatch")
        lines = encoded.splitlines(keepends=True)
        if len(lines) != expected_count:
            raise ValueError(
                "image reference inventory shard record count changed"
            )
        for line_index, line in enumerate(lines):
            try:
                raw_record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "image reference inventory contains invalid JSON"
                ) from exc
            if line != canonical_artifact_json_bytes(raw_record):
                raise ValueError(
                    "image reference inventory record is not canonical"
                )
            records.append(
                ImageFileRecord.from_dict(
                    raw_record,
                    path=(
                        "image reference inventory "
                        f"shard[{shard_index}][{line_index}]"
                    ),
                )
            )
    if len(records) != record_count:
        raise ValueError("image reference inventory record count changed")
    keys = [(record.tree, record.path.casefold()) for record in records]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError(
            "image reference inventory records must be sorted and unique"
        )
    return tuple(records)


def image_reference_materialization_digest(
    layout: Mapping[str, object],
) -> str:
    """Return the stable identity of the image indexing recipe."""

    return canonical_artifact_digest(
        {
            "name": IMAGE_REFERENCE_MATERIALIZER,
            "version": 3,
            "layout": dict(layout),
        }
    )


def build_image_reference(
    data_root: Path,
    *,
    roots: Mapping[str, Path],
    layout: Mapping[str, object],
) -> ReferencedDataArtifactBuild:
    """Build image-domain sidecars for a framework reference artifact."""

    records = tuple(
        sorted(
            (
                record
                for tree, root in sorted(roots.items())
                for record in _scan_image_tree(root, tree=tree)
            ),
            key=lambda record: (record.tree, record.path),
        )
    )
    keys = [(record.tree, record.path.casefold()) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("reference inventory contains duplicate paths")
    serialized = [record.to_dict() for record in records]
    content_digest = canonical_artifact_digest(serialized)
    inventory = _write_inventory(data_root, records)
    return ReferencedDataArtifactBuild(
        source_digest=content_digest,
        materialization_digest=image_reference_materialization_digest(layout),
        content_digest=content_digest,
        domain={
            "schema_version": 1,
            "layout": dict(layout),
            "inventory": inventory,
        },
    )


def load_image_reference(
    context: DataArtifactLoadContext,
    *,
    roots: Mapping[str, Path],
    layout: Mapping[str, object],
) -> tuple[ImageFileRecord, ...]:
    """Verify image-domain sidecars and represented external content."""

    try:
        domain = _strict_mapping(
            dict(context.domain),
            fields=_DOMAIN_FIELDS,
            path="image reference domain",
        )
        if (
            type(domain["schema_version"]) is not int
            or domain["schema_version"] != 1
        ):
            raise ValueError(
                "image reference domain.schema_version must be 1"
            )
        if domain["layout"] != dict(layout):
            raise ValueError(
                "image reference domain does not match the selected layout"
            )
        records = _read_inventory(context.data_root, domain["inventory"])
        live_paths = tuple(
            item
            for tree, root in sorted(roots.items())
            for item in _scan_image_tree_paths(root, tree=tree)
        )
        if live_paths != _records_without_hash(records):
            raise ValueError("referenced image paths or sizes changed")
        serialized = [record.to_dict() for record in records]
        content_digest = canonical_artifact_digest(serialized)
        if (
            context.identity.source_digest != content_digest
            or context.identity.content_digest != content_digest
        ):
            raise ValueError("referenced image content identity is invalid")
        expected_materialization = image_reference_materialization_digest(
            layout
        )
        if context.identity.materialization_digest != expected_materialization:
            raise ValueError(
                "referenced image materialization identity is invalid"
            )
        if context.verification == "full":
            live_records = tuple(
                sorted(
                    (
                        record
                        for tree, root in sorted(roots.items())
                        for record in _scan_image_tree(root, tree=tree)
                    ),
                    key=lambda record: (record.tree, record.path),
                )
            )
            if live_records != records:
                raise ValueError(
                    "referenced image content digest or dimensions changed"
                )
        return records
    except DataArtifactValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise DataArtifactValidationError(str(exc)) from exc


__all__ = [
    "IMAGE_REFERENCE_MATERIALIZER",
    "build_image_reference",
    "image_reference_materialization_digest",
    "load_image_reference",
]
