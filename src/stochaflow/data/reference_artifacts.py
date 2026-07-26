"""Referenced image artifact inventory and materialization store."""

from __future__ import annotations

import hashlib
import io
import json
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast
from uuid import uuid4

from PIL import Image

from stochaflow.data.artifact_io import (
    ArtifactFileSnapshot,
    create_cache_directory,
    publish_cache_directory,
    read_regular_file,
    remove_cache_directory,
    scan_regular_files,
    write_cache_file,
)
from stochaflow.data.artifact_store import (
    ArtifactMaterializationLock,
)
from stochaflow.data.artifact_store import (
    canonical_digest as _canonical_digest,
)
from stochaflow.data.artifact_store import (
    canonical_json_bytes as _canonical_json_bytes,
)
from stochaflow.data.artifact_store import (
    load_canonical_json as _load_canonical_json,
)
from stochaflow.data.artifact_store import (
    path_exists_without_following as _path_exists_without_following,
)
from stochaflow.data.artifact_store import (
    quarantine_path as _quarantine_path,
)
from stochaflow.data.artifact_store import (
    read_locator_for_policy as _read_locator_for_policy,
)
from stochaflow.data.artifact_store import (
    sha256_bytes as _sha256_bytes,
)
from stochaflow.data.artifact_store import (
    strict_mapping as _strict_mapping,
)
from stochaflow.data.artifact_store import (
    write_locator as _write_locator,
)
from stochaflow.data.artifacts import (
    DataSourceContext,
    ReferencedDataArtifactIdentity,
)
from stochaflow.data.image_contracts import (
    IMAGE_SUFFIXES,
    ImageFileRecord,
    validate_relative_image_path,
)

_INVENTORY_RECORD_LIMIT = 100_000
_REFERENCE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "artifact_type",
        "source_name",
        "source_digest",
        "materializer_name",
        "materialization_digest",
        "layout",
        "inventory",
        "artifact_digest",
    }
)
_INVENTORY_FIELDS = frozenset(
    {"record_limit", "record_count", "shards"}
)
_SHARD_FIELDS = frozenset({"path", "record_count", "sha256"})


def is_relative_to(path: Path, root: Path) -> bool:
    """Return whether one normalized path is nested below another."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _scan_regular_file_snapshots(
    root: Path,
    *,
    hash_contents: bool,
    label: str,
    path_filter: Callable[[str], bool] | None = None,
) -> tuple[ArtifactFileSnapshot, ...]:
    return scan_regular_files(
        root,
        hash_contents=hash_contents,
        label=label,
        path_filter=path_filter,
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
        label="referenced data",
        path_filter=(
            lambda relative: (
                PurePosixPath(relative).suffix.lower() in IMAGE_SUFFIXES
            )
        ),
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


def _assert_unique_records(records: Sequence[ImageFileRecord]) -> None:
    keys = [(record.tree, record.path.casefold()) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("reference inventory contains duplicate paths")


def _write_inventory(
    cache_root: Path,
    index_root: Path,
    records: Sequence[ImageFileRecord],
) -> dict[str, Any]:
    inventory_root = index_root / "inventory"
    create_cache_directory(
        cache_root,
        inventory_root,
        label="reference inventory directory",
    )
    shards: list[dict[str, Any]] = []
    for shard_index, offset in enumerate(
        range(0, len(records), _INVENTORY_RECORD_LIMIT)
    ):
        selected = records[offset : offset + _INVENTORY_RECORD_LIMIT]
        relative_path = f"inventory/{shard_index:06d}.jsonl"
        encoded = b"".join(
            _canonical_json_bytes(record.to_dict()) for record in selected
        )
        write_cache_file(
            cache_root,
            index_root / relative_path,
            encoded,
            label="reference inventory shard",
        )
        shards.append(
            {
                "path": relative_path,
                "record_count": len(selected),
                "sha256": _sha256_bytes(encoded),
            }
        )
    return {
        "record_limit": _INVENTORY_RECORD_LIMIT,
        "record_count": len(records),
        "shards": shards,
    }


def _read_inventory(
    index_root: Path,
    value: object,
) -> tuple[ImageFileRecord, ...]:
    inventory = _strict_mapping(
        value,
        fields=_INVENTORY_FIELDS,
        path="reference manifest.inventory",
    )
    if (
        type(inventory["record_limit"]) is not int
        or inventory["record_limit"] != _INVENTORY_RECORD_LIMIT
    ):
        raise ValueError("reference inventory record_limit is invalid")
    if (
        type(inventory["record_count"]) is not int
        or inventory["record_count"] <= 0
    ):
        raise ValueError("reference inventory record_count must be positive")
    record_count = cast(int, inventory["record_count"])
    serialized_shards = inventory["shards"]
    if not isinstance(serialized_shards, list) or not serialized_shards:
        raise ValueError("reference inventory shards must be a non-empty list")
    expected_shard_count = (
        record_count + _INVENTORY_RECORD_LIMIT - 1
    ) // _INVENTORY_RECORD_LIMIT
    if len(serialized_shards) != expected_shard_count:
        raise ValueError("reference inventory shard count is not canonical")
    records: list[ImageFileRecord] = []
    for shard_index, value in enumerate(serialized_shards):
        shard = _strict_mapping(
            value,
            fields=_SHARD_FIELDS,
            path=f"reference manifest.inventory.shards[{shard_index}]",
        )
        expected_path = f"inventory/{shard_index:06d}.jsonl"
        if shard["path"] != expected_path:
            raise ValueError("reference inventory shard paths are not canonical")
        expected_records = min(
            _INVENTORY_RECORD_LIMIT,
            record_count - shard_index * _INVENTORY_RECORD_LIMIT,
        )
        if (
            type(shard["record_count"]) is not int
            or shard["record_count"] != expected_records
        ):
            raise ValueError(
                "reference inventory shard record count is not canonical"
            )
        shard_digest = shard["sha256"]
        if (
            not isinstance(shard_digest, str)
            or len(shard_digest) != 64
            or shard_digest != shard_digest.lower()
            or any(
                character not in "0123456789abcdef"
                for character in shard_digest
            )
        ):
            raise ValueError("reference inventory shard digest is invalid")
        encoded, _ = read_regular_file(
            index_root,
            expected_path,
            label="reference inventory shard",
        )
        if _sha256_bytes(encoded) != shard_digest:
            raise ValueError("reference inventory shard digest mismatch")
        lines = encoded.splitlines(keepends=True)
        if len(lines) != shard["record_count"]:
            raise ValueError("reference inventory shard record count mismatch")
        for line_index, line in enumerate(lines):
            try:
                raw_record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("reference inventory contains invalid JSON") from exc
            if line != _canonical_json_bytes(raw_record):
                raise ValueError("reference inventory record is not canonical JSON")
            records.append(
                ImageFileRecord.from_dict(
                    raw_record,
                    path=(
                        "reference manifest.inventory."
                        f"shards[{shard_index}][{line_index}]"
                    ),
                )
            )
    if len(records) != record_count:
        raise ValueError("reference inventory total record count mismatch")
    if tuple(records) != tuple(
        sorted(records, key=lambda record: (record.tree, record.path))
    ):
        raise ValueError("reference inventory records must be sorted")
    _assert_unique_records(records)
    return tuple(records)


def _reference_locator_path(
    cache_root: Path,
    source_name: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
) -> Path:
    locator_digest = _canonical_digest(
        {
            "source_name": source_name,
            "roots": {
                name: str(root)
                for name, root in sorted(roots.items())
            },
            "layout": layout,
        }
    )
    return (
        cache_root
        / "references"
        / "locators"
        / _canonical_digest(source_name)[:16]
        / f"{locator_digest}.json"
    )


def _reference_index_path(
    cache_root: Path,
    source_name: str,
    artifact_digest: str,
) -> Path:
    return (
        cache_root
        / "references"
        / _canonical_digest(source_name)[:16]
        / artifact_digest
    )


def _reference_lock_path(
    cache_root: Path,
    source_name: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
) -> Path:
    lock_digest = _canonical_digest(
        {
            "source": source_name,
            "roots": {
                name: str(root) for name, root in sorted(roots.items())
            },
            "layout": layout,
        }
    )
    return cache_root / "references" / "locks" / f"{lock_digest}.lock"


def _reference_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> ReferencedDataArtifactIdentity:
    return ReferencedDataArtifactIdentity(
        artifact_type=manifest["artifact_type"],
        source_name=manifest["source_name"],
        source_digest=manifest["source_digest"],
        materializer_name=manifest["materializer_name"],
        materialization_digest=manifest["materialization_digest"],
        artifact_digest=manifest["artifact_digest"],
        manifest_sha256=manifest_sha256,
    )


def _load_reference_index(
    index_root: Path,
    *,
    source_name: str,
    artifact_type: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
    verification: Literal["manifest", "full"],
) -> tuple[ReferencedDataArtifactIdentity, tuple[ImageFileRecord, ...]]:
    manifest_path = index_root / "manifest.json"
    raw, manifest_bytes = _load_canonical_json(
        manifest_path,
        label="reference manifest",
    )
    manifest = _strict_mapping(
        raw,
        fields=_REFERENCE_MANIFEST_FIELDS,
        path="reference manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 2
        or manifest["kind"] != "referenced"
    ):
        raise ValueError("reference manifest has an unsupported schema")
    if (
        manifest["source_name"] != source_name
        or manifest["artifact_type"] != artifact_type
        or manifest["layout"] != dict(layout)
    ):
        raise ValueError("reference manifest does not match the selected source")
    identity = _reference_manifest_identity(
        manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )
    if index_root.name != identity.artifact_digest:
        raise ValueError(
            "reference index directory does not match its artifact digest"
        )
    records = _read_inventory(index_root, manifest["inventory"])
    live_paths = tuple(
        snapshot
        for tree, root in sorted(roots.items())
        for snapshot in _scan_image_tree_paths(root, tree=tree)
    )
    if live_paths != _records_without_hash(records):
        raise ValueError("referenced data paths or sizes changed")
    if verification == "full":
        live_records = tuple(
            record
            for tree, root in sorted(roots.items())
            for record in _scan_image_tree(root, tree=tree)
        )
        live_records = tuple(
            sorted(
                live_records,
                key=lambda record: (record.tree, record.path),
            )
        )
        if live_records != records:
            raise ValueError(
                "referenced data content digest or dimensions changed"
            )
    if identity.source_digest != _canonical_digest(
        [record.to_dict() for record in records]
    ):
        raise ValueError("reference manifest source digest is invalid")
    expected_materialization_digest = _canonical_digest(
        {
            "name": identity.materializer_name,
            "version": 2,
            "layout": layout,
        }
    )
    if identity.materialization_digest != expected_materialization_digest:
        raise ValueError(
            "reference manifest materialization digest is invalid"
        )
    expected_artifact_digest = _canonical_digest(
        {
            "kind": "referenced",
            "artifact_type": artifact_type,
            "source_name": source_name,
            "source_digest": identity.source_digest,
            "materializer_name": identity.materializer_name,
            "materialization_digest": identity.materialization_digest,
            "inventory_digest": _canonical_digest(
                [record.to_dict() for record in records]
            ),
        }
    )
    if identity.artifact_digest != expected_artifact_digest:
        raise ValueError("reference manifest artifact digest is invalid")
    return identity, records


def _build_reference_index(
    *,
    cache_root: Path,
    source_name: str,
    artifact_type: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
    expected_identity: ReferencedDataArtifactIdentity | None,
) -> tuple[Path, ReferencedDataArtifactIdentity, tuple[ImageFileRecord, ...]]:
    records = tuple(
        record
        for tree, root in sorted(roots.items())
        for record in _scan_image_tree(root, tree=tree)
    )
    records = tuple(
        sorted(records, key=lambda record: (record.tree, record.path))
    )
    _assert_unique_records(records)
    source_digest = _canonical_digest(
        [record.to_dict() for record in records]
    )
    materializer_name = "stochaflow.reference-image-inventory"
    materialization_digest = _canonical_digest(
        {
            "name": materializer_name,
            "version": 2,
            "layout": layout,
        }
    )
    artifact_digest = _canonical_digest(
        {
            "kind": "referenced",
            "artifact_type": artifact_type,
            "source_name": source_name,
            "source_digest": source_digest,
            "materializer_name": materializer_name,
            "materialization_digest": materialization_digest,
            "inventory_digest": _canonical_digest(
                [record.to_dict() for record in records]
            ),
        }
    )
    final_root = _reference_index_path(
        cache_root,
        source_name,
        artifact_digest,
    )
    staging_root = final_root.parent / f".{artifact_digest}.{uuid4().hex}.tmp"
    create_cache_directory(
        cache_root,
        staging_root,
        label="reference index staging directory",
    )
    try:
        inventory = _write_inventory(cache_root, staging_root, records)
        manifest = {
            "schema_version": 2,
            "kind": "referenced",
            "artifact_type": artifact_type,
            "source_name": source_name,
            "source_digest": source_digest,
            "materializer_name": materializer_name,
            "materialization_digest": materialization_digest,
            "layout": dict(layout),
            "inventory": inventory,
            "artifact_digest": artifact_digest,
        }
        manifest_path = staging_root / "manifest.json"
        manifest_bytes = _canonical_json_bytes(manifest)
        write_cache_file(
            cache_root,
            manifest_path,
            manifest_bytes,
            label="reference manifest",
        )
        identity = _reference_manifest_identity(
            manifest,
            manifest_sha256=_sha256_bytes(manifest_bytes),
        )
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(
                "strict resume referenced data identity does not match"
            )
        if _path_exists_without_following(cache_root, final_root):
            try:
                winner_identity, winner_records = _load_reference_index(
                    final_root,
                    source_name=source_name,
                    artifact_type=artifact_type,
                    roots=roots,
                    layout=layout,
                    verification="full",
                )
            except (FileNotFoundError, OSError, TypeError, ValueError):
                _quarantine_path(cache_root, final_root)
            else:
                remove_cache_directory(
                    cache_root,
                    staging_root,
                    label="reference index staging directory",
                )
                return final_root, winner_identity, winner_records
        try:
            publish_cache_directory(
                cache_root,
                staging_root,
                final_root,
                label="reference index",
            )
        except FileExistsError:
            winner_identity, winner_records = _load_reference_index(
                final_root,
                source_name=source_name,
                artifact_type=artifact_type,
                roots=roots,
                layout=layout,
                verification="full",
            )
            remove_cache_directory(
                cache_root,
                staging_root,
                label="reference index staging directory",
            )
            return final_root, winner_identity, winner_records
        return final_root, identity, records
    except BaseException:
        if _path_exists_without_following(cache_root, staging_root):
            remove_cache_directory(
                cache_root,
                staging_root,
                label="reference index staging directory",
            )
        raise


def materialize_reference(
    context: DataSourceContext,
    *,
    source_name: str,
    artifact_type: str,
    roots: Mapping[str, Path],
    layout: Mapping[str, Any],
) -> tuple[Path, ReferencedDataArtifactIdentity, tuple[ImageFileRecord, ...]]:
    cache_root = context.cache_root
    for root in roots.values():
        if is_relative_to(cache_root, root) or is_relative_to(
            root,
            cache_root,
        ):
            raise ValueError(
                "data source cache_root and referenced roots must not overlap"
            )
    expected = context.expected_identity
    locator = _reference_locator_path(
        cache_root,
        source_name,
        roots,
        layout,
    )
    if expected is not None:
        if not isinstance(expected, ReferencedDataArtifactIdentity):
            raise ValueError(
                "strict resume expected a different data artifact kind"
            )
        if (
            expected.source_name != source_name
            or expected.artifact_type != artifact_type
        ):
            raise ValueError(
                "strict resume expected a different referenced data source"
            )
        artifact_digest: str | None = expected.artifact_digest
    else:
        artifact_digest = _read_locator_for_policy(
            cache_root,
            locator,
            policy=context.policy,
            quarantine_on_error=False,
        )
    if artifact_digest is not None:
        index_root = _reference_index_path(
            cache_root,
            source_name,
            artifact_digest,
        )
        try:
            identity, records = _load_reference_index(
                index_root,
                source_name=source_name,
                artifact_type=artifact_type,
                roots=roots,
                layout=layout,
                verification=context.verification,
            )
            if expected is not None and identity != expected:
                raise ValueError(
                    "strict resume referenced data identity does not match"
                )
            return index_root, identity, records
        except (FileNotFoundError, OSError, TypeError, ValueError):
            if context.policy == "require":
                raise
    if context.policy == "require":
        raise FileNotFoundError(
            f"required reference artifact is not indexed for '{source_name}'"
        )
    lock_path = _reference_lock_path(
        cache_root,
        source_name,
        roots,
        layout,
    )
    with ArtifactMaterializationLock(lock_path, cache_root=cache_root):
        winner_digest = (
            expected.artifact_digest
            if expected is not None
            else _read_locator_for_policy(
                cache_root,
                locator,
                policy="ensure",
            )
        )
        if winner_digest is not None:
            winner_root = _reference_index_path(
                cache_root,
                source_name,
                winner_digest,
            )
            try:
                winner_identity, winner_records = _load_reference_index(
                    winner_root,
                    source_name=source_name,
                    artifact_type=artifact_type,
                    roots=roots,
                    layout=layout,
                    verification=context.verification,
                )
                if expected is not None and winner_identity != expected:
                    raise ValueError(
                        "strict resume referenced data identity does not match"
                    )
                return winner_root, winner_identity, winner_records
            except (FileNotFoundError, OSError, TypeError, ValueError):
                if _path_exists_without_following(cache_root, winner_root):
                    _quarantine_path(cache_root, winner_root)
        index_root, identity, records = _build_reference_index(
            cache_root=cache_root,
            source_name=source_name,
            artifact_type=artifact_type,
            roots=roots,
            layout=layout,
            expected_identity=expected,
        )
        if expected is not None and identity != expected:
            raise ValueError(
                "strict resume referenced data identity does not match"
            )
        if expected is None:
            _write_locator(cache_root, locator, identity.artifact_digest)
    return index_root, identity, records


__all__ = [
    "materialize_reference",
]
