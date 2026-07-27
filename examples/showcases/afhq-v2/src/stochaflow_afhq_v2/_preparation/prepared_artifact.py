"""Read and verify published AFHQ-v2 prepared artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from stochaflow.data.artifact_io import cache_entry_exists, canonical_directory

from .contracts import (
    PreparationError,
    PreparedArtifact,
    PreparedImageRecord,
    SourceArchive,
    SourceLock,
)
from .identity import _canonical_digest
from .planning import build_preparation_plan
from .safe_file import (
    _read_regular_file_without_links,
    _validate_relative_path,
)
from .safe_tree import (
    _enumerate_regular_files_without_links,
    _validate_prepared_root_layout,
)
from .source_lock import _require_exact_keys, _require_mapping

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INVENTORY_LINE_PATTERN = re.compile(
    r"^([0-9a-f]{64})  ([1-9][0-9]*)  ([^\r\n]+)$"
)

def require_prepared_artifact(
    *,
    lock: SourceLock,
    cache_root: Path,
    resolution: int = 128,
    full: bool = True,
) -> PreparedArtifact:
    """Verify a prepared cache hit without requiring raw archive bytes."""

    try:
        canonical_cache_root = canonical_directory(
            cache_root,
            label="AFHQ-v2 artifact cache root",
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise PreparationError(
            f"required AFHQ-v2 artifact cache does not exist: {cache_root}"
        ) from error
    plan = build_preparation_plan(
        lock=lock,
        resolution=resolution,
    )
    assert lock.expected_sha256 is not None
    source = SourceArchive(
        path=(
            canonical_cache_root
            / "raw"
            / "afhq-v2"
            / lock.expected_sha256
            / lock.archive_name
        ),
        sha256=lock.expected_sha256,
        size_bytes=lock.expected_bytes,
    )
    root = (
        canonical_cache_root
        / "prepared"
        / "afhq-v2"
        / str(resolution)
        / plan.preparation_key
    )
    if not cache_entry_exists(
        canonical_cache_root,
        root,
        label="required AFHQ-v2 prepared artifact",
    ):
        raise PreparationError(
            f"required AFHQ-v2 prepared artifact does not exist: {root}"
        )
    return verify_prepared_artifact(
        root,
        expected_preparation_key=plan.preparation_key,
        expected_recipe=plan.recipe,
        source_archive=source,
        source_lock=lock,
        expected_counts=plan.counts,
        full=full,
    )



def _load_manifest_bytes(payload: bytes, *, path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PreparationError(f"cannot read prepared manifest: {path}") from error
    return _require_mapping(value, label="prepared manifest")


def _parse_prepared_image_records(
    payload: bytes,
    *,
    path: Path,
) -> tuple[PreparedImageRecord, ...]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PreparationError(f"cannot read prepared inventory: {path}") from error
    records: list[PreparedImageRecord] = []
    for line_number, line in enumerate(lines, start=1):
        match = _INVENTORY_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise PreparationError(
                f"invalid files.sha256 line {line_number}: {line!r}"
            )
        records.append(
            PreparedImageRecord(
                relative_path=match.group(3),
                size_bytes=int(match.group(2)),
                sha256=match.group(1),
            )
        )
    paths = [record.relative_path for record in records]
    if paths != sorted(paths):
        raise PreparationError("files.sha256 paths are not sorted")
    if len(set(paths)) != len(records):
        raise PreparationError("files.sha256 contains duplicate paths")
    return tuple(records)


def load_prepared_image_records(path: Path) -> tuple[PreparedImageRecord, ...]:
    """Load the strict, sorted prepared-image inventory without following links."""

    payload, _ = _read_regular_file_without_links(
        path.parent,
        PurePosixPath(path.name),
        label="prepared inventory",
    )
    assert payload is not None
    return _parse_prepared_image_records(payload, path=path)


def _manifest_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, object]]:
    return {
        split: {
            "classes": dict(class_counts),
            "total": sum(class_counts.values()),
        }
        for split, class_counts in counts.items()
    }


def verify_prepared_artifact(
    root: Path,
    *,
    expected_preparation_key: str,
    expected_recipe: Mapping[str, object],
    source_archive: SourceArchive,
    source_lock: SourceLock,
    expected_counts: Mapping[str, Mapping[str, int]],
    full: bool = True,
) -> PreparedArtifact:
    """Verify manifest/layout identity and optionally every prepared file."""

    if type(full) is not bool:
        raise TypeError("full verification flag must be boolean")

    try:
        root = canonical_directory(root, label="prepared artifact root")
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise PreparationError(
            "prepared artifact root contains a symlink, reparse point, "
            f"invalid directory, or missing path: {root}"
        ) from error
    if full:
        _validate_prepared_root_layout(root)
    manifest_path = root / "dataset_manifest.yaml"
    inventory_path = root / "files.sha256"
    manifest_bytes, _ = _read_regular_file_without_links(
        root,
        PurePosixPath("dataset_manifest.yaml"),
        label="prepared manifest",
    )
    assert manifest_bytes is not None
    manifest = _load_manifest_bytes(manifest_bytes, path=manifest_path)
    _require_exact_keys(
        manifest,
        expected={
            "schema_version",
            "dataset",
            "source",
            "preparation",
            "counts",
            "inventory",
            "artifact_digest",
        },
        label="prepared manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise PreparationError("prepared manifest schema_version must be 1")
    dataset = _require_mapping(
        manifest["dataset"],
        label="prepared manifest dataset",
    )
    preparation = _require_mapping(
        manifest["preparation"],
        label="prepared manifest preparation",
    )
    source = _require_mapping(
        manifest["source"],
        label="prepared manifest source",
    )
    inventory = _require_mapping(
        manifest["inventory"],
        label="prepared manifest inventory",
    )
    manifest_counts = _require_mapping(
        manifest["counts"],
        label="prepared manifest counts",
    )
    expected_dataset = {
        "name": "AFHQ-v2",
        "version": 2,
        "homepage": source_lock.homepage,
        "license": {
            "name": source_lock.license_name,
            "url": source_lock.license_url,
        },
        "citation": source_lock.citation,
        "class_mapping": dict(source_lock.contract.class_mapping),
    }
    if dict(dataset) != expected_dataset:
        raise PreparationError("prepared manifest dataset metadata was modified")
    _require_exact_keys(
        preparation,
        expected={"key", "recipe", "recipe_sha256"},
        label="prepared manifest preparation",
    )
    if preparation.get("key") != expected_preparation_key:
        raise PreparationError("prepared manifest has the wrong preparation key")
    manifest_recipe = _require_mapping(
        preparation["recipe"],
        label="prepared manifest preparation.recipe",
    )
    expected_recipe_hash = _canonical_digest(expected_recipe)
    actual_recipe_hash = _canonical_digest(manifest_recipe)
    if (
        dict(manifest_recipe) != dict(expected_recipe)
        or actual_recipe_hash != expected_recipe_hash
        or preparation.get("recipe_sha256") != expected_recipe_hash
    ):
        raise PreparationError("prepared manifest has the wrong recipe digest")
    _require_exact_keys(
        source,
        expected={
            "type",
            "url",
            "archive",
            "source_splits",
            "source_class_counts",
            "total_count",
            "canonical_rgb_inventory_sha256",
        },
        label="prepared manifest source",
    )
    archive = _require_mapping(
        source["archive"],
        label="prepared manifest source.archive",
    )
    expected_source = {
        "type": "official_archive",
        "url": source_lock.url,
        "archive": {
            "name": source_lock.archive_name,
            "sha256": source_archive.sha256,
            "bytes": source_archive.size_bytes,
        },
        "source_splits": {
            "train": source_lock.contract.train_count,
            "test": source_lock.contract.test_count,
        },
        "source_class_counts": {
            split: dict(counts)
            for split, counts in (
                source_lock.contract.source_class_counts or {}
            ).items()
        },
        "total_count": source_lock.contract.total_count,
    }
    actual_source = dict(source)
    source_inventory_digest = actual_source.pop(
        "canonical_rgb_inventory_sha256",
        None,
    )
    if (
        not isinstance(source_inventory_digest, str)
        or _SHA256_PATTERN.fullmatch(source_inventory_digest) is None
    ):
        raise PreparationError(
            "prepared manifest has an invalid source inventory digest"
        )
    if actual_source != expected_source:
        raise PreparationError("prepared manifest source metadata was modified")
    _require_exact_keys(
        archive,
        expected={"name", "sha256", "bytes"},
        label="prepared manifest source.archive",
    )

    expected_manifest_counts = _manifest_counts(expected_counts)
    if dict(manifest_counts) != expected_manifest_counts:
        raise PreparationError("prepared manifest counts do not match the recipe")
    _require_exact_keys(
        inventory,
        expected={"path", "file_count", "sha256"},
        label="prepared manifest inventory",
    )
    if inventory["path"] != "files.sha256":
        raise PreparationError("prepared manifest has the wrong inventory path")
    inventory_bytes, _ = _read_regular_file_without_links(
        root,
        PurePosixPath("files.sha256"),
        label="prepared inventory",
    )
    assert inventory_bytes is not None
    inventory_digest = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory.get("sha256") != inventory_digest:
        raise PreparationError("files.sha256 digest does not match the manifest")
    records = _parse_prepared_image_records(
        inventory_bytes,
        path=inventory_path,
    )
    expected_count = source_lock.contract.total_count
    if len(records) != expected_count:
        raise PreparationError(
            f"prepared file count mismatch: {len(records)} != {expected_count}"
        )
    if inventory["file_count"] != expected_count:
        raise PreparationError("prepared manifest has the wrong inventory file count")

    actual_counts = {
        split: dict.fromkeys(source_lock.contract.classes, 0)
        for split in ("train", "test")
    }
    for record in records:
        _validate_relative_path(
            PurePosixPath(record.relative_path),
            label="prepared inventory record",
        )
        parts = PurePosixPath(record.relative_path).parts
        if (
            len(parts) != 3
            or parts[0] not in actual_counts
            or parts[1] not in source_lock.contract.classes
            or not parts[2].endswith(".png")
        ):
            raise PreparationError(
                f"invalid prepared inventory path: {record.relative_path!r}"
            )
        actual_counts[parts[0]][parts[1]] += 1
    if actual_counts != {
        split: dict(class_counts)
        for split, class_counts in expected_counts.items()
    }:
        raise PreparationError(
            "prepared inventory split/class counts do not match the recipe"
        )

    listed_paths = {record.relative_path for record in records}
    actual_paths: set[str] = set()
    for split in ("train", "test"):
        actual_paths.update(
            _enumerate_regular_files_without_links(
                root,
                PurePosixPath(split),
            )
        )
    if actual_paths != listed_paths:
        missing = sorted(listed_paths - actual_paths)
        unexpected = sorted(actual_paths - listed_paths)
        raise PreparationError(
            "prepared files do not match files.sha256; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for record in records:
        payload, metadata = _read_regular_file_without_links(
            root,
            PurePosixPath(record.relative_path),
            label="prepared image",
            read_content=full,
        )
        if metadata.st_size != record.size_bytes:
            raise PreparationError(
                f"prepared file size mismatch: {record.relative_path!r}"
            )
        if full:
            assert payload is not None
            actual_digest = hashlib.sha256(payload).hexdigest()
            if actual_digest != record.sha256:
                raise PreparationError(
                    f"prepared file digest mismatch: {record.relative_path!r}"
                )

    artifact_digest = _canonical_digest(
        {
            "inventory_sha256": inventory_digest,
            "recipe_sha256": expected_recipe_hash,
        }
    )
    if manifest.get("artifact_digest") != artifact_digest:
        raise PreparationError("prepared artifact digest does not match its content")
    return PreparedArtifact(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        artifact_digest=artifact_digest,
        preparation_key=expected_preparation_key,
        file_count=len(records),
        image_records=records,
        cache_hit=True,
    )
