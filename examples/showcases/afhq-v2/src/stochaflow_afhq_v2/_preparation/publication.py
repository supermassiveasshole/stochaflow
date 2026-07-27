"""Process and atomically publish a prepared AFHQ-v2 artifact."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from stochaflow.data.artifact_io import (
    cache_entry_exists,
    create_cache_directory,
    ensure_cache_directory,
    publish_cache_directory,
    quarantine_cache_entry,
    remove_cache_directory,
)

from .archive import _inspect_archive_stream
from .contracts import (
    PreparationError,
    PreparedArtifact,
    PreparedImageRecord,
    SourceArchive,
    SourceImage,
    SourceIntegrityError,
    SourceLock,
)
from .identity import _canonical_digest
from .image_transform import (
    _decode_and_resize,
    _manifest_text,
    _prepared_counts,
    _save_prepared_png,
    _write_text_atomic,
)
from .locking import ArtifactPreparationLock
from .planning import build_preparation_plan
from .prepared_artifact import _manifest_counts, verify_prepared_artifact
from .source_session import SourceArchiveSession


def _process_images(
    *,
    archive_stream: BinaryIO,
    archive_label: str,
    images: Sequence[SourceImage],
    staging_root: Path,
    cache_root: Path,
    input_resolution: int,
    output_resolution: int,
) -> tuple[list[PreparedImageRecord], str]:
    output_records: list[PreparedImageRecord] = []
    source_pixel_records: list[tuple[str, str]] = []
    try:
        archive_stream.seek(0)
        with ZipFile(archive_stream) as archive:
            for index, image in enumerate(images, start=1):
                try:
                    payload = archive.read(image.member_name)
                except (BadZipFile, KeyError, RuntimeError) as error:
                    raise PreparationError(
                        f"failed ZIP integrity check for {image.member_name!r}"
                    ) from error
                if len(payload) != image.file_size:
                    raise PreparationError(
                        f"source member size changed: {image.member_name!r}"
                    )
                prepared, pixel_digest = _decode_and_resize(
                    payload,
                    member_name=image.member_name,
                    input_resolution=input_resolution,
                    output_resolution=output_resolution,
                )
                output_split = image.source_split
                relative = f"{output_split}/{image.class_name}/{image.filename}"
                destination = staging_root / PurePosixPath(relative)
                output_digest, output_size = _save_prepared_png(
                    prepared,
                    destination,
                    cache_root=cache_root,
                )
                output_records.append(
                    PreparedImageRecord(
                        relative_path=relative,
                        size_bytes=output_size,
                        sha256=output_digest,
                    )
                )
                source_pixel_records.append((pixel_digest, image.relative_path))
                if index % 250 == 0 or index == len(images):
                    print(
                        f"Prepared {index:,}/{len(images):,} images",
                        flush=True,
                    )
    except BadZipFile as error:
        raise PreparationError(
            f"archive became invalid while preparing: {archive_label}"
        ) from error

    output_records.sort(key=lambda item: item.relative_path)
    source_pixel_records.sort(key=lambda item: item[1])
    source_inventory_digest = _canonical_digest(
        [
            {"path": relative, "rgb_sha256": digest}
            for digest, relative in source_pixel_records
        ]
    )
    return output_records, source_inventory_digest


def _quarantine_invalid_prepared_artifact(
    root: Path,
    *,
    cache_root: Path,
) -> Path:
    try:
        return quarantine_cache_entry(
            cache_root,
            root,
            suffix=f"{time.time_ns()}.invalid",
            label="invalid AFHQ-v2 prepared artifact",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot quarantine invalid prepared artifact: {root}"
        ) from error


def prepare_archive(
    *,
    source: SourceArchive,
    lock: SourceLock,
    cache_root: Path,
    resolution: int = 128,
    repair_invalid: bool = False,
) -> PreparedArtifact:
    """Validate and deterministically turn AFHQ-v2 into a training artifact."""

    if type(repair_invalid) is not bool:
        raise TypeError("repair_invalid must be boolean")
    plan = build_preparation_plan(
        lock=lock,
        resolution=resolution,
    )
    if (
        source.sha256 != lock.expected_sha256
        or source.size_bytes != lock.expected_bytes
    ):
        raise SourceIntegrityError(
            "source archive identity does not match the source lock",
            actual_sha256=source.sha256,
            actual_bytes=source.size_bytes,
        )
    recipe = plan.recipe
    recipe_hash = plan.recipe_sha256
    preparation_key = plan.preparation_key
    try:
        canonical_cache_root = ensure_cache_directory(
            cache_root,
            cache_root,
            label="AFHQ-v2 artifact cache root",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot create AFHQ-v2 artifact cache: {cache_root}"
        ) from error
    prepared_base = (
        canonical_cache_root / "prepared" / "afhq-v2" / str(resolution)
    )
    final_root = prepared_base / preparation_key
    lock_path = (
        canonical_cache_root
        / ".locks"
        / f"afhq-v2-prepare-{preparation_key}.lock"
    )

    with SourceArchiveSession(source, lock=lock) as source_session:
        archive_stream = source_session.stream
        assert archive_stream is not None
        images = _inspect_archive_stream(
            archive_stream,
            archive_label=str(source.path),
            contract=lock.contract,
        )
        counts = _prepared_counts(
            images,
            classes=lock.contract.classes,
        )
        if counts != plan.counts:
            raise PreparationError(
                "source archive class counts do not match the pinned "
                "preparation plan"
            )

        with ArtifactPreparationLock(
            lock_path,
            cache_root=canonical_cache_root,
        ):
            try:
                final_exists = cache_entry_exists(
                    canonical_cache_root,
                    final_root,
                    label="AFHQ-v2 prepared artifact",
                )
            except (OSError, ValueError) as error:
                raise PreparationError(
                    "cannot inspect AFHQ-v2 prepared artifact directory: "
                    f"{final_root}"
                ) from error
            if final_exists:
                try:
                    cached = verify_prepared_artifact(
                        final_root,
                        expected_preparation_key=preparation_key,
                        expected_recipe=recipe,
                        source_archive=source,
                        source_lock=lock,
                        expected_counts=counts,
                    )
                except PreparationError:
                    if not repair_invalid:
                        raise
                    _quarantine_invalid_prepared_artifact(
                        final_root,
                        cache_root=canonical_cache_root,
                    )
                else:
                    source_session.verify_unchanged()
                    return cached

            try:
                ensure_cache_directory(
                    canonical_cache_root,
                    prepared_base,
                    label="AFHQ-v2 prepared artifact directory",
                )
            except (OSError, ValueError) as error:
                raise PreparationError(
                    f"cannot create AFHQ-v2 prepared artifact directory: "
                    f"{prepared_base}"
                ) from error
            staging_root = prepared_base / (
                f".{preparation_key}.tmp-{os.getpid()}-{uuid4().hex}"
            )
            staging_created = False
            try:
                create_cache_directory(
                    canonical_cache_root,
                    staging_root,
                    label="AFHQ-v2 prepared artifact staging",
                )
                staging_created = True
                output_records, source_inventory_digest = _process_images(
                    archive_stream=archive_stream,
                    archive_label=str(source.path),
                    images=images,
                    staging_root=staging_root,
                    cache_root=canonical_cache_root,
                    input_resolution=lock.contract.input_resolution,
                    output_resolution=resolution,
                )
                inventory_text = "".join(
                    f"{record.sha256}  {record.size_bytes}  "
                    f"{record.relative_path}\n"
                    for record in output_records
                )
                inventory_path = staging_root / "files.sha256"
                _write_text_atomic(
                    inventory_path,
                    inventory_text,
                    cache_root=canonical_cache_root,
                    label="AFHQ-v2 prepared inventory",
                )
                inventory_digest = hashlib.sha256(
                    inventory_text.encode("utf-8")
                ).hexdigest()
                artifact_digest = _canonical_digest(
                    {
                        "inventory_sha256": inventory_digest,
                        "recipe_sha256": recipe_hash,
                    },
                )
                manifest: dict[str, object] = {
                    "schema_version": 1,
                    "dataset": {
                        "name": "AFHQ-v2",
                        "version": 2,
                        "homepage": lock.homepage,
                        "license": {
                            "name": lock.license_name,
                            "url": lock.license_url,
                        },
                        "citation": lock.citation,
                        "class_mapping": dict(lock.contract.class_mapping),
                    },
                    "source": {
                        "type": "official_archive",
                        "url": lock.url,
                        "archive": {
                            "name": lock.archive_name,
                            "sha256": source.sha256,
                            "bytes": source.size_bytes,
                        },
                        "source_splits": {
                            "train": lock.contract.train_count,
                            "test": lock.contract.test_count,
                        },
                        "source_class_counts": {
                            split: dict(source_counts)
                            for split, source_counts in (
                                lock.contract.source_class_counts or {}
                            ).items()
                        },
                        "total_count": lock.contract.total_count,
                        "canonical_rgb_inventory_sha256": (
                            source_inventory_digest
                        ),
                    },
                    "preparation": {
                        "key": preparation_key,
                        "recipe": recipe,
                        "recipe_sha256": recipe_hash,
                    },
                    "counts": _manifest_counts(counts),
                    "inventory": {
                        "path": "files.sha256",
                        "file_count": len(output_records),
                        "sha256": inventory_digest,
                    },
                    "artifact_digest": artifact_digest,
                }
                manifest_path = staging_root / "dataset_manifest.yaml"
                _write_text_atomic(
                    manifest_path,
                    _manifest_text(manifest),
                    cache_root=canonical_cache_root,
                    label="AFHQ-v2 prepared manifest",
                )
                source_session.verify_unchanged()
                try:
                    publish_cache_directory(
                        canonical_cache_root,
                        staging_root,
                        final_root,
                        label="AFHQ-v2 prepared artifact",
                    )
                except FileExistsError:
                    remove_cache_directory(
                        canonical_cache_root,
                        staging_root,
                        label="AFHQ-v2 losing prepared artifact staging",
                    )
                    return verify_prepared_artifact(
                        final_root,
                        expected_preparation_key=preparation_key,
                        expected_recipe=recipe,
                        source_archive=source,
                        source_lock=lock,
                        expected_counts=counts,
                    )
                published = verify_prepared_artifact(
                    final_root,
                    expected_preparation_key=preparation_key,
                    expected_recipe=recipe,
                    source_archive=source,
                    source_lock=lock,
                    expected_counts=counts,
                )
                return PreparedArtifact(
                    root=published.root,
                    manifest_path=published.manifest_path,
                    manifest_sha256=published.manifest_sha256,
                    artifact_digest=published.artifact_digest,
                    preparation_key=published.preparation_key,
                    file_count=published.file_count,
                    image_records=published.image_records,
                    cache_hit=False,
                )
            except BaseException:
                if staging_created and cache_entry_exists(
                    canonical_cache_root,
                    staging_root,
                    label="AFHQ-v2 prepared artifact staging cleanup",
                ):
                    remove_cache_directory(
                        canonical_cache_root,
                        staging_root,
                        label="AFHQ-v2 prepared artifact staging cleanup",
                    )
                raise
