"""Materialize the source-locked AFHQ-v2 image-folder artifact."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, cast

from stochaflow.extensions import (
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    DataSourceContext,
    ImageFileRecord,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
)
from stochaflow_afhq_v2 import preparation

AFHQV2_SOURCE_NAME = "afhq-v2.official"
AFHQV2_ARTIFACT_TYPE = "stochaflow.class-labeled-image-folder.v1"
AFHQV2_CLASS_MAPPING: Mapping[str, int] = MappingProxyType(
    {"cat": 0, "dog": 1, "wild": 2}
)
DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parent / "resources" / "afhq-v2.lock.yaml"
)


def _prepared_payload(
    prepared: preparation.PreparedArtifact,
    *,
    resolution: int,
) -> ClassLabeledImageFolderArtifactPayload:
    roots = {role: prepared.root / role for role in ("train", "test")}
    partitions: dict[str, list[ClassLabeledImageFileRecord]] = {
        role: [] for role in roots
    }
    for record in prepared.image_records:
        parts = PurePosixPath(record.relative_path).parts
        if (
            len(parts) != 3
            or parts[0] not in partitions
            or parts[1] not in AFHQV2_CLASS_MAPPING
        ):
            raise preparation.PreparationError(
                "prepared inventory has an invalid image path: "
                f"{record.relative_path!r}"
            )
        role = parts[0]
        partitions[role].append(
            ClassLabeledImageFileRecord(
                image=ImageFileRecord(
                    tree=role,
                    path=PurePosixPath(*parts[1:]).as_posix(),
                    size_bytes=record.size_bytes,
                    sha256=record.sha256,
                    width=resolution,
                    height=resolution,
                ),
                class_label=AFHQV2_CLASS_MAPPING[parts[1]],
            )
        )
    return ClassLabeledImageFolderArtifactPayload(
        roots=roots,
        class_mapping=AFHQV2_CLASS_MAPPING,
        train=tuple(partitions["train"]),
        test=tuple(partitions["test"]),
    )


def _expected_identity(
    context: DataSourceContext,
) -> ManagedDataArtifactIdentity | None:
    expected = context.expected_identity
    if expected is None:
        return None
    if not isinstance(expected, ManagedDataArtifactIdentity):
        raise TypeError(
            "AFHQ-v2 strict resume requires a managed data artifact identity"
        )
    if expected.source_name != AFHQV2_SOURCE_NAME:
        raise ValueError("AFHQ-v2 strict resume expected a different data source")
    if expected.artifact_type != AFHQV2_ARTIFACT_TYPE:
        raise ValueError(
            "AFHQ-v2 strict resume expected an incompatible artifact type"
        )
    return expected


def _preflight_expected_identity(
    expected: ManagedDataArtifactIdentity | None,
    *,
    source_digest: str,
    materializer_name: str,
    materialization_digest: str,
) -> None:
    if expected is None:
        return
    if expected.source_digest != source_digest:
        raise ValueError(
            "AFHQ-v2 strict resume expected a different source digest"
        )
    if expected.materializer_name != materializer_name:
        raise ValueError(
            "AFHQ-v2 strict resume expected a different materializer"
        )
    if expected.materialization_digest != materialization_digest:
        raise ValueError(
            "AFHQ-v2 strict resume expected a different preparation recipe"
        )


def materialize_afhq_v2_artifact(
    context: DataSourceContext,
    *,
    archive: Path | None = None,
    downloader: Literal["auto", "curl", "python"] = "auto",
    lock_file: Path | None = None,
    resolution: int = 128,
) -> ManagedDataArtifact[ClassLabeledImageFolderArtifactPayload]:
    """Prepare or verify AFHQ-v2 through the framework artifact lifecycle."""

    lock = preparation.load_source_lock(lock_file or DEFAULT_LOCK_PATH)
    if dict(lock.contract.class_mapping) != dict(AFHQV2_CLASS_MAPPING):
        raise preparation.PreparationError(
            "AFHQ-v2 source lock class mapping must be cat=0, dog=1, wild=2"
        )
    plan = preparation.build_preparation_plan(
        lock=lock,
        resolution=resolution,
    )
    if lock.expected_sha256 is None:
        raise preparation.PreparationError(
            "AFHQ-v2 source lock is missing SHA-256"
        )
    materializer_name = cast(str, plan.recipe["id"])
    expected = _expected_identity(context)
    _preflight_expected_identity(
        expected,
        source_digest=lock.expected_sha256,
        materializer_name=materializer_name,
        materialization_digest=plan.recipe_sha256,
    )

    prepared: preparation.PreparedArtifact | None
    try:
        prepared = preparation.require_prepared_artifact(
            lock=lock,
            cache_root=context.cache_root,
            resolution=resolution,
            full=context.verification == "full",
        )
    except preparation.PreparationError:
        if context.policy == "require":
            raise
        prepared = None

    if prepared is None:
        source = preparation.acquire_official_archive(
            lock=lock,
            cache_root=context.cache_root,
            proxy=None,
            archive_override=archive,
            downloader=downloader,
        )
        prepared = preparation.prepare_archive(
            source=source,
            lock=lock,
            cache_root=context.cache_root,
            resolution=resolution,
            repair_invalid=True,
        )

    identity = ManagedDataArtifactIdentity(
        artifact_type=AFHQV2_ARTIFACT_TYPE,
        source_name=AFHQV2_SOURCE_NAME,
        source_digest=lock.expected_sha256,
        materializer_name=materializer_name,
        materialization_digest=plan.recipe_sha256,
        artifact_digest=prepared.artifact_digest,
        manifest_sha256=prepared.manifest_sha256,
    )
    if expected is not None and identity != expected:
        raise ValueError(
            "AFHQ-v2 strict resume data artifact identity does not match"
        )
    return ManagedDataArtifact(
        artifact_root=prepared.root,
        manifest_path=prepared.manifest_path,
        identity=identity,
        payload=_prepared_payload(prepared, resolution=resolution),
    )


__all__ = [
    "AFHQV2_ARTIFACT_TYPE",
    "AFHQV2_CLASS_MAPPING",
    "AFHQV2_SOURCE_NAME",
    "DEFAULT_LOCK_PATH",
    "materialize_afhq_v2_artifact",
]
