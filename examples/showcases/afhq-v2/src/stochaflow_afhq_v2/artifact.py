"""Managed AFHQ-v2 producer built on the public artifact store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast
from zipfile import BadZipFile, ZipFile

from stochaflow.extensions import (
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    DataArtifact,
    DataArtifactLoadContext,
    DataArtifactStore,
    DataArtifactValidationError,
    DataSourceContext,
    ImageFileRecord,
    ManagedDataArtifactBuild,
    canonical_artifact_json_bytes,
)
from stochaflow_afhq_v2._preparation.archive import _inspect_archive_stream
from stochaflow_afhq_v2._preparation.contracts import (
    PreparationError,
    SourceArchive,
    SourceImage,
    SourceLock,
)
from stochaflow_afhq_v2._preparation.image_transform import (
    decode_and_resize,
    save_prepared_png,
)
from stochaflow_afhq_v2._preparation.materialization import (
    AFHQV2_MATERIALIZER_NAME,
    build_materialization_spec,
)
from stochaflow_afhq_v2._preparation.source_acquisition import (
    acquire_official_archive,
)
from stochaflow_afhq_v2._preparation.source_lock import load_source_lock
from stochaflow_afhq_v2._preparation.source_session import SourceArchiveSession

AFHQV2_SOURCE_NAME = "afhq-v2.official"
AFHQV2_ARTIFACT_TYPE = "stochaflow.class-labeled-image-folder.v1"
AFHQV2_CLASS_MAPPING: Mapping[str, int] = MappingProxyType(
    {"cat": 0, "dog": 1, "wild": 2}
)
DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parent / "resources" / "afhq-v2.lock.yaml"
)

_INDEX_PATH = PurePosixPath("_index/images.json")
_DOMAIN_FIELDS = frozenset(
    {
        "schema_version",
        "resolution",
        "class_mapping",
        "partitions",
        "index",
    }
)
_PARTITION_FIELDS = frozenset({"root", "count"})
_INDEX_FIELDS = frozenset({"path", "size_bytes", "sha256", "record_count"})
_RECORD_FIELDS = frozenset(
    {"partition", "path", "size_bytes", "sha256", "class_label"}
)


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataArtifactValidationError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise DataArtifactValidationError(
            f"{label} field names must be strings"
        )
    names = set(cast(dict[str, object], value))
    if names != fields:
        raise DataArtifactValidationError(
            f"{label} must contain exactly {sorted(fields)}"
        )
    return cast(dict[str, Any], value)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataArtifactValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _positive_int(value: object, *, label: str, allow_zero: bool = False) -> int:
    lower_bound = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise DataArtifactValidationError(f"{label} must be a {qualifier} integer")
    return value


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise DataArtifactValidationError(
            f"{label} must be a non-empty POSIX relative path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DataArtifactValidationError(
            f"{label} must be a safe POSIX relative path"
        )
    if path.as_posix() != value:
        raise DataArtifactValidationError(f"{label} is not canonical")
    return path


def _validate_lock(lock: SourceLock) -> str:
    if dict(lock.contract.class_mapping) != dict(AFHQV2_CLASS_MAPPING):
        raise PreparationError(
            "AFHQ-v2 source lock class mapping must be cat=0, dog=1, wild=2"
        )
    if lock.expected_sha256 is None:
        raise PreparationError("AFHQ-v2 source lock is missing SHA-256")
    return lock.expected_sha256


def _process_images(
    *,
    source: SourceArchive,
    lock: SourceLock,
    data_root: Path,
    resolution: int,
) -> tuple[dict[str, object], ...]:
    with SourceArchiveSession(source, lock=lock) as session:
        stream = session.stream
        assert stream is not None
        images = _inspect_archive_stream(
            stream,
            archive_label=str(source.path),
            contract=lock.contract,
        )
        records: list[dict[str, object]] = []
        try:
            stream.seek(0)
            with ZipFile(stream) as archive:
                for index, image in enumerate(images, start=1):
                    records.append(
                        _process_image(
                            archive=archive,
                            image=image,
                            data_root=data_root,
                            input_resolution=lock.contract.input_resolution,
                            output_resolution=resolution,
                        )
                    )
                    if index % 250 == 0 or index == len(images):
                        print(
                            f"Prepared {index:,}/{len(images):,} images",
                            flush=True,
                        )
        except BadZipFile as error:
            raise PreparationError(
                f"archive became invalid while preparing: {source.path}"
            ) from error
        session.verify_unchanged()
    records.sort(
        key=lambda item: (
            cast(str, item["partition"]),
            cast(str, item["path"]),
        )
    )
    return tuple(records)


def _process_image(
    *,
    archive: ZipFile,
    image: SourceImage,
    data_root: Path,
    input_resolution: int,
    output_resolution: int,
) -> dict[str, object]:
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
    prepared = decode_and_resize(
        payload,
        member_name=image.member_name,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
    )
    relative = PurePosixPath(image.class_name, image.filename)
    destination = data_root / image.source_split / relative
    digest, size_bytes = save_prepared_png(prepared, destination)
    return {
        "partition": image.source_split,
        "path": relative.as_posix(),
        "size_bytes": size_bytes,
        "sha256": digest,
        "class_label": AFHQV2_CLASS_MAPPING[image.class_name],
    }


def _domain(
    *,
    resolution: int,
    records: Sequence[Mapping[str, object]],
    index_bytes: bytes,
) -> dict[str, object]:
    counts = {"train": 0, "test": 0}
    for record in records:
        counts[cast(str, record["partition"])] += 1
    return {
        "schema_version": 1,
        "resolution": resolution,
        "class_mapping": dict(AFHQV2_CLASS_MAPPING),
        "partitions": {
            role: {"root": role, "count": counts[role]}
            for role in ("train", "test")
        },
        "index": {
            "path": _INDEX_PATH.as_posix(),
            "size_bytes": len(index_bytes),
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
            "record_count": len(records),
        },
    }


def _build_artifact(
    data_root: Path,
    *,
    acquisition_root: Path,
    lock: SourceLock,
    source_digest: str,
    materialization_digest: str,
    archive: Path | None,
    downloader: Literal["auto", "curl", "python"],
    resolution: int,
) -> ManagedDataArtifactBuild:
    source = acquire_official_archive(
        lock=lock,
        cache_root=acquisition_root,
        proxy=None,
        archive_override=archive,
        downloader=downloader,
    )
    records = _process_images(
        source=source,
        lock=lock,
        data_root=data_root,
        resolution=resolution,
    )

    index_bytes = canonical_artifact_json_bytes(
        {"schema_version": 1, "records": list(records)}
    )
    index_path = data_root / _INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(index_bytes)
    return ManagedDataArtifactBuild(
        source_digest=source_digest,
        materialization_digest=materialization_digest,
        domain=_domain(
            resolution=resolution,
            records=records,
            index_bytes=index_bytes,
        ),
    )


def _load_payload(
    context: DataArtifactLoadContext,
    *,
    source_digest: str,
    materialization_digest: str,
    resolution: int,
) -> ClassLabeledImageFolderArtifactPayload:
    if context.identity.source_digest != source_digest:
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact source identity does not match the current source lock"
        )
    if context.identity.materialization_digest != materialization_digest:
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact materialization identity does not match "
            "the current recipe"
        )
    domain = _strict_mapping(
        dict(context.domain),
        fields=_DOMAIN_FIELDS,
        label="AFHQ-v2 artifact domain",
    )
    if type(domain["schema_version"]) is not int or domain["schema_version"] != 1:
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact domain.schema_version must be 1"
        )
    domain_resolution = _positive_int(
        domain["resolution"],
        label="AFHQ-v2 artifact domain.resolution",
    )
    if domain_resolution != resolution:
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact resolution does not match the current recipe"
        )
    class_mapping = domain["class_mapping"]
    if (
        not isinstance(class_mapping, dict)
        or set(class_mapping) != set(AFHQV2_CLASS_MAPPING)
        or any(not isinstance(key, str) for key in class_mapping)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in class_mapping.values()
        )
        or class_mapping != dict(AFHQV2_CLASS_MAPPING)
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact class mapping does not match cat=0, dog=1, wild=2"
        )

    partitions_raw = _strict_mapping(
        domain["partitions"],
        fields=frozenset({"train", "test"}),
        label="AFHQ-v2 artifact domain.partitions",
    )
    partition_counts: dict[str, int] = {}
    roots: dict[str, Path] = {}
    for role in ("train", "test"):
        partition = _strict_mapping(
            partitions_raw[role],
            fields=_PARTITION_FIELDS,
            label=f"AFHQ-v2 artifact domain.partitions.{role}",
        )
        root_path = _safe_relative_path(
            partition["root"],
            label=f"AFHQ-v2 artifact domain.partitions.{role}.root",
        )
        if root_path.as_posix() != role:
            raise DataArtifactValidationError(
                f"AFHQ-v2 {role} partition root must be {role!r}"
            )
        roots[role] = context.data_root / root_path
        partition_counts[role] = _positive_int(
            partition["count"],
            label=f"AFHQ-v2 artifact domain.partitions.{role}.count",
            allow_zero=role == "test",
        )

    index = _strict_mapping(
        domain["index"],
        fields=_INDEX_FIELDS,
        label="AFHQ-v2 artifact domain.index",
    )
    index_path = _safe_relative_path(
        index["path"],
        label="AFHQ-v2 artifact domain.index.path",
    )
    if index_path != _INDEX_PATH:
        raise DataArtifactValidationError(
            f"AFHQ-v2 artifact index path must be {_INDEX_PATH.as_posix()!r}"
        )
    expected_size = _positive_int(
        index["size_bytes"],
        label="AFHQ-v2 artifact domain.index.size_bytes",
    )
    expected_digest = _sha256(
        index["sha256"],
        label="AFHQ-v2 artifact domain.index.sha256",
    )
    expected_count = _positive_int(
        index["record_count"],
        label="AFHQ-v2 artifact domain.index.record_count",
    )
    try:
        encoded = (context.data_root / index_path).read_bytes()
    except OSError as error:
        raise DataArtifactValidationError(
            "cannot read AFHQ-v2 artifact image index"
        ) from error
    if (
        len(encoded) != expected_size
        or hashlib.sha256(encoded).hexdigest() != expected_digest
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index does not match its descriptor"
        )
    try:
        raw_index = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index is not valid UTF-8 JSON"
        ) from error
    if encoded != canonical_artifact_json_bytes(raw_index):
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index is not canonical JSON"
        )
    index_mapping = _strict_mapping(
        raw_index,
        fields=frozenset({"schema_version", "records"}),
        label="AFHQ-v2 artifact image index",
    )
    if (
        type(index_mapping["schema_version"]) is not int
        or index_mapping["schema_version"] != 1
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index schema_version must be 1"
        )
    records_raw = index_mapping["records"]
    if not isinstance(records_raw, list):
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index records must be a list"
        )
    if len(records_raw) != expected_count:
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index record count changed"
        )

    partitions: dict[str, list[ClassLabeledImageFileRecord]] = {
        "train": [],
        "test": [],
    }
    ordering: list[tuple[str, str]] = []
    for record_index, value in enumerate(records_raw):
        record = _strict_mapping(
            value,
            fields=_RECORD_FIELDS,
            label=f"AFHQ-v2 artifact image index records[{record_index}]",
        )
        role = record["partition"]
        if role not in partitions:
            raise DataArtifactValidationError(
                "AFHQ-v2 artifact image record has an invalid partition"
            )
        path = _safe_relative_path(
            record["path"],
            label=f"AFHQ-v2 artifact image index records[{record_index}].path",
        )
        if len(path.parts) != 2 or path.parts[0] not in AFHQV2_CLASS_MAPPING:
            raise DataArtifactValidationError(
                "AFHQ-v2 artifact image record has an invalid class path"
            )
        class_label = record["class_label"]
        if (
            isinstance(class_label, bool)
            or not isinstance(class_label, int)
            or class_label != AFHQV2_CLASS_MAPPING[path.parts[0]]
        ):
            raise DataArtifactValidationError(
                "AFHQ-v2 artifact image record has an invalid class label"
            )
        key = (cast(str, role), path.as_posix())
        ordering.append(key)
        partitions[cast(str, role)].append(
            ClassLabeledImageFileRecord(
                image=ImageFileRecord(
                    tree=cast(str, role),
                    path=path.as_posix(),
                    size_bytes=_positive_int(
                        record["size_bytes"],
                        label="AFHQ-v2 artifact image record size_bytes",
                    ),
                    sha256=_sha256(
                        record["sha256"],
                        label="AFHQ-v2 artifact image record sha256",
                    ),
                    width=domain_resolution,
                    height=domain_resolution,
                ),
                class_label=class_label,
            )
        )
    if ordering != sorted(ordering) or len(ordering) != len(set(ordering)):
        raise DataArtifactValidationError(
            "AFHQ-v2 artifact image index must be sorted and unique"
        )
    for role in ("train", "test"):
        if len(partitions[role]) != partition_counts[role]:
            raise DataArtifactValidationError(
                f"AFHQ-v2 artifact {role} image count changed"
            )
    return ClassLabeledImageFolderArtifactPayload(
        roots=roots,
        class_mapping=AFHQV2_CLASS_MAPPING,
        train=tuple(partitions["train"]),
        test=tuple(partitions["test"]),
    )


def materialize_afhq_v2_artifact(
    context: DataSourceContext,
    *,
    archive: Path | None = None,
    downloader: Literal["auto", "curl", "python"] = "auto",
    lock_file: Path | None = None,
    resolution: int = 128,
) -> DataArtifact[ClassLabeledImageFolderArtifactPayload]:
    """Materialize or load source-locked AFHQ-v2 through the public store."""

    lock = load_source_lock(lock_file or DEFAULT_LOCK_PATH)
    source_digest = _validate_lock(lock)
    spec = build_materialization_spec(lock=lock, resolution=resolution)
    expected = context.expected_identity
    if expected is not None:
        if expected.source_digest != source_digest:
            raise DataArtifactValidationError(
                "strict resume AFHQ-v2 source identity does not match "
                "the current source lock"
            )
        if expected.materialization_digest != spec.digest:
            raise DataArtifactValidationError(
                "strict resume AFHQ-v2 materialization identity does not match "
                "the current recipe"
            )
    locator_key: dict[str, object] = {
        "source_digest": source_digest,
        "materialization_digest": spec.digest,
        "resolution": resolution,
    }
    return DataArtifactStore(context).materialize_managed(
        artifact_type=AFHQV2_ARTIFACT_TYPE,
        source_name=AFHQV2_SOURCE_NAME,
        materializer_name=AFHQV2_MATERIALIZER_NAME,
        locator_key=locator_key,
        build=lambda data_root: _build_artifact(
            data_root,
            acquisition_root=(
                context.cache_root / "source-acquisition" / "afhq-v2"
            ),
            lock=lock,
            source_digest=source_digest,
            materialization_digest=spec.digest,
            archive=archive,
            downloader=downloader,
            resolution=resolution,
        ),
        load=lambda load_context: _load_payload(
            load_context,
            source_digest=source_digest,
            materialization_digest=spec.digest,
            resolution=resolution,
        ),
    )


__all__ = [
    "AFHQV2_ARTIFACT_TYPE",
    "AFHQV2_CLASS_MAPPING",
    "AFHQV2_SOURCE_NAME",
    "DEFAULT_LOCK_PATH",
    "materialize_afhq_v2_artifact",
]
