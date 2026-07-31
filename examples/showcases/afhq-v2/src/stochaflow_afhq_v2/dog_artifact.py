"""Managed train/dog-only AFHQ-v2 artifact for the 256px benchmark."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast
from zipfile import BadZipFile, ZipFile

from stochaflow.extensions import (
    DataArtifact,
    DataArtifactLoadContext,
    DataArtifactStore,
    DataArtifactValidationError,
    DataSourceContext,
    ImageFileRecord,
    ImageFolderArtifactPayload,
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
from stochaflow_afhq_v2._preparation.dog_image_transform import (
    decode_and_center_crop,
    save_dog_png,
)
from stochaflow_afhq_v2._preparation.dog_materialization import (
    AFHQV2_DOG_MATERIALIZER_NAME,
    AFHQV2_DOG_RESOLUTION,
    build_dog_materialization_spec,
)
from stochaflow_afhq_v2._preparation.source_acquisition import (
    acquire_official_archive,
)
from stochaflow_afhq_v2._preparation.source_lock import load_source_lock
from stochaflow_afhq_v2._preparation.source_session import SourceArchiveSession

AFHQV2_DOG_SOURCE_NAME = "afhq-v2.dog"
AFHQV2_DOG_ARTIFACT_TYPE = "stochaflow.afhq-v2-dog-image-folder.v1"
DEFAULT_DOG_LOCK_PATH = (
    Path(__file__).resolve().parent / "resources" / "afhq-v2.lock.yaml"
)

_INDEX_PATH = PurePosixPath("_index/images.json")
_AFHQV2_CLASS_MAPPING = MappingProxyType({"cat": 0, "dog": 1, "wild": 2})
_DOMAIN_FIELDS = frozenset(
    {
        "schema_version",
        "resolution",
        "source_subset",
        "partition",
        "index",
    }
)
_SOURCE_SUBSET_FIELDS = frozenset({"partition", "class", "count"})
_PARTITION_FIELDS = frozenset({"root", "count"})
_INDEX_FIELDS = frozenset({"path", "size_bytes", "sha256", "record_count"})


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


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataArtifactValidationError(f"{label} must be a positive integer")
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
    if dict(lock.contract.class_mapping) != dict(_AFHQV2_CLASS_MAPPING):
        raise PreparationError(
            "AFHQ-v2 source lock class mapping must be cat=0, dog=1, wild=2"
        )
    if lock.expected_sha256 is None:
        raise PreparationError("AFHQ-v2 source lock is missing SHA-256")
    source_counts = lock.contract.source_class_counts
    if source_counts is None or "dog" not in source_counts.get("train", {}):
        raise PreparationError(
            "AFHQ-v2 source lock is missing the train/dog source count"
        )
    return lock.expected_sha256


def _process_dog_image(
    *,
    archive: ZipFile,
    image: SourceImage,
    data_root: Path,
    input_resolution: int,
) -> ImageFileRecord:
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
    prepared = decode_and_center_crop(
        payload,
        member_name=image.member_name,
        expected_input_size=(input_resolution, input_resolution),
        output_resolution=AFHQV2_DOG_RESOLUTION,
    )
    relative = PurePosixPath("dog", image.filename)
    digest, size_bytes = save_dog_png(
        prepared,
        data_root / "train" / relative,
    )
    return ImageFileRecord(
        tree="train",
        path=relative.as_posix(),
        size_bytes=size_bytes,
        sha256=digest,
        width=AFHQV2_DOG_RESOLUTION,
        height=AFHQV2_DOG_RESOLUTION,
    )


def _process_dog_images(
    *,
    source: SourceArchive,
    lock: SourceLock,
    data_root: Path,
) -> tuple[ImageFileRecord, ...]:
    source_counts = lock.contract.source_class_counts
    assert source_counts is not None
    expected_count = source_counts["train"]["dog"]
    with SourceArchiveSession(source, lock=lock) as session:
        stream = session.stream
        assert stream is not None
        inventory = _inspect_archive_stream(
            stream,
            archive_label=str(source.path),
            contract=lock.contract,
        )
        dog_images = tuple(
            image
            for image in inventory
            if image.source_split == "train" and image.class_name == "dog"
        )
        if len(dog_images) != expected_count:
            raise PreparationError(
                "authenticated AFHQ-v2 archive train/dog count changed"
            )
        records: list[ImageFileRecord] = []
        try:
            stream.seek(0)
            with ZipFile(stream) as archive:
                for index, image in enumerate(dog_images, start=1):
                    records.append(
                        _process_dog_image(
                            archive=archive,
                            image=image,
                            data_root=data_root,
                            input_resolution=lock.contract.input_resolution,
                        )
                    )
                    if index % 250 == 0 or index == len(dog_images):
                        print(
                            "Prepared "
                            f"{index:,}/{len(dog_images):,} AFHQ-v2 Dog images",
                            flush=True,
                        )
        except BadZipFile as error:
            raise PreparationError(
                f"archive became invalid while preparing: {source.path}"
            ) from error
        session.verify_unchanged()
    records.sort(key=lambda record: record.path)
    return tuple(records)


def _domain(
    *,
    records: Sequence[ImageFileRecord],
    index_bytes: bytes,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "resolution": AFHQV2_DOG_RESOLUTION,
        "source_subset": {
            "partition": "train",
            "class": "dog",
            "count": len(records),
        },
        "partition": {
            "root": "train",
            "count": len(records),
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
) -> ManagedDataArtifactBuild:
    source = acquire_official_archive(
        lock=lock,
        cache_root=acquisition_root,
        proxy=None,
        archive_override=archive,
        downloader=downloader,
    )
    records = _process_dog_images(
        source=source,
        lock=lock,
        data_root=data_root,
    )
    index_bytes = canonical_artifact_json_bytes(
        {
            "schema_version": 1,
            "records": [record.to_dict() for record in records],
        }
    )
    index_path = data_root / _INDEX_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(index_bytes)
    return ManagedDataArtifactBuild(
        source_digest=source_digest,
        materialization_digest=materialization_digest,
        domain=_domain(records=records, index_bytes=index_bytes),
    )


def _load_payload(
    context: DataArtifactLoadContext,
    *,
    source_digest: str,
    materialization_digest: str,
) -> ImageFolderArtifactPayload:
    if context.identity.source_digest != source_digest:
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact source identity does not match "
            "the current source lock"
        )
    if context.identity.materialization_digest != materialization_digest:
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact materialization identity does not match "
            "the current recipe"
        )
    domain = _strict_mapping(
        dict(context.domain),
        fields=_DOMAIN_FIELDS,
        label="AFHQ-v2 Dog artifact domain",
    )
    if type(domain["schema_version"]) is not int or domain["schema_version"] != 1:
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact domain.schema_version must be 1"
        )
    resolution = _positive_int(
        domain["resolution"],
        label="AFHQ-v2 Dog artifact domain.resolution",
    )
    if resolution != AFHQV2_DOG_RESOLUTION:
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact resolution must be 256"
        )
    source_subset = _strict_mapping(
        domain["source_subset"],
        fields=_SOURCE_SUBSET_FIELDS,
        label="AFHQ-v2 Dog artifact domain.source_subset",
    )
    if (
        source_subset["partition"] != "train"
        or source_subset["class"] != "dog"
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact source subset must be train/dog"
        )
    source_count = _positive_int(
        source_subset["count"],
        label="AFHQ-v2 Dog artifact domain.source_subset.count",
    )
    partition = _strict_mapping(
        domain["partition"],
        fields=_PARTITION_FIELDS,
        label="AFHQ-v2 Dog artifact domain.partition",
    )
    partition_root = _safe_relative_path(
        partition["root"],
        label="AFHQ-v2 Dog artifact domain.partition.root",
    )
    if partition_root.as_posix() != "train":
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact partition root must be 'train'"
        )
    partition_count = _positive_int(
        partition["count"],
        label="AFHQ-v2 Dog artifact domain.partition.count",
    )

    index = _strict_mapping(
        domain["index"],
        fields=_INDEX_FIELDS,
        label="AFHQ-v2 Dog artifact domain.index",
    )
    index_path = _safe_relative_path(
        index["path"],
        label="AFHQ-v2 Dog artifact domain.index.path",
    )
    if index_path != _INDEX_PATH:
        raise DataArtifactValidationError(
            f"AFHQ-v2 Dog artifact index path must be {_INDEX_PATH.as_posix()!r}"
        )
    expected_size = _positive_int(
        index["size_bytes"],
        label="AFHQ-v2 Dog artifact domain.index.size_bytes",
    )
    expected_digest = _sha256(
        index["sha256"],
        label="AFHQ-v2 Dog artifact domain.index.sha256",
    )
    expected_count = _positive_int(
        index["record_count"],
        label="AFHQ-v2 Dog artifact domain.index.record_count",
    )
    try:
        encoded = (context.data_root / index_path).read_bytes()
    except OSError as error:
        raise DataArtifactValidationError(
            "cannot read AFHQ-v2 Dog artifact image index"
        ) from error
    if (
        len(encoded) != expected_size
        or hashlib.sha256(encoded).hexdigest() != expected_digest
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index does not match its descriptor"
        )
    try:
        raw_index = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index is not valid UTF-8 JSON"
        ) from error
    if encoded != canonical_artifact_json_bytes(raw_index):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index is not canonical JSON"
        )
    index_mapping = _strict_mapping(
        raw_index,
        fields=frozenset({"schema_version", "records"}),
        label="AFHQ-v2 Dog artifact image index",
    )
    if (
        type(index_mapping["schema_version"]) is not int
        or index_mapping["schema_version"] != 1
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index schema_version must be 1"
        )
    records_raw = index_mapping["records"]
    if not isinstance(records_raw, list):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index records must be a list"
        )
    if len(records_raw) != expected_count:
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index record count changed"
        )

    records: list[ImageFileRecord] = []
    for record_index, value in enumerate(records_raw):
        try:
            record = ImageFileRecord.from_dict(
                value,
                path=(
                    "AFHQ-v2 Dog artifact image index "
                    f"records[{record_index}]"
                ),
            )
        except (TypeError, ValueError) as error:
            raise DataArtifactValidationError(str(error)) from error
        path = PurePosixPath(record.path)
        if (
            record.tree != "train"
            or len(path.parts) != 2
            or path.parts[0] != "dog"
        ):
            raise DataArtifactValidationError(
                "AFHQ-v2 Dog artifact image record must use train/dog"
            )
        if (
            record.width != AFHQV2_DOG_RESOLUTION
            or record.height != AFHQV2_DOG_RESOLUTION
        ):
            raise DataArtifactValidationError(
                "AFHQ-v2 Dog artifact image dimensions must be 256x256"
            )
        records.append(record)
    ordering = [record.path for record in records]
    if ordering != sorted(ordering) or len(ordering) != len(set(ordering)):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact image index must be sorted and unique"
        )
    if not (
        len(records)
        == source_count
        == partition_count
        == expected_count
    ):
        raise DataArtifactValidationError(
            "AFHQ-v2 Dog artifact train image count changed"
        )
    return ImageFolderArtifactPayload(
        roots={"train": context.data_root / partition_root},
        train=tuple(records),
    )


def materialize_afhq_v2_dog_artifact(
    context: DataSourceContext,
    *,
    archive: Path | None = None,
    downloader: Literal["auto", "curl", "python"] = "auto",
    lock_file: Path | None = None,
    resolution: int = AFHQV2_DOG_RESOLUTION,
) -> DataArtifact[ImageFolderArtifactPayload]:
    """Materialize or load the pinned AFHQ-v2 Dog benchmark artifact."""

    lock = load_source_lock(lock_file or DEFAULT_DOG_LOCK_PATH)
    source_digest = _validate_lock(lock)
    spec = build_dog_materialization_spec(lock=lock, resolution=resolution)
    expected = context.expected_identity
    if expected is not None:
        if expected.source_digest != source_digest:
            raise DataArtifactValidationError(
                "strict resume AFHQ-v2 Dog source identity does not match "
                "the current source lock"
            )
        if expected.materialization_digest != spec.digest:
            raise DataArtifactValidationError(
                "strict resume AFHQ-v2 Dog materialization identity does not "
                "match the current recipe"
            )
    locator_key: dict[str, object] = {
        "source_digest": source_digest,
        "materialization_digest": spec.digest,
        "resolution": resolution,
        "source_subset": "train/dog",
    }
    return DataArtifactStore(context).materialize_managed(
        artifact_type=AFHQV2_DOG_ARTIFACT_TYPE,
        source_name=AFHQV2_DOG_SOURCE_NAME,
        materializer_name=AFHQV2_DOG_MATERIALIZER_NAME,
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
        ),
        load=lambda load_context: _load_payload(
            load_context,
            source_digest=source_digest,
            materialization_digest=spec.digest,
        ),
    )


__all__ = [
    "AFHQV2_DOG_ARTIFACT_TYPE",
    "AFHQV2_DOG_SOURCE_NAME",
    "DEFAULT_DOG_LOCK_PATH",
    "materialize_afhq_v2_dog_artifact",
]
