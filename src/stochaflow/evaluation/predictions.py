"""Safe, replayable prediction artifacts for offline evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Literal, Protocol, cast, runtime_checkable

from stochaflow.data.artifact_io import canonical_directory, read_regular_file
from stochaflow.evaluation.artifacts import canonical_json_bytes, canonical_sha256
from stochaflow.evaluation.config import (
    EVALUATION_SPLITS,
    EvaluationSplit,
    _freeze_evaluation_mapping,
    _thaw_evaluation_value,
)
from stochaflow.evaluation.contracts import EvaluationStepOutput
from stochaflow.utils.config import StochaflowConfig, load_config_dict
from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    extension_plugin_provenance_to_dicts,
    parse_extension_plugin_provenance,
)

PREDICTION_ARTIFACT_SCHEMA_VERSION = 1
PREDICTION_MANIFEST_FILENAME = "prediction_manifest.json"
PREDICTION_RECORD_FORMAT = "stochaflow.prediction-record.v1"
PREDICTION_JSONL_MEDIA_TYPE = "application/x-ndjson"

PredictionArtifactStatus = Literal["complete"]
PredictionProducerKind = Literal["evaluation", "sampling"]


def _non_empty_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result:
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


def _sha256(value: object, *, path: str) -> str:
    digest = _non_empty_string(value, path=path)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _positive_integer(value: object, *, path: str) -> int:
    if type(value) is not int or cast(int, value) <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return cast(int, value)


def _schema_version(value: object, *, path: str) -> int:
    if type(value) is not int or cast(int, value) != 1:
        raise ValueError(f"{path} must be integer 1")
    return cast(int, value)


def _non_negative_integer(value: object, *, path: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return cast(int, value)


def _portable_relative_path(value: object, *, path: str) -> str:
    reference = _non_empty_string(value, path=path)
    posix = PurePosixPath(reference)
    windows = PureWindowsPath(reference)
    if (
        "\\" in reference
        or any(ord(character) < 32 for character in reference)
        or posix.is_absolute()
        or windows.is_absolute()
        or not posix.parts
        or any(
            part in {"", ".", ".."}
            or ":" in part
            or part.endswith((" ", "."))
            for part in posix.parts
        )
        or posix.as_posix() != reference
    ):
        raise ValueError(f"{path} must be a normalized portable relative path")
    return reference


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    declared = cast(Mapping[object, Any], value)
    for key in declared:
        if type(key) is not str:
            raise TypeError(f"{path} field names must be exact strings")
    keys = cast(set[str], set(declared))
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(keys - fields)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")
    return cast(Mapping[str, Any], value)


def _canonical_document_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _snapshot_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    return _freeze_evaluation_mapping(value, path=path)


def _mapping_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    thawed = _thaw_evaluation_value(value)
    if type(thawed) is not dict:
        raise TypeError("frozen prediction mapping did not thaw to a dictionary")
    return cast(dict[str, Any], thawed)


@dataclass(frozen=True, slots=True)
class PredictionSampleIdentity:
    """One exact sample-plan identity shared by live and offline scoring."""

    sample_id: str
    input_id: str
    replicate_index: int

    def __post_init__(self) -> None:
        _non_empty_string(self.sample_id, path="prediction sample sample_id")
        _non_empty_string(self.input_id, path="prediction sample input_id")
        _non_negative_integer(
            cast(object, self.replicate_index),
            path="prediction sample replicate_index",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the portable sample identity."""

        return {
            "sample_id": self.sample_id,
            "input_id": self.input_id,
            "replicate_index": self.replicate_index,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "prediction sample",
    ) -> PredictionSampleIdentity:
        """Parse one strict sample identity mapping."""

        raw = _strict_mapping(
            value,
            fields=frozenset({"sample_id", "input_id", "replicate_index"}),
            path=path,
        )
        return cls(
            sample_id=_non_empty_string(raw["sample_id"], path=f"{path}.sample_id"),
            input_id=_non_empty_string(raw["input_id"], path=f"{path}.input_id"),
            replicate_index=_non_negative_integer(
                raw["replicate_index"],
                path=f"{path}.replicate_index",
            ),
        )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One JSON-safe prediction payload bound to an exact sample identity."""

    sample_id: str
    input_id: str
    replicate_index: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        identity = PredictionSampleIdentity(
            sample_id=self.sample_id,
            input_id=self.input_id,
            replicate_index=self.replicate_index,
        )
        object.__setattr__(self, "sample_id", identity.sample_id)
        object.__setattr__(self, "input_id", identity.input_id)
        object.__setattr__(self, "replicate_index", identity.replicate_index)
        object.__setattr__(
            self,
            "payload",
            _snapshot_mapping(self.payload, path="prediction record payload"),
        )

    @property
    def identity(self) -> PredictionSampleIdentity:
        """Return the record's exact sample identity."""

        return PredictionSampleIdentity(
            sample_id=self.sample_id,
            input_id=self.input_id,
            replicate_index=self.replicate_index,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize one canonical JSONL record."""

        return {
            "schema_version": PREDICTION_ARTIFACT_SCHEMA_VERSION,
            **self.identity.to_dict(),
            "payload": _mapping_copy(self.payload),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "prediction record",
    ) -> PredictionRecord:
        """Parse one strict canonical JSONL record."""

        raw = _strict_mapping(
            value,
            fields=frozenset(
                {
                    "schema_version",
                    "sample_id",
                    "input_id",
                    "replicate_index",
                    "payload",
                }
            ),
            path=path,
        )
        _schema_version(raw["schema_version"], path=f"{path}.schema_version")
        return cls(
            sample_id=_non_empty_string(raw["sample_id"], path=f"{path}.sample_id"),
            input_id=_non_empty_string(raw["input_id"], path=f"{path}.input_id"),
            replicate_index=_non_negative_integer(
                raw["replicate_index"],
                path=f"{path}.replicate_index",
            ),
            payload=_snapshot_mapping(raw["payload"], path=f"{path}.payload"),
        )


@dataclass(frozen=True, slots=True)
class PredictionShard:
    """Content-addressed safe prediction shard reference."""

    path: str
    media_type: str
    format: str
    sha256: str
    size_bytes: int
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _portable_relative_path(self.path, path="prediction shard path"),
        )
        if self.media_type != PREDICTION_JSONL_MEDIA_TYPE:
            raise ValueError(
                "prediction shard media_type must be "
                f"{PREDICTION_JSONL_MEDIA_TYPE!r}"
            )
        if self.format != PREDICTION_RECORD_FORMAT:
            raise ValueError(
                f"prediction shard format must be {PREDICTION_RECORD_FORMAT!r}"
            )
        object.__setattr__(
            self,
            "sha256",
            _sha256(self.sha256, path="prediction shard sha256"),
        )
        _positive_integer(self.size_bytes, path="prediction shard size_bytes")
        _positive_integer(self.record_count, path="prediction shard record_count")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the portable shard reference."""

        return {
            "path": self.path,
            "media_type": self.media_type,
            "format": self.format,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "record_count": self.record_count,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "prediction shard",
    ) -> PredictionShard:
        """Parse one strict shard descriptor."""

        raw = _strict_mapping(
            value,
            fields=frozenset(
                {
                    "path",
                    "media_type",
                    "format",
                    "sha256",
                    "size_bytes",
                    "record_count",
                }
            ),
            path=path,
        )
        return cls(
            path=cast(str, raw["path"]),
            media_type=cast(str, raw["media_type"]),
            format=cast(str, raw["format"]),
            sha256=cast(str, raw["sha256"]),
            size_bytes=cast(int, raw["size_bytes"]),
            record_count=cast(int, raw["record_count"]),
        )


def _snapshot_samples(
    values: object,
    *,
    path: str,
) -> tuple[PredictionSampleIdentity, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{path} must be a sequence")
    samples = tuple(
        PredictionSampleIdentity(
            sample_id=value.sample_id,
            input_id=value.input_id,
            replicate_index=value.replicate_index,
        )
        if isinstance(value, PredictionSampleIdentity)
        else PredictionSampleIdentity.from_dict(value, path=f"{path}[{index}]")
        for index, value in enumerate(values)
    )
    if not samples:
        raise ValueError(f"{path} must be non-empty")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    input_replicates = [
        (sample.input_id, sample.replicate_index) for sample in samples
    ]
    if len(input_replicates) != len(set(input_replicates)):
        raise ValueError(f"{path} contains duplicate input/replicate identities")
    return samples


def select_prediction_gallery_sample_ids(
    samples: Sequence[PredictionSampleIdentity],
    *,
    protocol_id: str,
    count: int,
    declared_sample_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Select a fixed gallery by declaration or a stable protocol/sample hash."""

    frozen_samples = _snapshot_samples(samples, path="prediction gallery samples")
    protocol = _non_empty_string(protocol_id, path="prediction gallery protocol_id")
    selected_count = _non_negative_integer(
        cast(object, count),
        path="prediction gallery count",
    )
    if selected_count > len(frozen_samples):
        raise ValueError("prediction gallery count exceeds the sample plan")
    available = {sample.sample_id for sample in frozen_samples}
    if declared_sample_ids is not None:
        if isinstance(declared_sample_ids, (str, bytes)):
            raise TypeError("declared prediction gallery IDs must be a sequence")
        selected = tuple(
            _non_empty_string(
                value,
                path=f"declared prediction gallery IDs[{index}]",
            )
            for index, value in enumerate(declared_sample_ids)
        )
        if len(selected) != selected_count:
            raise ValueError(
                "declared prediction gallery ID count must match gallery count"
            )
        if len(selected) != len(set(selected)):
            raise ValueError("declared prediction gallery IDs must be unique")
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(
                "declared prediction gallery IDs are absent from the sample plan: "
                + ", ".join(missing)
            )
        return selected
    ranked = sorted(
        available,
        key=lambda sample_id: (
            hashlib.sha256(
                f"{protocol}\0{sample_id}".encode()
            ).hexdigest(),
            sample_id,
        ),
    )
    return tuple(ranked[:selected_count])


@dataclass(frozen=True, slots=True)
class PredictionArtifactDraft:
    """Finalized shards paired with their frozen exact sample plan."""

    samples: tuple[PredictionSampleIdentity, ...]
    shards: tuple[PredictionShard, ...]
    preprocess: Mapping[str, Any] = field(default_factory=dict)
    postprocess: Mapping[str, Any] = field(default_factory=dict)
    gallery_sample_ids: tuple[str, ...] | None = None
    gallery_count: int = 16
    status: PredictionArtifactStatus = "complete"

    def __post_init__(self) -> None:
        if type(cast(object, self.samples)) is not tuple:
            raise TypeError("prediction draft samples must be an exact tuple")
        if type(cast(object, self.shards)) is not tuple:
            raise TypeError("prediction draft shards must be an exact tuple")
        samples = _snapshot_samples(self.samples, path="prediction draft samples")
        shards: list[PredictionShard] = []
        seen_paths: set[str] = set()
        for index, declared in enumerate(self.shards):
            if not isinstance(cast(object, declared), PredictionShard):
                raise TypeError(f"prediction draft shards[{index}] is invalid")
            shard = PredictionShard.from_dict(
                declared.to_dict(),
                path=f"prediction draft shards[{index}]",
            )
            if shard.path in seen_paths:
                raise ValueError(
                    f"prediction draft contains duplicate shard path {shard.path!r}"
                )
            seen_paths.add(shard.path)
            shards.append(shard)
        if not shards:
            raise ValueError("prediction draft shards must be non-empty")
        if sum(shard.record_count for shard in shards) != len(samples):
            raise ValueError(
                "prediction draft shard record count must equal sample-plan count"
            )
        preprocess = _snapshot_mapping(
            self.preprocess,
            path="prediction draft preprocess",
        )
        postprocess = _snapshot_mapping(
            self.postprocess,
            path="prediction draft postprocess",
        )
        gallery_count = _non_negative_integer(
            cast(object, self.gallery_count),
            path="prediction draft gallery_count",
        )
        gallery_ids = self.gallery_sample_ids
        if gallery_ids is not None:
            if type(cast(object, gallery_ids)) is not tuple:
                raise TypeError(
                    "prediction draft gallery_sample_ids must be an exact tuple"
                )
            gallery_ids = select_prediction_gallery_sample_ids(
                samples,
                protocol_id="prediction-draft-validation-v1",
                count=len(gallery_ids),
                declared_sample_ids=gallery_ids,
            )
        if self.status != "complete":
            raise ValueError("prediction draft status must be 'complete'")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "shards", tuple(shards))
        object.__setattr__(self, "preprocess", preprocess)
        object.__setattr__(self, "postprocess", postprocess)
        object.__setattr__(self, "gallery_sample_ids", gallery_ids)
        object.__setattr__(self, "gallery_count", gallery_count)

    @property
    def sample_plan_digest(self) -> str:
        """Return the exact ordered sample-plan digest."""

        return canonical_sha256([sample.to_dict() for sample in self.samples])


@runtime_checkable
class EvaluationArtifactSink(Protocol):
    """Stream task-compatible evaluation outputs into a finalized draft."""

    def consume(self, output: EvaluationStepOutput) -> None:
        """Consume one already evaluated batch without invoking inference."""

        ...

    def finalize(self) -> PredictionArtifactDraft:
        """Flush and close all shards, returning immutable descriptors."""

        ...

    def abort(self) -> None:
        """Close handles and remove only unpublished files owned by this sink."""

        ...


class JsonlPredictionArtifactSink:
    """Safe streaming sink for canonical JSONL prediction records."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        expected_samples: Sequence[PredictionSampleIdentity],
        relative_path: str = "predictions.jsonl",
        preprocess: Mapping[str, Any] | None = None,
        postprocess: Mapping[str, Any] | None = None,
        gallery_sample_ids: Sequence[str] | None = None,
        gallery_count: int = 16,
    ) -> None:
        self._root = canonical_directory(
            Path(artifact_root),
            label="prediction artifact staging root",
        )
        self._samples = _snapshot_samples(
            expected_samples,
            path="prediction sink expected_samples",
        )
        self._expected = {sample.sample_id: sample for sample in self._samples}
        self._preprocess = _snapshot_mapping(
            {} if preprocess is None else preprocess,
            path="prediction sink preprocess",
        )
        self._postprocess = _snapshot_mapping(
            {} if postprocess is None else postprocess,
            path="prediction sink postprocess",
        )
        self._gallery_sample_ids = (
            None
            if gallery_sample_ids is None
            else select_prediction_gallery_sample_ids(
                self._samples,
                protocol_id="prediction-sink-validation-v1",
                count=len(gallery_sample_ids),
                declared_sample_ids=gallery_sample_ids,
            )
        )
        self._gallery_count = _non_negative_integer(
            cast(object, gallery_count),
            path="prediction sink gallery_count",
        )
        self._relative_path = _portable_relative_path(
            relative_path,
            path="prediction sink relative_path",
        )
        parts = PurePosixPath(self._relative_path).parts
        self._path = self._root.joinpath(*parts)
        canonical_directory(
            self._path.parent,
            label="prediction shard parent",
        )
        self._handle: BinaryIO | None = self._path.open("xb")
        self._digest = hashlib.sha256()
        self._size_bytes = 0
        self._record_count = 0
        self._observed: set[str] = set()
        self._finalized = False
        self._aborted = False

    def consume(self, output: EvaluationStepOutput) -> None:
        """Write one output's typed records after validating exact identities."""

        if self._finalized or self._aborted or self._handle is None:
            raise RuntimeError("prediction sink is not open")
        if not isinstance(cast(object, output), EvaluationStepOutput):
            raise TypeError("prediction sink output must be EvaluationStepOutput")
        records_value = output.records
        if isinstance(records_value, PredictionRecord):
            records = (records_value,)
        elif type(records_value) in {list, tuple}:
            records = tuple(cast(Sequence[object], records_value))
        else:
            raise TypeError(
                "prediction sink output.records must contain PredictionRecord values"
            )
        if len(records) != output.num_examples:
            raise ValueError(
                "prediction sink record count must equal output.num_examples"
            )
        validated: list[PredictionRecord] = []
        current_ids: set[str] = set()
        for index, declared in enumerate(records):
            if not isinstance(declared, PredictionRecord):
                raise TypeError(
                    f"prediction sink output.records[{index}] must be PredictionRecord"
                )
            record = PredictionRecord.from_dict(
                declared.to_dict(),
                path=f"prediction sink output.records[{index}]",
            )
            expected = self._expected.get(record.sample_id)
            if expected is None:
                raise ValueError(
                    f"prediction sink received unexpected sample ID {record.sample_id!r}"
                )
            if record.identity != expected:
                raise ValueError(
                    f"prediction sink identity mismatch for {record.sample_id!r}"
                )
            if record.sample_id in self._observed or record.sample_id in current_ids:
                raise ValueError(
                    f"prediction sink received duplicate sample ID {record.sample_id!r}"
                )
            current_ids.add(record.sample_id)
            validated.append(record)
        record_ids = tuple(record.sample_id for record in validated)
        if record_ids != output.sample_ids:
            raise ValueError(
                "prediction sink record IDs must match EvaluationStepOutput.sample_ids"
            )
        for record in validated:
            encoded = _canonical_document_bytes(record.to_dict())
            self._handle.write(encoded)
            self._digest.update(encoded)
            self._size_bytes += len(encoded)
            self._record_count += 1
            self._observed.add(record.sample_id)

    def finalize(self) -> PredictionArtifactDraft:
        """Flush one complete JSONL shard and return its content descriptor."""

        if self._finalized:
            raise RuntimeError("prediction sink is already finalized")
        if self._aborted or self._handle is None:
            raise RuntimeError("prediction sink is aborted")
        missing = sorted(set(self._expected) - self._observed)
        if missing:
            self.abort()
            raise ValueError(
                "prediction sink is missing expected sample IDs: " + ", ".join(missing)
            )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        self._finalized = True
        shard = PredictionShard(
            path=self._relative_path,
            media_type=PREDICTION_JSONL_MEDIA_TYPE,
            format=PREDICTION_RECORD_FORMAT,
            sha256=self._digest.hexdigest(),
            size_bytes=self._size_bytes,
            record_count=self._record_count,
        )
        return PredictionArtifactDraft(
            samples=self._samples,
            shards=(shard,),
            preprocess=self._preprocess,
            postprocess=self._postprocess,
            gallery_sample_ids=self._gallery_sample_ids,
            gallery_count=self._gallery_count,
        )

    def abort(self) -> None:
        """Close and remove the sink-owned unpublished shard."""

        if self._aborted:
            return
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        with suppress(FileNotFoundError):
            self._path.unlink()
        self._aborted = True


def _validated_producer(value: object, *, path: str) -> Mapping[str, Any]:
    producer = _snapshot_mapping(value, path=path)
    kind = producer.get("kind")
    if kind not in {"evaluation", "sampling"}:
        raise ValueError(f"{path}.kind must be 'evaluation' or 'sampling'")
    _non_empty_string(producer.get("id"), path=f"{path}.id")
    _sha256(producer.get("authority_sha256"), path=f"{path}.authority_sha256")
    return producer


def _validated_source_subject(value: object, *, path: str) -> Mapping[str, Any]:
    subject = _snapshot_mapping(value, path=path)
    _non_empty_string(subject.get("kind"), path=f"{path}.kind")
    return subject


def _validate_source_subject_digest(
    subject: Mapping[str, Any],
    digest: str,
    *,
    path: str,
) -> None:
    declared = subject.get("sha256", subject.get("artifact_digest"))
    if declared is not None and _sha256(declared, path=path) != digest:
        raise ValueError("prediction artifact source subject digest is inconsistent")


def _validate_data_split(
    data_identity: Mapping[str, Any],
    split: EvaluationSplit,
    *,
    path: str,
) -> None:
    if split not in EVALUATION_SPLITS:
        raise ValueError(f"{path} must be 'validation' or 'test'")
    declared = data_identity.get("split")
    if declared is not None and declared != split:
        raise ValueError("prediction artifact data identity split is inconsistent")


def _parse_record_lines(
    encoded: bytes,
    *,
    expected_count: int,
    path: str,
) -> tuple[PredictionRecord, ...]:
    lines = encoded.splitlines(keepends=True)
    if len(lines) != expected_count:
        raise ValueError(f"{path} record count does not match its shard descriptor")
    records: list[PredictionRecord] = []
    for index, line in enumerate(lines):
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}[{index}] is not valid UTF-8 JSON") from error
        if line != _canonical_document_bytes(raw):
            raise ValueError(f"{path}[{index}] is not canonical JSONL")
        records.append(
            PredictionRecord.from_dict(raw, path=f"{path}[{index}]")
        )
    return tuple(records)


def _read_prediction_shards(
    root: Path,
    shards: Sequence[PredictionShard],
) -> tuple[PredictionRecord, ...]:
    records: list[PredictionRecord] = []
    for index, shard in enumerate(shards):
        encoded, metadata = read_regular_file(
            root,
            shard.path,
            label=f"prediction shard {index}",
        )
        if metadata.st_size != shard.size_bytes or len(encoded) != shard.size_bytes:
            raise ValueError(f"prediction shard {index} size mismatch")
        if hashlib.sha256(encoded).hexdigest() != shard.sha256:
            raise ValueError(f"prediction shard {index} digest mismatch")
        records.extend(
            _parse_record_lines(
                encoded,
                expected_count=shard.record_count,
                path=f"prediction shard {index}",
            )
        )
    return tuple(records)


def _join_prediction_records(
    expected: Sequence[PredictionSampleIdentity],
    observed: Sequence[PredictionRecord],
) -> tuple[PredictionRecord, ...]:
    expected_by_id = {sample.sample_id: sample for sample in expected}
    observed_by_id: dict[str, PredictionRecord] = {}
    observed_input_replicates: set[tuple[str, int]] = set()
    duplicates: list[str] = []
    unexpected: list[str] = []
    for record in observed:
        if record.sample_id in observed_by_id:
            duplicates.append(record.sample_id)
            continue
        input_replicate = (record.input_id, record.replicate_index)
        if input_replicate in observed_input_replicates:
            raise ValueError(
                "prediction artifact contains duplicate input/replicate identity "
                f"{input_replicate!r}"
            )
        observed_input_replicates.add(input_replicate)
        observed_by_id[record.sample_id] = record
        if record.sample_id not in expected_by_id:
            unexpected.append(record.sample_id)
    if duplicates:
        raise ValueError(
            "prediction artifact contains duplicate sample IDs: "
            + ", ".join(sorted(set(duplicates)))
        )
    if unexpected:
        raise ValueError(
            "prediction artifact contains unexpected sample IDs: "
            + ", ".join(sorted(unexpected))
        )
    missing = sorted(set(expected_by_id) - set(observed_by_id))
    if missing:
        raise ValueError(
            "prediction artifact is missing expected sample IDs: "
            + ", ".join(missing)
        )
    for sample_id, expected_identity in expected_by_id.items():
        if observed_by_id[sample_id].identity != expected_identity:
            raise ValueError(
                f"prediction artifact identity mismatch for sample ID {sample_id!r}"
            )
    return tuple(observed_by_id[sample.sample_id] for sample in expected)


@dataclass(frozen=True, slots=True)
class PublishedPredictionArtifact:
    """Content identities exposed after a complete manifest is published."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    artifact_digest: str
    shards: tuple[PredictionShard, ...]


def materialize_prediction_manifest(
    artifact_root: Path,
    draft: PredictionArtifactDraft,
    *,
    producer: Mapping[str, Any],
    source_subject: Mapping[str, Any],
    source_subject_digest: str,
    resolved_weights: str,
    inference_profile: Mapping[str, Any],
    training_config: StochaflowConfig,
    extension_provenance: Sequence[ExtensionPluginProvenance],
    data_identity: Mapping[str, Any],
    split: EvaluationSplit,
    preprocess: Mapping[str, Any] | None = None,
    postprocess: Mapping[str, Any] | None = None,
    gallery_sample_ids: Sequence[str] | None = None,
    gallery_count: int | None = None,
) -> PublishedPredictionArtifact:
    """Validate all shards and publish the immutable completion manifest last."""

    root = canonical_directory(
        Path(artifact_root),
        label="prediction artifact root",
    )
    if not isinstance(cast(object, draft), PredictionArtifactDraft):
        raise TypeError("prediction artifact draft must be PredictionArtifactDraft")
    producer_snapshot = _validated_producer(
        producer,
        path="prediction artifact producer",
    )
    subject_snapshot = _validated_source_subject(
        source_subject,
        path="prediction artifact source_subject",
    )
    subject_digest = _sha256(
        source_subject_digest,
        path="prediction artifact source_subject_digest",
    )
    _validate_source_subject_digest(
        subject_snapshot,
        subject_digest,
        path="prediction artifact source_subject content digest",
    )
    weights = _non_empty_string(
        resolved_weights,
        path="prediction artifact resolved_weights",
    )
    if weights == "auto":
        raise ValueError("prediction artifact resolved_weights must not be 'auto'")
    profile = _snapshot_mapping(
        inference_profile,
        path="prediction artifact inference_profile",
    )
    profile_digest = canonical_sha256(profile)
    if not isinstance(cast(object, training_config), StochaflowConfig):
        raise TypeError("prediction artifact training_config must be StochaflowConfig")
    normalized_config = training_config.to_dict()
    normalized_config = load_config_dict(normalized_config).to_dict()
    provenance_dicts = extension_plugin_provenance_to_dicts(extension_provenance)
    validated_provenance = parse_extension_plugin_provenance(provenance_dicts)
    provenance_dicts = extension_plugin_provenance_to_dicts(validated_provenance)
    data_snapshot = _snapshot_mapping(
        data_identity,
        path="prediction artifact data_identity",
    )
    if not data_snapshot:
        raise ValueError("prediction artifact data_identity must be non-empty")
    _validate_data_split(
        data_snapshot,
        split,
        path="prediction artifact split",
    )
    preprocess_snapshot = _snapshot_mapping(
        draft.preprocess if preprocess is None else preprocess,
        path="prediction artifact preprocess",
    )
    postprocess_snapshot = _snapshot_mapping(
        draft.postprocess if postprocess is None else postprocess,
        path="prediction artifact postprocess",
    )

    observed = _read_prediction_shards(root, draft.shards)
    joined = _join_prediction_records(draft.samples, observed)
    if len(joined) != sum(shard.record_count for shard in draft.shards):
        raise ValueError("prediction artifact shard record count is inconsistent")

    samples_document = [sample.to_dict() for sample in draft.samples]
    sample_plan_digest = canonical_sha256(samples_document)
    selected_gallery_count = min(
        _non_negative_integer(
            cast(
                object,
                draft.gallery_count if gallery_count is None else gallery_count,
            ),
            path="prediction artifact gallery_count",
        ),
        len(draft.samples),
    )
    gallery_protocol_id = _non_empty_string(
        producer_snapshot.get("protocol_id", producer_snapshot["id"]),
        path="prediction artifact gallery protocol_id",
    )
    declared_gallery_ids = (
        draft.gallery_sample_ids
        if gallery_sample_ids is None
        else gallery_sample_ids
    )
    gallery_ids = select_prediction_gallery_sample_ids(
        draft.samples,
        protocol_id=gallery_protocol_id,
        count=(
            len(declared_gallery_ids)
            if declared_gallery_ids is not None
            else selected_gallery_count
        ),
        declared_sample_ids=declared_gallery_ids,
    )
    manifest_body: dict[str, Any] = {
        "schema_version": PREDICTION_ARTIFACT_SCHEMA_VERSION,
        "kind": "prediction_artifact",
        "status": "complete",
        "producer": {
            "identity": _mapping_copy(producer_snapshot),
            "training_config": normalized_config,
            "extension_plugins": provenance_dicts,
        },
        "source_subject": {
            "identity": _mapping_copy(subject_snapshot),
            "digest": subject_digest,
            "resolved_weights": weights,
        },
        "inference_profile": {
            "identity": _mapping_copy(profile),
            "digest": profile_digest,
        },
        "data": {
            "identity": _mapping_copy(data_snapshot),
            "split": split,
        },
        "sample_plan": {
            "schema_version": PREDICTION_ARTIFACT_SCHEMA_VERSION,
            "digest": sample_plan_digest,
            "count": len(draft.samples),
            "samples": samples_document,
        },
        "predictions": {
            "format": PREDICTION_RECORD_FORMAT,
            "record_count": len(joined),
            "shards": [shard.to_dict() for shard in draft.shards],
        },
        "preprocess": _mapping_copy(preprocess_snapshot),
        "postprocess": _mapping_copy(postprocess_snapshot),
        "gallery": {
            "method": (
                "declared_sample_ids_v1"
                if declared_gallery_ids is not None
                else "protocol_sample_hash_v1"
            ),
            "protocol_id": gallery_protocol_id,
            "count": len(gallery_ids),
            "sample_ids": list(gallery_ids),
        },
        "completeness": {
            "expected_count": len(draft.samples),
            "observed_count": len(joined),
            "missing_ids": [],
            "unexpected_ids": [],
            "duplicate_ids": [],
            "failed_ids": [],
            "skipped_ids": [],
            "complete": True,
        },
    }
    artifact_digest = canonical_sha256(manifest_body)
    manifest = {**manifest_body, "artifact_digest": artifact_digest}
    encoded = _canonical_document_bytes(manifest)
    manifest_path = root / PREDICTION_MANIFEST_FILENAME
    created_manifest = False
    try:
        with manifest_path.open("xb") as handle:
            created_manifest = True
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created_manifest:
            with suppress(FileNotFoundError):
                manifest_path.unlink()
        raise
    return PublishedPredictionArtifact(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        artifact_digest=artifact_digest,
        shards=draft.shards,
    )


def _parse_sample_plan(
    value: object,
) -> tuple[tuple[PredictionSampleIdentity, ...], str]:
    raw = _strict_mapping(
        value,
        fields=frozenset({"schema_version", "digest", "count", "samples"}),
        path="prediction manifest sample_plan",
    )
    _schema_version(
        raw["schema_version"],
        path="prediction manifest sample_plan.schema_version",
    )
    count = _positive_integer(
        raw["count"],
        path="prediction manifest sample_plan.count",
    )
    declared_samples = raw["samples"]
    samples = _snapshot_samples(
        declared_samples,
        path="prediction manifest sample_plan.samples",
    )
    if len(samples) != count:
        raise ValueError("prediction manifest sample plan count mismatch")
    digest = _sha256(
        raw["digest"],
        path="prediction manifest sample_plan.digest",
    )
    if canonical_sha256([sample.to_dict() for sample in samples]) != digest:
        raise ValueError("prediction manifest sample plan digest mismatch")
    return samples, digest


def _parse_shards(value: object) -> tuple[PredictionShard, ...]:
    raw = _strict_mapping(
        value,
        fields=frozenset({"format", "record_count", "shards"}),
        path="prediction manifest predictions",
    )
    if raw["format"] != PREDICTION_RECORD_FORMAT:
        raise ValueError("prediction manifest prediction format is unsupported")
    record_count = _positive_integer(
        raw["record_count"],
        path="prediction manifest predictions.record_count",
    )
    declared_shards = raw["shards"]
    if not isinstance(declared_shards, list) or not declared_shards:
        raise TypeError("prediction manifest predictions.shards must be a non-empty list")
    shards = tuple(
        PredictionShard.from_dict(
            value,
            path=f"prediction manifest predictions.shards[{index}]",
        )
        for index, value in enumerate(declared_shards)
    )
    paths = [shard.path for shard in shards]
    if len(paths) != len(set(paths)):
        raise ValueError("prediction manifest contains duplicate shard paths")
    if sum(shard.record_count for shard in shards) != record_count:
        raise ValueError("prediction manifest shard record count mismatch")
    return shards


def _validate_completeness(value: object, *, expected_count: int) -> None:
    raw = _strict_mapping(
        value,
        fields=frozenset(
            {
                "expected_count",
                "observed_count",
                "missing_ids",
                "unexpected_ids",
                "duplicate_ids",
                "failed_ids",
                "skipped_ids",
                "complete",
            }
        ),
        path="prediction manifest completeness",
    )
    if (
        type(raw["expected_count"]) is not int
        or raw["expected_count"] != expected_count
    ):
        raise ValueError("prediction manifest completeness expected_count mismatch")
    if (
        type(raw["observed_count"]) is not int
        or raw["observed_count"] != expected_count
    ):
        raise ValueError("prediction manifest completeness observed_count mismatch")
    for name in (
        "missing_ids",
        "unexpected_ids",
        "duplicate_ids",
        "failed_ids",
        "skipped_ids",
    ):
        if raw[name] != []:
            raise ValueError(
                f"prediction manifest completeness.{name} must be empty"
            )
    if raw["complete"] is not True:
        raise ValueError("prediction manifest completeness.complete must be true")


def _validate_gallery(
    value: object,
    *,
    samples: Sequence[PredictionSampleIdentity],
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        fields=frozenset({"method", "protocol_id", "count", "sample_ids"}),
        path="prediction manifest gallery",
    )
    method = raw["method"]
    if method not in {"declared_sample_ids_v1", "protocol_sample_hash_v1"}:
        raise ValueError("prediction manifest gallery method is unsupported")
    protocol_id = _non_empty_string(
        raw["protocol_id"],
        path="prediction manifest gallery.protocol_id",
    )
    count = _non_negative_integer(
        raw["count"],
        path="prediction manifest gallery.count",
    )
    declared = raw["sample_ids"]
    if type(declared) not in {list, tuple}:
        raise TypeError(
            "prediction manifest gallery.sample_ids must be a sequence"
        )
    declared_ids = cast(Sequence[str], declared)
    expected = select_prediction_gallery_sample_ids(
        samples,
        protocol_id=protocol_id,
        count=count,
        declared_sample_ids=(
            declared_ids
            if method == "declared_sample_ids_v1"
            else None
        ),
    )
    if tuple(declared_ids) != expected:
        raise ValueError("prediction manifest gallery selection is inconsistent")
    return _snapshot_mapping(raw, path="prediction manifest gallery")


@dataclass(frozen=True, slots=True)
class PredictionArtifactSubjectInputs:
    """Strictly verified, read-only prediction artifact inputs."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    artifact_digest: str
    producer: Mapping[str, Any]
    source_subject: Mapping[str, Any]
    source_subject_digest: str
    resolved_weights: str
    inference_profile: Mapping[str, Any]
    inference_profile_digest: str
    sample_plan_digest: str
    samples: tuple[PredictionSampleIdentity, ...]
    config: Mapping[str, Any]
    extension_provenance: tuple[ExtensionPluginProvenance, ...]
    data_identity: Mapping[str, Any]
    split: EvaluationSplit
    preprocess: Mapping[str, Any]
    postprocess: Mapping[str, Any]
    gallery: Mapping[str, Any]
    shards: tuple[PredictionShard, ...]
    records: tuple[PredictionRecord, ...]
    status: PredictionArtifactStatus = "complete"

    def __post_init__(self) -> None:
        root_value = cast(object, self.root)
        manifest_value = cast(object, self.manifest_path)
        if not isinstance(root_value, Path) or not self.root.is_absolute():
            raise ValueError("prediction artifact root must be an absolute Path")
        if not isinstance(manifest_value, Path) or not self.manifest_path.is_absolute():
            raise ValueError("prediction artifact manifest_path must be an absolute Path")
        if self.manifest_path.parent != self.root:
            raise ValueError("prediction artifact manifest_path must be inside root")
        object.__setattr__(
            self,
            "manifest_sha256",
            _sha256(self.manifest_sha256, path="prediction subject manifest_sha256"),
        )
        object.__setattr__(
            self,
            "artifact_digest",
            _sha256(self.artifact_digest, path="prediction subject artifact_digest"),
        )
        object.__setattr__(
            self,
            "producer",
            _validated_producer(self.producer, path="prediction subject producer"),
        )
        object.__setattr__(
            self,
            "source_subject",
            _validated_source_subject(
                self.source_subject,
                path="prediction subject source_subject",
            ),
        )
        object.__setattr__(
            self,
            "source_subject_digest",
            _sha256(
                self.source_subject_digest,
                path="prediction subject source_subject_digest",
            ),
        )
        _validate_source_subject_digest(
            cast(Mapping[str, Any], self.source_subject),
            self.source_subject_digest,
            path="prediction subject source_subject content digest",
        )
        weights = _non_empty_string(
            self.resolved_weights,
            path="prediction subject resolved_weights",
        )
        if weights == "auto":
            raise ValueError("prediction subject resolved_weights must not be 'auto'")
        object.__setattr__(self, "resolved_weights", weights)
        profile = _snapshot_mapping(
            self.inference_profile,
            path="prediction subject inference_profile",
        )
        profile_digest = _sha256(
            self.inference_profile_digest,
            path="prediction subject inference_profile_digest",
        )
        if canonical_sha256(profile) != profile_digest:
            raise ValueError("prediction subject inference profile digest mismatch")
        object.__setattr__(self, "inference_profile", profile)
        object.__setattr__(self, "inference_profile_digest", profile_digest)
        samples = _snapshot_samples(
            self.samples,
            path="prediction subject samples",
        )
        sample_plan_digest = _sha256(
            self.sample_plan_digest,
            path="prediction subject sample_plan_digest",
        )
        if canonical_sha256([sample.to_dict() for sample in samples]) != (
            sample_plan_digest
        ):
            raise ValueError("prediction subject sample plan digest mismatch")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_plan_digest", sample_plan_digest)
        config = _snapshot_mapping(self.config, path="prediction subject config")
        normalized_config = load_config_dict(_mapping_copy(config)).to_dict()
        if normalized_config != _mapping_copy(config):
            raise ValueError("prediction subject config must be normalized")
        object.__setattr__(self, "config", config)
        if type(cast(object, self.extension_provenance)) is not tuple:
            raise TypeError("prediction subject extension_provenance must be a tuple")
        provenance = parse_extension_plugin_provenance(
            extension_plugin_provenance_to_dicts(self.extension_provenance)
        )
        object.__setattr__(self, "extension_provenance", provenance)
        data_identity = _snapshot_mapping(
            self.data_identity,
            path="prediction subject data_identity",
        )
        if not data_identity:
            raise ValueError("prediction subject data_identity must be non-empty")
        object.__setattr__(self, "data_identity", data_identity)
        _validate_data_split(
            data_identity,
            self.split,
            path="prediction subject split",
        )
        object.__setattr__(
            self,
            "preprocess",
            _snapshot_mapping(
                self.preprocess,
                path="prediction subject preprocess",
            ),
        )
        object.__setattr__(
            self,
            "postprocess",
            _snapshot_mapping(
                self.postprocess,
                path="prediction subject postprocess",
            ),
        )
        object.__setattr__(
            self,
            "gallery",
            _validate_gallery(self.gallery, samples=samples),
        )
        if type(cast(object, self.shards)) is not tuple or not self.shards:
            raise TypeError("prediction subject shards must be a non-empty tuple")
        if type(cast(object, self.records)) is not tuple:
            raise TypeError("prediction subject records must be an exact tuple")
        shards: list[PredictionShard] = []
        for index, shard in enumerate(self.shards):
            if not isinstance(cast(object, shard), PredictionShard):
                raise TypeError(f"prediction subject shards[{index}] is invalid")
            shards.append(
                PredictionShard.from_dict(
                    shard.to_dict(),
                    path=f"prediction subject shards[{index}]",
                )
            )
        records: list[PredictionRecord] = []
        for index, record in enumerate(self.records):
            if not isinstance(cast(object, record), PredictionRecord):
                raise TypeError(f"prediction subject records[{index}] is invalid")
            records.append(
                PredictionRecord.from_dict(
                    record.to_dict(),
                    path=f"prediction subject records[{index}]",
                )
            )
        object.__setattr__(self, "shards", tuple(shards))
        object.__setattr__(self, "records", tuple(records))
        joined = _join_prediction_records(samples, records)
        if joined != self.records:
            raise ValueError("prediction subject records must use sample-plan order")
        if sum(shard.record_count for shard in self.shards) != len(self.records):
            raise ValueError("prediction subject shard record count mismatch")
        if self.status != "complete":
            raise ValueError("prediction subject status must be 'complete'")

    def training_config_copy(self) -> StochaflowConfig:
        """Return a fresh config for offline extension preflight and activation."""

        return load_config_dict(_mapping_copy(self.config))


@dataclass(frozen=True, slots=True)
class ResolvedPredictionArtifactSubject:
    """Verified prediction records and portable producer lineage."""

    inputs: PredictionArtifactSubjectInputs = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.inputs), PredictionArtifactSubjectInputs):
            raise TypeError(
                "resolved prediction artifact inputs must be "
                "PredictionArtifactSubjectInputs"
            )

    @property
    def kind(self) -> Literal["prediction_artifact"]:
        """Return the subject discriminator."""

        return "prediction_artifact"

    @property
    def records(self) -> tuple[PredictionRecord, ...]:
        """Return records joined into frozen sample-plan order."""

        return self.inputs.records

    @property
    def extension_provenance(self) -> tuple[ExtensionPluginProvenance, ...]:
        """Return producer-declared extension provenance."""

        return self.inputs.extension_provenance

    @property
    def data_identity(self) -> Mapping[str, Any]:
        """Return the producer's deeply immutable data identity."""

        return self.inputs.data_identity

    @property
    def split(self) -> EvaluationSplit:
        """Return the governed producer split."""

        return self.inputs.split

    @property
    def identity(self) -> Mapping[str, Any]:
        """Return portable artifact and producer identity for a new result."""

        return _snapshot_mapping(
            {
                "kind": self.kind,
                "artifact_digest": self.inputs.artifact_digest,
                "manifest_sha256": self.inputs.manifest_sha256,
                "producer": self.inputs.producer,
                "source_subject": self.inputs.source_subject,
                "source_subject_digest": self.inputs.source_subject_digest,
                "resolved_weights": self.inputs.resolved_weights,
                "inference_profile": self.inputs.inference_profile,
                "inference_profile_digest": self.inputs.inference_profile_digest,
                "sample_plan_digest": self.inputs.sample_plan_digest,
                "data": {
                    "identity": self.inputs.data_identity,
                    "split": self.inputs.split,
                },
                "preprocess": self.inputs.preprocess,
                "postprocess": self.inputs.postprocess,
                "gallery": self.inputs.gallery,
                "extension_plugins": extension_plugin_provenance_to_dicts(
                    self.inputs.extension_provenance
                ),
            },
            path="resolved prediction artifact identity",
        )

    def training_config_copy(self) -> StochaflowConfig:
        """Return a fresh normalized producer training config."""

        return self.inputs.training_config_copy()


_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "artifact_digest",
        "producer",
        "source_subject",
        "inference_profile",
        "data",
        "sample_plan",
        "predictions",
        "preprocess",
        "postprocess",
        "gallery",
        "completeness",
    }
)


def load_prediction_artifact_inputs(
    path: str | Path,
) -> PredictionArtifactSubjectInputs:
    """Strictly load and authenticate one complete prediction artifact."""

    declared = Path(path)
    if declared.name == PREDICTION_MANIFEST_FILENAME:
        root = canonical_directory(
            declared.parent.resolve(strict=True),
            label="prediction artifact root",
        )
    else:
        root = canonical_directory(
            declared.resolve(strict=True),
            label="prediction artifact root",
        )
    encoded, _ = read_regular_file(
        root,
        PREDICTION_MANIFEST_FILENAME,
        label="prediction artifact manifest",
    )
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    try:
        document_value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction artifact manifest is not valid UTF-8 JSON") from error
    if encoded != _canonical_document_bytes(document_value):
        raise ValueError("prediction artifact manifest is not canonical JSON")
    document = _strict_mapping(
        document_value,
        fields=_MANIFEST_FIELDS,
        path="prediction artifact manifest",
    )
    _schema_version(
        document["schema_version"],
        path="prediction artifact schema_version",
    )
    if document["kind"] != "prediction_artifact":
        raise ValueError("prediction artifact kind must be 'prediction_artifact'")
    if document["status"] != "complete":
        raise ValueError("prediction artifact status must be 'complete'")
    artifact_digest = _sha256(
        document["artifact_digest"],
        path="prediction artifact manifest artifact_digest",
    )
    digest_body = dict(document)
    del digest_body["artifact_digest"]
    if canonical_sha256(digest_body) != artifact_digest:
        raise ValueError("prediction artifact manifest digest mismatch")

    producer_raw = _strict_mapping(
        document["producer"],
        fields=frozenset({"identity", "training_config", "extension_plugins"}),
        path="prediction artifact manifest producer",
    )
    producer = _validated_producer(
        producer_raw["identity"],
        path="prediction artifact manifest producer.identity",
    )
    config_raw = producer_raw["training_config"]
    if type(config_raw) is not dict:
        raise TypeError("prediction artifact producer.training_config must be a mapping")
    training_config = load_config_dict(cast(dict[str, Any], config_raw))
    normalized_config = training_config.to_dict()
    if normalized_config != config_raw:
        raise ValueError("prediction artifact training_config is not normalized")
    extension_provenance = parse_extension_plugin_provenance(
        producer_raw["extension_plugins"]
    )

    subject_raw = _strict_mapping(
        document["source_subject"],
        fields=frozenset({"identity", "digest", "resolved_weights"}),
        path="prediction artifact manifest source_subject",
    )
    source_subject = _validated_source_subject(
        subject_raw["identity"],
        path="prediction artifact manifest source_subject.identity",
    )
    source_subject_digest = _sha256(
        subject_raw["digest"],
        path="prediction artifact manifest source_subject.digest",
    )
    _validate_source_subject_digest(
        source_subject,
        source_subject_digest,
        path="prediction artifact manifest source_subject content digest",
    )
    resolved_weights = _non_empty_string(
        subject_raw["resolved_weights"],
        path="prediction artifact manifest source_subject.resolved_weights",
    )
    if resolved_weights == "auto":
        raise ValueError(
            "prediction artifact source_subject.resolved_weights must not be 'auto'"
        )

    profile_raw = _strict_mapping(
        document["inference_profile"],
        fields=frozenset({"identity", "digest"}),
        path="prediction artifact manifest inference_profile",
    )
    inference_profile = _snapshot_mapping(
        profile_raw["identity"],
        path="prediction artifact manifest inference_profile.identity",
    )
    inference_profile_digest = _sha256(
        profile_raw["digest"],
        path="prediction artifact manifest inference_profile.digest",
    )
    if canonical_sha256(inference_profile) != inference_profile_digest:
        raise ValueError("prediction artifact inference profile digest mismatch")

    data_raw = _strict_mapping(
        document["data"],
        fields=frozenset({"identity", "split"}),
        path="prediction artifact manifest data",
    )
    data_identity = _snapshot_mapping(
        data_raw["identity"],
        path="prediction artifact manifest data.identity",
    )
    if not data_identity:
        raise ValueError("prediction artifact data identity must be non-empty")
    split = data_raw["split"]
    _validate_data_split(
        data_identity,
        cast(EvaluationSplit, split),
        path="prediction artifact data.split",
    )

    samples, sample_plan_digest = _parse_sample_plan(document["sample_plan"])
    shards = _parse_shards(document["predictions"])
    if sum(shard.record_count for shard in shards) != len(samples):
        raise ValueError("prediction artifact prediction/sample-plan count mismatch")
    preprocess = _snapshot_mapping(
        document["preprocess"],
        path="prediction artifact preprocess",
    )
    postprocess = _snapshot_mapping(
        document["postprocess"],
        path="prediction artifact postprocess",
    )
    gallery = _validate_gallery(document["gallery"], samples=samples)
    _validate_completeness(document["completeness"], expected_count=len(samples))
    observed = _read_prediction_shards(root, shards)
    records = _join_prediction_records(samples, observed)

    return PredictionArtifactSubjectInputs(
        root=root,
        manifest_path=root / PREDICTION_MANIFEST_FILENAME,
        manifest_sha256=manifest_sha256,
        artifact_digest=artifact_digest,
        producer=producer,
        source_subject=source_subject,
        source_subject_digest=source_subject_digest,
        resolved_weights=resolved_weights,
        inference_profile=inference_profile,
        inference_profile_digest=inference_profile_digest,
        sample_plan_digest=sample_plan_digest,
        samples=samples,
        config=normalized_config,
        extension_provenance=extension_provenance,
        data_identity=data_identity,
        split=cast(EvaluationSplit, split),
        preprocess=preprocess,
        postprocess=postprocess,
        gallery=gallery,
        shards=shards,
        records=records,
    )


def resolve_prediction_artifact(
    inputs: PredictionArtifactSubjectInputs,
) -> ResolvedPredictionArtifactSubject:
    """Resolve verified inputs without constructing a model or loading a checkpoint."""

    if not isinstance(cast(object, inputs), PredictionArtifactSubjectInputs):
        raise TypeError(
            "prediction artifact inputs must be PredictionArtifactSubjectInputs"
        )
    return ResolvedPredictionArtifactSubject(inputs)


def load_prediction_artifact(
    path: str | Path,
) -> ResolvedPredictionArtifactSubject:
    """Load and resolve a complete artifact for offline scoring."""

    return resolve_prediction_artifact(load_prediction_artifact_inputs(path))


__all__ = [
    "PREDICTION_ARTIFACT_SCHEMA_VERSION",
    "PREDICTION_JSONL_MEDIA_TYPE",
    "PREDICTION_MANIFEST_FILENAME",
    "PREDICTION_RECORD_FORMAT",
    "EvaluationArtifactSink",
    "JsonlPredictionArtifactSink",
    "PredictionArtifactDraft",
    "PredictionArtifactStatus",
    "PredictionArtifactSubjectInputs",
    "PredictionProducerKind",
    "PredictionRecord",
    "PredictionSampleIdentity",
    "PredictionShard",
    "PublishedPredictionArtifact",
    "ResolvedPredictionArtifactSubject",
    "load_prediction_artifact",
    "load_prediction_artifact_inputs",
    "materialize_prediction_manifest",
    "resolve_prediction_artifact",
    "select_prediction_gallery_sample_ids",
]
