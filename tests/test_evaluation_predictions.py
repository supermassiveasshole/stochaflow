"""Tests for replayable prediction artifacts and offline subject resolution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from stochaflow.evaluation.artifacts import canonical_json_bytes, canonical_sha256
from stochaflow.evaluation.contracts import EvaluationStepOutput
from stochaflow.evaluation.predictions import (
    PREDICTION_JSONL_MEDIA_TYPE,
    PREDICTION_MANIFEST_FILENAME,
    PREDICTION_RECORD_FORMAT,
    EvaluationArtifactSink,
    JsonlPredictionArtifactSink,
    PredictionArtifactDraft,
    PredictionRecord,
    PredictionSampleIdentity,
    PredictionShard,
    load_prediction_artifact,
    load_prediction_artifact_inputs,
    materialize_prediction_manifest,
    select_prediction_gallery_sample_ids,
)
from stochaflow.utils.config import StochaflowConfig, load_config_dict
from stochaflow.utils.plugins import ExtensionPluginProvenance


def training_config() -> StochaflowConfig:
    """Return one normalized producer training authority."""

    return load_config_dict(
        {
            "experiment": {
                "name": "prediction-producer",
                "seed": 17,
                "output_dir": "unused",
            },
            "extensions": {"plugins": ["acme_predictions"]},
            "data": {"name": "acme.data", "params": {"split_seed": 3}},
            "model": {"name": "acme.model", "params": {"width": 4}},
            "training": {"name": "acme.training", "params": {}},
            "trainer": {"precision": "fp32"},
        }
    )


def extension_provenance() -> tuple[ExtensionPluginProvenance, ...]:
    """Return one valid producer plugin identity."""

    return (
        ExtensionPluginProvenance(
            name="acme_predictions",
            distribution="acme-predictions",
            version="1.2.3",
            target="acme_predictions.plugin",
        ),
    )


def sample_plan() -> tuple[PredictionSampleIdentity, ...]:
    """Return one exact manifest order independent of shard order."""

    return (
        PredictionSampleIdentity("sample-a", "input-a", 0),
        PredictionSampleIdentity("sample-b", "input-b", 0),
        PredictionSampleIdentity("sample-c", "input-c", 0),
    )


def prediction_record(sample_id: str, value: float) -> PredictionRecord:
    """Build a record whose input identity is derived by the fixture."""

    return PredictionRecord(
        sample_id=sample_id,
        input_id=sample_id.replace("sample", "input"),
        replicate_index=0,
        payload={"prediction": value, "metadata": {"unit": "opaque"}},
    )


def evaluation_output(
    records: Sequence[PredictionRecord],
) -> EvaluationStepOutput:
    """Wrap typed prediction records in the evaluator/sink boundary."""

    return EvaluationStepOutput(
        num_examples=len(records),
        sample_ids=tuple(record.sample_id for record in records),
        metric_update_groups=(),
        records=tuple(records),
    )


def producer_identity() -> dict[str, Any]:
    """Return the minimum content-addressed producer declaration."""

    return {
        "kind": "evaluation",
        "id": "live-evaluation-run",
        "authority_sha256": "a" * 64,
        "protocol_id": "opaque-live-v1",
        "protocol_digest": "b" * 64,
    }


def publication_kwargs() -> dict[str, Any]:
    """Return immutable lineage required by manifest publication."""

    return {
        "producer": producer_identity(),
        "source_subject": {
            "kind": "checkpoint",
            "sha256": "c" * 64,
            "epoch": 7,
            "global_step": 31,
        },
        "source_subject_digest": "c" * 64,
        "resolved_weights": "ema",
        "inference_profile": {
            "method": "opaque.restore",
            "params": {"precision": "fp32"},
        },
        "training_config": training_config(),
        "extension_provenance": extension_provenance(),
        "data_identity": {
            "source": "checkpoint",
            "split": "test",
            "dataset_fingerprint": "d" * 64,
        },
        "split": "test",
        "preprocess": {"range": "zero_one"},
        "postprocess": {"quantization": "none"},
    }


def write_shard(
    root: Path,
    relative_path: str,
    records: Sequence[PredictionRecord],
) -> PredictionShard:
    """Write one canonical fixture shard and return its exact descriptor."""

    encoded = b"".join(
        canonical_json_bytes(record.to_dict()) + b"\n" for record in records
    )
    path = root / relative_path
    path.write_bytes(encoded)
    return PredictionShard(
        path=relative_path,
        media_type=PREDICTION_JSONL_MEDIA_TYPE,
        format=PREDICTION_RECORD_FORMAT,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        record_count=len(records),
    )


def publish_fixture(root: Path) -> Path:
    """Publish one complete artifact whose record order differs from its plan."""

    root.mkdir()
    first = write_shard(
        root,
        "part-0.jsonl",
        (prediction_record("sample-c", 3.0), prediction_record("sample-a", 1.0)),
    )
    second = write_shard(
        root,
        "part-1.jsonl",
        (prediction_record("sample-b", 2.0),),
    )
    draft = PredictionArtifactDraft(
        samples=sample_plan(),
        shards=(first, second),
    )
    publication = materialize_prediction_manifest(
        root,
        draft,
        **publication_kwargs(),
    )
    return publication.manifest_path


def rewrite_manifest(root: Path, document: Mapping[str, Any]) -> None:
    """Rewrite a manifest canonically for strict negative loader fixtures."""

    (root / PREDICTION_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(document) + b"\n"
    )


def snapshot_files(root: Path) -> dict[str, bytes]:
    """Capture producer-owned bytes without relying on directory order."""

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_prediction_record_is_a_deep_immutable_json_snapshot() -> None:
    payload: dict[str, Any] = {
        "prediction": [1.0, 2.0],
        "metadata": {"label": "cat"},
    }
    record = PredictionRecord("sample-a", "input-a", 0, payload)
    cast(list[float], payload["prediction"])[0] = 9.0
    cast(dict[str, str], payload["metadata"])["label"] = "dog"

    assert record.payload["prediction"] == (1.0, 2.0)
    assert cast(Mapping[str, str], record.payload["metadata"])["label"] == "cat"
    with pytest.raises(TypeError):
        cast(dict[str, Any], record.payload)["other"] = True
    with pytest.raises(ValueError, match="finite"):
        PredictionRecord("sample", "input", 0, {"value": float("nan")})
    with pytest.raises(TypeError, match="unsupported value type"):
        PredictionRecord("sample", "input", 0, {"unsafe": Path("payload.pt")})


def test_gallery_selection_is_stable_across_sample_plan_order() -> None:
    expected = select_prediction_gallery_sample_ids(
        sample_plan(),
        protocol_id="opaque-live-v1",
        count=2,
    )

    assert select_prediction_gallery_sample_ids(
        tuple(reversed(sample_plan())),
        protocol_id="opaque-live-v1",
        count=2,
    ) == expected
    assert len(expected) == 2
    assert set(expected) <= {"sample-a", "sample-b", "sample-c"}


def test_gallery_selection_preserves_declared_ids_and_rejects_invalid_plans() -> None:
    assert select_prediction_gallery_sample_ids(
        sample_plan(),
        protocol_id="opaque-live-v1",
        count=2,
        declared_sample_ids=("sample-c", "sample-a"),
    ) == ("sample-c", "sample-a")

    with pytest.raises(ValueError, match="must be unique"):
        select_prediction_gallery_sample_ids(
            sample_plan(),
            protocol_id="opaque-live-v1",
            count=2,
            declared_sample_ids=("sample-a", "sample-a"),
        )
    with pytest.raises(ValueError, match="absent from the sample plan"):
        select_prediction_gallery_sample_ids(
            sample_plan(),
            protocol_id="opaque-live-v1",
            count=1,
            declared_sample_ids=("sample-extra",),
        )


@pytest.mark.parametrize(
    "path",
    ["../predictions.jsonl", "/predictions.jsonl", "C:/predictions.jsonl", "a\\b"],
)
def test_prediction_shards_require_portable_relative_safe_paths(path: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        PredictionShard(
            path=path,
            media_type=PREDICTION_JSONL_MEDIA_TYPE,
            format=PREDICTION_RECORD_FORMAT,
            sha256="a" * 64,
            size_bytes=1,
            record_count=1,
        )


def test_prediction_shards_reject_pickle_like_formats() -> None:
    with pytest.raises(ValueError, match="media_type"):
        PredictionShard(
            path="predictions.pt",
            media_type="application/x-pytorch",
            format="torch.save",
            sha256="a" * 64,
            size_bytes=1,
            record_count=1,
        )


def test_jsonl_sink_streams_and_finalizes_exact_sample_plan(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    sink = JsonlPredictionArtifactSink(
        root,
        expected_samples=sample_plan(),
        preprocess={"source_range": [-1.0, 1.0]},
        postprocess={"codec": "opaque-u8-v1"},
        gallery_sample_ids=("sample-c", "sample-a"),
    )

    assert isinstance(sink, EvaluationArtifactSink)
    sink.consume(
        evaluation_output(
            (prediction_record("sample-c", 3.0), prediction_record("sample-a", 1.0))
        )
    )
    sink.consume(evaluation_output((prediction_record("sample-b", 2.0),)))
    draft = sink.finalize()

    assert draft.samples == sample_plan()
    assert draft.sample_plan_digest == canonical_sha256(
        [sample.to_dict() for sample in sample_plan()]
    )
    assert draft.shards[0].record_count == 3
    assert draft.preprocess == {"source_range": (-1.0, 1.0)}
    assert draft.postprocess == {"codec": "opaque-u8-v1"}
    assert draft.gallery_sample_ids == ("sample-c", "sample-a")
    assert hashlib.sha256((root / "predictions.jsonl").read_bytes()).hexdigest() == (
        draft.shards[0].sha256
    )
    with pytest.raises(RuntimeError, match="already finalized"):
        sink.finalize()


def test_materializer_uses_draft_transform_and_gallery_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    shard = write_shard(
        root,
        "predictions.jsonl",
        tuple(prediction_record(sample.sample_id, float(index)) for index, sample in enumerate(sample_plan())),
    )
    draft = PredictionArtifactDraft(
        samples=sample_plan(),
        shards=(shard,),
        preprocess={"source_range": [-1.0, 1.0]},
        postprocess={"codec": "opaque-u8-v1"},
        gallery_sample_ids=("sample-c", "sample-a"),
    )
    kwargs = publication_kwargs()
    kwargs.pop("preprocess")
    kwargs.pop("postprocess")

    publication = materialize_prediction_manifest(root, draft, **kwargs)
    manifest = json.loads(publication.manifest_path.read_bytes())

    assert manifest["preprocess"] == {"source_range": [-1.0, 1.0]}
    assert manifest["postprocess"] == {"codec": "opaque-u8-v1"}
    assert manifest["gallery"] == {
        "method": "declared_sample_ids_v1",
        "protocol_id": "opaque-live-v1",
        "count": 2,
        "sample_ids": ["sample-c", "sample-a"],
    }


def test_jsonl_sink_rejects_duplicate_unexpected_and_missing_ids(
    tmp_path: Path,
) -> None:
    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    duplicate = JsonlPredictionArtifactSink(
        duplicate_root,
        expected_samples=sample_plan(),
    )
    duplicate.consume(evaluation_output((prediction_record("sample-a", 1.0),)))
    with pytest.raises(ValueError, match="duplicate sample ID"):
        duplicate.consume(evaluation_output((prediction_record("sample-a", 2.0),)))
    duplicate.abort()
    assert not (duplicate_root / "predictions.jsonl").exists()
    duplicate.abort()

    unexpected_root = tmp_path / "unexpected"
    unexpected_root.mkdir()
    unexpected = JsonlPredictionArtifactSink(
        unexpected_root,
        expected_samples=sample_plan(),
    )
    with pytest.raises(ValueError, match="unexpected sample ID"):
        unexpected.consume(
            evaluation_output((prediction_record("sample-extra", 4.0),))
        )
    unexpected.abort()

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    missing = JsonlPredictionArtifactSink(
        missing_root,
        expected_samples=sample_plan(),
    )
    missing.consume(evaluation_output((prediction_record("sample-a", 1.0),)))
    with pytest.raises(ValueError, match="missing expected sample IDs"):
        missing.finalize()
    assert not (missing_root / "predictions.jsonl").exists()


def test_manifest_materialization_and_loader_preserve_identity_and_plan_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    manifest_path = publish_fixture(root)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    resolved = load_prediction_artifact(root)

    assert resolved.kind == "prediction_artifact"
    assert tuple(record.sample_id for record in resolved.records) == (
        "sample-a",
        "sample-b",
        "sample-c",
    )
    assert resolved.inputs.manifest_sha256 == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    digest_body = dict(manifest)
    artifact_digest = digest_body.pop("artifact_digest")
    assert canonical_sha256(digest_body) == artifact_digest
    assert resolved.inputs.artifact_digest == artifact_digest
    assert resolved.inputs.producer == producer_identity()
    assert resolved.inputs.source_subject_digest == "c" * 64
    assert resolved.inputs.resolved_weights == "ema"
    assert resolved.inputs.split == "test"
    assert resolved.data_identity["dataset_fingerprint"] == "d" * 64
    assert resolved.extension_provenance == extension_provenance()
    assert resolved.identity["kind"] == "prediction_artifact"
    assert resolved.identity["producer"] == producer_identity()
    assert manifest["status"] == "complete"
    assert manifest["completeness"]["missing_ids"] == []
    assert manifest["completeness"]["unexpected_ids"] == []
    assert manifest["completeness"]["duplicate_ids"] == []
    assert manifest["gallery"] == {
        "count": 3,
        "method": "protocol_sample_hash_v1",
        "protocol_id": "opaque-live-v1",
        "sample_ids": list(
            select_prediction_gallery_sample_ids(
                sample_plan(),
                protocol_id="opaque-live-v1",
                count=3,
            )
        ),
    }


def test_resolved_subject_returns_fresh_training_config_copies(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    publish_fixture(root)
    resolved = load_prediction_artifact(root)

    first = resolved.training_config_copy()
    second = resolved.training_config_copy()

    assert first is not second
    assert first.to_dict() == second.to_dict() == training_config().to_dict()
    first.model.params["width"] = 99
    assert second.model.params["width"] == 4
    with pytest.raises(TypeError):
        cast(dict[str, Any], resolved.inputs.config)["changed"] = True
    resolved_data = cast(Mapping[str, Any], resolved.identity["data"])
    with pytest.raises(TypeError):
        cast(dict[str, Any], resolved_data)["split"] = "validation"
    assert resolved.identity["preprocess"] == {"range": "zero_one"}
    assert resolved.identity["postprocess"] == {"quantization": "none"}


def test_offline_loading_never_modifies_producer_artifact(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    publish_fixture(root)
    before = snapshot_files(root)

    first = load_prediction_artifact_inputs(root)
    second = load_prediction_artifact(root / PREDICTION_MANIFEST_FILENAME)

    assert first.artifact_digest == second.inputs.artifact_digest
    assert snapshot_files(root) == before


def test_loader_rejects_corrupt_or_missing_shards(tmp_path: Path) -> None:
    corrupt_root = tmp_path / "corrupt"
    publish_fixture(corrupt_root)
    shard = corrupt_root / "part-0.jsonl"
    shard.write_bytes(shard.read_bytes().replace(b"3.0", b"9.0", 1))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_prediction_artifact(corrupt_root)

    missing_root = tmp_path / "missing"
    publish_fixture(missing_root)
    (missing_root / "part-1.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_prediction_artifact(missing_root)


def test_loader_rejects_manifest_digest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    publish_fixture(root)
    document = json.loads((root / PREDICTION_MANIFEST_FILENAME).read_bytes())
    document["producer"]["identity"]["id"] = "forged-producer"
    rewrite_manifest(root, document)

    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_prediction_artifact(root)


def test_loader_rejects_incomplete_artifact_even_with_valid_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    publish_fixture(root)
    document = json.loads((root / PREDICTION_MANIFEST_FILENAME).read_bytes())
    document["status"] = "incomplete"
    document["completeness"]["complete"] = False
    digest_body = dict(document)
    digest_body.pop("artifact_digest")
    document["artifact_digest"] = canonical_sha256(digest_body)
    rewrite_manifest(root, document)

    with pytest.raises(ValueError, match="status must be 'complete'"):
        load_prediction_artifact(root)


def test_materializer_never_replaces_or_deletes_existing_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    publish_fixture(root)
    inputs = load_prediction_artifact_inputs(root)
    draft = PredictionArtifactDraft(samples=inputs.samples, shards=inputs.shards)
    manifest_path = root / PREDICTION_MANIFEST_FILENAME
    before = manifest_path.read_bytes()

    with pytest.raises(FileExistsError):
        materialize_prediction_manifest(root, draft, **publication_kwargs())

    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_subject_digest", "e" * 64, "source subject digest"),
        (
            "data_identity",
            {"source": "checkpoint", "split": "validation"},
            "data identity split",
        ),
    ],
)
def test_materializer_rejects_inconsistent_producer_lineage(
    field: str,
    value: object,
    message: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / field
    root.mkdir()
    sink = JsonlPredictionArtifactSink(root, expected_samples=sample_plan())
    sink.consume(
        evaluation_output(
            tuple(
                prediction_record(sample.sample_id, float(index))
                for index, sample in enumerate(sample_plan())
            )
        )
    )
    draft = sink.finalize()
    kwargs = publication_kwargs()
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        materialize_prediction_manifest(root, draft, **kwargs)

    assert not (root / PREDICTION_MANIFEST_FILENAME).exists()


def test_materializer_rejects_duplicate_records_without_completion_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    samples = (
        PredictionSampleIdentity("sample-a", "input-a", 0),
        PredictionSampleIdentity("sample-b", "input-b", 0),
    )
    first = write_shard(
        root,
        "part-0.jsonl",
        (prediction_record("sample-a", 1.0),),
    )
    second = write_shard(
        root,
        "part-1.jsonl",
        (prediction_record("sample-a", 2.0),),
    )
    draft = PredictionArtifactDraft(samples=samples, shards=(first, second))

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        materialize_prediction_manifest(root, draft, **publication_kwargs())

    assert not (root / PREDICTION_MANIFEST_FILENAME).exists()


def test_materializer_rejects_unexpected_record_and_unresolved_weights(
    tmp_path: Path,
) -> None:
    unexpected_root = tmp_path / "unexpected"
    unexpected_root.mkdir()
    samples = (
        PredictionSampleIdentity("sample-a", "input-a", 0),
        PredictionSampleIdentity("sample-b", "input-b", 0),
    )
    shard = write_shard(
        unexpected_root,
        "predictions.jsonl",
        (
            prediction_record("sample-a", 1.0),
            prediction_record("sample-extra", 2.0),
        ),
    )
    draft = PredictionArtifactDraft(samples=samples, shards=(shard,))
    with pytest.raises(ValueError, match="unexpected sample IDs"):
        materialize_prediction_manifest(
            unexpected_root,
            draft,
            **publication_kwargs(),
        )

    auto_root = tmp_path / "auto"
    auto_root.mkdir()
    sink = JsonlPredictionArtifactSink(auto_root, expected_samples=sample_plan())
    sink.consume(
        evaluation_output(
            tuple(
                prediction_record(sample.sample_id, float(index))
                for index, sample in enumerate(sample_plan())
            )
        )
    )
    complete_draft = sink.finalize()
    kwargs = publication_kwargs()
    kwargs["resolved_weights"] = "auto"
    with pytest.raises(ValueError, match="must not be 'auto'"):
        materialize_prediction_manifest(auto_root, complete_draft, **kwargs)
    assert not (auto_root / PREDICTION_MANIFEST_FILENAME).exists()
