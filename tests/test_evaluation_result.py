"""Tests for portable evaluation results and atomic bundle publication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from stochaflow.evaluation import EvaluationResult, EvaluationRunOutcome
from stochaflow.evaluation import artifacts as evaluation_artifacts
from stochaflow.evaluation.artifacts import (
    canonical_json_bytes,
    canonical_sha256,
    evaluation_result_to_dict,
    publish_evaluation_bundle,
)


def make_result(
    *,
    subject: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    measurements: Mapping[str, float] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    completeness: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    """Build one representative portable result for focused contract tests."""

    return EvaluationResult(
        schema_version=1,
        evaluation_id="opaque-evaluation",
        protocol_id="opaque-v1",
        protocol_digest="a" * 64,
        status="complete",
        subject=(
            subject
            if subject is not None
            else {
                "kind": "checkpoint",
                "content_digest": "b" * 64,
                "weights": {"requested": "raw", "resolved": "raw"},
            }
        ),
        data=(
            data
            if data is not None
            else {"split": "validation", "sample_ids": ["sample-1"]}
        ),
        metrics=(
            metrics
            if metrics is not None
            else {"eval/metrics/score": 1.0}
        ),
        measurements=(
            measurements
            if measurements is not None
            else {"eval/measurements/latency_ms": 2.0}
        ),
        artifacts=(
            artifacts
            if artifacts is not None
            else {
                "predictions": {
                    "path": "predictions.jsonl",
                    "sha256": "c" * 64,
                }
            }
        ),
        completeness=(
            completeness
            if completeness is not None
            else {"expected": 1, "observed": 1, "missing_ids": []}
        ),
        provenance=(
            provenance
            if provenance is not None
            else {"builder": {"name": "tests.opaque", "version": 1}}
        ),
    )


def test_canonical_digest_is_stable_and_mapping_order_independent() -> None:
    left = {
        "z": [{"b": 2, "a": 1}],
        "a": "café",
    }
    right = {
        "a": "café",
        "z": [{"a": 1, "b": 2}],
    }
    expected = b'{"a":"caf\\u00e9","z":[{"a":1,"b":2}]}'

    assert canonical_json_bytes(left) == expected
    assert canonical_json_bytes(right) == expected
    assert canonical_sha256(left) == canonical_sha256(right)
    assert canonical_sha256(left) == hashlib.sha256(expected).hexdigest()


def test_result_serialization_excludes_local_outcome_paths(tmp_path: Path) -> None:
    result = make_result()
    outcome = EvaluationRunOutcome(
        evaluation_id=result.evaluation_id,
        protocol_id=result.protocol_id,
        status=result.status,
        output_dir=tmp_path.resolve(),
        subject=result.subject,
        split="validation",
        metrics=result.metrics,
        measurements=result.measurements,
        artifacts={"predictions": tmp_path.resolve() / "predictions.jsonl"},
        manifest_path=tmp_path.resolve() / "evaluation_manifest.yaml",
        result_path=tmp_path.resolve() / "result.json",
    )

    document = evaluation_result_to_dict(result)

    assert outcome.output_dir.is_absolute()
    assert {
        "output_dir",
        "manifest_path",
        "result_path",
        "gate_result_path",
    }.isdisjoint(document)
    assert document["artifacts"]["predictions"]["path"] == "predictions.jsonl"
    assert str(outcome.output_dir) not in json.dumps(document)
    with pytest.raises(TypeError, match="must be EvaluationResult"):
        evaluation_result_to_dict(cast(Any, outcome))


def test_result_rejects_absolute_local_artifact_references(tmp_path: Path) -> None:
    local_prediction_path = (tmp_path / "predictions.jsonl").resolve()

    with pytest.raises(ValueError, match="relative"):
        make_result(
            artifacts={
                "predictions": {
                    "path": str(local_prediction_path),
                    "sha256": "c" * 64,
                }
            }
        )


def test_publish_writes_completion_manifest_last_before_atomic_directory_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evaluation"
    result = make_result()
    staged_names: list[str] = []
    original_write = evaluation_artifacts._write_bytes

    def record_write(path: Path, content: bytes) -> None:
        assert not destination.exists()
        staged_names.append(path.name)
        original_write(path, content)

    monkeypatch.setattr(evaluation_artifacts, "_write_bytes", record_write)

    bundle = publish_evaluation_bundle(
        destination,
        result=result,
        resolved_config={"version": 1, "evaluation": {"name": "tests.opaque"}},
        manifest_metadata={"purpose": "benchmark", "split": "validation"},
    )

    assert staged_names == [
        "resolved_evaluation.yaml",
        "result.json",
        "evaluation_manifest.yaml",
    ]
    assert {path.name for path in destination.iterdir()} == {
        "resolved_evaluation.yaml",
        "result.json",
        "evaluation_manifest.yaml",
    }
    manifest = yaml.safe_load(bundle.manifest_path.read_text(encoding="utf-8"))
    result_bytes = bundle.result_path.read_bytes()
    result_digest = hashlib.sha256(result_bytes).hexdigest()
    assert manifest["status"] == "complete"
    assert manifest["resolved_config"] == "resolved_evaluation.yaml"
    assert manifest["result"] == {
        "path": "result.json",
        "sha256": result_digest,
    }
    assert bundle.result_sha256 == result_digest
    assert json.loads(result_bytes) == evaluation_result_to_dict(result)
    assert yaml.safe_load(bundle.resolved_config_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "evaluation": {"name": "tests.opaque"},
    }


def test_publish_moves_prepared_artifacts_before_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared-predictions"
    prepared.mkdir()
    prediction_manifest = prepared / "manifest.json"
    prediction_manifest.write_text('{"schema_version":1}\n', encoding="utf-8")
    destination = tmp_path / "evaluation"
    published_names: list[str] = []
    original_replace = Path.replace

    def record_replace(source: Path, target: Path) -> Path:
        published_names.append(target.name)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", record_replace)
    bundle = publish_evaluation_bundle(
        destination,
        result=make_result(
            artifacts={
                "predictions": {
                    "path": "predictions/manifest.json",
                    "sha256": "c" * 64,
                }
            }
        ),
        resolved_config={"version": 1},
        manifest_metadata={},
        prepared_artifacts={"predictions": prepared},
    )

    assert published_names == [
        "predictions",
    ]
    assert not prepared.exists()
    assert (destination / "predictions" / "manifest.json").read_text(
        encoding="utf-8"
    ) == '{"schema_version":1}\n'
    assert bundle.artifacts == {"predictions": destination / "predictions"}


def test_publish_rejects_existing_output_directory_without_modifying_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evaluation"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_evaluation_bundle(
            destination,
            result=make_result(),
            resolved_config={"version": 1},
            manifest_metadata={},
        )

    assert sentinel.read_text(encoding="utf-8") == "existing"
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "reserved_key",
    [
        "schema_version",
        "kind",
        "status",
        "evaluation_id",
        "protocol_id",
        "protocol_digest",
        "resolved_config",
        "result",
    ],
)
def test_publish_rejects_reserved_manifest_metadata_and_cleans_output(
    reserved_key: str,
    tmp_path: Path,
) -> None:
    destination = tmp_path / f"evaluation-{reserved_key}"

    with pytest.raises(ValueError, match="reserved field"):
        publish_evaluation_bundle(
            destination,
            result=make_result(),
            resolved_config={"version": 1},
            manifest_metadata={reserved_key: "forged"},
        )

    assert not destination.exists()


def test_write_failure_leaves_no_evaluation_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evaluation"
    original_write = evaluation_artifacts._write_bytes

    def fail_result_write(path: Path, content: bytes) -> None:
        if path.name == "result.json":
            raise OSError("simulated result write failure")
        original_write(path, content)

    monkeypatch.setattr(evaluation_artifacts, "_write_bytes", fail_result_write)

    with pytest.raises(OSError, match="simulated result write failure"):
        publish_evaluation_bundle(
            destination,
            result=make_result(),
            resolved_config={"version": 1},
            manifest_metadata={},
        )

    assert not destination.exists()


def test_atomic_publication_failure_leaves_no_evaluation_output_or_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evaluation"

    def fail_publish(*args: Any, **kwargs: Any) -> Path:
        del args, kwargs
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(
        evaluation_artifacts,
        "publish_cache_directory",
        fail_publish,
    )

    with pytest.raises(OSError, match="simulated atomic publication failure"):
        publish_evaluation_bundle(
            destination,
            result=make_result(),
            resolved_config={"version": 1},
            manifest_metadata={},
        )

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".evaluation.evaluation-*"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_result_rejects_nonfinite_metrics_and_nested_metadata(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_result(metrics={"eval/metrics/score": value})
    with pytest.raises(ValueError, match="finite"):
        make_result(measurements={"eval/measurements/latency_ms": value})
    with pytest.raises(ValueError, match="finite"):
        make_result(provenance={"runtime": {"seconds": value}})


@pytest.mark.parametrize(
    "key",
    [
        "metrics/score",
        "eval/metric/score",
        "eval/metrics/score/detail/extra",
        "eval/metrics/not canonical",
    ],
)
def test_result_rejects_noncanonical_metric_keys(key: str) -> None:
    with pytest.raises(ValueError, match="canonical"):
        make_result(metrics={key: 1.0})


def test_result_rejects_wrong_measurement_namespace() -> None:
    with pytest.raises(ValueError, match="canonical"):
        make_result(measurements={"eval/metrics/latency_ms": 1.0})


def test_result_is_a_deeply_immutable_snapshot() -> None:
    subject: dict[str, Any] = {
        "weights": {"requested": "raw"},
        "labels": ["cat", {"name": "dog"}],
    }
    data: dict[str, Any] = {"sample_ids": ["sample-1"]}
    artifacts: dict[str, Any] = {
        "predictions": {"path": "predictions.jsonl", "parts": ["part-0"]}
    }
    completeness: dict[str, Any] = {"missing_ids": []}
    provenance: dict[str, Any] = {"builder": {"name": "tests.opaque"}}
    result = make_result(
        subject=subject,
        data=data,
        artifacts=artifacts,
        completeness=completeness,
        provenance=provenance,
    )

    cast(dict[str, Any], subject["weights"])["requested"] = "ema"
    cast(list[Any], subject["labels"])[0] = "wild"
    cast(list[str], data["sample_ids"]).append("sample-2")
    cast(dict[str, Any], artifacts["predictions"])["path"] = "changed.jsonl"
    cast(list[str], completeness["missing_ids"]).append("sample-1")
    cast(dict[str, Any], provenance["builder"])["name"] = "changed"

    assert cast(Mapping[str, Any], result.subject["weights"])["requested"] == "raw"
    assert result.subject["labels"] == ("cat", {"name": "dog"})
    assert result.data["sample_ids"] == ("sample-1",)
    assert cast(Mapping[str, Any], result.artifacts["predictions"])["path"] == (
        "predictions.jsonl"
    )
    assert result.completeness["missing_ids"] == ()
    assert cast(Mapping[str, Any], result.provenance["builder"])["name"] == (
        "tests.opaque"
    )

    for frozen_mapping in (
        result.subject,
        result.data,
        result.metrics,
        result.measurements,
        result.artifacts,
        result.completeness,
        result.provenance,
    ):
        with pytest.raises(TypeError):
            cast(dict[str, Any], frozen_mapping)["changed"] = True

    frozen_labels = cast(tuple[Any, ...], result.subject["labels"])
    with pytest.raises(TypeError):
        cast(dict[str, Any], frozen_labels[1])["name"] = "changed"
    with pytest.raises(AttributeError):
        cast(Any, frozen_labels).append("changed")
