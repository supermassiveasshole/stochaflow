"""Black-box contracts for live prediction production and offline replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
import yaml
from torch import nn
from torchmetrics import Metric

from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.data.builder import DataBuilder
from stochaflow.data.dataloaders import DataLoaders
from stochaflow.evaluation import (
    EvaluationBuilder,
    EvaluationPlan,
    EvaluationStepOutput,
    JsonlPredictionArtifactSink,
    PredictionArtifactDraft,
    PredictionRecord,
    PredictionSampleIdentity,
    PredictionShard,
    ResolvedPredictionArtifactSubject,
    run_evaluation,
)
from stochaflow.evaluation.artifacts import canonical_json_bytes
from stochaflow.evaluation.predictions import (
    PREDICTION_JSONL_MEDIA_TYPE,
    PREDICTION_RECORD_FORMAT,
    materialize_prediction_manifest,
)
from stochaflow.metrics import MetricUpdate
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION
from stochaflow.utils.config import StochaflowConfig, load_config_dict
from stochaflow.utils.registry import REGISTRIES

MODEL_NAME = "test_e2_offline_runtime_model"
DATA_BUILDER_NAME = "test_e2_offline_runtime_data"
EVALUATION_BUILDER_NAME = "test_e2_offline_runtime_evaluator"
METRIC_NAME = "test_e2_offline_runtime_mean"

SAMPLE_PLAN = (
    PredictionSampleIdentity("sample-a", "input-a", 0),
    PredictionSampleIdentity("sample-b", "input-b", 0),
    PredictionSampleIdentity("sample-c", "input-c", 0),
)


@dataclass(frozen=True, slots=True)
class OfflineRuntimeBatch:
    """Task-owned live batch paired with stable prediction identities."""

    values: torch.Tensor
    sample_ids: tuple[str, ...]
    input_ids: tuple[str, ...]


class OfflineRuntimeModel(nn.Module):
    """Count construction and forward calls across live and offline runs."""

    constructor_calls: ClassVar[int] = 0
    forward_calls: ClassVar[int] = 0

    def __init__(self) -> None:
        super().__init__()
        type(self).constructor_calls += 1
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        type(self).forward_calls += 1
        return values + self.offset


class OfflineRuntimeDataBuilder(DataBuilder):
    """Expose live batches while recording whether offline replay rebuilds data."""

    build_calls: ClassVar[int] = 0

    def build(self) -> DataLoaders:
        type(self).build_calls += 1
        validation = [
            OfflineRuntimeBatch(
                values=torch.tensor([1.0, 3.0]),
                sample_ids=("sample-a", "sample-b"),
                input_ids=("input-a", "input-b"),
            ),
            OfflineRuntimeBatch(
                values=torch.tensor([5.0]),
                sample_ids=("sample-c",),
                input_ids=("input-c",),
            ),
        ]
        return DataLoaders(
            train=[
                OfflineRuntimeBatch(
                    values=torch.zeros(1),
                    sample_ids=("train-only",),
                    input_ids=("train-only",),
                )
            ],
            validation=validation,
            test=None,
            artifact_bindings=DataArtifactBindings(),
        )


class OfflineRuntimeMeanMetric(Metric):
    """Aggregate the identical scalar payload in live and offline modes."""

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        self.total += values.float().sum()
        self.count += values.numel()

    def compute(self) -> torch.Tensor:
        return self.total / self.count


class OfflineReplayEvaluator:
    """Run the selected model live or decode typed records offline."""

    metric_channels = frozenset({"offline.predictions"})

    def __init__(
        self,
        model: OfflineRuntimeModel | None,
        *,
        rewrite_offline_ids: bool = False,
    ) -> None:
        self.model = model
        self.rewrite_offline_ids = rewrite_offline_ids
        self.seen_sample_ids: list[str] = []

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        records: tuple[PredictionRecord, ...] | None
        if isinstance(batch, OfflineRuntimeBatch):
            if self.model is None:
                raise TypeError("live replay fixture requires injected model")
            predictions = self.model(batch.values)
            sample_ids = batch.sample_ids
            records = tuple(
                PredictionRecord(
                    sample_id=sample_id,
                    input_id=input_id,
                    replicate_index=0,
                    payload={"prediction": float(prediction)},
                )
                for sample_id, input_id, prediction in zip(
                    batch.sample_ids,
                    batch.input_ids,
                    predictions.detach().cpu(),
                    strict=True,
                )
            )
        elif isinstance(batch, PredictionRecord):
            if self.model is not None:
                raise TypeError("offline replay must not receive an inference model")
            prediction_value = cast(object, batch.payload.get("prediction"))
            if isinstance(prediction_value, bool) or not isinstance(
                prediction_value,
                (int, float),
            ):
                raise TypeError("offline prediction payload must be numeric")
            predictions = torch.tensor([float(prediction_value)])
            sample_ids = (
                f"rewritten-{batch.sample_id}"
                if self.rewrite_offline_ids
                else batch.sample_id,
            )
            records = None
        else:
            raise TypeError("offline replay received an unsupported work item")
        self.seen_sample_ids.extend(sample_ids)
        return EvaluationStepOutput(
            num_examples=len(sample_ids),
            sample_ids=sample_ids,
            metric_update_groups=(
                {
                    "offline.predictions": MetricUpdate(
                        args=(predictions,),
                    )
                },
            ),
            records=records,
        )


class OfflineReplayEvaluationBuilder(EvaluationBuilder):
    """Compose one Builder that supports both live and artifact subjects."""

    last_evaluator: ClassVar[OfflineReplayEvaluator | None] = None

    def build(self) -> EvaluationPlan:
        rewrite_offline_ids = self.context.params.get(
            "rewrite_offline_ids",
            False,
        )
        if type(rewrite_offline_ids) is not bool:
            raise TypeError("rewrite_offline_ids must be an exact bool")
        sink = None
        modules: Mapping[str, nn.Module]
        if isinstance(self.context.subject, ResolvedPredictionArtifactSubject):
            if self.context.inference is not None:
                raise ValueError("offline subject must not expose live inference")
            model = None
            modules = {}
        else:
            model_value = cast(object, self.context.inference)
            if not isinstance(model_value, OfflineRuntimeModel):
                raise TypeError("live evaluation requires OfflineRuntimeModel")
            if self.context.artifact_root is None:
                raise ValueError("live prediction production requires artifact root")
            model = model_value
            modules = {"primary": model}
            sink = JsonlPredictionArtifactSink(
                self.context.artifact_root,
                expected_samples=SAMPLE_PLAN,
            )
        evaluator = OfflineReplayEvaluator(
            model,
            rewrite_offline_ids=rewrite_offline_ids,
        )
        type(self).last_evaluator = evaluator
        return EvaluationPlan(
            evaluator=evaluator,
            data=self.context.data,
            metric_specs=self.context.metric_specs,
            protocol=self.context.protocol,
            subject=self.context.subject,
            data_identity=self.context.data_identity,
            artifact_sink=sink,
            modules=modules,
        )


REGISTRIES.models.add(MODEL_NAME, OfflineRuntimeModel)
REGISTRIES.data_builders.add(DATA_BUILDER_NAME, OfflineRuntimeDataBuilder)
REGISTRIES.evaluation_builders.add(
    EVALUATION_BUILDER_NAME,
    OfflineReplayEvaluationBuilder,
)
REGISTRIES.metrics.add(METRIC_NAME, OfflineRuntimeMeanMetric)


def training_config() -> StochaflowConfig:
    """Return the normalized authority retained by producer artifacts."""

    return load_config_dict(
        {
            "experiment": {
                "name": "offline-runtime-producer",
                "seed": 29,
                "output_dir": "unused",
            },
            "extensions": {"plugins": []},
            "data": {"name": DATA_BUILDER_NAME, "params": {}},
            "model": {"name": MODEL_NAME, "params": {}},
            "training": {"name": "offline_runtime_training", "params": {}},
            "trainer": {"precision": "fp32"},
        }
    )


def write_checkpoint(path: Path) -> Path:
    """Write a minimal valid v12 checkpoint for the live half of the test."""

    model = OfflineRuntimeModel()
    model.offset.data.fill_(2.0)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": 5,
            "global_step": 13,
            "model_state_dict": model.state_dict(),
            "precision_kind": "fp32",
            "inference_asset_descriptors": {},
            "inference_recipe": None,
            "config": training_config().to_dict(),
            "metadata": {
                "extension_plugins": [],
                "data_artifacts": DataArtifactBindings().to_dict(),
                "lineage": {"run_id": "offline-runtime-training-run"},
            },
        },
        path,
    )
    return path


def evaluation_document(
    *,
    name: str,
    subject: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    """Return the same evaluation method for live and offline authorities."""

    return {
        "version": 1,
        "name": name,
        "purpose": "benchmark",
        "extensions": {"plugins": []},
        "subject": dict(subject),
        "data": {"source": source, "split": "validation"},
        "evaluation": {
            "name": EVALUATION_BUILDER_NAME,
            "params": {"profile": "typed-offline-v1"},
        },
        "metrics": [
            {
                "id": "prediction_mean",
                "name": METRIC_NAME,
                "channel": "offline.predictions",
                "params": {},
            }
        ],
        "protocol": {
            "id": "typed-offline-v1",
            "expected_examples": len(SAMPLE_PLAN),
            "strict_complete": True,
        },
    }


def write_config(path: Path, document: Mapping[str, Any]) -> Path:
    path.write_text(
        yaml.safe_dump(dict(document), sort_keys=False),
        encoding="utf-8",
    )
    return path


def read_result(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(path.read_text(encoding="utf-8")),
    )


def snapshot_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    """Capture producer-owned bytes and mtimes without directory ordering."""

    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def prediction_record(sample_id: str, value: float) -> PredictionRecord:
    sample = next(item for item in SAMPLE_PLAN if item.sample_id == sample_id)
    return PredictionRecord(
        sample_id=sample.sample_id,
        input_id=sample.input_id,
        replicate_index=sample.replicate_index,
        payload={"prediction": value},
    )


def write_prediction_shard(
    root: Path,
    relative_path: str,
    records: Sequence[PredictionRecord],
) -> PredictionShard:
    """Write a canonical public-format shard for a shuffled replay fixture."""

    encoded = b"".join(
        canonical_json_bytes(record.to_dict()) + b"\n" for record in records
    )
    (root / relative_path).write_bytes(encoded)
    return PredictionShard(
        path=relative_path,
        media_type=PREDICTION_JSONL_MEDIA_TYPE,
        format=PREDICTION_RECORD_FORMAT,
        sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        record_count=len(records),
    )


def publish_shuffled_prediction_artifact(root: Path) -> Path:
    """Publish records whose shard, filename, and sample orders all differ."""

    root.mkdir()
    z_shard = write_prediction_shard(
        root,
        "z-first-in-manifest.jsonl",
        (prediction_record("sample-c", 7.0), prediction_record("sample-a", 3.0)),
    )
    a_shard = write_prediction_shard(
        root,
        "a-second-in-manifest.jsonl",
        (prediction_record("sample-b", 5.0),),
    )
    draft = PredictionArtifactDraft(
        samples=SAMPLE_PLAN,
        shards=(z_shard, a_shard),
    )
    published = materialize_prediction_manifest(
        root,
        draft,
        producer={
            "kind": "evaluation",
            "id": "shuffled-producer",
            "authority_sha256": "a" * 64,
            "protocol_id": "typed-offline-v1",
            "protocol_digest": "b" * 64,
        },
        source_subject={
            "kind": "checkpoint",
            "sha256": "c" * 64,
            "lineage": {"run_id": "shuffled-source-run"},
        },
        source_subject_digest="c" * 64,
        resolved_weights="raw",
        inference_profile={"profile": "typed-offline-v1"},
        training_config=training_config(),
        extension_provenance=(),
        data_identity={
            "source": "checkpoint",
            "split": "validation",
            "builder": {"name": DATA_BUILDER_NAME, "params": {}},
            "artifacts": DataArtifactBindings().to_dict(),
        },
        split="validation",
    )
    return published.manifest_path


def reset_runtime_counters() -> None:
    OfflineRuntimeModel.constructor_calls = 0
    OfflineRuntimeModel.forward_calls = 0
    OfflineRuntimeDataBuilder.build_calls = 0


def test_live_predictions_replay_offline_without_inference_or_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = write_checkpoint(tmp_path / "subject.pt")
    reset_runtime_counters()
    live_config = write_config(
        tmp_path / "live.yaml",
        evaluation_document(
            name="typed-live-producer",
            subject={
                "kind": "checkpoint",
                "path": str(checkpoint),
                "weights": "raw",
            },
            source="checkpoint",
        ),
    )
    live = run_evaluation(
        live_config,
        output_dir=tmp_path / "live-result",
        device_name="cpu",
    )

    assert live.metrics == {"eval/metrics/prediction_mean": 5.0}
    assert OfflineRuntimeModel.constructor_calls == 1
    assert OfflineRuntimeModel.forward_calls == 2
    assert OfflineRuntimeDataBuilder.build_calls == 1
    prediction_manifest = live.artifacts["predictions"]
    artifact_root = prediction_manifest.parent
    artifact_before = snapshot_tree(artifact_root)
    counters_before = (
        OfflineRuntimeModel.constructor_calls,
        OfflineRuntimeModel.forward_calls,
        OfflineRuntimeDataBuilder.build_calls,
    )

    offline_config = write_config(
        tmp_path / "offline.yaml",
        evaluation_document(
            name="typed-offline-replay",
            subject={
                "kind": "prediction_artifact",
                "path": str(prediction_manifest),
            },
            source="prediction_artifact",
        ),
    )
    offline = run_evaluation(
        offline_config,
        output_dir=tmp_path / "offline-result",
        device_name="cpu",
    )

    assert offline.metrics == live.metrics
    assert offline.measurements == live.measurements == {}
    assert (
        OfflineRuntimeModel.constructor_calls,
        OfflineRuntimeModel.forward_calls,
        OfflineRuntimeDataBuilder.build_calls,
    ) == counters_before
    assert snapshot_tree(artifact_root) == artifact_before
    assert set(offline.artifacts) == {"resolved_config"}

    live_result = read_result(live.result_path)
    offline_result = read_result(offline.result_path)
    producer_manifest = cast(
        dict[str, Any],
        json.loads(prediction_manifest.read_text(encoding="utf-8")),
    )
    offline_subject = cast(dict[str, Any], offline_result["subject"])
    assert offline_subject["kind"] == "prediction_artifact"
    assert offline_subject["producer"] == {
        "kind": "evaluation",
        "id": "typed-live-producer",
        "authority_sha256": hashlib.sha256(live_config.read_bytes()).hexdigest(),
        "protocol_id": "typed-offline-v1",
        "protocol_digest": live_result["protocol_digest"],
    }
    assert offline_subject["source_subject"] == live_result["subject"]
    assert offline_subject["source_subject_digest"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert offline_subject["sample_plan_digest"] == producer_manifest[
        "sample_plan"
    ]["digest"]
    assert offline_subject["artifact_digest"] == producer_manifest[
        "artifact_digest"
    ]
    assert offline_subject["data"] == {
        "identity": live_result["data"],
        "split": "validation",
    }
    assert offline_result["data"] == {
        "source": "prediction_artifact",
        "split": "validation",
        "artifact_digest": producer_manifest["artifact_digest"],
        "sample_plan_digest": producer_manifest["sample_plan"]["digest"],
        "producer_data": live_result["data"],
    }
    assert offline_result["artifacts"] == {}
    evaluator = OfflineReplayEvaluationBuilder.last_evaluator
    assert evaluator is not None
    assert evaluator.model is None
    assert evaluator.seen_sample_ids == ["sample-a", "sample-b", "sample-c"]


def test_offline_runtime_joins_shuffled_shards_by_sample_plan(
    tmp_path: Path,
) -> None:
    prediction_manifest = publish_shuffled_prediction_artifact(
        tmp_path / "shuffled-artifact"
    )
    artifact_before = snapshot_tree(prediction_manifest.parent)
    reset_runtime_counters()
    config = write_config(
        tmp_path / "shuffled-offline.yaml",
        evaluation_document(
            name="shuffled-offline-replay",
            subject={
                "kind": "prediction_artifact",
                "path": str(prediction_manifest),
            },
            source="prediction_artifact",
        ),
    )

    outcome = run_evaluation(
        config,
        output_dir=tmp_path / "shuffled-result",
        device_name="cpu",
    )

    assert outcome.metrics == {"eval/metrics/prediction_mean": 5.0}
    assert OfflineRuntimeModel.constructor_calls == 0
    assert OfflineRuntimeModel.forward_calls == 0
    assert OfflineRuntimeDataBuilder.build_calls == 0
    assert snapshot_tree(prediction_manifest.parent) == artifact_before
    evaluator = OfflineReplayEvaluationBuilder.last_evaluator
    assert evaluator is not None
    assert evaluator.seen_sample_ids == ["sample-a", "sample-b", "sample-c"]
    manifest = cast(
        dict[str, Any],
        json.loads(prediction_manifest.read_text(encoding="utf-8")),
    )
    assert [
        shard["path"] for shard in manifest["predictions"]["shards"]
    ] == ["z-first-in-manifest.jsonl", "a-second-in-manifest.jsonl"]


def test_offline_runtime_rejects_same_count_wrong_sample_ids(
    tmp_path: Path,
) -> None:
    prediction_manifest = publish_shuffled_prediction_artifact(
        tmp_path / "wrong-id-artifact"
    )
    artifact_before = snapshot_tree(prediction_manifest.parent)
    document = evaluation_document(
        name="wrong-id-offline-replay",
        subject={
            "kind": "prediction_artifact",
            "path": str(prediction_manifest),
        },
        source="prediction_artifact",
    )
    evaluation = cast(dict[str, Any], document["evaluation"])
    params = cast(dict[str, Any], evaluation["params"])
    params["rewrite_offline_ids"] = True
    config = write_config(tmp_path / "wrong-id.yaml", document)
    output_dir = tmp_path / "wrong-id-result"

    with pytest.raises(ValueError, match="sample IDs must match"):
        run_evaluation(config, output_dir=output_dir, device_name="cpu")

    assert not output_dir.exists()
    assert snapshot_tree(prediction_manifest.parent) == artifact_before
