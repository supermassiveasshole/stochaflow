"""Black-box contracts for the standalone evaluation operation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml
from torch import nn
from torchmetrics import Metric

from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.data.builder import DataBuilder
from stochaflow.data.dataloaders import DataLoaders
from stochaflow.evaluation import (
    PREDICTION_JSONL_MEDIA_TYPE,
    PREDICTION_RECORD_FORMAT,
    EvaluationBuilder,
    EvaluationPlan,
    EvaluationRunOutcome,
    EvaluationStepOutput,
    PredictionArtifactDraft,
    PredictionSampleIdentity,
    PredictionShard,
    resolve_evaluation_inputs,
    run_evaluation,
    run_resolved_evaluation,
)
from stochaflow.metrics import MetricUpdate
from stochaflow.scripts.cli import main
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION
from stochaflow.utils.config import load_config_dict
from stochaflow.utils.plugins import activate_extension_plugins
from stochaflow.utils.registry import REGISTRIES

MODEL_NAME = "test_e1_runtime_scalar_model"
DATA_BUILDER_NAME = "test_e1_runtime_opaque_data"
EVALUATION_BUILDER_NAME = "test_e1_runtime_evaluator"
METRIC_NAME = "test_e1_runtime_mean"


@dataclass(frozen=True, slots=True)
class OpaqueRuntimeBatch:
    """Task-owned batch that the core runtime must pass through unchanged."""

    token: str
    values: torch.Tensor
    sample_ids: tuple[str, ...]
    measurement: float


class TinyEvaluationModel(nn.Module):
    """One-parameter subject whose raw and EMA projections are distinguishable."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.observed_training: list[bool] = []
        self.observed_grad_enabled: list[bool] = []
        self.observed_inference_mode: list[bool] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.observed_training.append(self.training)
        self.observed_grad_enabled.append(torch.is_grad_enabled())
        self.observed_inference_mode.append(torch.is_inference_mode_enabled())
        return values + self.offset


class LifecycleProbeModule(nn.Module):
    """Auxiliary plan module used to observe the core-owned inference lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.observed_training: list[bool] = []
        self.observed_grad_enabled: list[bool] = []
        self.observed_inference_mode: list[bool] = []

    def forward(self) -> torch.Tensor:
        self.observed_training.append(self.training)
        self.observed_grad_enabled.append(torch.is_grad_enabled())
        self.observed_inference_mode.append(torch.is_inference_mode_enabled())
        return self.anchor


class TinyEvaluationDataBuilder(DataBuilder):
    """Provide sized opaque validation and test work batches."""

    def build(self) -> DataLoaders:
        validation = [
            OpaqueRuntimeBatch(
                token="validation-first",
                values=torch.tensor([1.0, 3.0]),
                sample_ids=("sample-a", "sample-b"),
                measurement=10.0,
            ),
            OpaqueRuntimeBatch(
                token="validation-second",
                values=torch.tensor([5.0]),
                sample_ids=("sample-c",),
                measurement=40.0,
            ),
        ]
        test = [
            OpaqueRuntimeBatch(
                token="test-only",
                values=torch.tensor([100.0]),
                sample_ids=("test-a",),
                measurement=100.0,
            )
        ]
        return DataLoaders(
            train=[OpaqueRuntimeBatch("train-only", torch.zeros(1), ("train",), 0.0)],
            validation=validation,
            test=test,
            artifact_bindings=(
                None
                if self.context.params.get("omit_artifact_bindings") is True
                else DataArtifactBindings()
            ),
        )


class TinyMeanMetric(Metric):
    """Example-weighted scalar mean with an injectable invalid result fixture."""

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self, *, emit_nonfinite: bool = False) -> None:
        super().__init__()
        self.emit_nonfinite = emit_nonfinite
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        self.total += values.float().sum()
        self.count += values.numel()

    def compute(self) -> torch.Tensor:
        if self.emit_nonfinite:
            return torch.tensor(float("nan"), device=self.total.device)
        return self.total / self.count


class TinyEvaluator:
    """Interpret the custom batch through one injected, preselected model."""

    metric_channels = frozenset({"tiny.predictions"})

    def __init__(
        self,
        model: TinyEvaluationModel,
        probe: LifecycleProbeModule,
        *,
        scenario: str,
    ) -> None:
        self.model = model
        self.probe = probe
        self.scenario = scenario
        self.seen_batches: list[OpaqueRuntimeBatch] = []

    def evaluate_batch(self, batch: Any) -> EvaluationStepOutput:
        assert isinstance(batch, OpaqueRuntimeBatch)
        self.seen_batches.append(batch)
        self.probe()
        predictions = self.model(batch.values)
        sample_ids = batch.sample_ids
        if self.scenario == "duplicate" and batch.token == "validation-second":
            sample_ids = ("sample-b",)
        return EvaluationStepOutput(
            num_examples=len(sample_ids),
            sample_ids=sample_ids,
            metric_update_groups=(
                {"tiny.predictions": MetricUpdate(args=(predictions,))},
            ),
            records={"opaque": batch.token},
            measurements={"latency_ms": batch.measurement},
        )


class ProbeEvaluationArtifactSink:
    """Inject sink failures while exposing cleanup state to the black-box test."""

    def __init__(self, root: Path, *, scenario: str) -> None:
        self.path = root / "probe.jsonl"
        self.path.write_bytes(b"probe\n")
        self.scenario = scenario
        self.aborted = False

    def consume(self, output: EvaluationStepOutput) -> None:
        if self.scenario == "sink_failure" and "sample-c" in output.sample_ids:
            raise RuntimeError("injected prediction sink failure")

    def finalize(self) -> PredictionArtifactDraft:
        encoded = self.path.read_bytes()
        samples = (
            PredictionSampleIdentity("sample-a", "input-a", 0),
            PredictionSampleIdentity("sample-b", "input-b", 0),
            PredictionSampleIdentity("sample-x", "input-x", 0),
        )
        return PredictionArtifactDraft(
            samples=samples,
            shards=(
                PredictionShard(
                    path="probe.jsonl",
                    media_type=PREDICTION_JSONL_MEDIA_TYPE,
                    format=PREDICTION_RECORD_FORMAT,
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    size_bytes=len(encoded),
                    record_count=len(samples),
                ),
            ),
        )

    def abort(self) -> None:
        self.path.unlink(missing_ok=True)
        self.aborted = True


class TinyEvaluationBuilder(EvaluationBuilder):
    """Compose the test evaluator exclusively from injected public contracts."""

    last_model: TinyEvaluationModel | None = None
    last_probe: LifecycleProbeModule | None = None
    last_evaluator: TinyEvaluator | None = None
    last_sink: ProbeEvaluationArtifactSink | None = None

    def build(self) -> EvaluationPlan:
        model = cast(object, self.context.inference)
        if not isinstance(model, TinyEvaluationModel):
            raise TypeError("tiny evaluation requires TinyEvaluationModel inference")
        scenario = cast(str, self.context.params.get("scenario", "normal"))
        probe = LifecycleProbeModule()
        evaluator = TinyEvaluator(model, probe, scenario=scenario)
        sink = None
        if scenario in {"sink_failure", "sink_wrong_plan"}:
            if self.context.artifact_root is None:
                raise ValueError("sink fixture requires an artifact staging root")
            sink = ProbeEvaluationArtifactSink(
                self.context.artifact_root,
                scenario=scenario,
            )
        type(self).last_model = model
        type(self).last_probe = probe
        type(self).last_evaluator = evaluator
        type(self).last_sink = sink
        return EvaluationPlan(
            evaluator=evaluator,
            data=self.context.data,
            metric_specs=self.context.metric_specs,
            protocol=self.context.protocol,
            subject=self.context.subject,
            data_identity=self.context.data_identity,
            modules={"primary": model, "lifecycle_probe": probe},
            artifact_sink=sink,
        )


REGISTRIES.models.add(MODEL_NAME, TinyEvaluationModel)
REGISTRIES.data_builders.add(DATA_BUILDER_NAME, TinyEvaluationDataBuilder)
REGISTRIES.evaluation_builders.add(
    EVALUATION_BUILDER_NAME,
    TinyEvaluationBuilder,
)
REGISTRIES.metrics.add(METRIC_NAME, TinyMeanMetric)


def _training_config() -> dict[str, Any]:
    """Return the complete training authority embedded in the checkpoint."""

    return load_config_dict(
        {
            "experiment": {
                "name": "e1-runtime-subject",
                "seed": 19,
                "output_dir": "unused",
            },
            "extensions": {"plugins": []},
            "data": {"name": DATA_BUILDER_NAME, "params": {}},
            "model": {"name": MODEL_NAME, "params": {}},
            "training": {"name": "not_required_for_evaluation", "params": {}},
            "ema": {
                "enabled": True,
                "decay": 0.9,
                "update_after_step": 0,
                "update_every": 1,
            },
            "trainer": {"precision": "fp32"},
        }
    ).to_dict()


def _write_checkpoint(path: Path) -> Path:
    """Write a valid inference-only v12 fixture without training lifecycle state."""

    raw_model = TinyEvaluationModel()
    raw_model.offset.data.fill_(2.0)
    ema_model = TinyEvaluationModel()
    ema_model.offset.data.fill_(20.0)
    ema = ExponentialMovingAverage(
        ema_model,
        decay=0.9,
        update_after_step=0,
        update_every=1,
    )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": 4,
        "global_step": 9,
        "model_state_dict": raw_model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "precision_kind": "fp32",
        "inference_asset_descriptors": {},
        "inference_recipe": None,
        "config": _training_config(),
        "metadata": {
            "extension_plugins": [],
            "data_artifacts": DataArtifactBindings().to_dict(),
            "lineage": {"run_id": "tiny-training-run"},
        },
        # Deliberately no optimizer, scheduler, scaler, or RNG state. Evaluation
        # consumes the inference projection rather than training restore topology.
    }
    torch.save(payload, path)
    return path


def _write_evaluation_config(
    path: Path,
    checkpoint: Path,
    *,
    weights: str = "raw",
    scenario: str = "normal",
    expected_examples: int = 3,
    emit_nonfinite: bool = False,
) -> Path:
    document = {
        "version": 1,
        "name": f"tiny-{weights}-{scenario}",
        "purpose": "benchmark",
        "extensions": {"plugins": []},
        "subject": {
            "kind": "checkpoint",
            "path": str(checkpoint),
            "weights": weights,
        },
        "data": {"source": "checkpoint", "split": "validation"},
        "evaluation": {
            "name": EVALUATION_BUILDER_NAME,
            "params": {"scenario": scenario},
        },
        "metrics": [
            {
                "id": "prediction_mean",
                "name": METRIC_NAME,
                "channel": "tiny.predictions",
                "params": {"emit_nonfinite": emit_nonfinite},
            }
        ],
        "protocol": {
            "id": "tiny-supervised-v1",
            "expected_examples": expected_examples,
            "strict_complete": True,
        },
    }
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _read_result(output_dir: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((output_dir / "result.json").read_text(encoding="utf-8")),
    )


def _assert_inference_lifecycle_restored() -> None:
    model = TinyEvaluationBuilder.last_model
    probe = TinyEvaluationBuilder.last_probe
    assert model is not None
    assert probe is not None
    assert model.observed_training
    assert not any(model.observed_training)
    assert model.observed_grad_enabled
    assert not any(model.observed_grad_enabled)
    assert model.observed_inference_mode
    assert all(model.observed_inference_mode)
    assert probe.observed_training
    assert not any(probe.observed_training)
    assert probe.observed_grad_enabled
    assert not any(probe.observed_grad_enabled)
    assert probe.observed_inference_mode
    assert all(probe.observed_inference_mode)
    assert model.training is False
    assert probe.training is True


@pytest.mark.parametrize(
    ("weights", "expected_mean"),
    [("raw", 5.0), ("ema", 23.0)],
)
def test_run_evaluation_resolves_weights_and_publishes_portable_bundle(
    tmp_path: Path,
    weights: str,
    expected_mean: float,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    checkpoint_bytes = checkpoint.read_bytes()
    config = _write_evaluation_config(
        tmp_path / f"{weights}.yaml",
        checkpoint,
        weights=weights,
    )
    output_dir = tmp_path / f"result-{weights}"

    outcome = run_evaluation(
        config,
        output_dir=output_dir,
        device_name="cpu",
    )

    assert isinstance(outcome, EvaluationRunOutcome)
    assert outcome.status == "complete"
    assert outcome.output_dir == output_dir
    assert outcome.split == "validation"
    assert outcome.metrics == {"eval/metrics/prediction_mean": expected_mean}
    assert outcome.measurements == {"eval/measurements/latency_ms": 20.0}
    assert outcome.manifest_path == output_dir / "evaluation_manifest.yaml"
    assert outcome.result_path == output_dir / "result.json"
    assert checkpoint.read_bytes() == checkpoint_bytes

    result = _read_result(output_dir)
    assert result["schema_version"] == 1
    assert result["evaluation_id"] == f"tiny-{weights}-normal"
    assert result["protocol_id"] == "tiny-supervised-v1"
    assert len(result["protocol_digest"]) == 64
    assert result["status"] == "complete"
    assert result["metrics"] == {
        "eval/metrics/prediction_mean": expected_mean
    }
    assert result["measurements"] == {
        "eval/measurements/latency_ms": 20.0
    }

    subject = result["subject"]
    assert subject["kind"] == "checkpoint"
    assert Path(subject["path"]).resolve() == checkpoint.resolve()
    assert subject["sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert subject["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert subject["epoch"] == 4
    assert subject["global_step"] == 9
    assert subject["requested_weights"] == weights
    assert subject["resolved_weights"] == weights
    assert subject["lineage"] == {"run_id": "tiny-training-run"}

    data = result["data"]
    assert data["source"] == "checkpoint"
    assert data["split"] == "validation"
    assert data["builder"] == {"name": DATA_BUILDER_NAME, "params": {}}
    assert data["artifacts"] == DataArtifactBindings().to_dict()

    completeness = result["completeness"]
    assert completeness["strict_complete"] is True
    assert completeness["expected_examples"] == 3
    assert completeness["observed_examples"] == 3
    assert completeness["unique_sample_ids"] == 3
    assert completeness["missing_examples"] == 0
    assert completeness["complete"] is True
    assert len(completeness["sample_ids_sha256"]) == 64

    provenance = result["provenance"]
    assert provenance["evaluation_builder"] == {
        "name": EVALUATION_BUILDER_NAME,
        "params": {"scenario": "normal"},
    }
    assert provenance["metrics"] == [
        {
            "id": "prediction_mean",
            "name": METRIC_NAME,
            "channel": "tiny.predictions",
            "params": {"emit_nonfinite": False},
        }
    ]
    assert provenance["device"] == "cpu"
    assert provenance["seed"] == 19
    assert provenance["extension_plugins"] == []
    assert "extension_version_acceptance" in provenance

    manifest = yaml.safe_load(
        outcome.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "evaluation"
    assert manifest["status"] == "complete"
    assert manifest["purpose"] == "benchmark"
    assert manifest["split"] == "validation"
    assert manifest["subject"] == subject
    assert manifest["data"] == data
    assert manifest["completeness"] == completeness
    assert manifest["provenance"] == provenance
    assert manifest["result"]["path"] == "result.json"
    assert manifest["result"]["sha256"] == hashlib.sha256(
        outcome.result_path.read_bytes()
    ).hexdigest()
    assert (output_dir / manifest["resolved_config"]).is_file()

    evaluator = TinyEvaluationBuilder.last_evaluator
    assert evaluator is not None
    assert [batch.token for batch in evaluator.seen_batches] == [
        "validation-first",
        "validation-second",
    ]
    _assert_inference_lifecycle_restored()


@pytest.mark.parametrize(
    ("scenario", "expected_examples", "emit_nonfinite", "message"),
    [
        ("duplicate", 3, False, "duplicate"),
        ("normal", 4, False, "expected|missing|complete"),
        ("normal", 3, True, "finite"),
    ],
)
def test_invalid_evaluation_fails_closed_without_completion_marker(
    tmp_path: Path,
    scenario: str,
    expected_examples: int,
    emit_nonfinite: bool,
    message: str,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    config = _write_evaluation_config(
        tmp_path / "evaluation.yaml",
        checkpoint,
        scenario=scenario,
        expected_examples=expected_examples,
        emit_nonfinite=emit_nonfinite,
    )
    output_dir = tmp_path / "invalid-result"

    with pytest.raises(ValueError, match=message):
        run_evaluation(config, output_dir=output_dir, device_name="cpu")

    assert not (output_dir / "evaluation_manifest.yaml").exists()
    assert not (output_dir / "result.json").exists()
    _assert_inference_lifecycle_restored()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("sink_failure", "injected prediction sink failure"),
        ("sink_wrong_plan", "sample plan must match"),
    ],
)
def test_prediction_sink_failure_aborts_and_publishes_nothing(
    tmp_path: Path,
    scenario: str,
    message: str,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    config = _write_evaluation_config(
        tmp_path / "evaluation.yaml",
        checkpoint,
        scenario=scenario,
    )
    output_dir = tmp_path / "invalid-sink-result"

    with pytest.raises((RuntimeError, ValueError), match=message):
        run_evaluation(config, output_dir=output_dir, device_name="cpu")

    sink = TinyEvaluationBuilder.last_sink
    assert sink is not None
    assert sink.aborted is True
    assert not sink.path.exists()
    assert not output_dir.exists()
    _assert_inference_lifecycle_restored()


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    config = _write_evaluation_config(
        tmp_path / "evaluation.yaml",
        checkpoint,
    )
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "user-owned.txt"
    sentinel.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_evaluation(config, output_dir=output_dir, device_name="cpu")

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert not (output_dir / "evaluation_manifest.yaml").exists()


def test_resolved_evaluation_rejects_swapped_activation_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    config = _write_evaluation_config(tmp_path / "evaluation.yaml", checkpoint)
    requested = resolve_evaluation_inputs(config)
    other = resolve_evaluation_inputs(config)
    swapped_receipt = activate_extension_plugins(other.extension_plan)
    output_dir = tmp_path / "swapped-receipt"

    with pytest.raises(ValueError, match="different extension plan"):
        run_resolved_evaluation(
            requested,
            swapped_receipt,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert not output_dir.exists()


def test_strict_evaluation_requires_actual_data_artifact_bindings(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["config"]["data"]["params"] = {"omit_artifact_bindings": True}
    torch.save(payload, checkpoint)
    config = _write_evaluation_config(tmp_path / "evaluation.yaml", checkpoint)
    output_dir = tmp_path / "missing-bindings"

    with pytest.raises(ValueError, match="must return artifact bindings"):
        run_evaluation(config, output_dir=output_dir, device_name="cpu")

    assert not output_dir.exists()


def test_evaluate_cli_and_library_api_publish_equivalent_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    config = _write_evaluation_config(
        tmp_path / "evaluation.yaml",
        checkpoint,
        weights="ema",
    )
    api_output = tmp_path / "api"
    cli_output = tmp_path / "cli"

    run_evaluation(
        config,
        output_dir=api_output,
        device_name="cpu",
        force_extension_version_mismatch=True,
    )
    main(
        [
            "evaluate",
            "--config",
            str(config),
            "--device",
            "cpu",
            "--output-dir",
            str(cli_output),
            "--force-extension-version-mismatch",
        ]
    )

    captured = capsys.readouterr()
    assert "complete" in captured.out.lower()
    assert str(cli_output) in captured.out
    assert _read_result(cli_output) == _read_result(api_output)


def test_outcome_mappings_are_read_only_snapshots(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "subject.pt")
    config = _write_evaluation_config(
        tmp_path / "evaluation.yaml",
        checkpoint,
    )
    outcome = run_evaluation(
        config,
        output_dir=tmp_path / "result",
        device_name="cpu",
    )

    with pytest.raises(TypeError):
        cast(dict[str, float], outcome.metrics)["other"] = 1.0
    with pytest.raises(TypeError):
        cast(dict[str, Any], outcome.subject)["weights"] = "ema"
    assert isinstance(outcome.artifacts, Mapping)
