"""Formal E3 contracts for the AFHQ-v2 EvaluationBuilder extension."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch
import yaml
from torch import nn
from torchmetrics import Metric

from stochaflow.data import DataArtifactBindings, DataLoaders
from stochaflow.data.builder import DataBuilder
from stochaflow.evaluation import (
    CheckpointSubjectConfig,
    CheckpointSubjectInputs,
    EvaluationPlan,
    EvaluationProtocol,
    EvaluationSamplingRequest,
    PredictionArtifactDraft,
    ResolvedCheckpointSubject,
    build_evaluation_plan,
    load_evaluation_config,
    load_prediction_artifact,
    materialize_prediction_manifest,
    run_evaluation,
)
from stochaflow.evaluation.config import _thaw_evaluation_value
from stochaflow.inference.checkpoint import InferenceCheckpointView
from stochaflow.metrics import MetricEngine, MetricSpec
from stochaflow.sampling import (
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION
from stochaflow.utils.config import ComponentConfig, StochaflowConfig, load_config_dict
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    sampling_recipe_to_dict,
)

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_SRC = _ROOT / "examples" / "showcases" / "afhq-v2" / "src"
_FORMAL_PROFILE = (
    _ROOT
    / "examples"
    / "showcases"
    / "afhq-v2"
    / "experiments"
    / "evaluation"
    / "formal-ddim50-cfg2-official-test.yaml"
)
_EPSILON_FORMAL_PROFILE = (
    _ROOT
    / "examples"
    / "showcases"
    / "afhq-v2"
    / "experiments"
    / "evaluation"
    / "formal-ddim50-cfg2-official-test-epsilon.yaml"
)
_EPSILON_SELECTION_PROFILE = (
    _ROOT
    / "examples"
    / "showcases"
    / "afhq-v2"
    / "experiments"
    / "evaluation"
    / "selection-ddim50-cfg2-validation-epsilon.yaml"
)
_P2_CLOSEOUT_POLICY = (
    _ROOT
    / "examples"
    / "showcases"
    / "afhq-v2"
    / "experiments"
    / "evaluation"
    / "p2-production-closeout-policy.yaml"
)
if str(_EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_SRC))

afhq_evaluation = importlib.import_module(
    "stochaflow_afhq_v2.stochaflow_ext.evaluation"
)

TEST_PROVIDER = "test_afhq_v2_e3_image_gap"
RUNTIME_MODEL = "test_afhq_v2_e3_runtime_model"
RUNTIME_DATA = "test_afhq_v2_e3_runtime_data"
RUNTIME_SAMPLING = "test_afhq_v2_e3_runtime_sampling"


class TinyImageGapMetric(Metric):
    """Compute a deterministic real/fake image-mean gap without quality extras."""

    real_total: torch.Tensor
    real_count: torch.Tensor
    fake_total: torch.Tensor
    fake_count: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("real_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("real_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fake_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fake_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        values = images.mean(dim=(1, 2, 3))
        if real:
            self.real_total += values.sum()
            self.real_count += values.numel()
        else:
            self.fake_total += values.sum()
            self.fake_count += values.numel()

    def compute(self) -> torch.Tensor:
        return torch.abs(
            self.real_total / self.real_count - self.fake_total / self.fake_count
        )


if TEST_PROVIDER not in REGISTRIES.metrics.names():
    REGISTRIES.metrics.add(TEST_PROVIDER, TinyImageGapMetric)


class TinyAFHQModel(nn.Module):
    """Primary model identity retained by the live evaluation plan."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


class FakeAFHQSamplingCapability:
    """Return class-blocked generated samples through the public narrow seam."""

    def __init__(self) -> None:
        self.requests: list[EvaluationSamplingRequest] = []

    def execute(self, request: EvaluationSamplingRequest) -> SamplingOutput:
        self.requests.append(request)
        samples = torch.stack(
            (
                torch.full((3, 128, 128), -0.6),
                torch.full((3, 128, 128), 0.0),
                torch.full((3, 128, 128), 0.6),
            )
        )
        return SamplingOutput(
            batches=(
                SamplingBatch(samples=samples, num_samples=samples.shape[0]),
            ),
            metadata={"fixture": "class-blocked"},
        )


class RuntimeAFHQModel(nn.Module):
    """Registry-built checkpoint model used to prove pinned-provider identity."""

    constructor_calls: ClassVar[int] = 0
    last_instance: ClassVar[RuntimeAFHQModel | None] = None

    def __init__(self) -> None:
        super().__init__()
        type(self).constructor_calls += 1
        type(self).last_instance = self
        self.anchor = nn.Parameter(torch.zeros(()))


class RuntimeAFHQDataBuilder(DataBuilder):
    """Expose a deterministic official-test-shaped tiny class batch."""

    build_calls: ClassVar[int] = 0

    def build(self) -> DataLoaders:
        type(self).build_calls += 1
        images = torch.stack(
            (
                torch.full((3, 128, 128), 0.8),
                torch.full((3, 128, 128), -0.8),
                torch.full((3, 128, 128), 0.2),
            )
        )
        test = [
            (
                images,
                {"class_label": torch.tensor([2, 0, 1], dtype=torch.long)},
            )
        ]
        return DataLoaders(
            train=test,
            validation=test,
            test=test,
            artifact_bindings=DataArtifactBindings(),
        )


class RuntimeAFHQSamplingBuilder(SamplingBuilder):
    """Resolve the pinned model then return class-blocked image samples."""

    run_calls: ClassVar[int] = 0
    resolved_model: ClassVar[nn.Module | None] = None

    def run(self) -> SamplingOutput:
        type(self).run_calls += 1
        model, weights = self.context.model_provider.resolve("raw")
        type(self).resolved_model = model
        assert weights == "raw"
        samples = torch.stack(
            (
                torch.full((3, 128, 128), -0.6),
                torch.full((3, 128, 128), 0.0),
                torch.full((3, 128, 128), 0.6),
            )
        )
        return SamplingOutput(
            batches=(
                SamplingBatch(samples=samples, num_samples=samples.shape[0]),
            ),
            metadata={"weights": weights},
        )


if RUNTIME_MODEL not in REGISTRIES.models.names():
    REGISTRIES.models.add(RUNTIME_MODEL, RuntimeAFHQModel)
if RUNTIME_DATA not in REGISTRIES.data_builders.names():
    REGISTRIES.data_builders.add(RUNTIME_DATA, RuntimeAFHQDataBuilder)
if RUNTIME_SAMPLING not in REGISTRIES.sampling_builders.names():
    REGISTRIES.sampling_builders.add(RUNTIME_SAMPLING, RuntimeAFHQSamplingBuilder)


def training_config() -> StochaflowConfig:
    """Return a complete normalized producer authority."""

    return load_config_dict(
        {
            "experiment": {
                "name": "afhq-e3-test",
                "seed": 31,
                "output_dir": "unused",
            },
            "extensions": {"plugins": []},
            "data": {"name": "unused.afhq", "params": {}},
            "model": {"name": "unused.model", "params": {}},
            "training": {"name": "unused.training", "params": {}},
            "trainer": {"precision": "fp32"},
        }
    )


def checkpoint_subject(tmp_path: Path) -> ResolvedCheckpointSubject:
    """Build one already-resolved raw checkpoint subject for Builder injection."""

    path = (tmp_path / "subject.pt").resolve()
    path.write_bytes(b"subject identity only")
    inputs = CheckpointSubjectInputs(
        path=path,
        content_digest="c" * 64,
        format_version=CHECKPOINT_FORMAT_VERSION,
        epoch=2,
        global_step=7,
        requested_weights="raw",
        config=training_config().to_dict(),
        metadata={},
        extension_provenance=(),
        selected_components={},
        lineage={"run_id": "afhq-e3-source"},
        data_artifacts=None,
        _checkpoint_view=cast(InferenceCheckpointView, {}),
    )
    model = TinyAFHQModel().eval()
    model.requires_grad_(False)
    return ResolvedCheckpointSubject(inputs, model, "raw")


def profile_params(
    *,
    recipe_name: str = "class_conditional_denoising",
    recipe_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a tiny full-class profile preserving the production topology."""

    resolved_contract = (
        {
            "prediction_type": "v",
            "variance": {"mode": "fixed"},
        }
        if recipe_contract is None
        else dict(recipe_contract)
    )
    return {
        "class_mapping": {"cat": 0, "dog": 1, "wild": 2},
        "expected_per_class": {"cat": 1, "dog": 1, "wild": 1},
        "sampling": {
            "recipe": {
                "name": recipe_name,
                "contract": resolved_contract,
            },
            "sampler": {
                "name": "ddim",
                "params": {"num_inference_steps": 2, "eta": 0.0},
            },
            "options": {
                "weights": "raw",
                "clip_denoised": True,
                "guidance_scale": 2.0,
                "conditions": [
                    {"class_label": 0, "count": 1},
                    {"class_label": 1, "count": 1},
                    {"class_label": 2, "count": 1},
                ],
                "trajectory": {"enabled": False, "every_steps": 1},
            },
            "shape": [3, 128, 128],
            "num_samples": 3,
            "batch_size": 3,
            "seed": 20260801,
        },
        "gallery_count": 3,
    }


def metric_spec() -> MetricSpec:
    """Bind the composite AFHQ metric to one injected lightweight provider."""

    return MetricSpec(
        id="distribution",
        name=afhq_evaluation.AFHQ_V2_DISTRIBUTION_METRIC,
        channel=afhq_evaluation.AFHQ_V2_IMAGE_PAIR_CHANNEL,
        params={
            "class_mapping": {"cat": 0, "dog": 1, "wild": 2},
            "expected_real": {"cat": 1, "dog": 1, "wild": 1},
            "expected_fake": {"cat": 1, "dog": 1, "wild": 1},
            "providers": [{"name": TEST_PROVIDER, "params": {}}],
        },
    )


def runtime_training_config() -> StochaflowConfig:
    """Return the checkpoint authority for the end-to-end E3 runtime test."""

    return load_config_dict(
        {
            "experiment": {
                "name": "afhq-e3-runtime",
                "seed": 37,
                "output_dir": "unused",
            },
            "extensions": {"plugins": []},
            "data": {"name": RUNTIME_DATA, "params": {}},
            "model": {"name": RUNTIME_MODEL, "params": {}},
            "training": {"name": "unused.training", "params": {}},
            "trainer": {"precision": "fp32"},
        }
    )


def write_runtime_checkpoint(path: Path) -> Path:
    """Write a supported checkpoint with one task sampling recipe."""

    model = RuntimeAFHQModel()
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": 3,
            "global_step": 11,
            "model_state_dict": model.state_dict(),
            "inference_recipe": sampling_recipe_to_dict(
                SamplingRecipe(name=RUNTIME_SAMPLING, contract={})
            ),
            "inference_asset_descriptors": {},
            "config": runtime_training_config().to_dict(),
            "metadata": {
                "extension_plugins": [],
                "data_artifacts": DataArtifactBindings().to_dict(),
                "lineage": {"run_id": "afhq-e3-runtime-source"},
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
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one shared live/offline formal AFHQ evaluation authority."""

    spec = metric_spec()
    return {
        "version": 1,
        "name": name,
        "purpose": "final_test",
        "extensions": {"plugins": []},
        "subject": dict(subject),
        "data": {"source": source, "split": "test"},
        "evaluation": {
            "name": afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
            "params": (
                profile_params() if profile is None else dict(profile)
            ),
        },
        "metrics": [
            {
                "id": spec.id,
                "name": spec.name,
                "channel": spec.channel,
                "params": spec.params,
            }
        ],
        "protocol": {
            "id": protocol().id,
            "expected_examples": 3,
            "strict_complete": True,
        },
    }


def write_evaluation_document(path: Path, document: Mapping[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(dict(document), sort_keys=False), encoding="utf-8")
    return path


def protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        id="afhq-v2-e3-test-v1",
        expected_examples=3,
        strict_complete=True,
    )


def run_plan(
    plan: EvaluationPlan,
) -> tuple[dict[str, float], tuple[str, ...], PredictionArtifactDraft | None]:
    """Execute the public evaluator/metric/sink contracts for one tiny plan."""

    runtime_specs = tuple(
        MetricSpec(
            id=spec.id,
            name=spec.name,
            channel=spec.channel,
            params=cast(dict[str, Any], _thaw_evaluation_value(spec.params)),
        )
        for spec in plan.metric_specs
    )
    engine = MetricEngine(runtime_specs).to("cpu")
    engine.reset()
    sample_ids: list[str] = []
    for batch in cast(Iterable[Any], plan.data):
        output = plan.evaluator.evaluate_batch(batch)
        sample_ids.extend(output.sample_ids)
        for updates in output.metric_update_groups:
            engine.update(updates)
        if plan.artifact_sink is not None:
            plan.artifact_sink.consume(output)
    metrics = engine.compute(reset=True)
    draft = (
        plan.artifact_sink.finalize()
        if plan.artifact_sink is not None
        else None
    )
    return metrics, tuple(sample_ids), draft


def test_afhq_e3_live_predictions_replay_with_identical_formal_metrics(
    tmp_path: Path,
) -> None:
    subject = checkpoint_subject(tmp_path)
    sampling = FakeAFHQSamplingCapability()
    real_images = torch.stack(
        (
            torch.full((3, 128, 128), 0.8),
            torch.full((3, 128, 128), -0.8),
            torch.full((3, 128, 128), 0.2),
        )
    )
    live_data = [
        (
            real_images,
            {"class_label": torch.tensor([2, 0, 1], dtype=torch.long)},
        )
    ]
    artifact_root = (tmp_path / "prediction-staging").resolve()
    artifact_root.mkdir()
    live_plan = build_evaluation_plan(
        ComponentConfig(
            name=afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
            params=profile_params(),
        ),
        subject=subject,
        data=live_data,
        data_identity={"source": "checkpoint", "split": "test"},
        inference=subject.model,
        metric_specs=(metric_spec(),),
        protocol=protocol(),
        artifact_root=artifact_root,
        sampling=sampling,
    )

    live_metrics, live_ids, draft = run_plan(live_plan)

    assert len(sampling.requests) == 1
    assert sampling.requests[0].num_samples == 3
    assert draft is not None
    assert draft.preprocess["pairing"] == "same-class official-test allocation"
    assert draft.postprocess["codec"] == afhq_evaluation.AFHQ_V2_IMAGE_CODEC
    assert draft.gallery_count == 3
    published = materialize_prediction_manifest(
        artifact_root,
        draft,
        producer={
            "kind": "evaluation",
            "id": "afhq-e3-live",
            "authority_sha256": "a" * 64,
            "protocol_id": protocol().id,
            "protocol_digest": "b" * 64,
        },
        source_subject=subject.identity,
        source_subject_digest=subject.content_digest,
        resolved_weights="raw",
        inference_profile={
            "evaluation": {
                "name": afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
                "params": profile_params(),
            },
            "protocol_id": protocol().id,
            "protocol_digest": "b" * 64,
        },
        training_config=training_config(),
        extension_provenance=(),
        data_identity={"source": "checkpoint", "split": "test"},
        split="test",
    )
    manifest = cast(
        dict[str, Any],
        json.loads(published.manifest_path.read_text(encoding="utf-8")),
    )
    assert manifest["preprocess"]["pairing"] == (
        "same-class official-test allocation"
    )
    assert manifest["postprocess"]["codec"] == (
        afhq_evaluation.AFHQ_V2_IMAGE_CODEC
    )
    assert manifest["gallery"]["count"] == 3

    offline_subject = load_prediction_artifact(published.manifest_path)
    offline_plan = build_evaluation_plan(
        ComponentConfig(
            name=afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
            params=profile_params(),
        ),
        subject=offline_subject,
        data=offline_subject.records,
        data_identity={
            "source": "prediction_artifact",
            "split": "test",
            "artifact_digest": offline_subject.inputs.artifact_digest,
        },
        inference=None,
        metric_specs=(metric_spec(),),
        protocol=protocol(),
        sampling=None,
    )
    offline_metrics, offline_ids, offline_draft = run_plan(offline_plan)

    assert offline_metrics == live_metrics
    assert offline_ids == live_ids
    assert offline_draft is None

    alternate_metric = metric_spec()
    alternate_metric.id = "alternate-distribution"
    build_evaluation_plan(
        ComponentConfig(
            name=afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
            params=profile_params(),
        ),
        subject=offline_subject,
        data=offline_subject.records,
        data_identity={"source": "prediction_artifact", "split": "test"},
        inference=None,
        metric_specs=(alternate_metric,),
        protocol=protocol(),
        sampling=None,
    )

    mismatched_profiles: list[dict[str, Any]] = []
    changed_recipe = deepcopy(profile_params())
    changed_recipe["sampling"]["recipe"]["name"] = "different-recipe"
    mismatched_profiles.append(changed_recipe)
    changed_sampler = deepcopy(profile_params())
    changed_sampler["sampling"]["sampler"]["params"]["num_inference_steps"] = 3
    mismatched_profiles.append(changed_sampler)
    changed_options = deepcopy(profile_params())
    changed_options["sampling"]["options"]["guidance_scale"] = 3.0
    mismatched_profiles.append(changed_options)
    changed_batch_size = deepcopy(profile_params())
    changed_batch_size["sampling"]["batch_size"] = 1
    mismatched_profiles.append(changed_batch_size)
    changed_seed = deepcopy(profile_params())
    changed_seed["sampling"]["seed"] += 1
    mismatched_profiles.append(changed_seed)

    for mismatched_profile in mismatched_profiles:
        with pytest.raises(ValueError, match="offline generation profile"):
            build_evaluation_plan(
                ComponentConfig(
                    name=afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
                    params=mismatched_profile,
                ),
                subject=offline_subject,
                data=offline_subject.records,
                data_identity={"source": "prediction_artifact", "split": "test"},
                inference=None,
                metric_specs=(metric_spec(),),
                protocol=protocol(),
                sampling=None,
            )


def test_afhq_validation_prediction_plan_uses_validation_identity(
    tmp_path: Path,
) -> None:
    subject = checkpoint_subject(tmp_path)
    sampling = FakeAFHQSamplingCapability()
    artifact_root = (tmp_path / "validation-predictions").resolve()
    artifact_root.mkdir()
    plan = build_evaluation_plan(
        ComponentConfig(
            name=afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER,
            params=profile_params(),
        ),
        subject=subject,
        data=[
            (
                torch.zeros(3, 3, 128, 128),
                {"class_label": torch.tensor([0, 1, 2])},
            )
        ],
        data_identity={"source": "checkpoint", "split": "validation"},
        inference=subject.model,
        metric_specs=(metric_spec(),),
        protocol=protocol(),
        artifact_root=artifact_root,
        sampling=sampling,
    )

    _, _, draft = run_plan(plan)

    assert draft is not None
    assert draft.preprocess["reference"] == (
        "authenticated AFHQ-v2 validation manifest order"
    )
    assert draft.preprocess["pairing"] == "same-class validation allocation"
    assert [sample.input_id for sample in draft.samples] == [
        "afhq-v2:validation:000000",
        "afhq-v2:validation:000001",
        "afhq-v2:validation:000002",
    ]


def test_afhq_e3_core_runtime_uses_pinned_sampling_and_offline_replay(
    tmp_path: Path,
) -> None:
    checkpoint = write_runtime_checkpoint(tmp_path / "subject.pt")
    RuntimeAFHQModel.constructor_calls = 0
    RuntimeAFHQModel.last_instance = None
    RuntimeAFHQDataBuilder.build_calls = 0
    RuntimeAFHQSamplingBuilder.run_calls = 0
    RuntimeAFHQSamplingBuilder.resolved_model = None
    runtime_profile = profile_params(
        recipe_name=RUNTIME_SAMPLING,
        recipe_contract={},
    )
    live_config = write_evaluation_document(
        tmp_path / "live.yaml",
        evaluation_document(
            name="afhq-e3-runtime-live",
            subject={
                "kind": "checkpoint",
                "path": str(checkpoint),
                "weights": "raw",
            },
            source="checkpoint",
            profile=runtime_profile,
        ),
    )

    live = run_evaluation(
        live_config,
        output_dir=tmp_path / "live-result",
        device_name="cpu",
    )

    assert live.status == "complete"
    assert RuntimeAFHQModel.constructor_calls == 1
    assert RuntimeAFHQDataBuilder.build_calls == 1
    assert RuntimeAFHQSamplingBuilder.run_calls == 1
    assert RuntimeAFHQSamplingBuilder.resolved_model is (
        RuntimeAFHQModel.last_instance
    )
    counters = (
        RuntimeAFHQModel.constructor_calls,
        RuntimeAFHQDataBuilder.build_calls,
        RuntimeAFHQSamplingBuilder.run_calls,
    )
    prediction_manifest = live.artifacts["predictions"]
    offline_config = write_evaluation_document(
        tmp_path / "offline.yaml",
        evaluation_document(
            name="afhq-e3-runtime-offline",
            subject={
                "kind": "prediction_artifact",
                "path": str(prediction_manifest),
            },
            source="prediction_artifact",
            profile=runtime_profile,
        ),
    )

    offline = run_evaluation(
        offline_config,
        output_dir=tmp_path / "offline-result",
        device_name="cpu",
    )

    assert offline.status == "complete"
    assert offline.metrics == live.metrics
    assert (
        RuntimeAFHQModel.constructor_calls,
        RuntimeAFHQDataBuilder.build_calls,
        RuntimeAFHQSamplingBuilder.run_calls,
    ) == counters


def test_afhq_e3_rejects_checkpoint_recipe_identity_before_sampling(
    tmp_path: Path,
) -> None:
    checkpoint = write_runtime_checkpoint(tmp_path / "subject.pt")
    RuntimeAFHQSamplingBuilder.run_calls = 0
    mismatched = profile_params(
        recipe_name="different.sampling.recipe",
        recipe_contract={},
    )
    config = write_evaluation_document(
        tmp_path / "mismatch.yaml",
        evaluation_document(
            name="afhq-e3-recipe-mismatch",
            subject={
                "kind": "checkpoint",
                "path": str(checkpoint),
                "weights": "raw",
            },
            source="checkpoint",
            profile=mismatched,
        ),
    )
    output_dir = tmp_path / "mismatch-result"

    with pytest.raises(ValueError, match="recipe name"):
        run_evaluation(config, output_dir=output_dir, device_name="cpu")

    assert RuntimeAFHQSamplingBuilder.run_calls == 0
    assert not output_dir.exists()


def test_afhq_image_codec_rejects_tampered_prediction_bytes() -> None:
    encoded = afhq_evaluation._encoded_image(torch.zeros(3, 128, 128))
    encoded["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="sha256"):
        afhq_evaluation._decoded_image(encoded, path="tampered image")


def test_checked_in_afhq_e3_profile_freezes_full_official_test_authority() -> None:
    config = load_evaluation_config(_FORMAL_PROFILE)

    assert config.purpose == "final_test"
    assert isinstance(config.subject, CheckpointSubjectConfig)
    assert config.subject.weights == "ema"
    assert config.data.split == "test"
    assert config.evaluation.name == afhq_evaluation.AFHQ_V2_EVALUATION_BUILDER
    assert config.evaluation.params["expected_per_class"] == {
        "cat": 493,
        "dog": 491,
        "wild": 483,
    }
    assert config.protocol.expected_examples == 1467
    assert config.protocol.strict_complete is True
    assert config.metrics[0].name == afhq_evaluation.AFHQ_V2_DISTRIBUTION_METRIC
    assert config.metrics[0].params["providers"] == (
        {
            "name": "kid",
            "params": {
                "feature": 2048,
                "subsets": 100,
                "subset_size": 300,
                "degree": 3,
                "coef": 1.0,
                "seed": 20260726,
            },
        },
        {
            "name": "fid",
            "params": {"feature": 2048, "antialias": True},
        },
    )


def test_checked_in_p2_official_profile_is_epsilon_compatible_and_fails_closed() -> None:
    config = load_evaluation_config(_EPSILON_FORMAL_PROFILE)
    baseline = load_evaluation_config(_FORMAL_PROFILE)

    assert isinstance(config.subject, CheckpointSubjectConfig)
    assert config.subject.weights == "ema"
    assert config.subject.path.name == "REPLACE_WITH_SELECTED_EPOCH_CHECKPOINT.pt"
    assert "REPLACE_WITH_RUN_ID" in config.subject.path.parts
    assert "p2-ab" not in config.subject.path.parts
    assert "latest.pt" not in config.subject.path.parts
    assert config.evaluation.params["sampling"]["recipe"] == {
        "name": "class_conditional_denoising",
        "contract": {
            "prediction_type": "epsilon",
            "variance": {"mode": "fixed"},
        },
    }
    assert config.evaluation.params["expected_per_class"] == (
        baseline.evaluation.params["expected_per_class"]
    )
    assert config.evaluation.params["sampling"]["sampler"] == (
        baseline.evaluation.params["sampling"]["sampler"]
    )
    assert config.evaluation.params["sampling"]["options"] == (
        baseline.evaluation.params["sampling"]["options"]
    )
    assert config.metrics == baseline.metrics
    assert config.protocol.expected_examples == 1467
    assert config.protocol.strict_complete is True


def test_checked_in_p2_selection_profile_uses_only_validation() -> None:
    config = load_evaluation_config(_EPSILON_SELECTION_PROFILE)
    final_test = load_evaluation_config(_EPSILON_FORMAL_PROFILE)

    assert config.purpose == "selection_candidate"
    assert isinstance(config.subject, CheckpointSubjectConfig)
    assert config.subject.weights == "ema"
    assert config.subject.path.name == "epoch_0200.pt"
    assert "REPLACE_WITH_RUN_ID" in config.subject.path.parts
    assert config.data.split == "validation"
    assert config.evaluation.params["expected_per_class"] == {
        "cat": 300,
        "dog": 300,
        "wild": 300,
    }
    sampling = config.evaluation.params["sampling"]
    assert sampling["recipe"] == final_test.evaluation.params["sampling"]["recipe"]
    assert sampling["sampler"] == final_test.evaluation.params["sampling"]["sampler"]
    assert sampling["num_samples"] == 900
    assert sampling["batch_size"] == 30
    assert sampling["seed"] == 20260726
    assert sampling["options"]["conditions"] == (
        {"class_label": 0, "count": 300},
        {"class_label": 1, "count": 300},
        {"class_label": 2, "count": 300},
    )
    assert config.metrics[0].params["providers"][0] == {
        "name": "kid",
        "params": {
            "feature": 2048,
            "subsets": 100,
            "subset_size": 200,
            "degree": 3,
            "coef": 1.0,
            "seed": 20260726,
        },
    }
    assert config.protocol.id == (
        "afhq-v2-adm-epsilon-ddim50-cfg2-validation-v1"
    )
    assert config.protocol.expected_examples == 900
    assert config.protocol.strict_complete is True


def test_checked_in_p2_closeout_policy_freezes_selection_and_acceptance() -> None:
    raw = yaml.safe_load(_P2_CLOSEOUT_POLICY.read_text(encoding="utf-8"))

    assert raw["version"] == 1
    assert raw["name"] == "afhq-v2-p2-production-closeout-v1"
    assert raw["training"] == {
        "profile": "../production/train-adm-128-p2.yaml",
        "seed": 20260726,
        "device": "cuda",
        "deterministic": True,
        "epochs": 200,
        "expected_optimizer_updates": 84_000,
    }
    assert raw["selection"] == {
        "profile": "selection-ddim50-cfg2-validation-epsilon.yaml",
        "split": "validation",
        "subject_weights": "ema",
        "eligible_epochs": list(range(20, 201, 20)),
        "checkpoint_filename_format": "epoch_{epoch:04d}.pt",
        "expected_examples": 900,
        "primary_metric": {
            "key": "eval/metrics/distribution/aggregate.fid",
            "direction": "minimize",
        },
        "tie_breakers": [
            {
                "key": "eval/metrics/distribution/aggregate.kid_mean",
                "direction": "minimize",
            },
            {"key": "checkpoint_epoch", "direction": "minimize"},
        ],
    }
    assert raw["official_test"] == {
        "profile": "formal-ddim50-cfg2-official-test-epsilon.yaml",
        "split": "test",
        "run_count": 1,
        "expected_examples": 1_467,
        "strict_complete": True,
        "acceptance": {
            "require_all": True,
            "thresholds": [
                {
                    "key": "eval/metrics/distribution/aggregate.fid",
                    "maximum": 35.0,
                },
                {
                    "key": "eval/metrics/distribution/aggregate.kid_mean",
                    "maximum": 0.01,
                },
                {
                    "key": "eval/metrics/distribution/cat.fid",
                    "maximum": 65.0,
                },
                {
                    "key": "eval/metrics/distribution/dog.fid",
                    "maximum": 65.0,
                },
                {
                    "key": "eval/metrics/distribution/wild.fid",
                    "maximum": 65.0,
                },
            ],
        },
        "aspirational": {
            "key": "eval/metrics/distribution/aggregate.fid",
            "maximum": 30.0,
        },
    }
    policy_root = _P2_CLOSEOUT_POLICY.parent
    assert (policy_root / raw["training"]["profile"]).resolve().is_file()
    assert (policy_root / raw["selection"]["profile"]).resolve().is_file()
    assert (policy_root / raw["official_test"]["profile"]).resolve().is_file()
