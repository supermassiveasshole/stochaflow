"""Standalone checkpoint evaluation orchestration."""

from __future__ import annotations

import hashlib
import platform
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torchmetrics import Metric

from stochaflow.data import DataArtifactBindings, DataLoaders, build_data_loaders
from stochaflow.evaluation.artifacts import (
    canonical_sha256,
    publish_evaluation_bundle,
)
from stochaflow.evaluation.builder import EvaluationPlan, build_evaluation_plan
from stochaflow.evaluation.config import (
    CheckpointSubjectConfig,
    EvaluationConfig,
    _thaw_evaluation_value,
    evaluation_config_to_dict,
    load_evaluation_config_snapshot,
)
from stochaflow.evaluation.contracts import (
    EvaluationResult,
    EvaluationRunOutcome,
    EvaluationStatus,
)
from stochaflow.evaluation.identity import build_protocol_implementation_identity
from stochaflow.evaluation.predictions import (
    PREDICTION_MANIFEST_FILENAME,
    PredictionArtifactDraft,
    PredictionArtifactSubjectInputs,
    PublishedPredictionArtifact,
    ResolvedPredictionArtifactSubject,
    load_prediction_artifact_inputs,
    materialize_prediction_manifest,
    resolve_prediction_artifact,
)
from stochaflow.evaluation.sampling import (
    CheckpointEvaluationSamplingCapability,
    EvaluationSamplingCapability,
)
from stochaflow.evaluation.subject import (
    CheckpointSubjectInputs,
    ResolvedCheckpointSubject,
    load_checkpoint_subject,
    resolve_checkpoint_subject,
)
from stochaflow.inference.extensions import prepare_checkpoint_extension_plan
from stochaflow.metrics.config import MetricSpec
from stochaflow.metrics.runtime import MetricEngine
from stochaflow.utils.config import StochaflowConfig
from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
    prepare_extension_plugins,
    require_resolved_extensions_for_plan,
)
from stochaflow.utils.registry import REGISTRIES, Registry
from stochaflow.utils.run_manifest import extension_runtime_metadata
from stochaflow.utils.seed import set_seed

EvaluationSubjectInputs = (
    CheckpointSubjectInputs | PredictionArtifactSubjectInputs
)
ResolvedEvaluationSubject = (
    ResolvedCheckpointSubject | ResolvedPredictionArtifactSubject
)


@dataclass(frozen=True, slots=True)
class ResolvedEvaluationInputs:
    """Preflighted config, checkpoint subject, and extension activation plan."""

    config: EvaluationConfig
    config_path: Path
    config_sha256: str
    subject_inputs: EvaluationSubjectInputs
    extension_plan: ExtensionActivationPlan


@dataclass(frozen=True, slots=True)
class EvaluationFacts:
    """Validated loop facts ready for immutable result publication."""

    status: EvaluationStatus
    metrics: Mapping[str, float]
    measurements: Mapping[str, float]
    completeness: Mapping[str, Any]
    sample_ids: tuple[str, ...]
    prediction_artifact: PredictionArtifactDraft | None = None


def resolve_evaluation_inputs(
    config_path: str | Path,
) -> ResolvedEvaluationInputs:
    """Preflight one standalone authority without importing extension code."""

    source = Path(config_path).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(f"evaluation config does not exist: {source}")
    config, encoded = load_evaluation_config_snapshot(source)
    if isinstance(config.subject, CheckpointSubjectConfig):
        subject_inputs: EvaluationSubjectInputs = load_checkpoint_subject(
            config.subject,
            base_dir=source.parent,
        )
    else:
        artifact_path = config.subject.path
        if not artifact_path.is_absolute():
            artifact_path = source.parent / artifact_path
        subject_inputs = load_prediction_artifact_inputs(artifact_path)
    extension_plan = prepare_checkpoint_extension_plan(
        subject_inputs.training_config_copy(),
        additions=tuple(config.extensions.plugins or ()),
        expected_provenance=subject_inputs.extension_provenance,
        plan_factory=prepare_extension_plugins,
    )
    return ResolvedEvaluationInputs(
        config=config,
        config_path=source,
        config_sha256=hashlib.sha256(encoded).hexdigest(),
        subject_inputs=subject_inputs,
        extension_plan=extension_plan,
    )


def run_evaluation(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    force_extension_version_mismatch: bool = False,
) -> EvaluationRunOutcome:
    """Run one strict standalone checkpoint evaluation from a YAML authority."""

    inputs = resolve_evaluation_inputs(config_path)
    extensions = activate_extension_plugins(
        inputs.extension_plan,
        policy=(
            ExtensionVersionPolicy.ALLOW
            if force_extension_version_mismatch
            else ExtensionVersionPolicy.REJECT
        ),
        acceptance_method=(
            "force-flag" if force_extension_version_mismatch else None
        ),
    )
    return run_resolved_evaluation(
        inputs,
        extensions,
        output_dir=output_dir,
        device_name=device_name,
    )


def run_resolved_evaluation(
    inputs: ResolvedEvaluationInputs,
    extensions: ResolvedExtensions,
    *,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
) -> EvaluationRunOutcome:
    """Execute evaluation after explicit extension activation."""

    if not isinstance(cast(object, inputs), ResolvedEvaluationInputs):
        raise TypeError("evaluation inputs must be ResolvedEvaluationInputs")
    if not isinstance(cast(object, extensions), ResolvedExtensions):
        raise TypeError("evaluation extensions must be ResolvedExtensions")
    require_resolved_extensions_for_plan(inputs.extension_plan, extensions)
    config = inputs.config
    training_config = extensions.config
    device = _resolve_evaluation_device(device_name)
    seed = training_config.experiment.seed
    set_seed(seed)
    subject_source = _subject_source_path(inputs.subject_inputs)
    target_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _default_evaluation_output_dir(subject_source)
    )
    artifact_root = _create_artifact_staging(target_dir)
    try:
        subject: ResolvedEvaluationSubject
        selected_data: Iterable[Any]
        data_identity: Mapping[str, Any]
        inference: object | None
        sampling: EvaluationSamplingCapability | None
        if isinstance(inputs.subject_inputs, CheckpointSubjectInputs):
            checkpoint_subject = resolve_checkpoint_subject(
                inputs.subject_inputs,
                device=device,
                resolved_config=training_config,
            )
            loaders = build_data_loaders(
                training_config.data,
                seed=seed,
                strict_resume=True,
                expected_artifacts=checkpoint_subject.data_artifacts,
            )
            selected_data = _selected_split(loaders, config.data.split)
            data_identity = _data_identity(
                training_config.data.name,
                training_config.data.params,
                split=config.data.split,
                expected_artifacts=checkpoint_subject.data_artifacts,
                actual_artifacts=loaders.artifact_bindings,
            )
            subject = checkpoint_subject
            inference = checkpoint_subject.model
            sampling = CheckpointEvaluationSamplingCapability(
                inputs=checkpoint_subject.inputs,
                config=training_config,
                model=checkpoint_subject.model,
                resolved_weights=checkpoint_subject.resolved_weights,
                device=device,
            )
        else:
            prediction_subject = resolve_prediction_artifact(
                inputs.subject_inputs,
            )
            if prediction_subject.split != config.data.split:
                raise ValueError(
                    "prediction artifact split does not match evaluation config"
                )
            subject = prediction_subject
            selected_data = prediction_subject.records
            data_identity = _offline_data_identity(prediction_subject)
            inference = None
            sampling = None

        plan = build_evaluation_plan(
            config.evaluation,
            subject=subject,
            data=selected_data,
            data_identity=data_identity,
            inference=inference,
            metric_specs=config.metrics,
            protocol=config.protocol,
            artifact_root=artifact_root,
            sampling=sampling,
        )
        if isinstance(subject, ResolvedCheckpointSubject) and all(
            module is not subject.model for module in plan.modules.values()
        ):
            raise ValueError(
                "EvaluationPlan.modules must declare the injected checkpoint model"
            )
        protocol_implementation = _protocol_implementation_identity(
            config,
            plan,
            extensions,
            seed=seed,
        )
        facts = execute_evaluation_plan(plan, device=device)
        if isinstance(subject, ResolvedPredictionArtifactSubject):
            expected_ids = tuple(
                sample.sample_id for sample in subject.inputs.samples
            )
            if facts.sample_ids != expected_ids:
                raise ValueError(
                    "offline evaluation sample IDs must match the prediction "
                    "artifact sample plan"
                )
        subject_identity = subject.identity
        provenance = _evaluation_provenance(
            config,
            extensions,
            device=device,
            seed=seed,
            protocol_implementation=protocol_implementation,
        )
        protocol_digest = _protocol_digest(
            config,
            data_identity=data_identity,
            protocol_implementation=protocol_implementation,
            sample_ids_sha256=canonical_sha256(facts.sample_ids),
        )
        prediction_artifact = _materialize_live_predictions(
            artifact_root,
            facts.prediction_artifact,
            config=config,
            config_sha256=inputs.config_sha256,
            protocol_digest=protocol_digest,
            subject=subject,
            data_identity=data_identity,
            training_config=training_config,
            extensions=extensions,
        )
        result_artifacts = (
            {
                "predictions": {
                    "path": f"predictions/{PREDICTION_MANIFEST_FILENAME}",
                    "sha256": prediction_artifact.manifest_sha256,
                    "artifact_digest": prediction_artifact.artifact_digest,
                }
            }
            if prediction_artifact is not None
            else {}
        )
        result = EvaluationResult(
            schema_version=1,
            evaluation_id=config.name,
            protocol_id=config.protocol.id,
            protocol_digest=protocol_digest,
            status=facts.status,
            subject=subject_identity,
            data=data_identity,
            metrics=facts.metrics,
            measurements=facts.measurements,
            artifacts=result_artifacts,
            completeness=facts.completeness,
            provenance=provenance,
        )
        resolved_config = evaluation_config_to_dict(config)
        cast(dict[str, Any], resolved_config["subject"])["path"] = str(
            subject_source
        )
        prepared_artifacts: Mapping[str, Path] | None = None
        if prediction_artifact is not None:
            prepared_artifacts = {"predictions": artifact_root}
        else:
            shutil.rmtree(artifact_root)
        published = publish_evaluation_bundle(
            target_dir,
            result=result,
            resolved_config=resolved_config,
            manifest_metadata={
                "config_source": str(inputs.config_path),
                "config_sha256": inputs.config_sha256,
                "purpose": config.purpose,
                "split": config.data.split,
                "subject": subject_identity,
                "data": data_identity,
                "artifacts": result_artifacts,
                "completeness": facts.completeness,
                "provenance": provenance,
                "runtime_options": {
                    "device": device_name,
                    "output_dir": (
                        str(output_dir) if output_dir is not None else None
                    ),
                },
            },
            prepared_artifacts=prepared_artifacts,
        )
        outcome_artifacts = {
            "resolved_config": published.resolved_config_path,
        }
        if prediction_artifact is not None:
            outcome_artifacts["predictions"] = (
                published.artifacts["predictions"]
                / PREDICTION_MANIFEST_FILENAME
            )
        return EvaluationRunOutcome(
            evaluation_id=config.name,
            protocol_id=config.protocol.id,
            status=result.status,
            output_dir=published.output_dir,
            subject=subject_identity,
            split=config.data.split,
            metrics=result.metrics,
            measurements=result.measurements,
            artifacts=outcome_artifacts,
            manifest_path=published.manifest_path,
            result_path=published.result_path,
        )
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def _subject_source_path(inputs: EvaluationSubjectInputs) -> Path:
    if isinstance(inputs, CheckpointSubjectInputs):
        return inputs.path
    return inputs.manifest_path


def _create_artifact_staging(target_dir: Path) -> Path:
    if not target_dir.name:
        raise ValueError("evaluation output directory must have a final name")
    if target_dir.exists():
        raise FileExistsError(
            f"evaluation output directory already exists: {target_dir}"
        )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.artifacts-",
            dir=target_dir.parent,
        )
    ).resolve()


def _offline_data_identity(
    subject: ResolvedPredictionArtifactSubject,
) -> Mapping[str, Any]:
    return subject.data_identity


def _materialize_live_predictions(
    artifact_root: Path,
    draft: PredictionArtifactDraft | None,
    *,
    config: EvaluationConfig,
    config_sha256: str,
    protocol_digest: str,
    subject: ResolvedEvaluationSubject,
    data_identity: Mapping[str, Any],
    training_config: StochaflowConfig,
    extensions: ResolvedExtensions,
) -> PublishedPredictionArtifact | None:
    if draft is None:
        return None
    encoded = evaluation_config_to_dict(config)
    subject_identity = subject.identity
    resolved_weights = (
        subject.resolved_weights
        if isinstance(subject, ResolvedCheckpointSubject)
        else subject.inputs.resolved_weights
    )
    source_subject_digest = (
        subject.content_digest
        if isinstance(subject, ResolvedCheckpointSubject)
        else subject.inputs.artifact_digest
    )
    return materialize_prediction_manifest(
        artifact_root,
        draft,
        producer={
            "kind": "evaluation",
            "id": config.name,
            "authority_sha256": config_sha256,
            "protocol_id": config.protocol.id,
            "protocol_digest": protocol_digest,
        },
        source_subject=subject_identity,
        source_subject_digest=source_subject_digest,
        resolved_weights=resolved_weights,
        inference_profile={
            "evaluation": encoded["evaluation"],
            "protocol_id": config.protocol.id,
            "protocol_digest": protocol_digest,
        },
        training_config=training_config,
        extension_provenance=extensions.provenance,
        data_identity=data_identity,
        split=config.data.split,
    )


def execute_evaluation_plan(
    plan: EvaluationPlan,
    *,
    device: torch.device,
    metric_registry: Registry[type[Metric]] = REGISTRIES.metrics,
) -> EvaluationFacts:
    """Execute one already-composed EvaluationPlan without publishing a bundle."""

    runtime_specs = tuple(
        MetricSpec(
            id=spec.id,
            name=spec.name,
            channel=spec.channel,
            params=cast(dict[str, Any], _thaw_evaluation_value(spec.params)),
        )
        for spec in plan.metric_specs
    )
    engine = MetricEngine(runtime_specs, registry=metric_registry).to(device)
    engine.reset()
    previous_modes = {
        name: module.training for name, module in plan.modules.items()
    }
    for module in plan.modules.values():
        module.eval()
    sample_ids: list[str] = []
    observed_ids: set[str] = set()
    observed_examples = 0
    measurement_names: frozenset[str] | None = None
    measurement_totals: dict[str, float] = {}
    sink = plan.artifact_sink
    sink_finalized = False
    try:
        with torch.inference_mode():
            for batch in plan.data:
                output = plan.evaluator.evaluate_batch(batch)
                for sample_id in output.sample_ids:
                    if sample_id in observed_ids:
                        raise ValueError(
                            f"evaluation contains duplicate sample id {sample_id!r}"
                        )
                    observed_ids.add(sample_id)
                    sample_ids.append(sample_id)
                observed_examples += output.num_examples
                if observed_examples > plan.protocol.expected_examples:
                    raise ValueError(
                        "evaluation observed more examples than the protocol allows"
                    )
                for updates in output.metric_update_groups:
                    engine.update(updates)
                if sink is not None:
                    sink.consume(output)
                current_names = frozenset(output.measurements)
                if measurement_names is None:
                    measurement_names = current_names
                elif current_names != measurement_names:
                    raise ValueError(
                        "evaluation measurement keys must be stable across batches"
                    )
                for name, value in output.measurements.items():
                    measurement_totals[name] = (
                        measurement_totals.get(name, 0.0)
                        + value * output.num_examples
                    )
        complete = observed_examples == plan.protocol.expected_examples
        if plan.protocol.strict_complete and not complete:
            raise ValueError(
                "strict evaluation is incomplete: expected "
                f"{plan.protocol.expected_examples}, observed {observed_examples}"
            )
        computed_metrics = engine.compute(reset=True)
        metrics = {
            f"eval/metrics/{name}": value
            for name, value in computed_metrics.items()
        }
        measurements = (
            {
                f"eval/measurements/{name}": total / observed_examples
                for name, total in measurement_totals.items()
            }
            if observed_examples > 0
            else {}
        )
        expected = plan.protocol.expected_examples
        completeness = {
            "strict_complete": plan.protocol.strict_complete,
            "expected_examples": expected,
            "observed_examples": observed_examples,
            "unique_sample_ids": len(observed_ids),
            "missing_examples": max(expected - observed_examples, 0),
            "complete": complete,
            "sample_ids_sha256": canonical_sha256(sample_ids),
        }
        prediction_artifact: PredictionArtifactDraft | None = None
        if sink is not None:
            draft_value = cast(object, sink.finalize())
            if not isinstance(draft_value, PredictionArtifactDraft):
                raise TypeError(
                    "EvaluationArtifactSink.finalize() must return "
                    "PredictionArtifactDraft"
                )
            prediction_artifact = draft_value
        if prediction_artifact is not None and tuple(
            sample.sample_id for sample in prediction_artifact.samples
        ) != tuple(sample_ids):
            raise ValueError(
                "prediction artifact sample plan must match evaluation order"
            )
        if prediction_artifact is not None:
            sink_finalized = True
        return EvaluationFacts(
            status="complete" if complete else "incomplete",
            metrics=metrics,
            measurements=measurements,
            completeness=completeness,
            sample_ids=tuple(sample_ids),
            prediction_artifact=prediction_artifact,
        )
    except BaseException as error:
        if sink is not None and not sink_finalized:
            try:
                sink.abort()
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(
                    f"prediction artifact cleanup failure: {cleanup_error}"
                )
        raise
    finally:
        engine.reset()
        for name, module in plan.modules.items():
            module.train(previous_modes[name])


def _selected_split(
    loaders: DataLoaders,
    split: str,
) -> Iterable[Any]:
    selected = (
        loaders.validation if split == "validation" else loaders.test
    )
    if selected is None:
        raise ValueError(f"evaluation data builder has no {split} split")
    return selected


def _data_identity(
    builder_name: str,
    builder_params: Mapping[str, Any],
    *,
    split: str,
    expected_artifacts: DataArtifactBindings | None,
    actual_artifacts: DataArtifactBindings | None,
) -> dict[str, Any]:
    if expected_artifacts is not None and actual_artifacts is None:
        raise ValueError(
            "evaluation data builder did not return checkpoint artifact identity"
        )
    if (
        expected_artifacts is not None
        and actual_artifacts is not None
        and expected_artifacts != actual_artifacts
    ):
        raise ValueError(
            "evaluation data artifacts do not match checkpoint identity"
        )
    artifacts = actual_artifacts
    return {
        "source": "checkpoint",
        "split": split,
        "builder": {
            "name": builder_name,
            "params": dict(builder_params),
        },
        "artifacts": artifacts.to_dict() if artifacts is not None else None,
    }


def _evaluation_provenance(
    config: EvaluationConfig,
    extensions: ResolvedExtensions,
    *,
    device: torch.device,
    seed: int,
    protocol_implementation: Mapping[str, Any],
) -> dict[str, Any]:
    encoded = evaluation_config_to_dict(config)
    extension_metadata = extension_runtime_metadata(extensions)
    return {
        "evaluation_builder": encoded["evaluation"],
        "metrics": encoded["metrics"],
        "device": str(device),
        "execution_environment": _execution_environment(device),
        "seed": seed,
        "protocol_implementation": protocol_implementation,
        **extension_metadata,
    }


def _protocol_implementation_identity(
    config: EvaluationConfig,
    plan: EvaluationPlan,
    extensions: ResolvedExtensions,
    *,
    seed: int,
    evaluation_builder_registry: Registry[type[Any]] = (
        REGISTRIES.evaluation_builders
    ),
    metric_registry: Registry[type[Metric]] = REGISTRIES.metrics,
) -> dict[str, Any]:
    """Bind task declarations to the exact registered and runtime providers."""

    return build_protocol_implementation_identity(
        evaluation_builder_name=config.evaluation.name,
        metric_specs=config.metrics,
        declared=plan.protocol_identity,
        evaluation_builder_registry=evaluation_builder_registry,
        metric_registry=metric_registry,
        runtime_parameters={"seed": seed},
        extension_provenance=extensions.provenance,
    )


def _execution_environment(device: torch.device) -> dict[str, Any]:
    """Record hardware facts without making hardware protocol compatibility."""

    result: dict[str, Any] = {
        "device": str(device),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        result.update(
            {
                "device_name": torch.cuda.get_device_name(index),
                "compute_capability": list(
                    torch.cuda.get_device_capability(index)
                ),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    elif device.type == "mps":
        result["mps_built"] = torch.backends.mps.is_built()
    return result


def _protocol_digest(
    config: EvaluationConfig,
    *,
    data_identity: Mapping[str, Any],
    protocol_implementation: Mapping[str, Any],
    sample_ids_sha256: str,
) -> str:
    encoded = evaluation_config_to_dict(config)
    return canonical_sha256(
        {
            "schema_version": 2,
            "purpose": config.purpose,
            "data": data_identity,
            "sample_ids_sha256": sample_ids_sha256,
            "evaluation": encoded["evaluation"],
            "metrics": encoded["metrics"],
            "protocol": encoded["protocol"],
            "implementation": protocol_implementation,
        }
    )


def _resolve_evaluation_device(device_name: str | None) -> torch.device:
    if device_name is None or device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _default_evaluation_output_dir(checkpoint_path: Path) -> Path:
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    root = checkpoint_path.parent.parent / "evaluations"
    result = root / timestamp
    suffix = 1
    while result.exists():
        result = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return result


__all__ = [
    "ResolvedEvaluationInputs",
    "execute_evaluation_plan",
    "resolve_evaluation_inputs",
    "run_evaluation",
    "run_resolved_evaluation",
]
