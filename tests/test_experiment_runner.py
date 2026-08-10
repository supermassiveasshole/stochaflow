"""Tests for shared experiment runner orchestration."""

import hashlib
import pickle
import random
from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

import stochaflow.training as training_api
from stochaflow.data import (
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataLoaders,
)
from stochaflow.metrics import MetricSpec
from stochaflow.scripts import experiment_runner
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.training import (
    Trainer,
    TrainingDiagnostic,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
)
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    CheckpointState,
    capture_rng_state,
)
from stochaflow.utils.config import (
    ComponentConfig,
    ConfigError,
    ValidationEvaluationConfig,
    ValidationEvaluationProtocolConfig,
    load_config,
)
from stochaflow.utils.logging import ExperimentLogger, LocalLogger
from stochaflow.utils.plugins import ResolvedExtensions
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    sampling_recipe_to_dict,
)

BUILTIN_CONFIGS = Path(
    "examples/built-in/image-generation/configs"
)
MNIST_TRAIN_CONFIG = BUILTIN_CONFIGS / "train/mnist.yaml"
MNIST_OBSERVABILITY_CONFIG = (
    BUILTIN_CONFIGS / "overlays/mnist-observability.yaml"
)


class RecordingTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.global_step = 0
        self.best_checkpoint_path = None
        self.best_epoch = 1
        self.best_metric_value = 0.5
        self.stopped_early = False
        self.restored_fit_state = None
        self.checkpoint_dir: Path | None = None
        self.checkpoint_config: dict[str, Any] | None = None
        self.checkpoint_metadata = {"extension_plugins": []}
        self.fit_kwargs = {}
        self.evaluate_calls = 0

    def fit(self, dataloader, **kwargs):
        del dataloader
        self.fit_kwargs = kwargs
        return [{"loss": 0.5, "train/loss": 0.5, "num_batches": 1.0}]

    def evaluate_epoch(self, dataloader, **kwargs):
        del dataloader, kwargs
        self.evaluate_calls += 1
        return {"loss": 0.25, "num_batches": 1.0, "duration_seconds": 0.0}

    def restore_fit_state(self, state, *, best_checkpoint_path=None):
        self.restored_fit_state = (state, best_checkpoint_path)
        self.best_epoch = state["best_epoch"]
        self.best_metric_value = state["best_metric_value"]
        self.stopped_early = state["stopped_early"]
        self.best_checkpoint_path = best_checkpoint_path


class RecordingLogger:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class StreamingTensorDataset(IterableDataset[torch.Tensor]):
    def __iter__(self):
        for value in range(3):
            yield torch.tensor([float(value)])


class UnconsumableTestIterable(IterableDataset[torch.Tensor]):
    """Test split that fails on either metadata access or iteration."""

    @property
    def dataset(self) -> object:
        raise AssertionError("disabled test split metadata was consumed")

    def __iter__(self):
        raise AssertionError("disabled test split was consumed")


def _loader() -> DataLoader:
    dataset = TensorDataset(torch.zeros(2, 1))
    return DataLoader(dataset, batch_size=1)


def _loaders(*, validation: bool = False, test: bool = False) -> DataLoaders:
    return DataLoaders(
        train=_loader(),
        validation=_loader() if validation else None,
        test=_loader() if test else None,
    )


def test_runner_uses_explicit_steps_for_pytorch_streaming_loader() -> None:
    train_loader = DataLoader(StreamingTensorDataset(), batch_size=None)
    loaders = DataLoaders(train=train_loader, steps_per_epoch=3)

    assert experiment_runner._effective_steps_per_epoch(
        loaders,
        max_batches=None,
    ) == 3
    assert experiment_runner._effective_steps_per_epoch(
        loaders,
        max_batches=2,
    ) == 2
    assert experiment_runner._dataset_size(train_loader) is None


def _data_artifact_binding(
    *,
    artifact_digest: str = "4" * 64,
) -> DataArtifactBinding:
    return DataArtifactBinding(
        id="source",
        identity=DataArtifactIdentity(
            kind="managed",
            artifact_type="image-folder-v1",
            source_name="example.images",
            source_digest="1" * 64,
            materializer_name="example.resize",
            materialization_digest="2" * 64,
            content_digest="5" * 64,
            artifact_digest=artifact_digest,
            manifest_sha256="3" * 64,
        ),
    )


def _data_artifacts(
    *,
    artifact_digest: str = "4" * 64,
) -> DataArtifactBindings:
    return DataArtifactBindings(
        (_data_artifact_binding(artifact_digest=artifact_digest),)
    )


def _args() -> Namespace:
    return Namespace(
        config=None,
        epochs=None,
        limit_batches=None,
        limit_validation_batches=None,
        limit_test_batches=None,
        deterministic=False,
        progress=False,
        no_progress=True,
        artifact_verification_workers=None,
        resume=None,
        observability_config=None,
        device=None,
        output_dir=None,
    )


def _options(
    config,
    args: Namespace | None = None,
) -> experiment_runner.ExperimentRunOptions:
    return experiment_runner.ExperimentRunOptions.from_namespace(
        args or _args(),
        configured_num_epochs=config.trainer.num_epochs,
        configured_show_progress=config.trainer.show_progress,
    )


def _training_components(
    trainer: RecordingTrainer,
    logger: Any,
) -> Any:
    return SimpleNamespace(
        trainer=trainer,
        logger=logger,
        checkpoint_manager=SimpleNamespace(
            load=lambda *args, **kwargs: None,
            restore_payload=lambda *args, **kwargs: None,
        ),
        ema=None,
        process=SimpleNamespace(),
        plan=SimpleNamespace(inference_recipe=_inference_recipe()),
    )


def _inference_recipe() -> SamplingRecipe:
    return SamplingRecipe(
        name="standard_denoising",
        contract={"prediction_type": "v"},
    )


def test_training_run_outcome_is_public_and_deeply_immutable(tmp_path) -> None:
    outcome_type = training_api.TrainingRunOutcome
    final_metrics = {"train/loss": 0.5}
    phase_test_metrics = {"test/loss": 0.25}
    outcome = outcome_type(
        output_dir=tmp_path,
        final_epoch=2,
        final_metrics=final_metrics,
        latest_checkpoint=None,
        best_epoch=None,
        best_metric_name=None,
        best_metric_value=None,
        best_checkpoint=None,
        selected_checkpoint=None,
        selected_checkpoint_kind=None,
        stopped_early=False,
        phase_test_metrics=phase_test_metrics,
        manifest_path=tmp_path / "run_manifest.yaml",
        metrics_path=None,
        log_path=None,
    )

    assert outcome_type.__module__ == "stochaflow.training.outcome"
    assert not hasattr(outcome, "__dict__")
    final_metrics["train/loss"] = 9.0
    phase_test_metrics["test/loss"] = 8.0
    assert outcome.final_metrics["train/loss"] == 0.5
    assert outcome.phase_test_metrics["test/loss"] == 0.25
    with pytest.raises(FrozenInstanceError):
        setattr(outcome, "final_" + "epoch", 3)
    with pytest.raises(TypeError):
        cast(dict[str, float], outcome.final_metrics)["train/loss"] = 1.0
    with pytest.raises(TypeError):
        cast(dict[str, float], outcome.phase_test_metrics)["test/loss"] = 1.0


def _minimal_outcome_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "output_dir": tmp_path,
        "final_epoch": 2,
        "final_metrics": {"train/loss": 0.5},
        "latest_checkpoint": None,
        "best_epoch": None,
        "best_metric_name": None,
        "best_metric_value": None,
        "best_checkpoint": None,
        "selected_checkpoint": None,
        "selected_checkpoint_kind": None,
        "stopped_early": False,
        "phase_test_metrics": {},
        "manifest_path": tmp_path / "run_manifest.yaml",
        "metrics_path": None,
        "log_path": None,
    }


def test_training_run_outcome_final_selection_must_match_latest(tmp_path) -> None:
    kwargs = _minimal_outcome_kwargs(tmp_path)
    kwargs.update(
        latest_checkpoint=tmp_path / "checkpoints" / "latest.pt",
        selected_checkpoint=tmp_path / "checkpoints" / "other.pt",
        selected_checkpoint_kind="final",
    )

    with pytest.raises(ValueError, match=r"final selection.*latest checkpoint"):
        training_api.TrainingRunOutcome(**kwargs)


def test_training_run_outcome_best_epoch_cannot_exceed_final_epoch(
    tmp_path,
) -> None:
    kwargs = _minimal_outcome_kwargs(tmp_path)
    kwargs.update(
        best_epoch=3,
        best_metric_name="valid/loss",
        best_metric_value=0.4,
    )

    with pytest.raises(ValueError, match=r"best_epoch.*final_epoch"):
        training_api.TrainingRunOutcome(**kwargs)


@pytest.mark.parametrize("value", [0, 1, "false"])
def test_training_run_outcome_requires_exact_stopped_early_bool(
    tmp_path,
    value: object,
) -> None:
    kwargs = _minimal_outcome_kwargs(tmp_path)
    kwargs["stopped_early"] = value

    with pytest.raises(TypeError, match=r"stopped_early.*bool"):
        training_api.TrainingRunOutcome(**kwargs)


def _training_loop_state(
    *,
    best_epoch: int | None = None,
    best_metric_value: float | None = None,
    observations_without_improvement: int = 0,
    monitor_observations: int | None = None,
    stopped_early: bool = False,
    monitor: str | None = None,
    mode: str = "min",
    min_delta: float = 0.0,
    early_stopping_patience: int | None = None,
) -> dict[str, Any]:
    tracking_enabled = monitor is not None
    if monitor_observations is None:
        monitor_observations = (
            0
            if not tracking_enabled
            else max(
                observations_without_improvement,
                1 if best_epoch is not None else 0,
            )
        )
    return {
        "best_epoch": best_epoch,
        "best_metric_value": best_metric_value,
        "observations_without_improvement": observations_without_improvement,
        "monitor_observations": monitor_observations,
        "stopped_early": stopped_early,
        "tracking_enabled": tracking_enabled,
        "monitor_policy": (
            {
                "metric": monitor,
                "mode": mode,
                "min_delta": min_delta,
            }
            if monitor is not None
            else None
        ),
        "early_stopping_patience": early_stopping_patience,
    }


def _epoch_validation_loop_state(
    *,
    last_evaluated_epoch: int | None,
    monitor_observations: int | None = None,
    off_cadence_final_epochs: list[int] | None = None,
    fid_by_epoch: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    monitor = "valid/metrics/distribution/aggregate.fid"
    kid = "valid/metrics/distribution/aggregate.kid_mean"
    result_epochs: list[int] = []
    if last_evaluated_epoch is not None:
        result_epochs.extend(range(100, last_evaluated_epoch + 1, 10))
        if off_cadence_final_epochs is None:
            if last_evaluated_epoch not in result_epochs:
                result_epochs.append(last_evaluated_epoch)
        else:
            result_epochs.extend(off_cadence_final_epochs)
        result_epochs.sort()
    results = [
        {
            "epoch": epoch,
            "global_step": epoch * 2,
            "metrics": {
                monitor: (
                    fid_by_epoch[epoch]
                    if fid_by_epoch is not None
                    else 12.5 + len(result_epochs) - index - 1
                ),
                kid: 0.125,
            },
        }
        for index, epoch in enumerate(result_epochs)
    ]
    best_epoch = result_epochs[-1] if results else None
    best_metric_value = (
        cast(dict[str, float], results[-1]["metrics"])[monitor]
        if results
        else None
    )
    state = _training_loop_state(
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        monitor_observations=(
            len(results)
            if monitor_observations is None
            else monitor_observations
        ),
        monitor=monitor,
    )
    epoch_validation = {
        "schema_version": 1,
        "identity": {
            "profile_digest": "a" * 64,
            "metric_keys": [monitor, kid],
            "cadence": {
                "first_epoch": 100,
                "every_n_epochs": 10,
                "include_final": True,
            },
        },
        "results": results,
    }
    state["epoch_validation"] = epoch_validation
    return state


def _best_payload(
    loop_state: dict[str, Any],
    *,
    epoch: int | None = None,
) -> CheckpointState:
    best_epoch = loop_state["best_epoch"] if epoch is None else epoch
    monitor_policy = loop_state["monitor_policy"]
    assert isinstance(monitor_policy, dict)
    monitor = monitor_policy["metric"]
    best_snapshot_loop = {
        **loop_state,
        "observations_without_improvement": 0,
        "stopped_early": False,
    }
    if epoch is None:
        epoch_validation = loop_state.get("epoch_validation")
        epoch_validation_keys = (
            cast(dict[str, Any], epoch_validation)
            .get("identity", {})
            .get("metric_keys", [])
            if isinstance(epoch_validation, dict)
            else []
        )
        if monitor not in epoch_validation_keys:
            best_snapshot_loop["monitor_observations"] = best_epoch
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "precision_kind": "fp32",
        "inference_asset_descriptors": {},
        "inference_recipe": sampling_recipe_to_dict(_inference_recipe()),
        "epoch": best_epoch,
        "global_step": best_epoch * 2,
        "rng_state": capture_rng_state(),
        "model_state_dict": {},
        "training_assets_state_dict": {},
        "config": {"identity": "selected-run"},
        "metrics": {monitor: loop_state["best_metric_value"]},
        "metadata": {
            "extension_plugins": [],
            "checkpoint_kind": "best",
            "training_loop": (
                best_snapshot_loop if epoch is None else deepcopy(loop_state)
            ),
        },
    }


def _resume_candidate_payload(
    *,
    epoch: int,
    kind: str,
    best_epoch: int,
    best_metric_value: float,
    observations_without_improvement: int | None = None,
    monitor_observations: int | None = None,
    global_step: int | None = None,
) -> CheckpointState:
    """Build one strict filename-bound checkpoint candidate for resolver tests."""

    wait = (
        epoch - best_epoch
        if observations_without_improvement is None
        else observations_without_improvement
    )
    loop_state = _training_loop_state(
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        observations_without_improvement=wait,
        monitor_observations=(
            epoch if monitor_observations is None else monitor_observations
        ),
        monitor="valid/loss",
    )
    payload = _best_payload(loop_state, epoch=epoch)
    payload["global_step"] = epoch * 2 if global_step is None else global_step
    metadata = _checkpoint_metadata(payload)
    if kind == "periodic":
        metadata["checkpoint_kind"] = None
        metadata["training_loop"] = loop_state
    elif kind in {"best", "latest"}:
        metadata["checkpoint_kind"] = kind
        metadata["training_loop"] = (
            {
                **loop_state,
                "observations_without_improvement": 0,
                "stopped_early": False,
            }
            if kind == "best"
            else loop_state
        )
    else:
        raise ValueError(f"unsupported test checkpoint kind: {kind}")
    return payload


def _epoch_validation_candidate_payload(
    *,
    epoch: int,
    kind: str,
    last_evaluated_epoch: int,
) -> CheckpointState:
    """Build a directory candidate with a complete validation result history."""

    result_epochs = list(range(100, last_evaluated_epoch + 1, 10))
    if last_evaluated_epoch not in result_epochs:
        result_epochs.append(last_evaluated_epoch)
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=last_evaluated_epoch,
        fid_by_epoch={epoch: 1_000.0 - epoch for epoch in result_epochs},
    )
    payload = _best_payload(loop_state, epoch=epoch)
    payload["global_step"] = epoch * 2
    payload["config"] = {"trainer": {"num_epochs": 200}}
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = loop_state
    if kind == "periodic":
        metadata["checkpoint_kind"] = None
    elif kind in {"best", "latest"}:
        metadata["checkpoint_kind"] = kind
    else:
        raise ValueError(f"unsupported test checkpoint kind: {kind}")
    if last_evaluated_epoch == epoch:
        epoch_validation = cast(dict[str, Any], loop_state["epoch_validation"])
        results = cast(list[dict[str, Any]], epoch_validation["results"])
        payload["metrics"] = cast(dict[str, float], results[-1]["metrics"])
    else:
        payload["metrics"] = {}
    return payload


def _strict_resume_fields(*, epoch: int, global_step: int) -> CheckpointState:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": capture_rng_state(),
        "metrics": {},
        "metadata": {
            "training_loop": _training_loop_state(),
        },
    }


def _epoch_validation_resume_fields(
    *,
    epoch: int,
    global_step: int,
    configured_final_epoch: int = 200,
) -> CheckpointState:
    payload = _strict_resume_fields(epoch=epoch, global_step=global_step)
    payload["config"] = {"trainer": {"num_epochs": configured_final_epoch}}
    return payload


def _checkpoint_metadata(payload: CheckpointState) -> dict[str, Any]:
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    return metadata


def _write_training_checkpoint(
    path: Path,
    config,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    checkpoint_metadata = {
        "extension_plugins": [],
        "checkpoint_kind": "latest",
        "training_loop": _training_loop_state(),
        **({} if metadata is None else metadata),
    }
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "precision_kind": config.trainer.precision,
            "inference_asset_descriptors": {},
            "inference_recipe": sampling_recipe_to_dict(_inference_recipe()),
            "epoch": 1,
            "global_step": 0,
            "config": config.to_dict(),
            "model_state_dict": {},
            "training_assets_state_dict": {},
            "rng_state": capture_rng_state(),
            "metrics": {},
            "metadata": checkpoint_metadata,
        },
        path,
    )
    return path


def _write_observability_config(path: Path, value: Any) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _load_mnist_config():
    return load_config(MNIST_TRAIN_CONFIG)


def _resume_args(
    checkpoint: Path,
    *,
    observability_config: Path | None = None,
) -> Namespace:
    args = _args()
    args.resume = checkpoint
    args.observability_config = observability_config
    return args


def _run_single(
    config,
    loaders,
    options,
    *,
    checkpoint_payload=None,
):
    return experiment_runner._run_single_run(
        config,
        loaders,
        options,
        extensions=ResolvedExtensions(config, (), ()),
        config_source="checkpoint" if checkpoint_payload is not None else "external",
        checkpoint_payload=checkpoint_payload,
        startup_cwd=Path.cwd(),
        runtime_options={},
    )


def test_runner_uses_canonical_validation_loss_when_validation_is_available(
    monkeypatch,
    tmp_path,
):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    build_kwargs = {}

    def build_training_components(config, **kwargs):
        del config
        build_kwargs.update(kwargs)
        return _training_components(trainer, logger)

    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        build_training_components,
    )

    loaders = _loaders(validation=True)
    _run_single(config, loaders, _options(config))

    assert trainer.fit_kwargs["validation_dataloader"] is not None
    assert trainer.fit_kwargs["early_stopping_monitor"] == "valid/loss"
    assert trainer.fit_kwargs["track_best"] is True
    assert trainer.fit_kwargs["num_epochs"] == config.trainer.num_epochs
    assert build_kwargs["checkpoint_metadata"]["extension_plugins"] == []
    assert "diagnostic_data_iterables" not in build_kwargs
    assert logger.closed


def test_runner_injects_configured_validation_evaluation(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "live-validation"
    config.trainer.early_stopping.monitor = "valid/metrics/quality"
    config.trainer.validation_evaluation = ValidationEvaluationConfig(
        enabled=True,
        start_epoch=1,
        every_epochs=1,
        weights="raw",
        evaluation=ComponentConfig(name="tests.live-evaluation"),
        metrics=[MetricSpec("quality", "mean", "tests.values")],
        metric_keys=["valid/metrics/quality"],
        protocol=ValidationEvaluationProtocolConfig(
            id="tests-live-validation-v1",
            expected_examples=2,
            strict_complete=True,
        ),
    )
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    observed: dict[str, Any] = {}
    marker = object()

    def build_validator(**kwargs: Any) -> object:
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(
        experiment_runner,
        "EvaluationBackedEpochValidator",
        build_validator,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )
    loaders = _loaders(validation=True)

    _run_single(config, loaders, _options(config))

    assert trainer.fit_kwargs["epoch_validation_evaluator"] is marker
    assert observed["trainer"] is trainer
    assert observed["config"] is config.trainer.validation_evaluation
    assert observed["validation_data"] is loaders.validation
    assert observed["data_identity"] == {
        "source": "training",
        "split": "validation",
        "builder": {
            "name": config.data.name,
            "params": config.data.params,
        },
        "artifacts": None,
    }


def test_runner_disables_best_and_skips_test_without_validation(
    monkeypatch,
    tmp_path,
):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    _run_single(
        config,
        _loaders(),
        _options(config),
    )

    assert trainer.fit_kwargs["validation_dataloader"] is None
    assert trainer.fit_kwargs["early_stopping_monitor"] == "valid/loss"
    assert trainer.fit_kwargs["track_best"] is False
    assert trainer.evaluate_calls == 0
    assert logger.closed


def test_phase_test_metrics_preserve_custom_and_system_facts() -> None:
    config = _load_mnist_config()
    trainer = RecordingTrainer()
    observed: dict[str, Any] = {}

    def evaluate_epoch(dataloader, **kwargs):
        del dataloader
        observed.update(kwargs)
        return {
            "loss": 0.25,
            "num_batches": 2.0,
            "duration_seconds": 0.125,
            "test/metrics/custom": 0.75,
        }

    trainer.evaluate_epoch = evaluate_epoch
    training = _training_components(trainer, RecordingLogger())

    metrics = experiment_runner._evaluate_test_split(
        training,
        _loaders(test=True),
        _options(config),
        reporter=experiment_runner.RichTrainingReporter(),
    )

    assert metrics == {
        "test/loss": 0.25,
        "test/metrics/custom": 0.75,
        "system/test/num_batches": 2.0,
        "system/test/duration_seconds": 0.125,
    }
    assert observed["metric_prefix"] == "test"


def test_runner_does_not_consume_test_split_when_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "no-test-after-fit"
    config.trainer.test_after_fit = False
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )
    loaders = DataLoaders(
        train=_loader(),
        test=UnconsumableTestIterable(),
    )

    outcome = _run_single(config, loaders, _options(config))

    assert trainer.evaluate_calls == 0
    assert dict(outcome.phase_test_metrics) == {}
    manifest = yaml.safe_load(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["outcome"]["phase_test_metrics"] == {}
    assert logger.closed


def test_local_log_paths_are_absent_without_local_backend(tmp_path) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.logging.backends = [ComponentConfig(name="tensorboard")]

    assert experiment_runner._local_log_paths(config) == (None, None)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("metrics_filename", "../escaped-metrics.jsonl"),
        ("text_filename", "ABSOLUTE_PATH"),
    ],
)
def test_local_log_paths_reject_paths_outside_run_directory(
    tmp_path,
    field: str,
    unsafe_value: str,
) -> None:
    config = _load_mnist_config()
    output_dir = tmp_path / "run"
    config.experiment.output_dir = str(output_dir)
    value = (
        str(tmp_path / "escaped.log")
        if unsafe_value == "ABSOLUTE_PATH"
        else unsafe_value
    )
    config.logging.backends = [
        ComponentConfig(name="local", params={field: value})
    ]

    with pytest.raises(ValueError, match=r"local.*(?:filename|path|output)"):
        experiment_runner._local_log_paths(config)


def test_resolve_monitor_returns_validated_configuration_value() -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.trainer.early_stopping.monitor = "valid/metrics/prediction_mae"

    assert experiment_runner._resolve_monitor(config) == (
        "valid/metrics/prediction_mae"
    )


def test_runner_does_not_pass_removed_missing_policy_to_trainer(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    _run_single(
        config,
        _loaders(validation=True),
        _options(config),
    )

    assert "monitor_missing" not in trainer.fit_kwargs


def test_resolve_monitor_does_not_depend_on_loader_availability() -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.trainer.early_stopping.monitor = (
        "valid/metrics/prediction_mae"
    )

    assert experiment_runner._resolve_monitor(config) == (
        "valid/metrics/prediction_mae"
    )


def test_runner_allows_cli_epochs_override(monkeypatch, tmp_path):
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    build_kwargs = {}

    def build_training_components(config, **kwargs):
        del config
        build_kwargs.update(kwargs)
        return _training_components(trainer, logger)

    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        build_training_components,
    )
    args = _args()
    args.epochs = 3
    options = _options(config, args)

    _run_single(
        config,
        _loaders(),
        options,
    )

    assert config.trainer.num_epochs == 3
    assert trainer.fit_kwargs["num_epochs"] == 3
    assert build_kwargs["checkpoint_metadata"]["extension_plugins"] == []
    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    manifest = yaml.safe_load((tmp_path / "run_manifest.yaml").read_text())
    assert resolved["trainer"]["num_epochs"] == 3
    assert resolved["trainer"]["show_progress"] is False
    assert resolved["data"] == {
        "name": config.data.name,
        "params": config.data.params,
    }
    assert manifest["kind"] == "training"
    assert manifest["config_source"] == "external"
    assert manifest["extension_plugins"] == []
    assert manifest["config"] == resolved
    selected = manifest["selected_components"]
    assert build_kwargs["checkpoint_metadata"]["selected_components"] == selected
    assert config.process is not None
    assert selected["data_builder"] == config.data.name
    assert selected["model"] == config.model.name
    assert selected["training_builder"] == config.training.name
    assert selected["process"] == config.process.name
    assert "sampling_builder" not in selected
    assert selected["inference_recipe"] == "standard_denoising"
    assert "sampler" not in selected
    assert "artifact_writers" not in selected


def test_runner_persists_effective_enabled_progress_override(
    monkeypatch,
    tmp_path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "progress-override"
    config.trainer.show_progress = False
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )
    args = _args()
    args.progress = True
    args.no_progress = False

    _run_single(
        config,
        _loaders(validation=True),
        _options(config, args),
    )

    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    manifest = yaml.safe_load((tmp_path / "run_manifest.yaml").read_text())
    assert trainer.fit_kwargs["show_progress"] is True
    assert resolved["trainer"]["show_progress"] is True
    assert manifest["config"]["trainer"]["show_progress"] is True


def test_runner_returns_complete_outcome_and_persists_it_in_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "outcome"
    metrics_filename = "outcome-metrics.jsonl"
    text_filename = "outcome.log"
    config.logging.backends = [
        ComponentConfig(
            name="local",
            params={
                "metrics_filename": metrics_filename,
                "text_filename": text_filename,
            },
        )
    ]
    final_metrics = {
        "train/loss": 0.5,
        "train/metrics/custom": 0.6,
        "valid/loss": 0.4,
        "valid/metrics/custom": 0.7,
        "system/trainer/epoch": 1.0,
        "system/train/num_batches": 2.0,
        "system/valid/num_batches": 1.0,
    }
    phase_test_metrics = {
        "test/loss": 0.25,
        "test/metrics/custom": 0.75,
        "system/test/num_batches": 2.0,
        "system/test/duration_seconds": 0.125,
    }
    trainer = RecordingTrainer()
    trainer.best_epoch = 1
    trainer.best_metric_value = 0.4
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    latest_checkpoint = checkpoint_dir / "latest.pt"
    latest_checkpoint.touch()
    best_checkpoint = checkpoint_dir / "best.pt"
    best_checkpoint.touch()
    trainer.checkpoint_dir = checkpoint_dir
    trainer.best_checkpoint_path = best_checkpoint
    trainer.fit = lambda *args, **kwargs: [dict(final_metrics)]
    trainer.evaluate_epoch = lambda *args, **kwargs: {
        "loss": 0.25,
        "num_batches": 2.0,
        "duration_seconds": 0.125,
        "test/metrics/custom": 0.75,
    }
    logger = LocalLogger(
        output_dir=str(tmp_path),
        run_name="outcome",
        console=False,
        metrics_filename=metrics_filename,
        text_filename=text_filename,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    outcome = _run_single(
        config,
        _loaders(validation=True, test=True),
        _options(config),
    )

    outcome_type = training_api.TrainingRunOutcome
    manifest_path = tmp_path / "run_manifest.yaml"
    metrics_path = tmp_path / metrics_filename
    log_path = tmp_path / text_filename
    assert isinstance(outcome, outcome_type)
    assert outcome.output_dir == tmp_path
    assert outcome.final_epoch == 1
    assert dict(outcome.final_metrics) == final_metrics
    assert outcome.latest_checkpoint == latest_checkpoint
    assert outcome.best_epoch == 1
    assert outcome.best_metric_name == "valid/loss"
    assert outcome.best_metric_value == 0.4
    assert outcome.best_checkpoint == best_checkpoint
    assert outcome.selected_checkpoint == best_checkpoint
    assert outcome.selected_checkpoint_kind == "best"
    assert outcome.stopped_early is False
    assert dict(outcome.phase_test_metrics) == phase_test_metrics
    assert outcome.manifest_path == manifest_path
    assert outcome.metrics_path == metrics_path
    assert outcome.log_path == log_path

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["outcome"] == {
        "output_dir": str(tmp_path),
        "final_epoch": 1,
        "final_metrics": final_metrics,
        "latest_checkpoint": str(latest_checkpoint),
        "best_epoch": 1,
        "best_metric_name": "valid/loss",
        "best_metric_value": 0.4,
        "best_checkpoint": str(best_checkpoint),
        "selected_checkpoint": str(best_checkpoint),
        "selected_checkpoint_kind": "best",
        "stopped_early": False,
        "phase_test_metrics": phase_test_metrics,
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "log_path": str(log_path),
    }


def test_run_manifest_stays_running_when_training_fails(
    monkeypatch,
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "failed-outcome"
    trainer = RecordingTrainer()
    logger = RecordingLogger()

    def fail_fit(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("training failed")

    trainer.fit = fail_fit
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        _run_single(config, _loaders(), _options(config))

    manifest = yaml.safe_load(
        (tmp_path / "run_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "running"
    assert "outcome" not in manifest
    assert logger.closed
    assert logger.closed


def test_run_manifest_stays_running_when_final_reporter_fails(
    monkeypatch,
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "reporter-failure"
    trainer = RecordingTrainer()
    logger = RecordingLogger()

    class FailingFinalReporter:
        def on_run_start(self, summary: object) -> None:
            del summary

        def on_run_end(self, summary: object) -> None:
            del summary
            raise RuntimeError("final reporter failed")

    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )
    monkeypatch.setattr(
        experiment_runner,
        "RichTrainingReporter",
        FailingFinalReporter,
    )

    with pytest.raises(RuntimeError, match="final reporter failed"):
        _run_single(config, _loaders(), _options(config))

    manifest = yaml.safe_load(
        (tmp_path / "run_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "running"
    assert "outcome" not in manifest
    assert logger.closed


def test_run_manifest_stays_running_when_logger_close_fails(
    monkeypatch,
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "logger-close-failure"
    trainer = RecordingTrainer()

    class CloseFailingLogger(RecordingLogger):
        def close(self) -> None:
            self.closed = True
            raise RuntimeError("logger close failed")

    logger = CloseFailingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    with pytest.raises(RuntimeError, match="logger close failed"):
        _run_single(config, _loaders(), _options(config))

    manifest = yaml.safe_load(
        (tmp_path / "run_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "running"
    assert "outcome" not in manifest
    assert logger.closed


def test_fit_result_preserves_terminal_early_stop_before_best_restore(
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    trainer = RecordingTrainer()
    best_checkpoint = tmp_path / "checkpoints" / "best.pt"
    best_checkpoint.parent.mkdir()
    best_checkpoint.touch()
    trainer.best_checkpoint_path = best_checkpoint

    def fit(*args, **kwargs):
        del args, kwargs
        trainer.stopped_early = True
        return [{"train/loss": 0.5, "system/trainer/epoch": 1.0}]

    def restore_best(*args, **kwargs):
        del args, kwargs
        trainer.stopped_early = False

    trainer.fit = fit
    training = _training_components(trainer, RecordingLogger())
    training.checkpoint_manager = SimpleNamespace(load=restore_best)

    result = experiment_runner._fit_and_select_best(
        training,
        config,
        _loaders(validation=True),
        _options(config),
        start_epoch=1,
        reporter=cast(Any, SimpleNamespace()),
    )

    assert result.stopped_early is True
    assert trainer.stopped_early is False


def test_runner_does_not_sample_selected_best_checkpoint(monkeypatch, tmp_path):
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    trainer.best_checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    logger = RecordingLogger()
    training = _training_components(trainer, logger)
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    _run_single(
        config,
        _loaders(validation=True),
        _options(config),
    )

    assert not hasattr(experiment_runner, "run_sampling")
    assert not (tmp_path / "samples").exists()
    assert logger.closed


def test_runner_does_not_sample_latest_checkpoint_without_validation(
    monkeypatch,
    tmp_path,
) -> None:
    config = _load_mnist_config()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    trainer.best_epoch = None
    trainer.best_metric_value = None
    checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir = checkpoint_dir
    checkpoint_dir.mkdir(parents=True)
    final_checkpoint = checkpoint_dir / "latest.pt"
    final_checkpoint.touch()
    logger = RecordingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )
    _run_single(config, _loaders(), _options(config))

    assert final_checkpoint.is_file()
    assert not (tmp_path / "samples").exists()
    assert trainer.fit_kwargs["track_best"] is False
    assert logger.closed


def test_training_parser_does_not_accept_skip_final_sample() -> None:
    parser = ArgumentParser()
    experiment_runner.add_training_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--config", str(MNIST_TRAIN_CONFIG), "--skip-final-sample"]
        )


def test_runner_closes_logger_when_resume_loading_fails(monkeypatch, tmp_path):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()

    def fail_load(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("broken checkpoint")

    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        load=lambda *args, **kwargs: None,
        restore_payload=fail_load,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    args = _args()
    args.resume = tmp_path / "checkpoint.pt"

    with pytest.raises(RuntimeError, match="broken checkpoint"):
        _run_single(
            config,
            _loaders(),
            _options(config, args),
            checkpoint_payload=_strict_resume_fields(epoch=1, global_step=2),
        )

    assert logger.closed


def test_runner_rejects_checkpoint_at_target_epoch(monkeypatch, tmp_path):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        monitor="valid/loss",
    )
    loaded = SimpleNamespace(
        epoch=config.trainer.num_epochs,
        global_step=123,
        metadata={"training_loop": loop_state},
    )
    training = _training_components(trainer, logger)
    ema_devices = []
    training.ema = SimpleNamespace(to=ema_devices.append)
    training.checkpoint_manager = SimpleNamespace(
        load=lambda *args, **kwargs: None,
        restore_payload=lambda *args, **kwargs: loaded,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    args = _args()
    args.resume = tmp_path / "checkpoint.pt"

    with pytest.raises(ValueError, match="increase --epochs to continue"):
        _run_single(
            config,
            _loaders(),
            _options(config, args),
            checkpoint_payload=_strict_resume_fields(
                epoch=config.trainer.num_epochs,
                global_step=123,
            ),
        )

    assert trainer.global_step == 123
    assert trainer.restored_fit_state is None
    assert ema_devices == [trainer.device]
    assert logger.closed


def test_strict_resume_requires_sibling_best_for_latest_checkpoint(tmp_path):
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        monitor="valid/loss",
    )
    loaded = SimpleNamespace(
        epoch=1,
        global_step=2,
        metadata={"checkpoint_kind": "latest", "training_loop": loop_state},
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    with pytest.raises(FileNotFoundError, match=r"sibling.*best\.pt"):
        experiment_runner._restore_training_state(
            training,
            checkpoint,
            _strict_resume_fields(epoch=1, global_step=2),
            target_epoch=2,
        )


def test_strict_resume_recognizes_renamed_best_from_metadata(tmp_path):
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        monitor="valid/loss",
    )
    loaded = SimpleNamespace(
        epoch=1,
        global_step=2,
        metadata={"checkpoint_kind": "best", "training_loop": loop_state},
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "renamed.pt"
    payload = _best_payload(loop_state)
    CheckpointManager.save_payload(payload, checkpoint)
    new_checkpoint_dir = tmp_path / "new-run" / "checkpoints"
    trainer.checkpoint_dir = new_checkpoint_dir

    start_epoch = experiment_runner._restore_training_state(
        training,
        checkpoint,
        payload,
        target_epoch=2,
    )

    assert start_epoch == 2
    inherited = new_checkpoint_dir / "best.pt"
    assert inherited.is_file()
    assert trainer.restored_fit_state == (loop_state, None)
    assert trainer.best_checkpoint_path == inherited


def test_strict_resume_rejects_sibling_best_from_future_epoch(tmp_path):
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    selected_loop = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        monitor="valid/loss",
    )
    future_loop = {
        **selected_loop,
        "best_epoch": 3,
        "best_metric_value": 0.25,
        "observations_without_improvement": 0,
    }
    loaded = SimpleNamespace(
        epoch=2,
        global_step=4,
        metadata={"checkpoint_kind": None, "training_loop": selected_loop},
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "checkpoints" / "epoch_0002.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    CheckpointManager.save_payload(
        _best_payload(future_loop),
        checkpoint.parent / "best.pt",
    )
    new_checkpoint_dir = tmp_path / "new-run" / "checkpoints"
    trainer.checkpoint_dir = new_checkpoint_dir
    selected_payload = _best_payload(selected_loop, epoch=2)
    _checkpoint_metadata(selected_payload)["checkpoint_kind"] = None

    with pytest.raises(ValueError, match="best epoch mismatch"):
        experiment_runner._restore_training_state(
            training,
            checkpoint,
            selected_payload,
            target_epoch=3,
        )

    assert not (new_checkpoint_dir / "best.pt").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("best_metric_value", 0.25),
        ("monitor", "valid/metrics/other"),
        ("mode", "max"),
    ],
)
def test_strict_resume_rejects_mismatched_sibling_best_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    selected_loop = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        monitor="valid/loss",
    )
    candidate_loop = deepcopy(selected_loop)
    if field == "best_metric_value":
        candidate_loop[field] = value
    else:
        policy = candidate_loop["monitor_policy"]
        assert isinstance(policy, dict)
        policy["metric" if field == "monitor" else "mode"] = value
    loaded = SimpleNamespace(
        epoch=2,
        global_step=4,
        metadata={"checkpoint_kind": "latest", "training_loop": selected_loop},
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "old-run" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    CheckpointManager.save_payload(
        _best_payload(candidate_loop),
        checkpoint.parent / "best.pt",
    )
    trainer.checkpoint_dir = tmp_path / "new-run" / "checkpoints"
    selected_payload = _best_payload(selected_loop, epoch=2)
    _checkpoint_metadata(selected_payload)["checkpoint_kind"] = "latest"

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        experiment_runner._restore_training_state(
            training,
            checkpoint,
            selected_payload,
            target_epoch=3,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "observations_without_improvement",
            1,
            "less than monitor_observations",
        ),
        (
            "stopped_early",
            True,
            "stopped_early requires",
        ),
    ],
)
def test_strict_resume_rejects_noncanonical_best_snapshot_state(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        monitor="valid/loss",
    )
    selected = _best_payload(loop_state, epoch=2)
    _checkpoint_metadata(selected)["checkpoint_kind"] = "latest"
    candidate = _best_payload(loop_state)
    candidate_loop = _checkpoint_metadata(candidate)["training_loop"]
    assert isinstance(candidate_loop, dict)
    candidate_loop[field] = value

    with pytest.raises(ValueError, match=message):
        experiment_runner._validate_inherited_best(
            candidate,
            source=tmp_path / "best.pt",
            selected_payload=selected,
            best_epoch=1,
            best_metric=0.5,
            monitor="valid/loss",
            mode="min",
        )


@pytest.mark.parametrize(
    "mismatch",
    ["config", "plugins", "overlays", "data_artifacts"],
)
def test_strict_resume_rejects_sibling_from_another_run(
    tmp_path: Path,
    mismatch: str,
) -> None:
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        monitor="valid/loss",
    )
    selected_payload = _best_payload(loop_state, epoch=2)
    _checkpoint_metadata(selected_payload)["checkpoint_kind"] = "latest"
    loaded = SimpleNamespace(
        epoch=2,
        global_step=4,
        metadata=_checkpoint_metadata(selected_payload),
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "old-run" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    candidate = _best_payload(loop_state)
    if mismatch == "config":
        candidate["config"] = {"identity": "another-run"}
        expected = "config does not match"
    elif mismatch == "plugins":
        _checkpoint_metadata(candidate)["extension_plugins"] = [
            {
                "name": "other",
                "distribution": "other",
                "version": "1.0",
                "target": "other.stochaflow_ext",
            }
        ]
        expected = "extension provenance does not match"
    elif mismatch == "overlays":
        _checkpoint_metadata(candidate)["config_overlays"] = [
            {
                "kind": "observability",
                "source_path": str(tmp_path / "observability.yaml"),
                "source_sha256": "a" * 64,
                "sections": ["logging"],
                "logging_fields": ["log_every"],
            }
        ]
        expected = "config overlay history does not match"
    else:
        _checkpoint_metadata(selected_payload)["data_artifacts"] = (
            _data_artifacts().to_dict()
        )
        _checkpoint_metadata(candidate)["data_artifacts"] = _data_artifacts(
            artifact_digest="5" * 64
        ).to_dict()
        expected = "data artifacts do not match"
    CheckpointManager.save_payload(candidate, checkpoint.parent / "best.pt")
    trainer.checkpoint_dir = tmp_path / "new-run" / "checkpoints"

    with pytest.raises(ValueError, match=expected):
        experiment_runner._restore_training_state(
            training,
            checkpoint,
            selected_payload,
            target_epoch=3,
        )


def test_strict_resume_materializes_matching_sibling_best_in_new_run(tmp_path):
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        monitor="valid/loss",
    )
    loaded = SimpleNamespace(
        epoch=2,
        global_step=4,
        metadata={"checkpoint_kind": "latest", "training_loop": loop_state},
    )
    training = _training_components(trainer, logger)
    ema_devices = []
    training.ema = SimpleNamespace(to=ema_devices.append)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "old-run" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    source_best = checkpoint.parent / "best.pt"
    source_payload = _best_payload(loop_state)
    _checkpoint_metadata(source_payload)["extension_plugins"] = [
        {
            "name": "example",
            "distribution": "example",
            "version": "1.0",
            "target": "example.stochaflow_ext",
        }
    ]
    CheckpointManager.save_payload(source_payload, source_best)
    new_checkpoint_dir = tmp_path / "new-run" / "checkpoints"
    trainer.checkpoint_dir = new_checkpoint_dir
    effective_config = {
        "identity": "selected-run",
        "logging": {"log_every": 7},
    }
    config_overlays = [
        {
            "kind": "observability",
            "source_path": str(tmp_path / "observability.yaml"),
            "source_sha256": "a" * 64,
            "sections": ["logging"],
            "logging_fields": ["log_every"],
        }
    ]
    trainer.checkpoint_config = effective_config
    trainer.checkpoint_metadata = {
        "extension_plugins": [
            {
                "name": "example",
                "distribution": "example",
                "version": "2.0",
                "target": "example.stochaflow_ext",
            }
        ],
        "extension_version_acceptance": [],
        "config_overlays": config_overlays,
    }

    selected_payload = _best_payload(loop_state, epoch=2)
    selected_metadata = _checkpoint_metadata(selected_payload)
    selected_metadata["checkpoint_kind"] = "latest"
    selected_metadata["extension_plugins"] = _checkpoint_metadata(source_payload)[
        "extension_plugins"
    ]
    start_epoch = experiment_runner._restore_training_state(
        training,
        checkpoint,
        selected_payload,
        target_epoch=3,
    )

    inherited = new_checkpoint_dir / "best.pt"
    assert start_epoch == 3
    assert inherited.is_file()
    inherited_payload = CheckpointManager.load_payload(inherited, map_location="cpu")
    source_payload = CheckpointManager.load_payload(source_best, map_location="cpu")
    assert inherited_payload.get("model_state_dict") == source_payload.get(
        "model_state_dict"
    )
    assert inherited_payload.get("epoch") == source_payload.get("epoch")
    assert inherited_payload.get("config") == effective_config
    inherited_metadata = _checkpoint_metadata(inherited_payload)
    assert inherited_metadata["extension_plugins"][0]["version"] == "2.0"
    assert inherited_metadata["config_overlays"] == config_overlays
    assert inherited_metadata["inherited_from"] == str(source_best)
    assert trainer.best_checkpoint_path == inherited
    assert ema_devices == [trainer.device, trainer.device]


def test_materialized_staged_final_best_preserves_result_history(
    tmp_path: Path,
) -> None:
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=105,
        monitor_observations=2,
    )
    source_payload = _best_payload(loop_state, epoch=105)
    source_payload["global_step"] = 210
    source_payload["config"] = {"trainer": {"num_epochs": 105}}
    source_payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    source = tmp_path / "old-run" / "checkpoints" / "best.pt"
    source.parent.mkdir(parents=True)
    CheckpointManager.save_payload(source_payload, source)

    trainer = RecordingTrainer()
    trainer.best_epoch = 105
    trainer.best_metric_value = 12.5
    trainer.checkpoint_dir = tmp_path / "new-run" / "checkpoints"
    trainer.checkpoint_config = {"trainer": {"num_epochs": 200}}
    trainer.checkpoint_metadata = {"extension_plugins": []}
    training = _training_components(trainer, RecordingLogger())
    fit_state = experiment_runner._checkpoint_training_fit_state(source_payload)

    inherited = experiment_runner._materialize_inherited_best(
        training,
        checkpoint=source,
        checkpoint_payload=source_payload,
        checkpoint_kind="best",
        fit_state=fit_state,
    )

    assert inherited == trainer.checkpoint_dir / "best.pt"
    assert inherited is not None
    inherited_payload = CheckpointManager.load_payload(inherited)
    inherited_loop = _checkpoint_metadata(inherited_payload)["training_loop"]
    assert isinstance(inherited_loop, dict)
    inherited_validation = inherited_loop["epoch_validation"]
    assert isinstance(inherited_validation, dict)
    assert inherited_validation["schema_version"] == 1
    assert [
        result["epoch"] for result in inherited_validation["results"]
    ] == [100, 105]
    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        inherited_payload,
        require_cuda_compatibility=False,
    )
    assert (epoch, global_step) == (105, 210)


def test_strict_resume_rejects_terminal_early_stopping_state(tmp_path):
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=2,
        monitor_observations=3,
        stopped_early=True,
        monitor="valid/loss",
        early_stopping_patience=2,
    )
    loaded = SimpleNamespace(
        epoch=1,
        global_step=2,
        metadata={"checkpoint_kind": "best", "training_loop": loop_state},
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    checkpoint = tmp_path / "best-copy.pt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="already stopped early"):
        experiment_runner._restore_training_state(
            training,
            checkpoint,
            _strict_resume_fields(epoch=1, global_step=2),
            target_epoch=2,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epoch", None, "epoch as a positive integer"),
        ("epoch", True, "epoch as a positive integer"),
        ("epoch", 0, "epoch as a positive integer"),
        ("global_step", None, "global_step as a non-negative integer"),
        ("global_step", True, "global_step as a non-negative integer"),
        ("global_step", -1, "global_step as a non-negative integer"),
    ],
)
def test_strict_resume_rejects_missing_or_invalid_progress(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _strict_resume_fields(epoch=1, global_step=2)
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value  # type: ignore[literal-required]

    with pytest.raises(TypeError, match=message):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_non_numeric_epoch_metrics() -> None:
    payload = _strict_resume_fields(epoch=1, global_step=2)
    payload["metrics"] = cast(Any, {"train/loss": "invalid"})

    with pytest.raises(
        TypeError,
        match=r"metrics\['train/loss'\] must be numeric",
    ):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_requires_dense_monitor_in_current_metrics() -> None:
    payload = _strict_resume_fields(epoch=1, global_step=2)
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _training_loop_state(
        monitor="valid/loss",
    )

    with pytest.raises(ValueError, match=r"missing monitor 'valid/loss'"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_explicit_file_resume_rejects_observations_without_best_state(
    tmp_path: Path,
) -> None:
    payload = _strict_resume_fields(epoch=1, global_step=2)
    payload["metrics"] = {"valid/loss": 0.5}
    _checkpoint_metadata(payload)["training_loop"] = _training_loop_state(
        monitor="valid/loss",
        monitor_observations=1,
    )

    with pytest.raises(ValueError, match="requires complete best state"):
        experiment_runner._preflight_inherited_best(
            tmp_path / "checkpoint.pt",
            payload,
            target_epoch=2,
        )


@pytest.mark.parametrize(
    ("monitor_observations", "wait", "message"),
    [
        (1, 0, "monitor_observations must equal the checkpoint epoch"),
        (2, 0, "must equal checkpoint_epoch minus best_epoch"),
        (2, 2, "less than monitor_observations"),
    ],
)
def test_explicit_file_resume_rejects_incomplete_dense_monitor_state(
    tmp_path: Path,
    monitor_observations: int,
    wait: int,
    message: str,
) -> None:
    payload = _strict_resume_fields(epoch=2, global_step=4)
    payload["metrics"] = {"valid/loss": 0.5}
    _checkpoint_metadata(payload)["training_loop"] = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=wait,
        monitor_observations=monitor_observations,
        monitor="valid/loss",
    )

    with pytest.raises(ValueError, match=message):
        experiment_runner._preflight_inherited_best(
            tmp_path / "checkpoint.pt",
            payload,
            target_epoch=3,
        )


@pytest.mark.parametrize(
    ("checkpoint_epoch", "last_evaluated_epoch"),
    [(99, None), (101, 100)],
)
def test_strict_resume_allows_sparse_monitor_between_epoch_evaluations(
    checkpoint_epoch: int,
    last_evaluated_epoch: int | None,
) -> None:
    payload = _epoch_validation_resume_fields(
        epoch=checkpoint_epoch,
        global_step=checkpoint_epoch * 2,
    )
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=last_evaluated_epoch,
    )

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (checkpoint_epoch, checkpoint_epoch * 2)


def test_strict_resume_requires_scheduled_epoch_validation_state() -> None:
    payload = _epoch_validation_resume_fields(epoch=100, global_step=200)
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=None,
    )

    with pytest.raises(
        ValueError,
        match="missing a declared interval or final observation",
    ):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_accepts_recorded_staged_final_observation() -> None:
    payload = _epoch_validation_resume_fields(epoch=101, global_step=202)
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=101,
    )

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (101, 202)


def test_strict_resume_requires_complete_current_epoch_validation_metrics() -> None:
    monitor = "valid/metrics/distribution/aggregate.fid"
    payload = _epoch_validation_resume_fields(epoch=100, global_step=200)
    payload["metrics"] = {monitor: 12.5}
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=100,
    )

    with pytest.raises(ValueError, match="missing epoch validation key"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_requires_current_epoch_validation_metrics_to_match() -> None:
    payload = _epoch_validation_resume_fields(epoch=100, global_step=200)
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 11.0,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=100,
    )

    with pytest.raises(ValueError, match="disagree with epoch validation"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_stale_metrics_at_non_evaluated_epoch() -> None:
    payload = _epoch_validation_resume_fields(epoch=101, global_step=202)
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
    }
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=100,
    )

    with pytest.raises(ValueError, match="contain stale epoch validation key"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_epoch_validation_state_from_future() -> None:
    payload = _epoch_validation_resume_fields(epoch=99, global_step=198)
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=100,
    )

    with pytest.raises(ValueError, match="ahead of the checkpoint epoch"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_accepts_evaluated_off_cadence_final_epoch() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=105,
        global_step=210,
        configured_final_epoch=105,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=105,
        monitor_observations=2,
    )

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (105, 210)


def test_strict_resume_preserves_staged_final_before_target_expansion() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=105,
        global_step=210,
        configured_final_epoch=105,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=105,
        monitor_observations=2,
    )

    persisted = experiment_runner._checkpoint_training_fit_state(
        payload
    ).to_dict()
    epoch_validation = persisted["epoch_validation"]
    assert isinstance(epoch_validation, dict)
    assert [
        result["epoch"] for result in epoch_validation["results"]
    ] == [100, 105]

    payload["config"] = {"trainer": {"num_epochs": 200}}
    metadata["training_loop"] = persisted
    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (105, 210)


@pytest.mark.parametrize(
    ("checkpoint_epoch", "monitor_observations"),
    [(110, 2), (200, 11)],
)
def test_strict_resume_round_trips_complete_interval_history(
    checkpoint_epoch: int,
    monitor_observations: int,
) -> None:
    payload = _epoch_validation_resume_fields(
        epoch=checkpoint_epoch,
        global_step=checkpoint_epoch * 2,
        configured_final_epoch=200,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    _checkpoint_metadata(payload)["training_loop"] = (
        _epoch_validation_loop_state(
            last_evaluated_epoch=checkpoint_epoch,
            monitor_observations=monitor_observations,
        )
    )

    persisted = experiment_runner._checkpoint_training_fit_state(
        payload
    ).to_dict()

    epoch_validation = persisted["epoch_validation"]
    assert isinstance(epoch_validation, dict)
    assert [
        result["epoch"] for result in epoch_validation["results"]
    ] == list(range(100, checkpoint_epoch + 1, 10))


@pytest.mark.parametrize(
    ("checkpoint_epoch", "monitor_observations"),
    [(110, 3), (200, 12)],
)
def test_strict_resume_rejects_monitor_count_mismatching_result_history(
    checkpoint_epoch: int,
    monitor_observations: int,
) -> None:
    payload = _epoch_validation_resume_fields(
        epoch=checkpoint_epoch,
        global_step=checkpoint_epoch * 2,
        configured_final_epoch=200,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    _checkpoint_metadata(payload)["training_loop"] = (
        _epoch_validation_loop_state(
            last_evaluated_epoch=checkpoint_epoch,
            monitor_observations=monitor_observations,
        )
    )

    with pytest.raises(ValueError, match=r"exactly match.*observation history"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


@pytest.mark.parametrize("monitor_kind", ["disabled", "ordinary"])
def test_strict_resume_accepts_history_without_selection_authority(
    monitor_kind: str,
) -> None:
    payload = _epoch_validation_resume_fields(
        epoch=110,
        global_step=220,
        configured_final_epoch=200,
    )
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=110,
        monitor_observations=2,
    )
    if monitor_kind == "disabled":
        epoch_validation = loop_state["epoch_validation"]
        loop_state = _training_loop_state()
        loop_state["epoch_validation"] = epoch_validation
        payload["metrics"] = {
            "valid/metrics/distribution/aggregate.fid": 12.5,
            "valid/metrics/distribution/aggregate.kid_mean": 0.125,
        }
    else:
        policy = loop_state["monitor_policy"]
        assert isinstance(policy, dict)
        policy["metric"] = "valid/loss"
        loop_state["best_metric_value"] = 0.5
        loop_state["monitor_observations"] = 110
        payload["metrics"] = {
            "valid/loss": 0.5,
            "valid/metrics/distribution/aggregate.fid": 12.5,
            "valid/metrics/distribution/aggregate.kid_mean": 0.125,
        }
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (110, 220)


def test_strict_resume_accepts_complete_staged_final_history() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=115,
        global_step=230,
        configured_final_epoch=200,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=115,
        monitor_observations=4,
        off_cadence_final_epochs=[105, 115],
        fid_by_epoch={100: 11.0, 105: 10.0, 110: 11.0, 115: 12.5},
    )
    loop_state["best_epoch"] = 105
    loop_state["best_metric_value"] = 10.0
    loop_state["observations_without_improvement"] = 2
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (115, 230)


def test_strict_resume_rejects_result_history_count_mismatch() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=115,
        global_step=230,
        configured_final_epoch=115,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=115,
        monitor_observations=4,
    )
    loop_state["best_epoch"] = 105
    loop_state["best_metric_value"] = 10.0
    loop_state["observations_without_improvement"] = 2
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    with pytest.raises(ValueError, match=r"exactly match.*observation history"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_preserves_staged_final_in_later_checkpoint() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=106,
        global_step=212,
        configured_final_epoch=105,
    )
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=105,
        monitor_observations=2,
    )
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (106, 212)


def test_strict_resume_accepts_complete_history_with_ordinary_monitor() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=105,
        global_step=210,
        configured_final_epoch=105,
    )
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=105,
        monitor_observations=2,
    )
    loop_state["best_metric_value"] = 0.5
    loop_state["monitor_observations"] = 105
    policy = loop_state["monitor_policy"]
    assert isinstance(policy, dict)
    policy["metric"] = "valid/loss"
    payload["metrics"] = {
        "valid/loss": 0.5,
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (105, 210)


def test_strict_resume_rejects_result_history_ahead_of_checkpoint_epoch() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=115,
        global_step=230,
        configured_final_epoch=200,
    )
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=116,
    )
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    with pytest.raises(ValueError, match="ahead of the checkpoint epoch"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_result_history_ahead_of_checkpoint_step() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=110,
        global_step=220,
    )
    payload["metrics"] = {
        "valid/metrics/distribution/aggregate.fid": 12.5,
        "valid/metrics/distribution/aggregate.kid_mean": 0.125,
    }
    loop_state = _epoch_validation_loop_state(last_evaluated_epoch=110)
    epoch_validation = loop_state["epoch_validation"]
    assert isinstance(epoch_validation, dict)
    results = epoch_validation["results"]
    assert isinstance(results, list)
    last_result = results[-1]
    assert isinstance(last_result, dict)
    last_result["global_step"] = 221
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    with pytest.raises(ValueError, match="ahead of the checkpoint global_step"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_non_list_result_history() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=100,
        global_step=200,
    )
    loop_state = _epoch_validation_loop_state(
        last_evaluated_epoch=100,
        monitor_observations=1,
        off_cadence_final_epochs=[],
    )
    epoch_validation = loop_state["epoch_validation"]
    assert isinstance(epoch_validation, dict)
    epoch_validation["results"] = None
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    with pytest.raises(TypeError, match="results must be a list"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_unversioned_legacy_result_summary() -> None:
    payload = _epoch_validation_resume_fields(epoch=100, global_step=200)
    loop_state = _epoch_validation_loop_state(last_evaluated_epoch=100)
    loop_state["epoch_validation"] = {
        "identity": cast(dict[str, Any], loop_state["epoch_validation"])[
            "identity"
        ],
        "last_evaluated_epoch": 100,
        "last_metrics": {
            "valid/metrics/distribution/aggregate.fid": 12.5,
            "valid/metrics/distribution/aggregate.kid_mean": 0.125,
        },
        "off_cadence_final_epochs": [],
    }
    _checkpoint_metadata(payload)["training_loop"] = loop_state

    with pytest.raises(ValueError, match="schema_version"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_unevaluated_off_cadence_final_epoch() -> None:
    payload = _epoch_validation_resume_fields(
        epoch=105,
        global_step=210,
        configured_final_epoch=105,
    )
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _epoch_validation_loop_state(
        last_evaluated_epoch=100,
    )

    with pytest.raises(
        ValueError,
        match="missing the scheduled observation at epoch 105",
    ):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_rejects_diagnostic_monitor_policy() -> None:
    payload = _strict_resume_fields(epoch=1, global_step=2)
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _training_loop_state(
        monitor="diagnostics/quality/fid",
    )

    with pytest.raises(ValueError, match="canonical validation metric key"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_resume_ignores_removed_metric_source_metadata() -> None:
    payload = _strict_resume_fields(epoch=1, global_step=2)
    payload["metrics"] = {"valid/loss": 0.5}
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        monitor_observations=1,
        monitor="valid/loss",
    )
    metadata["metric_sources"] = {
        "valid/loss": {
            "origin": "phase",
            "data_role": "validation",
            "protocol_id": None,
            "selection_eligible": False,
        }
    }

    epoch, global_step, _ = experiment_runner._parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )

    assert (epoch, global_step) == (1, 2)


def test_strict_resume_does_not_accept_diagnostic_source_provenance() -> None:
    monitor = "diagnostics/quality/fid"
    payload = _strict_resume_fields(epoch=1, global_step=2)
    payload["metrics"] = {monitor: 12.5}
    metadata = _checkpoint_metadata(payload)
    metadata["training_loop"] = _training_loop_state(
        monitor=monitor,
    )
    metadata["metric_sources"] = {
        monitor: {
            "origin": "diagnostic",
            "data_role": "external",
            "protocol_id": None,
            "selection_eligible": True,
        }
    }

    with pytest.raises(ValueError, match="canonical validation metric key"):
        experiment_runner._parse_strict_resume_state(
            payload,
            require_cuda_compatibility=False,
        )


def test_strict_mps_resume_warns_when_legacy_v8_has_no_mps_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = RecordingTrainer()
    trainer.device = torch.device("mps")
    trainer.best_epoch = None
    trainer.best_metric_value = None
    logger = RecordingLogger()
    loop_state = _training_loop_state()
    loaded = SimpleNamespace(
        epoch=1,
        global_step=2,
        metadata={"checkpoint_kind": "latest", "training_loop": loop_state},
    )
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        restore_payload=lambda *args, **kwargs: loaded
    )
    payload = _strict_resume_fields(epoch=1, global_step=2)
    rng_state = payload.get("rng_state")
    assert rng_state is not None
    rng_state.pop("torch_mps")
    restore_calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        experiment_runner,
        "restore_rng_state",
        lambda state, *, restore_cuda, restore_mps: restore_calls.append(
            (restore_cuda, restore_mps)
        ),
    )

    with pytest.warns(RuntimeWarning, match="does not contain MPS RNG state"):
        start_epoch = experiment_runner._restore_training_state(
            training,
            tmp_path / "latest.pt",
            payload,
            target_epoch=2,
        )

    assert start_epoch == 2
    assert restore_calls == [(False, True)]


class StochasticTestStrategy(TrainingStrategy):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(self, batch: Any) -> TrainStepOutput:
        prediction = self.model(batch)
        random_target = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
        )
        loss = (prediction - random_target).square().mean()
        return TrainStepOutput(loss=loss)


class RNGConsumingTestDiagnostic(TrainingDiagnostic):
    @staticmethod
    def _consume_rng() -> None:
        random.random()
        np.random.random()
        torch.rand(())

    def on_fit_start(self, event: Any) -> None:
        del event
        self._consume_rng()

    def on_train_batch_end(self, event: Any) -> None:
        del event
        self._consume_rng()

    def on_train_epoch_end(self, event: Any) -> None:
        del event
        self._consume_rng()


class BestRNGConsumingTestLogger(ExperimentLogger):
    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del step
        if "best/epoch" in metrics:
            RNGConsumingTestDiagnostic._consume_rng()

    def close(self) -> None:
        return None


class EpochEndRNGConsumingTestReporter:
    def on_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_phase_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_batch_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def on_phase_end(self) -> None:
        return None

    def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        RNGConsumingTestDiagnostic._consume_rng()


def _stochastic_trainer(
    checkpoint_dir: Path,
    *,
    logger: ExperimentLogger | None = None,
) -> Trainer:
    model = nn.Linear(1, 1)
    plan = TrainingPlan(
        strategy=StochasticTestStrategy(model),
        primary_model=model,
    )
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    manager = CheckpointManager(model=model, optimizer=optimizer)
    return Trainer(
        plan,
        optimizer,
        device="cpu",
        diagnostics=[RNGConsumingTestDiagnostic()],
        checkpoint_manager=manager,
        checkpoint_dir=checkpoint_dir,
        checkpoint_config={"identity": "rng-resume"},
        checkpoint_metadata={"extension_plugins": []},
        logger=logger,
        log_every=1000,
    )


def _seed_stochastic_test() -> None:
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)


@pytest.mark.parametrize("hook", ["fit_start", "batch_end", "epoch_end"])
def test_trainer_isolates_each_diagnostic_rng_callback(
    tmp_path: Path,
    hook: str,
) -> None:
    trainer = _stochastic_trainer(tmp_path / hook)
    _seed_stochastic_test()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    if hook == "fit_start":
        trainer._emit_fit_start_diagnostics(
            train_dataloader=[],
            validation_dataloader=None,
        )
    elif hook == "batch_end":
        trainer._emit_batch_diagnostics(
            batch=torch.ones(1, 1),
            output=TrainStepOutput(loss=torch.zeros(())),
            loss=0.0,
            global_step=1,
            epoch_index=1,
        )
    else:
        trainer._emit_epoch_diagnostics(
            epoch_index=1,
            metrics={},
        )

    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_strict_resume_matches_uninterrupted_stochastic_training(
    tmp_path: Path,
) -> None:
    loader = [torch.ones(4, 1), torch.full((4, 1), 2.0)]
    _seed_stochastic_test()
    uninterrupted = _stochastic_trainer(tmp_path / "uninterrupted")
    uninterrupted.fit(
        loader,
        num_epochs=2,
        show_progress=False,
        reporter=EpochEndRNGConsumingTestReporter(),
    )
    expected_state = {
        name: value.detach().clone()
        for name, value in uninterrupted.model.state_dict().items()
    }

    _seed_stochastic_test()
    interrupted = _stochastic_trainer(tmp_path / "interrupted")
    interrupted.fit(
        loader,
        num_epochs=1,
        show_progress=False,
        reporter=EpochEndRNGConsumingTestReporter(),
    )
    checkpoint = tmp_path / "interrupted" / "latest.pt"
    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed = _stochastic_trainer(tmp_path / "resumed")
    training: Any = SimpleNamespace(
        trainer=resumed,
        checkpoint_manager=resumed.checkpoint_manager,
        ema=None,
    )
    start_epoch = experiment_runner._restore_training_state(
        training,
        checkpoint,
        payload,
        target_epoch=2,
    )
    resumed.fit(
        loader,
        num_epochs=2,
        start_epoch=start_epoch,
        show_progress=False,
        reporter=EpochEndRNGConsumingTestReporter(),
    )

    assert resumed.global_step == uninterrupted.global_step
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected_state[name])


def test_strict_resume_from_best_isolates_post_checkpoint_logger_rng(
    tmp_path: Path,
) -> None:
    loader = [torch.ones(4, 1), torch.full((4, 1), 2.0)]
    _seed_stochastic_test()
    uninterrupted = _stochastic_trainer(
        tmp_path / "uninterrupted-best",
        logger=BestRNGConsumingTestLogger(),
    )
    uninterrupted.fit(
        loader,
        num_epochs=2,
        validation_dataloader=loader,
        show_progress=False,
        early_stopping_monitor="valid/loss",
        track_best=True,
    )
    expected_state = {
        name: value.detach().clone()
        for name, value in uninterrupted.model.state_dict().items()
    }

    _seed_stochastic_test()
    interrupted = _stochastic_trainer(
        tmp_path / "interrupted-best",
        logger=BestRNGConsumingTestLogger(),
    )
    interrupted.fit(
        loader,
        num_epochs=1,
        validation_dataloader=loader,
        show_progress=False,
        early_stopping_monitor="valid/loss",
        track_best=True,
    )
    checkpoint = tmp_path / "interrupted-best" / "best.pt"
    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")

    resumed = _stochastic_trainer(
        tmp_path / "resumed-best",
        logger=BestRNGConsumingTestLogger(),
    )
    training: Any = SimpleNamespace(
        trainer=resumed,
        checkpoint_manager=resumed.checkpoint_manager,
        ema=None,
    )
    start_epoch = experiment_runner._restore_training_state(
        training,
        checkpoint,
        payload,
        target_epoch=2,
    )
    resumed.fit(
        loader,
        num_epochs=2,
        start_epoch=start_epoch,
        validation_dataloader=loader,
        show_progress=False,
        early_stopping_monitor="valid/loss",
        track_best=True,
    )

    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected_state[name])


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("limit_batches", 0, "--limit-batches"),
        ("limit_validation_batches", -1, "--limit-validation-batches"),
        ("limit_test_batches", 0, "--limit-test-batches"),
    ],
)
def test_run_options_reject_non_positive_limits(
    attribute,
    value,
    message,
):
    args = _args()
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        experiment_runner.ExperimentRunOptions.from_namespace(
            args,
            configured_num_epochs=2,
            configured_show_progress=True,
        )


@pytest.mark.parametrize(
    (
        "configured_show_progress",
        "force_progress",
        "suppress_progress",
        "expected",
    ),
    [
        pytest.param(True, False, False, True, id="inherit-enabled"),
        pytest.param(False, False, False, False, id="inherit-disabled"),
        pytest.param(True, False, True, False, id="force-disabled"),
        pytest.param(False, True, False, True, id="force-enabled"),
    ],
)
def test_run_options_resolve_progress_override(
    configured_show_progress: bool,
    force_progress: bool,
    suppress_progress: bool,
    expected: bool,
) -> None:
    args = _args()
    args.progress = force_progress
    args.no_progress = suppress_progress

    options = experiment_runner.ExperimentRunOptions.from_namespace(
        args,
        configured_num_epochs=2,
        configured_show_progress=configured_show_progress,
    )

    assert options.show_progress is expected


def test_run_options_reject_conflicting_progress_overrides() -> None:
    args = _args()
    args.progress = True
    args.no_progress = True

    with pytest.raises(ValueError, match="mutually exclusive"):
        experiment_runner.ExperimentRunOptions.from_namespace(
            args,
            configured_num_epochs=2,
            configured_show_progress=False,
        )


def test_runner_builds_registered_data_builder(monkeypatch, tmp_path):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path / "outputs")
    args = _args()
    args.config = Path("unused.yaml")
    args.artifact_verification_workers = 3
    observed = {}
    expected_outcome = object()

    def stub_builder(
        data_config,
        *,
        seed,
        strict_resume,
        expected_artifacts,
        verification_observer,
        verification_workers,
    ):
        observed["builder_config"] = data_config
        observed["seed"] = seed
        observed["strict_resume"] = strict_resume
        observed["expected_artifacts"] = expected_artifacts
        observed["verification_observer"] = verification_observer
        observed["verification_workers"] = verification_workers
        return _loaders()

    monkeypatch.setattr(experiment_runner, "load_config", lambda path: config)
    monkeypatch.setattr(experiment_runner, "build_data_loaders", stub_builder)
    monkeypatch.setattr(
        experiment_runner,
        "_run_single_run",
        lambda *args, **kwargs: expected_outcome,
    )

    outcome = experiment_runner.run_experiment_from_args(args)

    assert outcome is expected_outcome
    assert observed == {
        "builder_config": config.data,
        "seed": config.experiment.seed,
        "strict_resume": False,
        "expected_artifacts": None,
        "verification_observer": None,
        "verification_workers": 3,
    }


def test_runner_rejects_unsupported_precision_before_data_or_run_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.trainer.precision = "fp16-mixed"
    args = _args()
    args.config = Path("unused.yaml")
    args.device = "cpu"
    args.output_dir = tmp_path / "outputs"

    monkeypatch.setattr(experiment_runner, "load_config", lambda path: config)

    def unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail("unsupported precision reached data or run creation")

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_run_single_run",
        unexpected_side_effect,
    )

    with pytest.raises(
        ValueError,
        match="fp16-mixed precision is supported only on CUDA",
    ):
        experiment_runner.run_experiment_from_args(args)

    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    ("device_name", "expected_error"),
    [
        ("cuda", "CUDA execution requires an available CUDA device"),
        ("mps", "MPS execution requires an available MPS device"),
        ("cuda:3", "CUDA device index 3 is outside the available range"),
    ],
)
def test_runner_rejects_unavailable_execution_device_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device_name: str,
    expected_error: str,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.trainer.precision = "fp32"
    args = _args()
    args.config = Path("unused.yaml")
    args.device = device_name
    args.output_dir = tmp_path / "outputs"

    monkeypatch.setattr(experiment_runner, "load_config", lambda path: config)
    if device_name == "mps":
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    elif device_name == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail("invalid execution device reached data or run creation")

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_run_single_run",
        unexpected_side_effect,
    )

    with pytest.raises(ValueError, match=expected_error):
        experiment_runner.run_experiment_from_args(args)

    assert not args.output_dir.exists()


def test_strict_resume_accepts_matching_data_artifacts() -> None:
    current = _data_artifacts()

    experiment_runner._validate_resume_data_artifacts(
        current,
        current,
        strict_resume=True,
    )


def test_strict_resume_rejects_missing_checkpoint_data_artifacts() -> None:
    with pytest.raises(
        ValueError,
        match="strict resume data artifacts do not match",
    ):
        experiment_runner._validate_resume_data_artifacts(
            None,
            _data_artifacts(),
            strict_resume=True,
        )


def test_strict_resume_rejects_data_artifact_mismatch_before_run_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    checkpoint = _write_training_checkpoint(
        tmp_path / "resume.pt",
        config,
        metadata={
            "data_artifacts": _data_artifacts(
                artifact_digest="5" * 64
            ).to_dict(),
        },
    )
    args = _resume_args(checkpoint)
    args.output_dir = tmp_path / "new-runs"
    current_binding = _data_artifact_binding()

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        lambda config,
        *,
        seed,
        strict_resume,
        expected_artifacts,
        verification_observer,
        verification_workers: DataLoaders(
            train=_loader(),
            artifact_bindings=DataArtifactBindings((current_binding,)),
        ),
    )

    def unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail("artifact mismatch reached run creation or training assets")

    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        unexpected_side_effect,
    )

    with pytest.raises(
        ValueError,
        match="strict resume data artifacts do not match",
    ):
        experiment_runner.run_experiment_from_args(args)

    assert not args.output_dir.exists()


def test_strict_resume_preflights_sibling_best_before_any_run_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    loop_state = _training_loop_state(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        monitor="valid/loss",
    )
    selected = _best_payload(loop_state, epoch=2)
    selected["config"] = config.to_dict()
    selected_metadata = _checkpoint_metadata(selected)
    selected_metadata["checkpoint_kind"] = "latest"
    selected_metadata["data_artifacts"] = _data_artifacts().to_dict()
    checkpoint = tmp_path / "old-run" / "checkpoints" / "latest.pt"
    CheckpointManager.save_payload(selected, checkpoint)

    inherited_best = _best_payload(loop_state)
    inherited_best["config"] = config.to_dict()
    _checkpoint_metadata(inherited_best)["data_artifacts"] = _data_artifacts(
        artifact_digest="5" * 64
    ).to_dict()
    CheckpointManager.save_payload(
        inherited_best,
        checkpoint.parent / "best.pt",
    )
    args = _resume_args(checkpoint)
    args.output_dir = tmp_path / "new-runs"

    def unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail(
            "sibling-best identity mismatch reached data, run, or training creation"
        )

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "activate_extensions_for_cli",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        unexpected_side_effect,
    )

    with pytest.raises(ValueError, match="data artifacts do not match"):
        experiment_runner.run_experiment_from_args(args)

    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        (
            "invalid_wait",
            TypeError,
            "observations_without_improvement",
        ),
        (
            "terminal_early_stop",
            ValueError,
            "stopped early",
        ),
        (
            "invalid_checkpoint_kind",
            ValueError,
            "checkpoint_kind",
        ),
        (
            "best_without_state",
            ValueError,
            "must record best_epoch",
        ),
        (
            "target_already_complete",
            ValueError,
            "increase --epochs",
        ),
    ],
)
def test_strict_resume_preflights_complete_loop_state_before_plugin_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")
    metadata = _checkpoint_metadata(payload)
    loop_state = metadata["training_loop"]
    assert isinstance(loop_state, dict)
    if case == "invalid_wait":
        loop_state["observations_without_improvement"] = []
    elif case == "terminal_early_stop":
        metadata["training_loop"] = _training_loop_state(
            best_epoch=1,
            best_metric_value=0.5,
            observations_without_improvement=1,
            monitor_observations=2,
            stopped_early=True,
            monitor="valid/loss",
            early_stopping_patience=1,
        )
        payload["epoch"] = 2
        payload["metrics"] = {"valid/loss": 0.5}
        metadata["metric_sources"] = {
            "valid/loss": {
                "origin": "phase",
                "data_role": "validation",
                "protocol_id": None,
                "selection_eligible": True,
            }
        }
    elif case == "invalid_checkpoint_kind":
        metadata["checkpoint_kind"] = []
    elif case == "best_without_state":
        metadata["checkpoint_kind"] = "best"
    CheckpointManager.save_payload(payload, checkpoint)
    args = _resume_args(checkpoint)
    if case == "target_already_complete":
        args.epochs = 1

    def unexpected_activation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("invalid strict-resume state reached plugin activation")

    monkeypatch.setattr(
        experiment_runner,
        "activate_extensions_for_cli",
        unexpected_activation,
    )

    with pytest.raises(error_type, match=message):
        experiment_runner.run_experiment_from_args(args)


def test_resolve_resume_checkpoint_requires_an_explicit_existing_target(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        experiment_runner._resolve_resume_checkpoint(tmp_path / "missing.pt")


def test_resolve_resume_checkpoint_accepts_run_directory(tmp_path):
    checkpoint_path = tmp_path / "run" / "checkpoints" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    payload = _resume_candidate_payload(
        epoch=1,
        kind="latest",
        best_epoch=1,
        best_metric_value=0.5,
    )
    CheckpointManager.save_payload(payload, checkpoint_path)

    resolved_path, resolved_payload = (
        experiment_runner._resolve_resume_checkpoint(tmp_path / "run")
    )

    assert resolved_path == checkpoint_path
    assert resolved_payload is not None
    assert resolved_payload.get("epoch") == 1


def test_resume_root_accepts_one_nested_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "runs" / "run-a" / "checkpoints" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="latest",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        checkpoint_path,
    )

    resolved_path, resolved_payload = experiment_runner._resolve_resume_checkpoint(
        tmp_path / "runs"
    )

    assert resolved_path == checkpoint_path
    assert resolved_payload is not None
    assert resolved_payload.get("epoch") == 1


def test_resume_root_rejects_multiple_nested_checkpoint_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    for run_name in ("run-a", "run-b"):
        checkpoint_path = root / run_name / "checkpoints" / "latest.pt"
        checkpoint_path.parent.mkdir(parents=True)
        CheckpointManager.save_payload(
            _resume_candidate_payload(
                epoch=1,
                kind="latest",
                best_epoch=1,
                best_metric_value=0.5,
            ),
            checkpoint_path,
        )

    with pytest.raises(
        ValueError,
        match="multiple nested checkpoint directories",
    ):
        experiment_runner._resolve_resume_checkpoint(root)


def test_resume_accepts_case_variant_checkpoint_directory(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "CHECKPOINTS"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="latest",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        latest,
    )

    for resume_target in (checkpoint_dir, checkpoint_dir.parent):
        resolved_path, resolved_payload = (
            experiment_runner._resolve_resume_checkpoint(resume_target)
        )

        assert resolved_path == latest
        assert resolved_payload is not None
        assert resolved_payload.get("epoch") == 1


def test_resume_directory_selects_future_best_after_latest_publish_failure(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="latest",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        latest,
    )
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=2,
            kind="best",
            best_epoch=2,
            best_metric_value=0.25,
            monitor_observations=2,
        ),
        best,
    )

    resolved, payload = experiment_runner._resolve_resume_checkpoint(
        checkpoint_dir.parent
    )

    assert resolved == best
    assert payload is not None
    assert payload.get("epoch") == 2


def test_resume_explicit_file_does_not_upgrade_to_future_sibling(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="latest",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        latest,
    )
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=2,
            kind="best",
            best_epoch=2,
            best_metric_value=0.25,
            monitor_observations=2,
        ),
        checkpoint_dir / "best.pt",
    )

    resolved, payload = experiment_runner._resolve_resume_checkpoint(latest)

    assert resolved == latest
    assert payload is not None
    assert payload.get("epoch") == 1


def test_resume_directory_selects_periodic_after_latest_publish_failure(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    periodic = checkpoint_dir / "epoch_0002.pt"
    epoch_one = _resume_candidate_payload(
        epoch=1,
        kind="latest",
        best_epoch=1,
        best_metric_value=0.5,
    )
    CheckpointManager.save_payload(epoch_one, latest)
    epoch_one_best = deepcopy(epoch_one)
    _checkpoint_metadata(epoch_one_best)["checkpoint_kind"] = "best"
    CheckpointManager.save_payload(epoch_one_best, best)
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=2,
            kind="periodic",
            best_epoch=1,
            best_metric_value=0.5,
            observations_without_improvement=1,
            monitor_observations=2,
        ),
        periodic,
    )

    resolved, payload = experiment_runner._resolve_resume_checkpoint(
        checkpoint_dir
    )

    assert resolved == periodic
    assert payload is not None
    assert payload.get("epoch") == 2


def test_resume_directory_prefers_latest_at_equal_or_greater_progress(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=2,
            kind="latest",
            best_epoch=1,
            best_metric_value=0.5,
            observations_without_improvement=1,
            monitor_observations=2,
        ),
        latest,
    )
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="best",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        best,
    )

    resolved, payload = experiment_runner._resolve_resume_checkpoint(
        checkpoint_dir.parent
    )

    assert resolved == latest
    assert payload is not None
    assert payload.get("epoch") == 2


def test_resume_directory_rejects_dense_monitor_counter_regression(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    periodic = _resume_candidate_payload(
        epoch=2,
        kind="periodic",
        best_epoch=2,
        best_metric_value=0.25,
    )
    latest = _resume_candidate_payload(
        epoch=3,
        kind="latest",
        best_epoch=2,
        best_metric_value=0.25,
        observations_without_improvement=1,
        monitor_observations=2,
    )
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0002.pt")
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")

    with pytest.raises(
        ValueError,
        match="monitor_observations must equal the checkpoint epoch",
    ):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir)


def test_resume_directory_rejects_ordinary_best_epoch_regression(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    periodic = _resume_candidate_payload(
        epoch=2,
        kind="periodic",
        best_epoch=2,
        best_metric_value=0.25,
    )
    latest = _resume_candidate_payload(
        epoch=3,
        kind="latest",
        best_epoch=1,
        best_metric_value=0.5,
    )
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0002.pt")
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")

    with pytest.raises(ValueError, match="best_epoch regresses"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir)


def test_resume_directory_rejects_changed_metric_for_same_best_epoch(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    periodic = _resume_candidate_payload(
        epoch=2,
        kind="periodic",
        best_epoch=1,
        best_metric_value=0.5,
    )
    latest = _resume_candidate_payload(
        epoch=3,
        kind="latest",
        best_epoch=1,
        best_metric_value=0.4,
    )
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0002.pt")
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")

    with pytest.raises(ValueError, match="changes without a new best epoch"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir)


def test_resume_directory_rejects_new_best_below_min_delta(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    periodic = _resume_candidate_payload(
        epoch=2,
        kind="periodic",
        best_epoch=1,
        best_metric_value=0.5,
    )
    latest = _resume_candidate_payload(
        epoch=3,
        kind="latest",
        best_epoch=3,
        best_metric_value=0.45,
    )
    for payload in (periodic, latest):
        state = cast(dict[str, Any], _checkpoint_metadata(payload)["training_loop"])
        policy = cast(dict[str, Any], state["monitor_policy"])
        policy["min_delta"] = 0.1
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0002.pt")
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")

    with pytest.raises(ValueError, match="mode and min_delta"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir)


def test_resume_directory_accepts_exact_validation_history_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    periodic = _epoch_validation_candidate_payload(
        epoch=100,
        kind="periodic",
        last_evaluated_epoch=100,
    )
    latest = _epoch_validation_candidate_payload(
        epoch=110,
        kind="latest",
        last_evaluated_epoch=110,
    )
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0100.pt")
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")

    resolved, payload = experiment_runner._resolve_resume_checkpoint(
        checkpoint_dir
    )

    assert resolved == checkpoint_dir / "latest.pt"
    assert payload is not None
    assert payload.get("epoch") == 110


def test_resume_directory_rejects_changed_validation_history_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    periodic = _epoch_validation_candidate_payload(
        epoch=100,
        kind="periodic",
        last_evaluated_epoch=100,
    )
    latest = _epoch_validation_candidate_payload(
        epoch=110,
        kind="latest",
        last_evaluated_epoch=110,
    )
    latest_state = cast(
        dict[str, Any],
        _checkpoint_metadata(latest)["training_loop"],
    )
    epoch_validation = cast(
        dict[str, Any],
        latest_state["epoch_validation"],
    )
    results = cast(list[dict[str, Any]], epoch_validation["results"])
    first_metrics = cast(dict[str, float], results[0]["metrics"])
    first_metrics["valid/metrics/distribution/aggregate.kid_mean"] = 0.5
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0100.pt")
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")

    with pytest.raises(ValueError, match="exact result prefix"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir)


def test_resume_directory_accepts_best_as_only_published_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "best.pt"
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="best",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        best,
    )

    resolved, payload = experiment_runner._resolve_resume_checkpoint(
        checkpoint_dir.parent
    )

    assert resolved == best
    assert payload is not None
    assert payload.get("epoch") == 1


def test_resume_directory_rejects_inherited_best_as_only_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    payload = _resume_candidate_payload(
        epoch=1,
        kind="best",
        best_epoch=1,
        best_metric_value=0.5,
    )
    _checkpoint_metadata(payload)["inherited_from"] = str(
        tmp_path / "parent" / "checkpoints" / "best.pt"
    )
    CheckpointManager.save_payload(payload, checkpoint_dir / "best.pt")

    with pytest.raises(ValueError, match="only an inherited best checkpoint"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)


def test_resume_directory_rejects_same_epoch_state_conflict(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = _resume_candidate_payload(
        epoch=2,
        kind="latest",
        best_epoch=2,
        best_metric_value=0.25,
        monitor_observations=2,
    )
    periodic = _resume_candidate_payload(
        epoch=2,
        kind="periodic",
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
    )
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")
    CheckpointManager.save_payload(periodic, checkpoint_dir / "epoch_0002.pt")

    with pytest.raises(ValueError, match="disagree on metrics or training-loop"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)


def test_resume_directory_rejects_corrupt_candidate(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="latest",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        checkpoint_dir / "latest.pt",
    )
    (checkpoint_dir / "best.pt").write_bytes(b"not-a-checkpoint")

    with pytest.raises((RuntimeError, pickle.UnpicklingError)) as error:
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)
    assert "mmap can only be used" in str(error.value) or (
        "Weights only load failed" in str(error.value)
    )


def test_resume_directory_rejects_mixed_lineage_configs(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = _resume_candidate_payload(
        epoch=1,
        kind="latest",
        best_epoch=1,
        best_metric_value=0.5,
    )
    best = deepcopy(latest)
    best["config"] = {"identity": "another-run"}
    _checkpoint_metadata(best)["checkpoint_kind"] = "best"
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")
    CheckpointManager.save_payload(best, checkpoint_dir / "best.pt")

    with pytest.raises(ValueError, match="different configs"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)


def test_resume_directory_rejects_mixed_sibling_lineage(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    latest = _resume_candidate_payload(
        epoch=1,
        kind="latest",
        best_epoch=1,
        best_metric_value=0.5,
    )
    best = deepcopy(latest)
    _checkpoint_metadata(latest)["config_source"] = "checkpoint"
    _checkpoint_metadata(latest)["lineage"] = {"resumed_from": "parent-a.pt"}
    best_metadata = _checkpoint_metadata(best)
    best_metadata["checkpoint_kind"] = "best"
    best_metadata["config_source"] = "checkpoint"
    best_metadata["lineage"] = {"resumed_from": "parent-b.pt"}
    CheckpointManager.save_payload(latest, checkpoint_dir / "latest.pt")
    CheckpointManager.save_payload(best, checkpoint_dir / "best.pt")

    with pytest.raises(ValueError, match="different lineage"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)


def test_resume_directory_rejects_global_step_regression(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=2,
            kind="latest",
            best_epoch=2,
            best_metric_value=0.5,
            monitor_observations=2,
            global_step=20,
        ),
        checkpoint_dir / "latest.pt",
    )
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=3,
            kind="best",
            best_epoch=3,
            best_metric_value=0.25,
            monitor_observations=3,
            global_step=19,
        ),
        checkpoint_dir / "best.pt",
    )

    with pytest.raises(ValueError, match="regress global_step"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)


def test_resume_directory_rejects_noncanonical_numbered_checkpoint_name(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    CheckpointManager.save_payload(
        _resume_candidate_payload(
            epoch=1,
            kind="periodic",
            best_epoch=1,
            best_metric_value=0.5,
        ),
        checkpoint_dir / "epoch_00001.pt",
    )

    with pytest.raises(ValueError, match="noncanonical numbered checkpoint"):
        experiment_runner._resolve_resume_checkpoint(checkpoint_dir.parent)


def test_resolve_training_inputs_uses_checkpoint_config_for_strict_resume(
    tmp_path,
):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.name = "saved-name"
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    args = _args()
    args.config = None
    args.resume = checkpoint

    inputs = experiment_runner._resolve_training_inputs(args)

    assert inputs.config_source == "checkpoint"
    assert inputs.config.experiment.name == "saved-name"
    assert inputs.config.experiment.output_dir == str(
        tmp_path / "runs" / "original"
    )
    assert inputs.checkpoint_path == checkpoint
    assert inputs.checkpoint is not None
    assert inputs.config_overlays == []


def test_training_preflight_activates_builtins_before_external_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_builtins = experiment_runner.activate_training_builtins
    original_extensions = experiment_runner.activate_extensions_for_cli

    def record_builtins() -> None:
        events.append("builtins")
        original_builtins()

    def record_extensions(*args: Any, **kwargs: Any) -> Any:
        events.append("extensions")
        return original_extensions(*args, **kwargs)

    monkeypatch.setattr(
        experiment_runner,
        "activate_training_builtins",
        record_builtins,
    )
    monkeypatch.setattr(
        experiment_runner,
        "activate_extensions_for_cli",
        record_extensions,
    )

    args = _args()
    args.config = MNIST_TRAIN_CONFIG
    experiment_runner._resolve_training_inputs(args)

    assert events[:2] == ["builtins", "extensions"]


def test_strict_resume_rejects_old_source_schema_before_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.data.params["source"] = {
        "kind": "torchvision",
        "dataset": "MNIST",
        "download": True,
    }
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    args = _resume_args(checkpoint)
    args.output_dir = tmp_path / "new-runs"

    def unexpected_side_effect(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("old source schema reached extension activation or run creation")

    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        unexpected_side_effect,
    )

    with pytest.raises(ConfigError, match=r"source\.kind"):
        experiment_runner.run_experiment_from_args(args)

    assert not args.output_dir.exists()


def test_resolve_training_inputs_rejects_config_and_resume_together(tmp_path):
    args = _args()
    args.config = Path("config.yaml")
    args.resume = tmp_path / "resume.pt"

    with pytest.raises(ValueError, match="exactly one"):
        experiment_runner._resolve_training_inputs(args)


def test_train_cli_requires_exactly_one_config_source() -> None:
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--observability-config",
                "observability.yaml",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train", "--config", "config.yaml", "--resume", "checkpoint.pt"]
        )

    assert parser.parse_args(["train", "--config", "config.yaml"]).config == Path(
        "config.yaml"
    )
    assert parser.parse_args(
        ["train", "--resume", "checkpoint.pt"]
    ).resume == Path("checkpoint.pt")


def test_train_cli_accepts_observability_config_with_resume() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "train",
            "--resume",
            "checkpoint.pt",
            "--observability-config",
            "observability.yaml",
        ]
    )

    assert args.resume == Path("checkpoint.pt")
    assert args.observability_config == Path("observability.yaml")


def test_train_cli_accepts_artifact_verification_workers() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "train",
            "--resume",
            "checkpoint.pt",
            "--artifact-verification-workers",
            "6",
        ]
    )

    assert args.artifact_verification_workers == 6


def test_run_options_reject_non_positive_artifact_verification_workers() -> None:
    args = _args()
    args.artifact_verification_workers = 0

    with pytest.raises(
        ValueError,
        match="--artifact-verification-workers must be positive",
    ):
        experiment_runner.ExperimentRunOptions.from_namespace(
            args,
            configured_num_epochs=2,
            configured_show_progress=False,
        )


def test_run_options_reject_excessive_artifact_verification_workers() -> None:
    args = _args()
    args.artifact_verification_workers = 9

    with pytest.raises(
        ValueError,
        match="--artifact-verification-workers must not exceed 8",
    ):
        experiment_runner.ExperimentRunOptions.from_namespace(
            args,
            configured_num_epochs=2,
            configured_show_progress=False,
        )


def test_train_cli_progress_overrides_are_mutually_exclusive() -> None:
    parser = build_argument_parser()

    inherited = parser.parse_args(["train", "--resume", "checkpoint.pt"])
    enabled = parser.parse_args(
        ["train", "--resume", "checkpoint.pt", "--progress"]
    )
    disabled = parser.parse_args(
        ["train", "--resume", "checkpoint.pt", "--no-progress"]
    )

    assert inherited.progress is False
    assert inherited.no_progress is False
    assert enabled.progress is True
    assert enabled.no_progress is False
    assert disabled.progress is False
    assert disabled.no_progress is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--resume",
                "checkpoint.pt",
                "--progress",
                "--no-progress",
            ]
        )


def test_progress_flag_reenables_checkpoint_disabled_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.trainer.show_progress = False
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    args = _resume_args(checkpoint)
    args.progress = True
    args.no_progress = False
    observed: dict[str, Any] = {}

    class FakeArtifactReporter:
        def __init__(self) -> None:
            observed["artifact_reporter"] = self
            self.closed = False

        def observe(self, event: object) -> None:
            observed["artifact_event"] = event

        def close(self) -> None:
            self.closed = True

    def build_with_observer(
        config,
        *,
        seed,
        strict_resume,
        expected_artifacts,
        verification_observer,
        verification_workers,
    ):
        del config, seed, strict_resume, expected_artifacts
        observed["verification_observer"] = verification_observer
        observed["verification_workers"] = verification_workers
        return _loaders()

    monkeypatch.setattr(
        experiment_runner,
        "RichArtifactVerificationReporter",
        FakeArtifactReporter,
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        build_with_observer,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        lambda output_dir: ("new-run", tmp_path / "runs" / "new-run"),
    )

    def record_run(config, loaders, options, **kwargs):
        del loaders
        observed["config"] = config
        observed["options"] = options
        observed.update(kwargs)

    monkeypatch.setattr(experiment_runner, "_run_single_run", record_run)

    experiment_runner.run_experiment_from_args(args)

    assert observed["config"].trainer.show_progress is False
    assert observed["options"].show_progress is True
    reporter = observed["artifact_reporter"]
    assert observed["verification_observer"].__self__ is reporter
    assert observed["verification_workers"] is None
    assert reporter.closed
    assert observed["runtime_options"]["progress"] is True
    assert observed["runtime_options"]["no_progress"] is False


def test_repository_mnist_observability_profile_is_valid() -> None:
    checkpoint_config = load_config(MNIST_TRAIN_CONFIG)
    overlay_path = MNIST_OBSERVABILITY_CONFIG

    overlay, audit = experiment_runner._load_observability_overlay(overlay_path)
    effective = experiment_runner._apply_observability_overlay(
        checkpoint_config,
        overlay,
    )

    assert [backend.name for backend in effective.logging.backends] == [
        "local",
        "tensorboard",
    ]
    assert effective.logging.log_every == checkpoint_config.logging.log_every
    assert effective.diagnostics[0].params["sampling"] == {
        "shape": [1, 32, 32],
        "sample_num": 32,
        "batch_size": 32,
        "seed": 123,
    }
    assert effective.diagnostics[0].params["samplers"][0]["params"] == {
        "num_inference_steps": 50,
        "eta": 0.0,
    }
    assert audit["sections"] == ["diagnostics", "logging"]
    assert audit["logging_fields"] == ["backends"]


def test_observability_config_requires_strict_resume(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    overlay_path = _write_observability_config(
        tmp_path / "observability.yaml",
        {"logging": {"log_every": 5}},
    )
    args = _args()
    args.config = config_path
    args.observability_config = overlay_path

    with pytest.raises(ValueError, match="observability"):
        experiment_runner._resolve_training_inputs(args)


def test_observability_config_requires_an_existing_file(tmp_path: Path) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)

    with pytest.raises(FileNotFoundError, match="observability config"):
        experiment_runner._resolve_training_inputs(
            _resume_args(
                checkpoint,
                observability_config=tmp_path / "missing.yaml",
            )
        )


def test_observability_overlay_replaces_diagnostics_and_shallow_merges_logging(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = load_config(MNIST_TRAIN_CONFIG)
    source.diagnostics[0].params["modules"] = ["trusted.providers"]
    source.logging.log_every = 41
    source.logging.torch_logs = {
        "graph_breaks": True,
        "recompiles": True,
    }
    source_before = deepcopy(source.to_dict())
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", source)
    raw_overlay = {
        "logging": {"log_every": 7},
        "diagnostics": [
            {
                "name": "diffusion_quality",
                "params": {
                    "modules": ["trusted.providers"],
                    "cadence": {"step_every": 3},
                },
            }
        ],
    }
    overlay_path = _write_observability_config(
        tmp_path / "observability.yaml",
        raw_overlay,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_load_training_checkpoint_config",
        lambda payload: source,
    )

    inputs = experiment_runner._resolve_training_inputs(
        _resume_args(checkpoint, observability_config=overlay_path)
    )

    resolved = inputs.config.to_dict()
    assert resolved["diagnostics"] == raw_overlay["diagnostics"]
    assert resolved["logging"]["log_every"] == 7
    assert resolved["logging"]["backends"] == source_before["logging"]["backends"]
    assert (
        resolved["logging"]["torch_logs"]
        == source_before["logging"]["torch_logs"]
    )
    assert source.to_dict() == source_before
    assert inputs.config is not source
    assert inputs.config_source == "checkpoint"
    assert inputs.config_overlays == [
        {
            "kind": "observability",
            "source_path": str(overlay_path.resolve()),
            "source_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
            "sections": ["diagnostics", "logging"],
            "logging_fields": ["log_every"],
        }
    ]


@pytest.mark.parametrize(
    ("torch_logs", "expected_torch_logs"),
    [
        pytest.param({}, {}, id="clear"),
        pytest.param(
            {"recompiles": False},
            {"recompiles": False},
            id="replace",
        ),
    ],
)
def test_observability_overlay_uses_atomic_collection_replacement(
    tmp_path: Path,
    torch_logs: dict[str, bool],
    expected_torch_logs: dict[str, bool],
) -> None:
    source = load_config(MNIST_TRAIN_CONFIG)
    source.logging.log_every = 37
    source.logging.torch_logs = {
        "graph_breaks": True,
        "recompiles": True,
    }
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", source)
    raw_backends = [
        {
            "name": "tensorboard",
            "params": {"subdir": "replacement"},
        }
    ]
    overlay_path = _write_observability_config(
        tmp_path / "observability.yaml",
        {
            "diagnostics": [],
            "logging": {
                "backends": raw_backends,
                "torch_logs": torch_logs,
            },
        },
    )

    inputs = experiment_runner._resolve_training_inputs(
        _resume_args(checkpoint, observability_config=overlay_path)
    )

    resolved = inputs.config.to_dict()
    assert resolved["diagnostics"] == []
    assert resolved["logging"]["log_every"] == 37
    assert resolved["logging"]["backends"] == raw_backends
    assert resolved["logging"]["torch_logs"] == expected_torch_logs


@pytest.mark.parametrize(
    "raw_overlay",
    [
        pytest.param(None, id="null-root"),
        pytest.param([], id="non-mapping-root"),
        pytest.param({}, id="empty-root"),
        pytest.param({"experiment": {"name": "other"}}, id="experiment"),
        pytest.param(
            {"data": {"name": "other", "params": {}}},
            id="data",
        ),
        pytest.param(
            {"model": {"name": "other", "params": {}}},
            id="model",
        ),
        pytest.param(
            {"process": {"name": "other", "params": {}}},
            id="process",
        ),
        pytest.param(
            {"training": {"name": "other", "params": {}}},
            id="training",
        ),
        pytest.param(
            {"objective": {"name": "other", "params": {}}},
            id="objective",
        ),
        pytest.param({"trainer": {"num_epochs": 1}}, id="trainer"),
        pytest.param(
            {"optimizer": {"name": "torch.optim.SGD", "params": {}}},
            id="optimizer",
        ),
        pytest.param(
            {
                "lr_scheduler": {
                    "name": "torch.optim.lr_scheduler.StepLR",
                    "interval": "epoch",
                    "params": {"step_size": 1},
                }
            },
            id="lr-scheduler",
        ),
        pytest.param({"ema": {"enabled": False}}, id="ema"),
        pytest.param({"sampling": {"num_samples": 1}}, id="sampling"),
        pytest.param({"artifacts": {"checkpoint_every": 1}}, id="artifacts"),
        pytest.param({"extensions": {"plugins": []}}, id="extensions"),
        pytest.param({"unknown": {}}, id="unknown-top-level"),
        pytest.param({"diagnostics": None}, id="null-diagnostics"),
        pytest.param({"diagnostics": {}}, id="non-list-diagnostics"),
        pytest.param({"diagnostics": [None]}, id="malformed-diagnostic"),
        pytest.param({"logging": None}, id="null-logging"),
        pytest.param({"logging": []}, id="non-mapping-logging"),
        pytest.param({"logging": {}}, id="empty-logging"),
        pytest.param(
            {"logging": {"unknown": True}},
            id="unknown-logging-field",
        ),
        pytest.param({"logging": {"log_every": 0}}, id="invalid-log-every"),
        pytest.param({"logging": {"backends": []}}, id="empty-backends"),
        pytest.param({"logging": {"backends": None}}, id="null-backends"),
        pytest.param({"logging": {"torch_logs": None}}, id="null-torch-logs"),
        pytest.param(
            {
                "diagnostics": [
                    {
                        "name": "diffusion_quality",
                        "params": {"modules": "not-a-list"},
                    }
                ]
            },
            id="modules-not-list",
        ),
        pytest.param(
            {
                "diagnostics": [
                    {
                        "name": "diffusion_quality",
                        "params": {"modules": [""]},
                    }
                ]
            },
            id="empty-module-name",
        ),
        pytest.param(
            {
                "diagnostics": [
                    {
                        "name": "diffusion_quality",
                        "params": {"modules": [1]},
                    }
                ]
            },
            id="non-string-module",
        ),
        pytest.param(
            {
                "diagnostics": [
                    {
                        "name": "diffusion_quality",
                        "params": {"modules": ["new.providers"]},
                    }
                ]
            },
            id="new-module",
        ),
    ],
)
def test_invalid_observability_overlay_fails_before_run_creation(
    monkeypatch,
    tmp_path: Path,
    raw_overlay: Any,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    overlay_path = _write_observability_config(
        tmp_path / "observability.yaml",
        raw_overlay,
    )
    args = _resume_args(checkpoint, observability_config=overlay_path)
    args.output_dir = tmp_path / "new-runs"

    def unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail("invalid overlay reached run creation")

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )

    with pytest.raises(ConfigError):
        experiment_runner.run_experiment_from_args(args)

    assert not args.output_dir.exists()


def test_malformed_observability_yaml_fails_before_run_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    overlay_path = tmp_path / "observability.yaml"
    overlay_path.write_text("logging: [unterminated", encoding="utf-8")
    args = _resume_args(checkpoint, observability_config=overlay_path)

    def unexpected_side_effect(*args, **kwargs):
        del args, kwargs
        pytest.fail("malformed overlay reached run creation")

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        unexpected_side_effect,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        unexpected_side_effect,
    )

    with pytest.raises(ConfigError):
        experiment_runner.run_experiment_from_args(args)


def test_observability_overlay_audit_accumulates_and_plain_resume_inherits(
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    prior_audit = {
        "kind": "observability",
        "source_path": str(tmp_path / "prior.yaml"),
        "source_sha256": "0" * 64,
        "sections": ["logging"],
        "logging_fields": ["log_every"],
    }
    first_checkpoint = _write_training_checkpoint(
        tmp_path / "first.pt",
        config,
        metadata={"config_overlays": [prior_audit]},
    )
    raw_overlay = {
        "logging": {
            "backends": [
                {
                    "name": "tensorboard",
                    "params": {"subdir": "resumed"},
                }
            ]
        }
    }
    overlay_path = _write_observability_config(
        tmp_path / "observability.yaml",
        raw_overlay,
    )

    overlaid = experiment_runner._resolve_training_inputs(
        _resume_args(first_checkpoint, observability_config=overlay_path)
    )

    expected_new_audit = {
        "kind": "observability",
        "source_path": str(overlay_path.resolve()),
        "source_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "sections": ["logging"],
        "logging_fields": ["backends"],
    }
    assert overlaid.config_overlays == [prior_audit, expected_new_audit]
    assert yaml.safe_load(yaml.safe_dump(overlaid.config_overlays)) == [
        prior_audit,
        expected_new_audit,
    ]

    second_checkpoint = _write_training_checkpoint(
        tmp_path / "second.pt",
        overlaid.config,
        metadata={"config_overlays": overlaid.config_overlays},
    )
    inherited = experiment_runner._resolve_training_inputs(
        _resume_args(second_checkpoint)
    )

    assert inherited.config_overlays == overlaid.config_overlays
    assert inherited.config_overlays is not overlaid.config_overlays
    assert inherited.config.to_dict()["logging"]["backends"] == raw_overlay[
        "logging"
    ]["backends"]


@pytest.mark.parametrize(
    "invalid_history",
    [
        pytest.param({}, id="not-list"),
        pytest.param([None], id="entry-not-mapping"),
        pytest.param(
            [
                {
                    "kind": "other",
                    "source_path": "old.yaml",
                    "source_sha256": "0" * 64,
                    "sections": ["logging"],
                    "logging_fields": ["log_every"],
                }
            ],
            id="wrong-kind",
        ),
        pytest.param(
            [
                {
                    "kind": "observability",
                    "source_path": "old.yaml",
                    "source_sha256": "not-a-sha256",
                    "sections": ["logging"],
                    "logging_fields": ["log_every"],
                }
            ],
            id="invalid-sha256",
        ),
        pytest.param(
            [
                {
                    "kind": "observability",
                    "source_path": "old.yaml",
                    "source_sha256": "0" * 64,
                    "sections": ["optimizer"],
                    "logging_fields": [],
                }
            ],
            id="invalid-sections",
        ),
    ],
)
def test_strict_resume_rejects_invalid_overlay_audit_history(
    tmp_path: Path,
    invalid_history: Any,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    checkpoint = _write_training_checkpoint(
        tmp_path / "resume.pt",
        config,
        metadata={"config_overlays": invalid_history},
    )

    with pytest.raises((TypeError, ValueError), match="config_overlays"):
        experiment_runner._resolve_training_inputs(_resume_args(checkpoint))


def test_run_experiment_passes_overlay_audit_and_absolute_runtime_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    checkpoint = _write_training_checkpoint(tmp_path / "resume.pt", config)
    monkeypatch.chdir(tmp_path)
    relative_overlay = Path("observability.yaml")
    _write_observability_config(
        relative_overlay,
        {"logging": {"log_every": 13}},
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        lambda config,
        *,
        seed,
        strict_resume,
        expected_artifacts,
        verification_observer,
        verification_workers: _loaders(),
    )
    monkeypatch.setattr(
        experiment_runner,
        "_make_timestamped_output_dir",
        lambda output_dir: ("new-run", tmp_path / "runs" / "new-run"),
    )

    def record_run(*args, **kwargs):
        del args
        observed.update(kwargs)

    monkeypatch.setattr(experiment_runner, "_run_single_run", record_run)

    experiment_runner.run_experiment_from_args(
        _resume_args(
            checkpoint,
            observability_config=relative_overlay,
        )
    )

    absolute_overlay = str(relative_overlay.resolve())
    assert observed["runtime_options"]["observability_config"] == absolute_overlay
    assert observed["config_overlays"] == [
        {
            "kind": "observability",
            "source_path": absolute_overlay,
            "source_sha256": hashlib.sha256(
                relative_overlay.read_bytes()
            ).hexdigest(),
            "sections": ["logging"],
            "logging_fields": ["log_every"],
        }
    ]


def test_run_manifest_and_checkpoint_metadata_record_overlay_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "overlay-audit"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    captured: dict[str, Any] = {}
    data_artifacts = _data_artifacts()
    config_overlays = [
        {
            "kind": "observability",
            "source_path": str(tmp_path / "observability.yaml"),
            "source_sha256": "a" * 64,
            "sections": ["diagnostics", "logging"],
            "logging_fields": ["log_every"],
        }
    ]

    def build_training_components(config, **kwargs):
        del config
        captured.update(kwargs)
        return _training_components(trainer, logger)

    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        build_training_components,
    )

    experiment_runner._run_single_run(
        config,
        _loaders(),
        _options(config),
        extensions=ResolvedExtensions(config, (), ()),
        config_source="checkpoint",
        config_overlays=config_overlays,
        data_artifacts=data_artifacts,
        checkpoint_payload=None,
        startup_cwd=Path.cwd(),
        runtime_options={
            "observability_config": str(tmp_path / "observability.yaml")
        },
    )

    manifest = yaml.safe_load((tmp_path / "run_manifest.yaml").read_text())
    assert manifest["config_overlays"] == config_overlays
    assert manifest["data_artifacts"] == data_artifacts.to_dict()
    assert captured["checkpoint_metadata"]["config_overlays"] == config_overlays
    assert (
        captured["checkpoint_metadata"]["data_artifacts"]
        == data_artifacts.to_dict()
    )
    assert manifest["runtime_options"]["observability_config"] == str(
        tmp_path / "observability.yaml"
    )
