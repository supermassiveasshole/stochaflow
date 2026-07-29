"""Tests for shared experiment runner orchestration."""

import hashlib
import random
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from stochaflow.data import (
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataLoaders,
)
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
from stochaflow.utils.config import ConfigError, load_config, load_config_dict
from stochaflow.utils.logging import ExperimentLogger
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
        return [{"loss": 0.5, "train_loss": 0.5, "num_batches": 1.0}]

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
        skip_final_sample=True,
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
    logger: RecordingLogger,
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


def _best_payload(
    loop_state: dict[str, Any],
    *,
    epoch: int | None = None,
) -> CheckpointState:
    best_epoch = loop_state["best_epoch"] if epoch is None else epoch
    monitor = loop_state["monitor"]
    best_snapshot_loop = {
        **loop_state,
        "epochs_without_improvement": 0,
        "stopped_early": False,
    }
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
            "training_loop": best_snapshot_loop,
        },
    }


def _strict_resume_fields(*, epoch: int, global_step: int) -> CheckpointState:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": capture_rng_state(),
    }


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
        "training_loop": {
            "best_epoch": None,
            "best_metric_value": None,
            "epochs_without_improvement": 0,
            "stopped_early": False,
            "monitor": None,
            "mode": None,
        },
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
            "metadata": checkpoint_metadata,
        },
        path,
    )
    return path


def _write_observability_config(path: Path, value: Any) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _load_mnist_config_with_final_sampling():
    raw = yaml.safe_load(MNIST_TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    raw["sampling"] = {
        "run_after_training": True,
        "sampler": {"name": "ddpm", "params": {}},
        "options": {
            "weights": "auto",
            "clip_denoised": True,
            "trajectory": {"enabled": False, "every_steps": 1},
        },
        "shape": [1, 32, 32],
        "num_samples": 64,
        "batch_size": 64,
        "seed": None,
        "writers": [
            {"name": "tensor", "params": {}},
            {
                "name": "image",
                "params": {
                    "grid_nrow": 8,
                    "gif_fps": 8,
                    "denormalize": True,
                },
            },
        ],
    }
    return load_config_dict(raw)


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


def test_runner_uses_valid_loss_when_validation_is_available(monkeypatch, tmp_path):
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

    _run_single(
        config,
        _loaders(validation=True),
        _options(config),
    )

    assert trainer.fit_kwargs["validation_dataloader"] is not None
    assert trainer.fit_kwargs["early_stopping_monitor"] == "valid_loss"
    assert trainer.fit_kwargs["num_epochs"] == config.trainer.num_epochs
    assert build_kwargs["checkpoint_metadata"]["extension_plugins"] == []
    assert logger.closed


def test_runner_uses_train_loss_and_skips_test_without_validation(monkeypatch, tmp_path):
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
    assert trainer.fit_kwargs["early_stopping_monitor"] == "train_loss"
    assert trainer.evaluate_calls == 0
    assert logger.closed


def test_runner_allows_cli_epochs_override(monkeypatch, tmp_path):
    config = _load_mnist_config_with_final_sampling()
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
    assert config.sampling.run_after_training
    assert selected["data_builder"] == config.data.name
    assert selected["model"] == config.model.name
    assert selected["training_builder"] == config.training.name
    assert selected["process"] == config.process.name
    assert "sampling_builder" not in selected
    assert selected["sampling_recipe"] == "standard_denoising"
    assert config.sampling.sampler is not None
    assert selected["sampling_sampler"] == config.sampling.sampler.name
    assert selected["sampling_artifact_writers"] == [
        writer.name for writer in config.sampling.writers
    ]


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
        _loaders(),
        _options(config, args),
    )

    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    manifest = yaml.safe_load((tmp_path / "run_manifest.yaml").read_text())
    assert trainer.fit_kwargs["show_progress"] is True
    assert resolved["trainer"]["show_progress"] is True
    assert manifest["config"]["trainer"]["show_progress"] is True


def test_runner_samples_selected_best_checkpoint(monkeypatch, tmp_path):
    config = _load_mnist_config_with_final_sampling()
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    trainer.best_checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    logger = RecordingLogger()
    training = _training_components(trainer, logger)
    observed = {}
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_sampling",
        lambda **kwargs: observed.update(kwargs)
        or SimpleNamespace(artifacts={"samples": tmp_path / "samples.png"}),
    )
    args = _args()
    args.skip_final_sample = False

    _run_single(
        config,
        _loaders(),
        _options(config, args),
    )

    assert observed["checkpoint"] == trainer.best_checkpoint_path
    assert observed["output_dir"] == tmp_path / "samples" / "final"
    assert logger.closed


def test_runner_skips_disabled_final_sample(monkeypatch, tmp_path):
    config = load_config(MNIST_TRAIN_CONFIG)
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    config.sampling.run_after_training = False
    trainer = RecordingTrainer()
    trainer.best_checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    logger = RecordingLogger()
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    def unexpected_sampling(**kwargs):
        del kwargs
        raise AssertionError("final sampling must be skipped without a builder")

    monkeypatch.setattr(experiment_runner, "run_sampling", unexpected_sampling)
    args = _args()
    args.skip_final_sample = False

    _run_single(
        config,
        _loaders(),
        _options(config, args),
    )

    assert logger.closed


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
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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
    selected_loop = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 1,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
    future_loop = {
        **selected_loop,
        "best_epoch": 3,
        "best_metric_value": 0.25,
        "epochs_without_improvement": 0,
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
        ("monitor", "train_loss"),
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
    selected_loop = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 1,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
    candidate_loop = {**selected_loop, field: value}
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
            "epochs_without_improvement",
            1,
            "epochs_without_improvement=0",
        ),
        (
            "stopped_early",
            True,
            "stopped_early=false",
        ),
    ],
)
def test_strict_resume_rejects_noncanonical_best_snapshot_state(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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
            monitor="valid_loss",
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
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 1,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 1,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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


def test_strict_resume_rejects_terminal_early_stopping_state(tmp_path):
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 2,
        "stopped_early": True,
        "monitor": "valid_loss",
        "mode": "min",
    }
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


def test_strict_mps_resume_warns_when_legacy_v8_has_no_mps_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = RecordingTrainer()
    trainer.device = torch.device("mps")
    trainer.best_epoch = None
    trainer.best_metric_value = None
    logger = RecordingLogger()
    loop_state = {
        "best_epoch": None,
        "best_metric_value": None,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "monitor": None,
        "mode": None,
    }
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
        trainer._emit_epoch_diagnostics(epoch_index=1, metrics={"loss": 0.0})

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
        show_progress=False,
        early_stopping_monitor="train_loss",
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
        show_progress=False,
        early_stopping_monitor="train_loss",
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
        show_progress=False,
        early_stopping_monitor="train_loss",
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
        lambda *args, **kwargs: None,
    )

    experiment_runner.run_experiment_from_args(args)

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
    loop_state = {
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "epochs_without_improvement": 1,
        "stopped_early": False,
        "monitor": "valid_loss",
        "mode": "min",
    }
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
            "epochs_without_improvement",
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
        loop_state["epochs_without_improvement"] = []
    elif case == "terminal_early_stop":
        loop_state["stopped_early"] = True
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
    checkpoint_path.write_bytes(b"latest")

    resolved = experiment_runner._resolve_resume_checkpoint(tmp_path / "run")

    assert resolved == checkpoint_path


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
