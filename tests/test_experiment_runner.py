"""Tests for shared experiment runner orchestration."""

from argparse import Namespace
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import yaml

from stochaflow.data import DataLoaders
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.scripts import experiment_runner
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
from stochaflow.utils.config import load_config
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.plugins import ResolvedExtensions


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
        self.checkpoint_config = None
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


def _loader() -> DataLoader:
    dataset = TensorDataset(torch.zeros(2, 1))
    return DataLoader(dataset, batch_size=1)


def _loaders(*, validation: bool = False, test: bool = False) -> DataLoaders:
    return DataLoaders(
        train=_loader(),
        validation=_loader() if validation else None,
        test=_loader() if test else None,
    )


def _args() -> Namespace:
    return Namespace(
        epochs=None,
        limit_batches=None,
        limit_validation_batches=None,
        limit_test_batches=None,
        deterministic=False,
        no_progress=True,
        resume=None,
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
    )


def _best_payload(
    loop_state: dict[str, Any],
    *,
    epoch: int | None = None,
) -> CheckpointState:
    best_epoch = loop_state["best_epoch"] if epoch is None else epoch
    monitor = loop_state["monitor"]
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
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
            "training_loop": dict(loop_state),
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
    config = load_config(Path("configs/ddpm_mnist.yaml"))
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
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
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
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
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
    assert config.sampling.builder is not None
    assert selected["data_builder"] == config.data.name
    assert selected["model"] == config.model.name
    assert selected["training_builder"] == config.training.name
    assert selected["process"] == config.process.name
    assert selected["sampling_builder"] == config.sampling.builder.name
    assert selected["sampling_artifact_writers"] == [
        writer.name for writer in config.sampling.writers
    ]


def test_runner_samples_selected_best_checkpoint(monkeypatch, tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
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


def test_runner_skips_default_final_sample_without_builder(monkeypatch, tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    config.sampling.builder = None
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
    config = load_config(Path("configs/ddpm_mnist.yaml"))
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
    config = load_config(Path("configs/ddpm_mnist.yaml"))
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

    with pytest.raises(FileNotFoundError, match="sibling.*best.pt"):
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


@pytest.mark.parametrize("mismatch", ["config", "plugins"])
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
    else:
        _checkpoint_metadata(candidate)["extension_plugins"] = [
            {
                "name": "other",
                "distribution": "other",
                "version": "1.0",
                "target": "other.stochaflow_ext",
            }
        ]
        expected = "extension provenance does not match"
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
    inherited_metadata = _checkpoint_metadata(inherited_payload)
    assert inherited_metadata["extension_plugins"][0]["version"] == "2.0"
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


class _StochasticStrategy(TrainingStrategy):
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


class _RNGConsumingDiagnostic(TrainingDiagnostic):
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


class _BestRNGConsumingLogger(ExperimentLogger):
    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del step
        if "best/epoch" in metrics:
            _RNGConsumingDiagnostic._consume_rng()

    def close(self) -> None:
        return None


class _EpochEndRNGConsumingReporter:
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
        _RNGConsumingDiagnostic._consume_rng()


def _stochastic_trainer(
    checkpoint_dir: Path,
    *,
    logger: ExperimentLogger | None = None,
) -> Trainer:
    model = nn.Linear(1, 1)
    plan = TrainingPlan(
        strategy=_StochasticStrategy(model),
        primary_model=model,
    )
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    manager = CheckpointManager(model=model, optimizer=optimizer)
    return Trainer(
        plan,
        optimizer,
        device="cpu",
        diagnostics=[_RNGConsumingDiagnostic()],
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
        reporter=_EpochEndRNGConsumingReporter(),
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
        reporter=_EpochEndRNGConsumingReporter(),
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
        reporter=_EpochEndRNGConsumingReporter(),
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
        logger=_BestRNGConsumingLogger(),
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
        logger=_BestRNGConsumingLogger(),
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
        logger=_BestRNGConsumingLogger(),
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
    "config_path",
    [Path("configs/ddpm_mnist.yaml"), Path("configs/ddim_cifar10.yaml")],
)
def test_runner_builds_registered_data_builder(monkeypatch, tmp_path, config_path):
    config = load_config(config_path)
    config.experiment.output_dir = str(tmp_path / "outputs")
    args = _args()
    args.config = Path("unused.yaml")
    observed = {}

    def stub_builder(data_config, *, seed):
        observed["builder_config"] = data_config
        observed["seed"] = seed
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
    }


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
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.experiment.name = "saved-name"
    config.experiment.output_dir = str(tmp_path / "runs" / "original")
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": config.to_dict(),
            "model_state_dict": {},
            "rng_state": capture_rng_state(),
            "metadata": {"extension_plugins": []},
        },
        checkpoint,
    )
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
            ["train", "--config", "config.yaml", "--resume", "checkpoint.pt"]
        )

    assert parser.parse_args(["train", "--config", "config.yaml"]).config == Path(
        "config.yaml"
    )
    assert parser.parse_args(
        ["train", "--resume", "checkpoint.pt"]
    ).resume == Path("checkpoint.pt")
