"""Tests for shared DDPM runner orchestration."""

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from stochaflow.data.pipeline import DataBundle, SplitData
from stochaflow.scripts import ddpm_runner
from stochaflow.utils.config import load_config


class RecordingTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.global_step = 0
        self.best_checkpoint_path = None
        self.best_epoch = 1
        self.best_metric_value = 0.5
        self.stopped_early = False
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


class RecordingLogger:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _split(name: str) -> SplitData:
    dataset = TensorDataset(torch.zeros(2, 1))
    return SplitData(
        name=name,
        dataset=dataset,
        dataloader=DataLoader(dataset, batch_size=1),
    )


def _args() -> Namespace:
    return Namespace(
        epochs=None,
        limit_batches=None,
        limit_validation_batches=None,
        limit_test_batches=None,
        no_progress=True,
        resume=None,
        skip_sampling=True,
        num_samples=1,
        sample_grid_size=1,
        trajectory_interval=1,
    )


def _training_components(trainer: RecordingTrainer, logger: RecordingLogger):
    return SimpleNamespace(
        trainer=trainer,
        logger=logger,
        checkpoint_manager=SimpleNamespace(load=lambda *args, **kwargs: None),
        diffusion=SimpleNamespace(),
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
        ddpm_runner,
        "build_training_components",
        build_training_components,
    )

    ddpm_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train"), valid=_split("valid")),
        _args(),
        sample_shape_fn=lambda config, num_samples: torch.Size((num_samples, 1, 2, 2)),
        script_name="test.py",
    )

    assert trainer.fit_kwargs["validation_dataloader"] is not None
    assert trainer.fit_kwargs["early_stopping_monitor"] == "valid_loss"
    assert trainer.fit_kwargs["num_epochs"] == config.trainer.num_epochs
    assert build_kwargs["num_epochs"] == config.trainer.num_epochs
    assert logger.closed


def test_runner_uses_train_loss_and_skips_test_without_validation(monkeypatch, tmp_path):
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    monkeypatch.setattr(
        ddpm_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    ddpm_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train")),
        _args(),
        sample_shape_fn=lambda config, num_samples: torch.Size((num_samples, 3, 2, 2)),
        script_name="test.py",
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
        ddpm_runner,
        "build_training_components",
        build_training_components,
    )
    args = _args()
    args.epochs = 3

    ddpm_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train")),
        args,
        sample_shape_fn=lambda config, num_samples: torch.Size((num_samples, 3, 2, 2)),
        script_name="test.py",
    )

    assert config.trainer.num_epochs == 3
    assert trainer.fit_kwargs["num_epochs"] == 3
    assert build_kwargs["num_epochs"] == 3


def test_resolve_resume_checkpoint_defaults_to_newest_latest(tmp_path):
    flat = tmp_path / "checkpoints" / "latest.pt"
    older = tmp_path / "older" / "checkpoints" / "latest.pt"
    newer = tmp_path / "newer" / "checkpoints" / "latest.pt"
    flat.parent.mkdir(parents=True)
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    flat.write_bytes(b"flat")
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(flat, (1, 1))
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = ddpm_runner._resolve_resume_checkpoint(
        Path("latest"),
        output_root=tmp_path,
    )

    assert resolved == newer


def test_resolve_resume_checkpoint_accepts_run_directory(tmp_path):
    checkpoint_path = tmp_path / "run" / "checkpoints" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"latest")

    resolved = ddpm_runner._resolve_resume_checkpoint(
        tmp_path / "run",
        output_root=tmp_path,
    )

    assert resolved == checkpoint_path
