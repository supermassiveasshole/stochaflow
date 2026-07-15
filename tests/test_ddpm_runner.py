"""Tests for shared DDPM runner orchestration."""

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
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
        deterministic=False,
        no_progress=True,
        resume=None,
        skip_sampling=True,
        num_samples=1,
        sample_grid_size=1,
        trajectory_interval=1,
    )


def _options(config, args: Namespace | None = None) -> ddpm_runner.DDPMRunOptions:
    return ddpm_runner.DDPMRunOptions.from_namespace(
        args or _args(),
        configured_num_epochs=config.trainer.num_epochs,
    )


def _training_components(trainer: RecordingTrainer, logger: RecordingLogger):
    return SimpleNamespace(
        trainer=trainer,
        logger=logger,
        checkpoint_manager=SimpleNamespace(load=lambda *args, **kwargs: None),
        diffusion=SimpleNamespace(),
    )


def test_image_sample_shape_uses_configured_sample_bucket() -> None:
    config = load_config(Path("configs/ddpm_mnist_flowers102.yaml"))

    shape = ddpm_runner.image_sample_shape(config, 5)

    assert shape == torch.Size((5, 3, 64, 64))


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
        _options(config),
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
        _options(config),
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
    options = _options(config, args)

    ddpm_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train")),
        options,
        sample_shape_fn=lambda config, num_samples: torch.Size((num_samples, 3, 2, 2)),
        script_name="test.py",
    )

    assert config.trainer.num_epochs == 3
    assert trainer.fit_kwargs["num_epochs"] == 3
    assert build_kwargs["num_epochs"] == 3


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
    training.checkpoint_manager = SimpleNamespace(load=fail_load)
    monkeypatch.setattr(
        ddpm_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    args = _args()
    args.resume = tmp_path / "checkpoint.pt"

    with pytest.raises(RuntimeError, match="broken checkpoint"):
        ddpm_runner._run_single_bundle(
            config,
            DataBundle(train=_split("train")),
            _options(config, args),
            sample_shape_fn=lambda config, num_samples: torch.Size(
                (num_samples, 1, 2, 2)
            ),
            script_name="test.py",
        )

    assert logger.closed


def test_runner_rejects_checkpoint_at_target_epoch(monkeypatch, tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.experiment.output_dir = str(tmp_path)
    config.experiment.exp_id = "test"
    trainer = RecordingTrainer()
    logger = RecordingLogger()
    loaded = SimpleNamespace(epoch=config.trainer.num_epochs, global_step=123)
    training = _training_components(trainer, logger)
    training.checkpoint_manager = SimpleNamespace(
        load=lambda *args, **kwargs: loaded
    )
    monkeypatch.setattr(
        ddpm_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    args = _args()
    args.resume = tmp_path / "checkpoint.pt"

    with pytest.raises(ValueError, match="increase --epochs to continue"):
        ddpm_runner._run_single_bundle(
            config,
            DataBundle(train=_split("train")),
            _options(config, args),
            sample_shape_fn=lambda config, num_samples: torch.Size(
                (num_samples, 1, 2, 2)
            ),
            script_name="test.py",
        )

    assert trainer.global_step == 123
    assert logger.closed


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("limit_batches", 0, "--limit-batches"),
        ("limit_validation_batches", -1, "--limit-validation-batches"),
        ("limit_test_batches", 0, "--limit-test-batches"),
        ("num_samples", 0, "--num-samples"),
        ("sample_grid_size", -1, "--sample-grid-size"),
        ("trajectory_interval", 0, "--trajectory-interval"),
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
        ddpm_runner.DDPMRunOptions.from_namespace(
            args,
            configured_num_epochs=2,
        )


def test_runner_builds_registered_data_pipeline(monkeypatch, tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.experiment.output_dir = str(tmp_path / "outputs")
    args = _args()
    args.config = Path("unused.yaml")
    observed = {}

    class StubPipeline:
        def __init__(self, data_config, *, seed):
            observed["pipeline_config"] = data_config
            observed["seed"] = seed

        def build(self):
            return [DataBundle(train=_split("train"))]

    monkeypatch.setattr(ddpm_runner, "load_config", lambda path: config)
    monkeypatch.setattr(ddpm_runner, "DataPipeline", StubPipeline)
    monkeypatch.setattr(ddpm_runner, "_run_single_bundle", lambda *args, **kwargs: None)

    ddpm_runner.run_ddpm_from_args(
        args,
        script_name="test.py",
    )

    assert observed == {
        "pipeline_config": config.data,
        "seed": config.experiment.seed,
    }


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


def test_resolve_resume_checkpoint_keeps_folds_isolated(tmp_path):
    fold_zero = tmp_path / "older" / "fold_00" / "checkpoints" / "latest.pt"
    fold_one = tmp_path / "newer" / "fold_01" / "checkpoints" / "latest.pt"
    fold_zero.parent.mkdir(parents=True)
    fold_one.parent.mkdir(parents=True)
    fold_zero.write_bytes(b"fold zero")
    fold_one.write_bytes(b"fold one")
    os.utime(fold_zero, (1, 1))
    os.utime(fold_one, (2, 2))

    resolved = ddpm_runner._resolve_resume_checkpoint(
        Path("latest"),
        output_root=tmp_path,
        fold_index=0,
    )

    assert resolved == fold_zero


def test_configure_run_output_scopes_an_individual_fold(tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))

    ddpm_runner._configure_run_output(
        config,
        base_exp_id="run",
        base_output_dir=tmp_path,
        bundle=DataBundle(train=_split("train"), fold_index=2, num_folds=3),
    )

    assert config.experiment.exp_id == "run_fold_02"
    assert config.experiment.output_dir == str(tmp_path / "fold_02")


def test_resume_scope_rejects_one_checkpoint_for_all_folds(tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.data.splits.mode = "kfold"
    config.data.splits.num_folds = 3
    config.data.splits.fold_index = None

    with pytest.raises(ValueError, match="fold-specific checkpoints"):
        ddpm_runner._validate_resume_scope(config, tmp_path / "latest.pt")


def test_resume_scope_accepts_a_fold_checkpoint_directory(tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.data.splits.mode = "kfold"
    config.data.splits.num_folds = 3
    config.data.splits.fold_index = None
    tmp_path.mkdir(exist_ok=True)

    ddpm_runner._validate_resume_scope(config, tmp_path)
