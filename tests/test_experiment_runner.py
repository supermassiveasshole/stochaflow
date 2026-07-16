"""Tests for shared experiment runner orchestration."""

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from stochaflow.data.pipeline import DataBundle, SplitData
from stochaflow.sampling.runtime import image_sample_shape
from stochaflow.scripts import experiment_runner
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


def _training_components(trainer: RecordingTrainer, logger: RecordingLogger):
    return SimpleNamespace(
        trainer=trainer,
        logger=logger,
        checkpoint_manager=SimpleNamespace(load=lambda *args, **kwargs: None),
        diffusion=SimpleNamespace(),
    )


def test_image_sample_shape_uses_configured_sample_bucket() -> None:
    config = load_config(Path("configs/ddpm_mnist_flowers102.yaml"))

    shape = image_sample_shape(config, 5)

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
        experiment_runner,
        "build_training_components",
        build_training_components,
    )

    experiment_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train"), valid=_split("valid")),
        _options(config),
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
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: _training_components(trainer, logger),
    )

    experiment_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train")),
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

    experiment_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train")),
        options,
    )

    assert config.trainer.num_epochs == 3
    assert trainer.fit_kwargs["num_epochs"] == 3
    assert build_kwargs["num_epochs"] == 3
    resolved = yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())
    assert resolved["trainer"]["num_epochs"] == 3
    assert resolved["trainer"]["show_progress"] is False


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

    experiment_runner._run_single_bundle(
        config,
        DataBundle(train=_split("train")),
        _options(config, args),
    )

    assert observed["checkpoint"] == trainer.best_checkpoint_path
    assert observed["output_dir"] == tmp_path / "samples" / "final"
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
    training.checkpoint_manager = SimpleNamespace(load=fail_load)
    monkeypatch.setattr(
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    args = _args()
    args.resume = tmp_path / "checkpoint.pt"

    with pytest.raises(RuntimeError, match="broken checkpoint"):
        experiment_runner._run_single_bundle(
            config,
            DataBundle(train=_split("train")),
            _options(config, args),
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
        experiment_runner,
        "build_training_components",
        lambda config, **kwargs: training,
    )
    args = _args()
    args.resume = tmp_path / "checkpoint.pt"

    with pytest.raises(ValueError, match="increase --epochs to continue"):
        experiment_runner._run_single_bundle(
            config,
            DataBundle(train=_split("train")),
            _options(config, args),
        )

    assert trainer.global_step == 123
    assert logger.closed


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
def test_runner_builds_registered_data_pipeline(monkeypatch, tmp_path, config_path):
    config = load_config(config_path)
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

    monkeypatch.setattr(experiment_runner, "load_config", lambda path: config)
    monkeypatch.setattr(experiment_runner, "DataPipeline", StubPipeline)
    monkeypatch.setattr(
        experiment_runner,
        "_run_single_bundle",
        lambda *args, **kwargs: None,
    )

    experiment_runner.run_experiment_from_args(args)

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

    resolved = experiment_runner._resolve_resume_checkpoint(
        Path("latest"),
        output_root=tmp_path,
    )

    assert resolved == newer


def test_resolve_resume_checkpoint_accepts_run_directory(tmp_path):
    checkpoint_path = tmp_path / "run" / "checkpoints" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"latest")

    resolved = experiment_runner._resolve_resume_checkpoint(
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

    resolved = experiment_runner._resolve_resume_checkpoint(
        Path("latest"),
        output_root=tmp_path,
        fold_index=0,
    )

    assert resolved == fold_zero


def test_configure_run_output_scopes_an_individual_fold(tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))

    experiment_runner._configure_run_output(
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
        experiment_runner._validate_resume_scope(config, tmp_path / "latest.pt")


def test_resume_scope_accepts_a_fold_checkpoint_directory(tmp_path):
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    config.data.splits.mode = "kfold"
    config.data.splits.num_folds = 3
    config.data.splits.fold_index = None
    tmp_path.mkdir(exist_ok=True)

    experiment_runner._validate_resume_scope(config, tmp_path)
