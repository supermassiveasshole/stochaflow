"""Tests for centralized config loading."""

from pathlib import Path

import pytest

from stochaflow.utils.config import (
    ConfigError,
    StochaflowConfig,
    load_config,
    load_config_dict,
)


def test_load_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.datasets[0].factory == "mnist"
    assert config.data.datasets[0].id == "mnist"
    assert config.data.image.channels == 1
    assert config.data.batching.sample_bucket == "square_32"
    assert config.data.splits.mode == "random_holdout"
    assert config.data.splits.validation_size == 10000
    assert config.diffusion.name == "ddpm"
    assert len(config.logging.backends) >= 1
    assert config.diffusion.noise_schedule.params["num_timesteps"] == 1000
    assert config.lr_scheduler.name is None


def test_config_loads_custom_dataset_modules() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"]["modules"] = ["my_project.datasets"]
    raw["data"]["datasets"][0]["factory"] = "manifest"

    config = load_config_dict(raw)

    assert config.data.modules == ["my_project.datasets"]
    assert config.data.datasets[0].factory == "manifest"


def test_load_ddpm_flowers102_config() -> None:
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.datasets[0].factory == "flowers102"
    assert config.data.batching.buckets[0].height == 64
    assert config.data.splits.mode == "official"
    assert config.data.datasets[0].splits.validation == "val"
    assert config.data.datasets[0].splits.test == "test"
    assert config.data.dataloader.batch_size == 64
    assert config.diffusion.name == "ddpm"
    assert config.diffusion.params["clip_denoised"] is True
    assert config.diffusion.noise_schedule.name == "linear_beta"
    assert config.diffusion.noise_schedule.params["num_timesteps"] == 1000
    assert config.optimizer.params["lr"] == 0.0001
    assert config.lr_scheduler.name == "warmup_cosine"
    assert config.lr_scheduler.interval == "step"
    assert config.lr_scheduler.params["total_steps"] == "auto"
    assert config.ema.enabled
    assert config.ema.use_for_sampling
    assert len(config.diagnostics) == 1
    assert config.diagnostics[0].name == "ddpm"
    assert config.trainer.num_epochs == 700
    assert config.trainer.early_stopping.monitor == "train_loss"
    assert not config.trainer.early_stopping.enabled


def test_load_ddim_cifar10_config() -> None:
    config = load_config(Path("configs/ddim_cifar10.yaml"))

    assert config.diffusion.name == "ddim"
    assert config.diffusion.noise_schedule.name == "linear_beta"
    assert config.diffusion.params["num_inference_steps"] == 100
    assert config.diffusion.params["eta"] == 0.0
    assert config.objective.name == "ddpm_epsilon"


def test_legacy_ddpm_scheduler_config_migrates_to_noise_schedule() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    diffusion = raw["diffusion"]
    diffusion["scheduler"] = diffusion.pop("noise_schedule")
    diffusion["scheduler"]["name"] = "linear_ddpm"

    config = load_config_dict(raw)

    assert config.diffusion.noise_schedule.name == "linear_beta"


def test_config_rejects_legacy_and_current_schedule_keys_together() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["diffusion"]["scheduler"] = {
        "name": "linear_ddpm",
        "params": {"num_timesteps": 1000},
    }

    with pytest.raises(ConfigError, match="both scheduler and noise_schedule"):
        load_config_dict(raw)


def test_config_to_dict_preserves_top_level_sections() -> None:
    config = load_config(Path("configs/ddpm_cifar10.yaml"))
    data = config.to_dict()
    assert "experiment" in data
    assert "model" in data
    assert "ema" in data
    assert "lr_scheduler" in data
    assert "diagnostics" in data
    assert "trainer" in data


def test_load_multi_source_weighted_config() -> None:
    config = load_config(Path("configs/ddpm_mnist_flowers102.yaml"))

    assert [source.id for source in config.data.datasets] == ["digits", "flowers"]
    assert [source.sampling_weight for source in config.data.datasets] == [0.4, 0.6]
    assert [bucket.name for bucket in config.data.batching.buckets] == [
        "square_32",
        "square_64",
    ]


def test_config_rejects_removed_single_dataset_schema() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"] = {
        "dataset": {"name": "mnist", "params": {}},
        "dataloader": raw["data"]["dataloader"],
        "splits": {"mode": "none"},
    }

    with pytest.raises(ConfigError, match="legacy data config"):
        load_config_dict(raw)


def test_config_requires_all_or_no_source_weights() -> None:
    raw = load_config(Path("configs/ddpm_mnist_flowers102.yaml")).to_dict()
    raw["data"]["datasets"][0]["sampling_weight"] = None

    with pytest.raises(ConfigError, match="sampling_weight"):
        load_config_dict(raw)


def test_config_rejects_bucket_incompatible_with_unet_depth() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"]["batching"]["buckets"][0]["height"] = 31

    with pytest.raises(ConfigError, match="divisible by"):
        load_config_dict(raw)


def test_lr_scheduler_config_rejects_invalid_interval() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["lr_scheduler"] = {"name": "step", "interval": "batch", "params": {}}

    with pytest.raises(ConfigError, match="lr_scheduler.interval"):
        load_config_dict(raw)


def test_disabled_lr_scheduler_ignores_interval() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["lr_scheduler"] = {"name": None, "interval": "batch", "params": {}}

    config = load_config_dict(raw)

    assert config.lr_scheduler.name is None
    assert config.lr_scheduler.interval == "batch"


def test_warmup_cosine_config_rejects_invalid_bounds() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["lr_scheduler"] = {
        "name": "warmup_cosine",
        "interval": "step",
        "params": {
            "warmup_steps": 10,
            "total_steps": 10,
            "min_lr_ratio": 0.05,
        },
    }

    with pytest.raises(ConfigError, match="greater than warmup_steps"):
        load_config_dict(raw)


def test_ema_config_rejects_invalid_decay() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["ema"] = {"enabled": True, "decay": 1.0}

    with pytest.raises(ConfigError, match="ema.decay"):
        load_config_dict(raw)
