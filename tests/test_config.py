"""Tests for centralized config loading."""

from pathlib import Path

import pytest

from stochaflow.utils.config import ConfigError, StochaflowConfig, load_config
from stochaflow.utils.config import load_config_dict


def test_load_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.dataset.name == "mnist"
    assert config.data.splits.mode == "random_holdout"
    assert config.data.splits.validation_size == 10000
    assert config.diffusion.name == "ddpm"
    assert len(config.logging.backends) >= 1
    assert config.diffusion.scheduler.params["num_timesteps"] == 1000
    assert config.lr_scheduler.name is None


def test_load_ddpm_flowers102_config() -> None:
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.dataset.name == "flowers102"
    assert config.data.dataset.params["image_size"] == 64
    assert config.data.splits.mode == "all"
    assert config.data.splits.train_splits == ["train", "val", "test"]
    assert config.data.splits.test_split is None
    assert config.data.dataloader.batch_size == 64
    assert config.diffusion.name == "ddpm"
    assert config.diffusion.params["clip_denoised"] is True
    assert config.diffusion.scheduler.name == "linear_ddpm"
    assert config.diffusion.scheduler.params["num_timesteps"] == 1000
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
    assert config.diffusion.scheduler.name == "linear_ddpm"
    assert config.diffusion.params["num_inference_steps"] == 100
    assert config.diffusion.params["eta"] == 0.0
    assert config.objective.name == "ddpm_epsilon"


def test_config_to_dict_preserves_top_level_sections() -> None:
    config = load_config(Path("configs/ddpm_cifar10.yaml"))
    data = config.to_dict()
    assert "experiment" in data
    assert "model" in data
    assert "ema" in data
    assert "lr_scheduler" in data
    assert "diagnostics" in data
    assert "trainer" in data


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
