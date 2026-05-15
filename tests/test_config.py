"""Tests for centralized config loading."""

from pathlib import Path

from stochaflow.utils.config import StochaflowConfig, load_config


def test_load_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.dataset.name == "mnist"
    assert config.diffusion.name == "ddpm"
    assert len(config.logging.backends) >= 1
    assert config.diffusion.scheduler.params["num_timesteps"] == 1000


def test_config_to_dict_preserves_top_level_sections() -> None:
    config = load_config(Path("configs/ddpm_cifar10.yaml"))
    data = config.to_dict()
    assert "experiment" in data
    assert "model" in data
    assert "trainer" in data
