"""Tests for centralized config loading."""

from copy import deepcopy
from pathlib import Path

import pytest

from stochaflow.data import build_data_loaders
from stochaflow.utils.config import (
    ConfigError,
    StochaflowConfig,
    load_config,
    load_config_dict,
)
from stochaflow.utils.registry import REGISTRIES, RegistryError


def test_load_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.name == "image"
    assert config.data.params["source"]["dataset"] == "MNIST"
    assert config.data.params["image"]["channels"] == 1
    assert config.data.params["image"]["size"] == [32, 32]
    assert config.data.params["partition"]["mode"] == "holdout"
    assert config.diffusion.name == "ddpm"
    assert len(config.logging.backends) >= 1
    assert config.diffusion.noise_schedule.params["num_timesteps"] == 1000
    assert config.data.params["loader"]["num_workers"] == 0
    assert config.lr_scheduler.name == "cosine"
    assert config.lr_scheduler.interval == "epoch"
    assert config.lr_scheduler.params == {
        "T_max": "auto",
        "eta_min": 0.00002,
    }
    assert config.sampling.sampler is not None
    assert config.sampling.sampler.name == "ddim"
    assert config.sampling.sampler.params == {
        "num_inference_steps": 100,
        "eta": 0.0,
    }
    assert config.ema.enabled
    assert config.ema.decay == 0.9995
    assert config.ema.update_after_step == 100
    assert config.ema.use_for_sampling
    assert config.sampling.num_samples == 64
    assert config.sampling.batch_size == 64
    assert config.sampling.shape == [1, 32, 32]
    assert [writer.name for writer in config.sampling.writers] == [
        "tensor",
        "image",
    ]
    assert config.sampling.debug.trajectory.enabled
    assert config.sampling.debug.trajectory.params == {"step_interval": 5}
    assert config.trainer.num_epochs == 30
    assert config.trainer.early_stopping.patience == 7
    assert config.trainer.early_stopping.min_delta == 0.00001
    assert config.artifacts.checkpoint_every == 5


def test_config_loads_custom_modules_before_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "config_extension.py"
    module_path.write_text(
        """
import torch.nn as nn

from stochaflow.extensions import (
    DataBuilder,
    DataLoaders,
    REGISTRIES,
    SamplingArtifactWriter,
)


@REGISTRIES.models.register("config_extension_model")
class ConfigExtensionModel(nn.Module):
    def forward(self, inputs):
        return inputs


@REGISTRIES.diffusions.register("config_extension_diffusion")
class ConfigExtensionDiffusion(nn.Module):
    def __init__(self, model, noise_schedule):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule


@REGISTRIES.data_builders.register("config_extension_builder")
class ConfigExtensionBuilder(DataBuilder):
    def build(self):
        return DataLoaders(train=[0])


@REGISTRIES.sampling_artifact_writers.register("config_extension_writer")
class ConfigExtensionWriter(SamplingArtifactWriter):
    def write(self, context):
        raise NotImplementedError
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"]["modules"] = ["config_extension"]
    raw["data"] = {"name": "config_extension_builder", "params": {}}

    config = load_config_dict(raw)
    repeated = load_config_dict(raw)

    assert config.extensions.modules == ["config_extension"]
    assert config.data.name == "config_extension_builder"
    assert repeated.data.name == "config_extension_builder"
    assert REGISTRIES.models.resolve("config_extension_model").__name__ == (
        "ConfigExtensionModel"
    )
    assert REGISTRIES.diffusions.resolve(
        "config_extension_diffusion"
    ).__name__ == "ConfigExtensionDiffusion"
    assert (
        REGISTRIES.data_builders.resolve("config_extension_builder").__name__
        == "ConfigExtensionBuilder"
    )
    assert REGISTRIES.sampling_artifact_writers.resolve(
        "config_extension_writer"
    ).__name__ == "ConfigExtensionWriter"


def test_config_rejects_removed_data_modules_without_mutating_input() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"]["modules"] = ["math"]
    original = deepcopy(raw)

    with pytest.raises(ConfigError, match=r"config\.data\.modules"):
        load_config_dict(raw)

    assert raw == original


@pytest.mark.parametrize("module", ["", "   ", 7, None])
def test_config_rejects_invalid_extension_module_declarations(module) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"]["modules"] = [module]

    with pytest.raises(ConfigError, match=r"extensions\.modules\[0\]"):
        load_config_dict(raw)


def test_config_reports_missing_extension_module() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"]["modules"] = ["stochaflow_missing_extension_for_test"]

    with pytest.raises(RegistryError, match="failed to import registry module"):
        load_config_dict(raw)


def test_config_rejects_empty_data_builder_name() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"]["name"] = ""

    with pytest.raises(ConfigError, match="non-empty registry name"):
        load_config_dict(raw)


def test_load_ddpm_flowers102_config() -> None:
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.model.name == "unet"
    assert config.data.params["source"]["dataset"] == "Flowers102"
    assert config.data.params["image"]["size"] == [64, 64]
    assert config.data.params["partition"]["mode"] == "official"
    assert config.data.params["loader"]["batch_size"] == 64
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
    assert config.diagnostics[0].name == "diffusion_quality"
    sampler_profiles = config.diagnostics[0].params["samplers"]
    assert [profile["id"] for profile in sampler_profiles] == [
        "ddpm_full",
        "ddim_50",
    ]
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
    assert data["extensions"] == {"modules": []}
    assert "model" in data
    assert "ema" in data
    assert "sampling" in data
    assert "lr_scheduler" in data
    assert "diagnostics" in data
    assert "trainer" in data


def test_load_multi_source_weighted_config() -> None:
    config = load_config(Path("configs/ddpm_mnist_flowers102.yaml"))

    sources = config.data.params["sources"]
    assert [source["id"] for source in sources] == ["digits", "flowers"]
    assert [source["sampling_weight"] for source in sources] == [0.4, 0.6]
    assert [
        bucket["name"] for bucket in config.data.params["batching"]["buckets"]
    ] == [
        "square_32",
        "square_64",
    ]


def test_config_rejects_removed_single_dataset_schema() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"] = {
        "dataset": {"name": "mnist", "params": {}},
        "dataloader": {},
        "splits": {"mode": "none"},
    }

    with pytest.raises(ConfigError, match=r"config\.data\.dataset"):
        load_config_dict(raw)


def test_config_requires_all_or_no_source_weights() -> None:
    raw = load_config(Path("configs/ddpm_mnist_flowers102.yaml")).to_dict()
    raw["data"]["params"]["sources"][0]["sampling_weight"] = None

    with pytest.raises(ConfigError, match="sampling_weight"):
        config = load_config_dict(raw)
        build_data_loaders(config.data, seed=config.experiment.seed)


def test_config_rejects_bucket_incompatible_with_unet_depth() -> None:
    raw = load_config(Path("configs/ddpm_mnist_flowers102.yaml")).to_dict()
    raw["data"]["params"]["batching"]["buckets"][0]["height"] = 0

    with pytest.raises(ConfigError, match="dimensions must be positive"):
        config = load_config_dict(raw)
        build_data_loaders(config.data, seed=config.experiment.seed)


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


def test_sampling_section_is_optional() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw.pop("sampling")

    config = load_config_dict(raw)

    assert config.sampling.sampler is None
    assert config.sampling.shape is None
    assert config.sampling.num_samples == 16
    assert config.sampling.batch_size == 16
    assert [writer.name for writer in config.sampling.writers] == ["tensor"]
    assert not config.sampling.debug.trajectory.enabled


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("num_samples",), 0),
        (("batch_size",), -1),
        (("shape",), []),
        (("shape",), [3, 0, 32]),
        (("writers", 0, "name"), ""),
    ],
)
def test_sampling_config_rejects_non_positive_values(path, value) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    target = raw["sampling"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ConfigError, match="sampling"):
        load_config_dict(raw)
