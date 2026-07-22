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
    assert config.process is not None
    assert config.model.name == "unet"
    assert config.data.name == "image"
    assert config.data.params["source"]["dataset"] == "MNIST"
    assert config.data.params["image"]["channels"] == 1
    assert config.data.params["image"]["size"] == [32, 32]
    assert config.data.params["partition"]["mode"] == "holdout"
    assert config.process.name == "discrete_gaussian"
    assert len(config.logging.backends) >= 1
    assert config.process.params["schedule"]["params"]["num_timesteps"] == 1000
    assert config.data.params["loader"]["num_workers"] == 0
    assert config.optimizer.name == "torch.optim.Adam"
    assert config.lr_scheduler is not None
    assert config.lr_scheduler.name == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert config.lr_scheduler.interval == "epoch"
    assert config.lr_scheduler.params == {
        "T_max": 30,
        "eta_min": 0.00002,
    }
    assert config.sampling.builder is not None
    sampler = config.sampling.builder.params["sampler"]
    assert sampler["name"] == "ddim"
    assert sampler["params"] == {
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
    assert config.sampling.builder.params["trajectory"] == {
        "enabled": True,
        "every_steps": 5,
    }
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
    Process,
    REGISTRIES,
    Sampler,
    SamplerResult,
    SamplingBuilder,
    SamplingOutput,
    SamplingArtifactWriter,
    TrainingBuilder,
)


@REGISTRIES.models.register("config_extension_model")
class ConfigExtensionModel(nn.Module):
    def forward(self, inputs):
        return inputs


@REGISTRIES.processes.register("config_extension_process")
class ConfigExtensionProcess(Process):
    pass


@REGISTRIES.samplers.register("config_extension_sampler")
class ConfigExtensionSampler(Sampler):
    def sample(self, dynamics, initial_state, **kwargs):
        return SamplerResult(initial_state, 0, {})


@REGISTRIES.sampling_builders.register("config_extension_sampling_builder")
class ConfigExtensionSamplingBuilder(SamplingBuilder):
    def run(self):
        return SamplingOutput((), {})


@REGISTRIES.data_builders.register("config_extension_builder")
class ConfigExtensionBuilder(DataBuilder):
    def build(self):
        return DataLoaders(train=[0])


@REGISTRIES.sampling_artifact_writers.register("config_extension_writer")
class ConfigExtensionWriter(SamplingArtifactWriter):
    def write(self, context):
        raise NotImplementedError


@REGISTRIES.training_builders.register("config_extension_training")
class ConfigExtensionTrainingBuilder(TrainingBuilder):
    def build(self):
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
    assert REGISTRIES.processes.resolve(
        "config_extension_process"
    ).__name__ == "ConfigExtensionProcess"
    assert REGISTRIES.samplers.resolve(
        "config_extension_sampler"
    ).__name__ == "ConfigExtensionSampler"
    assert REGISTRIES.sampling_builders.resolve(
        "config_extension_sampling_builder"
    ).__name__ == "ConfigExtensionSamplingBuilder"
    assert (
        REGISTRIES.data_builders.resolve("config_extension_builder").__name__
        == "ConfigExtensionBuilder"
    )
    assert REGISTRIES.sampling_artifact_writers.resolve(
        "config_extension_writer"
    ).__name__ == "ConfigExtensionWriter"
    assert REGISTRIES.training_builders.resolve(
        "config_extension_training"
    ).__name__ == "ConfigExtensionTrainingBuilder"


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
    assert config.process is not None
    assert config.model.name == "unet"
    assert config.data.params["source"]["dataset"] == "Flowers102"
    assert config.data.params["image"]["size"] == [64, 64]
    assert config.data.params["partition"]["mode"] == "official"
    assert config.data.params["loader"]["batch_size"] == 64
    assert config.process.name == "discrete_gaussian"
    assert config.process.params["schedule"]["name"] == "linear_beta"
    assert config.process.params["schedule"]["params"]["num_timesteps"] == 1000
    assert config.optimizer.params["lr"] == 0.0001
    assert config.lr_scheduler is not None
    assert config.lr_scheduler.name == "warmup_cosine"
    assert config.lr_scheduler.interval == "step"
    assert config.lr_scheduler.params["total_steps"] == 10500
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

    assert config.process is not None
    assert config.process.name == "discrete_gaussian"
    assert config.process.params["schedule"]["name"] == "linear_beta"
    assert config.sampling.builder is not None
    assert config.sampling.builder.params["sampler"]["name"] == "ddim"
    assert config.sampling.builder.params["sampler"]["params"]["eta"] == 0.0
    assert config.training.name == "gaussian_denoising"
    assert config.objective is not None
    assert config.objective.name == "mse"


def test_legacy_diffusion_config_is_rejected() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["diffusion"] = raw.pop("process")

    with pytest.raises(ConfigError, match=r"config\.diffusion"):
        load_config_dict(raw)


@pytest.mark.parametrize("declaration", [None, "missing"])
def test_process_is_optional_and_resolves_to_null(declaration: object) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    if declaration == "missing":
        raw.pop("process")
    else:
        raw["process"] = None

    config = load_config_dict(raw)

    assert config.process is None
    assert config.to_dict()["process"] is None


@pytest.mark.parametrize("declaration", [[], "discrete_gaussian", 3])
def test_optional_process_rejects_non_mapping_declarations(
    declaration: object,
) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["process"] = declaration

    with pytest.raises(ConfigError, match=r"config\.process must be a mapping"):
        load_config_dict(raw)


def test_config_rejects_empty_process_name_when_present() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["process"]["name"] = ""

    with pytest.raises(ConfigError, match="process.name"):
        load_config_dict(raw)


def test_config_to_dict_preserves_top_level_sections() -> None:
    config = load_config(Path("configs/ddpm_cifar10.yaml"))
    data = config.to_dict()
    assert "experiment" in data
    assert data["extensions"] == {"modules": []}
    assert "model" in data
    assert "training" in data
    assert "ema" in data
    assert "sampling" in data
    assert data["lr_scheduler"] is None
    assert "diagnostics" in data
    assert "trainer" in data


@pytest.mark.parametrize("declaration", [None, "missing"])
def test_objective_is_optional_and_resolves_to_null(declaration: object) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    if declaration == "missing":
        raw.pop("objective")
    else:
        raw["objective"] = None

    config = load_config_dict(raw)

    assert config.objective is None
    assert config.to_dict()["objective"] is None


def test_config_requires_training_declaration() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw.pop("training")

    with pytest.raises(TypeError, match="training"):
        load_config_dict(raw)


def test_config_rejects_empty_training_name() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["training"]["name"] = ""

    with pytest.raises(ConfigError, match="training.name"):
        load_config_dict(raw)


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


def test_optimizer_defaults_to_native_adam_with_minimal_explicit_params() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw.pop("optimizer")

    config = load_config_dict(raw)

    assert config.optimizer.name == "torch.optim.Adam"
    assert config.optimizer.params == {"lr": 0.0002}


def test_lr_scheduler_config_rejects_invalid_interval() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["lr_scheduler"] = {
        "name": "torch.optim.lr_scheduler.StepLR",
        "interval": "batch",
        "params": {"step_size": 1},
    }

    with pytest.raises(ConfigError, match="lr_scheduler.interval"):
        load_config_dict(raw)


@pytest.mark.parametrize("declaration", [None, "missing"])
def test_lr_scheduler_null_or_missing_disables_it(declaration: object) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    if declaration == "missing":
        raw.pop("lr_scheduler")
    else:
        raw["lr_scheduler"] = None

    config = load_config_dict(raw)

    assert config.lr_scheduler is None
    assert config.to_dict()["lr_scheduler"] is None


def test_lr_scheduler_rejects_legacy_null_name_form() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["lr_scheduler"] = {
        "name": None,
        "interval": "step",
        "params": {},
    }

    with pytest.raises(ConfigError, match="lr_scheduler.name"):
        load_config_dict(raw)


@pytest.mark.parametrize(
    ("section", "reserved_name"),
    [("optimizer", "params"), ("lr_scheduler", "optimizer")],
)
def test_optimization_config_rejects_runtime_injection_keys(
    section: str,
    reserved_name: str,
) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw[section]["params"][reserved_name] = "invalid"

    with pytest.raises(ConfigError, match=f"cannot override.*'{reserved_name}'"):
        load_config_dict(raw)


@pytest.mark.parametrize(
    ("section", "value"),
    [("optimizer", ""), ("optimizer", 3), ("lr_scheduler", "")],
)
def test_optimization_config_rejects_invalid_names(
    section: str,
    value: object,
) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw[section]["name"] = value

    with pytest.raises(ConfigError, match=rf"{section}\.name"):
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

    assert config.sampling.builder is None
    assert config.sampling.shape is None
    assert config.sampling.num_samples == 16
    assert config.sampling.batch_size == 16
    assert [writer.name for writer in config.sampling.writers] == ["tensor"]


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
