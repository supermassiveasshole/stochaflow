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


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("configs").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_all_builtin_configs_use_200_epochs(config_path: Path) -> None:
    config = load_config(config_path)

    assert config.trainer.num_epochs == 200


def test_load_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    assert isinstance(config, StochaflowConfig)
    assert config.process is not None
    assert config.model.name == "unet"
    assert config.data.name == "image"
    assert config.data.params["source"]["name"] == "torchvision"
    assert config.data.params["source"]["params"]["dataset"] == "MNIST"
    assert config.data.params["image"]["channels"] == 1
    assert config.data.params["image"]["size"] == [32, 32]
    assert config.data.params["partition"]["mode"] == "holdout"
    assert config.process.name == "discrete_gaussian"
    assert len(config.logging.backends) >= 1
    assert config.data.params["loader"] == {
        "batch_size": 128,
        "num_workers": 2,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "steps_per_epoch": "auto",
    }
    assert config.model.params == {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 96,
        "channel_multipliers": [1, 2, 4],
        "num_res_blocks": 3,
        "time_embedding_dim": 192,
        "dropout": 0.1,
        "attention_levels": [2],
        "attention_heads": 4,
    }
    assert config.process.params["schedule"] == {
        "name": "cosine_alpha_bar",
        "params": {
            "num_timesteps": 1000,
            "s": 0.008,
            "max_beta": 0.999,
        },
    }
    assert config.training.params == {"prediction_type": "v"}
    assert config.optimizer.name == "torch.optim.Adam"
    assert config.optimizer.params == {
        "lr": 0.0003,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
    }
    assert config.lr_scheduler is not None
    assert config.lr_scheduler.name == "warmup_cosine"
    assert config.lr_scheduler.interval == "step"
    assert config.lr_scheduler.params == {
        "warmup_steps": 2000,
        "total_steps": 78000,
        "min_lr_ratio": 0.05,
    }
    assert config.sampling.builder is not None
    assert config.sampling.builder.params["prediction_type"] == "v"
    sampler = config.sampling.builder.params["sampler"]
    assert sampler["name"] == "ddpm"
    assert sampler["params"] == {}
    assert config.ema.enabled
    assert config.ema.decay == 0.9995
    assert config.ema.update_after_step == 100
    assert config.ema.update_every == 1
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
        "every_steps": 50,
    }
    assert config.trainer.num_epochs == 200
    assert not config.trainer.early_stopping.enabled
    assert config.trainer.early_stopping.patience == 7
    assert config.trainer.early_stopping.min_delta == 0.00001
    assert [diagnostic.name for diagnostic in config.diagnostics] == [
        "diffusion_quality"
    ]
    diagnostic_params = config.diagnostics[0].params
    assert diagnostic_params["cadence"] == {
        "step_every": 100,
        "artifact_every_epochs": 10,
    }
    assert diagnostic_params["sampling"] == {
        "sample_num": 32,
        "batch_size": 32,
        "seed": 123,
    }
    assert diagnostic_params["use_ema"] is True
    assert diagnostic_params["failure_policy"] == "warn"
    assert [profile["id"] for profile in diagnostic_params["samplers"]] == [
        "ddim_50"
    ]
    assert diagnostic_params["samplers"][0] == {
        "id": "ddim_50",
        "name": "ddim",
        "params": {
            "num_inference_steps": 50,
            "eta": 0.0,
        },
        "trajectory": {
            "enabled": False,
            "every_steps": 5,
            "gif_fps": 8,
        },
    }
    providers = diagnostic_params["providers"]
    assert providers["step_metrics"] == [
        {"name": "timestep_bucket_loss", "params": {"buckets": 10}},
        {"name": "noise_alignment", "params": {}},
        {
            "name": "x0_reconstruction",
            "params": {"timesteps": [50, 250, 500, 750, 900]},
        },
    ]
    assert providers["sampler_artifacts"] == [
        {"name": "sample_grid", "params": {"nrow": 8}}
    ]
    assert diagnostic_params["reference"] == {"enabled": False}
    assert [backend.name for backend in config.logging.backends] == [
        "local",
        "tensorboard",
    ]
    assert config.artifacts.checkpoint_every == 50


def test_load_ddim_mnist_config() -> None:
    ddpm = load_config(Path("configs/ddpm_mnist.yaml"))
    ddim = load_config(Path("configs/ddim_mnist.yaml"))

    assert ddim.experiment.name == "ddim_mnist"
    assert ddim.experiment.output_dir == "outputs/ddim_mnist"
    assert ddim.data == ddpm.data
    assert ddim.model == ddpm.model
    assert ddim.process == ddpm.process
    assert ddim.training == ddpm.training
    assert ddim.objective == ddpm.objective
    assert ddim.optimizer == ddpm.optimizer
    assert ddim.lr_scheduler == ddpm.lr_scheduler
    assert ddim.ema == ddpm.ema
    assert ddim.diagnostics == ddpm.diagnostics
    assert ddim.logging == ddpm.logging
    assert ddim.sampling.builder is not None
    assert ddim.sampling.builder.params["prediction_type"] == "v"
    assert ddim.sampling.builder.params["sampler"] == {
        "name": "ddim",
        "params": {
            "num_inference_steps": 100,
            "eta": 0.0,
        },
    }
    assert ddim.sampling.builder.params["trajectory"] == {
        "enabled": True,
        "every_steps": 5,
    }


def test_config_parsing_does_not_import_declared_plugins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "imported"
    module_path = tmp_path / "config_extension.py"
    module_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"] = {"plugins": ["config_extension"]}
    original = deepcopy(raw)

    config = load_config_dict(raw)

    assert config.extensions.plugins == ["config_extension"]
    assert not marker.exists()
    assert raw == original


def test_config_rejects_removed_data_modules_without_mutating_input() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["data"]["modules"] = ["math"]
    original = deepcopy(raw)

    with pytest.raises(ConfigError, match=r"config\.data\.modules"):
        load_config_dict(raw)

    assert raw == original


@pytest.mark.parametrize("plugin", ["", "   ", " padded", 7])
def test_config_rejects_invalid_extension_plugin_declarations(plugin) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"] = {"plugins": [plugin]}

    with pytest.raises(ConfigError, match=r"extensions\.plugins\[0\]"):
        load_config_dict(raw)


def test_config_rejects_duplicate_extension_plugins() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"] = {"plugins": ["example", "example"]}

    with pytest.raises(ConfigError, match="duplicate entry-point name 'example'"):
        load_config_dict(raw)


def test_config_rejects_removed_extension_modules_schema() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"] = {"modules": ["example.extension"]}

    with pytest.raises(ConfigError, match=r"config\.extensions\.modules"):
        load_config_dict(raw)


@pytest.mark.parametrize(
    ("raw_extensions", "expected"),
    [({}, []), ({"plugins": []}, []), ({"plugins": None}, None)],
)
def test_config_preserves_extension_plugin_selection_semantics(
    raw_extensions: dict[str, object],
    expected: list[str] | None,
) -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["extensions"] = raw_extensions

    config = load_config_dict(raw)

    assert config.extensions.plugins == expected


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
    assert config.data.params["source"]["name"] == "torchvision"
    assert config.data.params["source"]["params"]["dataset"] == "Flowers102"
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
    assert config.lr_scheduler.params["warmup_steps"] == 150
    assert config.lr_scheduler.params["total_steps"] == 3000
    assert config.ema.enabled
    assert config.ema.use_for_sampling
    assert len(config.diagnostics) == 1
    assert config.diagnostics[0].name == "diffusion_quality"
    sampler_profiles = config.diagnostics[0].params["samplers"]
    assert [profile["id"] for profile in sampler_profiles] == [
        "ddpm_full",
        "ddim_50",
    ]
    assert config.trainer.num_epochs == 200
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

    with pytest.raises(ConfigError, match=r"process\.name"):
        load_config_dict(raw)


def test_config_to_dict_preserves_top_level_sections() -> None:
    config = load_config(Path("configs/ddpm_cifar10.yaml"))
    data = config.to_dict()
    assert "experiment" in data
    assert data["extensions"] == {"plugins": []}
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

    with pytest.raises(ConfigError, match="training"):
        load_config_dict(raw)


def test_config_rejects_empty_training_name() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["training"]["name"] = ""

    with pytest.raises(ConfigError, match=r"training\.name"):
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

    config = load_config_dict(raw)
    with pytest.raises(ConfigError, match="sampling_weight"):
        build_data_loaders(config.data, seed=config.experiment.seed)


def test_config_rejects_bucket_incompatible_with_unet_depth() -> None:
    raw = load_config(Path("configs/ddpm_mnist_flowers102.yaml")).to_dict()
    raw["data"]["params"]["batching"]["buckets"][0]["height"] = 0

    config = load_config_dict(raw)
    with pytest.raises(ConfigError, match="dimensions must be positive"):
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

    with pytest.raises(ConfigError, match=r"lr_scheduler\.interval"):
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

    with pytest.raises(ConfigError, match=r"lr_scheduler\.name"):
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

    with pytest.raises(ConfigError, match=r"ema\.decay"):
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
