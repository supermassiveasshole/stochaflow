"""Tests for centralized config loading."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from stochaflow.data import build_data_loaders
from stochaflow.utils.config import (
    ConfigError,
    StochaflowConfig,
    load_config,
    load_config_dict,
    parse_sample_request,
)

BUILTIN_CONFIGS = Path("examples/built-in/image-generation/configs")
BUILTIN_TRAIN_CONFIG = BUILTIN_CONFIGS / "train/mnist.yaml"


def _multi_resolution_config_raw() -> dict[str, Any]:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    source = deepcopy(raw["data"]["params"]["source"])
    raw["data"] = {
        "name": "multi_resolution_image",
        "params": {
            "sources": [
                {
                    "id": "first",
                    "sampling_weight": 0.4,
                    "source": deepcopy(source),
                },
                {
                    "id": "second",
                    "sampling_weight": 0.6,
                    "source": deepcopy(source),
                },
            ],
            "image": {
                "channels": 1,
                "normalize": True,
                "random_horizontal_flip": False,
            },
            "batching": {
                "buckets": [
                    {"name": "square_32", "height": 32, "width": 32},
                    {"name": "square_64", "height": 64, "width": 64},
                ],
                "base_bucket": "square_64",
                "dynamic_batch_size": True,
            },
            "loader": deepcopy(raw["data"]["params"]["loader"]),
            "partition": {
                "mode": "holdout",
                "validation_size": 0.1,
            },
        },
    }
    return raw


@pytest.mark.parametrize(
    "config_path",
    sorted((BUILTIN_CONFIGS / "train").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_all_builtin_train_configs_use_200_epochs(config_path: Path) -> None:
    config = load_config(config_path)

    assert config.trainer.num_epochs == 200


def test_load_mnist_train_config() -> None:
    config = load_config(BUILTIN_TRAIN_CONFIG)
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
    assert [
        {
            "id": metric.id,
            "name": metric.name,
            "channel": metric.channel,
            "phases": metric.phases,
            "params": metric.params,
        }
        for metric in config.metrics
    ] == [
        {
            "id": "prediction_mae",
            "name": "mae",
            "channel": "gaussian.prediction_target",
            "phases": ["validation", "test"],
            "params": {},
        },
        {
            "id": "clean_reconstruction_mse",
            "name": "mse",
            "channel": "gaussian.clean_reconstruction",
            "phases": ["validation", "test"],
            "params": {},
        },
    ]
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
    assert not config.sampling.run_after_training
    assert config.sampling.sampler is None
    assert "prediction_type" not in config.sampling.options
    assert config.ema.enabled
    assert config.ema.decay == 0.9995
    assert config.ema.update_after_step == 100
    assert config.ema.update_every == 1
    assert config.ema.use_for_sampling
    assert config.sampling.num_samples == 16
    assert config.sampling.batch_size == 16
    assert config.sampling.shape is None
    assert [writer.name for writer in config.sampling.writers] == ["tensor"]
    assert config.trainer.num_epochs == 200
    assert not config.trainer.early_stopping.enabled
    assert config.trainer.early_stopping.monitor == "valid/loss"
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
        "shape": [1, 32, 32],
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


def test_builtin_config_tree_separates_train_sample_and_overlays() -> None:
    train_paths = sorted((BUILTIN_CONFIGS / "train").glob("*.yaml"))
    sample_paths = sorted((BUILTIN_CONFIGS / "sample").glob("*.yaml"))
    overlay_paths = sorted((BUILTIN_CONFIGS / "overlays").glob("*.yaml"))

    assert [path.name for path in train_paths] == ["mnist.yaml"]
    assert [path.name for path in sample_paths] == [
        "mnist-ddim-50.yaml",
        "mnist-ddpm.yaml",
    ]
    assert [path.name for path in overlay_paths] == [
        "mnist-observability.yaml"
    ]

    train_document = yaml.safe_load(BUILTIN_TRAIN_CONFIG.read_text(encoding="utf-8"))
    assert "sampling" not in train_document

    for sample_path in sample_paths:
        document = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
        assert set(document) == {"sampling"}
        parsed = parse_sample_request(document["sampling"])
        assert parsed.provided_fields == {
            "sampler",
            "options",
            "shape",
            "num_samples",
            "batch_size",
            "seed",
            "writers",
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
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["extensions"] = {"plugins": ["config_extension"]}
    original = deepcopy(raw)

    config = load_config_dict(raw)

    assert config.extensions.plugins == ["config_extension"]
    assert not marker.exists()
    assert raw == original


def test_config_rejects_removed_data_modules_without_mutating_input() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["data"]["modules"] = ["math"]
    original = deepcopy(raw)

    with pytest.raises(ConfigError, match=r"config\.data\.modules"):
        load_config_dict(raw)

    assert raw == original


@pytest.mark.parametrize("plugin", ["", "   ", " padded", 7])
def test_config_rejects_invalid_extension_plugin_declarations(plugin) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["extensions"] = {"plugins": [plugin]}

    with pytest.raises(ConfigError, match=r"extensions\.plugins\[0\]"):
        load_config_dict(raw)


def test_config_rejects_duplicate_extension_plugins() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["extensions"] = {"plugins": ["example", "example"]}

    with pytest.raises(ConfigError, match="duplicate entry-point name 'example'"):
        load_config_dict(raw)


@pytest.mark.parametrize("monitor", ["train_loss", "valid_loss"])
def test_config_rejects_legacy_epoch_monitor_keys(monitor: str) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["trainer"]["early_stopping"]["monitor"] = monitor

    with pytest.raises(
        ConfigError,
        match=r"monitor.*(?:whitespace|canonical validation metric key)",
    ):
        load_config_dict(raw)


@pytest.mark.parametrize(
    "monitor",
    [
        " valid/loss ",
        "train/loss",
        "train/step/loss",
        "system/train/loss",
        "diagnostics/quality/fid",
        "valid/metrics/id/nested/subkey",
    ],
)
def test_config_rejects_noncanonical_epoch_monitor_keys(monitor: str) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["trainer"]["early_stopping"]["monitor"] = monitor

    with pytest.raises(
        ConfigError,
        match=r"monitor.*(?:whitespace|canonical validation metric key)",
    ):
        load_config_dict(raw)


def test_config_rejects_removed_early_stopping_missing_policy() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["trainer"]["early_stopping"]["missing"] = "skip"

    with pytest.raises(
        ConfigError,
        match=r"unknown config field.*early_stopping\.missing",
    ):
        load_config_dict(raw)


def test_config_without_metrics_keeps_backward_compatible_empty_default() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw.pop("metrics")

    config = load_config_dict(raw)

    assert config.metrics == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "invalid/id", r"metrics\[0\]\.id must match"),
        ("phases", [], r"metrics\[0\]\.phases must not be empty"),
        (
            "phases",
            ["validation", "validation"],
            "contains duplicate phase",
        ),
        ("phases", ["predict"], r"metrics\[0\]\.phases\[0\]"),
    ],
)
def test_config_rejects_invalid_metric_declarations(
    field: str,
    value: Any,
    message: str,
) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["metrics"][0][field] = value

    with pytest.raises(ConfigError, match=message):
        load_config_dict(raw)


def test_config_rejects_duplicate_metric_ids() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["metrics"][1]["id"] = raw["metrics"][0]["id"]

    with pytest.raises(ConfigError, match="duplicate metric id"):
        load_config_dict(raw)


def test_config_rejects_removed_extension_modules_schema() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
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
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["extensions"] = raw_extensions

    config = load_config_dict(raw)

    assert config.extensions.plugins == expected


def test_config_rejects_empty_data_builder_name() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["data"]["name"] = ""

    with pytest.raises(ConfigError, match="non-empty registry name"):
        load_config_dict(raw)


def test_legacy_diffusion_config_is_rejected() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["diffusion"] = raw.pop("process")

    with pytest.raises(ConfigError, match=r"config\.diffusion"):
        load_config_dict(raw)


@pytest.mark.parametrize("declaration", [None, "missing"])
def test_process_is_optional_and_resolves_to_null(declaration: object) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
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
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["process"] = declaration

    with pytest.raises(ConfigError, match=r"config\.process must be a mapping"):
        load_config_dict(raw)


def test_config_rejects_empty_process_name_when_present() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["process"]["name"] = ""

    with pytest.raises(ConfigError, match=r"process\.name"):
        load_config_dict(raw)


def test_config_to_dict_preserves_top_level_sections() -> None:
    config = load_config(BUILTIN_TRAIN_CONFIG)
    data = config.to_dict()
    assert "experiment" in data
    assert data["extensions"] == {"plugins": []}
    assert "model" in data
    assert "training" in data
    assert "ema" in data
    assert "sampling" in data
    assert data["lr_scheduler"] is not None
    assert "diagnostics" in data
    assert "trainer" in data


@pytest.mark.parametrize("declaration", [None, "missing"])
def test_objective_is_optional_and_resolves_to_null(declaration: object) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    if declaration == "missing":
        raw.pop("objective")
    else:
        raw["objective"] = None

    config = load_config_dict(raw)

    assert config.objective is None
    assert config.to_dict()["objective"] is None


def test_config_requires_training_declaration() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw.pop("training")

    with pytest.raises(ConfigError, match="training"):
        load_config_dict(raw)


def test_config_rejects_empty_training_name() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["training"]["name"] = ""

    with pytest.raises(ConfigError, match=r"training\.name"):
        load_config_dict(raw)


def test_load_multi_source_weighted_config() -> None:
    config = load_config_dict(_multi_resolution_config_raw())

    sources = config.data.params["sources"]
    assert [source["id"] for source in sources] == ["first", "second"]
    assert [source["sampling_weight"] for source in sources] == [0.4, 0.6]
    assert [
        bucket["name"] for bucket in config.data.params["batching"]["buckets"]
    ] == [
        "square_32",
        "square_64",
    ]


def test_config_rejects_removed_single_dataset_schema() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["data"] = {
        "dataset": {"name": "mnist", "params": {}},
        "dataloader": {},
        "splits": {"mode": "none"},
    }

    with pytest.raises(ConfigError, match=r"config\.data\.dataset"):
        load_config_dict(raw)


def test_config_requires_all_or_no_source_weights() -> None:
    raw = _multi_resolution_config_raw()
    raw["data"]["params"]["sources"][0]["sampling_weight"] = None

    config = load_config_dict(raw)
    with pytest.raises(ConfigError, match="sampling_weight"):
        build_data_loaders(config.data, seed=config.experiment.seed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_config_rejects_non_finite_source_sampling_weight(
    value: float,
) -> None:
    raw = _multi_resolution_config_raw()
    raw["data"]["params"]["sources"][0]["sampling_weight"] = value
    config = load_config_dict(raw)
    with pytest.raises(
        ConfigError,
        match=r"sources\[0\]\.sampling_weight.*finite positive number",
    ):
        build_data_loaders(config.data, seed=config.experiment.seed)


def test_config_rejects_bucket_incompatible_with_unet_depth() -> None:
    raw = _multi_resolution_config_raw()
    raw["data"]["params"]["batching"]["buckets"][0]["height"] = 0

    config = load_config_dict(raw)
    with pytest.raises(ConfigError, match="dimensions must be positive"):
        build_data_loaders(config.data, seed=config.experiment.seed)


def test_optimizer_defaults_to_native_adam_with_minimal_explicit_params() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw.pop("optimizer")

    config = load_config_dict(raw)

    assert config.optimizer.name == "torch.optim.Adam"
    assert config.optimizer.params == {"lr": 0.0002}


def test_lr_scheduler_config_rejects_invalid_interval() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["lr_scheduler"] = {
        "name": "torch.optim.lr_scheduler.StepLR",
        "interval": "batch",
        "params": {"step_size": 1},
    }

    with pytest.raises(ConfigError, match=r"lr_scheduler\.interval"):
        load_config_dict(raw)


@pytest.mark.parametrize("declaration", [None, "missing"])
def test_lr_scheduler_null_or_missing_disables_it(declaration: object) -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    if declaration == "missing":
        raw.pop("lr_scheduler")
    else:
        raw["lr_scheduler"] = None

    config = load_config_dict(raw)

    assert config.lr_scheduler is None
    assert config.to_dict()["lr_scheduler"] is None


def test_lr_scheduler_rejects_legacy_null_name_form() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
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
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
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
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw[section]["name"] = value

    with pytest.raises(ConfigError, match=rf"{section}\.name"):
        load_config_dict(raw)


def test_ema_config_rejects_invalid_decay() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw["ema"] = {"enabled": True, "decay": 1.0}

    with pytest.raises(ConfigError, match=r"ema\.decay"):
        load_config_dict(raw)


def test_sampling_section_is_optional() -> None:
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    raw.pop("sampling")

    config = load_config_dict(raw)

    assert not config.sampling.run_after_training
    assert config.sampling.sampler is None
    assert config.sampling.options == {}
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
    raw = load_config(BUILTIN_TRAIN_CONFIG).to_dict()
    target = raw["sampling"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ConfigError, match="sampling"):
        load_config_dict(raw)
