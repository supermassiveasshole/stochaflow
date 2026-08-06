from __future__ import annotations

import importlib
import math
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import yaml

from stochaflow.data import IMAGE_DATA_SOURCES
from stochaflow.evaluation import (
    CheckpointSubjectConfig,
    load_evaluation_config,
)
from stochaflow.models import ADMUNet
from stochaflow.utils.config import (
    StochaflowConfig,
    load_config,
    load_sample_config,
)
from stochaflow.utils.registry import REGISTRIES

_ROOT = Path(__file__).resolve().parents[1]
_SHOWCASE = _ROOT / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _SHOWCASE / "src"
_PROJECT_VERSION = tomllib.loads(
    (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
_RELEASE_WHEEL_URL = (
    "https://github.com/supermassiveasshole/stochaflow/releases/"
    f"download/v{_PROJECT_VERSION}/"
    f"stochaflow-{_PROJECT_VERSION}-py3-none-any.whl"
)
_ADM_CONFIG = _SHOWCASE / "experiments" / "production" / "train-adm-128.yaml"
_LEARNED_RANGE_ADM_CONFIG = (
    _SHOWCASE
    / "experiments"
    / "production"
    / "train-adm-128-learned-range-v.yaml"
)
_LEARNED_RANGE_OFFICIAL_TEST_CONFIG = (
    _SHOWCASE
    / "experiments"
    / "evaluation"
    / "formal-ddpm100-cfg2-official-test-learned-range-v.yaml"
)
_DIT_CONFIG = _SHOWCASE / "experiments" / "production" / "train-dit-128.yaml"
_SMOKE_CONFIG = _SHOWCASE / "experiments" / "smoke" / "train-adm-128.yaml"
_SAMPLING_CONFIG = (
    _SHOWCASE / "experiments" / "sampling" / "ddim50-cfg2.yaml"
)
_README_SAMPLING_CONFIG = (
    _SHOWCASE
    / "experiments"
    / "sampling"
    / "ddim50-cfg2-readme.yaml"
)
_LOCK = (
    _SHOWCASE
    / "src"
    / "stochaflow_afhq_v2"
    / "resources"
    / "afhq-v2.lock.yaml"
)
_EXPECTED_IMAGE_METRICS = [
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


def _raw(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _showcase_tool_module(name: str) -> ModuleType:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.invalidate_caches()
    return importlib.import_module(f"stochaflow_afhq_v2.tools.{name}")


def _capacity_module() -> ModuleType:
    return _showcase_tool_module("capacity")


def test_afhq_showcase_registers_source_and_formal_evaluation_extensions() -> None:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")
    declaration = tomllib.loads(
        (_SHOWCASE / "pyproject.toml").read_text(encoding="utf-8")
    )
    locked_packages = tomllib.loads(
        (_SHOWCASE / "uv.lock").read_text(encoding="utf-8")
    )["package"]
    locked_stochaflow = next(
        package
        for package in locked_packages
        if package["name"] == "stochaflow"
    )

    assert "class_labeled_image" in REGISTRIES.data_builders.names()
    assert "afhq-v2.class-images" not in REGISTRIES.data_builders.names()
    assert "afhq-v2.official" in IMAGE_DATA_SOURCES.names()
    assert (
        "afhq-v2.class-conditional-generation"
        in REGISTRIES.evaluation_builders.names()
    )
    assert "afhq-v2.class-aware-distribution" in REGISTRIES.metrics.names()
    assert (_SHOWCASE / "uv.lock").is_file()
    assert "torchmetrics" in {
        dependency["name"]
        for dependency in locked_stochaflow["dependencies"]
    }
    assert [
        dependency["name"]
        for dependency in locked_stochaflow["optional-dependencies"]["quality"]
    ] == ["torch-fidelity"]
    assert declaration["project"]["name"] == "stochaflow-afhq-v2"
    assert declaration["project"]["entry-points"]["stochaflow.extensions"] == {
        "stochaflow-afhq-v2": "stochaflow_afhq_v2.stochaflow_ext"
    }
    assert declaration["project"]["scripts"] == {
        "stochaflow-afhq-v2-prepare": (
            "stochaflow_afhq_v2.tools.prepare:main"
        ),
        "stochaflow-afhq-v2-capacity": (
            "stochaflow_afhq_v2.tools.capacity:main"
        ),
    }
    assert declaration["tool"]["setuptools"]["package-data"] == {
        "stochaflow_afhq_v2": ["resources/*.yaml"]
    }
    assert declaration["tool"]["uv"]["sources"]["stochaflow"] == {
        "path": "../../..",
        "editable": True,
    }
    assert declaration["project"]["optional-dependencies"]["quality"] == [
        f"stochaflow[quality] @ {_RELEASE_WHEEL_URL}"
    ]
    package = _SHOWCASE / "src" / "stochaflow_afhq_v2"
    assert (package / "resources" / "afhq-v2.lock.yaml").is_file()
    assert {path.name for path in package.glob("*.py")} == {
        "__init__.py",
        "artifact.py",
    }
    assert {
        path.name
        for path in (package / "_preparation").glob("*.py")
    } == {
        "__init__.py",
        "archive.py",
        "contracts.py",
        "downloading.py",
        "image_transform.py",
        "locking.py",
        "materialization.py",
        "safe_file.py",
        "source_acquisition.py",
        "source_lock.py",
        "source_session.py",
    }
    assert (package / "stochaflow_ext" / "source.py").is_file()
    assert (package / "stochaflow_ext" / "evaluation.py").is_file()
    assert not (package / "stochaflow_ext" / "builder.py").exists()
    assert not (package / "stochaflow_ext" / "batching.py").exists()
    assert not (package / "stochaflow_ext" / "partitioning.py").exists()
    assert not (package / "stochaflow_ext" / "config.py").exists()
    assert not (package / "stochaflow_ext" / "data.py").exists()
    assert not (package / "stochaflow_ext" / "dataset.py").exists()
    assert not (_SHOWCASE / "prepare.py").exists()
    assert not (_SHOWCASE / "experiments" / "ddpm_128.yaml").exists()


def test_afhq_production_configs_parse_and_follow_pipeline_contract() -> None:
    adm = _raw(_ADM_CONFIG)
    dit = _raw(_DIT_CONFIG)
    adm_config = load_config(_ADM_CONFIG)
    dit_config = load_config(_DIT_CONFIG)

    assert isinstance(adm_config, StochaflowConfig)
    assert isinstance(dit_config, StochaflowConfig)
    assert adm["experiment"]["name"] == "afhq_v2_adm_128"
    assert adm["experiment"]["output_dir"] == "outputs/afhq-v2/adm-128"
    assert dit["experiment"]["name"] == "afhq_v2_dit_b8_128"
    assert dit["experiment"]["output_dir"] == "outputs/afhq-v2/dit-b8-128"
    assert adm["extensions"]["plugins"] == ["stochaflow-afhq-v2"]
    assert dit["extensions"] == adm["extensions"]
    for raw, config, verification in (
        (adm, adm_config, "manifest"),
        (dit, dit_config, "full"),
    ):
        assert config.data.name == "class_labeled_image"
        source = raw["data"]["params"]["source"]
        assert source["name"] == "afhq-v2.official"
        assert source["params"] == {"resolution": 128}
        assert source["materialization"] == {
            "cache_root": "./data",
            "policy": "require",
            "verification": verification,
        }
        assert set(raw["data"]["params"]) == {
            "source",
            "partition",
            "image",
            "loader",
        }
        assert raw["data"]["params"]["partition"] == {
            "validation_per_class": 300,
            "seed": "stochaflow-afhq-v2-validation-v1",
        }
        assert raw["training"] == {
            "name": "class_conditional_gaussian_denoising",
            "params": {
                "prediction_type": "v",
                "condition_dropout": 0.1,
            },
        }
        assert raw["metrics"] == _EXPECTED_IMAGE_METRICS
        assert raw["trainer"]["precision"] == "bf16-mixed"
        assert raw["trainer"]["num_epochs"] == 200
        assert raw["trainer"]["early_stopping"]["monitor"] == "valid/loss"
        assert raw["artifacts"]["checkpoint_every"] == 5
        assert "sampling" not in raw
        assert "use_for_sampling" not in raw["ema"]
        assert raw["diagnostics"][0]["name"] == (
            "class_conditional_diffusion_quality"
        )
        assert raw["diagnostics"][0]["params"]["conditions"] == [
            {"class_label": 0, "count": 4},
            {"class_label": 1, "count": 4},
            {"class_label": 2, "count": 4},
        ]
        assert (
            raw["diagnostics"][0]["params"]["cadence"][
                "artifact_every_epochs"
            ]
            == 5
        )

    assert adm["model"]["name"] == "adm_unet"
    assert adm["model"]["params"] == {
        "input_size": 128,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 1, 2, 3, 4],
        "num_res_blocks": 2,
        "attention_resolutions": [32, 16, 8],
        "attention_head_channels": 64,
        "num_classes": 3,
        "dropout": 0.1,
    }
    assert dit["model"] == {
        "name": "dit",
        "params": {
            "input_size": 128,
            "patch_size": 8,
            "in_channels": 3,
            "out_channels": 3,
            "hidden_size": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "num_classes": 3,
        },
    }
    assert adm["data"]["params"]["loader"]["batch_size"] == 8
    assert dit["data"]["params"]["loader"]["batch_size"] == 32
    assert adm["trainer"]["accumulate_grad_batches"] == 4
    assert dit["trainer"]["accumulate_grad_batches"] == 1
    adm_shared = deepcopy(adm)
    dit_shared = deepcopy(dit)
    for config in (adm_shared, dit_shared):
        config.pop("model")
        config["experiment"].pop("name")
        config["experiment"].pop("output_dir")
        config["data"]["params"]["loader"].pop("batch_size")
        config["data"]["params"]["source"]["materialization"].pop(
            "verification"
        )
        config["trainer"].pop("accumulate_grad_batches")
        config["lr_scheduler"]["params"].pop("warmup_steps")
        config["lr_scheduler"]["params"].pop("total_steps")
    assert dit_shared == adm_shared


def test_afhq_learned_range_recipe_uses_live_validation_evaluation() -> None:
    raw = _raw(_LEARNED_RANGE_ADM_CONFIG)
    config = load_config(_LEARNED_RANGE_ADM_CONFIG)

    assert raw["model"]["params"]["out_channels"] == 6
    assert raw["model"]["params"]["channel_multipliers"] == [1, 2, 3, 4]
    assert raw["model"]["params"]["attention_resolutions"] == [32, 16]
    assert raw["training"]["params"] == {
        "prediction_type": "v",
        "variance": {"mode": "learned_range"},
        "condition_dropout": 0.1,
    }
    assert raw["trainer"]["test_after_fit"] is False
    validation = raw["trainer"]["validation_evaluation"]
    assert validation["enabled"] is True
    assert validation["start_epoch"] == 100
    assert validation["every_epochs"] == 10
    assert validation["include_final"] is True
    assert validation["weights"] == "ema"
    assert validation["protocol"] == {
        "id": "afhq-v2-adm-learned-range-v-ddpm100-validation-v1",
        "expected_examples": 900,
        "strict_complete": True,
    }
    profile = validation["evaluation"]["params"]
    assert profile["expected_per_class"] == {
        "cat": 300,
        "dog": 300,
        "wild": 300,
    }
    assert profile["sampling"]["recipe"] == {
        "name": "class_conditional_denoising",
        "contract": {
            "prediction_type": "v",
            "variance": {"mode": "learned_range"},
        },
    }
    assert profile["sampling"]["sampler"] == {
        "name": "ddpm",
        "params": {"num_inference_steps": 100},
    }
    assert profile["sampling"]["num_samples"] == 900
    assert raw["trainer"]["validation_evaluation"]["metrics"][0]["params"][
        "providers"
    ][0]["params"]["subset_size"] == 200
    assert profile["sampling"]["batch_size"] == 15
    providers = validation["metrics"][0]["params"]["providers"]
    assert [provider["name"] for provider in providers] == ["kid", "fid"]
    assert all(provider["params"]["antialias"] is True for provider in providers)
    assert validation["metric_keys"] == [
        "valid/metrics/distribution/aggregate.fid",
        "valid/metrics/distribution/aggregate.kid_mean",
        "valid/metrics/distribution/aggregate.kid_std",
        "valid/metrics/distribution/cat.fid",
        "valid/metrics/distribution/cat.kid_mean",
        "valid/metrics/distribution/cat.kid_std",
        "valid/metrics/distribution/dog.fid",
        "valid/metrics/distribution/dog.kid_mean",
        "valid/metrics/distribution/dog.kid_std",
        "valid/metrics/distribution/wild.fid",
        "valid/metrics/distribution/wild.kid_mean",
        "valid/metrics/distribution/wild.kid_std",
    ]
    monitor = raw["trainer"]["early_stopping"]["monitor"]
    assert monitor == "valid/metrics/distribution/aggregate.fid"
    assert monitor in validation["metric_keys"]
    assert raw["artifacts"]["checkpoint_every"] == 50
    assert raw["trainer"]["device"] == "cuda"
    diagnostic = raw["diagnostics"][0]["params"]
    assert diagnostic["cadence"] == {
        "step_every": 1000,
        "artifact_every_epochs": 10,
    }
    assert [sampler["id"] for sampler in diagnostic["samplers"]] == [
        "ddpm_100",
        "ddim_50",
    ]
    assert all(
        sampler["trajectory"]["enabled"] is False
        for sampler in diagnostic["samplers"]
    )
    assert [
        provider["name"]
        for provider in diagnostic["providers"]["sampler_artifacts"]
    ] == ["sample_grid"]
    assert config.trainer.validation_evaluation.enabled
    model = ADMUNet(**raw["model"]["params"])
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        100_351_366
    )


def test_afhq_learned_range_official_test_matches_frozen_protocol() -> None:
    raw = _raw(_LEARNED_RANGE_OFFICIAL_TEST_CONFIG)
    config = load_evaluation_config(_LEARNED_RANGE_OFFICIAL_TEST_CONFIG)

    assert config.purpose == "final_test"
    assert config.data.split == "test"
    assert isinstance(config.subject, CheckpointSubjectConfig)
    assert config.subject.weights == "ema"
    profile = raw["evaluation"]["params"]
    assert profile["expected_per_class"] == {
        "cat": 493,
        "dog": 491,
        "wild": 483,
    }
    assert profile["sampling"]["recipe"]["contract"] == {
        "prediction_type": "v",
        "variance": {"mode": "learned_range"},
    }
    assert profile["sampling"]["sampler"] == {
        "name": "ddpm",
        "params": {"num_inference_steps": 100},
    }
    assert profile["sampling"]["options"]["weights"] == "ema"
    assert profile["sampling"]["options"]["guidance_scale"] == 2.0
    assert profile["sampling"]["num_samples"] == 1467
    assert profile["sampling"]["batch_size"] == 15
    assert raw["metrics"][0]["params"]["providers"][0]["params"][
        "subset_size"
    ] == 200
    assert raw["protocol"]["expected_examples"] == 1467
    providers = raw["metrics"][0]["params"]["providers"]
    assert [provider["name"] for provider in providers] == ["kid", "fid"]
    assert all(provider["params"]["antialias"] is True for provider in providers)


@pytest.mark.parametrize(
    (
        "config_path",
        "expected_microbatches",
        "expected_updates",
        "expected_total_steps",
        "expected_warmup_steps",
    ),
    [
        (_ADM_CONFIG, 1_679, 420, 84_000, 1_680),
        (_LEARNED_RANGE_ADM_CONFIG, 1_679, 420, 84_000, 1_680),
        (_DIT_CONFIG, 419, 419, 83_800, 1_676),
    ],
)
def test_afhq_step_schedule_is_derived_from_the_locked_dataset_counts(
    config_path: Path,
    expected_microbatches: int,
    expected_updates: int,
    expected_total_steps: int,
    expected_warmup_steps: int,
) -> None:
    lock = _raw(_LOCK)
    raw = _raw(config_path)
    source_train = lock["dataset_contract"]["source_splits"]["train"]
    validation_per_class = raw["data"]["params"]["partition"][
        "validation_per_class"
    ]
    num_classes = len(lock["dataset_contract"]["class_mapping"])
    prepared_train = source_train - validation_per_class * num_classes
    batch_size = raw["data"]["params"]["loader"]["batch_size"]
    accumulation = raw["trainer"]["accumulate_grad_batches"]
    epochs = raw["trainer"]["num_epochs"]

    microbatches = prepared_train // batch_size
    updates = math.ceil(microbatches / accumulation)
    total_steps = updates * epochs
    warmup_steps = round(total_steps * 0.02)

    assert prepared_train == 13_436
    assert microbatches == expected_microbatches
    assert updates == expected_updates
    assert total_steps == expected_total_steps
    assert warmup_steps == expected_warmup_steps
    assert raw["lr_scheduler"]["params"]["total_steps"] == total_steps
    assert raw["lr_scheduler"]["params"]["warmup_steps"] == warmup_steps


def test_afhq_smoke_config_is_bounded_and_uses_the_real_data_contract() -> None:
    raw = _raw(_SMOKE_CONFIG)
    config = load_config(_SMOKE_CONFIG)

    assert config.data.name == "class_labeled_image"
    assert (
        raw["data"]["params"]["source"]["materialization"]["policy"]
        == "require"
    )
    assert raw["data"]["params"]["loader"] == {
        "batch_size": 2,
        "num_workers": 0,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "steps_per_epoch": 4,
    }
    assert raw["trainer"]["precision"] == "fp32"
    assert raw["trainer"]["accumulate_grad_batches"] == 2
    assert raw["trainer"]["num_epochs"] == 1
    assert raw["trainer"]["early_stopping"]["monitor"] == "valid/loss"
    assert raw["metrics"] == _EXPECTED_IMAGE_METRICS
    assert raw["lr_scheduler"]["params"]["total_steps"] == 2
    assert raw["model"] == {
        "name": "adm_unet",
        "params": {
            "input_size": 128,
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 32,
            "channel_multipliers": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [64],
            "attention_head_channels": 32,
            "num_classes": 3,
            "dropout": 0.0,
        },
    }
    assert raw["diagnostics"][0]["name"] == (
        "class_conditional_diffusion_quality"
    )
    assert "sampling" not in raw
    assert "use_for_sampling" not in raw["ema"]


def test_afhq_sampling_profile_is_complete_and_checkpoint_driven() -> None:
    raw = _raw(_SAMPLING_CONFIG)

    assert set(raw) == {"sample"}
    assert set(raw["sample"]) == {
        "sampler",
        "options",
        "shape",
        "num_samples",
        "batch_size",
        "seed",
        "writers",
    }
    assert "builder" not in raw["sample"]
    assert raw["sample"]["sampler"]["name"] == "ddim"
    assert raw["sample"]["options"]["guidance_scale"] == 2.0
    assert "prediction_type" not in raw["sample"]["options"]

    resolved = load_sample_config(_SAMPLING_CONFIG).sample
    assert resolved.num_samples == 36
    assert resolved.batch_size == 12
    assert resolved.seed == 20260726
    assert resolved.options["conditions"] == [
        {"class_label": 0, "count": 12},
        {"class_label": 1, "count": 12},
        {"class_label": 2, "count": 12},
    ]
    assert resolved.writers[1].params["grid_nrow"] == 12


def test_afhq_readme_sampling_request_is_explicit_and_reproducible() -> None:
    resolved = load_sample_config(_README_SAMPLING_CONFIG).sample

    assert resolved.num_samples == 36
    assert resolved.batch_size == 12
    assert resolved.seed == 20260726
    assert resolved.shape == [3, 128, 128]
    assert resolved.options == {
        "weights": "ema",
        "clip_denoised": True,
        "guidance_scale": 2.0,
        "conditions": [
            {"class_label": 0, "count": 12},
            {"class_label": 1, "count": 12},
            {"class_label": 2, "count": 12},
        ],
        "trajectory": {
            "enabled": False,
            "every_steps": 5,
        },
    }
    assert resolved.writers[1].name == "image"
    assert resolved.writers[1].params == {
        "grid_nrow": 6,
        "gif_fps": 8,
        "denormalize": True,
    }


def test_afhq_capacity_tool_runs_real_comparable_training_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _capacity_module()
    observed: dict[str, Any] = {}

    class RecordingBindings:
        """Provide the serialized artifact binding returned by DataBuilder."""

        def to_dict(self) -> dict[str, Any]:
            return {
                "schema_version": 2,
                "bindings": [{"id": "source"}],
            }

    def fake_profile_trials(
        config: StochaflowConfig,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], RecordingBindings]:
        observed["config"] = config
        observed.update(kwargs)
        return (
            [
                {
                    "status": "ok",
                    "micro_batch_size": 4,
                    "precision": "fp32",
                    "measurement": {
                        "images_per_second": 10.0,
                        "peak_allocated_vram_bytes": 1_000,
                        "peak_reserved_vram_bytes": 2_000,
                    },
                },
                {
                    "status": "ok",
                    "micro_batch_size": 4,
                    "precision": "bf16-mixed",
                    "measurement": {
                        "images_per_second": 15.0,
                        "peak_allocated_vram_bytes": 700,
                        "peak_reserved_vram_bytes": 1_600,
                    },
                },
            ],
            RecordingBindings(),
        )

    monkeypatch.setattr(module, "_activate_showcase_extension", lambda: None)
    monkeypatch.setattr(module, "_profile_trials", fake_profile_trials)
    report = module.capacity_report(
        _ADM_CONFIG,
        micro_batches=[4],
        device_name="cpu",
        run_root=tmp_path / "capacity",
    )

    assert report["schema_version"] == 3
    assert report["model"] == "adm_unet"
    assert (
        90_000_000
        <= report["primary_model_parameter_count"]
        <= 120_000_000
    )
    assert report["micro_batches"] == [4]
    assert report["precisions"] == ["fp32", "bf16-mixed"]
    assert report["warmup_updates"] == 5
    assert report["measured_updates"] == 25
    assert report["target_effective_batch_size"] == 32
    assert observed["warmup_updates"] == 5
    assert observed["measured_updates"] == 25
    assert observed["device_name"] == "cpu"
    assert report["seed"] == 20260726
    assert report["data_artifact_bindings"] == {
        "schema_version": 2,
        "bindings": [{"id": "source"}],
    }
    assert set(report["code_identity"]) == {"core", "extension"}
    assert report["precision_comparisons"] == [
        {
            "micro_batch_size": 4,
            "bf16_vs_fp32_images_per_second_delta": 5.0,
            "bf16_vs_fp32_throughput_ratio": 1.5,
            "bf16_vs_fp32_peak_allocated_vram_delta_bytes": -300,
            "bf16_vs_fp32_peak_reserved_vram_delta_bytes": -400,
        }
    ]


def test_afhq_capacity_tool_enforces_real_measurement_floor() -> None:
    module = _capacity_module()

    with pytest.raises(
        ValueError,
        match="measured_updates must be at least 25",
    ):
        module.capacity_report(
            _ADM_CONFIG,
            measured_updates=24,
            device_name="cpu",
        )


def test_afhq_capacity_rejects_device_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _capacity_module()
    calls: list[str] = []

    def reject_device(device: object) -> None:
        calls.append(f"device:{device}")
        raise ValueError("invalid execution device")

    monkeypatch.setattr(module, "validate_execution_device", reject_device)
    monkeypatch.setattr(
        module,
        "_activate_showcase_extension",
        lambda: calls.append("activate"),
    )
    monkeypatch.setattr(
        module,
        "build_model",
        lambda config: calls.append(f"model:{config}"),
    )
    monkeypatch.setattr(
        module,
        "_profile_trials",
        lambda *args, **kwargs: calls.append("data"),
    )

    with pytest.raises(ValueError, match="invalid execution device"):
        module.capacity_report(
            _ADM_CONFIG,
            micro_batches=[4],
            device_name="cuda:3",
        )

    assert calls == ["device:cuda:3"]


def test_afhq_capacity_all_unsupported_skips_model_data_and_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _capacity_module()
    provenance = _showcase_tool_module("capacity_provenance")
    calls: list[str] = []
    run_root = tmp_path / "capacity"

    monkeypatch.setattr(
        module,
        "validate_execution_device",
        lambda device: calls.append(f"device:{device}"),
    )

    def reject_precision(precision: str, device: object) -> None:
        calls.append(f"precision:{precision}:{device}")
        raise ValueError(f"{precision} unsupported")

    monkeypatch.setattr(
        module,
        "validate_precision_support",
        reject_precision,
    )
    monkeypatch.setattr(
        module,
        "_activate_showcase_extension",
        lambda: pytest.fail("extension activation must be skipped"),
    )
    monkeypatch.setattr(
        module,
        "build_model",
        lambda config: pytest.fail("meta model must be skipped"),
    )
    monkeypatch.setattr(
        module,
        "_profile_trials",
        lambda *args, **kwargs: pytest.fail("DataBuilder must be skipped"),
    )
    monkeypatch.setattr(module, "code_identity", lambda: {"test": True})

    report = module.capacity_report(
        _ADM_CONFIG,
        micro_batches=[4],
        device_name="cpu",
        run_root=run_root,
    )

    assert calls == [
        "device:cpu",
        "precision:fp32:cpu",
        "precision:bf16-mixed:cpu",
    ]
    assert report["primary_model_parameter_count"] is None
    assert report["data_artifact_bindings"] is None
    assert [trial["status"] for trial in report["trials"]] == [
        "unsupported_precision",
        "unsupported_precision",
    ]
    assert all(
        trial["resolved_config_sha256"]
        == provenance.canonical_sha256(trial["resolved_config"])
        for trial in report["trials"]
    )
    assert not run_root.exists()


def test_afhq_capacity_partial_support_builds_one_data_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _capacity_module()
    provenance = _showcase_tool_module("capacity_provenance")
    base = load_config(_ADM_CONFIG)
    bindings = object()
    build_calls: list[object] = []
    profile_calls: list[str] = []

    class RecordingLoaders:
        def __init__(self) -> None:
            self.train = [object()]
            self.artifact_bindings = bindings

    def fake_build_data_loaders(
        config: object,
        *,
        seed: int,
    ) -> RecordingLoaders:
        build_calls.extend((config, seed))
        return RecordingLoaders()

    def fake_profile_trial(
        config: StochaflowConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        profile_calls.append(config.trainer.precision)
        return {
            "status": "ok",
            "micro_batch_size": 4,
            "precision": config.trainer.precision,
            **provenance.trial_config_identity(config),
        }

    monkeypatch.setattr(
        module,
        "build_data_loaders",
        fake_build_data_loaders,
    )
    monkeypatch.setattr(
        module,
        "_profile_precision_trial",
        fake_profile_trial,
    )
    monkeypatch.setattr(module.gc, "collect", lambda: 0)

    trials, actual_bindings = module._profile_trials(
        base,
        micro_batches=[4],
        precisions=["fp32", "bf16-mixed"],
        precision_errors={
            "fp32": None,
            "bf16-mixed": ValueError("BF16 unavailable"),
        },
        warmup_updates=5,
        measured_updates=25,
        device_name="cpu",
        run_root=tmp_path / "capacity",
    )

    assert len(build_calls) == 2
    assert build_calls[1] == 20260726
    assert profile_calls == ["fp32"]
    assert actual_bindings is bindings
    assert [trial["status"] for trial in trials] == [
        "ok",
        "unsupported_precision",
    ]
    assert all("resolved_config" in trial for trial in trials)


def test_afhq_capacity_measurement_closes_training_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _capacity_module()
    capacity_config = _showcase_tool_module("capacity_config")
    provenance = _showcase_tool_module("capacity_provenance")
    config = capacity_config.trial_config(
        load_config(_ADM_CONFIG),
        micro_batch=4,
        precision="fp32",
        device_name="cpu",
        output_dir=tmp_path / "trial",
    )
    train_calls: list[dict[str, Any]] = []

    class RecordingTrainer:
        device = torch.device("cpu")

        def train_epoch(
            self,
            loader: object,
            **kwargs: Any,
        ) -> dict[str, float]:
            del loader
            train_calls.append(dict(kwargs))
            updates = int(kwargs["max_optimizer_steps"])
            accumulation = config.trainer.accumulate_grad_batches
            return {
                "loss": 1.0,
                "micro_batches": float(updates * accumulation),
                "optimizer_steps": float(updates),
                "skipped_optimizer_steps": 0.0,
                "optimizer_steps_per_second": 2.0,
                "data_wait_seconds": 1.0,
                "compute_seconds": 4.0,
                "duration_seconds": 5.0,
                "forward_seconds": 1.0,
                "backward_seconds": 2.0,
                "optimizer_seconds": 1.0,
                "non_finite_loss_count": 0.0,
                "non_finite_gradient_count": 0.0,
            }

    class RecordingLogger:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class RecordingTraining:
        def __init__(self) -> None:
            self.trainer = RecordingTrainer()
            self.logger = RecordingLogger()

    training = RecordingTraining()
    collections: list[bool] = []
    monkeypatch.setattr(
        module,
        "build_training_components",
        lambda resolved: training,
    )
    monkeypatch.setattr(
        module.gc,
        "collect",
        lambda: collections.append(True) or 0,
    )

    report = module._profile_precision_trial(
        config,
        train_loader=[object()],
        micro_batch=4,
        precision="fp32",
        warmup_updates=5,
        measured_updates=25,
    )

    assert training.logger.closed is True
    assert collections == [True]
    assert [call["max_optimizer_steps"] for call in train_calls] == [5, 25]
    assert all(call["profile_phases"] is True for call in train_calls)
    assert report["measurement"]["successful_optimizer_updates"] == 25
    assert report["measurement"]["processed_images"] == 800
    assert report["measurement"]["data_wait_compute_ratio"] == 0.25
    assert report["resolved_config_sha256"] == provenance.canonical_sha256(
        report["resolved_config"]
    )
    assert report["output_dir"] == str((tmp_path / "trial").resolve())


def test_afhq_capacity_trial_disables_live_validation_evaluation(
    tmp_path: Path,
) -> None:
    capacity_config = _showcase_tool_module("capacity_config")

    config = capacity_config.trial_config(
        load_config(_LEARNED_RANGE_ADM_CONFIG),
        micro_batch=4,
        precision="bf16-mixed",
        device_name="cuda",
        output_dir=tmp_path / "trial",
    )

    assert config.trainer.num_epochs == 1
    assert config.trainer.validation_evaluation.enabled is False


@pytest.mark.parametrize(
    ("micro_batch", "expected_accumulation"),
    [(4, 8), (6, 5), (8, 4)],
)
def test_afhq_capacity_keeps_effective_batch_near_32(
    micro_batch: int,
    expected_accumulation: int,
) -> None:
    capacity_config = _showcase_tool_module("capacity_config")

    assert (
        capacity_config.accumulation_for_micro_batch(micro_batch)
        == expected_accumulation
    )
