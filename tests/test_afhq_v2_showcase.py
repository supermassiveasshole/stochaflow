from __future__ import annotations

import importlib
import math
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
import yaml

from stochaflow.data import IMAGE_DATA_SOURCES
from stochaflow.utils.config import (
    StochaflowConfig,
    apply_sample_request,
    load_config,
    parse_sample_request,
)
from stochaflow.utils.registry import REGISTRIES

_ROOT = Path(__file__).resolve().parents[1]
_SHOWCASE = _ROOT / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _SHOWCASE / "src"
_ADM_CONFIG = _SHOWCASE / "experiments" / "production" / "train-adm-128.yaml"
_DIT_CONFIG = _SHOWCASE / "experiments" / "production" / "train-dit-128.yaml"
_SMOKE_CONFIG = _SHOWCASE / "experiments" / "smoke" / "train-adm-128.yaml"
_SAMPLING_CONFIG = (
    _SHOWCASE / "experiments" / "sampling" / "ddim50-cfg2.yaml"
)
_LOCK = (
    _SHOWCASE
    / "src"
    / "stochaflow_afhq_v2"
    / "resources"
    / "afhq-v2.lock.yaml"
)


def _raw(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _condition_allocations(raw: dict[str, Any]) -> list[dict[str, int]]:
    return raw["sampling"]["options"]["conditions"]


def _showcase_tool_module(name: str) -> ModuleType:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.invalidate_caches()
    return importlib.import_module(f"stochaflow_afhq_v2.tools.{name}")


def _capacity_module() -> ModuleType:
    return _showcase_tool_module("capacity")


def test_afhq_showcase_registers_only_the_source_extension() -> None:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")
    declaration = tomllib.loads(
        (_SHOWCASE / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "class_labeled_image" in REGISTRIES.data_builders.names()
    assert "afhq-v2.class-images" not in REGISTRIES.data_builders.names()
    assert "afhq-v2.official" in IMAGE_DATA_SOURCES.names()
    assert (_SHOWCASE / "uv.lock").is_file()
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
        "stochaflow-afhq-v2-evaluate": (
            "stochaflow_afhq_v2.tools.evaluate:main"
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
        "stochaflow[quality]==0.1.0"
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
    assert not (package / "stochaflow_ext" / "builder.py").exists()
    assert not (package / "stochaflow_ext" / "batching.py").exists()
    assert not (package / "stochaflow_ext" / "partitioning.py").exists()
    assert not (package / "stochaflow_ext" / "config.py").exists()
    assert not (package / "stochaflow_ext" / "data.py").exists()
    assert not (package / "stochaflow_ext" / "dataset.py").exists()
    assert not (_SHOWCASE / "prepare.py").exists()
    assert not (_SHOWCASE / "experiments" / "ddpm_128.yaml").exists()


def test_afhq_production_configs_parse_and_share_one_pipeline_contract() -> None:
    adm = _raw(_ADM_CONFIG)
    dit = _raw(_DIT_CONFIG)
    adm_config = load_config(_ADM_CONFIG)
    dit_config = load_config(_DIT_CONFIG)

    assert isinstance(adm_config, StochaflowConfig)
    assert isinstance(dit_config, StochaflowConfig)
    assert adm["extensions"]["plugins"] == ["stochaflow-afhq-v2"]
    assert dit["extensions"] == adm["extensions"]
    for raw, config in ((adm, adm_config), (dit, dit_config)):
        assert config.data.name == "class_labeled_image"
        source = raw["data"]["params"]["source"]
        assert source["name"] == "afhq-v2.official"
        assert source["params"] == {"resolution": 128}
        assert source["materialization"] == {
            "cache_root": "./data",
            "policy": "require",
            "verification": "full",
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
        assert raw["trainer"]["precision"] == "bf16-mixed"
        assert raw["trainer"]["accumulate_grad_batches"] == 4
        assert raw["trainer"]["num_epochs"] == 200
        assert raw["artifacts"]["checkpoint_every"] == 5
        assert raw["sampling"]["run_after_training"] is True
        assert raw["sampling"]["sampler"]["name"] == "ddim"
        assert "builder" not in raw["sampling"]
        assert "prediction_type" not in raw["sampling"]["options"]
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
        assert _condition_allocations(raw) == [
            {"class_label": 0, "count": 12},
            {"class_label": 1, "count": 12},
            {"class_label": 2, "count": 12},
        ]

    assert adm["model"]["name"] == "adm_unet"
    assert adm["model"]["params"]["num_classes"] == 3
    assert dit["model"] == {
        "name": "dit",
        "params": {
            "input_size": 128,
            "patch_size": 8,
            "in_channels": 3,
            "out_channels": 3,
            "hidden_size": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "num_classes": 3,
        },
    }
    for section in (
        "data",
        "process",
        "training",
        "objective",
        "optimizer",
        "lr_scheduler",
        "ema",
        "sampling",
        "diagnostics",
        "trainer",
        "artifacts",
    ):
        assert dit[section] == adm[section]


def test_afhq_step_schedule_is_derived_from_the_locked_dataset_counts() -> None:
    lock = _raw(_LOCK)
    raw = _raw(_ADM_CONFIG)
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
    assert microbatches == 1_679
    assert updates == 420
    assert total_steps == 84_000
    assert warmup_steps == 1_680
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
    assert raw["lr_scheduler"]["params"]["total_steps"] == 2
    assert raw["diagnostics"][0]["name"] == (
        "class_conditional_diffusion_quality"
    )
    assert _condition_allocations(raw) == [
        {"class_label": 0, "count": 1},
        {"class_label": 1, "count": 1},
        {"class_label": 2, "count": 1},
    ]


def test_afhq_sampling_request_is_minimal_and_checkpoint_driven() -> None:
    raw = _raw(_SAMPLING_CONFIG)

    assert set(raw) == {"sampling"}
    assert set(raw["sampling"]) == {"sampler", "options"}
    assert "run_after_training" not in raw["sampling"]
    assert "builder" not in raw["sampling"]
    assert raw["sampling"]["sampler"]["name"] == "ddim"
    assert raw["sampling"]["options"]["guidance_scale"] == 2.0
    assert "prediction_type" not in raw["sampling"]["options"]

    defaults = load_config(_ADM_CONFIG).sampling
    resolved = apply_sample_request(
        defaults,
        parse_sample_request(raw["sampling"]),
    )
    assert resolved.num_samples == 36
    assert resolved.batch_size == 12
    assert resolved.seed == 20260726
    assert resolved.options["conditions"] == [
        {"class_label": 0, "count": 12},
        {"class_label": 1, "count": 12},
        {"class_label": 2, "count": 12},
    ]
    assert resolved.writers[1].params["grid_nrow"] == 12


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
