from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from stochaflow.data.recipe_config import ImageDataBuilderConfig
from stochaflow.utils.config import (
    ConfigError,
    StochaflowConfig,
    apply_sample_request,
    coerce_config_section,
    load_config,
    parse_sample_request,
)

_ROOT = Path(__file__).resolve().parents[1]
_SHOWCASE = _ROOT / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _SHOWCASE / "src"
_RESEARCH = (
    _SHOWCASE
    / "experiments"
    / "research"
    / "p2-afhq-v2-dog-256"
)
_BASE_CONFIG = _RESEARCH / "train-base.yaml"
_P2_OVERRIDE = _RESEARCH / "p2-loss-weighting.yaml"
_DDPM_250_PROFILE = _RESEARCH / "sample-ddpm-250.yaml"
_DDPM_1000_PROFILE = _RESEARCH / "sample-ddpm-1000.yaml"


def _raw(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _benchmark_module() -> ModuleType:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.invalidate_caches()
    return importlib.import_module(
        "stochaflow_afhq_v2.tools.benchmark_config"
    )


def _without_weighting(raw: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(raw)
    result["training"]["params"].pop("loss_weighting")
    return result


def test_afhq_dog_research_base_freezes_official_compatible_budget() -> None:
    module = _benchmark_module()
    resolution = module.resolve_benchmark_training_config(_BASE_CONFIG)
    config = resolution.config
    raw = _raw(_BASE_CONFIG)

    assert isinstance(config, StochaflowConfig)
    assert raw["experiment"] == {
        "name": "p2_compatible_afhq_v2_dog_256",
        "seed": 20260730,
        "output_dir": "outputs/afhq-v2/research/p2-dog-256",
    }
    assert raw["data"]["name"] == "image"
    assert raw["data"]["params"]["source"] == {
        "name": "afhq-v2.dog",
        "materialization": {
            "cache_root": "./data",
            "policy": "require",
            "verification": "full",
        },
        "params": {"resolution": 256},
    }
    assert raw["data"]["params"]["partition"] == {"mode": "none"}
    assert raw["data"]["params"]["image"] == {
        "size": [256, 256],
        "channels": 3,
        "normalize": True,
        "random_horizontal_flip": True,
    }
    data_config = coerce_config_section(
        ImageDataBuilderConfig,
        raw["data"]["params"],
        "data.params",
    )
    data_config.validate()

    loader = raw["data"]["params"]["loader"]
    assert loader["batch_size"] == 8
    assert loader["steps_per_epoch"] == 500
    assert raw["trainer"]["num_epochs"] == 600
    assert raw["trainer"]["accumulate_grad_batches"] == 1
    assert raw["trainer"]["precision"] == "fp32"
    updates = (
        loader["steps_per_epoch"]
        * raw["trainer"]["num_epochs"]
        // raw["trainer"]["accumulate_grad_batches"]
    )
    assert updates == 300_000
    assert updates * loader["batch_size"] == 2_400_000

    assert raw["model"] == {
        "name": "adm_unet",
        "params": {
            "input_size": 256,
            "in_channels": 3,
            "out_channels": 6,
            "base_channels": 128,
            "channel_multipliers": [1, 1, 2, 2, 4, 4],
            "num_res_blocks": 1,
            "attention_resolutions": [16],
            "attention_head_channels": 64,
            "num_classes": None,
            "dropout": 0.1,
        },
    }
    assert raw["process"] == {
        "name": "discrete_gaussian",
        "params": {
            "schedule": {
                "name": "linear_beta",
                "params": {
                    "num_timesteps": 1000,
                    "beta_start": 1.0e-4,
                    "beta_end": 0.02,
                },
            }
        },
    }
    assert raw["training"] == {
        "name": "gaussian_denoising",
        "params": {
            "prediction_type": "epsilon",
            "variance": {
                "mode": "learned_range",
                "loss": "rescaled_variational_bound",
            },
            "loss_weighting": {"name": "constant"},
        },
    }
    assert raw["objective"] == {
        "name": "mse",
        "params": {"reduction": "mean"},
    }
    assert raw["optimizer"] == {
        "name": "torch.optim.AdamW",
        "params": {
            "lr": 2.0e-5,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        },
    }
    assert "lr_scheduler" not in raw
    assert raw["ema"] == {
        "enabled": True,
        "decay": 0.9999,
        "update_after_step": 0,
        "update_every": 1,
        "use_for_sampling": True,
    }
    assert raw["sampling"]["run_after_training"] is False
    assert raw["sampling"]["shape"] == [3, 256, 256]
    assert raw["sampling"]["num_samples"] == 50_000
    assert raw["sampling"]["batch_size"] == 8
    assert raw["sampling"]["seed"] == 20260731


def test_constant_and_p2_resolutions_differ_only_in_weighting() -> None:
    module = _benchmark_module()
    constant = module.resolve_benchmark_training_config(_BASE_CONFIG)
    p2 = module.resolve_benchmark_training_config(
        _BASE_CONFIG,
        override_path=_P2_OVERRIDE,
    )
    constant_raw = constant.config.to_dict()
    p2_raw = p2.config.to_dict()

    assert constant_raw["training"]["params"]["loss_weighting"] == {
        "name": "constant"
    }
    assert p2_raw["training"]["params"]["loss_weighting"] == {
        "name": "p2",
        "k": 1.0,
        "gamma": 1.0,
    }
    assert _without_weighting(constant_raw) == _without_weighting(p2_raw)
    assert constant_raw["experiment"]["seed"] == p2_raw["experiment"]["seed"]
    assert constant_raw["data"] == p2_raw["data"]
    assert constant_raw["model"] == p2_raw["model"]
    assert constant_raw["process"] == p2_raw["process"]
    assert constant_raw["optimizer"] == p2_raw["optimizer"]
    assert constant_raw["ema"] == p2_raw["ema"]


def test_benchmark_resolution_records_source_and_effective_provenance() -> None:
    module = _benchmark_module()
    constant = module.resolve_benchmark_training_config(_BASE_CONFIG)
    p2 = module.resolve_benchmark_training_config(
        _BASE_CONFIG,
        override_path=_P2_OVERRIDE,
    )

    assert constant.provenance.schema_version == 1
    assert constant.provenance.variant == "constant"
    assert constant.provenance.changed_paths == ()
    assert [source.role for source in constant.provenance.sources] == ["base"]
    assert p2.provenance.variant == "p2"
    assert p2.provenance.changed_paths == (
        "training.params.loss_weighting",
    )
    assert [source.role for source in p2.provenance.sources] == [
        "base",
        "override",
    ]
    assert (
        constant.provenance.sources[0]
        == p2.provenance.sources[0]
    )
    assert p2.provenance.sources[1].sha256 == hashlib.sha256(
        _P2_OVERRIDE.read_bytes()
    ).hexdigest()
    assert constant.provenance.resolved_config_sha256 == (
        module.canonical_config_sha256(constant.config)
    )
    assert p2.provenance.resolved_config_sha256 == (
        module.canonical_config_sha256(p2.config)
    )
    assert (
        constant.provenance.resolved_config_sha256
        != p2.provenance.resolved_config_sha256
    )
    serialized = p2.provenance.to_dict()
    assert serialized["sources"][0]["path"] == (
        "experiments/research/p2-afhq-v2-dog-256/train-base.yaml"
    )
    assert serialized["sources"][1]["path"] == (
        "experiments/research/p2-afhq-v2-dog-256/"
        "p2-loss-weighting.yaml"
    )


@pytest.mark.parametrize(
    "override",
    [
        {
            "training": {
                "params": {
                    "loss_weighting": {
                        "name": "p2",
                        "k": 1.0,
                        "gamma": 1.0,
                    }
                }
            },
            "trainer": {"precision": "bf16-mixed"},
        },
        {
            "training": {
                "params": {
                    "prediction_type": "v",
                    "loss_weighting": {
                        "name": "p2",
                        "k": 1.0,
                        "gamma": 1.0,
                    },
                }
            }
        },
        {
            "training": {
                "params": {
                    "loss_weighting": {
                        "name": "p2",
                        "k": 1.0,
                        "gamma": 0.5,
                    }
                }
            }
        },
    ],
)
def test_benchmark_override_fails_closed_outside_official_p2_leaf(
    tmp_path: Path,
    override: dict[str, Any],
) -> None:
    module = _benchmark_module()
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(override), encoding="utf-8")

    with pytest.raises(
        ConfigError,
        match=r"may change only|gamma must be 1\.0",
    ):
        module.resolve_benchmark_training_config(
            _BASE_CONFIG,
            override_path=path,
        )


def test_ddpm_research_profiles_are_training_free_and_checkpoint_driven() -> None:
    module = _benchmark_module()
    defaults = module.resolve_benchmark_training_config(
        _BASE_CONFIG
    ).config.sampling
    resolved_by_steps: dict[int, Any] = {}

    for steps, path in (
        (250, _DDPM_250_PROFILE),
        (1000, _DDPM_1000_PROFILE),
    ):
        raw = _raw(path)
        assert set(raw) == {"sampling"}
        assert set(raw["sampling"]) == {
            "sampler",
            "options",
            "seed",
            "writers",
        }
        assert raw["sampling"]["sampler"] == {
            "name": "ddpm",
            "params": {"num_inference_steps": steps},
        }
        assert raw["sampling"]["options"] == {"weights": "ema"}
        assert raw["sampling"]["seed"] == 20260731
        assert raw["sampling"]["writers"] == [
            {"name": "tensor", "params": {}}
        ]
        assert "training" not in raw
        assert "loss_weighting" not in path.read_text(encoding="utf-8")
        resolved = apply_sample_request(
            defaults,
            parse_sample_request(raw["sampling"]),
        )
        assert resolved.sampler is not None
        assert resolved.sampler.name == "ddpm"
        assert resolved.sampler.params == {"num_inference_steps": steps}
        assert resolved.options == {
            "weights": "ema",
            "clip_denoised": True,
            "trajectory": {"enabled": False, "every_steps": 1},
        }
        assert resolved.shape == [3, 256, 256]
        assert resolved.num_samples == 50_000
        assert resolved.batch_size == 8
        assert resolved.seed == 20260731
        assert [(writer.name, writer.params) for writer in resolved.writers] == [
            ("tensor", {})
        ]
        resolved_by_steps[steps] = resolved

    ddpm_250 = deepcopy(resolved_by_steps[250])
    ddpm_1000 = deepcopy(resolved_by_steps[1000])
    assert ddpm_250.sampler is not None
    assert ddpm_1000.sampler is not None
    ddpm_250.sampler.params["num_inference_steps"] = 1000
    assert ddpm_250 == ddpm_1000


@pytest.mark.parametrize(
    ("variant", "expected_weighting", "expected_source_roles"),
    [
        ("constant", {"name": "constant"}, ["base"]),
        (
            "p2",
            {"name": "p2", "k": 1.0, "gamma": 1.0},
            ["base", "override"],
        ),
    ],
)
def test_benchmark_cli_writes_trainable_yaml_and_provenance(
    tmp_path: Path,
    variant: str,
    expected_weighting: dict[str, Any],
    expected_source_roles: list[str],
) -> None:
    output = tmp_path / variant / "resolved.yaml"
    provenance = tmp_path / variant / "provenance.json"
    environment = os.environ.copy()
    python_path = [str(_EXAMPLE_SRC), str(_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stochaflow_afhq_v2.tools.benchmark_config",
            "--variant",
            variant,
            "--output",
            str(output),
            "--provenance",
            str(provenance),
        ],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "resolved_config_sha256:" in result.stdout
    resolved = load_config(output)
    assert resolved.training.params["loss_weighting"] == expected_weighting
    raw_provenance = json.loads(provenance.read_text(encoding="utf-8"))
    assert raw_provenance["variant"] == variant
    assert [
        source["role"] for source in raw_provenance["sources"]
    ] == expected_source_roles
    module = _benchmark_module()
    assert raw_provenance["resolved_config_sha256"] == (
        module.canonical_config_sha256(resolved)
    )


def test_benchmark_cli_validation_failure_writes_no_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _benchmark_module()
    output = tmp_path / "result" / "resolved.yaml"
    provenance = tmp_path / "result" / "provenance.json"

    def reject_variant(variant: str) -> Any:
        raise ConfigError(f"invalid {variant} benchmark authority")

    monkeypatch.setattr(
        module,
        "resolve_benchmark_variant",
        reject_variant,
    )

    with pytest.raises(SystemExit) as raised:
        module.main(
            [
                "--variant",
                "p2",
                "--output",
                str(output),
                "--provenance",
                str(provenance),
            ]
        )

    assert raised.value.code == 2
    assert not output.exists()
    assert not provenance.exists()


def test_benchmark_writer_rolls_back_both_outputs_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _benchmark_module()
    resolution = module.resolve_benchmark_variant(
        "p2",
        base_path=_BASE_CONFIG,
        p2_override_path=_P2_OVERRIDE,
    )
    output = tmp_path / "resolved.yaml"
    provenance = tmp_path / "provenance.json"
    real_link = os.link
    calls = 0

    def fail_second_link(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated provenance commit failure")
        real_link(source, target)

    monkeypatch.setattr(module.os, "link", fail_second_link)

    with pytest.raises(OSError, match="simulated provenance commit failure"):
        module.write_benchmark_resolution(
            resolution,
            output_path=output,
            provenance_path=provenance,
        )

    assert not output.exists()
    assert not provenance.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_benchmark_writer_refuses_existing_target_before_other_output(
    tmp_path: Path,
) -> None:
    module = _benchmark_module()
    resolution = module.resolve_benchmark_variant(
        "constant",
        base_path=_BASE_CONFIG,
    )
    output = tmp_path / "resolved.yaml"
    provenance = tmp_path / "provenance.json"
    provenance.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        module.write_benchmark_resolution(
            resolution,
            output_path=output,
            provenance_path=provenance,
        )

    assert not output.exists()
    assert provenance.read_text(encoding="utf-8") == "keep"


def test_benchmark_writer_never_overwrites_target_created_during_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _benchmark_module()
    resolution = module.resolve_benchmark_variant(
        "constant",
        base_path=_BASE_CONFIG,
    )
    output = tmp_path / "resolved.yaml"
    provenance = tmp_path / "provenance.json"
    real_link = os.link
    calls = 0

    def create_racing_target_then_link(
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(target).write_text("concurrent owner", encoding="utf-8")
        real_link(source, target)

    monkeypatch.setattr(
        module.os,
        "link",
        create_racing_target_then_link,
    )

    with pytest.raises(FileExistsError):
        module.write_benchmark_resolution(
            resolution,
            output_path=output,
            provenance_path=provenance,
        )

    assert output.read_text(encoding="utf-8") == "concurrent owner"
    assert not provenance.exists()
    assert list(tmp_path.glob(".*.tmp")) == []
