from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from stochaflow.utils.config import load_config

_ROOT = Path(__file__).resolve().parents[1]
_SHOWCASE = _ROOT / "examples" / "showcases" / "afhq-v2"


def test_afhq_showcase_is_an_installable_source_extension() -> None:
    declaration = tomllib.loads(
        (_SHOWCASE / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert declaration["project"]["name"] == "stochaflow-afhq-v2"
    assert declaration["project"]["entry-points"]["stochaflow.extensions"] == {
        "stochaflow-afhq-v2": "stochaflow_afhq_v2.stochaflow_ext"
    }
    assert declaration["tool"]["setuptools"]["package-data"] == {
        "stochaflow_afhq_v2": ["resources/*.yaml"]
    }
    assert (
        _SHOWCASE
        / "src"
        / "stochaflow_afhq_v2"
        / "resources"
        / "afhq-v2.lock.yaml"
    ).is_file()
    assert (
        _SHOWCASE
        / "src"
        / "stochaflow_afhq_v2"
        / "preparation.py"
    ).is_file()
    assert not (_SHOWCASE / "prepare.py").exists()


def test_afhq_showcase_config_selects_source_extension() -> None:
    path = _SHOWCASE / "experiments" / "ddpm_128.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_config(path)

    assert raw["extensions"]["plugins"] == ["stochaflow-afhq-v2"]
    assert raw["data"]["params"]["source"]["name"] == "afhq-v2.official"
    assert set(raw["data"]["params"]["source"]) == {
        "name",
        "params",
        "materialization",
    }
    assert raw["data"]["params"]["source"]["materialization"] == {
        "cache_root": "./data",
        "policy": "ensure",
        "verification": "full",
    }
    assert config.data.name == "image"
    assert config.trainer.num_epochs == 200
    assert config.lr_scheduler is not None
    assert config.lr_scheduler.params["total_steps"] == 167_800
