"""Configuration tests for automatic precision and accumulation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from stochaflow.utils.config import ConfigError, load_config_dict


def minimal_config() -> dict[str, Any]:
    """Return the smallest valid training configuration."""

    return {
        "experiment": {"name": "precision-config"},
        "data": {"name": "image", "params": {}},
        "model": {"name": "unet", "params": {}},
        "training": {"name": "gaussian_denoising", "params": {}},
    }


def test_trainer_precision_and_accumulation_defaults_are_compatible() -> None:
    config = load_config_dict(minimal_config())

    assert config.trainer.precision == "fp32"
    assert config.trainer.accumulate_grad_batches == 1


@pytest.mark.parametrize("precision", ["fp32", "bf16-mixed", "fp16-mixed"])
def test_trainer_accepts_supported_precision_kinds(precision: str) -> None:
    raw = minimal_config()
    raw["trainer"] = {
        "precision": precision,
        "accumulate_grad_batches": 4,
    }

    config = load_config_dict(raw)

    assert config.trainer.precision == precision
    assert config.trainer.accumulate_grad_batches == 4


@pytest.mark.parametrize(
    "precision",
    ["", "auto", "bf16", 1, True, [], {}],
)
def test_trainer_rejects_invalid_precision_values(precision: object) -> None:
    raw = minimal_config()
    raw["trainer"] = {"precision": precision}

    with pytest.raises(
        ConfigError,
        match=r"trainer\.precision must be",
    ):
        load_config_dict(raw)


@pytest.mark.parametrize(
    "accumulation",
    [0, -1, 1.5, "2", True, [], {}],
)
def test_trainer_rejects_invalid_accumulation_values(
    accumulation: object,
) -> None:
    raw = minimal_config()
    raw["trainer"] = {"accumulate_grad_batches": accumulation}

    with pytest.raises(
        ConfigError,
        match=r"accumulate_grad_batches must be a positive integer",
    ):
        load_config_dict(raw)


@pytest.mark.parametrize(
    "field",
    ["precision", "accumulate_grad_batches"],
)
def test_trainer_rejects_null_precision_fields(field: str) -> None:
    raw = minimal_config()
    raw["trainer"] = {field: None}

    with pytest.raises(ConfigError, match=rf"config\.trainer\.{field} must not be null"):
        load_config_dict(raw)


def test_precision_config_loading_does_not_mutate_input() -> None:
    raw = minimal_config()
    raw["trainer"] = {
        "precision": "bf16-mixed",
        "accumulate_grad_batches": 2,
    }
    original = deepcopy(raw)

    load_config_dict(raw)

    assert raw == original
