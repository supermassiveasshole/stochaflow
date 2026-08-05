"""Capacity profiling request validation and trial configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from stochaflow.utils.config import (
    StochaflowConfig,
    load_config_dict,
)

type CapacityPrecision = Literal["fp32", "bf16-mixed"]

DEFAULT_MICRO_BATCHES = (4, 6, 8)
DEFAULT_PRECISIONS: tuple[CapacityPrecision, ...] = (
    "fp32",
    "bf16-mixed",
)
DEFAULT_WARMUP_UPDATES = 5
MINIMUM_MEASURED_UPDATES = 25
TARGET_EFFECTIVE_BATCH_SIZE = 32


def positive_integer(value: object, *, label: str) -> int:
    """Validate one strictly positive integer option."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validated_micro_batches(values: Sequence[int]) -> tuple[int, ...]:
    """Validate a non-empty sequence of distinct micro-batch sizes."""

    batches = tuple(
        positive_integer(value, label="micro batch") for value in values
    )
    if not batches:
        raise ValueError("at least one micro batch value is required")
    if len(set(batches)) != len(batches):
        raise ValueError("micro batch values must not contain duplicates")
    return batches


def validated_precisions(
    values: Sequence[str],
) -> tuple[CapacityPrecision, ...]:
    """Validate supported, distinct capacity precision names."""

    if not values:
        raise ValueError("at least one precision is required")
    allowed = set(DEFAULT_PRECISIONS)
    precisions: list[CapacityPrecision] = []
    for value in values:
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"precision must be one of: {choices}")
        precisions.append(cast(CapacityPrecision, value))
    if len(set(precisions)) != len(precisions):
        raise ValueError("precisions must not contain duplicates")
    return tuple(precisions)


def accumulation_for_micro_batch(micro_batch: int) -> int:
    """Choose the nearest positive accumulation count to effective batch 32."""

    return max(
        1,
        round(TARGET_EFFECTIVE_BATCH_SIZE / micro_batch),
    )


def trial_config(
    base: StochaflowConfig,
    *,
    micro_batch: int,
    precision: CapacityPrecision,
    device_name: str | None,
    output_dir: Path,
) -> StochaflowConfig:
    """Derive an isolated, bounded training configuration for one trial."""

    raw = base.to_dict()
    data = cast(dict[str, Any], raw["data"])
    data_params = cast(dict[str, Any], data["params"])
    loader = cast(dict[str, Any], data_params["loader"])
    loader["batch_size"] = micro_batch
    loader["steps_per_epoch"] = "auto"
    trainer = cast(dict[str, Any], raw["trainer"])
    trainer["num_epochs"] = 1
    trainer["precision"] = precision
    trainer["accumulate_grad_batches"] = accumulation_for_micro_batch(
        micro_batch
    )
    trainer["show_progress"] = False
    validation_evaluation = cast(
        dict[str, Any], trainer["validation_evaluation"]
    )
    validation_evaluation["enabled"] = False
    if device_name is not None:
        trainer["device"] = device_name
    experiment = cast(dict[str, Any], raw["experiment"])
    experiment["name"] = (
        f"{experiment['name']}_capacity_{micro_batch}_{precision}"
    )
    experiment["output_dir"] = str(output_dir)
    raw["diagnostics"] = []
    raw["logging"] = {
        "log_every": 1_000_000_000,
        "backends": [
            {
                "name": "local",
                "params": {"console": False, "append": False},
            }
        ],
        "torch_logs": {},
    }
    return load_config_dict(raw)


__all__ = [
    "DEFAULT_MICRO_BATCHES",
    "DEFAULT_PRECISIONS",
    "DEFAULT_WARMUP_UPDATES",
    "MINIMUM_MEASURED_UPDATES",
    "TARGET_EFFECTIVE_BATCH_SIZE",
    "CapacityPrecision",
    "accumulation_for_micro_batch",
    "positive_integer",
    "trial_config",
    "validated_micro_batches",
    "validated_precisions",
]
