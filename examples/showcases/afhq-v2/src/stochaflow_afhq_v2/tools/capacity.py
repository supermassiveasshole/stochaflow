"""Profile AFHQ-v2 through registered DataSource and DataBuilder lifecycles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from stochaflow.data import DataArtifactBindings, build_data_loaders
from stochaflow.training.precision import validate_precision_support
from stochaflow.utils.config import StochaflowConfig, load_config
from stochaflow.utils.device import validate_execution_device
from stochaflow.utils.factory import (
    build_model,
    build_training_components,
    resolve_device,
)
from stochaflow_afhq_v2.tools.capacity_config import (
    DEFAULT_MICRO_BATCHES,
    DEFAULT_PRECISIONS,
    DEFAULT_WARMUP_UPDATES,
    MINIMUM_MEASURED_UPDATES,
    TARGET_EFFECTIVE_BATCH_SIZE,
    CapacityPrecision,
    positive_integer,
    validated_micro_batches,
    validated_precisions,
)
from stochaflow_afhq_v2.tools.capacity_measurement import (
    precision_comparisons,
)
from stochaflow_afhq_v2.tools.capacity_provenance import (
    canonical_sha256,
    code_identity,
    environment_report,
    normalize_non_finite_floats,
)
from stochaflow_afhq_v2.tools.capacity_trials import (
    profile_precision_trial,
    profile_trials,
    unsupported_trials,
)


def _activate_showcase_extension() -> None:
    """Import the installed showcase entry point target before composition."""

    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")


def _profile_precision_trial(
    config: StochaflowConfig,
    *,
    train_loader: Any,
    micro_batch: int,
    precision: CapacityPrecision,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, Any]:
    """Inject framework construction into one isolated precision trial."""

    return profile_precision_trial(
        config,
        train_loader=train_loader,
        micro_batch=micro_batch,
        precision=precision,
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        build_training_components_fn=build_training_components,
        collect_garbage=gc.collect,
    )


def _profile_trials(
    base: StochaflowConfig,
    *,
    micro_batches: Sequence[int],
    precisions: Sequence[CapacityPrecision],
    precision_errors: Mapping[CapacityPrecision, ValueError | None],
    warmup_updates: int,
    measured_updates: int,
    device_name: str,
    run_root: Path,
) -> tuple[list[dict[str, Any]], DataArtifactBindings]:
    """Inject framework construction into the capacity-trial matrix."""

    return profile_trials(
        base,
        micro_batches=micro_batches,
        precisions=precisions,
        precision_errors=precision_errors,
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        device_name=device_name,
        run_root=run_root,
        build_data_loaders_fn=build_data_loaders,
        profile_precision_trial_fn=_profile_precision_trial,
        collect_garbage=gc.collect,
    )


def _precision_preflight(
    precisions: Sequence[CapacityPrecision],
    device: torch.device,
) -> dict[CapacityPrecision, ValueError | None]:
    """Record precision support without suppressing unrelated failures."""

    results: dict[CapacityPrecision, ValueError | None] = {}
    for precision in precisions:
        try:
            validate_precision_support(precision, device)
        except ValueError as error:
            results[precision] = error
        else:
            results[precision] = None
    return results


def capacity_report(
    config_path: str | Path,
    *,
    micro_batches: Sequence[int] = DEFAULT_MICRO_BATCHES,
    precisions: Sequence[str] = DEFAULT_PRECISIONS,
    warmup_updates: int = DEFAULT_WARMUP_UPDATES,
    measured_updates: int = MINIMUM_MEASURED_UPDATES,
    device_name: str | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Run comparable framework-owned training trials and return their metrics."""

    source_path = Path(config_path)
    batches = validated_micro_batches(micro_batches)
    precision_values = validated_precisions(precisions)
    warmup = positive_integer(
        warmup_updates,
        label="warmup_updates",
    )
    measured = positive_integer(
        measured_updates,
        label="measured_updates",
    )
    if measured < MINIMUM_MEASURED_UPDATES:
        raise ValueError(
            "measured_updates must be at least "
            f"{MINIMUM_MEASURED_UPDATES}"
        )
    config = load_config(source_path)
    requested_device_name = device_name or "cuda"
    requested_device = resolve_device(requested_device_name)
    validate_execution_device(requested_device)
    if requested_device.type != "cuda" and device_name != "cpu":
        raise RuntimeError(
            "AFHQ-v2 capacity profiling requires CUDA by default; pass "
            "--device cpu explicitly only for bounded testing or debugging"
        )
    precision_errors = _precision_preflight(
        precision_values,
        requested_device,
    )
    supported_precisions = [
        precision
        for precision, error in precision_errors.items()
        if error is None
    ]
    resolved_run_root = (
        run_root
        if run_root is not None
        else Path(config.experiment.output_dir) / "capacity"
    ).resolve()
    source = source_path.read_bytes()
    if supported_precisions:
        _activate_showcase_extension()
        with torch.device("meta"):
            model = build_model(config.model)
        primary_model_parameter_count: int | None = sum(
            parameter.numel() for parameter in model.parameters()
        )
        del model
        trials, artifact_bindings = _profile_trials(
            config,
            micro_batches=batches,
            precisions=precision_values,
            precision_errors=precision_errors,
            warmup_updates=warmup,
            measured_updates=measured,
            device_name=requested_device_name,
            run_root=resolved_run_root,
        )
        serialized_bindings: dict[str, Any] | None = (
            artifact_bindings.to_dict()
        )
    else:
        primary_model_parameter_count = None
        trials = unsupported_trials(
            config,
            micro_batches=batches,
            precisions=precision_values,
            precision_errors=precision_errors,
            device_name=requested_device_name,
            run_root=resolved_run_root,
        )
        serialized_bindings = None
    report = {
        "schema_version": 3,
        "config": str(source_path.resolve()),
        "config_sha256": hashlib.sha256(source).hexdigest(),
        "model": config.model.name,
        "primary_model_parameter_count": primary_model_parameter_count,
        "seed": config.experiment.seed,
        "micro_batches": list(batches),
        "precisions": list(precision_values),
        "precision_support": {
            precision: {
                "supported": error is None,
                "error": None if error is None else str(error),
            }
            for precision, error in precision_errors.items()
        },
        "target_effective_batch_size": TARGET_EFFECTIVE_BATCH_SIZE,
        "warmup_updates": warmup,
        "measured_updates": measured,
        "environment": environment_report(requested_device),
        "code_identity": code_identity(),
        "data_artifact_bindings": serialized_bindings,
        "data_artifact_bindings_sha256": (
            None
            if serialized_bindings is None
            else canonical_sha256(serialized_bindings)
        ),
        "run_root": str(resolved_run_root),
        "trials": trials,
        "precision_comparisons": precision_comparisons(trials),
    }
    return cast(
        dict[str, Any],
        normalize_non_finite_floats(report),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the real training capacity command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--micro-batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_MICRO_BATCHES),
    )
    parser.add_argument(
        "--precisions",
        choices=DEFAULT_PRECISIONS,
        nargs="+",
        default=list(DEFAULT_PRECISIONS),
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=DEFAULT_WARMUP_UPDATES,
    )
    parser.add_argument(
        "--measured-updates",
        type=int,
        default=MINIMUM_MEASURED_UPDATES,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run capacity trials and print or persist one machine-readable report."""

    args = build_argument_parser().parse_args(argv)
    report = capacity_report(
        args.config,
        micro_batches=args.micro_batches,
        precisions=args.precisions,
        warmup_updates=args.warmup_updates,
        measured_updates=args.measured_updates,
        device_name=args.device,
        run_root=args.run_root,
    )
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()


__all__ = [
    "build_argument_parser",
    "capacity_report",
    "main",
]
