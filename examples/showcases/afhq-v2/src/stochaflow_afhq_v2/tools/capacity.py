"""Profile real AFHQ-v2 training capacity through framework lifecycles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch

import stochaflow
from stochaflow.data import DataArtifactBindings, build_data_loaders
from stochaflow.training.precision import validate_precision_support
from stochaflow.utils.config import (
    StochaflowConfig,
    load_config,
    load_config_dict,
)
from stochaflow.utils.device import validate_execution_device
from stochaflow.utils.factory import (
    TrainingComponents,
    build_model,
    build_training_components,
    resolve_device,
)
from stochaflow.utils.seed import set_seed

type CapacityPrecision = Literal["fp32", "bf16-mixed"]

_DEFAULT_MICRO_BATCHES = (4, 6, 8)
_DEFAULT_PRECISIONS: tuple[CapacityPrecision, ...] = (
    "fp32",
    "bf16-mixed",
)
_DEFAULT_WARMUP_UPDATES = 5
_MINIMUM_MEASURED_UPDATES = 25
_TARGET_EFFECTIVE_BATCH_SIZE = 32


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validated_micro_batches(values: Sequence[int]) -> tuple[int, ...]:
    batches = tuple(
        _positive_integer(value, label="micro batch") for value in values
    )
    if not batches:
        raise ValueError("at least one micro batch value is required")
    if len(set(batches)) != len(batches):
        raise ValueError("micro batch values must not contain duplicates")
    return batches


def _validated_precisions(
    values: Sequence[str],
) -> tuple[CapacityPrecision, ...]:
    if not values:
        raise ValueError("at least one precision is required")
    allowed = set(_DEFAULT_PRECISIONS)
    precisions: list[CapacityPrecision] = []
    for value in values:
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"precision must be one of: {choices}")
        precisions.append(cast(CapacityPrecision, value))
    if len(set(precisions)) != len(precisions):
        raise ValueError("precisions must not contain duplicates")
    return tuple(precisions)


def _accumulation_for_micro_batch(micro_batch: int) -> int:
    """Choose the nearest positive accumulation count to effective batch 32."""

    return max(
        1,
        round(_TARGET_EFFECTIVE_BATCH_SIZE / micro_batch),
    )


def _activate_showcase_extension() -> None:
    """Import the installed showcase entry point target before composition."""

    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _code_tree_sha256(root: Path) -> str:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    ]
    return _canonical_sha256(records)


def _code_identity() -> dict[str, Any]:
    core_root = Path(stochaflow.__file__).resolve().parent
    extension_root = Path(__file__).resolve().parents[1]
    return {
        "core": {
            "distribution": "stochaflow",
            "version": _distribution_version("stochaflow"),
            "package_root": str(core_root),
            "python_tree_sha256": _code_tree_sha256(core_root),
        },
        "extension": {
            "plugin_name": "stochaflow-afhq-v2",
            "distribution": "stochaflow-afhq-v2",
            "version": _distribution_version("stochaflow-afhq-v2"),
            "entry_point": "stochaflow_afhq_v2.stochaflow_ext",
            "package_root": str(extension_root),
            "python_tree_sha256": _code_tree_sha256(extension_root),
        },
    }


def _trial_config_identity(config: StochaflowConfig) -> dict[str, Any]:
    snapshot = config.to_dict()
    return {
        "seed": config.experiment.seed,
        "output_dir": str(Path(config.experiment.output_dir).resolve()),
        "resolved_config": snapshot,
        "resolved_config_sha256": _canonical_sha256(snapshot),
    }


def _trial_config(
    base: StochaflowConfig,
    *,
    micro_batch: int,
    precision: CapacityPrecision,
    device_name: str | None,
    output_dir: Path,
) -> StochaflowConfig:
    raw = base.to_dict()
    data = cast(dict[str, Any], raw["data"])
    data_params = cast(dict[str, Any], data["params"])
    loader = cast(dict[str, Any], data_params["loader"])
    loader["batch_size"] = micro_batch
    loader["steps_per_epoch"] = "auto"
    trainer = cast(dict[str, Any], raw["trainer"])
    trainer["num_epochs"] = 1
    trainer["precision"] = precision
    trainer["accumulate_grad_batches"] = (
        _accumulation_for_micro_batch(micro_batch)
    )
    trainer["show_progress"] = False
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


def _cuda_peak_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
        }
    return {
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
    }


def _environment_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["cuda_device"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        }
    else:
        report["cuda_device"] = None
    return report


def _measurement_report(
    metrics: Mapping[str, float],
    *,
    micro_batch: int,
    accumulation: int,
    requested_updates: int,
    memory: Mapping[str, int | None],
) -> dict[str, Any]:
    successful_updates = int(metrics["optimizer_steps"])
    skipped_updates = int(metrics["skipped_optimizer_steps"])
    processed_images = int(metrics["micro_batches"]) * micro_batch
    duration_seconds = metrics["duration_seconds"]
    if successful_updates < requested_updates:
        raise RuntimeError(
            "capacity trial ended before the requested successful optimizer "
            f"updates: {successful_updates} < {requested_updates}"
        )
    non_finite_loss_count = int(metrics["non_finite_loss_count"])
    non_finite_gradient_count = int(
        metrics["non_finite_gradient_count"]
    )
    return {
        "requested_optimizer_updates": requested_updates,
        "successful_optimizer_updates": successful_updates,
        "skipped_optimizer_updates": skipped_updates,
        "processed_micro_batches": int(metrics["micro_batches"]),
        "processed_images": processed_images,
        "effective_batch_size": micro_batch * accumulation,
        "loss": metrics["loss"],
        "duration_seconds": duration_seconds,
        "images_per_second": (
            processed_images / duration_seconds
            if duration_seconds > 0.0
            else 0.0
        ),
        "optimizer_updates_per_second": metrics[
            "optimizer_steps_per_second"
        ],
        "data_wait_seconds": metrics["data_wait_seconds"],
        "compute_seconds": metrics["compute_seconds"],
        "data_wait_compute_ratio": (
            metrics["data_wait_seconds"] / metrics["compute_seconds"]
            if metrics["compute_seconds"] > 0.0
            else None
        ),
        "forward_seconds": metrics["forward_seconds"],
        "backward_seconds": metrics["backward_seconds"],
        "optimizer_seconds": metrics["optimizer_seconds"],
        "non_finite_loss_count": non_finite_loss_count,
        "non_finite_gradient_count": non_finite_gradient_count,
        **memory,
    }


def _precision_comparisons(
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    by_batch: dict[int, dict[str, Mapping[str, Any]]] = {}
    for trial in trials:
        if trial.get("status") not in {"ok", "non_finite"}:
            continue
        micro_batch = cast(int, trial["micro_batch_size"])
        precision = cast(str, trial["precision"])
        by_batch.setdefault(micro_batch, {})[precision] = trial
    for micro_batch in sorted(by_batch):
        pair = by_batch[micro_batch]
        if set(_DEFAULT_PRECISIONS) - set(pair):
            continue
        fp32 = cast(Mapping[str, Any], pair["fp32"]["measurement"])
        bf16 = cast(
            Mapping[str, Any],
            pair["bf16-mixed"]["measurement"],
        )
        fp32_throughput = cast(float, fp32["images_per_second"])
        bf16_throughput = cast(float, bf16["images_per_second"])
        fp32_allocated = cast(
            int | None,
            fp32["peak_allocated_vram_bytes"],
        )
        bf16_allocated = cast(
            int | None,
            bf16["peak_allocated_vram_bytes"],
        )
        fp32_reserved = cast(
            int | None,
            fp32["peak_reserved_vram_bytes"],
        )
        bf16_reserved = cast(
            int | None,
            bf16["peak_reserved_vram_bytes"],
        )
        comparisons.append(
            {
                "micro_batch_size": micro_batch,
                "bf16_vs_fp32_images_per_second_delta": (
                    bf16_throughput - fp32_throughput
                ),
                "bf16_vs_fp32_throughput_ratio": (
                    bf16_throughput / fp32_throughput
                    if fp32_throughput > 0.0
                    else None
                ),
                "bf16_vs_fp32_peak_allocated_vram_delta_bytes": (
                    bf16_allocated - fp32_allocated
                    if bf16_allocated is not None
                    and fp32_allocated is not None
                    else None
                ),
                "bf16_vs_fp32_peak_reserved_vram_delta_bytes": (
                    bf16_reserved - fp32_reserved
                    if bf16_reserved is not None
                    and fp32_reserved is not None
                    else None
                ),
            }
        )
    return comparisons


def _profile_precision_trial(
    config: StochaflowConfig,
    *,
    train_loader: Any,
    micro_batch: int,
    precision: CapacityPrecision,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, Any]:
    training: TrainingComponents | None = None
    device = resolve_device(config.trainer.device)
    set_seed(config.experiment.seed)
    try:
        training = build_training_components(config)
        actual_device = training.trainer.device
        warmup = training.trainer.train_epoch(
            train_loader,
            epoch_index=0,
            show_progress=False,
            max_optimizer_steps=warmup_updates,
            profile_phases=True,
        )
        if int(warmup["optimizer_steps"]) < warmup_updates:
            raise RuntimeError(
                "capacity warmup ended before the requested successful "
                f"optimizer updates: {int(warmup['optimizer_steps'])} "
                f"< {warmup_updates}"
            )
        if actual_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(actual_device)
        measured = training.trainer.train_epoch(
            train_loader,
            epoch_index=1,
            show_progress=False,
            max_optimizer_steps=measured_updates,
            profile_phases=True,
        )
        memory = _cuda_peak_memory(actual_device)
        measurement = _measurement_report(
            measured,
            micro_batch=micro_batch,
            accumulation=config.trainer.accumulate_grad_batches,
            requested_updates=measured_updates,
            memory=memory,
        )
        non_finite = (
            int(warmup["non_finite_loss_count"])
            + int(warmup["non_finite_gradient_count"])
            + measurement["non_finite_loss_count"]
            + measurement["non_finite_gradient_count"]
        )
        return {
            "status": "ok" if non_finite == 0 else "non_finite",
            "micro_batch_size": micro_batch,
            "precision": precision,
            "device": str(actual_device),
            "accumulate_grad_batches": config.trainer.accumulate_grad_batches,
            **_trial_config_identity(config),
            "warmup": {
                "requested_optimizer_updates": warmup_updates,
                "successful_optimizer_updates": int(
                    warmup["optimizer_steps"]
                ),
                "skipped_optimizer_updates": int(
                    warmup["skipped_optimizer_steps"]
                ),
                "non_finite_loss_count": int(
                    warmup["non_finite_loss_count"]
                ),
                "non_finite_gradient_count": int(
                    warmup["non_finite_gradient_count"]
                ),
            },
            "measurement": measurement,
        }
    finally:
        if training is not None:
            training.logger.close()
        del training
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _failed_trial(
    *,
    micro_batch: int,
    precision: CapacityPrecision,
    device: torch.device,
    status: str,
    error: Exception,
    config: StochaflowConfig,
) -> dict[str, Any]:
    return {
        "status": status,
        "micro_batch_size": micro_batch,
        "precision": precision,
        "device": str(device),
        "error_type": type(error).__name__,
        "error": str(error),
        **_trial_config_identity(config),
    }


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
    trials: list[dict[str, Any]] = []
    artifact_bindings: DataArtifactBindings | None = None
    for micro_batch in micro_batches:
        configs = {
            precision: _trial_config(
                base,
                micro_batch=micro_batch,
                precision=precision,
                device_name=device_name,
                output_dir=run_root / f"batch-{micro_batch}" / precision,
            )
            for precision in precisions
        }
        supported = [
            precision
            for precision in precisions
            if precision_errors[precision] is None
        ]
        if not supported:
            raise RuntimeError(
                "capacity trial composition requires a supported precision"
            )
        data_config = configs[supported[0]]
        set_seed(data_config.experiment.seed)
        loaders = build_data_loaders(
            data_config.data,
            seed=data_config.experiment.seed,
        )
        if loaders.artifact_bindings is None:
            raise RuntimeError(
                "AFHQ-v2 capacity DataBuilder must return artifact bindings"
            )
        if artifact_bindings is None:
            artifact_bindings = loaders.artifact_bindings
        elif loaders.artifact_bindings != artifact_bindings:
            raise RuntimeError(
                "AFHQ-v2 capacity trials resolved different data artifacts"
            )
        try:
            for precision in precisions:
                config = configs[precision]
                device = resolve_device(config.trainer.device)
                precision_error = precision_errors[precision]
                if precision_error is not None:
                    trials.append(
                        _failed_trial(
                            micro_batch=micro_batch,
                            precision=precision,
                            device=device,
                            status="unsupported_precision",
                            error=precision_error,
                            config=config,
                        )
                    )
                    continue
                try:
                    trial = _profile_precision_trial(
                        config,
                        train_loader=loaders.train,
                        micro_batch=micro_batch,
                        precision=precision,
                        warmup_updates=warmup_updates,
                        measured_updates=measured_updates,
                    )
                except torch.OutOfMemoryError as error:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    trial = _failed_trial(
                        micro_batch=micro_batch,
                        precision=precision,
                        device=device,
                        status="out_of_memory",
                        error=error,
                        config=config,
                    )
                trials.append(trial)
        finally:
            del loaders
            gc.collect()
    if artifact_bindings is None:
        raise RuntimeError("AFHQ-v2 capacity did not build any data artifact")
    return trials, artifact_bindings


def _precision_preflight(
    precisions: Sequence[CapacityPrecision],
    device: torch.device,
) -> dict[CapacityPrecision, ValueError | None]:
    results: dict[CapacityPrecision, ValueError | None] = {}
    for precision in precisions:
        try:
            validate_precision_support(precision, device)
        except ValueError as error:
            results[precision] = error
        else:
            results[precision] = None
    return results


def _unsupported_trials(
    base: StochaflowConfig,
    *,
    micro_batches: Sequence[int],
    precisions: Sequence[CapacityPrecision],
    precision_errors: Mapping[CapacityPrecision, ValueError | None],
    device_name: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    device = resolve_device(device_name)
    for micro_batch in micro_batches:
        for precision in precisions:
            error = precision_errors[precision]
            if error is None:
                continue
            config = _trial_config(
                base,
                micro_batch=micro_batch,
                precision=precision,
                device_name=device_name,
                output_dir=run_root / f"batch-{micro_batch}" / precision,
            )
            trials.append(
                _failed_trial(
                    micro_batch=micro_batch,
                    precision=precision,
                    device=device,
                    status="unsupported_precision",
                    error=error,
                    config=config,
                )
            )
    return trials


def capacity_report(
    config_path: Path,
    *,
    micro_batches: Sequence[int] = _DEFAULT_MICRO_BATCHES,
    precisions: Sequence[str] = _DEFAULT_PRECISIONS,
    warmup_updates: int = _DEFAULT_WARMUP_UPDATES,
    measured_updates: int = _MINIMUM_MEASURED_UPDATES,
    device_name: str | None = None,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Run comparable framework-owned training trials and return their metrics."""

    batches = _validated_micro_batches(micro_batches)
    precision_values = _validated_precisions(precisions)
    warmup = _positive_integer(
        warmup_updates,
        label="warmup_updates",
    )
    measured = _positive_integer(
        measured_updates,
        label="measured_updates",
    )
    if measured < _MINIMUM_MEASURED_UPDATES:
        raise ValueError(
            "measured_updates must be at least "
            f"{_MINIMUM_MEASURED_UPDATES}"
        )
    config = load_config(config_path)
    requested_device_name = device_name or "cuda"
    requested_device = resolve_device(requested_device_name)
    validate_execution_device(requested_device)
    if (
        requested_device.type != "cuda"
        and device_name != "cpu"
    ):
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
    source = config_path.read_bytes()
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
        trials = _unsupported_trials(
            config,
            micro_batches=batches,
            precisions=precision_values,
            precision_errors=precision_errors,
            device_name=requested_device_name,
            run_root=resolved_run_root,
        )
        serialized_bindings = None
    return {
        "schema_version": 3,
        "config": str(config_path.resolve()),
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
        "target_effective_batch_size": _TARGET_EFFECTIVE_BATCH_SIZE,
        "warmup_updates": warmup,
        "measured_updates": measured,
        "environment": _environment_report(requested_device),
        "code_identity": _code_identity(),
        "data_artifact_bindings": serialized_bindings,
        "data_artifact_bindings_sha256": (
            None
            if serialized_bindings is None
            else _canonical_sha256(serialized_bindings)
        ),
        "run_root": str(resolved_run_root),
        "trials": trials,
        "precision_comparisons": _precision_comparisons(trials),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the real training capacity command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--micro-batches",
        type=int,
        nargs="+",
        default=list(_DEFAULT_MICRO_BATCHES),
    )
    parser.add_argument(
        "--precisions",
        choices=_DEFAULT_PRECISIONS,
        nargs="+",
        default=list(_DEFAULT_PRECISIONS),
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=_DEFAULT_WARMUP_UPDATES,
    )
    parser.add_argument(
        "--measured-updates",
        type=int,
        default=_MINIMUM_MEASURED_UPDATES,
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
    encoded = json.dumps(report, indent=2, sort_keys=True)
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
