"""Execute and summarize bounded AFHQ-v2 capacity trials."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from stochaflow.data import DataArtifactBindings
from stochaflow.utils.config import StochaflowConfig
from stochaflow.utils.factory import resolve_device
from stochaflow.utils.seed import set_seed
from stochaflow_afhq_v2.tools.capacity_config import (
    CapacityPrecision,
    trial_config,
)
from stochaflow_afhq_v2.tools.capacity_measurement import (
    cuda_peak_memory,
    measurement_report,
)
from stochaflow_afhq_v2.tools.capacity_provenance import (
    trial_config_identity,
)

type DataLoadersFactory = Callable[..., Any]
type GarbageCollector = Callable[[], int]
type PrecisionTrialRunner = Callable[..., dict[str, Any]]
type TrainingComponentsFactory = Callable[[StochaflowConfig], Any]


def profile_precision_trial(
    config: StochaflowConfig,
    *,
    train_loader: Any,
    micro_batch: int,
    precision: CapacityPrecision,
    warmup_updates: int,
    measured_updates: int,
    build_training_components_fn: TrainingComponentsFactory,
    collect_garbage: GarbageCollector,
) -> dict[str, Any]:
    """Run one warmup and measured precision trial through the core trainer."""

    training: Any = None
    device = resolve_device(config.trainer.device)
    set_seed(config.experiment.seed)
    try:
        training_runtime = build_training_components_fn(config)
        training = training_runtime
        actual_device = training_runtime.trainer.device
        warmup = training_runtime.trainer.train_epoch(
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
        measured = training_runtime.trainer.train_epoch(
            train_loader,
            epoch_index=1,
            show_progress=False,
            max_optimizer_steps=measured_updates,
            profile_phases=True,
        )
        measurement = measurement_report(
            measured,
            micro_batch=micro_batch,
            accumulation=config.trainer.accumulate_grad_batches,
            requested_updates=measured_updates,
            memory=cuda_peak_memory(actual_device),
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
            **trial_config_identity(config),
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
        collect_garbage()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def failed_trial(
    *,
    micro_batch: int,
    precision: CapacityPrecision,
    device: torch.device,
    status: str,
    error: Exception,
    config: StochaflowConfig,
) -> dict[str, Any]:
    """Describe one unsupported or out-of-memory trial."""

    return {
        "status": status,
        "micro_batch_size": micro_batch,
        "precision": precision,
        "device": str(device),
        "error_type": type(error).__name__,
        "error": str(error),
        **trial_config_identity(config),
    }


def profile_trials(
    base: StochaflowConfig,
    *,
    micro_batches: Sequence[int],
    precisions: Sequence[CapacityPrecision],
    precision_errors: Mapping[CapacityPrecision, ValueError | None],
    warmup_updates: int,
    measured_updates: int,
    device_name: str,
    run_root: Path,
    build_data_loaders_fn: DataLoadersFactory,
    profile_precision_trial_fn: PrecisionTrialRunner,
    collect_garbage: GarbageCollector,
) -> tuple[list[dict[str, Any]], DataArtifactBindings]:
    """Run every trial through one source-backed data recipe per batch size."""

    trials: list[dict[str, Any]] = []
    artifact_bindings: DataArtifactBindings | None = None
    for micro_batch in micro_batches:
        configs = {
            precision: trial_config(
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
        loaders = build_data_loaders_fn(
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
                        failed_trial(
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
                    trial = profile_precision_trial_fn(
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
                    trial = failed_trial(
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
            collect_garbage()
    if artifact_bindings is None:
        raise RuntimeError("AFHQ-v2 capacity did not build any data artifact")
    return trials, artifact_bindings


def unsupported_trials(
    base: StochaflowConfig,
    *,
    micro_batches: Sequence[int],
    precisions: Sequence[CapacityPrecision],
    precision_errors: Mapping[CapacityPrecision, ValueError | None],
    device_name: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    """Describe a trial matrix when no requested precision is supported."""

    trials: list[dict[str, Any]] = []
    device = resolve_device(device_name)
    for micro_batch in micro_batches:
        for precision in precisions:
            error = precision_errors[precision]
            if error is None:
                continue
            config = trial_config(
                base,
                micro_batch=micro_batch,
                precision=precision,
                device_name=device_name,
                output_dir=run_root / f"batch-{micro_batch}" / precision,
            )
            trials.append(
                failed_trial(
                    micro_batch=micro_batch,
                    precision=precision,
                    device=device,
                    status="unsupported_precision",
                    error=error,
                    config=config,
                )
            )
    return trials


__all__ = [
    "failed_trial",
    "profile_precision_trial",
    "profile_trials",
    "unsupported_trials",
]
