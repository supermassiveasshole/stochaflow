"""Generic training loop utilities."""

import math
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from copy import copy
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.metrics import (
    EpochMetricSnapshot,
    MetricSource,
    detach_metric_updates,
    validate_training_monitor_key,
)
from stochaflow.training.builder import (
    ManagedTrainingModule,
    TrainingPlan,
    validate_training_plan,
)
from stochaflow.training.builder import (
    trainable_parameters as plan_trainable_parameters,
)
from stochaflow.training.diagnostics.contracts import (
    FitStartEvent,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainingDiagnostic,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.metric_binding import (
    TrainingMetricPhase,
    TrainingMetricRuntime,
)
from stochaflow.training.precision import (
    PrecisionRuntime,
    build_precision_runtime,
)
from stochaflow.training.strategy import (
    Batch,
    DeviceTransferableBatch,
    ScalarMetric,
    TrainStepOutput,
    loss_aggregation_weight_to_float,
    validate_train_step_output,
)
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    inference_asset_descriptors_equal,
    inference_asset_descriptors_from_projections,
)
from stochaflow.utils.device import move_module_to_device
from stochaflow.utils.iterables import try_length
from stochaflow.utils.logging import ExperimentLogger, NullLogger
from stochaflow.utils.seed import preserve_global_rng_state


def _move_to_device(batch: Batch, device: torch.device) -> Batch:
    """Recursively move tensors in a batch structure onto a device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, DeviceTransferableBatch):
        return batch.to_device(device)
    if isinstance(batch, Mapping):
        moved_items = {
            key: _move_to_device(value, device) for key, value in batch.items()
        }
        if isinstance(batch, MutableMapping):
            moved_mapping = copy(batch)
            moved_mapping.clear()
            moved_mapping.update(moved_items)
            return moved_mapping
        try:
            mapping_type = cast(
                Callable[[Mapping[Any, Any]], Mapping[Any, Any]],
                type(batch),
            )
            return mapping_type(moved_items)
        except TypeError:
            return moved_items
    if isinstance(batch, tuple):
        moved_tuple = tuple(_move_to_device(item, device) for item in batch)
        if hasattr(batch, "_fields"):
            return type(batch)(*moved_tuple)
        return moved_tuple
    if isinstance(batch, list):
        moved_list = [_move_to_device(item, device) for item in batch]
        if type(batch) is list:
            return moved_list
        try:
            return type(batch)(moved_list)
        except TypeError:
            return moved_list
    return batch


def _optimizer_metrics(optimizer: Optimizer) -> dict[str, float]:
    """Extract a flat set of optimizer-side scalar metrics."""

    metrics: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        if "lr" in group:
            metrics[f"train/lr/group_{index}"] = float(group["lr"])
    return metrics


def _first_lr(optimizer: Optimizer) -> float | None:
    """Return the learning rate of the first optimizer parameter group."""

    if not optimizer.param_groups:
        return None
    lr = optimizer.param_groups[0].get("lr")
    if lr is None:
        return None
    return float(lr)


def _resolve_total_batches(
    dataloader: Iterable[Batch],
    max_batches: int | None,
) -> int | None:
    """Resolve the number of displayed batches for a progress reporter."""

    total = try_length(dataloader)
    if total is not None:
        if max_batches is not None:
            return min(total, max_batches)
        return total
    return max_batches


def _set_dataloader_epoch(dataloader: Iterable[Batch], epoch: int) -> None:
    """Propagate an epoch to distinct duck-typed PyTorch sampler objects."""

    seen: set[int] = set()
    for attribute in ("sampler", "batch_sampler"):
        sampler = getattr(dataloader, attribute, None)
        if sampler is None or id(sampler) in seen:
            continue
        seen.add(id(sampler))
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


type TrainingPhaseToken = float | torch.cuda.Event


class TrainingPhaseProfiler:
    """Record opt-in CPU timings or asynchronous CUDA event intervals."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.phase_seconds = {
            "forward_seconds": 0.0,
            "backward_seconds": 0.0,
            "optimizer_seconds": 0.0,
        }
        self.cuda_events: dict[
            str,
            list[tuple[torch.cuda.Event, torch.cuda.Event]],
        ] = {}

    def start_measurement(self) -> float:
        """Start one wall-clock measurement after pending CUDA work drains."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start_phase(self) -> TrainingPhaseToken:
        """Record one phase start without synchronizing the execution stream."""

        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def end_phase(
        self,
        name: str,
        started_at: TrainingPhaseToken,
    ) -> None:
        """Record one phase end without forcing CUDA work to finish."""

        if self.device.type == "cuda":
            if isinstance(started_at, float):
                raise TypeError("CUDA phase timing requires a CUDA event")
            started_event = cast(torch.cuda.Event, started_at)
            ended_at = torch.cuda.Event(enable_timing=True)
            ended_at.record()
            self.cuda_events.setdefault(name, []).append(
                (started_event, ended_at)
            )
            return
        if not isinstance(started_at, float):
            raise TypeError("CPU phase timing requires a perf-counter value")
        self.phase_seconds[name] += time.perf_counter() - started_at

    def finish_measurement(
        self,
        started_at: float,
    ) -> tuple[float, dict[str, float]]:
        """Synchronize once and materialize wall-clock and phase durations."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            for name, intervals in self.cuda_events.items():
                self.phase_seconds[name] = sum(
                    start.elapsed_time(end) / 1_000.0
                    for start, end in intervals
                )
        return time.perf_counter() - started_at, dict(self.phase_seconds)


def _gradients_are_finite(parameters: tuple[nn.Parameter, ...]) -> bool:
    """Return whether every materialized gradient contains finite values."""

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
            return False
    return True


def _accumulate_scalar_metrics(
    values: dict[str, list[float | torch.Tensor]],
    metrics: Mapping[str, ScalarMetric],
) -> None:
    """Collect detached scalars without retaining their step outputs."""

    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            detached_value: float | torch.Tensor = value.detach().clone()
        else:
            detached_value = float(value)
        values.setdefault(name, []).append(detached_value)


def _mean_accumulated_metrics(
    values: Mapping[str, list[float | torch.Tensor]],
) -> dict[str, float]:
    """Materialize accumulated metrics after every backward has completed."""

    result: dict[str, float] = {}
    for name, metric_values in values.items():
        total = sum(
            float(value.detach().item())
            if isinstance(value, torch.Tensor)
            else value
            for value in metric_values
        )
        result[name] = total / len(metric_values)
    return result


def _epoch_metric_snapshot(
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float] | None,
    *,
    epoch: int,
) -> EpochMetricSnapshot:
    """Build one canonical, source-typed snapshot for a completed epoch."""

    values: dict[str, float] = {
        "train/loss": float(train_metrics["loss"]),
        "system/epoch": float(epoch),
    }
    sources: dict[str, MetricSource] = {}
    train_source = MetricSource(
        origin="phase",
        data_role="train",
        protocol_id=None,
        selection_eligible=True,
    )
    validation_source = MetricSource(
        origin="phase",
        data_role="validation",
        protocol_id=None,
        selection_eligible=True,
    )
    system_source = MetricSource(
        origin="system",
        data_role=None,
        protocol_id=None,
        selection_eligible=False,
    )
    sources["train/loss"] = train_source
    sources["system/epoch"] = system_source

    for name, value in train_metrics.items():
        if name.startswith("train/metrics/"):
            values[name] = float(value)
            sources[name] = train_source
    train_system_names = {
        "num_batches",
        "micro_batches",
        "optimizer_steps",
        "skipped_optimizer_steps",
        "optimizer_steps_per_second",
        "data_wait_seconds",
        "compute_seconds",
        "duration_seconds",
        "forward_seconds",
        "backward_seconds",
        "optimizer_seconds",
        "non_finite_loss_count",
        "non_finite_gradient_count",
    }
    for name in train_system_names:
        if name in train_metrics:
            key = f"system/train/{name}"
            values[key] = float(train_metrics[name])
            sources[key] = system_source

    if validation_metrics is not None:
        values["valid/loss"] = float(validation_metrics["loss"])
        sources["valid/loss"] = validation_source
        for name, value in validation_metrics.items():
            if name.startswith("valid/metrics/"):
                values[name] = float(value)
                sources[name] = validation_source
        for name in ("num_batches", "duration_seconds"):
            key = f"system/valid/{name}"
            values[key] = float(validation_metrics[name])
            sources[key] = system_source
    return EpochMetricSnapshot(values=values, sources=sources)


def _checkpoint_metric_sources(
    snapshot: EpochMetricSnapshot,
) -> dict[str, dict[str, object]]:
    """Serialize snapshot source metadata for checkpoint provenance."""

    return {
        name: source.to_dict()
        for name, source in snapshot.sources.items()
    }


def _validate_optimizer_parameters(
    optimizer: Optimizer,
    expected_parameters: tuple[nn.Parameter, ...],
) -> None:
    """Require the optimizer to own exactly the Plan-selected parameters."""

    actual_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    actual_ids = tuple(map(id, actual_parameters))
    expected_ids = tuple(map(id, expected_parameters))
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("optimizer parameters must exactly match TrainingPlan")


def _validate_checkpoint_manager(
    manager: CheckpointManager,
    *,
    plan: TrainingPlan,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler | None,
    ema: ExponentialMovingAverage | None,
    precision: PrecisionRuntime,
) -> None:
    """Keep checkpoint ownership identical to the validated training Plan."""

    if manager.model is not plan.primary_model:
        raise ValueError("CheckpointManager model must match TrainingPlan")
    if manager.process is not plan.process:
        raise ValueError("CheckpointManager process must match TrainingPlan")
    if manager.objective is not plan.objective:
        raise ValueError("CheckpointManager objective must match TrainingPlan")
    if set(manager.auxiliary_modules) != set(plan.auxiliary_modules):
        raise ValueError("CheckpointManager auxiliary names must match TrainingPlan")
    for name, asset in plan.auxiliary_modules.items():
        if manager.auxiliary_modules[name] is not asset.module:
            raise ValueError(
                f"CheckpointManager auxiliary '{name}' must match TrainingPlan"
            )
    if manager.optimizer is not optimizer:
        raise ValueError("CheckpointManager optimizer must match Trainer")
    if manager.lr_scheduler is not lr_scheduler:
        raise ValueError("CheckpointManager lr_scheduler must match Trainer")
    if manager.ema is not ema:
        raise ValueError("CheckpointManager EMA must match Trainer")
    if manager.precision_kind != precision.kind:
        raise ValueError("CheckpointManager precision must match Trainer")
    if manager.grad_scaler is not precision.grad_scaler:
        raise ValueError("CheckpointManager GradScaler must match Trainer")
    if manager.inference_recipe != plan.inference_recipe:
        raise ValueError(
            "CheckpointManager inference recipe must match TrainingPlan"
        )
    expected_descriptors = inference_asset_descriptors_from_projections(
        plan.inference_assets
    )
    if not inference_asset_descriptors_equal(
        manager.inference_asset_descriptors,
        expected_descriptors,
    ):
        raise ValueError(
            "CheckpointManager inference asset descriptors must match TrainingPlan"
        )


def _validate_checkpoint_training_config(
    config: object,
    *,
    precision: PrecisionRuntime,
    accumulate_grad_batches: int,
) -> None:
    """Keep serialized automatic-loop topology identical to this Trainer."""

    if config is None:
        return
    if type(config) is not dict:
        raise TypeError("checkpoint_config must be an exact dictionary or None")
    trainer = cast(dict[str, Any], config).get("trainer")
    if trainer is None:
        return
    if type(trainer) is not dict:
        raise TypeError("checkpoint_config.trainer must be an exact dictionary")
    trainer_config = cast(dict[str, Any], trainer)
    configured_precision = trainer_config.get("precision")
    if (
        configured_precision is not None
        and configured_precision != precision.kind
    ):
        raise ValueError(
            "checkpoint_config trainer precision must match Trainer"
        )
    configured_accumulation = trainer_config.get(
        "accumulate_grad_batches"
    )
    if (
        configured_accumulation is not None
        and configured_accumulation != accumulate_grad_batches
    ):
        raise ValueError(
            "checkpoint_config accumulation must match Trainer"
        )


@dataclass(frozen=True, slots=True)
class TrainingFitState:
    """Validated best-selection and early-stopping checkpoint state."""

    best_epoch: int | None
    best_metric_value: float | None
    epochs_without_improvement: int
    stopped_early: bool
    monitor: str | None
    mode: str | None

    @classmethod
    def from_mapping(cls, state: object) -> "TrainingFitState":
        """Parse the strict persisted mapping without mutating trainer state."""

        if not isinstance(state, Mapping):
            raise TypeError("checkpoint metadata.training_loop must be a mapping")
        required = {
            "best_epoch",
            "best_metric_value",
            "epochs_without_improvement",
            "stopped_early",
            "monitor",
            "mode",
        }
        if set(state) != required:
            missing = sorted(required - set(state))
            unknown = sorted(set(state) - required, key=str)
            raise ValueError(
                "checkpoint metadata.training_loop has invalid fields: "
                f"missing={missing or '<none>'}, unknown={unknown or '<none>'}"
            )
        best_epoch = state["best_epoch"]
        if best_epoch is not None and (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch <= 0
        ):
            raise TypeError(
                "training_loop.best_epoch must be a positive int or null"
            )
        best_metric = state["best_metric_value"]
        if best_metric is not None and (
            isinstance(best_metric, bool)
            or not isinstance(best_metric, (int, float))
        ):
            raise TypeError(
                "training_loop.best_metric_value must be numeric or null"
            )
        if best_metric is not None and not math.isfinite(float(best_metric)):
            raise ValueError(
                "training_loop.best_metric_value must be finite or null"
            )
        if (best_epoch is None) != (best_metric is None):
            raise ValueError(
                "training_loop best_epoch and best_metric_value must both be null "
                "or both be set"
            )
        wait = state["epochs_without_improvement"]
        if isinstance(wait, bool) or not isinstance(wait, int) or wait < 0:
            raise TypeError(
                "training_loop.epochs_without_improvement must be a non-negative int"
            )
        stopped_early = state["stopped_early"]
        if not isinstance(stopped_early, bool):
            raise TypeError("training_loop.stopped_early must be a bool")
        monitor = state["monitor"]
        if monitor is not None and (
            not isinstance(monitor, str) or not monitor
        ):
            raise TypeError(
                "training_loop.monitor must be a non-empty string or null"
            )
        mode = state["mode"]
        if mode is not None and (
            not isinstance(mode, str) or mode not in {"min", "max"}
        ):
            raise ValueError("training_loop.mode must be 'min', 'max', or null")
        if (monitor is None) != (mode is None):
            raise ValueError(
                "training_loop monitor and mode must both be null or both be set"
            )
        if best_epoch is not None and monitor is None:
            raise ValueError(
                "training_loop monitor and mode are required when a best "
                "checkpoint is recorded"
            )
        return cls(
            best_epoch=best_epoch,
            best_metric_value=(
                None if best_metric is None else float(best_metric)
            ),
            epochs_without_improvement=wait,
            stopped_early=stopped_early,
            monitor=monitor,
            mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the validated state using the checkpoint schema."""

        return {
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
            "monitor": self.monitor,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class OptimizerWindowResult:
    """One complete automatic optimizer-update attempt."""

    microbatch_losses: tuple[float, ...]
    loss_aggregation_weights: tuple[float, ...]
    metrics: Mapping[str, float]
    final_batch: Batch
    final_output: TrainStepOutput
    succeeded: bool
    grad_norm: float | None
    non_finite_loss_count: int = 0
    non_finite_gradient_count: int = 0

    @property
    def loss(self) -> float:
        """Return the equally weighted mean of micro-batch scalar losses."""

        return sum(self.microbatch_losses) / len(self.microbatch_losses)

    @property
    def weighted_loss_sum(self) -> float:
        """Return the detached weighted loss numerator for epoch reporting."""

        return sum(
            loss * weight
            for loss, weight in zip(
                self.microbatch_losses,
                self.loss_aggregation_weights,
                strict=True,
            )
        )

    @property
    def loss_aggregation_weight(self) -> float:
        """Return the detached epoch aggregation denominator contribution."""

        return sum(self.loss_aggregation_weights)


class Trainer:
    """Automatic optimization and experiment lifecycle wrapper.

    The trainer owns loop mechanics such as:
    - device placement
    - zeroing gradients
    - backward / optimizer step
    - optional gradient clipping
    - optional scheduler stepping

    Algorithm-specific batch handling is delegated to the injected strategy.
    """

    def __init__(
        self,
        plan: TrainingPlan,
        optimizer: Optimizer,
        *,
        device: torch.device | str,
        lr_scheduler: LRScheduler | None = None,
        lr_scheduler_interval: str = "step",
        ema: ExponentialMovingAverage | None = None,
        diagnostics: Iterable[TrainingDiagnostic] | None = None,
        metric_runtime: TrainingMetricRuntime | None = None,
        max_grad_norm: float | None = None,
        logger: ExperimentLogger | None = None,
        log_every: int = 100,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        checkpoint_config: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
        precision: PrecisionRuntime | None = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.plan = validate_training_plan(plan)
        self.strategy = self.plan.strategy
        self.model = self.plan.primary_model
        self.process = self.plan.process
        self.objective = self.plan.objective
        self.optimizer = optimizer
        self.device = torch.device(device)
        accumulation_value = cast(object, accumulate_grad_batches)
        if (
            not isinstance(accumulation_value, int)
            or isinstance(accumulation_value, bool)
            or accumulation_value <= 0
        ):
            raise ValueError("accumulate_grad_batches must be a positive integer")
        self.accumulate_grad_batches = accumulation_value
        self.precision = precision or build_precision_runtime("fp32", self.device)
        precision_value = cast(object, self.precision)
        if not isinstance(precision_value, PrecisionRuntime):
            raise TypeError("precision must be a PrecisionRuntime")
        self.precision = precision_value
        if self.precision.device_type != self.device.type:
            raise ValueError("precision runtime device must match Trainer device")
        self.trainable_parameters = plan_trainable_parameters(self.plan)
        _validate_optimizer_parameters(self.optimizer, self.trainable_parameters)
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval
        self.ema = ema
        self.ema_model = self.model
        managed_modules: dict[str, ManagedTrainingModule] = {
            "primary_model": ManagedTrainingModule(self.model),
        }
        if self.process is not None:
            managed_modules["process"] = ManagedTrainingModule(self.process)
        if self.objective is not None:
            managed_modules["objective"] = ManagedTrainingModule(self.objective)
        managed_modules.update(self.plan.auxiliary_modules)
        self.managed_modules: Mapping[str, ManagedTrainingModule] = MappingProxyType(
            managed_modules
        )
        self.diagnostics = list(diagnostics or [])
        metric_runtime_value = cast(object, metric_runtime)
        if metric_runtime_value is not None and not isinstance(
            metric_runtime_value,
            TrainingMetricRuntime,
        ):
            raise TypeError(
                "metric_runtime must be TrainingMetricRuntime or None"
            )
        self.metric_runtime = metric_runtime
        self.max_grad_norm = max_grad_norm
        self.logger = logger or NullLogger()
        self.log_every = log_every
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every = checkpoint_every
        self.checkpoint_config = checkpoint_config
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.global_step = 0
        self.best_checkpoint_path: Path | None = None
        self.best_epoch: int | None = None
        self.best_metric_value: float | None = None
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self._best_monitor: str | None = None
        self._best_mode: str | None = None
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive when provided")
        _validate_checkpoint_training_config(
            self.checkpoint_config,
            precision=self.precision,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        if self.checkpoint_manager is not None and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_manager is provided")
        if self.checkpoint_manager is not None:
            _validate_checkpoint_manager(
                self.checkpoint_manager,
                plan=self.plan,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                ema=self.ema,
                precision=self.precision,
            )
        if self.lr_scheduler_interval not in {"step", "epoch"}:
            raise ValueError("lr_scheduler_interval must be 'step' or 'epoch'")

        for name, asset in self.managed_modules.items():
            move_module_to_device(
                asset.module,
                self.device,
                role=f"training module '{name}'",
            )
        if self.ema is not None:
            self.ema.to(self.device)

    def restore_fit_state(
        self,
        state: object,
        *,
        best_checkpoint_path: str | Path | None = None,
    ) -> None:
        """Restore best-selection and early-stopping state for strict resume."""

        parsed = TrainingFitState.from_mapping(state)
        self.best_epoch = parsed.best_epoch
        self.best_metric_value = parsed.best_metric_value
        self.epochs_without_improvement = parsed.epochs_without_improvement
        self.stopped_early = parsed.stopped_early
        self._best_monitor = parsed.monitor
        self._best_mode = parsed.mode
        self.best_checkpoint_path = (
            Path(best_checkpoint_path)
            if best_checkpoint_path is not None
            else None
        )

    def _fit_state_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_metric_value": self.best_metric_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_early,
            "monitor": self._best_monitor,
            "mode": self._best_mode,
        }

    def _set_module_modes(self, *, training: bool) -> None:
        for asset in self.managed_modules.values():
            if training and asset.mode == "follow":
                asset.module.train()
            else:
                asset.module.eval()

    def _step_lr_scheduler(self, interval: str) -> None:
        if self.lr_scheduler is None or self.lr_scheduler_interval != interval:
            return
        self.lr_scheduler.step()

    def _emit_batch_diagnostics(
        self,
        *,
        batch: Batch,
        output: TrainStepOutput,
        loss: float,
        global_step: int,
        epoch_index: int | None,
    ) -> None:
        event = TrainBatchEndEvent(
            trainer=self,
            batch=batch,
            output=output,
            loss=loss,
            global_step=global_step,
            epoch_index=epoch_index,
        )
        for diagnostic in self.diagnostics:
            with preserve_global_rng_state(self.device):
                diagnostic.on_train_batch_end(event)

    def _emit_fit_start_diagnostics(
        self,
        *,
        train_dataloader: Iterable[Batch],
        validation_dataloader: Iterable[Batch] | None,
    ) -> None:
        event = FitStartEvent(
            trainer=self,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
        )
        for diagnostic in self.diagnostics:
            with preserve_global_rng_state(self.device):
                diagnostic.on_fit_start(event)

    def _emit_epoch_diagnostics(
        self,
        *,
        epoch_index: int,
        metrics: dict[str, float],
    ) -> None:
        event = TrainEpochEndEvent(
            trainer=self,
            epoch_index=epoch_index,
            metrics=metrics,
        )
        for diagnostic in self.diagnostics:
            with preserve_global_rng_state(self.device):
                diagnostic.on_train_epoch_end(event)

    def _finish_optimizer_step(self) -> tuple[bool, float | None]:
        grad_norm: float | None = None
        if self.max_grad_norm is not None:
            self.precision.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters,
                self.max_grad_norm,
            )
            grad_norm = float(norm.detach().item())
        succeeded = self.precision.step(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        if not succeeded:
            return False, grad_norm
        if self.ema is not None:
            self.ema.update(self.ema_model)
        self._step_lr_scheduler("step")
        self.global_step += 1
        return True, grad_norm

    def _run_accumulation_window(
        self,
        batches: tuple[Batch, ...],
        *,
        phase_profiler: TrainingPhaseProfiler | None = None,
        metric_phase: TrainingMetricPhase | None = None,
    ) -> OptimizerWindowResult:
        if not batches:
            raise ValueError("an accumulation window must contain a batch")
        collect_phase_metrics = (
            metric_phase is not None
            and self.metric_runtime is not None
            and self.metric_runtime.has_phase(metric_phase)
        )
        loss_tensors: list[torch.Tensor] = []
        loss_aggregation_weights: list[float] = []
        metric_update_values = []
        metric_values: dict[str, list[float | torch.Tensor]] = {}
        final_batch: Batch = batches[-1]
        final_output: TrainStepOutput | None = None
        try:
            for index, batch in enumerate(batches):
                prepared_batch = _move_to_device(batch, self.device)
                forward_started_at = (
                    phase_profiler.start_phase()
                    if phase_profiler is not None
                    else None
                )
                with self.precision.autocast():
                    output = validate_train_step_output(
                        cast(object, self.strategy.training_step(prepared_batch))
                    )
                    normalized_loss = output.loss / len(batches)
                if (
                    phase_profiler is not None
                    and forward_started_at is not None
                ):
                    phase_profiler.end_phase(
                        "forward_seconds",
                        forward_started_at,
                    )
                loss_tensors.append(output.loss.detach().clone())
                loss_aggregation_weights.append(
                    loss_aggregation_weight_to_float(
                        output.loss_aggregation_weight
                    )
                )
                if collect_phase_metrics:
                    metric_update_values.append(
                        detach_metric_updates(output.metric_updates)
                    )
                _accumulate_scalar_metrics(
                    metric_values,
                    output.metrics,
                )
                backward_started_at = (
                    phase_profiler.start_phase()
                    if phase_profiler is not None
                    else None
                )
                self.precision.backward(normalized_loss)
                if (
                    phase_profiler is not None
                    and backward_started_at is not None
                ):
                    phase_profiler.end_phase(
                        "backward_seconds",
                        backward_started_at,
                    )
                del normalized_loss
                del prepared_batch
                if index == len(batches) - 1:
                    final_output = output
                else:
                    del output
            if final_output is None:
                raise RuntimeError(
                    "accumulation window did not produce an output"
                )
            stacked_losses = (
                torch.stack(tuple(loss_tensors))
                .to(device="cpu")
                .to(dtype=torch.float64)
            )
            losses = tuple(
                float(value) for value in stacked_losses.tolist()
            )
            metrics = _mean_accumulated_metrics(metric_values)
            gradients_finite = (
                _gradients_are_finite(self.trainable_parameters)
                if phase_profiler is not None and self.max_grad_norm is None
                else None
            )
            optimizer_started_at = (
                phase_profiler.start_phase()
                if phase_profiler is not None
                else None
            )
            succeeded, grad_norm = self._finish_optimizer_step()
            if (
                succeeded
                and collect_phase_metrics
                and self.metric_runtime is not None
            ):
                assert metric_phase is not None
                for updates in metric_update_values:
                    self.metric_runtime.update_phase(metric_phase, updates)
            if (
                phase_profiler is not None
                and optimizer_started_at is not None
            ):
                phase_profiler.end_phase(
                    "optimizer_seconds",
                    optimizer_started_at,
                )
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        if grad_norm is not None:
            gradients_finite = math.isfinite(grad_norm)
        return OptimizerWindowResult(
            microbatch_losses=losses,
            loss_aggregation_weights=tuple(loss_aggregation_weights),
            metrics=metrics,
            final_batch=final_batch,
            final_output=final_output,
            succeeded=succeeded,
            grad_norm=grad_norm,
            non_finite_loss_count=sum(
                not math.isfinite(loss) for loss in losses
            ),
            non_finite_gradient_count=int(gradients_finite is False),
        )

    def train_batch(self, batch: Batch) -> float:
        """Run one complete optimizer-update attempt and return its loss."""

        self._set_module_modes(training=True)
        self.optimizer.zero_grad(set_to_none=True)
        return self._run_accumulation_window((batch,)).loss

    def train_epoch(
        self,
        dataloader: Iterable[Batch],
        *,
        epoch_index: int | None = None,
        show_progress: bool = True,
        max_batches: int | None = None,
        max_optimizer_steps: int | None = None,
        profile_phases: bool = False,
        reporter: Any | None = None,
    ) -> dict[str, float]:
        """Train for one epoch and return aggregate metrics."""

        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive when provided")
        max_optimizer_steps_value = cast(object, max_optimizer_steps)
        if (
            max_optimizer_steps_value is not None
            and (
                isinstance(max_optimizer_steps_value, bool)
                or not isinstance(max_optimizer_steps_value, int)
                or max_optimizer_steps_value <= 0
            )
        ):
            raise ValueError(
                "max_optimizer_steps must be a positive integer when provided"
            )
        profile_phases_value = cast(object, profile_phases)
        if not isinstance(profile_phases_value, bool):
            raise TypeError("profile_phases must be boolean")
        if epoch_index is not None:
            _set_dataloader_epoch(dataloader, epoch_index)

        progress_reporter = reporter
        if progress_reporter is not None:
            progress_reporter.on_phase_start(
                phase="train",
                epoch=epoch_index,
                total_batches=_resolve_total_batches(dataloader, max_batches),
                enabled=show_progress,
            )

        iterator = (
            islice(dataloader, max_batches)
            if max_batches is not None
            else iter(dataloader)
        )

        total_loss = 0.0
        total_loss_aggregation_weight = 0.0
        num_batches = 0
        optimizer_steps = 0
        skipped_optimizer_steps = 0
        non_finite_loss_count = 0
        non_finite_gradient_count = 0
        data_wait_seconds = 0.0
        compute_seconds = 0.0
        phase_profiler = (
            TrainingPhaseProfiler(self.device) if profile_phases else None
        )
        started_at = (
            phase_profiler.start_measurement()
            if phase_profiler is not None
            else time.perf_counter()
        )
        self._set_module_modes(training=True)
        self.optimizer.zero_grad(set_to_none=True)
        has_train_metrics = (
            self.metric_runtime is not None
            and self.metric_runtime.has_phase("train")
        )
        if has_train_metrics and self.metric_runtime is not None:
            self.metric_runtime.reset_phase("train")
        try:
            while True:
                data_wait_started_at = time.perf_counter()
                window = tuple(
                    islice(iterator, self.accumulate_grad_batches)
                )
                data_wait_seconds += time.perf_counter() - data_wait_started_at
                if not window:
                    break
                compute_started_at = time.perf_counter()
                result = self._run_accumulation_window(
                    window,
                    phase_profiler=phase_profiler,
                    metric_phase="train",
                )
                compute_seconds += time.perf_counter() - compute_started_at
                total_loss += result.weighted_loss_sum
                total_loss_aggregation_weight += (
                    result.loss_aggregation_weight
                )
                non_finite_loss_count += result.non_finite_loss_count
                non_finite_gradient_count += (
                    result.non_finite_gradient_count
                )
                num_batches += len(result.microbatch_losses)
                if result.succeeded:
                    optimizer_steps += 1
                    self._emit_batch_diagnostics(
                        batch=result.final_batch,
                        output=result.final_output,
                        loss=result.loss,
                        global_step=self.global_step,
                        epoch_index=epoch_index,
                    )
                else:
                    skipped_optimizer_steps += 1
                if (
                    result.succeeded
                    and self.global_step % self.log_every == 0
                ):
                    metrics = {
                        "train/step/loss": result.loss,
                        "train/epoch": (
                            float(epoch_index) if epoch_index is not None else 0.0
                        ),
                    }
                    metrics.update(
                        {
                            f"train/step/strategy/{name}": value
                            for name, value in result.metrics.items()
                        }
                    )
                    metrics.update(_optimizer_metrics(self.optimizer))
                    if result.grad_norm is not None:
                        metrics["train/grad_norm"] = result.grad_norm
                    if self.precision.grad_scaler is not None:
                        metrics["train/loss_scale"] = (
                            self.precision.grad_scaler.get_scale()
                        )
                    self.logger.log_metrics(metrics, step=self.global_step)
                if progress_reporter is not None:
                    running_total = (
                        total_loss - result.weighted_loss_sum
                    )
                    running_weight = (
                        total_loss_aggregation_weight
                        - result.loss_aggregation_weight
                    )
                    step_before_window = self.global_step - int(
                        result.succeeded
                    )
                    for index, (batch_loss, batch_weight) in enumerate(
                        zip(
                            result.microbatch_losses,
                            result.loss_aggregation_weights,
                            strict=True,
                        ),
                        start=1,
                    ):
                        running_total += batch_loss * batch_weight
                        running_weight += batch_weight
                        progress_reporter.on_batch_end(
                            phase="train",
                            loss=batch_loss,
                            avg_loss=(
                                running_total / running_weight
                                if running_weight > 0.0
                                else batch_loss
                            ),
                            lr=_first_lr(self.optimizer),
                            global_step=(
                                self.global_step
                                if index == len(result.microbatch_losses)
                                else step_before_window
                            ),
                        )
                reached_optimizer_limit = (
                    max_optimizer_steps is not None
                    and optimizer_steps >= max_optimizer_steps
                )
                del result
                if reached_optimizer_limit:
                    break
        except BaseException:
            if has_train_metrics and self.metric_runtime is not None:
                self.metric_runtime.reset_phase("train")
            raise
        finally:
            if progress_reporter is not None:
                progress_reporter.on_phase_end()

        if phase_profiler is not None:
            duration_seconds, phase_timings = (
                phase_profiler.finish_measurement(started_at)
            )
            if self.device.type == "cuda":
                compute_seconds = max(
                    0.0,
                    duration_seconds - data_wait_seconds,
                )
        else:
            duration_seconds = time.perf_counter() - started_at
            phase_timings = None
        if num_batches == 0:
            raise ValueError("dataloader yielded no batches")
        if total_loss_aggregation_weight <= 0.0:
            if has_train_metrics and self.metric_runtime is not None:
                self.metric_runtime.reset_phase("train")
            raise ValueError(
                "training epoch loss aggregation weight must be positive"
            )
        optimizer_steps_per_second = (
            optimizer_steps / duration_seconds
            if duration_seconds > 0.0
            else 0.0
        )
        epoch_metrics = {
            "loss": total_loss / total_loss_aggregation_weight,
            "num_batches": float(num_batches),
            "micro_batches": float(num_batches),
            "optimizer_steps": float(optimizer_steps),
            "skipped_optimizer_steps": float(skipped_optimizer_steps),
            "optimizer_steps_per_second": optimizer_steps_per_second,
            "data_wait_seconds": data_wait_seconds,
            "compute_seconds": compute_seconds,
            "duration_seconds": duration_seconds,
        }
        if has_train_metrics and self.metric_runtime is not None:
            if optimizer_steps > 0:
                epoch_metrics.update(
                    self.metric_runtime.compute_phase("train", reset=True)
                )
            else:
                self.metric_runtime.reset_phase("train")
        if phase_timings is not None:
            epoch_metrics.update(phase_timings)
            epoch_metrics["non_finite_loss_count"] = float(
                non_finite_loss_count
            )
            epoch_metrics["non_finite_gradient_count"] = float(
                non_finite_gradient_count
            )
        logged_epoch_metrics = {
            "train/loss": epoch_metrics["loss"],
            "system/train/num_batches": epoch_metrics["num_batches"],
            "system/train/micro_batches": epoch_metrics["micro_batches"],
            "system/train/optimizer_steps": epoch_metrics["optimizer_steps"],
            "system/train/skipped_optimizer_steps": epoch_metrics[
                "skipped_optimizer_steps"
            ],
            "system/train/optimizer_steps_per_second": epoch_metrics[
                "optimizer_steps_per_second"
            ],
            "system/train/data_wait_seconds": epoch_metrics[
                "data_wait_seconds"
            ],
            "system/train/compute_seconds": epoch_metrics["compute_seconds"],
            "system/train/duration_seconds": epoch_metrics["duration_seconds"],
        }
        logged_epoch_metrics.update(
            {
                name: value
                for name, value in epoch_metrics.items()
                if name.startswith("train/metrics/")
            }
        )
        if phase_timings is not None:
            logged_epoch_metrics.update(
                {
                    "system/train/forward_seconds": epoch_metrics[
                        "forward_seconds"
                    ],
                    "system/train/backward_seconds": epoch_metrics[
                        "backward_seconds"
                    ],
                    "system/train/optimizer_seconds": epoch_metrics[
                        "optimizer_seconds"
                    ],
                    "system/train/non_finite_loss_count": epoch_metrics[
                        "non_finite_loss_count"
                    ],
                    "system/train/non_finite_gradient_count": epoch_metrics[
                        "non_finite_gradient_count"
                    ],
                }
            )
        if self.precision.grad_scaler is not None:
            logged_epoch_metrics["system/train/loss_scale"] = (
                self.precision.grad_scaler.get_scale()
            )
        if epoch_index is not None:
            logged_epoch_metrics["system/epoch"] = float(epoch_index)
        self.logger.log_metrics(logged_epoch_metrics, step=self.global_step)
        if optimizer_steps == 0 and skipped_optimizer_steps > 0:
            overflow_warning = (
                "all optimizer windows were skipped in the training epoch; "
                f"skipped_windows={skipped_optimizer_steps}"
            )
            self.logger.log_text(
                "training/optimizer_overflow",
                overflow_warning,
                step=self.global_step,
            )
            warnings.warn(
                overflow_warning,
                RuntimeWarning,
                stacklevel=2,
            )
        return epoch_metrics

    def evaluate_epoch(
        self,
        dataloader: Iterable[Batch],
        *,
        epoch_index: int | None = None,
        show_progress: bool = True,
        max_batches: int | None = None,
        metric_prefix: str = "valid",
        log_metrics: bool = True,
        reporter: Any | None = None,
    ) -> dict[str, float]:
        """Evaluate one epoch without gradient updates and return aggregate metrics."""

        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive when provided")
        metric_phase_by_prefix: dict[str, TrainingMetricPhase] = {
            "valid": "validation",
            "test": "test",
        }
        metric_phase = metric_phase_by_prefix.get(metric_prefix)
        has_phase_metrics = (
            self.metric_runtime is not None
            and metric_phase is not None
            and self.metric_runtime.has_phase(metric_phase)
        )

        progress_reporter = reporter
        if progress_reporter is not None:
            progress_reporter.on_phase_start(
                phase=metric_prefix,
                epoch=epoch_index,
                total_batches=_resolve_total_batches(dataloader, max_batches),
                enabled=show_progress,
            )

        iterator = (
            islice(dataloader, max_batches)
            if max_batches is not None
            else iter(dataloader)
        )

        self._set_module_modes(training=False)
        total_loss = 0.0
        total_loss_aggregation_weight = 0.0
        num_batches = 0
        started_at = time.perf_counter()
        if has_phase_metrics and self.metric_runtime is not None:
            assert metric_phase is not None
            self.metric_runtime.reset_phase(metric_phase)
        try:
            with torch.no_grad():
                for batch in iterator:
                    prepared_batch = _move_to_device(batch, self.device)
                    with self.precision.autocast():
                        output = validate_train_step_output(
                            cast(
                                object,
                                self.strategy.evaluation_step(prepared_batch),
                            )
                        )
                    batch_loss = float(output.loss.detach().item())
                    batch_weight = loss_aggregation_weight_to_float(
                        output.loss_aggregation_weight
                    )
                    total_loss += batch_loss * batch_weight
                    total_loss_aggregation_weight += batch_weight
                    if has_phase_metrics and self.metric_runtime is not None:
                        assert metric_phase is not None
                        self.metric_runtime.update_phase(
                            metric_phase,
                            output.metric_updates,
                        )
                    num_batches += 1
                    running_loss = (
                        total_loss / total_loss_aggregation_weight
                        if total_loss_aggregation_weight > 0.0
                        else batch_loss
                    )
                    if progress_reporter is not None:
                        progress_reporter.on_batch_end(
                            phase=metric_prefix,
                            loss=batch_loss,
                            avg_loss=running_loss,
                            lr=_first_lr(self.optimizer),
                            global_step=self.global_step,
                        )
        except BaseException:
            if has_phase_metrics and self.metric_runtime is not None:
                assert metric_phase is not None
                self.metric_runtime.reset_phase(metric_phase)
            raise
        finally:
            if progress_reporter is not None:
                progress_reporter.on_phase_end()

        if num_batches == 0:
            raise ValueError("dataloader yielded no batches")
        if total_loss_aggregation_weight <= 0.0:
            if has_phase_metrics and self.metric_runtime is not None:
                assert metric_phase is not None
                self.metric_runtime.reset_phase(metric_phase)
            raise ValueError(
                "evaluation epoch loss aggregation weight must be positive"
            )

        epoch_metrics = {
            "loss": total_loss / total_loss_aggregation_weight,
            "num_batches": float(num_batches),
            "duration_seconds": time.perf_counter() - started_at,
        }
        if has_phase_metrics and self.metric_runtime is not None:
            assert metric_phase is not None
            epoch_metrics.update(
                self.metric_runtime.compute_phase(metric_phase, reset=True)
            )
        if log_metrics:
            logged_epoch_metrics = {
                f"{metric_prefix}/loss": epoch_metrics["loss"],
                f"system/{metric_prefix}/num_batches": epoch_metrics[
                    "num_batches"
                ],
                f"system/{metric_prefix}/duration_seconds": epoch_metrics[
                    "duration_seconds"
                ],
            }
            logged_epoch_metrics.update(
                {
                    name: value
                    for name, value in epoch_metrics.items()
                    if name.startswith(f"{metric_prefix}/metrics/")
                }
            )
            if epoch_index is not None:
                logged_epoch_metrics["system/epoch"] = float(epoch_index)
            self.logger.log_metrics(logged_epoch_metrics, step=self.global_step)
        return epoch_metrics

    def fit(
        self,
        dataloader: Iterable[Batch],
        *,
        num_epochs: int,
        show_progress: bool = True,
        max_batches_per_epoch: int | None = None,
        validation_dataloader: Iterable[Batch] | None = None,
        max_validation_batches: int | None = None,
        start_epoch: int = 1,
        close_logger: bool = True,
        early_stopping_patience: int | None = None,
        early_stopping_monitor: str = "valid/loss",
        early_stopping_mode: str = "min",
        early_stopping_min_delta: float = 0.0,
        best_checkpoint_filename: str = "best.pt",
        reporter: Any | None = None,
        track_best: bool | None = None,
    ) -> list[dict[str, float]]:
        """Train for multiple epochs and return per-epoch metric summaries."""

        if start_epoch <= 0:
            raise ValueError("start_epoch must be positive")
        if start_epoch > num_epochs:
            raise ValueError("start_epoch must be less than or equal to num_epochs")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when provided")
        if early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be 'min' or 'max'")
        if early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        validate_training_monitor_key(
            early_stopping_monitor,
            path="early_stopping_monitor",
        )
        if early_stopping_patience is not None and track_best is False:
            raise ValueError("early stopping requires best tracking")
        should_track_best = (
            validation_dataloader is not None if track_best is None else track_best
        )
        if early_stopping_patience is not None:
            should_track_best = True
        if should_track_best and early_stopping_monitor.startswith("test/"):
            raise ValueError("test metrics cannot be used for best tracking")

        history: list[dict[str, float]] = []
        if start_epoch == 1:
            self.best_checkpoint_path = None
            self.best_epoch = None
            self.best_metric_value = None
            self.epochs_without_improvement = 0
            self._best_monitor = None
            self._best_mode = None
        elif self._best_monitor is not None and (
            self._best_monitor != early_stopping_monitor
            or self._best_mode != early_stopping_mode
        ):
            raise ValueError(
                "restored best-tracking monitor/mode do not match this fit"
            )
        if start_epoch > 1 and self.stopped_early:
            raise ValueError(
                "restored training already stopped early and cannot be strictly "
                "resumed"
            )
        best_value = self.best_metric_value
        epochs_without_improvement = self.epochs_without_improvement
        self._best_monitor = early_stopping_monitor if should_track_best else None
        self._best_mode = early_stopping_mode if should_track_best else None
        self.stopped_early = False
        try:
            self._emit_fit_start_diagnostics(
                train_dataloader=dataloader,
                validation_dataloader=validation_dataloader,
            )
            for epoch in range(start_epoch, num_epochs + 1):
                if reporter is not None:
                    reporter.on_epoch_start(epoch, num_epochs)
                train_metrics = self.train_epoch(
                    dataloader,
                    epoch_index=epoch,
                    show_progress=show_progress,
                    max_batches=max_batches_per_epoch,
                    reporter=reporter,
                )
                validation_metrics: dict[str, float] | None = None
                if validation_dataloader is not None:
                    validation_metrics = self.evaluate_epoch(
                        validation_dataloader,
                        epoch_index=epoch,
                        show_progress=show_progress,
                        max_batches=max_validation_batches,
                        metric_prefix="valid",
                        reporter=reporter,
                    )
                successful_updates = train_metrics.get(
                    "optimizer_steps",
                    train_metrics["num_batches"],
                )
                if successful_updates > 0:
                    self._step_lr_scheduler("epoch")
                snapshot = _epoch_metric_snapshot(
                    train_metrics,
                    validation_metrics,
                    epoch=epoch,
                )
                history.append(dict(snapshot.values))

                status = "-"
                if should_track_best:
                    current_value = snapshot.values.get(
                        early_stopping_monitor
                    )
                    if current_value is None:
                        raise ValueError(
                            f"best tracking monitor '{early_stopping_monitor}' "
                            "was not found in epoch metrics"
                        )
                    source = snapshot.sources[early_stopping_monitor]
                    if (
                        not source.selection_eligible
                        or source.data_role == "test"
                    ):
                        raise ValueError(
                            f"best tracking monitor '{early_stopping_monitor}' "
                            "is not selection eligible"
                        )
                    if not math.isfinite(float(current_value)):
                        raise ValueError(
                            f"best tracking monitor '{early_stopping_monitor}' "
                            f"is non-finite at epoch {epoch}"
                        )
                    improved = self._is_metric_improved(
                        current=float(current_value),
                        best=best_value,
                        mode=early_stopping_mode,
                        min_delta=early_stopping_min_delta,
                    )
                    if improved:
                        best_value = float(current_value)
                        epochs_without_improvement = 0
                        self.epochs_without_improvement = 0
                        status = "BEST"
                        self.best_epoch = epoch
                        self.best_metric_value = best_value
                        self.best_checkpoint_path = self._save_named_checkpoint(
                            best_checkpoint_filename,
                            epoch=epoch,
                            snapshot=snapshot,
                            metadata={
                                **self.checkpoint_metadata,
                                "checkpoint_kind": "best",
                                "monitor": early_stopping_monitor,
                                "mode": early_stopping_mode,
                            },
                        )
                        with preserve_global_rng_state(self.device):
                            self.logger.log_metrics(
                                {
                                    "best/epoch": float(epoch),
                                    f"best/{early_stopping_monitor}": best_value,
                                },
                                step=self.global_step,
                            )
                    else:
                        if early_stopping_patience is not None:
                            epochs_without_improvement += 1
                            self.epochs_without_improvement = (
                                epochs_without_improvement
                            )
                            status = (
                                f"WAIT {epochs_without_improvement}/"
                                f"{early_stopping_patience}"
                            )
                        if (
                            early_stopping_patience is not None
                            and epochs_without_improvement >= early_stopping_patience
                        ):
                            self.stopped_early = True
                            status = "EARLY STOP"
                            early_stopping_text = (
                                f"stopped at epoch {epoch}; best_epoch="
                                f"{self.best_epoch}; monitor="
                                f"{early_stopping_monitor}; best="
                                f"{self.best_metric_value}"
                            )
                            self.logger.log_text(
                                "early_stopping",
                                early_stopping_text,
                                step=self.global_step,
                            )
                            if reporter is not None:
                                reporter.on_early_stopping(early_stopping_text)
                self._maybe_save_checkpoint(epoch, snapshot)
                self._save_latest_checkpoint(epoch, snapshot)
                self._emit_epoch_diagnostics(
                    epoch_index=epoch,
                    metrics=dict(snapshot.values),
                )
                if reporter is not None:
                    with preserve_global_rng_state(self.device):
                        reporter.on_epoch_end(
                            epoch=epoch,
                            total_epochs=num_epochs,
                            train_loss=train_metrics["loss"],
                            valid_loss=(
                                validation_metrics["loss"]
                                if validation_metrics is not None
                                else None
                            ),
                            best_metric_value=best_value,
                            lr=_first_lr(self.optimizer),
                            train_batches=int(train_metrics["num_batches"]),
                            valid_batches=(
                                int(validation_metrics["num_batches"])
                                if validation_metrics is not None
                                else None
                            ),
                            epoch_time=train_metrics["duration_seconds"]
                            + (
                                validation_metrics["duration_seconds"]
                                if validation_metrics is not None
                                else 0.0
                            ),
                            status=status,
                        )
                if self.stopped_early:
                    break
        finally:
            if close_logger:
                self.logger.close()
        return history

    @staticmethod
    def _is_metric_improved(
        *,
        current: float,
        best: float | None,
        mode: str,
        min_delta: float,
    ) -> bool:
        """Return whether a monitored metric improved enough to count."""

        if best is None:
            return True
        if mode == "min":
            return current < best - min_delta
        return current > best + min_delta

    def _maybe_save_checkpoint(
        self,
        epoch: int,
        snapshot: EpochMetricSnapshot,
    ) -> None:
        """Save an epoch checkpoint when checkpointing is configured."""

        if self.checkpoint_manager is None or self.checkpoint_every is None:
            return
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is required for checkpoint saving")
        if epoch % self.checkpoint_every != 0:
            return

        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        self._save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            snapshot=snapshot,
            metadata=self.checkpoint_metadata,
        )

    def _save_latest_checkpoint(
        self,
        epoch: int,
        snapshot: EpochMetricSnapshot,
    ) -> None:
        """Save a stable latest checkpoint after every completed epoch."""

        if self.checkpoint_manager is None:
            return
        self._save_named_checkpoint(
            "latest.pt",
            epoch=epoch,
            snapshot=snapshot,
            metadata={**self.checkpoint_metadata, "checkpoint_kind": "latest"},
        )

    def _save_named_checkpoint(
        self,
        filename: str,
        *,
        epoch: int,
        snapshot: EpochMetricSnapshot,
        metadata: dict[str, Any],
    ) -> Path:
        """Save a checkpoint with a stable filename in the checkpoint directory."""

        if self.checkpoint_manager is None:
            raise RuntimeError("checkpoint_manager is required for checkpoint saving")
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is required for checkpoint saving")
        return self._save_checkpoint(
            self.checkpoint_dir / filename,
            epoch=epoch,
            snapshot=snapshot,
            metadata=metadata,
        )

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        epoch: int,
        snapshot: EpochMetricSnapshot,
        metadata: dict[str, Any],
    ) -> Path:
        """Save a checkpoint using the trainer's common runtime metadata."""

        if self.checkpoint_manager is None:
            raise RuntimeError("checkpoint_manager is required for checkpoint saving")
        return self.checkpoint_manager.save(
            checkpoint_path,
            epoch=epoch,
            global_step=self.global_step,
            config=self.checkpoint_config,
            metrics=dict(snapshot.values),
            metadata={
                **metadata,
                "metric_sources": _checkpoint_metric_sources(snapshot),
                "training_loop": self._fit_state_dict(),
            },
        )
