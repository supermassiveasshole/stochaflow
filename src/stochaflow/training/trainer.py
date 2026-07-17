"""Generic training loop utilities."""

import time
from collections.abc import Callable, Iterable, Mapping, Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from stochaflow.training.diagnostics.contracts import (
    FitStartEvent,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainingDiagnostic,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.logging import ExperimentLogger, NullLogger

Batch = Any


@dataclass(slots=True)
class TrainStepOutput:
    """Structured result from an algorithm-specific training step."""

    loss: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


TrainStepResult = torch.Tensor | TrainStepOutput
TrainStepFn = Callable[[nn.Module, nn.Module, Batch, torch.device], TrainStepResult]


def _move_to_device(batch: Batch, device: torch.device) -> Batch:
    """Recursively move tensors in a batch structure onto a device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [_move_to_device(item, device) for item in batch]
    return batch


def _default_train_step(
    model: nn.Module,
    criterion: nn.Module,
    batch: Batch,
    device: torch.device,
) -> torch.Tensor:
    """Run a default supervised train step for ``(inputs, targets)`` batches."""

    batch = _move_to_device(batch, device)
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(
            "default train step expects batches shaped like (inputs, targets); "
            "provide a custom train_step_fn for other batch formats"
        )
    inputs, targets = batch
    predictions = model(inputs)
    return criterion(predictions, targets)


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

    if isinstance(dataloader, Sized):
        total = len(dataloader)
        if max_batches is not None:
            return min(total, max_batches)
        return total
    return max_batches


def _normalize_train_step_result(result: TrainStepResult) -> TrainStepOutput:
    if isinstance(result, TrainStepOutput):
        return result
    if isinstance(result, torch.Tensor):
        return TrainStepOutput(loss=result)
    raise TypeError("train_step_fn must return a Tensor or TrainStepOutput")


class Trainer:
    """Generic optimization loop wrapper.

    The trainer owns loop mechanics such as:
    - device placement
    - zeroing gradients
    - backward / optimizer step
    - optional gradient clipping
    - optional scheduler stepping

    Algorithm-specific batch handling is delegated to ``train_step_fn`` so the
    trainer itself stays generic.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        *,
        device: torch.device | str,
        train_step_fn: TrainStepFn | None = None,
        lr_scheduler: Any | None = None,
        lr_scheduler_interval: str = "step",
        ema: ExponentialMovingAverage | None = None,
        diagnostics: Iterable[TrainingDiagnostic] | None = None,
        max_grad_norm: float | None = None,
        logger: ExperimentLogger | None = None,
        log_every: int = 100,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        checkpoint_config: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.train_step_fn = train_step_fn or _default_train_step
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval
        self.ema = ema
        self.diagnostics = list(diagnostics or [])
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
        self.stopped_early = False
        self._last_train_step_output: TrainStepOutput | None = None

        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive when provided")
        if self.checkpoint_manager is not None and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_manager is provided")
        if self.lr_scheduler_interval not in {"step", "epoch"}:
            raise ValueError("lr_scheduler_interval must be 'step' or 'epoch'")

        self.model.to(self.device)
        if self.ema is not None:
            self.ema.to(self.device)

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
            diagnostic.on_train_epoch_end(event)

    def train_batch(self, batch: Batch) -> float:
        """Run one optimization step and return the scalar loss."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = _normalize_train_step_result(
            self.train_step_fn(self.model, self.criterion, batch, self.device)
        )
        loss = output.loss
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self._step_lr_scheduler("step")

        self._last_train_step_output = output
        return float(loss.detach().item())

    def train_epoch(
        self,
        dataloader: Iterable[Batch],
        *,
        epoch_index: int | None = None,
        show_progress: bool = True,
        max_batches: int | None = None,
        reporter: Any | None = None,
    ) -> dict[str, float]:
        """Train for one epoch and return aggregate metrics."""

        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive when provided")
        if epoch_index is not None:
            batch_sampler = getattr(dataloader, "batch_sampler", None)
            set_epoch = getattr(batch_sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch_index)

        progress_reporter = reporter
        if progress_reporter is not None:
            progress_reporter.on_phase_start(
                phase="train",
                epoch=epoch_index,
                total_batches=_resolve_total_batches(dataloader, max_batches),
                enabled=show_progress,
            )

        iterator = dataloader

        total_loss = 0.0
        num_batches = 0
        started_at = time.perf_counter()
        try:
            for batch in iterator:
                if max_batches is not None and num_batches >= max_batches:
                    break
                batch_loss = self.train_batch(batch)
                train_step_output = self._last_train_step_output
                if train_step_output is None:
                    raise RuntimeError("train_batch did not produce a TrainStepOutput")
                total_loss += batch_loss
                num_batches += 1
                self.global_step += 1
                self._emit_batch_diagnostics(
                    batch=batch,
                    output=train_step_output,
                    loss=batch_loss,
                    global_step=self.global_step,
                    epoch_index=epoch_index,
                )
                running_loss = total_loss / num_batches
                if self.global_step % self.log_every == 0:
                    metrics = {
                        "train/loss": batch_loss,
                        "train/epoch": (
                            float(epoch_index) if epoch_index is not None else 0.0
                        ),
                    }
                    metrics.update(_optimizer_metrics(self.optimizer))
                    self.logger.log_metrics(metrics, step=self.global_step)
                if progress_reporter is not None:
                    progress_reporter.on_batch_end(
                        phase="train",
                        loss=batch_loss,
                        avg_loss=running_loss,
                        lr=_first_lr(self.optimizer),
                        global_step=self.global_step,
                    )
        finally:
            if progress_reporter is not None:
                progress_reporter.on_phase_end()

        if num_batches == 0:
            raise ValueError("dataloader yielded no batches")

        epoch_metrics = {
            "loss": total_loss / num_batches,
            "num_batches": float(num_batches),
            "duration_seconds": time.perf_counter() - started_at,
        }
        logged_epoch_metrics = {
            "train/epoch_loss": epoch_metrics["loss"],
            "train/epoch_batches": epoch_metrics["num_batches"],
            "train/epoch_duration_seconds": epoch_metrics["duration_seconds"],
        }
        if epoch_index is not None:
            logged_epoch_metrics["train/epoch"] = float(epoch_index)
        self.logger.log_metrics(logged_epoch_metrics, step=self.global_step)
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

        progress_reporter = reporter
        if progress_reporter is not None:
            progress_reporter.on_phase_start(
                phase=metric_prefix,
                epoch=epoch_index,
                total_batches=_resolve_total_batches(dataloader, max_batches),
                enabled=show_progress,
            )

        iterator = dataloader

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        started_at = time.perf_counter()
        try:
            with torch.no_grad():
                for batch in iterator:
                    if max_batches is not None and num_batches >= max_batches:
                        break
                    output = _normalize_train_step_result(
                        self.train_step_fn(
                            self.model,
                            self.criterion,
                            batch,
                            self.device,
                        )
                    )
                    batch_loss = float(output.loss.detach().item())
                    total_loss += batch_loss
                    num_batches += 1
                    running_loss = total_loss / num_batches
                    if progress_reporter is not None:
                        progress_reporter.on_batch_end(
                            phase=metric_prefix,
                            loss=batch_loss,
                            avg_loss=running_loss,
                            lr=_first_lr(self.optimizer),
                            global_step=self.global_step,
                        )
        finally:
            if progress_reporter is not None:
                progress_reporter.on_phase_end()

        if num_batches == 0:
            raise ValueError("dataloader yielded no batches")

        epoch_metrics = {
            "loss": total_loss / num_batches,
            "num_batches": float(num_batches),
            "duration_seconds": time.perf_counter() - started_at,
        }
        if log_metrics:
            logged_epoch_metrics = {
                f"{metric_prefix}/epoch_loss": epoch_metrics["loss"],
                f"{metric_prefix}/epoch_batches": epoch_metrics["num_batches"],
                f"{metric_prefix}/epoch_duration_seconds": epoch_metrics[
                    "duration_seconds"
                ],
            }
            if epoch_index is not None:
                logged_epoch_metrics[f"{metric_prefix}/epoch"] = float(epoch_index)
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
        early_stopping_monitor: str = "valid_loss",
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
        if early_stopping_patience is not None and track_best is False:
            raise ValueError("early stopping requires best tracking")
        should_track_best = (
            validation_dataloader is not None if track_best is None else track_best
        )
        if early_stopping_patience is not None:
            should_track_best = True

        history: list[dict[str, float]] = []
        best_value: float | None = None
        epochs_without_improvement = 0
        self.best_checkpoint_path = None
        self.best_epoch = None
        self.best_metric_value = None
        self.stopped_early = False
        try:
            self._emit_fit_start_diagnostics(
                train_dataloader=dataloader,
                validation_dataloader=validation_dataloader,
            )
            for epoch in range(start_epoch, num_epochs + 1):
                if reporter is not None:
                    reporter.on_epoch_start(epoch, num_epochs)
                metrics = self.train_epoch(
                    dataloader,
                    epoch_index=epoch,
                    show_progress=show_progress,
                    max_batches=max_batches_per_epoch,
                    reporter=reporter,
                )
                metrics["train_loss"] = metrics["loss"]
                if validation_dataloader is not None:
                    validation_metrics = self.evaluate_epoch(
                        validation_dataloader,
                        epoch_index=epoch,
                        show_progress=show_progress,
                        max_batches=max_validation_batches,
                        metric_prefix="valid",
                        reporter=reporter,
                    )
                    metrics = {
                        **metrics,
                        "valid_loss": validation_metrics["loss"],
                        "valid_num_batches": validation_metrics["num_batches"],
                        "valid_duration_seconds": validation_metrics[
                            "duration_seconds"
                        ],
                    }
                self._step_lr_scheduler("epoch")
                history.append(metrics)
                self._maybe_save_checkpoint(epoch, metrics)
                self._save_latest_checkpoint(epoch, metrics)

                status = "-"
                if should_track_best:
                    current_value = metrics.get(early_stopping_monitor)
                    if current_value is None:
                        raise ValueError(
                            f"best tracking monitor '{early_stopping_monitor}' "
                            "was not found in epoch metrics"
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
                        status = "BEST"
                        self.best_epoch = epoch
                        self.best_metric_value = best_value
                        self.best_checkpoint_path = self._save_named_checkpoint(
                            best_checkpoint_filename,
                            epoch=epoch,
                            metrics=metrics,
                            metadata={
                                **self.checkpoint_metadata,
                                "checkpoint_kind": "best",
                                "monitor": early_stopping_monitor,
                                "mode": early_stopping_mode,
                            },
                        )
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
                self._emit_epoch_diagnostics(epoch_index=epoch, metrics=metrics)
                if reporter is not None:
                    reporter.on_epoch_end(
                        epoch=epoch,
                        total_epochs=num_epochs,
                        train_loss=metrics["loss"],
                        valid_loss=metrics.get("valid_loss"),
                        best_valid_loss=best_value,
                        lr=_first_lr(self.optimizer),
                        train_batches=int(metrics["num_batches"]),
                        valid_batches=(
                            int(metrics["valid_num_batches"])
                            if "valid_num_batches" in metrics
                            else None
                        ),
                        epoch_time=metrics["duration_seconds"]
                        + metrics.get("valid_duration_seconds", 0.0),
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

    def _maybe_save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
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
            metrics=metrics,
            metadata=self.checkpoint_metadata,
        )

    def _save_latest_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save a stable latest checkpoint after every completed epoch."""

        if self.checkpoint_manager is None:
            return
        self._save_named_checkpoint(
            "latest.pt",
            epoch=epoch,
            metrics=metrics,
            metadata={**self.checkpoint_metadata, "checkpoint_kind": "latest"},
        )

    def _save_named_checkpoint(
        self,
        filename: str,
        *,
        epoch: int,
        metrics: dict[str, float],
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
            metrics=metrics,
            metadata=metadata,
        )

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        *,
        epoch: int,
        metrics: dict[str, float],
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
            metrics=metrics,
            metadata=metadata,
        )
