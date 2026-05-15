"""Generic training loop utilities."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm.auto import tqdm

from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.logging import ExperimentLogger, NullLogger

Batch = Any
TrainStepFn = Callable[[nn.Module, nn.Module, Batch, torch.device], torch.Tensor]


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
        self.max_grad_norm = max_grad_norm
        self.logger = logger or NullLogger()
        self.log_every = log_every
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_every = checkpoint_every
        self.checkpoint_config = checkpoint_config
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.global_step = 0

        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive when provided")
        if self.checkpoint_manager is not None and self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_manager is provided")

        self.model.to(self.device)

    def train_batch(self, batch: Batch) -> float:
        """Run one optimization step and return the scalar loss."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.train_step_fn(self.model, self.criterion, batch, self.device)
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return float(loss.detach().item())

    def train_epoch(
        self,
        dataloader: Iterable[Batch],
        *,
        epoch_index: int | None = None,
        show_progress: bool = True,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Train for one epoch and return aggregate metrics."""

        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive when provided")

        iterator = dataloader
        if show_progress:
            desc = f"Epoch {epoch_index}" if epoch_index is not None else "Train"
            iterator = tqdm(dataloader, desc=desc, total=max_batches)

        total_loss = 0.0
        num_batches = 0
        for batch in iterator:
            if max_batches is not None and num_batches >= max_batches:
                break
            batch_loss = self.train_batch(batch)
            total_loss += batch_loss
            num_batches += 1
            self.global_step += 1
            if self.global_step % self.log_every == 0:
                metrics = {
                    "train/loss": batch_loss,
                    "train/epoch": float(epoch_index) if epoch_index is not None else 0.0,
                }
                metrics.update(_optimizer_metrics(self.optimizer))
                self.logger.log_metrics(metrics, step=self.global_step)
            if show_progress:
                iterator.set_postfix(loss=f"{batch_loss:.4f}")  # type: ignore[attr-defined]

        if num_batches == 0:
            raise ValueError("dataloader yielded no batches")

        epoch_metrics = {
            "loss": total_loss / num_batches,
            "num_batches": float(num_batches),
        }
        logged_epoch_metrics = {
            "train/epoch_loss": epoch_metrics["loss"],
            "train/epoch_batches": epoch_metrics["num_batches"],
        }
        if epoch_index is not None:
            logged_epoch_metrics["train/epoch"] = float(epoch_index)
        self.logger.log_metrics(logged_epoch_metrics, step=self.global_step)
        return epoch_metrics

    def fit(
        self,
        dataloader: Iterable[Batch],
        *,
        num_epochs: int,
        show_progress: bool = True,
        max_batches_per_epoch: int | None = None,
        start_epoch: int = 1,
    ) -> list[dict[str, float]]:
        """Train for multiple epochs and return per-epoch metric summaries."""

        if start_epoch <= 0:
            raise ValueError("start_epoch must be positive")
        if start_epoch > num_epochs:
            raise ValueError("start_epoch must be less than or equal to num_epochs")

        history: list[dict[str, float]] = []
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                metrics = self.train_epoch(
                    dataloader,
                    epoch_index=epoch,
                    show_progress=show_progress,
                    max_batches=max_batches_per_epoch,
                )
                history.append(metrics)
                self._maybe_save_checkpoint(epoch, metrics)
        finally:
            self.logger.close()
        return history

    def _maybe_save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save an epoch checkpoint when checkpointing is configured."""

        if self.checkpoint_manager is None or self.checkpoint_every is None:
            return
        if self.checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is required for checkpoint saving")
        if epoch % self.checkpoint_every != 0:
            return

        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        self.checkpoint_manager.save(
            checkpoint_path,
            epoch=epoch,
            global_step=self.global_step,
            config=self.checkpoint_config,
            metrics=metrics,
            metadata=self.checkpoint_metadata,
        )
