"""Tests for trainer reporting and validation checkpoint behavior."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stochaflow.training import Trainer
from stochaflow.utils.checkpoint import CheckpointManager


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class RecordingReporter:
    def __init__(self) -> None:
        self.phase_enabled: list[bool] = []
        self.epoch_summaries = 0

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        del epoch, total_epochs

    def on_phase_start(
        self,
        *,
        phase: str,
        epoch: int | None,
        total_batches: int | None,
        enabled: bool = True,
    ) -> None:
        del phase, epoch, total_batches
        self.phase_enabled.append(enabled)

    def on_batch_end(
        self,
        *,
        phase: str,
        loss: float,
        avg_loss: float,
        lr: float | None,
        global_step: int,
    ) -> None:
        del phase, loss, avg_loss, lr, global_step

    def on_phase_end(self) -> None:
        return None

    def on_epoch_end(self, **kwargs) -> None:
        del kwargs
        self.epoch_summaries += 1


def _make_loader() -> DataLoader:
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)


def _make_trainer(tmp_path) -> Trainer:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_manager = CheckpointManager(model=model, optimizer=optimizer)
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device="cpu",
        checkpoint_manager=checkpoint_manager,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )


def test_fit_reports_epoch_summary_when_progress_bars_are_disabled(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    reporter = RecordingReporter()

    trainer.fit(
        _make_loader(),
        num_epochs=1,
        validation_dataloader=_make_loader(),
        show_progress=False,
        reporter=reporter,
    )

    assert reporter.phase_enabled == [False, False]
    assert reporter.epoch_summaries == 1


def test_fit_tracks_best_checkpoint_without_early_stopping(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)

    trainer.fit(
        _make_loader(),
        num_epochs=1,
        validation_dataloader=_make_loader(),
        show_progress=False,
        early_stopping_patience=None,
    )

    assert trainer.best_epoch == 1
    assert trainer.best_metric_value is not None
    best_checkpoint_path = trainer.best_checkpoint_path
    assert best_checkpoint_path is not None
    assert best_checkpoint_path == tmp_path / "checkpoints" / "best.pt"
    assert best_checkpoint_path.exists()
