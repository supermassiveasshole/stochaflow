"""Tests for trainer reporting and validation checkpoint behavior."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stochaflow.training import (
    FitStartEvent,
    Trainer,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainStepOutput,
    TrainingDiagnostic,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import CheckpointManager


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class ModuleWithFloatingBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running", torch.tensor([2.0]))


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


class CountingScheduler:
    def __init__(self) -> None:
        self.count = 0

    def step(self) -> None:
        self.count += 1

    def state_dict(self) -> dict[str, int]:
        return {"count": self.count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.count = state["count"]


class EpochRecorder:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class EpochAwareLoader:
    def __init__(self) -> None:
        self.sampler = EpochRecorder()
        self.batch_sampler = self.sampler

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield torch.tensor([[1.0]]), torch.tensor([[1.0]])


class RecordingDiagnostic(TrainingDiagnostic):
    def __init__(self) -> None:
        self.fit_started = False
        self.batch_steps: list[int] = []
        self.epoch_indices: list[int] = []

    def on_fit_start(self, event: FitStartEvent) -> None:
        assert event.validation_dataloader is None
        self.fit_started = True

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        assert isinstance(event.output, TrainStepOutput)
        assert "custom" in event.output.diagnostics
        self.batch_steps.append(event.global_step)

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        self.epoch_indices.append(event.epoch_index)


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


def _make_trainer_with_scheduler(
    tmp_path,
    scheduler: CountingScheduler,
    *,
    interval: str,
) -> Trainer:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device="cpu",
        lr_scheduler=scheduler,
        lr_scheduler_interval=interval,
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


def test_fit_can_track_best_train_loss_without_validation(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)

    trainer.fit(
        _make_loader(),
        num_epochs=1,
        show_progress=False,
        early_stopping_monitor="train_loss",
        track_best=True,
    )

    assert trainer.best_epoch == 1
    assert trainer.best_metric_value is not None
    best_checkpoint_path = trainer.best_checkpoint_path
    assert best_checkpoint_path is not None
    assert best_checkpoint_path == tmp_path / "checkpoints" / "best.pt"
    assert best_checkpoint_path.exists()


def test_fit_early_stopping_enables_train_loss_tracking_without_validation(
    tmp_path,
) -> None:
    trainer = _make_trainer(tmp_path)

    trainer.fit(
        _make_loader(),
        num_epochs=1,
        show_progress=False,
        early_stopping_patience=1,
        early_stopping_monitor="train_loss",
    )

    assert trainer.best_epoch == 1
    assert trainer.best_metric_value is not None
    assert trainer.best_checkpoint_path == tmp_path / "checkpoints" / "best.pt"


def test_step_lr_scheduler_steps_once_per_batch(tmp_path) -> None:
    scheduler = CountingScheduler()
    trainer = _make_trainer_with_scheduler(tmp_path, scheduler, interval="step")

    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)

    assert scheduler.count == 2


def test_structured_train_step_runs_diagnostics_hooks(tmp_path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    diagnostic = RecordingDiagnostic()

    def train_step(
        model: nn.Module,
        criterion: nn.Module,
        batch,
        device: torch.device,
    ) -> TrainStepOutput:
        inputs, targets = batch
        predictions = model(inputs.to(device))
        loss = criterion(predictions, targets.to(device))
        return TrainStepOutput(loss=loss, diagnostics={"custom": torch.tensor(1.0)})

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device="cpu",
        train_step_fn=train_step,
        diagnostics=[diagnostic],
        checkpoint_manager=CheckpointManager(model=model, optimizer=optimizer),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )

    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)

    assert diagnostic.fit_started
    assert diagnostic.batch_steps == [1, 2]
    assert diagnostic.epoch_indices == [1]
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_structured_batch_reaches_custom_train_step(tmp_path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    observed: list[object] = []

    def train_step(
        model: nn.Module,
        criterion: nn.Module,
        batch,
        device: torch.device,
    ) -> torch.Tensor:
        observed.append(batch)
        assert batch["state"].device == device
        assert batch["condition"]["scale"].device == device
        assert batch["metadata"] == {"source": "physics"}
        prediction = model(batch["state"])
        return criterion(prediction, batch["target"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device="cpu",
        train_step_fn=train_step,
        checkpoint_manager=CheckpointManager(model=model, optimizer=optimizer),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )
    batch = {
        "state": torch.tensor([[1.0]]),
        "target": torch.tensor([[0.0]]),
        "condition": {"scale": torch.tensor([2.0])},
        "metadata": {"source": "physics"},
    }

    trainer.train_epoch([batch], show_progress=False)

    assert len(observed) == 1


def test_ema_updates_once_per_train_batch(tmp_path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ema = ExponentialMovingAverage(model, decay=0.0)
    checkpoint_manager = CheckpointManager(model=model, optimizer=optimizer, ema=ema)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device="cpu",
        ema=ema,
        checkpoint_manager=checkpoint_manager,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )

    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)

    assert ema.num_updates == 2
    assert "ema_state_dict" in torch.load(
        tmp_path / "checkpoints" / "epoch_0001.pt",
        weights_only=False,
    )


def test_ema_store_restore_round_trips_floating_buffers() -> None:
    model = ModuleWithFloatingBuffer()
    ema = ExponentialMovingAverage(model, decay=0.0)
    ema.shadow_params["weight"].fill_(10.0)
    ema.shadow_buffers["running"].fill_(20.0)

    ema.store(model)
    ema.copy_to(model)
    assert model.weight.item() == pytest.approx(10.0)
    assert model.get_buffer("running").item() == pytest.approx(20.0)

    ema.restore(model)

    assert model.weight.item() == pytest.approx(1.0)
    assert model.get_buffer("running").item() == pytest.approx(2.0)


def test_epoch_lr_scheduler_steps_once_per_epoch(tmp_path) -> None:
    scheduler = CountingScheduler()
    trainer = _make_trainer_with_scheduler(tmp_path, scheduler, interval="epoch")

    trainer.fit(_make_loader(), num_epochs=2, show_progress=False, track_best=False)

    assert scheduler.count == 2


def test_train_loader_set_epoch_is_called_once_per_epoch(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    loader = EpochAwareLoader()

    trainer.fit(loader, num_epochs=2, show_progress=False, track_best=False)

    assert loader.sampler.epochs == [1, 2]


def test_checkpoint_saves_and_restores_lr_scheduler_state(tmp_path) -> None:
    scheduler = CountingScheduler()
    trainer = _make_trainer_with_scheduler(tmp_path, scheduler, interval="epoch")
    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)
    checkpoint_path = tmp_path / "checkpoints" / "epoch_0001.pt"

    scheduler.count = 99
    checkpoint_manager = trainer.checkpoint_manager
    assert checkpoint_manager is not None
    checkpoint_manager.load(checkpoint_path)

    assert scheduler.count == 1


def test_checkpoint_manager_rejects_v3(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    checkpoint_manager = trainer.checkpoint_manager
    assert checkpoint_manager is not None
    checkpoint = tmp_path / "v3.pt"
    payload = checkpoint_manager.build_state()
    payload["format_version"] = 3
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="expected version 4"):
        checkpoint_manager.load(checkpoint)
