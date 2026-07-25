"""Tests for trainer reporting and validation checkpoint behavior."""

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from stochaflow.training import (
    FitStartEvent,
    ManagedTrainingModule,
    SupervisedTrainingStrategy,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    Trainer,
    TrainingDiagnostic,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
    trainable_parameters,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.logging import ExperimentLogger


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


class MetricRecordingLogger(ExperimentLogger):
    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del step
        self.metrics.append(metrics)

    def close(self) -> None:
        return None


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


class CountingScheduler(LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.count = -1
        super().__init__(optimizer)

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


class CountingLoader:
    def __init__(self) -> None:
        self.num_pulls = 0

    def __iter__(self):
        for _ in range(5):
            self.num_pulls += 1
            yield torch.tensor([[1.0]]), torch.tensor([[1.0]])


class NamedBatch(NamedTuple):
    inputs: torch.Tensor
    targets: torch.Tensor


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


class CallableTrainingStrategy(TrainingStrategy):
    def __init__(self, step: Callable[[Any], TrainStepOutput]) -> None:
        self.step = step

    def training_step(self, batch: Any) -> TrainStepOutput:
        return self.step(batch)


def _make_loader() -> DataLoader:
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)


def _build_trainer(
    tmp_path,
    *,
    model: nn.Module | None = None,
    objective: nn.Module | None = None,
    strategy: TrainingStrategy | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    **kwargs: Any,
) -> Trainer:
    model = model or TinyRegressor()
    objective = objective or nn.MSELoss()
    strategy = strategy or SupervisedTrainingStrategy(model, objective)
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.01)
    checkpoint_manager = checkpoint_manager or CheckpointManager(
        model=model,
        objective=objective,
        optimizer=optimizer,
    )
    return Trainer(
        plan=TrainingPlan(
            strategy=strategy,
            primary_model=model,
            objective=objective,
        ),
        optimizer=optimizer,
        device="cpu",
        checkpoint_manager=checkpoint_manager,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=kwargs.pop("checkpoint_every", 1),
        **kwargs,
    )


def _make_trainer(tmp_path) -> Trainer:
    return _build_trainer(tmp_path)


def _make_trainer_with_scheduler(
    tmp_path,
    *,
    interval: str,
) -> tuple[Trainer, CountingScheduler]:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = CountingScheduler(optimizer)
    objective = nn.MSELoss()
    checkpoint_manager = CheckpointManager(
        model=model,
        objective=objective,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    return (
        _build_trainer(
            tmp_path,
            model=model,
            objective=objective,
            optimizer=optimizer,
            checkpoint_manager=checkpoint_manager,
            lr_scheduler=scheduler,
            lr_scheduler_interval=interval,
        ),
        scheduler,
    )


def test_trainer_rejects_optimizer_with_parameters_outside_plan(tmp_path) -> None:
    model = TinyRegressor()
    unrelated_model = TinyRegressor()
    optimizer = torch.optim.SGD(unrelated_model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match=r"optimizer parameters.*TrainingPlan"):
        _build_trainer(tmp_path, model=model, optimizer=optimizer)


def test_trainer_rejects_checkpoint_manager_outside_plan(tmp_path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mismatched_manager = CheckpointManager(
        model=TinyRegressor(),
        objective=nn.MSELoss(),
        optimizer=optimizer,
    )

    with pytest.raises(
        ValueError,
        match=r"CheckpointManager model.*TrainingPlan",
    ):
        _build_trainer(
            tmp_path,
            model=model,
            optimizer=optimizer,
            checkpoint_manager=mismatched_manager,
        )


def test_empty_primary_model_ema_and_managed_assets_are_safe(tmp_path) -> None:
    primary = nn.Identity()
    learner = nn.Linear(1, 1)

    def step(batch: torch.Tensor) -> TrainStepOutput:
        return TrainStepOutput(learner(batch).square().mean())

    plan = TrainingPlan(
        CallableTrainingStrategy(step),
        primary,
        auxiliary_modules={"learner": ManagedTrainingModule(learner)},
    )
    parameters = trainable_parameters(plan)
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    ema = ExponentialMovingAverage(primary)
    manager = CheckpointManager(
        model=primary,
        auxiliary_modules={"learner": learner},
        optimizer=optimizer,
        ema=ema,
    )
    trainer = Trainer(
        plan,
        optimizer,
        device="cpu",
        ema=ema,
        checkpoint_manager=manager,
        checkpoint_dir=tmp_path,
    )

    trainer.train_batch(torch.ones(1, 1))
    state = manager.build_state()

    ema_state = state.get("ema_state_dict")
    assert ema_state is not None
    assert ema_state["shadow_params"] == {}
    with pytest.raises(TypeError):
        cast(Any, trainer.managed_modules)["late"] = ManagedTrainingModule(
            nn.Linear(1, 1)
        )


def test_strategy_metrics_use_a_nonconflicting_namespace(tmp_path) -> None:
    model = TinyRegressor()
    objective = nn.MSELoss()
    logger = MetricRecordingLogger()

    def step(batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepOutput:
        inputs, targets = batch
        loss = objective(model(inputs), targets)
        return TrainStepOutput(loss, metrics={"loss": 123.0, "epoch": 456.0})

    trainer = _build_trainer(
        tmp_path,
        model=model,
        objective=objective,
        strategy=CallableTrainingStrategy(step),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        logger=logger,
        log_every=1,
    )
    trainer.train_epoch(_make_loader(), show_progress=False, max_batches=1)
    batch_metrics = next(
        metrics for metrics in logger.metrics if "train/strategy/loss" in metrics
    )

    assert batch_metrics["train/strategy/loss"] == 123.0
    assert batch_metrics["train/strategy/epoch"] == 456.0
    assert batch_metrics["train/loss"] != 123.0


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


def test_fit_resume_preserves_best_and_early_stopping_wait(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    previous_best = tmp_path / "checkpoints" / "best.pt"
    previous_best.parent.mkdir(parents=True)
    previous_best.touch()
    trainer.restore_fit_state(
        {
            "best_epoch": 1,
            "best_metric_value": 0.1,
            "epochs_without_improvement": 1,
            "stopped_early": False,
            "monitor": "valid_loss",
            "mode": "min",
        },
        best_checkpoint_path=previous_best,
    )
    trainer.train_epoch = lambda *args, **kwargs: {
        "loss": 0.2,
        "num_batches": 1.0,
        "duration_seconds": 0.0,
    }
    trainer.evaluate_epoch = lambda *args, **kwargs: {
        "loss": 0.2,
        "num_batches": 1.0,
        "duration_seconds": 0.0,
    }

    trainer.fit(
        _make_loader(),
        num_epochs=2,
        start_epoch=2,
        validation_dataloader=_make_loader(),
        show_progress=False,
        early_stopping_patience=2,
        early_stopping_monitor="valid_loss",
        track_best=True,
    )

    assert trainer.best_epoch == 1
    assert trainer.best_metric_value == 0.1
    assert trainer.best_checkpoint_path == previous_best
    assert trainer.epochs_without_improvement == 2
    assert trainer.stopped_early
    latest = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "latest.pt"
    )
    metadata = latest.get("metadata")
    assert metadata is not None
    assert metadata["training_loop"] == {
        "best_epoch": 1,
        "best_metric_value": 0.1,
        "epochs_without_improvement": 2,
        "stopped_early": True,
        "monitor": "valid_loss",
        "mode": "min",
    }


def test_step_lr_scheduler_steps_once_per_batch(tmp_path) -> None:
    trainer, scheduler = _make_trainer_with_scheduler(tmp_path, interval="step")

    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)

    assert scheduler.count == 2


def test_structured_train_step_runs_diagnostics_hooks(tmp_path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    diagnostic = RecordingDiagnostic()

    objective = nn.MSELoss()

    def train_step(batch: Any) -> TrainStepOutput:
        inputs, targets = batch
        predictions = model(inputs)
        loss = objective(predictions, targets)
        return TrainStepOutput(loss=loss, diagnostics={"custom": torch.tensor(1.0)})

    trainer = _build_trainer(
        tmp_path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        strategy=CallableTrainingStrategy(train_step),
        diagnostics=[diagnostic],
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

    objective = nn.MSELoss()

    def train_step(batch: Any) -> TrainStepOutput:
        observed.append(batch)
        assert batch["state"].device == trainer.device
        assert batch["condition"]["scale"].device == trainer.device
        assert batch["metadata"] == {"source": "physics"}
        prediction = model(batch["state"])
        return TrainStepOutput(loss=objective(prediction, batch["target"]))

    trainer = _build_trainer(
        tmp_path,
        model=model,
        optimizer=optimizer,
        objective=objective,
        strategy=CallableTrainingStrategy(train_step),
    )
    batch = {
        "state": torch.tensor([[1.0]]),
        "target": torch.tensor([[0.0]]),
        "condition": {"scale": torch.tensor([2.0])},
        "metadata": {"source": "physics"},
    }

    trainer.train_epoch([batch], show_progress=False)

    assert len(observed) == 1


def test_batch_limit_does_not_consume_an_extra_item(tmp_path) -> None:
    model = TinyRegressor()
    trainer = _build_trainer(
        tmp_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        checkpoint_every=None,
    )
    train_loader = CountingLoader()
    validation_loader = CountingLoader()

    trainer.train_epoch(train_loader, show_progress=False, max_batches=2)
    trainer.evaluate_epoch(
        validation_loader,
        show_progress=False,
        max_batches=2,
        log_metrics=False,
    )

    assert train_loader.num_pulls == 2
    assert validation_loader.num_pulls == 2


def test_device_transfer_preserves_namedtuple_batch(tmp_path) -> None:
    model = TinyRegressor()
    observed: list[NamedBatch] = []

    objective = nn.MSELoss()

    def train_step(batch: Any) -> TrainStepOutput:
        assert isinstance(batch, NamedBatch)
        assert batch.inputs.device == trainer.device
        observed.append(batch)
        return TrainStepOutput(loss=objective(model(batch.inputs), batch.targets))

    trainer = _build_trainer(
        tmp_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        objective=objective,
        strategy=CallableTrainingStrategy(train_step),
        checkpoint_every=None,
    )

    trainer.train_epoch(
        [NamedBatch(torch.tensor([[1.0]]), torch.tensor([[0.0]]))],
        show_progress=False,
    )

    assert len(observed) == 1


def test_ema_updates_once_per_train_batch(tmp_path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ema = ExponentialMovingAverage(model, decay=0.0)
    objective = nn.MSELoss()
    checkpoint_manager = CheckpointManager(
        model=model,
        objective=objective,
        optimizer=optimizer,
        ema=ema,
    )
    trainer = _build_trainer(
        tmp_path,
        model=model,
        objective=objective,
        optimizer=optimizer,
        ema=ema,
        checkpoint_manager=checkpoint_manager,
    )

    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)

    assert ema.num_updates == 2
    assert "ema_state_dict" in CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "epoch_0001.pt"
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
    trainer, scheduler = _make_trainer_with_scheduler(tmp_path, interval="epoch")

    trainer.fit(_make_loader(), num_epochs=2, show_progress=False, track_best=False)

    assert scheduler.count == 2


def test_train_loader_set_epoch_is_called_once_per_epoch(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    loader = EpochAwareLoader()

    trainer.fit(loader, num_epochs=2, show_progress=False, track_best=False)

    assert loader.sampler.epochs == [1, 2]


def test_checkpoint_saves_and_restores_lr_scheduler_state(tmp_path) -> None:
    trainer, scheduler = _make_trainer_with_scheduler(tmp_path, interval="epoch")
    trainer.fit(_make_loader(), num_epochs=1, show_progress=False, track_best=False)
    checkpoint_path = tmp_path / "checkpoints" / "epoch_0001.pt"

    scheduler.count = 99
    checkpoint_manager = trainer.checkpoint_manager
    assert checkpoint_manager is not None
    checkpoint_manager.load(checkpoint_path)

    assert scheduler.count == 1


def test_checkpoint_manager_rejects_v7(tmp_path) -> None:
    trainer = _make_trainer(tmp_path)
    checkpoint_manager = trainer.checkpoint_manager
    assert checkpoint_manager is not None
    checkpoint = tmp_path / "v7.pt"
    payload = checkpoint_manager.build_state()
    payload["format_version"] = 7
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="expected version 8"):
        checkpoint_manager.load(checkpoint)
