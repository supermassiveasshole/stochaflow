"""Boundary tests for observation-only training diagnostics."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.training import (
    FitStartEvent,
    SupervisedTrainingStrategy,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    Trainer,
    TrainingDiagnostic,
    TrainingPlan,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.logging import ExperimentLogger


class DiagnosticBoundaryLogger(ExperimentLogger):
    """Retain scalar payloads emitted during one fit."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, float]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del step
        self.payloads.append(
            {key: float(value) for key, value in metrics.items()}
        )

    def close(self) -> None:
        return None


class DiagnosticBoundaryRegressor(nn.Module):
    """One-parameter deterministic regression model."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class ObservationLoggingDiagnostic(TrainingDiagnostic):
    """Log an observation while retaining the epoch input for assertions."""

    def __init__(self, logger: DiagnosticBoundaryLogger) -> None:
        self.logger = logger
        self.observed: list[dict[str, float]] = []
        self.fit_started = False
        self.fit_global_step: int | None = None

    def on_fit_start(self, event: FitStartEvent) -> None:
        self.fit_started = True
        self.fit_global_step = event.global_step

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        self.observed.append(dict(event.metrics))
        self.logger.log_metrics(
            {"diagnostics/quality/runtime_seconds": float(event.epoch_index)},
            step=event.global_step,
        )


class ReturningDiagnostic(TrainingDiagnostic):
    """Violate the observation-only return contract intentionally."""

    def __init__(self, callback_name: str) -> None:
        self.callback_name = callback_name

    def _result(self, callback_name: str) -> None:
        if self.callback_name != callback_name:
            return None
        torch.rand(())
        return cast(Any, {"diagnostics/quality/score": 1.0})

    def on_fit_start(self, event: FitStartEvent) -> None:
        del event
        return self._result("on_fit_start")

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        del event
        return self._result("on_train_batch_end")

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        del event
        return self._result("on_train_epoch_end")


class RaisingDiagnostic(TrainingDiagnostic):
    """Raise one pre-created error from a selected callback."""

    def __init__(self, callback_name: str, error: RuntimeError) -> None:
        self.callback_name = callback_name
        self.error = error

    def _raise(self, callback_name: str) -> None:
        if self.callback_name != callback_name:
            return
        torch.rand(())
        raise self.error

    def on_fit_start(self, event: FitStartEvent) -> None:
        del event
        self._raise("on_fit_start")

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        del event
        self._raise("on_train_batch_end")

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        del event
        self._raise("on_train_epoch_end")


class CallbackRecordingDiagnostic(TrainingDiagnostic):
    """Record callbacks that reach a later Diagnostic."""

    def __init__(self) -> None:
        self.callbacks: list[str] = []

    def on_fit_start(self, event: FitStartEvent) -> None:
        del event
        self.callbacks.append("on_fit_start")

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        del event
        self.callbacks.append("on_train_batch_end")

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        del event
        self.callbacks.append("on_train_epoch_end")


class MutatingDiagnostic(TrainingDiagnostic):
    """Attempt to write into the read-only epoch observation."""

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        cast(dict[str, float], event.metrics)["valid/loss"] = -1.0


def diagnostic_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return one tiny train or validation batch."""

    return [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))]


def emit_diagnostic_callback(trainer: Trainer, callback_name: str) -> None:
    """Invoke one private callback owner for a focused contract test."""

    if callback_name == "on_fit_start":
        trainer._emit_fit_start_diagnostics(
            train_dataloader=diagnostic_loader(),
            validation_dataloader=diagnostic_loader(),
        )
    elif callback_name == "on_train_batch_end":
        trainer._emit_batch_diagnostics(
            batch=diagnostic_loader()[0],
            diagnostic_observation=object(),
            loss=0.0,
            global_step=7,
            epoch_index=1,
        )
    else:
        trainer._emit_epoch_diagnostics(epoch_index=1, metrics={})


def build_diagnostic_trainer(
    tmp_path,
    diagnostic: TrainingDiagnostic,
    logger: DiagnosticBoundaryLogger,
) -> Trainer:
    """Build a deterministic trainer with one observation callback."""

    model = DiagnosticBoundaryRegressor()
    objective = nn.MSELoss()
    strategy = SupervisedTrainingStrategy(model, objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    manager = CheckpointManager(
        model=model,
        objective=objective,
        optimizer=optimizer,
    )
    return Trainer(
        TrainingPlan(
            strategy=strategy,
            primary_model=model,
            objective=objective,
        ),
        optimizer,
        device="cpu",
        diagnostics=[diagnostic],
        logger=logger,
        checkpoint_manager=manager,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )


def test_diagnostic_observations_do_not_enter_training_state(tmp_path) -> None:
    logger = DiagnosticBoundaryLogger()
    diagnostic = ObservationLoggingDiagnostic(logger)
    trainer = build_diagnostic_trainer(tmp_path, diagnostic, logger)

    history = trainer.fit(
        diagnostic_loader(),
        num_epochs=1,
        validation_dataloader=diagnostic_loader(),
        show_progress=False,
    )

    diagnostic_key = "diagnostics/quality/runtime_seconds"
    assert diagnostic.fit_started
    assert diagnostic.observed == history
    assert diagnostic_key not in history[0]
    assert any(payload.get(diagnostic_key) == 1.0 for payload in logger.payloads)
    epoch_payloads = [
        payload for payload in logger.payloads if "train/loss" in payload
    ]
    assert epoch_payloads == history

    latest = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "latest.pt"
    )
    latest_metrics = latest.get("metrics")
    latest_metadata = latest.get("metadata")
    assert isinstance(latest_metrics, dict)
    assert isinstance(latest_metadata, dict)
    assert latest_metrics == history[0]
    assert diagnostic_key not in latest_metrics
    assert "metric_sources" not in latest_metadata


@pytest.mark.parametrize(
    "callback_name",
    ["on_fit_start", "on_train_batch_end", "on_train_epoch_end"],
)
def test_diagnostic_non_none_callback_result_is_rejected(
    tmp_path,
    callback_name: str,
) -> None:
    logger = DiagnosticBoundaryLogger()
    trainer = build_diagnostic_trainer(
        tmp_path,
        ReturningDiagnostic(callback_name),
        logger,
    )
    torch.manual_seed(2468)
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(TypeError, match="must return None"):
        emit_diagnostic_callback(trainer, callback_name)

    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "callback_name",
    ["on_fit_start", "on_train_batch_end", "on_train_epoch_end"],
)
def test_diagnostic_callback_preserves_exception_and_stops_dispatch(
    tmp_path,
    callback_name: str,
) -> None:
    logger = DiagnosticBoundaryLogger()
    error = RuntimeError(f"injected {callback_name} failure")
    raising = RaisingDiagnostic(callback_name, error)
    follower = CallbackRecordingDiagnostic()
    trainer = build_diagnostic_trainer(tmp_path, raising, logger)
    trainer.diagnostics = (raising, follower)
    torch.manual_seed(1357)
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError) as raised:
        emit_diagnostic_callback(trainer, callback_name)

    assert raised.value is error
    assert follower.callbacks == []
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_diagnostic_cannot_mutate_epoch_metrics(tmp_path) -> None:
    logger = DiagnosticBoundaryLogger()
    trainer = build_diagnostic_trainer(
        tmp_path,
        MutatingDiagnostic(),
        logger,
    )

    with pytest.raises(TypeError, match="mappingproxy"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=diagnostic_loader(),
            show_progress=False,
        )


def test_fit_start_observes_restored_global_step(tmp_path) -> None:
    logger = DiagnosticBoundaryLogger()
    diagnostic = ObservationLoggingDiagnostic(logger)
    trainer = build_diagnostic_trainer(tmp_path, diagnostic, logger)
    trainer.global_step = 37

    trainer._emit_fit_start_diagnostics(
        train_dataloader=diagnostic_loader(),
        validation_dataloader=diagnostic_loader(),
    )

    assert diagnostic.fit_global_step == 37


def test_diagnostic_monitor_is_rejected_before_callbacks_and_loop(
    tmp_path,
) -> None:
    logger = DiagnosticBoundaryLogger()
    diagnostic = ObservationLoggingDiagnostic(logger)
    trainer = build_diagnostic_trainer(tmp_path, diagnostic, logger)
    trainer.train_epoch = lambda *args, **kwargs: pytest.fail(
        "training loop must not start"
    )

    with pytest.raises(ValueError, match="canonical validation metric key"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=diagnostic_loader(),
            show_progress=False,
            early_stopping_monitor="diagnostics/quality/score",
            track_best=True,
        )

    assert not diagnostic.fit_started
