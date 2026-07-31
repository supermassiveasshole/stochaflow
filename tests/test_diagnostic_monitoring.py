"""Boundary tests for observation-only training diagnostics."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.training import (
    SupervisedTrainingStrategy,
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

    def on_fit_start(self, event) -> None:
        del event
        self.fit_started = True

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        self.observed.append(dict(event.metrics))
        self.logger.log_metrics(
            {"diagnostics/quality/runtime_seconds": float(event.epoch_index)},
            step=event.trainer.global_step,
        )


class ReturningDiagnostic(TrainingDiagnostic):
    """Violate the observation-only return contract intentionally."""

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        del event
        return cast(Any, {"diagnostics/quality/score": 1.0})


class MutatingDiagnostic(TrainingDiagnostic):
    """Attempt to write into the read-only epoch observation."""

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        cast(dict[str, float], event.metrics)["valid/loss"] = -1.0


def diagnostic_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return one tiny train or validation batch."""

    return [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))]


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


def test_diagnostic_non_none_epoch_result_is_rejected(tmp_path) -> None:
    logger = DiagnosticBoundaryLogger()
    trainer = build_diagnostic_trainer(
        tmp_path,
        ReturningDiagnostic(),
        logger,
    )

    with pytest.raises(TypeError, match="must return None"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=diagnostic_loader(),
            show_progress=False,
        )


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
