"""Tests for live validation evaluation in the epoch training lifecycle."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from stochaflow.training import (
    EpochValidationCadence,
    EpochValidationEvaluator,
    EpochValidationIdentity,
    EpochValidationResult,
    SupervisedTrainingStrategy,
    TrainEpochEndEvent,
    Trainer,
    TrainingDiagnostic,
    TrainingPlan,
)
from stochaflow.utils.checkpoint import CheckpointManager

PROFILE_DIGEST = "a" * 64


class ScalarRegressor(nn.Module):
    """One-parameter regressor for deterministic trainer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class RecordingEpochValidationEvaluator(EpochValidationEvaluator):
    """Return configured epoch metrics while retaining evaluator calls."""

    def __init__(
        self,
        *,
        identity: EpochValidationIdentity,
        metrics_by_epoch: Mapping[int, Mapping[str, float]],
        fail_epoch: int | None = None,
        consume_rng: bool = False,
    ) -> None:
        self._identity = identity
        self.metrics_by_epoch = metrics_by_epoch
        self.fail_epoch = fail_epoch
        self.consume_rng = consume_rng
        self.calls: list[tuple[int, int]] = []

    @property
    def identity(self) -> EpochValidationIdentity:
        return self._identity

    def evaluate(
        self,
        *,
        epoch: int,
        global_step: int,
    ) -> EpochValidationResult:
        self.calls.append((epoch, global_step))
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        if epoch == self.fail_epoch:
            raise RuntimeError("validation evaluation failed")
        return EpochValidationResult(
            epoch=epoch,
            global_step=global_step,
            metrics=self.metrics_by_epoch[epoch],
        )


class FailingEpochDiagnostic(TrainingDiagnostic):
    """Fail after observing one configured completed epoch."""

    def __init__(self, fail_epoch: int) -> None:
        self.fail_epoch = fail_epoch

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        if event.epoch_index == self.fail_epoch:
            raise RuntimeError("epoch diagnostic failed")


def _identity(
    *,
    first_epoch: int = 2,
    every_n_epochs: int = 2,
    include_final: bool = False,
    digest: str = PROFILE_DIGEST,
    metric_keys: tuple[str, ...] = (
        "valid/metrics/fid",
        "valid/metrics/kid",
    ),
) -> EpochValidationIdentity:
    return EpochValidationIdentity(
        profile_digest=digest,
        metric_keys=metric_keys,
        cadence=EpochValidationCadence(
            first_epoch=first_epoch,
            every_n_epochs=every_n_epochs,
            include_final=include_final,
        ),
    )


def _trainer(
    tmp_path,
    *,
    diagnostic: TrainingDiagnostic | None = None,
) -> Trainer:
    model = ScalarRegressor()
    objective = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    strategy = SupervisedTrainingStrategy(model, objective)
    return Trainer(
        TrainingPlan(
            strategy=strategy,
            primary_model=model,
            objective=objective,
        ),
        optimizer,
        device="cpu",
        diagnostics=() if diagnostic is None else (diagnostic,),
        checkpoint_manager=CheckpointManager(
            model=model,
            objective=objective,
            optimizer=optimizer,
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )


def _loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(torch.tensor([[1.0]]), torch.tensor([[0.0]]))]


def _training_loop_state(path) -> dict[str, Any]:
    payload = CheckpointManager.load_payload(path)
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    state = metadata.get("training_loop")
    assert isinstance(state, dict)
    return state


def test_epoch_evaluator_metrics_select_and_persist_best_checkpoint(
    tmp_path,
) -> None:
    trainer = _trainer(tmp_path)
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(),
        metrics_by_epoch={
            2: {"valid/metrics/fid": 20.0, "valid/metrics/kid": 0.02},
            4: {"valid/metrics/fid": 10.0, "valid/metrics/kid": 0.01},
        },
    )

    history = trainer.fit(
        _loader(),
        num_epochs=4,
        show_progress=False,
        epoch_validation_evaluator=evaluator,
        early_stopping_monitor="valid/metrics/fid",
        early_stopping_mode="min",
    )

    assert [epoch for epoch, _ in evaluator.calls] == [2, 4]
    assert "valid/metrics/fid" not in history[0]
    assert history[1]["valid/metrics/fid"] == 20.0
    assert "valid/metrics/fid" not in history[2]
    assert history[3]["valid/metrics/kid"] == 0.01
    assert trainer.best_epoch == 4
    assert trainer.best_metric_value == 10.0
    assert trainer.monitor_observations == 2
    best_path = tmp_path / "checkpoints" / "best.pt"
    best_payload = CheckpointManager.load_payload(best_path)
    assert best_payload.get("epoch") == 4
    best_metrics = best_payload.get("metrics")
    assert isinstance(best_metrics, dict)
    assert best_metrics["valid/metrics/fid"] == 10.0
    state = _training_loop_state(best_path)
    assert state["epoch_validation"] == {
        "identity": evaluator.identity.to_dict(),
        "last_evaluated_epoch": 4,
        "last_metrics": {
            "valid/metrics/fid": 10.0,
            "valid/metrics/kid": 0.01,
        },
        "off_cadence_final_epochs": [],
    }


def test_external_monitor_patience_counts_observations_not_epochs(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(),
        metrics_by_epoch={
            2: {"valid/metrics/fid": 10.0, "valid/metrics/kid": 0.01},
            4: {"valid/metrics/fid": 11.0, "valid/metrics/kid": 0.01},
        },
    )

    history = trainer.fit(
        _loader(),
        num_epochs=5,
        show_progress=False,
        epoch_validation_evaluator=evaluator,
        early_stopping_monitor="valid/metrics/fid",
        early_stopping_patience=1,
    )

    assert len(history) == 4
    assert trainer.monitor_observations == 2
    assert trainer.observations_without_improvement == 1
    assert trainer.stopped_early


def test_epoch_evaluator_include_final_runs_unscheduled_final_epoch(
    tmp_path,
) -> None:
    trainer = _trainer(tmp_path)
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(
            first_epoch=3,
            every_n_epochs=3,
            include_final=True,
        ),
        metrics_by_epoch={
            3: {"valid/metrics/fid": 3.0, "valid/metrics/kid": 0.03},
            4: {"valid/metrics/fid": 4.0, "valid/metrics/kid": 0.04},
        },
    )

    trainer.fit(
        _loader(),
        num_epochs=4,
        show_progress=False,
        epoch_validation_evaluator=evaluator,
        early_stopping_monitor="valid/metrics/fid",
    )

    assert [epoch for epoch, _ in evaluator.calls] == [3, 4]
    state = _training_loop_state(tmp_path / "checkpoints" / "latest.pt")
    assert state["epoch_validation"]["off_cadence_final_epochs"] == [4]


def test_epoch_evaluator_staged_resume_preserves_all_final_observations(
    tmp_path,
) -> None:
    identity = _identity(
        first_epoch=3,
        every_n_epochs=3,
        include_final=True,
    )
    first = _trainer(tmp_path)
    first_evaluator = RecordingEpochValidationEvaluator(
        identity=identity,
        metrics_by_epoch={
            3: {"valid/metrics/fid": 5.0, "valid/metrics/kid": 0.05},
            4: {"valid/metrics/fid": 4.0, "valid/metrics/kid": 0.04},
        },
    )
    first.fit(
        _loader(),
        num_epochs=4,
        show_progress=False,
        epoch_validation_evaluator=first_evaluator,
        early_stopping_monitor="valid/metrics/fid",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    first_state = _training_loop_state(checkpoint_dir / "latest.pt")

    resumed = _trainer(tmp_path)
    resumed.restore_fit_state(
        first_state,
        best_checkpoint_path=checkpoint_dir / "best.pt",
    )
    resumed_evaluator = RecordingEpochValidationEvaluator(
        identity=identity,
        metrics_by_epoch={
            6: {"valid/metrics/fid": 3.0, "valid/metrics/kid": 0.03},
            8: {"valid/metrics/fid": 2.0, "valid/metrics/kid": 0.02},
        },
    )
    resumed.fit(
        _loader(),
        num_epochs=8,
        start_epoch=5,
        show_progress=False,
        epoch_validation_evaluator=resumed_evaluator,
        early_stopping_monitor="valid/metrics/fid",
    )

    assert [epoch for epoch, _ in resumed_evaluator.calls] == [6, 8]
    state = _training_loop_state(checkpoint_dir / "latest.pt")
    assert state["monitor_observations"] == 4
    assert state["epoch_validation"]["off_cadence_final_epochs"] == [4, 8]


def test_epoch_evaluator_preserves_global_rng_streams(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(first_epoch=1, every_n_epochs=1),
        metrics_by_epoch={
            1: {"valid/metrics/fid": 1.0, "valid/metrics/kid": 0.01},
        },
        consume_rng=True,
    )
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    trainer.fit(
        _loader(),
        num_epochs=1,
        show_progress=False,
        epoch_validation_evaluator=evaluator,
        early_stopping_monitor="valid/metrics/fid",
    )
    actual = (random.random(), float(np.random.random()), torch.rand(1))
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    expected = (random.random(), float(np.random.random()), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_epoch_evaluator_failure_cannot_publish_completed_checkpoints(
    tmp_path,
) -> None:
    trainer = _trainer(tmp_path)
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(first_epoch=1, every_n_epochs=1),
        metrics_by_epoch={},
        fail_epoch=1,
    )

    with pytest.raises(RuntimeError, match="validation evaluation failed"):
        trainer.fit(
            _loader(),
            num_epochs=1,
            show_progress=False,
            epoch_validation_evaluator=evaluator,
            early_stopping_monitor="valid/metrics/fid",
        )

    assert trainer.best_epoch is None
    assert not (tmp_path / "checkpoints" / "best.pt").exists()
    assert not (tmp_path / "checkpoints" / "latest.pt").exists()
    assert not (tmp_path / "checkpoints" / "epoch_0001.pt").exists()


def test_epoch_diagnostic_failure_follows_due_evaluation_checkpoint_publication(
    tmp_path,
) -> None:
    trainer = _trainer(
        tmp_path,
        diagnostic=FailingEpochDiagnostic(fail_epoch=2),
    )
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(first_epoch=1, every_n_epochs=1),
        metrics_by_epoch={
            1: {"valid/metrics/fid": 20.0, "valid/metrics/kid": 0.02},
            2: {"valid/metrics/fid": 10.0, "valid/metrics/kid": 0.01},
        },
    )

    with pytest.raises(RuntimeError, match="epoch diagnostic failed"):
        trainer.fit(
            _loader(),
            num_epochs=2,
            show_progress=False,
            epoch_validation_evaluator=evaluator,
            early_stopping_monitor="valid/metrics/fid",
        )

    checkpoint_dir = tmp_path / "checkpoints"
    expected_metrics = {
        "valid/metrics/fid": 10.0,
        "valid/metrics/kid": 0.01,
    }
    for filename in ("best.pt", "latest.pt", "epoch_0002.pt"):
        checkpoint = checkpoint_dir / filename
        payload = CheckpointManager.load_payload(checkpoint)
        assert payload.get("epoch") == 2
        metrics = payload.get("metrics")
        assert isinstance(metrics, dict)
        assert {
            key: metrics[key] for key in expected_metrics
        } == expected_metrics
        state = _training_loop_state(checkpoint)
        assert state["best_epoch"] == 2
        assert state["best_metric_value"] == 10.0
        assert state["monitor_observations"] == 2
        assert state["observations_without_improvement"] == 0
        assert state["epoch_validation"]["last_evaluated_epoch"] == 2
        assert state["epoch_validation"]["last_metrics"] == expected_metrics


def test_epoch_evaluator_rejects_metric_collision_before_checkpointing(
    tmp_path,
) -> None:
    trainer = _trainer(tmp_path)
    trainer.evaluate_epoch = lambda *args, **kwargs: {
        "loss": 1.0,
        "num_batches": 1.0,
        "duration_seconds": 0.0,
        "valid/metrics/fid": 9.0,
    }
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(
            first_epoch=1,
            every_n_epochs=1,
            metric_keys=("valid/metrics/fid",),
        ),
        metrics_by_epoch={1: {"valid/metrics/fid": 8.0}},
    )

    with pytest.raises(ValueError, match="metrics collide"):
        trainer.fit(
            _loader(),
            num_epochs=1,
            validation_dataloader=_loader(),
            show_progress=False,
            epoch_validation_evaluator=evaluator,
            early_stopping_monitor="valid/metrics/fid",
        )

    assert not (tmp_path / "checkpoints" / "latest.pt").exists()


def test_epoch_evaluator_rejects_missing_declared_metric(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(first_epoch=1, every_n_epochs=1),
        metrics_by_epoch={1: {"valid/metrics/fid": 1.0}},
    )

    with pytest.raises(ValueError, match="exactly match"):
        trainer.fit(
            _loader(),
            num_epochs=1,
            show_progress=False,
            epoch_validation_evaluator=evaluator,
            early_stopping_monitor="valid/metrics/fid",
        )

    assert not (tmp_path / "checkpoints" / "latest.pt").exists()


def test_epoch_evaluator_strict_resume_preserves_identity_and_cadence(
    tmp_path,
) -> None:
    identity = _identity()
    first_trainer = _trainer(tmp_path)
    first_evaluator = RecordingEpochValidationEvaluator(
        identity=identity,
        metrics_by_epoch={
            2: {"valid/metrics/fid": 2.0, "valid/metrics/kid": 0.02},
        },
    )
    first_trainer.fit(
        _loader(),
        num_epochs=2,
        show_progress=False,
        epoch_validation_evaluator=first_evaluator,
        early_stopping_monitor="valid/metrics/fid",
    )
    state = _training_loop_state(tmp_path / "checkpoints" / "latest.pt")

    resumed = _trainer(tmp_path)
    resumed.restore_fit_state(
        state,
        best_checkpoint_path=tmp_path / "checkpoints" / "best.pt",
    )
    resumed_evaluator = RecordingEpochValidationEvaluator(
        identity=identity,
        metrics_by_epoch={
            4: {"valid/metrics/fid": 1.0, "valid/metrics/kid": 0.01},
        },
    )
    resumed.fit(
        _loader(),
        num_epochs=4,
        start_epoch=3,
        show_progress=False,
        epoch_validation_evaluator=resumed_evaluator,
        early_stopping_monitor="valid/metrics/fid",
    )

    assert [epoch for epoch, _ in resumed_evaluator.calls] == [4]
    assert resumed.best_epoch == 4
    resumed_state = _training_loop_state(
        tmp_path / "checkpoints" / "latest.pt"
    )
    assert resumed_state["epoch_validation"]["last_evaluated_epoch"] == 4


def test_epoch_evaluator_resume_rejects_changed_profile_before_training(
    tmp_path,
) -> None:
    trainer = _trainer(tmp_path)
    trainer.restore_fit_state(
        {
            "best_epoch": 2,
            "best_metric_value": 2.0,
            "observations_without_improvement": 0,
            "monitor_observations": 1,
            "stopped_early": False,
            "tracking_enabled": True,
            "monitor_policy": {
                "metric": "valid/metrics/fid",
                "mode": "min",
                "min_delta": 0.0,
            },
            "early_stopping_patience": None,
            "epoch_validation": {
                "identity": _identity().to_dict(),
                "last_evaluated_epoch": 2,
                "last_metrics": {
                    "valid/metrics/fid": 2.0,
                    "valid/metrics/kid": 0.02,
                },
            },
        }
    )
    evaluator = RecordingEpochValidationEvaluator(
        identity=_identity(digest="b" * 64),
        metrics_by_epoch={},
    )
    trainer.train_epoch = lambda *args, **kwargs: pytest.fail(
        "training must not start"
    )

    with pytest.raises(ValueError, match="identity must exactly match"):
        trainer.fit(
            _loader(),
            num_epochs=4,
            start_epoch=3,
            show_progress=False,
            epoch_validation_evaluator=evaluator,
            early_stopping_monitor="valid/metrics/fid",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_epoch_validation_result_rejects_non_finite_metrics(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        EpochValidationResult(
            epoch=1,
            global_step=0,
            metrics={"valid/metrics/fid": value},
        )
