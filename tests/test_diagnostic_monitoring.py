"""Integration tests for verified epoch-diagnostic monitoring."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from stochaflow.metrics import MetricDataRole
from stochaflow.training import (
    DiagnosticResult,
    DiagnosticSourceRequest,
    SupervisedTrainingStrategy,
    TrainEpochEndEvent,
    Trainer,
    TrainingDiagnostic,
    TrainingPlan,
    bind_training_diagnostic,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.logging import ExperimentLogger


class DiagnosticMonitorLogger(ExperimentLogger):
    """Retain metric payloads emitted by the Trainer."""

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


class DiagnosticMonitorRegressor(nn.Module):
    """One-parameter deterministic regression model."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class ScheduledQualityDiagnostic(TrainingDiagnostic):
    """Emit validation quality and external observations on selected epochs."""

    def __init__(
        self,
        values: Mapping[int, float | None],
        *,
        data_role: MetricDataRole = "validation",
        undeclared_source: bool = False,
    ) -> None:
        self.values = dict(values)
        self.data_role: MetricDataRole = data_role
        self.undeclared_source = undeclared_source

    @property
    def metric_source_requests(self) -> tuple[DiagnosticSourceRequest, ...]:
        return (
            DiagnosticSourceRequest(
                id="observation",
                data_role="external",
                protocol={"kind": "runtime-observation", "version": 1},
            ),
            DiagnosticSourceRequest(
                id="quality",
                data_role=self.data_role,
                protocol={
                    "kind": "class-quality",
                    "version": 2,
                    "due_epochs": sorted(self.values),
                },
            ),
        )

    def on_train_epoch_end(
        self,
        event: TrainEpochEndEvent,
    ) -> tuple[DiagnosticResult, ...] | None:
        if event.epoch_index not in self.values:
            return None
        value = self.values[event.epoch_index]
        quality_metrics = (
            {}
            if value is None
            else {"diagnostics/quality/class_score": value}
        )
        source_id = "unknown" if self.undeclared_source else "quality"
        return (
            DiagnosticResult(
                source_id="observation",
                metrics={
                    "diagnostics/quality/runtime_seconds": (
                        float(event.epoch_index)
                    )
                },
            ),
            DiagnosticResult(
                source_id=source_id,
                metrics=quality_metrics,
            ),
        )


class DuplicateSourceDiagnostic(ScheduledQualityDiagnostic):
    """Return one bound source twice to exercise fail-closed merging."""

    def on_train_epoch_end(
        self,
        event: TrainEpochEndEvent,
    ) -> tuple[DiagnosticResult, ...] | None:
        result = DiagnosticResult(
            source_id="quality",
            metrics={"diagnostics/quality/class_score": 0.5},
        )
        return result, result


class WrongScopeDiagnostic(ScheduledQualityDiagnostic):
    """Return a canonical key owned by a different diagnostic id."""

    def on_train_epoch_end(
        self,
        event: TrainEpochEndEvent,
    ) -> tuple[DiagnosticResult, ...] | None:
        del event
        return (
            DiagnosticResult(
                source_id="quality",
                metrics={"diagnostics/other/class_score": 0.5},
            ),
        )


class AmbiguousValidationDiagnostic(TrainingDiagnostic):
    """Request two validation sources that cannot select unambiguously."""

    @property
    def metric_source_requests(self) -> tuple[DiagnosticSourceRequest, ...]:
        return (
            DiagnosticSourceRequest(
                id="first",
                data_role="validation",
                protocol={"kind": "first"},
            ),
            DiagnosticSourceRequest(
                id="second",
                data_role="validation",
                protocol={"kind": "second"},
            ),
        )


def diagnostic_protocol_provenance(
    *,
    partition: int = 0,
    artifact_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return the strict composition provenance used by test bindings."""

    artifacts = (
        None
        if artifact_fingerprint is None
        else {
            "schema_version": 2,
            "bindings": [
                {
                    "id": "source",
                    "identity": {
                        "schema_version": 2,
                        "kind": "managed",
                        "artifact_type": "diagnostic-test",
                        "source_name": "in-memory",
                        "source_digest": hashlib.sha256(
                            f"{artifact_fingerprint}:source".encode()
                        ).hexdigest(),
                        "materializer_name": "test",
                        "materialization_digest": hashlib.sha256(
                            f"{artifact_fingerprint}:materialization".encode()
                        ).hexdigest(),
                        "content_digest": hashlib.sha256(
                            f"{artifact_fingerprint}:content".encode()
                        ).hexdigest(),
                        "artifact_digest": hashlib.sha256(
                            f"{artifact_fingerprint}:artifact".encode()
                        ).hexdigest(),
                        "manifest_sha256": hashlib.sha256(
                            f"{artifact_fingerprint}:manifest".encode()
                        ).hexdigest(),
                    },
                }
            ],
        }
    )
    return {
        "schema_version": 1,
        "data_config": {
            "name": "diagnostic_test_data",
            "params": {"partition": partition},
        },
        "data_artifacts": artifacts,
        "extension_plugins": [],
    }


def build_diagnostic_monitor_trainer(
    tmp_path,
    diagnostic: ScheduledQualityDiagnostic,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[Trainer, DiagnosticMonitorLogger, list[Any]]:
    """Build a deterministic Trainer with one composition-bound diagnostic."""

    model = DiagnosticMonitorRegressor()
    objective = nn.MSELoss()
    strategy = SupervisedTrainingStrategy(model, objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    plan = TrainingPlan(
        strategy=strategy,
        primary_model=model,
        objective=objective,
    )
    logger = DiagnosticMonitorLogger()
    manager = CheckpointManager(
        model=model,
        objective=objective,
        optimizer=optimizer,
    )
    validation_loader = diagnostic_loader()
    binding = bind_training_diagnostic(
        "quality",
        diagnostic,
        protocol_provenance=(
            diagnostic_protocol_provenance()
            if provenance is None
            else provenance
        ),
        data_iterables={"validation": validation_loader},
    )
    trainer = Trainer(
        plan,
        optimizer,
        device="cpu",
        diagnostics=[binding],
        logger=logger,
        checkpoint_manager=manager,
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_every=1,
    )
    return trainer, logger, validation_loader


def diagnostic_loader() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return one tiny train/validation batch."""

    return [(torch.tensor([[1.0]]), torch.tensor([[1.0]]))]


def test_low_frequency_diagnostic_controls_best_before_checkpoint(tmp_path) -> None:
    provenance = diagnostic_protocol_provenance(
        artifact_fingerprint="abc123"
    )
    diagnostic = ScheduledQualityDiagnostic({2: 0.8, 4: 0.7})
    trainer, logger, validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        diagnostic,
        provenance=provenance,
    )

    history = trainer.fit(
        diagnostic_loader(),
        num_epochs=4,
        validation_dataloader=validation_loader,
        show_progress=False,
        early_stopping_patience=1,
        early_stopping_monitor="diagnostics/quality/class_score",
        early_stopping_mode="max",
        monitor_missing="skip",
        track_best=True,
    )

    assert len(history) == 4
    assert "diagnostics/quality/class_score" not in history[0]
    assert history[1]["diagnostics/quality/class_score"] == pytest.approx(0.8)
    assert "diagnostics/quality/class_score" not in history[2]
    assert history[3]["diagnostics/quality/class_score"] == pytest.approx(0.7)
    assert trainer.best_epoch == 2
    assert trainer.monitor_observations == 2
    assert trainer.observations_without_improvement == 1
    assert trainer.stopped_early

    best = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "best.pt"
    )
    assert best.get("epoch") == 2
    best_metrics = best.get("metrics")
    assert isinstance(best_metrics, dict)
    assert best_metrics["diagnostics/quality/class_score"] == pytest.approx(
        0.8
    )
    best_metadata = best.get("metadata")
    assert isinstance(best_metadata, dict)
    source = best_metadata["metric_sources"][
        "diagnostics/quality/class_score"
    ]
    assert source["origin"] == "diagnostic"
    assert source["data_role"] == "validation"
    assert source["selection_eligible"] is True
    assert source["protocol_id"].startswith("sha256:")

    latest = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "latest.pt"
    )
    assert latest.get("epoch") == 4
    latest_metadata = latest.get("metadata")
    assert isinstance(latest_metadata, dict)
    assert latest_metadata["training_loop"] == {
        "best_epoch": 2,
        "best_metric_value": 0.8,
        "observations_without_improvement": 1,
        "monitor_observations": 2,
        "stopped_early": True,
        "tracking_enabled": True,
        "monitor_policy": {
            "metric": "diagnostics/quality/class_score",
            "mode": "max",
            "missing": "skip",
            "min_delta": 0.0,
        },
        "early_stopping_patience": 1,
    }
    assert any(
        payload.get("diagnostics/quality/class_score") == pytest.approx(0.8)
        for payload in logger.payloads
    )


def test_diagnostic_skip_state_resumes_by_observation_count(tmp_path) -> None:
    first, _, first_validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        ScheduledQualityDiagnostic({2: 0.8, 4: 0.7}),
    )
    fit_kwargs = {
        "validation_dataloader": first_validation_loader,
        "show_progress": False,
        "early_stopping_patience": 2,
        "early_stopping_monitor": "diagnostics/quality/class_score",
        "early_stopping_mode": "max",
        "monitor_missing": "skip",
        "track_best": True,
    }
    first.fit(
        diagnostic_loader(),
        num_epochs=2,
        **fit_kwargs,
    )
    latest_path = tmp_path / "checkpoints" / "latest.pt"
    latest = CheckpointManager.load_payload(latest_path)
    latest_metadata = latest.get("metadata")
    assert isinstance(latest_metadata, dict)

    resumed, _, resumed_validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        ScheduledQualityDiagnostic({2: 0.8, 4: 0.7}),
    )
    resumed.restore_fit_state(
        latest_metadata["training_loop"],
        best_checkpoint_path=tmp_path / "checkpoints" / "best.pt",
    )
    resumed.fit(
        diagnostic_loader(),
        num_epochs=4,
        start_epoch=3,
        **{
            **fit_kwargs,
            "validation_dataloader": resumed_validation_loader,
        },
    )

    assert resumed.best_epoch == 2
    assert resumed.monitor_observations == 2
    assert resumed.observations_without_improvement == 1
    assert not resumed.stopped_early


def test_due_diagnostic_missing_value_is_not_a_cadence_skip(tmp_path) -> None:
    trainer, _, validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        ScheduledQualityDiagnostic({1: None}),
    )

    with pytest.raises(ValueError, match=r"was due.*returned no value"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=validation_loader,
            show_progress=False,
            early_stopping_monitor="diagnostics/quality/class_score",
            monitor_missing="skip",
            track_best=True,
        )


def test_diagnostic_monitor_requires_at_least_one_observation(tmp_path) -> None:
    trainer, _, validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        ScheduledQualityDiagnostic({3: 0.5}),
    )

    with pytest.raises(ValueError, match="produced no observations"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=2,
            validation_dataloader=validation_loader,
            show_progress=False,
            early_stopping_monitor="diagnostics/quality/class_score",
            monitor_missing="skip",
            track_best=True,
        )


def test_diagnostic_result_rejects_unknown_bound_source(tmp_path) -> None:
    trainer, _, validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        ScheduledQualityDiagnostic(
            {1: 0.5},
            undeclared_source=True,
        ),
    )

    with pytest.raises(ValueError, match="returned undeclared source"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=validation_loader,
            show_progress=False,
            track_best=False,
        )


def test_test_role_diagnostic_cannot_control_selection(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="does not inject a test iterable",
    ):
        build_diagnostic_monitor_trainer(
            tmp_path,
            ScheduledQualityDiagnostic({1: 0.5}, data_role="test"),
        )


def test_binding_protocol_digest_changes_with_composition_provenance() -> None:
    diagnostic = ScheduledQualityDiagnostic({1: 0.5})
    validation_loader = diagnostic_loader()

    first = bind_training_diagnostic(
        "quality",
        diagnostic,
        protocol_provenance=diagnostic_protocol_provenance(
            partition=0,
            artifact_fingerprint="first",
        ),
        data_iterables={"validation": validation_loader},
    )
    second = bind_training_diagnostic(
        "quality",
        diagnostic,
        protocol_provenance=diagnostic_protocol_provenance(
            partition=1,
            artifact_fingerprint="second",
        ),
        data_iterables={"validation": validation_loader},
    )

    assert (
        first.sources["quality"].protocol_digest
        != second.sources["quality"].protocol_digest
    )


def test_protocol_digest_ignores_per_run_iterable_identity() -> None:
    diagnostic = ScheduledQualityDiagnostic({1: 0.5})
    provenance = diagnostic_protocol_provenance()
    first_loader = diagnostic_loader()
    second_loader = diagnostic_loader()

    first = bind_training_diagnostic(
        "quality",
        diagnostic,
        protocol_provenance=provenance,
        data_iterables={"validation": first_loader},
    )
    second = bind_training_diagnostic(
        "quality",
        diagnostic,
        protocol_provenance=provenance,
        data_iterables={"validation": second_loader},
    )

    assert (
        first.sources["quality"].protocol_digest
        == second.sources["quality"].protocol_digest
    )
    assert first.source_iterables["quality"] is first_loader
    assert second.source_iterables["quality"] is second_loader


def test_binding_rejects_validation_claim_without_composition_provenance() -> None:
    with pytest.raises(TypeError, match="must be a composition mapping"):
        bind_training_diagnostic(
            "quality",
            ScheduledQualityDiagnostic({1: 0.5}),
            data_iterables={"validation": diagnostic_loader()},
        )

    with pytest.raises(ValueError, match="has invalid fields"):
        bind_training_diagnostic(
            "quality",
            ScheduledQualityDiagnostic({1: 0.5}),
            protocol_provenance={},
            data_iterables={"validation": diagnostic_loader()},
        )


def test_binding_rejects_validation_claim_without_actual_iterable() -> None:
    with pytest.raises(ValueError, match="actual validation fit iterable"):
        bind_training_diagnostic(
            "quality",
            ScheduledQualityDiagnostic({1: 0.5}),
            protocol_provenance=diagnostic_protocol_provenance(),
        )


@pytest.mark.parametrize("data_role", ["train", "validation"])
def test_trainer_rejects_different_bound_phase_iterable(
    data_role: MetricDataRole,
) -> None:
    model = DiagnosticMonitorRegressor()
    objective = nn.MSELoss()
    strategy = SupervisedTrainingStrategy(model, objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    bound_loader = diagnostic_loader()
    binding = bind_training_diagnostic(
        "quality",
        ScheduledQualityDiagnostic({1: 0.5}, data_role=data_role),
        protocol_provenance=diagnostic_protocol_provenance(),
        data_iterables={data_role: bound_loader},
    )
    trainer = Trainer(
        TrainingPlan(
            strategy=strategy,
            primary_model=model,
            objective=objective,
        ),
        optimizer,
        device="cpu",
        diagnostics=[binding],
    )
    train_loader = (
        diagnostic_loader() if data_role == "train" else bound_loader
    )
    validation_loader = (
        diagnostic_loader() if data_role == "validation" else None
    )

    with pytest.raises(
        ValueError,
        match=rf"different {data_role} iterable",
    ):
        trainer.fit(
            train_loader,
            num_epochs=1,
            validation_dataloader=validation_loader,
            show_progress=False,
            track_best=False,
        )


def test_trainer_rejects_raw_diagnostic_that_declares_metric_sources() -> None:
    model = DiagnosticMonitorRegressor()
    objective = nn.MSELoss()
    strategy = SupervisedTrainingStrategy(model, objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    with pytest.raises(
        ValueError,
        match=r"must be supplied as.*composition-bound",
    ):
        Trainer(
            TrainingPlan(
                strategy=strategy,
                primary_model=model,
                objective=objective,
            ),
            optimizer,
            device="cpu",
            diagnostics=[ScheduledQualityDiagnostic({1: 0.5})],
        )


def test_diagnostic_result_rejects_duplicate_source(tmp_path) -> None:
    trainer, _, validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        DuplicateSourceDiagnostic({1: 0.5}),
    )

    with pytest.raises(ValueError, match="returned source 'quality' more than once"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=validation_loader,
            show_progress=False,
            track_best=False,
        )


def test_diagnostic_result_rejects_another_diagnostic_scope(tmp_path) -> None:
    trainer, _, validation_loader = build_diagnostic_monitor_trainer(
        tmp_path,
        WrongScopeDiagnostic({1: 0.5}),
    )

    with pytest.raises(ValueError, match="must start with 'diagnostics/quality/'"):
        trainer.fit(
            diagnostic_loader(),
            num_epochs=1,
            validation_dataloader=validation_loader,
            show_progress=False,
            track_best=False,
        )


def test_binding_rejects_multiple_selection_sources() -> None:
    validation_loader = diagnostic_loader()
    with pytest.raises(ValueError, match="at most one selection-eligible source"):
        bind_training_diagnostic(
            "quality",
            AmbiguousValidationDiagnostic(),
            protocol_provenance=diagnostic_protocol_provenance(),
            data_iterables={"validation": validation_loader},
        )
