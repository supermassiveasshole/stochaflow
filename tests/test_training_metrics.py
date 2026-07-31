"""Integration tests for phase metrics in the automatic training lifecycle."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import nn
from torchmetrics import Metric

from stochaflow.metrics import MetricConfig, MetricUpdate
from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training import (
    GaussianDenoisingTrainingStrategy,
    MSEObjective,
    SupervisedTrainingStrategy,
    Trainer,
    TrainingPlan,
)
from stochaflow.training.gaussian_loss import (
    GaussianLossWeightingConfig,
    GaussianVarianceConfig,
)
from stochaflow.training.metric_binding import TrainingMetricRuntime
from stochaflow.training.precision import PrecisionRuntime
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import REGISTRIES


class MetricRecordingLogger(ExperimentLogger):
    """Retain flat metric payloads for lifecycle assertions."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del step
        self.payloads.append(dict(metrics))

    def close(self) -> None:
        return None


class ScalarRegressor(nn.Module):
    """One-parameter regressor whose output remains deterministic at lr=0."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class TinyGaussianDenoiser(nn.Module):
    """One-parameter denoiser for a network-free Gaussian smoke run."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        return self.scale * state


class TinyLearnedVarianceGaussianDenoiser(TinyGaussianDenoiser):
    """Emit distinct prediction and learned-range variance heads."""

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        mean = super().forward(state, model_time)
        variance = torch.full_like(mean, -0.5)
        return torch.cat((mean, variance), dim=1)


class MappingMeanAbsoluteErrorMetric(Metric):
    """Return a dynamic mapping around one mean absolute error."""

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "total",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        absolute_error = (prediction - target).abs()
        self.total += absolute_error.sum()
        self.count += absolute_error.numel()

    def compute(self) -> dict[str, torch.Tensor]:
        mean = self.total / self.count
        return {"mae": mean, "twice": mean * 2.0}


class BinaryAccuracyMetric(Metric):
    """Compute binary accuracy for a custom higher-is-better metric."""

    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "correct",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        predicted_class = prediction >= 0.5
        target_class = target >= 0.5
        self.correct += (predicted_class == target_class).sum()
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


REGISTRIES.metrics.add(
    "test.training_mapping_mae",
    MappingMeanAbsoluteErrorMetric,
)
REGISTRIES.metrics.add(
    "test.training_binary_accuracy",
    BinaryAccuracyMetric,
)


class SkipFirstOptimizerStep(PrecisionRuntime):
    """Simulate one overflow before allowing optimizer progress."""

    def __init__(self) -> None:
        super().__init__(
            kind="fp32",
            device_type="cpu",
            autocast_dtype=None,
            grad_scaler=None,
        )
        self.attempts = 0

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        self.attempts += 1
        if self.attempts == 1:
            return False
        optimizer.step()
        return True


class IncreasingTargetLoader:
    """Yield a larger validation error each time a new epoch iterates it."""

    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield (
            torch.tensor([[0.0]]),
            torch.tensor([[float(self.iterations)]]),
        )


class ImprovingAccuracyLoader:
    """Yield incorrect labels first and correct labels on the next epoch."""

    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        targets = (
            torch.tensor([[1.0], [0.0]])
            if self.iterations == 1
            else torch.tensor([[0.0], [1.0]])
        )
        yield torch.tensor([[0.0], [1.0]]), targets


class IterationCountingLoader:
    """Record whether semantic preflight allowed iteration to start."""

    def __init__(self, batches: list[Any]) -> None:
        self.batches = batches
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from self.batches


def build_metric_trainer(
    tmp_path,
    *,
    declarations: list[MetricConfig],
    accumulate_grad_batches: int = 1,
    precision: PrecisionRuntime | None = None,
) -> tuple[Trainer, MetricRecordingLogger]:
    """Build a deterministic supervised trainer with configured metrics."""

    model = ScalarRegressor()
    objective = nn.MSELoss()
    strategy = SupervisedTrainingStrategy(model, objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    plan = TrainingPlan(
        strategy=strategy,
        primary_model=model,
        objective=objective,
    )
    logger = MetricRecordingLogger()
    manager = CheckpointManager(
        model=model,
        objective=objective,
        optimizer=optimizer,
    )
    runtime = TrainingMetricRuntime(
        declarations,
        strategy,
        device="cpu",
    )
    return (
        Trainer(
            plan,
            optimizer,
            device="cpu",
            logger=logger,
            metric_runtime=runtime,
            checkpoint_manager=manager,
            checkpoint_dir=tmp_path / "checkpoints",
            checkpoint_every=1,
            accumulate_grad_batches=accumulate_grad_batches,
            precision=precision,
        ),
        logger,
    )


def metric_declarations() -> list[MetricConfig]:
    """Return one declaration shared across all three training phases."""

    return [
        MetricConfig(
            id="prediction_mae",
            name="mae",
            channel="supervised.prediction_target",
            phases=["train", "validation", "test"],
        )
    ]


def test_phase_metrics_share_canonical_history_logger_and_checkpoint_keys(
    tmp_path,
) -> None:
    trainer, logger = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )
    train = [
        (torch.tensor([[0.0], [0.0]]), torch.tensor([[1.0], [1.0]])),
    ]
    validation = [
        (torch.tensor([[0.0]]), torch.tensor([[2.0]])),
    ]

    history = trainer.fit(
        train,
        num_epochs=1,
        validation_dataloader=validation,
        show_progress=False,
        early_stopping_monitor="valid/metrics/prediction_mae",
        track_best=True,
    )

    assert history[0]["train/metrics/prediction_mae"] == pytest.approx(1.0)
    assert history[0]["valid/metrics/prediction_mae"] == pytest.approx(2.0)
    assert any(
        payload.get("valid/metrics/prediction_mae") == pytest.approx(2.0)
        for payload in logger.payloads
    )
    payload = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "best.pt"
    )
    assert "metrics" in payload
    assert payload["metrics"] == history[0]
    assert "metadata" in payload
    sources = payload["metadata"]["metric_sources"]
    assert sources["valid/metrics/prediction_mae"] == {
        "origin": "phase",
        "data_role": "validation",
        "protocol_id": None,
        "selection_eligible": True,
    }


def test_test_metric_state_is_isolated_from_train_and_validation(tmp_path) -> None:
    trainer, logger = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )
    trainer.fit(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))],
        num_epochs=1,
        validation_dataloader=[
            (torch.tensor([[0.0]]), torch.tensor([[2.0]]))
        ],
        show_progress=False,
        track_best=False,
    )

    result = trainer.evaluate_epoch(
        [(torch.tensor([[0.0]]), torch.tensor([[7.0]]))],
        metric_prefix="test",
        show_progress=False,
    )

    assert result["test/metrics/prediction_mae"] == pytest.approx(7.0)
    assert any(
        payload.get("test/metrics/prediction_mae") == pytest.approx(7.0)
        for payload in logger.payloads
    )


def test_builtin_strategy_weights_variable_batches_for_epoch_loss(
    tmp_path,
) -> None:
    trainer, _ = build_metric_trainer(tmp_path, declarations=[])

    result = trainer.evaluate_epoch(
        [
            (
                torch.zeros(2, 1),
                torch.ones(2, 1),
            ),
            (
                torch.zeros(1, 1),
                torch.full((1, 1), 3.0),
            ),
        ],
        show_progress=False,
        log_metrics=False,
    )

    assert result["loss"] == pytest.approx(11.0 / 3.0)


def test_train_metrics_include_every_microbatch_in_accumulation_window(
    tmp_path,
) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="supervised.prediction_target",
                phases=["train"],
            )
        ],
        accumulate_grad_batches=2,
    )

    history = trainer.fit(
        [
            (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[0.0]]), torch.tensor([[3.0]])),
        ],
        num_epochs=1,
        show_progress=False,
        track_best=False,
    )

    assert history[0]["train/metrics/prediction_mae"] == pytest.approx(2.0)


def test_train_metrics_exclude_skipped_optimizer_windows(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="supervised.prediction_target",
                phases=["train"],
            )
        ],
        precision=SkipFirstOptimizerStep(),
    )

    history = trainer.fit(
        [
            (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
            (torch.tensor([[0.0]]), torch.tensor([[3.0]])),
        ],
        num_epochs=1,
        show_progress=False,
        track_best=False,
    )

    assert history[0]["train/metrics/prediction_mae"] == pytest.approx(3.0)
    assert history[0]["system/train/skipped_optimizer_steps"] == 1.0


def test_validation_only_metrics_add_no_train_payload_retention(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="supervised.prediction_target",
                phases=["validation"],
            )
        ],
    )

    def unexpected_detach(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("validation-only metrics detached train payloads")

    monkeypatch.setattr(
        "stochaflow.training.trainer.detach_metric_updates",
        unexpected_detach,
    )
    trainer.train_epoch(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))],
        show_progress=False,
    )


def test_configured_metrics_preserve_custom_evaluation_prefix(
    tmp_path,
) -> None:
    trainer, logger = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="supervised.prediction_target",
                phases=["validation"],
            )
        ],
    )

    result = trainer.evaluate_epoch(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))],
        metric_prefix="holdout",
        show_progress=False,
    )

    assert result["loss"] == pytest.approx(1.0)
    assert any(
        payload.get("holdout/loss") == pytest.approx(1.0)
        for payload in logger.payloads
    )
    assert all(
        "holdout/metrics/prediction_mae" not in payload
        for payload in logger.payloads
    )


def test_best_checkpoint_supports_max_mode_phase_metric(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )

    trainer.fit(
        [(torch.tensor([[0.0]]), torch.tensor([[0.0]]))],
        num_epochs=2,
        validation_dataloader=IncreasingTargetLoader(),
        show_progress=False,
        early_stopping_monitor="valid/metrics/prediction_mae",
        early_stopping_mode="max",
        track_best=True,
    )

    assert trainer.best_epoch == 2
    assert trainer.best_metric_value == pytest.approx(2.0)


def test_best_checkpoint_supports_max_mode_custom_accuracy(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="binary_accuracy",
                name="test.training_binary_accuracy",
                channel="supervised.prediction_target",
                phases=["validation"],
            )
        ],
    )

    trainer.fit(
        [(torch.tensor([[0.0]]), torch.tensor([[0.0]]))],
        num_epochs=2,
        validation_dataloader=ImprovingAccuracyLoader(),
        show_progress=False,
        early_stopping_monitor="valid/metrics/binary_accuracy",
        early_stopping_mode="max",
        track_best=True,
    )

    assert trainer.best_epoch == 2
    assert trainer.best_metric_value == pytest.approx(1.0)


def test_non_finite_phase_metric_fails_closed_when_monitored(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )

    with pytest.raises(
        ValueError,
        match=r"prediction_mae.*non-finite at epoch 1",
    ):
        trainer.fit(
            [(torch.tensor([[0.0]]), torch.tensor([[0.0]]))],
            num_epochs=1,
            validation_dataloader=[
                (torch.tensor([[0.0]]), torch.tensor([[float("nan")]]))
            ],
            show_progress=False,
            early_stopping_monitor="valid/metrics/prediction_mae",
            track_best=True,
        )


def test_gaussian_phase_metrics_run_end_to_end_without_external_data(
    tmp_path,
) -> None:
    torch.manual_seed(7)
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 4,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )
    model = TinyGaussianDenoiser()
    objective = MSEObjective()
    strategy = GaussianDenoisingTrainingStrategy(
        model,
        process,
        objective,
        prediction_type="epsilon",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime = TrainingMetricRuntime(
        [
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="gaussian.prediction_target",
                phases=["validation", "test"],
            ),
            MetricConfig(
                id="clean_reconstruction_mse",
                name="mse",
                channel="gaussian.clean_reconstruction",
                phases=["validation", "test"],
            ),
        ],
        strategy,
        device="cpu",
    )
    manager = CheckpointManager(
        model=model,
        process=process,
        objective=objective,
        optimizer=optimizer,
    )
    trainer = Trainer(
        TrainingPlan(
            strategy=strategy,
            primary_model=model,
            process=process,
            objective=objective,
        ),
        optimizer,
        device="cpu",
        metric_runtime=runtime,
        checkpoint_manager=manager,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    clean = torch.zeros(2, 1, 4, 4)

    history = trainer.fit(
        [clean],
        num_epochs=1,
        validation_dataloader=[clean],
        show_progress=False,
        early_stopping_monitor=(
            "valid/metrics/clean_reconstruction_mse"
        ),
        track_best=True,
    )
    test_metrics = trainer.evaluate_epoch(
        [clean],
        metric_prefix="test",
        show_progress=False,
    )

    for key in (
        "valid/metrics/prediction_mae",
        "valid/metrics/clean_reconstruction_mse",
    ):
        assert math.isfinite(history[0][key])
    for key in (
        "test/metrics/prediction_mae",
        "test/metrics/clean_reconstruction_mse",
    ):
        assert math.isfinite(test_metrics[key])
    assert trainer.best_checkpoint_path == tmp_path / "checkpoints" / "best.pt"


def test_learned_range_p2_metrics_aggregate_uneven_batches_by_samples(
    tmp_path,
) -> None:
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 4,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )
    model = TinyLearnedVarianceGaussianDenoiser()
    objective = MSEObjective()
    strategy = GaussianDenoisingTrainingStrategy(
        model,
        process,
        objective,
        prediction_type="epsilon",
        variance=GaussianVarianceConfig(
            mode="learned_range",
            loss="rescaled_variational_bound",
        ),
        loss_weighting=GaussianLossWeightingConfig(
            name="p2",
            k=1.0,
            gamma=1.0,
        ),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    runtime = TrainingMetricRuntime(
        [
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="gaussian.prediction_target",
                phases=["validation"],
            )
        ],
        strategy,
        device="cpu",
    )
    trainer = Trainer(
        TrainingPlan(
            strategy=strategy,
            primary_model=model,
            process=process,
            objective=objective,
        ),
        optimizer,
        device="cpu",
        metric_runtime=runtime,
    )
    batches = [
        torch.zeros(2, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
    ]

    torch.manual_seed(29)
    expected_outputs = [
        strategy.evaluation_step(batch)
        for batch in batches
    ]
    expected_loss = sum(
        output.loss.item() * float(output.loss_aggregation_weight)
        for output in expected_outputs
    ) / 3.0
    assert [
        output.loss_aggregation_weight
        for output in expected_outputs
    ] == [2, 1]

    torch.manual_seed(29)
    result = trainer.evaluate_epoch(
        batches,
        show_progress=False,
        log_metrics=False,
    )

    assert result["loss"] == pytest.approx(expected_loss)
    assert math.isfinite(result["valid/metrics/prediction_mae"])


def test_metric_compatibility_fails_at_training_composition_boundary(
    tmp_path,
) -> None:
    del tmp_path
    model = ScalarRegressor()
    strategy = SupervisedTrainingStrategy(model, nn.MSELoss())

    with pytest.raises(ValueError, match="not provided"):
        TrainingMetricRuntime(
            [
                MetricConfig(
                    id="bad",
                    name="mae",
                    channel="custom.missing",
                )
            ],
            strategy,
            device="cpu",
        )


def test_metric_runtime_reports_ids_by_phase(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )
    runtime = trainer.metric_runtime

    assert runtime is not None
    assert runtime.has_metric("train", "prediction_mae")
    assert runtime.has_metric("validation", "prediction_mae")
    assert runtime.has_metric("test", "prediction_mae")
    assert not runtime.has_metric("validation", "missing")


def test_metric_runtime_moves_every_phase_engine_and_preserves_state() -> None:
    model = ScalarRegressor()
    strategy = SupervisedTrainingStrategy(model, nn.MSELoss())
    runtime = TrainingMetricRuntime(
        metric_declarations(),
        strategy,
        device="cpu",
    )
    runtime.update_phase(
        "train",
        {
            "supervised.prediction_target": MetricUpdate(
                args=(torch.tensor([3.0]), torch.tensor([1.0])),
            )
        },
    )

    assert runtime.to(torch.device("cpu")) is runtime
    assert runtime.compute_phase("train") == {
        "train/metrics/prediction_mae": pytest.approx(2.0)
    }


def test_unknown_metric_monitor_fails_before_loader_iteration(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )
    train = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))]
    )
    validation = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[2.0]]))]
    )

    with pytest.raises(
        ValueError,
        match=r"metric id 'typo'.*validation phase",
    ):
        trainer.fit(
            train,
            num_epochs=1,
            validation_dataloader=validation,
            show_progress=False,
            early_stopping_monitor="valid/metrics/typo",
            track_best=True,
        )

    assert train.iterations == 0
    assert validation.iterations == 0


def test_metric_monitor_phase_mismatch_fails_before_iteration(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="prediction_mae",
                name="mae",
                channel="supervised.prediction_target",
                phases=["train"],
            )
        ],
    )
    train = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))]
    )
    validation = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[2.0]]))]
    )

    with pytest.raises(
        ValueError,
        match=r"prediction_mae.*validation phase",
    ):
        trainer.fit(
            train,
            num_epochs=1,
            validation_dataloader=validation,
            show_progress=False,
            early_stopping_monitor="valid/metrics/prediction_mae",
            track_best=True,
        )

    assert train.iterations == 0
    assert validation.iterations == 0


def test_validation_monitor_requires_loader_before_training(tmp_path) -> None:
    trainer, _ = build_metric_trainer(tmp_path, declarations=[])
    train = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))]
    )

    with pytest.raises(ValueError, match="requires a validation dataloader"):
        trainer.fit(
            train,
            num_epochs=1,
            show_progress=False,
            early_stopping_monitor="valid/loss",
            track_best=True,
        )

    assert train.iterations == 0


def test_unconsumed_metric_monitor_skips_semantic_preflight(tmp_path) -> None:
    trainer, _ = build_metric_trainer(tmp_path, declarations=[])
    train = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))]
    )

    history = trainer.fit(
        train,
        num_epochs=1,
        show_progress=False,
        early_stopping_monitor="valid/metrics/typo",
        track_best=False,
    )

    assert train.iterations == 1
    assert history[0]["train/loss"] == pytest.approx(1.0)


def test_mapping_metric_subkey_passes_base_id_preflight(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="scorecard",
                name="test.training_mapping_mae",
                channel="supervised.prediction_target",
                phases=["validation"],
            )
        ],
    )

    history = trainer.fit(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))],
        num_epochs=1,
        validation_dataloader=[
            (torch.tensor([[0.0]]), torch.tensor([[2.0]]))
        ],
        show_progress=False,
        early_stopping_monitor="valid/metrics/scorecard/mae",
        track_best=True,
    )

    assert history[0]["valid/metrics/scorecard/mae"] == pytest.approx(2.0)
    assert history[0]["valid/metrics/scorecard/twice"] == pytest.approx(4.0)
    assert trainer.best_metric_value == pytest.approx(2.0)


def test_missing_dynamic_metric_subkey_fails_after_compute(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=[
            MetricConfig(
                id="scorecard",
                name="test.training_mapping_mae",
                channel="supervised.prediction_target",
                phases=["validation"],
            )
        ],
    )
    train = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))]
    )
    validation = IterationCountingLoader(
        [(torch.tensor([[0.0]]), torch.tensor([[2.0]]))]
    )

    with pytest.raises(ValueError, match=r"scorecard/missing.*not found"):
        trainer.fit(
            train,
            num_epochs=1,
            validation_dataloader=validation,
            show_progress=False,
            early_stopping_monitor="valid/metrics/scorecard/missing",
            track_best=True,
        )

    assert train.iterations == 1
    assert validation.iterations == 1


def test_test_phase_monitor_is_rejected_before_training(tmp_path) -> None:
    trainer, _ = build_metric_trainer(
        tmp_path,
        declarations=metric_declarations(),
    )

    with pytest.raises(ValueError, match="test metrics cannot be used"):
        trainer.fit(
            [(torch.tensor([[0.0]]), torch.tensor([[1.0]]))],
            num_epochs=1,
            show_progress=False,
            early_stopping_monitor="test/metrics/prediction_mae",
            track_best=True,
        )
