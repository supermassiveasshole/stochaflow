"""Tests for automatic gradient-accumulation lifecycle semantics."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.training.builder import TrainingPlan
from stochaflow.training.diagnostics.contracts import (
    TrainBatchEndEvent,
    TrainingDiagnostic,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.precision import PrecisionRuntime
from stochaflow.training.strategy import (
    TrainingStrategy,
    TrainStepOutput,
)
from stochaflow.training.trainer import Trainer, TrainingPhaseProfiler
from stochaflow.utils.logging import ExperimentLogger


class LinearMeanSquaredStrategy(TrainingStrategy):
    """Compute a logical-sample mean loss for one linear model."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        inputs, targets = batch
        loss = (self.model(inputs) - targets).square().mean()
        return TrainStepOutput(loss=loss)


class CountingStepScheduler(LRScheduler):
    """Count lifecycle calls without changing the learning rate."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.count = -1
        super().__init__(optimizer)

    def step(self) -> None:
        self.count += 1


class SkippingPrecisionRuntime(PrecisionRuntime):
    """Simulate an FP16 overflow after backward."""

    def __init__(self) -> None:
        super().__init__(
            kind="fp32",
            device_type="cpu",
            autocast_dtype=None,
            grad_scaler=None,
        )

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        del optimizer
        return False


class SkipFirstPrecisionRuntime(PrecisionRuntime):
    """Skip one optimizer attempt before allowing later updates."""

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


class OrderedPrecisionRuntime(PrecisionRuntime):
    """Record the unscale and optimizer-step ordering."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(
            kind="fp32",
            device_type="cpu",
            autocast_dtype=None,
            grad_scaler=None,
        )
        self.events = events

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        self.events.append("unscale")

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        self.events.append("step")
        optimizer.step()
        return True


class FailingSecondStepStrategy(TrainingStrategy):
    """Raise after one micro-batch has accumulated gradients."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.calls = 0

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("planned failure")
        inputs, targets = batch
        return TrainStepOutput((self.model(inputs) - targets).square().mean())


class RecordingTrainingDiagnostic(TrainingDiagnostic):
    """Record successful optimizer-step diagnostics."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        self.steps.append(event.global_step)


class RecordingMetricLogger(ExperimentLogger):
    """Collect scalar metric payloads for window aggregation assertions."""

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.texts: list[tuple[str, str, int | None]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del step
        self.metrics.append(dict(metrics))

    def log_text(
        self,
        tag: str,
        text: str,
        *,
        step: int | None = None,
    ) -> None:
        self.texts.append((tag, text, step))

    def close(self) -> None:
        return None


class RecordingProgressReporter:
    """Record the successful-update step visible at each micro-batch."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def on_phase_start(self, **kwargs: Any) -> None:
        del kwargs

    def on_batch_end(self, **kwargs: Any) -> None:
        self.steps.append(kwargs["global_step"])

    def on_phase_end(self) -> None:
        return None


class WindowMetricStrategy(TrainingStrategy):
    """Emit partially overlapping metric keys across one window."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.calls = 0

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        self.calls += 1
        inputs, targets = batch
        metrics = {"shared": float(2 * self.calls - 1)}
        if self.calls == 1:
            metrics["first_only"] = 4.0
        return TrainStepOutput(
            (self.model(inputs) - targets).square().mean(),
            metrics=metrics,
        )


class PayloadRetentionStrategy(TrainingStrategy):
    """Track whether non-current diagnostic payloads remain strongly referenced."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.previous_payload: weakref.ReferenceType[torch.Tensor] | None = None
        self.release_checks: list[bool] = []

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        if self.previous_payload is not None:
            gc.collect()
            self.release_checks.append(self.previous_payload() is None)
        inputs, targets = batch
        payload = torch.ones(128, 128, device=inputs.device)
        self.previous_payload = weakref.ref(payload)
        return TrainStepOutput(
            (self.model(inputs) - targets).square().mean(),
            diagnostics={"large_payload": payload},
        )


class LargeHalfMetricStrategy(TrainingStrategy):
    """Emit half-precision metrics whose window sum would overflow."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        inputs, targets = batch
        return TrainStepOutput(
            (self.model(inputs) - targets).square().mean(),
            metrics={"large_half": torch.tensor(40_000.0, dtype=torch.float16)},
        )


class ComplexMetricStrategy(TrainingStrategy):
    """Emit a metric that cannot be represented by the reporting contract."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        inputs, targets = batch
        return TrainStepOutput(
            (self.model(inputs) - targets).square().mean(),
            metrics={"invalid": torch.tensor(1.0 + 2.0j)},
        )


def regression_batches(count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return deterministic one-sample regression micro-batches."""

    return [
        (
            torch.tensor([[float(index + 1)]]),
            torch.tensor([[0.5 * float(index)]]),
        )
        for index in range(count)
    ]


def build_trainer(
    *,
    model: nn.Linear,
    strategy: TrainingStrategy,
    accumulate_grad_batches: int,
    precision: PrecisionRuntime | None = None,
    lr_scheduler_interval: str = "step",
    diagnostics: Sequence[TrainingDiagnostic] = (),
    logger: ExperimentLogger | None = None,
    max_grad_norm: float | None = None,
) -> tuple[Trainer, CountingStepScheduler, ExponentialMovingAverage]:
    """Build one lifecycle-complete CPU trainer."""

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = CountingStepScheduler(optimizer)
    ema = ExponentialMovingAverage(model, decay=0.0)
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cpu",
        lr_scheduler=scheduler,
        lr_scheduler_interval=lr_scheduler_interval,
        ema=ema,
        diagnostics=diagnostics,
        logger=logger,
        log_every=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        max_grad_norm=max_grad_norm,
    )
    return trainer, scheduler, ema


def test_five_microbatches_with_k_two_produce_three_updates() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, scheduler, ema = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
    )

    metrics = trainer.train_epoch(
        regression_batches(5),
        show_progress=False,
    )

    assert metrics["num_batches"] == 5.0
    assert metrics["optimizer_steps"] == 3.0
    assert metrics["skipped_optimizer_steps"] == 0.0
    assert trainer.global_step == 3
    assert scheduler.count == 3
    assert ema.num_updates == 3


def test_progress_reports_microbatches_and_window_boundary_steps() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
    )
    reporter = RecordingProgressReporter()

    trainer.train_epoch(
        regression_batches(5),
        show_progress=False,
        reporter=reporter,
    )

    assert reporter.steps == [0, 1, 1, 2, 3]


def test_accumulated_updates_match_corresponding_physical_batches() -> None:
    initial_weight = torch.tensor([[0.25]])
    accumulated = nn.Linear(1, 1, bias=False)
    physical = nn.Linear(1, 1, bias=False)
    accumulated.weight.data.copy_(initial_weight)
    physical.weight.data.copy_(initial_weight)
    microbatches = regression_batches(5)
    physical_batches = [
        (
            torch.cat([microbatches[0][0], microbatches[1][0]]),
            torch.cat([microbatches[0][1], microbatches[1][1]]),
        ),
        (
            torch.cat([microbatches[2][0], microbatches[3][0]]),
            torch.cat([microbatches[2][1], microbatches[3][1]]),
        ),
        microbatches[4],
    ]
    accumulated_trainer, _, _ = build_trainer(
        model=accumulated,
        strategy=LinearMeanSquaredStrategy(accumulated),
        accumulate_grad_batches=2,
    )
    physical_trainer, _, _ = build_trainer(
        model=physical,
        strategy=LinearMeanSquaredStrategy(physical),
        accumulate_grad_batches=1,
    )

    accumulated_trainer.train_epoch(microbatches, show_progress=False)
    physical_trainer.train_epoch(physical_batches, show_progress=False)

    assert torch.equal(accumulated.weight, physical.weight)


def test_different_microbatch_sizes_use_equal_scalar_weighting() -> None:
    initial_weight = torch.tensor([[0.75]])
    accumulated = nn.Linear(1, 1, bias=False)
    expected = nn.Linear(1, 1, bias=False)
    accumulated.weight.data.copy_(initial_weight)
    expected.weight.data.copy_(initial_weight)
    batches = [
        (torch.tensor([[1.0]]), torch.tensor([[0.0]])),
        (
            torch.tensor([[2.0], [3.0], [4.0]]),
            torch.tensor([[0.0], [1.0], [1.0]]),
        ),
    ]
    trainer, _, _ = build_trainer(
        model=accumulated,
        strategy=LinearMeanSquaredStrategy(accumulated),
        accumulate_grad_batches=2,
    )
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.05)
    losses = [
        (expected(inputs) - targets).square().mean()
        for inputs, targets in batches
    ]
    torch.stack(losses).mean().backward()
    expected_optimizer.step()

    trainer.train_epoch(batches, show_progress=False)

    assert torch.equal(accumulated.weight, expected.weight)


def test_max_batch_limit_flushes_a_partial_window_without_extra_pull() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=4,
    )

    metrics = trainer.train_epoch(
        iter(regression_batches(5)),
        max_batches=2,
        show_progress=False,
    )

    assert metrics["num_batches"] == 2.0
    assert metrics["optimizer_steps"] == 1.0
    assert trainer.global_step == 1


def test_epoch_logs_explicit_accumulation_throughput_and_timing_metrics() -> None:
    model = nn.Linear(1, 1, bias=False)
    logger = RecordingMetricLogger()
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
        logger=logger,
    )

    metrics = trainer.train_epoch(
        regression_batches(3),
        show_progress=False,
        profile_phases=True,
    )

    assert metrics["micro_batches"] == 3.0
    assert metrics["optimizer_steps"] == 2.0
    assert metrics["optimizer_steps_per_second"] > 0.0
    assert metrics["data_wait_seconds"] >= 0.0
    assert metrics["compute_seconds"] > 0.0
    assert metrics["forward_seconds"] > 0.0
    assert metrics["backward_seconds"] > 0.0
    assert metrics["optimizer_seconds"] > 0.0
    assert metrics["non_finite_loss_count"] == 0.0
    assert metrics["non_finite_gradient_count"] == 0.0
    epoch_log = next(
        item for item in logger.metrics if "system/train/micro_batches" in item
    )
    assert epoch_log["system/train/micro_batches"] == 3.0
    assert epoch_log["system/train/optimizer_steps"] == 2.0
    assert epoch_log["system/train/optimizer_steps_per_second"] > 0.0
    assert epoch_log["system/train/data_wait_seconds"] >= 0.0
    assert epoch_log["system/train/compute_seconds"] > 0.0
    assert epoch_log["system/train/forward_seconds"] > 0.0
    assert epoch_log["system/train/backward_seconds"] > 0.0
    assert epoch_log["system/train/optimizer_seconds"] > 0.0


def test_epoch_can_stop_after_successful_optimizer_step_budget() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
    )

    metrics = trainer.train_epoch(
        regression_batches(10),
        show_progress=False,
        max_optimizer_steps=3,
    )

    assert metrics["micro_batches"] == 6.0
    assert metrics["optimizer_steps"] == 3.0
    assert trainer.global_step == 3


def test_skipped_optimizer_attempt_does_not_consume_success_budget() -> None:
    model = nn.Linear(1, 1, bias=False)
    runtime = SkipFirstPrecisionRuntime()
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=1,
        precision=runtime,
    )

    metrics = trainer.train_epoch(
        regression_batches(5),
        show_progress=False,
        max_optimizer_steps=2,
    )

    assert runtime.attempts == 3
    assert metrics["micro_batches"] == 3.0
    assert metrics["optimizer_steps"] == 2.0
    assert metrics["skipped_optimizer_steps"] == 1.0
    assert trainer.global_step == 2


def test_finite_loader_can_end_before_success_budget() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=1,
    )

    metrics = trainer.train_epoch(
        regression_batches(2),
        show_progress=False,
        max_optimizer_steps=3,
    )

    assert metrics["micro_batches"] == 2.0
    assert metrics["optimizer_steps"] == 2.0
    assert trainer.global_step == 2


def test_cuda_phase_profiler_synchronizes_only_at_measurement_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronizations: list[torch.device] = []

    class RecordingCudaEvent:
        """Provide deterministic elapsed CUDA event time without a GPU."""

        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            return None

        def elapsed_time(self, end: object) -> float:
            assert isinstance(end, RecordingCudaEvent)
            return 2.0

    monkeypatch.setattr(torch.cuda, "Event", RecordingCudaEvent)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronizations.append(torch.device(device)),
    )
    profiler = TrainingPhaseProfiler(torch.device("cuda:0"))

    measurement_started_at = profiler.start_measurement()
    for phase in (
        "forward_seconds",
        "backward_seconds",
        "optimizer_seconds",
    ):
        phase_started_at = profiler.start_phase()
        profiler.end_phase(phase, phase_started_at)
    duration, phases = profiler.finish_measurement(
        measurement_started_at
    )

    assert synchronizations == [
        torch.device("cuda:0"),
        torch.device("cuda:0"),
    ]
    assert duration >= 0.0
    assert phases == {
        "forward_seconds": 0.002,
        "backward_seconds": 0.002,
        "optimizer_seconds": 0.002,
    }


def test_overflow_skips_all_success_owned_lifecycle() -> None:
    model = nn.Linear(1, 1, bias=False)
    diagnostic = RecordingTrainingDiagnostic()
    logger = RecordingMetricLogger()
    trainer, scheduler, ema = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
        precision=SkippingPrecisionRuntime(),
        diagnostics=(diagnostic,),
        logger=logger,
    )
    before = model.weight.detach().clone()

    with pytest.warns(RuntimeWarning, match="all optimizer windows"):
        metrics = trainer.train_epoch(
            regression_batches(2),
            show_progress=False,
        )

    assert torch.equal(model.weight, before)
    assert metrics["optimizer_steps"] == 0.0
    assert metrics["skipped_optimizer_steps"] == 1.0
    assert trainer.global_step == 0
    assert scheduler.count == 0
    assert ema.num_updates == 0
    assert diagnostic.steps == []
    assert all(parameter.grad is None for parameter in model.parameters())
    assert logger.texts == [
        (
            "training/optimizer_overflow",
            (
                "all optimizer windows were skipped in the training epoch; "
                "skipped_windows=1"
            ),
            0,
        )
    ]


def test_epoch_scheduler_does_not_advance_when_every_window_overflows() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, scheduler, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=1,
        precision=SkippingPrecisionRuntime(),
        lr_scheduler_interval="epoch",
    )

    with pytest.warns(RuntimeWarning, match="all optimizer windows"):
        trainer.fit(
            regression_batches(2),
            num_epochs=1,
            show_progress=False,
            track_best=False,
        )

    assert scheduler.count == 0


def test_exception_clears_partial_window_gradients() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=FailingSecondStepStrategy(model),
        accumulate_grad_batches=2,
    )

    with pytest.raises(RuntimeError, match="planned failure"):
        trainer.train_epoch(regression_batches(2), show_progress=False)

    assert all(parameter.grad is None for parameter in model.parameters())
    assert trainer.global_step == 0


def test_zero_grad_occurs_only_at_window_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
    )
    calls = 0
    original_zero_grad = trainer.optimizer.zero_grad

    def recording_zero_grad(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original_zero_grad(*args, **kwargs)

    monkeypatch.setattr(trainer.optimizer, "zero_grad", recording_zero_grad)

    trainer.train_epoch(regression_batches(3), show_progress=False)

    assert calls == 3


def test_clipping_orders_unscale_before_clip_and_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    events: list[str] = []
    runtime = OrderedPrecisionRuntime(events)
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(*args: Any, **kwargs: Any) -> torch.Tensor:
        events.append("clip")
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=2,
        precision=runtime,
        max_grad_norm=1.0,
    )

    trainer.train_epoch(regression_batches(2), show_progress=False)

    assert events == ["unscale", "clip", "step"]


def test_window_metrics_average_each_key_by_its_appearance_count() -> None:
    model = nn.Linear(1, 1, bias=False)
    logger = RecordingMetricLogger()
    trainer, _, _ = build_trainer(
        model=model,
        strategy=WindowMetricStrategy(model),
        accumulate_grad_batches=2,
        logger=logger,
    )

    trainer.train_epoch(regression_batches(2), show_progress=False)

    step_metrics = next(
        metrics
        for metrics in logger.metrics
        if "train/step/strategy/shared" in metrics
    )
    assert step_metrics["train/step/strategy/shared"] == 2.0
    assert step_metrics["train/step/strategy/first_only"] == 4.0


def test_window_metric_sum_does_not_overflow_source_scalar_dtype() -> None:
    model = nn.Linear(1, 1, bias=False)
    logger = RecordingMetricLogger()
    trainer, _, _ = build_trainer(
        model=model,
        strategy=LargeHalfMetricStrategy(model),
        accumulate_grad_batches=2,
        logger=logger,
    )

    trainer.train_epoch(regression_batches(2), show_progress=False)

    step_metrics = next(
        metrics
        for metrics in logger.metrics
        if "train/step/strategy/large_half" in metrics
    )
    assert step_metrics["train/step/strategy/large_half"] == 40_000.0


def test_invalid_metric_fails_before_optimizer_lifecycle_commits() -> None:
    model = nn.Linear(1, 1, bias=False)
    trainer, scheduler, ema = build_trainer(
        model=model,
        strategy=ComplexMetricStrategy(model),
        accumulate_grad_batches=1,
    )
    before = model.weight.detach().clone()

    with pytest.raises(TypeError, match="real numeric"):
        trainer.train_epoch(regression_batches(1), show_progress=False)

    assert torch.equal(model.weight, before)
    assert trainer.global_step == 0
    assert scheduler.count == 0
    assert ema.num_updates == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_accumulation_retains_only_the_current_microbatch_payload() -> None:
    model = nn.Linear(1, 1, bias=False)
    strategy = PayloadRetentionStrategy(model)
    trainer, _, _ = build_trainer(
        model=model,
        strategy=strategy,
        accumulate_grad_batches=2,
    )

    trainer.train_epoch(regression_batches(3), show_progress=False)

    assert strategy.release_checks == [True, True]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_accumulation_materializes_mps_losses_without_mps_float64() -> None:
    model = nn.Linear(1, 1, bias=False, device="mps")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer = Trainer(
        TrainingPlan(
            strategy=LinearMeanSquaredStrategy(model),
            primary_model=model,
        ),
        optimizer,
        device="mps",
        accumulate_grad_batches=2,
    )

    metrics = trainer.train_epoch(regression_batches(2), show_progress=False)

    assert metrics["optimizer_steps"] == 1.0
    assert torch.isfinite(model.weight).all()


def test_direct_train_batch_owns_one_successful_global_step() -> None:
    model = nn.Linear(1, 1, bias=False)
    diagnostic = RecordingTrainingDiagnostic()
    trainer, scheduler, ema = build_trainer(
        model=model,
        strategy=LinearMeanSquaredStrategy(model),
        accumulate_grad_batches=4,
        diagnostics=(diagnostic,),
    )

    trainer.train_batch(regression_batches(1)[0])

    assert trainer.global_step == 1
    assert scheduler.count == 1
    assert ema.num_updates == 1
    assert diagnostic.steps == []
