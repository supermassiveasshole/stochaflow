"""CUDA integration tests for mixed-precision optimizer semantics."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.training.builder import TrainingPlan
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.precision import (
    PrecisionRuntime,
    build_precision_runtime,
)
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput
from stochaflow.training.trainer import Trainer
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    parse_rng_state,
    restore_rng_state,
)
from stochaflow.utils.logging import ExperimentLogger


class CudaRegressionStrategy(TrainingStrategy):
    """Record the autocast output dtype for a tiny regression objective."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.prediction_dtypes: list[torch.dtype] = []

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        inputs, targets = batch
        prediction = self.model(inputs)
        self.prediction_dtypes.append(prediction.dtype)
        return TrainStepOutput((prediction.float() - targets).square().mean())


class ToggleOverflowStrategy(TrainingStrategy):
    """Produce a deliberate non-finite gradient for one FP16 window."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.overflow = True

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        inputs, targets = batch
        loss = (self.model(inputs).float() - targets).square().mean()
        if self.overflow:
            loss = loss * torch.tensor(float("inf"), device=loss.device)
        return TrainStepOutput(loss)


class SelectedCallOverflowStrategy(TrainingStrategy):
    """Use CUDA RNG every call and overflow only selected micro-batches."""

    def __init__(
        self,
        model: nn.Linear,
        *,
        overflow_calls: set[int],
    ) -> None:
        self.model = model
        self.overflow_calls = set(overflow_calls)
        self.calls = 0

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        self.calls += 1
        inputs, targets = batch
        random_factor = 0.5 + torch.rand((), device=inputs.device)
        loss = (
            (self.model(inputs).float() - targets).square().mean()
            * random_factor
        )
        if self.calls in self.overflow_calls:
            loss = loss * torch.tensor(float("inf"), device=loss.device)
        return TrainStepOutput(loss)


class CudaCountingScheduler(LRScheduler):
    """Count successful optimizer updates in the CUDA resume integration test."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.count = -1
        super().__init__(optimizer)

    def step(self) -> None:
        self.count += 1

    def state_dict(self) -> dict[str, int]:
        return {"count": self.count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.count = state["count"]


class CudaMetricLogger(ExperimentLogger):
    """Capture epoch-level CUDA precision metrics."""

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
    ) -> None:
        del step
        self.metrics.append(dict(metrics))

    def close(self) -> None:
        return None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_cuda_bf16_keeps_model_and_optimizer_state_in_fp32() -> None:
    model = nn.Linear(8, 8)
    strategy = CudaRegressionStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = build_precision_runtime("bf16-mixed", "cuda")
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cuda",
        precision=precision,
    )
    batch = (torch.randn(4, 8), torch.randn(4, 8))

    trainer.train_batch(batch)

    assert strategy.prediction_dtypes == [torch.bfloat16]
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    optimizer_tensors = (
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
    )
    assert all(value.dtype == torch.float32 for value in optimizer_tensors)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_overflow_skips_then_recovers_next_update() -> None:
    model = nn.Linear(8, 8)
    strategy = ToggleOverflowStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = build_precision_runtime("fp16-mixed", "cuda")
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cuda",
        precision=precision,
    )
    batch = (torch.randn(4, 8), torch.randn(4, 8))
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    scaler = precision.grad_scaler
    assert scaler is not None
    scale_before = scaler.get_scale()

    trainer.train_batch(batch)

    assert trainer.global_step == 0
    assert scaler.get_scale() < scale_before
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    strategy.overflow = False
    trainer.train_batch(batch)

    assert trainer.global_step == 1
    assert any(
        not torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_scale_underflow_fails_before_false_success() -> None:
    minimum_positive_float32 = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    ).item()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scaler = torch.cuda.amp.GradScaler(
            init_scale=minimum_positive_float32,
        )
    precision = PrecisionRuntime(
        kind="fp16-mixed",
        device_type="cuda",
        autocast_dtype=torch.float16,
        grad_scaler=scaler,
    )
    model = nn.Linear(8, 8)
    strategy = ToggleOverflowStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = ExponentialMovingAverage(model, decay=0.0)
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cuda",
        precision=precision,
        ema=ema,
    )
    batch = (torch.randn(4, 8), torch.randn(4, 8))
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    with pytest.raises(
        RuntimeError,
        match="GradScaler scale after optimizer step must be a finite positive",
    ):
        trainer.train_batch(batch)

    assert scaler.get_scale() == 0.0
    assert trainer.global_step == 0
    assert ema.num_updates == 0
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())

    with pytest.raises(
        RuntimeError,
        match="GradScaler scale before optimizer step must be a finite positive",
    ):
        trainer.train_batch(batch)

    assert trainer.global_step == 0
    assert ema.num_updates == 0
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_scale_growth_counts_as_successful_update() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scaler = torch.cuda.amp.GradScaler(
            init_scale=8.0,
            growth_interval=1,
        )
    precision = PrecisionRuntime(
        kind="fp16-mixed",
        device_type="cuda",
        autocast_dtype=torch.float16,
        grad_scaler=scaler,
    )
    model = nn.Linear(8, 8)
    strategy = CudaRegressionStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cuda",
        precision=precision,
    )
    scale_before = scaler.get_scale()

    trainer.train_batch((torch.randn(4, 8), torch.randn(4, 8)))

    assert trainer.global_step == 1
    assert scaler.get_scale() > scale_before


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_checkpoint_manager_must_share_runtime_scaler(
    tmp_path: Path,
) -> None:
    model = nn.Linear(8, 8)
    strategy = CudaRegressionStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = build_precision_runtime("fp16-mixed", "cuda")
    checkpoint_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        precision_kind=precision.kind,
        grad_scaler=precision.grad_scaler,
    )
    Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cuda",
        precision=precision,
        checkpoint_manager=checkpoint_manager,
        checkpoint_dir=tmp_path,
    )

    other_precision = build_precision_runtime("fp16-mixed", "cuda")
    mismatched_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        precision_kind=other_precision.kind,
        grad_scaler=other_precision.grad_scaler,
    )
    with pytest.raises(
        ValueError,
        match=r"CheckpointManager GradScaler.*Trainer",
    ):
        Trainer(
            TrainingPlan(strategy=strategy, primary_model=model),
            optimizer,
            device="cuda",
            precision=precision,
            checkpoint_manager=mismatched_manager,
            checkpoint_dir=tmp_path,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_epoch_logs_current_loss_scale() -> None:
    model = nn.Linear(8, 8)
    strategy = CudaRegressionStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    precision = build_precision_runtime("fp16-mixed", "cuda")
    logger = CudaMetricLogger()
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cuda",
        precision=precision,
        logger=logger,
    )

    trainer.train_epoch(
        [(torch.randn(4, 8), torch.randn(4, 8))],
        show_progress=False,
    )

    scaler = precision.grad_scaler
    assert scaler is not None
    epoch_metrics = next(
        metrics
        for metrics in logger.metrics
        if "train/epoch_loss_scale" in metrics
    )
    assert epoch_metrics["train/epoch_loss_scale"] == scaler.get_scale()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_ema_checkpoint_uses_one_synchronized_host_snapshot() -> None:
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.BatchNorm1d(8),
    ).cuda()
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.update(model)

    payload = CheckpointManager(model=model, ema=ema).build_state()
    ema_state = payload.get("ema_state_dict")
    ema_projection = payload.get("ema_model_state_dict")

    assert ema_state is not None
    assert ema_projection is not None
    shadow_values = {
        **ema_state["shadow_params"],
        **ema_state["shadow_buffers"],
    }
    assert all(tensor.device.type == "cpu" for tensor in shadow_values.values())
    assert all(
        tensor.device.type == "cpu"
        for tensor in ema_projection.values()
        if isinstance(tensor, torch.Tensor)
    )
    for name, shadow in shadow_values.items():
        torch.testing.assert_close(
            ema_projection[name],
            shadow,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_overflow_checkpoint_resume_matches_next_success(
    tmp_path: Path,
) -> None:
    def build_stack(checkpoint_dir: Path):
        model = nn.Linear(8, 8)
        strategy = ToggleOverflowStrategy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CudaCountingScheduler(optimizer)
        ema = ExponentialMovingAverage(model, decay=0.9)
        precision = build_precision_runtime("fp16-mixed", "cuda")
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            ema=ema,
            precision_kind=precision.kind,
            grad_scaler=precision.grad_scaler,
        )
        trainer = Trainer(
            TrainingPlan(strategy=strategy, primary_model=model),
            optimizer,
            device="cuda",
            lr_scheduler=scheduler,
            ema=ema,
            precision=precision,
            checkpoint_manager=manager,
            checkpoint_dir=checkpoint_dir,
        )
        return trainer, strategy, manager

    torch.manual_seed(123)
    uninterrupted, uninterrupted_strategy, uninterrupted_manager = build_stack(
        tmp_path / "uninterrupted"
    )
    batch = (
        torch.linspace(-1.0, 1.0, steps=32).reshape(4, 8),
        torch.zeros(4, 8),
    )

    uninterrupted.train_batch(batch)

    assert uninterrupted.global_step == 0
    checkpoint = uninterrupted_manager.save(
        tmp_path / "overflow.pt",
        epoch=1,
        global_step=uninterrupted.global_step,
        config={
            "trainer": {
                "precision": "fp16-mixed",
                "accumulate_grad_batches": 1,
            }
        },
    )
    uninterrupted_strategy.overflow = False
    uninterrupted.train_batch(batch)
    expected = uninterrupted_manager.build_state()

    resumed, resumed_strategy, resumed_manager = build_stack(
        tmp_path / "resumed"
    )
    resumed_strategy.overflow = False
    loaded = resumed_manager.load(checkpoint, map_location="cuda")
    assert loaded.global_step == 0
    resumed.global_step = loaded.global_step
    resumed.train_batch(batch)
    actual = resumed_manager.build_state()

    assert resumed.global_step == uninterrupted.global_step == 1
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "ema_state_dict",
        "grad_scaler_state_dict",
    ):
        torch.testing.assert_close(
            actual.get(key),
            expected.get(key),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_fp16_partial_window_overflow_resume_preserves_rng_and_state(
    tmp_path: Path,
) -> None:
    def build_stack(
        checkpoint_dir: Path,
        *,
        overflow_calls: set[int],
    ):
        model = nn.Linear(8, 8)
        strategy = SelectedCallOverflowStrategy(
            model,
            overflow_calls=overflow_calls,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CudaCountingScheduler(optimizer)
        ema = ExponentialMovingAverage(model, decay=0.9)
        precision = build_precision_runtime("fp16-mixed", "cuda")
        manager = CheckpointManager(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            ema=ema,
            precision_kind=precision.kind,
            grad_scaler=precision.grad_scaler,
        )
        trainer = Trainer(
            TrainingPlan(strategy=strategy, primary_model=model),
            optimizer,
            device="cuda",
            lr_scheduler=scheduler,
            ema=ema,
            precision=precision,
            accumulate_grad_batches=2,
            checkpoint_manager=manager,
            checkpoint_dir=checkpoint_dir,
        )
        return trainer, manager

    torch.manual_seed(321)
    uninterrupted, uninterrupted_manager = build_stack(
        tmp_path / "uninterrupted-partial",
        overflow_calls={3},
    )
    microbatches = [
        (
            torch.linspace(-1.0, 1.0, steps=32).reshape(4, 8),
            torch.full((4, 8), float(index) / 10.0),
        )
        for index in range(3)
    ]

    history = uninterrupted.fit(
        microbatches,
        num_epochs=1,
        show_progress=False,
        track_best=False,
    )
    first_epoch = history[0]

    assert first_epoch["optimizer_steps"] == 1.0
    assert first_epoch["skipped_optimizer_steps"] == 1.0
    assert uninterrupted.global_step == 1
    checkpoint = tmp_path / "uninterrupted-partial" / "latest.pt"
    assert checkpoint.is_file()
    continuation = microbatches[:2]
    uninterrupted.train_epoch(
        continuation,
        epoch_index=2,
        show_progress=False,
    )
    expected = uninterrupted_manager.build_state()

    resumed, resumed_manager = build_stack(
        tmp_path / "resumed-partial",
        overflow_calls=set(),
    )
    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")
    loaded = resumed_manager.restore_payload(payload, path=checkpoint)
    assert loaded.global_step == 1
    resumed.global_step = loaded.global_step
    assert resumed.ema is not None
    resumed.ema.to("cuda")
    rng_state = payload.get("rng_state")
    assert rng_state is not None
    restore_rng_state(
        parse_rng_state(
            rng_state,
            require_cuda_compatibility=True,
        ),
        restore_cuda=True,
        restore_mps=False,
    )
    resumed.train_epoch(
        continuation,
        epoch_index=2,
        show_progress=False,
    )
    actual = resumed_manager.build_state()

    assert resumed.global_step == uninterrupted.global_step == 2
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "ema_state_dict",
        "grad_scaler_state_dict",
    ):
        torch.testing.assert_close(
            actual.get(key),
            expected.get(key),
            rtol=0.0,
            atol=0.0,
        )
    actual_rng_value = actual.get("rng_state")
    expected_rng_value = expected.get("rng_state")
    assert actual_rng_value is not None
    assert expected_rng_value is not None
    actual_rng = parse_rng_state(actual_rng_value)
    expected_rng = parse_rng_state(expected_rng_value)
    assert actual_rng.python == expected_rng.python
    assert actual_rng.numpy[0] == expected_rng.numpy[0]
    assert np.array_equal(actual_rng.numpy[1], expected_rng.numpy[1])
    assert actual_rng.numpy[2:] == expected_rng.numpy[2:]
    assert torch.equal(actual_rng.torch_cpu, expected_rng.torch_cpu)
    assert len(actual_rng.torch_cuda) == len(expected_rng.torch_cuda)
    assert all(
        torch.equal(actual_state, expected_state)
        for actual_state, expected_state in zip(
            actual_rng.torch_cuda,
            expected_rng.torch_cuda,
            strict=True,
        )
    )
