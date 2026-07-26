"""Tests for automatic-loop precision policy."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from stochaflow.training.builder import TrainingPlan
from stochaflow.training.precision import (
    PrecisionKind,
    PrecisionRuntime,
    build_precision_runtime,
)
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput
from stochaflow.training.trainer import Trainer


class EvaluationDtypeStrategy(TrainingStrategy):
    """Record model output dtype during evaluation."""

    def __init__(self, model: nn.Linear) -> None:
        self.model = model
        self.dtypes: list[torch.dtype] = []

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        return self.evaluation_step(batch)

    def evaluation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepOutput:
        inputs, targets = batch
        prediction = self.model(inputs)
        self.dtypes.append(prediction.dtype)
        return TrainStepOutput((prediction.float() - targets).square().mean())


def test_fp32_runtime_runs_plain_backward_and_step() -> None:
    parameter = nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.25)
    runtime = build_precision_runtime("fp32", "cpu")

    with runtime.autocast():
        loss = parameter.square()
    runtime.backward(loss)
    runtime.unscale_(optimizer)
    succeeded = runtime.step(optimizer)

    assert succeeded
    assert parameter.item() == pytest.approx(1.0)
    assert runtime.grad_scaler is None
    assert runtime.autocast_dtype is None


def test_cpu_bf16_runtime_autocasts_supported_operations() -> None:
    runtime = build_precision_runtime("bf16-mixed", "cpu")
    left = torch.randn(4, 4)
    right = torch.randn(4, 4)

    with runtime.autocast():
        result = left @ right

    assert result.dtype == torch.bfloat16
    assert runtime.grad_scaler is None


def test_evaluation_uses_autocast_without_scaler_or_optimizer_state() -> None:
    model = nn.Linear(4, 4)
    strategy = EvaluationDtypeStrategy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = build_precision_runtime("bf16-mixed", "cpu")
    trainer = Trainer(
        TrainingPlan(strategy=strategy, primary_model=model),
        optimizer,
        device="cpu",
        precision=runtime,
    )

    metrics = trainer.evaluate_epoch(
        [(torch.randn(2, 4), torch.randn(2, 4))],
        show_progress=False,
    )

    assert strategy.dtypes == [torch.bfloat16]
    assert metrics["num_batches"] == 1
    assert runtime.grad_scaler is None
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("kind", "device", "message"),
    [
        ("fp16-mixed", "cpu", "only on CUDA"),
        ("fp16-mixed", "mps", "only on CUDA"),
        ("bf16-mixed", "mps", "not supported on MPS"),
        ("bf16-mixed", "meta", "only cpu, cuda, and mps"),
        ("unknown", "cpu", "precision must be one of"),
    ],
)
def test_precision_runtime_rejects_unsupported_combinations(
    kind: str,
    device: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_precision_runtime(kind, device)


def test_precision_runtime_enforces_internal_topology() -> None:
    with pytest.raises(ValueError, match="cannot use autocast"):
        PrecisionRuntime(
            kind="fp32",
            device_type="cpu",
            autocast_dtype=torch.bfloat16,
            grad_scaler=None,
        )


@pytest.mark.parametrize("invalid_scale", [0.0, float("nan"), float("inf")])
def test_precision_runtime_rejects_unusable_grad_scaler_at_construction(
    invalid_scale: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    monkeypatch.setattr(scaler, "is_enabled", lambda: True)
    monkeypatch.setattr(scaler, "get_scale", lambda: invalid_scale)

    with pytest.raises(
        RuntimeError,
        match="GradScaler scale at precision runtime construction "
        "must be a finite positive",
    ):
        PrecisionRuntime(
            kind="fp16-mixed",
            device_type="cuda",
            autocast_dtype=torch.float16,
            grad_scaler=scaler,
        )


@pytest.mark.parametrize(
    ("kind", "device_type", "autocast_dtype", "message"),
    [
        ("fp32", "xpu", None, "only cpu, cuda, and mps"),
        ("bf16-mixed", "cpu", torch.float16, "requires BF16 autocast"),
        ("bf16-mixed", "mps", torch.bfloat16, "not supported on MPS"),
        ("fp16-mixed", "cpu", torch.float16, "only on CUDA"),
    ],
)
def test_precision_runtime_constructor_cannot_bypass_device_policy(
    kind: PrecisionKind,
    device_type: str,
    autocast_dtype: torch.dtype | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PrecisionRuntime(
            kind=kind,
            device_type=device_type,
            autocast_dtype=autocast_dtype,
            grad_scaler=None,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fp16_runtime_uses_grad_scaler() -> None:
    runtime = build_precision_runtime("fp16-mixed", "cuda")

    assert runtime.autocast_dtype == torch.float16
    assert runtime.grad_scaler is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_cuda_bf16_runtime_does_not_use_grad_scaler() -> None:
    runtime = build_precision_runtime("bf16-mixed", "cuda")

    assert runtime.autocast_dtype == torch.bfloat16
    assert runtime.grad_scaler is None
