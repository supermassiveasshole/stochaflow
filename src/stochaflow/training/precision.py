"""Mixed-precision policy for the automatic single-optimizer loop."""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from typing import Literal, cast

import torch
from torch.optim import Optimizer

type PrecisionKind = Literal["fp32", "bf16-mixed", "fp16-mixed"]

PRECISION_KINDS: tuple[PrecisionKind, ...] = (
    "fp32",
    "bf16-mixed",
    "fp16-mixed",
)


def _usable_grad_scale(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite positive number")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"{path} must be a finite positive number")
    return scale


def _cuda_grad_scaler() -> torch.cuda.amp.GradScaler:
    # The unified torch.amp.GradScaler constructor is unavailable in the
    # supported PyTorch 2.2 line. Keep the compatible constructor without
    # surfacing its newer-version deprecation warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.cuda\.amp\.GradScaler",
            category=FutureWarning,
        )
        return torch.cuda.amp.GradScaler()


class PrecisionRuntime:
    """Own autocast and optional gradient-scaling policy for one device."""

    def __init__(
        self,
        *,
        kind: PrecisionKind,
        device_type: str,
        autocast_dtype: torch.dtype | None,
        grad_scaler: torch.cuda.amp.GradScaler | None,
    ) -> None:
        if kind not in PRECISION_KINDS:
            raise ValueError(f"unsupported precision kind: {kind!r}")
        device_type_value = cast(object, device_type)
        if not isinstance(device_type_value, str) or not device_type_value:
            raise ValueError("precision device_type must be non-empty")
        if device_type_value not in {"cpu", "cuda", "mps"}:
            raise ValueError(
                "automatic precision supports only cpu, cuda, and mps devices"
            )
        scaler_value = cast(object, grad_scaler)
        if scaler_value is not None and not isinstance(
            scaler_value,
            torch.cuda.amp.GradScaler,
        ):
            raise TypeError(
                "precision grad_scaler must be torch.cuda.amp.GradScaler or None"
            )
        if kind == "fp32":
            if autocast_dtype is not None or grad_scaler is not None:
                raise ValueError("fp32 precision cannot use autocast or a GradScaler")
        elif kind == "bf16-mixed":
            if autocast_dtype != torch.bfloat16:
                raise ValueError("bf16-mixed precision requires BF16 autocast")
            if device_type_value == "mps":
                raise ValueError("bf16-mixed precision is not supported on MPS")
        else:
            if autocast_dtype != torch.float16:
                raise ValueError("fp16-mixed precision requires FP16 autocast")
            if device_type_value != "cuda":
                raise ValueError("fp16-mixed precision is supported only on CUDA")
        if kind != "fp16-mixed" and grad_scaler is not None:
            raise ValueError("only fp16-mixed precision can use a GradScaler")
        if kind == "fp16-mixed" and grad_scaler is None:
            raise ValueError("fp16-mixed precision requires a GradScaler")
        if grad_scaler is not None and not grad_scaler.is_enabled():
            raise ValueError("fp16-mixed precision requires an enabled GradScaler")
        if grad_scaler is not None:
            _usable_grad_scale(
                grad_scaler.get_scale(),
                path="GradScaler scale at precision runtime construction",
            )
        self.kind = kind
        self.device_type = device_type
        self.autocast_dtype = autocast_dtype
        self.grad_scaler = grad_scaler

    def autocast(self):
        """Return the forward/evaluation precision context."""

        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
        )

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate a scalar loss with the configured scale."""

        loss_value = cast(object, loss)
        if not isinstance(loss_value, torch.Tensor):
            raise TypeError("precision backward loss must be a Tensor")
        loss = loss_value
        if loss.numel() != 1:
            raise ValueError("precision backward loss must be scalar")
        if self.grad_scaler is None:
            loss.backward()
            return
        self.grad_scaler.scale(loss).backward()

    def unscale_(self, optimizer: Optimizer) -> None:
        """Unscale gradients before inspection or clipping when necessary."""

        optimizer_value = cast(object, optimizer)
        if not isinstance(optimizer_value, Optimizer):
            raise TypeError("precision optimizer must be a torch Optimizer")
        optimizer = optimizer_value
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer) -> bool:
        """Attempt one optimizer update and report whether it succeeded."""

        optimizer_value = cast(object, optimizer)
        if not isinstance(optimizer_value, Optimizer):
            raise TypeError("precision optimizer must be a torch Optimizer")
        optimizer = optimizer_value
        if self.grad_scaler is None:
            optimizer.step()
            return True
        scale_before = _usable_grad_scale(
            self.grad_scaler.get_scale(),
            path="GradScaler scale before optimizer step",
        )
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        scale_after = _usable_grad_scale(
            self.grad_scaler.get_scale(),
            path="GradScaler scale after optimizer step",
        )
        return scale_after >= scale_before


def _resolve_precision_policy(
    kind: str,
    device: torch.device | str,
) -> tuple[PrecisionKind, str, torch.dtype | None]:
    """Resolve one supported precision policy without constructing runtime state."""

    kind_value = cast(object, kind)
    if not isinstance(kind_value, str) or kind_value not in PRECISION_KINDS:
        choices = ", ".join(PRECISION_KINDS)
        raise ValueError(f"precision must be one of: {choices}")
    precision_kind = cast(PrecisionKind, kind_value)
    resolved_device = torch.device(device)
    device_type = resolved_device.type
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "automatic precision supports only cpu, cuda, and mps devices"
        )
    if precision_kind == "fp32":
        return precision_kind, device_type, None
    if precision_kind == "bf16-mixed":
        if device_type == "mps":
            raise ValueError("bf16-mixed precision is not supported on MPS")
        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("bf16-mixed precision requires an available CUDA device")
            if not torch.cuda.is_bf16_supported():
                raise ValueError(
                    "bf16-mixed precision requires CUDA BF16 support"
                )
        return precision_kind, device_type, torch.bfloat16
    if device_type != "cuda":
        raise ValueError("fp16-mixed precision is supported only on CUDA")
    if not torch.cuda.is_available():
        raise ValueError("fp16-mixed precision requires an available CUDA device")
    return precision_kind, device_type, torch.float16


def validate_precision_support(
    kind: str,
    device: torch.device | str,
) -> None:
    """Validate one precision/device combination without creating runtime state."""

    _resolve_precision_policy(kind, device)


def build_precision_runtime(
    kind: str,
    device: torch.device | str,
) -> PrecisionRuntime:
    """Validate a precision/device combination and construct its runtime."""

    precision_kind, device_type, autocast_dtype = _resolve_precision_policy(
        kind,
        device,
    )
    return PrecisionRuntime(
        kind=precision_kind,
        device_type=device_type,
        autocast_dtype=autocast_dtype,
        grad_scaler=(
            _cuda_grad_scaler()
            if precision_kind == "fp16-mixed"
            else None
        ),
    )


__all__ = [
    "PRECISION_KINDS",
    "PrecisionKind",
    "PrecisionRuntime",
    "build_precision_runtime",
    "validate_precision_support",
]
