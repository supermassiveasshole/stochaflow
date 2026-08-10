"""Execution-device validation and managed-module placement helpers."""

from __future__ import annotations

import torch
from torch import nn

_MPS_UNSUPPORTED_DTYPES = frozenset({torch.float64, torch.complex128})


def resolve_device(device_name: str) -> torch.device:
    """Resolve a configured device name into one concrete Torch device."""

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def move_module_to_device[ModuleT: nn.Module](
    module: ModuleT,
    device: torch.device | str,
    *,
    role: str,
) -> ModuleT:
    """Move a managed module after validating target-device dtype support."""

    target = torch.device(device)
    if target.type == "mps":
        incompatible: list[str] = []
        for name, parameter in module.named_parameters():
            if parameter.dtype in _MPS_UNSUPPORTED_DTYPES:
                incompatible.append(f"parameter '{name}' ({parameter.dtype})")
        for name, buffer in module.named_buffers():
            if buffer.dtype in _MPS_UNSUPPORTED_DTYPES:
                incompatible.append(f"buffer '{name}' ({buffer.dtype})")
        if incompatible:
            details = ", ".join(incompatible)
            raise TypeError(
                f"{role} cannot be moved to MPS because it contains "
                f"unsupported state: {details}. Use float32/complex64 state "
                "for MPS, or select a CPU/CUDA device."
            )
    module.to(target)
    return module


def validate_execution_device(device: torch.device | str) -> None:
    """Fail early when an explicitly resolved execution device is unusable."""

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA execution requires an available CUDA device")
        device_count = torch.cuda.device_count()
        if (
            resolved.index is not None
            and not 0 <= resolved.index < device_count
        ):
            raise ValueError(
                f"CUDA device index {resolved.index} is outside the available "
                f"range [0, {device_count})"
            )
        return
    if resolved.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS execution requires an available MPS device")
        if resolved.index not in {None, 0}:
            raise ValueError("MPS execution supports only device index 0")


__all__ = ["move_module_to_device", "resolve_device", "validate_execution_device"]
