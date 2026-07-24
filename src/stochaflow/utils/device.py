"""Device-placement helpers for framework-managed modules."""

from __future__ import annotations

from typing import TypeVar

import torch
import torch.nn as nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)

_MPS_UNSUPPORTED_DTYPES = frozenset({torch.float64, torch.complex128})


def move_module_to_device(
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
