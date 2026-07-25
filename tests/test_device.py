"""Tests for framework-owned device placement."""

import pytest
import torch
from torch import nn

from stochaflow.utils.device import move_module_to_device


class _StatefulModule(nn.Module):
    def __init__(
        self,
        *,
        parameter_dtype: torch.dtype = torch.float32,
        buffer_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=parameter_dtype))
        self.register_buffer("coefficient", torch.ones(1, dtype=buffer_dtype))


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_mps_placement_rejects_unsupported_parameter_dtype(
    dtype: torch.dtype,
) -> None:
    module = _StatefulModule(parameter_dtype=dtype)

    with pytest.raises(
        TypeError,
        match=r"training module 'primary_model'.*parameter 'weight'.*CPU/CUDA",
    ):
        move_module_to_device(
            module,
            "mps",
            role="training module 'primary_model'",
        )

    assert module.weight.device.type == "cpu"


def test_mps_placement_rejects_unsupported_buffer_dtype() -> None:
    module = _StatefulModule(buffer_dtype=torch.float64)

    with pytest.raises(
        TypeError,
        match=r"sampling process.*buffer 'coefficient'.*float32",
    ):
        move_module_to_device(module, "mps", role="sampling process")

    assert module.coefficient.device.type == "cpu"


def test_cpu_placement_preserves_explicit_float64_state() -> None:
    module = _StatefulModule(
        parameter_dtype=torch.float64,
        buffer_dtype=torch.float64,
    )

    result = move_module_to_device(module, "cpu", role="test module")

    assert result is module
    assert module.weight.dtype == torch.float64
    assert module.coefficient.dtype == torch.float64
