"""Regression tests for fixed-DDP runtime-state admission."""

from __future__ import annotations

from collections import OrderedDict, UserDict
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from stochaflow.training.distributed.state_fingerprint import (
    require_clone_safe_runtime_state,
    require_no_distinct_shared_storage_across_modules,
    require_relocatable_module_states,
    require_runtime_state_disjoint_from_modules,
    runtime_state_fingerprint,
)


class TiedParameterModule(nn.Module):
    """Expose two registered names for one canonical Parameter object."""

    def __init__(self) -> None:
        super().__init__()
        weight = nn.Parameter(torch.tensor(1.0))
        self.left = weight
        self.right = weight


class RegisteredBufferModule(nn.Module):
    """Register one externally supplied Tensor for alias admission tests."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)


class ExtraStateModule(nn.Module):
    """Persist one caller-supplied value through the PyTorch extra-state hook."""

    def __init__(self, value: object) -> None:
        super().__init__()
        self.register_buffer("registered", torch.arange(3.0))
        self.value = value

    def get_extra_state(self) -> object:
        return self.value

    def set_extra_state(self, state: object) -> None:
        self.value = state


class OverlappingExtraStateModule(nn.Module):
    """Expose registered and extra Tensor views into overlapping storage."""

    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        super().__init__()
        self.register_buffer("registered", torch.from_numpy(array[:3]))
        self.extra = torch.from_numpy(array[1:])

    def get_extra_state(self) -> dict[str, torch.Tensor]:
        return {"overlap": self.extra}

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise TypeError("overlap state must be a dictionary")
        self.extra = cast(torch.Tensor, state["overlap"])


def test_runtime_fingerprint_preserves_mapping_type_and_iteration_order() -> None:
    forward = OrderedDict((("a", 1), ("b", 2)))
    reverse = OrderedDict((("b", 2), ("a", 1)))

    assert runtime_state_fingerprint(forward, path="forward") != (
        runtime_state_fingerprint(reverse, path="reverse")
    )
    assert runtime_state_fingerprint(dict(forward), path="dict") != (
        runtime_state_fingerprint(forward, path="ordered")
    )

    forward._metadata = {"version": 1}  # type: ignore[attr-defined]
    reverse_metadata = OrderedDict((("a", 1), ("b", 2)))
    reverse_metadata._metadata = {"version": 2}  # type: ignore[attr-defined]
    assert runtime_state_fingerprint(forward, path="metadata-one") != (
        runtime_state_fingerprint(reverse_metadata, path="metadata-two")
    )
    require_clone_safe_runtime_state(forward, path="ordered runtime state")


def test_clone_safe_state_rejects_semantic_tensor_changes() -> None:
    with pytest.raises(ValueError, match="must not require gradients"):
        require_clone_safe_runtime_state(
            torch.tensor(1.0, requires_grad=True),
            path="optimizer state",
        )

    with pytest.raises(ValueError, match="storage topology"):
        require_clone_safe_runtime_state(
            torch.arange(10.0)[:5],
            path="optimizer state",
        )

    channels_last = torch.zeros(2, 3, 4, 5).to(memory_format=torch.channels_last)
    require_clone_safe_runtime_state(channels_last, path="optimizer state")

    with pytest.raises(ValueError, match="lazy conjugate"):
        require_clone_safe_runtime_state(
            torch.tensor([1.0 + 2.0j]).conj(),
            path="optimizer state",
        )

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="pinned host memory"):
            require_clone_safe_runtime_state(
                torch.ones(1, pin_memory=True),
                path="optimizer state",
            )

    shared = torch.ones(1).share_memory_()
    with pytest.raises(ValueError, match="shared host memory"):
        require_clone_safe_runtime_state(shared, path="optimizer state")

    with torch.inference_mode():
        inference = torch.ones(1)
    with pytest.raises(ValueError, match="inference Tensor"):
        require_clone_safe_runtime_state(inference, path="optimizer state")

    quantized = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0]),
        scale=0.1,
        zero_point=3,
        dtype=torch.quint8,
    )
    with pytest.raises(ValueError, match="quantized state"):
        require_clone_safe_runtime_state(quantized, path="optimizer state")


def test_managed_roots_reject_cross_checkpoint_aliases() -> None:
    shared = torch.arange(3.0)

    with pytest.raises(ValueError, match="shared across managed roots"):
        require_no_distinct_shared_storage_across_modules(
            {
                "primary": RegisteredBufferModule(shared),
                "process": RegisteredBufferModule(shared),
            },
            path="managed roots",
        )

    array = np.arange(4.0, dtype=np.float32)
    with pytest.raises(ValueError, match="overlapping storage"):
        require_no_distinct_shared_storage_across_modules(
            {
                "primary": RegisteredBufferModule(torch.from_numpy(array[:3])),
                "process": RegisteredBufferModule(torch.from_numpy(array[1:])),
            },
            path="managed roots",
        )

    empty = torch.empty(0)
    with pytest.raises(ValueError, match="shared storage"):
        require_no_distinct_shared_storage_across_modules(
            {
                "primary": RegisteredBufferModule(empty),
                "process": RegisteredBufferModule(empty.view(0)),
            },
            path="managed roots",
        )


def test_runtime_state_rejects_managed_parameter_storage_alias() -> None:
    module = TiedParameterModule()
    parameter = next(module.parameters())

    with pytest.raises(ValueError, match="managed module state"):
        require_runtime_state_disjoint_from_modules(
            {"optimizer": {"alias": parameter.detach()}},
            {"primary": module},
            path="runtime state",
        )

    state_with_metadata: OrderedDict[str, object] = OrderedDict()
    state_with_metadata._metadata = {  # type: ignore[attr-defined]
        "alias": parameter.detach()
    }
    with pytest.raises(ValueError, match="managed module state"):
        require_runtime_state_disjoint_from_modules(
            state_with_metadata,
            {"primary": module},
            path="runtime state",
        )

    shared_extra = torch.arange(3.0)
    process = ExtraStateModule({"authority": shared_extra})
    with pytest.raises(ValueError, match="managed module state"):
        require_runtime_state_disjoint_from_modules(
            {"optimizer": {"alias": shared_extra.detach()}},
            {"process": process},
            path="runtime state",
        )

    shared_container = {"counter": 1}
    process = ExtraStateModule({"authority": shared_container})
    with pytest.raises(ValueError, match="managed module state"):
        require_runtime_state_disjoint_from_modules(
            {"optimizer": {"alias": shared_container}},
            {"process": process},
            path="runtime state",
        )


def test_module_checkpoint_admission_keeps_tied_parameter_objects() -> None:
    require_relocatable_module_states(
        {"primary": TiedParameterModule()},
        path="managed roots",
    )


def test_module_checkpoint_admission_rejects_hidden_registered_storage() -> None:
    backing = torch.tensor([7.0, 1.0, 2.0, 3.0, 17.0])

    with pytest.raises(ValueError, match="storage topology"):
        require_relocatable_module_states(
            {"process": RegisteredBufferModule(backing[1:4])},
            path="managed roots",
        )


def test_module_checkpoint_admission_rejects_grad_buffer() -> None:
    with pytest.raises(ValueError, match="must not require gradients"):
        require_relocatable_module_states(
            {"process": RegisteredBufferModule(torch.ones(1, requires_grad=True))},
            path="managed roots",
        )


def test_module_checkpoint_admission_rejects_unsupported_extra_state() -> None:
    with pytest.raises(TypeError, match="unsupported checkpoint value type"):
        require_relocatable_module_states(
            {"process": ExtraStateModule(UserDict({"version": 1}))},
            path="managed roots",
        )


def test_module_checkpoint_admission_rejects_extra_registered_storage_alias() -> None:
    module = ExtraStateModule(None)
    module.value = {"alias": cast(torch.Tensor, module.registered).detach()}

    with pytest.raises(ValueError, match="shares storage with registered"):
        require_relocatable_module_states(
            {"process": module},
            path="managed roots",
        )

    array = np.arange(4.0, dtype=np.float32)
    module = OverlappingExtraStateModule(array)
    with pytest.raises(ValueError, match="overlaps registered"):
        require_relocatable_module_states(
            {"process": module},
            path="managed roots",
        )


def test_module_checkpoint_admission_rejects_registered_tensor_as_extra_state() -> None:
    module = ExtraStateModule(None)
    module.value = module.registered

    with pytest.raises(ValueError, match="extra state"):
        require_relocatable_module_states(
            {"process": module},
            path="managed roots",
        )
