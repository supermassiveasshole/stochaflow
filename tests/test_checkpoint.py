"""Tests for the versioned data-only checkpoint contract."""

from __future__ import annotations

import math
import pickle
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, StepLR

from stochaflow.sampling import SamplingRecipe
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    InferenceAssetDescriptor,
    capture_rng_state,
    parse_rng_state,
    restore_rng_state,
    validate_inference_asset_descriptors,
)
from stochaflow.utils.plugins import ExtensionIdentityError


def _write_load_marker(path: str) -> object:
    Path(path).write_text("loaded", encoding="utf-8")
    return object()


class FixtureExecutablePickleValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_load_marker, (str(self.marker),)


class FixtureUnsafeExtraState:
    pass


class FixtureUnsafeExtraStateModule(nn.Module):
    def get_extra_state(self) -> object:
        return FixtureUnsafeExtraState()

    def set_extra_state(self, state: object) -> None:
        del state


class FixtureTensorSubclass(torch.Tensor):
    pass


class FixtureFailingLoadModule(nn.Module):
    """Raise once after PyTorch has applied an otherwise valid state dict."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.failures_remaining = 0
        self.register_load_state_dict_post_hook(self._raise_once_after_load)

    def _raise_once_after_load(
        self,
        module: nn.Module,
        incompatible_keys: object,
    ) -> None:
        del module, incompatible_keys
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("planned post-load failure")


class FixtureFailingScheduler(LRScheduler):
    """Mutate then fail once so cross-asset rollback can be verified."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.count = -1
        self.failures_remaining = 0
        super().__init__(optimizer)

    def step(self) -> None:
        self.count += 1

    def state_dict(self) -> dict[str, int]:
        return {"count": self.count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.count = state["count"]
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("planned scheduler load failure")


class FixtureTensorStateScheduler(LRScheduler):
    """Expose tensor state so checkpoint aliasing can be tested directly."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.tensor_state = torch.tensor(3.0)
        super().__init__(optimizer)

    def step(self) -> None:
        pass


class FixtureFailingTensorStateScheduler(LRScheduler):
    """Assign tensor state then fail once to exercise device-safe rollback."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.tensor_state = torch.tensor(3.0, device=optimizer.param_groups[0]["params"][0].device)
        self.failures_remaining = 0
        super().__init__(optimizer)

    def step(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"tensor_state": self.tensor_state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.tensor_state = cast(torch.Tensor, state["tensor_state"])
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("planned tensor scheduler load failure")


class FixtureExtraStateOptimizer(torch.optim.Optimizer):
    """Optimizer whose public state mapping carries one extension-owned field."""

    def __init__(self, params, *, lr: float = 0.1) -> None:
        self.extra_counter = 0
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        self.extra_counter += 1
        return closure() if closure is not None else None

    def state_dict(self):
        state = OrderedDict(super().state_dict())
        state["extra_counter"] = self.extra_counter
        return state

    def load_state_dict(self, state_dict):
        state = OrderedDict(state_dict)
        self.extra_counter = int(state.pop("extra_counter"))
        super().load_state_dict(state)


class FixtureTiedStateModule(nn.Module):
    """Model with tied parameter and floating-buffer state-dict aliases."""

    shared_buffer_a: torch.Tensor
    shared_buffer_b: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(1, 1, bias=False)
        self.second = nn.Linear(1, 1, bias=False)
        self.second.weight = self.first.weight
        shared_buffer = torch.zeros(1)
        self.register_buffer("shared_buffer_a", shared_buffer)
        self.register_buffer("shared_buffer_b", shared_buffer)


def _legacy_v8_payload(
    manager: CheckpointManager,
    *,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(manager.build_state(config=config, metadata=metadata))
    payload["format_version"] = 8
    payload.pop("precision_kind")
    payload.pop("inference_asset_descriptors")
    payload.pop("grad_scaler_class", None)
    payload.pop("grad_scaler_state_dict", None)
    return payload


def _cuda_grad_scaler(*, initial_scale: float = 65_536.0) -> torch.cuda.amp.GradScaler:
    # PyTorch <= 2.2 imports this probe into grad_scaler; newer releases look it
    # up through common. Patch both call sites so CPU-only CI exercises enabled
    # scaler checkpoint semantics instead of PyTorch's CUDA availability policy.
    with (
        patch(
            "torch.cuda.amp.common.amp_definitely_not_available",
            return_value=False,
        ),
        patch(
            "torch.cuda.amp.grad_scaler.amp_definitely_not_available",
            return_value=False,
            create=True,
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", FutureWarning)
        return torch.cuda.amp.GradScaler(init_scale=initial_scale)


def _inference_asset_descriptors() -> dict[str, InferenceAssetDescriptor]:
    return {
        "codec": {
            "training_asset_name": "codec",
            "declaration": {
                "name": "test_codec",
                "params": {"channels": 3},
            },
            "capability_role": "image_codec",
            "persistence": "embedded_state",
        }
    }


def test_v10_checkpoint_always_records_header_recipe_and_plugin_metadata() -> None:
    manager = CheckpointManager(nn.Linear(1, 1))

    default_state = manager.build_state()
    populated_state = manager.build_state(
        metadata={
            "extension_plugins": [
                {
                    "name": "example",
                    "distribution": "example-project",
                    "version": "1.2.3",
                    "target": "example_project.stochaflow_ext",
                }
            ],
            "checkpoint_kind": "latest",
        }
    )

    assert CHECKPOINT_FORMAT_VERSION == 10
    assert default_state.get("precision_kind") == "fp32"
    assert default_state.get("inference_asset_descriptors") == {}
    assert "inference_recipe" in default_state
    assert default_state.get("inference_recipe") is None
    assert "grad_scaler_class" not in default_state
    assert "grad_scaler_state_dict" not in default_state
    default_metadata = default_state.get("metadata")
    populated_metadata = populated_state.get("metadata")
    assert default_metadata == {"extension_plugins": []}
    assert populated_metadata is not None
    assert populated_metadata["extension_plugins"] == [
        {
            "name": "example",
            "distribution": "example-project",
            "version": "1.2.3",
            "target": "example_project.stochaflow_ext",
        }
    ]
    rng_state = default_state.get("rng_state")
    assert rng_state is not None
    assert "torch_mps" in rng_state
    if torch.backends.mps.is_available():
        assert isinstance(rng_state["torch_mps"], torch.Tensor)
    else:
        assert rng_state["torch_mps"] is None


def test_v10_checkpoint_round_trips_inference_recipe(tmp_path: Path) -> None:
    recipe = SamplingRecipe(
        name="project.generate",
        contract={
            "prediction_type": "v",
            "schedule": {"steps": [4, 2, 0]},
        },
    )
    manager = CheckpointManager(nn.Linear(1, 1), inference_recipe=recipe)
    checkpoint = manager.save(tmp_path / "recipe.pt")

    payload = CheckpointManager.load_payload(checkpoint)

    assert payload.get("inference_recipe") == {
        "schema_version": 1,
        "name": "project.generate",
        "contract": {
            "prediction_type": "v",
            "schedule": {"steps": [4, 2, 0]},
        },
    }
    CheckpointManager(
        nn.Linear(1, 1),
        inference_recipe=recipe,
    ).restore_payload(payload, path=checkpoint)


def test_v10_checkpoint_rejects_non_json_recipe_contract_tuple(
    tmp_path: Path,
) -> None:
    payload = CheckpointManager(nn.Linear(1, 1)).build_state()
    payload["inference_recipe"] = {
        "schema_version": 1,
        "name": "project.generate",
        "contract": {"steps": (4, 2, 0)},
    }
    checkpoint = tmp_path / "tuple-contract.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(TypeError, match=r"unsupported value type.*tuple"):
        CheckpointManager.load_payload(checkpoint)


def test_build_state_detaches_caller_owned_payload_metadata() -> None:
    config = {"trainer": {"precision": "fp32"}}
    metrics = {"train/loss": 1.0}
    metadata = {
        "extension_plugins": [],
        "nested": {"values": [torch.ones(1)]},
    }

    state = CheckpointManager(nn.Linear(1, 1)).build_state(
        config=config,
        metrics=metrics,
        metadata=metadata,
    )
    config["trainer"]["precision"] = "bf16-mixed"
    metrics["train/loss"] = 9.0
    nested_values = cast(list[torch.Tensor], metadata["nested"]["values"])
    nested_values[0].fill_(9.0)

    assert state.get("config") == {"trainer": {"precision": "fp32"}}
    assert state.get("metrics") == {"train/loss": 1.0}
    state_metadata = state.get("metadata")
    assert state_metadata is not None
    state_nested = cast(dict[str, list[torch.Tensor]], state_metadata["nested"])
    assert torch.equal(state_nested["values"][0], torch.ones(1))


def test_checkpoint_boundary_strictly_validates_plugin_provenance() -> None:
    manager = CheckpointManager(nn.Linear(1, 1))

    with pytest.raises(ExtensionIdentityError, match="invalid fields"):
        manager.build_state(
            metadata={
                "extension_plugins": [
                    {
                        "name": "example",
                        "distribution": "example",
                        "version": "1.0",
                        "typo": "example.stochaflow_ext",
                    }
                ]
            }
        )


@pytest.mark.parametrize("precision_kind", ["fp32", "bf16-mixed"])
def test_non_fp16_checkpoints_forbid_scaler_state(
    precision_kind: str,
) -> None:
    manager = CheckpointManager(
        nn.Linear(1, 1),
        precision_kind=precision_kind,
    )

    state = manager.build_state()

    assert state.get("precision_kind") == precision_kind
    assert "grad_scaler_class" not in state
    assert "grad_scaler_state_dict" not in state

    state["grad_scaler_class"] = "torch.cuda.amp.grad_scaler.GradScaler"
    state["grad_scaler_state_dict"] = {}
    with pytest.raises(ValueError, match="cannot contain GradScaler field"):
        manager.restore_payload(state, path="unexpected-scaler.pt")


def test_checkpoint_manager_rejects_invalid_runtime_scaler_topology() -> None:
    scaler = _cuda_grad_scaler()

    with pytest.raises(ValueError, match=r"fp16-mixed.*requires a GradScaler"):
        CheckpointManager(nn.Linear(1, 1), precision_kind="fp16-mixed")
    with pytest.raises(ValueError, match=r"fp32.*cannot use a GradScaler"):
        CheckpointManager(
            nn.Linear(1, 1),
            precision_kind="fp32",
            grad_scaler=scaler,
        )
    with pytest.raises(ValueError, match="precision_kind must be one of"):
        CheckpointManager(nn.Linear(1, 1), precision_kind="unknown")


@pytest.mark.parametrize("invalid_scale", [0.0, float("nan"), float("inf")])
def test_unusable_grad_scaler_scale_is_rejected_at_checkpoint_boundaries(
    invalid_scale: float,
    tmp_path: Path,
) -> None:
    invalid_runtime_scaler = _cuda_grad_scaler()
    invalid_runtime_state = invalid_runtime_scaler.state_dict()
    invalid_runtime_state["scale"] = invalid_scale
    invalid_runtime_scaler.load_state_dict(invalid_runtime_state)

    with pytest.raises(
        ValueError,
        match="checkpoint manager GradScaler scale must be a finite positive",
    ):
        CheckpointManager(
            nn.Linear(1, 1),
            precision_kind="fp16-mixed",
            grad_scaler=invalid_runtime_scaler,
        )

    mutable_scaler = _cuda_grad_scaler()
    mutable_manager = CheckpointManager(
        nn.Linear(1, 1),
        precision_kind="fp16-mixed",
        grad_scaler=mutable_scaler,
    )
    mutable_model_state = {
        name: value.detach().clone()
        for name, value in mutable_manager.model.state_dict().items()
    }
    mutable_state = mutable_scaler.state_dict()
    mutable_state["scale"] = invalid_scale
    mutable_scaler.load_state_dict(mutable_state)
    with pytest.raises(
        ValueError,
        match=r"grad_scaler_state_dict\.scale must be a finite positive",
    ):
        mutable_manager.build_state()

    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(7.0)
    source_manager = CheckpointManager(
        source_model,
        precision_kind="fp16-mixed",
        grad_scaler=_cuda_grad_scaler(initial_scale=32.0),
    )
    valid_payload = source_manager.build_state()
    with pytest.raises(
        ValueError,
        match="runtime GradScaler scale must be a finite positive",
    ):
        mutable_manager.restore_payload(
            valid_payload,
            path="invalid-runtime-scale.pt",
        )
    assert all(
        torch.equal(value, mutable_model_state[name])
        for name, value in mutable_manager.model.state_dict().items()
    )
    assert mutable_scaler.get_scale() == invalid_scale or (
        math.isnan(mutable_scaler.get_scale()) and math.isnan(invalid_scale)
    )

    payload = source_manager.build_state()
    payload_scaler_state = payload.get("grad_scaler_state_dict")
    assert payload_scaler_state is not None
    payload_scaler_state["scale"] = invalid_scale

    raw_path = tmp_path / "invalid-scale-raw.pt"
    torch.save(payload, raw_path)
    with pytest.raises(
        ValueError,
        match=r"grad_scaler_state_dict\.scale must be a finite positive",
    ):
        CheckpointManager.load_payload(raw_path)

    with pytest.raises(
        ValueError,
        match=r"grad_scaler_state_dict\.scale must be a finite positive",
    ):
        CheckpointManager.save_payload(payload, tmp_path / "invalid-scale.pt")

    target_model = nn.Linear(1, 1)
    target_model.weight.data.fill_(-3.0)
    target_scaler = _cuda_grad_scaler(initial_scale=8.0)
    target_manager = CheckpointManager(
        target_model,
        precision_kind="fp16-mixed",
        grad_scaler=target_scaler,
    )
    original_model_state = {
        name: value.detach().clone()
        for name, value in target_model.state_dict().items()
    }
    original_scaler_state = target_scaler.state_dict()

    with pytest.raises(
        ValueError,
        match=r"grad_scaler_state_dict\.scale must be a finite positive",
    ):
        target_manager.restore_payload(payload, path="invalid-scale.pt")

    assert all(
        torch.equal(value, original_model_state[name])
        for name, value in target_model.state_dict().items()
    )
    assert target_scaler.state_dict() == original_scaler_state


def test_fp16_scaler_save_load_round_trip(tmp_path: Path) -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(4.0)
    source_scaler = _cuda_grad_scaler(initial_scale=256.0)
    checkpoint = CheckpointManager(
        source_model,
        precision_kind="fp16-mixed",
        grad_scaler=source_scaler,
    ).save(tmp_path / "fp16.pt")

    payload = CheckpointManager.load_payload(checkpoint)
    assert payload.get("precision_kind") == "fp16-mixed"
    assert payload.get("grad_scaler_class") == (
        "torch.cuda.amp.grad_scaler.GradScaler"
    )
    assert payload.get("grad_scaler_state_dict") == source_scaler.state_dict()

    target_model = nn.Linear(1, 1)
    target_model.weight.data.zero_()
    target_scaler = _cuda_grad_scaler(initial_scale=8.0)
    CheckpointManager(
        target_model,
        precision_kind="fp16-mixed",
        grad_scaler=target_scaler,
    ).load(checkpoint)

    assert torch.equal(target_model.weight, source_model.weight)
    assert target_scaler.state_dict() == source_scaler.state_dict()


def test_scaler_payload_is_detached_before_and_after_restore() -> None:
    source_scaler = _cuda_grad_scaler(initial_scale=256.0)
    source_manager = CheckpointManager(
        nn.Linear(1, 1),
        precision_kind="fp16-mixed",
        grad_scaler=source_scaler,
    )
    payload = source_manager.build_state()
    scaler_state = payload.get("grad_scaler_state_dict")
    assert scaler_state is not None
    scaler_state["scale"] = 8.0
    assert source_scaler.get_scale() == 256.0

    clean_payload = source_manager.build_state()
    target_scaler = _cuda_grad_scaler(initial_scale=4.0)
    CheckpointManager(
        nn.Linear(1, 1),
        precision_kind="fp16-mixed",
        grad_scaler=target_scaler,
    ).restore_payload(clean_payload, path="detached-scaler.pt")
    restored_scale = target_scaler.get_scale()
    restored_scaler_state = clean_payload.get("grad_scaler_state_dict")
    assert restored_scaler_state is not None
    restored_scaler_state["scale"] = 16.0

    assert target_scaler.get_scale() == restored_scale


def test_inference_asset_descriptors_default_and_round_trip(
    tmp_path: Path,
) -> None:
    descriptors = _inference_asset_descriptors()
    codec = nn.Linear(1, 1)
    manager = CheckpointManager(
        nn.Linear(1, 1),
        auxiliary_modules={"codec": codec},
        inference_asset_descriptors=descriptors,
    )
    descriptors["codec"]["declaration"]["params"]["channels"] = 4

    checkpoint = manager.save(tmp_path / "descriptors.pt")
    state = CheckpointManager.load_payload(checkpoint)

    assert state.get("inference_asset_descriptors") == (
        _inference_asset_descriptors()
    )
    CheckpointManager(
        nn.Linear(1, 1),
        auxiliary_modules={"codec": nn.Linear(1, 1)},
        inference_asset_descriptors=_inference_asset_descriptors(),
    ).load(checkpoint)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([], TypeError, "exact dictionary"),
        (
            {"": _inference_asset_descriptors()["codec"]},
            ValueError,
            "slot names must be non-empty",
        ),
        (
            {
                "codec": {
                    **_inference_asset_descriptors()["codec"],
                    "unknown": True,
                }
            },
            ValueError,
            "invalid fields",
        ),
        (
            {
                "codec": {
                    **_inference_asset_descriptors()["codec"],
                    "declaration": {"name": "test_codec", "params": []},
                }
            },
            TypeError,
            r"declaration\.params must be an exact dictionary",
        ),
        (
            {
                "codec": {
                    **_inference_asset_descriptors()["codec"],
                    "persistence": "immutable_reference",
                }
            },
            ValueError,
            "persistence must be 'embedded_state'",
        ),
    ],
)
def test_inference_asset_descriptor_validator_rejects_malformed_schema(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        validate_inference_asset_descriptors(value)


def test_inference_asset_descriptor_requires_managed_training_asset() -> None:
    with pytest.raises(ValueError, match="missing training assets: codec"):
        CheckpointManager(
            nn.Linear(1, 1),
            inference_asset_descriptors=_inference_asset_descriptors(),
        )


@pytest.mark.parametrize("runtime_descriptor_mode", ["empty", "different-role"])
def test_inference_asset_descriptor_topology_mismatch_is_atomic(
    runtime_descriptor_mode: str,
) -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(42.0)
    state = CheckpointManager(
        source_model,
        auxiliary_modules={"codec": nn.Linear(1, 1)},
        inference_asset_descriptors=_inference_asset_descriptors(),
    ).build_state()
    target_model = nn.Linear(1, 1)
    original_weight = target_model.weight.detach().clone()
    runtime_descriptors = (
        {}
        if runtime_descriptor_mode == "empty"
        else _inference_asset_descriptors()
    )
    if runtime_descriptors:
        runtime_descriptors["codec"]["capability_role"] = "decoder"
    manager = CheckpointManager(
        target_model,
        auxiliary_modules={"codec": nn.Linear(1, 1)},
        inference_asset_descriptors=runtime_descriptors,
    )

    with pytest.raises(
        ValueError,
        match="inference asset descriptors do not match runtime",
    ):
        manager.restore_payload(state, path="descriptor-mismatch.pt")

    assert torch.equal(target_model.weight, original_weight)


@pytest.mark.parametrize(
    "missing_field",
    ["precision_kind", "inference_asset_descriptors", "inference_recipe"],
)
def test_v10_header_rejects_missing_required_fields(
    missing_field: str,
) -> None:
    manager = CheckpointManager(nn.Linear(1, 1))
    state = manager.build_state()
    state.pop(missing_field)

    with pytest.raises((TypeError, ValueError), match=missing_field):
        manager.restore_payload(state, path="missing-v10-field.pt")


@pytest.mark.parametrize(
    "missing_field",
    ["grad_scaler_class", "grad_scaler_state_dict"],
)
def test_fp16_header_requires_complete_scaler_topology(
    missing_field: str,
) -> None:
    scaler = _cuda_grad_scaler()
    manager = CheckpointManager(
        nn.Linear(1, 1),
        precision_kind="fp16-mixed",
        grad_scaler=scaler,
    )
    state = manager.build_state()
    state.pop(missing_field)

    with pytest.raises(TypeError, match=missing_field):
        manager.restore_payload(state, path="missing-scaler-field.pt")


def test_precision_and_scaler_class_mismatch_fail_before_model_restore() -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(7.0)
    bf16_state = CheckpointManager(
        source_model,
        precision_kind="bf16-mixed",
    ).build_state()
    target_model = nn.Linear(1, 1)
    target_model.weight.data.fill_(-3.0)
    original_weight = target_model.weight.detach().clone()

    with pytest.raises(ValueError, match="precision kind does not match runtime"):
        CheckpointManager(target_model).restore_payload(
            bf16_state,
            path="precision-mismatch.pt",
        )
    assert torch.equal(target_model.weight, original_weight)

    scaler = _cuda_grad_scaler()
    fp16_state = CheckpointManager(
        source_model,
        precision_kind="fp16-mixed",
        grad_scaler=scaler,
    ).build_state()
    fp16_state["grad_scaler_class"] = "example.OtherScaler"
    with pytest.raises(ValueError, match="GradScaler class does not match runtime"):
        CheckpointManager(
            target_model,
            precision_kind="fp16-mixed",
            grad_scaler=_cuda_grad_scaler(),
        ).restore_payload(fp16_state, path="scaler-class-mismatch.pt")
    assert torch.equal(target_model.weight, original_weight)


def test_malformed_scaler_state_fails_atomically_before_managed_state() -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(9.0)
    source_state = CheckpointManager(
        source_model,
        precision_kind="fp16-mixed",
        grad_scaler=_cuda_grad_scaler(initial_scale=512.0),
    ).build_state()
    source_scaler_state = source_state.get("grad_scaler_state_dict")
    assert source_scaler_state is not None
    scaler_state = dict(source_scaler_state)
    scaler_state.pop("growth_factor")
    source_state["grad_scaler_state_dict"] = scaler_state

    target_model = nn.Linear(1, 1)
    target_model.weight.data.fill_(-5.0)
    target_scaler = _cuda_grad_scaler(initial_scale=16.0)
    original_weight = target_model.weight.detach().clone()
    original_scaler_state = target_scaler.state_dict()

    with pytest.raises(KeyError, match="growth_factor"):
        CheckpointManager(
            target_model,
            precision_kind="fp16-mixed",
            grad_scaler=target_scaler,
        ).restore_payload(source_state, path="malformed-scaler.pt")

    assert torch.equal(target_model.weight, original_weight)
    assert target_scaler.state_dict() == original_scaler_state


def test_malformed_module_state_fails_before_any_parameter_is_changed() -> None:
    source_model = nn.Linear(1, 1)
    state = CheckpointManager(source_model).build_state()
    model_state = state.get("model_state_dict")
    assert model_state is not None
    model_state["weight"] = torch.full_like(source_model.weight, 42.0)
    model_state.pop("bias")
    target_model = nn.Linear(1, 1)
    original = {
        name: tensor.detach().clone()
        for name, tensor in target_model.state_dict().items()
    }

    with pytest.raises(ValueError, match="keys do not match runtime"):
        CheckpointManager(target_model).restore_payload(
            state,
            path="malformed-module.pt",
        )

    for name, tensor in target_model.state_dict().items():
        assert torch.equal(tensor, original[name])


def test_meta_module_tensor_fails_before_load_version_changes() -> None:
    source_model = nn.Linear(1, 1)
    state = CheckpointManager(source_model).build_state()
    model_state = state.get("model_state_dict")
    assert model_state is not None
    model_state["weight"] = torch.empty_like(source_model.weight, device="meta")
    target_model = nn.Linear(1, 1)
    original_weight = target_model.weight.detach().clone()
    original_version = target_model.weight._version

    with pytest.raises(ValueError, match="cannot use the meta device"):
        CheckpointManager(target_model).restore_payload(
            state,
            path="meta-module-state.pt",
        )

    assert torch.equal(target_model.weight, original_weight)
    assert target_model.weight._version == original_version


def test_unforeseen_module_load_failure_rolls_back_every_parameter() -> None:
    source_model = FixtureFailingLoadModule()
    source_model.weight.data.fill_(42.0)
    state = CheckpointManager(source_model).build_state()
    target_model = FixtureFailingLoadModule()
    target_model.weight.data.fill_(-7.0)
    original_weight = target_model.weight.detach().clone()
    target_model.failures_remaining = 1

    with pytest.raises(RuntimeError, match="planned post-load failure"):
        CheckpointManager(target_model).restore_payload(
            state,
            path="transactional-module.pt",
        )

    assert torch.equal(target_model.weight, original_weight)


def test_late_scheduler_failure_rolls_back_model_and_optimizer_state() -> None:
    source_model = nn.Linear(1, 1)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    source_scheduler = FixtureFailingScheduler(source_optimizer)
    source_model(torch.ones(1, 1)).square().mean().backward()
    source_optimizer.step()
    source_scheduler.step()
    source_state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
    ).build_state()

    target_model = nn.Linear(1, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.02)
    target_scheduler = FixtureFailingScheduler(target_optimizer)
    target_model(torch.full((1, 1), 2.0)).square().mean().backward()
    target_optimizer.step()
    target_scheduler.step()
    manager = CheckpointManager(
        target_model,
        optimizer=target_optimizer,
        lr_scheduler=target_scheduler,
    )
    before = manager.build_state()
    target_scheduler.failures_remaining = 1

    with pytest.raises(RuntimeError, match="planned scheduler load failure"):
        manager.restore_payload(source_state, path="scheduler-failure.pt")

    after = manager.build_state()
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
    ):
        torch.testing.assert_close(
            after.get(key),
            before.get(key),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_failed_restore_preserves_optimizer_and_scheduler_tensor_devices() -> None:
    source_model = nn.Linear(1, 1).cuda()
    source_optimizer = torch.optim.AdamW(source_model.parameters())
    source_scheduler = FixtureFailingTensorStateScheduler(source_optimizer)
    source_model(torch.ones(1, 1, device="cuda")).square().mean().backward()
    source_optimizer.step()
    source_state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
    ).build_state()

    target_model = nn.Linear(1, 1).cuda()
    target_optimizer = torch.optim.AdamW(target_model.parameters())
    target_scheduler = FixtureFailingTensorStateScheduler(target_optimizer)
    target_model(torch.ones(1, 1, device="cuda")).square().mean().backward()
    target_optimizer.step()
    target_scheduler.failures_remaining = 1
    manager = CheckpointManager(
        target_model,
        optimizer=target_optimizer,
        lr_scheduler=target_scheduler,
    )
    optimizer_devices_before = [
        value.device
        for values in target_optimizer.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    ]

    with pytest.raises(RuntimeError, match="planned tensor scheduler load failure"):
        manager.restore_payload(source_state, path="cuda-device-rollback.pt")

    optimizer_tensors = [
        value
        for values in target_optimizer.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    ]
    assert optimizer_tensors
    assert [value.device for value in optimizer_tensors] == optimizer_devices_before
    assert target_scheduler.tensor_state.device.type == "cuda"


def test_optimizer_topology_fails_before_model_state_is_applied() -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(17.0)
    source_optimizer = torch.optim.AdamW(source_model.parameters())
    source_state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
    ).build_state()
    optimizer_state = source_state.get("optimizer_state_dict")
    assert optimizer_state is not None
    param_groups = optimizer_state["param_groups"]
    assert isinstance(param_groups, list)
    param_groups[0]["params"].append(999)

    target_model = nn.Linear(1, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters())
    original_weight = target_model.weight.detach().clone()

    with pytest.raises(ValueError, match="parameter count does not match"):
        CheckpointManager(
            target_model,
            optimizer=target_optimizer,
        ).restore_payload(source_state, path="optimizer-topology.pt")

    assert torch.equal(target_model.weight, original_weight)


def test_missing_optimizer_state_fails_before_module_load_version_changes() -> None:
    source_model = nn.Linear(1, 1)
    source_optimizer = torch.optim.AdamW(source_model.parameters())
    source_state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
    ).build_state()
    optimizer_state = source_state.get("optimizer_state_dict")
    assert optimizer_state is not None
    optimizer_state.pop("state")

    target_model = nn.Linear(1, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters())
    original_weight = target_model.weight.detach().clone()
    original_version = target_model.weight._version

    with pytest.raises(TypeError, match=r"optimizer_state_dict\.state"):
        CheckpointManager(
            target_model,
            optimizer=target_optimizer,
        ).restore_payload(source_state, path="missing-optimizer-state.pt")

    assert torch.equal(target_model.weight, original_weight)
    assert target_model.weight._version == original_version


def test_missing_optimizer_group_key_is_preflighted_and_runtime_stays_usable() -> None:
    source_model = nn.Linear(1, 1)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1)
    source_state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
    ).build_state()
    optimizer_state = source_state.get("optimizer_state_dict")
    assert optimizer_state is not None
    param_groups = optimizer_state["param_groups"]
    assert isinstance(param_groups, list)
    param_groups[0].pop("lr")

    target_model = nn.Linear(1, 1)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.2)
    original_weight = target_model.weight.detach().clone()
    original_version = target_model.weight._version

    with pytest.raises(ValueError, match=r"group 0.*lr"):
        CheckpointManager(
            target_model,
            optimizer=target_optimizer,
        ).restore_payload(source_state, path="missing-optimizer-group-key.pt")

    assert torch.equal(target_model.weight, original_weight)
    assert target_model.weight._version == original_version
    target_model(torch.ones(1, 1)).sum().backward()
    target_optimizer.step()
    assert not torch.equal(target_model.weight, original_weight)


def test_extension_optimizer_extra_state_round_trips_from_ordered_mapping() -> None:
    source_model = nn.Linear(1, 1)
    source_optimizer = FixtureExtraStateOptimizer(source_model.parameters())
    source_optimizer.extra_counter = 7
    state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
    ).build_state()
    optimizer_state = state.get("optimizer_state_dict")
    assert isinstance(optimizer_state, OrderedDict)
    assert optimizer_state["extra_counter"] == 7

    target_model = nn.Linear(1, 1)
    target_optimizer = FixtureExtraStateOptimizer(target_model.parameters())
    target_optimizer.extra_counter = 1
    CheckpointManager(
        target_model,
        optimizer=target_optimizer,
    ).restore_payload(state, path="extension-optimizer-state.pt")

    assert target_optimizer.extra_counter == 7
    target_optimizer.step()
    assert target_optimizer.extra_counter == 8


def test_missing_scheduler_key_is_preflighted_and_runtime_stays_usable() -> None:
    source_model = nn.Linear(1, 1)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1)
    source_scheduler = StepLR(source_optimizer, step_size=2)
    source_state = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
    ).build_state()
    scheduler_state = source_state.get("lr_scheduler_state_dict")
    assert scheduler_state is not None
    scheduler_state.pop("last_epoch")

    target_model = nn.Linear(1, 1)
    target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.2)
    target_scheduler = StepLR(target_optimizer, step_size=2)
    original_weight = target_model.weight.detach().clone()
    original_version = target_model.weight._version
    original_last_epoch = target_scheduler.last_epoch

    with pytest.raises(ValueError, match=r"missing runtime key.*last_epoch"):
        CheckpointManager(
            target_model,
            optimizer=target_optimizer,
            lr_scheduler=target_scheduler,
        ).restore_payload(source_state, path="missing-scheduler-key.pt")

    assert torch.equal(target_model.weight, original_weight)
    assert target_model.weight._version == original_version
    target_model(torch.ones(1, 1)).sum().backward()
    target_optimizer.step()
    target_scheduler.step()
    assert target_scheduler.last_epoch == original_last_epoch + 1


def test_build_and_restore_payloads_do_not_alias_live_runtime_state() -> None:
    source_model = nn.Linear(1, 1)
    source_optimizer = torch.optim.AdamW(source_model.parameters())
    source_scheduler = FixtureTensorStateScheduler(source_optimizer)
    source_ema = ExponentialMovingAverage(source_model)
    source_model(torch.ones(1, 1)).square().mean().backward()
    source_optimizer.step()
    manager = CheckpointManager(
        source_model,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
        ema=source_ema,
    )
    payload = manager.build_state()
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("lr_scheduler_state_dict")
    ema_state = payload.get("ema_state_dict")
    assert model_state is not None
    assert optimizer_state is not None
    assert scheduler_state is not None
    assert ema_state is not None
    original_model_weight = source_model.weight.detach().clone()
    source_optimizer_tensor = next(
        value
        for values in source_optimizer.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    )
    original_optimizer_tensor = source_optimizer_tensor.detach().clone()
    original_scheduler_tensor = source_scheduler.tensor_state.detach().clone()
    original_ema_tensor = source_ema.shadow_params["weight"].detach().clone()

    model_state["weight"].add_(10.0)
    payload_optimizer_tensor = next(
        value
        for values in optimizer_state["state"].values()
        for value in values.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    )
    payload_optimizer_tensor.add_(10.0)
    scheduler_state["tensor_state"].add_(10.0)
    ema_state["shadow_params"]["weight"].add_(10.0)

    assert torch.equal(source_model.weight, original_model_weight)
    assert torch.equal(source_optimizer_tensor, original_optimizer_tensor)
    assert torch.equal(source_scheduler.tensor_state, original_scheduler_tensor)
    assert torch.equal(source_ema.shadow_params["weight"], original_ema_tensor)

    target_model = nn.Linear(1, 1)
    target_optimizer = torch.optim.AdamW(target_model.parameters())
    target_scheduler = FixtureTensorStateScheduler(target_optimizer)
    target_ema = ExponentialMovingAverage(target_model)
    clean_payload = manager.build_state()
    CheckpointManager(
        target_model,
        optimizer=target_optimizer,
        lr_scheduler=target_scheduler,
        ema=target_ema,
    ).restore_payload(clean_payload, path="detached-payload.pt")
    restored_model_weight = target_model.weight.detach().clone()
    restored_optimizer_tensor = next(
        value
        for values in target_optimizer.state.values()
        for value in values.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    )
    restored_optimizer_snapshot = restored_optimizer_tensor.detach().clone()
    restored_scheduler_tensor = target_scheduler.tensor_state.detach().clone()
    restored_ema_tensor = target_ema.shadow_params["weight"].detach().clone()

    clean_model_state = clean_payload.get("model_state_dict")
    clean_optimizer_state = clean_payload.get("optimizer_state_dict")
    clean_scheduler_state = clean_payload.get("lr_scheduler_state_dict")
    clean_ema_state = clean_payload.get("ema_state_dict")
    assert clean_model_state is not None
    assert clean_optimizer_state is not None
    assert clean_scheduler_state is not None
    assert clean_ema_state is not None
    clean_model_state["weight"].add_(10.0)
    clean_optimizer_tensor = next(
        value
        for values in clean_optimizer_state["state"].values()
        for value in values.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    )
    clean_optimizer_tensor.add_(10.0)
    clean_scheduler_state["tensor_state"].add_(10.0)
    clean_ema_state["shadow_params"]["weight"].add_(10.0)

    assert torch.equal(target_model.weight, restored_model_weight)
    assert torch.equal(restored_optimizer_tensor, restored_optimizer_snapshot)
    assert torch.equal(target_scheduler.tensor_state, restored_scheduler_tensor)
    assert torch.equal(target_ema.shadow_params["weight"], restored_ema_tensor)


def test_ema_topology_is_strict_and_checked_before_model_restore() -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(42.0)
    source_ema = ExponentialMovingAverage(source_model)
    state = CheckpointManager(source_model, ema=source_ema).build_state()
    target_model = nn.Linear(1, 1)
    original_weight = target_model.weight.detach().clone()

    with pytest.raises(ValueError, match="runtime has no EMA"):
        CheckpointManager(target_model).restore_payload(
            state,
            path="unexpected-ema.pt",
        )

    assert torch.equal(target_model.weight, original_weight)

    missing_projection = dict(state)
    missing_projection.pop("ema_model_state_dict")
    target_ema = ExponentialMovingAverage(target_model)
    with pytest.raises(TypeError, match="ema_model_state_dict"):
        CheckpointManager(target_model, ema=target_ema).restore_payload(
            missing_projection,
            path="missing-ema-projection.pt",
        )
    assert torch.equal(target_model.weight, original_weight)


def test_ema_projection_must_match_canonical_shadow_state() -> None:
    model = nn.Linear(1, 1)
    ema = ExponentialMovingAverage(model)
    manager = CheckpointManager(model, ema=ema)
    state = manager.build_state()
    ema_model_state = state.get("ema_model_state_dict")
    assert ema_model_state is not None
    ema_model_state["weight"] = torch.full_like(model.weight, 99.0)

    with pytest.raises(ValueError, match="does not match EMA state"):
        manager.restore_payload(state, path="inconsistent-ema.pt")


def test_matching_nan_ema_projection_remains_structurally_restorable() -> None:
    source_model = nn.Linear(1, 1)
    source_model.weight.data.fill_(float("nan"))
    source_ema = ExponentialMovingAverage(source_model)
    state = CheckpointManager(source_model, ema=source_ema).build_state()
    target_model = nn.Linear(1, 1)
    target_ema = ExponentialMovingAverage(target_model)

    CheckpointManager(target_model, ema=target_ema).restore_payload(
        state,
        path="matching-nan-ema.pt",
    )

    assert torch.isnan(target_model.weight).all()
    assert torch.isnan(target_ema.shadow_params["weight"]).all()


def test_tied_parameter_and_buffer_ema_round_trip(tmp_path: Path) -> None:
    source_model = FixtureTiedStateModule()
    source_model.first.weight.data.fill_(2.0)
    source_model.shared_buffer_a.fill_(4.0)
    source_ema = ExponentialMovingAverage(source_model)
    source_ema.shadow_params["first.weight"].fill_(3.0)
    source_ema.shadow_buffers["shared_buffer_a"].fill_(5.0)
    checkpoint = CheckpointManager(
        source_model,
        ema=source_ema,
    ).save(tmp_path / "tied-ema.pt")

    loaded = CheckpointManager.load_payload(checkpoint)
    model_state = loaded.get("model_state_dict")
    ema_model_state = loaded.get("ema_model_state_dict")
    assert model_state is not None
    assert ema_model_state is not None
    assert model_state["first.weight"].data_ptr() == (
        model_state["second.weight"].data_ptr()
    )
    assert ema_model_state["first.weight"].data_ptr() == (
        ema_model_state["second.weight"].data_ptr()
    )

    target_model = FixtureTiedStateModule()
    target_ema = ExponentialMovingAverage(target_model)
    CheckpointManager(target_model, ema=target_ema).restore_payload(
        loaded,
        path=checkpoint,
    )

    assert target_model.first.weight is target_model.second.weight
    assert target_model.shared_buffer_a is target_model.shared_buffer_b
    assert torch.equal(target_model.first.weight, torch.full((1, 1), 2.0))
    assert torch.equal(target_ema.shadow_params["first.weight"], torch.full((1, 1), 3.0))
    assert torch.equal(
        target_ema.shadow_buffers["shared_buffer_a"],
        torch.full((1,), 5.0),
    )
    target_model.first.weight.data.fill_(7.0)
    target_model.shared_buffer_a.fill_(8.0)
    target_ema.update(target_model)


def test_legacy_v8_checkpoint_is_not_migrated(tmp_path: Path) -> None:
    legacy = _legacy_v8_payload(CheckpointManager(nn.Linear(1, 1)))
    checkpoint = tmp_path / "legacy-v8.pt"
    torch.save(legacy, checkpoint)

    with pytest.raises(ValueError, match="expected version 10"):
        CheckpointManager.load_payload(checkpoint)


def test_native_v10_tied_ema_does_not_infer_aliases_from_equal_values() -> None:
    source_model = FixtureTiedStateModule()
    source_model.first.weight.data.fill_(2.0)
    source_ema = ExponentialMovingAverage(source_model)
    source_ema.shadow_params["first.weight"].fill_(3.0)
    state = CheckpointManager(source_model, ema=source_ema).build_state()
    model_state = state.get("model_state_dict")
    assert model_state is not None
    state["model_state_dict"] = type(model_state)(
        (name, tensor.detach().clone())
        for name, tensor in model_state.items()
    )

    with pytest.raises(ValueError, match=r"second\.weight.*EMA state"):
        CheckpointManager.save_payload(state, "native-v10-tied-invalid.pt")


@pytest.mark.parametrize("missing_field", ["ema_state_dict", "ema_model_state_dict"])
def test_shared_v10_header_rejects_incomplete_ema_topology_on_save_and_load(
    tmp_path: Path,
    missing_field: str,
) -> None:
    model = nn.Linear(1, 1)
    state = CheckpointManager(
        model,
        ema=ExponentialMovingAverage(model),
    ).build_state()
    state.pop(missing_field)
    destination = tmp_path / f"missing-{missing_field}.pt"

    with pytest.raises(TypeError, match=missing_field):
        CheckpointManager.save_payload(state, destination)
    torch.save(state, destination)
    with pytest.raises(TypeError, match=missing_field):
        CheckpointManager.load_payload(destination)


def test_shared_v10_header_rejects_malformed_ema_and_projection(
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    manager = CheckpointManager(model, ema=ExponentialMovingAverage(model))
    malformed_schema = manager.build_state()
    malformed_ema = malformed_schema.get("ema_state_dict")
    assert malformed_ema is not None
    cast(dict[str, object], malformed_ema)["unexpected"] = True
    malformed_path = tmp_path / "malformed-ema.pt"
    torch.save(malformed_schema, malformed_path)

    with pytest.raises(ValueError, match="invalid fields"):
        CheckpointManager.load_payload(malformed_path)

    inconsistent_projection = manager.build_state()
    ema_projection = inconsistent_projection.get("ema_model_state_dict")
    assert ema_projection is not None
    ema_projection["weight"].fill_(99.0)
    projection_path = tmp_path / "inconsistent-projection.pt"
    torch.save(inconsistent_projection, projection_path)

    with pytest.raises(ValueError, match="does not match EMA state"):
        CheckpointManager.load_payload(projection_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_payload_restores_ema_to_cuda_runtime_and_remains_updatable() -> None:
    source_model = nn.Linear(1, 1)
    source_ema = ExponentialMovingAverage(source_model)
    payload = CheckpointManager(source_model, ema=source_ema).build_state()

    target_model = nn.Linear(1, 1).cuda()
    target_ema = ExponentialMovingAverage(target_model)
    CheckpointManager(target_model, ema=target_ema).restore_payload(
        payload,
        path="cpu-to-cuda.pt",
    )

    assert all(
        tensor.device.type == "cuda"
        for tensor in (
            *target_ema.shadow_params.values(),
            *target_ema.shadow_buffers.values(),
        )
    )
    target_model.weight.data.add_(1.0)
    target_ema.update(target_model)


def test_legacy_v9_checkpoint_is_not_migrated(tmp_path: Path) -> None:
    payload = CheckpointManager(nn.Linear(1, 1)).build_state()
    payload["format_version"] = 9
    payload.pop("inference_recipe")
    checkpoint = tmp_path / "legacy-v9.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="expected version 10"):
        CheckpointManager.load_payload(checkpoint)


@pytest.mark.parametrize(
    "field_name",
    [
        "precision_kind",
        "grad_scaler_class",
        "grad_scaler_state_dict",
        "inference_asset_descriptors",
    ],
)
def test_v8_is_rejected_regardless_of_newer_payload_fields(
    field_name: str,
) -> None:
    manager = CheckpointManager(nn.Linear(1, 1))
    legacy = _legacy_v8_payload(manager)
    legacy[field_name] = {} if field_name.endswith(("dict", "descriptors")) else "fp32"

    with pytest.raises(ValueError, match="expected version 10"):
        manager.restore_payload(legacy, path="smuggled-v8.pt")


@pytest.mark.parametrize(
    "field_name",
    ["precision", "accumulate_grad_batches"],
)
def test_v8_rejects_smuggled_trainer_precision_fields(
    field_name: str,
) -> None:
    manager = CheckpointManager(nn.Linear(1, 1))
    legacy = _legacy_v8_payload(
        manager,
        config={"trainer": {}},
    )
    legacy["config"]["trainer"][field_name] = (
        "fp32" if field_name == "precision" else 1
    )

    with pytest.raises(ValueError, match="expected version 10"):
        manager.restore_payload(legacy, path="smuggled-config-v8.pt")


def test_v8_cannot_resume_into_fp16_runtime() -> None:
    source = CheckpointManager(nn.Linear(1, 1))
    legacy = _legacy_v8_payload(
        source,
        config={"trainer": {"device": "cpu"}},
    )
    target_model = nn.Linear(1, 1)
    original_weight = target_model.weight.detach().clone()

    with pytest.raises(ValueError, match="expected version 10"):
        CheckpointManager(
            target_model,
            precision_kind="fp16-mixed",
            grad_scaler=_cuda_grad_scaler(),
        ).restore_payload(legacy, path="legacy-into-fp16.pt")
    assert torch.equal(target_model.weight, original_weight)


def test_writer_rejects_legacy_v8_payload(tmp_path: Path) -> None:
    legacy = _legacy_v8_payload(CheckpointManager(nn.Linear(1, 1)))
    destination = tmp_path / "legacy.pt"

    with pytest.raises(ValueError, match="writer requires format version 10"):
        CheckpointManager.save_payload(legacy, destination)

    assert not destination.exists()


def test_restore_payload_reuses_loaded_payload_without_reading_disk(
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1)
    manager = CheckpointManager(model)
    expected_weight = model.weight.detach().clone()
    payload = manager.build_state(epoch=3, global_step=17)
    model.weight.data.zero_()

    loaded = manager.restore_payload(payload, path=tmp_path / "already-loaded.pt")

    assert torch.equal(model.weight, expected_weight)
    assert loaded.path == tmp_path / "already-loaded.pt"
    assert loaded.epoch == 3
    assert loaded.global_step == 17


def test_restore_payload_rejects_v7_and_missing_plugin_metadata(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(nn.Linear(1, 1))
    v7 = manager.build_state()
    v7["format_version"] = 7
    with pytest.raises(ValueError, match=r"version 7.*expected version 10"):
        manager.restore_payload(v7, path=tmp_path / "v7.pt")

    missing_plugins = manager.build_state()
    missing_metadata = missing_plugins.get("metadata")
    assert missing_metadata is not None
    missing_metadata.pop("extension_plugins")
    with pytest.raises(
        TypeError,
        match=r"metadata\.extension_plugins must be a list",
    ):
        manager.restore_payload(missing_plugins, path=tmp_path / "missing.pt")


def test_restore_payload_rejects_legacy_v8_rng_state_without_mps(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(nn.Linear(1, 1))
    legacy_v8 = _legacy_v8_payload(manager)
    rng_state = legacy_v8.get("rng_state")
    assert rng_state is not None
    rng_state.pop("torch_mps")

    with pytest.raises(ValueError, match="expected version 10"):
        manager.restore_payload(
            legacy_v8,
            path=tmp_path / "legacy-v8.pt",
        )


def test_save_rejects_custom_extra_state_with_precise_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nested" / "unsafe.pt"
    manager = CheckpointManager(
        nn.Linear(1, 1),
        auxiliary_modules={"asset": FixtureUnsafeExtraStateModule()},
    )

    with pytest.raises(
        TypeError,
        match=(
            r"checkpoint\['training_assets_state_dict'\]\['asset'\]"
            r"\['_extra_state'\].*FixtureUnsafeExtraState"
        ),
    ):
        manager.save(checkpoint)

    assert not checkpoint.parent.exists()


def test_save_rejects_tensor_subclasses_with_precise_path(tmp_path: Path) -> None:
    model = nn.Module()
    value = torch.Tensor._make_subclass(
        FixtureTensorSubclass,
        torch.ones(1),
        require_grad=False,
    )
    model.register_buffer("state", value)

    with pytest.raises(
        TypeError,
        match=r"checkpoint\['model_state_dict'\]\['state'\].*FixtureTensorSubclass",
    ):
        CheckpointManager(model).save(tmp_path / "tensor-subclass.pt")


def test_load_payload_does_not_execute_pickle_globals(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    checkpoint = tmp_path / "unsafe-pickle.pt"
    torch.save({"value": FixtureExecutablePickleValue(marker)}, checkpoint)

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        CheckpointManager.load_payload(checkpoint)

    assert not marker.exists()


def test_load_payload_rejects_non_contract_set_across_torch_versions(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "set.pt"
    torch.save({"value": {1, 2}}, checkpoint)

    with pytest.raises((TypeError, pickle.UnpicklingError)) as error:
        CheckpointManager.load_payload(checkpoint)
    if isinstance(error.value, TypeError):
        assert "checkpoint['value']" in str(error.value)
        assert "builtins.set" in str(error.value)
    else:
        assert "Weights only load failed" in str(error.value)
        assert "set" in str(error.value)


def test_data_only_extra_state_round_trips(tmp_path: Path) -> None:
    class DataOnlyExtraStateModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.extra_state: dict[str, Any] = {
                "version": 1,
                "labels": ["initial"],
            }

        def get_extra_state(self) -> dict[str, Any]:
            return self.extra_state

        def set_extra_state(self, state: dict[str, Any]) -> None:
            self.extra_state = state

    module = DataOnlyExtraStateModule()
    manager = CheckpointManager(module)
    checkpoint = manager.save(tmp_path / "safe-extra-state.pt")
    module.extra_state = {"version": 99, "labels": ["mutated"]}

    manager.load(checkpoint)

    assert module.extra_state == {"version": 1, "labels": ["initial"]}


def test_save_payload_failure_preserves_destination_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"previous-checkpoint")
    payload = CheckpointManager(nn.Linear(1, 1)).build_state()

    def fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(OSError, match="simulated serialization failure"):
        CheckpointManager.save_payload(payload, checkpoint)

    assert checkpoint.read_bytes() == b"previous-checkpoint"
    assert tuple(tmp_path.iterdir()) == (checkpoint,)


def test_rng_state_round_trips_through_weights_only_checkpoint(tmp_path: Path) -> None:
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    checkpoint = CheckpointManager(nn.Linear(1, 1)).save(tmp_path / "rng.pt")

    expected = (random.random(), float(np.random.random()), torch.rand(4))
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")

    encoded = payload.get("rng_state")
    assert isinstance(encoded, dict)
    numpy_state = encoded["numpy"]
    assert isinstance(numpy_state, dict)
    assert type(numpy_state["keys"]) is list
    restore_rng_state(parse_rng_state(encoded, require_cuda_compatibility=True))

    actual = (random.random(), float(np.random.random()), torch.rand(4))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_mps_rng_state_round_trips_through_checkpoint() -> None:
    torch.mps.manual_seed(123)
    encoded = capture_rng_state()
    expected = torch.rand(4, device="mps").cpu()
    torch.mps.manual_seed(456)

    parsed = parse_rng_state(encoded, require_mps_compatibility=True)
    restore_rng_state(parsed, restore_cuda=False, restore_mps=True)

    actual = torch.rand(4, device="mps").cpu()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_mps_rng_restore_failure_rolls_back_all_rng_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    torch.mps.manual_seed(44)
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    previous_cpu = torch.random.get_rng_state().clone()
    previous_mps = torch.mps.get_rng_state().clone()

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    torch.mps.manual_seed(404)
    target = parse_rng_state(
        capture_rng_state(),
        require_mps_compatibility=True,
    )

    random.setstate(previous_python)
    np.random.set_state(previous_numpy)
    torch.random.set_rng_state(previous_cpu)
    torch.mps.set_rng_state(previous_mps)
    original_set_rng_state = torch.mps.set_rng_state
    calls = 0

    def fail_once(
        state: torch.Tensor,
        device: int | str | torch.device = "mps",
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated MPS restore failure")
        original_set_rng_state(state, device=device)

    monkeypatch.setattr(torch.mps, "set_rng_state", fail_once)

    with pytest.raises(RuntimeError, match="simulated MPS restore failure"):
        restore_rng_state(target, restore_cuda=False, restore_mps=True)

    assert random.getstate() == previous_python
    np.testing.assert_equal(np.random.get_state(), previous_numpy)
    assert torch.equal(torch.random.get_rng_state(), previous_cpu)
    assert torch.equal(torch.mps.get_rng_state(), previous_mps)


def test_generic_checkpoint_load_does_not_restore_rng(tmp_path: Path) -> None:
    checkpoint = CheckpointManager(nn.Linear(1, 1)).save(tmp_path / "rng.pt")
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)

    CheckpointManager.load_payload(checkpoint, map_location="cpu")
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda state: state.pop("python"),
            r"rng_state has invalid fields.*python",
        ),
        (
            lambda state: state["numpy"].__setitem__("keys", [True]),
            r"numpy\.keys\[0\] must be an exact integer",
        ),
        (
            lambda state: state.__setitem__("torch_cpu", torch.ones(4)),
            r"torch_cpu must be a non-empty one-dimensional uint8 Tensor",
        ),
    ],
)
def test_parse_rng_state_rejects_malformed_snapshots(
    mutate: Any,
    error: str,
) -> None:
    encoded = capture_rng_state()
    mutate(encoded)

    with pytest.raises((TypeError, ValueError), match=error):
        parse_rng_state(encoded)
