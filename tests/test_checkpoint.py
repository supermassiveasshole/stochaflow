"""Tests for the v8 data-only checkpoint contract."""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    capture_rng_state,
    parse_rng_state,
    restore_rng_state,
)
from stochaflow.utils.plugins import ExtensionIdentityError


def _write_load_marker(path: str) -> object:
    Path(path).write_text("loaded", encoding="utf-8")
    return object()


class _ExecutablePickleValue:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_load_marker, (str(self.marker),)


class _UnsafeExtraState:
    pass


class _UnsafeExtraStateModule(nn.Module):
    def get_extra_state(self) -> object:
        return _UnsafeExtraState()

    def set_extra_state(self, state: object) -> None:
        del state


class _TensorSubclass(torch.Tensor):
    pass


def test_v8_checkpoint_always_records_extension_plugin_metadata() -> None:
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

    assert CHECKPOINT_FORMAT_VERSION == 8
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
    with pytest.raises(ValueError, match=r"version 7.*expected version 8"):
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


def test_restore_payload_accepts_legacy_v8_rng_state_without_mps(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(nn.Linear(1, 1))
    legacy_v8 = manager.build_state()
    rng_state = legacy_v8.get("rng_state")
    assert rng_state is not None
    rng_state.pop("torch_mps")

    loaded = manager.restore_payload(
        legacy_v8,
        path=tmp_path / "legacy-v8.pt",
    )

    assert loaded.path == tmp_path / "legacy-v8.pt"


def test_save_rejects_custom_extra_state_with_precise_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nested" / "unsafe.pt"
    manager = CheckpointManager(
        nn.Linear(1, 1),
        auxiliary_modules={"asset": _UnsafeExtraStateModule()},
    )

    with pytest.raises(
        TypeError,
        match=(
            r"checkpoint\['training_assets_state_dict'\]\['asset'\]"
            r"\['_extra_state'\].*_UnsafeExtraState"
        ),
    ):
        manager.save(checkpoint)

    assert not checkpoint.parent.exists()


def test_save_rejects_tensor_subclasses_with_precise_path(tmp_path: Path) -> None:
    model = nn.Module()
    value = torch.Tensor._make_subclass(
        _TensorSubclass,
        torch.ones(1),
        require_grad=False,
    )
    model.register_buffer("state", value)

    with pytest.raises(
        TypeError,
        match=r"checkpoint\['model_state_dict'\]\['state'\].*_TensorSubclass",
    ):
        CheckpointManager(model).save(tmp_path / "tensor-subclass.pt")


def test_load_payload_does_not_execute_pickle_globals(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    checkpoint = tmp_path / "unsafe-pickle.pt"
    torch.save({"value": _ExecutablePickleValue(marker)}, checkpoint)

    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        CheckpointManager.load_payload(checkpoint)

    assert not marker.exists()


def test_load_payload_rejects_weights_only_safe_but_non_contract_container(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "set.pt"
    torch.save({"value": {1, 2}}, checkpoint)

    with pytest.raises(TypeError, match=r"checkpoint\['value'\].*builtins\.set"):
        CheckpointManager.load_payload(checkpoint)


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
