"""Tests for shared read-only checkpoint inference projection."""

import hashlib
from typing import cast

import pytest
import torch
from torch import nn

import stochaflow.inference.checkpoint as checkpoint_module
from stochaflow.inference.checkpoint import (
    InferenceCheckpointView,
    build_checkpointed_process,
    checkpoint_content_digest,
    checkpoint_epoch_and_step,
    project_inference_checkpoint,
)
from stochaflow.inference.model import InferenceModelProvider
from stochaflow.processes import Process
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION, CheckpointState
from stochaflow.utils.config import load_config_dict


class StatefulInferenceProcess(Process):
    """Small process fixture with strict checkpointed buffer identity."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))


def test_inference_projection_excludes_training_lifecycle_state() -> None:
    model_state = {"weight": torch.ones(1)}
    payload = cast(
        CheckpointState,
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": {"experiment": {"name": "projection"}},
            "inference_recipe": None,
            "metadata": {"extension_plugins": []},
            "model_state_dict": model_state,
            "epoch": 3,
            "global_step": 17,
            "inference_asset_descriptors": {},
            "optimizer_state_dict": {"state": {0: {"moment": torch.ones(1)}}},
            "lr_scheduler_state_dict": {"last_epoch": 2},
            "grad_scaler_state_dict": {"scale": 4.0},
            "rng_state": {"python": "opaque"},
            "training_assets_state_dict": {},
        },
    )

    projected = project_inference_checkpoint(payload)

    assert set(projected) == {
        "format_version",
        "config",
        "inference_recipe",
        "model_state_dict",
        "inference_asset_descriptors",
        "inference_asset_state_dicts",
    }
    assert "metadata" not in projected
    assert projected.get("model_state_dict") is model_state
    assert checkpoint_epoch_and_step(payload) == (3, 17)


def test_inference_projection_excludes_legacy_epoch_validation_summary() -> None:
    legacy_training_loop = {
        "epoch_validation": {
            "identity": {
                "profile_digest": "a" * 64,
                "metric_keys": ["valid/metrics/fid"],
                "cadence": {
                    "first_epoch": 100,
                    "every_n_epochs": 10,
                    "include_final": True,
                },
            },
            "last_evaluated_epoch": 200,
            "last_metrics": {"valid/metrics/fid": 20.0},
            "off_cadence_final_epochs": [],
        }
    }
    metadata = {
        "extension_plugins": [],
        "training_loop": legacy_training_loop,
    }
    payload = cast(
        CheckpointState,
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "config": {"experiment": {"name": "legacy-e200"}},
            "inference_recipe": None,
            "metadata": metadata,
            "model_state_dict": {"weight": torch.ones(1)},
            "inference_asset_descriptors": {},
        },
    )

    projected = project_inference_checkpoint(payload)

    assert projected.get("format_version") == CHECKPOINT_FORMAT_VERSION
    assert "metadata" not in projected


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"epoch": 0, "global_step": 1}, "epoch"),
        ({"epoch": 1, "global_step": -1}, "global_step"),
        ({"epoch": True, "global_step": 1}, "epoch"),
    ],
)
def test_checkpoint_progress_identity_is_strict(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        checkpoint_epoch_and_step(payload)


def test_checkpoint_content_digest_hashes_exact_bytes(tmp_path) -> None:
    path = tmp_path / "subject.pt"
    content = b"portable checkpoint identity\x00\xff"
    path.write_bytes(content)

    assert checkpoint_content_digest(path) == hashlib.sha256(content).hexdigest()


def test_primary_inference_model_rejects_state_dtype_conversion() -> None:
    model = nn.Linear(2, 1)
    state = {
        name: value.to(dtype=torch.float64)
        for name, value in model.state_dict().items()
    }
    provider = InferenceModelProvider(
        model_factory=lambda: nn.Linear(2, 1),
        raw_state_dict=state,
        ema_state_dict=None,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="dtype does not match runtime"):
        provider.resolve("raw")


def test_inference_process_rejects_state_dtype_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config_dict(
        {
            "experiment": {"name": "strict-inference-state"},
            "data": {"name": "unused.data", "params": {}},
            "model": {"name": "unused.model", "params": {}},
            "process": {"name": "unused.process", "params": {}},
            "training": {"name": "unused.training", "params": {}},
            "trainer": {"precision": "fp32"},
        }
    )
    payload = cast(
        InferenceCheckpointView,
        {"process_state_dict": {"scale": torch.ones(1, dtype=torch.float64)}},
    )
    monkeypatch.setattr(
        checkpoint_module,
        "build_process",
        lambda _config: StatefulInferenceProcess(),
    )

    with pytest.raises(ValueError, match="dtype does not match runtime"):
        build_checkpointed_process(config, payload, device=torch.device("cpu"))
