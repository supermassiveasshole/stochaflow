"""Shared inference-only projection of portable training checkpoints."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Required, TypedDict, cast

import torch
from torch import nn

from stochaflow.processes.base import Process
from stochaflow.sampling.assets import InferenceAssetProvider
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    CheckpointState,
    InferenceAssetDescriptor,
    validate_inference_asset_descriptors,
    validate_module_state_dict_compatibility,
)
from stochaflow.utils.config import (
    ComponentConfig,
    StochaflowConfig,
    load_config_dict,
)
from stochaflow.utils.device import move_module_to_device
from stochaflow.utils.factory import build_model, build_process
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    sampling_recipe_from_dict,
)

from .model import InferenceModelProvider


class InferenceCheckpointView(TypedDict, total=False):
    """Validated state retained for read-only inference operations."""

    format_version: int
    config: dict[str, Any]
    inference_recipe: dict[str, Any] | None
    metadata: dict[str, Any]
    model_state_dict: dict[str, Any]
    ema_model_state_dict: dict[str, Any]
    process_state_dict: dict[str, Any]
    inference_asset_descriptors: Required[
        dict[str, InferenceAssetDescriptor]
    ]
    inference_asset_state_dicts: Required[dict[str, dict[str, Any]]]


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve an explicit checkpoint file or one best checkpoint directory."""

    path = Path(checkpoint)
    if path.is_dir():
        return CheckpointManager.find_best(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def load_stable_checkpoint_snapshot(
    path: str | Path,
) -> tuple[CheckpointState, str]:
    """Snapshot, hash, and load the same stable checkpoint bytes exactly once."""

    source_path = Path(path)
    digest = hashlib.sha256()
    copied = 0
    with source_path.open("rb") as source:
        before = os.fstat(source.fileno())
        with tempfile.TemporaryFile(mode="w+b") as snapshot:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                snapshot.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            after = os.fstat(source.fileno())
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise RuntimeError("checkpoint changed while its snapshot was read")
            if copied != before.st_size:
                raise RuntimeError("checkpoint byte count changed during snapshot")
            snapshot.flush()
            snapshot.seek(0)
            raw = torch.load(snapshot, map_location="cpu", weights_only=True)
    if type(raw) is not dict:
        raise TypeError(
            f"checkpoint at '{source_path}' must contain a dictionary payload"
        )
    payload = cast(CheckpointState, raw)
    version = cast(object, payload.get("format_version"))
    if type(version) is not int:
        raise TypeError("checkpoint format_version must be an exact integer")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version!r} is unsupported; "
            f"expected version {CHECKPOINT_FORMAT_VERSION}"
        )
    return payload, digest.hexdigest()


def checkpoint_content_digest(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one checkpoint file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_config(payload: CheckpointState) -> StochaflowConfig:
    """Load the training authority retained by a supported checkpoint."""

    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version!r} is unsupported; "
            f"expected version {CHECKPOINT_FORMAT_VERSION}"
        )
    raw = cast(object, payload.get("config"))
    if raw is None:
        raise ValueError("checkpoint does not contain a Stochaflow config")
    if not isinstance(raw, dict):
        raise TypeError("checkpoint config must be a mapping")
    return load_config_dict(raw)


def load_checkpoint_recipe(payload: CheckpointState) -> SamplingRecipe:
    """Load the optional task inference recipe from a checkpoint."""

    if "inference_recipe" not in payload:
        raise ValueError("checkpoint does not contain an inference recipe")
    raw = cast(object, payload["inference_recipe"])
    if raw is None:
        raise ValueError("checkpoint does not support task inference")
    return sampling_recipe_from_dict(raw)


def project_inference_checkpoint(
    payload: CheckpointState,
) -> InferenceCheckpointView:
    """Drop optimizer, scheduler, scaler, RNG, and training-loop state."""

    retained_keys = (
        "format_version",
        "config",
        "inference_recipe",
        "metadata",
        "model_state_dict",
        "ema_model_state_dict",
        "process_state_dict",
    )
    raw_payload = cast(dict[str, Any], payload)
    descriptors = validate_inference_asset_descriptors(
        payload.get("inference_asset_descriptors"),
        path="checkpoint.inference_asset_descriptors",
    )
    projected_states: dict[str, dict[str, Any]] = {}
    if descriptors:
        asset_states_value = cast(
            object,
            payload.get("training_assets_state_dict"),
        )
        if type(asset_states_value) is not dict:
            raise TypeError(
                "checkpoint with inference assets requires an exact "
                "training_assets_state_dict"
            )
        asset_states = cast(dict[object, object], asset_states_value)
        for descriptor in descriptors.values():
            asset_name = descriptor["training_asset_name"]
            state_value = asset_states.get(asset_name)
            if not isinstance(state_value, dict):
                raise TypeError(
                    "checkpoint embedded inference asset state "
                    f"{asset_name!r} must be a state dictionary"
                )
            projected_states[asset_name] = cast(dict[str, Any], state_value)
    view = cast(
        InferenceCheckpointView,
        {key: raw_payload[key] for key in retained_keys if key in raw_payload},
    )
    view["inference_asset_descriptors"] = descriptors
    view["inference_asset_state_dicts"] = projected_states
    return view


def checkpoint_epoch_and_step(
    payload: Mapping[str, object],
) -> tuple[int, int]:
    """Validate and return the immutable checkpoint progress identity."""

    epoch = payload.get("epoch")
    if type(epoch) is not int or epoch <= 0:
        raise TypeError("checkpoint epoch must be a positive integer")
    global_step = payload.get("global_step")
    if type(global_step) is not int or global_step < 0:
        raise TypeError("checkpoint global_step must be a non-negative integer")
    return epoch, global_step


def build_checkpointed_process(
    config: StochaflowConfig,
    payload: InferenceCheckpointView,
    *,
    device: torch.device,
) -> Process | None:
    """Reconstruct only the process state required for task inference."""

    has_state = "process_state_dict" in payload
    if config.process is None:
        if has_state:
            raise ValueError(
                "checkpoint contains 'process_state_dict' but config.process is null"
            )
        return None
    if not has_state:
        raise ValueError("checkpoint is missing required 'process_state_dict'")
    state = payload.get("process_state_dict")
    if not isinstance(state, dict):
        raise TypeError("checkpoint process_state_dict must be a mapping")
    process = build_process(config.process)
    validate_module_state_dict_compatibility(
        process,
        state,
        path="inference.process_state_dict",
        allow_lazy_state=False,
    )
    process.load_state_dict(state, strict=True)
    move_module_to_device(process, device, role="inference process")
    process.eval()
    return process


def build_inference_model_provider(
    config: StochaflowConfig,
    payload: InferenceCheckpointView,
    *,
    device: torch.device,
    model_factory: Callable[[ComponentConfig], nn.Module] = build_model,
) -> InferenceModelProvider:
    """Build a lazy raw/EMA primary-model provider from projected state."""

    raw = cast(object, payload.get("model_state_dict"))
    if raw is None:
        raise ValueError("checkpoint is missing required 'model_state_dict'")
    if not isinstance(raw, dict):
        raise TypeError("checkpoint model_state_dict must be a mapping")
    ema = cast(object, payload.get("ema_model_state_dict"))
    if ema is not None and not isinstance(ema, dict):
        raise TypeError("checkpoint ema_model_state_dict must be a mapping")
    return InferenceModelProvider(
        model_factory=lambda: model_factory(config.model),
        raw_state_dict=raw,
        ema_state_dict=ema,
        device=device,
    )


def build_inference_asset_provider(
    payload: InferenceCheckpointView,
    *,
    device: torch.device,
) -> InferenceAssetProvider:
    """Build the lazy provider for checkpoint-declared inference assets."""

    descriptors = cast(
        Mapping[str, InferenceAssetDescriptor],
        payload.get("inference_asset_descriptors", {}),
    )
    state_dicts = cast(
        Mapping[str, Mapping[str, object]],
        payload.get("inference_asset_state_dicts", {}),
    )
    return InferenceAssetProvider(
        descriptors=descriptors,
        state_dicts=state_dicts,
        device=device,
        model_factory=build_model,
    )


__all__ = [
    "InferenceCheckpointView",
    "build_checkpointed_process",
    "build_inference_asset_provider",
    "build_inference_model_provider",
    "checkpoint_content_digest",
    "checkpoint_epoch_and_step",
    "load_checkpoint_config",
    "load_checkpoint_recipe",
    "load_stable_checkpoint_snapshot",
    "project_inference_checkpoint",
    "resolve_checkpoint_path",
]
