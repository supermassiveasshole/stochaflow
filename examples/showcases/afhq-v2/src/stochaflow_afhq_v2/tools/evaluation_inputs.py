"""Authenticated checkpoint and dataset inputs for AFHQ-v2 evaluation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch

from stochaflow.data import (
    DataArtifactBindings,
    DataLoaders,
    build_data_loaders,
)
from stochaflow.sampling.runtime import (
    ResolvedSamplingInputs,
    SamplingCheckpointView,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.plugins import ResolvedExtensions

BUILDER_NAME = "class_labeled_image"
SOURCE_NAME = "afhq-v2.official"


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """One authenticated byte snapshot used by every evaluation lifecycle."""

    source_path: Path
    snapshot_path: Path
    sha256: str
    size_bytes: int


def resolve_checkpoint_source(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint file or a run directory's best checkpoint."""

    source = Path(checkpoint)
    if source.is_dir():
        return CheckpointManager.find_best(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    return source.resolve()


def snapshot_checkpoint(
    source: Path,
    destination: Path,
) -> CheckpointSnapshot:
    """Copy and authenticate a checkpoint before loading mutable source bytes."""

    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as source_handle:
        before = os.fstat(source_handle.fileno())
        with destination.open("xb") as snapshot_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                snapshot_handle.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            snapshot_handle.flush()
            os.fsync(snapshot_handle.fileno())
        after = os.fstat(source_handle.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise RuntimeError("checkpoint changed while its evaluation snapshot was read")
    if copied != before.st_size:
        raise RuntimeError("checkpoint snapshot byte count changed during capture")
    return CheckpointSnapshot(
        source_path=source,
        snapshot_path=destination,
        sha256=digest.hexdigest(),
        size_bytes=copied,
    )


def verify_checkpoint_snapshot(snapshot: CheckpointSnapshot) -> None:
    """Reject changes to the private snapshot between evaluation phases."""

    if snapshot.snapshot_path.stat().st_size != snapshot.size_bytes:
        raise RuntimeError("private checkpoint snapshot size changed")
    digest = hashlib.sha256()
    with snapshot.snapshot_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != snapshot.sha256:
        raise RuntimeError("private checkpoint snapshot digest changed")


def checkpoint_progress(checkpoint_path: Path) -> dict[str, int]:
    """Read only scalar training progress through a memory-mapped payload."""

    payload = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if type(payload) is not dict:
        raise TypeError(
            f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
        )
    epoch = cast(object, payload.get("epoch"))
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("AFHQ-v2 evaluation checkpoint epoch must be positive")
    global_step = cast(object, payload.get("global_step"))
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(
            "AFHQ-v2 evaluation checkpoint global_step must be non-negative"
        )
    return {"epoch": epoch, "global_step": global_step}


def checkpoint_data_bindings(
    inputs: ResolvedSamplingInputs,
) -> DataArtifactBindings:
    """Recover and validate the source identity frozen in a checkpoint."""

    metadata = cast(object, inputs.checkpoint.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    raw = metadata.get("data_artifacts")
    if raw is None:
        raise ValueError(
            "AFHQ-v2 evaluation requires checkpoint data artifact bindings"
        )
    bindings = DataArtifactBindings.from_dict(
        raw,
        path="checkpoint metadata.data_artifacts",
    )
    bindings.assert_ids(("source",))
    if bindings.identity_for("source").source_name != SOURCE_NAME:
        raise ValueError(
            "checkpoint data artifact binding is not the official AFHQ-v2 source"
        )
    return bindings


def validate_data_config(inputs: ResolvedSamplingInputs) -> None:
    """Enforce the frozen AFHQ-v2 test-data protocol."""

    config = inputs.config
    if config.data.name != BUILDER_NAME:
        raise ValueError(
            "AFHQ-v2 evaluation requires the built-in "
            "class_labeled_image DataBuilder"
        )
    params = cast(object, config.data.params)
    if not isinstance(params, dict):
        raise TypeError("checkpoint data.params must be a mapping")
    source = cast(object, params.get("source"))
    if not isinstance(source, dict):
        raise TypeError("checkpoint data source must be a mapping")
    if source.get("name") != SOURCE_NAME:
        raise ValueError(
            "AFHQ-v2 evaluation requires the official AFHQ-v2 DataSource"
        )
    materialization = cast(object, source.get("materialization"))
    if not isinstance(materialization, dict):
        raise TypeError(
            "checkpoint data source materialization must be a mapping"
        )
    if materialization.get("policy") != "require":
        raise ValueError("AFHQ-v2 evaluation requires artifact policy: require")
    # Formal evaluation binds the checkpoint's exact artifact identity below.
    # ``strict_resume=True`` plus ``expected_artifacts`` upgrades materialization
    # to full content verification even when training used manifest verification.
    source_params = cast(object, source.get("params"))
    if not isinstance(source_params, dict):
        raise TypeError("checkpoint data source params must be a mapping")
    if source_params.get("resolution") != 128:
        raise ValueError(
            "AFHQ-v2 evaluation requires a 128x128 source artifact"
        )
    image = cast(object, params.get("image"))
    if not isinstance(image, dict):
        raise TypeError("checkpoint data image recipe must be a mapping")
    if (
        image.get("size") != [128, 128]
        or image.get("channels") != 3
        or image.get("normalize") is not True
    ):
        raise ValueError(
            "AFHQ-v2 evaluation requires exact normalized 128x128 RGB test images"
        )


def build_strict_test_loaders(
    extensions: ResolvedExtensions,
    expected: DataArtifactBindings,
) -> DataLoaders:
    """Build test data while requiring the checkpoint's exact artifact identity."""

    loaders = build_data_loaders(
        extensions.config.data,
        seed=extensions.config.experiment.seed,
        strict_resume=True,
        expected_artifacts=expected,
    )
    if loaders.artifact_bindings != expected:
        raise ValueError(
            "evaluation data artifact bindings changed during strict build"
        )
    if loaders.test is None:
        raise ValueError("AFHQ-v2 DataBuilder must expose the official test split")
    return loaders


def retain_result_checkpoint_header(
    inputs: ResolvedSamplingInputs,
) -> ResolvedSamplingInputs:
    """Release tensor state while retaining result-manifest metadata."""

    format_version = cast(object, inputs.checkpoint.get("format_version"))
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise TypeError("AFHQ-v2 evaluation checkpoint format_version must be integer")
    checkpoint: SamplingCheckpointView = {
        "format_version": format_version,
        "inference_asset_descriptors": {},
        "inference_asset_state_dicts": {},
    }
    return replace(inputs, checkpoint=checkpoint)


__all__ = [
    "CheckpointSnapshot",
    "build_strict_test_loaders",
    "checkpoint_data_bindings",
    "checkpoint_progress",
    "resolve_checkpoint_source",
    "retain_result_checkpoint_header",
    "snapshot_checkpoint",
    "validate_data_config",
    "verify_checkpoint_snapshot",
]
