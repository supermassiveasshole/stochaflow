"""Fixed-topology distributed resume bundles and portable v12 exports.

The bundle is deliberately separate from an ordinary checkpoint-v12 file.  It
contains one common checkpoint plus one RNG/data-plan file per rank, and only a
committed manifest makes those files discoverable as an exact-resume snapshot.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from stochaflow.data.ranked import (
    RankedEpochDataIdentity,
    RankedTrainEpochPlan,
)
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    CheckpointState,
    LoadedCheckpoint,
    ParsedRNGState,
    capture_local_rng_state,
    parse_rng_state,
    restore_local_rng_state,
    validate_checkpoint_payload,
    validate_inference_asset_descriptors,
)
from stochaflow.utils.run_manifest import write_yaml_manifest

from .contracts import DistributedTopology

DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION = 2
DISTRIBUTED_CHECKPOINT_BUNDLE_KIND = (
    "stochaflow.fixed_single_node_distributed_resume_bundle"
)
DISTRIBUTED_CHECKPOINT_MANIFEST_NAME = "bundle-manifest.yaml"
DISTRIBUTED_COMMON_CHECKPOINT_NAME = "common.pt"
DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME = "best-portable.pt"
DISTRIBUTED_COMMON_CHECKPOINT_ROLE = "distributed_common"
DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE = "distributed_portable"

_RANK_LOCAL_KIND = "stochaflow.distributed_rank_local_state"
_RANK_LOCAL_DIRECTORY = "ranks"
_PORTABLE_ATTACHMENT_SOURCE_KEY = "distributed_portable_attachment_source"
_SUPPORTED_RUNTIME_PAIRS = frozenset({("gloo", "cpu"), ("nccl", "cuda")})


@dataclass(frozen=True, slots=True)
class DistributedCheckpointBundlePaths:
    """Canonical hidden staging and absent final paths for one save attempt."""

    bundle_id: str
    completed_epoch: int
    global_step: int
    staging_directory: Path
    final_directory: Path


@dataclass(frozen=True, slots=True)
class DistributedCheckpointBundle:
    """Identity and committed files for one exact-resume bundle."""

    directory: Path
    manifest_path: Path
    bundle_id: str
    completed_epoch: int
    global_step: int
    world_size: int
    local_world_size: int
    backend: str
    device_type: str
    common_checkpoint_path: Path
    common_checkpoint_sha256: str
    manifest_sha256: str = ""
    best_portable_checkpoint_path: Path | None = None
    best_portable_selected_epoch: int | None = None
    best_portable_checkpoint_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class DistributedCheckpointRestore:
    """Fully validated state ready for coordinated all-rank restoration."""

    bundle: DistributedCheckpointBundle
    common_payload: CheckpointState
    rank_rng_state: ParsedRNGState
    next_plan: RankedTrainEpochPlan


@dataclass(frozen=True, slots=True)
class DistributedCheckpointPreflight:
    """Immutable common authority from one fully committed bundle.

    Preflight is suitable for resolving configuration before the current
    DataBuilder runs.  It is not restoration authority: callers must still
    obtain a fresh ranked next plan and call
    :func:`load_distributed_checkpoint_bundle` before mutating runtime state.
    """

    bundle: DistributedCheckpointBundle
    common_payload: CheckpointState


@dataclass(frozen=True, slots=True)
class DistributedCheckpointFile:
    """One content-addressed file in a committed bundle inventory."""

    relative_path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the strict data-only manifest representation."""

        return {
            "path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LoadedDistributedBundle:
    """Internal fully validated committed bundle contents."""

    bundle: DistributedCheckpointBundle
    common_payload: CheckpointState
    rank_plans: dict[int, RankedTrainEpochPlan]
    rank_rng_states: dict[int, ParsedRNGState]
    manifest_sha256: str


def new_distributed_checkpoint_bundle_id() -> str:
    """Return a canonical unpredictable identity for one save attempt."""

    return uuid.uuid4().hex


def distributed_checkpoint_bundle_paths(
    root: str | Path,
    *,
    bundle_id: str,
    completed_epoch: int,
    global_step: int,
) -> DistributedCheckpointBundlePaths:
    """Derive same-filesystem staging and final paths without creating them."""

    validated_bundle_id = _bundle_id(bundle_id, path="bundle_id")
    epoch = _positive_int(completed_epoch, path="completed_epoch")
    step = _nonnegative_int(global_step, path="global_step")
    root_path = Path(root)
    return DistributedCheckpointBundlePaths(
        bundle_id=validated_bundle_id,
        completed_epoch=epoch,
        global_step=step,
        staging_directory=root_path / f".staging-{validated_bundle_id}",
        final_directory=(
            root_path / f"epoch-{epoch:08d}-{validated_bundle_id[:12]}"
        ),
    )


def stage_distributed_common_checkpoint(
    paths: DistributedCheckpointBundlePaths,
    payload: object,
) -> DistributedCheckpointFile:
    """Write rank-zero's common valid-v12 state into the hidden staging area."""

    _validate_bundle_paths(paths)
    destination = paths.staging_directory / DISTRIBUTED_COMMON_CHECKPOINT_NAME
    if destination.exists():
        raise FileExistsError(
            f"distributed common checkpoint already exists: '{destination}'"
        )
    common_payload = _common_checkpoint_payload(payload, paths=paths)
    CheckpointManager.save_payload(common_payload, destination)
    return _file_record(destination, root=paths.staging_directory)


def stage_distributed_best_portable_checkpoint(
    paths: DistributedCheckpointBundlePaths,
    *,
    selected_epoch: int,
    source_portable_checkpoint: str | Path | None = None,
    expected_source_sha256: str | None = None,
) -> DistributedCheckpointFile:
    """Stage the selected portable checkpoint inside the resume bundle.

    When ``source_portable_checkpoint`` is omitted, the selected epoch must be
    the bundle's current completed epoch and the portable state is projected
    from the staged common checkpoint.  Otherwise the source must already be a
    validated distributed portable checkpoint for the selected epoch.  The
    canonical attachment is optional, but a bundle that contains it owns all
    state required to preserve an earlier best selection after copying.
    """

    _validate_bundle_paths(paths)
    epoch = _positive_int(selected_epoch, path="selected_epoch")
    if epoch > paths.completed_epoch:
        raise ValueError("selected_epoch cannot be later than completed_epoch")
    destination = (
        paths.staging_directory / DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME
    )
    if destination.exists():
        raise FileExistsError(
            "distributed best portable checkpoint already exists: "
            f"'{destination}'"
        )

    if source_portable_checkpoint is None:
        if expected_source_sha256 is not None:
            raise ValueError(
                "a current-epoch best projection cannot declare a source digest"
            )
        if epoch != paths.completed_epoch:
            raise ValueError(
                "an earlier selected epoch requires a source portable checkpoint"
            )
        common_path = (
            paths.staging_directory / DISTRIBUTED_COMMON_CHECKPOINT_NAME
        )
        common_payload = CheckpointManager.load_payload(
            common_path,
            map_location="cpu",
        )
        _validate_common_checkpoint(
            common_payload,
            paths=paths,
            source=common_path,
        )
        attachment_source = {
            "bundle_id": paths.bundle_id,
            "common_sha256": _file_sha256(common_path),
            "completed_epoch": paths.completed_epoch,
            "global_step": paths.global_step,
        }
        portable = _project_portable_checkpoint(
            common_payload,
            committed_source=None,
            attachment_source=attachment_source,
        )
        _atomic_torch_save(cast(dict[str, object], portable), destination)
    else:
        source_path = Path(source_portable_checkpoint)
        expected_digest = _sha256_digest(
            expected_source_sha256,
            path="expected_source_sha256",
        )
        if _file_sha256(source_path) != expected_digest:
            raise ValueError("source portable checkpoint digest does not match")
        source_payload = CheckpointManager.load_payload(
            source_path,
            map_location="cpu",
        )
        _validate_distributed_portable_checkpoint(
            source_payload,
            selected_epoch=epoch,
            require_provenance=True,
            path=f"source portable checkpoint at '{source_path}'",
        )
        _atomic_copy_file(source_path, destination)
        if _file_sha256(destination) != expected_digest:
            destination.unlink(missing_ok=True)
            raise ValueError(
                "staged best portable checkpoint differs from its source authority"
            )

    staged_payload = CheckpointManager.load_payload(
        destination,
        map_location="cpu",
    )
    _validate_distributed_portable_checkpoint(
        staged_payload,
        selected_epoch=epoch,
        require_provenance=True,
        path=f"staged best portable checkpoint at '{destination}'",
    )
    return _file_record(destination, root=paths.staging_directory)


def stage_distributed_rank_checkpoint(
    paths: DistributedCheckpointBundlePaths,
    *,
    topology: DistributedTopology,
    next_plan: RankedTrainEpochPlan,
    rng_state: object | None = None,
    device: torch.device | str = "cpu",
) -> DistributedCheckpointFile:
    """Write one rank's exact RNG and freshly planned next-epoch data state."""

    _validate_bundle_paths(paths)
    _validate_next_plan(
        next_plan,
        rank=topology.rank,
        world_size=topology.world_size,
        completed_epoch=paths.completed_epoch,
        path="next_plan",
    )
    raw_rng_state = (
        capture_local_rng_state(device) if rng_state is None else rng_state
    )
    parse_rng_state(raw_rng_state)
    rank_payload = {
        "format_version": DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION,
        "kind": _RANK_LOCAL_KIND,
        "bundle_id": paths.bundle_id,
        "completed_epoch": paths.completed_epoch,
        "global_step": paths.global_step,
        "rank": topology.rank,
        "world_size": topology.world_size,
        "rng_state": deepcopy(raw_rng_state),
        "next_plan": _ranked_train_plan_to_dict(next_plan),
    }
    destination = paths.staging_directory / _rank_local_relative_path(
        topology.rank
    )
    if destination.exists():
        raise FileExistsError(
            f"distributed rank-local checkpoint already exists: '{destination}'"
        )
    _atomic_torch_save(rank_payload, destination)
    return _file_record(destination, root=paths.staging_directory)


def commit_distributed_checkpoint_bundle(
    paths: DistributedCheckpointBundlePaths,
    *,
    topology: DistributedTopology,
    backend: str,
    device_type: str,
) -> DistributedCheckpointBundle:
    """Validate every staged file and atomically publish the manifest last."""

    _validate_bundle_paths(paths)
    if not topology.is_primary:
        raise PermissionError(
            "only distributed rank zero may commit a checkpoint bundle"
        )
    validated_backend, validated_device_type = _runtime_pair(
        backend,
        device_type,
    )
    if paths.final_directory.exists():
        raise FileExistsError(
            f"distributed checkpoint bundle already exists: "
            f"'{paths.final_directory}'"
        )
    if not paths.staging_directory.is_dir():
        raise FileNotFoundError(
            f"distributed checkpoint staging directory is missing: "
            f"'{paths.staging_directory}'"
        )

    manifest_path = (
        paths.staging_directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    )
    if manifest_path.exists():
        raise FileExistsError(
            f"distributed checkpoint staging manifest already exists: "
            f"'{manifest_path}'"
        )
    try:
        common_path = (
            paths.staging_directory / DISTRIBUTED_COMMON_CHECKPOINT_NAME
        )
        common_payload = CheckpointManager.load_payload(
            common_path,
            map_location="cpu",
        )
        _validate_common_checkpoint(
            common_payload,
            paths=paths,
            source=common_path,
        )
        common_file = _file_record(common_path, root=paths.staging_directory)

        rank_files: list[dict[str, object]] = []
        plans: list[RankedTrainEpochPlan] = []
        for rank in range(topology.world_size):
            rank_path = paths.staging_directory / _rank_local_relative_path(
                rank
            )
            rank_payload = _load_rank_local_payload(rank_path)
            plan, _ = _validate_rank_local_payload(
                rank_payload,
                paths=paths,
                expected_rank=rank,
                expected_world_size=topology.world_size,
                expected_device_type=validated_device_type,
            )
            plans.append(plan)
            rank_file = _file_record(
                rank_path,
                root=paths.staging_directory,
            )
            rank_files.append({"rank": rank, **rank_file.to_dict()})
        _validate_cross_rank_plans(plans)

        best_portable: dict[str, object] | None = None
        best_portable_path = (
            paths.staging_directory
            / DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME
        )
        if best_portable_path.exists():
            if not best_portable_path.is_file():
                raise ValueError(
                    "distributed best portable checkpoint must be a regular file"
                )
            best_payload = CheckpointManager.load_payload(
                best_portable_path,
                map_location="cpu",
            )
            selected_epoch = _validate_distributed_portable_checkpoint(
                best_payload,
                selected_epoch=None,
                require_provenance=True,
                path=(
                    "staged best portable checkpoint at "
                    f"'{best_portable_path}'"
                ),
            )
            _validate_current_attachment_projection(
                best_payload,
                common_payload=common_payload,
                paths=paths,
                common_sha256=common_file.sha256,
                path=(
                    "staged best portable checkpoint at "
                    f"'{best_portable_path}'"
                ),
            )
            if selected_epoch > paths.completed_epoch:
                raise ValueError(
                    "distributed best portable selected epoch cannot be later "
                    "than completed_epoch"
                )
            best_file = _file_record(
                best_portable_path,
                root=paths.staging_directory,
            )
            best_portable = {
                "selected_epoch": selected_epoch,
                **best_file.to_dict(),
            }
        _validate_bundle_file_set(
            paths.staging_directory,
            world_size=topology.world_size,
            include_manifest=False,
            include_best_portable=best_portable is not None,
        )

        manifest = {
            "format_version": DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION,
            "kind": DISTRIBUTED_CHECKPOINT_BUNDLE_KIND,
            "status": "committed",
            "bundle_id": paths.bundle_id,
            "completed_epoch": paths.completed_epoch,
            "global_step": paths.global_step,
            "topology": {
                "single_node": True,
                "world_size": topology.world_size,
                "local_world_size": topology.local_world_size,
                "rank_mapping": [
                    {"rank": rank, "local_rank": rank}
                    for rank in range(topology.world_size)
                ],
                "backend": validated_backend,
                "device_type": validated_device_type,
            },
            "common": common_file.to_dict(),
            "best_portable": best_portable,
            "rank_local": rank_files,
        }
        write_yaml_manifest(manifest_path, manifest)
        _parse_manifest(manifest_path)
        paths.staging_directory.replace(paths.final_directory)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise

    return _bundle_from_manifest(paths.final_directory)


def load_distributed_checkpoint_bundle(
    bundle_directory: str | Path,
    *,
    topology: DistributedTopology,
    backend: str,
    device_type: str,
    fresh_next_plan: RankedTrainEpochPlan,
    expected_bundle: DistributedCheckpointBundle | None = None,
) -> DistributedCheckpointRestore:
    """Validate a committed fixed-topology bundle without mutating runtime state.

    ``fresh_next_plan`` must come from the current DataBuilder execution.  Its
    equality check binds resume to current data verification while keeping the
    runtime-only verification receipt out of every serialized bundle file.
    """

    validated_backend, validated_device_type = _runtime_pair(
        backend,
        device_type,
    )
    loaded = _load_committed_bundle(Path(bundle_directory))
    bundle = loaded.bundle
    if expected_bundle is not None and (
        bundle.bundle_id != expected_bundle.bundle_id
        or bundle.completed_epoch != expected_bundle.completed_epoch
        or bundle.global_step != expected_bundle.global_step
        or bundle.common_checkpoint_sha256
        != expected_bundle.common_checkpoint_sha256
        or not expected_bundle.manifest_sha256
        or _file_sha256(bundle.manifest_path) != expected_bundle.manifest_sha256
    ):
        raise ValueError(
            "distributed checkpoint changed after its initial preflight"
        )
    if (
        bundle.world_size != topology.world_size
        or bundle.local_world_size != topology.local_world_size
    ):
        raise ValueError(
            "distributed checkpoint topology does not match the current "
            "world/local-world size"
        )
    if bundle.backend != validated_backend:
        raise ValueError(
            "distributed checkpoint backend does not match the current runtime"
        )
    if bundle.device_type != validated_device_type:
        raise ValueError(
            "distributed checkpoint device type does not match the current runtime"
        )
    _validate_next_plan(
        fresh_next_plan,
        rank=topology.rank,
        world_size=topology.world_size,
        completed_epoch=bundle.completed_epoch,
        path="fresh_next_plan",
    )
    stored_plan = loaded.rank_plans[topology.rank]
    if _ranked_train_plan_to_dict(stored_plan) != _ranked_train_plan_to_dict(
        fresh_next_plan
    ):
        raise ValueError(
            "distributed checkpoint next data plan does not match the freshly "
            "verified runtime plan"
        )
    return DistributedCheckpointRestore(
        bundle=bundle,
        common_payload=loaded.common_payload,
        rank_rng_state=loaded.rank_rng_states[topology.rank],
        next_plan=fresh_next_plan,
    )


def preflight_distributed_checkpoint_bundle(
    bundle_directory: str | Path,
) -> DistributedCheckpointPreflight:
    """Read a complete bundle's common authority without enabling restore."""

    loaded = _load_committed_bundle(Path(bundle_directory))
    return DistributedCheckpointPreflight(
        bundle=loaded.bundle,
        common_payload=loaded.common_payload,
    )


def apply_distributed_checkpoint_restore(
    restore: DistributedCheckpointRestore,
    *,
    checkpoint_manager: CheckpointManager,
    local_device: torch.device | str = "cpu",
) -> LoadedCheckpoint:
    """Apply a previously validated common state and this rank's RNG state.

    Callers must first reach an all-rank success consensus for
    :func:`load_distributed_checkpoint_bundle`.  Common runtime objects use the
    existing transactional v12 restore.  If the subsequent RNG restore fails,
    this helper attempts to restore both the previous common state and RNG.
    """

    selected_device = torch.device(local_device)
    if selected_device.type != restore.bundle.device_type:
        raise ValueError(
            "distributed restore local device type does not match the bundle"
        )
    if selected_device.type == "cuda" and selected_device.index is None:
        raise ValueError("distributed restore requires an indexed local CUDA device")
    previous_rng_raw = capture_local_rng_state(selected_device)
    previous_payload = checkpoint_manager.build_state(rng_state=previous_rng_raw)
    previous_rng = parse_rng_state(
        previous_rng_raw,
    )
    loaded = checkpoint_manager.restore_payload(
        restore.common_payload,
        path=restore.bundle.common_checkpoint_path,
    )
    try:
        restore_local_rng_state(
            restore.rank_rng_state,
            device=selected_device,
        )
    except Exception as primary_error:
        cleanup_errors: list[str] = []
        try:
            checkpoint_manager.restore_payload(
                previous_payload,
                path="distributed-resume-runtime-rollback.pt",
            )
        except Exception as rollback_error:  # noqa: BLE001 - rollback is best effort
            cleanup_errors.append(
                f"common-state rollback failed: {rollback_error!r}"
            )
        try:
            restore_local_rng_state(
                previous_rng,
                device=selected_device,
            )
        except Exception as rollback_error:  # noqa: BLE001 - rollback is best effort
            cleanup_errors.append(f"RNG rollback failed: {rollback_error!r}")
        for note in cleanup_errors:
            primary_error.add_note(note)
        if cleanup_errors:
            raise RuntimeError(
                "distributed checkpoint restoration failed and runtime rollback "
                "did not complete; runtime state is poisoned"
            ) from primary_error
        raise
    return loaded


def export_distributed_portable_checkpoint(
    bundle_directory: str | Path,
    destination: str | Path,
) -> Path:
    """Retryably export one committed common state as a valid v12 checkpoint.

    The v12-required precision, scaler, and RNG fields remain present for format
    compatibility.  Metadata identifies this file as a portable projection, so
    it cannot stand in for the exact distributed resume bundle.
    """

    loaded = _load_committed_bundle(Path(bundle_directory))
    destination_path = Path(destination)
    source = {
        "bundle_id": loaded.bundle.bundle_id,
        "manifest_sha256": loaded.manifest_sha256,
        "common_sha256": loaded.bundle.common_checkpoint_sha256,
        "completed_epoch": loaded.bundle.completed_epoch,
        "global_step": loaded.bundle.global_step,
    }
    portable = _project_portable_checkpoint(
        loaded.common_payload,
        committed_source=source,
        attachment_source=None,
    )
    if destination_path.exists():
        existing = CheckpointManager.load_payload(
            destination_path,
            map_location="cpu",
        )
        if (
            _portable_source(existing) == source
            and _checkpoint_values_equal(existing, portable)
        ):
            return destination_path
        raise FileExistsError(
            "portable checkpoint destination contains a different snapshot or "
            f"is not the exact projection: '{destination_path}'"
        )
    CheckpointManager.save_payload(portable, destination_path)
    published = CheckpointManager.load_payload(
        destination_path,
        map_location="cpu",
    )
    if (
        _portable_source(published) != source
        or not _checkpoint_values_equal(published, portable)
    ):
        destination_path.unlink(missing_ok=True)
        raise RuntimeError(
            "published portable checkpoint does not match the expected projection"
        )
    return destination_path


def _project_portable_checkpoint(
    common_payload: CheckpointState,
    *,
    committed_source: dict[str, object] | None,
    attachment_source: dict[str, object] | None,
) -> CheckpointState:
    if (committed_source is None) == (attachment_source is None):
        raise ValueError(
            "portable projection requires exactly one provenance source"
        )
    portable = cast(CheckpointState, dict(common_payload))
    descriptors = validate_inference_asset_descriptors(
        portable.get("inference_asset_descriptors"),
        path="portable checkpoint.inference_asset_descriptors",
    )
    projected_asset_names = {
        descriptor["training_asset_name"] for descriptor in descriptors.values()
    }
    if projected_asset_names:
        raw_assets = cast(object, portable.get("training_assets_state_dict"))
        if type(raw_assets) is not dict:
            raise TypeError(
                "portable checkpoint inference assets require training asset state"
            )
        assets = cast(dict[str, dict[str, Any]], raw_assets)
        missing = projected_asset_names - set(assets)
        if missing:
            raise ValueError(
                "portable checkpoint is missing projected training assets: "
                f"{sorted(missing)}"
            )
        portable["training_assets_state_dict"] = {
            name: assets[name] for name in sorted(projected_asset_names)
        }
    else:
        portable.pop("training_assets_state_dict", None)
    for training_only_field in (
        "objective_state_dict",
        "optimizer_class",
        "optimizer_state_dict",
        "lr_scheduler_class",
        "lr_scheduler_state_dict",
    ):
        portable.pop(training_only_field, None)
    metadata = deepcopy(_checkpoint_metadata(portable))
    metadata["checkpoint_role"] = DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE
    metadata.pop("distributed_portable_source", None)
    metadata.pop(_PORTABLE_ATTACHMENT_SOURCE_KEY, None)
    if committed_source is not None:
        metadata["distributed_portable_source"] = deepcopy(committed_source)
    else:
        metadata[_PORTABLE_ATTACHMENT_SOURCE_KEY] = deepcopy(attachment_source)
    metadata["rng_authority"] = "v12_format_compatibility_only"
    portable["metadata"] = metadata
    validated = validate_checkpoint_payload(portable)
    _validate_distributed_portable_checkpoint(
        validated,
        selected_epoch=None,
        require_provenance=True,
        path="projected portable checkpoint",
    )
    return validated


def _validate_distributed_portable_checkpoint(
    payload: object,
    *,
    selected_epoch: int | None,
    require_provenance: bool,
    path: str,
) -> int:
    portable = validate_checkpoint_payload(payload)
    metadata = _checkpoint_metadata(portable)
    if metadata.get("checkpoint_role") != DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE:
        raise ValueError(f"{path} has the wrong checkpoint role")
    if metadata.get("rng_authority") != "v12_format_compatibility_only":
        raise ValueError(f"{path} has the wrong RNG authority")
    epoch = _positive_int(portable.get("epoch"), path=f"{path}.epoch")
    if selected_epoch is not None and epoch != selected_epoch:
        raise ValueError(f"{path} epoch does not match selected_epoch")

    training_only = {
        "objective_state_dict",
        "optimizer_class",
        "optimizer_state_dict",
        "lr_scheduler_class",
        "lr_scheduler_state_dict",
    }
    present_training_only = training_only.intersection(portable)
    if present_training_only:
        raise ValueError(
            f"{path} contains training-only fields: "
            f"{sorted(present_training_only)}"
        )
    descriptors = validate_inference_asset_descriptors(
        portable.get("inference_asset_descriptors"),
        path=f"{path}.inference_asset_descriptors",
    )
    expected_assets = {
        descriptor["training_asset_name"] for descriptor in descriptors.values()
    }
    assets_value = cast(object, portable.get("training_assets_state_dict"))
    if expected_assets:
        if type(assets_value) is not dict:
            raise TypeError(f"{path} is missing projected training asset state")
        actual_assets = set(cast(dict[object, object], assets_value))
        if actual_assets != expected_assets:
            raise ValueError(
                f"{path} training asset projection does not match descriptors"
            )
    elif "training_assets_state_dict" in portable:
        raise ValueError(f"{path} contains unreferenced training asset state")

    committed_value_present = "distributed_portable_source" in metadata
    attachment_value_present = _PORTABLE_ATTACHMENT_SOURCE_KEY in metadata
    if committed_value_present and attachment_value_present:
        raise ValueError(f"{path} declares more than one portable provenance")
    committed_source = _portable_source(portable)
    attachment_source = _portable_attachment_source(portable)
    if committed_value_present and committed_source is None:
        raise ValueError(f"{path} has invalid committed-bundle provenance")
    if attachment_value_present and attachment_source is None:
        raise ValueError(f"{path} has invalid attachment provenance")
    provenance = committed_source or attachment_source
    if require_provenance and provenance is None:
        raise ValueError(f"{path} has no distributed portable provenance")
    if provenance is not None:
        if provenance["completed_epoch"] != epoch:
            raise ValueError(f"{path} provenance epoch does not match payload epoch")
        if provenance["global_step"] != portable.get("global_step"):
            raise ValueError(
                f"{path} provenance global_step does not match payload global_step"
            )
    return epoch


def _validate_current_attachment_projection(
    payload: CheckpointState,
    *,
    common_payload: CheckpointState,
    paths: DistributedCheckpointBundlePaths,
    common_sha256: str,
    path: str,
) -> None:
    attachment_source = _portable_attachment_source(payload)
    if attachment_source is None:
        return
    if attachment_source["bundle_id"] != paths.bundle_id:
        return
    expected_source = {
        "bundle_id": paths.bundle_id,
        "common_sha256": common_sha256,
        "completed_epoch": paths.completed_epoch,
        "global_step": paths.global_step,
    }
    if attachment_source != expected_source:
        raise ValueError(f"{path} has the wrong current-bundle provenance")
    expected_payload = _project_portable_checkpoint(
        common_payload,
        committed_source=None,
        attachment_source=expected_source,
    )
    if not _checkpoint_values_equal(payload, expected_payload):
        raise ValueError(f"{path} does not match the staged common projection")


def _checkpoint_values_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if set(left) != set(right):
            return False
        return all(
            _checkpoint_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            _checkpoint_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return False
    if type(left) is not type(right):
        return False
    result = left == right
    return result if type(result) is bool else False


def _validate_bundle_paths(paths: object) -> None:
    if not isinstance(paths, DistributedCheckpointBundlePaths):
        raise TypeError("paths must be DistributedCheckpointBundlePaths")
    bundle_id = _bundle_id(paths.bundle_id, path="paths.bundle_id")
    _positive_int(paths.completed_epoch, path="paths.completed_epoch")
    _nonnegative_int(paths.global_step, path="paths.global_step")
    expected_staging = f".staging-{bundle_id}"
    expected_final = (
        f"epoch-{paths.completed_epoch:08d}-{bundle_id[:12]}"
    )
    if paths.staging_directory.name != expected_staging:
        raise ValueError("distributed checkpoint staging path is not canonical")
    if paths.final_directory.name != expected_final:
        raise ValueError("distributed checkpoint final path is not canonical")
    if (
        paths.staging_directory.parent.resolve()
        != paths.final_directory.parent.resolve()
    ):
        raise ValueError(
            "distributed checkpoint staging and final paths must share a parent"
        )
    if paths.final_directory.exists():
        raise FileExistsError(
            f"distributed checkpoint bundle already exists: "
            f"'{paths.final_directory}'"
        )


def _common_checkpoint_payload(
    payload: object,
    *,
    paths: DistributedCheckpointBundlePaths,
) -> CheckpointState:
    validated = validate_checkpoint_payload(payload)
    _checkpoint_progress(validated, paths=paths, source="common payload")
    common = cast(CheckpointState, dict(validated))
    metadata = deepcopy(_checkpoint_metadata(common))
    role = metadata.get("checkpoint_role")
    if role not in (None, DISTRIBUTED_COMMON_CHECKPOINT_ROLE):
        raise ValueError(
            "common checkpoint payload already has an incompatible checkpoint role"
        )
    metadata["checkpoint_role"] = DISTRIBUTED_COMMON_CHECKPOINT_ROLE
    metadata["distributed_bundle"] = {
        "bundle_id": paths.bundle_id,
        "completed_epoch": paths.completed_epoch,
        "global_step": paths.global_step,
    }
    metadata["rng_authority"] = "rank_local_bundle_files"
    common["metadata"] = metadata
    return validate_checkpoint_payload(common)


def _validate_common_checkpoint(
    payload: CheckpointState,
    *,
    paths: DistributedCheckpointBundlePaths,
    source: Path,
) -> None:
    _checkpoint_progress(payload, paths=paths, source=str(source))
    metadata = _checkpoint_metadata(payload)
    if metadata.get("checkpoint_role") != DISTRIBUTED_COMMON_CHECKPOINT_ROLE:
        raise ValueError(
            f"distributed common checkpoint at '{source}' has the wrong role"
        )
    binding = _exact_dict(
        metadata.get("distributed_bundle"),
        path=f"checkpoint at '{source}'.metadata.distributed_bundle",
        fields={"bundle_id", "completed_epoch", "global_step"},
    )
    parsed_binding = {
        "bundle_id": _bundle_id(
            binding["bundle_id"],
            path="checkpoint.metadata.distributed_bundle.bundle_id",
        ),
        "completed_epoch": _positive_int(
            binding["completed_epoch"],
            path="checkpoint.metadata.distributed_bundle.completed_epoch",
        ),
        "global_step": _nonnegative_int(
            binding["global_step"],
            path="checkpoint.metadata.distributed_bundle.global_step",
        ),
    }
    if parsed_binding != {
        "bundle_id": paths.bundle_id,
        "completed_epoch": paths.completed_epoch,
        "global_step": paths.global_step,
    }:
        raise ValueError(
            f"distributed common checkpoint at '{source}' has the wrong "
            "bundle binding"
        )
    if metadata.get("rng_authority") != "rank_local_bundle_files":
        raise ValueError(
            f"distributed common checkpoint at '{source}' has the wrong RNG "
            "authority"
        )
    runtime_digest = metadata.get("common_runtime_state_sha256")
    if (
        type(runtime_digest) is not str
        or len(runtime_digest) != 64
        or runtime_digest != runtime_digest.lower()
        or any(character not in "0123456789abcdef" for character in runtime_digest)
    ):
        raise ValueError(
            f"distributed common checkpoint at '{source}' is missing a valid "
            "common runtime state digest"
        )


def _checkpoint_progress(
    payload: CheckpointState,
    *,
    paths: DistributedCheckpointBundlePaths,
    source: str,
) -> None:
    epoch = payload.get("epoch")
    step = payload.get("global_step")
    if epoch != paths.completed_epoch or type(epoch) is not int:
        raise ValueError(
            f"{source} epoch does not match the completed distributed epoch"
        )
    if step != paths.global_step or type(step) is not int:
        raise ValueError(
            f"{source} global_step does not match the distributed sync point"
        )


def _ranked_train_plan_to_dict(
    plan: RankedTrainEpochPlan,
) -> dict[str, object]:
    return {
        "data_identity": plan.data_identity.to_dict(),
        "plan_digest": plan.plan_digest,
        "expected_terminal_token": plan.expected_terminal_token,
        "epoch": plan.epoch,
        "rank": plan.rank,
        "world_size": plan.world_size,
        "microbatches_per_window": plan.microbatches_per_window,
        "window_count": plan.window_count,
        "samples_per_microbatch": plan.samples_per_microbatch,
        "local_assigned_samples": plan.local_assigned_samples,
        "global_assigned_samples": plan.global_assigned_samples,
        "global_dropped_samples": plan.global_dropped_samples,
        "assignment_digest": plan.assignment_digest,
        "requested_max_microbatches": plan.requested_max_microbatches,
    }


def _ranked_train_plan_from_dict(
    value: object,
    *,
    path: str,
) -> RankedTrainEpochPlan:
    plan = _exact_dict(
        value,
        path=path,
        fields={
            "data_identity",
            "plan_digest",
            "expected_terminal_token",
            "epoch",
            "rank",
            "world_size",
            "microbatches_per_window",
            "window_count",
            "samples_per_microbatch",
            "local_assigned_samples",
            "global_assigned_samples",
            "global_dropped_samples",
            "assignment_digest",
            "requested_max_microbatches",
        },
    )
    identity_value = _exact_dict(
        plan["data_identity"],
        path=f"{path}.data_identity",
        fields={"provider", "digest"},
    )
    provider = _exact_string(
        identity_value["provider"],
        path=f"{path}.data_identity.provider",
    )
    digest = _sha256_digest(
        identity_value["digest"],
        path=f"{path}.data_identity.digest",
    )
    maximum_value = plan["requested_max_microbatches"]
    maximum = (
        None
        if maximum_value is None
        else _positive_int(
            maximum_value,
            path=f"{path}.requested_max_microbatches",
        )
    )
    return RankedTrainEpochPlan(
        data_identity=RankedEpochDataIdentity(provider=provider, digest=digest),
        plan_digest=_sha256_digest(
            plan["plan_digest"],
            path=f"{path}.plan_digest",
        ),
        expected_terminal_token=_sha256_digest(
            plan["expected_terminal_token"],
            path=f"{path}.expected_terminal_token",
        ),
        epoch=_nonnegative_int(plan["epoch"], path=f"{path}.epoch"),
        rank=_nonnegative_int(plan["rank"], path=f"{path}.rank"),
        world_size=_positive_int(
            plan["world_size"],
            path=f"{path}.world_size",
        ),
        microbatches_per_window=_positive_int(
            plan["microbatches_per_window"],
            path=f"{path}.microbatches_per_window",
        ),
        window_count=_positive_int(
            plan["window_count"],
            path=f"{path}.window_count",
        ),
        samples_per_microbatch=_positive_int(
            plan["samples_per_microbatch"],
            path=f"{path}.samples_per_microbatch",
        ),
        local_assigned_samples=_positive_int(
            plan["local_assigned_samples"],
            path=f"{path}.local_assigned_samples",
        ),
        global_assigned_samples=_positive_int(
            plan["global_assigned_samples"],
            path=f"{path}.global_assigned_samples",
        ),
        global_dropped_samples=_nonnegative_int(
            plan["global_dropped_samples"],
            path=f"{path}.global_dropped_samples",
        ),
        assignment_digest=_sha256_digest(
            plan["assignment_digest"],
            path=f"{path}.assignment_digest",
        ),
        requested_max_microbatches=maximum,
    )


def _validate_next_plan(
    plan: object,
    *,
    rank: int,
    world_size: int,
    completed_epoch: int,
    path: str,
) -> None:
    if not isinstance(plan, RankedTrainEpochPlan):
        raise TypeError(f"{path} must be RankedTrainEpochPlan")
    if plan.rank != rank or plan.world_size != world_size:
        raise ValueError(f"{path} rank/world_size does not match runtime topology")
    if plan.epoch != completed_epoch + 1:
        raise ValueError(
            f"{path} must describe the epoch immediately after the committed "
            "boundary"
        )


def _validate_cross_rank_plans(plans: list[RankedTrainEpochPlan]) -> None:
    if not plans:
        raise ValueError("distributed checkpoint requires rank-local plans")
    expected_ranks = set(range(len(plans)))
    actual_ranks = {plan.rank for plan in plans}
    if actual_ranks != expected_ranks:
        raise ValueError("distributed checkpoint rank plans are incomplete")
    baseline = _ranked_train_plan_to_dict(plans[0])
    baseline.pop("rank")
    baseline.pop("expected_terminal_token")
    baseline.pop("assignment_digest")
    for plan in plans[1:]:
        candidate = _ranked_train_plan_to_dict(plan)
        candidate.pop("rank")
        candidate.pop("expected_terminal_token")
        candidate.pop("assignment_digest")
        if candidate != baseline:
            raise ValueError(
                "distributed checkpoint rank-local next plans do not describe "
                "one common data assignment"
            )


def _rank_local_relative_path(rank: int) -> Path:
    return Path(_RANK_LOCAL_DIRECTORY) / f"rank-{rank:05d}.pt"


def _atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as source_stream, os.fdopen(
            descriptor,
            "wb",
        ) as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        temporary_path.replace(destination)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_rank_local_payload(path: Path) -> dict[str, object]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return _exact_dict(
        raw,
        path=f"rank-local checkpoint at '{path}'",
        fields={
            "format_version",
            "kind",
            "bundle_id",
            "completed_epoch",
            "global_step",
            "rank",
            "world_size",
            "rng_state",
            "next_plan",
        },
    )


def _validate_rank_local_payload(
    payload: dict[str, object],
    *,
    paths: DistributedCheckpointBundlePaths,
    expected_rank: int,
    expected_world_size: int,
    expected_device_type: str,
) -> tuple[RankedTrainEpochPlan, ParsedRNGState]:
    version = _nonnegative_int(
        payload["format_version"],
        path="rank-local checkpoint format_version",
    )
    if version != DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION:
        raise ValueError("rank-local checkpoint format version is unsupported")
    if _exact_string(payload["kind"], path="rank-local checkpoint kind") != (
        _RANK_LOCAL_KIND
    ):
        raise ValueError("rank-local checkpoint has the wrong kind")
    actual = {
        "bundle_id": _bundle_id(
            payload["bundle_id"],
            path="rank-local checkpoint bundle_id",
        ),
        "completed_epoch": _positive_int(
            payload["completed_epoch"],
            path="rank-local checkpoint completed_epoch",
        ),
        "global_step": _nonnegative_int(
            payload["global_step"],
            path="rank-local checkpoint global_step",
        ),
        "rank": _nonnegative_int(
            payload["rank"],
            path="rank-local checkpoint rank",
        ),
        "world_size": _positive_int(
            payload["world_size"],
            path="rank-local checkpoint world_size",
        ),
    }
    if actual != {
        "bundle_id": paths.bundle_id,
        "completed_epoch": paths.completed_epoch,
        "global_step": paths.global_step,
        "rank": expected_rank,
        "world_size": expected_world_size,
    }:
        raise ValueError("rank-local checkpoint has the wrong bundle/topology binding")
    plan = _ranked_train_plan_from_dict(
        payload["next_plan"],
        path=f"rank_local[{expected_rank}].next_plan",
    )
    _validate_next_plan(
        plan,
        rank=expected_rank,
        world_size=expected_world_size,
        completed_epoch=paths.completed_epoch,
        path=f"rank_local[{expected_rank}].next_plan",
    )
    rng = parse_rng_state(payload["rng_state"])
    if expected_device_type == "cuda":
        if len(rng.torch_cuda) != 1 or rng.torch_mps is not None:
            raise ValueError(
                "rank-local CUDA checkpoint must contain exactly one CUDA "
                "RNG stream and no MPS stream"
            )
    elif expected_device_type == "cpu":
        if not rng.torch_cuda and rng.torch_mps is None:
            return plan, rng
        raise ValueError(
            "rank-local CPU checkpoint must not contain accelerator RNG streams"
        )
    else:
        raise ValueError(
            "rank-local checkpoint has an unsupported declared device type"
        )
    return plan, rng


def _file_record(path: Path, *, root: Path) -> DistributedCheckpointFile:
    relative = path.relative_to(root).as_posix()
    return DistributedCheckpointFile(
        relative_path=relative,
        size=path.stat().st_size,
        sha256=_file_sha256(path),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_manifest(path: Path) -> dict[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"distributed checkpoint manifest at '{path}' is invalid YAML"
        ) from exc
    manifest = _exact_dict(
        raw,
        path=f"distributed checkpoint manifest at '{path}'",
        fields={
            "format_version",
            "kind",
            "status",
            "bundle_id",
            "completed_epoch",
            "global_step",
            "topology",
            "common",
            "best_portable",
            "rank_local",
        },
    )
    version = _nonnegative_int(
        manifest["format_version"],
        path="manifest.format_version",
    )
    if version != DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION:
        raise ValueError("distributed checkpoint bundle format version is unsupported")
    if _exact_string(manifest["kind"], path="manifest.kind") != (
        DISTRIBUTED_CHECKPOINT_BUNDLE_KIND
    ):
        raise ValueError("distributed checkpoint manifest has the wrong kind")
    if _exact_string(manifest["status"], path="manifest.status") != "committed":
        raise ValueError("distributed checkpoint manifest is not committed")
    _bundle_id(manifest["bundle_id"], path="manifest.bundle_id")
    _positive_int(
        manifest["completed_epoch"],
        path="manifest.completed_epoch",
    )
    _nonnegative_int(manifest["global_step"], path="manifest.global_step")
    return manifest


def _bundle_from_manifest(directory: Path) -> DistributedCheckpointBundle:
    manifest_path = directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    manifest = _parse_manifest(manifest_path)
    topology = _parse_manifest_topology(manifest["topology"])
    common = _parse_file_record(
        manifest["common"],
        path="manifest.common",
        expected_path=DISTRIBUTED_COMMON_CHECKPOINT_NAME,
    )
    best = _parse_best_portable_record(
        manifest["best_portable"],
        completed_epoch=cast(int, manifest["completed_epoch"]),
    )
    return DistributedCheckpointBundle(
        directory=directory,
        manifest_path=manifest_path,
        bundle_id=cast(str, manifest["bundle_id"]),
        completed_epoch=cast(int, manifest["completed_epoch"]),
        global_step=cast(int, manifest["global_step"]),
        world_size=cast(int, topology["world_size"]),
        local_world_size=cast(int, topology["local_world_size"]),
        backend=cast(str, topology["backend"]),
        device_type=cast(str, topology["device_type"]),
        common_checkpoint_path=directory / common.relative_path,
        common_checkpoint_sha256=common.sha256,
        manifest_sha256=_file_sha256(manifest_path),
        best_portable_checkpoint_path=(
            directory / best[1].relative_path if best is not None else None
        ),
        best_portable_selected_epoch=(best[0] if best is not None else None),
        best_portable_checkpoint_sha256=(
            best[1].sha256 if best is not None else None
        ),
    )


def _parse_manifest_topology(value: object) -> dict[str, object]:
    topology = _exact_dict(
        value,
        path="manifest.topology",
        fields={
            "single_node",
            "world_size",
            "local_world_size",
            "rank_mapping",
            "backend",
            "device_type",
        },
    )
    if topology["single_node"] is not True:
        raise ValueError("distributed checkpoint topology must be single-node")
    world_size = _positive_int(
        topology["world_size"],
        path="manifest.topology.world_size",
    )
    local_world_size = _positive_int(
        topology["local_world_size"],
        path="manifest.topology.local_world_size",
    )
    if world_size != local_world_size:
        raise ValueError(
            "distributed checkpoint world_size must equal local_world_size"
        )
    backend, device_type = _runtime_pair(
        topology["backend"],
        topology["device_type"],
    )
    rank_mapping = topology["rank_mapping"]
    if type(rank_mapping) is not list:
        raise TypeError("manifest.topology.rank_mapping must be a list")
    expected_mapping = [
        {"rank": rank, "local_rank": rank} for rank in range(world_size)
    ]
    parsed_mapping: list[dict[str, int]] = []
    for index, value in enumerate(cast(list[object], rank_mapping)):
        raw_mapping = _exact_dict(
            value,
            path=f"manifest.topology.rank_mapping[{index}]",
            fields={"rank", "local_rank"},
        )
        parsed_mapping.append(
            {
                "rank": _nonnegative_int(
                    raw_mapping["rank"],
                    path=f"manifest.topology.rank_mapping[{index}].rank",
                ),
                "local_rank": _nonnegative_int(
                    raw_mapping["local_rank"],
                    path=f"manifest.topology.rank_mapping[{index}].local_rank",
                ),
            }
        )
    if parsed_mapping != expected_mapping:
        raise ValueError(
            "distributed checkpoint rank mapping is not the fixed single-node "
            "mapping"
        )
    return {
        "single_node": True,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "rank_mapping": parsed_mapping,
        "backend": backend,
        "device_type": device_type,
    }


def _parse_file_record(
    value: object,
    *,
    path: str,
    expected_path: str,
) -> DistributedCheckpointFile:
    record = _exact_dict(
        value,
        path=path,
        fields={"path", "size", "sha256"},
    )
    relative_path = _exact_string(record["path"], path=f"{path}.path")
    if relative_path != expected_path:
        raise ValueError(f"{path}.path is not the canonical bundle path")
    size = _nonnegative_int(record["size"], path=f"{path}.size")
    digest = _sha256_digest(record["sha256"], path=f"{path}.sha256")
    return DistributedCheckpointFile(relative_path, size, digest)


def _parse_best_portable_record(
    value: object,
    *,
    completed_epoch: int,
) -> tuple[int, DistributedCheckpointFile] | None:
    if value is None:
        return None
    raw = _exact_dict(
        value,
        path="manifest.best_portable",
        fields={"selected_epoch", "path", "size", "sha256"},
    )
    selected_epoch = _positive_int(
        raw["selected_epoch"],
        path="manifest.best_portable.selected_epoch",
    )
    if selected_epoch > completed_epoch:
        raise ValueError(
            "manifest.best_portable.selected_epoch cannot be later than "
            "manifest.completed_epoch"
        )
    record = _parse_file_record(
        {
            "path": raw["path"],
            "size": raw["size"],
            "sha256": raw["sha256"],
        },
        path="manifest.best_portable",
        expected_path=DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME,
    )
    return selected_epoch, record


def _load_committed_bundle(directory: Path) -> LoadedDistributedBundle:
    if directory.name.startswith(".staging-"):
        raise ValueError("hidden distributed checkpoint staging is not resumable")
    if not directory.is_dir():
        raise FileNotFoundError(
            f"distributed checkpoint bundle is missing: '{directory}'"
        )
    manifest_path = directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    manifest = _parse_manifest(manifest_path)
    topology = _parse_manifest_topology(manifest["topology"])
    bundle_id = cast(str, manifest["bundle_id"])
    completed_epoch = cast(int, manifest["completed_epoch"])
    global_step = cast(int, manifest["global_step"])
    paths = DistributedCheckpointBundlePaths(
        bundle_id=bundle_id,
        completed_epoch=completed_epoch,
        global_step=global_step,
        staging_directory=directory.parent / f".staging-{bundle_id}",
        final_directory=directory,
    )

    common_record = _parse_file_record(
        manifest["common"],
        path="manifest.common",
        expected_path=DISTRIBUTED_COMMON_CHECKPOINT_NAME,
    )
    _verify_file_record(directory, common_record, path="manifest.common")
    common_path = directory / common_record.relative_path
    common_payload = CheckpointManager.load_payload(
        common_path,
        map_location="cpu",
    )
    _validate_common_checkpoint(
        common_payload,
        paths=paths,
        source=common_path,
    )

    best = _parse_best_portable_record(
        manifest["best_portable"],
        completed_epoch=completed_epoch,
    )
    best_path: Path | None = None
    if best is not None:
        selected_epoch, best_record = best
        _verify_file_record(
            directory,
            best_record,
            path="manifest.best_portable",
        )
        best_path = directory / best_record.relative_path
        best_payload = CheckpointManager.load_payload(
            best_path,
            map_location="cpu",
        )
        _validate_distributed_portable_checkpoint(
            best_payload,
            selected_epoch=selected_epoch,
            require_provenance=True,
            path=f"best portable checkpoint at '{best_path}'",
        )
        _validate_current_attachment_projection(
            best_payload,
            common_payload=common_payload,
            paths=paths,
            common_sha256=common_record.sha256,
            path=f"best portable checkpoint at '{best_path}'",
        )

    rank_records_value = manifest["rank_local"]
    if type(rank_records_value) is not list:
        raise TypeError("manifest.rank_local must be a list")
    rank_records = cast(list[object], rank_records_value)
    world_size = cast(int, topology["world_size"])
    if len(rank_records) != world_size:
        raise ValueError(
            "distributed checkpoint rank-local inventory does not match world_size"
        )
    rank_plans: dict[int, RankedTrainEpochPlan] = {}
    rank_rng_states: dict[int, ParsedRNGState] = {}
    for expected_rank, value in enumerate(rank_records):
        raw_record = _exact_dict(
            value,
            path=f"manifest.rank_local[{expected_rank}]",
            fields={"rank", "path", "size", "sha256"},
        )
        rank = _nonnegative_int(
            raw_record["rank"],
            path=f"manifest.rank_local[{expected_rank}].rank",
        )
        if rank != expected_rank:
            raise ValueError("distributed checkpoint rank inventory is not ordered")
        record = _parse_file_record(
            {
                "path": raw_record["path"],
                "size": raw_record["size"],
                "sha256": raw_record["sha256"],
            },
            path=f"manifest.rank_local[{rank}]",
            expected_path=_rank_local_relative_path(rank).as_posix(),
        )
        _verify_file_record(
            directory,
            record,
            path=f"manifest.rank_local[{rank}]",
        )
        rank_payload = _load_rank_local_payload(
            directory / record.relative_path
        )
        plan, rng = _validate_rank_local_payload(
            rank_payload,
            paths=paths,
            expected_rank=rank,
            expected_world_size=world_size,
            expected_device_type=cast(str, topology["device_type"]),
        )
        rank_plans[rank] = plan
        rank_rng_states[rank] = rng
    _validate_cross_rank_plans(list(rank_plans.values()))
    _validate_bundle_file_set(
        directory,
        world_size=world_size,
        include_manifest=True,
        include_best_portable=best is not None,
    )

    bundle = DistributedCheckpointBundle(
        directory=directory,
        manifest_path=manifest_path,
        bundle_id=bundle_id,
        completed_epoch=completed_epoch,
        global_step=global_step,
        world_size=world_size,
        local_world_size=cast(int, topology["local_world_size"]),
        backend=cast(str, topology["backend"]),
        device_type=cast(str, topology["device_type"]),
        common_checkpoint_path=common_path,
        common_checkpoint_sha256=common_record.sha256,
        manifest_sha256=_file_sha256(manifest_path),
        best_portable_checkpoint_path=best_path,
        best_portable_selected_epoch=(best[0] if best is not None else None),
        best_portable_checkpoint_sha256=(
            best[1].sha256 if best is not None else None
        ),
    )
    return LoadedDistributedBundle(
        bundle=bundle,
        common_payload=common_payload,
        rank_plans=rank_plans,
        rank_rng_states=rank_rng_states,
        manifest_sha256=_file_sha256(manifest_path),
    )


def _verify_file_record(
    directory: Path,
    record: DistributedCheckpointFile,
    *,
    path: str,
) -> None:
    candidate = directory / record.relative_path
    if not candidate.is_file():
        raise FileNotFoundError(f"{path} file is missing: '{candidate}'")
    if candidate.stat().st_size != record.size:
        raise ValueError(f"{path} file size does not match the manifest")
    if _file_sha256(candidate) != record.sha256:
        raise ValueError(f"{path} file digest does not match the manifest")


def _validate_bundle_file_set(
    directory: Path,
    *,
    world_size: int,
    include_manifest: bool,
    include_best_portable: bool,
) -> None:
    expected = {DISTRIBUTED_COMMON_CHECKPOINT_NAME}
    if include_best_portable:
        expected.add(DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME)
    expected.update(
        _rank_local_relative_path(rank).as_posix()
        for rank in range(world_size)
    )
    if include_manifest:
        expected.add(DISTRIBUTED_CHECKPOINT_MANIFEST_NAME)
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError(
            "distributed checkpoint bundle has an invalid file inventory: "
            f"missing={sorted(expected - actual) or '<none>'}, "
            f"unknown={sorted(actual - expected) or '<none>'}"
        )


def _portable_source(payload: CheckpointState) -> dict[str, object] | None:
    metadata = _checkpoint_metadata(payload)
    if metadata.get("checkpoint_role") != DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE:
        return None
    value = metadata.get("distributed_portable_source")
    if type(value) is not dict:
        return None
    source = cast(dict[object, object], value)
    if set(source) != {
        "bundle_id",
        "manifest_sha256",
        "common_sha256",
        "completed_epoch",
        "global_step",
    }:
        return None
    if any(type(key) is not str for key in source):
        return None
    parsed = cast(dict[str, object], source)
    try:
        return {
            "bundle_id": _bundle_id(
                parsed["bundle_id"],
                path="portable source bundle_id",
            ),
            "manifest_sha256": _sha256_digest(
                parsed["manifest_sha256"],
                path="portable source manifest_sha256",
            ),
            "common_sha256": _sha256_digest(
                parsed["common_sha256"],
                path="portable source common_sha256",
            ),
            "completed_epoch": _positive_int(
                parsed["completed_epoch"],
                path="portable source completed_epoch",
            ),
            "global_step": _nonnegative_int(
                parsed["global_step"],
                path="portable source global_step",
            ),
        }
    except (TypeError, ValueError):
        return None


def _portable_attachment_source(
    payload: CheckpointState,
) -> dict[str, object] | None:
    metadata = _checkpoint_metadata(payload)
    value = metadata.get(_PORTABLE_ATTACHMENT_SOURCE_KEY)
    if type(value) is not dict:
        return None
    source = cast(dict[object, object], value)
    if set(source) != {
        "bundle_id",
        "common_sha256",
        "completed_epoch",
        "global_step",
    }:
        return None
    if any(type(key) is not str for key in source):
        return None
    parsed = cast(dict[str, object], source)
    try:
        return {
            "bundle_id": _bundle_id(
                parsed["bundle_id"],
                path="portable attachment source bundle_id",
            ),
            "common_sha256": _sha256_digest(
                parsed["common_sha256"],
                path="portable attachment source common_sha256",
            ),
            "completed_epoch": _positive_int(
                parsed["completed_epoch"],
                path="portable attachment source completed_epoch",
            ),
            "global_step": _nonnegative_int(
                parsed["global_step"],
                path="portable attachment source global_step",
            ),
        }
    except (TypeError, ValueError):
        return None


def _checkpoint_metadata(payload: CheckpointState) -> dict[str, Any]:
    value = cast(object, payload.get("metadata"))
    if type(value) is not dict:
        raise TypeError("checkpoint metadata must be an exact dictionary")
    metadata = cast(dict[object, object], value)
    if any(type(key) is not str for key in metadata):
        raise TypeError("checkpoint metadata field names must be exact strings")
    return cast(dict[str, Any], metadata)


def _runtime_pair(backend: object, device_type: object) -> tuple[str, str]:
    backend_name = _exact_string(backend, path="distributed backend")
    device_name = _exact_string(device_type, path="distributed device type")
    pair = (backend_name, device_name)
    if pair not in _SUPPORTED_RUNTIME_PAIRS:
        raise ValueError(
            "distributed checkpoint runtime must be CPU/Gloo or CUDA/NCCL"
        )
    return pair


def _exact_dict(
    value: object,
    *,
    path: str,
    fields: set[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dictionary")
    raw = cast(dict[object, object], value)
    if any(type(key) is not str for key in raw):
        raise TypeError(f"{path} field names must be exact strings")
    actual = set(cast(dict[str, object], raw))
    if actual != fields:
        raise ValueError(
            f"{path} has invalid fields: missing={sorted(fields - actual) or '<none>'}, "
            f"unknown={sorted(actual - fields) or '<none>'}"
        )
    return cast(dict[str, object], raw)


def _exact_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result or result != result.strip():
        raise ValueError(f"{path} must be non-empty without surrounding whitespace")
    return result


def _nonnegative_int(value: object, *, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an exact integer")
    result = cast(int, value)
    if result < 0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _positive_int(value: object, *, path: str) -> int:
    result = _nonnegative_int(value, path=path)
    if result == 0:
        raise ValueError(f"{path} must be positive")
    return result


def _sha256_digest(value: object, *, path: str) -> str:
    result = _exact_string(value, path=path)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _bundle_id(value: object, *, path: str) -> str:
    result = _exact_string(value, path=path)
    if (
        len(result) != 32
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{path} must be 32 lowercase hexadecimal characters")
    return result


__all__ = [
    "DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME",
    "DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION",
    "DISTRIBUTED_CHECKPOINT_BUNDLE_KIND",
    "DISTRIBUTED_CHECKPOINT_MANIFEST_NAME",
    "DISTRIBUTED_COMMON_CHECKPOINT_NAME",
    "DISTRIBUTED_COMMON_CHECKPOINT_ROLE",
    "DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE",
    "DistributedCheckpointBundle",
    "DistributedCheckpointBundlePaths",
    "DistributedCheckpointFile",
    "DistributedCheckpointPreflight",
    "DistributedCheckpointRestore",
    "apply_distributed_checkpoint_restore",
    "commit_distributed_checkpoint_bundle",
    "distributed_checkpoint_bundle_paths",
    "export_distributed_portable_checkpoint",
    "load_distributed_checkpoint_bundle",
    "new_distributed_checkpoint_bundle_id",
    "preflight_distributed_checkpoint_bundle",
    "stage_distributed_best_portable_checkpoint",
    "stage_distributed_common_checkpoint",
    "stage_distributed_rank_checkpoint",
]
