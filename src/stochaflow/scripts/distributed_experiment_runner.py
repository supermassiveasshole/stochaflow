"""Fixed single-node DDP training operation selected by ``train --ddp``."""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from collections.abc import Callable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from stochaflow._builtin_activation import activate_training_builtins
from stochaflow.data import (
    DataArtifactBindings,
    DataLoaders,
    RankedDataExecution,
    RankedTrainEpochPlan,
    RankedTrainExecution,
    build_data_loaders,
)
from stochaflow.scripts import experiment_runner as single_runner
from stochaflow.scripts.artifact_reporting import RichArtifactVerificationReporter
from stochaflow.training.composition import build_logger
from stochaflow.training.distributed.checkpoint_bundle import (
    DistributedCheckpointBundle,
    DistributedCheckpointBundlePaths,
    apply_distributed_checkpoint_restore,
    commit_distributed_checkpoint_bundle,
    distributed_checkpoint_bundle_paths,
    export_distributed_portable_checkpoint,
    load_distributed_checkpoint_bundle,
    new_distributed_checkpoint_bundle_id,
    preflight_distributed_checkpoint_bundle,
    stage_distributed_best_portable_checkpoint,
    stage_distributed_common_checkpoint,
    stage_distributed_rank_checkpoint,
)
from stochaflow.training.distributed.composition import (
    DDPTrainingComponents,
    build_ddp_training_components,
)
from stochaflow.training.distributed.contracts import DistributedTopology
from stochaflow.training.distributed.session import DistributedSession
from stochaflow.training.outcome import TrainingRunOutcome
from stochaflow.training.trainer import MonitorPolicy, TrainingFitState
from stochaflow.utils.checkpoint import CheckpointState, capture_local_rng_state
from stochaflow.utils.config import StochaflowConfig, load_config
from stochaflow.utils.logging_contracts import ExperimentLogger, NullLogger
from stochaflow.utils.plugins import (
    ExtensionSelectionPolicy,
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
    prepare_extension_plugins,
)
from stochaflow.utils.run_manifest import (
    extension_runtime_metadata,
    selected_training_component_identities,
    write_yaml_manifest,
)
from stochaflow.utils.seed import set_cpu_seed, set_local_seed

DistributedSessionFactory = Callable[[], DistributedSession]
DistributedComponentFactory = Callable[
    [StochaflowConfig, DistributedSession],
    DDPTrainingComponents,
]


@dataclass(frozen=True, slots=True)
class DistributedTrainingInputs:
    """Resolved DDP config and optional exact bundle resume authority."""

    config: StochaflowConfig
    extensions: ResolvedExtensions
    config_source: str
    resume_bundle: DistributedCheckpointBundle | None
    resume_payload: CheckpointState | None
    startup_cwd: Path
    config_overlays: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class DistributedRunPaths:
    """One rank-zero-owned hidden workspace and absent final run path."""

    run_id: str
    workspace: Path
    final_directory: Path


@dataclass(slots=True)
class DistributedSelection:
    """Validation-only selection and early-stopping state for the DDP loop."""

    best_epoch: int | None
    best_metric_value: float | None
    observations_without_improvement: int
    monitor_observations: int
    stopped_early: bool
    policy: MonitorPolicy | None
    patience: int | None
    best_portable_relative_path: str | None = None
    best_portable_sha256: str | None = None

    def observe(self, metrics: dict[str, float], *, epoch: int) -> bool:
        """Apply one globally broadcast validation observation."""

        if self.policy is None:
            return False
        value = metrics.get(self.policy.metric)
        if value is None:
            raise ValueError(
                f"fixed DDP monitor {self.policy.metric!r} was not produced"
            )
        if not math.isfinite(value):
            raise ValueError(f"fixed DDP monitor {self.policy.metric!r} is non-finite")
        self.monitor_observations += 1
        improved = self.best_metric_value is None or (
            value < self.best_metric_value - self.policy.min_delta
            if self.policy.mode == "min"
            else value > self.best_metric_value + self.policy.min_delta
        )
        if improved:
            self.best_epoch = epoch
            self.best_metric_value = value
            self.observations_without_improvement = 0
            self.best_portable_relative_path = None
            self.best_portable_sha256 = None
        else:
            self.observations_without_improvement += 1
        self.stopped_early = (
            self.patience is not None
            and self.observations_without_improvement >= self.patience
        )
        return improved

    def fit_state(self) -> TrainingFitState:
        """Project current selection facts into the existing strict schema."""

        return TrainingFitState(
            best_epoch=self.best_epoch,
            best_metric_value=self.best_metric_value,
            observations_without_improvement=self.observations_without_improvement,
            monitor_observations=self.monitor_observations,
            stopped_early=self.stopped_early,
            tracking_enabled=self.policy is not None,
            monitor_policy=self.policy,
            early_stopping_patience=self.patience,
        )


def _default_session_factory() -> DistributedSession:
    return DistributedSession.from_environment(backend="nccl")


def _failure_summary(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    try:
        message = str(error)
    except BaseException:  # noqa: BLE001 - reporting cannot replace the cause
        message = "<exception text could not be rendered>"
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message[:1000],
    }


def _add_failure_note(error: BaseException, note: str) -> None:
    with suppress(BaseException):
        BaseException.add_note(error, note)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _resume_runtime_state_digest(payload: CheckpointState) -> str:
    metadata = cast(object, payload.get("metadata"))
    if type(metadata) is not dict:
        raise TypeError("distributed resume metadata must be an exact dictionary")
    value = cast(dict[object, object], metadata).get(
        "common_runtime_state_sha256"
    )
    if not _is_sha256(value):
        raise ValueError(
            "distributed resume is missing its common runtime state digest"
        )
    return cast(str, value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distributed_runtime_seed(base_seed: int, *, rank: int) -> int:
    """Derive one stable NumPy-compatible runtime seed for a DDP rank."""

    if type(base_seed) is not int:
        raise TypeError("distributed base seed must be an integer")
    if type(rank) is not int or rank < 0:
        raise ValueError("distributed rank must be a non-negative integer")
    digest = hashlib.sha256()
    digest.update(b"stochaflow.fixed-ddp.runtime-seed.v1\0")
    digest.update(str(base_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(rank).encode("ascii"))
    return int.from_bytes(digest.digest()[:4], byteorder="big")


def _committed_bundle_exists(workspace: Path) -> bool:
    resume_root = workspace / "checkpoints" / "resume"
    if not resume_root.is_dir():
        return False
    for candidate in resume_root.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith("epoch-"):
            continue
        try:
            preflight_distributed_checkpoint_bundle(candidate)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        return True
    return False


def _handle_failed_workspace(
    paths: DistributedRunPaths,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    error: BaseException,
) -> None:
    """Preserve recoverable bundles; remove a workspace with no commit."""

    if not paths.workspace.exists():
        return
    if (
        paths.workspace.parent != paths.final_directory.parent
        or paths.workspace.name != f".staging-ddp-{paths.run_id}"
        or paths.final_directory.name != paths.run_id
    ):
        raise RuntimeError("refusing to clean a non-canonical DDP workspace")
    if _committed_bundle_exists(paths.workspace):
        manifest["status"] = "failed"
        manifest["failure"] = _failure_summary(error)
        write_yaml_manifest(manifest_path, manifest)
        return
    shutil.rmtree(paths.workspace)


def _all_rank_stage[DistributedStageResult](
    session: DistributedSession,
    *,
    phase: str,
    action: Callable[[], DistributedStageResult],
) -> DistributedStageResult:
    """Run one local action and fail every rank before the next phase."""

    local_error: BaseException | None = None
    result: DistributedStageResult | None = None
    try:
        result = action()
    except BaseException as error:  # noqa: BLE001 - preserve local traceback
        local_error = error
    try:
        succeeded = session.collectives.all_true(local_error is None)
    except BaseException as collective_error:
        if local_error is None:
            raise
        _add_failure_note(
            local_error,
            f"fixed DDP {phase} failure consensus also failed: "
            f"{_failure_summary(collective_error)}"
        )
        raise local_error from collective_error
    if succeeded:
        if local_error is not None:
            raise RuntimeError(
                f"fixed DDP {phase} reported impossible success consensus"
            ) from local_error
        return cast(DistributedStageResult, result)

    try:
        gathered = session.collectives.gather_to_primary(
            _failure_summary(local_error)
        )
        summary: object = None
        if session.is_primary:
            assert gathered is not None
            summary = {
                "phase": phase,
                "failures": [
                    {"rank": rank, "error": value}
                    for rank, value in enumerate(gathered)
                    if value is not None
                ],
            }
        summary = session.collectives.broadcast_from_primary(summary)
    except BaseException as collective_error:
        if local_error is None:
            raise
        _add_failure_note(
            local_error,
            f"fixed DDP {phase} failure reporting also failed: "
            f"{_failure_summary(collective_error)}",
        )
        raise local_error from collective_error
    if local_error is not None:
        _add_failure_note(
            local_error,
            f"fixed DDP failure summary: {summary!r}",
        )
        raise local_error
    raise RuntimeError(f"fixed DDP {phase} failed on another rank: {summary!r}")


def _resolve_inputs(
    args: argparse.Namespace,
) -> DistributedTrainingInputs:
    config_path = cast(Path | None, args.config)
    resume_path = cast(Path | None, args.resume)
    observability_path = cast(
        Path | None,
        getattr(args, "observability_config", None),
    )
    if (config_path is None) == (resume_path is None):
        raise ValueError("fixed DDP requires exactly one of --config or --resume")
    if observability_path is not None and resume_path is None:
        raise ValueError("--observability-config requires --resume")
    startup_cwd = Path.cwd().resolve()
    resume_bundle: DistributedCheckpointBundle | None = None
    resume_payload: CheckpointState | None = None
    overlays: list[dict[str, Any]] = []
    if resume_path is None:
        assert config_path is not None
        unresolved = load_config(config_path)
        extension_plan = prepare_extension_plugins(unresolved)
        config_source = "external"
    else:
        preflight = preflight_distributed_checkpoint_bundle(resume_path)
        resume_bundle = preflight.bundle
        resume_payload = preflight.common_payload
        checkpoint_config = single_runner._load_training_checkpoint_config(
            resume_payload
        )
        overlays = single_runner._load_checkpoint_config_overlays(resume_payload)
        if observability_path is None:
            unresolved = checkpoint_config
        else:
            overlay, audit = single_runner._load_observability_overlay(
                observability_path
            )
            unresolved = single_runner._apply_observability_overlay(
                checkpoint_config,
                overlay,
            )
            overlays.append(audit)
        extension_plan = prepare_extension_plugins(
            unresolved,
            expected_provenance=(
                single_runner._load_checkpoint_extension_provenance(resume_payload)
            ),
            selection_policy=ExtensionSelectionPolicy.EXACT,
        )
        config_source = "distributed_checkpoint"

    activate_training_builtins()
    force_mismatch = bool(
        getattr(args, "force_extension_version_mismatch", False)
    )
    extensions = (
        activate_extension_plugins(
            extension_plan,
            policy=ExtensionVersionPolicy.ALLOW,
            acceptance_method="force-flag",
        )
        if force_mismatch and extension_plan.version_mismatches
        else activate_extension_plugins(extension_plan)
    )
    return DistributedTrainingInputs(
        config=extensions.config,
        extensions=extensions,
        config_source=config_source,
        resume_bundle=resume_bundle,
        resume_payload=resume_payload,
        startup_cwd=startup_cwd,
        config_overlays=overlays,
    )


def _validate_operation_scope(
    config: StochaflowConfig,
    options: single_runner.ExperimentRunOptions,
    *,
    session: DistributedSession,
) -> None:
    """Reject unsupported first-version semantics before DataSource access."""

    if session.topology.world_size < 2:
        raise ValueError("fixed DDP requires at least two torchrun processes")
    if options.device is not None:
        raise ValueError(
            "fixed DDP assigns devices from LOCAL_RANK; do not pass --device"
        )
    if config.diagnostics:
        raise ValueError("fixed DDP does not yet support training Diagnostics")
    if config.trainer.validation_evaluation.enabled:
        raise ValueError("fixed DDP does not yet support epoch live Evaluation")
    if config.trainer.test_after_fit:
        raise ValueError("fixed DDP does not yet support test-after-fit")
    if config.trainer.precision == "fp16-mixed":
        raise ValueError("fixed DDP currently supports fp32 and bf16-mixed")
    unsupported_phases = sorted(
        {
            phase
            for declaration in config.metrics
            for phase in declaration.phases
            if phase != "validation"
        }
    )
    if unsupported_phases:
        raise ValueError(
            "fixed DDP accepts validation-only Metrics; unsupported phase(s): "
            + ", ".join(unsupported_phases)
        )
    if options.max_validation_batches is not None:
        raise ValueError(
            "fixed DDP requires complete validation; "
            "--limit-validation-batches is unsupported"
        )
    if options.max_test_batches is not None:
        raise ValueError(
            "fixed DDP does not run a test phase; --limit-test-batches is unsupported"
        )
    if (
        options.max_train_batches is not None
        and options.max_train_batches % config.trainer.accumulate_grad_batches != 0
    ):
        raise ValueError(
            "fixed DDP --limit-batches must contain complete gradient "
            "accumulation windows"
        )
    if config.artifacts.checkpoint_every != 1:
        raise ValueError(
            "fixed DDP currently commits an exact resume bundle after every "
            "epoch; artifacts.checkpoint_every must be 1"
        )
    if config.lr_scheduler is not None and config.lr_scheduler.interval == "epoch":
        raise ValueError("fixed DDP does not yet support epoch-interval lr schedulers")
    if session.backend != "nccl" or session.device.type != "cuda":
        raise ValueError(
            "the maintained fixed DDP operation requires CUDA/NCCL; "
            "CPU/Gloo is test-only"
        )


def _allocate_run_paths(
    output_root: Path,
    session: DistributedSession,
) -> DistributedRunPaths:
    def allocate_primary() -> dict[str, str] | None:
        if not session.is_primary:
            return None
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
        suffix = 0
        while True:
            run_id = timestamp if suffix == 0 else f"{timestamp}_{suffix:02d}"
            final_directory = output_root / run_id
            workspace = output_root / f".staging-ddp-{run_id}"
            if not final_directory.exists() and not workspace.exists():
                return {
                    "run_id": run_id,
                    "workspace": str(workspace.resolve()),
                    "final_directory": str(final_directory.resolve()),
                }
            suffix += 1

    local_value = _all_rank_stage(
        session,
        phase="rank-zero run workspace creation",
        action=allocate_primary,
    )
    value = session.collectives.broadcast_from_primary(local_value)

    def validate_broadcast() -> DistributedRunPaths:
        if type(value) is not dict:
            raise TypeError("rank zero broadcast an invalid DDP run path payload")
        raw = cast(dict[object, object], value)
        if set(raw) != {"run_id", "workspace", "final_directory"}:
            raise ValueError("rank zero broadcast incomplete DDP run paths")
        run_id = raw["run_id"]
        workspace = raw["workspace"]
        final_directory = raw["final_directory"]
        if not all(isinstance(item, str) and item for item in raw.values()):
            raise TypeError("rank zero broadcast invalid DDP run path strings")
        result = DistributedRunPaths(
            run_id=cast(str, run_id),
            workspace=Path(cast(str, workspace)),
            final_directory=Path(cast(str, final_directory)),
        )
        if result.workspace.parent != result.final_directory.parent:
            raise ValueError("DDP workspace and final directory must share a parent")
        if result.workspace.name != f".staging-ddp-{result.run_id}":
            raise ValueError("DDP workspace name is not canonical")
        if result.final_directory.name != result.run_id:
            raise ValueError("DDP final directory name is not canonical")
        if result.workspace.exists():
            raise FileExistsError("rank-zero DDP workspace already exists")
        if result.final_directory.exists():
            raise FileExistsError("DDP final run directory already exists")
        return result

    return _all_rank_stage(
        session,
        phase="run workspace broadcast validation",
        action=validate_broadcast,
    )


def _epoch_metrics(
    train: dict[str, float],
    validation: dict[str, float] | None,
    *,
    epoch: int,
) -> dict[str, float]:
    values = {
        "train/loss": float(train["loss"]),
        "system/trainer/epoch": float(epoch),
    }
    for name, value in train.items():
        if name != "loss":
            values[f"system/train/{name}"] = float(value)
    if validation is not None:
        values["valid/loss"] = float(validation["loss"])
        for name, value in validation.items():
            if name.startswith("valid/metrics/"):
                values[name] = float(value)
            elif name != "loss":
                values[f"system/valid/{name}"] = float(value)
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("fixed DDP epoch metrics must all be finite")
    return values


def _initial_selection(
    config: StochaflowConfig,
    *,
    validation_available: bool,
    resume_payload: CheckpointState | None,
    resume_bundle: DistributedCheckpointBundle | None,
) -> DistributedSelection:
    policy = (
        MonitorPolicy(
            metric=config.trainer.early_stopping.monitor,
            mode=config.trainer.early_stopping.mode,
            min_delta=config.trainer.early_stopping.min_delta,
        )
        if validation_available
        else None
    )
    patience = (
        config.trainer.early_stopping.patience
        if validation_available and config.trainer.early_stopping.enabled
        else None
    )
    if resume_payload is None:
        if resume_bundle is not None:
            raise ValueError("distributed resume bundle is missing its common payload")
        return DistributedSelection(
            best_epoch=None,
            best_metric_value=None,
            observations_without_improvement=0,
            monitor_observations=0,
            stopped_early=False,
            policy=policy,
            patience=patience,
        )
    if resume_bundle is None:
        raise ValueError("distributed resume payload is missing its committed bundle")
    metadata = cast(object, resume_payload.get("metadata"))
    if type(metadata) is not dict:
        raise TypeError("distributed resume metadata must be an exact dictionary")
    state = TrainingFitState.from_mapping(
        cast(dict[str, Any], metadata).get("training_loop")
    )
    if (
        state.tracking_enabled != validation_available
        or state.monitor_policy != policy
        or state.early_stopping_patience != patience
    ):
        raise ValueError(
            "distributed resume selection policy does not match current config/data"
        )
    if state.stopped_early:
        raise ValueError(
            "distributed checkpoint already stopped early and cannot continue"
        )
    attachment_fields = (
        resume_bundle.best_portable_checkpoint_path,
        resume_bundle.best_portable_selected_epoch,
        resume_bundle.best_portable_checkpoint_sha256,
    )
    has_attachment = all(value is not None for value in attachment_fields)
    if any(value is not None for value in attachment_fields) and not has_attachment:
        raise ValueError("distributed best portable attachment is incomplete")
    if state.best_epoch is None:
        if has_attachment:
            raise ValueError(
                "distributed bundle has a best attachment without selection state"
            )
        best_sha256 = None
    else:
        if not has_attachment:
            raise ValueError(
                "distributed resume selection is missing its bundled best portable"
            )
        if resume_bundle.best_portable_selected_epoch != state.best_epoch:
            raise ValueError(
                "bundled best portable epoch does not match selection state"
            )
        if not _is_sha256(resume_bundle.best_portable_checkpoint_sha256):
            raise ValueError("bundled best portable digest is invalid")
        best_sha256 = resume_bundle.best_portable_checkpoint_sha256
    return DistributedSelection(
        best_epoch=state.best_epoch,
        best_metric_value=state.best_metric_value,
        observations_without_improvement=state.observations_without_improvement,
        monitor_observations=state.monitor_observations,
        stopped_early=False,
        policy=policy,
        patience=patience,
        best_portable_relative_path=None,
        best_portable_sha256=best_sha256,
    )


def _checkpoint_metadata(
    *,
    inputs: DistributedTrainingInputs,
    options: single_runner.ExperimentRunOptions,
    data_artifacts: DataArtifactBindings | None,
    selected_components: dict[str, str | list[str] | None],
    runtime_options: dict[str, Any],
    selection: DistributedSelection,
) -> dict[str, Any]:
    metadata = {
        **extension_runtime_metadata(inputs.extensions),
        "config_source": inputs.config_source,
        "config_overlays": deepcopy(inputs.config_overlays),
        "lineage": {
            "resumed_from": (
                str(options.resume_checkpoint)
                if options.resume_checkpoint is not None
                else None
            )
        },
        "startup_cwd": str(inputs.startup_cwd),
        "runtime_options": deepcopy(runtime_options),
        "data_artifacts": (
            data_artifacts.to_dict() if data_artifacts is not None else None
        ),
        "selected_components": deepcopy(selected_components),
        "checkpoint_kind": "latest",
        "training_loop": selection.fit_state().to_dict(),
    }
    if selection.policy is not None:
        metadata.update(
            {
                "monitor": selection.policy.metric,
                "mode": selection.policy.mode,
                "min_delta": selection.policy.min_delta,
            }
        )
    return metadata


def _save_epoch_bundle(
    *,
    session: DistributedSession,
    components: DDPTrainingComponents,
    config: StochaflowConfig,
    paths: DistributedRunPaths,
    epoch: int,
    metrics: dict[str, float],
    next_plan: RankedTrainEpochPlan,
    metadata: dict[str, Any],
    best_selected_epoch: int | None,
    best_portable_source: Path | None,
    best_portable_sha256: str | None,
) -> tuple[DistributedCheckpointBundlePaths, Path]:
    common_runtime_state_sha256 = (
        components.trainer.checkpoint_state_fingerprint()
    )
    metadata = deepcopy(metadata)
    metadata["common_runtime_state_sha256"] = common_runtime_state_sha256
    local_bundle_id = _all_rank_stage(
        session,
        phase="checkpoint bundle identity creation",
        action=lambda: (
            new_distributed_checkpoint_bundle_id() if session.is_primary else None
        ),
    )
    raw_bundle_id = session.collectives.broadcast_from_primary(local_bundle_id)

    def resolve_bundle_paths() -> DistributedCheckpointBundlePaths:
        if not isinstance(raw_bundle_id, str):
            raise TypeError("rank zero broadcast an invalid checkpoint bundle id")
        return distributed_checkpoint_bundle_paths(
            paths.workspace / "checkpoints" / "resume",
            bundle_id=raw_bundle_id,
            completed_epoch=epoch,
            global_step=components.trainer.global_step,
        )

    bundle_paths = _all_rank_stage(
        session,
        phase="checkpoint bundle path validation",
        action=resolve_bundle_paths,
    )

    def stage_common() -> None:
        if not session.is_primary:
            return
        payload = components.checkpoint_manager.build_state(
            epoch=epoch,
            global_step=components.trainer.global_step,
            config=config.to_dict(),
            metrics=metrics,
            metadata=metadata,
            rng_state=capture_local_rng_state(session.device),
        )
        stage_distributed_common_checkpoint(bundle_paths, payload)

    _all_rank_stage(session, phase="common checkpoint staging", action=stage_common)

    def stage_best_portable() -> None:
        if not session.is_primary:
            return
        if best_selected_epoch is None:
            if best_portable_source is not None or best_portable_sha256 is not None:
                raise ValueError("best portable state exists without a selected epoch")
            return
        if best_selected_epoch > epoch:
            raise ValueError("selected best epoch cannot be later than the save epoch")
        if best_selected_epoch == epoch:
            if best_portable_source is not None or best_portable_sha256 is not None:
                raise ValueError(
                    "current best epoch must be projected from the common checkpoint"
                )
        else:
            if best_portable_source is None:
                raise ValueError("earlier best epoch requires portable source state")
            if not _is_sha256(best_portable_sha256):
                raise ValueError("earlier best portable digest is invalid")
            if _file_sha256(best_portable_source) != best_portable_sha256:
                raise ValueError("earlier best portable changed before bundle staging")
        stage_distributed_best_portable_checkpoint(
            bundle_paths,
            selected_epoch=best_selected_epoch,
            source_portable_checkpoint=best_portable_source,
            expected_source_sha256=(
                best_portable_sha256 if best_portable_source is not None else None
            ),
        )

    _all_rank_stage(
        session,
        phase="best portable checkpoint staging",
        action=stage_best_portable,
    )
    _all_rank_stage(
        session,
        phase="rank-local checkpoint staging",
        action=lambda: stage_distributed_rank_checkpoint(
            bundle_paths,
            topology=session.topology,
            next_plan=next_plan,
            device=session.device,
        ),
    )
    committed = _all_rank_stage(
        session,
        phase="checkpoint bundle commit",
        action=lambda: (
            commit_distributed_checkpoint_bundle(
                bundle_paths,
                topology=session.topology,
                backend=session.backend,
                device_type=session.device.type,
            )
            if session.is_primary
            else None
        ),
    )

    def validate_commit_receipt() -> None:
        if session.is_primary and not isinstance(
            committed,
            DistributedCheckpointBundle,
        ):
            raise TypeError("rank zero did not commit a distributed bundle")

    _all_rank_stage(
        session,
        phase="checkpoint bundle commit receipt",
        action=validate_commit_receipt,
    )
    portable = paths.workspace / "checkpoints" / "portable" / (f"epoch_{epoch:04d}.pt")
    _all_rank_stage(
        session,
        phase="portable checkpoint export",
        action=lambda: (
            export_distributed_portable_checkpoint(
                bundle_paths.final_directory,
                portable,
            )
            if session.is_primary
            else None
        ),
    )
    return bundle_paths, portable


def _restore_if_requested(
    *,
    session: DistributedSession,
    inputs: DistributedTrainingInputs,
    components: DDPTrainingComponents,
    train_execution: RankedTrainExecution,
    options: single_runner.ExperimentRunOptions,
    fresh_plan: RankedTrainEpochPlan | None = None,
) -> int:
    resume_bundle = inputs.resume_bundle
    if resume_bundle is None:
        return 1
    assert inputs.resume_payload is not None
    completed_epoch = resume_bundle.completed_epoch

    def validate_resume_progress() -> None:
        if completed_epoch >= options.num_epochs:
            raise ValueError(
                f"distributed checkpoint already completed epoch {completed_epoch}, "
                f"which meets or exceeds target {options.num_epochs}"
            )

    _all_rank_stage(
        session,
        phase="resume progress validation",
        action=validate_resume_progress,
    )
    if fresh_plan is None:
        fresh_plan = _all_rank_stage(
            session,
            phase="resume next-epoch planning",
            action=lambda: train_execution.plan_epoch(
                completed_epoch + 1,
                microbatches_per_window=configured_accumulation(components),
                max_microbatches=options.max_train_batches,
            ),
        )
    restore = _all_rank_stage(
        session,
        phase="resume bundle validation",
        action=lambda: load_distributed_checkpoint_bundle(
            resume_bundle.directory,
            topology=session.topology,
            backend=session.backend,
            device_type=session.device.type,
            fresh_next_plan=fresh_plan,
            expected_bundle=resume_bundle,
        ),
    )
    expected_runtime_digest = _all_rank_stage(
        session,
        phase="resume runtime digest preflight",
        action=lambda: _resume_runtime_state_digest(restore.common_payload),
    )

    def apply_restore() -> None:
        apply_distributed_checkpoint_restore(
            restore,
            checkpoint_manager=components.checkpoint_manager,
            local_device=session.device,
        )
        if components.ema is not None:
            components.ema.to(session.device)

    _all_rank_stage(session, phase="resume state apply", action=apply_restore)

    def resolve_restore_acceptor() -> Callable[..., object]:
        acceptor = getattr(
            components.trainer,
            "accept_restored_state",
            None,
        )
        if not callable(acceptor):
            raise TypeError(
                "DDPTrainer lacks the one-shot restored-state acceptance contract"
            )
        return acceptor

    accept_restored_state = _all_rank_stage(
        session,
        phase="resume acceptance contract validation",
        action=resolve_restore_acceptor,
    )
    _all_rank_stage(
        session,
        phase="resume state acceptance",
        action=lambda: accept_restored_state(
            global_step=resume_bundle.global_step,
            common_checkpoint_sha256=(resume_bundle.common_checkpoint_sha256),
            common_runtime_state_sha256=expected_runtime_digest,
        ),
    )
    return completed_epoch + 1


def configured_accumulation(components: DDPTrainingComponents) -> int:
    """Return the DDPTrainer's validated accumulation factor."""

    value = cast(object, components.trainer.accumulate_grad_batches)
    if type(value) is not int or value <= 0:
        raise RuntimeError("DDPTrainer has an invalid accumulation factor")
    return cast(int, value)


def _certified_effective_batch(
    plan: RankedTrainEpochPlan,
    *,
    topology: DistributedTopology,
    accumulation: int,
) -> dict[str, int]:
    """Project effective-batch facts from an authenticated ranked-data plan."""

    if plan.rank != topology.rank or plan.world_size != topology.world_size:
        raise ValueError("ranked train plan does not match the active DDP topology")
    if plan.microbatches_per_window != accumulation:
        raise ValueError(
            "ranked train plan accumulation does not match the DDPTrainer"
        )
    return {
        "per_rank_batch_size": plan.samples_per_microbatch,
        "gradient_accumulation": accumulation,
        "effective_global_batch": (
            topology.world_size * plan.samples_per_microbatch * accumulation
        ),
    }


def _run_active_session(
    args: argparse.Namespace,
    session: DistributedSession,
    *,
    component_factory: DistributedComponentFactory,
) -> TrainingRunOutcome:
    inputs = _all_rank_stage(
        session,
        phase="DDP input and extension resolution",
        action=lambda: _resolve_inputs(args),
    )
    config = inputs.config

    def resolve_options() -> single_runner.ExperimentRunOptions:
        resolved = single_runner.ExperimentRunOptions.from_namespace(
            args,
            configured_num_epochs=config.trainer.num_epochs,
            configured_show_progress=config.trainer.show_progress,
        )
        resolved = replace(
            resolved,
            resume_checkpoint=(
                inputs.resume_bundle.directory
                if inputs.resume_bundle is not None
                else None
            ),
        )
        _validate_operation_scope(config, resolved, session=session)
        return resolved

    options = _all_rank_stage(
        session,
        phase="DDP invocation validation",
        action=resolve_options,
    )
    config.trainer.num_epochs = options.num_epochs
    config.trainer.show_progress = options.show_progress
    config.trainer.device = session.device.type
    if inputs.resume_payload is not None:
        resume_metadata = cast(object, inputs.resume_payload.get("metadata"))
        if type(resume_metadata) is not dict:
            raise TypeError("distributed resume metadata must be an exact dictionary")
        saved_options = cast(dict[object, object], resume_metadata).get(
            "runtime_options"
        )
        if type(saved_options) is not dict:
            raise TypeError(
                "distributed resume metadata is missing runtime_options"
            )
        saved_deterministic = cast(dict[object, object], saved_options).get(
            "deterministic"
        )
        if type(saved_deterministic) is not bool:
            raise TypeError(
                "distributed resume deterministic option must be a boolean"
            )
        if saved_deterministic != options.deterministic:
            raise ValueError(
                "distributed resume must preserve the saved deterministic policy"
            )
    invocation_facts = {
        "config": config.to_dict(),
        "extensions": extension_runtime_metadata(inputs.extensions),
        "epochs": options.num_epochs,
        "max_train_batches": options.max_train_batches,
        "deterministic": options.deterministic,
        "output_override": (
            str(args.output_dir) if args.output_dir is not None else None
        ),
        "resume_bundle_id": (
            inputs.resume_bundle.bundle_id if inputs.resume_bundle is not None else None
        ),
    }
    if not session.collectives.all_equal(invocation_facts):
        raise ValueError("fixed DDP ranks resolved different invocation authority")

    strict_resume = inputs.resume_bundle is not None
    expected_artifacts = _all_rank_stage(
        session,
        phase="resume data authority validation",
        action=lambda: (
            single_runner._checkpoint_data_artifacts(inputs.resume_payload)
            if inputs.resume_payload is not None
            else None
        ),
    )
    _all_rank_stage(
        session,
        phase="pre-data RNG initialization",
        action=lambda: set_cpu_seed(
            config.experiment.seed,
            deterministic=options.deterministic,
        ),
    )

    def compose_ranked_data() -> DataLoaders:
        reporter: RichArtifactVerificationReporter | None = None
        local_error: BaseException | None = None
        loaders: DataLoaders | None = None
        try:
            if session.is_primary and options.show_progress:
                reporter = RichArtifactVerificationReporter()
            loaders = build_data_loaders(
                config.data,
                seed=config.experiment.seed,
                strict_resume=strict_resume,
                expected_artifacts=expected_artifacts,
                verification_observer=(
                    reporter.observe if reporter is not None else None
                ),
                verification_workers=options.artifact_verification_workers,
                rank_context=session.data_rank_context,
            )
        except BaseException as error:  # noqa: BLE001 - preserve local cause
            local_error = error
        if reporter is not None:
            try:
                reporter.close()
            except BaseException as cleanup_error:  # noqa: BLE001
                if local_error is None:
                    local_error = cleanup_error
                else:
                    _add_failure_note(
                        local_error,
                        "artifact verification reporter close also failed: "
                        f"{_failure_summary(cleanup_error)!r}"
                    )
        if local_error is not None:
            raise local_error
        assert loaders is not None
        return loaders

    loaders = _all_rank_stage(
        session,
        phase="ranked data composition",
        action=compose_ranked_data,
    )

    def validate_ranked_data() -> tuple[
        DataArtifactBindings | None,
        RankedDataExecution,
    ]:
        data_artifacts = loaders.artifact_bindings
        single_runner._validate_resume_data_artifacts(
            expected_artifacts,
            data_artifacts,
            strict_resume=strict_resume,
        )
        ranked = loaders.ranked_execution
        if ranked is None:
            raise RuntimeError("fixed DDP requires ranked data execution")
        if config.trainer.early_stopping.enabled and ranked.validation is None:
            raise ValueError("fixed DDP early stopping requires validation")
        return data_artifacts, ranked

    data_artifacts, ranked = _all_rank_stage(
        session,
        phase="ranked data authority validation",
        action=validate_ranked_data,
    )
    artifact_facts = data_artifacts.to_dict() if data_artifacts is not None else None
    if not session.collectives.all_equal(artifact_facts):
        raise ValueError("fixed DDP ranks verified different data artifacts")
    validation_available = ranked.validation is not None
    if not session.collectives.all_equal(
        {"validation_available": validation_available}
    ):
        raise ValueError("fixed DDP ranks received different validation capabilities")

    selection = _all_rank_stage(
        session,
        phase="validation selection and bundled-best preflight",
        action=lambda: _initial_selection(
            config,
            validation_available=validation_available,
            resume_payload=inputs.resume_payload,
            resume_bundle=inputs.resume_bundle,
        ),
    )
    initial_selection_facts = _all_rank_stage(
        session,
        phase="validation selection state projection",
        action=lambda: {
            "state": selection.fit_state().to_dict(),
            "best_portable_sha256": selection.best_portable_sha256,
        },
    )
    if not session.collectives.all_equal(initial_selection_facts):
        raise ValueError(
            "fixed DDP ranks preflighted different validation selection state"
        )

    startup_epoch = (
        inputs.resume_bundle.completed_epoch + 1
        if inputs.resume_bundle is not None
        else 1
    )

    def plan_startup_epoch() -> RankedTrainEpochPlan:
        if startup_epoch > options.num_epochs:
            raise ValueError(
                f"distributed checkpoint already completed epoch "
                f"{startup_epoch - 1}, which meets or exceeds target "
                f"{options.num_epochs}"
            )
        return ranked.train.plan_epoch(
            startup_epoch,
            microbatches_per_window=config.trainer.accumulate_grad_batches,
            max_microbatches=options.max_train_batches,
        )

    startup_plan = _all_rank_stage(
        session,
        phase="startup ranked-data planning",
        action=plan_startup_epoch,
    )

    configured_output = Path(config.experiment.output_dir)
    output_root = configured_output.parent if strict_resume else configured_output
    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    output_root = output_root.resolve()
    run_paths = _allocate_run_paths(output_root, session)

    def bind_run_identity() -> None:
        config.experiment.exp_id = run_paths.run_id
        config.experiment.output_dir = str(run_paths.final_directory)
        set_cpu_seed(config.experiment.seed, deterministic=options.deterministic)

    _all_rank_stage(
        session,
        phase="run identity and RNG binding",
        action=bind_run_identity,
    )
    if not session.collectives.all_equal(config.to_dict()):
        raise ValueError("fixed DDP ranks bound different resolved run configs")
    components = component_factory(config, session)
    accumulation = configured_accumulation(components)
    batch_facts = _all_rank_stage(
        session,
        phase="effective global batch certification",
        action=lambda: _certified_effective_batch(
            startup_plan,
            topology=session.topology,
            accumulation=accumulation,
        ),
    )
    if not session.collectives.all_equal(batch_facts):
        raise ValueError("fixed DDP ranks certified different effective batch facts")
    selected_components = _all_rank_stage(
        session,
        phase="selected component identity projection",
        action=lambda: selected_training_component_identities(
            config,
            inference_recipe=(
                components.plan.inference_recipe.name
                if components.plan.inference_recipe is not None
                else None
            ),
        ),
    )
    runtime_options = {
        "ddp": True,
        "device": None,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "epochs": args.epochs,
        "limit_batches": args.limit_batches,
        "limit_validation_batches": args.limit_validation_batches,
        "limit_test_batches": args.limit_test_batches,
        "deterministic": args.deterministic,
        "progress": args.progress,
        "no_progress": args.no_progress,
        "artifact_verification_workers": options.artifact_verification_workers,
        "force_extension_version_mismatch": bool(
            getattr(args, "force_extension_version_mismatch", False)
        ),
        **batch_facts,
        "topology": {
            "world_size": session.topology.world_size,
            "local_world_size": session.topology.local_world_size,
            "backend": session.backend,
            "device_type": session.device.type,
        },
    }
    extension_metadata = extension_runtime_metadata(inputs.extensions)
    manifest_path = run_paths.workspace / "run_manifest.yaml"
    manifest: dict[str, Any] = {
        "kind": "distributed_training",
        "status": "running",
        "config_source": inputs.config_source,
        "config": config.to_dict(),
        **extension_metadata,
        "selected_components": selected_components,
        "config_overlays": deepcopy(inputs.config_overlays),
        "lineage": {
            "resumed_from": (
                str(inputs.resume_bundle.directory)
                if inputs.resume_bundle is not None
                else None
            )
        },
        "startup_cwd": str(inputs.startup_cwd),
        "runtime_options": runtime_options,
        "data_artifacts": artifact_facts,
    }
    logger: ExperimentLogger = NullLogger()
    logger_close_attempted = False
    operation_error: BaseException | None = None
    try:

        def start_primary_side_effects() -> None:
            nonlocal logger
            if not session.is_primary:
                return
            run_paths.workspace.mkdir(parents=False, exist_ok=False)
            write_yaml_manifest(
                run_paths.workspace / "resolved_config.yaml",
                config.to_dict(),
            )
            write_yaml_manifest(manifest_path, manifest)
            logger_experiment = deepcopy(config.experiment)
            logger_experiment.output_dir = str(run_paths.workspace)
            logger = build_logger(
                config.logging,
                experiment=logger_experiment,
                resolved_config=config,
            )

        _all_rank_stage(
            session,
            phase="rank-zero manifest and logger creation",
            action=start_primary_side_effects,
        )
        if inputs.resume_bundle is not None:
            start_epoch = _restore_if_requested(
                session=session,
                inputs=inputs,
                components=components,
                train_execution=ranked.train,
                options=options,
                fresh_plan=startup_plan,
            )
        else:
            start_epoch = startup_epoch
            _all_rank_stage(
                session,
                phase="rank-local runtime RNG initialization",
                action=lambda: set_local_seed(
                    _distributed_runtime_seed(
                        config.experiment.seed,
                        rank=session.topology.rank,
                    ),
                    device=session.device,
                    deterministic=options.deterministic,
                ),
            )
        latest_relative: str | None = None
        resume_bundle = inputs.resume_bundle
        if selection.best_epoch is not None and resume_bundle is not None:
            inherited_destination = (
                run_paths.workspace / "checkpoints" / "portable" / "inherited_best.pt"
            )

            def materialize_inherited_best() -> None:
                if not session.is_primary:
                    return
                source = resume_bundle.best_portable_checkpoint_path
                expected_digest = resume_bundle.best_portable_checkpoint_sha256
                if source is None or expected_digest is None:
                    raise ValueError(
                        "distributed resume is missing its bundled best portable"
                    )
                if _file_sha256(source) != expected_digest:
                    raise ValueError(
                        "bundled best portable content changed after preflight"
                    )
                inherited_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, inherited_destination)
                if _file_sha256(inherited_destination) != expected_digest:
                    inherited_destination.unlink(missing_ok=True)
                    raise ValueError(
                        "copied best portable does not match the bundled authority"
                    )

            _all_rank_stage(
                session,
                phase="inherited best portable materialization",
                action=materialize_inherited_best,
            )
            selection.best_portable_relative_path = inherited_destination.relative_to(
                run_paths.workspace
            ).as_posix()
            local_best_digest = _all_rank_stage(
                session,
                phase="inherited best portable digest calculation",
                action=lambda: (
                    _file_sha256(inherited_destination) if session.is_primary else None
                ),
            )
            selection.best_portable_sha256 = cast(
                str,
                session.collectives.broadcast_from_primary(local_best_digest),
            )
            if not _is_sha256(selection.best_portable_sha256):
                raise ValueError("rank zero broadcast an invalid best portable digest")
            if (
                selection.best_portable_sha256
                != resume_bundle.best_portable_checkpoint_sha256
            ):
                raise ValueError(
                    "materialized best portable digest differs from the resume bundle"
                )

        final_metrics: dict[str, float] | None = None
        latest_resume_relative: str | None = None
        current_plan = startup_plan
        for epoch in range(start_epoch, options.num_epochs + 1):
            train_result = components.trainer.train_epoch(
                ranked.train,
                epoch_index=epoch,
                max_microbatches=options.max_train_batches,
                preplanned=current_plan,
            )
            validation = (
                components.trainer.evaluate_epoch(
                    ranked.validation,
                    epoch_index=epoch,
                )
                if validation_available and ranked.validation is not None
                else None
            )
            metrics = _all_rank_stage(
                session,
                phase=f"epoch {epoch} metric projection",
                action=lambda train_result=train_result, validation=validation, epoch=epoch: (
                    _epoch_metrics(
                        dict(train_result.metrics),
                        validation,
                        epoch=epoch,
                    )
                ),
            )
            if not session.collectives.all_equal(metrics):
                raise ValueError(
                    f"fixed DDP ranks projected different epoch {epoch} metrics"
                )
            improved = _all_rank_stage(
                session,
                phase=f"epoch {epoch} selection update",
                action=lambda metrics=metrics, epoch=epoch: selection.observe(
                    metrics,
                    epoch=epoch,
                ),
            )
            selection_facts = _all_rank_stage(
                session,
                phase=f"epoch {epoch} selection state projection",
                action=lambda improved=improved: {
                    "improved": improved,
                    "state": selection.fit_state().to_dict(),
                    "best_portable": selection.best_portable_relative_path,
                    "best_portable_sha256": selection.best_portable_sha256,
                },
            )
            if not session.collectives.all_equal(selection_facts):
                raise ValueError(
                    f"fixed DDP ranks produced different epoch {epoch} selection state"
                )
            _all_rank_stage(
                session,
                phase="rank-zero epoch logging",
                action=lambda metrics=metrics: (
                    logger.log_metrics(metrics, step=components.trainer.global_step)
                    if session.is_primary
                    else None
                ),
            )
            next_plan = _all_rank_stage(
                session,
                phase="next epoch data planning",
                action=lambda epoch=epoch: ranked.train.plan_epoch(
                    epoch + 1,
                    microbatches_per_window=config.trainer.accumulate_grad_batches,
                    max_microbatches=options.max_train_batches,
                ),
            )
            current_plan = next_plan
            metadata = _all_rank_stage(
                session,
                phase=f"epoch {epoch} checkpoint metadata projection",
                action=lambda epoch=epoch: _checkpoint_metadata(
                    inputs=inputs,
                    options=options,
                    data_artifacts=data_artifacts,
                    selected_components=selected_components,
                    runtime_options=runtime_options,
                    selection=selection,
                ),
            )
            if not session.collectives.all_equal(metadata):
                raise ValueError(
                    f"fixed DDP ranks produced different epoch {epoch} metadata"
                )
            best_portable_source: Path | None = None
            if selection.best_epoch is not None and selection.best_epoch != epoch:
                best_relative = selection.best_portable_relative_path
                if best_relative is None:
                    raise ValueError(
                        "earlier selected epoch has no materialized portable state"
                    )
                best_portable_source = run_paths.workspace / best_relative
            bundle_paths, portable = _save_epoch_bundle(
                session=session,
                components=components,
                config=config,
                paths=run_paths,
                epoch=epoch,
                metrics=metrics,
                next_plan=next_plan,
                metadata=metadata,
                best_selected_epoch=selection.best_epoch,
                best_portable_source=best_portable_source,
                best_portable_sha256=selection.best_portable_sha256,
            )
            latest_resume_relative = bundle_paths.final_directory.relative_to(
                run_paths.workspace
            ).as_posix()
            latest_relative = portable.relative_to(run_paths.workspace).as_posix()
            local_portable_digest = _all_rank_stage(
                session,
                phase=f"epoch {epoch} portable digest calculation",
                action=lambda portable=portable: (
                    _file_sha256(portable) if session.is_primary else None
                ),
            )
            portable_sha256 = cast(
                str,
                session.collectives.broadcast_from_primary(local_portable_digest),
            )
            if not _is_sha256(portable_sha256):
                raise ValueError("rank zero broadcast an invalid portable digest")
            if improved:
                selection.best_portable_relative_path = latest_relative
                selection.best_portable_sha256 = portable_sha256
            final_metrics = metrics
            if selection.stopped_early:
                break
        if (
            final_metrics is None
            or latest_relative is None
            or latest_resume_relative is None
        ):
            raise RuntimeError("fixed DDP completed no epochs")
        final_epoch = int(final_metrics["system/trainer/epoch"])
        latest_checkpoint = run_paths.final_directory / latest_relative
        best_checkpoint = (
            run_paths.final_directory / selection.best_portable_relative_path
            if selection.best_portable_relative_path is not None
            else None
        )
        selected_checkpoint = best_checkpoint or latest_checkpoint
        metrics_path, log_path = single_runner._local_log_paths(config)
        outcome = TrainingRunOutcome(
            output_dir=run_paths.final_directory,
            final_epoch=final_epoch,
            final_metrics=final_metrics,
            latest_checkpoint=latest_checkpoint,
            best_epoch=selection.best_epoch,
            best_metric_name=(
                selection.policy.metric if selection.policy is not None else None
            ),
            best_metric_value=selection.best_metric_value,
            best_checkpoint=best_checkpoint,
            selected_checkpoint=selected_checkpoint,
            selected_checkpoint_kind=("best" if best_checkpoint else "final"),
            stopped_early=selection.stopped_early,
            phase_test_metrics={},
            manifest_path=run_paths.final_directory / "run_manifest.yaml",
            metrics_path=metrics_path,
            log_path=log_path,
        )

        def publish_success() -> None:
            nonlocal logger_close_attempted
            if not session.is_primary:
                return
            logger_close_attempted = True
            logger.close()
            manifest["status"] = "completed"
            manifest["outcome"] = outcome.to_manifest()
            manifest["distributed_resume"] = {
                "latest_bundle": latest_resume_relative,
                "portable_checkpoint": latest_relative,
                "portable_checkpoint_role": "sample_or_evaluate_only",
            }
            write_yaml_manifest(manifest_path, manifest)
            run_paths.workspace.replace(run_paths.final_directory)

        _all_rank_stage(
            session,
            phase="final run publication readiness",
            action=lambda: None,
        )
        # Publication is the last rank-zero action.  It must not be followed by
        # a process-group collective: once the atomic rename succeeds, a later
        # reporting failure must not turn a completed public directory into a
        # failed command.  A rank-zero publication error is propagated by the
        # launcher, while every peer has already certified readiness above.
        publish_success()
        return outcome
    except BaseException as error:
        operation_error = error
        raise
    finally:
        if session.is_primary and not logger_close_attempted:
            logger_close_attempted = True
            try:
                logger.close()
            except Exception as cleanup_error:
                if operation_error is None:
                    raise
                _add_failure_note(
                    operation_error,
                    "rank-zero logger close also failed: "
                    f"{_failure_summary(cleanup_error)!r}"
                )
        if session.is_primary and operation_error is not None:
            try:
                _handle_failed_workspace(
                    run_paths,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    error=operation_error,
                )
            except Exception as cleanup_error:  # noqa: BLE001 - keep root failure
                _add_failure_note(
                    operation_error,
                    "failed DDP workspace cleanup/preservation also failed: "
                    f"{_failure_summary(cleanup_error)!r}"
                )


def run_distributed_experiment_from_args(
    args: argparse.Namespace,
    *,
    session_factory: DistributedSessionFactory = _default_session_factory,
    component_factory: DistributedComponentFactory = build_ddp_training_components,
) -> TrainingRunOutcome:
    """Run fixed single-node DDP inside one explicit process-group session."""

    if not bool(getattr(args, "ddp", False)):
        raise ValueError("distributed experiment runner requires --ddp")

    def execute() -> TrainingRunOutcome:
        session = session_factory()
        with session:
            return _run_active_session(
                args,
                session,
                component_factory=component_factory,
            )

    if (
        session_factory is _default_session_factory
        and component_factory is build_ddp_training_components
    ):
        from torch.distributed.elastic.multiprocessing.errors import (  # noqa: PLC0415
            record,
        )

        recorded_result = record(execute)()
        if recorded_result is None:
            raise RuntimeError("torchrun error recording returned no DDP outcome")
        return recorded_result
    return execute()


__all__ = ["run_distributed_experiment_from_args"]
