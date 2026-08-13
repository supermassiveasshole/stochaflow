"""Focused operation tests for the fixed single-node DDP entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import pytest
import torch

from stochaflow.data import RankedEpochDataIdentity, RankedTrainEpochPlan
from stochaflow.scripts import distributed_experiment_runner, experiment_runner
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.training.distributed.checkpoint_bundle import (
    DistributedCheckpointBundle,
)
from stochaflow.training.distributed.contracts import DistributedTopology
from stochaflow.training.distributed.session import DistributedSession
from stochaflow.utils.config import (
    ArtifactConfig,
    ComponentConfig,
    ExperimentConfig,
    StochaflowConfig,
    TrainerConfig,
)


class RecordingCollectives:
    """Single-process simulation of successful two-rank control collectives."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def broadcast_from_primary(self, value: object) -> object:
        self.events.append("broadcast")
        return value

    def gather_to_primary(self, value: object) -> tuple[object, ...]:
        self.events.append("gather")
        return (value, None)

    def all_true(self, value: bool) -> bool:
        self.events.append(f"all_true:{value}")
        return value

    def all_equal(self, value: object) -> bool:
        self.events.append("all_equal")
        return True


class RecordingSession:
    """Small active-session stand-in used only at the operation boundary."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.collectives = RecordingCollectives()
        self.topology = DistributedTopology(
            rank=0,
            local_rank=0,
            world_size=2,
            local_world_size=2,
        )
        self.backend = "nccl"
        self.device = torch.device("cuda:0")

    @property
    def is_primary(self) -> bool:
        return True

    def __enter__(self) -> Self:
        self.events.append("enter")
        return self

    def __exit__(self, *args: object) -> None:
        self.events.append("exit")


def minimal_config() -> StochaflowConfig:
    """Return a config that satisfies the first DDP operation admission."""

    return StochaflowConfig(
        experiment=ExperimentConfig(name="test"),
        data=ComponentConfig(name="data"),
        model=ComponentConfig(name="model"),
        training=ComponentConfig(name="training"),
        trainer=TrainerConfig(
            num_epochs=2,
            device="auto",
            test_after_fit=False,
        ),
        artifacts=ArtifactConfig(checkpoint_every=1),
    )


def test_training_parser_selects_ddp_without_changing_default() -> None:
    parser = build_argument_parser()

    ordinary = parser.parse_args(["train", "--config", "train.yaml"])
    distributed = parser.parse_args(["train", "--config", "train.yaml", "--ddp"])

    assert ordinary.ddp is False
    assert distributed.ddp is True


def test_experiment_entry_routes_ddp_without_entering_single_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    monkeypatch.setattr(
        distributed_experiment_runner,
        "run_distributed_experiment_from_args",
        lambda args: marker,
    )

    result = experiment_runner.run_experiment_from_args(argparse.Namespace(ddp=True))

    assert result is marker


def test_multi_process_launch_without_ddp_is_rejected_before_input_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "8")

    with pytest.raises(RuntimeError, match="without --ddp"):
        experiment_runner.run_experiment_from_args(argparse.Namespace(ddp=False))


@pytest.mark.parametrize("role", ["distributed_common", "distributed_portable"])
def test_single_process_resume_rejects_distributed_checkpoint_roles(
    role: str,
) -> None:
    with pytest.raises(ValueError, match="not resumable training state"):
        experiment_runner._require_training_resume_checkpoint_role(
            cast(Any, {"metadata": {"checkpoint_role": role}})
        )


def test_distributed_entry_establishes_session_before_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    session = RecordingSession()

    def run_active(
        args: argparse.Namespace,
        active: DistributedSession,
        *,
        component_factory: object,
    ) -> Any:
        assert args.ddp is True
        assert active is session
        assert session.events == ["enter"]
        assert component_factory is not None
        session.events.append("operation")
        return marker

    monkeypatch.setattr(
        distributed_experiment_runner,
        "_run_active_session",
        run_active,
    )

    result = distributed_experiment_runner.run_distributed_experiment_from_args(
        argparse.Namespace(ddp=True),
        session_factory=lambda: cast(DistributedSession, session),
    )

    assert result is marker
    assert session.events == ["enter", "operation", "exit"]


def test_all_rank_stage_preserves_the_local_exception() -> None:
    session = RecordingSession()
    expected = RuntimeError("local failure")

    def fail() -> None:
        raise expected

    with pytest.raises(RuntimeError) as exc_info:
        distributed_experiment_runner._all_rank_stage(
            cast(DistributedSession, session),
            phase="test phase",
            action=fail,
        )

    assert exc_info.value is expected
    assert "fixed DDP failure summary" in "\n".join(expected.__notes__)
    assert session.collectives.events == ["all_true:False", "gather", "broadcast"]


def test_all_rank_stage_preserves_local_error_when_reporting_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = RecordingSession()
    expected = RuntimeError("local failure")

    def fail_gather(value: object) -> tuple[object, ...]:
        raise OSError("reporting transport failed")

    def fail_action() -> None:
        raise expected

    monkeypatch.setattr(session.collectives, "gather_to_primary", fail_gather)

    with pytest.raises(RuntimeError) as exc_info:
        distributed_experiment_runner._all_rank_stage(
            cast(DistributedSession, session),
            phase="test phase",
            action=fail_action,
        )

    assert exc_info.value is expected
    assert "failure reporting also failed" in "\n".join(expected.__notes__)


def test_workspace_is_rank_zero_owned_and_hidden_until_publication(
    tmp_path: Path,
) -> None:
    session = RecordingSession()

    paths = distributed_experiment_runner._allocate_run_paths(
        tmp_path,
        cast(DistributedSession, session),
    )

    assert not paths.workspace.exists()
    assert paths.workspace.parent == tmp_path
    assert paths.workspace.name == f".staging-ddp-{paths.run_id}"
    assert not paths.final_directory.exists()
    assert session.collectives.events == [
        "all_true:True",
        "broadcast",
        "all_true:True",
    ]


def test_failed_workspace_without_committed_bundle_is_removed(tmp_path: Path) -> None:
    workspace = tmp_path / ".staging-ddp-run"
    workspace.mkdir()
    paths = distributed_experiment_runner.DistributedRunPaths(
        run_id="run",
        workspace=workspace,
        final_directory=tmp_path / "run",
    )

    distributed_experiment_runner._handle_failed_workspace(
        paths,
        manifest_path=workspace / "run_manifest.yaml",
        manifest={"status": "running"},
        error=RuntimeError("failure"),
    )

    assert not workspace.exists()
    assert not paths.final_directory.exists()


def test_failed_workspace_with_committed_bundle_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".staging-ddp-run"
    bundle = workspace / "checkpoints" / "resume" / "epoch-00000001-bundle"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(
        distributed_experiment_runner,
        "preflight_distributed_checkpoint_bundle",
        lambda path: object(),
    )
    paths = distributed_experiment_runner.DistributedRunPaths(
        run_id="run",
        workspace=workspace,
        final_directory=tmp_path / "run",
    )
    manifest = {"status": "running"}

    distributed_experiment_runner._handle_failed_workspace(
        paths,
        manifest_path=workspace / "run_manifest.yaml",
        manifest=manifest,
        error=RuntimeError("failure"),
    )

    assert workspace.is_dir()
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "type": "builtins.RuntimeError",
        "message": "failure",
    }
    assert (workspace / "run_manifest.yaml").is_file()


def test_exact_resume_validates_fresh_plan_before_apply_and_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []
    digest = "a" * 64
    bundle = DistributedCheckpointBundle(
        directory=tmp_path / "bundle",
        manifest_path=tmp_path / "bundle" / "bundle-manifest.yaml",
        bundle_id="b" * 32,
        completed_epoch=2,
        global_step=17,
        world_size=2,
        local_world_size=2,
        backend="nccl",
        device_type="cuda",
        common_checkpoint_path=tmp_path / "bundle" / "common.pt",
        common_checkpoint_sha256=digest,
    )
    fresh_plan = object()
    restore = SimpleNamespace(
        common_payload={
            "metadata": {"common_runtime_state_sha256": "c" * 64}
        }
    )

    class RecordingExecution:
        def plan_epoch(
            self,
            epoch: int,
            *,
            microbatches_per_window: int,
            max_microbatches: int | None,
        ) -> object:
            events.append(
                ("fresh-plan", epoch, microbatches_per_window, max_microbatches)
            )
            return fresh_plan

    class RecordingTrainer:
        accumulate_grad_batches = 4

        def accept_restored_state(
            self,
            *,
            global_step: int,
            common_checkpoint_sha256: str,
            common_runtime_state_sha256: str,
        ) -> None:
            events.append(
                (
                    "accept",
                    global_step,
                    common_checkpoint_sha256,
                    common_runtime_state_sha256,
                )
            )

    def load_bundle(
        directory: Path,
        **kwargs: object,
    ) -> object:
        assert kwargs["expected_bundle"] is bundle
        events.append(("validate", directory, kwargs["fresh_next_plan"]))
        return restore

    def apply_restore(
        value: object,
        *,
        checkpoint_manager: object,
        local_device: torch.device | str,
    ) -> None:
        events.append(("apply", value, checkpoint_manager, local_device))

    monkeypatch.setattr(
        distributed_experiment_runner,
        "load_distributed_checkpoint_bundle",
        load_bundle,
    )
    monkeypatch.setattr(
        distributed_experiment_runner,
        "apply_distributed_checkpoint_restore",
        apply_restore,
    )
    checkpoint_manager = object()
    components = SimpleNamespace(
        trainer=RecordingTrainer(),
        checkpoint_manager=checkpoint_manager,
        ema=None,
    )
    options = experiment_runner.ExperimentRunOptions(
        num_epochs=5,
        max_train_batches=8,
        max_validation_batches=None,
        max_test_batches=None,
        deterministic=False,
        show_progress=False,
        artifact_verification_workers=None,
        resume_checkpoint=bundle.directory,
        device=None,
    )

    start_epoch = distributed_experiment_runner._restore_if_requested(
        session=cast(DistributedSession, RecordingSession()),
        inputs=cast(
            Any,
            SimpleNamespace(resume_bundle=bundle, resume_payload={}),
        ),
        components=cast(Any, components),
        train_execution=cast(Any, RecordingExecution()),
        options=options,
    )

    assert start_epoch == 3
    assert events == [
        ("fresh-plan", 3, 4, 8),
        ("validate", bundle.directory, fresh_plan),
        ("apply", restore, checkpoint_manager, torch.device("cuda:0")),
        ("accept", 17, digest, "c" * 64),
    ]


def test_checkpoint_stage_proves_common_state_before_manifest_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    session = RecordingSession()
    final_bundle: DistributedCheckpointBundle | None = None

    class RecordingTrainer:
        global_step = 11

        def checkpoint_state_fingerprint(self) -> str:
            events.append("state-consensus")
            return "f" * 64

    class RecordingCheckpointManager:
        def build_state(self, **kwargs: object) -> dict[str, object]:
            events.append("common-build")
            return {"epoch": kwargs["epoch"]}

    def stage_common(*args: object, **kwargs: object) -> None:
        events.append("common-stage")

    def stage_rank(*args: object, **kwargs: object) -> None:
        events.append("rank-stage")

    def stage_best(*args: object, **kwargs: object) -> None:
        events.append("best-stage")

    def commit(paths: Any, **kwargs: object) -> DistributedCheckpointBundle:
        nonlocal final_bundle
        events.append("manifest-commit")
        final_bundle = DistributedCheckpointBundle(
            directory=paths.final_directory,
            manifest_path=paths.final_directory / "bundle-manifest.yaml",
            bundle_id=paths.bundle_id,
            completed_epoch=paths.completed_epoch,
            global_step=paths.global_step,
            world_size=2,
            local_world_size=2,
            backend="nccl",
            device_type="cuda",
            common_checkpoint_path=paths.final_directory / "common.pt",
            common_checkpoint_sha256="c" * 64,
        )
        return final_bundle

    def export(*args: object, **kwargs: object) -> None:
        events.append("portable-export")

    monkeypatch.setattr(
        distributed_experiment_runner,
        "new_distributed_checkpoint_bundle_id",
        lambda: "d" * 32,
    )
    monkeypatch.setattr(
        distributed_experiment_runner,
        "stage_distributed_common_checkpoint",
        stage_common,
    )
    monkeypatch.setattr(
        distributed_experiment_runner,
        "stage_distributed_rank_checkpoint",
        stage_rank,
    )
    monkeypatch.setattr(
        distributed_experiment_runner,
        "stage_distributed_best_portable_checkpoint",
        stage_best,
    )
    monkeypatch.setattr(
        distributed_experiment_runner,
        "commit_distributed_checkpoint_bundle",
        commit,
    )
    monkeypatch.setattr(
        distributed_experiment_runner,
        "export_distributed_portable_checkpoint",
        export,
    )
    components = SimpleNamespace(
        trainer=RecordingTrainer(),
        checkpoint_manager=RecordingCheckpointManager(),
    )
    paths = distributed_experiment_runner.DistributedRunPaths(
        run_id="run",
        workspace=tmp_path / ".staging-ddp-run",
        final_directory=tmp_path / "run",
    )

    bundle_paths, portable = distributed_experiment_runner._save_epoch_bundle(
        session=cast(DistributedSession, session),
        components=cast(Any, components),
        config=minimal_config(),
        paths=paths,
        epoch=2,
        metrics={"train/loss": 1.0},
        next_plan=cast(Any, object()),
        metadata={},
        best_selected_epoch=2,
        best_portable_source=None,
        best_portable_sha256=None,
    )

    assert final_bundle is not None
    assert bundle_paths.final_directory == final_bundle.directory
    assert portable == paths.workspace / "checkpoints/portable/epoch_0002.pt"
    assert events == [
        "state-consensus",
        "common-build",
        "common-stage",
        "best-stage",
        "rank-stage",
        "manifest-commit",
        "portable-export",
    ]


def test_selection_resume_uses_self_contained_bundle_attachment(
    tmp_path: Path,
) -> None:
    config = minimal_config()
    policy = distributed_experiment_runner.MonitorPolicy(
        metric="valid/loss",
        mode="min",
        min_delta=0.0,
    )
    state = distributed_experiment_runner.TrainingFitState(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=1,
        monitor_observations=2,
        stopped_early=False,
        tracking_enabled=True,
        monitor_policy=policy,
        early_stopping_patience=None,
    )
    bundle = DistributedCheckpointBundle(
        directory=tmp_path / "bundle",
        manifest_path=tmp_path / "bundle" / "bundle-manifest.yaml",
        bundle_id="b" * 32,
        completed_epoch=2,
        global_step=4,
        world_size=2,
        local_world_size=2,
        backend="nccl",
        device_type="cuda",
        common_checkpoint_path=tmp_path / "bundle" / "common.pt",
        common_checkpoint_sha256="c" * 64,
        best_portable_checkpoint_path=tmp_path / "bundle" / "best-portable.pt",
        best_portable_selected_epoch=1,
        best_portable_checkpoint_sha256="d" * 64,
    )

    selection = distributed_experiment_runner._initial_selection(
        config,
        validation_available=True,
        resume_payload={
            "metadata": {
                "training_loop": state.to_dict(),
                "distributed_best_portable": {
                    "epoch": 999,
                    "path": "outside-the-bundle.pt",
                    "sha256": "e" * 64,
                },
            }
        },
        resume_bundle=bundle,
    )

    assert selection.best_epoch == 1
    assert selection.best_portable_relative_path is None
    assert selection.best_portable_sha256 == "d" * 64


def test_selection_resume_rejects_missing_bundle_attachment_before_restore(
    tmp_path: Path,
) -> None:
    config = minimal_config()
    policy = distributed_experiment_runner.MonitorPolicy(
        metric="valid/loss",
        mode="min",
        min_delta=0.0,
    )
    state = distributed_experiment_runner.TrainingFitState(
        best_epoch=1,
        best_metric_value=0.5,
        observations_without_improvement=0,
        monitor_observations=1,
        stopped_early=False,
        tracking_enabled=True,
        monitor_policy=policy,
        early_stopping_patience=None,
    )
    bundle = DistributedCheckpointBundle(
        directory=tmp_path / "bundle",
        manifest_path=tmp_path / "bundle" / "bundle-manifest.yaml",
        bundle_id="b" * 32,
        completed_epoch=1,
        global_step=2,
        world_size=2,
        local_world_size=2,
        backend="nccl",
        device_type="cuda",
        common_checkpoint_path=tmp_path / "bundle" / "common.pt",
        common_checkpoint_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="bundled best portable"):
        distributed_experiment_runner._initial_selection(
            config,
            validation_available=True,
            resume_payload={"metadata": {"training_loop": state.to_dict()}},
            resume_bundle=bundle,
        )


def test_rank_runtime_seed_is_stable_distinct_and_numpy_compatible() -> None:
    rank_zero = distributed_experiment_runner._distributed_runtime_seed(42, rank=0)
    rank_one = distributed_experiment_runner._distributed_runtime_seed(42, rank=1)

    assert rank_zero == distributed_experiment_runner._distributed_runtime_seed(
        42,
        rank=0,
    )
    assert rank_zero != rank_one
    assert 0 <= rank_zero <= 2**32 - 1
    assert 0 <= rank_one <= 2**32 - 1


def test_effective_global_batch_is_certified_from_ranked_plan() -> None:
    plan = RankedTrainEpochPlan(
        data_identity=RankedEpochDataIdentity(provider="test", digest="a" * 64),
        plan_digest="b" * 64,
        expected_terminal_token="c" * 64,
        epoch=1,
        rank=0,
        world_size=2,
        microbatches_per_window=4,
        window_count=3,
        samples_per_microbatch=8,
        local_assigned_samples=96,
        global_assigned_samples=192,
        global_dropped_samples=0,
        assignment_digest="d" * 64,
    )

    facts = distributed_experiment_runner._certified_effective_batch(
        plan,
        topology=DistributedTopology(
            rank=0,
            local_rank=0,
            world_size=2,
            local_world_size=2,
        ),
        accumulation=4,
    )

    assert facts == {
        "per_rank_batch_size": 8,
        "gradient_accumulation": 4,
        "effective_global_batch": 64,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config, options: setattr(config, "diagnostics", [object()]),
            "Diagnostics",
        ),
        (
            lambda config, options: setattr(
                config.trainer,
                "test_after_fit",
                True,
            ),
            "test-after-fit",
        ),
        (
            lambda config, options: object.__setattr__(
                options,
                "max_validation_batches",
                1,
            ),
            "complete validation",
        ),
    ],
)
def test_first_ddp_operation_rejects_unsupported_semantics(
    mutate: Any,
    message: str,
) -> None:
    config = minimal_config()
    options = experiment_runner.ExperimentRunOptions(
        num_epochs=2,
        max_train_batches=None,
        max_validation_batches=None,
        max_test_batches=None,
        deterministic=False,
        show_progress=False,
        artifact_verification_workers=None,
        resume_checkpoint=None,
        device=None,
    )
    mutate(config, options)

    with pytest.raises(ValueError, match=message):
        distributed_experiment_runner._validate_operation_scope(
            config,
            options,
            session=cast(DistributedSession, RecordingSession()),
        )
