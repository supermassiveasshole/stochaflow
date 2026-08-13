"""Fixed-topology distributed checkpoint bundle contracts."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml
from torch import nn

from stochaflow.data.ranked import (
    RankedEpochDataIdentity,
    RankedTrainEpochPlan,
)
from stochaflow.inference.checkpoint import project_inference_checkpoint
from stochaflow.training.distributed import checkpoint_bundle
from stochaflow.training.distributed.checkpoint_bundle import (
    DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME,
    DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION,
    DISTRIBUTED_CHECKPOINT_MANIFEST_NAME,
    DISTRIBUTED_COMMON_CHECKPOINT_ROLE,
    DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE,
    DistributedCheckpointBundle,
    DistributedCheckpointBundlePaths,
    apply_distributed_checkpoint_restore,
    commit_distributed_checkpoint_bundle,
    distributed_checkpoint_bundle_paths,
    export_distributed_portable_checkpoint,
    load_distributed_checkpoint_bundle,
    preflight_distributed_checkpoint_bundle,
    stage_distributed_best_portable_checkpoint,
    stage_distributed_common_checkpoint,
    stage_distributed_rank_checkpoint,
)
from stochaflow.training.distributed.contracts import DistributedTopology
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    ParsedRNGState,
    capture_local_rng_state,
)

_TEST_RUNTIME_DIGEST = "a" * 64


def checkpoint_paths(
    root: Path,
    *,
    bundle_id: str = "1" * 32,
    completed_epoch: int = 2,
    global_step: int = 8,
) -> DistributedCheckpointBundlePaths:
    """Return one deterministic epoch-boundary bundle path fixture."""

    return distributed_checkpoint_bundle_paths(
        root,
        bundle_id=bundle_id,
        completed_epoch=completed_epoch,
        global_step=global_step,
    )


def topology(rank: int, *, world_size: int = 2) -> DistributedTopology:
    """Return one fixed single-node topology fixture."""

    return DistributedTopology(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        local_world_size=world_size,
    )


def common_payload(
    manager: CheckpointManager,
    *,
    epoch: int = 2,
    global_step: int = 8,
) -> dict[str, object]:
    """Build common state carrying its required saved-runtime authority."""

    return cast(
        dict[str, object],
        manager.build_state(
            epoch=epoch,
            global_step=global_step,
            metadata={"common_runtime_state_sha256": _TEST_RUNTIME_DIGEST},
        ),
    )


def next_plan(
    rank: int,
    *,
    world_size: int = 2,
    identity_digest: str = "a" * 64,
    plan_digest: str = "b" * 64,
    epoch: int = 3,
) -> RankedTrainEpochPlan:
    """Return one exact rank-local next-epoch plan fixture."""

    local_samples = 2 * 2 * 3
    return RankedTrainEpochPlan(
        data_identity=RankedEpochDataIdentity(
            provider="tests.fixed-ranked-data.v1",
            digest=identity_digest,
        ),
        plan_digest=plan_digest,
        expected_terminal_token=(f"{rank + 1:x}" * 64),
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        microbatches_per_window=2,
        window_count=2,
        samples_per_microbatch=3,
        local_assigned_samples=local_samples,
        global_assigned_samples=local_samples * world_size,
        global_dropped_samples=1,
        assignment_digest=(f"{rank + 3:x}" * 64),
        requested_max_microbatches=4,
    )


def checkpoint_manager(*, precision_kind: str = "fp32") -> CheckpointManager:
    """Return a stateful v12 manager used by bundle tests."""

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, momentum=0.9)
    return CheckpointManager(
        model=model,
        optimizer=optimizer,
        precision_kind=precision_kind,
    )


def stage_complete_bundle(
    root: Path,
    *,
    bundle_id: str = "1" * 32,
    manager: CheckpointManager | None = None,
    completed_epoch: int = 2,
    global_step: int = 8,
    include_best_portable: bool = False,
    best_portable_source: str | Path | None = None,
    best_epoch: int | None = None,
) -> tuple[
    CheckpointManager,
    DistributedCheckpointBundlePaths,
    DistributedCheckpointBundle,
]:
    """Stage and commit one two-rank CPU/Gloo bundle."""

    runtime_manager = checkpoint_manager() if manager is None else manager
    paths = checkpoint_paths(
        root,
        bundle_id=bundle_id,
        completed_epoch=completed_epoch,
        global_step=global_step,
    )
    payload = runtime_manager.build_state(
        epoch=completed_epoch,
        global_step=global_step,
        config={
            "trainer": {
                "precision": runtime_manager.precision_kind,
                "accumulate_grad_batches": 2,
            }
        },
        metrics={"valid/loss": 0.5},
        metadata={
            "run_identity": "tests.distributed",
            "common_runtime_state_sha256": _TEST_RUNTIME_DIGEST,
        },
    )
    stage_distributed_common_checkpoint(paths, payload)
    if include_best_portable:
        stage_distributed_best_portable_checkpoint(
            paths,
            selected_epoch=(
                completed_epoch if best_epoch is None else best_epoch
            ),
            source_portable_checkpoint=best_portable_source,
            expected_source_sha256=(
                checkpoint_bundle._file_sha256(Path(best_portable_source))
                if best_portable_source is not None
                else None
            ),
        )
    for rank in range(2):
        stage_distributed_rank_checkpoint(
            paths,
            topology=topology(rank),
            next_plan=next_plan(rank, epoch=completed_epoch + 1),
            rng_state=capture_local_rng_state("cpu"),
        )
    bundle = commit_distributed_checkpoint_bundle(
        paths,
        topology=topology(0),
        backend="gloo",
        device_type="cpu",
    )
    return runtime_manager, paths, bundle


def test_bundle_publishes_manifest_only_after_complete_inventory(tmp_path: Path) -> None:
    manager = checkpoint_manager()
    paths = checkpoint_paths(tmp_path / "resume")
    payload = common_payload(manager)

    common = stage_distributed_common_checkpoint(paths, payload)
    rank_files = [
        stage_distributed_rank_checkpoint(
            paths,
            topology=topology(rank),
            next_plan=next_plan(rank),
        )
        for rank in range(2)
    ]

    assert common.relative_path == "common.pt"
    assert [item.relative_path for item in rank_files] == [
        "ranks/rank-00000.pt",
        "ranks/rank-00001.pt",
    ]
    assert not (
        paths.staging_directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    ).exists()
    assert not paths.final_directory.exists()

    bundle = commit_distributed_checkpoint_bundle(
        paths,
        topology=topology(0),
        backend="gloo",
        device_type="cpu",
    )

    assert not paths.staging_directory.exists()
    assert bundle.directory == paths.final_directory
    assert bundle.manifest_path.is_file()
    manifest = yaml.safe_load(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == DISTRIBUTED_CHECKPOINT_BUNDLE_FORMAT_VERSION
    assert manifest["status"] == "committed"
    assert manifest["best_portable"] is None
    assert manifest["topology"]["rank_mapping"] == [
        {"rank": 0, "local_rank": 0},
        {"rank": 1, "local_rank": 1},
    ]
    assert [item["rank"] for item in manifest["rank_local"]] == [0, 1]
    assert bundle.best_portable_checkpoint_path is None
    assert bundle.best_portable_selected_epoch is None
    assert bundle.best_portable_checkpoint_sha256 is None


def test_commit_failure_never_publishes_or_leaves_a_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = checkpoint_manager()
    paths = checkpoint_paths(tmp_path / "resume")
    payload = common_payload(manager)
    stage_distributed_common_checkpoint(paths, payload)
    for rank in range(2):
        stage_distributed_rank_checkpoint(
            paths,
            topology=topology(rank),
            next_plan=next_plan(rank),
        )

    def fail_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
        del manifest
        manifest_path = Path(path)
        manifest_path.write_text("partial: true\n", encoding="utf-8")
        raise OSError("manifest publication failed")

    monkeypatch.setattr(checkpoint_bundle, "write_yaml_manifest", fail_manifest)

    with pytest.raises(OSError, match="manifest publication failed"):
        commit_distributed_checkpoint_bundle(
            paths,
            topology=topology(0),
            backend="gloo",
            device_type="cpu",
        )

    assert not paths.final_directory.exists()
    assert not (
        paths.staging_directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    ).exists()
    assert (paths.staging_directory / "common.pt").is_file()


def test_incomplete_bundle_cannot_be_committed_or_discovered(tmp_path: Path) -> None:
    manager = checkpoint_manager()
    paths = checkpoint_paths(tmp_path / "resume")
    stage_distributed_common_checkpoint(
        paths,
        common_payload(manager),
    )
    stage_distributed_rank_checkpoint(
        paths,
        topology=topology(0),
        next_plan=next_plan(0),
    )

    with pytest.raises(FileNotFoundError):
        commit_distributed_checkpoint_bundle(
            paths,
            topology=topology(0),
            backend="gloo",
            device_type="cpu",
        )

    assert not paths.final_directory.exists()
    assert not (
        paths.staging_directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    ).exists()
    with pytest.raises(ValueError, match="staging is not resumable"):
        load_distributed_checkpoint_bundle(
            paths.staging_directory,
            topology=topology(0),
            backend="gloo",
            device_type="cpu",
            fresh_next_plan=next_plan(0),
        )


def test_only_rank_zero_can_commit_and_extra_files_fail_closed(
    tmp_path: Path,
) -> None:
    manager = checkpoint_manager()
    paths = checkpoint_paths(tmp_path / "resume")
    stage_distributed_common_checkpoint(
        paths,
        common_payload(manager),
    )
    for rank in range(2):
        stage_distributed_rank_checkpoint(
            paths,
            topology=topology(rank),
            next_plan=next_plan(rank),
        )

    with pytest.raises(PermissionError, match="rank zero"):
        commit_distributed_checkpoint_bundle(
            paths,
            topology=topology(1),
            backend="gloo",
            device_type="cpu",
        )

    (paths.staging_directory / "unlisted.pt").write_bytes(b"not inventory")
    with pytest.raises(ValueError, match="invalid file inventory"):
        commit_distributed_checkpoint_bundle(
            paths,
            topology=topology(0),
            backend="gloo",
            device_type="cpu",
        )
    assert not paths.final_directory.exists()
    assert not (
        paths.staging_directory / DISTRIBUTED_CHECKPOINT_MANIFEST_NAME
    ).exists()


def test_restore_requires_fixed_topology_and_fresh_exact_next_plan(
    tmp_path: Path,
) -> None:
    _, _, bundle = stage_complete_bundle(tmp_path / "resume")

    restored = load_distributed_checkpoint_bundle(
        bundle.directory,
        topology=topology(1),
        backend="gloo",
        device_type="cpu",
        fresh_next_plan=next_plan(1),
    )

    assert restored.bundle.bundle_id == "1" * 32
    assert restored.next_plan.rank == 1
    assert restored.next_plan.epoch == 3

    wrong_identity = replace(
        next_plan(1),
        data_identity=RankedEpochDataIdentity(
            provider="tests.fixed-ranked-data.v1",
            digest="c" * 64,
        ),
    )
    with pytest.raises(ValueError, match="freshly verified runtime plan"):
        load_distributed_checkpoint_bundle(
            bundle.directory,
            topology=topology(1),
            backend="gloo",
            device_type="cpu",
            fresh_next_plan=wrong_identity,
        )

    with pytest.raises(ValueError, match="world/local-world size"):
        load_distributed_checkpoint_bundle(
            bundle.directory,
            topology=topology(0, world_size=1),
            backend="gloo",
            device_type="cpu",
            fresh_next_plan=next_plan(0, world_size=1),
        )


def test_preflight_exposes_common_config_but_cannot_apply_state(
    tmp_path: Path,
) -> None:
    _, _, bundle = stage_complete_bundle(tmp_path / "resume")

    preflight = preflight_distributed_checkpoint_bundle(bundle.directory)

    assert preflight.bundle == bundle
    assert preflight.common_payload.get("config") == {
        "trainer": {
            "precision": "fp32",
            "accumulate_grad_batches": 2,
        }
    }
    assert not hasattr(preflight, "rank_rng_state")
    assert not hasattr(preflight, "next_plan")


def test_current_best_portable_is_projected_into_the_bundle(
    tmp_path: Path,
) -> None:
    _, _, bundle = stage_complete_bundle(
        tmp_path / "resume",
        include_best_portable=True,
    )

    assert bundle.best_portable_checkpoint_path == (
        bundle.directory / DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME
    )
    best_path = bundle.best_portable_checkpoint_path
    assert isinstance(best_path, Path)
    assert bundle.best_portable_selected_epoch == 2
    assert bundle.best_portable_checkpoint_sha256 is not None
    manifest = yaml.safe_load(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["best_portable"] == {
        "selected_epoch": 2,
        "path": DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME,
        "size": best_path.stat().st_size,
        "sha256": bundle.best_portable_checkpoint_sha256,
    }
    payload = CheckpointManager.load_payload(
        best_path,
        map_location="cpu",
    )
    metadata = cast(dict[str, Any], payload.get("metadata"))
    assert payload.get("epoch") == 2
    assert metadata["checkpoint_role"] == DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE
    assert metadata["distributed_portable_attachment_source"] == {
        "bundle_id": bundle.bundle_id,
        "common_sha256": bundle.common_checkpoint_sha256,
        "completed_epoch": 2,
        "global_step": 8,
    }
    assert "optimizer_state_dict" not in payload

    preflight = preflight_distributed_checkpoint_bundle(bundle.directory)
    restored = load_distributed_checkpoint_bundle(
        bundle.directory,
        topology=topology(0),
        backend="gloo",
        device_type="cpu",
        fresh_next_plan=next_plan(0),
    )
    assert preflight.bundle.best_portable_checkpoint_path == (
        bundle.best_portable_checkpoint_path
    )
    assert restored.bundle.best_portable_selected_epoch == 2
    assert restored.bundle.best_portable_checkpoint_sha256 == (
        bundle.best_portable_checkpoint_sha256
    )


def test_earlier_best_portable_is_copied_into_a_later_bundle(
    tmp_path: Path,
) -> None:
    _, _, earlier = stage_complete_bundle(
        tmp_path / "resume",
        bundle_id="1" * 32,
        completed_epoch=1,
        global_step=4,
    )
    earlier_portable = tmp_path / "portable" / "epoch-1.pt"
    export_distributed_portable_checkpoint(earlier.directory, earlier_portable)

    _, _, later = stage_complete_bundle(
        tmp_path / "resume",
        bundle_id="2" * 32,
        completed_epoch=2,
        global_step=8,
        include_best_portable=True,
        best_portable_source=earlier_portable,
        best_epoch=1,
    )

    assert later.best_portable_selected_epoch == 1
    assert later.best_portable_checkpoint_path is not None
    copied = CheckpointManager.load_payload(
        later.best_portable_checkpoint_path,
        map_location="cpu",
    )
    source = CheckpointManager.load_payload(earlier_portable, map_location="cpu")
    assert copied.get("epoch") == 1
    assert torch.equal(
        cast(dict[str, torch.Tensor], copied.get("model_state_dict"))["weight"],
        cast(dict[str, torch.Tensor], source.get("model_state_dict"))["weight"],
    )


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_copied_bundle_detects_missing_or_tampered_best_attachment(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, _, bundle = stage_complete_bundle(
        tmp_path / "resume",
        include_best_portable=True,
    )
    copied_directory = tmp_path / f"copied-{mutation}"
    shutil.copytree(bundle.directory, copied_directory)
    copied_best = copied_directory / DISTRIBUTED_BEST_PORTABLE_CHECKPOINT_NAME
    if mutation == "missing":
        copied_best.unlink()
    else:
        contents = bytearray(copied_best.read_bytes())
        contents[-1] ^= 1
        copied_best.write_bytes(contents)

    expected = "file is missing" if mutation == "missing" else "file digest"
    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        preflight_distributed_checkpoint_bundle(copied_directory)


def test_staging_best_portable_rejects_wrong_role_or_epoch(
    tmp_path: Path,
) -> None:
    manager = checkpoint_manager()
    paths = checkpoint_paths(tmp_path / "resume")
    stage_distributed_common_checkpoint(
        paths,
        common_payload(manager),
    )
    ordinary = tmp_path / "ordinary.pt"
    CheckpointManager.save_payload(
        manager.build_state(epoch=1, global_step=4),
        ordinary,
    )
    with pytest.raises(ValueError, match="wrong checkpoint role"):
        stage_distributed_best_portable_checkpoint(
            paths,
            selected_epoch=1,
            source_portable_checkpoint=ordinary,
            expected_source_sha256=checkpoint_bundle._file_sha256(ordinary),
        )

    _, _, source_bundle = stage_complete_bundle(
        tmp_path / "source-resume",
        completed_epoch=1,
        global_step=4,
    )
    portable = tmp_path / "portable.pt"
    export_distributed_portable_checkpoint(source_bundle.directory, portable)
    with pytest.raises(ValueError, match="epoch does not match"):
        stage_distributed_best_portable_checkpoint(
            paths,
            selected_epoch=2,
            source_portable_checkpoint=portable,
            expected_source_sha256=checkpoint_bundle._file_sha256(portable),
        )


def test_bundle_contains_only_plan_state_not_runtime_receipts(tmp_path: Path) -> None:
    _, _, bundle = stage_complete_bundle(tmp_path / "resume")
    manifest = yaml.safe_load(bundle.manifest_path.read_text(encoding="utf-8"))
    rank_path = bundle.directory / manifest["rank_local"][0]["path"]
    rank_payload = torch.load(rank_path, map_location="cpu", weights_only=True)

    assert set(rank_payload) == {
        "format_version",
        "kind",
        "bundle_id",
        "completed_epoch",
        "global_step",
        "rank",
        "world_size",
        "rng_state",
        "next_plan",
    }
    assert set(rank_payload["next_plan"]["data_identity"]) == {
        "provider",
        "digest",
    }
    assert rank_payload["next_plan"]["assignment_digest"] == "3" * 64
    assert all("receipt" not in str(key).casefold() for key in rank_payload)
    assert "receipt" not in bundle.manifest_path.read_text(
        encoding="utf-8"
    ).casefold()


def test_restore_rejects_any_corrupted_inventory_file(tmp_path: Path) -> None:
    _, _, bundle = stage_complete_bundle(tmp_path / "resume")
    rank_path = bundle.directory / "ranks" / "rank-00001.pt"
    with rank_path.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="file size does not match"):
        load_distributed_checkpoint_bundle(
            bundle.directory,
            topology=topology(0),
            backend="gloo",
            device_type="cpu",
            fresh_next_plan=next_plan(0),
        )


def test_apply_restores_common_v12_state_after_all_validation(tmp_path: Path) -> None:
    manager = checkpoint_manager()
    model = cast(nn.Linear, manager.model)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(4.0)
    _, _, bundle = stage_complete_bundle(
        tmp_path / "resume",
        manager=manager,
    )
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    restore = load_distributed_checkpoint_bundle(
        bundle.directory,
        topology=topology(0),
        backend="gloo",
        device_type="cpu",
        fresh_next_plan=next_plan(0),
    )
    loaded = apply_distributed_checkpoint_restore(
        restore,
        checkpoint_manager=manager,
    )

    assert loaded.epoch == 2
    assert loaded.global_step == 8
    assert torch.equal(model.weight, torch.full_like(model.weight, 3))
    assert torch.equal(model.bias, torch.full_like(model.bias, 4))


def test_rng_apply_failure_rolls_common_state_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = checkpoint_manager()
    model = cast(nn.Linear, manager.model)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(4.0)
    _, _, bundle = stage_complete_bundle(
        tmp_path / "resume",
        manager=manager,
    )
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    restore = load_distributed_checkpoint_bundle(
        bundle.directory,
        topology=topology(0),
        backend="gloo",
        device_type="cpu",
        fresh_next_plan=next_plan(0),
    )
    original_restore_rng = checkpoint_bundle.restore_local_rng_state
    calls = 0

    def fail_first_rng_restore(
        state: ParsedRNGState,
        *,
        device: torch.device | str,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected RNG restore failure")
        original_restore_rng(
            state,
            device=device,
        )

    monkeypatch.setattr(
        checkpoint_bundle,
        "restore_local_rng_state",
        fail_first_rng_restore,
    )

    with pytest.raises(RuntimeError, match="injected RNG restore failure"):
        apply_distributed_checkpoint_restore(
            restore,
            checkpoint_manager=manager,
        )

    assert calls == 2
    assert torch.equal(model.weight, torch.zeros_like(model.weight))
    assert torch.equal(model.bias, torch.zeros_like(model.bias))


@pytest.mark.parametrize("precision_kind", ["fp32", "bf16-mixed"])
def test_portable_export_is_valid_v12_and_retryable(
    tmp_path: Path,
    precision_kind: str,
) -> None:
    manager = checkpoint_manager(precision_kind=precision_kind)
    _, _, bundle = stage_complete_bundle(
        tmp_path / "resume",
        manager=manager,
    )
    destination = tmp_path / "portable" / "latest.pt"

    first = export_distributed_portable_checkpoint(bundle.directory, destination)
    second = export_distributed_portable_checkpoint(bundle.directory, destination)
    payload = CheckpointManager.load_payload(destination, map_location="cpu")
    metadata = payload.get("metadata")

    assert first == second == destination
    assert isinstance(metadata, dict)
    assert metadata["checkpoint_role"] == DISTRIBUTED_PORTABLE_CHECKPOINT_ROLE
    assert metadata["distributed_portable_source"] == {
        "bundle_id": bundle.bundle_id,
        "manifest_sha256": metadata["distributed_portable_source"][
            "manifest_sha256"
        ],
        "common_sha256": bundle.common_checkpoint_sha256,
        "completed_epoch": 2,
        "global_step": 8,
    }
    assert metadata["rng_authority"] == "v12_format_compatibility_only"
    assert payload.get("precision_kind") == precision_kind
    assert "rng_state" in payload
    assert "optimizer_class" not in payload
    assert "optimizer_state_dict" not in payload
    assert "lr_scheduler_class" not in payload
    assert "lr_scheduler_state_dict" not in payload
    assert "objective_state_dict" not in payload
    inference = project_inference_checkpoint(payload)
    assert "rng_state" not in inference
    assert "optimizer_state_dict" not in inference


def test_portable_collision_does_not_replace_another_bundle(tmp_path: Path) -> None:
    _, _, first_bundle = stage_complete_bundle(
        tmp_path / "resume",
        bundle_id="1" * 32,
    )
    _, _, second_bundle = stage_complete_bundle(
        tmp_path / "resume",
        bundle_id="2" * 32,
    )
    destination = tmp_path / "latest.pt"
    export_distributed_portable_checkpoint(first_bundle.directory, destination)
    original = destination.read_bytes()

    with pytest.raises(FileExistsError, match="different snapshot"):
        export_distributed_portable_checkpoint(
            second_bundle.directory,
            destination,
        )

    assert destination.read_bytes() == original


def test_portable_retry_rejects_self_reported_source_with_wrong_projection(
    tmp_path: Path,
) -> None:
    _, _, bundle = stage_complete_bundle(tmp_path / "resume")
    destination = tmp_path / "latest.pt"
    export_distributed_portable_checkpoint(bundle.directory, destination)
    payload = CheckpointManager.load_payload(destination, map_location="cpu")
    model_state = cast(dict[str, torch.Tensor], payload.get("model_state_dict"))
    model_state["weight"] = model_state["weight"] + 1
    CheckpointManager.save_payload(payload, destination)

    with pytest.raises(FileExistsError, match="exact projection"):
        export_distributed_portable_checkpoint(bundle.directory, destination)


def test_common_checkpoint_remains_valid_v12_with_explicit_role(
    tmp_path: Path,
) -> None:
    _, _, bundle = stage_complete_bundle(tmp_path / "resume")
    common = CheckpointManager.load_payload(
        bundle.common_checkpoint_path,
        map_location="cpu",
    )
    metadata = common.get("metadata")

    assert isinstance(metadata, dict)
    assert metadata["checkpoint_role"] == DISTRIBUTED_COMMON_CHECKPOINT_ROLE
    assert metadata["rng_authority"] == "rank_local_bundle_files"
    assert common.get("epoch") == 2
    assert common.get("global_step") == 8
