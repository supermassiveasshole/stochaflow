"""Real two-process Gloo integration for the fixed DDP trainer."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
import torch.distributed as torch_distributed
import torch.multiprocessing as torch_multiprocessing
from torch import nn
from torch.multiprocessing.spawn import ProcessContext
from torch.nn.parallel import DistributedDataParallel

from stochaflow.data.ranked import (
    ExactCoverageReceipt,
    ExactCoverageSpan,
    ExactValidationBatch,
    ExactValidationEpochPlan,
    RankedBatchFacts,
    RankedEpochCompletion,
    RankedEpochDataIdentity,
    RankedTrainEpochPlan,
    RankedTrainWindow,
)
from stochaflow.training.builder import TrainingPlan, TrainingPlanAssembly
from stochaflow.training.ddp_trainer import DDPExecutionBinding, DDPTrainer
from stochaflow.training.distributed import DistributedSession
from stochaflow.training.precision import build_precision_runtime
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput

GlooScenario = Literal[
    "accumulation",
    "backward_error",
    "reader_error",
    "validation_finalization_error",
]

_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 10
_SPAWN_TIMEOUT_SECONDS = 30


def gloo_digest(value: str) -> str:
    """Return one deterministic SHA-256 test identity."""

    return hashlib.sha256(value.encode()).hexdigest()


class GlooRegressionStrategy(TrainingStrategy):
    """Compute one mean squared regression loss through an injected model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(self, batch: Any) -> TrainStepOutput:
        inputs, targets = batch
        prediction = self.model(inputs)
        return TrainStepOutput(
            loss=torch.nn.functional.mse_loss(prediction, targets),
            loss_aggregation_weight=inputs.shape[0],
        )


class GlooRegressionExecutionBindingBuilder:
    """Bind the regression Strategy to the real DDP execution wrapper."""

    def build_primary_execution_module(self, plan: TrainingPlan) -> nn.Module:
        return plan.primary_model

    def bind_primary_execution_model(
        self,
        plan: TrainingPlan,
        execution_model: nn.Module,
    ) -> TrainingStrategy:
        del plan
        return GlooRegressionStrategy(execution_model)


class RecordingDistributedModel(nn.Module):
    """Delegate to real DDP while recording accumulation no-sync entries."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.distributed = DistributedDataParallel(
            module,
            device_ids=None,
            broadcast_buffers=False,
        )
        self.no_sync_entries = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.distributed(value)

    @contextmanager
    def no_sync(self) -> Generator[None]:
        self.no_sync_entries += 1
        with self.distributed.no_sync():
            yield


class GlooTrainReader:
    """Return one complete window or inject one rank-local early read error."""

    def __init__(
        self,
        *,
        plan: RankedTrainEpochPlan,
        window: RankedTrainWindow,
        fail_early: bool,
    ) -> None:
        self.plan = plan
        self.window = window
        self.fail_early = fail_early
        self.read = False
        self.closed = False

    def read_window(self) -> RankedTrainWindow | None:
        if self.fail_early and not self.read:
            self.read = True
            raise RuntimeError("injected rank-one reader failure")
        if self.read:
            return None
        self.read = True
        return self.window

    def finish(self) -> RankedEpochCompletion:
        if not self.read:
            raise RuntimeError("Gloo train reader was not consumed")
        return RankedEpochCompletion(
            plan_digest=self.plan.plan_digest,
            rank=self.plan.rank,
            observed_windows=self.plan.window_count,
            observed_microbatches=self.plan.microbatch_count,
            observed_samples=self.plan.local_assigned_samples,
            assignment_digest=self.plan.assignment_digest,
            terminal_token=self.plan.expected_terminal_token,
        )

    def close(self) -> None:
        self.closed = True


class GlooTrainExecution:
    """Independent public-protocol fake with rank-local opaque batches."""

    def __init__(self, *, rank: int, fail_early: bool) -> None:
        self._plan, self._window = gloo_train_plan_and_window(rank)
        self.batches = self._window.batches
        self.resume_identity = self._plan.data_identity
        self.fail_early = fail_early

    def plan_epoch(
        self,
        epoch: int,
        *,
        microbatches_per_window: int,
        max_microbatches: int | None,
    ) -> RankedTrainEpochPlan:
        if epoch != self._plan.epoch:
            raise ValueError("unexpected Gloo test epoch")
        if microbatches_per_window != self._plan.microbatches_per_window:
            raise ValueError("unexpected Gloo test accumulation")
        if max_microbatches is not None:
            raise ValueError("Gloo test does not accept a batch cap")
        return self._plan

    def open_epoch(self, plan: RankedTrainEpochPlan) -> GlooTrainReader:
        if plan != self._plan:
            raise ValueError("Gloo train plan does not belong to this rank")
        return GlooTrainReader(
            plan=plan,
            window=self._window,
            fail_early=self.fail_early,
        )


class GlooValidationReader:
    """Read rank-zero full validation and optionally forge its receipt digest."""

    def __init__(
        self,
        *,
        plan: ExactValidationEpochPlan,
        batches: tuple[ExactValidationBatch, ...],
        wrong_receipt_digest: bool,
    ) -> None:
        self.plan = plan
        self.batches = batches
        self.wrong_receipt_digest = wrong_receipt_digest
        self.index = 0
        self.closed = False

    def read_batch(self) -> ExactValidationBatch | None:
        if self.index == len(self.batches):
            return None
        item = self.batches[self.index]
        self.index += 1
        return item

    def finish(self) -> ExactCoverageReceipt:
        if self.index != len(self.batches):
            raise RuntimeError("Gloo validation reader was not exhausted")
        return ExactCoverageReceipt(
            plan_digest=(
                gloo_digest("wrong-validation-plan")
                if self.wrong_receipt_digest
                else self.plan.plan_digest
            ),
            rank=self.plan.rank,
            completed_spans=self.plan.local_spans,
            observed_samples=self.plan.local_expected_samples,
        )

    def close(self) -> None:
        self.closed = True


class GlooValidationExecution:
    """Expose full validation on rank zero and an empty peer view."""

    def __init__(self, *, rank: int, wrong_receipt_digest: bool) -> None:
        self._plan, self._batches = gloo_validation_plan_and_batches(rank)
        self.batches = tuple(item.batch for item in self._batches)
        self.coverage_identity = self._plan.coverage_identity
        self.wrong_receipt_digest = wrong_receipt_digest

    def plan_epoch(self, epoch: int) -> ExactValidationEpochPlan:
        if epoch != self._plan.epoch:
            raise ValueError("unexpected Gloo validation epoch")
        return self._plan

    def open_epoch(
        self,
        plan: ExactValidationEpochPlan,
    ) -> GlooValidationReader:
        if plan != self._plan:
            raise ValueError("Gloo validation plan does not belong to this rank")
        return GlooValidationReader(
            plan=plan,
            batches=self._batches,
            wrong_receipt_digest=self.wrong_receipt_digest,
        )


def gloo_train_plan_and_window(
    rank: int,
) -> tuple[RankedTrainEpochPlan, RankedTrainWindow]:
    """Build one equal two-microbatch assignment for a global batch of four."""

    identity = RankedEpochDataIdentity(
        provider="tests.gloo.ranked.v1",
        digest=gloo_digest("two-rank-data-identity"),
    )
    plan = RankedTrainEpochPlan(
        data_identity=identity,
        plan_digest=gloo_digest("two-rank-train-plan"),
        expected_terminal_token=gloo_digest(f"terminal-rank-{rank}"),
        epoch=0,
        rank=rank,
        world_size=_WORLD_SIZE,
        microbatches_per_window=2,
        window_count=1,
        samples_per_microbatch=1,
        local_assigned_samples=2,
        global_assigned_samples=4,
        global_dropped_samples=0,
        assignment_digest=gloo_digest(f"assignments-rank-{rank}"),
    )
    rank_inputs = ((1.0, 2.0), (3.0, 4.0))[rank]
    batches = tuple(
        (
            torch.tensor([[value]], dtype=torch.float32),
            torch.tensor([[2.0 * value]], dtype=torch.float32),
        )
        for value in rank_inputs
    )
    facts = tuple(
        RankedBatchFacts(
            ordinal=ordinal,
            sample_count=1,
            loss_weight=1.0,
            assignment_token=gloo_digest(f"rank-{rank}-batch-{ordinal}"),
        )
        for ordinal in range(2)
    )
    return plan, RankedTrainWindow(
        ordinal=0,
        batches=batches,
        batch_facts=facts,
    )


def gloo_validation_plan_and_batches(
    rank: int,
) -> tuple[ExactValidationEpochPlan, tuple[ExactValidationBatch, ...]]:
    """Build rank-zero full coverage and an exact empty peer assignment."""

    identity = RankedEpochDataIdentity(
        provider="tests.gloo.validation.v1",
        digest=gloo_digest("two-rank-validation-identity"),
    )
    expected_samples = 2
    full_span = ExactCoverageSpan(0, expected_samples)
    spans = (full_span,) if rank == 0 else ()
    plan = ExactValidationEpochPlan(
        coverage_identity=identity,
        plan_digest=gloo_digest("two-rank-validation-plan"),
        epoch=0,
        rank=rank,
        world_size=_WORLD_SIZE,
        global_expected_samples=expected_samples,
        primary_batch_count=1,
        local_expected_samples=expected_samples if rank == 0 else 0,
        local_spans=spans,
    )
    if rank != 0:
        return plan, ()
    batch = (
        torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        torch.tensor([[2.0], [4.0]], dtype=torch.float32),
    )
    return plan, (
        ExactValidationBatch(
            batch=batch,
            facts=RankedBatchFacts(
                ordinal=0,
                sample_count=2,
                loss_weight=2.0,
                assignment_token=gloo_digest("rank-zero-validation-batch"),
            ),
            coverage_span=full_span,
        ),
    )


def gloo_environment(rank: int, port: int) -> dict[str, str]:
    """Build the fixed single-node environment consumed by the real session."""

    return {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(_WORLD_SIZE),
        "LOCAL_WORLD_SIZE": str(_WORLD_SIZE),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "ROLE_RANK": str(rank),
        "ROLE_WORLD_SIZE": str(_WORLD_SIZE),
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
    }


def install_gloo_environment(environment: dict[str, str]) -> None:
    """Replace every distributed launch variable for one spawned worker."""

    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
    ):
        os.environ.pop(name, None)
    os.environ.update(environment)


def build_gloo_trainer(
    session: DistributedSession,
    *,
    fail_backward: bool = False,
) -> tuple[DDPTrainer, nn.Linear, RecordingDistributedModel]:
    """Build one canonical model and bind a real CPU DDP execution model."""

    canonical = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        canonical.weight.fill_(0.5)
    plan = TrainingPlan(
        strategy=GlooRegressionStrategy(canonical),
        primary_model=canonical,
    )
    assembly = TrainingPlanAssembly(
        plan=plan,
        builder_name="tests.gloo.regression",
        _execution_binding=GlooRegressionExecutionBindingBuilder(),
    )
    binding = DDPExecutionBinding.from_assembly(
        assembly,
        wrap=RecordingDistributedModel,
    )
    execution_model = binding.execution_model
    if not isinstance(execution_model, RecordingDistributedModel):
        raise TypeError("Gloo test wrapper returned an unexpected model")
    optimizer = torch.optim.SGD(canonical.parameters(), lr=0.1)
    if fail_backward and session.topology.rank == 1:
        backward_calls = 0

        def fail_final_backward(gradient: torch.Tensor) -> torch.Tensor:
            nonlocal backward_calls
            backward_calls += 1
            if backward_calls == 2:
                raise RuntimeError("injected rank-one backward failure")
            return gradient

        canonical.weight.register_hook(fail_final_backward)
    trainer = DDPTrainer(
        binding=binding,
        optimizer=optimizer,
        collectives=session.collectives,
        topology=session.topology,
        device=session.device,
        precision=build_precision_runtime("fp32", session.device),
        accumulate_grad_batches=2,
    )
    return trainer, canonical, execution_model


def write_gloo_result(output_directory: str, rank: int, value: object) -> None:
    """Write one rank-owned bounded result file for parent-process assertions."""

    path = Path(output_directory) / f"rank-{rank}.json"
    path.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )


def run_gloo_scenario(
    rank: int,
    scenario: GlooScenario,
    port: int,
) -> dict[str, object]:
    """Run one scenario inside an active real Gloo process group."""

    environment = gloo_environment(rank, port)
    install_gloo_environment(environment)
    payload: dict[str, object] = {
        "rank": rank,
        "scenario": scenario,
        "error_type": None,
        "error_message": None,
    }
    with DistributedSession.from_environment(
        backend="gloo",
        timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        environ=environment,
    ) as session:
        trainer, canonical, execution_model = build_gloo_trainer(
            session,
            fail_backward=scenario == "backward_error",
        )
        try:
            if scenario == "validation_finalization_error":
                trainer.evaluate_epoch(
                    GlooValidationExecution(
                        rank=rank,
                        wrong_receipt_digest=rank == 0,
                    ),
                    epoch_index=0,
                )
            else:
                result = trainer.train_epoch(
                    GlooTrainExecution(
                        rank=rank,
                        fail_early=scenario == "reader_error" and rank == 1,
                    ),
                    epoch_index=0,
                )
                payload["metrics"] = dict(result.metrics)
        except BaseException as error:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
            if scenario == "backward_error":
                raise

        parameters = [
            float(value)
            for value in canonical.weight.detach().cpu().flatten().tolist()
        ]
        payload["parameters"] = parameters
        payload["parameters_consistent"] = session.collectives.all_equal(parameters)
        payload["global_step"] = trainer.global_step
        payload["no_sync_entries"] = execution_model.no_sync_entries
    return payload


def gloo_spawn_worker(
    rank: int,
    scenario: GlooScenario,
    port: int,
    output_directory: str,
) -> None:
    """Top-level Windows-spawn-safe worker with explicit failure evidence."""

    try:
        result = run_gloo_scenario(rank, scenario, port)
    except BaseException as error:
        write_gloo_result(
            output_directory,
            rank,
            {
                "rank": rank,
                "scenario": scenario,
                "unexpected_error_type": type(error).__name__,
                "unexpected_error_message": str(error),
            },
        )
        raise
    write_gloo_result(output_directory, rank, result)


def free_local_port() -> int:
    """Reserve and release one loopback port immediately before spawning."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def terminate_spawned_processes(processes: list[Any]) -> None:
    """Bound cleanup for a failed or hung integration-test process group."""

    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5.0)


def spawn_gloo_scenario(
    scenario: GlooScenario,
    output_directory: Path,
) -> list[dict[str, object]]:
    """Spawn two ranks and fail with bounded cleanup instead of hanging pytest."""

    output_directory.mkdir(parents=True)
    process_context = cast(
        ProcessContext,
        torch_multiprocessing.spawn(
            gloo_spawn_worker,
            args=(scenario, free_local_port(), str(output_directory)),
            nprocs=_WORLD_SIZE,
            join=False,
            daemon=False,
            start_method="spawn",
        ),
    )
    deadline = time.monotonic() + _SPAWN_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if process_context.join(timeout=0.25, grace_period=2.0):
                break
        else:
            terminate_spawned_processes(process_context.processes)
            pytest.fail(
                f"two-rank Gloo scenario {scenario!r} exceeded "
                f"{_SPAWN_TIMEOUT_SECONDS} seconds"
            )
    except BaseException:
        terminate_spawned_processes(process_context.processes)
        raise

    results = [
        json.loads(
            (output_directory / f"rank-{rank}.json").read_text(encoding="utf-8")
        )
        for rank in range(_WORLD_SIZE)
    ]
    assert [result["rank"] for result in results] == [0, 1]
    assert not any("unexpected_error_type" in result for result in results)
    return results


def reference_global_batch_weight() -> float:
    """Apply one SGD step to the same effective global batch without DDP."""

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    targets = 2.0 * inputs
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return float(model.weight.detach().item())


GLOO_PROCESS_ACCEPTANCE_UNAVAILABLE = (
    not torch_distributed.is_available()
    or not torch_distributed.is_gloo_available()
    or sys.platform == "darwin"
)
GLOO_PROCESS_ACCEPTANCE_REASON = (
    "two-rank Gloo process acceptance requires a supported Linux or Windows "
    "Gloo build"
)


@pytest.mark.skipif(
    GLOO_PROCESS_ACCEPTANCE_UNAVAILABLE,
    reason=GLOO_PROCESS_ACCEPTANCE_REASON,
)
def test_real_gloo_accumulation_matches_one_effective_global_batch(
    tmp_path: Path,
) -> None:
    results = spawn_gloo_scenario("accumulation", tmp_path / "accumulation")
    expected_weight = reference_global_batch_weight()

    assert all(result["error_type"] is None for result in results)
    assert all(result["parameters_consistent"] is True for result in results)
    assert all(result["global_step"] == 1 for result in results)
    assert all(result["no_sync_entries"] == 1 for result in results)
    parameters = [cast(list[float], result["parameters"]) for result in results]
    assert [value[0] for value in parameters] == pytest.approx(
        [expected_weight, expected_weight]
    )


@pytest.mark.skipif(
    GLOO_PROCESS_ACCEPTANCE_UNAVAILABLE,
    reason=GLOO_PROCESS_ACCEPTANCE_REASON,
)
def test_real_gloo_rank_local_reader_error_exits_every_rank_without_update(
    tmp_path: Path,
) -> None:
    results = spawn_gloo_scenario("reader_error", tmp_path / "reader-error")

    assert all(result["error_type"] is not None for result in results)
    assert "training data window 0 failed on another rank" in str(
        results[0]["error_message"]
    )
    assert "injected rank-one reader failure" in str(results[1]["error_message"])
    assert all(result["parameters_consistent"] is True for result in results)
    assert all(result["parameters"] == [0.5] for result in results)
    assert all(result["global_step"] == 0 for result in results)
    assert all(result["no_sync_entries"] == 0 for result in results)


@pytest.mark.skipif(
    GLOO_PROCESS_ACCEPTANCE_UNAVAILABLE,
    reason=GLOO_PROCESS_ACCEPTANCE_REASON,
)
def test_real_gloo_rank_local_backward_error_has_bounded_launcher_exit(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "backward-error"
    output_directory.mkdir(parents=True)
    process_context = cast(
        ProcessContext,
        torch_multiprocessing.spawn(
            gloo_spawn_worker,
            args=("backward_error", free_local_port(), str(output_directory)),
            nprocs=_WORLD_SIZE,
            join=False,
            daemon=False,
            start_method="spawn",
        ),
    )
    deadline = time.monotonic() + _SPAWN_TIMEOUT_SECONDS
    exited = False
    try:
        while time.monotonic() < deadline:
            try:
                if process_context.join(timeout=0.25, grace_period=2.0):
                    exited = True
                    break
            except Exception:  # noqa: BLE001 - child failure is the expected outcome
                exited = True
                break
        assert exited, "rank-local backward failure left the Gloo job hanging"
    finally:
        terminate_spawned_processes(process_context.processes)

    rank_one = json.loads(
        (output_directory / "rank-1.json").read_text(encoding="utf-8")
    )
    assert (
        rank_one.get("error_message") == "injected rank-one backward failure"
        or rank_one.get("unexpected_error_message")
        == "injected rank-one backward failure"
    )


@pytest.mark.skipif(
    GLOO_PROCESS_ACCEPTANCE_UNAVAILABLE,
    reason=GLOO_PROCESS_ACCEPTANCE_REASON,
)
def test_real_gloo_rank_zero_validation_finalization_error_exits_every_rank(
    tmp_path: Path,
) -> None:
    results = spawn_gloo_scenario(
        "validation_finalization_error",
        tmp_path / "validation-finalization-error",
    )

    assert all(result["error_type"] is not None for result in results)
    assert "validation receipt does not match its plan" in str(
        results[0]["error_message"]
    )
    assert "validation finalization failed on another rank" in str(
        results[1]["error_message"]
    )
    assert all(result["parameters_consistent"] is True for result in results)
    assert all(result["parameters"] == [0.5] for result in results)
    assert all(result["global_step"] == 0 for result in results)
