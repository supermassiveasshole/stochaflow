"""Behavioral tests for the independent fixed-topology DDP trainer."""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch
from torch import nn

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
from stochaflow.processes.base import Process
from stochaflow.training.builder import TrainingPlan, TrainingPlanAssembly
from stochaflow.training.ddp_trainer import DDPExecutionBinding, DDPTrainer
from stochaflow.training.distributed import DistributedTopology
from stochaflow.training.precision import build_precision_runtime
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class RecordingExecutionModel(nn.Module):
    """Small state-sharing execution wrapper with visible no-sync scopes."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.no_sync_active = False
        self.forward_scopes: list[bool] = []
        self.backward_scopes: list[bool] = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(self._record_backward_scope)

    def _record_backward_scope(self, gradient: torch.Tensor) -> torch.Tensor:
        self.backward_scopes.append(self.no_sync_active)
        return gradient

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.forward_scopes.append(self.no_sync_active)
        return self.model(value)

    @contextmanager
    def no_sync(self) -> Generator[None]:
        assert not self.no_sync_active
        self.no_sync_active = True
        try:
            yield
        finally:
            self.no_sync_active = False


class RegressionStrategy(TrainingStrategy):
    """Compute a mean squared loss through one injected model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(self, batch: Any) -> TrainStepOutput:
        inputs, targets = batch
        prediction = self.model(inputs)
        return TrainStepOutput(
            loss=torch.nn.functional.mse_loss(prediction, targets),
            loss_aggregation_weight=inputs.shape[0],
        )


class FrozenResumeProcess(Process):
    """Frozen state used to prove restore-version baselines are one-shot."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("coefficient", torch.tensor(1.0))


class UnusedParameterRegressionModel(nn.Module):
    """Keep one trainable parameter outside the forward graph."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5]]))
        self.unused = nn.Parameter(torch.tensor([[1.5]]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


class RegressionExecutionBindingBuilder:
    """Independent test Builder capability for execution rebinding."""

    def build_primary_execution_module(self, plan: TrainingPlan) -> nn.Module:
        return plan.primary_model

    def bind_primary_execution_model(
        self,
        plan: TrainingPlan,
        execution_model: nn.Module,
    ) -> TrainingStrategy:
        del plan
        return RegressionStrategy(execution_model)


class SingleRankCollectives:
    """Exact world-size-one collective oracle."""

    def broadcast_from_primary(self, value: object) -> object:
        return value

    def gather_to_primary(self, value: object) -> tuple[object, ...]:
        return (value,)

    def all_true(self, value: bool) -> bool:
        return value

    def all_equal(self, value: object) -> bool:
        del value
        return True

    def sum_int(self, value: int) -> int:
        return value

    def sum_float(self, value: float) -> float:
        return value

    def min_int(self, value: int) -> int:
        return value

    def max_int(self, value: int) -> int:
        return value


class RestoreMismatchCollectives(SingleRankCollectives):
    """Reject only the restored common-state authority comparison."""

    def all_equal(self, value: object) -> bool:
        return not (
            isinstance(value, dict)
            and "common_checkpoint_sha256" in value
        )


@dataclass
class TrainReader:
    plan: RankedTrainEpochPlan
    windows: list[RankedTrainWindow]
    index: int = 0
    closed: bool = False

    def read_window(self) -> RankedTrainWindow | None:
        if self.index == len(self.windows):
            return None
        value = self.windows[self.index]
        self.index += 1
        return value

    def finish(self) -> RankedEpochCompletion:
        if self.index != len(self.windows):
            raise RuntimeError("reader not exhausted")
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


class TrainExecution:
    """Test ranked execution returning one exact prebuilt plan."""

    def __init__(
        self,
        *,
        plan: RankedTrainEpochPlan,
        windows: list[RankedTrainWindow],
    ) -> None:
        self._plan = plan
        self._windows = windows
        self.batches = tuple(window.batches for window in windows)
        self.resume_identity = plan.data_identity
        self.reader: TrainReader | None = None

    def plan_epoch(
        self,
        epoch: int,
        *,
        microbatches_per_window: int,
        max_microbatches: int | None,
    ) -> RankedTrainEpochPlan:
        assert epoch == self._plan.epoch
        assert microbatches_per_window == self._plan.microbatches_per_window
        assert max_microbatches == self._plan.requested_max_microbatches
        return self._plan

    def open_epoch(self, plan: RankedTrainEpochPlan) -> TrainReader:
        assert plan == self._plan
        self.reader = TrainReader(plan=plan, windows=list(self._windows))
        return self.reader


class UntrustedTrainExecution(TrainExecution):
    """Structurally compatible extension that may return unrelated plan facts."""

    def plan_epoch(
        self,
        epoch: int,
        *,
        microbatches_per_window: int,
        max_microbatches: int | None,
    ) -> RankedTrainEpochPlan:
        del epoch, microbatches_per_window, max_microbatches
        return self._plan


class ValidationReader:
    """Rank-zero validation reader for one exact coverage span."""

    def __init__(
        self,
        plan: ExactValidationEpochPlan,
        batches: list[ExactValidationBatch],
    ) -> None:
        self.plan = plan
        self.batches = batches
        self.index = 0

    def read_batch(self) -> ExactValidationBatch | None:
        if self.index == len(self.batches):
            return None
        value = self.batches[self.index]
        self.index += 1
        return value

    def finish(self) -> ExactCoverageReceipt:
        return ExactCoverageReceipt(
            plan_digest=self.plan.plan_digest,
            rank=self.plan.rank,
            completed_spans=self.plan.local_spans,
            observed_samples=self.plan.local_expected_samples,
        )

    def close(self) -> None:
        pass


class ValidationExecution:
    """Test exact validation execution."""

    def __init__(
        self,
        *,
        plan: ExactValidationEpochPlan,
        batches: list[ExactValidationBatch],
    ) -> None:
        self._plan = plan
        self._batches = batches
        self.batches = tuple(item.batch for item in batches)
        self.coverage_identity = plan.coverage_identity

    def plan_epoch(self, epoch: int) -> ExactValidationEpochPlan:
        assert epoch == self._plan.epoch
        return self._plan

    def open_epoch(self, plan: ExactValidationEpochPlan) -> ValidationReader:
        return ValidationReader(plan, list(self._batches))


class UntrustedValidationExecution(ValidationExecution):
    """Structurally compatible extension that may return another epoch's plan."""

    def plan_epoch(self, epoch: int) -> ExactValidationEpochPlan:
        del epoch
        return self._plan


def make_plan(*, accumulation: int = 2) -> tuple[RankedTrainEpochPlan, list[RankedTrainWindow]]:
    identity = RankedEpochDataIdentity("test.ranked.v1", digest("identity"))
    plan = RankedTrainEpochPlan(
        data_identity=identity,
        plan_digest=digest("plan"),
        expected_terminal_token=digest("terminal"),
        epoch=1,
        rank=0,
        world_size=1,
        microbatches_per_window=accumulation,
        window_count=1,
        samples_per_microbatch=2,
        local_assigned_samples=2 * accumulation,
        global_assigned_samples=2 * accumulation,
        global_dropped_samples=0,
        assignment_digest=digest("assignments"),
    )
    batches = tuple(
        (
            torch.tensor([[1.0], [2.0]]) + index,
            torch.tensor([[2.0], [4.0]]) + index,
        )
        for index in range(accumulation)
    )
    facts = tuple(
        RankedBatchFacts(
            ordinal=index,
            sample_count=2,
            loss_weight=2.0,
            assignment_token=digest(f"batch-{index}"),
        )
        for index in range(accumulation)
    )
    return plan, [RankedTrainWindow(0, batches, facts)]


def make_trainer(
    *,
    accumulation: int = 2,
    process: Process | None = None,
    canonical: nn.Module | None = None,
    max_grad_norm: float | None = None,
) -> tuple[DDPTrainer, RecordingExecutionModel]:
    canonical = canonical or nn.Linear(1, 1, bias=False)
    if isinstance(canonical, nn.Linear):
        canonical.weight.data.fill_(0.5)
    canonical_strategy = RegressionStrategy(canonical)
    plan = TrainingPlan(
        strategy=canonical_strategy,
        primary_model=canonical,
        process=process,
    )
    binding = DDPExecutionBinding.from_assembly(
        TrainingPlanAssembly(
            plan=plan,
            builder_name="test_regression",
            _execution_binding=RegressionExecutionBindingBuilder(),
        ),
        wrap=RecordingExecutionModel,
    )
    execution = binding.execution_model
    assert isinstance(execution, RecordingExecutionModel)
    trainer = DDPTrainer(
        binding=binding,
        optimizer=torch.optim.SGD(canonical.parameters(), lr=0.1),
        collectives=SingleRankCollectives(),
        topology=DistributedTopology(0, 0, 1, 1),
        device="cpu",
        precision=build_precision_runtime("fp32", "cpu"),
        accumulate_grad_batches=accumulation,
        max_grad_norm=max_grad_norm,
    )
    return trainer, execution


def make_two_parameter_trainer(
    *,
    reverse_optimizer_order: bool = False,
    separate_parameter_groups: bool = False,
) -> DDPTrainer:
    """Build a trainer whose optimizer ordering can be controlled exactly."""

    canonical = nn.Linear(1, 1, bias=True)
    plan = TrainingPlan(
        strategy=RegressionStrategy(canonical),
        primary_model=canonical,
    )
    binding = DDPExecutionBinding.from_assembly(
        TrainingPlanAssembly(
            plan=plan,
            builder_name="test_regression",
            _execution_binding=RegressionExecutionBindingBuilder(),
        ),
        wrap=RecordingExecutionModel,
    )
    parameters = tuple(canonical.parameters())
    if reverse_optimizer_order:
        optimizer = torch.optim.SGD(tuple(reversed(parameters)), lr=0.1)
    elif separate_parameter_groups:
        optimizer = torch.optim.SGD(
            [{"params": [parameters[0]]}, {"params": [parameters[1]]}],
            lr=0.1,
        )
    else:
        optimizer = torch.optim.SGD(parameters, lr=0.1)
    return DDPTrainer(
        binding=binding,
        optimizer=optimizer,
        collectives=SingleRankCollectives(),
        topology=DistributedTopology(0, 0, 1, 1),
        device="cpu",
        precision=build_precision_runtime("fp32", "cpu"),
        accumulate_grad_batches=1,
    )


def test_ddp_trainer_requires_exact_optimizer_parameter_order() -> None:
    with pytest.raises(ValueError, match="canonical plan in exact order"):
        make_two_parameter_trainer(reverse_optimizer_order=True)

    trainer = make_two_parameter_trainer(separate_parameter_groups=True)
    assert len(trainer.optimizer.param_groups) == 2


def test_ddp_trainer_rechecks_optimizer_order_during_restore_acceptance() -> None:
    trainer = make_two_parameter_trainer(separate_parameter_groups=True)
    saved_runtime_digest = trainer.checkpoint_state_fingerprint()
    trainer.optimizer.param_groups.reverse()

    with pytest.raises(ValueError, match="canonical plan in exact order"):
        trainer.accept_restored_state(
            global_step=3,
            common_checkpoint_sha256=digest("common-checkpoint"),
            common_runtime_state_sha256=saved_runtime_digest,
        )

    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.assert_checkpoint_publishable()


def test_ddp_trainer_rechecks_optimizer_order_before_checkpoint_publish() -> None:
    trainer = make_two_parameter_trainer(separate_parameter_groups=True)
    saved_runtime_digest = trainer.checkpoint_state_fingerprint()
    trainer.accept_restored_state(
        global_step=3,
        common_checkpoint_sha256=digest("common-checkpoint"),
        common_runtime_state_sha256=saved_runtime_digest,
    )
    trainer.optimizer.param_groups.reverse()

    with pytest.raises(ValueError, match="canonical plan in exact order"):
        trainer.assert_checkpoint_publishable()


def test_ddp_trainer_rejects_checkpoint_runtime_tensor_aliases() -> None:
    trainer = make_two_parameter_trainer()
    parameters = tuple(trainer.trainable_parameters)
    shared_momentum = torch.ones_like(parameters[0])
    trainer.optimizer.state[parameters[0]]["momentum_buffer"] = shared_momentum
    trainer.optimizer.state[parameters[1]]["momentum_buffer"] = shared_momentum

    with pytest.raises(ValueError, match="runtime tensor aliases"):
        trainer.assert_checkpoint_publishable()


def test_ddp_trainer_rejects_checkpoint_runtime_tensor_views() -> None:
    trainer = make_two_parameter_trainer()
    parameter = trainer.trainable_parameters[0]
    trainer.optimizer.state[parameter]["momentum_buffer"] = torch.arange(5.0)[1:4]

    with pytest.raises(ValueError, match="storage topology"):
        trainer.assert_checkpoint_publishable()

    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.assert_checkpoint_publishable()


def test_ddp_trainer_accepts_one_verified_restore_before_runtime() -> None:
    source_process = FrozenResumeProcess()
    source_process.load_state_dict({"coefficient": torch.tensor(2.0)})
    source_trainer, _ = make_trainer(accumulation=1, process=source_process)
    saved_runtime_digest = source_trainer.checkpoint_state_fingerprint()

    process = FrozenResumeProcess()
    trainer, _ = make_trainer(accumulation=1, process=process)
    trainer.plan.primary_model.load_state_dict(
        source_trainer.plan.primary_model.state_dict()
    )
    process.load_state_dict(source_process.state_dict())

    trainer.accept_restored_state(
        global_step=7,
        common_checkpoint_sha256=digest("common-checkpoint"),
        common_runtime_state_sha256=saved_runtime_digest,
    )

    assert trainer.global_step == 7
    plan, windows = make_plan(accumulation=1)
    trainer.train_epoch(
        TrainExecution(plan=plan, windows=windows),
        epoch_index=1,
    )
    assert trainer.global_step == 8
    with pytest.raises(RuntimeError, match="only be accepted once"):
        trainer.accept_restored_state(
            global_step=8,
            common_checkpoint_sha256=digest("common-checkpoint"),
            common_runtime_state_sha256=saved_runtime_digest,
        )


def test_ddp_trainer_rejects_restore_acceptance_after_runtime_started() -> None:
    trainer, _ = make_trainer(accumulation=1)
    plan, windows = make_plan(accumulation=1)
    trainer.train_epoch(
        TrainExecution(plan=plan, windows=windows),
        epoch_index=1,
    )

    with pytest.raises(RuntimeError, match="before run"):
        trainer.accept_restored_state(
            global_step=trainer.global_step,
            common_checkpoint_sha256=digest("common-checkpoint"),
            common_runtime_state_sha256=digest("runtime-state"),
        )


def test_failed_restore_acceptance_poison_runtime() -> None:
    trainer, _ = make_trainer(accumulation=1)
    trainer.collectives = RestoreMismatchCollectives()

    with pytest.raises(ValueError, match="restored common state differs"):
        trainer.accept_restored_state(
            global_step=7,
            common_checkpoint_sha256=digest("common-checkpoint"),
            common_runtime_state_sha256=digest("runtime-state"),
        )

    plan, windows = make_plan(accumulation=1)
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.train_epoch(
            TrainExecution(plan=plan, windows=windows),
            epoch_index=1,
        )
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.assert_checkpoint_publishable()


def test_ddp_trainer_uses_no_sync_for_forward_and_backward_window() -> None:
    trainer, execution_model = make_trainer(accumulation=2)
    plan, windows = make_plan(accumulation=2)
    execution = TrainExecution(plan=plan, windows=windows)

    result = trainer.train_epoch(execution, epoch_index=1)

    assert execution_model.forward_scopes == [True, False]
    assert execution_model.backward_scopes == [True, False]
    assert trainer.global_step == 1
    assert result.metrics["optimizer_steps"] == 1.0
    assert result.metrics["num_batches"] == 2.0
    assert execution.reader is not None
    assert execution.reader.closed


def test_ddp_trainer_rejects_untrusted_train_plan_for_another_epoch() -> None:
    trainer, execution_model = make_trainer(accumulation=1)
    plan, windows = make_plan(accumulation=1)
    unrelated = replace(plan, epoch=2)

    with pytest.raises(ValueError, match="plan epoch does not match"):
        trainer.train_epoch(
            UntrustedTrainExecution(plan=unrelated, windows=windows),
            epoch_index=1,
        )

    assert execution_model.forward_scopes == []


def test_ddp_trainer_rejects_untrusted_train_plan_with_another_limit() -> None:
    trainer, execution_model = make_trainer(accumulation=1)
    plan, windows = make_plan(accumulation=1)
    unrelated = replace(plan, requested_max_microbatches=2)

    with pytest.raises(ValueError, match="requested_max_microbatches"):
        trainer.train_epoch(
            UntrustedTrainExecution(plan=unrelated, windows=windows),
            epoch_index=1,
            max_microbatches=None,
        )

    assert execution_model.forward_scopes == []


def test_ddp_trainer_rejects_partial_window_before_forward() -> None:
    trainer, execution_model = make_trainer(accumulation=2)
    plan, windows = make_plan(accumulation=2)
    invalid = RankedTrainWindow(
        ordinal=0,
        batches=windows[0].batches[:1],
        batch_facts=windows[0].batch_facts[:1],
    )

    with pytest.raises(ValueError, match="complete accumulation"):
        trainer.train_epoch(
            TrainExecution(plan=plan, windows=[invalid]),
            epoch_index=1,
        )

    assert execution_model.forward_scopes == []
    assert trainer.global_step == 0


def test_ddp_trainer_rejects_non_finite_loss_without_publishing() -> None:
    trainer, _ = make_trainer(accumulation=1)
    plan, windows = make_plan(accumulation=1)
    inputs, targets = windows[0].batches[0]
    invalid_window = RankedTrainWindow(
        ordinal=0,
        batches=((inputs, torch.full_like(targets, float("inf"))),),
        batch_facts=windows[0].batch_facts,
    )
    before = tuple(
        parameter.detach().clone()
        for parameter in trainer.plan.primary_model.parameters()
    )

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        trainer.train_epoch(
            TrainExecution(plan=plan, windows=[invalid_window]),
            epoch_index=1,
        )

    assert trainer.global_step == 0
    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(
            trainer.plan.primary_model.parameters(),
            before,
            strict=True,
        )
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.assert_checkpoint_publishable()


@pytest.mark.parametrize("max_grad_norm", [None, 1.0])
def test_ddp_trainer_rejects_missing_gradient_before_commit(
    max_grad_norm: float | None,
) -> None:
    trainer, _ = make_trainer(
        accumulation=1,
        canonical=UnusedParameterRegressionModel(),
        max_grad_norm=max_grad_norm,
    )
    plan, windows = make_plan(accumulation=1)
    before = tuple(
        parameter.detach().clone()
        for parameter in trainer.plan.primary_model.parameters()
    )

    with pytest.raises(FloatingPointError, match="missing or non-finite gradients"):
        trainer.train_epoch(
            TrainExecution(plan=plan, windows=windows),
            epoch_index=1,
        )

    assert trainer.global_step == 0
    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(
            trainer.plan.primary_model.parameters(),
            before,
            strict=True,
        )
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.assert_checkpoint_publishable()


def test_ddp_trainer_rejects_ranked_weight_mismatch_before_backward() -> None:
    trainer, _ = make_trainer(accumulation=1)
    plan, windows = make_plan(accumulation=1)
    wrong_fact = RankedBatchFacts(
        ordinal=0,
        sample_count=2,
        loss_weight=1.0,
        assignment_token=digest("wrong"),
    )
    window = RankedTrainWindow(0, windows[0].batches, (wrong_fact,))

    with pytest.raises(ValueError, match="loss weight"):
        trainer.train_epoch(
            TrainExecution(plan=plan, windows=[window]),
            epoch_index=1,
        )

    assert trainer.global_step == 0


def test_ddp_validation_uses_canonical_model_and_full_rank_zero_view() -> None:
    trainer, execution_model = make_trainer(accumulation=1)
    identity = RankedEpochDataIdentity("test.validation.v1", digest("validation"))
    plan = ExactValidationEpochPlan(
        coverage_identity=identity,
        plan_digest=digest("validation-plan"),
        epoch=1,
        rank=0,
        world_size=1,
        global_expected_samples=2,
        primary_batch_count=1,
        local_expected_samples=2,
        local_spans=(ExactCoverageSpan(0, 2),),
    )
    batch = (
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[2.0], [4.0]]),
    )
    item = ExactValidationBatch(
        batch=batch,
        facts=RankedBatchFacts(0, 2, 2.0, digest("validation-batch")),
        coverage_span=ExactCoverageSpan(0, 2),
    )

    metrics = trainer.evaluate_epoch(
        ValidationExecution(plan=plan, batches=[item]),
        epoch_index=1,
    )

    assert metrics == {"loss": pytest.approx(5.625), "num_batches": 1.0}
    assert execution_model.forward_scopes == []


def test_ddp_validation_rejects_untrusted_plan_for_another_epoch() -> None:
    trainer, execution_model = make_trainer(accumulation=1)
    identity = RankedEpochDataIdentity("test.validation.v1", digest("validation"))
    plan = ExactValidationEpochPlan(
        coverage_identity=identity,
        plan_digest=digest("validation-plan"),
        epoch=2,
        rank=0,
        world_size=1,
        global_expected_samples=2,
        primary_batch_count=1,
        local_expected_samples=2,
        local_spans=(ExactCoverageSpan(0, 2),),
    )

    with pytest.raises(ValueError, match="validation plan epoch does not match"):
        trainer.evaluate_epoch(
            UntrustedValidationExecution(plan=plan, batches=[]),
            epoch_index=1,
        )

    assert execution_model.forward_scopes == []


def test_ddp_trainer_rejects_fp16_scaler_contract() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to construct an fp16 GradScaler")
    canonical = nn.Linear(1, 1, bias=False).cuda()
    plan = TrainingPlan(
        strategy=RegressionStrategy(canonical),
        primary_model=canonical,
    )
    binding = DDPExecutionBinding.from_assembly(
        TrainingPlanAssembly(
            plan=plan,
            builder_name="test_regression",
            _execution_binding=RegressionExecutionBindingBuilder(),
        ),
        wrap=RecordingExecutionModel,
    )
    with pytest.raises(ValueError, match="fp16-mixed"):
        DDPTrainer(
            binding=binding,
            optimizer=torch.optim.SGD(canonical.parameters(), lr=0.1),
            collectives=SingleRankCollectives(),
            topology=DistributedTopology(0, 0, 1, 1),
            device="cuda:0",
            precision=build_precision_runtime("fp16-mixed", "cuda:0"),
            accumulate_grad_batches=1,
        )
