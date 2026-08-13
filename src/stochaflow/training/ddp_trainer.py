"""Fixed-topology DDP training without changing the single-process Trainer."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import AbstractContextManager, nullcontext, suppress
from copy import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.data.ranked import (
    ExactCoverageReceipt,
    ExactCoverageSpan,
    ExactValidationExecution,
    RankedBatchFacts,
    RankedEpochCompletion,
    RankedTrainEpochPlan,
    RankedTrainExecution,
    RankedTrainWindow,
)
from stochaflow.training.builder import (
    ManagedTrainingModule,
    TrainingPlan,
    TrainingPlanAssembly,
    trainable_parameters,
    validate_training_plan,
)
from stochaflow.training.distributed.contracts import (
    DistributedCollectives,
    DistributedTopology,
)
from stochaflow.training.distributed.state_fingerprint import (
    module_runtime_state,
    require_clone_safe_runtime_state,
    require_no_distinct_shared_storage_across_modules,
    require_relocatable_module_states,
    require_runtime_state_disjoint_from_modules,
    require_tensor_free_runtime_state,
    runtime_state_fingerprint,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.metric_binding import TrainingMetricRuntime
from stochaflow.training.precision import PrecisionRuntime
from stochaflow.training.strategy import (
    Batch,
    DeviceTransferableBatch,
    TrainingStrategy,
    loss_aggregation_weight_to_float,
    validate_train_step_output,
)

StageResultT = TypeVar("StageResultT")


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
_DDP_BINDING_PROVENANCE = object()


@runtime_checkable
class GradientSynchronizationModel(Protocol):
    """Execution-model capability used for accumulation without early sync."""

    def no_sync(self) -> AbstractContextManager[None]:
        """Disable gradient synchronization around forward and backward."""

        ...


@dataclass(frozen=True, slots=True)
class DDPExecutionBinding:
    """Validated canonical plan and the Builder-bound DDP execution path."""

    plan: TrainingPlan
    execution_model: nn.Module
    execution_strategy: TrainingStrategy
    _provenance: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._provenance is not _DDP_BINDING_PROVENANCE:
            raise TypeError(
                "DDPExecutionBinding must be created from TrainingPlanAssembly"
            )

    @classmethod
    def from_assembly(
        cls,
        assembly: TrainingPlanAssembly,
        *,
        wrap: Callable[[nn.Module], nn.Module],
    ) -> DDPExecutionBinding:
        """Wrap one Builder-owned execution root and bind its Strategy."""

        assembly_value = cast(object, assembly)
        if not isinstance(assembly_value, TrainingPlanAssembly):
            raise TypeError("DDP execution binding requires TrainingPlanAssembly")
        if not callable(wrap):
            raise TypeError("DDP execution wrapper factory must be callable")
        execution_root = assembly.build_primary_execution_module()
        execution_model = cast(object, wrap(execution_root))
        if not isinstance(execution_model, nn.Module):
            raise TypeError("DDP execution wrapper must return an nn.Module")
        execution_strategy = assembly.bind_primary_execution_model(execution_model)
        return cls(
            plan=assembly.plan,
            execution_model=execution_model,
            execution_strategy=execution_strategy,
            _provenance=_DDP_BINDING_PROVENANCE,
        )

    @classmethod
    def from_prepared(
        cls,
        assembly: TrainingPlanAssembly,
        *,
        execution_root: nn.Module,
        execution_model: nn.Module,
    ) -> DDPExecutionBinding:
        """Bind an already wrapped root after composition-stage consensus."""

        assembly_value = cast(object, assembly)
        if not isinstance(assembly_value, TrainingPlanAssembly):
            raise TypeError("DDP execution binding requires TrainingPlanAssembly")
        root_value = cast(object, execution_root)
        model_value = cast(object, execution_model)
        if not isinstance(root_value, nn.Module):
            raise TypeError("prepared DDP execution root must be an nn.Module")
        if not isinstance(model_value, nn.Module):
            raise TypeError("prepared DDP execution wrapper must be an nn.Module")
        canonical_parameters = tuple(assembly.plan.primary_model.parameters())
        canonical_buffers = tuple(assembly.plan.primary_model.buffers())
        if tuple(map(id, execution_root.parameters())) != tuple(
            map(id, canonical_parameters)
        ) or tuple(map(id, execution_root.buffers())) != tuple(
            map(id, canonical_buffers)
        ):
            raise ValueError(
                "prepared DDP execution root must preserve canonical state identity"
            )
        if tuple(map(id, execution_model.parameters())) != tuple(
            map(id, canonical_parameters)
        ) or tuple(map(id, execution_model.buffers())) != tuple(
            map(id, canonical_buffers)
        ):
            raise ValueError(
                "prepared DDP execution wrapper must preserve canonical state identity"
            )
        execution_strategy = assembly.bind_primary_execution_model(execution_model)
        return cls(
            plan=assembly.plan,
            execution_model=execution_model,
            execution_strategy=execution_strategy,
            _provenance=_DDP_BINDING_PROVENANCE,
        )


@dataclass(frozen=True, slots=True)
class DDPTrainEpochResult:
    """Globally aggregated facts from one completed ranked training epoch."""

    metrics: Mapping[str, float]
    local_completion: RankedEpochCompletion
    global_completions: tuple[RankedEpochCompletion, ...]

    def __post_init__(self) -> None:
        metrics = cast(object, self.metrics)
        if not isinstance(metrics, Mapping):
            raise TypeError("distributed train metrics must be a mapping")
        completion = cast(object, self.local_completion)
        if not isinstance(completion, RankedEpochCompletion):
            raise TypeError("distributed train completion has the wrong type")
        completions = cast(object, self.global_completions)
        if not isinstance(completions, tuple) or not completions:
            raise TypeError("distributed global completions must be a non-empty tuple")
        if any(
            not isinstance(value, RankedEpochCompletion)
            for value in cast(tuple[object, ...], completions)
        ):
            raise TypeError("distributed global completions have the wrong type")
        if tuple(value.rank for value in self.global_completions) != tuple(
            range(len(self.global_completions))
        ):
            raise ValueError("distributed global completions are not rank ordered")
        if self.global_completions[self.local_completion.rank] != self.local_completion:
            raise ValueError("local completion does not match the global inventory")
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(self.metrics)),
        )

    @property
    def completion(self) -> RankedEpochCompletion:
        """Compatibility spelling for the explicitly rank-local completion."""

        return self.local_completion


def _move_to_device(batch: Batch, device: torch.device) -> Batch:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, DeviceTransferableBatch):
        return batch.to_device(device)
    if isinstance(batch, Mapping):
        moved_items = {
            key: _move_to_device(value, device) for key, value in batch.items()
        }
        if isinstance(batch, MutableMapping):
            moved_mapping = copy(batch)
            moved_mapping.clear()
            moved_mapping.update(moved_items)
            return moved_mapping
        try:
            mapping_type = cast(
                Callable[[Mapping[Any, Any]], Mapping[Any, Any]],
                type(batch),
            )
            return mapping_type(moved_items)
        except TypeError:
            return moved_items
    if isinstance(batch, tuple):
        moved = tuple(_move_to_device(item, device) for item in batch)
        if hasattr(batch, "_fields"):
            return type(batch)(*moved)
        return moved
    if isinstance(batch, list):
        moved_list = [_move_to_device(item, device) for item in batch]
        if type(batch) is list:
            return moved_list
        try:
            return type(batch)(moved_list)
        except TypeError:
            return moved_list
    return batch


def _gradient_state_is_finite(parameters: tuple[nn.Parameter, ...]) -> bool:
    return all(
        gradient is not None and bool(torch.isfinite(gradient).all().item())
        for gradient in (parameter.grad for parameter in parameters)
    )


def _optimizer_parameters(optimizer: Optimizer) -> tuple[object, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _validate_optimizer_parameter_order(
    optimizer: Optimizer,
    expected: tuple[nn.Parameter, ...],
) -> None:
    parameters = _optimizer_parameters(optimizer)
    if any(not isinstance(parameter, nn.Parameter) for parameter in parameters):
        raise TypeError("DDP optimizer params must contain only Parameter objects")
    if tuple(map(id, parameters)) != tuple(map(id, expected)):
        raise ValueError(
            "DDP optimizer parameters must match the canonical plan in exact order"
        )


def _failure_summary(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    try:
        message = str(error)
    except BaseException:  # noqa: BLE001
        message = "<exception text could not be rendered>"
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message[:1000],
    }


def _add_failure_note(error: BaseException, note: str) -> None:
    with suppress(BaseException):
        BaseException.add_note(error, note)


class DDPTrainer:
    """Automatic fixed-world DDP loop for one primary trainable model.

    This class deliberately does not inherit :class:`Trainer`. It consumes a
    Builder-bound execution Strategy and ranked data capabilities while the
    canonical plan remains responsible for state, EMA, and checkpoint identity.
    """

    def __init__(
        self,
        *,
        binding: DDPExecutionBinding,
        optimizer: Optimizer,
        collectives: DistributedCollectives,
        topology: DistributedTopology,
        device: torch.device | str,
        precision: PrecisionRuntime,
        accumulate_grad_batches: int,
        lr_scheduler: LRScheduler | None = None,
        lr_scheduler_interval: str = "step",
        ema: ExponentialMovingAverage | None = None,
        metric_runtime: TrainingMetricRuntime | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        binding_value = cast(object, binding)
        if not isinstance(binding_value, DDPExecutionBinding):
            raise TypeError("DDPTrainer requires a validated DDPExecutionBinding")
        self.plan = validate_training_plan(binding.plan)
        execution_model = binding.execution_model
        execution_strategy = binding.execution_strategy
        execution_model_value = cast(object, execution_model)
        if not isinstance(execution_model_value, nn.Module):
            raise TypeError("DDP execution_model must be an nn.Module")
        if not isinstance(execution_model_value, GradientSynchronizationModel):
            raise TypeError("DDP execution_model must provide no_sync()")
        execution_strategy_value = cast(object, execution_strategy)
        if not isinstance(execution_strategy_value, TrainingStrategy):
            raise TypeError("DDP execution_strategy must be TrainingStrategy")
        if isinstance(execution_strategy, nn.Module):
            raise TypeError("DDP execution_strategy must not inherit nn.Module")
        topology_value = cast(object, topology)
        if not isinstance(topology_value, DistributedTopology):
            raise TypeError("DDP topology has the wrong type")
        precision_value = cast(object, precision)
        if not isinstance(precision_value, PrecisionRuntime):
            raise TypeError("DDP precision has the wrong type")
        self.device = torch.device(device)
        if precision.device_type != self.device.type:
            raise ValueError("DDP precision device must match the local device")
        if precision.grad_scaler is not None:
            raise ValueError(
                "fixed DDP currently supports fp32 and bf16-mixed; "
                "fp16-mixed needs a global GradScaler commit contract"
            )
        if type(accumulate_grad_batches) is not int or accumulate_grad_batches <= 0:
            raise ValueError(
                "DDP accumulate_grad_batches must be a positive integer"
            )
        if lr_scheduler_interval not in {"step", "epoch"}:
            raise ValueError("DDP lr_scheduler_interval must be 'step' or 'epoch'")
        max_grad_norm_value = cast(object, max_grad_norm)
        if max_grad_norm_value is not None and (
            isinstance(max_grad_norm_value, bool)
            or not isinstance(max_grad_norm_value, (int, float))
            or not math.isfinite(float(max_grad_norm_value))
            or max_grad_norm_value <= 0
        ):
            raise ValueError("DDP max_grad_norm must be finite and positive")
        if metric_runtime is not None and metric_runtime.has_phase("train"):
            raise ValueError(
                "fixed DDP does not yet accept train-phase Metric providers"
            )
        if metric_runtime is not None and metric_runtime.has_phase("test"):
            raise ValueError(
                "fixed DDP does not yet accept test-phase Metric providers"
            )

        selected_parameters = trainable_parameters(self.plan)
        primary_parameters = tuple(
            parameter
            for parameter in self.plan.primary_model.parameters()
            if parameter.requires_grad
        )
        if tuple(map(id, selected_parameters)) != tuple(map(id, primary_parameters)):
            raise ValueError(
                "fixed DDP requires the primary model to be the only trainable root"
            )
        _validate_optimizer_parameter_order(optimizer, selected_parameters)
        execution_parameters = tuple(
            parameter
            for parameter in execution_model.parameters()
            if parameter.requires_grad
        )
        if tuple(map(id, execution_parameters)) != tuple(
            map(id, primary_parameters)
        ):
            raise ValueError(
                "DDP execution model parameters must be the canonical primary "
                "parameters in the same order"
            )
        if tuple(map(id, execution_model.buffers())) != tuple(
            map(id, self.plan.primary_model.buffers())
        ):
            raise ValueError(
                "DDP execution model buffers must be the canonical primary "
                "buffers in the same order"
            )
        if lr_scheduler is not None and lr_scheduler.optimizer is not optimizer:
            raise ValueError("DDP lr_scheduler must retain the injected optimizer")

        self.execution_model = execution_model
        self.synchronization_model = cast(
            GradientSynchronizationModel,
            execution_model,
        )
        self.execution_strategy = execution_strategy
        self.optimizer = optimizer
        self.collectives = collectives
        self.topology = topology
        self.precision = precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_interval = lr_scheduler_interval
        self.ema = ema
        self.metric_runtime = metric_runtime
        self.max_grad_norm = (
            float(max_grad_norm) if max_grad_norm is not None else None
        )
        self.trainable_parameters = selected_parameters
        self.global_step = 0
        self._runtime_started = False
        self._restored_state_accepted = False
        self._poisoned = False
        self._collectives_unsafe = False
        self.managed_modules: Mapping[str, ManagedTrainingModule] = MappingProxyType(
            {
                "primary_model": ManagedTrainingModule(self.plan.primary_model),
                **(
                    {"process": ManagedTrainingModule(self.plan.process)}
                    if self.plan.process is not None
                    else {}
                ),
                **(
                    {"objective": ManagedTrainingModule(self.plan.objective)}
                    if self.plan.objective is not None
                    else {}
                ),
                **dict(self.plan.auxiliary_modules),
            }
        )
        self._primary_read_only_state_fingerprint = (
            self._snapshot_primary_read_only_fingerprint()
        )
        self._non_primary_state_fingerprint = (
            self._snapshot_non_primary_state_fingerprint()
        )

    def _common_runtime_state_fingerprint(self) -> str:
        state: dict[str, object] = {
            "primary_model": module_runtime_state(self.plan.primary_model),
            "process": (
                module_runtime_state(self.plan.process)
                if self.plan.process is not None
                else None
            ),
            "objective": (
                module_runtime_state(self.plan.objective)
                if self.plan.objective is not None
                else None
            ),
            "auxiliary_modules": {
                name: module_runtime_state(asset.module)
                for name, asset in self.plan.auxiliary_modules.items()
            },
            "optimizer": {
                "type": _type_identity(self.optimizer),
                "state": self.optimizer.state_dict(),
            },
            "lr_scheduler": (
                {
                    "type": _type_identity(self.lr_scheduler),
                    "state": self.lr_scheduler.state_dict(),
                }
                if self.lr_scheduler is not None
                else None
            ),
            "ema": (
                {
                    "type": _type_identity(self.ema),
                    "state": self.ema.state_dict(),
                }
                if self.ema is not None
                else None
            ),
            "precision": {
                "type": _type_identity(self.precision),
                "kind": self.precision.kind,
                "grad_scaler_type": (
                    _type_identity(self.precision.grad_scaler)
                    if self.precision.grad_scaler is not None
                    else None
                ),
            },
        }
        return runtime_state_fingerprint(state, path="distributed runtime state")

    def _require_optimizer_parameter_order(self) -> None:
        _validate_optimizer_parameter_order(
            self.optimizer,
            self.trainable_parameters,
        )

    def accept_restored_state(
        self,
        *,
        global_step: int,
        common_checkpoint_sha256: str,
        common_runtime_state_sha256: str,
    ) -> None:
        """Accept one all-rank-verified restore before any runtime stage.

        This one-shot boundary verifies that every rank restored the same saved
        content before refreshing checkpoint-boundary content guards.
        """

        self._raise_if_poisoned()
        try:
            self._accept_restored_state(
                global_step=global_step,
                common_checkpoint_sha256=common_checkpoint_sha256,
                common_runtime_state_sha256=common_runtime_state_sha256,
            )
        except BaseException:
            self._poisoned = True
            raise

    def _accept_restored_state(
        self,
        *,
        global_step: int,
        common_checkpoint_sha256: str,
        common_runtime_state_sha256: str,
    ) -> None:
        """Validate and bind one restore after the public poison guard."""

        local_error: BaseException | None = None
        fingerprint: str | None = None
        try:
            if self._runtime_started or self._restored_state_accepted:
                raise RuntimeError(
                    "distributed restored state may only be accepted once before run"
                )
            if type(global_step) is not int or global_step < 0:
                raise ValueError(
                    "distributed restored global_step must be non-negative"
                )
            if (
                len(common_checkpoint_sha256) != 64
                or common_checkpoint_sha256 != common_checkpoint_sha256.lower()
                or any(
                    character not in "0123456789abcdef"
                    for character in common_checkpoint_sha256
                )
            ):
                raise ValueError(
                    "distributed common checkpoint digest must be lowercase SHA-256"
                )
            if (
                len(common_runtime_state_sha256) != 64
                or common_runtime_state_sha256
                != common_runtime_state_sha256.lower()
                or any(
                    character not in "0123456789abcdef"
                    for character in common_runtime_state_sha256
                )
            ):
                raise ValueError(
                    "distributed runtime state digest must be lowercase SHA-256"
                )
            if any(parameter.grad is not None for parameter in self.trainable_parameters):
                raise RuntimeError(
                    "distributed restore acceptance requires empty gradients"
                )
            self._require_optimizer_parameter_order()
            execution_parameters = tuple(
                parameter
                for parameter in self.execution_model.parameters()
                if parameter.requires_grad
            )
            if tuple(map(id, execution_parameters)) != tuple(
                map(id, self.trainable_parameters)
            ):
                raise RuntimeError(
                    "distributed execution binding changed during restore"
                )
            if tuple(map(id, self.execution_model.buffers())) != tuple(
                map(id, self.plan.primary_model.buffers())
            ):
                raise RuntimeError(
                    "distributed execution buffers changed during restore"
                )
            fingerprint = self._common_runtime_state_fingerprint()
        except BaseException as error:  # noqa: BLE001
            local_error = error
        self._require_collective_success(
            local_error=local_error,
            phase="distributed restore acceptance",
        )
        assert fingerprint is not None
        authority = {
            "common_checkpoint_sha256": common_checkpoint_sha256,
            "expected_runtime_state_sha256": common_runtime_state_sha256,
            "global_step": global_step,
            "runtime_state_sha256": fingerprint,
        }
        if not self.collectives.all_equal(authority):
            raise ValueError(
                "distributed restored common state differs across ranks"
            )
        if fingerprint != common_runtime_state_sha256:
            raise ValueError(
                "distributed restored runtime state does not match its saved digest"
            )
        self.global_step = global_step
        self._primary_read_only_state_fingerprint = (
            self._snapshot_primary_read_only_fingerprint()
        )
        self._non_primary_state_fingerprint = (
            self._snapshot_non_primary_state_fingerprint()
        )
        self._restored_state_accepted = True

    @staticmethod
    def _module_fingerprint(module: nn.Module, *, path: str) -> str:
        return runtime_state_fingerprint(module_runtime_state(module), path=path)

    def _primary_model_fingerprint(self) -> str:
        return self._module_fingerprint(
            self.plan.primary_model,
            path="primary model state",
        )

    def _snapshot_primary_read_only_fingerprint(self) -> str:
        state = {
            "frozen_parameters": {
                name: parameter
                for name, parameter in self.plan.primary_model.named_parameters()
                if not parameter.requires_grad
            },
            "buffers": dict(self.plan.primary_model.named_buffers()),
        }
        return runtime_state_fingerprint(state, path="primary read-only state")

    def _snapshot_non_primary_state_fingerprint(self) -> str:
        state = {
            role: module_runtime_state(asset.module)
            for role, asset in self.managed_modules.items()
            if role != "primary_model"
        }
        return runtime_state_fingerprint(state, path="non-primary training state")

    def _require_read_only_managed_state(self) -> None:
        if (
            self._snapshot_primary_read_only_fingerprint()
            != self._primary_read_only_state_fingerprint
        ):
            raise RuntimeError(
                "fixed DDP requires frozen primary state to remain read-only"
            )
        if (
            self._snapshot_non_primary_state_fingerprint()
            != self._non_primary_state_fingerprint
        ):
            raise RuntimeError(
                "fixed DDP requires non-primary training state to remain read-only"
            )

    def _raise_if_poisoned(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "distributed training runtime is poisoned after an earlier failure"
            )

    def checkpoint_state_fingerprint(self) -> str:
        """Return the all-rank-equal common state approved for publication."""

        self._raise_if_poisoned()
        try:
            self._synchronized_stage(
                phase="checkpoint read-only state admission",
                action=self._require_read_only_managed_state,
            )
            self._synchronized_stage(
                phase="checkpoint optimizer parameter admission",
                action=self._require_optimizer_parameter_order,
            )
            runtime_state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "ema": self.ema.state_dict() if self.ema is not None else None,
            }
            managed_modules = {
                role: asset.module for role, asset in self.managed_modules.items()
            }
            self._synchronized_stage(
                phase="checkpoint managed storage admission",
                action=lambda: require_no_distinct_shared_storage_across_modules(
                    managed_modules,
                    path="distributed checkpoint managed state",
                ),
            )
            self._synchronized_stage(
                phase="checkpoint module portability admission",
                action=lambda: require_relocatable_module_states(
                    managed_modules,
                    path="distributed checkpoint managed state",
                ),
            )
            self._synchronized_stage(
                phase="checkpoint state ownership admission",
                action=lambda: require_runtime_state_disjoint_from_modules(
                    runtime_state,
                    managed_modules,
                    path="distributed checkpoint runtime state",
                ),
            )
            self._synchronized_stage(
                phase="checkpoint clone-safe runtime admission",
                action=lambda: require_clone_safe_runtime_state(
                    runtime_state,
                    path="distributed checkpoint runtime state",
                ),
            )
            self._synchronized_stage(
                phase="checkpoint scheduler portability admission",
                action=lambda: require_tensor_free_runtime_state(
                    (
                        self.lr_scheduler.state_dict()
                        if self.lr_scheduler is not None
                        else None
                    ),
                    path="distributed lr scheduler state",
                ),
            )
            fingerprint = self._synchronized_stage(
                phase="checkpoint state fingerprint",
                action=self._common_runtime_state_fingerprint,
            )
            authority = {
                "global_step": self.global_step,
                "common_runtime_state_sha256": fingerprint,
            }
            if not self.collectives.all_equal(authority):
                raise RuntimeError(
                    "distributed common runtime state differs across ranks"
                )
            return fingerprint
        except BaseException:
            self._poisoned = True
            raise

    def assert_checkpoint_publishable(self) -> None:
        """Reject publishing state after any distributed runtime failure."""

        self.checkpoint_state_fingerprint()

    @property
    def is_primary(self) -> bool:
        """Return whether this rank owns user-visible side effects."""

        return self.topology.is_primary

    def _set_training_modes(self) -> None:
        self.execution_model.train()
        for name, asset in self.managed_modules.items():
            if name == "primary_model":
                continue
            if asset.mode == "follow":
                asset.module.train()
            else:
                asset.module.eval()

    def _set_evaluation_modes(self) -> None:
        self.plan.primary_model.eval()
        for name, asset in self.managed_modules.items():
            if name != "primary_model":
                asset.module.eval()

    def _require_collective_success(
        self,
        *,
        local_error: BaseException | None,
        phase: str,
    ) -> None:
        try:
            succeeded = self.collectives.all_true(local_error is None)
        except BaseException as collective_error:
            if local_error is None:
                raise
            _add_failure_note(
                local_error,
                "distributed failure consensus also failed: "
                f"{type(collective_error).__name__}: "
                f"{_failure_summary(collective_error)}",
            )
            raise local_error from collective_error
        if succeeded:
            if local_error is not None:
                raise RuntimeError(
                    "distributed all_true returned success for a local failure"
                ) from local_error
            return
        try:
            gathered = self.collectives.gather_to_primary(
                _failure_summary(local_error)
            )
            summary: object = None
            if self.is_primary:
                assert gathered is not None
                failures = [value for value in gathered if value is not None]
                summary = {"phase": phase, "failures": failures}
            summary = self.collectives.broadcast_from_primary(summary)
        except BaseException as collective_error:
            if local_error is None:
                raise
            _add_failure_note(
                local_error,
                "distributed failure reporting also failed: "
                f"{type(collective_error).__name__}: "
                f"{_failure_summary(collective_error)}",
            )
            raise local_error from collective_error
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"distributed {phase} failed on another rank: {summary}")

    def _synchronized_stage(
        self,
        *,
        phase: str,
        action: Callable[[], StageResultT],
    ) -> StageResultT:
        local_error: BaseException | None = None
        result: StageResultT | None = None
        try:
            result = action()
        except BaseException as error:  # noqa: BLE001
            local_error = error
        self._require_collective_success(local_error=local_error, phase=phase)
        if local_error is not None:
            raise local_error
        return cast(StageResultT, result)

    def _synchronized_acquire(
        self,
        *,
        phase: str,
        acquire: Callable[[], StageResultT],
        release: Callable[[StageResultT], None],
    ) -> StageResultT:
        """Acquire one rank-local resource and release it on peer failure."""

        local_error: BaseException | None = None
        result: StageResultT | None = None
        try:
            result = acquire()
        except BaseException as error:  # noqa: BLE001
            local_error = error
        try:
            acquired_everywhere = self.collectives.all_true(local_error is None)
        except BaseException as collective_error:
            if result is not None:
                with suppress(BaseException):
                    release(result)
            if local_error is not None:
                _add_failure_note(
                    local_error,
                    f"{phase} acquisition consensus also failed: "
                    f"{_failure_summary(collective_error)}",
                )
                raise local_error from collective_error
            raise
        if acquired_everywhere:
            assert result is not None
            return result
        if result is not None:
            try:
                release(result)
            except BaseException as cleanup_error:  # noqa: BLE001
                if local_error is None:
                    local_error = cleanup_error
                else:
                    _add_failure_note(
                        local_error,
                        f"{phase} peer-failure cleanup also failed: "
                        f"{_failure_summary(cleanup_error)}",
                    )
        if local_error is None:
            local_error = RuntimeError(f"{phase} failed on another rank")
        self._require_collective_success(local_error=local_error, phase=phase)
        raise local_error

    def _synchronized_enter_context(
        self,
        context: AbstractContextManager[None],
        *,
        phase: str,
    ) -> None:
        """Enter a local context and unwind it if any peer cannot enter."""

        entered = False
        local_error: BaseException | None = None
        try:
            context.__enter__()
            entered = True
        except BaseException as error:  # noqa: BLE001
            local_error = error
        try:
            entered_everywhere = self.collectives.all_true(local_error is None)
        except BaseException as collective_error:
            if entered:
                with suppress(BaseException):
                    context.__exit__(None, None, None)
            if local_error is not None:
                _add_failure_note(
                    local_error,
                    f"{phase} consensus also failed: {_failure_summary(collective_error)}",
                )
                raise local_error from collective_error
            raise
        if entered_everywhere:
            return
        if entered:
            try:
                context.__exit__(None, None, None)
            except BaseException as cleanup_error:  # noqa: BLE001
                if local_error is None:
                    local_error = cleanup_error
                else:
                    _add_failure_note(
                        local_error,
                        f"{phase} peer-failure exit also failed: "
                        f"{_failure_summary(cleanup_error)}",
                    )
        if local_error is None:
            local_error = RuntimeError(f"{phase} failed on another rank")
        self._require_collective_success(local_error=local_error, phase=phase)
        raise local_error

    def _validate_local_train_plan(
        self,
        plan: RankedTrainEpochPlan,
        *,
        epoch_index: int,
        max_microbatches: int | None,
    ) -> None:
        if plan.rank != self.topology.rank or plan.world_size != self.topology.world_size:
            raise ValueError("ranked train plan does not match DDP topology")
        if plan.epoch != epoch_index:
            raise ValueError("ranked train plan epoch does not match the request")
        if plan.microbatches_per_window != self.accumulate_grad_batches:
            raise ValueError("ranked train plan accumulation does not match DDPTrainer")
        if plan.requested_max_microbatches != max_microbatches:
            raise ValueError(
                "ranked train plan requested_max_microbatches does not match "
                "the request"
            )

    def _require_equal_train_plans(self, plan: RankedTrainEpochPlan) -> None:
        common = {
            "data_identity": plan.data_identity.to_dict(),
            "plan_digest": plan.plan_digest,
            "epoch": plan.epoch,
            "world_size": plan.world_size,
            "microbatches_per_window": plan.microbatches_per_window,
            "window_count": plan.window_count,
            "samples_per_microbatch": plan.samples_per_microbatch,
            "local_assigned_samples": plan.local_assigned_samples,
            "global_assigned_samples": plan.global_assigned_samples,
            "global_dropped_samples": plan.global_dropped_samples,
            "requested_max_microbatches": plan.requested_max_microbatches,
        }
        if not self.collectives.all_equal(common):
            raise ValueError("ranked train plans disagree across DDP ranks")

    def _read_window(
        self,
        reader: Any,
        *,
        plan: RankedTrainEpochPlan,
        ordinal: int,
    ) -> RankedTrainWindow:
        local_error: BaseException | None = None
        window: RankedTrainWindow | None = None
        try:
            window = reader.read_window()
            if window is None:
                raise RuntimeError(
                    "ranked train reader ended before its declared window count"
                )
            self._validate_window(window, plan=plan, ordinal=ordinal)
        except BaseException as error:  # noqa: BLE001
            local_error = error
        self._require_collective_success(
            local_error=local_error,
            phase=f"training data window {ordinal}",
        )
        assert window is not None
        return window

    @staticmethod
    def _validate_window(
        window: RankedTrainWindow,
        *,
        plan: RankedTrainEpochPlan,
        ordinal: int,
    ) -> None:
        window_value = cast(object, window)
        if not isinstance(window_value, RankedTrainWindow):
            raise TypeError("ranked train reader returned the wrong window type")
        if window.ordinal != ordinal:
            raise ValueError("ranked train window ordinal is not monotonic")
        if len(window.batches) != plan.microbatches_per_window:
            raise ValueError("ranked train window is not a complete accumulation window")
        expected_first = ordinal * plan.microbatches_per_window
        for offset, facts in enumerate(window.batch_facts):
            facts_value = cast(object, facts)
            if not isinstance(facts_value, RankedBatchFacts):
                raise TypeError("ranked train window has invalid batch facts")
            if facts.ordinal != expected_first + offset:
                raise ValueError("ranked train batch ordinal is not monotonic")
            if facts.sample_count != plan.samples_per_microbatch:
                raise ValueError("ranked train batch sample count differs from its plan")
            if facts.loss_weight != float(plan.samples_per_microbatch):
                raise ValueError("ranked train batch loss weight differs from its plan")

    def _run_window(self, window: RankedTrainWindow) -> tuple[float, float, bool]:
        local_weighted_loss = 0.0
        local_weight = 0.0
        self._synchronized_stage(
            phase="gradient reset before optimizer window",
            action=lambda: self.optimizer.zero_grad(set_to_none=True),
        )
        try:
            for index, (batch, facts) in enumerate(
                zip(window.batches, window.batch_facts, strict=True)
            ):
                prepared_batch = self._synchronized_stage(
                    phase=f"training batch transfer {facts.ordinal}",
                    action=lambda batch=batch: _move_to_device(batch, self.device),
                )
                synchronize = index == len(window.batches) - 1
                context_error: BaseException | None = None
                synchronization_context: AbstractContextManager[None] | None = None
                try:
                    synchronization_context = (
                        nullcontext()
                        if synchronize
                        else self.synchronization_model.no_sync()
                    )
                except BaseException as error:  # noqa: BLE001
                    context_error = error
                self._require_collective_success(
                    local_error=context_error,
                    phase=f"training synchronization context {facts.ordinal}",
                )
                assert synchronization_context is not None
                self._synchronized_enter_context(
                    synchronization_context,
                    phase=f"training synchronization entry {facts.ordinal}",
                )
                microbatch_error: BaseException | None = None
                microbatch_skipped = False
                distributed_compute_error = False
                try:
                    def forward(
                        prepared_batch: Any = prepared_batch,
                        facts: RankedBatchFacts = facts,
                    ) -> Any:
                        with self.precision.autocast():
                            return self.execution_strategy.training_step(
                                prepared_batch
                            )

                    def validate_forward_output(
                        raw_output: object,
                        facts: RankedBatchFacts = facts,
                    ) -> tuple[torch.Tensor, float, float]:
                        output = validate_train_step_output(raw_output)
                        normalized_loss = output.loss / len(window.batches)
                        actual_weight = loss_aggregation_weight_to_float(
                            output.loss_aggregation_weight
                        )
                        if (
                            actual_weight != facts.loss_weight
                            or actual_weight != float(facts.sample_count)
                        ):
                            raise ValueError(
                                "Strategy loss weight disagrees with ranked batch facts"
                            )
                        loss_value = float(output.loss.detach().item())
                        return normalized_loss, actual_weight, loss_value

                    if synchronize:
                        try:
                            raw_output = forward()
                        except BaseException:
                            distributed_compute_error = True
                            raise
                        normalized_loss, actual_weight, loss_value = (
                            self._synchronized_stage(
                                phase=f"training output validation {facts.ordinal}",
                                action=lambda raw_output=raw_output: (
                                    validate_forward_output(raw_output)
                                ),
                            )
                        )
                    else:
                        raw_output = self._synchronized_stage(
                            phase=f"training forward {facts.ordinal}", action=forward
                        )
                        normalized_loss, actual_weight, loss_value = (
                            self._synchronized_stage(
                                phase=f"training output validation {facts.ordinal}",
                                action=lambda raw_output=raw_output: (
                                    validate_forward_output(raw_output)
                                ),
                            )
                        )
                    globally_finite_loss = self.collectives.all_true(
                        math.isfinite(loss_value)
                    )
                    if not globally_finite_loss:
                        microbatch_skipped = True
                    else:
                        if synchronize:
                            try:
                                self.precision.backward(normalized_loss)
                            except BaseException:
                                distributed_compute_error = True
                                raise
                        else:
                            self._synchronized_stage(
                                phase=f"training backward {facts.ordinal}",
                                action=lambda loss=normalized_loss: (
                                    self.precision.backward(loss)
                                ),
                            )
                        local_weighted_loss += loss_value * actual_weight
                        local_weight += actual_weight
                except BaseException as error:  # noqa: BLE001
                    microbatch_error = error
                exit_error: BaseException | None = None
                try:
                    synchronization_context.__exit__(
                        type(microbatch_error) if microbatch_error is not None else None,
                        microbatch_error,
                        (
                            microbatch_error.__traceback__
                            if microbatch_error is not None
                            else None
                        ),
                    )
                except BaseException as error:  # noqa: BLE001
                    exit_error = error
                if exit_error is not None:
                    if microbatch_error is None:
                        microbatch_error = exit_error
                    else:
                        _add_failure_note(
                            microbatch_error,
                            "training synchronization exit also failed: "
                            f"{type(exit_error).__name__}: "
                            f"{_failure_summary(exit_error)}",
                        )
                if distributed_compute_error:
                    self._collectives_unsafe = True
                    assert microbatch_error is not None
                    raise microbatch_error
                self._require_collective_success(
                    local_error=microbatch_error,
                    phase=f"training microbatch cleanup {facts.ordinal}",
                )
                if microbatch_error is not None:
                    raise microbatch_error
                if microbatch_skipped:
                    self._synchronized_stage(
                        phase="gradient reset after non-finite loss",
                        action=lambda: self.optimizer.zero_grad(set_to_none=True),
                    )
                    raise FloatingPointError(
                        "fixed DDP encountered a non-finite loss on at least one rank"
                    )

            def inspect_gradients() -> bool:
                self.precision.unscale_(self.optimizer)
                if any(
                    parameter.grad is None
                    for parameter in self.trainable_parameters
                ):
                    return False
                if self.max_grad_norm is not None:
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        self.max_grad_norm,
                    )
                    return math.isfinite(
                        float(norm.detach().item())
                    ) and _gradient_state_is_finite(self.trainable_parameters)
                return _gradient_state_is_finite(self.trainable_parameters)

            local_finite = self._synchronized_stage(
                phase="gradient inspection",
                action=inspect_gradients,
            )
            globally_finite = self.collectives.all_true(local_finite)
            if not globally_finite:
                self._synchronized_stage(
                    phase="gradient reset after non-finite gradients",
                    action=lambda: self.optimizer.zero_grad(set_to_none=True),
                )
                raise FloatingPointError(
                    "fixed DDP encountered missing or non-finite gradients on "
                    "at least one rank"
                )

            self._synchronized_stage(
                phase="read-only state check before optimizer commit",
                action=self._require_read_only_managed_state,
            )

            self._synchronized_stage(
                phase="optimizer commit",
                action=self.optimizer.step,
            )

            def commit_followup_state() -> None:
                if self.ema is not None:
                    self.ema.update(self.plan.primary_model)
                if (
                    self.lr_scheduler is not None
                    and self.lr_scheduler_interval == "step"
                ):
                    self.lr_scheduler.step()

            self._synchronized_stage(
                phase="optimizer follow-up state",
                action=commit_followup_state,
            )
            self._synchronized_stage(
                phase="gradient reset after optimizer window",
                action=lambda: self.optimizer.zero_grad(set_to_none=True),
            )
            self.global_step += 1
            return local_weighted_loss, local_weight, True
        except BaseException as error:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:  # noqa: BLE001
                _add_failure_note(
                    error,
                    "gradient cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {_failure_summary(cleanup_error)}",
                )
            raise

    def train_epoch(
        self,
        execution: RankedTrainExecution,
        *,
        epoch_index: int,
        max_microbatches: int | None = None,
        preplanned: RankedTrainEpochPlan | None = None,
    ) -> DDPTrainEpochResult:
        """Train one exact ranked epoch and poison this runtime on failure."""

        self._raise_if_poisoned()
        try:
            return self._train_epoch_impl(
                execution,
                epoch_index=epoch_index,
                max_microbatches=max_microbatches,
                preplanned=preplanned,
            )
        except BaseException:
            self._poisoned = True
            raise

    def _train_epoch_impl(
        self,
        execution: RankedTrainExecution,
        *,
        epoch_index: int,
        max_microbatches: int | None = None,
        preplanned: RankedTrainEpochPlan | None = None,
    ) -> DDPTrainEpochResult:
        """Train one exact ranked epoch and return global loss facts."""

        self._runtime_started = True
        if not self.collectives.all_equal({"global_step": self.global_step}):
            raise ValueError("DDP ranks entered training with different global_step")

        def plan_epoch() -> RankedTrainEpochPlan:
            execution_value = cast(object, execution)
            if not isinstance(execution_value, RankedTrainExecution):
                raise TypeError("DDP training requires RankedTrainExecution")
            return execution.plan_epoch(
                epoch_index,
                microbatches_per_window=self.accumulate_grad_batches,
                max_microbatches=max_microbatches,
            )

        plan = (
            self._synchronized_stage(
                phase="training epoch planning",
                action=plan_epoch,
            )
            if preplanned is None
            else preplanned
        )
        self._synchronized_stage(
            phase="training local plan validation",
            action=lambda: self._validate_local_train_plan(
                plan,
                epoch_index=epoch_index,
                max_microbatches=max_microbatches,
            ),
        )
        self._require_equal_train_plans(plan)

        def open_reader() -> Any:
            reader = execution.open_epoch(plan)
            if reader.plan != plan:
                raise ValueError("ranked train reader changed its issued plan")
            return reader

        reader = self._synchronized_acquire(
            phase="training reader open",
            acquire=open_reader,
            release=lambda value: value.close(),
        )
        self._synchronized_stage(
            phase="training mode entry",
            action=self._set_training_modes,
        )
        started_at = time.perf_counter()
        local_weighted_loss = 0.0
        local_weight = 0.0
        successful_steps = 0
        skipped_steps = 0
        body_error: BaseException | None = None
        completion: RankedEpochCompletion | None = None
        try:
            for ordinal in range(plan.window_count):
                window = self._read_window(
                    reader,
                    plan=plan,
                    ordinal=ordinal,
                )
                weighted_loss, weight, succeeded = self._run_window(window)
                local_weighted_loss += weighted_loss
                local_weight += weight
                successful_steps += int(succeeded)
                skipped_steps += int(not succeeded)
            finished = reader.finish()
            self._validate_completion(finished, plan=plan)
            completion = finished
        except BaseException as error:  # noqa: BLE001
            body_error = error
        finally:
            try:
                reader.close()
            except BaseException as cleanup_error:  # noqa: BLE001
                if body_error is None:
                    body_error = cleanup_error
                else:
                    _add_failure_note(
                        body_error,
                        "ranked train reader close also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    )
        if self._collectives_unsafe:
            if body_error is None:
                raise RuntimeError(
                    "distributed compute failed after process-group communication"
                )
            raise body_error
        self._require_collective_success(
            local_error=body_error,
            phase="training epoch completion",
        )
        if body_error is not None:
            raise body_error
        assert completion is not None

        expected_completion = {
            "rank": plan.rank,
            "plan_digest": plan.plan_digest,
            "assignment_digest": plan.assignment_digest,
            "terminal_token": plan.expected_terminal_token,
        }
        gathered_expectations = self.collectives.gather_to_primary(
            expected_completion
        )
        gathered_completions = self.collectives.gather_to_primary(completion)
        completion_validation_error: BaseException | None = None
        global_completions: object = None
        if self.is_primary:
            assert gathered_completions is not None
            assert gathered_expectations is not None
            try:
                ordered: list[RankedEpochCompletion] = []
                for rank, value in enumerate(gathered_completions):
                    if not isinstance(value, RankedEpochCompletion):
                        raise TypeError(
                            "distributed completion gather contains an invalid value"
                        )
                    if value.rank != rank or value.plan_digest != plan.plan_digest:
                        raise ValueError(
                            "distributed completion does not match rank or plan"
                        )
                    expected_value = gathered_expectations[rank]
                    if not isinstance(expected_value, dict):
                        raise TypeError(
                            "distributed completion expectation has invalid type"
                        )
                    if expected_value != {
                        "rank": rank,
                        "plan_digest": plan.plan_digest,
                        "assignment_digest": value.assignment_digest,
                        "terminal_token": value.terminal_token,
                    }:
                        raise ValueError(
                            "distributed completion does not match its expected "
                            "terminal commitment"
                        )
                    ordered.append(value)
                global_completions = tuple(ordered)
            except BaseException as error:  # noqa: BLE001
                completion_validation_error = error
        self._require_collective_success(
            local_error=completion_validation_error,
            phase="training completion inventory",
        )
        global_completions = self.collectives.broadcast_from_primary(
            global_completions
        )
        if not isinstance(global_completions, tuple) or not all(
            isinstance(value, RankedEpochCompletion)
            for value in global_completions
        ):
            raise TypeError("rank zero broadcast an invalid completion inventory")

        finite_aggregation = (
            math.isfinite(local_weighted_loss)
            and math.isfinite(local_weight)
            and local_weight >= 0.0
        )
        if not self.collectives.all_true(finite_aggregation):
            raise RuntimeError("distributed training loss aggregation is non-finite")
        global_weighted_loss = self.collectives.sum_float(local_weighted_loss)
        global_weight = self.collectives.sum_float(local_weight)
        if global_weight <= 0.0:
            raise RuntimeError("distributed training produced no loss weight")
        minimum_successful_steps = self.collectives.min_int(successful_steps)
        maximum_successful_steps = self.collectives.max_int(successful_steps)
        minimum_skipped_steps = self.collectives.min_int(skipped_steps)
        maximum_skipped_steps = self.collectives.max_int(skipped_steps)
        if minimum_successful_steps != maximum_successful_steps:
            raise RuntimeError("DDP ranks disagree about successful optimizer steps")
        if minimum_skipped_steps != maximum_skipped_steps:
            raise RuntimeError("DDP ranks disagree about skipped optimizer steps")
        global_successful_steps = self.collectives.sum_int(successful_steps)
        global_skipped_steps = self.collectives.sum_int(skipped_steps)
        expected_multiplier = self.topology.world_size
        if global_successful_steps % expected_multiplier != 0:
            raise RuntimeError("DDP ranks disagree about successful optimizer steps")
        if global_skipped_steps % expected_multiplier != 0:
            raise RuntimeError("DDP ranks disagree about skipped optimizer steps")
        completed_optimizer_steps = global_successful_steps // expected_multiplier
        if (
            completed_optimizer_steps > 0
            and self.lr_scheduler is not None
            and self.lr_scheduler_interval == "epoch"
        ):
            self._synchronized_stage(
                phase="epoch scheduler commit",
                action=self.lr_scheduler.step,
            )
        duration_microseconds = self.collectives.max_int(
            math.ceil((time.perf_counter() - started_at) * 1_000_000)
        )
        metrics = {
            "loss": global_weighted_loss / global_weight,
            "num_batches": float(plan.microbatch_count * self.topology.world_size),
            "micro_batches": float(plan.microbatch_count * self.topology.world_size),
            "optimizer_steps": float(completed_optimizer_steps),
            "skipped_optimizer_steps": float(
                global_skipped_steps // expected_multiplier
            ),
            "duration_seconds": duration_microseconds / 1_000_000.0,
        }
        return DDPTrainEpochResult(
            metrics=metrics,
            local_completion=completion,
            global_completions=cast(
                tuple[RankedEpochCompletion, ...],
                global_completions,
            ),
        )

    @staticmethod
    def _validate_completion(
        completion: RankedEpochCompletion,
        *,
        plan: RankedTrainEpochPlan,
    ) -> None:
        completion_value = cast(object, completion)
        if not isinstance(completion_value, RankedEpochCompletion):
            raise TypeError("ranked train reader returned invalid completion")
        if completion.plan_digest != plan.plan_digest or completion.rank != plan.rank:
            raise ValueError("ranked train completion does not match its plan")
        if completion.observed_windows != plan.window_count:
            raise ValueError("ranked train completion has the wrong window count")
        if completion.observed_microbatches != plan.microbatch_count:
            raise ValueError("ranked train completion has the wrong microbatch count")
        if completion.observed_samples != plan.local_assigned_samples:
            raise ValueError("ranked train completion has the wrong sample count")
        if completion.assignment_digest != plan.assignment_digest:
            raise ValueError("ranked train completion has the wrong assignment digest")
        if completion.terminal_token != plan.expected_terminal_token:
            raise ValueError("ranked train completion has the wrong terminal token")

    def evaluate_epoch(
        self,
        execution: ExactValidationExecution,
        *,
        epoch_index: int,
    ) -> dict[str, float]:
        """Evaluate one exact epoch and poison this runtime on failure."""

        self._raise_if_poisoned()
        try:
            return self._evaluate_epoch_impl(
                execution,
                epoch_index=epoch_index,
            )
        except BaseException:
            self._poisoned = True
            raise

    def _evaluate_epoch_impl(
        self,
        execution: ExactValidationExecution,
        *,
        epoch_index: int,
    ) -> dict[str, float]:
        """Evaluate a rank-zero full view with no DDP forward collectives."""

        self._runtime_started = True
        if not self.collectives.all_equal({"global_step": self.global_step}):
            raise ValueError("DDP ranks entered validation with different global_step")

        def plan_epoch() -> Any:
            execution_value = cast(object, execution)
            if not isinstance(execution_value, ExactValidationExecution):
                raise TypeError("DDP validation requires ExactValidationExecution")
            return execution.plan_epoch(epoch_index)

        plan = self._synchronized_stage(
            phase="validation epoch planning",
            action=plan_epoch,
        )

        def validate_local_plan() -> None:
            if plan.epoch != epoch_index:
                raise ValueError(
                    "validation plan epoch does not match the request"
                )
            if (
                plan.rank != self.topology.rank
                or plan.world_size != self.topology.world_size
            ):
                raise ValueError("validation plan does not match DDP topology")
            expected_spans = (
                (ExactCoverageSpan(0, plan.global_expected_samples),)
                if self.is_primary
                else ()
            )
            if plan.local_spans != expected_spans:
                raise ValueError(
                    "fixed DDP validation requires rank zero full coverage and "
                    "empty peer coverage"
                )

        self._synchronized_stage(
            phase="validation local plan admission",
            action=validate_local_plan,
        )
        common = {
            "coverage_identity": plan.coverage_identity.to_dict(),
            "plan_digest": plan.plan_digest,
            "epoch": plan.epoch,
            "world_size": plan.world_size,
            "global_expected_samples": plan.global_expected_samples,
            "primary_batch_count": plan.primary_batch_count,
        }
        if not self.collectives.all_equal(common):
            raise ValueError("validation plans disagree across DDP ranks")

        def open_reader() -> Any:
            reader = execution.open_epoch(plan)
            if reader.plan != plan:
                raise ValueError("validation reader changed its issued plan")
            return reader

        reader = self._synchronized_acquire(
            phase="validation reader open",
            acquire=open_reader,
            release=lambda value: value.close(),
        )
        self._synchronized_stage(
            phase="validation mode entry",
            action=self._set_evaluation_modes,
        )

        def reset_metrics() -> None:
            if self.metric_runtime is not None and self.is_primary:
                self.metric_runtime.reset_phase("validation")

        self._synchronized_stage(
            phase="validation metric reset",
            action=reset_metrics,
        )
        local_weighted_loss = 0.0
        local_weight = 0.0
        local_batches = 0
        body_error: BaseException | None = None
        receipt: ExactCoverageReceipt | None = None
        try:
            with torch.inference_mode():
                for batch_ordinal in range(plan.primary_batch_count):
                    batch_error: BaseException | None = None
                    try:
                        item = reader.read_batch()
                        if self.is_primary:
                            if item is None:
                                raise ValueError(
                                    "rank-zero validation reader ended before its "
                                    "declared batch count"
                                )
                            if item.facts.ordinal != batch_ordinal:
                                raise ValueError(
                                    "validation batch ordinal is not monotonic"
                                )
                            prepared_batch = _move_to_device(item.batch, self.device)
                            with self.precision.autocast():
                                output = validate_train_step_output(
                                    self.plan.strategy.evaluation_step(prepared_batch)
                                )
                            weight = loss_aggregation_weight_to_float(
                                output.loss_aggregation_weight
                            )
                            if weight != item.facts.loss_weight or weight != float(
                                item.facts.sample_count
                            ):
                                raise ValueError(
                                    "validation Strategy loss weight disagrees with "
                                    "data facts"
                                )
                            local_weighted_loss += (
                                float(output.loss.detach().item()) * weight
                            )
                            local_weight += weight
                            local_batches += 1
                            if self.metric_runtime is not None:
                                self.metric_runtime.update_phase(
                                    "validation",
                                    output.metric_updates,
                                )
                        elif item is not None:
                            raise ValueError(
                                "non-primary rank returned a validation batch"
                            )
                    except BaseException as error:  # noqa: BLE001
                        batch_error = error
                    self._require_collective_success(
                        local_error=batch_error,
                        phase=f"validation batch {batch_ordinal}",
                    )
                    if batch_error is not None:
                        raise batch_error
                extra_item = reader.read_batch()
                if extra_item is not None:
                    raise ValueError(
                        "validation reader exceeded its declared batch count"
                    )
            receipt = reader.finish()
        except BaseException as error:  # noqa: BLE001
            body_error = error
        finally:
            try:
                reader.close()
            except BaseException as cleanup_error:  # noqa: BLE001
                if body_error is None:
                    body_error = cleanup_error
                else:
                    _add_failure_note(
                        body_error,
                        "validation reader close also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    )
        self._require_collective_success(
            local_error=body_error,
            phase="validation",
        )
        assert receipt is not None
        gathered = self.collectives.gather_to_primary(receipt)
        primary_result: object = None
        finalization_error: BaseException | None = None
        if self.is_primary:
            assert gathered is not None
            try:
                self._validate_primary_full_coverage(
                    gathered,
                    expected_samples=plan.global_expected_samples,
                    expected_plan_digest=plan.plan_digest,
                )
                if local_weight <= 0.0:
                    raise ValueError("rank-zero validation view yielded no samples")
                result = {
                    "loss": local_weighted_loss / local_weight,
                    "num_batches": float(local_batches),
                }
                if self.metric_runtime is not None:
                    result.update(
                        self.metric_runtime.compute_phase("validation", reset=True)
                    )
                primary_result = result
            except BaseException as error:  # noqa: BLE001
                finalization_error = error
        self._require_collective_success(
            local_error=finalization_error,
            phase="validation finalization",
        )
        self._synchronized_stage(
            phase="validation read-only state check",
            action=self._require_read_only_managed_state,
        )
        broadcast_result = self.collectives.broadcast_from_primary(primary_result)
        if not isinstance(broadcast_result, dict):
            raise TypeError("rank zero broadcast an invalid validation result")
        return {str(name): float(value) for name, value in broadcast_result.items()}

    @staticmethod
    def _validate_primary_full_coverage(
        gathered: tuple[object, ...],
        *,
        expected_samples: int,
        expected_plan_digest: str,
    ) -> None:
        receipts: dict[int, ExactCoverageReceipt] = {}
        for value in gathered:
            if not isinstance(value, ExactCoverageReceipt):
                raise TypeError("validation gather contains an invalid receipt")
            if value.rank in receipts:
                raise ValueError("validation gather contains duplicate ranks")
            if value.plan_digest != expected_plan_digest:
                raise ValueError("validation receipt does not match its plan")
            receipts[value.rank] = value
        if set(receipts) != set(range(len(gathered))):
            raise ValueError("validation gather is missing a rank receipt")
        expected = (ExactCoverageSpan(0, expected_samples),)
        if receipts[0].completed_spans != expected:
            raise ValueError("rank zero did not complete the full validation view")
        if receipts[0].observed_samples != expected_samples:
            raise ValueError("rank zero validation count is incomplete")
        for rank, receipt in receipts.items():
            if rank != 0 and (
                receipt.completed_spans or receipt.observed_samples != 0
            ):
                raise ValueError("non-primary rank observed validation samples")


__all__ = [
    "DDPExecutionBinding",
    "DDPTrainEpochResult",
    "DDPTrainer",
    "GradientSynchronizationModel",
]
