"""Runtime composition for the first fixed-topology DDP training path."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow._builtin_activation import require_training_builtins
from stochaflow._component_factory import (
    build_model,
    build_objective,
    build_process,
)
from stochaflow.processes.base import Process
from stochaflow.training.builder import (
    TrainingPlan,
    TrainingPlanAssembly,
    build_training_plan_assembly,
    trainable_parameters,
    training_module_roots,
)
from stochaflow.training.ddp_trainer import DDPExecutionBinding, DDPTrainer
from stochaflow.training.distributed.session import DistributedSession
from stochaflow.training.distributed.state_fingerprint import (
    module_runtime_state,
    require_no_distinct_shared_storage,
    require_no_distinct_shared_storage_across_modules,
    require_relocatable_module_states,
    runtime_state_fingerprint,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.metric_binding import TrainingMetricRuntime
from stochaflow.training.optimization import build_lr_scheduler, build_optimizer
from stochaflow.training.precision import PrecisionRuntime, build_precision_runtime
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    inference_asset_descriptors_from_projections,
)
from stochaflow.utils.config import StochaflowConfig
from stochaflow.utils.device import move_module_to_device
from stochaflow.utils.logging_contracts import ExperimentLogger, NullLogger

type DDPWrapperFactory = Callable[[nn.Module, DistributedSession], nn.Module]


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _error_summary(error: BaseException | None) -> dict[str, str] | None:
    if error is None:
        return None
    try:
        message = str(error)
    except BaseException:  # noqa: BLE001 - reporting must not replace the cause
        message = "<exception text could not be rendered>"
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": message[:1000],
    }


def _add_error_note(error: BaseException, note: str) -> None:
    with suppress(BaseException):
        BaseException.add_note(error, note)


def _composition_stage[CompositionResultT](
    session: DistributedSession,
    *,
    phase: str,
    action: Callable[[], CompositionResultT],
) -> CompositionResultT:
    """Finish one local composition stage before any rank advances."""

    local_error: BaseException | None = None
    result: CompositionResultT | None = None
    try:
        result = action()
    except BaseException as error:  # noqa: BLE001
        local_error = error
    try:
        succeeded = session.collectives.all_true(local_error is None)
    except BaseException as collective_error:
        if local_error is None:
            raise
        _add_error_note(
            local_error,
            f"distributed composition {phase} consensus also failed: "
            f"{_error_summary(collective_error)}",
        )
        raise local_error from collective_error
    if succeeded:
        if local_error is not None:
            raise RuntimeError(
                f"distributed composition {phase} reported impossible success"
            ) from local_error
        return cast(CompositionResultT, result)
    try:
        gathered = session.collectives.gather_to_primary(
            _error_summary(local_error)
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
        _add_error_note(
            local_error,
            f"distributed composition {phase} reporting also failed: "
            f"{_error_summary(collective_error)}",
        )
        raise local_error from collective_error
    if local_error is not None:
        _add_error_note(local_error, f"distributed composition summary: {summary!r}")
        raise local_error
    raise RuntimeError(
        f"distributed composition {phase} failed on another rank: {summary!r}"
    )


@dataclass(slots=True)
class DDPTrainingComponents:
    """Complete side-effect-free composition for one active DDP rank."""

    model: nn.Module
    process: Process | None
    objective: nn.Module | None
    assembly: TrainingPlanAssembly
    binding: DDPExecutionBinding
    optimizer: Optimizer
    lr_scheduler: LRScheduler | None
    ema: ExponentialMovingAverage | None
    precision: PrecisionRuntime
    logger: ExperimentLogger
    metric_runtime: TrainingMetricRuntime
    checkpoint_manager: CheckpointManager
    trainer: DDPTrainer
    checkpoint_metadata: dict[str, Any] | None

    @property
    def plan(self) -> TrainingPlan:
        """Return the canonical plan retained by the assembly and binding."""

        return self.assembly.plan


def _validate_first_ddp_scope(config: StochaflowConfig) -> None:
    if config.diagnostics:
        raise ValueError("fixed DDP does not yet support training Diagnostics")
    if config.trainer.validation_evaluation.enabled:
        raise ValueError(
            "fixed DDP does not yet support epoch live Evaluation"
        )
    if config.trainer.test_after_fit:
        raise ValueError("fixed DDP does not yet support test-after-fit")
    unsupported_metric_phases = sorted(
        {
            phase
            for declaration in config.metrics
            for phase in declaration.phases
            if phase != "validation"
        }
    )
    if unsupported_metric_phases:
        raise ValueError(
            "fixed DDP accepts validation-only training Metrics; unsupported "
            f"phase(s): {', '.join(unsupported_metric_phases)}"
        )
    if config.lr_scheduler is not None and config.lr_scheduler.interval == "epoch":
        raise ValueError(
            "fixed DDP does not yet support epoch-interval lr schedulers"
        )
    if config.trainer.precision == "fp16-mixed":
        raise ValueError(
            "fixed DDP currently supports fp32 and bf16-mixed; fp16-mixed "
            "needs a global GradScaler commit contract"
        )


def _module_state_entries(
    module: nn.Module,
) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        *((f"parameter:{name}", value) for name, value in module.named_parameters()),
        *((f"buffer:{name}", value) for name, value in module.named_buffers()),
    )


def _state_fingerprint(module: nn.Module, *, role: str) -> str:
    return runtime_state_fingerprint(
        module_runtime_state(module),
        path=f"fixed DDP {role} state",
    )


def _validate_plan_state_admission(plan: TrainingPlan) -> None:
    require_no_distinct_shared_storage(
        plan.primary_model,
        path="fixed DDP primary model",
    )
    selected = trainable_parameters(plan)
    primary_trainable = tuple(
        parameter
        for parameter in plan.primary_model.parameters()
        if parameter.requires_grad
    )
    if tuple(map(id, selected)) != tuple(map(id, primary_trainable)):
        raise ValueError(
            "fixed DDP requires the primary model to be the only trainable root"
        )

    primary_buffers = tuple(name for name, _ in plan.primary_model.named_buffers())
    if primary_buffers:
        raise ValueError(
            "fixed DDP disables buffer broadcasts and therefore does not yet "
            "accept primary-model buffers: " + ", ".join(primary_buffers)
        )

    for role, module in training_module_roots(plan):
        if role == "primary_model":
            continue
        state_entries = _module_state_entries(module)
        state_names = [name for name, _ in state_entries]
        if module is plan.process:
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError("fixed DDP requires Process state to be frozen")
            require_no_distinct_shared_storage(
                module,
                path=f"fixed DDP {role}",
            )
            _state_fingerprint(module, role=role)
            continue
        state_dict_names = set(module.state_dict())
        registered_state_names = {
            name.partition(":")[2] for name in state_names
        }
        extra_state_names = [
            f"state:{name}"
            for name in sorted(state_dict_names - registered_state_names)
        ]
        if extra_state_names:
            details = ", ".join(extra_state_names)
            raise ValueError(
                "fixed DDP does not yet accept extra state on non-primary "
                f"training root '{role}': {details}"
            )
        if state_names:
            details = ", ".join(state_names)
            raise ValueError(
                "fixed DDP does not yet accept stateful non-primary training "
                f"root '{role}': {details}"
            )


def _require_equal_frozen_state(
    frozen_roots: tuple[tuple[str, str, str], ...],
    session: DistributedSession,
) -> None:
    if not session.collectives.all_equal(frozen_roots):
        raise ValueError(
            "fixed DDP non-primary frozen state differs across ranks"
        )


def _wrap_with_distributed_data_parallel(
    module: nn.Module,
    session: DistributedSession,
) -> nn.Module:
    device_ids: list[int] | None = None
    output_device: int | None = None
    if session.backend == "nccl":
        if session.device.type != "cuda":
            raise ValueError("NCCL DDP requires a CUDA session device")
        device_index = session.device.index
        if device_index is None:
            raise ValueError("NCCL DDP requires an indexed local CUDA device")
        device_ids = [device_index]
        output_device = device_index
    elif session.backend != "gloo":
        raise ValueError("fixed DDP supports only NCCL and Gloo backends")
    elif session.device.type != "cpu":
        raise ValueError("Gloo DDP requires a CPU session device")
    return DistributedDataParallel(
        module,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=False,
        init_sync=True,
        find_unused_parameters=True,
    )


def _build_ema(
    config: StochaflowConfig,
    model: nn.Module,
) -> ExponentialMovingAverage | None:
    if not config.ema.enabled:
        return None
    return ExponentialMovingAverage(
        model,
        decay=config.ema.decay,
        update_after_step=config.ema.update_after_step,
        update_every=config.ema.update_every,
    )


def build_ddp_training_components(
    config: StochaflowConfig,
    session: DistributedSession,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> DDPTrainingComponents:
    """Compose one active rank with the maintained real DDP wrapper."""

    return _build_ddp_training_components(
        config,
        session,
        checkpoint_metadata=checkpoint_metadata,
        wrapper_factory=_wrap_with_distributed_data_parallel,
    )


def _build_ddp_training_components(
    config: StochaflowConfig,
    session: DistributedSession,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
    wrapper_factory: DDPWrapperFactory,
) -> DDPTrainingComponents:
    """Internal test seam for staged fixed-DDP composition."""

    config_value = cast(object, config)
    if not isinstance(config_value, StochaflowConfig):
        raise TypeError("DDP composition requires StochaflowConfig")
    session_value = cast(object, session)
    if not isinstance(session_value, DistributedSession):
        raise TypeError("DDP composition requires DistributedSession")
    if not callable(wrapper_factory):
        raise TypeError("DDP wrapper_factory must be callable")
    if checkpoint_metadata is not None and type(checkpoint_metadata) is not dict:
        raise TypeError("DDP checkpoint_metadata must be a dictionary or None")

    # Reading these properties requires an active, fully initialized session.
    device = session.device
    collectives = session.collectives

    def validate_config() -> None:
        _validate_first_ddp_scope(config)
        config.validate()
        require_training_builtins()

    _composition_stage(
        session,
        phase="configuration admission",
        action=validate_config,
    )
    if not collectives.all_equal(config.to_dict()):
        raise ValueError("fixed DDP configuration differs across ranks")

    def build_assets() -> tuple[nn.Module, Process | None, nn.Module | None]:
        built_model = build_model(config.model)
        built_process = (
            build_process(config.process) if config.process is not None else None
        )
        built_objective = (
            build_objective(config.objective)
            if config.objective is not None
            else None
        )
        return built_model, built_process, built_objective

    model, process, objective = _composition_stage(
        session,
        phase="primary asset construction",
        action=build_assets,
    )

    def build_assembly() -> TrainingPlanAssembly:
        result = build_training_plan_assembly(
            config.training,
            primary_model=model,
            process=process,
            objective=objective,
            model_factory=build_model,
            objective_factory=build_objective,
        )
        _validate_plan_state_admission(result.plan)
        return result

    assembly = _composition_stage(
        session,
        phase="training plan construction",
        action=build_assembly,
    )
    semantic_authority = {
        "model_type": _type_identity(model),
        "process_type": _type_identity(process) if process is not None else None,
        "objective_type": (
            _type_identity(objective) if objective is not None else None
        ),
        "training_builder": assembly.builder_name,
        "strategy_type": _type_identity(assembly.plan.strategy),
    }
    if not collectives.all_equal(semantic_authority):
        raise ValueError("fixed DDP component identities differ across ranks")

    _composition_stage(
        session,
        phase="pre-placement storage admission",
        action=lambda: (
            require_no_distinct_shared_storage_across_modules(
                dict(training_module_roots(assembly.plan)),
                path="fixed DDP managed state",
            ),
            require_relocatable_module_states(
                dict(training_module_roots(assembly.plan)),
                path="fixed DDP checkpoint state",
            )
        ),
    )

    preplacement_state = _composition_stage(
        session,
        phase="pre-placement state fingerprint",
        action=lambda: tuple(
            (
                role,
                _state_fingerprint(module, role=role),
            )
            for role, module in training_module_roots(assembly.plan)
        ),
    )
    if not collectives.all_equal(preplacement_state):
        raise ValueError("fixed DDP managed state differs across ranks")

    def move_roots() -> None:
        for role, module in training_module_roots(assembly.plan):
            move_module_to_device(module, device, role=role)

    _composition_stage(
        session,
        phase="managed module placement",
        action=move_roots,
    )
    frozen_roots = _composition_stage(
        session,
        phase="frozen state fingerprint",
        action=lambda: tuple(
            (
                role,
                _type_identity(module),
                _state_fingerprint(module, role=role),
            )
            for role, module in training_module_roots(assembly.plan)
            if role != "primary_model"
        ),
    )
    _require_equal_frozen_state(frozen_roots, session)

    execution_root = _composition_stage(
        session,
        phase="execution root construction",
        action=assembly.build_primary_execution_module,
    )
    # The real DDP constructor performs collectives internally. If one rank
    # fails after entering that protocol, starting a second control collective
    # here would mismatch peers still inside DDP. Let the launcher terminate
    # the job; every purely local prerequisite above already reached consensus.
    execution_model = wrapper_factory(execution_root, session)
    binding = _composition_stage(
        session,
        phase="execution strategy binding",
        action=lambda: DDPExecutionBinding.from_prepared(
            assembly,
            execution_root=execution_root,
            execution_model=execution_model,
        ),
    )
    execution_authority = _composition_stage(
        session,
        phase="execution semantic fingerprint",
        action=lambda: {
            "root": module_runtime_state(execution_root),
            "wrapper_type": _type_identity(execution_model),
            "strategy_type": _type_identity(binding.execution_strategy),
        },
    )
    execution_authority_sha256 = _composition_stage(
        session,
        phase="execution semantic digest",
        action=lambda: runtime_state_fingerprint(
            execution_authority,
            path="fixed DDP execution authority",
        ),
    )
    if not collectives.all_equal(execution_authority_sha256):
        raise ValueError("fixed DDP execution semantics differ across ranks")

    def build_runtime() -> DDPTrainingComponents:
        parameters = trainable_parameters(assembly.plan)
        optimizer = build_optimizer(config.optimizer, parameters)
        lr_scheduler = build_lr_scheduler(config.lr_scheduler, optimizer)
        ema = _build_ema(config, assembly.plan.primary_model)
        precision = build_precision_runtime(config.trainer.precision, device)
        metric_runtime = TrainingMetricRuntime(
            config.metrics,
            binding.execution_strategy,
            device=device,
        )
        checkpoint_manager = CheckpointManager(
            model=assembly.plan.primary_model,
            process=assembly.plan.process,
            objective=assembly.plan.objective,
            auxiliary_modules={
                name: asset.module
                for name, asset in assembly.plan.auxiliary_modules.items()
            },
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            precision_kind=precision.kind,
            grad_scaler=precision.grad_scaler,
            inference_asset_descriptors=inference_asset_descriptors_from_projections(
                assembly.plan.inference_assets
            ),
            inference_recipe=assembly.plan.inference_recipe,
        )
        logger = NullLogger()
        trainer = DDPTrainer(
            binding=binding,
            optimizer=optimizer,
            collectives=collectives,
            topology=session.topology,
            device=device,
            precision=precision,
            accumulate_grad_batches=config.trainer.accumulate_grad_batches,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=(
                config.lr_scheduler.interval
                if config.lr_scheduler is not None
                else "step"
            ),
            ema=ema,
            metric_runtime=metric_runtime,
            max_grad_norm=config.trainer.max_grad_norm,
        )
        metadata_snapshot = (
            deepcopy(checkpoint_metadata)
            if checkpoint_metadata is not None
            else None
        )
        return DDPTrainingComponents(
            model=model,
            process=process,
            objective=objective,
            assembly=assembly,
            binding=binding,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            precision=precision,
            logger=logger,
            metric_runtime=metric_runtime,
            checkpoint_manager=checkpoint_manager,
            trainer=trainer,
            checkpoint_metadata=metadata_snapshot,
        )

    components = _composition_stage(
        session,
        phase="distributed runtime construction",
        action=build_runtime,
    )
    components.trainer.checkpoint_state_fingerprint()
    return components


__all__ = [
    "DDPTrainingComponents",
    "DDPWrapperFactory",
    "build_ddp_training_components",
]
