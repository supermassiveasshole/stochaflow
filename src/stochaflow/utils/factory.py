"""Component registries and builder utilities."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.processes import Process
from stochaflow.training import (
    DiagnosticBuildContext,
    PrecisionRuntime,
    Trainer,
    TrainingDiagnostic,
    TrainingPlan,
    build_precision_runtime,
    build_training_plan,
    trainable_parameters,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.optimization import build_lr_scheduler, build_optimizer
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ComponentConfig,
    EMAConfig,
    ExperimentConfig,
    LoggingConfig,
    StochaflowConfig,
)
from stochaflow.utils.logging import (
    CompositeLogger,
    ExperimentLogger,
    configure_torch_logging,
)
from stochaflow.utils.registry import REGISTRIES, Registry, RegistryError

BUILTIN_COMPONENT_MODULES = (
    "stochaflow.data",
    "stochaflow.models",
    "stochaflow.processes",
    "stochaflow.sampling",
    "stochaflow.training",
    "stochaflow.training.diagnostics",
)


def load_builtin_components() -> None:
    """Import built-in component modules so their registry decorators run."""

    for module_name in BUILTIN_COMPONENT_MODULES:
        import_module(module_name)


load_builtin_components()


@dataclass(slots=True)
class TrainingComponents:
    """Fully built training components for an experiment."""

    model: nn.Module
    process: Process | None
    objective: nn.Module | None
    plan: TrainingPlan
    optimizer: Optimizer
    lr_scheduler: LRScheduler | None
    ema: ExponentialMovingAverage | None
    use_ema_for_sampling: bool
    precision: PrecisionRuntime
    logger: ExperimentLogger
    diagnostics: list[TrainingDiagnostic]
    checkpoint_manager: CheckpointManager
    trainer: Trainer


def _build_from_registry(
    registry: Registry[Any],
    component: ComponentConfig,
    *,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    extra_kwargs = extra_kwargs or {}
    kwargs = {**component.params, **extra_kwargs}
    return registry.create(component.name, **kwargs)


def build_model(component: ComponentConfig) -> nn.Module:
    """Instantiate a model from the model registry."""

    return cast(nn.Module, _build_from_registry(REGISTRIES.models, component))


def build_process(component: ComponentConfig) -> Process:
    """Instantiate a model-free probability process."""

    return cast(Process, _build_from_registry(REGISTRIES.processes, component))


def build_objective(component: ComponentConfig) -> nn.Module:
    """Instantiate a training objective from the objective registry."""

    return cast(
        nn.Module,
        _build_from_registry(
            REGISTRIES.objectives,
            component,
        ),
    )


def build_logger(
    config: LoggingConfig,
    *,
    experiment: ExperimentConfig,
    resolved_config: StochaflowConfig,
) -> ExperimentLogger:
    """Instantiate and compose experiment logging backends."""

    configure_torch_logging(config.torch_logs)
    backends: list[ExperimentLogger] = []
    for backend_config in config.backends:
        backend = cast(
            ExperimentLogger,
            _build_from_registry(
                REGISTRIES.loggers,
                backend_config,
                extra_kwargs={
                    "output_dir": experiment.output_dir,
                    "run_name": experiment.name,
                },
            ),
        )
        backends.append(backend)

    logger: ExperimentLogger = (
        backends[0] if len(backends) == 1 else CompositeLogger(backends)
    )
    logger.log_config(resolved_config.to_dict())
    return logger


def build_diagnostics(
    configs: list[ComponentConfig],
    *,
    logger: ExperimentLogger,
    output_dir: str,
    sample_shape: tuple[int, ...] | None,
) -> list[TrainingDiagnostic]:
    """Instantiate training diagnostic plugins from configuration."""

    context = DiagnosticBuildContext(
        logger=logger,
        output_dir=output_dir,
        sample_shape=sample_shape,
    )
    diagnostics: list[TrainingDiagnostic] = []
    for diagnostic_config in configs:
        diagnostic_cls = REGISTRIES.diagnostics.resolve(diagnostic_config.name)
        context_parameters = getattr(
            diagnostic_cls,
            "context_parameters",
            None,
        )
        runtime_params: dict[str, Any] = {
            "logger": logger,
            "output_dir": output_dir,
        }
        if callable(context_parameters):
            provided = context_parameters(context)
            if not isinstance(provided, Mapping):
                raise RegistryError(
                    f"diagnostic '{diagnostic_config.name}' context_parameters "
                    "must return a mapping"
                )
            runtime_params.update(provided)
        conflicts = sorted(set(diagnostic_config.params).intersection(runtime_params))
        if conflicts:
            raise RegistryError(
                f"diagnostic '{diagnostic_config.name}' config cannot override "
                "runtime parameter(s): " + ", ".join(conflicts)
            )
        constructor_params = {
            **diagnostic_config.params,
            **runtime_params,
        }
        diagnostic = cast(
            TrainingDiagnostic,
            REGISTRIES.diagnostics.create(
                diagnostic_config.name,
                **constructor_params,
            ),
        )
        diagnostics.append(diagnostic)
    return diagnostics


def build_ema(config: EMAConfig, model: nn.Module) -> ExponentialMovingAverage | None:
    """Instantiate an EMA tracker for a model when configured."""

    if not config.enabled:
        return None
    return ExponentialMovingAverage(
        model,
        decay=config.decay,
        update_after_step=config.update_after_step,
        update_every=config.update_every,
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve special device keywords into concrete torch devices."""

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def build_training_components(
    config: StochaflowConfig,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> TrainingComponents:
    """Build model-side training components without dataset I/O side effects."""

    model = build_model(config.model)
    process = build_process(config.process) if config.process is not None else None
    objective = (
        build_objective(config.objective) if config.objective is not None else None
    )
    plan = build_training_plan(
        config.training,
        primary_model=model,
        process=process,
        objective=objective,
        model_factory=build_model,
        objective_factory=build_objective,
    )
    parameters = trainable_parameters(plan)
    optimizer = build_optimizer(config.optimizer, parameters)
    lr_scheduler = build_lr_scheduler(config.lr_scheduler, optimizer)
    ema = build_ema(config.ema, model)
    device = resolve_device(config.trainer.device)
    precision = build_precision_runtime(
        config.trainer.precision,
        device,
    )
    checkpoint_manager = CheckpointManager(
        model=model,
        process=process,
        objective=objective,
        auxiliary_modules={
            name: asset.module for name, asset in plan.auxiliary_modules.items()
        },
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        precision_kind=precision.kind,
        grad_scaler=precision.grad_scaler,
    )
    logger = build_logger(
        config.logging,
        experiment=config.experiment,
        resolved_config=config,
    )
    diagnostics = build_diagnostics(
        config.diagnostics,
        logger=logger,
        output_dir=config.experiment.output_dir,
        sample_shape=(
            tuple(config.sampling.shape)
            if config.sampling.shape is not None
            else None
        ),
    )
    trainer = Trainer(
        plan=plan,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler,
        lr_scheduler_interval=(
            config.lr_scheduler.interval
            if config.lr_scheduler is not None
            else "step"
        ),
        ema=ema,
        max_grad_norm=config.trainer.max_grad_norm,
        logger=logger,
        diagnostics=diagnostics,
        log_every=config.logging.log_every,
        checkpoint_manager=checkpoint_manager,
        checkpoint_dir=f"{config.experiment.output_dir}/checkpoints",
        checkpoint_every=config.artifacts.checkpoint_every,
        checkpoint_config=config.to_dict(),
        checkpoint_metadata=checkpoint_metadata,
        precision=precision,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    )
    return TrainingComponents(
        model=model,
        process=process,
        objective=objective,
        plan=plan,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        use_ema_for_sampling=config.ema.use_for_sampling,
        precision=precision,
        logger=logger,
        diagnostics=diagnostics,
        checkpoint_manager=checkpoint_manager,
        trainer=trainer,
    )
