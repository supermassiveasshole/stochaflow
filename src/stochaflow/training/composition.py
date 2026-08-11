"""Complete runtime composition for the training operation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

import torch
from torch import nn
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
    build_training_plan,
    trainable_parameters,
    training_module_roots,
)
from stochaflow.training.diagnostics.contracts import (
    DiagnosticBuildContext,
    DiagnosticModelAccess,
    TrainingDiagnostic,
)
from stochaflow.training.diagnostics.runtime import TrainingDiagnosticModelAccess
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.metric_binding import TrainingMetricRuntime
from stochaflow.training.optimization import build_lr_scheduler, build_optimizer
from stochaflow.training.precision import PrecisionRuntime, build_precision_runtime
from stochaflow.training.trainer import Trainer
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    inference_asset_descriptors_from_projections,
)
from stochaflow.utils.config import (
    ComponentConfig,
    EMAConfig,
    ExperimentConfig,
    LoggingConfig,
    StochaflowConfig,
)
from stochaflow.utils.device import resolve_device
from stochaflow.utils.logging_contracts import CompositeLogger, ExperimentLogger
from stochaflow.utils.registry import REGISTRIES, RegistryError
from stochaflow.utils.torch_logging import configure_torch_logging


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
    precision: PrecisionRuntime
    logger: ExperimentLogger
    diagnostics: list[TrainingDiagnostic]
    metric_runtime: TrainingMetricRuntime
    checkpoint_manager: CheckpointManager
    trainer: Trainer


# Preserve the established import and pickle location. Annotations are made
# concrete first so reflection is independent of import order.
TrainingComponents.__annotations__ = get_type_hints(TrainingComponents)
TrainingComponents.__module__ = "stochaflow.utils.factory"


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
        constructor_params = {
            **backend_config.params,
            "output_dir": experiment.output_dir,
            "run_name": experiment.name,
        }
        backend = cast(
            ExperimentLogger,
            REGISTRIES.loggers.create(
                backend_config.name,
                **constructor_params,
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
) -> list[TrainingDiagnostic]:
    """Instantiate diagnostics without a complete Training composition.

    Diagnostics that request model, Strategy, or Process capabilities fail at
    construction. The complete training operation injects those capabilities
    through its private composition path.
    """

    return _build_diagnostics(
        configs,
        logger=logger,
        output_dir=output_dir,
        model_access=UnavailableDiagnosticModelAccess(),
        strategy=UnavailableDiagnosticCapability(),
        process=None,
    )


class UnavailableDiagnosticModelAccess:
    """Reject model use outside complete Training runtime composition."""

    @property
    def device(self) -> torch.device:
        raise RuntimeError(
            "DiagnosticModelAccess requires complete Training composition"
        )

    @property
    def ema_available(self) -> bool:
        return False

    def evaluation(
        self,
        *,
        seed: int,
        prefer_ema: bool,
    ) -> AbstractContextManager[None]:
        del seed, prefer_ema
        raise RuntimeError(
            "DiagnosticModelAccess requires complete Training composition"
        )


class UnavailableDiagnosticCapability:
    """Represent unavailable Strategy capabilities in legacy construction."""


def _build_diagnostics(
    configs: list[ComponentConfig],
    *,
    logger: ExperimentLogger,
    output_dir: str,
    model_access: DiagnosticModelAccess,
    strategy: object,
    process: object | None,
) -> list[TrainingDiagnostic]:
    """Instantiate diagnostics with explicit runtime-owned capabilities."""

    diagnostics: list[TrainingDiagnostic] = []
    for diagnostic_config in configs:
        context = DiagnosticBuildContext(
            component_name=diagnostic_config.name,
            logger=logger,
            output_dir=output_dir,
            model_access=model_access,
            _strategy=strategy,
            _process=process,
        )
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
            invalid_keys = sorted(
                repr(key)
                for key in provided
                if not isinstance(key, str) or not key
            )
            if invalid_keys:
                raise RegistryError(
                    f"diagnostic '{diagnostic_config.name}' context_parameters "
                    "returned invalid key(s): " + ", ".join(invalid_keys)
                )
            protected = sorted(
                set(provided).intersection({"logger", "output_dir", "model_access"})
            )
            if protected:
                raise RegistryError(
                    f"diagnostic '{diagnostic_config.name}' context_parameters "
                    "cannot replace runtime parameter(s): "
                    + ", ".join(protected)
                )
            if (
                "build_context" in provided
                and provided["build_context"] is not context
            ):
                raise RegistryError(
                    f"diagnostic '{diagnostic_config.name}' context_parameters "
                    "must return its supplied DiagnosticBuildContext"
                )
            runtime_params.update(provided)
        conflicts = sorted(set(diagnostic_config.params).intersection(runtime_params))
        reserved_conflicts = sorted(
            set(diagnostic_config.params).intersection(
                {
                    "build_context",
                    "component_name",
                    "logger",
                    "model_access",
                    "output_dir",
                    "process",
                    "strategy",
                }
            )
        )
        conflicts = sorted(set(conflicts).union(reserved_conflicts))
        if conflicts:
            raise RegistryError(
                f"diagnostic '{diagnostic_config.name}' config cannot override "
                "runtime parameter(s): " + ", ".join(conflicts)
            )
        diagnostic = cast(
            TrainingDiagnostic,
            REGISTRIES.diagnostics.create(
                diagnostic_config.name,
                **diagnostic_config.params,
                **runtime_params,
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


def build_training_components(
    config: StochaflowConfig,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> TrainingComponents:
    """Build the complete training runtime without dataset I/O side effects."""

    require_training_builtins()
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
    inference_asset_descriptors = inference_asset_descriptors_from_projections(
        plan.inference_assets
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
        inference_asset_descriptors=inference_asset_descriptors,
        inference_recipe=plan.inference_recipe,
    )
    logger = build_logger(
        config.logging,
        experiment=config.experiment,
        resolved_config=config,
    )
    managed_diagnostic_modules = training_module_roots(plan)
    diagnostic_model_access = TrainingDiagnosticModelAccess(
        device=device,
        model=plan.primary_model,
        ema=ema,
        managed_modules=managed_diagnostic_modules[1:],
    )
    diagnostics = _build_diagnostics(
        config.diagnostics,
        logger=logger,
        output_dir=config.experiment.output_dir,
        model_access=diagnostic_model_access,
        strategy=plan.strategy,
        process=process,
    )
    metric_runtime = TrainingMetricRuntime(
        config.metrics,
        plan.strategy,
        device=device,
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
        metric_runtime=metric_runtime,
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
        precision=precision,
        logger=logger,
        diagnostics=diagnostics,
        metric_runtime=metric_runtime,
        checkpoint_manager=checkpoint_manager,
        trainer=trainer,
    )


__all__ = ["TrainingComponents", "build_training_components"]
