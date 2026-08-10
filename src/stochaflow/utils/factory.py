"""Compatibility forwards for component and training runtime construction."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow import _component_factory
from stochaflow._builtin_activation import (
    activate_all_builtins,
    activate_training_builtins,
)
from stochaflow.processes.base import Process
from stochaflow.training.composition import TrainingComponents
from stochaflow.training.diagnostics.contracts import TrainingDiagnostic
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.config import (
    ComponentConfig,
    EMAConfig,
    ExperimentConfig,
    LoggingConfig,
    LRSchedulerConfig,
    StochaflowConfig,
)
from stochaflow.utils.device import resolve_device as _resolve_device
from stochaflow.utils.logging_contracts import ExperimentLogger


def load_builtin_components() -> None:
    """Activate every built-in through the process lifecycle owner."""

    activate_all_builtins()


def build_model(component: ComponentConfig) -> nn.Module:
    """Forward model construction to the narrow component factory."""

    activate_all_builtins()
    return _component_factory.build_model(component)


def build_process(component: ComponentConfig) -> Process:
    """Forward Process construction to the narrow component factory."""

    activate_all_builtins()
    return _component_factory.build_process(component)


def build_objective(component: ComponentConfig) -> nn.Module:
    """Forward objective construction to the narrow component factory."""

    activate_all_builtins()
    return _component_factory.build_objective(component)


def build_logger(
    config: LoggingConfig,
    *,
    experiment: ExperimentConfig,
    resolved_config: StochaflowConfig,
) -> ExperimentLogger:
    """Forward logger construction to training composition."""

    activate_training_builtins()
    composition = cast(
        Any,
        import_module("stochaflow.training.composition"),
    )
    return composition.build_logger(
        config,
        experiment=experiment,
        resolved_config=resolved_config,
    )


def build_diagnostics(
    configs: list[ComponentConfig],
    *,
    logger: ExperimentLogger,
    output_dir: str,
) -> list[TrainingDiagnostic]:
    """Forward Diagnostic construction to training composition."""

    activate_training_builtins()
    composition = cast(
        Any,
        import_module("stochaflow.training.composition"),
    )
    return composition.build_diagnostics(
        configs,
        logger=logger,
        output_dir=output_dir,
    )


def build_ema(
    config: EMAConfig,
    model: nn.Module,
) -> ExponentialMovingAverage | None:
    """Forward EMA construction to training composition."""

    activate_training_builtins()
    composition = cast(
        Any,
        import_module("stochaflow.training.composition"),
    )
    return composition.build_ema(config, model)


def build_optimizer(config: ComponentConfig, parameters: Any) -> Optimizer:
    """Forward optimizer construction to training composition support."""

    activate_training_builtins()
    optimization = cast(
        Any,
        import_module("stochaflow.training.optimization"),
    )
    return optimization.build_optimizer(config, parameters)


def build_lr_scheduler(
    config: LRSchedulerConfig | None,
    optimizer: Optimizer,
) -> LRScheduler | None:
    """Forward scheduler construction to training composition support."""

    activate_training_builtins()
    optimization = cast(
        Any,
        import_module("stochaflow.training.optimization"),
    )
    return optimization.build_lr_scheduler(config, optimizer)


def resolve_device(device_name: str) -> torch.device:
    """Forward device resolution to the device utility."""

    return _resolve_device(device_name)


def build_training_components(
    config: StochaflowConfig,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> TrainingComponents:
    """Preserve the legacy training composition entry as a thin forward."""

    activate_training_builtins()
    composition = cast(
        Any,
        import_module("stochaflow.training.composition"),
    )
    return composition.build_training_components(
        config,
        checkpoint_metadata=checkpoint_metadata,
    )


__all__ = [
    "TrainingComponents",
    "build_diagnostics",
    "build_ema",
    "build_logger",
    "build_lr_scheduler",
    "build_model",
    "build_objective",
    "build_optimizer",
    "build_process",
    "build_training_components",
    "load_builtin_components",
    "resolve_device",
]
