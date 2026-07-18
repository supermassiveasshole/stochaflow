"""Component registries and builder utilities."""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiStepLR,
    StepLR,
)

from stochaflow.diffusion import NoiseSchedule
from stochaflow.training import DiagnosticBuildContext, Trainer, TrainingDiagnostic
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ComponentConfig,
    EMAConfig,
    ExperimentConfig,
    LRSchedulerConfig,
    LoggingConfig,
    OptimizerConfig,
    StochaflowConfig,
)
from stochaflow.utils.logging import CompositeLogger, ExperimentLogger, configure_torch_logging
from stochaflow.utils.registry import REGISTRIES, Registry, RegistryError


BUILTIN_COMPONENT_MODULES = (
    "stochaflow.data",
    "stochaflow.diffusion",
    "stochaflow.models",
    "stochaflow.sampling",
    "stochaflow.training.diagnostics",
)


def load_builtin_components() -> None:
    """Import built-in component modules so their registry decorators run."""

    REGISTRIES.load_modules(BUILTIN_COMPONENT_MODULES)


load_builtin_components()
REGISTRIES.models.require_base(nn.Module)
REGISTRIES.noise_schedules.require_base(NoiseSchedule)
REGISTRIES.diffusions.require_base(nn.Module)
REGISTRIES.objectives.require_base(nn.Module)
REGISTRIES.optimizers.require_base(Optimizer)
REGISTRIES.loggers.require_base(ExperimentLogger)
REGISTRIES.diagnostics.require_base(TrainingDiagnostic)
REGISTRIES.optimizers.add("adam", Adam)
REGISTRIES.optimizers.add("adamw", AdamW)
REGISTRIES.lr_schedulers.add("cosine", CosineAnnealingLR)
REGISTRIES.lr_schedulers.add("step", StepLR)
REGISTRIES.lr_schedulers.add("multistep", MultiStepLR)
REGISTRIES.lr_schedulers.add("exponential", ExponentialLR)
REGISTRIES.lr_schedulers.add("linear", LinearLR)


@dataclass(slots=True)
class TrainingComponents:
    """Fully built training components for an experiment."""

    model: nn.Module
    noise_schedule: NoiseSchedule
    diffusion: nn.Module
    objective: nn.Module
    optimizer: Optimizer
    lr_scheduler: Any | None
    ema: ExponentialMovingAverage | None
    use_ema_for_sampling: bool
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

    model = _build_from_registry(REGISTRIES.models, component)
    if not isinstance(model, nn.Module):
        raise RegistryError(f"registered model '{component.name}' did not produce nn.Module")
    return model


def build_noise_schedule(component: ComponentConfig) -> NoiseSchedule:
    """Instantiate a forward noise path from the noise-schedule registry."""

    schedule = _build_from_registry(
        REGISTRIES.noise_schedules,
        component,
    )
    if not isinstance(schedule, NoiseSchedule):
        raise RegistryError(
            f"registered noise schedule '{component.name}' did not produce "
            "NoiseSchedule"
        )
    return schedule


def build_diffusion(
    diffusion_name: str,
    *,
    model: nn.Module,
    noise_schedule: NoiseSchedule,
    params: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a diffusion/process object."""

    params = params or {}
    component = ComponentConfig(name=diffusion_name, params=params)
    diffusion = _build_from_registry(
        REGISTRIES.diffusions,
        component,
        extra_kwargs={"model": model, "noise_schedule": noise_schedule},
    )
    if not isinstance(diffusion, nn.Module):
        raise RegistryError(
            f"registered diffusion '{diffusion_name}' did not produce nn.Module"
        )
    return diffusion


def build_objective(component: ComponentConfig) -> nn.Module:
    """Instantiate a training objective from the objective registry."""

    objective = _build_from_registry(
        REGISTRIES.objectives,
        component,
    )
    if not isinstance(objective, nn.Module):
        raise RegistryError(
            f"registered objective '{component.name}' did not produce nn.Module"
        )
    return objective


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
        backend = _build_from_registry(
            REGISTRIES.loggers,
            backend_config,
            extra_kwargs={
                "output_dir": experiment.output_dir,
                "run_name": experiment.name,
            },
        )
        if not isinstance(backend, ExperimentLogger):
            raise RegistryError(
                f"registered logger '{backend_config.name}' did not produce ExperimentLogger"
            )
        backends.append(backend)

    logger: ExperimentLogger
    if len(backends) == 1:
        logger = backends[0]
    else:
        logger = CompositeLogger(backends)
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
        diagnostic = REGISTRIES.diagnostics.create(
            diagnostic_config.name,
            **constructor_params,
        )
        if not isinstance(diagnostic, TrainingDiagnostic):
            raise RegistryError(
                f"registered diagnostic '{diagnostic_config.name}' did not produce "
                "TrainingDiagnostic"
            )
        diagnostics.append(diagnostic)
    return diagnostics


def build_optimizer(config: OptimizerConfig, parameters: Any) -> Optimizer:
    """Instantiate an optimizer from configuration."""

    optimizer_cls = REGISTRIES.optimizers.resolve(config.name)
    try:
        return optimizer_cls(parameters, **config.params)
    except TypeError as exc:
        raise RegistryError(
            f"failed to initialize optimizer '{config.name}' with params {config.params}: {exc}"
        ) from exc


def _resolve_warmup_cosine_total_steps(
    params: dict[str, Any],
    *,
    steps_per_epoch: int | None,
    num_epochs: int | None,
) -> int:
    total_steps = params.get("total_steps")
    if total_steps == "auto":
        if steps_per_epoch is None or num_epochs is None:
            raise RegistryError(
                "lr_scheduler warmup_cosine with total_steps='auto' requires "
                "steps_per_epoch and num_epochs"
            )
        if steps_per_epoch <= 0 or num_epochs <= 0:
            raise RegistryError("steps_per_epoch and num_epochs must be positive")
        return steps_per_epoch * num_epochs
    if not isinstance(total_steps, int) or total_steps <= 0:
        raise RegistryError(
            "lr_scheduler warmup_cosine requires a positive integer total_steps "
            "or 'auto'"
        )
    return total_steps


def _resolve_cosine_t_max(
    params: dict[str, Any],
    *,
    interval: str,
    steps_per_epoch: int | None,
    num_epochs: int | None,
) -> int:
    t_max = params.get("T_max")
    if t_max == "auto":
        if num_epochs is None or num_epochs <= 0:
            raise RegistryError(
                "lr_scheduler cosine with T_max='auto' requires positive num_epochs"
            )
        if interval == "epoch":
            return num_epochs
        if interval == "step":
            if steps_per_epoch is None or steps_per_epoch <= 0:
                raise RegistryError(
                    "lr_scheduler cosine with T_max='auto' and interval='step' "
                    "requires positive steps_per_epoch"
                )
            return steps_per_epoch * num_epochs
        raise RegistryError("lr_scheduler interval must be 'step' or 'epoch'")
    if isinstance(t_max, bool) or not isinstance(t_max, int) or t_max <= 0:
        raise RegistryError(
            "lr_scheduler cosine requires a positive integer T_max or 'auto'"
        )
    return t_max


def _build_warmup_cosine_lr_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    if warmup_steps <= 0:
        raise RegistryError("warmup_cosine warmup_steps must be positive")
    if total_steps <= warmup_steps:
        raise RegistryError("warmup_cosine total_steps must be greater than warmup_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise RegistryError("warmup_cosine min_lr_ratio must be between 0 and 1")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(
            1.0,
            float(step + 1 - warmup_steps) / float(total_steps - warmup_steps),
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


REGISTRIES.lr_schedulers.add("warmup_cosine", _build_warmup_cosine_lr_scheduler)


def build_lr_scheduler(
    config: LRSchedulerConfig,
    optimizer: Optimizer,
    *,
    steps_per_epoch: int | None = None,
    num_epochs: int | None = None,
) -> Any | None:
    """Instantiate an optimizer LR scheduler from configuration."""

    if config.name is None:
        return None
    params = dict(config.params)
    if config.name == "cosine":
        params["T_max"] = _resolve_cosine_t_max(
            params,
            interval=config.interval,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
        )
    elif config.name == "warmup_cosine":
        params["total_steps"] = _resolve_warmup_cosine_total_steps(
            params,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
        )

    scheduler_builder = REGISTRIES.lr_schedulers.resolve(config.name)
    try:
        return scheduler_builder(optimizer, **params)
    except TypeError as exc:
        raise RegistryError(
            f"failed to initialize lr scheduler '{config.name}' with params "
            f"{params}: {exc}"
        ) from exc


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


def resolve_train_step_fn(diffusion_name: str, objective_name: str):
    """Resolve an algorithm-specific train step function when needed."""

    if diffusion_name in {"ddpm", "ddim"} and objective_name == "ddpm_epsilon":
        return ddpm_epsilon_train_step
    return None


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
    steps_per_epoch: int | None = None,
    num_epochs: int | None = None,
) -> TrainingComponents:
    """Build model-side training components without dataset I/O side effects."""

    model = build_model(config.model)
    noise_schedule = build_noise_schedule(config.diffusion.noise_schedule)
    diffusion = build_diffusion(
        config.diffusion.name,
        model=model,
        noise_schedule=noise_schedule,
        params=config.diffusion.params,
    )
    objective = build_objective(config.objective)
    optimizer = build_optimizer(config.optimizer, diffusion.parameters())
    lr_scheduler = build_lr_scheduler(
        config.lr_scheduler,
        optimizer,
        steps_per_epoch=steps_per_epoch,
        num_epochs=num_epochs or config.trainer.num_epochs,
    )
    ema = build_ema(config.ema, diffusion)
    checkpoint_manager = CheckpointManager(
        model=diffusion,
        denoiser=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
    )
    device = resolve_device(config.trainer.device)
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
    train_step_fn = resolve_train_step_fn(config.diffusion.name, config.objective.name)
    trainer = Trainer(
        model=diffusion,
        optimizer=optimizer,
        criterion=objective,
        device=device,
        train_step_fn=train_step_fn,
        lr_scheduler=lr_scheduler,
        lr_scheduler_interval=(
            config.lr_scheduler.interval if lr_scheduler is not None else "step"
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
    )
    return TrainingComponents(
        model=model,
        noise_schedule=noise_schedule,
        diffusion=diffusion,
        objective=objective,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        use_ema_for_sampling=config.ema.use_for_sampling,
        logger=logger,
        diagnostics=diagnostics,
        checkpoint_manager=checkpoint_manager,
        trainer=trainer,
    )
