"""Component registries and builder utilities."""

from dataclasses import dataclass
from importlib import import_module
import math
import random
from typing import Any

import numpy as np
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
from torch.utils.data import DataLoader, Dataset

from stochaflow.diffusion import DiffusionScheduler
from stochaflow.training import Trainer
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ExperimentConfig,
    ComponentConfig,
    DataloaderConfig,
    EMAConfig,
    LRSchedulerConfig,
    LoggingConfig,
    OptimizerConfig,
    StochaflowConfig,
)
from stochaflow.utils.logging import CompositeLogger, ExperimentLogger, configure_torch_logging
from stochaflow.utils.registry import (
    DATASET_REGISTRY,
    DIFFUSION_REGISTRY,
    LOGGER_REGISTRY,
    DIAGNOSTIC_REGISTRY,
    LR_SCHEDULER_REGISTRY,
    MODEL_REGISTRY,
    OBJECTIVE_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    RegistryError,
    register_lr_scheduler,
    register_optimizer,
)


BUILTIN_COMPONENT_MODULES = (
    "stochaflow.data",
    "stochaflow.diffusion",
    "stochaflow.models",
    "stochaflow.training.diagnostics",
)


def load_builtin_components() -> None:
    """Import built-in component modules so their registry decorators run."""

    for module_name in BUILTIN_COMPONENT_MODULES:
        import_module(module_name)


load_builtin_components()
register_optimizer("adam", Adam)
register_optimizer("adamw", AdamW)
register_lr_scheduler("cosine", CosineAnnealingLR)
register_lr_scheduler("step", StepLR)
register_lr_scheduler("multistep", MultiStepLR)
register_lr_scheduler("exponential", ExponentialLR)
register_lr_scheduler("linear", LinearLR)


@dataclass(slots=True)
class TrainingComponents:
    """Fully built training components for an experiment."""

    model: nn.Module
    scheduler: DiffusionScheduler
    diffusion: nn.Module
    objective: nn.Module
    optimizer: Optimizer
    lr_scheduler: Any | None
    ema: ExponentialMovingAverage | None
    use_ema_for_sampling: bool
    logger: ExperimentLogger
    diagnostics: list[Any]
    checkpoint_manager: CheckpointManager
    trainer: Trainer


def _build_from_registry(
    registry: dict[str, Any],
    component: ComponentConfig,
    *,
    kind: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    extra_kwargs = extra_kwargs or {}
    if component.name not in registry:
        available = ", ".join(sorted(registry)) or "<empty>"
        raise RegistryError(f"unknown {kind} '{component.name}'. Available: {available}")

    cls = registry[component.name]
    kwargs = {**component.params, **extra_kwargs}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise RegistryError(
            f"failed to initialize {kind} '{component.name}' with params {kwargs}: {exc}"
        ) from exc


def build_model(component: ComponentConfig) -> nn.Module:
    """Instantiate a model from the model registry."""

    model = _build_from_registry(MODEL_REGISTRY, component, kind="model")
    if not isinstance(model, nn.Module):
        raise RegistryError(f"registered model '{component.name}' did not produce nn.Module")
    return model


def build_dataset(component: ComponentConfig) -> Dataset[Any]:
    """Instantiate a dataset from the dataset registry."""

    dataset = _build_from_registry(DATASET_REGISTRY, component, kind="dataset")
    if not isinstance(dataset, Dataset):
        raise RegistryError(
            f"registered dataset '{component.name}' did not produce torch Dataset"
        )
    return dataset


def build_scheduler(component: ComponentConfig) -> DiffusionScheduler:
    """Instantiate a scheduler from the scheduler registry."""

    scheduler = _build_from_registry(SCHEDULER_REGISTRY, component, kind="scheduler")
    if not isinstance(scheduler, DiffusionScheduler):
        raise RegistryError(
            f"registered scheduler '{component.name}' did not produce DiffusionScheduler"
        )
    return scheduler


def build_diffusion(
    diffusion_name: str,
    *,
    model: nn.Module,
    scheduler: DiffusionScheduler,
    params: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate a diffusion/process object."""

    params = params or {}
    component = ComponentConfig(name=diffusion_name, params=params)
    diffusion = _build_from_registry(
        DIFFUSION_REGISTRY,
        component,
        kind="diffusion",
        extra_kwargs={"model": model, "scheduler": scheduler},
    )
    if not isinstance(diffusion, nn.Module):
        raise RegistryError(
            f"registered diffusion '{diffusion_name}' did not produce nn.Module"
        )
    return diffusion


def build_objective(component: ComponentConfig) -> nn.Module:
    """Instantiate a training objective from the objective registry."""

    objective = _build_from_registry(OBJECTIVE_REGISTRY, component, kind="objective")
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
            LOGGER_REGISTRY,
            backend_config,
            kind="logger",
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
) -> list[Any]:
    """Instantiate training diagnostic plugins from configuration."""

    diagnostics: list[Any] = []
    for diagnostic_config in configs:
        if diagnostic_config.name not in DIAGNOSTIC_REGISTRY:
            available = ", ".join(sorted(DIAGNOSTIC_REGISTRY)) or "<empty>"
            raise RegistryError(
                f"unknown diagnostic '{diagnostic_config.name}'. Available: {available}"
            )
        diagnostic_cls = DIAGNOSTIC_REGISTRY[diagnostic_config.name]
        try:
            diagnostic = diagnostic_cls(
                logger=logger,
                output_dir=output_dir,
                **diagnostic_config.params,
            )
        except TypeError as exc:
            raise RegistryError(
                f"failed to initialize diagnostic '{diagnostic_config.name}' "
                f"with params {diagnostic_config.params}: {exc}"
            ) from exc
        diagnostics.append(diagnostic)
    return diagnostics


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch RNG state inside a dataloader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset[Any],
    config: DataloaderConfig,
    *,
    seed: int | None = None,
) -> DataLoader[Any]:
    """Instantiate a dataloader with reproducible worker seeding."""

    dataloader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": config.shuffle,
        "num_workers": config.num_workers,
        "drop_last": config.drop_last,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = config.prefetch_factor
        dataloader_kwargs["worker_init_fn"] = _seed_dataloader_worker
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader_kwargs["generator"] = generator
    return DataLoader(dataset, **dataloader_kwargs)


def build_optimizer(config: OptimizerConfig, parameters: Any) -> Optimizer:
    """Instantiate an optimizer from configuration."""

    if config.name not in OPTIMIZER_REGISTRY:
        available = ", ".join(sorted(OPTIMIZER_REGISTRY))
        raise RegistryError(
            f"unknown optimizer '{config.name}'. Available: {available}"
        )
    optimizer_cls = OPTIMIZER_REGISTRY[config.name]
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


register_lr_scheduler("warmup_cosine", _build_warmup_cosine_lr_scheduler)


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
    if config.name not in LR_SCHEDULER_REGISTRY:
        available = ", ".join(sorted(LR_SCHEDULER_REGISTRY)) or "<empty>"
        raise RegistryError(
            f"unknown lr scheduler '{config.name}'. Available: {available}"
        )

    params = dict(config.params)
    if config.name == "warmup_cosine":
        params["total_steps"] = _resolve_warmup_cosine_total_steps(
            params,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
        )

    scheduler_builder = LR_SCHEDULER_REGISTRY[config.name]
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
    scheduler = build_scheduler(config.diffusion.scheduler)
    diffusion = build_diffusion(
        config.diffusion.name,
        model=model,
        scheduler=scheduler,
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
        scheduler=scheduler,
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
