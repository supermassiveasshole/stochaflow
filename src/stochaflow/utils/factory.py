"""Component registries and builder utilities."""

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW, Optimizer

import stochaflow.data  # noqa: F401
import stochaflow.diffusion  # noqa: F401
import stochaflow.models  # noqa: F401
from stochaflow.diffusion import DiffusionScheduler
from stochaflow.training import Trainer
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ExperimentConfig,
    ComponentConfig,
    DataConfig,
    DataloaderConfig,
    LoggingConfig,
    OptimizerConfig,
    StochaflowConfig,
)
from stochaflow.utils.logging import CompositeLogger, ExperimentLogger, configure_torch_logging
from stochaflow.utils.registry import (
    DATASET_REGISTRY,
    DIFFUSION_REGISTRY,
    LOGGER_REGISTRY,
    MODEL_REGISTRY,
    OBJECTIVE_REGISTRY,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    RegistryError,
    register_optimizer,
)

register_optimizer("adam", Adam)
register_optimizer("adamw", AdamW)


@dataclass(slots=True)
class TrainingComponents:
    """Fully built training components for an experiment."""

    model: nn.Module
    scheduler: DiffusionScheduler
    diffusion: nn.Module
    objective: nn.Module
    optimizer: Optimizer
    logger: ExperimentLogger
    checkpoint_manager: CheckpointManager
    trainer: Trainer


@dataclass(slots=True)
class DataComponents:
    """Dataset and dataloader built for an experiment split."""

    dataset: Dataset[Any]
    dataloader: DataLoader[Any]


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


def build_data_components(
    config: DataConfig,
    *,
    seed: int | None = None,
) -> DataComponents:
    """Build dataset and dataloader from configuration."""

    dataset = build_dataset(config.dataset)
    dataloader = build_dataloader(dataset, config.dataloader, seed=seed)
    return DataComponents(dataset=dataset, dataloader=dataloader)


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


def resolve_train_step_fn(diffusion_name: str, objective_name: str):
    """Resolve an algorithm-specific train step function when needed."""

    if diffusion_name == "ddpm" and objective_name == "ddpm_epsilon":
        return ddpm_epsilon_train_step
    return None


def resolve_device(device_name: str) -> torch.device:
    """Resolve special device keywords into concrete torch devices."""

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def build_training_components(config: StochaflowConfig) -> TrainingComponents:
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
    checkpoint_manager = CheckpointManager(model=diffusion, optimizer=optimizer)
    device = resolve_device(config.trainer.device)
    logger = build_logger(
        config.logging,
        experiment=config.experiment,
        resolved_config=config,
    )
    train_step_fn = resolve_train_step_fn(config.diffusion.name, config.objective.name)
    trainer = Trainer(
        model=diffusion,
        optimizer=optimizer,
        criterion=objective,
        device=device,
        train_step_fn=train_step_fn,
        max_grad_norm=config.trainer.max_grad_norm,
        logger=logger,
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
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        trainer=trainer,
    )
