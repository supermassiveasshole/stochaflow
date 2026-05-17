"""Component registries and registration decorators."""

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when a registry operation fails."""


MODEL_REGISTRY: dict[str, type[Any]] = {}
DATASET_REGISTRY: dict[str, Callable[..., Any]] = {}
SCHEDULER_REGISTRY: dict[str, type[Any]] = {}
DIFFUSION_REGISTRY: dict[str, type[Any]] = {}
OBJECTIVE_REGISTRY: dict[str, type[Any]] = {}
LOGGER_REGISTRY: dict[str, type[Any]] = {}
OPTIMIZER_REGISTRY: dict[str, type[Any]] = {}
LR_SCHEDULER_REGISTRY: dict[str, Callable[..., Any]] = {}
DIAGNOSTIC_REGISTRY: dict[str, type[Any]] = {}


def _register(
    registry: dict[str, type[Any]],
    *,
    kind: str,
    name: str,
) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in registry:
            existing = registry[name]
            raise RegistryError(
                f"{kind} '{name}' already registered by {existing.__module__}.{existing.__name__}"
            )
        registry[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Register a model class under a stable config-facing name."""

    return _register(MODEL_REGISTRY, kind="model", name=name)


def register_dataset(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Register a dataset builder under a stable config-facing name."""

    def decorator(builder: Callable[..., T]) -> Callable[..., T]:
        if name in DATASET_REGISTRY:
            existing = DATASET_REGISTRY[name]
            raise RegistryError(
                f"dataset '{name}' already registered by "
                f"{existing.__module__}.{existing.__name__}"
            )
        DATASET_REGISTRY[name] = builder
        return builder

    return decorator


def register_scheduler(name: str) -> Callable[[type[T]], type[T]]:
    """Register a scheduler class under a stable config-facing name."""

    return _register(SCHEDULER_REGISTRY, kind="scheduler", name=name)


def register_diffusion(name: str) -> Callable[[type[T]], type[T]]:
    """Register a diffusion/process class under a stable config-facing name."""

    return _register(DIFFUSION_REGISTRY, kind="diffusion", name=name)


def register_objective(name: str) -> Callable[[type[T]], type[T]]:
    """Register an objective class under a stable config-facing name."""

    return _register(OBJECTIVE_REGISTRY, kind="objective", name=name)


def register_logger(name: str) -> Callable[[type[T]], type[T]]:
    """Register an experiment logger class under a stable config-facing name."""

    return _register(LOGGER_REGISTRY, kind="logger", name=name)


def register_diagnostic(name: str) -> Callable[[type[T]], type[T]]:
    """Register a training diagnostic plugin under a stable config name."""

    return _register(DIAGNOSTIC_REGISTRY, kind="diagnostic", name=name)


def register_optimizer(name: str, optimizer_cls: type[Any]) -> None:
    """Register an optimizer class under a stable config-facing name."""

    if name in OPTIMIZER_REGISTRY:
        existing = OPTIMIZER_REGISTRY[name]
        if existing is optimizer_cls:
            return
        raise RegistryError(
            f"optimizer '{name}' already registered by {existing.__module__}.{existing.__name__}"
        )
    OPTIMIZER_REGISTRY[name] = optimizer_cls


def register_lr_scheduler(name: str, scheduler_builder: Callable[..., Any]) -> None:
    """Register an optimizer LR scheduler builder under a stable config name."""

    if name in LR_SCHEDULER_REGISTRY:
        existing = LR_SCHEDULER_REGISTRY[name]
        if existing is scheduler_builder:
            return
        raise RegistryError(
            f"lr scheduler '{name}' already registered by "
            f"{existing.__module__}.{existing.__name__}"
        )
    LR_SCHEDULER_REGISTRY[name] = scheduler_builder
