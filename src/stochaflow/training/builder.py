"""Training composition contracts and plan validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from torch import nn

from stochaflow.processes import Process
from stochaflow.training.strategy import TrainingStrategy
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    validate_sampling_recipe,
)

ModelFactory = Callable[[ComponentConfig], nn.Module]
ObjectiveFactory = Callable[[ComponentConfig], nn.Module]
ModuleMode = Literal["follow", "eval"]
_RESERVED_TRAINING_ASSET_NAMES = frozenset(
    {"primary_model", "process", "objective"}
)


@dataclass(frozen=True, slots=True)
class ManagedTrainingModule:
    """A named auxiliary module and its core-managed mode policy."""

    module: nn.Module
    mode: ModuleMode = "follow"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Fully assembled assets and computation for one training run."""

    strategy: TrainingStrategy
    primary_model: nn.Module
    process: Process | None = None
    objective: nn.Module | None = None
    auxiliary_modules: Mapping[str, ManagedTrainingModule] = field(
        default_factory=dict
    )
    inference_recipe: SamplingRecipe | None = None


class TrainingBuilderContext:
    """Injected primary assets and factories available to a TrainingBuilder."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any],
        primary_model: nn.Module,
        process: Process | None,
        objective: nn.Module | None,
        model_factory: ModelFactory,
        objective_factory: ObjectiveFactory,
    ) -> None:
        self.params = deepcopy(dict(params))
        self.primary_model = primary_model
        self.process = process
        self.objective = objective
        self.model_factory = model_factory
        self.objective_factory = objective_factory


class TrainingBuilder(ABC):
    """Compose a complete TrainingPlan from injected and private assets."""

    def __init__(self, context: TrainingBuilderContext) -> None:
        self.context = context

    @abstractmethod
    def build(self) -> TrainingPlan:
        """Return one complete plan without starting the training loop."""


REGISTRIES.training_builders.require_base(TrainingBuilder)


def build_training_plan(
    declaration: ComponentConfig,
    *,
    primary_model: nn.Module,
    process: Process | None,
    objective: nn.Module | None,
    model_factory: ModelFactory,
    objective_factory: ObjectiveFactory,
    registries: RegistryCatalog = REGISTRIES,
) -> TrainingPlan:
    """Construct and validate one registered TrainingBuilder result."""

    context = TrainingBuilderContext(
        params=declaration.params,
        primary_model=primary_model,
        process=process,
        objective=objective,
        model_factory=model_factory,
        objective_factory=objective_factory,
    )
    builder = cast(
        TrainingBuilder,
        registries.training_builders.create(declaration.name, context),
    )
    plan = validate_training_plan(cast(object, builder.build()))
    if plan.primary_model is not primary_model:
        raise ValueError("TrainingPlan.primary_model must preserve the injected model")
    if plan.process is not process:
        raise ValueError("TrainingPlan.process must preserve the injected process")
    if plan.objective is not objective:
        raise ValueError("TrainingPlan.objective must preserve the injected objective")
    return plan


def validate_training_plan(value: object) -> TrainingPlan:
    """Validate and snapshot an extension-produced plan."""

    if not isinstance(value, TrainingPlan):
        raise TypeError("TrainingBuilder.build() must return TrainingPlan")
    strategy_value = cast(object, value.strategy)
    if not isinstance(strategy_value, TrainingStrategy):
        raise TypeError("TrainingPlan.strategy must be TrainingStrategy")
    if isinstance(strategy_value, nn.Module):
        raise TypeError("TrainingStrategy must not inherit nn.Module")
    primary_model_value = cast(object, value.primary_model)
    if not isinstance(primary_model_value, nn.Module):
        raise TypeError("TrainingPlan.primary_model must be an nn.Module")
    process_value = cast(object, value.process)
    if process_value is not None and not isinstance(process_value, Process):
        raise TypeError("TrainingPlan.process must be Process or None")
    objective_value = cast(object, value.objective)
    if objective_value is not None and not isinstance(objective_value, nn.Module):
        raise TypeError("TrainingPlan.objective must be an nn.Module or None")
    recipe = (
        validate_sampling_recipe(
            value.inference_recipe,
            path="TrainingPlan.inference_recipe",
        )
        if value.inference_recipe is not None
        else None
    )
    declared_auxiliaries = cast(object, value.auxiliary_modules)
    if not isinstance(declared_auxiliaries, Mapping):
        raise TypeError("TrainingPlan.auxiliary_modules must be a mapping")

    roots: list[tuple[str, nn.Module]] = [("primary_model", value.primary_model)]
    if value.process is not None:
        roots.append(("process", value.process))
    if value.objective is not None:
        roots.append(("objective", value.objective))
    for declared_name, declared_asset in declared_auxiliaries.items():
        name = cast(object, declared_name)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("training auxiliary names must be non-empty strings")
        if name in _RESERVED_TRAINING_ASSET_NAMES:
            raise ValueError(
                f"training auxiliary name '{name}' is reserved by TrainingPlan"
            )
        asset = cast(object, declared_asset)
        if not isinstance(asset, ManagedTrainingModule):
            raise TypeError(
                f"training auxiliary '{name}' must be ManagedTrainingModule"
            )
        module_value = cast(object, asset.module)
        if not isinstance(module_value, nn.Module):
            raise TypeError(f"training auxiliary '{name}' must contain an nn.Module")
        if asset.mode not in ("follow", "eval"):
            raise ValueError(
                f"training auxiliary '{name}' mode must be 'follow' or 'eval'"
            )
        roots.append((f"auxiliary_modules.{name}", asset.module))
    _validate_distinct_state_roots(roots)
    if not trainable_parameters(value):
        raise ValueError("TrainingPlan must contain at least one trainable parameter")
    return TrainingPlan(
        strategy=value.strategy,
        primary_model=value.primary_model,
        process=value.process,
        objective=value.objective,
        inference_recipe=recipe,
        auxiliary_modules=MappingProxyType(dict(value.auxiliary_modules)),
    )


def training_module_roots(plan: TrainingPlan) -> tuple[tuple[str, nn.Module], ...]:
    """Return every module whose lifecycle is managed by core."""

    roots: list[tuple[str, nn.Module]] = [("primary_model", plan.primary_model)]
    if plan.process is not None:
        roots.append(("process", plan.process))
    if plan.objective is not None:
        roots.append(("objective", plan.objective))
    roots.extend(
        (name, plan.auxiliary_modules[name].module)
        for name in sorted(plan.auxiliary_modules)
    )
    return tuple(roots)


def trainable_parameters(plan: TrainingPlan) -> tuple[nn.Parameter, ...]:
    """Return stable, deduplicated parameters selected by requires_grad."""

    result: list[nn.Parameter] = []
    seen: set[int] = set()
    for _, module in training_module_roots(plan):
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return tuple(result)


def _validate_distinct_state_roots(roots: list[tuple[str, nn.Module]]) -> None:
    seen_modules: dict[int, str] = {}
    seen_state: dict[int, str] = {}
    for name, module in roots:
        previous_module = seen_modules.setdefault(id(module), name)
        if previous_module != name:
            raise ValueError(
                f"training assets '{previous_module}' and '{name}' are the same module"
            )
        for tensor_name, tensor in [
            *module.named_parameters(recurse=True),
            *module.named_buffers(recurse=True),
        ]:
            state_name = f"{name}.{tensor_name}"
            previous_state = seen_state.setdefault(id(tensor), state_name)
            if previous_state != state_name:
                raise ValueError(
                    f"training state is shared by '{previous_state}' and '{state_name}'"
                )


__all__ = [
    "ManagedTrainingModule",
    "ModelFactory",
    "ObjectiveFactory",
    "TrainingBuilder",
    "TrainingBuilderContext",
    "TrainingPlan",
    "build_training_plan",
    "trainable_parameters",
    "training_module_roots",
    "validate_training_plan",
]
