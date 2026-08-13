"""Training composition contracts and plan validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, runtime_checkable

import torch
from torch import nn

from stochaflow._builtin_activation import activate_training_component_builtins
from stochaflow.processes.base import Process
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
class InferenceAssetProjection:
    """Reconstructable inference projection of one managed training asset."""

    training_asset_name: str
    declaration: ComponentConfig
    capability_role: str


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
    inference_assets: Mapping[str, InferenceAssetProjection] = field(
        default_factory=dict
    )


@runtime_checkable
class TrainingExecutionBindingBuilder(Protocol):
    """Optionally bind a canonical training plan to an execution wrapper.

    The canonical primary model remains the authority for identity, state,
    EMA, and checkpointing. Parallel training runtimes may wrap a state-sharing
    execution module, then ask the owning Builder for a Strategy that invokes
    that wrapper. This keeps task-specific model signatures out of core.
    """

    def build_primary_execution_module(self, plan: TrainingPlan) -> nn.Module:
        """Return the state-sharing module that a runtime may wrap."""

        ...

    def bind_primary_execution_model(
        self,
        plan: TrainingPlan,
        execution_model: nn.Module,
    ) -> TrainingStrategy:
        """Return a Strategy whose trainable calls use ``execution_model``."""

        ...


@dataclass(frozen=True, slots=True)
class TrainingPlanAssembly:
    """One canonical plan plus its optional runtime execution binding."""

    plan: TrainingPlan
    builder_name: str
    _execution_binding: TrainingExecutionBindingBuilder | None = field(
        repr=False,
        compare=False,
    )

    @property
    def supports_primary_execution_binding(self) -> bool:
        """Return whether the owning Builder can bind a wrapped primary model."""

        return self._execution_binding is not None

    def build_primary_execution_module(self) -> nn.Module:
        """Build and validate the module that a parallel runtime may wrap."""

        binding = self._require_execution_binding()
        module = cast(object, binding.build_primary_execution_module(self.plan))
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"training builder '{self.builder_name}' execution module "
                "must be an nn.Module"
            )
        _validate_primary_execution_state(self.plan.primary_model, module)
        return module

    def bind_primary_execution_model(
        self,
        execution_model: nn.Module,
    ) -> TrainingStrategy:
        """Bind one state-sharing wrapper without changing the canonical plan."""

        execution_model_value = cast(object, execution_model)
        if not isinstance(execution_model_value, nn.Module):
            raise TypeError("primary execution model must be an nn.Module")
        _validate_primary_execution_state(
            self.plan.primary_model,
            execution_model_value,
        )
        binding = self._require_execution_binding()
        strategy = cast(
            object,
            binding.bind_primary_execution_model(
                self.plan,
                execution_model_value,
            ),
        )
        if not isinstance(strategy, TrainingStrategy):
            raise TypeError(
                f"training builder '{self.builder_name}' execution binding "
                "must return TrainingStrategy"
            )
        if isinstance(strategy, nn.Module):
            raise TypeError("bound TrainingStrategy must not inherit nn.Module")
        return strategy

    def _require_execution_binding(self) -> TrainingExecutionBindingBuilder:
        binding = self._execution_binding
        if binding is None:
            raise TypeError(
                f"training builder '{self.builder_name}' does not support "
                "primary execution binding"
            )
        return binding


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

    return build_training_plan_assembly(
        declaration,
        primary_model=primary_model,
        process=process,
        objective=objective,
        model_factory=model_factory,
        objective_factory=objective_factory,
        registries=registries,
    ).plan


def build_training_plan_assembly(
    declaration: ComponentConfig,
    *,
    primary_model: nn.Module,
    process: Process | None,
    objective: nn.Module | None,
    model_factory: ModelFactory,
    objective_factory: ObjectiveFactory,
    registries: RegistryCatalog = REGISTRIES,
) -> TrainingPlanAssembly:
    """Construct one canonical plan and retain its optional execution binding."""

    if registries is REGISTRIES:
        activate_training_component_builtins()
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
    execution_binding = (
        cast(TrainingExecutionBindingBuilder, builder)
        if isinstance(builder, TrainingExecutionBindingBuilder)
        else None
    )
    return TrainingPlanAssembly(
        plan=plan,
        builder_name=declaration.name,
        _execution_binding=execution_binding,
    )


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
    declared_inference_assets = cast(object, value.inference_assets)
    if not isinstance(declared_inference_assets, Mapping):
        raise TypeError("TrainingPlan.inference_assets must be a mapping")

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

    inference_assets: dict[str, InferenceAssetProjection] = {}
    projected_training_assets: dict[str, str] = {}
    for declared_slot, declared_projection in declared_inference_assets.items():
        slot = _validate_projection_string(
            cast(object, declared_slot),
            path="TrainingPlan.inference_assets slot",
        )
        projection = cast(object, declared_projection)
        if not isinstance(projection, InferenceAssetProjection):
            raise TypeError(
                f"TrainingPlan.inference_assets[{slot!r}] must be "
                "InferenceAssetProjection"
            )
        projection_path = f"TrainingPlan.inference_assets[{slot!r}]"
        training_asset_name = _validate_projection_string(
            cast(object, projection.training_asset_name),
            path=f"{projection_path}.training_asset_name",
        )
        if training_asset_name not in declared_auxiliaries:
            raise ValueError(
                f"{projection_path} references missing training auxiliary "
                f"{training_asset_name!r}"
            )
        previous_slot = projected_training_assets.setdefault(
            training_asset_name,
            slot,
        )
        if previous_slot != slot:
            raise ValueError(
                "TrainingPlan.inference_assets cannot project training auxiliary "
                f"{training_asset_name!r} more than once "
                f"(slots {previous_slot!r} and {slot!r})"
            )
        declaration_value = cast(object, projection.declaration)
        if not isinstance(declaration_value, ComponentConfig):
            raise TypeError(
                f"{projection_path}.declaration must be ComponentConfig"
            )
        declaration_name = _validate_projection_string(
            cast(object, declaration_value.name),
            path=f"{projection_path}.declaration.name",
        )
        declaration_params = cast(object, declaration_value.params)
        if type(declaration_params) is not dict:
            raise TypeError(
                f"{projection_path}.declaration.params must be an exact dictionary"
            )
        capability_role = _validate_projection_string(
            cast(object, projection.capability_role),
            path=f"{projection_path}.capability_role",
        )
        inference_assets[slot] = InferenceAssetProjection(
            training_asset_name=training_asset_name,
            declaration=ComponentConfig(
                name=declaration_name,
                params=deepcopy(cast(dict[str, Any], declaration_params)),
            ),
            capability_role=capability_role,
        )
    return TrainingPlan(
        strategy=value.strategy,
        primary_model=value.primary_model,
        process=value.process,
        objective=value.objective,
        inference_recipe=recipe,
        auxiliary_modules=MappingProxyType(dict(value.auxiliary_modules)),
        inference_assets=MappingProxyType(inference_assets),
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


def _validate_projection_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result:
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


def _validate_primary_execution_state(
    canonical_model: nn.Module,
    execution_model: nn.Module,
) -> None:
    """Require an execution wrapper to expose only canonical model state."""

    canonical_parameters = tuple(map(id, canonical_model.parameters()))
    execution_parameters = tuple(map(id, execution_model.parameters()))
    if execution_parameters != canonical_parameters:
        raise ValueError(
            "primary execution model parameters must preserve canonical "
            "primary parameter identities and order"
        )
    canonical_buffers = tuple(map(id, canonical_model.buffers()))
    execution_buffers = tuple(map(id, execution_model.buffers()))
    if execution_buffers != canonical_buffers:
        raise ValueError(
            "primary execution model buffers must preserve canonical primary "
            "buffer identities and order"
        )
    _validate_registered_execution_state(
        canonical_model,
        canonical_parameters=canonical_parameters,
        canonical_buffers=canonical_buffers,
        path="primary model",
    )
    _validate_registered_execution_state(
        execution_model,
        canonical_parameters=canonical_parameters,
        canonical_buffers=canonical_buffers,
        path="primary execution model",
    )


def _validate_registered_execution_state(
    module: nn.Module,
    *,
    canonical_parameters: tuple[int, ...],
    canonical_buffers: tuple[int, ...],
    path: str,
) -> None:
    """Reject state that DDP cannot synchronize through canonical tensors."""

    allowed_ids = {*canonical_parameters, *canonical_buffers}
    state = module.state_dict(keep_vars=True)
    present_parameter_ids: set[int] = set()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or id(value) not in allowed_ids:
            raise ValueError(
                f"{path} state entry {name!r} is not a registered canonical "
                "parameter or buffer"
            )
        if id(value) in canonical_parameters:
            present_parameter_ids.add(id(value))
    missing_parameters = set(canonical_parameters) - present_parameter_ids
    if missing_parameters:
        raise ValueError(
            f"{path} state_dict must contain every canonical parameter"
        )




__all__ = [
    "InferenceAssetProjection",
    "ManagedTrainingModule",
    "ModelFactory",
    "ObjectiveFactory",
    "TrainingBuilder",
    "TrainingBuilderContext",
    "TrainingExecutionBindingBuilder",
    "TrainingPlan",
    "TrainingPlanAssembly",
    "build_training_plan",
    "build_training_plan_assembly",
    "trainable_parameters",
    "training_module_roots",
    "validate_training_plan",
]
