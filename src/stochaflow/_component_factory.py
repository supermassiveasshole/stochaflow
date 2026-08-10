"""Narrow registry-backed construction for shared runtime components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from torch import nn

from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, Registry

if TYPE_CHECKING:
    from stochaflow.processes.base import Process


def _build_from_registry(
    registry: Registry[Any],
    component: ComponentConfig,
) -> Any:
    return registry.create(component.name, **component.params)


def build_model(component: ComponentConfig) -> nn.Module:
    """Instantiate a model from the model registry."""

    return cast(nn.Module, _build_from_registry(REGISTRIES.models, component))


def build_process(component: ComponentConfig) -> Process:
    """Instantiate a model-free probability process."""

    return cast(Any, _build_from_registry(REGISTRIES.processes, component))


def build_objective(component: ComponentConfig) -> nn.Module:
    """Instantiate a training objective from the objective registry."""

    return cast(nn.Module, _build_from_registry(REGISTRIES.objectives, component))
