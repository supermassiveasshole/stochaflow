"""Shared implementation identity for formal and live Evaluation protocols."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata
from typing import Any

from packaging.utils import canonicalize_name
from torchmetrics import Metric

from stochaflow.evaluation.builder import EvaluationProtocolIdentity
from stochaflow.metrics.config import MetricSpec
from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    extension_plugin_provenance_to_dicts,
)
from stochaflow.utils.registry import Registry


def build_protocol_implementation_identity(
    *,
    evaluation_builder_name: str,
    metric_specs: Sequence[MetricSpec],
    declared: EvaluationProtocolIdentity,
    evaluation_builder_registry: Registry[type[Any]],
    metric_registry: Registry[type[Metric]],
    runtime_parameters: Mapping[str, Any],
    extension_provenance: Sequence[ExtensionPluginProvenance] = (),
) -> dict[str, Any]:
    """Bind task-declared facts to exact component and distribution versions."""

    import_module("stochaflow.metrics.builtin")
    import_module("stochaflow.metrics.reference")
    package_distributions = metadata.packages_distributions()
    builder = evaluation_builder_registry.resolve(evaluation_builder_name)
    metrics = [
        {
            "id": spec.id,
            "name": spec.name,
            "implementation": _component_implementation_identity(
                metric_registry.resolve(spec.name),
                package_distributions=package_distributions,
            ),
        }
        for spec in metric_specs
    ]
    declared_metric_providers = [
        {
            "name": name,
            "implementation": _component_implementation_identity(
                metric_registry.resolve(name),
                package_distributions=package_distributions,
            ),
        }
        for name in declared.metric_providers
    ]
    runtime_distributions = tuple(
        sorted(
            {
                "stochaflow",
                "torch",
                "torchmetrics",
                *declared.dependencies,
            }
        )
    )
    return {
        "schema_version": 1,
        "declared": {
            "providers": declared.providers,
            "preprocessing": declared.preprocessing,
            "metric_providers": list(declared.metric_providers),
            "dependencies": list(declared.dependencies),
        },
        "components": {
            "evaluation_builder": {
                "name": evaluation_builder_name,
                "implementation": _component_implementation_identity(
                    builder,
                    package_distributions=package_distributions,
                ),
            },
            "metrics": metrics,
            "declared_metric_providers": declared_metric_providers,
        },
        "extensions": extension_plugin_provenance_to_dicts(
            tuple(extension_provenance)
        ),
        "runtime": {
            "python": platform.python_version(),
            "distributions": [
                {
                    "name": name,
                    "version": _installed_distribution_version(name),
                }
                for name in runtime_distributions
            ],
            "parameters": dict(runtime_parameters),
        },
    }


def _component_implementation_identity(
    component: object,
    *,
    package_distributions: Mapping[str, list[str]],
) -> dict[str, Any]:
    module_name = getattr(component, "__module__", None)
    qualified_name = getattr(component, "__qualname__", None)
    if type(module_name) is not str or not module_name:
        raise TypeError("registered evaluation component must declare __module__")
    if type(qualified_name) is not str or not qualified_name:
        raise TypeError("registered evaluation component must declare __qualname__")
    package_name = module_name.partition(".")[0]
    distribution_names = sorted(
        {
            canonicalize_name(name)
            for name in package_distributions.get(package_name, ())
        }
    )
    return {
        "module": module_name,
        "qualname": qualified_name,
        "distributions": [
            {
                "name": name,
                "version": _installed_distribution_version(name),
            }
            for name in distribution_names
        ],
    }


def _installed_distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "evaluation protocol dependency is not installed: " + name
        ) from error


__all__ = ["build_protocol_implementation_identity"]
