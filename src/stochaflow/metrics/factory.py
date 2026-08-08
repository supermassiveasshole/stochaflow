"""Metric construction through the Stochaflow extension registry."""

from __future__ import annotations

from contextvars import ContextVar
from importlib import import_module

from torchmetrics import Metric

from stochaflow.metrics.config import MetricSpec, validate_metric_spec
from stochaflow.utils.registry import REGISTRIES, Registry

_METRIC_REGISTRY_AUTHORITY: ContextVar[Registry[type[Metric]] | None] = (
    ContextVar("stochaflow_metric_registry_authority", default=None)
)


def build_metric(
    spec: MetricSpec,
    *,
    registry: Registry[type[Metric]] | None = None,
) -> Metric:
    """Construct one metric under an explicit or inherited registry authority."""

    validate_metric_spec(spec)
    import_module("stochaflow.metrics.builtin")
    import_module("stochaflow.metrics.reference")
    selected_registry = registry
    if selected_registry is None:
        selected_registry = _METRIC_REGISTRY_AUTHORITY.get()
    if selected_registry is None:
        selected_registry = REGISTRIES.metrics
    token = _METRIC_REGISTRY_AUTHORITY.set(selected_registry)
    try:
        metric = selected_registry.create(spec.name, **spec.params)
    finally:
        _METRIC_REGISTRY_AUTHORITY.reset(token)
    if not isinstance(metric, Metric):
        raise TypeError(
            f"registered metric {spec.name!r} constructed "
            f"{type(metric).__name__}, expected torchmetrics.Metric"
        )
    return metric


__all__ = ["build_metric"]
