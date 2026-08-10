"""Metric construction through the Stochaflow extension registry."""

from __future__ import annotations

from contextvars import ContextVar

from torchmetrics import Metric

from stochaflow._builtin_activation import activate_metric_builtins
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
    selected_registry = registry
    if selected_registry is None:
        selected_registry = _METRIC_REGISTRY_AUTHORITY.get()
    if selected_registry is None:
        selected_registry = REGISTRIES.metrics
    if selected_registry is REGISTRIES.metrics:
        activate_metric_builtins()
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
