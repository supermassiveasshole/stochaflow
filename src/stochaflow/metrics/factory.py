"""Metric construction through the Stochaflow extension registry."""

from __future__ import annotations

from importlib import import_module

from torchmetrics import Metric

from stochaflow.metrics.config import MetricSpec, validate_metric_spec
from stochaflow.utils.registry import REGISTRIES, Registry


def build_metric(
    spec: MetricSpec,
    *,
    registry: Registry[type[Metric]] = REGISTRIES.metrics,
) -> Metric:
    """Construct one validated metric without importing arbitrary targets."""

    validate_metric_spec(spec)
    import_module("stochaflow.metrics.builtin")
    import_module("stochaflow.metrics.reference")
    metric = registry.create(spec.name, **spec.params)
    if not isinstance(metric, Metric):
        raise TypeError(
            f"registered metric {spec.name!r} constructed "
            f"{type(metric).__name__}, expected torchmetrics.Metric"
        )
    return metric


__all__ = ["build_metric"]
