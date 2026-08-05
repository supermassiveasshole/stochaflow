"""Task-neutral metric declarations, payloads, construction, and runtime."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from stochaflow.metrics.config import MetricSpec
from stochaflow.metrics.contracts import MetricUpdate

if TYPE_CHECKING:
    from stochaflow.metrics.builtin import (
        ErrorOnNanMeanMetric,
        SingleOutputMeanAbsoluteError,
        SingleOutputMeanSquaredError,
    )
    from stochaflow.metrics.factory import build_metric
    from stochaflow.metrics.reference import (
        FrechetInceptionDistanceMetric,
        KernelInceptionDistanceMetric,
        ShareableImageFeatureMetric,
    )
    from stochaflow.metrics.runtime import MetricEngine, MetricRuntimeError

_LAZY_METRIC_EXPORTS = {
    "ErrorOnNanMeanMetric": (
        "stochaflow.metrics.builtin",
        "ErrorOnNanMeanMetric",
    ),
    "FrechetInceptionDistanceMetric": (
        "stochaflow.metrics.reference",
        "FrechetInceptionDistanceMetric",
    ),
    "KernelInceptionDistanceMetric": (
        "stochaflow.metrics.reference",
        "KernelInceptionDistanceMetric",
    ),
    "ShareableImageFeatureMetric": (
        "stochaflow.metrics.reference",
        "ShareableImageFeatureMetric",
    ),
    "SingleOutputMeanAbsoluteError": (
        "stochaflow.metrics.builtin",
        "SingleOutputMeanAbsoluteError",
    ),
    "SingleOutputMeanSquaredError": (
        "stochaflow.metrics.builtin",
        "SingleOutputMeanSquaredError",
    ),
    "MetricEngine": ("stochaflow.metrics.runtime", "MetricEngine"),
    "MetricRuntimeError": (
        "stochaflow.metrics.runtime",
        "MetricRuntimeError",
    ),
    "build_metric": ("stochaflow.metrics.factory", "build_metric"),
}


def __getattr__(name: str) -> Any:
    """Load TorchMetrics-backed exports only when explicitly requested."""

    target = _LAZY_METRIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery."""

    return sorted(set(globals()).union(_LAZY_METRIC_EXPORTS))


__all__ = [
    "ErrorOnNanMeanMetric",
    "FrechetInceptionDistanceMetric",
    "KernelInceptionDistanceMetric",
    "MetricEngine",
    "MetricRuntimeError",
    "MetricSpec",
    "MetricUpdate",
    "ShareableImageFeatureMetric",
    "SingleOutputMeanAbsoluteError",
    "SingleOutputMeanSquaredError",
    "build_metric",
]
