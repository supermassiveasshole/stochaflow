"""Small stable set of Stochaflow-owned metric registrations."""

from __future__ import annotations

from typing import Any

from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric as TorchMeanMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from stochaflow.utils.registry import REGISTRIES

REGISTRIES.metrics.require_base(Metric)


@REGISTRIES.metrics.register("mean")
class ErrorOnNanMeanMetric(TorchMeanMetric):
    """Aggregate a mean while failing immediately on NaN inputs."""

    def __init__(self, **kwargs: Any) -> None:
        configured_strategy = kwargs.pop("nan_strategy", "error")
        if configured_strategy != "error":
            raise ValueError(
                "mean metric fixes nan_strategy='error'; "
                "other strategies are not supported"
            )
        super().__init__(nan_strategy="error", **kwargs)


REGISTRIES.metrics.add("mse", MeanSquaredError)
REGISTRIES.metrics.add("mae", MeanAbsoluteError)


__all__ = ["ErrorOnNanMeanMetric"]
