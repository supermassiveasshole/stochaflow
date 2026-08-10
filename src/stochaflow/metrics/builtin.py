"""Small stable set of Stochaflow-owned metric registrations."""

from __future__ import annotations

from typing import Any

from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric as TorchMeanMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from stochaflow.utils.registry import REGISTRIES

REGISTRIES.metrics.require_base(Metric)


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


class SingleOutputMeanSquaredError(MeanSquaredError):
    """Compute scalar MSE with fixed single-output semantics."""

    def __init__(
        self,
        *,
        squared: bool = True,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        if squared is not True:
            raise ValueError("mse metric fixes squared=True")
        if type(num_outputs) is not int or num_outputs != 1:
            raise ValueError("mse metric fixes num_outputs=1")
        super().__init__(
            squared=True,
            num_outputs=1,
            **kwargs,
        )


class SingleOutputMeanAbsoluteError(MeanAbsoluteError):
    """Compute scalar MAE with fixed single-output semantics."""

    def __init__(
        self,
        *,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        if type(num_outputs) is not int or num_outputs != 1:
            raise ValueError("mae metric fixes num_outputs=1")
        super().__init__(
            num_outputs=1,
            **kwargs,
        )


__all__ = [
    "ErrorOnNanMeanMetric",
    "SingleOutputMeanAbsoluteError",
    "SingleOutputMeanSquaredError",
]
