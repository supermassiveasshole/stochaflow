"""Built-in reusable training objectives."""

from typing import Protocol, cast, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from stochaflow.utils.registry import REGISTRIES


@runtime_checkable
class PerSampleObjective(Protocol):
    """Optional capability for training policies and diagnostics."""

    def per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return one scalar loss per leading batch item."""

        ...


@REGISTRIES.objectives.register("mse")
class MSEObjective(nn.Module):
    """Task-neutral mean-squared-error objective."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("MSE objective reduction must be mean or sum")
        self.reduction = reduction

    def per_sample_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return batch-aligned MSE values for policies and diagnostics."""

        elementwise = F.mse_loss(prediction, target, reduction="none")
        if elementwise.ndim == 0:
            raise ValueError("per-sample MSE requires a batch dimension")
        flattened = elementwise.reshape(elementwise.shape[0], -1)
        if self.reduction == "sum":
            return flattened.sum(dim=1)
        return flattened.mean(dim=1)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare prediction and target with configured scalar reduction."""

        return F.mse_loss(prediction, target, reduction=self.reduction)


def compute_objective(
    objective: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate a generic Objective and its optional per-sample capability."""

    loss_value: object = objective(prediction, target)
    if not isinstance(loss_value, torch.Tensor):
        raise TypeError("training objective must return a Tensor")
    if loss_value.ndim != 0:
        raise ValueError("training objective must return a scalar Tensor")
    per_sample_value: object | None = None
    if isinstance(objective, PerSampleObjective):
        per_sample_value = cast(
            object,
            objective.per_sample_loss(prediction, target),
        )
    if per_sample_value is not None:
        if not isinstance(per_sample_value, torch.Tensor):
            raise TypeError("per-sample objective capability must return a Tensor")
        if (
            per_sample_value.ndim != 1
            or per_sample_value.shape[0] != prediction.shape[0]
        ):
            raise ValueError(
                "per-sample objective output must match the prediction batch"
            )
    return loss_value, per_sample_value


__all__ = ["MSEObjective", "PerSampleObjective", "compute_objective"]
