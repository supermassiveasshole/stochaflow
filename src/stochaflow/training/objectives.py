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


@runtime_checkable
class BatchReduciblePerSampleObjective(PerSampleObjective, Protocol):
    """Objective capability with an explicit batch-reduction policy."""

    def reduce_per_sample_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce one scalar loss per batch item to a scalar loss."""

        ...


def validate_per_sample_loss(
    value: object,
    *,
    prediction: torch.Tensor,
) -> torch.Tensor:
    """Validate a batch-aligned floating-point objective result."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("per-sample objective capability must return a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(
            "per-sample objective capability must return a floating-point Tensor"
        )
    if prediction.ndim == 0:
        raise ValueError("per-sample objective validation requires a batch dimension")
    if value.ndim != 1 or value.shape[0] != prediction.shape[0]:
        raise ValueError("per-sample objective output must match the prediction batch")
    if value.device != prediction.device:
        raise ValueError(
            "per-sample objective output must be on the prediction device"
        )
    return value


def validate_reduced_loss(
    value: object,
    *,
    per_sample_loss: torch.Tensor,
) -> torch.Tensor:
    """Validate a floating-point scalar produced by a batch reducer."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("objective batch reducer must return a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError("objective batch reducer must return a floating-point Tensor")
    if value.ndim != 0:
        raise ValueError("objective batch reducer must return a scalar Tensor")
    if value.device != per_sample_loss.device:
        raise ValueError(
            "objective batch reducer output must be on the per-sample loss device"
        )
    return value


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

    def reduce_per_sample_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce batch-aligned MSE values with the configured semantics."""

        per_sample_loss = validate_per_sample_loss(loss, prediction=loss)
        if self.reduction == "sum":
            reduced = per_sample_loss.sum()
        else:
            reduced = per_sample_loss.mean()
        return validate_reduced_loss(reduced, per_sample_loss=per_sample_loss)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare prediction and target with configured scalar reduction."""

        per_sample_loss = self.per_sample_loss(prediction, target)
        return self.reduce_per_sample_loss(per_sample_loss)


def compute_objective(
    objective: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate a generic Objective and its optional per-sample capability."""

    loss_value: object = objective(prediction, target)
    if not isinstance(loss_value, torch.Tensor):
        raise TypeError("training objective must return a Tensor")
    if not torch.is_floating_point(loss_value):
        raise TypeError("training objective must return a floating-point Tensor")
    if loss_value.ndim != 0:
        raise ValueError("training objective must return a scalar Tensor")
    per_sample_value: object | None = None
    if isinstance(objective, PerSampleObjective):
        per_sample_value = cast(
            object,
            objective.per_sample_loss(prediction, target),
        )
    if per_sample_value is not None:
        per_sample_value = validate_per_sample_loss(
            per_sample_value,
            prediction=prediction,
        )
    return loss_value, per_sample_value


__all__ = [
    "BatchReduciblePerSampleObjective",
    "MSEObjective",
    "PerSampleObjective",
    "compute_objective",
    "validate_per_sample_loss",
    "validate_reduced_loss",
]
