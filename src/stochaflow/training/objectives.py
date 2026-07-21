"""Built-in training objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stochaflow.utils.registry import REGISTRIES


@REGISTRIES.objectives.register("ddpm_epsilon")
class DDPMEpsilonObjective(nn.Module):
    """Mean-squared-error objective for epsilon prediction."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("DDPM epsilon objective reduction must be mean, sum, or none")
        self.reduction = reduction

    def compute(
        self,
        predicted_noise: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the configured loss and batch-aligned per-sample losses."""

        elementwise = F.mse_loss(predicted_noise, noise, reduction="none")
        if elementwise.ndim == 0:
            raise ValueError("DDPM epsilon objective requires a batch dimension")
        per_sample = elementwise.reshape(elementwise.shape[0], -1).mean(dim=1)
        if self.reduction == "mean":
            loss = elementwise.mean()
        elif self.reduction == "sum":
            loss = elementwise.sum()
        else:
            loss = elementwise
        return loss, per_sample

    def forward(self, predicted_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compare predicted and target Gaussian noise."""

        return self.compute(predicted_noise, noise)[0]


__all__ = ["DDPMEpsilonObjective"]
