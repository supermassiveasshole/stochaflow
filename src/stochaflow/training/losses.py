"""Algorithm-specific training step functions."""

from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion.gaussian import GaussianDiffusion
from stochaflow.diffusion.objectives import DDPMEpsilonObjective
from stochaflow.training.trainer import TrainStepOutput


def ddpm_epsilon_train_step(
    model: nn.Module,
    criterion: nn.Module,
    batch: Any,
    device: torch.device,
) -> TrainStepOutput:
    """Run one DDPM epsilon-prediction training step.

    Expected inputs:
    - ``model`` must implement the shared discrete Gaussian training contract
    - ``criterion`` must be a :class:`DDPMEpsilonObjective`
    - ``batch`` must be a tensor of clean samples or a tuple/list whose first
      element is the clean-sample tensor
    """

    if not isinstance(model, GaussianDiffusion):
        raise TypeError(
            "ddpm_epsilon_train_step expects a DDPM-compatible model"
        )
    if not isinstance(criterion, DDPMEpsilonObjective):
        raise TypeError(
            "ddpm_epsilon_train_step expects criterion to be an instance of DDPMEpsilonObjective"
        )

    if isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise TypeError("batch tuple/list must contain at least one tensor")
        x0 = batch[0]
    else:
        x0 = batch

    if not isinstance(x0, torch.Tensor):
        raise TypeError("ddpm_epsilon_train_step expects batch to provide a tensor of clean samples")

    x0 = x0.to(device)
    ddpm_batch = model(x0)
    loss = criterion(ddpm_batch)
    per_sample_loss = (ddpm_batch.predicted_noise - ddpm_batch.noise).pow(2)
    per_sample_loss = per_sample_loss.flatten(start_dim=1).mean(dim=1)
    return TrainStepOutput(
        loss=loss,
        diagnostics={
            "timesteps": ddpm_batch.timesteps.detach(),
            "per_sample_loss": per_sample_loss.detach(),
            "predicted_noise": ddpm_batch.predicted_noise.detach(),
            "target_noise": ddpm_batch.noise.detach(),
            "clean_samples": x0.detach(),
        },
    )
