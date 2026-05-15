"""Algorithm-specific training step functions."""

from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion.ddpm import DDPM
from stochaflow.diffusion.objectives import DDPMEpsilonObjective


def ddpm_epsilon_train_step(
    model: nn.Module,
    criterion: nn.Module,
    batch: Any,
    device: torch.device,
) -> torch.Tensor:
    """Run one DDPM epsilon-prediction training step.

    Expected inputs:
    - ``model`` must be a :class:`DDPM`
    - ``criterion`` must be a :class:`DDPMEpsilonObjective`
    - ``batch`` must be a tensor of clean samples or a tuple/list whose first
      element is the clean-sample tensor
    """

    if not isinstance(model, DDPM):
        raise TypeError("ddpm_epsilon_train_step expects model to be an instance of DDPM")
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
    return criterion(ddpm_batch)
