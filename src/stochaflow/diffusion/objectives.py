"""Training objectives for diffusion processes."""

import torch
import torch.nn as nn

from stochaflow.diffusion.gaussian import DiffusionForwardOutput
from stochaflow.utils.registry import register_objective


@register_objective("ddpm_epsilon")
class DDPMEpsilonObjective(nn.Module):
    """Epsilon-prediction objective for DDPM training.

    This criterion assumes the DDPM process is responsible for:

    ```python
    batch = ddpm(x0)
    loss = criterion(batch)
    ```

    The objective only compares the DDPM forward prediction against the target
    noise prepared by the same forward pass.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        batch: DiffusionForwardOutput,
    ) -> torch.Tensor:
        """Compute epsilon-prediction MSE loss for a DDPM batch."""

        return self.loss_fn(batch.predicted_noise, batch.noise)
