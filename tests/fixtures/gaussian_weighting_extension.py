"""Installed-plugin fixture for Gaussian weighting strict-resume tests."""

from __future__ import annotations

import torch
from torch import nn

from stochaflow.extensions import (
    REGISTRIES,
    GaussianSimpleLossContext,
    GaussianSimpleLossWeighting,
    register_gaussian_simple_loss_weighting,
)
from stochaflow.families.gaussian import PredictionType

MODEL_NAME = "tests.gaussian-weighting-plugin.denoiser"
WEIGHTING_NAME = "tests.gaussian-weighting-plugin.scaled-snr"


class PluginGaussianDenoiser(nn.Module):
    """Tiny parameter-bearing denoiser used by the CLI lifecycle fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        """Return a batch-aligned clean-state prediction."""

        del model_time
        return torch.zeros_like(state) + self.offset


@register_gaussian_simple_loss_weighting(WEIGHTING_NAME)
class PluginScaledSnrWeighting(GaussianSimpleLossWeighting):
    """Scale an SNR-shaped identity tensor through the public policy API."""

    def __init__(self, *, scale: float) -> None:
        self.scale = float(scale)

    @property
    def requires_per_sample_loss(self) -> bool:
        """Require explicit per-sample Objective reduction."""

        return True

    def validate_contract(self, *, prediction_type: PredictionType) -> None:
        """Support every Gaussian prediction representation."""

    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Return one configured floating-point weight per sample."""

        return torch.ones_like(context.signal_to_noise_ratio) * self.scale


REGISTRIES.models.add(MODEL_NAME, PluginGaussianDenoiser)
