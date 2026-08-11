"""Installed-plugin fixture proving a complete Gaussian training extension."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from stochaflow.extensions import (
    REGISTRIES,
    DiscreteGaussianDenoisingProcess,
    MSEObjective,
    SamplingRecipe,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
)

MODEL_NAME = "tests.gaussian-strategy-plugin.denoiser"
TRAINING_NAME = "tests.gaussian-strategy-plugin.scaled-x0"


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


class PluginScaledGaussianStrategy(TrainingStrategy):
    """Own a complete extension-specific Gaussian x0 training algorithm."""

    def __init__(
        self,
        model: nn.Module,
        process: DiscreteGaussianDenoisingProcess,
        objective: MSEObjective,
        *,
        scale: float,
    ) -> None:
        self.model = model
        self.process = process
        self.objective = objective
        self.scale = float(scale)

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Sample one marginal and scale the complete per-sample x0 loss."""

        if not isinstance(batch, torch.Tensor):
            raise TypeError("plugin Gaussian strategy expects a Tensor batch")
        state_times = torch.randint(
            self.process.clean_time + 1,
            self.process.terminal_time + 1,
            (batch.shape[0],),
            device=batch.device,
        )
        noisy, _ = self.process.sample_marginal(batch, state_times)
        prediction = self.model(
            noisy,
            state_times - self.process.clean_time - 1,
        )
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("plugin Gaussian model must return a Tensor")
        per_sample = self.objective.per_sample_loss(prediction, batch)
        weighted = per_sample * self.scale
        loss = (
            weighted.sum()
            if self.objective.reduction == "sum"
            else weighted.mean()
        )
        return TrainStepOutput(
            loss=loss,
            diagnostic_observation={
                "timestep_loss_weight": torch.full_like(
                    per_sample,
                    self.scale,
                )
            },
            loss_aggregation_weight=batch.shape[0],
        )


@REGISTRIES.training_builders.register(TRAINING_NAME)
class PluginScaledGaussianTrainingBuilder(TrainingBuilder):
    """Compose the plugin model with its concrete TrainingStrategy."""

    def build(self) -> TrainingPlan:
        """Validate private parameters and return the plugin training plan."""

        params = dict(self.context.params)
        scale = params.pop("scale", 1.0)
        if params:
            raise ValueError(
                "unknown plugin Gaussian training parameter(s): "
                + ", ".join(sorted(params))
            )
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("plugin Gaussian scale must be numeric")
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError("plugin Gaussian training requires a Gaussian process")
        objective = self.context.objective
        if not isinstance(objective, MSEObjective):
            raise TypeError("plugin Gaussian training requires MSEObjective")
        strategy = PluginScaledGaussianStrategy(
            self.context.primary_model,
            process,
            objective,
            scale=float(scale),
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=process,
            objective=objective,
            inference_recipe=SamplingRecipe(
                name="standard_denoising",
                contract={
                    "prediction_type": "x0",
                    "variance": {"mode": "fixed"},
                },
            ),
        )


REGISTRIES.models.add(MODEL_NAME, PluginGaussianDenoiser)
