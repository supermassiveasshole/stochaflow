"""Independent Strategy and Metric registrations for vertical extension tests."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchmetrics import Metric

from stochaflow.extensions import (
    REGISTRIES,
    MetricUpdate,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
)

MODEL_NAME = "test.vertical-extension.linear-model"
TRAINING_BUILDER_NAME = "test.vertical-extension.vector-training"
METRIC_NAME = "test.vertical-extension.relative-l2"
METRIC_CHANNEL = "test.vertical-extension.vector-pair"


class VerticalExtensionLinearModel(nn.Module):
    """A tiny plugin-owned linear model used by the vertical training fixture."""

    def __init__(self, *, features: int = 2) -> None:
        super().__init__()
        self.projection = nn.Linear(features, features, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project one vector batch."""

        return self.projection(inputs)


class VerticalExtensionRelativeL2Metric(Metric):
    """A plugin-owned relative L2 metric with additive distributed state."""

    relative_error: torch.Tensor
    observations: torch.Tensor

    def __init__(self) -> None:
        super().__init__(
            dist_sync_on_step=False,
            sync_on_compute=True,
        )
        self.add_state(
            "relative_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "observations",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        """Accumulate relative error for a vector batch."""

        numerator = torch.linalg.vector_norm(
            prediction - reference,
            dim=-1,
        )
        denominator = torch.linalg.vector_norm(reference, dim=-1).clamp_min(
            torch.finfo(reference.dtype).eps
        )
        relative = numerator / denominator
        self.relative_error += relative.sum()
        self.observations += relative.numel()

    def compute(self) -> torch.Tensor:
        """Return the mean relative L2 error."""

        return self.relative_error / self.observations


class VerticalExtensionVectorStrategy(TrainingStrategy):
    """Interpret a plugin-owned vector pair and emit its metric channel."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @property
    def metric_channels(self) -> frozenset[str]:
        """Declare the plugin-owned update channel."""

        return frozenset({METRIC_CHANNEL})

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Train the injected model against paired vector targets."""

        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise TypeError("vertical extension requires an input/target pair")
        inputs, reference = batch
        if not isinstance(inputs, torch.Tensor) or not isinstance(
            reference,
            torch.Tensor,
        ):
            raise TypeError("vertical extension pair members must be tensors")
        prediction = self.model(inputs)
        return TrainStepOutput(
            loss=(prediction - reference).square().mean(),
            metric_updates={
                METRIC_CHANNEL: MetricUpdate(
                    args=(prediction, reference),
                )
            },
            loss_aggregation_weight=inputs.shape[0],
        )


class VerticalExtensionTrainingBuilder(TrainingBuilder):
    """Compose the plugin-owned Strategy with the injected primary model."""

    def build(self) -> TrainingPlan:
        """Return a task-local training plan."""

        return TrainingPlan(
            strategy=VerticalExtensionVectorStrategy(
                self.context.primary_model,
            ),
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=self.context.objective,
        )


REGISTRIES.models.add(MODEL_NAME, VerticalExtensionLinearModel)
REGISTRIES.training_builders.add(
    TRAINING_BUILDER_NAME,
    VerticalExtensionTrainingBuilder,
)
REGISTRIES.metrics.add(METRIC_NAME, VerticalExtensionRelativeL2Metric)
