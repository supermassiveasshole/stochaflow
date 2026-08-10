"""Built-in task-neutral training builders."""

from typing import Any

import torch
from torch import nn

from stochaflow.metrics import MetricUpdate
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.training.objectives import compute_objective
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput


class SupervisedTrainingStrategy(TrainingStrategy):
    """Compute a scalar objective for conventional input/target batches."""

    def __init__(self, model: nn.Module, objective: nn.Module) -> None:
        self.model = model
        self.objective = objective

    @property
    def metric_channels(self) -> frozenset[str]:
        """Return the supervised prediction/target update channel."""

        return frozenset(("supervised.prediction_target",))

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Run one supervised forward and objective computation."""

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError("supervised training expects an (inputs, targets) batch")
        inputs, target_value = batch
        prediction_value: object = self.model(inputs)
        if not isinstance(prediction_value, torch.Tensor):
            raise TypeError("supervised model must return a Tensor")
        if not isinstance(target_value, torch.Tensor):
            raise TypeError("supervised target must be a Tensor")
        loss, _ = compute_objective(
            self.objective,
            prediction_value,
            target_value,
        )
        batch_size = prediction_value.shape[0] if prediction_value.ndim > 0 else 1
        return TrainStepOutput(
            loss=loss,
            metric_updates={
                "supervised.prediction_target": MetricUpdate(
                    args=(prediction_value, target_value)
                )
            },
            loss_aggregation_weight=batch_size,
        )


class SupervisedTrainingBuilder(TrainingBuilder):
    """Assemble the built-in supervised strategy."""

    def build(self) -> TrainingPlan:
        """Return a plan using the injected primary model and objective."""

        if self.context.params:
            unknown = ", ".join(sorted(self.context.params))
            raise ValueError(f"unknown supervised training parameter(s): {unknown}")
        objective = self.context.objective
        if objective is None:
            raise TypeError("supervised training requires objective")
        return TrainingPlan(
            strategy=SupervisedTrainingStrategy(
                self.context.primary_model,
                objective,
            ),
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=objective,
        )


__all__ = ["SupervisedTrainingBuilder", "SupervisedTrainingStrategy"]
