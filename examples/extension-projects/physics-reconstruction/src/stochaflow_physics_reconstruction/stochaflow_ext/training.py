"""Physics-conditioned Gaussian training composition."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from stochaflow.extensions import (
    REGISTRIES,
    DiscreteGaussianDenoisingProcess,
    PredictionType,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
    compute_objective,
    gaussian_training_target,
)

from ._config import copied_mapping, pop_float, pop_string, reject_unknown
from .model import ConditionalDenoiser
from .physics import conditioning_gradient


class PhysicsDenoisingStrategy(TrainingStrategy):
    """Interpret raw triplets and compute one conditional Gaussian objective."""

    def __init__(
        self,
        model: ConditionalDenoiser,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        *,
        prediction_type: PredictionType,
        conditioning_strength: float,
    ) -> None:
        self.model = model
        self.process = process
        self.objective = objective
        self.prediction_type: PredictionType = prediction_type
        self.conditioning_strength = conditioning_strength

    def training_step(self, batch: Any) -> TrainStepOutput:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("physics training batches must be raw Tensor triplets")
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError("physics training batches must have shape [B, 3, H, W]")
        clean = self.model.normalize(batch)
        state_times = torch.randint(
            self.process.clean_time + 1,
            self.process.terminal_time + 1,
            (clean.shape[0],),
            device=clean.device,
        )
        noisy, noise = self.process.sample_marginal(clean, state_times)
        condition, physics_loss = conditioning_gradient(
            noisy,
            self.model,
            strength=self.conditioning_strength,
        )
        model_times = state_times - self.process.clean_time - 1
        prediction = self.model(noisy, model_times, condition)
        target = gaussian_training_target(
            self.process,
            clean=clean,
            noise=noise,
            state_times=state_times,
            prediction_type=self.prediction_type,
        )
        loss_value, per_sample = compute_objective(
            self.objective,
            prediction,
            target,
        )
        diagnostics: dict[str, Any] = {
            "state_times": state_times.detach(),
            "condition_norm": condition.flatten(1).norm(dim=1).detach(),
        }
        if per_sample is not None:
            diagnostics["per_sample_loss"] = per_sample.detach()
        return TrainStepOutput(
            loss=loss_value,
            metrics={
                "denoising_loss": loss_value.detach(),
                "condition_residual": physics_loss,
            },
            diagnostics=diagnostics,
        )


@REGISTRIES.training_builders.register(
    "physics-reconstruction.gaussian-denoising"
)
class PhysicsTrainingBuilder(TrainingBuilder):
    """Compose the project model, Gaussian process, and injected objective."""

    def build(self) -> TrainingPlan:
        params = copied_mapping(self.context.params, path="training.params")
        prediction_type_value = pop_string(
            params,
            "prediction_type",
            path="training.params",
            default="epsilon",
        )
        if prediction_type_value not in {"epsilon", "x0", "v", "score"}:
            raise ValueError(
                "training.params.prediction_type must be epsilon, x0, v, or score"
            )
        conditioning_strength = pop_float(
            params,
            "conditioning_strength",
            path="training.params",
            default=1.0,
            minimum=0.0,
        )
        reject_unknown(params, path="training.params")
        model = self.context.primary_model
        if not isinstance(model, ConditionalDenoiser):
            raise TypeError(
                "physics-reconstruction training requires ConditionalDenoiser"
            )
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "physics-reconstruction training requires "
                "DiscreteGaussianDenoisingProcess"
            )
        objective = self.context.objective
        if objective is None:
            raise TypeError("physics-reconstruction training requires an objective")
        strategy = PhysicsDenoisingStrategy(
            model,
            process,
            objective,
            prediction_type=cast(PredictionType, prediction_type_value),
            conditioning_strength=conditioning_strength,
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=model,
            process=process,
            objective=objective,
        )


__all__ = ["PhysicsDenoisingStrategy", "PhysicsTrainingBuilder"]
