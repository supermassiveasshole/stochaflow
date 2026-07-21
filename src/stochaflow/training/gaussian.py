"""Internal bridge for existing Gaussian epsilon training."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from stochaflow.processes import DiscreteGaussianDenoisingProcess, Process
from stochaflow.training.objectives import DDPMEpsilonObjective
from stochaflow.training.trainer import TrainStepOutput


@dataclass(frozen=True, slots=True)
class GaussianEpsilonTrainingOutput:
    """Tensors produced by one Gaussian epsilon training forward pass."""

    timesteps: torch.Tensor
    noisy: torch.Tensor
    noise: torch.Tensor
    predicted_noise: torch.Tensor


class GaussianEpsilonTrainingSystem(nn.Module):
    """Capability bridge combining an inference model and Gaussian process."""

    def __init__(
        self,
        inference_model: nn.Module,
        process: Process,
    ) -> None:
        super().__init__()
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "Gaussian epsilon training requires "
                "DiscreteGaussianDenoisingProcess capability"
            )
        if process.terminal_time <= process.clean_time:
            raise ValueError("Gaussian training requires a non-empty noisy time range")
        self.inference_model = inference_model
        self.process = process

    def forward(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> GaussianEpsilonTrainingOutput:
        """Sample a marginal state and predict its epsilon target."""

        if timesteps is None:
            timesteps = torch.randint(
                self.process.clean_time + 1,
                self.process.terminal_time + 1,
                (clean.shape[0],),
                device=clean.device,
            )
        noisy, noise = self.process.sample_marginal(clean, timesteps)
        model_times = timesteps - self.process.clean_time - 1
        predicted_value: object = self.inference_model(noisy, model_times)
        if not isinstance(predicted_value, torch.Tensor):
            raise TypeError("inference model must return a Tensor")
        if predicted_value.shape != noisy.shape:
            raise ValueError("inference model output must match the noisy sample shape")
        return GaussianEpsilonTrainingOutput(
            timesteps, noisy, noise, predicted_value
        )


def ddpm_epsilon_train_step(
    model: nn.Module,
    criterion: nn.Module,
    batch: Any,
    device: torch.device,
) -> TrainStepOutput:
    """Adapt the temporary Gaussian epsilon system to the generic Trainer."""

    if not isinstance(model, GaussianEpsilonTrainingSystem):
        raise TypeError(
            "ddpm_epsilon_train_step expects GaussianEpsilonTrainingSystem"
        )
    if not isinstance(criterion, DDPMEpsilonObjective):
        raise TypeError(
            "ddpm_epsilon_train_step expects DDPMEpsilonObjective"
        )
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise TypeError("batch tuple/list must contain at least one tensor")
        clean = batch[0]
    else:
        clean = batch
    if not isinstance(clean, torch.Tensor):
        raise TypeError(
            "ddpm_epsilon_train_step expects a clean-sample Tensor"
        )

    output = model(clean.to(device))
    loss, per_sample_loss = criterion.compute(
        output.predicted_noise,
        output.noise,
    )
    if loss.ndim != 0:
        raise ValueError(
            "ddpm_epsilon training requires an objective with scalar reduction"
        )
    return TrainStepOutput(
        loss=loss,
        diagnostics={
            "timesteps": output.timesteps.detach(),
            "per_sample_loss": per_sample_loss.detach(),
            "predicted_noise": output.predicted_noise.detach(),
            "target_noise": output.noise.detach(),
            "clean_samples": clean.detach(),
        },
    )


__all__ = [
    "GaussianEpsilonTrainingOutput",
    "GaussianEpsilonTrainingSystem",
    "ddpm_epsilon_train_step",
]
