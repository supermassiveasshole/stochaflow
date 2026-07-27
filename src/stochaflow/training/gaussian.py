"""Gaussian-family training strategy and registered builder."""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.sampling import GaussianModelDynamics, PredictionType
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.training.objectives import compute_objective
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.sampling_recipe import SamplingRecipe


@runtime_checkable
class GaussianDiagnosticSemantics(Protocol):
    """Optional Gaussian model-invocation capability for diagnostics."""

    @property
    def prediction_type(self) -> PredictionType:
        """Return the model prediction parameterization used for training."""

        ...

    def predict_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the task-adapted model for a Gaussian diagnostic."""

        ...


class GaussianDenoisingTrainingStrategy(TrainingStrategy):
    """Train a model against one discrete Gaussian marginal target."""

    def __init__(
        self,
        model: nn.Module,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        *,
        prediction_type: PredictionType = "epsilon",
    ) -> None:
        if prediction_type not in ("epsilon", "x0", "v", "score"):
            raise ValueError(
                "Gaussian prediction_type must be epsilon, x0, v, or score"
            )
        if process.terminal_time <= process.clean_time:
            raise ValueError("Gaussian training requires a non-empty noisy time range")
        self.model = model
        self.process = process
        self.objective = objective
        self._prediction_type: PredictionType = cast(
            PredictionType,
            prediction_type,
        )
        self.dynamics = GaussianModelDynamics(
            self.process,
            self.predict_gaussian_model,
            prediction_type=self.prediction_type,
            clip_denoised=False,
        )

    @property
    def prediction_type(self) -> PredictionType:
        """Return the configured Gaussian model parameterization."""

        return self._prediction_type

    def predict_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the built-in unconditional Gaussian model signature."""

        prediction: object = self.model(state, model_time)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Gaussian model must return a Tensor")
        return prediction

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Sample a noisy marginal and compare the configured model target."""

        clean = _clean_samples(batch)
        state_times = torch.randint(
            self.process.clean_time + 1,
            self.process.terminal_time + 1,
            (clean.shape[0],),
            device=clean.device,
        )
        noisy, noise = self.process.sample_marginal(clean, state_times)
        prediction = self.dynamics.predict(noisy, state_times)
        target = gaussian_training_target(
            self.process,
            clean=clean,
            noise=noise,
            state_times=state_times,
            prediction_type=self.prediction_type,
        )
        loss, per_sample = compute_objective(
            self.objective,
            prediction.model_output,
            target,
        )
        diagnostics: dict[str, Any] = {
            "timesteps": state_times.detach(),
            "predicted_noise": prediction.epsilon.detach(),
            "target_noise": noise.detach(),
            "predicted_clean": prediction.clean.detach(),
            "clean_samples": clean.detach(),
        }
        if per_sample is not None:
            diagnostics["per_sample_loss"] = per_sample.detach()
        return TrainStepOutput(loss=loss, diagnostics=diagnostics)


@REGISTRIES.training_builders.register("gaussian_denoising")
class GaussianDenoisingTrainingBuilder(TrainingBuilder):
    """Assemble the built-in discrete Gaussian training strategy."""

    def build(self) -> TrainingPlan:
        """Validate family assets and return a Gaussian training plan."""

        params = dict(self.context.params)
        prediction_type = params.pop("prediction_type", "epsilon")
        if params:
            unknown = ", ".join(sorted(params))
            raise ValueError(
                f"unknown gaussian_denoising training parameter(s): {unknown}"
            )
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "gaussian_denoising training requires "
                "DiscreteGaussianDenoisingProcess"
            )
        objective = self.context.objective
        if objective is None:
            raise TypeError("gaussian_denoising training requires objective")
        if not isinstance(prediction_type, str) or prediction_type not in (
            "epsilon",
            "x0",
            "v",
            "score",
        ):
            raise ValueError(
                "training.params.prediction_type must be epsilon, x0, v, or score"
            )
        strategy = GaussianDenoisingTrainingStrategy(
            self.context.primary_model,
            process,
            objective,
            prediction_type=cast(PredictionType, prediction_type),
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=process,
            objective=objective,
            inference_recipe=SamplingRecipe(
                name="standard_denoising",
                contract={"prediction_type": prediction_type},
            ),
        )


def _clean_samples(batch: Any) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise TypeError(
                "gaussian_denoising expects a Tensor or (Tensor, conditions) batch"
            )
        clean, conditions = batch
        if not isinstance(conditions, dict) or conditions:
            raise TypeError(
                "built-in gaussian_denoising supports only empty condition mappings; "
                "use a custom TrainingBuilder for conditional training"
            )
    else:
        clean = batch
    if not isinstance(clean, torch.Tensor):
        raise TypeError("gaussian_denoising clean samples must be a Tensor")
    if clean.ndim == 0:
        raise ValueError("gaussian_denoising samples must have a batch dimension")
    return clean


def gaussian_training_target(
    process: DiscreteGaussianDenoisingProcess,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    prediction_type: PredictionType,
) -> torch.Tensor:
    """Build the configured discrete Gaussian model-training target."""

    process_value = cast(object, process)
    if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "Gaussian training target requires DiscreteGaussianDenoisingProcess"
        )
    process = process_value
    clean_value = cast(object, clean)
    noise_value = cast(object, noise)
    if not isinstance(clean_value, torch.Tensor) or not isinstance(
        noise_value,
        torch.Tensor,
    ):
        raise TypeError("Gaussian training clean state and noise must be Tensors")
    clean = clean_value
    noise = noise_value
    if clean.ndim == 0:
        raise ValueError("Gaussian training clean state must have a batch dimension")
    if not torch.is_floating_point(clean) or not torch.is_floating_point(noise):
        raise TypeError("Gaussian training clean state and noise must be floating-point")
    if noise.shape != clean.shape:
        raise ValueError("Gaussian training noise must match the clean state shape")
    if noise.device != clean.device:
        raise ValueError("Gaussian training noise must share the clean state device")
    if noise.dtype != clean.dtype:
        raise ValueError("Gaussian training noise must share the clean state dtype")
    state_times = process.validate_noisy_state_times(state_times)
    if state_times.shape[0] != clean.shape[0]:
        raise ValueError("Gaussian training state times must match the batch")
    if state_times.device != clean.device:
        raise ValueError(
            "Gaussian training state times must share the clean state device"
        )
    if prediction_type not in ("epsilon", "x0", "v", "score"):
        raise ValueError(
            "Gaussian prediction_type must be epsilon, x0, v, or score"
        )
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "x0":
        return clean
    scales = process.marginal_scales(state_times, clean.size())
    if prediction_type == "v":
        return scales.signal * noise - scales.noise * clean
    return -noise / scales.noise


__all__ = [
    "GaussianDenoisingTrainingBuilder",
    "GaussianDenoisingTrainingStrategy",
    "GaussianDiagnosticSemantics",
    "gaussian_training_target",
]
