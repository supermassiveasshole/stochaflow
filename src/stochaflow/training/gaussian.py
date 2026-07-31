"""Gaussian-family training strategy and registered builder."""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.processes.gaussian import LearnedRangeGaussianVarianceProcess
from stochaflow.sampling import PredictionType, VarianceMode
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.training.gaussian_loss import (
    GaussianLossWeightingConfig,
    GaussianVarianceConfig,
    compute_gaussian_training_loss,
    gaussian_loss_diagnostics,
    gaussian_training_target,
    parse_gaussian_loss_weighting,
    parse_gaussian_variance,
    validate_gaussian_model_output_layout,
    validate_per_sample_objective,
)
from stochaflow.training.strategy import (
    Batch,
    TrainingStrategy,
    TrainStepOutput,
)
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.sampling_recipe import SamplingRecipe


@runtime_checkable
class GaussianDiagnosticSemantics(Protocol):
    """Optional Gaussian model-invocation capability for diagnostics."""

    @property
    def prediction_type(self) -> PredictionType:
        """Return the model prediction parameterization used for training."""

        ...

    @property
    def variance_mode(self) -> VarianceMode:
        """Return the raw Gaussian model-output variance layout."""

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
        variance: GaussianVarianceConfig | None = None,
        loss_weighting: GaussianLossWeightingConfig | None = None,
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
        self.variance = variance or GaussianVarianceConfig()
        self.loss_weighting = loss_weighting or GaussianLossWeightingConfig()
        if (
            self.loss_weighting.name == "p2"
            and self.prediction_type != "epsilon"
        ):
            raise ValueError("P2 weighting requires epsilon prediction")
        if self.variance.mode == "learned_range" and not isinstance(
            self.process,
            LearnedRangeGaussianVarianceProcess,
        ):
            raise TypeError(
                "learned_range variance requires "
                "LearnedRangeGaussianVarianceProcess capability"
            )
        validate_per_sample_objective(
            self.objective,
            variance=self.variance,
            loss_weighting=self.loss_weighting,
            path="Gaussian training policy",
        )

    @property
    def prediction_type(self) -> PredictionType:
        """Return the configured Gaussian model parameterization."""

        return self._prediction_type

    @property
    def variance_mode(self) -> VarianceMode:
        """Return the configured Gaussian model-output variance layout."""

        return self.variance.mode

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

    def extract_reference_images(self, batch: Batch) -> torch.Tensor:
        """Extract clean images using this strategy's batch semantics."""

        return _clean_samples(batch)

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
        model_times = state_times - self.process.clean_time - 1
        raw_model_output = self.predict_gaussian_model(noisy, model_times)
        computation = compute_gaussian_training_loss(
            objective=self.objective,
            process=self.process,
            clean=clean,
            noisy=noisy,
            noise=noise,
            state_times=state_times,
            raw_model_output=raw_model_output,
            prediction_type=self.prediction_type,
            variance=self.variance,
            loss_weighting=self.loss_weighting,
        )
        prediction = computation.prediction
        diagnostics: dict[str, Any] = {
            "timesteps": state_times.detach(),
            "predicted_noise": prediction.epsilon.detach(),
            "target_noise": noise.detach(),
            "predicted_clean": prediction.clean.detach(),
            "clean_samples": clean.detach(),
        }
        diagnostics.update(gaussian_loss_diagnostics(computation))
        return TrainStepOutput(loss=computation.loss, diagnostics=diagnostics)


@REGISTRIES.training_builders.register("gaussian_denoising")
class GaussianDenoisingTrainingBuilder(TrainingBuilder):
    """Assemble the built-in discrete Gaussian training strategy."""

    def build(self) -> TrainingPlan:
        """Validate family assets and return a Gaussian training plan."""

        params = dict(self.context.params)
        prediction_type = params.pop("prediction_type", "epsilon")
        variance = parse_gaussian_variance(
            params.pop("variance", None),
            path="training.params.variance",
        )
        loss_weighting = parse_gaussian_loss_weighting(
            params.pop("loss_weighting", None),
            path="training.params.loss_weighting",
        )
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
        if loss_weighting.name == "p2" and prediction_type != "epsilon":
            raise ValueError(
                "training.params.loss_weighting p2 requires "
                "training.params.prediction_type epsilon"
            )
        if variance.mode == "learned_range" and not isinstance(
            process,
            LearnedRangeGaussianVarianceProcess,
        ):
            raise TypeError(
                "training.params.variance learned_range requires "
                "LearnedRangeGaussianVarianceProcess capability"
            )
        validate_gaussian_model_output_layout(
            self.context.primary_model,
            variance_mode=variance.mode,
            path="gaussian_denoising primary model",
        )
        validate_per_sample_objective(
            objective,
            variance=variance,
            loss_weighting=loss_weighting,
            path="gaussian_denoising training policy",
        )
        strategy = GaussianDenoisingTrainingStrategy(
            self.context.primary_model,
            process,
            objective,
            prediction_type=cast(PredictionType, prediction_type),
            variance=variance,
            loss_weighting=loss_weighting,
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=process,
            objective=objective,
            inference_recipe=SamplingRecipe(
                name="standard_denoising",
                contract={
                    "prediction_type": prediction_type,
                    "variance": {"mode": variance.mode},
                },
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


__all__ = [
    "GaussianDenoisingTrainingBuilder",
    "GaussianDenoisingTrainingStrategy",
    "GaussianDiagnosticSemantics",
    "gaussian_training_target",
]
