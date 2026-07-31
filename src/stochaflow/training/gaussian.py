"""Gaussian-family training strategy and registered builder."""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from stochaflow.families.gaussian import PredictionType, VarianceMode
from stochaflow.metrics import MetricUpdate
from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.training.gaussian_loss import (
    GaussianLossComposer,
    build_gaussian_loss_composer,
    gaussian_loss_diagnostics,
    gaussian_training_target,
    prepare_gaussian_loss_inputs,
)
from stochaflow.training.gaussian_variance import (
    parse_gaussian_variance,
    validate_gaussian_model_output_layout,
)
from stochaflow.training.gaussian_weighting import (
    build_gaussian_simple_loss_weighting,
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
        loss_composer: GaussianLossComposer,
    ) -> None:
        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "Gaussian training requires DiscreteGaussianDenoisingProcess"
            )
        if process_value.terminal_time <= process_value.clean_time:
            raise ValueError("Gaussian training requires a non-empty noisy time range")
        composer_value = cast(object, loss_composer)
        if not isinstance(composer_value, GaussianLossComposer):
            raise TypeError("Gaussian training requires GaussianLossComposer")
        if (
            composer_value.bound_variance_process is not None
            and composer_value.bound_variance_process is not process_value
        ):
            raise ValueError(
                "Gaussian training loss composer is bound to a different Process"
            )
        self.model = model
        self.process = process_value
        self.loss_composer = composer_value

    @property
    def prediction_type(self) -> PredictionType:
        """Return the configured Gaussian model parameterization."""

        return self.loss_composer.prediction_type

    @property
    def variance_mode(self) -> VarianceMode:
        """Return the configured Gaussian model-output variance layout."""

        return self.loss_composer.variance.mode

    @property
    def metric_channels(self) -> frozenset[str]:
        """Return Gaussian prediction and clean-reconstruction channels."""

        return frozenset(
            (
                "gaussian.prediction_target",
                "gaussian.clean_reconstruction",
            )
        )

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
        inputs = prepare_gaussian_loss_inputs(
            self.process,
            clean=clean,
            noisy=noisy,
            noise=noise,
            state_times=state_times,
            raw_model_output=raw_model_output,
        )
        computation = self.loss_composer.compute(inputs)
        prediction = computation.prediction
        diagnostics: dict[str, Any] = {
            "timesteps": state_times.detach(),
            "predicted_noise": prediction.epsilon.detach(),
            "target_noise": noise.detach(),
            "predicted_clean": prediction.clean.detach(),
            "clean_samples": clean.detach(),
        }
        diagnostics.update(gaussian_loss_diagnostics(computation))
        return TrainStepOutput(
            loss=computation.loss,
            diagnostics=diagnostics,
            metric_updates={
                "gaussian.prediction_target": MetricUpdate(
                    args=(
                        computation.prediction.model_output.contiguous(),
                        computation.target.contiguous(),
                    )
                ),
                "gaussian.clean_reconstruction": MetricUpdate(
                    args=(computation.prediction.clean, clean)
                ),
            },
            loss_aggregation_weight=clean.shape[0],
        )


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
        loss_weighting = build_gaussian_simple_loss_weighting(
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
        validate_gaussian_model_output_layout(
            self.context.primary_model,
            variance_mode=variance.mode,
            path="gaussian_denoising primary model",
        )
        composer = build_gaussian_loss_composer(
            objective=objective,
            process=process,
            prediction_type=cast(PredictionType, prediction_type),
            variance=variance,
            loss_weighting=loss_weighting,
            path="gaussian_denoising training policy",
        )
        strategy = GaussianDenoisingTrainingStrategy(
            self.context.primary_model,
            process,
            composer,
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
