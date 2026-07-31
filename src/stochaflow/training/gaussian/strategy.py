"""Template implementation shared by concrete Gaussian training strategies."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from stochaflow.families.gaussian import (
    PredictionType,
    normalize_gaussian_prediction,
)
from stochaflow.metrics.contracts import MetricUpdate
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
    LearnedRangeGaussianVarianceProcess,
)
from stochaflow.training.objectives import (
    MSEObjective,
    PerSampleObjective,
    validate_per_sample_loss,
    validate_reduced_loss,
)
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput

from .contracts import VarianceMode
from .loss import (
    GaussianLossComputation,
    gaussian_loss_diagnostics,
    gaussian_signal_to_noise_ratio,
    gaussian_training_target,
    split_gaussian_training_output,
    validate_gaussian_timestep_weights,
    validate_scalar_objective_loss,
)
from .variance import (
    GaussianVarianceConfig,
    learned_range_variational_bound,
    validate_gaussian_model_output_layout,
)


class GaussianTrainingStrategyBase(TrainingStrategy):
    """Own the common Gaussian step while subclasses own task semantics."""

    def __init__(
        self,
        model: nn.Module,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        *,
        prediction_type: PredictionType,
        variance: GaussianVarianceConfig | None = None,
    ) -> None:
        model_value = cast(object, model)
        if not isinstance(model_value, nn.Module):
            raise TypeError("Gaussian training model must be an nn.Module")
        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "Gaussian training requires DiscreteGaussianDenoisingProcess"
            )
        if process_value.terminal_time <= process_value.clean_time:
            raise ValueError(
                "Gaussian training requires a non-empty noisy time range"
            )
        objective_value = cast(object, objective)
        if not isinstance(objective_value, nn.Module):
            raise TypeError("Gaussian training objective must be an nn.Module")
        if prediction_type not in ("epsilon", "x0", "v", "score"):
            raise ValueError(
                "Gaussian prediction_type must be epsilon, x0, v, or score"
            )
        variance_value = cast(object, variance or GaussianVarianceConfig())
        if not isinstance(variance_value, GaussianVarianceConfig):
            raise TypeError(
                "Gaussian training variance must be GaussianVarianceConfig"
            )
        if variance_value.mode == "learned_range":
            if not isinstance(process_value, LearnedRangeGaussianVarianceProcess):
                raise TypeError(
                    "learned_range variance requires "
                    "LearnedRangeGaussianVarianceProcess capability"
                )
            if not isinstance(objective_value, MSEObjective):
                raise TypeError(
                    "learned_range Gaussian training requires MSEObjective"
                )
        validate_gaussian_model_output_layout(
            model_value,
            variance_mode=variance_value.mode,
            path="Gaussian training model",
        )
        self.model = model_value
        self.process = process_value
        self.objective = objective_value
        self._prediction_type: PredictionType = cast(
            PredictionType,
            prediction_type,
        )
        self.variance = variance_value

    @property
    def prediction_type(self) -> PredictionType:
        """Return the configured Gaussian model parameterization."""

        return self._prediction_type

    @property
    def variance_mode(self) -> VarianceMode:
        """Return the configured Gaussian model-output variance layout."""

        return self.variance.mode

    @property
    def metric_channels(self) -> frozenset[str]:
        """Return Gaussian prediction and clean-reconstruction channels."""

        return frozenset(
            (
                "gaussian.prediction_target",
                "gaussian.clean_reconstruction",
            )
        )

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Compute one step with task-specific training-time batch policy."""

        return self._step(batch, apply_training_policy=True)

    def evaluation_step(self, batch: Any) -> TrainStepOutput:
        """Compute one step without training-only batch randomization."""

        return self._step(batch, apply_training_policy=False)

    @abstractmethod
    def _prepare_batch(
        self,
        batch: Any,
        *,
        apply_training_policy: bool,
    ) -> tuple[torch.Tensor, object, Mapping[str, Any]]:
        """Return clean samples, model context, and task diagnostics."""

    @abstractmethod
    def _predict_training_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        model_context: object,
    ) -> torch.Tensor:
        """Invoke one task-specific model signature."""

    def _timestep_loss_weights(self, snr: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(snr)

    def _step(
        self,
        batch: Any,
        *,
        apply_training_policy: bool,
    ) -> TrainStepOutput:
        clean, model_context, task_diagnostics = self._prepare_batch(
            batch,
            apply_training_policy=apply_training_policy,
        )
        state_times = torch.randint(
            self.process.clean_time + 1,
            self.process.terminal_time + 1,
            (clean.shape[0],),
            device=clean.device,
        )
        noisy, noise = self.process.sample_marginal(clean, state_times)
        model_times = state_times - self.process.clean_time - 1
        raw_model_output = self._predict_training_model(
            noisy,
            model_times,
            model_context,
        )
        computation = self._compute_loss(
            clean=clean,
            noisy=noisy,
            noise=noise,
            state_times=state_times,
            raw_model_output=raw_model_output,
        )
        prediction = computation.prediction
        diagnostics: dict[str, Any] = {
            "timesteps": state_times.detach(),
            "predicted_noise": prediction.epsilon.detach(),
            "target_noise": noise.detach(),
            "predicted_clean": prediction.clean.detach(),
            "clean_samples": clean.detach(),
            **task_diagnostics,
        }
        diagnostics.update(gaussian_loss_diagnostics(computation))
        return TrainStepOutput(
            loss=computation.loss,
            diagnostics=diagnostics,
            metric_updates={
                "gaussian.prediction_target": MetricUpdate(
                    args=(
                        prediction.model_output.contiguous(),
                        computation.target.contiguous(),
                    )
                ),
                "gaussian.clean_reconstruction": MetricUpdate(
                    args=(prediction.clean, clean)
                ),
            },
            loss_aggregation_weight=clean.shape[0],
        )

    def _compute_loss(
        self,
        *,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        noise: torch.Tensor,
        state_times: torch.Tensor,
        raw_model_output: object,
    ) -> GaussianLossComputation:
        mean_output, variance_values = split_gaussian_training_output(
            raw_model_output,
            state=noisy,
            variance_mode=self.variance.mode,
        )
        scales = self.process.marginal_scales(state_times, noisy.size())
        prediction = normalize_gaussian_prediction(
            noisy,
            mean_output,
            signal_scale=scales.signal,
            noise_scale=scales.noise,
            prediction_type=self.prediction_type,
        )
        target = gaussian_training_target(
            self.process,
            clean=clean,
            noise=noise,
            state_times=state_times,
            prediction_type=self.prediction_type,
        )
        snr = gaussian_signal_to_noise_ratio(self.process, state_times)
        weights = validate_gaussian_timestep_weights(
            self._timestep_loss_weights(snr),
            snr=snr,
        )
        objective = self.objective
        if not isinstance(objective, MSEObjective):
            if variance_values is not None:
                raise TypeError(
                    "learned-range Gaussian training requires MSEObjective"
                )
            loss = validate_scalar_objective_loss(
                objective(prediction.model_output, target),
                prediction=prediction.model_output,
            )
            per_sample: torch.Tensor | None = None
            if isinstance(objective, PerSampleObjective):
                per_sample = validate_per_sample_loss(
                    objective.per_sample_loss(prediction.model_output, target),
                    prediction=prediction.model_output,
                )
            return GaussianLossComputation(
                loss=loss,
                prediction=prediction,
                target=target,
                snr=snr,
                timestep_loss_weight=weights,
                per_sample_simple_loss=per_sample,
                per_sample_weighted_simple_loss=per_sample,
                per_sample_variational_bound=None,
                per_sample_loss=per_sample,
            )

        per_sample_simple = validate_per_sample_loss(
            objective.per_sample_loss(prediction.model_output, target),
            prediction=prediction.model_output,
        )
        weighted_simple = weights.to(dtype=per_sample_simple.dtype) * per_sample_simple
        variational_bound: torch.Tensor | None = None
        per_sample_loss = weighted_simple
        if variance_values is not None:
            variational_bound = validate_per_sample_loss(
                learned_range_variational_bound(
                    self.process,
                    clean=clean,
                    noisy=noisy,
                    state_times=state_times,
                    mean_output=mean_output,
                    variance_values=variance_values,
                    prediction_type=self.prediction_type,
                ),
                prediction=prediction.model_output,
            ).to(dtype=per_sample_simple.dtype)
            per_sample_loss = weighted_simple + variational_bound
        loss = validate_reduced_loss(
            objective.reduce_per_sample_loss(per_sample_loss),
            per_sample_loss=per_sample_loss,
        )
        return GaussianLossComputation(
            loss=loss,
            prediction=prediction,
            target=target,
            snr=snr,
            timestep_loss_weight=weights,
            per_sample_simple_loss=per_sample_simple,
            per_sample_weighted_simple_loss=weighted_simple,
            per_sample_variational_bound=variational_bound,
            per_sample_loss=per_sample_loss,
        )


__all__: list[str] = []
