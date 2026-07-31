"""Unconditional standard and P2 Gaussian training strategies."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from stochaflow.families.gaussian import PredictionType
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
)
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.training.objectives import MSEObjective
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.sampling_recipe import SamplingRecipe

from .p2 import p2_timestep_loss_weights, validate_p2_parameters
from .strategy import GaussianTrainingStrategyBase
from .variance import GaussianVarianceConfig, parse_gaussian_variance


class GaussianDenoisingTrainingStrategy(GaussianTrainingStrategyBase):
    """Train an unconditional model against one Gaussian marginal target."""

    def _prepare_batch(
        self,
        batch: Any,
        *,
        apply_training_policy: bool,
    ) -> tuple[torch.Tensor, object, dict[str, Any]]:
        del apply_training_policy
        return _clean_samples(batch), None, {}

    def _predict_training_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        model_context: object,
    ) -> torch.Tensor:
        del model_context
        return self.predict_gaussian_model(state, model_time)

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

    def extract_reference_images(self, batch: Any) -> torch.Tensor:
        """Extract clean images using this strategy's batch semantics."""

        return _clean_samples(batch)


class P2GaussianDenoisingTrainingStrategy(GaussianDenoisingTrainingStrategy):
    """Train an epsilon-predicting Gaussian model with P2 weighting."""

    def __init__(
        self,
        model: nn.Module,
        process: DiscreteGaussianDenoisingProcess,
        objective: MSEObjective,
        *,
        variance: GaussianVarianceConfig | None = None,
        k: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        objective_value = cast(object, objective)
        if not isinstance(objective_value, MSEObjective):
            raise TypeError("P2 Gaussian training requires MSEObjective")
        validated_k, validated_gamma = validate_p2_parameters(k, gamma)
        super().__init__(
            model,
            process,
            objective_value,
            prediction_type="epsilon",
            variance=variance,
        )
        self.k = validated_k
        self.gamma = validated_gamma

    def _timestep_loss_weights(self, snr: torch.Tensor) -> torch.Tensor:
        return p2_timestep_loss_weights(snr, k=self.k, gamma=self.gamma)


@REGISTRIES.training_builders.register("gaussian_denoising")
class GaussianDenoisingTrainingBuilder(TrainingBuilder):
    """Assemble the standard unconditional Gaussian training strategy."""

    def build(self) -> TrainingPlan:
        """Validate family assets and return a standard training plan."""

        params = dict(self.context.params)
        prediction_type = _prediction_type(
            params.pop("prediction_type", "epsilon"),
            path="training.params.prediction_type",
        )
        variance = parse_gaussian_variance(
            params.pop("variance", None),
            path="training.params.variance",
        )
        _reject_unknown(params, builder="gaussian_denoising")
        model, process, objective = self._components()
        strategy = GaussianDenoisingTrainingStrategy(
            model,
            process,
            objective,
            prediction_type=prediction_type,
            variance=variance,
        )
        return self._plan(
            strategy,
            model=model,
            process=process,
            objective=objective,
            prediction_type=prediction_type,
            variance=variance,
        )

    def _components(
        self,
    ) -> tuple[nn.Module, DiscreteGaussianDenoisingProcess, nn.Module]:
        model = self.context.primary_model
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "gaussian_denoising training requires "
                "DiscreteGaussianDenoisingProcess"
            )
        objective = self.context.objective
        if objective is None:
            raise TypeError("Gaussian denoising training requires objective")
        return model, process, objective

    def _plan(
        self,
        strategy: GaussianTrainingStrategyBase,
        *,
        model: nn.Module,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        prediction_type: PredictionType,
        variance: GaussianVarianceConfig,
    ) -> TrainingPlan:
        return TrainingPlan(
            strategy=strategy,
            primary_model=model,
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


@REGISTRIES.training_builders.register("p2_gaussian_denoising")
class P2GaussianDenoisingTrainingBuilder(GaussianDenoisingTrainingBuilder):
    """Assemble unconditional epsilon prediction with P2 weighting."""

    def build(self) -> TrainingPlan:
        """Validate the P2 recipe and return its concrete strategy."""

        params = dict(self.context.params)
        variance = parse_gaussian_variance(
            params.pop("variance", None),
            path="training.params.variance",
        )
        k, gamma = validate_p2_parameters(
            params.pop("k", 1.0),
            params.pop("gamma", 1.0),
        )
        _reject_unknown(params, builder="p2_gaussian_denoising")
        model, process, objective = self._components()
        if not isinstance(objective, MSEObjective):
            raise TypeError("p2_gaussian_denoising requires MSEObjective")
        strategy = P2GaussianDenoisingTrainingStrategy(
            model,
            process,
            objective,
            variance=variance,
            k=k,
            gamma=gamma,
        )
        return self._plan(
            strategy,
            model=model,
            process=process,
            objective=objective,
            prediction_type="epsilon",
            variance=variance,
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


def _prediction_type(value: object, *, path: str) -> PredictionType:
    if not isinstance(value, str) or value not in ("epsilon", "x0", "v", "score"):
        raise ValueError(f"{path} must be epsilon, x0, v, or score")
    return cast(PredictionType, value)


def _reject_unknown(params: dict[str, Any], *, builder: str) -> None:
    if params:
        unknown = ", ".join(sorted(params))
        raise ValueError(f"unknown {builder} training parameter(s): {unknown}")


__all__ = [
    "GaussianDenoisingTrainingBuilder",
    "GaussianDenoisingTrainingStrategy",
    "P2GaussianDenoisingTrainingBuilder",
    "P2GaussianDenoisingTrainingStrategy",
]
