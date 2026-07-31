"""Concrete unconditional and class-conditional P2 training recipes."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from stochaflow.models.conditioning import ClassConditionalDenoiser
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
)
from stochaflow.training.builder import TrainingPlan
from stochaflow.training.objectives import MSEObjective, validate_per_sample_loss
from stochaflow.utils.registry import REGISTRIES

from .class_conditional import (
    ClassConditionalGaussianDenoisingTrainingBuilder,
    ClassConditionalGaussianDenoisingTrainingStrategy,
    _condition_dropout,
)
from .loss import GaussianLossComputation
from .strategy import GaussianTrainingStrategyBase, _reduce_mse_per_sample
from .unconditional import (
    GaussianDenoisingTrainingBuilder,
    GaussianDenoisingTrainingStrategy,
)
from .variance import (
    GaussianVarianceConfig,
    learned_range_variational_bound,
    parse_gaussian_variance,
)


class P2GaussianDenoisingTrainingStrategy(GaussianDenoisingTrainingStrategy):
    """Train an unconditional epsilon predictor with P2 weighting."""

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
        self.k, self.gamma = _validate_p2_parameters(k, gamma)
        super().__init__(
            model,
            process,
            objective_value,
            prediction_type="epsilon",
            variance=variance,
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
        if self.gamma == 0.0:
            return super()._compute_loss(
                clean=clean,
                noisy=noisy,
                noise=noise,
                state_times=state_times,
                raw_model_output=raw_model_output,
            )
        return _compute_p2_loss(
            self,
            clean=clean,
            noisy=noisy,
            noise=noise,
            state_times=state_times,
            raw_model_output=raw_model_output,
            k=self.k,
            gamma=self.gamma,
        )


class ClassConditionalP2GaussianDenoisingTrainingStrategy(
    ClassConditionalGaussianDenoisingTrainingStrategy
):
    """Train a class-conditional epsilon predictor with P2 weighting."""

    def __init__(
        self,
        model: ClassConditionalDenoiser,
        process: DiscreteGaussianDenoisingProcess,
        objective: MSEObjective,
        *,
        variance: GaussianVarianceConfig | None = None,
        condition_dropout: float = 0.0,
        k: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        objective_value = cast(object, objective)
        if not isinstance(objective_value, MSEObjective):
            raise TypeError("class-conditional P2 training requires MSEObjective")
        self.k, self.gamma = _validate_p2_parameters(k, gamma)
        super().__init__(
            model,
            process,
            objective_value,
            prediction_type="epsilon",
            variance=variance,
            condition_dropout=condition_dropout,
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
        if self.gamma == 0.0:
            return super()._compute_loss(
                clean=clean,
                noisy=noisy,
                noise=noise,
                state_times=state_times,
                raw_model_output=raw_model_output,
            )
        return _compute_p2_loss(
            self,
            clean=clean,
            noisy=noisy,
            noise=noise,
            state_times=state_times,
            raw_model_output=raw_model_output,
            k=self.k,
            gamma=self.gamma,
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
        k, gamma = _validate_p2_parameters(
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


@REGISTRIES.training_builders.register(
    "class_conditional_p2_gaussian_denoising"
)
class ClassConditionalP2GaussianDenoisingTrainingBuilder(
    ClassConditionalGaussianDenoisingTrainingBuilder
):
    """Assemble class-conditional epsilon prediction with P2 weighting."""

    def build(self) -> TrainingPlan:
        """Validate the P2 recipe and return its concrete strategy."""

        params = dict(self.context.params)
        condition_dropout = _condition_dropout(
            params.pop("condition_dropout", 0.0),
            path="training.params.condition_dropout",
        )
        variance = parse_gaussian_variance(
            params.pop("variance", None),
            path="training.params.variance",
        )
        k, gamma = _validate_p2_parameters(
            params.pop("k", 1.0),
            params.pop("gamma", 1.0),
        )
        _reject_unknown(
            params,
            builder="class_conditional_p2_gaussian_denoising",
        )
        model, process, objective = self._components()
        if not isinstance(objective, MSEObjective):
            raise TypeError(
                "class_conditional_p2_gaussian_denoising requires MSEObjective"
            )
        strategy = ClassConditionalP2GaussianDenoisingTrainingStrategy(
            model,
            process,
            objective,
            variance=variance,
            condition_dropout=condition_dropout,
            k=k,
            gamma=gamma,
        )
        return self._plan(
            strategy,
            model=cast(nn.Module, model),
            process=process,
            objective=objective,
            prediction_type="epsilon",
            variance=variance,
        )


def _compute_p2_loss(
    strategy: GaussianTrainingStrategyBase,
    *,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    raw_model_output: object,
    k: float,
    gamma: float,
) -> GaussianLossComputation:
    mean_output, variance_values, prediction, target = (
        strategy._prediction_and_target(
            clean=clean,
            noisy=noisy,
            noise=noise,
            state_times=state_times,
            raw_model_output=raw_model_output,
        )
    )
    objective = strategy.objective
    if not isinstance(objective, MSEObjective):
        raise TypeError("P2 Gaussian training requires MSEObjective")
    per_sample_simple = validate_per_sample_loss(
        objective.per_sample_loss(prediction.model_output, target),
        prediction=prediction.model_output,
    )
    snr = _gaussian_signal_to_noise_ratio(strategy.process, state_times)
    weights = _validate_gaussian_timestep_weights(
        _p2_timestep_loss_weights(snr, k=k, gamma=gamma),
        snr=snr,
    ).to(dtype=per_sample_simple.dtype)
    per_sample_loss = weights * per_sample_simple
    if variance_values is not None:
        variational_bound = validate_per_sample_loss(
            learned_range_variational_bound(
                strategy.process,
                clean=clean,
                noisy=noisy,
                state_times=state_times,
                mean_output=mean_output,
                variance_values=variance_values,
                prediction_type=strategy.prediction_type,
            ),
            prediction=prediction.model_output,
        ).to(dtype=per_sample_simple.dtype)
        per_sample_loss = per_sample_loss + variational_bound
    return GaussianLossComputation(
        loss=_reduce_mse_per_sample(objective, per_sample_loss),
        prediction=prediction,
        target=target,
        per_sample_loss=per_sample_loss,
    )


def _validate_p2_parameters(k: object, gamma: object) -> tuple[float, float]:
    validated_k = _finite_number(k, path="P2 k")
    validated_gamma = _finite_number(gamma, path="P2 gamma")
    if validated_k <= 0.0:
        raise ValueError("P2 k must be greater than zero")
    if validated_gamma < 0.0:
        raise ValueError("P2 gamma must be non-negative")
    return validated_k, validated_gamma


def _p2_timestep_loss_weights(
    signal_to_noise_ratio: torch.Tensor,
    *,
    k: float,
    gamma: float,
) -> torch.Tensor:
    return (k + signal_to_noise_ratio).pow(-gamma)


def _gaussian_signal_to_noise_ratio(
    process: DiscreteGaussianDenoisingProcess,
    state_times: torch.Tensor,
) -> torch.Tensor:
    state_times = process.validate_noisy_state_times(state_times)
    scales = process.marginal_scales(state_times, state_times.size())
    return scales.signal.square() / scales.noise.square()


def _validate_gaussian_timestep_weights(
    value: object,
    *,
    snr: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian timestep loss weights must be a Tensor")
    if value.ndim != 1 or value.shape != snr.shape:
        raise ValueError("Gaussian timestep loss weights must have shape [B]")
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian timestep loss weights must be floating-point")
    if value.device != snr.device:
        raise ValueError("Gaussian timestep loss weights must share the SNR device")
    if value.dtype != snr.dtype:
        raise ValueError("Gaussian timestep loss weights must share the SNR dtype")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Gaussian timestep loss weights must be finite")
    if bool(torch.any(value < 0)):
        raise ValueError("Gaussian timestep loss weights must be non-negative")
    return value


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _reject_unknown(params: dict[str, Any], *, builder: str) -> None:
    if params:
        unknown = ", ".join(sorted(params))
        raise ValueError(f"unknown {builder} training parameter(s): {unknown}")


__all__ = [
    "ClassConditionalP2GaussianDenoisingTrainingBuilder",
    "ClassConditionalP2GaussianDenoisingTrainingStrategy",
    "P2GaussianDenoisingTrainingBuilder",
    "P2GaussianDenoisingTrainingStrategy",
]
