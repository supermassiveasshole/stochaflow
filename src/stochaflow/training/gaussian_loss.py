"""Gaussian-family training loss composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from stochaflow.families import gaussian as gaussian_family
from stochaflow.families.gaussian import (
    GaussianPrediction,
    PredictionType,
    normalize_gaussian_prediction,
    split_gaussian_model_output,
)
from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.training.gaussian_variance import (
    GaussianVarianceConfig,
    GaussianVarianceLoss,
    build_gaussian_variance_loss,
    learned_range_log_variance,
    parse_gaussian_variance,
    validate_gaussian_model_output_layout,
)
from stochaflow.training.gaussian_weighting import (
    GaussianSimpleLossContext,
    GaussianSimpleLossWeighting,
    compute_gaussian_simple_loss_weights,
)
from stochaflow.training.objectives import (
    BatchReduciblePerSampleObjective,
    PerSampleObjective,
    validate_per_sample_loss,
    validate_reduced_loss,
)


@dataclass(frozen=True, slots=True)
class GaussianLossInputs:
    """Process-prepared tensors consumed by one Gaussian loss computation."""

    clean: torch.Tensor
    noisy: torch.Tensor
    noise: torch.Tensor
    state_times: torch.Tensor
    raw_model_output: object
    signal_scale: torch.Tensor
    noise_scale: torch.Tensor
    signal_to_noise_ratio: torch.Tensor


@dataclass(frozen=True, slots=True)
class GaussianLossComputation:
    """One normalized prediction and its batch-aligned loss components."""

    loss: torch.Tensor
    prediction: GaussianPrediction
    target: torch.Tensor
    snr: torch.Tensor
    timestep_loss_weight: torch.Tensor
    per_sample_simple_loss: torch.Tensor | None
    per_sample_weighted_simple_loss: torch.Tensor | None
    per_sample_variational_bound: torch.Tensor | None
    per_sample_loss: torch.Tensor | None


class GaussianLossComposer:
    """Compose a Gaussian Objective, simple-loss weighting, and variance term."""

    def __init__(
        self,
        objective: nn.Module,
        *,
        prediction_type: PredictionType,
        variance: GaussianVarianceConfig,
        loss_weighting: GaussianSimpleLossWeighting,
        variance_loss: GaussianVarianceLoss | None = None,
        path: str = "Gaussian training policy",
    ) -> None:
        objective_value = cast(object, objective)
        if not isinstance(objective_value, nn.Module):
            raise TypeError(f"{path} objective must be an nn.Module")
        prediction_type_value = cast(object, prediction_type)
        if not isinstance(prediction_type_value, str) or prediction_type_value not in (
            "epsilon",
            "x0",
            "v",
            "score",
        ):
            raise ValueError(
                f"{path} prediction_type must be epsilon, x0, v, or score"
            )
        validated_prediction_type = cast(PredictionType, prediction_type_value)
        loss_weighting_value = cast(object, loss_weighting)
        if not isinstance(loss_weighting_value, GaussianSimpleLossWeighting):
            raise TypeError(
                f"{path} loss weighting must inherit "
                "GaussianSimpleLossWeighting"
            )
        loss_weighting_value.validate_contract(
            prediction_type=validated_prediction_type
        )
        variance_value = cast(object, variance)
        if not isinstance(variance_value, GaussianVarianceConfig):
            raise TypeError(f"{path} variance must be GaussianVarianceConfig")
        variance_loss_value = cast(object, variance_loss)
        if variance_loss_value is not None and not isinstance(
            variance_loss_value,
            GaussianVarianceLoss,
        ):
            raise TypeError(
                f"{path} variance loss must satisfy GaussianVarianceLoss"
            )
        validated_variance_loss = cast(
            GaussianVarianceLoss | None,
            variance_loss_value,
        )
        if variance.mode == "fixed" and validated_variance_loss is not None:
            raise ValueError(f"{path} fixed variance cannot define a variance loss")
        if variance.mode == "learned_range" and validated_variance_loss is None:
            raise ValueError(
                f"{path} learned_range variance requires a variance loss"
            )
        needs_reducer = (
            loss_weighting_value.requires_per_sample_loss
            or validated_variance_loss is not None
        )
        if needs_reducer and not isinstance(
            objective,
            BatchReduciblePerSampleObjective,
        ):
            raise TypeError(
                f"{path} requires objective satisfying "
                "BatchReduciblePerSampleObjective"
            )
        self.objective = objective_value
        self.prediction_type: PredictionType = validated_prediction_type
        self.variance = variance
        self.loss_weighting = loss_weighting_value
        self.variance_loss = validated_variance_loss

    @property
    def bound_variance_process(
        self,
    ) -> DiscreteGaussianDenoisingProcess | None:
        """Return the Process bound to the optional variance collaborator."""

        if self.variance_loss is None:
            return None
        return self.variance_loss.bound_process

    @property
    def requires_per_sample_objective(self) -> bool:
        """Return whether this composition requires explicit batch reduction."""

        return (
            self.loss_weighting.requires_per_sample_loss
            or self.variance_loss is not None
        )

    def compute(self, inputs: GaussianLossInputs) -> GaussianLossComputation:
        """Compute one loss without depending on a Process or model signature."""

        mean_output, variance_values = split_gaussian_model_output(
            inputs.raw_model_output,
            state=inputs.noisy,
            variance_mode=self.variance.mode,
        )
        prediction = normalize_gaussian_prediction(
            inputs.noisy,
            mean_output,
            signal_scale=inputs.signal_scale,
            noise_scale=inputs.noise_scale,
            prediction_type=self.prediction_type,
        )
        target = gaussian_family.gaussian_training_target(
            clean=inputs.clean,
            noise=inputs.noise,
            signal_scale=inputs.signal_scale,
            noise_scale=inputs.noise_scale,
            prediction_type=self.prediction_type,
        )
        context = GaussianSimpleLossContext(
            prediction_type=self.prediction_type,
            signal_to_noise_ratio=inputs.signal_to_noise_ratio,
        )
        if context.signal_to_noise_ratio.shape[0] != prediction.model_output.shape[0]:
            raise ValueError(
                "Gaussian simple-loss SNR must match the prediction batch"
            )
        if context.signal_to_noise_ratio.device != prediction.model_output.device:
            raise ValueError(
                "Gaussian simple-loss SNR must share the prediction device"
            )
        weights = compute_gaussian_simple_loss_weights(
            self.loss_weighting,
            context,
        )

        objective = self.objective
        if not isinstance(objective, BatchReduciblePerSampleObjective):
            if not torch.equal(weights, torch.ones_like(weights)):
                raise ValueError(
                    "Gaussian simple-loss policies that produce non-identity "
                    "weights must declare requires_per_sample_loss=True"
                )
            loss = _compute_scalar_objective(
                objective,
                prediction.model_output,
                target,
            )
            per_sample_simple: torch.Tensor | None = None
            if isinstance(objective, PerSampleObjective):
                per_sample_simple = validate_per_sample_loss(
                    objective.per_sample_loss(prediction.model_output, target),
                    prediction=prediction.model_output,
                )
            return GaussianLossComputation(
                loss=loss,
                prediction=prediction,
                target=target,
                snr=inputs.signal_to_noise_ratio,
                timestep_loss_weight=weights,
                per_sample_simple_loss=per_sample_simple,
                per_sample_weighted_simple_loss=per_sample_simple,
                per_sample_variational_bound=None,
                per_sample_loss=per_sample_simple,
            )

        per_sample_simple = validate_per_sample_loss(
            objective.per_sample_loss(prediction.model_output, target),
            prediction=prediction.model_output,
        )
        weighted_simple = weights.to(dtype=per_sample_simple.dtype) * per_sample_simple
        variational_bound: torch.Tensor | None = None
        per_sample_loss = weighted_simple
        if variance_values is not None:
            variance_loss = self.variance_loss
            if variance_loss is None:
                raise RuntimeError(
                    "learned_range variance composition is missing its loss"
                )
            variational_bound = validate_per_sample_loss(
                variance_loss.per_sample_loss(
                    clean=inputs.clean,
                    noisy=inputs.noisy,
                    state_times=inputs.state_times,
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
            snr=inputs.signal_to_noise_ratio,
            timestep_loss_weight=weights,
            per_sample_simple_loss=per_sample_simple,
            per_sample_weighted_simple_loss=weighted_simple,
            per_sample_variational_bound=variational_bound,
            per_sample_loss=per_sample_loss,
        )


def build_gaussian_loss_composer(
    *,
    objective: nn.Module,
    process: DiscreteGaussianDenoisingProcess,
    prediction_type: PredictionType,
    variance: GaussianVarianceConfig,
    loss_weighting: GaussianSimpleLossWeighting,
    path: str,
) -> GaussianLossComposer:
    """Validate one complete Gaussian training-loss composition."""

    variance_loss = build_gaussian_variance_loss(process, variance)
    return GaussianLossComposer(
        objective,
        prediction_type=prediction_type,
        variance=variance,
        loss_weighting=loss_weighting,
        variance_loss=variance_loss,
        path=path,
    )


def prepare_gaussian_loss_inputs(
    process: DiscreteGaussianDenoisingProcess,
    *,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    raw_model_output: object,
) -> GaussianLossInputs:
    """Prepare Process-owned marginal facts for a process-free Composer."""

    state_times = process.validate_noisy_state_times(state_times)
    state_scales = process.marginal_scales(state_times, noisy.size())
    batch_scales = process.marginal_scales(state_times, state_times.size())
    snr = gaussian_family.gaussian_signal_to_noise_ratio(
        signal_scale=batch_scales.signal,
        noise_scale=batch_scales.noise,
    )
    return GaussianLossInputs(
        clean=clean,
        noisy=noisy,
        noise=noise,
        state_times=state_times,
        raw_model_output=raw_model_output,
        signal_scale=state_scales.signal,
        noise_scale=state_scales.noise,
        signal_to_noise_ratio=snr,
    )


def gaussian_loss_diagnostics(
    computation: GaussianLossComputation,
) -> dict[str, torch.Tensor]:
    """Detach standardized Gaussian loss-policy diagnostics."""

    diagnostics = {
        "snr": computation.snr.detach(),
        "timestep_loss_weight": computation.timestep_loss_weight.detach(),
    }
    for name in (
        "per_sample_simple_loss",
        "per_sample_weighted_simple_loss",
        "per_sample_variational_bound",
        "per_sample_loss",
    ):
        value = getattr(computation, name)
        if value is not None:
            diagnostics[name] = value.detach()
    return diagnostics


def gaussian_signal_to_noise_ratio(
    process: DiscreteGaussianDenoisingProcess,
    state_times: torch.Tensor,
) -> torch.Tensor:
    """Return cumulative VP signal-to-noise ratios at public state times."""

    process = _validate_process(process)
    state_times = process.validate_noisy_state_times(state_times)
    scales = process.marginal_scales(state_times, state_times.size())
    return gaussian_family.gaussian_signal_to_noise_ratio(
        signal_scale=scales.signal,
        noise_scale=scales.noise,
    )


def gaussian_training_target(
    process: DiscreteGaussianDenoisingProcess,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    prediction_type: PredictionType,
) -> torch.Tensor:
    """Build a target through the public Process-facing extension helper."""

    process = _validate_process(process)
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
        raise TypeError(
            "Gaussian training clean state and noise must be floating-point"
        )
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
    scales = process.marginal_scales(state_times, clean.size())
    return gaussian_family.gaussian_training_target(
        clean=clean,
        noise=noise,
        signal_scale=scales.signal,
        noise_scale=scales.noise,
        prediction_type=prediction_type,
    )


def _compute_scalar_objective(
    objective: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    value: object = objective(prediction, target)
    if not isinstance(value, torch.Tensor):
        raise TypeError("training objective must return a Tensor")
    if not torch.is_floating_point(value):
        raise TypeError("training objective must return a floating-point Tensor")
    if value.ndim != 0:
        raise ValueError("training objective must return a scalar Tensor")
    if value.device != prediction.device:
        raise ValueError("training objective loss must be on the prediction device")
    return value


def _validate_process(
    value: object,
) -> DiscreteGaussianDenoisingProcess:
    if not isinstance(value, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "Gaussian training requires DiscreteGaussianDenoisingProcess"
        )
    return cast(DiscreteGaussianDenoisingProcess, value)


__all__ = [
    "GaussianLossComposer",
    "GaussianLossComputation",
    "GaussianLossInputs",
    "GaussianVarianceConfig",
    "build_gaussian_loss_composer",
    "gaussian_loss_diagnostics",
    "gaussian_signal_to_noise_ratio",
    "gaussian_training_target",
    "learned_range_log_variance",
    "parse_gaussian_variance",
    "prepare_gaussian_loss_inputs",
    "validate_gaussian_model_output_layout",
]
