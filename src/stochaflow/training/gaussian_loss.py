"""Gaussian-family loss policies shared by built-in training strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn

from stochaflow.models.denoising import DenoiserChannelLayout
from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.processes.gaussian import LearnedRangeGaussianVarianceProcess
from stochaflow.sampling.gaussian import (
    GaussianPrediction,
    PredictionType,
    VarianceMode,
    normalize_gaussian_prediction,
)
from stochaflow.training.objectives import (
    PerSampleObjective,
    compute_objective,
)

LossWeightingName = Literal["constant", "p2"]
VarianceLossName = Literal["rescaled_variational_bound"]


@dataclass(frozen=True, slots=True)
class GaussianLossWeightingConfig:
    """Validated Gaussian-local simple-loss weighting policy."""

    name: LossWeightingName = "constant"
    k: float = 1.0
    gamma: float = 0.0


@dataclass(frozen=True, slots=True)
class GaussianVarianceConfig:
    """Validated Gaussian model-variance and hybrid-loss policy."""

    mode: VarianceMode = "fixed"
    loss: VarianceLossName | None = None


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


def validate_gaussian_model_output_layout(
    model: object,
    *,
    variance_mode: VarianceMode,
    path: str,
) -> None:
    """Preflight a static denoiser channel declaration when one is available."""

    if not isinstance(model, DenoiserChannelLayout):
        return
    in_channels = _positive_channel_count(
        model.in_channels,
        path=f"{path}.in_channels",
    )
    out_channels = _positive_channel_count(
        model.out_channels,
        path=f"{path}.out_channels",
    )
    expected_out_channels = in_channels * (
        2 if variance_mode == "learned_range" else 1
    )
    if out_channels != expected_out_channels:
        raise ValueError(
            f"{path} declares {out_channels} output channels, but "
            f"{variance_mode} Gaussian variance requires "
            f"{expected_out_channels} output channels for "
            f"{in_channels} input channels"
        )


def parse_gaussian_loss_weighting(
    value: object,
    *,
    path: str,
) -> GaussianLossWeightingConfig:
    """Parse a private ``constant`` or ``p2`` Gaussian training policy."""

    if value is None:
        return GaussianLossWeightingConfig()
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    name = value.get("name")
    if name == "constant":
        unknown = sorted(set(value) - {"name"})
        if unknown:
            raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
        return GaussianLossWeightingConfig()
    if name != "p2":
        raise ValueError(f"{path}.name must be constant or p2")
    unknown = sorted(set(value) - {"name", "k", "gamma"})
    if unknown:
        raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
    k = _finite_number(value.get("k", 1.0), path=f"{path}.k")
    gamma = _finite_number(value.get("gamma", 1.0), path=f"{path}.gamma")
    if k <= 0.0:
        raise ValueError(f"{path}.k must be greater than zero")
    if gamma < 0.0:
        raise ValueError(f"{path}.gamma must be non-negative")
    return GaussianLossWeightingConfig(name="p2", k=k, gamma=gamma)


def parse_gaussian_variance(
    value: object,
    *,
    path: str,
) -> GaussianVarianceConfig:
    """Parse fixed or learned-range variance as one Gaussian recipe fact."""

    if value is None:
        return GaussianVarianceConfig()
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    mode = value.get("mode")
    if mode == "fixed":
        unknown = sorted(set(value) - {"mode"})
        if unknown:
            raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
        return GaussianVarianceConfig()
    if mode != "learned_range":
        raise ValueError(f"{path}.mode must be fixed or learned_range")
    unknown = sorted(set(value) - {"mode", "loss"})
    if unknown:
        raise ValueError(f"unknown {path} field(s): " + ", ".join(unknown))
    loss = value.get("loss")
    if loss != "rescaled_variational_bound":
        raise ValueError(
            f"{path}.loss must be rescaled_variational_bound for learned_range"
        )
    return GaussianVarianceConfig(
        mode="learned_range",
        loss="rescaled_variational_bound",
    )


def gaussian_signal_to_noise_ratio(
    process: DiscreteGaussianDenoisingProcess,
    state_times: torch.Tensor,
) -> torch.Tensor:
    """Return cumulative VP signal-to-noise ratios at public state times."""

    state_times = process.validate_noisy_state_times(state_times)
    scales = process.marginal_scales(state_times, state_times.size())
    return scales.signal.square() / scales.noise.square()


def gaussian_timestep_loss_weights(
    process: DiscreteGaussianDenoisingProcess,
    state_times: torch.Tensor,
    config: GaussianLossWeightingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SNR and the exact unnormalized constant/P2 sample weights."""

    snr = gaussian_signal_to_noise_ratio(process, state_times)
    if config.name == "constant" or config.gamma == 0.0:
        return snr, torch.ones_like(snr)
    return snr, (config.k + snr).pow(-config.gamma)


def learned_range_log_variance(
    process: DiscreteGaussianDenoisingProcess,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    variance_values: torch.Tensor,
) -> torch.Tensor:
    """Interpolate selected-pair posterior/beta log-variance endpoints."""

    if not isinstance(process, LearnedRangeGaussianVarianceProcess):
        raise TypeError(
            "learned_range variance requires "
            "LearnedRangeGaussianVarianceProcess capability"
        )
    bounds = process.reverse_log_variance_bounds(
        source_times,
        target_times,
        variance_values.size(),
    )
    fraction = (variance_values + 1.0) / 2.0
    return fraction * bounds.upper + (1.0 - fraction) * bounds.lower


def compute_gaussian_training_loss(
    *,
    objective: nn.Module,
    process: DiscreteGaussianDenoisingProcess,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    noise: torch.Tensor,
    state_times: torch.Tensor,
    raw_model_output: object,
    prediction_type: PredictionType,
    variance: GaussianVarianceConfig,
    loss_weighting: GaussianLossWeightingConfig,
) -> GaussianLossComputation:
    """Compute fixed or learned-range Gaussian training semantics once."""

    state_times = process.validate_noisy_state_times(state_times)
    mean_output, variance_values = _split_model_output(
        raw_model_output,
        state=noisy,
        variance_mode=variance.mode,
    )
    prediction = normalize_gaussian_prediction(
        process,
        noisy,
        state_times,
        mean_output,
        prediction_type=prediction_type,
        clip_denoised=False,
    )
    target = gaussian_training_target(
        process,
        clean=clean,
        noise=noise,
        state_times=state_times,
        prediction_type=prediction_type,
    )
    snr, weights = gaussian_timestep_loss_weights(
        process,
        state_times,
        loss_weighting,
    )
    scalar_simple, per_sample_simple = compute_objective(
        objective,
        prediction.model_output,
        target,
    )
    requires_per_sample = (
        loss_weighting.name != "constant" or variance.mode == "learned_range"
    )
    if requires_per_sample and per_sample_simple is None:
        raise TypeError(
            "P2 weighting and learned_range variance require an objective "
            "satisfying PerSampleObjective"
        )

    if per_sample_simple is None:
        return GaussianLossComputation(
            loss=scalar_simple,
            prediction=prediction,
            target=target,
            snr=snr,
            timestep_loss_weight=weights,
            per_sample_simple_loss=None,
            per_sample_weighted_simple_loss=None,
            per_sample_variational_bound=None,
            per_sample_loss=None,
        )

    sample_weights = weights.to(
        device=per_sample_simple.device,
        dtype=per_sample_simple.dtype,
    )
    weighted_simple = sample_weights * per_sample_simple
    if variance_values is None:
        variational_bound = None
        per_sample_loss = weighted_simple
    else:
        variational_bound = _rescaled_variational_bound(
            process=process,
            clean=clean,
            noisy=noisy,
            state_times=state_times,
            mean_output=mean_output,
            variance_values=variance_values,
            prediction_type=prediction_type,
        ).to(dtype=per_sample_simple.dtype)
        per_sample_loss = weighted_simple + variational_bound
    if loss_weighting.name == "constant" and variance.mode == "fixed":
        loss = scalar_simple
    else:
        loss = per_sample_loss.mean()
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
        raise ValueError("Gaussian prediction_type must be epsilon, x0, v, or score")
    if prediction_type == "epsilon":
        return noise
    if prediction_type == "x0":
        return clean
    scales = process.marginal_scales(state_times, clean.size())
    if prediction_type == "v":
        return scales.signal * noise - scales.noise * clean
    return -noise / scales.noise


def _split_model_output(
    raw_model_output: object,
    *,
    state: torch.Tensor,
    variance_mode: VarianceMode,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(raw_model_output, torch.Tensor):
        raise TypeError("Gaussian model must return a Tensor")
    if raw_model_output.device != state.device:
        raise ValueError("Gaussian model output must share the noisy state device")
    if not torch.is_floating_point(raw_model_output):
        raise TypeError("Gaussian model output must be floating-point")
    if variance_mode == "fixed":
        if raw_model_output.shape != state.shape:
            raise ValueError(
                "fixed-variance Gaussian model output must match the state shape"
            )
        return raw_model_output, None
    if state.ndim < 2:
        raise ValueError(
            "learned_range Gaussian states must include a channel dimension"
        )
    expected = (state.shape[0], state.shape[1] * 2, *state.shape[2:])
    if raw_model_output.shape != expected:
        raise ValueError(
            "learned_range Gaussian model output must have shape "
            f"{expected}, got {tuple(raw_model_output.shape)}"
        )
    mean_output, variance_values = raw_model_output.chunk(2, dim=1)
    return mean_output, variance_values


def _rescaled_variational_bound(
    *,
    process: DiscreteGaussianDenoisingProcess,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    state_times: torch.Tensor,
    mean_output: torch.Tensor,
    variance_values: torch.Tensor,
    prediction_type: PredictionType,
) -> torch.Tensor:
    target_times = state_times - 1
    model_log_variance = learned_range_log_variance(
        process,
        state_times,
        target_times,
        variance_values,
    )
    frozen_prediction = normalize_gaussian_prediction(
        process,
        noisy,
        state_times,
        mean_output.detach(),
        prediction_type=prediction_type,
        clip_denoised=False,
    )
    true_mean = process.posterior_mean(noisy, state_times, clean)
    model_mean = process.posterior_mean(
        noisy,
        state_times,
        frozen_prediction.clean,
    )
    if not isinstance(process, LearnedRangeGaussianVarianceProcess):
        raise TypeError(
            "learned_range variance requires "
            "LearnedRangeGaussianVarianceProcess capability"
        )
    true_log_variance = process.reverse_log_variance_bounds(
        state_times,
        target_times,
        clean.size(),
    ).lower
    kl = _mean_flat(
        _normal_kl(
            true_mean,
            true_log_variance,
            model_mean,
            model_log_variance,
        )
    ) / math.log(2.0)
    decoder_nll = -_discretized_gaussian_log_likelihood(
        clean,
        means=model_mean,
        log_scales=0.5 * model_log_variance,
    )
    decoder_nll = _mean_flat(decoder_nll) / math.log(2.0)
    per_sample = torch.where(
        state_times == process.clean_time + 1,
        decoder_nll,
        kl,
    )
    transition_count = process.terminal_time - process.clean_time
    return per_sample * (transition_count / 1000.0)


def _normal_kl(
    mean1: torch.Tensor,
    log_variance1: torch.Tensor,
    mean2: torch.Tensor,
    log_variance2: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        -1.0
        + log_variance2
        - log_variance1
        + torch.exp(log_variance1 - log_variance2)
        + (mean1 - mean2).square() * torch.exp(-log_variance2)
    )


def _approx_standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi)
            * (value + 0.044715 * value.pow(3))
        )
    )


def _discretized_gaussian_log_likelihood(
    value: torch.Tensor,
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    if value.shape != means.shape:
        raise ValueError("decoder likelihood means must match clean samples")
    centered = value - means
    inverse_standard_deviation = torch.exp(-log_scales)
    plus_input = inverse_standard_deviation * (centered + 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_input)
    minimum_input = inverse_standard_deviation * (centered - 1.0 / 255.0)
    cdf_minimum = _approx_standard_normal_cdf(minimum_input)
    log_cdf_plus = torch.log(cdf_plus.clamp_min(1e-12))
    log_one_minus_cdf_minimum = torch.log(
        (1.0 - cdf_minimum).clamp_min(1e-12)
    )
    cdf_delta = cdf_plus - cdf_minimum
    return torch.where(
        value < -0.999,
        log_cdf_plus,
        torch.where(
            value > 0.999,
            log_one_minus_cdf_minimum,
            torch.log(cdf_delta.clamp_min(1e-12)),
        ),
    )


def _mean_flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).mean(dim=1)


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _positive_channel_count(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def requires_per_sample_objective(
    variance: GaussianVarianceConfig,
    loss_weighting: GaussianLossWeightingConfig,
) -> bool:
    """Return whether the configured policy consumes per-sample losses."""

    return variance.mode == "learned_range" or loss_weighting.name == "p2"


def validate_per_sample_objective(
    objective: nn.Module,
    *,
    variance: GaussianVarianceConfig,
    loss_weighting: GaussianLossWeightingConfig,
    path: str,
) -> None:
    """Fail at the composition boundary when a policy needs this capability."""

    if requires_per_sample_objective(variance, loss_weighting) and not isinstance(
        objective,
        PerSampleObjective,
    ):
        raise TypeError(
            f"{path} requires objective satisfying PerSampleObjective"
        )


__all__ = [
    "GaussianLossComputation",
    "GaussianLossWeightingConfig",
    "GaussianVarianceConfig",
    "compute_gaussian_training_loss",
    "gaussian_loss_diagnostics",
    "gaussian_signal_to_noise_ratio",
    "gaussian_timestep_loss_weights",
    "gaussian_training_target",
    "learned_range_log_variance",
    "parse_gaussian_loss_weighting",
    "parse_gaussian_variance",
    "requires_per_sample_objective",
    "validate_gaussian_model_output_layout",
    "validate_per_sample_objective",
]
