"""Gaussian learned-variance training policy isolated from simple loss weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

import torch

from stochaflow.families.gaussian import (
    PredictionType,
    VarianceMode,
    normalize_gaussian_prediction,
)
from stochaflow.models.denoising import DenoiserChannelLayout
from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.processes.gaussian import LearnedRangeGaussianVarianceProcess

VarianceLossName = Literal["rescaled_variational_bound"]


@dataclass(frozen=True, slots=True)
class GaussianVarianceConfig:
    """Validated Gaussian model-variance and hybrid-loss policy."""

    mode: VarianceMode = "fixed"
    loss: VarianceLossName | None = None

    def __post_init__(self) -> None:
        """Fail closed when constructed outside the YAML parser."""

        mode_value = cast(object, self.mode)
        if not isinstance(mode_value, str) or mode_value not in (
            "fixed",
            "learned_range",
        ):
            raise ValueError("GaussianVarianceConfig.mode must be fixed or learned_range")
        if mode_value == "fixed":
            if self.loss is not None:
                raise ValueError("fixed Gaussian variance cannot define a loss")
            return
        if self.loss != "rescaled_variational_bound":
            raise ValueError(
                "learned_range Gaussian variance requires "
                "loss='rescaled_variational_bound'"
            )


@runtime_checkable
class GaussianVarianceLoss(Protocol):
    """Compute an optional per-sample Gaussian variance-training term."""

    @property
    def bound_process(self) -> DiscreteGaussianDenoisingProcess:
        """Return the exact Process whose coefficients this loss consumes."""

        ...

    def per_sample_loss(
        self,
        *,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        state_times: torch.Tensor,
        mean_output: torch.Tensor,
        variance_values: torch.Tensor,
        prediction_type: PredictionType,
    ) -> torch.Tensor:
        """Return one variance loss for every leading batch item."""

        ...


class LearnedRangeGaussianVarianceLoss:
    """Compute the improved-DDPM learned-range variational-bound term."""

    def __init__(self, process: DiscreteGaussianDenoisingProcess) -> None:
        if not isinstance(process, LearnedRangeGaussianVarianceProcess):
            raise TypeError(
                "learned_range variance requires "
                "LearnedRangeGaussianVarianceProcess capability"
            )
        self.process = process
        self.variance_process = cast(LearnedRangeGaussianVarianceProcess, process)

    @property
    def bound_process(self) -> DiscreteGaussianDenoisingProcess:
        """Return the Process bound at the Builder composition boundary."""

        return self.process

    def per_sample_loss(
        self,
        *,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        state_times: torch.Tensor,
        mean_output: torch.Tensor,
        variance_values: torch.Tensor,
        prediction_type: PredictionType,
    ) -> torch.Tensor:
        """Return the detached-mean decoder NLL or KL term in bits."""

        process = self.process
        target_times = state_times - 1
        model_log_variance = learned_range_log_variance(
            self.variance_process,
            state_times,
            target_times,
            variance_values,
        )
        scales = process.marginal_scales(state_times, noisy.size())
        frozen_prediction = normalize_gaussian_prediction(
            noisy,
            mean_output.detach(),
            signal_scale=scales.signal,
            noise_scale=scales.noise,
            prediction_type=prediction_type,
        )
        true_mean = process.posterior_mean(noisy, state_times, clean)
        model_mean = process.posterior_mean(
            noisy,
            state_times,
            frozen_prediction.clean,
        )
        true_log_variance = self.variance_process.reverse_log_variance_bounds(
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


def build_gaussian_variance_loss(
    process: DiscreteGaussianDenoisingProcess,
    config: GaussianVarianceConfig,
) -> GaussianVarianceLoss | None:
    """Build the configured variance collaborator at the composition boundary."""

    if config.mode == "fixed":
        return None
    return LearnedRangeGaussianVarianceLoss(process)


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


def learned_range_log_variance(
    process: LearnedRangeGaussianVarianceProcess,
    source_times: torch.Tensor,
    target_times: torch.Tensor,
    variance_values: torch.Tensor,
) -> torch.Tensor:
    """Interpolate selected-pair posterior/beta log-variance endpoints."""

    process_value = cast(object, process)
    if not isinstance(process_value, LearnedRangeGaussianVarianceProcess):
        raise TypeError(
            "learned_range variance requires "
            "LearnedRangeGaussianVarianceProcess capability"
        )
    bounds = process_value.reverse_log_variance_bounds(
        source_times,
        target_times,
        variance_values.size(),
    )
    fraction = (variance_values + 1.0) / 2.0
    return fraction * bounds.upper + (1.0 - fraction) * bounds.lower


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


def _positive_channel_count(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


__all__ = [
    "GaussianVarianceConfig",
    "GaussianVarianceLoss",
    "LearnedRangeGaussianVarianceLoss",
    "build_gaussian_variance_loss",
    "learned_range_log_variance",
    "parse_gaussian_variance",
    "validate_gaussian_model_output_layout",
]
