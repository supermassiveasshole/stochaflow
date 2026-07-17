"""Built-in step-level denoiser metric providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from stochaflow.diffusion import GaussianDiffusion
from stochaflow.training.diagnostics.contracts import (
    ProviderValidationContext,
    StepMetricContext,
    StepMetricProvider,
)
from stochaflow.training.diagnostics.registry import DIAGNOSTIC_PROVIDERS


def parse_timesteps(raw: Sequence[int], *, provider: str) -> tuple[int, ...]:
    """Validate one provider's ordered fixed-timestep list."""

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"{provider} timesteps must be a sequence")
    timesteps: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{provider} timesteps must contain positive integers")
        if value in timesteps:
            raise ValueError(f"{provider} timesteps must be unique")
        timesteps.append(value)
    if not timesteps:
        raise ValueError(f"{provider} timesteps must not be empty")
    return tuple(timesteps)


def validate_timesteps(
    timesteps: Sequence[int],
    context: ProviderValidationContext,
    *,
    provider: str,
) -> None:
    """Validate fixed timesteps against the training diffusion schedule."""

    diffusion = context.diffusion
    if not isinstance(diffusion, GaussianDiffusion):
        raise TypeError(f"{provider} requires a GaussianDiffusion")
    invalid = [value for value in timesteps if value > diffusion.num_timesteps]
    if invalid:
        rendered = ", ".join(str(value) for value in invalid)
        raise ValueError(
            f"{provider} timesteps exceed the training schedule "
            f"({diffusion.num_timesteps}): {rendered}"
        )


@DIAGNOSTIC_PROVIDERS.step_metrics.register("timestep_bucket_loss")
class TimestepBucketLossProvider(StepMetricProvider):
    """Aggregate per-sample denoising loss into timestep buckets."""

    def __init__(self, *, buckets: int = 10) -> None:
        if isinstance(buckets, bool) or not isinstance(buckets, int) or buckets <= 0:
            raise ValueError("timestep_bucket_loss buckets must be a positive integer")
        self.buckets = buckets

    def collect(self, context: StepMetricContext) -> Mapping[str, float]:
        timesteps = context.diagnostics.get("timesteps")
        per_sample_loss = context.diagnostics.get("per_sample_loss")
        if not isinstance(timesteps, torch.Tensor) or not isinstance(
            per_sample_loss,
            torch.Tensor,
        ):
            raise ValueError(
                "timestep_bucket_loss requires timesteps and per_sample_loss tensors"
            )
        timesteps = timesteps.detach().flatten().cpu()
        losses = per_sample_loss.detach().flatten().cpu()
        if timesteps.shape != losses.shape:
            raise ValueError(
                "timestep_bucket_loss timesteps and per_sample_loss must align"
            )
        num_timesteps = context.diffusion.num_timesteps
        width = max(1, (num_timesteps + self.buckets - 1) // self.buckets)
        digits = max(3, len(str(num_timesteps)))
        metrics: dict[str, float] = {}
        for bucket_index in range(self.buckets):
            start = 1 + bucket_index * width
            end = min(num_timesteps, start + width - 1)
            if start > end:
                break
            mask = (timesteps >= start) & (timesteps <= end)
            if bool(mask.any()):
                tag = (
                    f"diagnostics/denoiser/loss_t_"
                    f"{start:0{digits}d}_{end:0{digits}d}"
                )
                metrics[tag] = float(losses[mask].mean())
        return metrics


@DIAGNOSTIC_PROVIDERS.step_metrics.register("noise_alignment")
class NoiseAlignmentProvider(StepMetricProvider):
    """Measure predicted and target noise distribution alignment."""

    def collect(self, context: StepMetricContext) -> Mapping[str, float]:
        predicted = context.diagnostics.get("predicted_noise")
        target = context.diagnostics.get("target_noise")
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            target,
            torch.Tensor,
        ):
            raise ValueError(
                "noise_alignment requires predicted_noise and target_noise tensors"
            )
        if predicted.shape != target.shape or predicted.ndim < 2:
            raise ValueError(
                "noise_alignment predicted and target tensors must have matching "
                "batched shapes"
            )
        predicted_float = predicted.detach().float()
        target_float = target.detach().float()
        return {
            "diagnostics/denoiser/pred_noise_mean": float(predicted_float.mean()),
            "diagnostics/denoiser/pred_noise_std": float(
                predicted_float.std(unbiased=False)
            ),
            "diagnostics/denoiser/target_noise_mean": float(target_float.mean()),
            "diagnostics/denoiser/target_noise_std": float(
                target_float.std(unbiased=False)
            ),
            "diagnostics/denoiser/noise_cosine_similarity": float(
                F.cosine_similarity(
                    predicted_float.flatten(start_dim=1),
                    target_float.flatten(start_dim=1),
                    dim=1,
                ).mean()
            ),
        }


@DIAGNOSTIC_PROVIDERS.step_metrics.register("x0_reconstruction")
class X0ReconstructionMetricProvider(StepMetricProvider):
    """Measure fixed-timestep clean-sample reconstruction quality."""

    def __init__(self, *, timesteps: Sequence[int]) -> None:
        self.timesteps = parse_timesteps(
            timesteps,
            provider="x0_reconstruction",
        )

    def validate(self, context: ProviderValidationContext) -> None:
        validate_timesteps(
            self.timesteps,
            context,
            provider="x0_reconstruction",
        )

    def collect(self, context: StepMetricContext) -> Mapping[str, float]:
        if context.clean_samples is None:
            raise ValueError("x0_reconstruction requires clean_samples diagnostics")
        result = context.reconstruct(
            clean_samples=context.clean_samples,
            timesteps=self.timesteps,
            max_samples=context.sample_num,
            use_ema=context.use_ema,
        )
        metrics: dict[str, float] = {}
        for frame in result.frames:
            prefix = (
                f"diagnostics/denoiser/reconstruction_t_{frame.timestep:04d}"
            )
            metrics[f"{prefix}/mse"] = frame.mse
            metrics[f"{prefix}/psnr"] = frame.psnr
        return metrics


__all__ = [
    "NoiseAlignmentProvider",
    "TimestepBucketLossProvider",
    "X0ReconstructionMetricProvider",
    "parse_timesteps",
    "validate_timesteps",
]
