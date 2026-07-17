"""Built-in sampler metric providers."""

from __future__ import annotations

from collections.abc import Mapping

from stochaflow.training.diagnostics.contracts import (
    SamplerMetricContext,
    SamplerMetricProvider,
)
from stochaflow.training.diagnostics.registry import DIAGNOSTIC_PROVIDERS


@DIAGNOSTIC_PROVIDERS.sampler_metrics.register("sample_statistics")
class SampleStatisticsProvider(SamplerMetricProvider):
    """Measure value distribution, saturation, and batch diversity."""

    def collect(self, context: SamplerMetricContext) -> Mapping[str, float]:
        values = context.result.samples.float()
        if values.ndim != 4 or values.shape[0] == 0:
            raise ValueError("sample_statistics requires a non-empty image batch")
        prefix = f"diagnostics/samplers/{context.profile_id}"
        return {
            f"{prefix}/sample_mean": float(values.mean()),
            f"{prefix}/sample_std": float(values.std(unbiased=False)),
            f"{prefix}/saturation_fraction": float(
                (values.abs() >= 0.99).float().mean()
            ),
            f"{prefix}/batch_diversity": float(
                values.var(dim=0, unbiased=False).mean()
            ),
        }


@DIAGNOSTIC_PROVIDERS.sampler_metrics.register("sampling_performance")
class SamplingPerformanceProvider(SamplerMetricProvider):
    """Report wall-clock sampling duration and throughput."""

    def collect(self, context: SamplerMetricContext) -> Mapping[str, float]:
        count = context.result.samples.shape[0]
        duration = context.result.duration_seconds
        if count <= 0 or duration < 0:
            raise ValueError("sampling_performance received an invalid sampling result")
        prefix = f"diagnostics/samplers/{context.profile_id}"
        return {
            f"{prefix}/sampling_seconds": duration,
            f"{prefix}/samples_per_second": count / max(duration, 1e-12),
        }


__all__ = ["SampleStatisticsProvider", "SamplingPerformanceProvider"]
