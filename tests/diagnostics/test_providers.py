"""Focused tests for built-in metric and artifact providers."""

import pytest
import torch

from stochaflow.diffusion import DDPM, LinearBetaSchedule
from stochaflow.training.diagnostics.contracts import (
    DenoiserArtifactContext,
    ProviderValidationContext,
    ReconstructionFrame,
    ReconstructionResult,
    SamplerArtifactContext,
    SamplerMetricContext,
    SamplingResult,
    StepMetricContext,
)
from stochaflow.training.diagnostics.manifest import EpochArtifactStore
from stochaflow.training.diagnostics.providers.artifacts import (
    ReconstructionPanelProvider,
    SampleGridProvider,
    TrajectoryArtifactProvider,
)
from stochaflow.training.diagnostics.providers.denoiser import (
    NoiseAlignmentProvider,
    TimestepBucketLossProvider,
    X0ReconstructionMetricProvider,
)
from stochaflow.training.diagnostics.providers.sampler import (
    SampleStatisticsProvider,
    SamplingPerformanceProvider,
)

from .helpers import ZeroDenoiser


def _reconstruct(**kwargs) -> ReconstructionResult:
    clean = kwargs["clean_samples"][: kwargs["max_samples"]]
    frames = tuple(
        ReconstructionFrame(
            timestep=timestep,
            clean=clean,
            noisy=torch.zeros_like(clean),
            predicted_clean=torch.ones_like(clean),
            mse=float(timestep),
            psnr=float(timestep + 10),
        )
        for timestep in kwargs["timesteps"]
    )
    return ReconstructionResult(frames=frames)


def _step_context() -> StepMetricContext:
    diffusion = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=4),
        model=ZeroDenoiser(),
    )
    return StepMetricContext(
        diffusion=diffusion,
        diagnostics={
            "timesteps": torch.tensor([1, 4]),
            "per_sample_loss": torch.tensor([1.0, 3.0]),
            "predicted_noise": torch.zeros(2, 1, 2, 2),
            "target_noise": torch.ones(2, 1, 2, 2),
        },
        clean_samples=torch.zeros(2, 1, 2, 2),
        sample_num=2,
        use_ema=True,
        reconstruct=_reconstruct,
    )


def test_step_metric_providers_produce_disjoint_namespaced_metrics() -> None:
    context = _step_context()
    bucket_metrics = TimestepBucketLossProvider(buckets=2).collect(context)
    noise_metrics = NoiseAlignmentProvider().collect(context)
    reconstruction = X0ReconstructionMetricProvider(timesteps=[1, 4])
    reconstruction.validate(
        ProviderValidationContext(
            diffusion=context.diffusion,
            sample_shape=(1, 2, 2),
        )
    )
    reconstruction_metrics = reconstruction.collect(context)

    assert bucket_metrics["diagnostics/denoiser/loss_t_001_002"] == 1.0
    assert bucket_metrics["diagnostics/denoiser/loss_t_003_004"] == 3.0
    assert noise_metrics["diagnostics/denoiser/pred_noise_std"] == 0.0
    assert "diagnostics/denoiser/noise_cosine_similarity" in noise_metrics
    assert reconstruction_metrics[
        "diagnostics/denoiser/reconstruction_t_0001/mse"
    ] == 1.0
    assert not set(bucket_metrics) & set(noise_metrics)


def test_x0_reconstruction_uses_configured_ema_setting() -> None:
    received: list[bool] = []

    def reconstruct(**kwargs) -> ReconstructionResult:
        received.append(kwargs["use_ema"])
        return _reconstruct(**kwargs)

    context = _step_context()
    context = StepMetricContext(
        diffusion=context.diffusion,
        diagnostics=context.diagnostics,
        clean_samples=context.clean_samples,
        sample_num=context.sample_num,
        use_ema=True,
        reconstruct=reconstruct,
    )

    X0ReconstructionMetricProvider(timesteps=[1]).collect(context)

    assert received == [True]


def test_sampler_metric_providers_share_one_sampling_result() -> None:
    result = SamplingResult(
        samples=torch.tensor([[[[-1.0, 1.0]]], [[[0.0, 0.0]]]]),
        trajectory=None,
        duration_seconds=2.0,
    )
    context = SamplerMetricContext(
        profile_id="test",
        profile_name="ddim",
        result=result,
    )

    statistics = SampleStatisticsProvider().collect(context)
    performance = SamplingPerformanceProvider().collect(context)

    assert statistics["diagnostics/samplers/test/sample_mean"] == 0.0
    assert statistics["diagnostics/samplers/test/saturation_fraction"] == 0.5
    assert performance["diagnostics/samplers/test/sampling_seconds"] == 2.0
    assert performance["diagnostics/samplers/test/samples_per_second"] == 1.0


def test_artifact_providers_write_expected_layout_and_detect_collisions(tmp_path) -> None:
    store = EpochArtifactStore(tmp_path / "diagnostics" / "diffusion_quality", 1)
    clean = torch.zeros(2, 1, 4, 4)
    reconstruction_records = ReconstructionPanelProvider(
        timesteps=[1],
        max_samples=2,
    ).render(
        DenoiserArtifactContext(
            store=store,
            clean_samples=clean,
            reconstruct=_reconstruct,
            use_ema=False,
        )
    )
    result = SamplingResult(
        samples=torch.zeros(2, 1, 4, 4),
        trajectory={2: torch.ones(2, 1, 4, 4), 0: torch.zeros(2, 1, 4, 4)},
        duration_seconds=0.1,
    )
    context = SamplerArtifactContext(
        store=store,
        profile_id="profile",
        profile_name="ddpm",
        trajectory_enabled=True,
        trajectory_gif_fps=4,
        result=result,
    )
    sample_provider = SampleGridProvider(nrow=2)
    sample_records = sample_provider.render(context)
    trajectory_records = TrajectoryArtifactProvider(nrow=2).render(context)

    for record in (*reconstruction_records, *sample_records, *trajectory_records):
        assert record.path.is_file()
    assert any(record.image_tag for record in sample_records)
    assert any(record.path.suffix == ".gif" for record in trajectory_records)
    with pytest.raises(ValueError, match="collision"):
        sample_provider.render(context)
