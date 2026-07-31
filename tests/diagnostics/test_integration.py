"""End-to-end integration tests for the diffusion quality orchestrator."""

import pytest
import torch
import yaml

from stochaflow.training.diagnostics import DiffusionQualityDiagnostic
from stochaflow.training.ema import ExponentialMovingAverage

from .helpers import (
    RecordingLogger,
    RecordingSampler,
    TinyDenoiser,
    ZeroDenoiser,
    batch_event,
    epoch_event,
    fit_event,
    gaussian_system,
    profiles,
    provider_config,
    trainer,
)


def _diagnostic(tmp_path, logger, **overrides) -> DiffusionQualityDiagnostic:
    params = {
        "logger": logger,
        "output_dir": tmp_path,
        "samplers": profiles(),
        "cadence": {"step_every": 1, "artifact_every_epochs": 1},
        "sampling": {
            "shape": [1, 4, 4],
            "sample_num": 2,
            "batch_size": 1,
            "seed": 123,
        },
        "providers": provider_config(),
        "use_ema": False,
        "failure_policy": "raise",
    }
    params.update(overrides)
    return DiffusionQualityDiagnostic(**params)


def test_step_pipeline_logs_all_denoiser_provider_metrics(tmp_path) -> None:
    logger = RecordingLogger()
    diagnostic = _diagnostic(
        tmp_path,
        logger,
        samplers=profiles()[:1],
        cadence={"step_every": 1, "artifact_every_epochs": 5},
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=4)
    runtime = trainer(model)
    diagnostic.on_fit_start(fit_event(runtime))

    diagnostic.on_train_batch_end(batch_event(runtime))

    metrics = logger.metrics[-1][1]
    prefix = "diagnostics/diffusion_quality/denoiser"
    assert metrics[f"{prefix}/loss_t_001_002"] == 1.0
    assert metrics[f"{prefix}/loss_t_003_004"] == 3.0
    assert f"{prefix}/noise_cosine_similarity" in metrics
    assert f"{prefix}/reconstruction_t_0001/mse" in metrics


def test_gaussian_runtime_compares_ddpm_and_ddim_artifacts(tmp_path) -> None:
    logger = RecordingLogger()
    diagnostic = _diagnostic(
        tmp_path,
        logger,
        samplers=profiles(trajectory=True),
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=4)
    runtime = trainer(model)
    diagnostic.on_fit_start(fit_event(runtime))
    diagnostic.on_train_batch_end(batch_event(runtime))

    result = diagnostic.on_train_epoch_end(epoch_event(runtime))

    epoch_dir = tmp_path / "diagnostics" / "diffusion_quality" / "epoch_0001"
    manifest_path = epoch_dir / "manifest.yaml"
    assert manifest_path.is_file()
    assert (epoch_dir / "denoiser" / "reconstruction.png").is_file()
    for profile_id in ("ddpm_full", "ddim_2"):
        target = epoch_dir / profile_id
        for name in (
            "samples.pt",
            "samples.png",
            "trajectory.pt",
            "trajectory.png",
            "trajectory.gif",
        ):
            assert (target / name).is_file()
    tags = {tag for tag, _, _, _ in logger.images}
    assert "diagnostics/samplers/ddpm_full/samples" in tags
    assert "diagnostics/samplers/ddim_2/trajectory" in tags
    assert result is None
    combined = set(logger.metrics[-1][1])
    assert (
        "diagnostics/diffusion_quality/samplers/ddpm_full/sampling_seconds"
        in combined
    )
    assert (
        "diagnostics/diffusion_quality/samplers/ddim_2/batch_diversity"
        in combined
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["combined_metrics"]) == {
        "observation",
        "validation_quality",
    }
    assert set(manifest["combined_metrics"]["observation"]) == combined
    assert manifest["combined_metrics"]["validation_quality"] == {}
    assert "metrics" not in manifest


def test_profiles_share_noise_and_restore_ema_model_and_rng(tmp_path) -> None:
    RecordingSampler.records.clear()
    logger = RecordingLogger()
    diagnostic = _diagnostic(
        tmp_path,
        logger,
        samplers=[
            {
                "id": "first",
                "name": "test_recording_diagnostic",
                "params": {"marker": "first"},
            },
            {
                "id": "second",
                "name": "test_recording_diagnostic",
                "params": {"marker": "second"},
            },
        ],
        use_ema=True,
    )
    model = gaussian_system(TinyDenoiser(), num_timesteps=2)
    ema = ExponentialMovingAverage(model.inference_model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    runtime = trainer(model, ema=ema)
    diagnostic.on_fit_start(fit_event(runtime))
    diagnostic.on_train_batch_end(batch_event(runtime))
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    model.train()
    torch.manual_seed(987)
    rng_before = torch.random.get_rng_state().clone()

    diagnostic.on_train_epoch_end(epoch_event(runtime))

    first = torch.cat(RecordingSampler.records["first"], dim=0)
    second = torch.cat(RecordingSampler.records["second"], dim=0)
    assert torch.equal(first, second)
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters_before[name])


@pytest.mark.parametrize("training", [True, False])
def test_fit_start_preserves_shared_denoiser_mode(tmp_path, training) -> None:
    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        samplers=profiles()[:1],
    )
    model = gaussian_system(TinyDenoiser(), num_timesteps=2)
    model.train(training)

    diagnostic.on_fit_start(fit_event(trainer(model)))

    assert model.training is training
    assert model.inference_model.training is training


def test_fixed_seed_repeats_stochastic_sampler_results(tmp_path) -> None:
    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        samplers=[{"id": "ddpm_full", "name": "ddpm"}],
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)
    runtime = trainer(model)
    diagnostic.on_fit_start(fit_event(runtime))
    diagnostic.on_train_batch_end(batch_event(runtime))

    diagnostic.on_train_epoch_end(epoch_event(runtime, 1))
    diagnostic.on_train_epoch_end(epoch_event(runtime, 2))

    root = tmp_path / "diagnostics" / "diffusion_quality"
    first = torch.load(
        root / "epoch_0001" / "ddpm_full" / "samples.pt",
        weights_only=True,
    )
    second = torch.load(
        root / "epoch_0002" / "ddpm_full" / "samples.pt",
        weights_only=True,
    )
    assert torch.equal(first, second)


def test_warn_policy_isolates_profile_failure_and_records_manifest_error(tmp_path) -> None:
    logger = RecordingLogger()
    diagnostic = _diagnostic(
        tmp_path,
        logger,
        samplers=[{"id": "broken", "name": "test_failing_diagnostic"}],
        failure_policy="warn",
        use_ema=True,
        providers={
            **provider_config(),
            "denoiser_artifacts": [],
        },
    )
    model = gaussian_system(TinyDenoiser(), num_timesteps=2)
    ema = ExponentialMovingAverage(model.inference_model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    runtime = trainer(model, ema=ema)
    diagnostic.on_fit_start(fit_event(runtime))
    model.train()
    torch.manual_seed(321)
    rng_before = torch.random.get_rng_state().clone()

    result = diagnostic.on_train_epoch_end(epoch_event(runtime))

    assert result is None
    returned_metrics = logger.metrics[-1][1]
    assert (
        returned_metrics[
            "diagnostics/diffusion_quality/system/error_count"
        ]
        == 1.0
    )
    assert "sampling failed" in logger.text[-1][1]
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters_before[name])
    manifest_path = (
        tmp_path
        / "diagnostics"
        / "diffusion_quality"
        / "epoch_0001"
        / "manifest.yaml"
    )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["errors"][0]["provider"] == "broken"
