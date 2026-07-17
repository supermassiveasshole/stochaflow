"""Configuration and registry validation for diagnostic providers."""

from typing import Any

import pytest

from stochaflow.diffusion import DDPM, LinearBetaSchedule
from stochaflow.training.diagnostics import DiffusionQualityDiagnostic
from stochaflow.utils.registry import RegistryError

from .helpers import (
    RecordingLogger,
    ZeroDenoiser,
    fit_event,
    profiles,
    provider_config,
    trainer,
)


def _params(tmp_path) -> dict[str, Any]:
    return {
        "logger": RecordingLogger(),
        "output_dir": tmp_path,
        "sample_shape": (1, 4, 4),
        "samplers": profiles()[:1],
        "providers": provider_config(),
    }


@pytest.mark.parametrize(
    ("override", "exception", "match"),
    [
        ({"cadence": {"step_every": 0}}, ValueError, "step_every"),
        (
            {"cadence": {"artifact_every_epochs": 0}},
            ValueError,
            "artifact_every_epochs",
        ),
        ({"sampling": {"sample_num": 0}}, ValueError, "sample_num"),
        ({"failure_policy": "ignore"}, ValueError, "failure_policy"),
        (
            {
                "samplers": [
                    {"id": "same", "name": "ddpm"},
                    {"id": "same", "name": "ddim"},
                ]
            },
            ValueError,
            "duplicate",
        ),
        (
            {
                "providers": {
                    "step_metrics": [
                        {"name": "noise_alignment"},
                        {"name": "noise_alignment"},
                    ],
                    "sampler_metrics": [],
                    "denoiser_artifacts": [],
                    "sampler_artifacts": [],
                }
            },
            ValueError,
            "duplicate provider",
        ),
        (
            {
                "providers": {
                    "step_metrics": [{"name": "missing"}],
                    "sampler_metrics": [],
                    "denoiser_artifacts": [],
                    "sampler_artifacts": [],
                }
            },
            RegistryError,
            "unknown diagnostic step metric provider",
        ),
        (
            {
                "reference": {
                    "enabled": True,
                    "num_real": 1,
                    "num_fake": 2,
                }
            },
            ValueError,
            "at least 2",
        ),
        (
            {"modules": ["diagnostic_module_that_does_not_exist"]},
            RegistryError,
            "failed to import diagnostic provider module",
        ),
    ],
)
def test_diffusion_quality_rejects_invalid_configuration(
    tmp_path,
    override,
    exception,
    match,
) -> None:
    params = _params(tmp_path)
    params.update(override)
    with pytest.raises(exception, match=match):
        DiffusionQualityDiagnostic(**params)


def test_explicit_empty_provider_categories_disable_defaults(tmp_path) -> None:
    diagnostic = DiffusionQualityDiagnostic(
        **{
            **_params(tmp_path),
            "providers": {
                "step_metrics": [],
                "sampler_metrics": [],
                "denoiser_artifacts": [],
                "sampler_artifacts": [],
            },
        }
    )

    assert diagnostic.step_metrics == ()
    assert diagnostic.sampler_metrics == ()
    assert diagnostic.denoiser_artifacts == ()
    assert diagnostic.sampler_artifacts == ()


def test_unknown_sampler_fails_at_fit_start(tmp_path) -> None:
    params = _params(tmp_path)
    params["samplers"] = [{"id": "missing", "name": "missing"}]
    diagnostic = DiffusionQualityDiagnostic(**params)
    model = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=2),
        model=ZeroDenoiser(),
    )

    with pytest.raises(RegistryError, match="unknown diffusion"):
        diagnostic.on_fit_start(fit_event(trainer(model)))


def test_missing_trajectory_capability_fails_at_fit_start(tmp_path) -> None:
    params = _params(tmp_path)
    params["samplers"] = [
        {
            "id": "no_trace",
            "name": "test_recording_diagnostic",
            "params": {"marker": "no_trace"},
            "trajectory": {"enabled": True},
        }
    ]
    diagnostic = DiffusionQualityDiagnostic(**params)
    model = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=2),
        model=ZeroDenoiser(),
    )

    with pytest.raises(TypeError, match="sample_trajectory_from_noise"):
        diagnostic.on_fit_start(fit_event(trainer(model)))
