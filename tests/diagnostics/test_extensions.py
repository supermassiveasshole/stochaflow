"""Open-closed acceptance tests for externally registered providers."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from stochaflow.training.diagnostics import (
    DIAGNOSTIC_PROVIDERS,
    DiffusionQualityDiagnostic,
)
from stochaflow.utils.registry import RegistryError

from .helpers import (
    RecordingLogger,
    ZeroDenoiser,
    batch_event,
    epoch_event,
    fit_event,
    gaussian_system,
    trainer,
)

MODULE_SOURCE = '''
from stochaflow.training.diagnostics import (
    ArtifactRecord,
    DIAGNOSTIC_PROVIDERS,
    DenoiserArtifactProvider,
    ReferenceMetricProvider,
    SamplerArtifactProvider,
    SamplerMetricProvider,
    StepMetricProvider,
)


@DIAGNOSTIC_PROVIDERS.step_metrics.register("ocp_custom_step")
class CustomStep(StepMetricProvider):
    def collect(self, context):
        return {"diagnostics/custom/step": 7.0}


@DIAGNOSTIC_PROVIDERS.step_metrics.register("ocp_custom_collision")
class CustomCollision(StepMetricProvider):
    def collect(self, context):
        return {"diagnostics/custom/step": 8.0}


@DIAGNOSTIC_PROVIDERS.sampler_metrics.register("ocp_custom_sampler")
class CustomSampler(SamplerMetricProvider):
    def collect(self, context):
        return {f"diagnostics/samplers/{context.profile_id}/custom": 3.0}


@DIAGNOSTIC_PROVIDERS.denoiser_artifacts.register("ocp_custom_denoiser_artifact")
class CustomDenoiserArtifact(DenoiserArtifactProvider):
    def render(self, context):
        path = context.store.reserve("denoiser/custom.txt")
        path.write_text("denoiser", encoding="utf-8")
        return [ArtifactRecord(kind="custom_denoiser", path=path)]


@DIAGNOSTIC_PROVIDERS.sampler_artifacts.register("ocp_custom_sampler_artifact")
class CustomSamplerArtifact(SamplerArtifactProvider):
    def render(self, context):
        path = context.store.reserve(f"{context.profile_id}/custom.txt")
        path.write_text("sampler", encoding="utf-8")
        return [ArtifactRecord(kind="custom_sampler", path=path)]


@DIAGNOSTIC_PROVIDERS.reference_metrics.register("ocp_custom_reference")
class CustomReference(ReferenceMetricProvider):
    def __init__(self, *, device, num_real, num_fake):
        del device, num_real, num_fake
        self.real = 0
        self.fake = 0

    def update(self, images, *, real):
        if real:
            self.real += images.shape[0]
        else:
            self.fake += images.shape[0]

    def compute(self):
        return {"custom_reference": float(self.real + self.fake)}

    def reset_fake(self):
        self.fake = 0
'''


def _write_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    module_name = "ocp_diagnostic_extension"
    (tmp_path / f"{module_name}.py").write_text(MODULE_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return module_name


def test_external_module_registers_every_provider_type_without_orchestrator_changes(
    monkeypatch,
    tmp_path,
) -> None:
    module_name = _write_extension(tmp_path, monkeypatch)
    logger = RecordingLogger()
    diagnostic = DiffusionQualityDiagnostic(
        logger=logger,
        output_dir=tmp_path,
        sample_shape=(1, 4, 4),
        modules=[module_name],
        cadence={"step_every": 1, "artifact_every_epochs": 1},
        sampling={"sample_num": 2, "batch_size": 1, "seed": 123},
        samplers=[{"id": "ddpm", "name": "ddpm"}],
        providers={
            "step_metrics": [{"name": "ocp_custom_step"}],
            "sampler_metrics": [{"name": "ocp_custom_sampler"}],
            "denoiser_artifacts": [
                {"name": "ocp_custom_denoiser_artifact"}
            ],
            "sampler_artifacts": [{"name": "ocp_custom_sampler_artifact"}],
        },
        reference={
            "enabled": True,
            "every_epochs": 1,
            "num_real": 2,
            "num_fake": 2,
            "batch_size": 1,
            "metrics": [{"name": "ocp_custom_reference"}],
        },
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)
    runtime = trainer(model)
    diagnostic.on_fit_start(
        fit_event(runtime, validation=[torch.zeros(2, 1, 4, 4)])
    )
    diagnostic.on_train_batch_end(batch_event(runtime))

    diagnostic.on_train_epoch_end(epoch_event(runtime))

    all_metrics = {key for _, payload in logger.metrics for key in payload}
    assert "diagnostics/custom/step" in all_metrics
    assert "diagnostics/samplers/ddpm/custom" in all_metrics
    assert "diagnostics/samplers/ddpm/custom_reference" in all_metrics
    epoch = tmp_path / "diagnostics" / "diffusion_quality" / "epoch_0001"
    assert (epoch / "denoiser" / "custom.txt").is_file()
    assert (epoch / "ddpm" / "custom.txt").is_file()


def test_provider_metric_collision_isolated_by_warn_policy(monkeypatch, tmp_path) -> None:
    module_name = _write_extension(tmp_path, monkeypatch)
    logger = RecordingLogger()
    diagnostic = DiffusionQualityDiagnostic(
        logger=logger,
        output_dir=tmp_path,
        sample_shape=(1, 4, 4),
        modules=[module_name],
        cadence={"step_every": 1},
        samplers=[{"id": "ddpm", "name": "ddpm"}],
        providers={
            "step_metrics": [
                {"name": "ocp_custom_step"},
                {"name": "ocp_custom_collision"},
            ],
            "sampler_metrics": [],
            "denoiser_artifacts": [],
            "sampler_artifacts": [],
        },
        failure_policy="warn",
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)
    runtime = trainer(model)
    diagnostic.on_fit_start(fit_event(runtime))

    diagnostic.on_train_batch_end(batch_event(runtime))

    assert any("metric tag collision" in text for _, text, _ in logger.text)
    assert logger.metrics[-1][1]["diagnostics/custom/step"] == 7.0


def test_provider_registry_rejects_wrong_base() -> None:
    class WrongBase:
        pass

    with pytest.raises(RegistryError, match="must inherit"):
        DIAGNOSTIC_PROVIDERS.step_metrics.add(
            "wrong_base",
            cast(Any, WrongBase),
        )
