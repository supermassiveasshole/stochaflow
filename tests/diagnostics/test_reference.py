"""Reference metric provider and feature-cache tests."""

import sys
from types import ModuleType
from typing import cast

import pytest
import torch

from stochaflow.training.diagnostics import DiffusionQualityDiagnostic
from stochaflow.training.diagnostics.config import ReferencePipelineConfig
from stochaflow.training.diagnostics.contracts import ReferenceMetricProvider
from stochaflow.training.diagnostics.providers.reference import ReferenceMetricSuite
from stochaflow.training.diagnostics.runtime import BoundSampler, SeedPolicy

from .helpers import (
    RecordingLogger,
    RecordingSampler,
    ZeroDenoiser,
    epoch_event,
    fit_event,
    gaussian_system,
    provider_config,
    trainer,
)


def _reference_config() -> dict:
    return {
        "enabled": True,
        "every_epochs": 1,
        "num_real": 2,
        "num_fake": 3,
        "batch_size": 2,
        "metrics": [
            {"name": "kid", "params": {"subsets": 2, "subset_size": 2}},
            {"name": "fid", "params": {}},
        ],
    }


def _diagnostic(tmp_path, logger, *, reference=None):
    return DiffusionQualityDiagnostic(
        logger=logger,
        output_dir=tmp_path,
        sample_shape=(1, 4, 4),
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
        cadence={"step_every": 1, "artifact_every_epochs": 5},
        sampling={"sample_num": 2, "batch_size": 1, "seed": 123},
        providers=provider_config(),
        reference=reference,
    )


def _install_fake_torchmetrics(monkeypatch):
    class FakeMetric:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.real_count = 0
            self.fake_count = 0

        def to(self, device):
            del device
            return self

        def set_dtype(self, dtype) -> None:
            del dtype

        def update(self, images: torch.Tensor, *, real: bool) -> None:
            torch.rand(1)
            assert images.shape[1] == 3
            assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0
            if real:
                self.real_count += images.shape[0]
            else:
                self.fake_count += images.shape[0]

        def reset(self) -> None:
            self.fake_count = 0

    class FakeFID(FakeMetric):
        def compute(self) -> torch.Tensor:
            return torch.tensor(float(self.real_count + self.fake_count))

    class FakeKID(FakeMetric):
        def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor(0.25), torch.tensor(0.05)

    torchmetrics = ModuleType("torchmetrics")
    image_module = ModuleType("torchmetrics.image")
    fid_module = ModuleType("torchmetrics.image.fid")
    kid_module = ModuleType("torchmetrics.image.kid")
    setattr(fid_module, "FrechetInceptionDistance", FakeFID)
    setattr(kid_module, "KernelInceptionDistance", FakeKID)
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", image_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", fid_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", kid_module)


def test_reference_metrics_require_validation_data(monkeypatch, tmp_path) -> None:
    _install_fake_torchmetrics(monkeypatch)
    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        reference=_reference_config(),
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)

    with pytest.raises(ValueError, match="validation dataloader"):
        diagnostic.on_fit_start(fit_event(trainer(model)))


def test_reference_providers_cache_multibatch_real_features_and_score_profiles(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_torchmetrics(monkeypatch)
    RecordingSampler.records.clear()
    logger = RecordingLogger()
    diagnostic = _diagnostic(
        tmp_path,
        logger,
        reference=_reference_config(),
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)
    runtime = trainer(model)
    torch.manual_seed(654)
    rng_before = torch.random.get_rng_state().clone()

    diagnostic.on_fit_start(
        fit_event(
            runtime,
            validation=[
                torch.zeros(1, 1, 4, 4),
                torch.zeros(1, 1, 4, 4),
            ],
        )
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    diagnostic.on_train_epoch_end(epoch_event(runtime))

    payload = logger.metrics[-1][1]
    for profile_id in ("first", "second"):
        prefix = f"diagnostics/samplers/{profile_id}"
        assert payload[f"{prefix}/fid"] == 5.0
        assert payload[f"{prefix}/kid_mean"] == pytest.approx(0.25)
        assert payload[f"{prefix}/kid_std"] == pytest.approx(0.05)
        assert payload[f"{prefix}/reference_fake_samples"] == 3.0
    assert diagnostic._reference_suite is not None
    for _, provider in diagnostic._reference_suite.providers:
        metric = getattr(provider, "metric")
        assert metric.real_count == 2
        assert metric.fake_count == 0


def test_enabled_reference_reports_missing_optional_dependencies(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", None)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", None)
    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        reference=_reference_config(),
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)

    with pytest.raises(RuntimeError, match="quality"):
        diagnostic.on_fit_start(
            fit_event(
                trainer(model),
                validation=[torch.zeros(2, 1, 4, 4)],
            )
        )


def test_disabled_reference_does_not_import_torchmetrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", None)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", None)
    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        reference={"enabled": False},
    )
    model = gaussian_system(ZeroDenoiser(), num_timesteps=2)

    diagnostic.on_fit_start(fit_event(trainer(model)))


def test_real_cache_warn_failure_disables_only_the_failing_provider() -> None:
    class FailingProvider(ReferenceMetricProvider):
        def __init__(self) -> None:
            self.fake_updates = 0

        def update(self, images: torch.Tensor, *, real: bool) -> None:
            if real:
                raise RuntimeError("real update failed")
            self.fake_updates += images.shape[0]

        def compute(self) -> dict[str, float]:
            return {"failed": 1.0}

        def reset_fake(self) -> None:
            self.fake_updates = 0

    class HealthyProvider(ReferenceMetricProvider):
        def __init__(self) -> None:
            self.real = 0
            self.fake = 0

        def update(self, images: torch.Tensor, *, real: bool) -> None:
            if real:
                self.real += images.shape[0]
            else:
                self.fake += images.shape[0]

        def compute(self) -> dict[str, float]:
            return {"healthy": float(self.real + self.fake)}

        def reset_fake(self) -> None:
            self.fake = 0

    failing = FailingProvider()
    healthy = HealthyProvider()
    errors: list[tuple[str, str, str]] = []
    suite = ReferenceMetricSuite(
        (("failing", failing), ("healthy", healthy)),
        ReferencePipelineConfig(
            enabled=True,
            every_epochs=1,
            num_real=2,
            num_fake=2,
            batch_size=1,
        ),
        device=torch.device("cpu"),
        seed_policy=SeedPolicy(123),
        handle_error=lambda phase, provider, error: errors.append(
            (phase, provider, str(error))
        ),
    )

    suite.cache_real(
        [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)]
    )
    metrics = suite.evaluate(
        profile_id="profile",
        sampler=cast(BoundSampler, torch.nn.Identity()),
        sample_shape=(1, 4, 4),
        visual_samples=torch.zeros(2, 1, 4, 4),
    )

    assert errors == [("reference_real_update", "failing", "real update failed")]
    assert healthy.real == 2
    assert metrics["diagnostics/samplers/profile/healthy"] == 4.0
    assert "diagnostics/samplers/profile/failed" not in metrics
    assert failing.fake_updates == 0
