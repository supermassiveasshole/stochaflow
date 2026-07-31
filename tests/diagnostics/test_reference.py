"""Reference metric provider and feature-cache tests."""

import sys
from collections.abc import Sequence
from types import ModuleType
from typing import cast

import pytest
import torch

from stochaflow.training import (
    Trainer,
    TrainingPlan,
    bind_training_diagnostic,
)
from stochaflow.training.diagnostics import DiffusionQualityDiagnostic
from stochaflow.training.diagnostics.config import ReferencePipelineConfig
from stochaflow.training.diagnostics.contracts import ReferenceMetricProvider
from stochaflow.training.diagnostics.providers.reference import (
    FIDReferenceMetricProvider,
    KIDReferenceMetricProvider,
    ReferenceMetricSuite,
)
from stochaflow.training.diagnostics.runtime import BoundSampler, SeedPolicy
from stochaflow.training.gaussian import GaussianDenoisingTrainingStrategy
from stochaflow.utils.checkpoint import CheckpointManager

from .helpers import (
    RecordingLogger,
    RecordingSampler,
    TinyDenoiser,
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
        sampling={
            "shape": [1, 4, 4],
            "sample_num": 2,
            "batch_size": 1,
            "seed": 123,
        },
        providers=provider_config(),
        reference=reference,
    )


def _install_fake_torchmetrics(
    monkeypatch,
    *,
    fid_values: Sequence[float] | None = None,
) -> None:
    class FakeMetric:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.real_count = 0
            self.fake_count = 0
            self.device = torch.device("cpu")
            self.update_devices: list[torch.device] = []
            self.compute_device: torch.device | None = None
            self.compute_count = 0
            self.real_updates: list[torch.Tensor] = []

        def to(self, device):
            self.device = torch.device(device)
            return self

        def set_dtype(self, dtype) -> None:
            del dtype

        def update(self, images: torch.Tensor, *, real: bool) -> None:
            torch.rand(1)
            self.update_devices.append(images.device)
            assert images.shape[1] == 3
            assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0
            if real:
                self.real_count += images.shape[0]
                self.real_updates.append(images.detach().cpu().clone())
            else:
                self.fake_count += images.shape[0]

        def reset(self) -> None:
            self.fake_count = 0

    class FakeFID(FakeMetric):
        def compute(self) -> torch.Tensor:
            self.compute_device = self.device
            if fid_values is None:
                value = float(self.real_count + self.fake_count)
            else:
                value = float(fid_values[self.compute_count])
                self.compute_count += 1
            return torch.tensor(value)

    class FakeKID(FakeMetric):
        def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor(0.25), torch.tensor(0.05)

    torchmetrics = ModuleType("torchmetrics")
    image_module = ModuleType("torchmetrics.image")
    fid_module = ModuleType("torchmetrics.image.fid")
    kid_module = ModuleType("torchmetrics.image.kid")
    monkeypatch.setattr(
        fid_module,
        "FrechetInceptionDistance",
        FakeFID,
        raising=False,
    )
    monkeypatch.setattr(
        kid_module,
        "KernelInceptionDistance",
        FakeKID,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", image_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", fid_module)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", kid_module)


def test_fid_uses_cpu_for_mps_without_changing_kid_device(monkeypatch) -> None:
    _install_fake_torchmetrics(monkeypatch)

    fid = FIDReferenceMetricProvider(
        device=torch.device("mps"),
        num_real=2,
        num_fake=2,
    )
    kid = KIDReferenceMetricProvider(
        device=torch.device("mps"),
        num_real=2,
        num_fake=2,
        subsets=1,
        subset_size=2,
    )
    images = torch.zeros(2, 3, 4, 4)

    fid.update(images, real=True)
    fid.update(images, real=False)

    assert fid.metric.device == torch.device("cpu")
    assert fid.metric.update_devices == [torch.device("cpu"), torch.device("cpu")]
    assert fid.compute() == {"fid": 4.0}
    assert fid.metric.compute_device == torch.device("cpu")
    assert kid.metric.device == torch.device("mps")


def test_fid_preserves_parameter_validation_errors(monkeypatch) -> None:
    _install_fake_torchmetrics(monkeypatch)
    fid_module = sys.modules["torchmetrics.image.fid"]

    class InvalidFID:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise ValueError("feature must identify a supported extractor")

    monkeypatch.setattr(fid_module, "FrechetInceptionDistance", InvalidFID)

    with pytest.raises(ValueError, match="supported extractor"):
        FIDReferenceMetricProvider(
            device=torch.device("mps"),
            num_real=2,
            num_fake=2,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_fid_cpu_fallback_accepts_real_mps_images(monkeypatch) -> None:
    _install_fake_torchmetrics(monkeypatch)
    provider = FIDReferenceMetricProvider(
        device=torch.device("mps"),
        num_real=2,
        num_fake=2,
    )

    provider.update(torch.zeros(2, 3, 4, 4, device="mps"), real=True)

    assert provider.metric.update_devices == [torch.device("cpu")]


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

    results = diagnostic.on_train_epoch_end(epoch_event(runtime))

    assert results is not None
    assert {result.source_id for result in results} == {
        "observation",
        "validation_quality",
    }
    requests = {
        request.id: request
        for request in diagnostic.metric_source_requests
    }
    assert requests["observation"].data_role == "external"
    assert requests["validation_quality"].data_role == "validation"
    result_by_source = {
        result.source_id: result.metrics
        for result in results
    }
    assert all(
        key.endswith(("/fid", "/kid_mean", "/kid_std"))
        for key in result_by_source["validation_quality"]
    )
    assert any(
        key.endswith("/reference_fake_samples")
        for key in result_by_source["observation"]
    )
    payload = {
        key: value
        for result in results
        for key, value in result.metrics.items()
    }
    for profile_id in ("first", "second"):
        prefix = f"diagnostics/diffusion_quality/samplers/{profile_id}"
        assert payload[f"{prefix}/fid"] == 5.0
        assert payload[f"{prefix}/kid_mean"] == pytest.approx(0.25)
        assert payload[f"{prefix}/kid_std"] == pytest.approx(0.05)
        assert payload[f"{prefix}/reference_fake_samples"] == 3.0
    assert diagnostic._reference_suite is not None
    for _, provider in diagnostic._reference_suite.providers:
        assert isinstance(
            provider,
            (FIDReferenceMetricProvider, KIDReferenceMetricProvider),
        )
        metric = provider.metric
        assert metric.real_count == 2
        assert metric.fake_count == 0


def test_builtin_reference_cadence_warn_failure_and_best_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_torchmetrics(
        monkeypatch,
        fid_values=(9.0, 7.0),
    )
    RecordingSampler.records.clear()
    logger = RecordingLogger()
    diagnostic = DiffusionQualityDiagnostic(
        logger=logger,
        output_dir=tmp_path,
        samplers=[
            {
                "id": "healthy",
                "name": "test_recording_diagnostic",
                "params": {"marker": "healthy"},
            },
            {
                "id": "broken",
                "name": "test_failing_diagnostic",
            },
        ],
        cadence={"step_every": 100, "artifact_every_epochs": 100},
        sampling={
            "shape": [1, 4, 4],
            "sample_num": 1,
            "batch_size": 1,
            "seed": 123,
        },
        providers={
            "step_metrics": [],
            "sampler_metrics": [],
            "denoiser_artifacts": [],
            "sampler_artifacts": [],
        },
        reference={
            "enabled": True,
            "every_epochs": 2,
            "num_real": 2,
            "num_fake": 2,
            "batch_size": 1,
            "metrics": [
                {
                    "name": "kid",
                    "params": {"subsets": 2, "subset_size": 2},
                },
                {"name": "fid", "params": {}},
            ],
        },
        use_ema=False,
        failure_policy="warn",
    )
    assets = gaussian_system(TinyDenoiser(), num_timesteps=2)
    optimizer = torch.optim.SGD(
        assets.inference_model.parameters(),
        lr=0.0,
    )
    manager = CheckpointManager(
        model=assets.inference_model,
        process=assets.process,
        objective=assets.objective,
        optimizer=optimizer,
    )
    loader = [torch.zeros(2, 1, 4, 4)]
    trainer_runtime = Trainer(
        TrainingPlan(
            strategy=assets.strategy,
            primary_model=assets.inference_model,
            process=assets.process,
            objective=assets.objective,
        ),
        optimizer,
        device="cpu",
        diagnostics=[
            bind_training_diagnostic(
                "diffusion_quality",
                diagnostic,
                protocol_provenance={
                    "schema_version": 1,
                    "data_config": {
                        "name": "in_memory_reference_test",
                        "params": {},
                    },
                    "data_artifacts": None,
                    "extension_plugins": [],
                },
                data_iterables={"validation": loader},
            )
        ],
        logger=logger,
        checkpoint_manager=manager,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    fid_key = (
        "diagnostics/diffusion_quality/samplers/healthy/fid"
    )
    kid_mean_key = (
        "diagnostics/diffusion_quality/samplers/healthy/kid_mean"
    )
    kid_std_key = (
        "diagnostics/diffusion_quality/samplers/healthy/kid_std"
    )

    history = trainer_runtime.fit(
        loader,
        num_epochs=4,
        validation_dataloader=loader,
        show_progress=False,
        early_stopping_monitor=fid_key,
        early_stopping_mode="min",
        monitor_missing="skip",
        track_best=True,
    )

    assert all(
        key not in history[index]
        for index in (0, 2)
        for key in (fid_key, kid_mean_key, kid_std_key)
    )
    assert history[1][fid_key] == pytest.approx(9.0)
    assert history[1][kid_mean_key] == pytest.approx(0.25)
    assert history[1][kid_std_key] == pytest.approx(0.05)
    assert history[3][fid_key] == pytest.approx(7.0)
    assert history[3][kid_mean_key] == pytest.approx(0.25)
    assert history[3][kid_std_key] == pytest.approx(0.05)
    assert history[1][
        "diagnostics/diffusion_quality/system/error_count"
    ] == pytest.approx(1.0)
    assert history[3][
        "diagnostics/diffusion_quality/system/error_count"
    ] == pytest.approx(2.0)
    assert trainer_runtime.monitor_observations == 2
    assert trainer_runtime.best_epoch == 4
    assert trainer_runtime.best_metric_value == pytest.approx(7.0)
    assert trainer_runtime.best_checkpoint_path == (
        tmp_path / "checkpoints" / "best.pt"
    )

    best = CheckpointManager.load_payload(
        tmp_path / "checkpoints" / "best.pt"
    )
    assert best.get("epoch") == 4
    best_metrics = best.get("metrics")
    assert isinstance(best_metrics, dict)
    assert best_metrics[fid_key] == pytest.approx(7.0)
    assert len(logger.text) == 2
    assert all(
        tag == "diagnostics/system/error"
        and "sampling failed" in text
        for tag, text, _ in logger.text
    )


def test_reference_cache_uses_strategy_before_label_and_4d_condition(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_torchmetrics(monkeypatch)

    class MappingReferenceImageStrategy(GaussianDenoisingTrainingStrategy):
        """Test strategy with an explicit mapping-batch image contract."""

        def extract_reference_images(
            self,
            batch: object,
        ) -> torch.Tensor:
            if not isinstance(batch, dict):
                raise TypeError("expected a mapping batch")
            images = batch["image"]
            if not isinstance(images, torch.Tensor):
                raise TypeError("image must be a Tensor")
            return images

    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        reference=_reference_config(),
    )
    assets = gaussian_system(ZeroDenoiser(), num_timesteps=2)
    runtime = trainer(assets)
    runtime.strategy = MappingReferenceImageStrategy(
        assets.inference_model,
        assets.process,
        assets.loss_composer,
    )
    reference_images = -torch.ones(2, 1, 4, 4)

    diagnostic.on_fit_start(
        fit_event(
            runtime,
            validation=[
                {
                    "class_label": torch.tensor([1, 0]),
                    "condition": torch.ones(2, 3, 4, 4),
                    "image": reference_images,
                }
            ],
        )
    )

    assert diagnostic._reference_suite is not None
    for _, provider in diagnostic._reference_suite.providers:
        assert isinstance(
            provider,
            (FIDReferenceMetricProvider, KIDReferenceMetricProvider),
        )
        assert torch.count_nonzero(provider.metric.real_updates[0]) == 0


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
        extract_images=lambda batch: cast(torch.Tensor, batch),
    )

    suite.cache_real(
        [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)]
    )
    result = suite.evaluate(
        profile_id="profile",
        sampler=cast(BoundSampler, torch.nn.Identity()),
        sample_shape=(1, 4, 4),
        visual_samples=torch.zeros(2, 1, 4, 4),
    )

    assert errors == [("reference_real_update", "failing", "real update failed")]
    assert healthy.real == 2
    assert result.metrics["diagnostics/samplers/profile/healthy"] == 4.0
    assert "diagnostics/samplers/profile/failed" not in result.metrics
    assert failing.fake_updates == 0
