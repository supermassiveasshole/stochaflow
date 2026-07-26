from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml

import stochaflow.utils.plugins as plugin_runtime
from stochaflow.data import (
    DataArtifactBinding,
    DataArtifactBindings,
    DataLoaders,
    ManagedDataArtifactIdentity,
)
from stochaflow.sampling.runtime import (
    ResolvedSamplingInputs,
    SamplingRunResult,
)
from stochaflow.training.diagnostics import ReferenceMetricProvider
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
)
from stochaflow.utils.config import load_config, load_config_dict
from stochaflow.utils.factory import build_model, build_process
from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionSelectionPolicy,
    ResolvedExtensions,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_SHOWCASE = _REPOSITORY / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _SHOWCASE / "src"
_PRODUCTION_CONFIG = (
    _SHOWCASE / "experiments" / "production" / "train-adm-128.yaml"
)
_EVALUATION_CONFIG = (
    _SHOWCASE
    / "experiments"
    / "evaluation"
    / "ddim50-cfg2-kid-fid.yaml"
)

example_src = str(_EXAMPLE_SRC)
if example_src not in sys.path:
    sys.path.insert(0, example_src)

evaluation = importlib.import_module(
    "stochaflow_afhq_v2.tools.evaluation"
)
evaluation_metrics = importlib.import_module(
    "stochaflow_afhq_v2.tools.evaluation_metrics"
)
build_argument_parser = importlib.import_module(
    "stochaflow_afhq_v2.tools.evaluate"
).build_argument_parser


class FakeReferenceMetricProvider(ReferenceMetricProvider):
    """Small injected provider that never imports optional quality packages."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.real_total = 0.0
        self.fake_total = 0.0
        self.real_count = 0
        self.fake_count = 0
        self.reset_count = 0

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        total = float(images.sum())
        if real:
            self.real_total += total
            self.real_count += images.shape[0]
        else:
            self.fake_total += total
            self.fake_count += images.shape[0]

    def compute(self) -> dict[str, float]:
        assert self.real_count > 0
        assert self.fake_count > 0
        return {
            f"{self.name}_score": abs(
                self.real_total / self.real_count
                - self.fake_total / self.fake_count
            )
        }

    def reset_fake(self) -> None:
        self.fake_total = 0.0
        self.fake_count = 0
        self.reset_count += 1


class CleanupFailingReferenceMetricProvider(FakeReferenceMetricProvider):
    """Provider exposing update and cleanup failures for lifecycle tests."""

    def __init__(self, name: str, *, fail_fake_update: bool) -> None:
        super().__init__(name)
        self.fail_fake_update = fail_fake_update

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        if not real and self.fail_fake_update:
            raise RuntimeError("primary fake update failure")
        super().update(images, real=real)

    def reset_fake(self) -> None:
        self.reset_count += 1
        raise RuntimeError(f"{self.name} reset failure")


@dataclass
class FakeDistribution:
    """Installed distribution metadata used by plugin discovery."""

    name: str = "stochaflow-afhq-v2"
    version: str = "0.1.0"

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


@dataclass
class FakeEntryPoint:
    """AFHQ-v2 extension entry point used by the real sampling lifecycle."""

    name: str = "stochaflow-afhq-v2"
    value: str = "stochaflow_afhq_v2.stochaflow_ext"

    @property
    def dist(self) -> FakeDistribution:
        return FakeDistribution()


def _identity() -> ManagedDataArtifactIdentity:
    return ManagedDataArtifactIdentity(
        artifact_type="image-folder",
        source_name="afhq-v2.official",
        source_digest="1" * 64,
        materializer_name="afhq-v2.prepare",
        materialization_digest="2" * 64,
        artifact_digest="3" * 64,
        manifest_sha256="4" * 64,
    )


def _bindings() -> DataArtifactBindings:
    return DataArtifactBindings(
        (DataArtifactBinding(id="source", identity=_identity()),)
    )


def _small_evaluation_config(path: Path) -> Path:
    raw = yaml.safe_load(_EVALUATION_CONFIG.read_text(encoding="utf-8"))
    raw["sampling"]["num_samples"] = 6
    raw["sampling"]["batch_size"] = 3
    raw["sampling"]["seed"] = 123
    raw["sampling"]["builder"]["params"]["conditions"] = [
        {"class_label": 0, "count": 2},
        {"class_label": 1, "count": 2},
        {"class_label": 2, "count": 2},
    ]
    raw["evaluation"]["real_per_class"] = 2
    raw["evaluation"]["fake_per_class"] = 2
    raw["evaluation"]["metric_batch_size"] = 2
    raw["evaluation"]["metric_seed"] = 456
    raw["evaluation"]["metrics"][0]["params"]["subset_size"] = 2
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _resolved_inputs(
    overlay_path: Path,
    checkpoint_path: Path,
    bindings: DataArtifactBindings,
) -> ResolvedSamplingInputs:
    base = load_config(_PRODUCTION_CONFIG)
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    merged = base.to_dict()
    merged["extensions"] = overlay["extensions"]
    merged["sampling"] = overlay["sampling"]
    config = load_config_dict(merged)
    plan = ExtensionActivationPlan(
        config=config,
        provenance=(),
        version_mismatches=(),
        selection_policy=ExtensionSelectionPolicy.EXACT,
    )
    return ResolvedSamplingInputs(
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint={
            "format_version": 9,
            "metadata": {"data_artifacts": bindings.to_dict()},
        },
        config_source="sampling-overlay",
        extension_plan=plan,
    )


def _tiny_evaluation_config(path: Path) -> Path:
    _small_evaluation_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = raw["sampling"]["builder"]["params"]
    params["weights"] = "raw"
    params["sampler"]["params"]["num_inference_steps"] = 2
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _tiny_checkpoint(
    path: Path,
    bindings: DataArtifactBindings,
) -> Path:
    raw = load_config(_PRODUCTION_CONFIG).to_dict()
    raw["model"] = {
        "name": "adm_unet",
        "params": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 8,
            "channel_multipliers": [1],
            "num_res_blocks": 1,
            "transformer_depths": [0],
            "middle_transformer_depth": 0,
            "attention_head_dim": 8,
            "time_embedding_dim": 32,
            "num_classes": 3,
            "dropout": 0.0,
            "scale_shift_norm": True,
            "residual_resampling": True,
            "zero_init_residual": True,
            "zero_init_output": True,
        },
    }
    raw["process"] = {
        "name": "discrete_gaussian",
        "params": {
            "schedule": {
                "name": "linear_beta",
                "params": {
                    "num_timesteps": 4,
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                },
            }
        },
    }
    raw["trainer"]["precision"] = "fp32"
    config = load_config_dict(raw)
    model = build_model(config.model)
    assert config.process is not None
    process = build_process(config.process)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "process_state_dict": process.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config.to_dict(),
        "metadata": {
            "extension_plugins": [
                {
                    "name": "stochaflow-afhq-v2",
                    "distribution": "stochaflow-afhq-v2",
                    "version": "0.1.0",
                    "target": "stochaflow_afhq_v2.stochaflow_ext",
                }
            ],
            "data_artifacts": bindings.to_dict(),
        },
        "precision_kind": "fp32",
        "inference_asset_descriptors": {},
        "epoch": 1,
        "global_step": 3,
    }
    torch.save(payload, path)
    return path


def test_checked_in_evaluation_protocol_is_frozen_and_balanced() -> None:
    document = evaluation.load_evaluation_document(_EVALUATION_CONFIG)
    protocol = document.protocol
    sampling = document.sampling_overlay["sampling"]
    params = sampling["builder"]["params"]

    assert protocol.class_mapping == {"cat": 0, "dog": 1, "wild": 2}
    assert protocol.split == "test"
    assert protocol.real_per_class == protocol.fake_per_class == 300
    assert [(spec.name, spec.params) for spec in protocol.metrics] == [
        ("kid", {"subsets": 100, "subset_size": 300}),
        ("fid", {"feature": 2048}),
    ]
    assert sampling["num_samples"] == 900
    assert sampling["seed"] == 20260726
    assert params["weights"] == "ema"
    assert params["guidance_scale"] == 2.0
    assert params["conditions"] == [
        {"class_label": 0, "count": 300},
        {"class_label": 1, "count": 300},
        {"class_label": 2, "count": 300},
    ]
    assert params["sampler"] == {
        "name": "ddim",
        "params": {"num_inference_steps": 50, "eta": 0.0},
    }
    assert params["trajectory"]["enabled"] is False


def test_evaluation_uses_core_sampling_strict_test_data_and_fake_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _small_evaluation_config(tmp_path / "evaluation.yaml")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"frozen-checkpoint")
    bindings = _bindings()
    calls: dict[str, Any] = {}
    events: list[str] = []

    def fake_resolve(
        *,
        config_path: str | Path | None,
        checkpoint: str | Path | None,
    ) -> ResolvedSamplingInputs:
        assert config_path is not None
        assert checkpoint is not None
        snapshot_path = Path(checkpoint)
        assert snapshot_path != checkpoint_path
        assert snapshot_path.read_bytes() == b"frozen-checkpoint"
        replacement = tmp_path / "replacement.pt"
        replacement.write_bytes(b"atomically-replaced-checkpoint")
        replacement.replace(checkpoint_path)
        events.append("resolve")
        calls["resolve"] = Path(config_path).read_text(encoding="utf-8")
        return _resolved_inputs(Path(config_path), snapshot_path, bindings)

    def fake_activate(
        plan: ExtensionActivationPlan,
        **kwargs: Any,
    ) -> ResolvedExtensions:
        events.append("activate")
        calls["activate"] = kwargs
        return ResolvedExtensions(
            config=plan.config,
            provenance=(),
            acceptance_audit=(),
        )

    test_batches = [
        (
            torch.full((3, 3, 128, 128), -0.5),
            {"class_label": torch.tensor([0, 1, 2], dtype=torch.long)},
        ),
        (
            torch.full((3, 3, 128, 128), 0.5),
            {"class_label": torch.tensor([0, 1, 2], dtype=torch.long)},
        ),
    ]

    def fake_build_data(
        config: Any,
        *,
        seed: int,
        strict_resume: bool,
        expected_artifacts: DataArtifactBindings,
    ) -> DataLoaders:
        events.append("data")
        calls["data"] = {
            "name": config.name,
            "seed": seed,
            "strict_resume": strict_resume,
            "expected": expected_artifacts,
        }
        return DataLoaders(
            train=[0],
            test=test_batches,
            artifact_bindings=bindings,
        )

    def fake_run_sampling(
        inputs: ResolvedSamplingInputs,
        extensions: ResolvedExtensions,
        *,
        output_dir: str | Path | None,
        device_name: str | None,
    ) -> SamplingRunResult:
        assert inputs.config.data.name == "afhq-v2.class-images"
        assert extensions.config.sampling.num_samples == 6
        assert device_name == "cpu"
        events.append("sampling")
        target = Path(cast(Path, output_dir))
        target.mkdir(parents=True)
        samples_path = target / "samples.pt"
        samples = torch.cat(
            (
                torch.full((2, 3, 128, 128), -1.0),
                torch.zeros((2, 3, 128, 128)),
                torch.ones((2, 3, 128, 128)),
            )
        )
        torch.save(samples, samples_path)
        manifest_path = target / "resolved_sampling.yaml"
        manifest_path.write_text("kind: sampling\n", encoding="utf-8")
        metadata = {
            "builder": "class_conditional_denoising",
            "weights": "ema",
            "guidance_scale": 2.0,
            "conditions": [
                {"class_label": 0, "count": 2},
                {"class_label": 1, "count": 2},
                {"class_label": 2, "count": 2},
            ],
            "sampler": {
                "name": "ddim",
                "params": {"num_inference_steps": 50, "eta": 0.0},
            },
        }
        calls["sampling"] = target
        return SamplingRunResult(
            checkpoint_path=checkpoint_path,
            output_dir=target,
            builder_name="class_conditional_denoising",
            device=torch.device("cpu"),
            seed=123,
            metadata=metadata,
            artifacts={"samples": samples_path, "config": manifest_path},
        )

    created: list[tuple[str, int, int, FakeReferenceMetricProvider]] = []

    def fake_provider_factory(
        spec: Any,
        device: torch.device,
        num_real: int,
        num_fake: int,
    ) -> ReferenceMetricProvider:
        assert device == torch.device("cpu")
        events.append(f"provider:{num_real}:{num_fake}")
        provider = FakeReferenceMetricProvider(spec.name)
        created.append((spec.name, num_real, num_fake, provider))
        return provider

    monkeypatch.setattr(evaluation, "resolve_sampling_inputs", fake_resolve)
    monkeypatch.setattr(evaluation, "activate_extension_plugins", fake_activate)
    monkeypatch.setattr(evaluation, "build_data_loaders", fake_build_data)
    monkeypatch.setattr(evaluation, "run_resolved_sampling", fake_run_sampling)
    monkeypatch.setattr(
        evaluation,
        "_checkpoint_progress",
        lambda path: {"epoch": 17, "global_step": 420},
    )

    output_dir = tmp_path / "evaluation-result"
    result = evaluation.evaluate_checkpoint(
        config_path=config_path,
        checkpoint=checkpoint_path,
        output_dir=output_dir,
        device_name="cpu",
        provider_factory=fake_provider_factory,
    )

    assert set(calls) == {"resolve", "activate", "data", "sampling"}
    assert events[:7] == [
        "resolve",
        "provider:6:6",
        "provider:6:6",
        "provider:2:2",
        "provider:2:2",
        "activate",
        "data",
    ]
    assert events[7] == "sampling"
    assert calls["data"] == {
        "name": "afhq-v2.class-images",
        "seed": 20260726,
        "strict_resume": True,
        "expected": bindings,
    }
    assert len(created) == 12
    assert sorted((name, real, fake) for name, real, fake, _ in created) == [
        ("fid", 2, 2),
        ("fid", 2, 2),
        ("fid", 2, 2),
        ("fid", 2, 2),
        ("fid", 6, 6),
        ("fid", 6, 6),
        ("kid", 2, 2),
        ("kid", 2, 2),
        ("kid", 2, 2),
        ("kid", 2, 2),
        ("kid", 6, 6),
        ("kid", 6, 6),
    ]
    assert all(provider.reset_count == 1 for *_, provider in created)

    payload = json.loads(result.result_path.read_text(encoding="utf-8"))
    encoded = result.result_path.read_bytes()
    assert result.result_sha256 == hashlib.sha256(encoded).hexdigest()
    assert result.digest_path.read_text(encoding="ascii") == (
        f"{result.result_sha256}  evaluation-result.json\n"
    )
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))[
        "result"
    ]["sha256"] == result.result_sha256
    assert payload["checkpoint"]["sha256"] == hashlib.sha256(
        b"frozen-checkpoint"
    ).hexdigest()
    assert payload["checkpoint"]["path"] == str(checkpoint_path.resolve())
    assert checkpoint_path.read_bytes() == b"atomically-replaced-checkpoint"
    assert payload["checkpoint"]["weights"] == "ema"
    assert payload["checkpoint"]["epoch"] == 17
    assert payload["checkpoint"]["global_step"] == 420
    assert payload["extensions"]["extension_plugins"] == []
    assert payload["protocol"]["allocation"] == {
        "real": {"cat": 2, "dog": 2, "wild": 2},
        "fake": {"cat": 2, "dog": 2, "wild": 2},
    }
    assert payload["data"]["artifact_bindings"] == bindings.to_dict()
    assert payload["protocol"]["sampling_seed"] == 123
    assert payload["protocol"]["selection"] == {
        "real": "authenticated-manifest-order",
        "fake": "ordered-class-label-blocks",
    }
    assert set(payload["metrics"]["aggregate"]) == {"kid_score", "fid_score"}
    assert set(payload["metrics"]["per_class"]) == {"cat", "dog", "wild"}
    assert payload["sampling"]["artifacts"]["samples"]["sha256"] == (
        hashlib.sha256(
            (output_dir / "sampling" / "samples.pt").read_bytes()
        ).hexdigest()
    )


def test_evaluation_requires_explicit_frozen_weights(tmp_path: Path) -> None:
    config_path = _small_evaluation_config(tmp_path / "evaluation.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["sampling"]["builder"]["params"]["weights"] = "auto"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly select raw or ema"):
        evaluation.load_evaluation_document(config_path)


def test_checkpoint_progress_requires_positive_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation.CheckpointManager,
        "load_payload",
        lambda *args, **kwargs: {"epoch": 0, "global_step": 0},
    )

    with pytest.raises(ValueError, match="epoch must be positive"):
        evaluation._checkpoint_progress(tmp_path / "best.pt")


def test_unavailable_execution_device_fails_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _small_evaluation_config(tmp_path / "evaluation.yaml")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    bindings = _bindings()
    output_dir = tmp_path / "must-not-exist"
    calls: list[str] = []

    def fake_resolve(
        *,
        config_path: str | Path | None,
        checkpoint: str | Path | None,
    ) -> ResolvedSamplingInputs:
        assert config_path is not None
        assert checkpoint is not None
        snapshot_path = Path(checkpoint)
        assert snapshot_path != checkpoint_path
        assert snapshot_path.read_bytes() == b"checkpoint"
        return _resolved_inputs(Path(config_path), snapshot_path, bindings)

    def provider_factory(*args: Any, **kwargs: Any) -> ReferenceMetricProvider:
        del args, kwargs
        calls.append("provider")
        return FakeReferenceMetricProvider("unexpected")

    monkeypatch.setattr(evaluation, "resolve_sampling_inputs", fake_resolve)
    monkeypatch.setattr(
        evaluation,
        "_checkpoint_progress",
        lambda path: {"epoch": 1, "global_step": 0},
    )
    monkeypatch.setattr(
        evaluation,
        "activate_extension_plugins",
        lambda *args, **kwargs: calls.append("activate"),
    )
    monkeypatch.setattr(
        evaluation,
        "build_data_loaders",
        lambda *args, **kwargs: calls.append("data"),
    )
    monkeypatch.setattr(
        evaluation,
        "run_resolved_sampling",
        lambda *args, **kwargs: calls.append("sampling"),
    )

    with pytest.raises(
        ValueError,
        match=r"CUDA (execution requires|device index)",
    ):
        evaluation.evaluate_checkpoint(
            config_path=config_path,
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            device_name="cuda:999999",
            provider_factory=provider_factory,
        )

    assert calls == []
    assert not output_dir.exists()


def test_quality_dependency_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise RuntimeError(
            "FID requires the optional 'quality' dependencies and available "
            "Inception weights"
        )

    monkeypatch.setattr(
        evaluation_metrics.DIAGNOSTIC_PROVIDERS.reference_metrics,
        "create",
        unavailable,
    )
    spec = evaluation.AFHQV2MetricSpec("fid", {"feature": 2048})

    with pytest.raises(
        RuntimeError,
        match=r"evaluation metric 'fid' is unavailable.*quality",
    ):
        evaluation_metrics.default_provider_factory(
            spec,
            torch.device("cpu"),
            300,
            300,
        )


def test_evaluation_cli_requires_checkpoint_and_protocol() -> None:
    args = build_argument_parser().parse_args(
        [
            "--config",
            str(_EVALUATION_CONFIG),
            "--checkpoint",
            "best.pt",
            "--device",
            "cpu",
        ]
    )
    assert args.config == _EVALUATION_CONFIG
    assert args.checkpoint == Path("best.pt")
    assert args.device == "cpu"


def test_metric_cleanup_preserves_primary_and_releases_every_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = evaluation.load_evaluation_document(
        _small_evaluation_config(tmp_path / "evaluation.yaml")
    )
    real_images = {
        name: torch.zeros((2, 3, 4, 4))
        for name in document.protocol.class_mapping
    }
    fake_images = {
        name: torch.ones((2, 3, 4, 4))
        for name in document.protocol.class_mapping
    }
    providers: list[CleanupFailingReferenceMetricProvider] = []

    def factory(
        spec: Any,
        device: torch.device,
        num_real: int,
        num_fake: int,
    ) -> ReferenceMetricProvider:
        del device, num_real, num_fake
        provider = CleanupFailingReferenceMetricProvider(
            spec.name,
            fail_fake_update=spec.name == "kid",
        )
        providers.append(provider)
        return provider

    releases: list[torch.device] = []

    def failing_release(device: torch.device) -> None:
        releases.append(device)
        raise RuntimeError("release failure")

    monkeypatch.setattr(
        evaluation_metrics,
        "release_metric_device",
        failing_release,
    )

    with pytest.raises(
        RuntimeError,
        match="aggregate fake update: primary fake update failure",
    ) as error:
        evaluation_metrics.evaluate_reference_metrics(
            real_images=real_images,
            fake_images=fake_images,
            protocol=document.protocol,
            device=torch.device("cpu"),
            factory=factory,
        )

    notes = getattr(error.value, "__notes__", ())
    assert any("kid reset failure" in note for note in notes)
    assert any("fid reset failure" in note for note in notes)
    assert any("release failure" in note for note in notes)
    assert [provider.reset_count for provider in providers] == [1, 1]
    assert releases == [torch.device("cpu")]


def test_aggregate_metrics_stream_class_groups_without_cat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = evaluation.load_evaluation_document(
        _small_evaluation_config(tmp_path / "evaluation.yaml")
    )
    real_images = {
        name: torch.full((2, 3, 4, 4), float(label))
        for name, label in document.protocol.class_mapping.items()
    }
    fake_images = {
        name: torch.full((2, 3, 4, 4), float(label + 1))
        for name, label in document.protocol.class_mapping.items()
    }

    def forbidden_cat(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("aggregate evaluation must not concatenate groups")

    monkeypatch.setattr(evaluation_metrics.torch, "cat", forbidden_cat)

    metrics, identities = evaluation_metrics.evaluate_reference_metrics(
        real_images=real_images,
        fake_images=fake_images,
        protocol=document.protocol,
        device=torch.device("cpu"),
        factory=lambda spec, device, num_real, num_fake: (
            FakeReferenceMetricProvider(spec.name)
        ),
    )

    assert set(metrics["aggregate"]) == {"kid_score", "fid_score"}
    assert set(metrics["per_class"]) == {"cat", "dog", "wild"}
    assert set(identities) == {"kid", "fid"}


def test_sampling_failure_never_publishes_formal_evaluation_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _small_evaluation_config(tmp_path / "evaluation.yaml")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    bindings = _bindings()
    output_dir = tmp_path / "evaluation-result"

    def fake_resolve(
        *,
        config_path: str | Path | None,
        checkpoint: str | Path | None,
    ) -> ResolvedSamplingInputs:
        assert config_path is not None
        assert checkpoint is not None
        return _resolved_inputs(
            Path(config_path),
            Path(checkpoint),
            bindings,
        )

    def failing_sampling(
        inputs: ResolvedSamplingInputs,
        extensions: ResolvedExtensions,
        *,
        output_dir: str | Path | None,
        device_name: str | None,
    ) -> SamplingRunResult:
        del inputs, extensions, device_name
        partial = Path(cast(Path, output_dir))
        partial.mkdir(parents=True)
        (partial / "partial.pt").write_bytes(b"incomplete")
        raise RuntimeError("sampling failed")

    test_batches = [
        (
            torch.zeros((6, 3, 128, 128)),
            {
                "class_label": torch.tensor(
                    [0, 0, 1, 1, 2, 2],
                    dtype=torch.long,
                )
            },
        )
    ]
    monkeypatch.setattr(evaluation, "resolve_sampling_inputs", fake_resolve)
    monkeypatch.setattr(
        evaluation,
        "_checkpoint_progress",
        lambda path: {"epoch": 1, "global_step": 0},
    )
    monkeypatch.setattr(
        evaluation,
        "activate_extension_plugins",
        lambda plan, **kwargs: ResolvedExtensions(
            config=plan.config,
            provenance=(),
            acceptance_audit=(),
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "build_data_loaders",
        lambda *args, **kwargs: DataLoaders(
            train=[0],
            test=test_batches,
            artifact_bindings=bindings,
        ),
    )
    monkeypatch.setattr(evaluation, "run_resolved_sampling", failing_sampling)

    with pytest.raises(RuntimeError, match="sampling failed"):
        evaluation.evaluate_checkpoint(
            config_path=config_path,
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            device_name="cpu",
            provider_factory=lambda spec, device, num_real, num_fake: (
                FakeReferenceMetricProvider(spec.name)
            ),
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".evaluation-result.staging-*"))


def test_atomic_publish_does_not_replace_concurrent_destination(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".result.staging"
    staging.mkdir()
    (staging / "evaluation-result.json").write_text(
        "{}",
        encoding="utf-8",
    )
    destination = tmp_path / "result"
    destination.mkdir()
    foreign = destination / "foreign-owner.txt"
    foreign.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        evaluation._atomic_publish_directory(staging, destination)

    assert staging.is_dir()
    assert foreign.read_text(encoding="utf-8") == "preserve"
    assert not (destination / "evaluation-result.json").exists()


def test_real_tiny_checkpoint_and_core_sampling_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    config_path = _tiny_evaluation_config(tmp_path / "evaluation.yaml")
    checkpoint_path = _tiny_checkpoint(tmp_path / "best.pt", bindings)
    output_dir = tmp_path / "evaluation-result"
    test_batches = [
        (
            torch.linspace(
                -1.0,
                1.0,
                6 * 3 * 128 * 128,
            ).reshape(6, 3, 128, 128),
            {
                "class_label": torch.tensor(
                    [0, 0, 1, 1, 2, 2],
                    dtype=torch.long,
                )
            },
        )
    ]

    def discover(*, group: str) -> tuple[FakeEntryPoint, ...]:
        assert group == "stochaflow.extensions"
        return (FakeEntryPoint(),)

    def fake_build_data(
        config: Any,
        *,
        seed: int,
        strict_resume: bool,
        expected_artifacts: DataArtifactBindings,
    ) -> DataLoaders:
        assert config.name == "afhq-v2.class-images"
        assert seed == 20260726
        assert strict_resume is True
        assert expected_artifacts == bindings
        return DataLoaders(
            train=[0],
            test=test_batches,
            artifact_bindings=bindings,
        )

    monkeypatch.setattr(plugin_runtime.metadata, "entry_points", discover)
    monkeypatch.setattr(evaluation, "build_data_loaders", fake_build_data)
    plugin_runtime._reset_extension_activation_state_for_testing()
    try:
        result = evaluation.evaluate_checkpoint(
            config_path=config_path,
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            device_name="cpu",
            provider_factory=lambda spec, device, num_real, num_fake: (
                FakeReferenceMetricProvider(spec.name)
            ),
        )
    finally:
        plugin_runtime._reset_extension_activation_state_for_testing()

    samples_path = result.sampling.artifacts["samples"]
    samples = torch.load(samples_path, map_location="cpu", weights_only=True)
    payload = json.loads(result.result_path.read_text(encoding="utf-8"))
    resolved_sampling = yaml.safe_load(
        result.sampling.artifacts["config"].read_text(encoding="utf-8")
    )

    assert result.output_dir == output_dir.resolve()
    assert samples.shape == (6, 3, 128, 128)
    assert payload["checkpoint"]["path"] == str(checkpoint_path.resolve())
    assert payload["checkpoint"]["epoch"] == 1
    assert payload["checkpoint"]["global_step"] == 3
    assert payload["checkpoint"]["weights"] == "raw"
    assert resolved_sampling["checkpoint"] == str(checkpoint_path.resolve())
    assert not list(output_dir.rglob("checkpoint.snapshot.pt"))
    assert result.digest_path.is_file()
    assert result.manifest_path.is_file()
