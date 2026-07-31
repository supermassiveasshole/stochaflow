"""Tests for the stable third-party extension API."""

from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Metric

from stochaflow import data, metrics, models, processes, sampling, training
from stochaflow import extensions as public
from stochaflow.scripts import experiment_runner
from stochaflow.training import diagnostics
from stochaflow.utils import config, logging, plugins, registry


@dataclass(frozen=True, slots=True)
class ExtensionVectorBatch:
    """A plugin-owned batch shape that core must not interpret."""

    prediction: torch.Tensor
    reference: torch.Tensor


class ExtensionRelativeL2Metric(Metric):
    """A third-party metric with explicit additive distributed state."""

    relative_error: torch.Tensor
    observations: torch.Tensor

    def __init__(self) -> None:
        super().__init__(
            dist_sync_on_step=False,
            sync_on_compute=True,
        )
        self.add_state(
            "relative_error",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "observations",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        numerator = torch.linalg.vector_norm(
            prediction - reference,
            dim=-1,
        )
        denominator = torch.linalg.vector_norm(reference, dim=-1).clamp_min(
            torch.finfo(reference.dtype).eps
        )
        relative = numerator / denominator
        self.relative_error += relative.sum()
        self.observations += relative.numel()

    def compute(self) -> torch.Tensor:
        return self.relative_error / self.observations


class ExtensionVectorStrategy(public.TrainingStrategy):
    """A third-party strategy that owns its structured-batch interpretation."""

    @property
    def metric_channels(self) -> frozenset[str]:
        """Declare the plugin-owned update channel."""

        return frozenset({"example.vector_pair"})

    def training_step(self, batch: Any) -> public.TrainStepOutput:
        """Produce optimization loss and an opaque metric update."""

        if not isinstance(batch, ExtensionVectorBatch):
            raise TypeError("extension strategy requires ExtensionVectorBatch")
        error = batch.prediction - batch.reference
        return public.TrainStepOutput(
            loss=error.square().mean(),
            metric_updates={
                "example.vector_pair": public.MetricUpdate(
                    args=(batch.prediction, batch.reference),
                )
            },
        )


EXTENSION_METRIC_NAME = "test.extensions.relative_l2"
public.REGISTRIES.metrics.add(
    EXTENSION_METRIC_NAME,
    ExtensionRelativeL2Metric,
)

VERTICAL_PLUGIN_NAME = "vertical-metrics"
VERTICAL_PLUGIN_DISTRIBUTION = "vertical-metrics-extension"
VERTICAL_PLUGIN_VERSION = "2.3.1"
VERTICAL_PLUGIN_TARGET = "fixtures.metrics_extension"
VERTICAL_MODEL_NAME = "test.vertical-extension.linear-model"
VERTICAL_TRAINING_BUILDER_NAME = "test.vertical-extension.vector-training"
VERTICAL_METRIC_NAME = "test.vertical-extension.relative-l2"
VERTICAL_METRIC_CHANNEL = "test.vertical-extension.vector-pair"


@dataclass(frozen=True, slots=True)
class InstalledExtensionDistribution:
    """Distribution metadata exposed by one discovered test entry point."""

    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        """Expose the canonical package metadata field used by discovery."""

        return {"Name": self.name}


@dataclass(frozen=True, slots=True)
class InstalledExtensionEntryPoint:
    """Installed entry-point metadata for the vertical extension fixture."""

    name: str
    value: str
    distribution: str
    version: str

    @property
    def dist(self) -> InstalledExtensionDistribution:
        """Return the owning distribution metadata."""

        return InstalledExtensionDistribution(
            name=self.distribution,
            version=self.version,
        )


def test_public_extension_contracts_reexport_runtime_types() -> None:
    expected = {
        "DIAGNOSTIC_PROVIDERS": diagnostics.DIAGNOSTIC_PROVIDERS,
        "ArtifactRecord": diagnostics.ArtifactRecord,
        "ArtifactVerificationEvent": data.ArtifactVerificationEvent,
        "ArtifactVerificationObserver": data.ArtifactVerificationObserver,
        "ArtifactVerificationPhase": data.ArtifactVerificationPhase,
        "ComponentConfig": config.ComponentConfig,
        "ConfigError": config.ConfigError,
        "ContextAwareDiagnostic": training.ContextAwareDiagnostic,
        "ClassLabeledImageFileRecord": data.ClassLabeledImageFileRecord,
        "ClassLabeledImageFolderArtifactPayload": (
            data.ClassLabeledImageFolderArtifactPayload
        ),
        "DataArtifact": data.DataArtifact,
        "DataArtifactBinding": data.DataArtifactBinding,
        "DataArtifactBindings": data.DataArtifactBindings,
        "DataArtifactIdentity": data.DataArtifactIdentity,
        "DataArtifactLoadContext": data.DataArtifactLoadContext,
        "DataArtifactStore": data.DataArtifactStore,
        "DataArtifactValidationError": data.DataArtifactValidationError,
        "DataBuilder": data.DataBuilder,
        "DataBuilderContext": data.DataBuilderContext,
        "DataLoaders": data.DataLoaders,
        "DataSource": data.DataSource,
        "DataSourceContext": data.DataSourceContext,
        "DataSourceMaterializationConfig": (
            data.DataSourceMaterializationConfig
        ),
        "DDIMSampler": sampling.DDIMSampler,
        "DDPMAncestralSampler": sampling.DDPMAncestralSampler,
        "DenoiserChannelLayout": models.DenoiserChannelLayout,
        "DenoiserArtifactContext": diagnostics.DenoiserArtifactContext,
        "DenoiserArtifactProvider": diagnostics.DenoiserArtifactProvider,
        "DiagnosticBuildContext": training.DiagnosticBuildContext,
        "DiagnosticCadenceConfig": diagnostics.DiagnosticCadenceConfig,
        "DiagnosticProviderCatalog": diagnostics.DiagnosticProviderCatalog,
        "DiagnosticSamplingConfig": diagnostics.DiagnosticSamplingConfig,
        "DeviceTransferableBatch": training.DeviceTransferableBatch,
        "DiffusionQualityConfig": diagnostics.DiffusionQualityConfig,
        "ExperimentLogger": logging.ExperimentLogger,
        "ExtensionActivationError": plugins.ExtensionActivationError,
        "ExtensionActivationPlan": plugins.ExtensionActivationPlan,
        "ExtensionActivationStateError": plugins.ExtensionActivationStateError,
        "ExtensionDiscoveryError": plugins.ExtensionDiscoveryError,
        "ExtensionIdentityError": plugins.ExtensionIdentityError,
        "ExtensionPluginError": plugins.ExtensionPluginError,
        "ExtensionPluginProvenance": plugins.ExtensionPluginProvenance,
        "ExtensionSelectionPolicy": plugins.ExtensionSelectionPolicy,
        "ExtensionVersionAcceptance": plugins.ExtensionVersionAcceptance,
        "ExtensionVersionMismatch": plugins.ExtensionVersionMismatch,
        "ExtensionVersionMismatchError": plugins.ExtensionVersionMismatchError,
        "ExtensionVersionPolicy": plugins.ExtensionVersionPolicy,
        "FitStartEvent": training.FitStartEvent,
        "DiscreteGaussianDenoisingProcess": (
            processes.DiscreteGaussianDenoisingProcess
        ),
        "DiscreteGaussianProcess": processes.DiscreteGaussianProcess,
        "DiscreteVPCoefficients": processes.DiscreteVPCoefficients,
        "DiscreteVPSchedule": processes.DiscreteVPSchedule,
        "GaussianDiagnosticSemantics": training.GaussianDiagnosticSemantics,
        "GaussianDenoisingDynamics": sampling.GaussianDenoisingDynamics,
        "GaussianModelDynamics": sampling.GaussianModelDynamics,
        "GaussianNoiseSchedule": processes.GaussianNoiseSchedule,
        "GaussianPrediction": sampling.GaussianPrediction,
        "GaussianScales": processes.GaussianScales,
        "GaussianTransition": sampling.GaussianTransition,
        "GenerativeDynamics": sampling.GenerativeDynamics,
        "InferenceAssetProjection": training.InferenceAssetProjection,
        "InferenceAssetProvider": sampling.InferenceAssetProvider,
        "InferenceModelProvider": sampling.InferenceModelProvider,
        "IMAGE_DATA_SOURCES": data.IMAGE_DATA_SOURCES,
        "ImageDataSource": data.ImageDataSource,
        "ImageDimensionTable": data.ImageDimensionTable,
        "ImageDimensions": data.ImageDimensions,
        "ImageFilePair": data.ImageFilePair,
        "ImageFileRecord": data.ImageFileRecord,
        "ImageFolderArtifactPayload": data.ImageFolderArtifactPayload,
        "ManagedDataArtifactBuild": data.ManagedDataArtifactBuild,
        "MSEObjective": training.MSEObjective,
        "ManagedTrainingModule": training.ManagedTrainingModule,
        "MetricChannelProvider": training.MetricChannelProvider,
        "MetricUpdate": metrics.MetricUpdate,
        "PerSampleObjective": training.PerSampleObjective,
        "PredictionType": sampling.PredictionType,
        "Process": processes.Process,
        "ProviderPipelineConfig": diagnostics.ProviderPipelineConfig,
        "ProviderSpec": diagnostics.ProviderSpec,
        "ProviderValidationContext": diagnostics.ProviderValidationContext,
        "PairedImageFolderArtifactPayload": (
            data.PairedImageFolderArtifactPayload
        ),
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
        "ReferencedDataArtifactBuild": data.ReferencedDataArtifactBuild,
        "ReferenceImageBatchSemantics": training.ReferenceImageBatchSemantics,
        "ReferenceMetricProvider": diagnostics.ReferenceMetricProvider,
        "ReferencePipelineConfig": diagnostics.ReferencePipelineConfig,
        "ResolvedExtensions": plugins.ResolvedExtensions,
        "Sampler": sampling.Sampler,
        "SamplerArtifactContext": diagnostics.SamplerArtifactContext,
        "SamplerArtifactProvider": diagnostics.SamplerArtifactProvider,
        "SamplerResult": sampling.SamplerResult,
        "SamplingArtifactContext": sampling.SamplingArtifactContext,
        "SamplingArtifactWriter": sampling.SamplingArtifactWriter,
        "SamplingBatch": sampling.SamplingBatch,
        "SamplingBuilder": sampling.SamplingBuilder,
        "SamplingBuilderContext": sampling.SamplingBuilderContext,
        "SamplingObservation": sampling.SamplingObservation,
        "SamplingObserver": sampling.SamplingObserver,
        "SamplingOutput": sampling.SamplingOutput,
        "SamplingRecipe": sampling.SamplingRecipe,
        "SamplerMetricContext": diagnostics.SamplerMetricContext,
        "SamplerMetricProvider": diagnostics.SamplerMetricProvider,
        "SamplerProfileConfig": diagnostics.SamplerProfileConfig,
        "StepMetricContext": diagnostics.StepMetricContext,
        "StepMetricProvider": diagnostics.StepMetricProvider,
        "TrainBatchEndEvent": training.TrainBatchEndEvent,
        "TrainEpochEndEvent": training.TrainEpochEndEvent,
        "TrainStepOutput": training.TrainStepOutput,
        "TrainingBuilder": training.TrainingBuilder,
        "TrainingBuilderContext": training.TrainingBuilderContext,
        "TrainingDiagnostic": training.TrainingDiagnostic,
        "TrainingPlan": training.TrainingPlan,
        "TrainingStrategy": training.TrainingStrategy,
        "TrajectoryProviderConfig": diagnostics.TrajectoryProviderConfig,
        "TorchvisionImageArtifactPayload": (
            data.TorchvisionImageArtifactPayload
        ),
        "TrajectoryObserver": sampling.TrajectoryObserver,
        "TabulatedDiscreteVPSchedule": processes.TabulatedDiscreteVPSchedule,
        "activate_extension_plugins": plugins.activate_extension_plugins,
        "canonical_artifact_digest": data.canonical_artifact_digest,
        "canonical_artifact_json_bytes": (
            data.canonical_artifact_json_bytes
        ),
        "compute_objective": training.compute_objective,
        "extension_plugin_provenance_to_dicts": (
            plugins.extension_plugin_provenance_to_dicts
        ),
        "parse_extension_plugin_provenance": (
            plugins.parse_extension_plugin_provenance
        ),
        "prepare_extension_plugins": plugins.prepare_extension_plugins,
        "gaussian_training_target": training.gaussian_training_target,
        "normalize_gaussian_prediction": sampling.normalize_gaussian_prediction,
    }

    assert set(public.__all__) == set(expected)
    for name, component in expected.items():
        assert getattr(public, name) is component

    for removed in (
        "ArtifactMaterializationLock",
        "DataPipeline",
        "DataBundle",
        "SplitData",
        "DatasetFactory",
        "DatasetView",
        "DatasetBuildRequest",
        "ManagedDataArtifact",
        "ManagedDataArtifactIdentity",
        "ReferencedDataArtifact",
        "ReferencedDataArtifactIdentity",
        "BoundTrainingDiagnostic",
        "DiagnosticResult",
        "DiagnosticSourceProvider",
        "DiagnosticSourceRequest",
        "EpochMetricSnapshot",
        "MetricConfig",
        "MetricDataRole",
        "MetricEngine",
        "MetricOrigin",
        "MetricPayloadDetachable",
        "MetricRuntimeError",
        "MetricSource",
        "MetricSpec",
        "VerifiedMetricSource",
        "bind_training_diagnostic",
        "bind_training_diagnostics",
        "build_metric",
        "detach_metric_update",
        "detach_metric_updates",
        "detach_metric_value",
        "validate_metric_configs",
        "validate_metric_spec",
        "validate_metric_updates",
        "validate_training_monitor_key",
    ):
        assert not hasattr(public, removed)
    assert not hasattr(registry.REGISTRIES, "data_pipelines")
    assert not hasattr(registry.REGISTRIES, "dataset_factories")
    assert not hasattr(registry.REGISTRIES, "diffusions")
    assert not hasattr(registry.REGISTRIES, "dynamics")
    assert not hasattr(processes, "GaussianDenoisingProcess")
    assert not hasattr(public, "GaussianDenoisingProcess")
    assert not inspect.isabstract(public.GenerativeDynamics)
    assert not hasattr(public.Process, "denoising_dynamics")
    for sampling_contract in (
        "GenerativeDynamics",
        "GaussianDenoisingDynamics",
        "GaussianModelDynamics",
        "GaussianPrediction",
    ):
        assert not hasattr(processes, sampling_contract)


@pytest.mark.parametrize(
    "component_registry",
    [
        registry.REGISTRIES.models,
        registry.REGISTRIES.data_builders,
        registry.REGISTRIES.sampling_artifact_writers,
        registry.REGISTRIES.noise_schedules,
        registry.REGISTRIES.processes,
        registry.REGISTRIES.samplers,
        registry.REGISTRIES.sampling_builders,
        registry.REGISTRIES.training_builders,
        registry.REGISTRIES.objectives,
        registry.REGISTRIES.optimizers,
        registry.REGISTRIES.lr_schedulers,
        registry.REGISTRIES.loggers,
        registry.REGISTRIES.diagnostics,
    ],
)
def test_public_registries_reject_wrong_base_at_public_import_time(
    component_registry: registry.Registry,
) -> None:
    with pytest.raises(registry.RegistryError, match="must inherit"):
        component_registry.add("public_wrong_base", object)


def test_public_import_installs_contracts_without_factory_side_effect() -> None:
    script = """
import sys
from stochaflow import extensions

assert "stochaflow.utils.factory" not in sys.modules
registries = (
    extensions.REGISTRIES.models,
    extensions.REGISTRIES.noise_schedules,
    extensions.REGISTRIES.objectives,
    extensions.REGISTRIES.loggers,
    extensions.REGISTRIES.diagnostics,
)
for index, component_registry in enumerate(registries):
    try:
        component_registry.add(f"wrong_base_{index}", object)
    except extensions.RegistryError:
        continue
    raise AssertionError(f"{component_registry.kind} accepted the wrong base")
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_plugin_strategy_drives_registered_metric() -> None:
    strategy = ExtensionVectorStrategy()
    assert isinstance(strategy, public.MetricChannelProvider)
    assert strategy.metric_channels == frozenset({"example.vector_pair"})

    engine = metrics.MetricEngine(
        (
            metrics.MetricSpec(
                id="relative_l2",
                name=EXTENSION_METRIC_NAME,
                channel="example.vector_pair",
            ),
        )
    )
    batches = (
        ExtensionVectorBatch(
            prediction=torch.tensor(
                [[2.0, 0.0], [0.0, 2.0]],
                requires_grad=True,
            ),
            reference=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        ExtensionVectorBatch(
            prediction=torch.tensor([[1.0, 0.0]], requires_grad=True),
            reference=torch.tensor([[2.0, 0.0]]),
        ),
    )
    for batch in batches:
        output = strategy.training_step(batch)
        assert output.loss.requires_grad
        engine.update(output.metric_updates)

    values = engine.compute(reset=True)
    assert values == {"relative_l2": pytest.approx(5.0 / 6.0)}



def test_selected_plugin_metric_runs_and_persists_installed_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise discovery, activation, factory construction, and checkpointing."""

    plugins._reset_extension_activation_state_for_testing()

    entry_point = InstalledExtensionEntryPoint(
        name=VERTICAL_PLUGIN_NAME,
        value=VERTICAL_PLUGIN_TARGET,
        distribution=VERTICAL_PLUGIN_DISTRIBUTION,
        version=VERTICAL_PLUGIN_VERSION,
    )

    def discover(*, group: str) -> tuple[InstalledExtensionEntryPoint, ...]:
        assert group == "stochaflow.extensions"
        return (entry_point,)

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    monkeypatch.setattr(plugins.metadata, "entry_points", discover)
    unresolved = config.load_config_dict(
        {
            "experiment": {
                "name": "vertical-extension-test",
                "output_dir": str(tmp_path),
            },
            "extensions": {"plugins": [VERTICAL_PLUGIN_NAME]},
            "data": {"name": "test.vertical-extension.data", "params": {}},
            "model": {
                "name": VERTICAL_MODEL_NAME,
                "params": {"features": 2},
            },
            "training": {
                "name": VERTICAL_TRAINING_BUILDER_NAME,
                "params": {},
            },
            "metrics": [
                {
                    "id": "relative_l2",
                    "name": VERTICAL_METRIC_NAME,
                    "channel": VERTICAL_METRIC_CHANNEL,
                    "phases": ["train", "validation"],
                }
            ],
            "optimizer": {
                "name": "torch.optim.SGD",
                "params": {"lr": 0.05},
            },
            "ema": {"enabled": False},
            "trainer": {
                "num_epochs": 1,
                "device": "cpu",
                "show_progress": False,
                "early_stopping": {
                    "enabled": False,
                    "monitor": "valid/metrics/relative_l2",
                    "mode": "min",
                },
            },
            "logging": {
                "log_every": 1,
                "backends": [
                    {
                        "name": "local",
                        "params": {"console": False},
                    }
                ],
            },
            "artifacts": {"checkpoint_every": 1},
        }
    )

    try:
        assert VERTICAL_PLUGIN_TARGET not in sys.modules
        activation_plan = public.prepare_extension_plugins(unresolved)
        assert activation_plan.provenance == (
            public.ExtensionPluginProvenance(
                name=VERTICAL_PLUGIN_NAME,
                distribution=VERTICAL_PLUGIN_DISTRIBUTION,
                version=VERTICAL_PLUGIN_VERSION,
                target=VERTICAL_PLUGIN_TARGET,
            ),
        )

        resolved = public.activate_extension_plugins(activation_plan)
        assert VERTICAL_PLUGIN_TARGET in sys.modules
        assert resolved.config.extensions.plugins == [VERTICAL_PLUGIN_NAME]
        assert resolved.provenance == activation_plan.provenance
        assert VERTICAL_MODEL_NAME in public.REGISTRIES.models
        assert (
            VERTICAL_TRAINING_BUILDER_NAME
            in public.REGISTRIES.training_builders
        )
        assert VERTICAL_METRIC_NAME in public.REGISTRIES.metrics

        inputs = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        )
        references = inputs * 2.0
        train_loader = DataLoader(
            TensorDataset(inputs, references),
            batch_size=2,
        )
        validation_loader = DataLoader(
            TensorDataset(inputs, references),
            batch_size=2,
        )
        loaders = data.DataLoaders(
            train=train_loader,
            validation=validation_loader,
        )
        options = experiment_runner.ExperimentRunOptions(
            num_epochs=1,
            max_train_batches=None,
            max_validation_batches=None,
            max_test_batches=None,
            deterministic=True,
            show_progress=False,
            artifact_verification_workers=None,
            resume_checkpoint=None,
            device=None,
            sample_after_training=False,
        )

        experiment_runner._run_single_run(
            resolved.config,
            loaders,
            options,
            extensions=resolved,
            config_source="external",
            checkpoint_payload=None,
            startup_cwd=Path.cwd(),
            runtime_options={},
        )

        expected_provenance = [
            {
                "name": VERTICAL_PLUGIN_NAME,
                "distribution": VERTICAL_PLUGIN_DISTRIBUTION,
                "version": VERTICAL_PLUGIN_VERSION,
                "target": VERTICAL_PLUGIN_TARGET,
            }
        ]
        resolved_config = yaml.safe_load(
            (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        assert resolved_config["extensions"]["plugins"] == [
            VERTICAL_PLUGIN_NAME
        ]
        assert resolved_config["model"]["name"] == VERTICAL_MODEL_NAME
        assert (
            resolved_config["training"]["name"]
            == VERTICAL_TRAINING_BUILDER_NAME
        )
        assert resolved_config["metrics"] == [
            {
                "id": "relative_l2",
                "name": VERTICAL_METRIC_NAME,
                "channel": VERTICAL_METRIC_CHANNEL,
                "params": {},
                "phases": ["train", "validation"],
            }
        ]

        manifest = yaml.safe_load(
            (tmp_path / "run_manifest.yaml").read_text(encoding="utf-8")
        )
        assert manifest["extension_plugins"] == expected_provenance
        assert manifest["config"] == resolved_config
        assert manifest["selected_components"]["training_builder"] == (
            VERTICAL_TRAINING_BUILDER_NAME
        )
        assert manifest["selected_components"]["metrics"] == [
            VERTICAL_METRIC_NAME
        ]

        checkpoint = torch.load(
            tmp_path / "checkpoints/latest.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["config"] == resolved_config
        assert checkpoint["metadata"]["extension_plugins"] == (
            expected_provenance
        )
        assert checkpoint["metadata"]["selected_components"] == (
            manifest["selected_components"]
        )
        assert "valid/metrics/relative_l2" in checkpoint["metrics"]
    finally:
        plugins._reset_extension_activation_state_for_testing()


@pytest.mark.parametrize(
    ("metric_name", "state_names"),
    [
        pytest.param(
            "mean",
            frozenset({"mean_value", "weight"}),
            id="builtin-mean",
        ),
        pytest.param(
            "mse",
            frozenset({"sum_squared_error", "total"}),
            id="builtin-mse",
        ),
        pytest.param(
            "mae",
            frozenset({"sum_abs_error", "total"}),
            id="builtin-mae",
        ),
        pytest.param(
            EXTENSION_METRIC_NAME,
            frozenset({"relative_error", "observations"}),
            id="extension-relative-l2",
        ),
    ],
)
def test_metric_ddp_declaration_matrix_has_explicit_additive_reduction(
    metric_name: str,
    state_names: frozenset[str],
) -> None:
    """Audit reduction declarations without claiming distributed Trainer support."""

    metric = metrics.build_metric(
        metrics.MetricSpec(
            id="contract",
            name=metric_name,
            channel="contract.payload",
        )
    )
    metric_internals = cast(Any, metric)
    defaults = cast(dict[str, Any], metric_internals._defaults)
    reductions = cast(dict[str, Any], metric_internals._reductions)

    assert frozenset(defaults) == state_names
    assert frozenset(reductions) == state_names
    assert all(
        callable(reduction)
        and getattr(reduction, "__name__", None) == "dim_zero_sum"
        for reduction in reductions.values()
    )
    assert metric.sync_on_compute is True
    assert metric.dist_sync_on_step is False
