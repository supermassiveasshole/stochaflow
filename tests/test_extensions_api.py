"""Tests for the stable third-party extension API."""

import inspect
import subprocess
import sys

import pytest

from stochaflow import data, models, processes, sampling, training
from stochaflow import extensions as public
from stochaflow.utils import config, logging, plugins, registry


def test_public_extension_contracts_reexport_runtime_types() -> None:
    expected = {
        "ArtifactVerificationEvent": data.ArtifactVerificationEvent,
        "ArtifactVerificationObserver": data.ArtifactVerificationObserver,
        "ArtifactVerificationPhase": data.ArtifactVerificationPhase,
        "ComponentConfig": config.ComponentConfig,
        "ConfigError": config.ConfigError,
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
        "DiagnosticBuildContext": training.DiagnosticBuildContext,
        "DeviceTransferableBatch": training.DeviceTransferableBatch,
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
        "PerSampleObjective": training.PerSampleObjective,
        "PredictionType": sampling.PredictionType,
        "Process": processes.Process,
        "PairedImageFolderArtifactPayload": (
            data.PairedImageFolderArtifactPayload
        ),
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
        "ReferencedDataArtifactBuild": data.ReferencedDataArtifactBuild,
        "ReferenceImageBatchSemantics": training.ReferenceImageBatchSemantics,
        "ResolvedExtensions": plugins.ResolvedExtensions,
        "Sampler": sampling.Sampler,
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
        "TrainBatchEndEvent": training.TrainBatchEndEvent,
        "TrainEpochEndEvent": training.TrainEpochEndEvent,
        "TrainStepOutput": training.TrainStepOutput,
        "TrainingBuilder": training.TrainingBuilder,
        "TrainingBuilderContext": training.TrainingBuilderContext,
        "TrainingDiagnostic": training.TrainingDiagnostic,
        "TrainingPlan": training.TrainingPlan,
        "TrainingStrategy": training.TrainingStrategy,
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
