"""Tests for the stable third-party extension API."""

import pytest

from stochaflow import data, processes, sampling, training
from stochaflow import extensions as public
from stochaflow.utils import config, logging, plugins, registry


def test_public_extension_contracts_reexport_runtime_types() -> None:
    expected = {
        "ComponentConfig": config.ComponentConfig,
        "DataBuilder": data.DataBuilder,
        "DataBuilderContext": data.DataBuilderContext,
        "DataLoaders": data.DataLoaders,
        "DDIMSampler": sampling.DDIMSampler,
        "DDPMAncestralSampler": sampling.DDPMAncestralSampler,
        "DiagnosticBuildContext": training.DiagnosticBuildContext,
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
        "InferenceModelProvider": sampling.InferenceModelProvider,
        "MSEObjective": training.MSEObjective,
        "ManagedTrainingModule": training.ManagedTrainingModule,
        "PerSampleObjective": training.PerSampleObjective,
        "PredictionType": sampling.PredictionType,
        "Process": processes.Process,
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
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
        "TrainBatchEndEvent": training.TrainBatchEndEvent,
        "TrainEpochEndEvent": training.TrainEpochEndEvent,
        "TrainStepOutput": training.TrainStepOutput,
        "TrainingBuilder": training.TrainingBuilder,
        "TrainingBuilderContext": training.TrainingBuilderContext,
        "TrainingDiagnostic": training.TrainingDiagnostic,
        "TrainingPlan": training.TrainingPlan,
        "TrainingStrategy": training.TrainingStrategy,
        "TrajectoryObserver": sampling.TrajectoryObserver,
        "TabulatedDiscreteVPSchedule": processes.TabulatedDiscreteVPSchedule,
        "activate_extension_plugins": plugins.activate_extension_plugins,
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
        "DataPipeline",
        "DataBundle",
        "SplitData",
        "DatasetFactory",
        "DatasetView",
        "DatasetBuildRequest",
    ):
        assert not hasattr(public, removed)
    assert not hasattr(registry.REGISTRIES, "data_pipelines")
    assert not hasattr(registry.REGISTRIES, "dataset_factories")
    assert not hasattr(registry.REGISTRIES, "diffusions")
    assert not hasattr(registry.REGISTRIES, "dynamics")
    assert not hasattr(processes, "GaussianDenoisingProcess")
    assert not hasattr(public, "GaussianDenoisingProcess")
    assert public.GenerativeDynamics.__abstractmethods__ == frozenset()
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
    (
        registry.REGISTRIES.processes,
        registry.REGISTRIES.samplers,
        registry.REGISTRIES.sampling_builders,
    ),
)
def test_stage3_registries_reject_wrong_base_at_public_import_time(
    component_registry: registry.Registry,
) -> None:
    with pytest.raises(registry.RegistryError, match="must inherit"):
        component_registry.add("stage3_wrong_base", object)
