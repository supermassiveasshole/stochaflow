"""Tests for the stable third-party extension API."""

import pytest

from stochaflow import data, processes, sampling, training
from stochaflow import extensions as public
from stochaflow.utils import config, logging, registry


def test_public_extension_contracts_reexport_runtime_types() -> None:
    expected = {
        "ComponentConfig": config.ComponentConfig,
        "DataBuilder": data.DataBuilder,
        "DataBuilderContext": data.DataBuilderContext,
        "DataLoaders": data.DataLoaders,
        "DiagnosticBuildContext": training.DiagnosticBuildContext,
        "ExperimentLogger": logging.ExperimentLogger,
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
        "GenerativeDynamics": sampling.GenerativeDynamics,
        "InferenceModelProvider": sampling.InferenceModelProvider,
        "MSEObjective": training.MSEObjective,
        "ManagedTrainingModule": training.ManagedTrainingModule,
        "PredictionType": sampling.PredictionType,
        "Process": processes.Process,
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
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
