"""Tests for the stable third-party extension API."""

from stochaflow import data, diffusion, sampling, training
from stochaflow import extensions as public
from stochaflow.utils import config, logging, registry


def test_public_extension_contracts_reexport_runtime_types() -> None:
    expected = {
        "ComponentConfig": config.ComponentConfig,
        "DataBundle": data.DataBundle,
        "DataPipeline": data.DataPipeline,
        "DataPipelineContext": data.DataPipelineContext,
        "DatasetBuildRequest": data.DatasetBuildRequest,
        "DatasetFactory": data.DatasetFactory,
        "DatasetFactoryContext": data.DatasetFactoryContext,
        "DatasetView": data.DatasetView,
        "DiagnosticBuildContext": training.DiagnosticBuildContext,
        "ExperimentLogger": logging.ExperimentLogger,
        "FitStartEvent": training.FitStartEvent,
        "NoiseSchedule": diffusion.NoiseSchedule,
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
        "SamplingArtifactContext": sampling.SamplingArtifactContext,
        "SamplingArtifactWriter": sampling.SamplingArtifactWriter,
        "SamplingBatch": sampling.SamplingBatch,
        "SplitData": data.SplitData,
        "TrainBatchEndEvent": training.TrainBatchEndEvent,
        "TrainEpochEndEvent": training.TrainEpochEndEvent,
        "TrainingDiagnostic": training.TrainingDiagnostic,
    }

    assert set(public.__all__) == set(expected)
    for name, component in expected.items():
        assert getattr(public, name) is component
