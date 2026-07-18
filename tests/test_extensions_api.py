"""Tests for the stable third-party extension API."""

from stochaflow import data, diffusion, sampling, training
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
        "NoiseSchedule": diffusion.NoiseSchedule,
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
        "SamplingArtifactContext": sampling.SamplingArtifactContext,
        "SamplingArtifactWriter": sampling.SamplingArtifactWriter,
        "SamplingBatch": sampling.SamplingBatch,
        "TrainBatchEndEvent": training.TrainBatchEndEvent,
        "TrainEpochEndEvent": training.TrainEpochEndEvent,
        "TrainingDiagnostic": training.TrainingDiagnostic,
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
