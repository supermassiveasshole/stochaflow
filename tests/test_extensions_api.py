"""Tests for the stable third-party extension API."""

from stochaflow import data, diffusion, training
from stochaflow import extensions as public
from stochaflow.utils import config, logging, registry


def test_public_extension_contracts_reexport_runtime_types() -> None:
    expected = {
        "ComponentConfig": config.ComponentConfig,
        "DataPartitions": data.DataPartitions,
        "DatasetBuildRequest": data.DatasetBuildRequest,
        "DatasetFactory": data.DatasetFactory,
        "DatasetFactoryContext": data.DatasetFactoryContext,
        "DatasetMaterializer": data.DatasetMaterializer,
        "DatasetSelection": data.DatasetSelection,
        "DatasetView": data.DatasetView,
        "DiagnosticBuildContext": training.DiagnosticBuildContext,
        "ExperimentLogger": logging.ExperimentLogger,
        "FitStartEvent": training.FitStartEvent,
        "NoiseSchedule": diffusion.NoiseSchedule,
        "REGISTRIES": registry.REGISTRIES,
        "Registry": registry.Registry,
        "RegistryError": registry.RegistryError,
        "SplitContext": data.SplitContext,
        "SplitStrategy": data.SplitStrategy,
        "TrainBatchEndEvent": training.TrainBatchEndEvent,
        "TrainEpochEndEvent": training.TrainEpochEndEvent,
        "TrainingDiagnostic": training.TrainingDiagnostic,
    }

    assert set(public.__all__) == set(expected)
    for name, component in expected.items():
        assert getattr(public, name) is component
