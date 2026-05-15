"""Tests for registry and builder utilities."""

from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.utils.data import TensorDataset

from stochaflow.diffusion import DDPM, DDPMEpsilonObjective, LinearDDPMScheduler
from stochaflow.models import UNet
from stochaflow.training import Trainer
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import ComponentConfig, DataConfig, DataloaderConfig, load_config
from stochaflow.utils.factory import (
    build_data_components,
    build_training_components,
)
from stochaflow.utils.logging import ExperimentLogger
from stochaflow.utils.registry import (
    register_dataset,
)


def test_build_training_components_from_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    components = build_training_components(config)

    assert isinstance(components.model, UNet)
    assert isinstance(components.scheduler, LinearDDPMScheduler)
    assert isinstance(components.diffusion, DDPM)
    assert isinstance(components.objective, DDPMEpsilonObjective)
    assert isinstance(components.optimizer, Optimizer)
    assert isinstance(components.logger, ExperimentLogger)
    assert isinstance(components.checkpoint_manager, CheckpointManager)
    assert isinstance(components.trainer, Trainer)


@register_dataset("toy_tensor_dataset")
def build_toy_tensor_dataset(*, num_samples: int = 8, feature_dim: int = 4) -> TensorDataset:
    features = torch.randn(num_samples, feature_dim)
    return TensorDataset(features)


def test_build_data_components_from_registered_dataset() -> None:
    config = DataConfig(
        dataset=ComponentConfig(
            name="toy_tensor_dataset",
            params={"num_samples": 10, "feature_dim": 3},
        ),
        dataloader=DataloaderConfig(
            batch_size=4,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=False,
        ),
    )

    components = build_data_components(config, seed=7)

    assert isinstance(components.dataset, TensorDataset)
    assert len(components.dataset) == 10
    batch = next(iter(components.dataloader))
    assert isinstance(batch, (tuple, list))
    assert batch[0].shape == (4, 3)
