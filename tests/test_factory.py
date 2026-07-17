"""Tests for registry and builder utilities."""

from pathlib import Path

import pytest
import torch
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR

from stochaflow.diffusion import (
    DDPM,
    DDPMEpsilonObjective,
    LinearBetaSchedule,
)
from stochaflow.models import UNet
from stochaflow.training import Trainer, TrainingDiagnostic
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ComponentConfig,
    LRSchedulerConfig,
    load_config,
    load_config_dict,
)
from stochaflow.utils.factory import (
    build_diagnostics,
    build_lr_scheduler,
    build_training_components,
    resolve_device,
)
from stochaflow.utils.logging import ExperimentLogger, NullLogger
from stochaflow.utils.registry import REGISTRIES, RegistryError


class MinimalDiagnostic(TrainingDiagnostic):
    """Custom diagnostic that only accepts the generic constructor contract."""

    def __init__(
        self,
        *,
        logger: ExperimentLogger,
        output_dir: str,
        marker: str,
    ) -> None:
        self.logger = logger
        self.output_dir = output_dir
        self.marker = marker


REGISTRIES.diagnostics.add("test_minimal", MinimalDiagnostic)


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")


def test_resolve_device_auto_uses_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("mps")


def test_resolve_device_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_device("auto") == torch.device("cpu")


def test_build_training_components_from_ddpm_mnist_config() -> None:
    config = load_config(Path("configs/ddpm_mnist.yaml"))
    components = build_training_components(config)

    assert isinstance(components.model, UNet)
    assert isinstance(components.noise_schedule, LinearBetaSchedule)
    assert isinstance(components.diffusion, DDPM)
    assert isinstance(components.objective, DDPMEpsilonObjective)
    assert isinstance(components.optimizer, Optimizer)
    assert components.ema is not None
    assert isinstance(components.lr_scheduler, CosineAnnealingLR)
    assert components.lr_scheduler.T_max == config.trainer.num_epochs
    assert components.trainer.lr_scheduler_interval == "epoch"
    assert isinstance(components.logger, ExperimentLogger)
    assert isinstance(components.checkpoint_manager, CheckpointManager)
    assert isinstance(components.trainer, Trainer)


def test_build_training_components_from_ddpm_flowers102_config() -> None:
    config = load_config(Path("configs/ddpm_flowers102.yaml"))
    components = build_training_components(config, steps_per_epoch=10, num_epochs=200)

    assert isinstance(components.model, UNet)
    assert isinstance(components.noise_schedule, LinearBetaSchedule)
    assert isinstance(components.diffusion, DDPM)
    assert components.diffusion.clip_denoised
    assert isinstance(components.objective, DDPMEpsilonObjective)
    assert isinstance(components.optimizer, Optimizer)
    assert isinstance(components.lr_scheduler, LambdaLR)
    assert components.ema is not None
    assert len(components.diagnostics) == 1
    assert isinstance(components.logger, ExperimentLogger)
    assert isinstance(components.checkpoint_manager, CheckpointManager)
    assert isinstance(components.trainer, Trainer)


def test_custom_diagnostic_receives_only_generic_runtime_parameters(tmp_path) -> None:
    logger = NullLogger()

    diagnostics = build_diagnostics(
        [ComponentConfig(name="test_minimal", params={"marker": "ready"})],
        logger=logger,
        output_dir=str(tmp_path),
        sample_shape=(3, 32, 32),
    )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert isinstance(diagnostic, MinimalDiagnostic)
    assert diagnostic.logger is logger
    assert diagnostic.output_dir == str(tmp_path)
    assert diagnostic.marker == "ready"


def test_warmup_cosine_lr_scheduler_uses_auto_total_steps() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="warmup_cosine",
            interval="step",
            params={
                "warmup_steps": 2,
                "total_steps": "auto",
                "min_lr_ratio": 0.1,
            },
        ),
        optimizer,
        steps_per_epoch=3,
        num_epochs=2,
    )

    assert isinstance(scheduler, LambdaLR)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(0.5)
    assert lrs[1] == pytest.approx(1.0)
    assert lrs[2] < 1.0
    assert lrs[-1] == pytest.approx(0.1)


def test_cosine_lr_scheduler_uses_effective_epoch_override() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="cosine",
            interval="epoch",
            params={"T_max": "auto", "eta_min": 0.1},
        ),
        optimizer,
        steps_per_epoch=3,
        num_epochs=60,
    )

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 60


def test_torch_builtin_lr_scheduler_can_be_built() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="step",
            interval="epoch",
            params={"step_size": 1, "gamma": 0.5},
        ),
        optimizer,
    )

    assert isinstance(scheduler, StepLR)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_disabled_lr_scheduler_uses_safe_trainer_interval() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["lr_scheduler"] = {"name": None, "interval": "batch", "params": {}}
    config = load_config_dict(raw)

    components = build_training_components(config)

    assert components.lr_scheduler is None
    assert components.trainer.lr_scheduler_interval == "step"


def test_unknown_diagnostic_raises_registry_error() -> None:
    raw = load_config(Path("configs/ddpm_mnist.yaml")).to_dict()
    raw["diagnostics"] = [{"name": "missing", "params": {}}]
    config = load_config_dict(raw)

    with pytest.raises(RegistryError, match="unknown diagnostic"):
        build_training_components(config)
