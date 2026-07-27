"""Tests for registry and builder utilities."""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR

from stochaflow.models import UNet
from stochaflow.processes import (
    DiscreteGaussianProcess,
    Process,
)
from stochaflow.training import (
    GaussianDenoisingTrainingStrategy,
    MSEObjective,
    Trainer,
    TrainingDiagnostic,
    WarmupCosineLR,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import (
    ComponentConfig,
    ConfigError,
    LRSchedulerConfig,
    load_config,
    load_config_dict,
)
from stochaflow.utils.factory import (
    build_diagnostics,
    build_lr_scheduler,
    build_optimizer,
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

TINY_UNET_PARAMS = {
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 16,
    "channel_multipliers": [1, 2],
    "num_res_blocks": 1,
    "time_embedding_dim": 32,
    "dropout": 0.0,
}


class RegisteredOptimizer(Optimizer):
    """Test extension optimizer using the native constructor convention."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        options: dict[str, object] | None = None,
    ) -> None:
        if options is not None:
            options["constructed"] = True
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        return closure() if closure is not None else None


class RegisteredScheduler(LRScheduler):
    """Test extension scheduler using the native constructor convention."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        factor: float,
        options: dict[str, object] | None = None,
    ) -> None:
        self.factor = factor
        if options is not None:
            options["constructed"] = True
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        return [float(base_lr) * self.factor**self.last_epoch for base_lr in self.base_lrs]


class VarArgsScheduler(LRScheduler):
    """Scheduler whose step accepts the trainer's zero-argument call."""

    def step(self, *args) -> None:
        del args


class RequiredKeywordScheduler(LRScheduler):
    """Scheduler incompatible with the automatic zero-argument lifecycle."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer

    def step(self, *, metric: float) -> None:
        del metric


class MisboundScheduler(LRScheduler):
    """Invalid extension that discards the optimizer injected by core."""

    def __init__(self, optimizer: Optimizer) -> None:
        del optimizer
        unrelated_parameter = torch.nn.Parameter(torch.ones(()))
        unrelated_optimizer = SGD([unrelated_parameter], lr=0.1)
        super().__init__(unrelated_optimizer)

    def get_lr(self) -> list[float]:
        return [float(base_lr) for base_lr in self.base_lrs]


class KeyErrorScheduler(LRScheduler):
    """Extension whose constructor fails outside TypeError/ValueError."""

    def __init__(self, optimizer: Optimizer) -> None:
        del optimizer
        raise KeyError("constructor failed")


REGISTRIES.optimizers.add("test_registered_optimizer", RegisteredOptimizer)
REGISTRIES.lr_schedulers.add("test_registered_scheduler", RegisteredScheduler)
REGISTRIES.lr_schedulers.add("test_varargs_scheduler", VarArgsScheduler)
REGISTRIES.lr_schedulers.add(
    "test_required_keyword_scheduler",
    RequiredKeywordScheduler,
)
REGISTRIES.lr_schedulers.add("test_misbound_scheduler", MisboundScheduler)
REGISTRIES.lr_schedulers.add("test_key_error_scheduler", KeyErrorScheduler)


@REGISTRIES.processes.register("test_learnable_gaussian")
class LearnableGaussianProcess(DiscreteGaussianProcess):
    """Gaussian process with an additional trainable process parameter."""

    def __init__(self, schedule) -> None:
        super().__init__(schedule)
        self.process_gain = torch.nn.Parameter(torch.tensor(1.0))


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
    config = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    )
    components = build_training_components(config)

    assert isinstance(components.model, UNet)
    assert isinstance(components.process, DiscreteGaussianProcess)
    assert not hasattr(components.process, "schedule")
    assert components.process.num_timesteps == 1000
    assert isinstance(components.plan.strategy, GaussianDenoisingTrainingStrategy)
    assert components.plan.strategy.prediction_type == "v"
    assert isinstance(components.objective, MSEObjective)
    assert isinstance(components.optimizer, torch.optim.Adam)
    assert components.ema is not None
    assert components.ema.decay == 0.9995
    assert components.ema.update_after_step == 100
    assert components.ema.update_every == 1
    assert isinstance(components.lr_scheduler, WarmupCosineLR)
    assert components.lr_scheduler.warmup_steps == 2000
    assert components.lr_scheduler.total_steps == 78000
    assert components.lr_scheduler.min_lr_ratio == pytest.approx(0.05)
    assert components.lr_scheduler.optimizer is components.optimizer
    assert components.trainer.lr_scheduler_interval == "step"
    assert isinstance(components.logger, ExperimentLogger)
    assert isinstance(components.checkpoint_manager, CheckpointManager)
    assert components.checkpoint_manager.precision_kind == components.precision.kind
    assert (
        components.checkpoint_manager.grad_scaler
        is components.precision.grad_scaler
    )
    assert isinstance(components.trainer, Trainer)


def test_build_training_components_from_ddpm_flowers102_config() -> None:
    config = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_flowers102.yaml")
    )
    components = build_training_components(config)

    assert isinstance(components.model, UNet)
    assert isinstance(components.process, DiscreteGaussianProcess)
    assert not hasattr(components.process, "schedule")
    assert components.process.num_timesteps == 1000
    assert isinstance(components.plan.strategy, GaussianDenoisingTrainingStrategy)
    assert isinstance(components.objective, MSEObjective)
    assert isinstance(components.optimizer, Optimizer)
    assert isinstance(components.lr_scheduler, WarmupCosineLR)
    assert components.lr_scheduler.warmup_steps == 150
    assert components.lr_scheduler.total_steps == 3000
    assert components.ema is not None
    assert len(components.diagnostics) == 1
    assert isinstance(components.logger, ExperimentLogger)
    assert isinstance(components.checkpoint_manager, CheckpointManager)
    assert isinstance(components.trainer, Trainer)


def test_factory_injects_bf16_runtime_into_trainer_and_checkpoint_manager() -> None:
    raw = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    ).to_dict()
    raw["model"]["params"] = dict(TINY_UNET_PARAMS)
    raw["trainer"].update({"device": "cpu", "precision": "bf16-mixed"})
    components = build_training_components(load_config_dict(raw))

    assert components.precision is components.trainer.precision
    assert components.precision.kind == "bf16-mixed"
    assert components.checkpoint_manager.precision_kind == "bf16-mixed"
    assert components.checkpoint_manager.grad_scaler is None


def test_gaussian_training_requires_a_configured_process() -> None:
    raw = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    ).to_dict()
    raw["model"]["params"] = dict(TINY_UNET_PARAMS)
    raw["process"] = None
    config = load_config_dict(raw)

    with pytest.raises(
        TypeError,
        match=r"gaussian_denoising.*DiscreteGaussianDenoisingProcess",
    ):
        build_training_components(config)


def test_mse_objective_owns_scalar_and_per_sample_loss_semantics() -> None:
    objective = MSEObjective(reduction="mean")
    predicted = torch.tensor([[0.0, 2.0], [1.0, 5.0]])
    target = torch.tensor([[0.0, 0.0], [3.0, 1.0]])

    loss = objective(predicted, target)
    per_sample = objective.per_sample_loss(predicted, target)

    assert torch.equal(per_sample, torch.tensor([2.0, 10.0]))
    assert loss.item() == pytest.approx(6.0)
    assert torch.equal(objective(predicted, target), loss)

    summed = MSEObjective(reduction="sum")
    assert torch.equal(
        summed.per_sample_loss(predicted, target),
        torch.tensor([4.0, 20.0]),
    )
    assert summed(predicted, target).item() == pytest.approx(24.0)


def test_process_parameters_are_optimized_checkpointed_but_not_ema(tmp_path) -> None:
    raw = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    ).to_dict()
    raw["model"]["params"] = dict(TINY_UNET_PARAMS)
    raw["process"]["name"] = "test_learnable_gaussian"
    raw["experiment"]["output_dir"] = str(tmp_path)
    config = load_config_dict(raw)

    components = build_training_components(config)
    process = components.process
    assert isinstance(process, LearnableGaussianProcess)
    optimizer_parameters = {
        id(parameter)
        for group in components.optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(process.process_gain) in optimizer_parameters
    assert components.ema is not None
    assert "process_gain" not in components.ema.shadow_params

    state = components.checkpoint_manager.build_state()
    assert state.get("format_version") == 10
    process_state = state.get("process_state_dict")
    assert process_state is not None
    assert "process_gain" in process_state
    assert "ema_model_state_dict" in state
    assert "objective_state_dict" in state
    assert state.get("training_assets_state_dict") == {}
    assert "denoiser_state_dict" not in state
    assert "ema_denoiser_state_dict" not in state

    checkpoint = tmp_path / "checkpoint.pt"
    expected_signal = process.marginal_signal_t.detach().clone()
    components.checkpoint_manager.save(checkpoint)
    process.process_gain.data.fill_(7.0)
    process.marginal_signal_t.fill_(0.0)
    components.checkpoint_manager.load(checkpoint)
    assert process.process_gain.item() == pytest.approx(1.0)
    assert torch.equal(process.marginal_signal_t, expected_signal)


def test_sampling_defaults_cannot_override_inference_recipe_contract(
    tmp_path: Path,
) -> None:
    raw = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    ).to_dict()
    raw["model"]["params"] = dict(TINY_UNET_PARAMS)
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["sampling"]["options"]["prediction_type"] = "epsilon"

    with pytest.raises(
        ConfigError,
        match=r"cannot override fixed inference contract.*prediction_type",
    ):
        build_training_components(load_config_dict(raw))


def test_checkpoint_manager_omits_absent_process_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    manager = CheckpointManager(model=model)
    state = manager.build_state()

    assert "process_state_dict" not in state
    checkpoint = manager.save(tmp_path / "model-only.pt")
    manager.load(checkpoint)


def test_checkpoint_manager_preserves_present_empty_process_state(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(model=torch.nn.Linear(2, 2), process=Process())

    state = manager.build_state()

    assert "process_state_dict" in state
    assert state["process_state_dict"] == {}
    checkpoint = manager.save(tmp_path / "empty-process.pt")
    manager.load(checkpoint)


def test_checkpoint_manager_rejects_process_presence_mismatches(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 2)
    no_process_manager = CheckpointManager(model=model)
    unexpected = no_process_manager.build_state()
    unexpected["process_state_dict"] = {}
    unexpected_path = tmp_path / "unexpected-process.pt"
    torch.save(unexpected, unexpected_path)
    with pytest.raises(ValueError, match="runtime has no process"):
        no_process_manager.load(unexpected_path)

    process = DiscreteGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )
    process_manager = CheckpointManager(model=model, process=process)
    missing = process_manager.build_state()
    missing.pop("process_state_dict")
    missing_path = tmp_path / "missing-process.pt"
    torch.save(missing, missing_path)
    with pytest.raises(TypeError, match="missing process_state_dict"):
        process_manager.load(missing_path)


def test_checkpoint_restores_real_optimizer_and_scheduler_continuously(
    tmp_path: Path,
) -> None:
    first_model = torch.nn.Linear(1, 1)
    first_optimizer = SGD(first_model.parameters(), lr=0.8, momentum=0.9)
    first_scheduler = CosineAnnealingLR(first_optimizer, T_max=6, eta_min=0.2)
    for _ in range(2):
        first_optimizer.step()
        first_scheduler.step()
    checkpoint = CheckpointManager(
        model=first_model,
        optimizer=first_optimizer,
        lr_scheduler=first_scheduler,
    ).save(tmp_path / "optimizer-scheduler.pt")

    second_model = torch.nn.Linear(1, 1)
    second_optimizer = SGD(second_model.parameters(), lr=0.1, momentum=0.0)
    second_scheduler = CosineAnnealingLR(second_optimizer, T_max=6, eta_min=0.2)
    CheckpointManager(
        model=second_model,
        optimizer=second_optimizer,
        lr_scheduler=second_scheduler,
    ).load(checkpoint)

    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(
        first_optimizer.param_groups[0]["lr"]
    )
    assert second_optimizer.param_groups[0]["momentum"] == pytest.approx(0.9)
    assert second_scheduler.state_dict() == first_scheduler.state_dict()

    first_optimizer.step()
    first_scheduler.step()
    second_optimizer.step()
    second_scheduler.step()
    assert second_optimizer.param_groups[0]["lr"] == pytest.approx(
        first_optimizer.param_groups[0]["lr"]
    )


def test_checkpoint_rejects_scheduler_state_without_a_runtime_scheduler(
    tmp_path: Path,
) -> None:
    source_model = torch.nn.Linear(1, 1)
    source_optimizer = SGD(source_model.parameters(), lr=0.1)
    source_scheduler = StepLR(source_optimizer, step_size=1)
    checkpoint = CheckpointManager(
        model=source_model,
        optimizer=source_optimizer,
        lr_scheduler=source_scheduler,
    ).save(tmp_path / "with-scheduler.pt")

    runtime_model = torch.nn.Linear(1, 1)
    runtime_optimizer = SGD(runtime_model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="runtime has no lr scheduler"):
        CheckpointManager(
            model=runtime_model,
            optimizer=runtime_optimizer,
        ).load(checkpoint)


def test_checkpoint_rejects_optimizer_and_scheduler_class_mismatches(
    tmp_path: Path,
) -> None:
    optimizer_model = torch.nn.Linear(1, 1)
    optimizer_checkpoint = CheckpointManager(
        model=optimizer_model,
        optimizer=SGD(optimizer_model.parameters(), lr=0.1),
    ).save(tmp_path / "sgd.pt")
    runtime_optimizer_model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="optimizer class does not match runtime"):
        CheckpointManager(
            model=runtime_optimizer_model,
            optimizer=torch.optim.Adam(runtime_optimizer_model.parameters()),
        ).load(optimizer_checkpoint)

    scheduler_model = torch.nn.Linear(1, 1)
    scheduler_optimizer = SGD(scheduler_model.parameters(), lr=0.1)
    scheduler_checkpoint = CheckpointManager(
        model=scheduler_model,
        optimizer=scheduler_optimizer,
        lr_scheduler=CosineAnnealingLR(scheduler_optimizer, T_max=2),
    ).save(tmp_path / "cosine.pt")
    runtime_scheduler_model = torch.nn.Linear(1, 1)
    runtime_scheduler_optimizer = SGD(runtime_scheduler_model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="lr scheduler class does not match runtime"):
        CheckpointManager(
            model=runtime_scheduler_model,
            optimizer=runtime_scheduler_optimizer,
            lr_scheduler=StepLR(runtime_scheduler_optimizer, step_size=1),
        ).load(scheduler_checkpoint)


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


def test_warmup_cosine_lr_scheduler_uses_explicit_total_steps() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="warmup_cosine",
            interval="step",
            params={
                "warmup_steps": 2,
                "total_steps": 6,
                "min_lr_ratio": 0.1,
            },
        ),
        optimizer,
    )

    assert isinstance(scheduler, WarmupCosineLR)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] == pytest.approx(0.5)
    assert lrs[1] == pytest.approx(1.0)
    assert lrs[2] < 1.0
    assert lrs[-1] == pytest.approx(0.1)


def test_warmup_cosine_rejects_auto_total_steps() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    with pytest.raises(RegistryError, match="total_steps must be a positive integer"):
        build_lr_scheduler(
            LRSchedulerConfig(
                name="warmup_cosine",
                interval="step",
                params={"warmup_steps": 2, "total_steps": "auto"},
            ),
            optimizer,
        )


def test_native_cosine_lr_scheduler_uses_explicit_params() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="torch.optim.lr_scheduler.CosineAnnealingLR",
            interval="epoch",
            params={"T_max": 60, "eta_min": 0.1},
        ),
        optimizer,
    )

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.optimizer is optimizer
    assert scheduler.T_max == 60


def test_torch_builtin_lr_scheduler_can_be_built() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="torch.optim.lr_scheduler.StepLR",
            interval="epoch",
            params={"step_size": 1, "gamma": 0.5},
        ),
        optimizer,
    )

    assert isinstance(scheduler, StepLR)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_native_optimizer_resolver_needs_no_per_class_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.optim,
        "Stage41Optimizer",
        RegisteredOptimizer,
        raising=False,
    )
    parameter = torch.nn.Parameter(torch.ones(()))

    optimizer = build_optimizer(
        ComponentConfig(
            name="torch.optim.Stage41Optimizer",
            params={"lr": 0.25},
        ),
        [parameter],
    )

    assert isinstance(optimizer, RegisteredOptimizer)
    assert optimizer.param_groups[0]["params"] == [parameter]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "target",
    [
        "torch.optim.MissingStage41Optimizer",
        "torch.optim.lr_scheduler.StepLR",
        "torch.optim._Stage41Optimizer",
    ],
)
def test_native_optimizer_rejects_unknown_nested_or_private_targets(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    monkeypatch.setattr(
        torch.optim,
        "_Stage41Optimizer",
        RegisteredOptimizer,
        raising=False,
    )
    parameter = torch.nn.Parameter(torch.ones(()))

    with pytest.raises(RegistryError, match="optimizer target"):
        build_optimizer(
            ComponentConfig(name=target, params={"lr": 0.1}),
            [parameter],
        )


def test_native_optimizer_rejects_wrong_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.optim, "Stage41WrongBase", str, raising=False)
    parameter = torch.nn.Parameter(torch.ones(()))

    with pytest.raises(RegistryError, match=r"must inherit.*Optimizer"):
        build_optimizer(
            ComponentConfig(
                name="torch.optim.Stage41WrongBase",
                params={"lr": 0.1},
            ),
            [parameter],
        )


def test_optimizer_and_scheduler_registries_reject_wrong_bases() -> None:
    with pytest.raises(RegistryError, match="optimizer registrations must inherit"):
        REGISTRIES.optimizers.add(
            "test_wrong_optimizer",
            cast(type[Optimizer], str),
        )
    with pytest.raises(RegistryError, match="lr scheduler registrations must inherit"):
        REGISTRIES.lr_schedulers.add(
            "test_wrong_scheduler",
            cast(type[LRScheduler], str),
        )


def test_optimizer_and_scheduler_registries_reject_native_namespaces() -> None:
    with pytest.raises(
        RegistryError,
        match=r"reserved namespace 'torch\.optim\.'",
    ):
        REGISTRIES.optimizers.add(
            "torch.optim.Stage41Optimizer",
            RegisteredOptimizer,
        )
    with pytest.raises(
        RegistryError,
        match=r"reserved namespace 'torch\.optim\.lr_scheduler\.'",
    ):
        REGISTRIES.lr_schedulers.add(
            "torch.optim.lr_scheduler.Stage41Scheduler",
            RegisteredScheduler,
        )


def test_non_native_target_is_not_imported_as_a_python_path() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))

    with pytest.raises(RegistryError, match=r"unknown optimizer 'os\.system'"):
        build_optimizer(
            ComponentConfig(name="os.system", params={}),
            [parameter],
        )


def test_registered_optimizer_and_scheduler_use_native_constructor_contract() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = build_optimizer(
        ComponentConfig(
            name="test_registered_optimizer",
            params={"lr": 0.5},
        ),
        [parameter],
    )
    scheduler = build_lr_scheduler(
        LRSchedulerConfig(
            name="test_registered_scheduler",
            interval="step",
            params={"factor": 0.5},
        ),
        optimizer,
    )

    assert isinstance(optimizer, RegisteredOptimizer)
    assert optimizer.step() is None
    assert isinstance(scheduler, RegisteredScheduler)
    assert scheduler.optimizer is optimizer


def test_optimizer_requiring_a_closure_is_rejected_at_build_boundary() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))

    with pytest.raises(RegistryError, match=r"optimizer .*step\(\).*without arguments"):
        build_optimizer(
            ComponentConfig(name="torch.optim.LBFGS", params={"lr": 0.1}),
            [parameter],
        )


def test_scheduler_must_retain_the_injected_optimizer() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    with pytest.raises(RegistryError, match="must retain the injected optimizer"):
        build_lr_scheduler(
            LRSchedulerConfig(name="test_misbound_scheduler"),
            optimizer,
        )


def test_optimizer_and_scheduler_constructor_params_are_deep_copied() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer_options: dict[str, object] = {"label": "original"}
    optimizer_config = ComponentConfig(
        name="test_registered_optimizer",
        params={"lr": 0.5, "options": optimizer_options},
    )

    optimizer = build_optimizer(optimizer_config, [parameter])

    scheduler_options: dict[str, object] = {"label": "original"}
    scheduler_config = LRSchedulerConfig(
        name="test_registered_scheduler",
        interval="step",
        params={"factor": 0.5, "options": scheduler_options},
    )
    build_lr_scheduler(scheduler_config, optimizer)

    assert optimizer_options == {"label": "original"}
    assert optimizer_config.params["options"] == {"label": "original"}
    assert scheduler_options == {"label": "original"}
    assert scheduler_config.params["options"] == {"label": "original"}


def test_runtime_constructor_parameters_cannot_be_overridden() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    with pytest.raises(RegistryError, match=r"cannot override.*'params'"):
        build_optimizer(
            ComponentConfig(
                name="torch.optim.SGD",
                params={"params": [parameter], "lr": 0.1},
            ),
            [parameter],
        )
    with pytest.raises(RegistryError, match=r"cannot override.*'optimizer'"):
        build_lr_scheduler(
            LRSchedulerConfig(
                name="torch.optim.lr_scheduler.StepLR",
                params={"optimizer": optimizer, "step_size": 1},
            ),
            optimizer,
        )


def test_native_constructor_type_error_preserves_original_cause() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))

    with pytest.raises(RegistryError, match=r"torch\.optim\.SGD") as raised:
        build_optimizer(
            ComponentConfig(
                name="torch.optim.SGD",
                params={"unknown_parameter": True},
            ),
            [parameter],
        )

    assert isinstance(raised.value.__cause__, TypeError)


def test_constructor_failure_is_wrapped_with_target_and_original_cause() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=0.1)

    with pytest.raises(RegistryError, match="test_key_error_scheduler") as raised:
        build_lr_scheduler(
            LRSchedulerConfig(name="test_key_error_scheduler"),
            optimizer,
        )

    assert isinstance(raised.value.__cause__, KeyError)


def test_metric_driven_scheduler_is_rejected_at_build_boundary() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    with pytest.raises(RegistryError, match=r"step\(\).*without arguments"):
        build_lr_scheduler(
            LRSchedulerConfig(
                name="torch.optim.lr_scheduler.ReduceLROnPlateau",
                interval="epoch",
            ),
            optimizer,
        )


def test_zero_argument_step_contract_accepts_varargs() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    scheduler = build_lr_scheduler(
        LRSchedulerConfig(name="test_varargs_scheduler"),
        optimizer,
    )

    assert isinstance(scheduler, VarArgsScheduler)


def test_zero_argument_step_contract_rejects_required_keyword() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    with pytest.raises(RegistryError, match=r"step\(\).*without arguments"):
        build_lr_scheduler(
            LRSchedulerConfig(name="test_required_keyword_scheduler"),
            optimizer,
        )


def test_zero_argument_step_contract_rejects_uninspectable_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = SGD([parameter], lr=1.0)

    def unavailable_signature(_callable):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(
        "stochaflow.training.optimization.inspect.signature",
        unavailable_signature,
    )
    with pytest.raises(RegistryError, match="no inspectable step"):
        build_lr_scheduler(
            LRSchedulerConfig(
                name="torch.optim.lr_scheduler.StepLR",
                params={"step_size": 1},
            ),
            optimizer,
        )


def test_removed_native_aliases_are_not_compatibility_names() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))

    with pytest.raises(RegistryError, match="unknown optimizer 'adam'"):
        build_optimizer(
            ComponentConfig(name="adam", params={"lr": 0.1}),
            [parameter],
        )
    with pytest.raises(RegistryError, match="unknown lr scheduler 'step'"):
        build_lr_scheduler(
            LRSchedulerConfig(name="step", params={"step_size": 1}),
            SGD([parameter], lr=0.1),
        )


def test_disabled_lr_scheduler_uses_safe_trainer_interval() -> None:
    raw = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    ).to_dict()
    raw["model"]["params"] = dict(TINY_UNET_PARAMS)
    raw["lr_scheduler"] = None
    config = load_config_dict(raw)

    components = build_training_components(config)

    assert components.lr_scheduler is None
    assert components.trainer.lr_scheduler_interval == "step"


def test_unknown_diagnostic_raises_registry_error() -> None:
    raw = load_config(
        Path("examples/built-in/image-generation/experiments/ddpm_mnist.yaml")
    ).to_dict()
    raw["model"]["params"] = dict(TINY_UNET_PARAMS)
    raw["diagnostics"] = [{"name": "missing", "params": {}}]
    config = load_config_dict(raw)

    with pytest.raises(RegistryError, match="unknown diagnostic"):
        build_training_components(config)
