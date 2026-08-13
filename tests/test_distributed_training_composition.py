"""Tests for side-effect-free fixed-DDP runtime composition."""

from __future__ import annotations

import os
import socket
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import stochaflow.training.distributed.composition as ddp_composition
from stochaflow._builtin_activation import activate_training_builtins
from stochaflow.processes.base import Process
from stochaflow.training import (
    ManagedTrainingModule,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
)
from stochaflow.training.distributed import DistributedSession
from stochaflow.training.distributed.composition import (
    _build_ddp_training_components,
    build_ddp_training_components,
)
from stochaflow.utils.config import (
    ComponentConfig,
    EMAConfig,
    ExperimentConfig,
    LRSchedulerConfig,
    StochaflowConfig,
    TrainerConfig,
    TrainingMetricConfig,
)
from stochaflow.utils.logging_contracts import NullLogger
from stochaflow.utils.registry import REGISTRIES

DistributedReduction = Literal["sum", "min", "max"]
MODEL_NAME = "test.ddp_composition_primary"
BUFFERED_MODEL_NAME = "test.ddp_composition_buffered_primary"
SHARED_STORAGE_MODEL_NAME = "test.ddp_composition_shared_storage_primary"
PROCESS_NAME = "test.ddp_composition_frozen_process"
STATEFUL_BUILDER_NAME = "test.ddp_composition_stateful_auxiliary"


class CompositionPrimaryModel(nn.Module):
    """Small state-bearing primary model used by composition tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class BufferedCompositionPrimaryModel(CompositionPrimaryModel):
    """Primary model rejected because DDP buffer broadcast is disabled."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("runtime_mean", torch.zeros(()))


class SharedStorageCompositionPrimaryModel(nn.Module):
    """Expose distinct parameters backed by one storage allocation."""

    def __init__(self) -> None:
        super().__init__()
        storage = torch.ones(2)
        self.left = nn.Parameter(storage[:1])
        self.right = nn.Parameter(storage[1:])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * (self.left + self.right)


class FrozenCompositionProcess(Process):
    """Process with read-only tensor state synchronized by fingerprint."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("schedule", torch.linspace(0.0, 1.0, 4))


class StateFreeExecutionStrategy(TrainingStrategy):
    """Provide a valid Strategy for auxiliary-state admission tests."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(self, batch: Any) -> TrainStepOutput:
        output = self.model(batch)
        if not isinstance(output, torch.Tensor):
            raise TypeError("test model must return a Tensor")
        return TrainStepOutput(loss=output.square().mean())


class NonPersistentStateModule(nn.Module):
    """Expose mutable runtime state that is absent from state_dict()."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cache", torch.ones(()), persistent=False)


class StatefulAuxiliaryTrainingBuilder(TrainingBuilder):
    """Return a plan whose auxiliary state is unsupported by first DDP."""

    def build(self) -> TrainingPlan:
        return TrainingPlan(
            strategy=StateFreeExecutionStrategy(self.context.primary_model),
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=self.context.objective,
            auxiliary_modules={
                "runtime_cache": ManagedTrainingModule(NonPersistentStateModule())
            },
        )


class StateSharingExecutionWrapper(nn.Module):
    """Test double for DDP that preserves canonical state and no_sync()."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def no_sync(self) -> AbstractContextManager[None]:
        return nullcontext()


class ForeignStateExecutionWrapper(StateSharingExecutionWrapper):
    """Invalid wrapper that adds a parameter outside canonical state."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)
        self.foreign = nn.Parameter(torch.zeros(()))


class CompositionProcessGroupRuntime:
    """One-rank process-group stand-in for injectable-wrapper tests."""

    def __init__(
        self,
        *,
        world_size: int = 1,
        disagree_on_all_equal_call: int | None = None,
    ) -> None:
        self.initialized = False
        self.configured_world_size = world_size
        self.disagree_on_all_equal_call = disagree_on_all_equal_call
        self.all_gather_calls = 0

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def initialize(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ) -> None:
        assert backend == "gloo"
        assert rank == 0
        assert world_size == self.configured_world_size
        assert timeout > timedelta(0)
        self.initialized = True

    def destroy(self) -> None:
        self.initialized = False

    def rank(self) -> int:
        return 0

    def world_size(self) -> int:
        return self.configured_world_size

    def backend(self) -> str:
        return "gloo"

    def broadcast_object_list(
        self,
        values: list[object],
        *,
        source_rank: int,
    ) -> None:
        del values
        assert source_rank == 0

    def gather_object(
        self,
        value: object,
        output: list[object] | None,
        *,
        destination_rank: int,
    ) -> None:
        assert destination_rank == 0
        if output is not None:
            output[:] = [value]

    def all_gather_object(self, output: list[object], value: object) -> None:
        self.all_gather_calls += 1
        output[:] = [value for _ in range(self.configured_world_size)]
        if (
            self.disagree_on_all_equal_call == self.all_gather_calls
            and self.configured_world_size > 1
        ):
            output[-1] = "0" * 64 if value != "0" * 64 else "1" * 64

    def all_reduce(
        self,
        value: torch.Tensor,
        *,
        reduction: DistributedReduction,
    ) -> None:
        del value
        assert reduction in {"sum", "min", "max"}


REGISTRIES.models.add(MODEL_NAME, CompositionPrimaryModel)
REGISTRIES.models.add(BUFFERED_MODEL_NAME, BufferedCompositionPrimaryModel)
REGISTRIES.models.add(SHARED_STORAGE_MODEL_NAME, SharedStorageCompositionPrimaryModel)
REGISTRIES.processes.add(PROCESS_NAME, FrozenCompositionProcess)
REGISTRIES.training_builders.add(
    STATEFUL_BUILDER_NAME,
    StatefulAuxiliaryTrainingBuilder,
)


def torchrun_environment(
    *,
    world_size: int = 1,
    port: int = 29500,
) -> dict[str, str]:
    """Return a valid fixed single-node torchrun environment."""

    return {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "ROLE_RANK": "0",
        "ROLE_WORLD_SIZE": str(world_size),
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
    }


def build_config(
    output_dir: Path,
    *,
    training_name: str = "supervised",
) -> StochaflowConfig:
    """Build the smallest supported first-DDP configuration."""

    return StochaflowConfig(
        experiment=ExperimentConfig(
            name="ddp-composition-test",
            output_dir=str(output_dir),
        ),
        data=ComponentConfig("unused-by-composition"),
        model=ComponentConfig(MODEL_NAME),
        training=ComponentConfig(training_name),
        objective=ComponentConfig("mse"),
        metrics=[
            TrainingMetricConfig(
                id="prediction_mae",
                name="mae",
                channel="supervised.prediction_target",
                phases=["validation"],
            )
        ],
        optimizer=ComponentConfig("torch.optim.SGD", {"lr": 0.1}),
        ema=EMAConfig(enabled=True, decay=0.9),
        trainer=TrainerConfig(
            device="cpu",
            precision="fp32",
            show_progress=False,
            test_after_fit=False,
        ),
    )


def fake_session(
    *,
    world_size: int = 1,
    disagree_on_all_equal_call: int | None = None,
) -> DistributedSession:
    """Construct one unentered fake-backed Gloo session."""

    return DistributedSession.from_environment(
        backend="gloo",
        environ=torchrun_environment(world_size=world_size),
        runtime=CompositionProcessGroupRuntime(
            world_size=world_size,
            disagree_on_all_equal_call=disagree_on_all_equal_call,
        ),
    )


def test_composition_orders_placement_wrap_optimizer_and_ema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activate_training_builtins()
    output_dir = tmp_path / "must-not-be-created"
    config = build_config(output_dir)
    events: list[str] = []
    original_move = ddp_composition.move_module_to_device
    original_optimizer = ddp_composition.build_optimizer

    def recording_move(
        module: nn.Module,
        device: torch.device | str,
        *,
        role: str,
    ) -> nn.Module:
        events.append(f"move:{role}")
        return original_move(module, device, role=role)

    def recording_optimizer(config_value: ComponentConfig, parameters: Any):
        events.append("optimizer")
        return original_optimizer(config_value, parameters)

    def synchronized_wrapper(
        module: nn.Module,
        session: DistributedSession,
    ) -> nn.Module:
        assert session.device == torch.device("cpu")
        events.append("wrap")
        with torch.no_grad():
            cast_model = module
            assert isinstance(cast_model, CompositionPrimaryModel)
            cast_model.weight.fill_(7.0)
        return StateSharingExecutionWrapper(module)

    monkeypatch.setattr(ddp_composition, "move_module_to_device", recording_move)
    monkeypatch.setattr(ddp_composition, "build_optimizer", recording_optimizer)
    metadata = {"nested": {"value": 1}}
    with fake_session() as session:
        components = _build_ddp_training_components(
            config,
            session,
            checkpoint_metadata=metadata,
            wrapper_factory=synchronized_wrapper,
        )

    assert events == ["move:primary_model", "move:objective", "wrap", "optimizer"]
    assert components.binding.plan is components.plan
    assert components.trainer.plan.primary_model is components.model
    assert components.trainer.plan.process is components.process
    assert components.trainer.plan.objective is components.objective
    assert components.binding.execution_model is components.trainer.execution_model
    assert tuple(components.binding.execution_model.parameters()) == tuple(
        components.model.parameters()
    )
    assert tuple(components.binding.execution_model.buffers()) == tuple(
        components.model.buffers()
    )
    assert components.checkpoint_manager.model is components.model
    assert components.checkpoint_manager.optimizer is components.optimizer
    assert components.checkpoint_manager.ema is components.ema
    assert components.ema is not None
    assert components.ema.shadow_params["weight"].item() == 7.0
    assert not components.ema.shadow_buffers
    assert components.metric_runtime.has_phase("validation")
    assert not components.metric_runtime.has_phase("train")
    assert not components.metric_runtime.has_phase("test")
    assert isinstance(components.logger, NullLogger)
    metadata["nested"]["value"] = 2
    assert components.checkpoint_metadata == {"nested": {"value": 1}}
    assert not output_dir.exists()


def test_composition_rejects_wrapper_state_before_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activate_training_builtins()
    optimizer_built = False

    def forbidden_optimizer(config_value: ComponentConfig, parameters: Any):
        del config_value, parameters
        nonlocal optimizer_built
        optimizer_built = True
        raise AssertionError("optimizer must not precede wrapper validation")

    monkeypatch.setattr(ddp_composition, "build_optimizer", forbidden_optimizer)
    with fake_session() as session, pytest.raises(
        ValueError,
        match="preserve canonical state identity",
    ):
        _build_ddp_training_components(
            build_config(tmp_path / "unused"),
            session,
            wrapper_factory=lambda module, unused: ForeignStateExecutionWrapper(
                module
            ),
        )

    assert not optimizer_built


def test_composition_rejects_nonpersistent_auxiliary_state(
    tmp_path: Path,
) -> None:
    activate_training_builtins()
    config = build_config(
        tmp_path / "unused",
        training_name=STATEFUL_BUILDER_NAME,
    )
    config.metrics = []

    with fake_session() as session, pytest.raises(
        ValueError,
        match=r"stateful non-primary.*runtime_cache.*buffer:cache",
    ):
        _build_ddp_training_components(
            config,
            session,
            wrapper_factory=lambda module, unused: StateSharingExecutionWrapper(
                module
            ),
        )


def test_composition_accepts_equal_frozen_process_state_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    activate_training_builtins()
    config = build_config(tmp_path / "unused")
    config.process = ComponentConfig(PROCESS_NAME)

    with fake_session() as session:
        components = _build_ddp_training_components(
            config,
            session,
            wrapper_factory=lambda module, unused: StateSharingExecutionWrapper(
                module
            ),
        )
    assert isinstance(components.process, FrozenCompositionProcess)
    assert components.checkpoint_manager.process is components.process

    with fake_session(
        world_size=2,
        disagree_on_all_equal_call=3,
    ) as session, pytest.raises(
        ValueError,
        match="managed state differs across ranks",
    ):
        _build_ddp_training_components(
            config,
            session,
            wrapper_factory=lambda module, unused: StateSharingExecutionWrapper(
                module
            ),
        )


def test_composition_accepts_builtin_gaussian_process_runtime_state(
    tmp_path: Path,
) -> None:
    activate_training_builtins()
    config = build_config(tmp_path / "unused", training_name="gaussian_denoising")
    config.process = ComponentConfig(
        "discrete_gaussian",
        {
            "schedule": {
                "name": "linear_beta",
                "params": {"num_timesteps": 4},
            }
        },
    )
    config.training.params = {"prediction_type": "epsilon"}
    config.metrics = []

    with fake_session() as session:
        components = _build_ddp_training_components(
            config,
            session,
            wrapper_factory=lambda module, unused: StateSharingExecutionWrapper(
                module
            ),
        )

    assert components.process is not None
    assert "_extra_state" in components.process.state_dict()


def test_composition_rejects_distinct_shared_storage_before_placement(
    tmp_path: Path,
) -> None:
    activate_training_builtins()
    config = build_config(tmp_path / "unused")
    config.model = ComponentConfig(SHARED_STORAGE_MODEL_NAME)

    with fake_session() as session, pytest.raises(
        ValueError,
        match=r"(?:shared|overlapping) storage",
    ):
        _build_ddp_training_components(
            config,
            session,
            wrapper_factory=lambda module, unused: StateSharingExecutionWrapper(
                module
            ),
        )


def test_composition_rejects_primary_buffers_when_broadcast_is_disabled(
    tmp_path: Path,
) -> None:
    activate_training_builtins()
    config = build_config(tmp_path / "unused")
    config.model = ComponentConfig(BUFFERED_MODEL_NAME)

    with fake_session() as session, pytest.raises(
        ValueError,
        match=r"does not yet accept primary-model buffers.*runtime_mean",
    ):
        _build_ddp_training_components(
            config,
            session,
            wrapper_factory=lambda module, unused: StateSharingExecutionWrapper(
                module
            ),
        )


def test_composition_rejects_unclosed_first_iteration_features(
    tmp_path: Path,
) -> None:
    configs: list[tuple[StochaflowConfig, str]] = []

    diagnostics = build_config(tmp_path / "diagnostics")
    diagnostics.diagnostics = [ComponentConfig("unsupported")]
    configs.append((diagnostics, "Diagnostics"))

    live_evaluation = build_config(tmp_path / "live-evaluation")
    live_evaluation.trainer.validation_evaluation.enabled = True
    configs.append((live_evaluation, "live Evaluation"))

    test_after_fit = build_config(tmp_path / "test")
    test_after_fit.trainer.test_after_fit = True
    configs.append((test_after_fit, "test-after-fit"))

    train_metric = build_config(tmp_path / "train-metric")
    train_metric.metrics[0].phases = ["train"]
    configs.append((train_metric, "validation-only"))

    epoch_scheduler = build_config(tmp_path / "epoch-scheduler")
    epoch_scheduler.lr_scheduler = LRSchedulerConfig(
        name="torch.optim.lr_scheduler.StepLR",
        interval="epoch",
        params={"step_size": 1},
    )
    configs.append((epoch_scheduler, "epoch-interval"))

    fp16 = build_config(tmp_path / "fp16")
    fp16.trainer.precision = "fp16-mixed"
    configs.append((fp16, "global GradScaler"))

    for config, message in configs:
        with fake_session() as session, pytest.raises(ValueError, match=message):
            build_ddp_training_components(config, session)


def available_port() -> int:
    """Reserve and release one localhost port for a world-one Gloo group."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def test_real_gloo_world_one_uses_ddp_without_output_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Torch Gloo distributed support is unavailable")
    if torch.distributed.is_initialized():
        pytest.skip("another test owns the process group")
    activate_training_builtins()
    environment = torchrun_environment(port=available_port())
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    output_dir = tmp_path / "must-not-be-created"
    config = build_config(output_dir)

    with DistributedSession.from_environment(
        backend="gloo",
        environ=os.environ,
    ) as session:
        components = build_ddp_training_components(config, session)
        execution_model = components.binding.execution_model
        assert isinstance(execution_model, DistributedDataParallel)
        assert execution_model.broadcast_buffers is False
        assert execution_model.find_unused_parameters is True
        assert tuple(execution_model.parameters()) == tuple(
            components.model.parameters()
        )
        assert tuple(execution_model.buffers()) == tuple(components.model.buffers())
        assert isinstance(components.logger, NullLogger)
        del components, execution_model

    assert not torch.distributed.is_initialized()
    assert not output_dir.exists()
