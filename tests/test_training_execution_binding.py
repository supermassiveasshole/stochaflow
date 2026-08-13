"""Tests for task-owned primary execution-model binding."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training import (
    MSEObjective,
    TrainingBuilder,
    TrainingPlan,
    TrainingPlanAssembly,
    TrainingStrategy,
    TrainStepOutput,
    build_training_plan_assembly,
)
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import RegistryCatalog


class RecordingExecutionWrapper(nn.Module):
    """Record calls while preserving one wrapped module's exact state."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.module(*args, **kwargs)


class ExtraStateExecutionWrapper(RecordingExecutionWrapper):
    """Add state that a primary execution wrapper may not introduce."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)
        self.extra = nn.Parameter(torch.zeros(()))


class ExtraPythonStateExecutionWrapper(RecordingExecutionWrapper):
    """Add non-tensor state that parallel execution cannot checkpoint."""

    def get_extra_state(self) -> dict[str, int]:
        return {"revision": 1}

    def set_extra_state(self, state: object) -> None:
        del state


class CustomStateExecutionWrapper(RecordingExecutionWrapper):
    """Add tensor state without registering it as a parameter or buffer."""

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        revision = torch.tensor(1.0)
        destination[f"{prefix}revision"] = revision if keep_vars else revision.detach()


class ScalarSupervisedModel(nn.Module):
    """Provide a small supervised model for execution binding tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight


class ExtraPythonStatePrimaryModel(ScalarSupervisedModel):
    """Expose primary extra state that DDP cannot synchronize."""

    def get_extra_state(self) -> dict[str, int]:
        return {"revision": 1}

    def set_extra_state(self, state: object) -> None:
        del state


class MissingParameterStatePrimaryModel(ScalarSupervisedModel):
    """Illegally omit a trainable parameter from serialized model state."""

    def __init__(self) -> None:
        super().__init__()

        def remove_weight(
            module: nn.Module,
            state: dict[str, Any],
            prefix: str,
            local_metadata: dict[str, Any],
        ) -> None:
            del module, local_metadata
            state.pop(f"{prefix}weight")

        self.register_state_dict_post_hook(remove_weight)


class TiedWeightPrimaryModel(nn.Module):
    """Expose one legal shared parameter under two state-dict names."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(1, 1, bias=False)
        self.right = nn.Linear(1, 1, bias=False)
        self.right.weight = self.left.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.left(inputs) + self.right(inputs)


class GaussianExecutionModel(nn.Module):
    """Return a shape-preserving unconditional Gaussian prediction."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        return torch.zeros_like(state) + self.offset


class CapabilityOnlyConditionalModel(nn.Module):
    """Implement conditional prediction without a compatible forward method."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.capability_calls = 0

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def null_class_id(self) -> int:
        return 3

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        del model_time, class_labels
        self.capability_calls += 1
        return torch.zeros_like(state) + self.offset


class ExtensionExecutionStrategy(TrainingStrategy):
    """Exercise a third-party Builder without built-in task knowledge."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(self, batch: Any) -> TrainStepOutput:
        prediction = self.model(batch)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("extension execution model must return a Tensor")
        return TrainStepOutput(prediction.square().mean())


class ExtensionExecutionBindingBuilder(TrainingBuilder):
    """Provide execution binding structurally as an external Builder would."""

    def build(self) -> TrainingPlan:
        return TrainingPlan(
            strategy=ExtensionExecutionStrategy(self.context.primary_model),
            primary_model=self.context.primary_model,
        )

    def build_primary_execution_module(self, plan: TrainingPlan) -> nn.Module:
        if plan.primary_model is not self.context.primary_model:
            raise ValueError("extension binding received a foreign plan")
        return plan.primary_model

    def bind_primary_execution_model(
        self,
        plan: TrainingPlan,
        execution_model: nn.Module,
    ) -> TrainingStrategy:
        if plan.primary_model is not self.context.primary_model:
            raise ValueError("extension binding received a foreign plan")
        return ExtensionExecutionStrategy(execution_model)


def unavailable_model_factory(config: ComponentConfig) -> nn.Module:
    del config
    raise AssertionError("test execution binding must not construct another model")


def unavailable_objective_factory(config: ComponentConfig) -> nn.Module:
    del config
    raise AssertionError("test execution binding must not construct an Objective")


def gaussian_process() -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 4}}
    )


def build_assembly(
    name: str,
    *,
    model: nn.Module,
    process: DiscreteGaussianProcess | None,
    objective: nn.Module,
    params: dict[str, Any] | None = None,
) -> TrainingPlanAssembly:
    return build_training_plan_assembly(
        ComponentConfig(name, params or {}),
        primary_model=model,
        process=process,
        objective=objective,
        model_factory=unavailable_model_factory,
        objective_factory=unavailable_objective_factory,
    )


def test_execution_binding_is_an_optional_builder_capability() -> None:
    model = ScalarSupervisedModel()
    assembly = build_assembly(
        "supervised",
        model=model,
        process=None,
        objective=MSEObjective(),
    )

    assert assembly.supports_primary_execution_binding
    assert assembly.build_primary_execution_module() is model

    unbound = TrainingPlanAssembly(
        plan=assembly.plan,
        builder_name="test.unbound",
        _execution_binding=None,
    )
    with pytest.raises(TypeError, match="does not support primary execution"):
        unbound.build_primary_execution_module()


def test_structural_extension_builder_can_bind_without_core_dispatch() -> None:
    catalog = RegistryCatalog()
    catalog.training_builders.require_base(TrainingBuilder)
    catalog.training_builders.add(
        "extension_execution_binding",
        ExtensionExecutionBindingBuilder,
    )
    model = ScalarSupervisedModel()
    assembly = build_training_plan_assembly(
        ComponentConfig("extension_execution_binding"),
        primary_model=model,
        process=None,
        objective=None,
        model_factory=unavailable_model_factory,
        objective_factory=unavailable_objective_factory,
        registries=catalog,
    )
    wrapper = RecordingExecutionWrapper(
        assembly.build_primary_execution_module()
    )

    strategy = assembly.bind_primary_execution_model(wrapper)
    output = strategy.training_step(torch.ones(2, 1))

    assert output.loss.ndim == 0
    assert wrapper.calls == 1
    assert assembly.plan.primary_model is model


def test_supervised_binding_uses_wrapper_without_changing_canonical_plan() -> None:
    model = ScalarSupervisedModel()
    assembly = build_assembly(
        "supervised",
        model=model,
        process=None,
        objective=MSEObjective(),
    )
    canonical_strategy = assembly.plan.strategy
    wrapper = RecordingExecutionWrapper(
        assembly.build_primary_execution_module()
    )

    strategy = assembly.bind_primary_execution_model(wrapper)
    output = strategy.training_step(
        (torch.ones(2, 1), torch.zeros(2, 1))
    )

    assert output.loss.ndim == 0
    assert wrapper.calls == 1
    assert strategy is not canonical_strategy
    assert assembly.plan.primary_model is model


def test_execution_binding_rejects_added_or_reordered_primary_state() -> None:
    model = ScalarSupervisedModel()
    assembly = build_assembly(
        "supervised",
        model=model,
        process=None,
        objective=MSEObjective(),
    )

    with pytest.raises(ValueError, match="parameter identities and order"):
        assembly.bind_primary_execution_model(ExtraStateExecutionWrapper(model))

    with pytest.raises(ValueError, match="not a registered canonical"):
        assembly.bind_primary_execution_model(
            ExtraPythonStateExecutionWrapper(model)
        )

    with pytest.raises(ValueError, match="not a registered canonical"):
        assembly.bind_primary_execution_model(CustomStateExecutionWrapper(model))


def test_execution_binding_rejects_primary_extra_state() -> None:
    model = ExtraPythonStatePrimaryModel()
    assembly = build_assembly(
        "supervised",
        model=model,
        process=None,
        objective=MSEObjective(),
    )

    with pytest.raises(ValueError, match="not a registered canonical"):
        assembly.build_primary_execution_module()


def test_execution_binding_rejects_missing_canonical_parameter_state() -> None:
    model = MissingParameterStatePrimaryModel()
    assembly = build_assembly(
        "supervised",
        model=model,
        process=None,
        objective=MSEObjective(),
    )

    with pytest.raises(ValueError, match="contain every canonical parameter"):
        assembly.build_primary_execution_module()


def test_execution_binding_accepts_tied_primary_parameter_aliases() -> None:
    model = TiedWeightPrimaryModel()
    assembly = build_assembly(
        "supervised",
        model=model,
        process=None,
        objective=MSEObjective(),
    )

    assert assembly.build_primary_execution_module() is model


def test_unconditional_gaussian_binding_uses_wrapper_forward() -> None:
    model = GaussianExecutionModel()
    process = gaussian_process()
    assembly = build_assembly(
        "gaussian_denoising",
        model=model,
        process=process,
        objective=MSEObjective(),
        params={"prediction_type": "epsilon"},
    )
    wrapper = RecordingExecutionWrapper(
        assembly.build_primary_execution_module()
    )

    strategy = assembly.bind_primary_execution_model(wrapper)
    output = strategy.training_step(torch.zeros(2, 1, 2, 2))

    assert output.loss.ndim == 0
    assert wrapper.calls == 1
    assert strategy is not assembly.plan.strategy
    assert assembly.plan.primary_model is model


def test_class_conditional_binding_adapts_capability_only_model_to_forward() -> None:
    model = CapabilityOnlyConditionalModel()
    process = gaussian_process()
    assembly = build_assembly(
        "class_conditional_gaussian_denoising",
        model=model,
        process=process,
        objective=MSEObjective(),
        params={"prediction_type": "epsilon", "condition_dropout": 0.0},
    )
    execution_root = assembly.build_primary_execution_module()
    assert tuple(execution_root.parameters()) == tuple(model.parameters())
    wrapper = RecordingExecutionWrapper(execution_root)

    strategy = assembly.bind_primary_execution_model(wrapper)
    output = strategy.training_step(
        (
            torch.zeros(2, 1, 2, 2),
            {"class_label": torch.tensor([0, 2])},
        )
    )

    assert output.loss.ndim == 0
    assert wrapper.calls == 1
    assert model.capability_calls == 1
    assert strategy is not assembly.plan.strategy
    assert assembly.plan.primary_model is model
