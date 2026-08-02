"""Tests for registered training composition and managed assets."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.sampling import SamplingRecipe
from stochaflow.training import (
    InferenceAssetProjection,
    ManagedTrainingModule,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
    build_training_plan,
    compute_objective,
    trainable_parameters,
    validate_training_plan,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import ComponentConfig, load_config, load_config_dict
from stochaflow.utils.factory import (
    build_model,
    build_objective,
    build_training_components,
)
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog, RegistryError
from stochaflow.utils.sampling_recipe import sampling_recipe_to_dict

BUILTIN_MNIST_TRAIN_CONFIG = Path(
    "examples/built-in/image-generation/configs/train/mnist.yaml"
)


class ScalarStrategy(TrainingStrategy):
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def training_step(self, batch: Any) -> TrainStepOutput:
        prediction = self.model(batch)
        assert isinstance(prediction, torch.Tensor)
        return TrainStepOutput(prediction.square().mean())


class ModuleStrategy(nn.Module, TrainingStrategy):
    def __init__(self) -> None:
        super().__init__()

    def training_step(self, batch: Any) -> TrainStepOutput:
        del batch
        return TrainStepOutput(torch.zeros((), requires_grad=True))


class ExtraStateModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.extra_state = {"version": 1, "labels": ["initial"]}

    def get_extra_state(self) -> dict[str, Any]:
        return self.extra_state

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.extra_state = state


class VersionedStateModule(nn.Module):
    _version = 7
    value: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(1.0))
        self.loaded_version: int | None = None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        self.loaded_version = local_metadata.get("version")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


@REGISTRIES.training_builders.register("stage4_mutating_context")
class MutatingContextBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        self.context.params["nested"]["value"] = 2
        return TrainingPlan(
            ScalarStrategy(self.context.primary_model),
            self.context.primary_model,
            self.context.process,
            self.context.objective,
        )


@REGISTRIES.training_builders.register("stage4_wrong_result")
class WrongResultBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        return cast(TrainingPlan, object())


@REGISTRIES.training_builders.register("stage4_replaces_primary")
class ReplacingPrimaryBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        replacement = nn.Linear(1, 1)
        return TrainingPlan(ScalarStrategy(replacement), replacement)


@REGISTRIES.models.register("stage4_student")
class StudentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


@REGISTRIES.models.register("stage4_teacher")
class TeacherModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


@REGISTRIES.models.register("stage4_inference_asset_model")
class InferenceAssetModel(nn.Module):
    def __init__(self, *, num_features: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))


class KnowledgeDistillationStrategy(TrainingStrategy):
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        task_objective: nn.Module,
        distill_objective: nn.Module,
        alpha: float,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.task_objective = task_objective
        self.distill_objective = distill_objective
        self.alpha = alpha

    def training_step(self, batch: Any) -> TrainStepOutput:
        inputs, targets = batch
        with torch.no_grad():
            teacher_prediction = self.teacher(inputs)
        student_prediction = self.student(inputs)
        task_loss, _ = compute_objective(
            self.task_objective,
            student_prediction,
            targets,
        )
        distill_loss, _ = compute_objective(
            self.distill_objective,
            student_prediction,
            teacher_prediction,
        )
        total = (1.0 - self.alpha) * task_loss + self.alpha * distill_loss
        return TrainStepOutput(
            total,
            metrics={
                "task_loss": task_loss.detach(),
                "distill_loss": distill_loss.detach(),
                "total_loss": total.detach(),
            },
        )


@REGISTRIES.training_builders.register("stage4_distillation")
class DistillationBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        params = dict(self.context.params)
        alpha = float(params.pop("alpha", 0.5))
        if params:
            raise ValueError("unexpected distillation params")
        objective = self.context.objective
        if objective is None:
            raise TypeError("distillation requires a task objective")
        teacher = self.context.model_factory(ComponentConfig("stage4_teacher"))
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        distill_objective = self.context.objective_factory(ComponentConfig("mse"))
        strategy = KnowledgeDistillationStrategy(
            student=self.context.primary_model,
            teacher=teacher,
            task_objective=objective,
            distill_objective=distill_objective,
            alpha=alpha,
        )
        return TrainingPlan(
            strategy,
            self.context.primary_model,
            self.context.process,
            objective,
            {
                "teacher": ManagedTrainingModule(teacher, mode="eval"),
                "distill_objective": ManagedTrainingModule(distill_objective),
            },
        )


@REGISTRIES.training_builders.register("stage4_direct_loss")
class DirectLossBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        if self.context.process is not None or self.context.objective is not None:
            raise TypeError("direct loss expects no Process or Objective")
        return TrainingPlan(
            ScalarStrategy(self.context.primary_model),
            self.context.primary_model,
        )


@REGISTRIES.training_builders.register("stage4_inference_asset")
class InferenceAssetTrainingBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        if self.context.process is not None or self.context.objective is not None:
            raise TypeError("inference asset test expects no Process or Objective")
        asset = self.context.model_factory(
            ComponentConfig(
                "stage4_inference_asset_model",
                {"num_features": 2},
            )
        )
        asset.requires_grad_(False)
        return TrainingPlan(
            strategy=ScalarStrategy(self.context.primary_model),
            primary_model=self.context.primary_model,
            auxiliary_modules={
                "calibrator": ManagedTrainingModule(asset, mode="eval"),
            },
            inference_assets={
                "calibrator": InferenceAssetProjection(
                    training_asset_name="calibrator",
                    declaration=ComponentConfig(
                        "stage4_inference_asset_model",
                        {"num_features": 2},
                    ),
                    capability_role="classification_logit_calibrator",
                ),
            },
        )


def _linear_plan(**kwargs: Any) -> TrainingPlan:
    model = kwargs.pop("primary_model", nn.Linear(1, 1))
    return TrainingPlan(ScalarStrategy(model), model, **kwargs)


def test_training_builder_registry_rejects_wrong_base() -> None:
    catalog = RegistryCatalog()
    catalog.training_builders.require_base(TrainingBuilder)

    with pytest.raises(RegistryError, match="must inherit"):
        catalog.training_builders.add("wrong", cast(Any, object))


def test_builder_context_deep_copies_params() -> None:
    declaration = ComponentConfig(
        "stage4_mutating_context",
        {"nested": {"value": 1}},
    )
    original = deepcopy(declaration.params)
    model = nn.Linear(1, 1)

    build_training_plan(
        declaration,
        primary_model=model,
        process=None,
        objective=None,
        model_factory=build_model,
        objective_factory=build_objective,
    )

    assert declaration.params == original


def test_builder_rejects_wrong_result_and_replaced_primary() -> None:
    model = nn.Linear(1, 1)
    common = {
        "primary_model": model,
        "process": None,
        "objective": None,
        "model_factory": build_model,
        "objective_factory": build_objective,
    }
    with pytest.raises(TypeError, match="must return TrainingPlan"):
        build_training_plan(ComponentConfig("stage4_wrong_result"), **common)
    with pytest.raises(ValueError, match="preserve the injected model"):
        build_training_plan(ComponentConfig("stage4_replaces_primary"), **common)


@pytest.mark.parametrize("name", ["", "   ", cast(Any, 1)])
def test_plan_rejects_invalid_auxiliary_names(name: Any) -> None:
    model = nn.Linear(1, 1)
    plan = _linear_plan(
        primary_model=model,
        auxiliary_modules={name: ManagedTrainingModule(nn.Linear(1, 1))},
    )

    with pytest.raises(ValueError, match="auxiliary names"):
        validate_training_plan(plan)


@pytest.mark.parametrize("name", ["primary_model", "process", "objective"])
def test_plan_rejects_reserved_auxiliary_names(name: str) -> None:
    plan = _linear_plan(
        auxiliary_modules={name: ManagedTrainingModule(nn.Linear(1, 1))},
    )

    with pytest.raises(ValueError, match="reserved by TrainingPlan"):
        validate_training_plan(plan)


def test_plan_rejects_module_strategy_and_snapshots_auxiliaries() -> None:
    model = nn.Linear(1, 1)
    with pytest.raises(TypeError, match=r"must not inherit nn\.Module"):
        validate_training_plan(TrainingPlan(ModuleStrategy(), model))

    auxiliaries = {"teacher": ManagedTrainingModule(nn.Linear(1, 1))}
    validated = validate_training_plan(
        _linear_plan(primary_model=model, auxiliary_modules=auxiliaries)
    )
    auxiliaries["late"] = ManagedTrainingModule(nn.Linear(1, 1))

    assert set(validated.auxiliary_modules) == {"teacher"}
    with pytest.raises(TypeError):
        cast(Any, validated.auxiliary_modules)["late"] = ManagedTrainingModule(
            nn.Linear(1, 1)
        )


def test_plan_validates_and_snapshots_inference_recipe() -> None:
    contract = {
        "prediction_type": "v",
        "schedule": {"steps": [4, 2, 0]},
    }
    validated = validate_training_plan(
        _linear_plan(
            inference_recipe=SamplingRecipe(
                name="project.generate",
                contract=contract,
            )
        )
    )
    contract["prediction_type"] = "epsilon"
    cast(dict[str, list[int]], contract["schedule"])["steps"].append(-1)

    recipe = validated.inference_recipe
    assert recipe is not None
    assert sampling_recipe_to_dict(recipe) == {
        "schema_version": 1,
        "name": "project.generate",
        "contract": {
            "prediction_type": "v",
            "schedule": {"steps": [4, 2, 0]},
        },
    }
    schedule = cast(Mapping[str, Any], recipe.contract["schedule"])
    steps = cast(tuple[int, ...], schedule["steps"])
    with pytest.raises(TypeError):
        cast(Any, schedule)["late"] = True
    with pytest.raises(AttributeError):
        cast(Any, steps).append(-1)
    with pytest.raises(TypeError, match="must be SamplingRecipe"):
        validate_training_plan(
            _linear_plan(inference_recipe=cast(Any, {"name": "invalid"}))
        )
    with pytest.raises(TypeError, match=r"unsupported value type.*tuple"):
        validate_training_plan(
            _linear_plan(
                inference_recipe=SamplingRecipe(
                    name="project.generate",
                    contract={"steps": (4, 2, 0)},
                )
            )
        )


def test_plan_validates_and_snapshots_inference_assets() -> None:
    asset = nn.Linear(1, 1)
    params = {"nested": {"value": 1}}
    declaration = ComponentConfig("test_codec", params)
    inference_assets = {
        "codec": InferenceAssetProjection(
            training_asset_name="codec",
            declaration=declaration,
            capability_role="image_codec",
        )
    }
    validated = validate_training_plan(
        _linear_plan(
            auxiliary_modules={
                "codec": ManagedTrainingModule(asset, mode="eval"),
            },
            inference_assets=inference_assets,
        )
    )

    declaration.name = "mutated"
    params["nested"]["value"] = 2
    inference_assets["late"] = InferenceAssetProjection(
        training_asset_name="codec",
        declaration=ComponentConfig("late"),
        capability_role="late",
    )

    assert set(validated.inference_assets) == {"codec"}
    projection = validated.inference_assets["codec"]
    assert projection.training_asset_name == "codec"
    assert projection.declaration.name == "test_codec"
    assert projection.declaration.params == {"nested": {"value": 1}}
    assert projection.capability_role == "image_codec"
    with pytest.raises(TypeError):
        cast(Any, validated.inference_assets)["late"] = projection


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("slot", 1, TypeError, "slot must be an exact string"),
        ("slot", "", ValueError, "slot must be non-empty"),
        (
            "slot",
            " codec ",
            ValueError,
            "slot must not contain surrounding whitespace",
        ),
        (
            "training_asset_name",
            1,
            TypeError,
            r"training_asset_name must be an exact string",
        ),
        (
            "training_asset_name",
            " codec ",
            ValueError,
            r"training_asset_name.*surrounding whitespace",
        ),
        (
            "declaration_name",
            1,
            TypeError,
            r"declaration\.name must be an exact string",
        ),
        (
            "declaration_name",
            "",
            ValueError,
            r"declaration\.name must be non-empty",
        ),
        (
            "capability_role",
            1,
            TypeError,
            r"capability_role must be an exact string",
        ),
        (
            "capability_role",
            " image_codec",
            ValueError,
            r"capability_role.*surrounding whitespace",
        ),
    ],
)
def test_plan_rejects_invalid_inference_asset_strings(
    field: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    slot: Any = value if field == "slot" else "codec"
    projection = InferenceAssetProjection(
        training_asset_name=cast(
            str,
            value if field == "training_asset_name" else "codec",
        ),
        declaration=ComponentConfig(
            cast(str, value if field == "declaration_name" else "test_codec"),
        ),
        capability_role=cast(
            str,
            value if field == "capability_role" else "image_codec",
        ),
    )

    with pytest.raises(error, match=message):
        validate_training_plan(
            _linear_plan(
                auxiliary_modules={
                    "codec": ManagedTrainingModule(nn.Linear(1, 1)),
                },
                inference_assets={slot: projection},
            )
        )


def test_plan_rejects_invalid_or_duplicate_inference_asset_projection() -> None:
    auxiliary_modules = {
        "codec": ManagedTrainingModule(nn.Linear(1, 1)),
    }
    projection = InferenceAssetProjection(
        training_asset_name="codec",
        declaration=ComponentConfig("test_codec"),
        capability_role="image_codec",
    )
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_training_plan(
            _linear_plan(inference_assets=cast(Any, []))
        )
    with pytest.raises(TypeError, match="must be InferenceAssetProjection"):
        validate_training_plan(
            _linear_plan(
                auxiliary_modules=auxiliary_modules,
                inference_assets={"codec": cast(Any, object())},
            )
        )
    with pytest.raises(ValueError, match="missing training auxiliary"):
        validate_training_plan(
            _linear_plan(inference_assets={"codec": projection})
        )
    with pytest.raises(TypeError, match=r"declaration\.params"):
        validate_training_plan(
            _linear_plan(
                auxiliary_modules=auxiliary_modules,
                inference_assets={
                    "codec": InferenceAssetProjection(
                        training_asset_name="codec",
                        declaration=ComponentConfig(
                            "test_codec",
                            cast(Any, []),
                        ),
                        capability_role="image_codec",
                    )
                },
            )
        )
    with pytest.raises(ValueError, match="more than once"):
        validate_training_plan(
            _linear_plan(
                auxiliary_modules=auxiliary_modules,
                inference_assets={
                    "encoder": projection,
                    "decoder": projection,
                },
            )
        )


def test_plan_rejects_invalid_mode_duplicate_and_shared_state() -> None:
    model = nn.Linear(1, 1)
    bad_mode = _linear_plan(
        primary_model=model,
        auxiliary_modules={
            "bad": ManagedTrainingModule(model, mode=cast(Any, "frozen"))
        },
    )
    with pytest.raises(ValueError, match="mode"):
        validate_training_plan(bad_mode)

    duplicate = _linear_plan(
        primary_model=model,
        auxiliary_modules={"duplicate": ManagedTrainingModule(model)},
    )
    with pytest.raises(ValueError, match="same module"):
        validate_training_plan(duplicate)

    first = nn.Linear(1, 1)
    second = nn.Linear(1, 1)
    second.weight = first.weight
    shared = _linear_plan(
        primary_model=first,
        auxiliary_modules={"second": ManagedTrainingModule(second)},
    )
    with pytest.raises(ValueError, match="state is shared"):
        validate_training_plan(shared)


def test_plan_rejects_no_trainable_parameters() -> None:
    model = nn.Linear(1, 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="at least one trainable"):
        validate_training_plan(_linear_plan(primary_model=model))


def test_auxiliary_name_order_does_not_change_optimizer_state_binding(
    tmp_path: Path,
) -> None:
    first_primary = nn.Linear(1, 1)
    first_primary.requires_grad_(False)
    first_alpha = nn.Linear(1, 1)
    first_omega = nn.Linear(1, 1)
    first_plan = validate_training_plan(
        TrainingPlan(
            ScalarStrategy(first_primary),
            first_primary,
            auxiliary_modules={
                "omega": ManagedTrainingModule(first_omega),
                "alpha": ManagedTrainingModule(first_alpha),
            },
        )
    )
    first_optimizer = torch.optim.Adam(trainable_parameters(first_plan), lr=0.1)
    for parameter in first_alpha.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in first_omega.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    first_optimizer.step()
    expected_alpha = [
        first_optimizer.state[parameter]["exp_avg"].clone()
        for parameter in first_alpha.parameters()
    ]
    expected_omega = [
        first_optimizer.state[parameter]["exp_avg"].clone()
        for parameter in first_omega.parameters()
    ]
    checkpoint = tmp_path / "ordered-assets.pt"
    CheckpointManager(
        model=first_primary,
        auxiliary_modules={"omega": first_omega, "alpha": first_alpha},
        optimizer=first_optimizer,
    ).save(checkpoint)

    second_primary = nn.Linear(1, 1)
    second_primary.requires_grad_(False)
    second_alpha = nn.Linear(1, 1)
    second_omega = nn.Linear(1, 1)
    second_plan = validate_training_plan(
        TrainingPlan(
            ScalarStrategy(second_primary),
            second_primary,
            auxiliary_modules={
                "alpha": ManagedTrainingModule(second_alpha),
                "omega": ManagedTrainingModule(second_omega),
            },
        )
    )
    second_optimizer = torch.optim.Adam(trainable_parameters(second_plan), lr=0.1)
    CheckpointManager(
        model=second_primary,
        auxiliary_modules={"alpha": second_alpha, "omega": second_omega},
        optimizer=second_optimizer,
    ).load(checkpoint)

    for expected, parameter in zip(
        expected_alpha,
        second_alpha.parameters(),
        strict=True,
    ):
        assert torch.equal(second_optimizer.state[parameter]["exp_avg"], expected)
    for expected, parameter in zip(
        expected_omega,
        second_omega.parameters(),
        strict=True,
    ):
        assert torch.equal(second_optimizer.state[parameter]["exp_avg"], expected)


def test_checkpoint_preserves_data_only_module_extra_state(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    asset = ExtraStateModule()
    manager = CheckpointManager(model=model, auxiliary_modules={"asset": asset})
    checkpoint = manager.save(tmp_path / "extra-state.pt")
    asset.extra_state["version"] = 99
    cast(list[str], asset.extra_state["labels"]).append("mutated")

    manager.load(checkpoint)

    assert asset.extra_state == {"version": 1, "labels": ["initial"]}


def test_checkpoint_preserves_pytorch_state_dict_metadata(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    asset = VersionedStateModule()
    manager = CheckpointManager(model=model, auxiliary_modules={"asset": asset})
    checkpoint = manager.save(tmp_path / "versioned-state.pt")
    asset.value.zero_()

    manager.load(checkpoint)

    assert asset.loaded_version == VersionedStateModule._version
    assert asset.value.item() == pytest.approx(1.0)


def test_distillation_builder_drives_runtime_and_checkpoint(tmp_path: Path) -> None:
    raw = load_config(BUILTIN_MNIST_TRAIN_CONFIG).to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["model"] = {"name": "stage4_student", "params": {}}
    raw["process"] = None
    raw["training"] = {
        "name": "stage4_distillation",
        "params": {"alpha": 0.25},
    }
    raw["objective"] = {"name": "mse", "params": {}}
    raw["metrics"] = []
    raw["diagnostics"] = []
    raw["lr_scheduler"] = None
    raw["ema"]["enabled"] = False
    components = build_training_components(load_config_dict(raw))

    strategy = components.plan.strategy
    assert isinstance(strategy, KnowledgeDistillationStrategy)
    teacher = components.plan.auxiliary_modules["teacher"].module
    optimizer_parameter_ids = {
        id(parameter)
        for group in components.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(
        id(parameter) not in optimizer_parameter_ids
        for parameter in teacher.parameters()
    )
    assert {id(parameter) for parameter in trainable_parameters(components.plan)} == {
        id(parameter) for parameter in components.model.parameters()
    }

    before = [parameter.detach().clone() for parameter in components.model.parameters()]
    components.trainer.train_batch(
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[0.0], [0.0]]))
    )
    assert components.model.training
    assert not teacher.training
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, components.model.parameters(), strict=True)
    )
    output = strategy.training_step(
        (
            torch.tensor([[1.0]], device=components.trainer.device),
            torch.tensor([[0.0]], device=components.trainer.device),
        )
    )
    assert set(output.metrics) == {"task_loss", "distill_loss", "total_loss"}

    checkpoint = tmp_path / "distillation.pt"
    components.checkpoint_manager.save(checkpoint)
    payload = CheckpointManager.load_payload(checkpoint)
    assets_state = payload.get("training_assets_state_dict")
    assert isinstance(assets_state, dict)
    assert set(assets_state) == {
        "teacher",
        "distill_objective",
    }
    expected_teacher = {
        name: value.detach().clone() for name, value in teacher.state_dict().items()
    }
    for parameter in teacher.parameters():
        parameter.data.add_(10.0)
    components.checkpoint_manager.load(checkpoint)
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, expected_teacher[name])

    mismatched = deepcopy(payload)
    mismatched_assets = mismatched.get("training_assets_state_dict")
    assert isinstance(mismatched_assets, dict)
    mismatched_assets["renamed"] = mismatched_assets.pop("teacher")
    mismatched_path = tmp_path / "mismatched.pt"
    torch.save(mismatched, mismatched_path)
    with pytest.raises(ValueError, match="asset names do not match"):
        components.checkpoint_manager.load(mismatched_path)


def test_custom_builder_trains_without_process_or_objective(tmp_path: Path) -> None:
    raw = load_config(BUILTIN_MNIST_TRAIN_CONFIG).to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["model"] = {"name": "stage4_student", "params": {}}
    raw["process"] = None
    raw["training"] = {"name": "stage4_direct_loss", "params": {}}
    raw["objective"] = None
    raw["metrics"] = []
    raw["diagnostics"] = []
    raw["lr_scheduler"] = None
    raw["ema"]["enabled"] = False

    components = build_training_components(load_config_dict(raw))
    loss = components.trainer.train_batch(torch.tensor([[1.0], [2.0]]))
    state = components.checkpoint_manager.build_state()

    assert loss >= 0.0
    assert components.process is None
    assert components.objective is None
    assert "process_state_dict" not in state
    assert "objective_state_dict" not in state
    assert state.get("training_assets_state_dict") == {}


def test_standard_factory_projects_inference_assets_into_checkpoint(
    tmp_path: Path,
) -> None:
    raw = load_config(BUILTIN_MNIST_TRAIN_CONFIG).to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["model"] = {"name": "stage4_student", "params": {}}
    raw["process"] = None
    raw["training"] = {"name": "stage4_inference_asset", "params": {}}
    raw["objective"] = None
    raw["metrics"] = []
    raw["diagnostics"] = []
    raw["lr_scheduler"] = None
    raw["ema"]["enabled"] = False

    components = build_training_components(load_config_dict(raw))
    state = components.checkpoint_manager.build_state()

    assert components.checkpoint_manager.inference_asset_descriptors == {
        "calibrator": {
            "training_asset_name": "calibrator",
            "declaration": {
                "name": "stage4_inference_asset_model",
                "params": {"num_features": 2},
            },
            "capability_role": "classification_logit_calibrator",
            "persistence": "embedded_state",
        }
    }
    assert state.get("inference_asset_descriptors") == (
        components.checkpoint_manager.inference_asset_descriptors
    )
    assets_state = state.get("training_assets_state_dict")
    assert isinstance(assets_state, dict)
    assert set(assets_state) == {"calibrator"}
    assert set(assets_state["calibrator"]) == {"scale"}


def test_strategy_has_no_lifecycle_or_state_api() -> None:
    strategy = ScalarStrategy(nn.Linear(1, 1))

    for name in ("to", "train", "eval", "parameters", "state_dict"):
        assert not hasattr(strategy, name)
