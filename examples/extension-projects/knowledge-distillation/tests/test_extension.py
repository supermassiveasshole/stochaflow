"""Focused contract tests for the installed reference extension."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from stochaflow.extensions import (
    DataBuilderContext,
    InferenceModelProvider,
    REGISTRIES,
    SamplingBuilderContext,
    TrainingBuilderContext,
    TrainingPlan,
)
from stochaflow.training import trainable_parameters
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.factory import build_model, build_objective
from stochaflow_knowledge_distillation import stochaflow_ext
from stochaflow_knowledge_distillation.stochaflow_ext.data import (
    ClassificationDataBuilder,
    _EpochShuffleSampler,
)
from stochaflow_knowledge_distillation.stochaflow_ext.models import (
    StudentClassifier,
    TeacherClassifier,
)
from stochaflow_knowledge_distillation.stochaflow_ext.objectives import (
    ClassificationCrossEntropy,
    TemperatureKLDistillation,
)
from stochaflow_knowledge_distillation.stochaflow_ext.sampling import (
    StudentPredictionBuilder,
)
from stochaflow_knowledge_distillation.stochaflow_ext.training import (
    DistillationTrainingBuilder,
)

del stochaflow_ext

_PREFIX = "stochaflow-knowledge-distillation"


def test_components_are_namespaced_and_registered() -> None:
    assert f"{_PREFIX}.classification" in REGISTRIES.data_builders
    assert f"{_PREFIX}.student" in REGISTRIES.models
    assert f"{_PREFIX}.teacher" in REGISTRIES.models
    assert f"{_PREFIX}.cross-entropy" in REGISTRIES.objectives
    assert f"{_PREFIX}.temperature-kl" in REGISTRIES.objectives
    assert f"{_PREFIX}.training" in REGISTRIES.training_builders
    assert f"{_PREFIX}.predictions" in REGISTRIES.sampling_builders


def test_synthetic_data_is_deterministic() -> None:
    params = {
        "source": {
            "kind": "synthetic",
            "input_features": 3,
            "num_classes": 2,
            "train_samples": 8,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "loader": {"batch_size": 4, "shuffle": False},
    }
    first = ClassificationDataBuilder(DataBuilderContext(params, seed=7)).build()
    second = ClassificationDataBuilder(DataBuilderContext(params, seed=7)).build()
    first_inputs, first_labels = next(iter(first.train))
    second_inputs, second_labels = next(iter(second.train))
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_labels, second_labels)


@pytest.mark.parametrize("noise_std", [float("nan"), float("inf")])
def test_synthetic_data_rejects_non_finite_noise(noise_std: float) -> None:
    builder = ClassificationDataBuilder(
        DataBuilderContext(
            {
                "source": {
                    "kind": "synthetic",
                    "train_samples": 8,
                    "validation_samples": 4,
                    "test_samples": 4,
                    "noise_std": noise_std,
                },
                "loader": {"batch_size": 4},
            },
            seed=7,
        )
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        builder.build()


def test_shuffle_order_is_epoch_derived_and_resume_rebuild_safe() -> None:
    params = {
        "source": {
            "kind": "synthetic",
            "input_features": 3,
            "num_classes": 2,
            "train_samples": 12,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "loader": {"batch_size": 4, "shuffle": True},
    }
    uninterrupted = ClassificationDataBuilder(
        DataBuilderContext(params, seed=17)
    ).build()
    assert isinstance(uninterrupted.train, DataLoader)
    sampler = uninterrupted.train.sampler
    assert isinstance(sampler, _EpochShuffleSampler)
    sampler.set_epoch(1)
    epoch_one = tuple(sampler)
    sampler.set_epoch(2)
    uninterrupted_epoch_two = tuple(sampler)

    rebuilt = ClassificationDataBuilder(DataBuilderContext(params, seed=17)).build()
    assert isinstance(rebuilt.train, DataLoader)
    rebuilt_sampler = rebuilt.train.sampler
    assert isinstance(rebuilt_sampler, _EpochShuffleSampler)
    rebuilt_sampler.set_epoch(2)

    assert epoch_one != uninterrupted_epoch_two
    assert tuple(rebuilt_sampler) == uninterrupted_epoch_two


def test_shuffle_false_preserves_sequential_order() -> None:
    params = {
        "source": {
            "kind": "synthetic",
            "input_features": 3,
            "num_classes": 2,
            "train_samples": 8,
            "validation_samples": 4,
            "test_samples": 4,
        },
        "loader": {"batch_size": 4, "shuffle": False},
    }
    loaders = ClassificationDataBuilder(DataBuilderContext(params, seed=17)).build()
    assert isinstance(loaders.train, DataLoader)
    assert isinstance(loaders.train.sampler, SequentialSampler)
    assert tuple(loaders.train.sampler) == tuple(range(8))


def test_objectives_return_scalar_and_temperature_is_stateful() -> None:
    task = ClassificationCrossEntropy()
    distillation = TemperatureKLDistillation(temperature=3.0)
    student_logits = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
    teacher_logits = torch.tensor([[1.5, -0.5], [-0.5, 1.5]])
    labels = torch.tensor([0, 1])

    assert task(student_logits, labels).ndim == 0
    assert distillation(student_logits, teacher_logits).ndim == 0
    assert distillation.state_dict()["temperature"].item() == 3.0


def _teacher_state(path: Path, *, fill: float) -> None:
    teacher = TeacherClassifier(
        input_features=3,
        hidden_features=5,
        num_classes=2,
    )
    for value in teacher.state_dict().values():
        value.fill_(fill)
    torch.save(teacher.state_dict(), path)


def _distillation_plan(
    bootstrap: Path,
    *,
    temperature: float = 2.0,
) -> TrainingPlan:
    student = StudentClassifier(
        input_features=3,
        hidden_features=4,
        num_classes=2,
    )
    task = ClassificationCrossEntropy()
    context = TrainingBuilderContext(
        params={
            "teacher": {
                "name": f"{_PREFIX}.teacher",
                "params": {
                    "input_features": 3,
                    "hidden_features": 5,
                    "num_classes": 2,
                },
            },
            "teacher_bootstrap": str(bootstrap),
            "distillation_objective": {
                "name": f"{_PREFIX}.temperature-kl",
                "params": {"temperature": temperature},
            },
            "distillation_weight": 0.4,
        },
        primary_model=student,
        process=None,
        objective=task,
        model_factory=build_model,
        objective_factory=build_objective,
    )
    return DistillationTrainingBuilder(context).build()


def test_builder_freezes_teacher_and_strategy_updates_only_student(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "teacher.pt"
    _teacher_state(bootstrap, fill=0.25)
    plan = _distillation_plan(bootstrap)
    student = plan.primary_model
    teacher = plan.auxiliary_modules["teacher"].module
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    selected = trainable_parameters(plan)
    assert {id(parameter) for parameter in selected} == {
        id(parameter) for parameter in student.parameters()
    }
    optimizer = torch.optim.SGD(selected, lr=0.1)
    before = [parameter.detach().clone() for parameter in student.parameters()]
    teacher_before = {
        name: value.detach().clone() for name, value in teacher.state_dict().items()
    }
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    output = plan.strategy.training_step((inputs, labels))
    optimizer.zero_grad()
    output.loss.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not teacher.training
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_before[name])
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, student.parameters(), strict=True)
    )


def test_checkpoint_assets_override_new_bootstrap_and_objective_state(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "teacher.pt"
    _teacher_state(bootstrap, fill=0.25)
    original = _distillation_plan(bootstrap, temperature=2.0)
    original_assets = {
        name: asset.module for name, asset in original.auxiliary_modules.items()
    }
    original_optimizer = torch.optim.SGD(trainable_parameters(original), lr=0.1)
    original_manager = CheckpointManager(
        model=original.primary_model,
        process=None,
        objective=original.objective,
        auxiliary_modules=original_assets,
        optimizer=original_optimizer,
    )
    payload = original_manager.build_state()
    asset_state = payload.get("training_assets_state_dict")
    assert isinstance(asset_state, dict)
    assert set(asset_state) == {"teacher", "distillation_objective"}

    expected_teacher = {
        name: value.detach().clone()
        for name, value in original_assets["teacher"].state_dict().items()
    }
    _teacher_state(bootstrap, fill=9.0)
    resumed = _distillation_plan(bootstrap, temperature=7.0)
    resumed_assets = {
        name: asset.module for name, asset in resumed.auxiliary_modules.items()
    }
    resumed_temperature = resumed_assets["distillation_objective"].state_dict()[
        "temperature"
    ]
    assert resumed_temperature.item() == 7.0
    assert any(
        not torch.equal(value, expected_teacher[name])
        for name, value in resumed_assets["teacher"].state_dict().items()
    )

    resumed_manager = CheckpointManager(
        model=resumed.primary_model,
        process=None,
        objective=resumed.objective,
        auxiliary_modules=resumed_assets,
        optimizer=torch.optim.SGD(trainable_parameters(resumed), lr=0.1),
    )
    resumed_manager.restore_payload(payload, path=tmp_path / "resume.pt")
    for name, value in resumed_assets["teacher"].state_dict().items():
        assert torch.equal(value, expected_teacher[name])
    assert resumed_assets["distillation_objective"].state_dict()[
        "temperature"
    ].item() == 2.0


def test_sampling_builds_only_student_predictions() -> None:
    model = StudentClassifier(input_features=3, hidden_features=4, num_classes=2)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    provider = InferenceModelProvider(
        model_factory=lambda: StudentClassifier(
            input_features=3,
            hidden_features=4,
            num_classes=2,
        ),
        raw_state_dict=state,
        ema_state_dict=None,
        device=torch.device("cpu"),
        prefer_ema=False,
    )
    builder = StudentPredictionBuilder(
        SamplingBuilderContext(
            params={"input_features": 3, "weights": "raw"},
            process=None,
            model_provider=provider,
            device=torch.device("cpu"),
            seed=9,
            shape=None,
            num_samples=5,
            batch_size=2,
        )
    )
    output = builder.run()
    assert [batch.samples.shape for batch in output.batches] == [
        torch.Size((2, 2)),
        torch.Size((2, 2)),
        torch.Size((1, 2)),
    ]
    assert output.metadata["workflow"] == "student-only-classification"
