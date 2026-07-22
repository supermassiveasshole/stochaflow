"""Frozen-teacher training composition for the reference project."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from stochaflow.extensions import (
    ComponentConfig,
    ManagedTrainingModule,
    REGISTRIES,
    TrainStepOutput,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    compute_objective,
)

_PREFIX = "stochaflow-knowledge-distillation"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return dict(cast(Mapping[str, Any], value))


def _component(value: object, name: str) -> ComponentConfig:
    params = _mapping(value, name)
    component_name = params.pop("name", None)
    if not isinstance(component_name, str) or not component_name.strip():
        raise ValueError(f"{name}.name must be a non-empty registry name")
    component_params = _mapping(params.pop("params", {}), f"{name}.params")
    if params:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(params))}")
    return ComponentConfig(component_name, component_params)


def _plain_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"teacher bootstrap state does not exist: {path}. "
            "Run tools/create_teacher_bootstrap.py before a fresh train or resume."
        )
    value: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError("teacher bootstrap must contain a plain model state mapping")
    state: dict[str, torch.Tensor] = {}
    for declared_name, declared_tensor in value.items():
        if not isinstance(declared_name, str) or not declared_name:
            raise TypeError("teacher bootstrap state names must be non-empty strings")
        if not isinstance(declared_tensor, torch.Tensor):
            raise TypeError("teacher bootstrap values must all be Tensors")
        state[declared_name] = declared_tensor
    return state


class KnowledgeDistillationStrategy(TrainingStrategy):
    """Interpret classification batches and combine task and teacher losses."""

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        task_objective: nn.Module,
        distillation_objective: nn.Module,
        distillation_weight: float,
    ) -> None:
        self._student = student
        self._teacher = teacher
        self._task_objective = task_objective
        self._distillation_objective = distillation_objective
        self._distillation_weight = distillation_weight

    @staticmethod
    def _batch(value: object) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise TypeError("classification batches must contain inputs and labels")
        inputs, labels = value
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("classification inputs and labels must be Tensors")
        return inputs, labels

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Compute one scalar weighted distillation loss."""

        inputs, labels = self._batch(batch)
        with torch.no_grad():
            teacher_logits = self._teacher(inputs)
        student_logits = self._student(inputs)
        if not isinstance(student_logits, torch.Tensor) or not isinstance(
            teacher_logits,
            torch.Tensor,
        ):
            raise TypeError("student and teacher models must return Tensors")
        task_loss, _ = compute_objective(
            self._task_objective,
            student_logits,
            labels,
        )
        distillation_loss, _ = compute_objective(
            self._distillation_objective,
            student_logits,
            teacher_logits,
        )
        weight = self._distillation_weight
        total = (1.0 - weight) * task_loss + weight * distillation_loss
        accuracy = (student_logits.argmax(dim=1) == labels).float().mean()
        return TrainStepOutput(
            loss=total,
            metrics={
                "task_loss": task_loss.detach(),
                "distillation_loss": distillation_loss.detach(),
                "classification_accuracy": accuracy.detach(),
            },
        )


@REGISTRIES.training_builders.register(f"{_PREFIX}.training")
class DistillationTrainingBuilder(TrainingBuilder):
    """Build, initialize, freeze, and declare a managed teacher asset."""

    def build(self) -> TrainingPlan:
        """Return a plan whose auxiliary state is checkpoint-managed by core."""

        if self.context.process is not None:
            raise ValueError("classification distillation does not use a Process")
        task_objective = self.context.objective
        if task_objective is None:
            raise ValueError("classification distillation requires a task Objective")

        params = dict(self.context.params)
        teacher_declaration = _component(params.pop("teacher", None), "teacher")
        bootstrap_path = params.pop("teacher_bootstrap", None)
        if not isinstance(bootstrap_path, str) or not bootstrap_path.strip():
            raise ValueError("teacher_bootstrap must be a non-empty path string")
        distillation_declaration = _component(
            params.pop("distillation_objective", None),
            "distillation_objective",
        )
        declared_weight = params.pop("distillation_weight", 0.5)
        if (
            isinstance(declared_weight, bool)
            or not isinstance(declared_weight, (int, float))
            or not 0.0 < declared_weight < 1.0
        ):
            raise ValueError("distillation_weight must satisfy 0 < value < 1")
        if params:
            raise ValueError(f"unknown training params: {', '.join(sorted(params))}")

        teacher = self.context.model_factory(teacher_declaration)
        teacher.load_state_dict(_plain_state_dict(Path(bootstrap_path)), strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
        distillation_objective = self.context.objective_factory(
            distillation_declaration
        )
        strategy = KnowledgeDistillationStrategy(
            student=self.context.primary_model,
            teacher=teacher,
            task_objective=task_objective,
            distillation_objective=distillation_objective,
            distillation_weight=float(declared_weight),
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=task_objective,
            auxiliary_modules={
                "teacher": ManagedTrainingModule(teacher, mode="eval"),
                "distillation_objective": ManagedTrainingModule(
                    distillation_objective
                ),
            },
        )


__all__ = ["DistillationTrainingBuilder", "KnowledgeDistillationStrategy"]
