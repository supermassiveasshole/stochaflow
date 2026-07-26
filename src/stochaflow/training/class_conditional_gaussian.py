"""Class-conditional Gaussian-family training composition."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from stochaflow.models.conditioning import (
    ClassConditionalDenoiser,
    predict_prevalidated_class_conditioned,
)
from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.sampling.gaussian import GaussianModelDynamics, PredictionType
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.training.gaussian import gaussian_training_target
from stochaflow.training.objectives import compute_objective
from stochaflow.training.strategy import TrainingStrategy, TrainStepOutput
from stochaflow.utils.registry import REGISTRIES


@runtime_checkable
class ClassConditionalGaussianDiagnosticSemantics(Protocol):
    """Optional class-conditional Gaussian invocation used by diagnostics."""

    @property
    def prediction_type(self) -> PredictionType:
        """Return the model prediction parameterization used for training."""

        ...

    @property
    def num_classes(self) -> int:
        """Return the number of non-null classes accepted by the model."""

        ...

    @property
    def null_class_id(self) -> int:
        """Return the reserved classifier-free unconditional identifier."""

        ...

    def predict_class_conditional_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the task-adapted model for a conditional diagnostic."""

        ...


class ClassConditionalGaussianDenoisingTrainingStrategy(TrainingStrategy):
    """Train a class-conditional denoiser against a Gaussian marginal target."""

    def __init__(
        self,
        model: ClassConditionalDenoiser,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        *,
        prediction_type: PredictionType = "epsilon",
        condition_dropout: float = 0.0,
    ) -> None:
        self.model = _validate_denoiser(model, role="class-conditional training model")
        process_value = cast(object, process)
        if not isinstance(process_value, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "class-conditional Gaussian training requires "
                "DiscreteGaussianDenoisingProcess"
            )
        if process_value.terminal_time <= process_value.clean_time:
            raise ValueError(
                "class-conditional Gaussian training requires a non-empty "
                "noisy time range"
            )
        objective_value = cast(object, objective)
        if not isinstance(objective_value, nn.Module):
            raise TypeError("class-conditional Gaussian training requires objective")
        self.process = process_value
        self.objective = objective_value
        self._prediction_type: PredictionType = _prediction_type(
            prediction_type,
            path="class-conditional Gaussian prediction_type",
        )
        self.condition_dropout = _condition_dropout(condition_dropout)

    @property
    def prediction_type(self) -> PredictionType:
        """Return the configured Gaussian model parameterization."""

        return self._prediction_type

    @property
    def num_classes(self) -> int:
        """Return the number of non-null classes accepted by the model."""

        return self.model.num_classes

    @property
    def null_class_id(self) -> int:
        """Return the model's classifier-free null class identifier."""

        return self.model.null_class_id

    def predict_class_conditional_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the conditional denoiser without applying dropout or guidance."""

        output = cast(
            object,
            predict_prevalidated_class_conditioned(
                self.model,
                state,
                model_time,
                class_labels,
            ),
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError("class-conditional Gaussian model must return a Tensor")
        return output

    def training_step(self, batch: Any) -> TrainStepOutput:
        """Apply per-sample class dropout and compute one training loss."""

        return self._step(batch, apply_condition_dropout=True)

    def evaluation_step(self, batch: Any) -> TrainStepOutput:
        """Compute one evaluation loss without classifier-free dropout."""

        return self._step(batch, apply_condition_dropout=False)

    def _step(
        self,
        batch: Any,
        *,
        apply_condition_dropout: bool,
    ) -> TrainStepOutput:
        clean, class_labels = _class_conditional_batch(
            batch,
            num_classes=self.num_classes,
        )
        dropout_mask = torch.zeros_like(class_labels, dtype=torch.bool)
        if apply_condition_dropout and self.condition_dropout > 0.0:
            if self.condition_dropout == 1.0:
                dropout_mask = torch.ones_like(class_labels, dtype=torch.bool)
            else:
                dropout_mask = (
                    torch.rand(class_labels.shape, device=class_labels.device)
                    < self.condition_dropout
                )
        model_labels = torch.where(
            dropout_mask,
            torch.full_like(class_labels, self.null_class_id),
            class_labels,
        )
        state_times = torch.randint(
            self.process.clean_time + 1,
            self.process.terminal_time + 1,
            (clean.shape[0],),
            device=clean.device,
        )
        noisy, noise = self.process.sample_marginal(clean, state_times)
        dynamics = GaussianModelDynamics(
            self.process,
            lambda state, model_time: self.predict_class_conditional_gaussian_model(
                state,
                model_time,
                model_labels,
            ),
            prediction_type=self.prediction_type,
            clip_denoised=False,
        )
        prediction = dynamics.predict(noisy, state_times)
        target = gaussian_training_target(
            self.process,
            clean=clean,
            noise=noise,
            state_times=state_times,
            prediction_type=self.prediction_type,
        )
        loss, per_sample = compute_objective(
            self.objective,
            prediction.model_output,
            target,
        )
        diagnostics: dict[str, Any] = {
            "timesteps": state_times.detach(),
            "predicted_noise": prediction.epsilon.detach(),
            "target_noise": noise.detach(),
            "predicted_clean": prediction.clean.detach(),
            "clean_samples": clean.detach(),
            "class_labels": class_labels.detach(),
            "model_class_labels": model_labels.detach(),
            "condition_dropout_mask": dropout_mask.detach(),
        }
        if per_sample is not None:
            diagnostics["per_sample_loss"] = per_sample.detach()
        return TrainStepOutput(loss=loss, diagnostics=diagnostics)


@REGISTRIES.training_builders.register("class_conditional_gaussian_denoising")
class ClassConditionalGaussianDenoisingTrainingBuilder(TrainingBuilder):
    """Assemble built-in class-conditional discrete Gaussian training."""

    def build(self) -> TrainingPlan:
        """Validate the complete family composition and return its plan."""

        params = dict(self.context.params)
        prediction_type = _prediction_type(
            params.pop("prediction_type", "epsilon"),
            path="training.params.prediction_type",
        )
        condition_dropout = _condition_dropout(
            params.pop("condition_dropout", 0.0),
            path="training.params.condition_dropout",
        )
        if params:
            unknown = ", ".join(sorted(params))
            raise ValueError(
                "unknown class_conditional_gaussian_denoising training "
                f"parameter(s): {unknown}"
            )
        model = _validate_denoiser(
            self.context.primary_model,
            role="class_conditional_gaussian_denoising primary model",
        )
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "class_conditional_gaussian_denoising training requires "
                "DiscreteGaussianDenoisingProcess"
            )
        objective = self.context.objective
        if objective is None:
            raise TypeError(
                "class_conditional_gaussian_denoising training requires objective"
            )
        strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
            model,
            process,
            objective,
            prediction_type=prediction_type,
            condition_dropout=condition_dropout,
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=process,
            objective=objective,
        )


def _validate_denoiser(
    value: object,
    *,
    role: str,
) -> ClassConditionalDenoiser:
    if not isinstance(value, ClassConditionalDenoiser):
        raise TypeError(f"{role} must satisfy ClassConditionalDenoiser")
    num_classes = cast(object, value.num_classes)
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError(f"{role} num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError(f"{role} num_classes must be positive")
    null_class_id = cast(object, value.null_class_id)
    if isinstance(null_class_id, bool) or not isinstance(null_class_id, int):
        raise TypeError(f"{role} null_class_id must be an integer")
    if null_class_id != num_classes:
        raise ValueError(f"{role} null_class_id must equal num_classes")
    return value


def _class_conditional_batch(
    batch: object,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(
            "class_conditional_gaussian_denoising expects "
            "(Tensor, {'class_label': labels})"
        )
    clean_value, conditions_value = batch
    if not isinstance(clean_value, torch.Tensor):
        raise TypeError(
            "class_conditional_gaussian_denoising clean samples must be a Tensor"
        )
    if clean_value.ndim == 0:
        raise ValueError(
            "class_conditional_gaussian_denoising samples must have a batch dimension"
        )
    if not isinstance(conditions_value, Mapping):
        raise TypeError(
            "class_conditional_gaussian_denoising conditions must be a mapping"
        )
    if set(conditions_value) != {"class_label"}:
        raise ValueError(
            "class_conditional_gaussian_denoising conditions must contain only "
            "'class_label'"
        )
    labels_value = cast(object, conditions_value["class_label"])
    if not isinstance(labels_value, torch.Tensor):
        raise TypeError(
            "class_conditional_gaussian_denoising class_label must be a Tensor"
        )
    if labels_value.ndim != 1:
        raise ValueError(
            "class_conditional_gaussian_denoising class_label must be a 1D Tensor"
        )
    if (
        labels_value.dtype == torch.bool
        or torch.is_floating_point(labels_value)
        or torch.is_complex(labels_value)
    ):
        raise TypeError(
            "class_conditional_gaussian_denoising class_label must contain integers"
        )
    if labels_value.shape[0] != clean_value.shape[0]:
        raise ValueError(
            "class_conditional_gaussian_denoising class_label must match the batch"
        )
    if labels_value.device != clean_value.device:
        raise ValueError(
            "class_conditional_gaussian_denoising class_label must share the "
            "sample device"
        )
    class_labels = labels_value.to(dtype=torch.long)
    invalid_labels = (class_labels < 0) | (class_labels >= num_classes)
    if bool(torch.any(invalid_labels)):
        raise ValueError(
            "class_conditional_gaussian_denoising class_label values must lie in "
            f"[0, {num_classes})"
        )
    return clean_value, class_labels


def _prediction_type(value: object, *, path: str) -> PredictionType:
    if not isinstance(value, str) or value not in ("epsilon", "x0", "v", "score"):
        raise ValueError(f"{path} must be epsilon, x0, v, or score")
    return cast(PredictionType, value)


def _condition_dropout(
    value: object,
    *,
    path: str = "class-conditional Gaussian condition_dropout",
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be in [0, 1]")
    return result


__all__ = [
    "ClassConditionalGaussianDenoisingTrainingBuilder",
    "ClassConditionalGaussianDenoisingTrainingStrategy",
    "ClassConditionalGaussianDiagnosticSemantics",
]
