"""Standard class-conditional Gaussian training strategy and builder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from stochaflow.families.gaussian import PredictionType
from stochaflow.models.conditioning import (
    ClassConditionalDenoiser,
    predict_prevalidated_class_conditioned,
)
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
)
from stochaflow.training.builder import TrainingBuilder, TrainingPlan
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.sampling_recipe import SamplingRecipe

from .strategy import GaussianTrainingStrategyBase
from .variance import GaussianVarianceConfig, parse_gaussian_variance


class ClassConditionalGaussianDenoisingTrainingStrategy(
    GaussianTrainingStrategyBase
):
    """Train a class-conditional denoiser against a Gaussian marginal."""

    def __init__(
        self,
        model: ClassConditionalDenoiser,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        *,
        prediction_type: PredictionType,
        variance: GaussianVarianceConfig | None = None,
        condition_dropout: float = 0.0,
    ) -> None:
        denoiser = _validate_denoiser(
            model,
            role="class-conditional training model",
        )
        super().__init__(
            cast(nn.Module, denoiser),
            process,
            objective,
            prediction_type=prediction_type,
            variance=variance,
        )
        self.denoiser = denoiser
        self.condition_dropout = _condition_dropout(condition_dropout)

    @property
    def num_classes(self) -> int:
        """Return the number of non-null classes accepted by the model."""

        return self.denoiser.num_classes

    @property
    def null_class_id(self) -> int:
        """Return the model's classifier-free null class identifier."""

        return self.denoiser.null_class_id

    def _prepare_batch(
        self,
        batch: Any,
        *,
        apply_training_policy: bool,
    ) -> tuple[torch.Tensor, object, dict[str, Any]]:
        clean, class_labels = _class_conditional_batch(
            batch,
            num_classes=self.num_classes,
        )
        dropout_mask = torch.zeros_like(class_labels, dtype=torch.bool)
        if apply_training_policy and self.condition_dropout > 0.0:
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
        return clean, model_labels, {
            "class_labels": class_labels.detach(),
            "model_class_labels": model_labels.detach(),
            "condition_dropout_mask": dropout_mask.detach(),
        }

    def _predict_training_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        model_context: object,
    ) -> torch.Tensor:
        if not isinstance(model_context, torch.Tensor):
            raise TypeError("class-conditional model context must be a Tensor")
        return self.predict_class_conditional_gaussian_model(
            state,
            model_time,
            model_context,
        )

    def predict_class_conditional_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the conditional denoiser without dropout or guidance."""

        output = cast(
            object,
            predict_prevalidated_class_conditioned(
                self.denoiser,
                state,
                model_time,
                class_labels,
            ),
        )
        if not isinstance(output, torch.Tensor):
            raise TypeError("class-conditional Gaussian model must return a Tensor")
        return output

    def extract_reference_images(self, batch: Any) -> torch.Tensor:
        """Extract clean class-conditional images without applying dropout."""

        return _class_conditional_batch(batch, num_classes=self.num_classes)[0]


@REGISTRIES.training_builders.register("class_conditional_gaussian_denoising")
class ClassConditionalGaussianDenoisingTrainingBuilder(TrainingBuilder):
    """Assemble standard class-conditional Gaussian training."""

    def build(self) -> TrainingPlan:
        """Validate the complete composition and return its plan."""

        params = dict(self.context.params)
        prediction_type = _prediction_type(
            params.pop("prediction_type", "epsilon"),
            path="training.params.prediction_type",
        )
        condition_dropout = _condition_dropout(
            params.pop("condition_dropout", 0.0),
            path="training.params.condition_dropout",
        )
        variance = parse_gaussian_variance(
            params.pop("variance", None),
            path="training.params.variance",
        )
        _reject_unknown(
            params,
            builder="class_conditional_gaussian_denoising",
        )
        model, process, objective = self._components()
        strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
            model,
            process,
            objective,
            prediction_type=prediction_type,
            variance=variance,
            condition_dropout=condition_dropout,
        )
        return self._plan(
            strategy,
            model=cast(nn.Module, model),
            process=process,
            objective=objective,
            prediction_type=prediction_type,
            variance=variance,
        )

    def _components(
        self,
    ) -> tuple[
        ClassConditionalDenoiser,
        DiscreteGaussianDenoisingProcess,
        nn.Module,
    ]:
        model = _validate_denoiser(
            self.context.primary_model,
            role="class-conditional Gaussian primary model",
        )
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "class-conditional Gaussian training requires "
                "DiscreteGaussianDenoisingProcess"
            )
        objective = self.context.objective
        if objective is None:
            raise TypeError("class-conditional Gaussian training requires objective")
        return model, process, objective

    def _plan(
        self,
        strategy: GaussianTrainingStrategyBase,
        *,
        model: nn.Module,
        process: DiscreteGaussianDenoisingProcess,
        objective: nn.Module,
        prediction_type: PredictionType,
        variance: GaussianVarianceConfig,
    ) -> TrainingPlan:
        return TrainingPlan(
            strategy=strategy,
            primary_model=model,
            process=process,
            objective=objective,
            inference_recipe=SamplingRecipe(
                name="class_conditional_denoising",
                contract={
                    "prediction_type": prediction_type,
                    "variance": {"mode": variance.mode},
                },
            ),
        )


def _validate_denoiser(
    value: object,
    *,
    role: str,
) -> ClassConditionalDenoiser:
    if not isinstance(value, nn.Module) or not isinstance(
        value, ClassConditionalDenoiser
    ):
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


def _reject_unknown(params: dict[str, Any], *, builder: str) -> None:
    if params:
        unknown = ", ".join(sorted(params))
        raise ValueError(f"unknown {builder} training parameter(s): {unknown}")


__all__ = [
    "ClassConditionalGaussianDenoisingTrainingBuilder",
    "ClassConditionalGaussianDenoisingTrainingStrategy",
]
