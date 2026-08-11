"""Narrow diagnostic contracts exposed by Gaussian training strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import torch

from stochaflow.families.gaussian import (
    GaussianPrediction,
    PredictionType,
    VarianceMode,
)


@dataclass(frozen=True, slots=True)
class GaussianStepObservation:
    """Detached Gaussian-family facts from one completed strategy step."""

    state_times: torch.Tensor
    prediction: GaussianPrediction
    noise_target: torch.Tensor
    clean_samples: torch.Tensor
    per_sample_loss: torch.Tensor | None

    def __post_init__(self) -> None:
        """Validate detached, batch-aligned Gaussian observations."""

        state_times = cast(object, self.state_times)
        if not isinstance(state_times, torch.Tensor):
            raise TypeError("GaussianStepObservation.state_times must be a Tensor")
        if self.state_times.ndim != 1:
            raise ValueError("GaussianStepObservation.state_times must be 1D")
        if (
            self.state_times.dtype == torch.bool
            or torch.is_floating_point(self.state_times)
            or torch.is_complex(self.state_times)
        ):
            raise TypeError(
                "GaussianStepObservation.state_times must contain integers"
            )
        prediction = cast(object, self.prediction)
        if not isinstance(prediction, GaussianPrediction):
            raise TypeError(
                "GaussianStepObservation.prediction must be GaussianPrediction"
            )
        noise_target = cast(object, self.noise_target)
        clean_samples = cast(object, self.clean_samples)
        if not isinstance(noise_target, torch.Tensor) or not isinstance(
            clean_samples,
            torch.Tensor,
        ):
            raise TypeError(
                "GaussianStepObservation noise_target and clean_samples must be "
                "Tensors"
            )
        batch_size = self.state_times.shape[0]
        if self.clean_samples.ndim == 0:
            raise ValueError(
                "GaussianStepObservation.clean_samples must have a batch dimension"
            )
        if self.clean_samples.shape[0] != batch_size:
            raise ValueError(
                "GaussianStepObservation state_times and clean_samples must align"
            )
        if self.noise_target.shape != self.clean_samples.shape:
            raise ValueError(
                "GaussianStepObservation noise_target must match clean_samples"
            )
        if self.state_times.device != self.clean_samples.device:
            raise ValueError(
                "GaussianStepObservation state_times and clean_samples must share "
                "a device"
            )
        if self.noise_target.device != self.clean_samples.device:
            raise ValueError(
                "GaussianStepObservation noise_target and clean_samples must share "
                "a device"
            )
        if not torch.is_floating_point(self.clean_samples) or not torch.is_floating_point(
            self.noise_target
        ):
            raise TypeError(
                "GaussianStepObservation clean_samples and noise_target must use "
                "floating dtypes"
            )
        prediction_tensors = (
            self.prediction.clean,
            self.prediction.epsilon,
            self.prediction.model_output,
        )
        if any(value.shape != self.clean_samples.shape for value in prediction_tensors):
            raise ValueError(
                "GaussianStepObservation prediction must match clean_samples"
            )
        if any(
            value.device != self.clean_samples.device
            for value in prediction_tensors
        ):
            raise ValueError(
                "GaussianStepObservation prediction and clean_samples must share "
                "a device"
            )
        if any(not torch.is_floating_point(value) for value in prediction_tensors):
            raise TypeError(
                "GaussianStepObservation prediction tensors must use floating dtypes"
            )
        tensors = (
            self.state_times,
            self.noise_target,
            self.clean_samples,
            *prediction_tensors,
        )
        if any(value.requires_grad for value in tensors):
            raise ValueError("GaussianStepObservation tensors must be detached")
        for name in ("per_sample_loss",):
            value = cast(object, getattr(self, name))
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"GaussianStepObservation.{name} must be a Tensor")
            if value.shape != (batch_size,):
                raise ValueError(
                    f"GaussianStepObservation.{name} must align with the batch"
                )
            if value.device != self.clean_samples.device:
                raise ValueError(
                    f"GaussianStepObservation.{name} must share the batch device"
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"GaussianStepObservation.{name} must use a floating dtype"
                )
            if value.requires_grad:
                raise ValueError(
                    f"GaussianStepObservation.{name} must be detached"
                )


@dataclass(frozen=True, slots=True)
class ClassConditionalGaussianStepObservation(GaussianStepObservation):
    """Gaussian step facts plus explicit class-conditioning decisions."""

    class_labels: torch.Tensor
    model_class_labels: torch.Tensor
    condition_dropout_mask: torch.Tensor

    def __post_init__(self) -> None:
        """Validate label and dropout facts against the Gaussian batch."""

        GaussianStepObservation.__post_init__(self)
        batch_size = self.state_times.shape[0]
        for name in ("class_labels", "model_class_labels"):
            value = cast(object, getattr(self, name))
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"ClassConditionalGaussianStepObservation.{name} must be a "
                    "Tensor"
                )
            if value.shape != (batch_size,) or value.dtype != torch.long:
                raise ValueError(
                    f"ClassConditionalGaussianStepObservation.{name} must be a "
                    "batch-aligned 1D long Tensor"
                )
            if value.device != self.clean_samples.device:
                raise ValueError(
                    f"ClassConditionalGaussianStepObservation.{name} must share "
                    "the batch device"
                )
            if value.requires_grad:
                raise ValueError(
                    f"ClassConditionalGaussianStepObservation.{name} must be "
                    "detached"
                )
        mask = cast(object, self.condition_dropout_mask)
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                "ClassConditionalGaussianStepObservation.condition_dropout_mask "
                "must be a Tensor"
            )
        if mask.shape != (batch_size,) or mask.dtype != torch.bool:
            raise ValueError(
                "ClassConditionalGaussianStepObservation.condition_dropout_mask "
                "must be a batch-aligned 1D bool Tensor"
            )
        if mask.device != self.clean_samples.device:
            raise ValueError(
                "ClassConditionalGaussianStepObservation.condition_dropout_mask "
                "must share the batch device"
            )
        if mask.requires_grad:
            raise ValueError(
                "ClassConditionalGaussianStepObservation.condition_dropout_mask "
                "must be detached"
            )


@runtime_checkable
class GaussianDiagnosticSemantics(Protocol):
    """Unconditional Gaussian model invocation used by diagnostics."""

    @property
    def prediction_type(self) -> PredictionType:
        """Return the model prediction parameterization used for training."""

        ...

    @property
    def variance_mode(self) -> VarianceMode:
        """Return the raw Gaussian model-output variance layout."""

        ...

    def predict_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the task-adapted model for a Gaussian diagnostic."""

        ...


@runtime_checkable
class ClassConditionalGaussianDiagnosticSemantics(Protocol):
    """Class-conditional Gaussian invocation used by diagnostics."""

    @property
    def prediction_type(self) -> PredictionType:
        """Return the model prediction parameterization used for training."""

        ...

    @property
    def variance_mode(self) -> VarianceMode:
        """Return the raw Gaussian model-output variance layout."""

        ...

    @property
    def num_classes(self) -> int:
        """Return the number of non-null classes accepted by the model."""

        ...

    @property
    def null_class_id(self) -> int:
        """Return the classifier-free unconditional identifier."""

        ...

    def predict_class_conditional_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the task-adapted model for a conditional diagnostic."""

        ...


__all__ = [
    "ClassConditionalGaussianDiagnosticSemantics",
    "ClassConditionalGaussianStepObservation",
    "GaussianDiagnosticSemantics",
    "GaussianStepObservation",
    "VarianceMode",
]
