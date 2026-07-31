"""Narrow diagnostic contracts exposed by Gaussian training strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from stochaflow.families.gaussian import PredictionType, VarianceMode


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
    "GaussianDiagnosticSemantics",
    "VarianceMode",
]
