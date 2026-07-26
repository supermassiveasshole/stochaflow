"""Narrow model capabilities for conditioned generation."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ClassConditionalDenoiser(Protocol):
    """Predict a denoising target from state, time, and explicit class labels."""

    @property
    def num_classes(self) -> int:
        """Return the number of real classes accepted by the model."""

        ...

    @property
    def null_class_id(self) -> int:
        """Return the reserved class identifier for unconditional prediction."""

        ...

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return one class-conditioned denoising prediction."""

        ...


@runtime_checkable
class PrevalidatedClassConditionalDenoiser(Protocol):
    """Optional fast path for values certified by a composition boundary."""

    def predict_prevalidated_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict without repeating data-dependent host-side validation."""

        ...


def predict_prevalidated_class_conditioned(
    model: ClassConditionalDenoiser,
    state: torch.Tensor,
    model_time: torch.Tensor,
    class_labels: torch.Tensor,
) -> torch.Tensor:
    """Use an optional prevalidated fast path while preserving extensions."""

    if isinstance(model, PrevalidatedClassConditionalDenoiser):
        return model.predict_prevalidated_class_conditioned(
            state,
            model_time,
            class_labels,
        )
    return model.predict_class_conditioned(
        state,
        model_time,
        class_labels,
    )


__all__ = [
    "ClassConditionalDenoiser",
    "PrevalidatedClassConditionalDenoiser",
    "predict_prevalidated_class_conditioned",
]
