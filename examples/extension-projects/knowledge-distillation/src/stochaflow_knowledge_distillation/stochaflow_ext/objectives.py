"""Task and distillation objectives owned by the reference extension."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

from stochaflow.extensions import REGISTRIES

_PREFIX = "stochaflow-knowledge-distillation"


@REGISTRIES.objectives.register(f"{_PREFIX}.cross-entropy")
class ClassificationCrossEntropy(nn.Module):
    """Cross-entropy objective for integer class labels."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return scalar supervised classification loss."""

        if logits.ndim != 2:
            raise ValueError("classification logits must have shape [batch, classes]")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("classification targets must have shape [batch]")
        if targets.dtype != torch.long:
            raise TypeError("classification targets must use torch.long dtype")
        return F.cross_entropy(logits, targets)


@REGISTRIES.objectives.register(f"{_PREFIX}.temperature-kl")
class TemperatureKLDistillation(nn.Module):
    """Temperature-scaled KL objective with checkpointed temperature state."""

    temperature: torch.Tensor

    def __init__(self, *, temperature: float = 2.0) -> None:
        super().__init__()
        runtime_temperature = cast(object, temperature)
        if (
            isinstance(runtime_temperature, bool)
            or not isinstance(runtime_temperature, (int, float))
            or not math.isfinite(float(runtime_temperature))
            or runtime_temperature <= 0
        ):
            raise ValueError("temperature must be a positive finite number")
        self.register_buffer("temperature", torch.tensor(float(runtime_temperature)))

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return temperature-scaled batch-mean KL divergence."""

        if student_logits.shape != teacher_logits.shape:
            raise ValueError("student and teacher logits must share the same shape")
        if student_logits.ndim != 2:
            raise ValueError("distillation logits must have shape [batch, classes]")
        if (
            not student_logits.is_floating_point()
            or not teacher_logits.is_floating_point()
        ):
            raise TypeError("distillation logits must be floating-point Tensors")
        temperature = self.temperature.to(
            device=student_logits.device,
            dtype=student_logits.dtype,
        )
        teacher_probabilities = F.softmax(
            teacher_logits.detach() / temperature,
            dim=-1,
        )
        student_log_probabilities = F.log_softmax(
            student_logits / temperature,
            dim=-1,
        )
        return F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        ) * temperature.square()


__all__ = ["ClassificationCrossEntropy", "TemperatureKLDistillation"]
