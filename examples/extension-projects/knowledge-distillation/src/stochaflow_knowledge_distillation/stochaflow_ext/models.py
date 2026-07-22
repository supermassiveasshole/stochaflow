"""Student and teacher classifiers for the distillation reference project."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import cast

from stochaflow.extensions import REGISTRIES

_PREFIX = "stochaflow-knowledge-distillation"


def _positive_int(value: int, name: str) -> int:
    runtime_value = cast(object, value)
    if (
        isinstance(runtime_value, bool)
        or not isinstance(runtime_value, int)
        or runtime_value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return runtime_value


class _FlattenClassifier(nn.Module):
    """Small MLP base that accepts either vectors or image-shaped inputs."""

    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: tuple[int, ...],
        num_classes: int,
    ) -> None:
        super().__init__()
        input_features = _positive_int(input_features, "input_features")
        num_classes = _positive_int(num_classes, "num_classes")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not hidden_features:
            raise ValueError("hidden_features must not be empty")
        for index, width in enumerate(hidden_features):
            _positive_int(width, f"hidden_features[{index}]")

        layers: list[nn.Module] = []
        previous = input_features
        for width in hidden_features:
            layers.extend((nn.Linear(previous, width), nn.GELU()))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_features = input_features
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for one batch."""

        runtime_inputs = cast(object, inputs)
        if not isinstance(runtime_inputs, torch.Tensor):
            raise TypeError("classifier inputs must be a Tensor")
        if runtime_inputs.ndim < 2:
            raise ValueError("classifier inputs must include batch and feature axes")
        flattened = runtime_inputs.reshape(runtime_inputs.shape[0], -1)
        if flattened.shape[1] != self.input_features:
            raise ValueError(
                "flattened classifier input has "
                f"{flattened.shape[1]} features; expected {self.input_features}"
            )
        return self.network(flattened)


@REGISTRIES.models.register(f"{_PREFIX}.student")
class StudentClassifier(_FlattenClassifier):
    """Compact classifier optimized by the training loop."""

    def __init__(
        self,
        *,
        input_features: int = 8,
        hidden_features: int = 12,
        num_classes: int = 4,
    ) -> None:
        super().__init__(
            input_features=input_features,
            hidden_features=(_positive_int(hidden_features, "hidden_features"),),
            num_classes=num_classes,
        )


@REGISTRIES.models.register(f"{_PREFIX}.teacher")
class TeacherClassifier(_FlattenClassifier):
    """Larger classifier loaded from a plain PyTorch state dictionary."""

    def __init__(
        self,
        *,
        input_features: int = 8,
        hidden_features: int = 24,
        num_classes: int = 4,
    ) -> None:
        width = _positive_int(hidden_features, "hidden_features")
        super().__init__(
            input_features=input_features,
            hidden_features=(width, width),
            num_classes=num_classes,
        )


__all__ = ["StudentClassifier", "TeacherClassifier"]
