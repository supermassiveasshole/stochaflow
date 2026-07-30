"""Classifier and calibration models for the distillation reference project."""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import torch
from torch import nn

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


class FlattenClassifier(nn.Module):
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


@runtime_checkable
class LogitCalibrationCapability(Protocol):
    """Narrow extension-owned capability for calibrating class logits."""

    @property
    def num_classes(self) -> int:
        """Return the class width required by the calibrator."""

        ...

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated logits with the same batch and class axes."""

        ...


@REGISTRIES.models.register(f"{_PREFIX}.student")
class StudentClassifier(FlattenClassifier):
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
class TeacherClassifier(FlattenClassifier):
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


@REGISTRIES.models.register(f"{_PREFIX}.calibrator")
class LogitCalibrator(nn.Module):
    """Apply checkpointed per-class affine calibration to classifier logits."""

    def __init__(self, *, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = _positive_int(num_classes, "num_classes")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.scale = nn.Parameter(torch.ones(self.num_classes))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the fixed affine transform to a batch of class logits."""

        runtime_logits = cast(object, logits)
        if not isinstance(runtime_logits, torch.Tensor):
            raise TypeError("calibrator logits must be a Tensor")
        if runtime_logits.ndim != 2:
            raise ValueError("calibrator logits must have shape [batch, classes]")
        if runtime_logits.shape[1] != self.num_classes:
            raise ValueError(
                "calibrator logits have "
                f"{runtime_logits.shape[1]} classes; expected {self.num_classes}"
            )
        return runtime_logits * self.scale + self.bias

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Delegate module calls to the calibration capability."""

        return self.calibrate_logits(logits)


__all__ = [
    "LogitCalibrationCapability",
    "LogitCalibrator",
    "StudentClassifier",
    "TeacherClassifier",
]
