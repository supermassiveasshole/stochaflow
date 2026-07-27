"""Student-only direct prediction sampling for the reference project."""

from __future__ import annotations

from typing import Any

import torch

from stochaflow.extensions import (
    REGISTRIES,
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)

_PREFIX = "stochaflow-knowledge-distillation"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite")
    return result


@REGISTRIES.sampling_builders.register(f"{_PREFIX}.predictions")
class StudentPredictionBuilder(SamplingBuilder):
    """Evaluate only the checkpointed student on deterministic synthetic inputs."""

    def run(self) -> SamplingOutput:
        """Return batched student logits without constructing training assets."""

        if self.context.process is not None:
            raise ValueError("student prediction sampling does not use a Process")
        if self.context.shape is not None:
            raise ValueError(
                "student prediction sampling requires sampling.shape: null"
            )

        params: dict[str, Any] = dict(self.context.params)
        if "input_features" not in params:
            raise ValueError(
                "student prediction inference recipe requires input_features"
            )
        input_features = _positive_int(
            params.pop("input_features"),
            "input_features",
        )
        mean = _number(params.pop("mean", 0.0), "mean")
        std = _number(params.pop("std", 1.0), "std")
        if std < 0:
            raise ValueError("std must be non-negative")
        weights = params.pop("weights", "auto")
        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("weights must be auto, raw, or ema")
        if params:
            raise ValueError(f"unknown sampling params: {', '.join(sorted(params))}")

        model, resolved_weights = self.context.model_provider.resolve(weights)
        generator = torch.Generator().manual_seed(self.context.seed)
        values = torch.randn(
            self.context.num_samples,
            input_features,
            generator=generator,
        )
        values.mul_(std).add_(mean)
        batches: list[SamplingBatch] = []
        predicted_classes: list[int] = []
        for start in range(0, self.context.num_samples, self.context.batch_size):
            inputs = values[start : start + self.context.batch_size].to(
                self.context.device
            )
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("student model must return a Tensor")
            if logits.ndim != 2 or logits.shape[0] != inputs.shape[0]:
                raise ValueError("student logits must have shape [batch, classes]")
            logits = logits.detach().cpu()
            predicted_classes.extend(logits.argmax(dim=1).tolist())
            batches.append(SamplingBatch(samples=logits))
        return SamplingOutput(
            batches=tuple(batches),
            metadata={
                "workflow": "student-only-classification",
                "weights": resolved_weights,
                "input_features": input_features,
                "input_mean": mean,
                "input_std": std,
                "predicted_classes": predicted_classes,
            },
        )


__all__ = ["StudentPredictionBuilder"]
