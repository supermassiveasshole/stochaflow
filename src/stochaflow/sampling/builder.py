"""Task-neutral sampling-builder contracts and inference model provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
from torch import nn

from stochaflow.processes.base import Process
from stochaflow.utils.device import move_module_to_device
from stochaflow.utils.registry import REGISTRIES

from .assets import InferenceAssetProvider
from .writers import SamplingBatch

WeightSelection = Literal["auto", "raw", "ema"]


class InferenceModelProvider:
    """Construct raw or EMA inference models from portable checkpoint state."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        raw_state_dict: Mapping[str, torch.Tensor],
        ema_state_dict: Mapping[str, torch.Tensor] | None,
        device: torch.device,
        prefer_ema: bool,
    ) -> None:
        self._model_factory = model_factory
        self._raw_state_dict = raw_state_dict
        self._ema_state_dict = ema_state_dict
        self.device = device
        self.prefer_ema = prefer_ema

    def resolve(self, weights: WeightSelection) -> tuple[nn.Module, str]:
        """Build the selected model and return its resolved weight label."""

        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("sampling weights must be auto, raw, or ema")
        resolved = (
            "ema"
            if weights == "ema" or (weights == "auto" and self.prefer_ema)
            else "raw"
        )
        state = self._ema_state_dict if resolved == "ema" else self._raw_state_dict
        if state is None:
            raise ValueError("EMA weights were requested but are unavailable")
        model_value = cast(object, self._model_factory())
        if not isinstance(model_value, nn.Module):
            raise TypeError("inference model factory must return nn.Module")
        model_value.load_state_dict(state)
        move_module_to_device(
            model_value,
            self.device,
            role="inference model",
        )
        model_value.eval()
        return model_value, resolved

    def get(self, weights: WeightSelection = "auto") -> nn.Module:
        """Build and return a selected inference model."""

        return self.resolve(weights)[0]


@dataclass(frozen=True, slots=True)
class SamplingBuilderContext:
    """Runtime services and user parameters supplied to a sampling builder."""

    params: dict[str, Any]
    process: Process | None
    model_provider: InferenceModelProvider
    device: torch.device
    seed: int
    shape: tuple[int, ...] | None
    num_samples: int
    batch_size: int
    inference_assets: InferenceAssetProvider = field(
        default_factory=InferenceAssetProvider.empty
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", deepcopy(self.params))


@dataclass(frozen=True, slots=True)
class SamplingOutput:
    """Generated batches and recipe-resolved metadata."""

    batches: tuple[SamplingBatch, ...]
    metadata: Mapping[str, Any]


class SamplingBuilder(ABC):
    """Assemble and execute one complete task-specific sampling workflow."""

    def __init__(self, context: SamplingBuilderContext) -> None:
        self.context = context

    @abstractmethod
    def run(self) -> SamplingOutput:
        """Execute the workflow once and return writer-ready batches."""


REGISTRIES.sampling_builders.require_base(SamplingBuilder)


__all__ = [
    "InferenceAssetProvider",
    "InferenceModelProvider",
    "SamplingBuilder",
    "SamplingBuilderContext",
    "SamplingOutput",
]
