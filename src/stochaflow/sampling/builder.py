"""Task-neutral sampling-builder contracts and inference model provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from stochaflow.inference import InferenceAssetProvider
from stochaflow.inference.model import InferenceModelProvider, WeightSelection
from stochaflow.processes.base import Process
from stochaflow.utils.registry import REGISTRIES

from .writers import SamplingBatch


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
    "WeightSelection",
]
