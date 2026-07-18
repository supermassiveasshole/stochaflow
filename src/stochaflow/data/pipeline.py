"""Modality-neutral data-pipeline extension contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sized
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog


@dataclass(frozen=True, slots=True)
class DataPipelineContext:
    """Configuration and deterministic seed supplied to a data pipeline."""

    params: dict[str, Any]
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.params, dict):
            raise TypeError("data pipeline params must be a mapping")
        object.__setattr__(self, "params", deepcopy(self.params))


@dataclass(frozen=True, slots=True)
class SplitData:
    """One named loader with optional size and dataset metadata."""

    name: str
    dataloader: Iterable[Any]
    dataset: Any | None = None
    num_samples: int | None = None
    num_batches: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("split name must be non-empty")
        if not isinstance(self.dataloader, Iterable):
            raise TypeError(f"split '{self.name}' dataloader must be iterable")
        if self.num_samples is not None and self.num_samples < 0:
            raise ValueError(f"split '{self.name}' num_samples must be non-negative")
        if self.num_batches is not None and self.num_batches <= 0:
            raise ValueError(f"split '{self.name}' num_batches must be positive")

    def resolved_num_samples(self) -> int | None:
        """Return explicit sample count or a safe dataset length."""

        if self.num_samples is not None:
            return self.num_samples
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        return None

    def resolved_num_batches(self) -> int | None:
        """Return explicit batch count or a safe loader length."""

        if self.num_batches is not None:
            return self.num_batches
        if isinstance(self.dataloader, Sized):
            return len(self.dataloader)
        return None


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Loaders required for one independent training run."""

    train: SplitData
    valid: SplitData | None = None
    test: SplitData | None = None
    fold_index: int | None = None
    num_folds: int | None = None


class DataPipeline(ABC):
    """Extension point owning all data construction and batching semantics."""

    def __init__(self, context: DataPipelineContext) -> None:
        self.context = context

    @abstractmethod
    def build(self) -> list[DataBundle]:
        """Build one or more non-empty training bundles."""


REGISTRIES.data_pipelines.require_base(DataPipeline)


def build_data_pipeline(
    config: ComponentConfig,
    *,
    seed: int,
    registries: RegistryCatalog = REGISTRIES,
) -> list[DataBundle]:
    """Construct and validate a registered data pipeline."""

    pipeline = registries.data_pipelines.create(
        config.name,
        DataPipelineContext(params=config.params, seed=seed),
    )
    if not isinstance(pipeline, DataPipeline):
        raise TypeError(
            f"registered data pipeline '{config.name}' did not produce DataPipeline"
        )
    bundles = pipeline.build()
    if not isinstance(bundles, list) or not bundles:
        raise ValueError(
            f"data pipeline '{config.name}' must return a non-empty list[DataBundle]"
        )
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, DataBundle):
            raise TypeError(
                f"data pipeline '{config.name}' bundle {index} is not DataBundle"
            )
        num_batches = bundle.train.resolved_num_batches()
        if num_batches is None:
            raise ValueError(
                f"data pipeline '{config.name}' training split must expose a "
                "finite loader length or num_batches"
            )
        if num_batches <= 0:
            raise ValueError(
                f"data pipeline '{config.name}' training split must contain a batch"
            )
    return bundles


__all__ = [
    "DataBundle",
    "DataPipeline",
    "DataPipelineContext",
    "SplitData",
    "build_data_pipeline",
]
