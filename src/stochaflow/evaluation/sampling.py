"""Narrow checkpoint sampling capability injected into task evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import nn

from stochaflow.evaluation.config import (
    CheckpointWeightVariant,
    _freeze_evaluation_mapping,
    _thaw_evaluation_value,
)
from stochaflow.evaluation.subject import CheckpointSubjectInputs
from stochaflow.inference.checkpoint import (
    build_checkpointed_process,
    build_inference_asset_provider,
    load_checkpoint_recipe,
)
from stochaflow.inference.model import PinnedInferenceModelProvider
from stochaflow.sampling.builder import SamplingBuilderContext, SamplingOutput
from stochaflow.sampling.execution import execute_sampling_builder
from stochaflow.utils.checkpoint import CheckpointState
from stochaflow.utils.config import StochaflowConfig
from stochaflow.utils.sampling_recipe import (
    resolve_sampling_recipe_params,
    sampling_recipe_to_dict,
)


def _positive_integer(value: object, *, path: str) -> int:
    if type(value) is not int or cast(int, value) <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class EvaluationSamplingRequest:
    """Writer-free sampling inputs owned by one EvaluationBuilder profile."""

    options: Mapping[str, Any]
    sampler: Mapping[str, Any] | None
    shape: tuple[int, ...]
    num_samples: int
    batch_size: int
    seed: int
    expected_recipe_name: str | None = None
    expected_recipe_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        options = _freeze_evaluation_mapping(
            self.options,
            path="evaluation sampling options",
        )
        sampler = (
            None
            if self.sampler is None
            else _freeze_evaluation_mapping(
                self.sampler,
                path="evaluation sampling sampler",
            )
        )
        if type(cast(object, self.shape)) is not tuple or not self.shape:
            raise ValueError("evaluation sampling shape must be a non-empty tuple")
        shape = tuple(
            _positive_integer(value, path=f"evaluation sampling shape[{index}]")
            for index, value in enumerate(self.shape)
        )
        num_samples = _positive_integer(
            cast(object, self.num_samples),
            path="evaluation sampling num_samples",
        )
        batch_size = _positive_integer(
            cast(object, self.batch_size),
            path="evaluation sampling batch_size",
        )
        if batch_size > num_samples:
            raise ValueError(
                "evaluation sampling batch_size must not exceed num_samples"
            )
        if type(cast(object, self.seed)) is not int:
            raise TypeError("evaluation sampling seed must be an exact integer")
        recipe_name = cast(object, self.expected_recipe_name)
        recipe_contract = cast(object, self.expected_recipe_contract)
        if (recipe_name is None) != (recipe_contract is None):
            raise ValueError(
                "evaluation sampling expected recipe name and contract must be "
                "declared together"
            )
        if recipe_name is not None:
            if (
                type(recipe_name) is not str
                or not cast(str, recipe_name)
                or cast(str, recipe_name).strip() != recipe_name
            ):
                raise ValueError(
                    "evaluation sampling expected recipe name must be non-empty"
                )
            recipe_contract = _freeze_evaluation_mapping(
                recipe_contract,
                path="evaluation sampling expected recipe contract",
            )
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "expected_recipe_contract", recipe_contract)


@runtime_checkable
class EvaluationSamplingCapability(Protocol):
    """Execute an already checkpoint-bound sampling method without publishing."""

    def execute(self, request: EvaluationSamplingRequest) -> SamplingOutput:
        """Return validated in-memory batches without writers or a manifest."""

        ...


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationSamplingCapability:
    """Lazily compose checkpoint process/assets with one pinned primary model."""

    inputs: CheckpointSubjectInputs = field(repr=False)
    config: StochaflowConfig = field(repr=False)
    model: nn.Module = field(repr=False, compare=False)
    resolved_weights: CheckpointWeightVariant
    device: torch.device

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.inputs), CheckpointSubjectInputs):
            raise TypeError("checkpoint evaluation sampling inputs are invalid")
        if not isinstance(cast(object, self.config), StochaflowConfig):
            raise TypeError("checkpoint evaluation sampling config is invalid")
        if not isinstance(cast(object, self.model), nn.Module):
            raise TypeError("checkpoint evaluation sampling model must be nn.Module")
        if self.resolved_weights not in {"raw", "ema"}:
            raise ValueError(
                "checkpoint evaluation sampling weights must be raw or ema"
            )
        object.__setattr__(self, "device", torch.device(self.device))

    def execute(self, request: EvaluationSamplingRequest) -> SamplingOutput:
        """Run the checkpoint recipe through the shared SamplingBuilder seam."""

        if not isinstance(cast(object, request), EvaluationSamplingRequest):
            raise TypeError("evaluation sampling request is invalid")
        recipe = load_checkpoint_recipe(
            cast(CheckpointState, self.inputs._checkpoint_view)
        )
        if request.expected_recipe_name is not None:
            recipe_document = sampling_recipe_to_dict(recipe)
            expected_contract = _thaw_evaluation_value(
                request.expected_recipe_contract
            )
            if recipe.name != request.expected_recipe_name:
                raise ValueError(
                    "checkpoint sampling recipe name does not match the "
                    "evaluation profile"
                )
            if recipe_document["contract"] != expected_contract:
                raise ValueError(
                    "checkpoint sampling recipe contract does not match the "
                    "evaluation profile"
                )
        options_value = _thaw_evaluation_value(request.options)
        if type(options_value) is not dict:
            raise TypeError("evaluation sampling options must thaw to a mapping")
        sampler_value = (
            None
            if request.sampler is None
            else _thaw_evaluation_value(request.sampler)
        )
        if sampler_value is not None and type(sampler_value) is not dict:
            raise TypeError("evaluation sampling sampler must thaw to a mapping")
        params = resolve_sampling_recipe_params(
            recipe,
            options=cast(dict[str, Any], options_value),
            sampler=cast(dict[str, Any] | None, sampler_value),
        )
        process = build_checkpointed_process(
            self.config,
            self.inputs._checkpoint_view,
            device=self.device,
        )
        assets = build_inference_asset_provider(
            self.inputs._checkpoint_view,
            device=self.device,
        )
        provider = PinnedInferenceModelProvider(
            model=self.model,
            weights=self.resolved_weights,
            device=self.device,
        )
        context = SamplingBuilderContext(
            params=deepcopy(params),
            process=process,
            model_provider=provider,
            device=self.device,
            seed=request.seed,
            shape=request.shape,
            num_samples=request.num_samples,
            batch_size=request.batch_size,
            inference_assets=assets,
        )
        return execute_sampling_builder(recipe.name, context)


__all__ = [
    "CheckpointEvaluationSamplingCapability",
    "EvaluationSamplingCapability",
    "EvaluationSamplingRequest",
]
