"""Tests for artifact-free sampling execution and pinned model authority."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.inference import PinnedInferenceModelProvider
from stochaflow.sampling import (
    SamplingBatch,
    SamplingBuilder,
    SamplingBuilderContext,
    SamplingOutput,
    execute_sampling_builder,
)
from stochaflow.sampling.execution import validate_sampling_output
from stochaflow.sampling.runtime import (
    validate_sampling_output as runtime_validate_sampling_output,
)
from stochaflow.utils.registry import REGISTRIES


class PinnedLinearModel(nn.Module):
    """Small stateful model used to verify pinned-object identity."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class GradientGuardSamplingBuilder(SamplingBuilder):
    """Record whether the reusable execution seam disables gradients."""

    constructor_calls = 0

    def __init__(self, context: SamplingBuilderContext) -> None:
        super().__init__(context)
        type(self).constructor_calls += 1

    def run(self) -> SamplingOutput:
        if torch.is_grad_enabled():
            raise AssertionError("sampling execution must disable gradients")
        model = self.context.model_provider.get("raw")
        samples = model(torch.ones(1)).detach()
        return SamplingOutput(
            (SamplingBatch(samples, num_samples=1),),
            {"weights": "raw"},
        )


class InvalidOutputSamplingBuilder(SamplingBuilder):
    """Return a malformed value so execution validation can reject it."""

    def run(self) -> SamplingOutput:
        return cast(Any, object())


REGISTRIES.sampling_builders.add(
    "test_execution_gradient_guard",
    GradientGuardSamplingBuilder,
)
REGISTRIES.sampling_builders.add(
    "test_execution_invalid_output",
    InvalidOutputSamplingBuilder,
)


def _pinned_provider(
    *,
    weights: str = "raw",
) -> tuple[PinnedInferenceModelProvider, PinnedLinearModel]:
    model = PinnedLinearModel()
    model.eval()
    provider = PinnedInferenceModelProvider(
        model=model,
        weights=cast(Any, weights),
        device=torch.device("cpu"),
    )
    return provider, model


def _context(provider: PinnedInferenceModelProvider) -> SamplingBuilderContext:
    return SamplingBuilderContext(
        params={},
        process=None,
        model_provider=provider,
        device=torch.device("cpu"),
        seed=7,
        shape=None,
        num_samples=1,
        batch_size=1,
    )


def test_execute_sampling_builder_constructs_runs_and_validates() -> None:
    provider, _ = _pinned_provider()
    before = GradientGuardSamplingBuilder.constructor_calls

    output = execute_sampling_builder(
        "test_execution_gradient_guard",
        _context(provider),
    )

    assert GradientGuardSamplingBuilder.constructor_calls == before + 1
    assert torch.equal(output.batches[0].samples, torch.tensor([2.0]))
    assert output.metadata == {"weights": "raw"}


def test_execute_sampling_builder_rejects_invalid_output() -> None:
    provider, _ = _pinned_provider()

    with pytest.raises(TypeError, match="must return SamplingOutput"):
        execute_sampling_builder(
            "test_execution_invalid_output",
            _context(provider),
        )


def test_runtime_reexports_shared_sampling_output_validator() -> None:
    assert runtime_validate_sampling_output is validate_sampling_output


def test_sampling_output_requires_exact_declared_sample_count() -> None:
    output = SamplingOutput(
        (SamplingBatch(torch.zeros(1), num_samples=1),),
        {},
    )

    with pytest.raises(ValueError, match="expected 2, observed 1"):
        validate_sampling_output(output, expected_num_samples=2)
    with pytest.raises(ValueError, match="positive integer"):
        SamplingBatch(torch.zeros(1), num_samples=0)


def test_pinned_provider_returns_same_model_for_matching_authority() -> None:
    provider, model = _pinned_provider(weights="ema")

    automatic, automatic_label = provider.resolve("auto")
    explicit, explicit_label = provider.resolve("ema")

    assert automatic is model
    assert explicit is model
    assert provider.get("auto") is model
    assert automatic_label == explicit_label == "ema"


def test_pinned_provider_rejects_opposite_explicit_weight_request() -> None:
    provider, _ = _pinned_provider(weights="ema")

    with pytest.raises(ValueError, match="cannot satisfy explicit raw"):
        provider.resolve("raw")


def test_pinned_provider_requires_resolved_eval_model() -> None:
    model = PinnedLinearModel()

    with pytest.raises(ValueError, match="already be in eval mode"):
        PinnedInferenceModelProvider(
            model=model,
            weights="raw",
            device=torch.device("cpu"),
        )

    model.eval()
    with pytest.raises(ValueError, match="must be raw or ema"):
        PinnedInferenceModelProvider(
            model=model,
            weights=cast(Any, "auto"),
            device=torch.device("cpu"),
        )
