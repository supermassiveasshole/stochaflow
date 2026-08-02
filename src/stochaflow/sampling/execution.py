"""Artifact-free execution of one registered sampling workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import torch

from stochaflow.sampling.builder import (
    SamplingBuilder,
    SamplingBuilderContext,
    SamplingOutput,
)
from stochaflow.sampling.sampler import SamplingObservation
from stochaflow.sampling.writers import SamplingBatch
from stochaflow.utils.registry import REGISTRIES


def execute_sampling_builder(
    builder_name: str,
    context: SamplingBuilderContext,
) -> SamplingOutput:
    """Construct and execute one builder without writing runtime artifacts."""

    builder = cast(
        SamplingBuilder,
        REGISTRIES.sampling_builders.create(builder_name, context),
    )
    with torch.no_grad():
        output_value = cast(object, builder.run())
    return validate_sampling_output(
        output_value,
        expected_num_samples=context.num_samples,
    )


def validate_sampling_output(
    output: object,
    *,
    expected_num_samples: int | None = None,
) -> SamplingOutput:
    """Validate the modality-neutral result contract owned by core runtime."""

    if expected_num_samples is not None and (
        type(expected_num_samples) is not int or expected_num_samples <= 0
    ):
        raise ValueError("expected_num_samples must be a positive integer")

    if not isinstance(output, SamplingOutput):
        raise TypeError("SamplingBuilder.run() must return SamplingOutput")
    if not output.batches:
        raise ValueError("SamplingOutput.batches must not be empty")
    observed_num_samples = 0
    for batch_index, declared_batch in enumerate(output.batches):
        batch_value = cast(object, declared_batch)
        if not isinstance(batch_value, SamplingBatch):
            raise TypeError(
                f"SamplingOutput.batches[{batch_index}] must be SamplingBatch"
            )
        batch = batch_value
        observed_num_samples += batch.num_samples
        if batch.trajectory is not None:
            previous = -1
            for observation_index, declared_observation in enumerate(
                batch.trajectory
            ):
                observation_value = cast(object, declared_observation)
                if not isinstance(observation_value, SamplingObservation):
                    raise TypeError(
                        f"trajectory[{observation_index}] must be "
                        "SamplingObservation"
                    )
                observation = observation_value
                if observation.step_index <= previous:
                    raise ValueError(
                        "trajectory observation step indices must increase"
                    )
                previous = observation.step_index
    if (
        expected_num_samples is not None
        and observed_num_samples != expected_num_samples
    ):
        raise ValueError(
            "SamplingOutput sample count does not match the request: "
            f"expected {expected_num_samples}, observed {observed_num_samples}"
        )
    metadata = cast(object, output.metadata)
    if not isinstance(metadata, Mapping):
        raise TypeError("SamplingOutput.metadata must be a mapping")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("SamplingOutput.metadata keys must be strings")
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise TypeError("SamplingOutput.metadata must be JSON serializable") from exc
    return output


__all__ = ["execute_sampling_builder", "validate_sampling_output"]
