"""Shared read-only inference capabilities.

Checkpoint helpers live in :mod:`stochaflow.inference.checkpoint`; keeping the
package root lightweight avoids importing runtime factories during contract
imports.
"""

from .model import (
    InferenceModelProvider,
    PinnedInferenceModelProvider,
    PinnedWeightSelection,
    WeightSelection,
)

__all__ = [
    "InferenceModelProvider",
    "PinnedInferenceModelProvider",
    "PinnedWeightSelection",
    "WeightSelection",
]
