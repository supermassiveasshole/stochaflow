"""Core model-free probability-process contract."""

from abc import ABC

import torch.nn as nn

from stochaflow.utils.registry import REGISTRIES


class Process(nn.Module, ABC):
    """Semantic root for registered, checkpointable probability processes.

    Mathematical behavior belongs to algorithm-family contracts derived from
    this type. The root deliberately defines no universal probability API.
    """


REGISTRIES.processes.require_base(Process)


__all__ = ["Process"]
