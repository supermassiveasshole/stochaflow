"""Exponential moving average utilities for model parameters."""

from __future__ import annotations

from collections import OrderedDict
from typing import TypedDict

import torch
import torch.nn as nn


class EMAStateDict(TypedDict):
    """Serialized EMA state schema."""

    decay: float
    update_after_step: int
    update_every: int
    num_updates: int
    shadow_params: OrderedDict[str, torch.Tensor]
    shadow_buffers: OrderedDict[str, torch.Tensor]


class ExponentialMovingAverage:
    """Track an exponential moving average of a module's trainable state.

    The EMA object owns shadow copies of:
    - parameters
    - floating-point buffers

    It does not mutate the source module unless ``copy_to`` is called.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        if update_after_step < 0:
            raise ValueError("update_after_step must be non-negative")
        if update_every <= 0:
            raise ValueError("update_every must be positive")

        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.num_updates = 0

        self.shadow_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.shadow_buffers: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._stored_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._register_from_module(module)

    def _register_from_module(self, module: nn.Module) -> None:
        """Clone the initial parameter and buffer state from a module."""

        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self.shadow_params[name] = parameter.detach().clone()

        for name, buffer in module.named_buffers():
            if not torch.is_floating_point(buffer):
                continue
            self.shadow_buffers[name] = buffer.detach().clone()

    def _named_trainable_parameters(
        self, module: nn.Module
    ) -> list[tuple[str, nn.Parameter]]:
        """Collect trainable parameters that should participate in EMA updates."""

        return [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]

    def _named_floating_buffers(self, module: nn.Module) -> list[tuple[str, torch.Tensor]]:
        """Collect floating-point buffers that should be mirrored into EMA state."""

        return [
            (name, buffer)
            for name, buffer in module.named_buffers()
            if torch.is_floating_point(buffer)
        ]

    def _current_decay(self) -> float:
        """Return the decay factor for the next EMA update."""

        if self.num_updates < self.update_after_step:
            return 0.0
        return self.decay

    def update(self, module: nn.Module) -> None:
        """Update the EMA shadow state from the current module weights."""

        self.num_updates += 1
        if self.num_updates % self.update_every != 0:
            return

        decay = self._current_decay()
        one_minus_decay = 1.0 - decay

        for name, parameter in self._named_trainable_parameters(module):
            if name not in self.shadow_params:
                raise KeyError(f"EMA parameter '{name}' was not registered at initialization")
            shadow = self.shadow_params[name]
            shadow.mul_(decay).add_(parameter.detach(), alpha=one_minus_decay)

        for name, buffer in self._named_floating_buffers(module):
            if name not in self.shadow_buffers:
                raise KeyError(f"EMA buffer '{name}' was not registered at initialization")
            shadow = self.shadow_buffers[name]
            shadow.copy_(buffer.detach())

    def copy_to(self, module: nn.Module) -> None:
        """Overwrite a module's parameters and floating buffers with EMA state."""

        module_parameters = dict(module.named_parameters())
        for name, shadow in self.shadow_params.items():
            if name not in module_parameters:
                raise KeyError(f"module is missing EMA parameter '{name}'")
            module_parameters[name].data.copy_(shadow.data)

        module_buffers = dict(module.named_buffers())
        for name, shadow in self.shadow_buffers.items():
            if name not in module_buffers:
                raise KeyError(f"module is missing EMA buffer '{name}'")
            module_buffers[name].data.copy_(shadow.data)

    def store(self, module: nn.Module) -> None:
        """Snapshot the current module parameters for temporary EMA evaluation."""

        self._stored_params = OrderedDict(
            (
                name,
                parameter.detach().clone(),
            )
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )

    def restore(self, module: nn.Module) -> None:
        """Restore parameters previously captured by ``store``."""

        if not self._stored_params:
            raise RuntimeError("no stored parameters available; call store(module) first")

        module_parameters = dict(module.named_parameters())
        for name, parameter in self._stored_params.items():
            if name not in module_parameters:
                raise KeyError(f"module is missing stored parameter '{name}'")
            module_parameters[name].data.copy_(parameter.data)
        self._stored_params = OrderedDict()

    def state_dict(self) -> EMAStateDict:
        """Serialize the EMA state for checkpointing."""

        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "shadow_params": OrderedDict(
                (name, tensor.detach().clone()) for name, tensor in self.shadow_params.items()
            ),
            "shadow_buffers": OrderedDict(
                (name, tensor.detach().clone())
                for name, tensor in self.shadow_buffers.items()
            ),
        }

    def load_state_dict(self, state_dict: EMAStateDict) -> None:
        """Load an EMA state produced by ``state_dict``."""

        self.decay = float(state_dict["decay"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.update_every = int(state_dict["update_every"])
        self.num_updates = int(state_dict["num_updates"])

        shadow_params = state_dict["shadow_params"]
        if not isinstance(shadow_params, OrderedDict):
            raise TypeError("shadow_params must be an OrderedDict")
        shadow_buffers = state_dict["shadow_buffers"]
        if not isinstance(shadow_buffers, OrderedDict):
            raise TypeError("shadow_buffers must be an OrderedDict")

        self.shadow_params = OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in shadow_params.items()
        )
        self.shadow_buffers = OrderedDict(
            (name, tensor.detach().clone()) for name, tensor in shadow_buffers.items()
        )
