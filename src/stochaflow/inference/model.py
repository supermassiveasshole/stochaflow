"""Portable primary-model reconstruction for read-only inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, cast

import torch
from torch import nn

from stochaflow.utils.checkpoint import validate_module_state_dict_compatibility
from stochaflow.utils.device import move_module_to_device

WeightSelection = Literal["auto", "raw", "ema"]
PinnedWeightSelection = Literal["raw", "ema"]


class InferenceModelProvider:
    """Construct raw or EMA inference models from portable checkpoint state."""

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        raw_state_dict: Mapping[str, torch.Tensor],
        ema_state_dict: Mapping[str, torch.Tensor] | None,
        device: torch.device,
    ) -> None:
        if not callable(model_factory):
            raise TypeError("inference model factory must be callable")
        raw_value = cast(object, raw_state_dict)
        if not isinstance(raw_value, Mapping):
            raise TypeError("raw inference state must be a mapping")
        ema_value = cast(object, ema_state_dict)
        if ema_value is not None and not isinstance(
            ema_value,
            Mapping,
        ):
            raise TypeError("EMA inference state must be a mapping or null")
        self._model_factory = model_factory
        self._raw_state_dict = cast(Mapping[str, torch.Tensor], raw_value)
        self._ema_state_dict = cast(
            Mapping[str, torch.Tensor] | None,
            ema_value,
        )
        self.device = torch.device(device)

    def resolve(self, weights: WeightSelection) -> tuple[nn.Module, str]:
        """Build the selected model and return its resolved weight label."""

        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("inference weights must be auto, raw, or ema")
        resolved = (
            "ema"
            if weights == "ema"
            or (weights == "auto" and self._ema_state_dict is not None)
            else "raw"
        )
        state = self._ema_state_dict if resolved == "ema" else self._raw_state_dict
        if state is None:
            raise ValueError("EMA weights were requested but are unavailable")
        model_value = cast(object, self._model_factory())
        if not isinstance(model_value, nn.Module):
            raise TypeError("inference model factory must return nn.Module")
        validate_module_state_dict_compatibility(
            model_value,
            state,
            path=f"inference.{resolved}_model_state_dict",
            allow_lazy_state=False,
        )
        model_value.load_state_dict(state, strict=True)
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


class PinnedInferenceModelProvider(InferenceModelProvider):
    """Expose one already-selected inference model without reconstructing it."""

    def __init__(
        self,
        *,
        model: nn.Module,
        weights: PinnedWeightSelection,
        device: torch.device,
    ) -> None:
        model_value = cast(object, model)
        if not isinstance(model_value, nn.Module):
            raise TypeError("pinned inference model must be nn.Module")
        if weights not in {"raw", "ema"}:
            raise ValueError("pinned inference weights must be raw or ema")
        if model_value.training:
            raise ValueError("pinned inference model must already be in eval mode")
        self.device = torch.device(device)
        incompatible_state = [
            name
            for name, tensor in (
                *model_value.named_parameters(),
                *model_value.named_buffers(),
            )
            if tensor.device.type != self.device.type
            or (
                self.device.index is not None
                and tensor.device.index != self.device.index
            )
        ]
        if incompatible_state:
            raise ValueError(
                "pinned inference model state must already be on device "
                f"{self.device}: {', '.join(incompatible_state)}"
            )
        self._pinned_model = model_value
        self._pinned_weights = weights

    def resolve(self, weights: WeightSelection) -> tuple[nn.Module, str]:
        """Return the pinned model when the request preserves its authority."""

        if weights not in {"auto", "raw", "ema"}:
            raise ValueError("inference weights must be auto, raw, or ema")
        if weights not in ("auto", self._pinned_weights):
            raise ValueError(
                f"pinned inference model uses {self._pinned_weights} weights; "
                f"cannot satisfy explicit {weights} request"
            )
        return self._pinned_model, self._pinned_weights


__all__ = [
    "InferenceModelProvider",
    "PinnedInferenceModelProvider",
    "PinnedWeightSelection",
    "WeightSelection",
]
