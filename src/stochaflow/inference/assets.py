"""Lazy reconstruction of checkpoint-embedded inference assets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Self, cast

import torch
from torch import nn

from stochaflow.utils.checkpoint import (
    InferenceAssetDescriptor,
    validate_inference_asset_descriptors,
    validate_module_state_dict_compatibility,
)
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.device import move_module_to_device

InferenceAssetModelFactory = Callable[[ComponentConfig], nn.Module]


def _unavailable_model_factory(component: ComponentConfig) -> nn.Module:
    del component
    raise RuntimeError("empty inference asset provider has no model factory")


def _nonempty_exact_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result or result.strip() != result:
        raise ValueError(
            f"{path} must be non-empty and contain no surrounding whitespace"
        )
    return result


class InferenceAssetProvider:
    """Lazily reconstruct declared inference modules from embedded state."""

    def __init__(
        self,
        *,
        descriptors: Mapping[str, InferenceAssetDescriptor],
        state_dicts: Mapping[str, Mapping[str, object]],
        device: torch.device,
        model_factory: InferenceAssetModelFactory,
    ) -> None:
        descriptor_value = cast(object, descriptors)
        if not isinstance(descriptor_value, Mapping):
            raise TypeError("inference asset descriptors must be a mapping")
        state_dict_value = cast(object, state_dicts)
        if not isinstance(state_dict_value, Mapping):
            raise TypeError("inference asset state dictionaries must be a mapping")
        if not callable(model_factory):
            raise TypeError("inference asset model_factory must be callable")

        validated_descriptors = validate_inference_asset_descriptors(
            dict(descriptor_value),
            path="inference asset descriptors",
        )
        validated_states: dict[str, Mapping[str, object]] = {}
        for asset_name_value, state_value in state_dict_value.items():
            asset_name = _nonempty_exact_string(
                asset_name_value,
                path="inference asset state name",
            )
            if not isinstance(state_value, Mapping):
                raise TypeError(
                    "inference asset state "
                    f"{asset_name!r} must be a mapping"
                )
            validated_states[asset_name] = cast(Mapping[str, object], state_value)

        referenced_assets = {
            descriptor["training_asset_name"]
            for descriptor in validated_descriptors.values()
        }
        state_assets = set(validated_states)
        if state_assets != referenced_assets:
            missing = sorted(referenced_assets - state_assets)
            unexpected = sorted(state_assets - referenced_assets)
            raise ValueError(
                "inference asset states do not match descriptors: "
                f"missing={missing or '<none>'}, "
                f"unexpected={unexpected or '<none>'}"
            )

        self._descriptors = validated_descriptors
        self._state_dicts = validated_states
        self._device = torch.device(device)
        self._model_factory = model_factory
        self._cache: dict[str, nn.Module] = {}

    @classmethod
    def empty(cls) -> Self:
        """Create a provider with no declared inference assets."""

        return cls(
            descriptors={},
            state_dicts={},
            device=torch.device("cpu"),
            model_factory=_unavailable_model_factory,
        )

    def get(
        self,
        slot: str,
        *,
        expected_capability_role: str,
    ) -> nn.Module:
        """Resolve one declared slot after validating its semantic role."""

        slot_name = _nonempty_exact_string(slot, path="inference asset slot")
        expected_role = _nonempty_exact_string(
            expected_capability_role,
            path="expected inference asset capability role",
        )
        try:
            descriptor = self._descriptors[slot_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._descriptors)) or "<none>"
            raise KeyError(
                f"unknown inference asset slot {slot_name!r}; "
                f"available: {available}"
            ) from exc
        actual_role = descriptor["capability_role"]
        if actual_role != expected_role:
            raise ValueError(
                f"inference asset slot {slot_name!r} has capability role "
                f"{actual_role!r}, expected {expected_role!r}"
            )
        cached = self._cache.get(slot_name)
        if cached is not None:
            return cached

        declaration = descriptor["declaration"]
        component = ComponentConfig(
            name=declaration["name"],
            params=deepcopy(declaration["params"]),
        )
        module_value = cast(object, self._model_factory(component))
        if not isinstance(module_value, nn.Module):
            raise TypeError(
                f"inference asset slot {slot_name!r} factory must return nn.Module"
            )
        module = module_value
        training_asset_name = descriptor["training_asset_name"]
        state_dict = self._state_dicts[training_asset_name]
        state_path = f"inference asset state {training_asset_name!r}"
        validate_module_state_dict_compatibility(
            module,
            state_dict,
            path=state_path,
            allow_lazy_state=False,
        )
        module.load_state_dict(cast(Mapping[str, Any], state_dict), strict=True)
        move_module_to_device(
            module,
            self._device,
            role=f"inference asset {slot_name!r}",
        )
        module.eval()
        self._cache[slot_name] = module
        return module


__all__ = ["InferenceAssetProvider"]
