"""Checkpoint-owned sampling recipe declarations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class SamplingRecipe:
    """Internal SamplingBuilder identity and its fixed inference contract."""

    name: str
    contract: Mapping[str, Any] = field(default_factory=dict)


def validate_sampling_recipe(
    value: object,
    *,
    path: str = "sampling recipe",
) -> SamplingRecipe:
    """Validate and detach a TrainingPlan-provided sampling recipe."""

    if not isinstance(value, SamplingRecipe):
        raise TypeError(f"{path} must be SamplingRecipe")
    name = _nonempty_string(
        cast(object, value.name),
        path=f"{path}.name",
    )
    declared_contract = cast(object, value.contract)
    contract = _contract_mapping(
        declared_contract,
        path=f"{path}.contract",
        allow_frozen=isinstance(declared_contract, MappingProxyType),
    )
    return SamplingRecipe(
        name=name,
        contract=MappingProxyType(contract),
    )


def sampling_recipe_to_dict(value: SamplingRecipe) -> dict[str, Any]:
    """Serialize a validated recipe into its strict checkpoint representation."""

    recipe = validate_sampling_recipe(value)
    return {
        "schema_version": 1,
        "name": recipe.name,
        "contract": _thaw_contract_mapping(recipe.contract),
    }


def sampling_recipe_from_dict(
    value: object,
    *,
    path: str = "checkpoint.inference_recipe",
) -> SamplingRecipe:
    """Parse a strict checkpoint recipe representation."""

    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dictionary")
    raw = cast(dict[object, object], value)
    if any(type(key) is not str for key in raw):
        raise TypeError(f"{path} field names must be exact strings")
    expected = {"schema_version", "name", "contract"}
    actual = cast(set[str], set(raw))
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{path} has invalid fields: missing={missing or '<none>'}, "
            f"unknown={unknown or '<none>'}"
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError(f"{path}.schema_version must be the exact integer 1")
    return validate_sampling_recipe(
        SamplingRecipe(
            name=_nonempty_string(
                raw["name"],
                path=f"{path}.name",
            ),
            contract=cast(Any, raw["contract"]),
        ),
        path=path,
    )


def resolve_sampling_recipe_params(
    recipe: SamplingRecipe,
    *,
    options: Mapping[str, Any],
    sampler: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge mutable invocation values without overriding the fixed contract."""

    validated = validate_sampling_recipe(recipe)
    option_values = deepcopy(dict(options))
    contract_values = _thaw_contract_mapping(validated.contract)
    collisions = sorted(set(option_values).intersection(validated.contract))
    if collisions:
        raise ValueError(
            "sampling options cannot override fixed inference contract field(s): "
            + ", ".join(collisions)
        )
    if "sampler" in option_values:
        raise ValueError(
            "sampling options cannot contain sampler; use sampling.sampler"
        )
    if sampler is not None:
        if "sampler" in validated.contract:
            raise ValueError(
                "sampling sampler cannot override the fixed inference contract"
            )
        option_values["sampler"] = deepcopy(dict(sampler))
    option_values.update(contract_values)
    return option_values


def _nonempty_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result.strip():
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


def _contract_mapping(
    value: object,
    *,
    path: str,
    allow_frozen: bool,
) -> dict[str, Any]:
    if type(value) is not dict and not (
        allow_frozen and isinstance(value, MappingProxyType)
    ):
        raise TypeError(f"{path} must be an exact dictionary")
    declared = cast(Mapping[object, object], value)
    result: dict[str, Any] = {}
    for key, item in declared.items():
        if type(key) is not str or not key:
            raise TypeError(f"{path} keys must be non-empty exact strings")
        result[cast(str, key)] = _contract_value(
            cast(object, item),
            path=f"{path}[{key!r}]",
            allow_frozen=allow_frozen,
        )
    return result


def _contract_value(
    value: object,
    *,
    path: str,
    allow_frozen: bool,
) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        number = cast(float, value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite")
        return number
    if type(value) is list or (allow_frozen and type(value) is tuple):
        return tuple(
            _contract_value(
                item,
                path=f"{path}[{index}]",
                allow_frozen=allow_frozen,
            )
            for index, item in enumerate(cast(list[object] | tuple[object, ...], value))
        )
    if type(value) is dict or (
        allow_frozen and isinstance(value, MappingProxyType)
    ):
        raw = cast(Mapping[object, object], value)
        result: dict[str, Any] = {}
        for key, item in raw.items():
            if type(key) is not str or not key:
                raise TypeError(f"{path} keys must be non-empty exact strings")
            result[cast(str, key)] = _contract_value(
                item,
                path=f"{path}[{key!r}]",
                allow_frozen=allow_frozen,
            )
        return MappingProxyType(result)
    raise TypeError(
        f"{path} contains unsupported value type "
        f"'{type(value).__module__}.{type(value).__qualname__}'"
    )


def _thaw_contract_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _thaw_contract_value(item)
        for key, item in value.items()
    }


def _thaw_contract_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            cast(str, key): _thaw_contract_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_contract_value(item) for item in value]
    return value


__all__ = [
    "SamplingRecipe",
    "resolve_sampling_recipe_params",
    "sampling_recipe_from_dict",
    "sampling_recipe_to_dict",
    "validate_sampling_recipe",
]
