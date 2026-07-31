"""Extensible simple-loss weighting policies for Gaussian training."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import torch

from stochaflow.families.gaussian import PredictionType
from stochaflow.utils.registry import Registry


@dataclass(frozen=True, slots=True)
class GaussianSimpleLossContext:
    """Batch-aligned Gaussian facts available to a weighting policy."""

    prediction_type: PredictionType
    signal_to_noise_ratio: torch.Tensor

    def __post_init__(self) -> None:
        prediction_type = _validate_prediction_type(self.prediction_type)
        snr_value = cast(object, self.signal_to_noise_ratio)
        if not isinstance(snr_value, torch.Tensor):
            raise TypeError(
                "GaussianSimpleLossContext.signal_to_noise_ratio must be a Tensor"
            )
        if snr_value.ndim != 1:
            raise ValueError(
                "GaussianSimpleLossContext.signal_to_noise_ratio must have "
                "shape [B]"
            )
        if not torch.is_floating_point(snr_value):
            raise TypeError(
                "GaussianSimpleLossContext.signal_to_noise_ratio must be "
                "floating-point"
            )
        if bool(torch.any(torch.isnan(snr_value))):
            raise ValueError(
                "GaussianSimpleLossContext.signal_to_noise_ratio must not "
                "contain NaN"
            )
        if bool(torch.any(snr_value < 0)):
            raise ValueError(
                "GaussianSimpleLossContext.signal_to_noise_ratio must be "
                "non-negative"
            )
        object.__setattr__(self, "prediction_type", prediction_type)


class GaussianSimpleLossWeighting(ABC):
    """Gaussian-family policy that computes one simple-loss weight per sample."""

    __slots__ = ()

    @property
    @abstractmethod
    def requires_per_sample_loss(self) -> bool:
        """Return whether this policy requires per-sample Objective values."""

    @abstractmethod
    def validate_contract(self, *, prediction_type: PredictionType) -> None:
        """Validate compatibility with a configured Gaussian prediction contract."""

    @abstractmethod
    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Compute unnormalized weights aligned with the context batch."""


GaussianSimpleLossWeightingType = TypeVar(
    "GaussianSimpleLossWeightingType",
    bound=GaussianSimpleLossWeighting,
)

_GAUSSIAN_SIMPLE_LOSS_WEIGHTINGS = Registry[
    type[GaussianSimpleLossWeighting]
](
    "Gaussian simple-loss weighting",
    expected_type=GaussianSimpleLossWeighting,
)


def _register_framework_gaussian_simple_loss_weighting(
    name: str,
) -> Callable[
    [type[GaussianSimpleLossWeightingType]],
    type[GaussianSimpleLossWeightingType],
]:
    """Register one framework-owned unqualified policy through the registry."""

    if not name or name != name.strip() or "." in name:
        raise ValueError(
            "framework Gaussian simple-loss weighting names must be "
            "non-empty and unqualified"
        )

    def decorator(
        policy_type: type[GaussianSimpleLossWeightingType],
    ) -> type[GaussianSimpleLossWeightingType]:
        _GAUSSIAN_SIMPLE_LOSS_WEIGHTINGS.register(name)(policy_type)
        return policy_type

    return decorator


@_register_framework_gaussian_simple_loss_weighting("constant")
class ConstantGaussianSimpleLossWeighting(GaussianSimpleLossWeighting):
    """Leave every Gaussian simple-loss contribution unchanged."""

    __slots__ = ()

    @property
    def requires_per_sample_loss(self) -> bool:
        """Return false because scalar Objectives can use the constant fast path."""

        return False

    def validate_contract(self, *, prediction_type: PredictionType) -> None:
        """Accept every supported Gaussian prediction contract."""

        _validate_prediction_type(prediction_type)

    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Return an exact one for each sample."""

        return torch.ones_like(context.signal_to_noise_ratio)


@_register_framework_gaussian_simple_loss_weighting("p2")
@dataclass(frozen=True, slots=True)
class P2GaussianSimpleLossWeighting(GaussianSimpleLossWeighting):
    """Apply the P2 signal-to-noise weighting policy to epsilon prediction."""

    k: float = 1.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        k = _finite_number(self.k, path="P2GaussianSimpleLossWeighting.k")
        gamma = _finite_number(
            self.gamma,
            path="P2GaussianSimpleLossWeighting.gamma",
        )
        if k <= 0.0:
            raise ValueError(
                "P2GaussianSimpleLossWeighting.k must be greater than zero"
            )
        if gamma < 0.0:
            raise ValueError(
                "P2GaussianSimpleLossWeighting.gamma must be non-negative"
            )
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "gamma", gamma)

    @property
    def requires_per_sample_loss(self) -> bool:
        """Return true because P2 assigns a distinct weight to each sample."""

        return True

    def validate_contract(self, *, prediction_type: PredictionType) -> None:
        """Require the epsilon-prediction formulation derived by P2."""

        prediction_type = _validate_prediction_type(prediction_type)
        if prediction_type != "epsilon":
            raise ValueError(
                "P2 Gaussian simple-loss weighting requires "
                "prediction_type='epsilon'"
            )

    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Return the exact unnormalized ``(k + SNR) ** (-gamma)`` weights."""

        return (self.k + context.signal_to_noise_ratio).pow(-self.gamma)


def register_gaussian_simple_loss_weighting(
    name: str,
) -> Callable[
    [type[GaussianSimpleLossWeightingType]],
    type[GaussianSimpleLossWeightingType],
]:
    """Register one namespaced third-party Gaussian weighting policy class."""

    _validate_extension_name(name)

    def decorator(
        policy_type: type[GaussianSimpleLossWeightingType],
    ) -> type[GaussianSimpleLossWeightingType]:
        _GAUSSIAN_SIMPLE_LOSS_WEIGHTINGS.register(name)(policy_type)
        return policy_type

    return decorator


def build_gaussian_simple_loss_weighting(
    value: object,
    *,
    path: str,
    registry: Registry[
        type[GaussianSimpleLossWeighting]
    ] = _GAUSSIAN_SIMPLE_LOSS_WEIGHTINGS,
) -> GaussianSimpleLossWeighting:
    """Build a policy from the strict ``{name, params}`` family declaration."""

    name, params = _parse_weighting_declaration(value, path=path)
    policy = registry.create(name, **params)
    if not isinstance(policy, GaussianSimpleLossWeighting):
        raise TypeError(
            f"registered Gaussian simple-loss weighting '{name}' constructed "
            f"{type(policy).__name__}, expected GaussianSimpleLossWeighting"
        )
    return policy


def compute_gaussian_simple_loss_weights(
    policy: GaussianSimpleLossWeighting,
    context: GaussianSimpleLossContext,
) -> torch.Tensor:
    """Compute and centrally validate one policy's batch-aligned weights."""

    policy_value = cast(object, policy)
    if not isinstance(policy_value, GaussianSimpleLossWeighting):
        raise TypeError(
            "Gaussian simple-loss policy must inherit "
            "GaussianSimpleLossWeighting"
        )
    context_value = cast(object, context)
    if not isinstance(context_value, GaussianSimpleLossContext):
        raise TypeError(
            "Gaussian simple-loss context must be GaussianSimpleLossContext"
        )
    policy.validate_contract(prediction_type=context.prediction_type)
    weights = policy.sample_weights(context)
    return _validate_sample_weights(weights, context=context)


def _parse_weighting_declaration(
    value: object,
    *,
    path: str,
) -> tuple[str, dict[str, Any]]:
    if value is None:
        return "constant", {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    unknown = sorted(
        (key for key in value if key not in {"name", "params"}),
        key=str,
    )
    if unknown:
        raise ValueError(
            f"unknown {path} field(s): "
            + ", ".join(map(str, unknown))
            + f"; component parameters must be nested under {path}.params"
        )
    name_value = value.get("name")
    if not isinstance(name_value, str) or not name_value.strip():
        raise TypeError(f"{path}.name must be a non-empty string")
    if name_value != name_value.strip():
        raise ValueError(
            f"{path}.name must not contain leading or trailing whitespace"
        )
    if "params" not in value:
        raise ValueError(
            f"{path} must use the canonical {{name, params}} declaration; "
            f"{path}.params is required and may be an empty mapping"
        )
    params_value = value["params"]
    if not isinstance(params_value, Mapping):
        raise TypeError(f"{path}.params must be a mapping")
    params: dict[str, Any] = {}
    for key, parameter in params_value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{path}.params keys must be non-empty strings")
        params[key] = parameter
    return name_value, params


def _validate_sample_weights(
    value: object,
    *,
    context: GaussianSimpleLossContext,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Gaussian simple-loss weights must be a Tensor")
    expected = context.signal_to_noise_ratio
    if value.ndim != 1 or value.shape != expected.shape:
        raise ValueError(
            "Gaussian simple-loss weights must have shape [B] matching SNR; "
            f"expected {tuple(expected.shape)}, got {tuple(value.shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError("Gaussian simple-loss weights must be floating-point")
    if value.device != expected.device:
        raise ValueError(
            "Gaussian simple-loss weights must share the SNR device"
        )
    if value.dtype != expected.dtype:
        raise ValueError("Gaussian simple-loss weights must share the SNR dtype")
    if not bool(torch.all(torch.isfinite(value))):
        raise ValueError("Gaussian simple-loss weights must be finite")
    if bool(torch.any(value < 0)):
        raise ValueError("Gaussian simple-loss weights must be non-negative")
    return value


def _validate_prediction_type(value: object) -> PredictionType:
    if not isinstance(value, str) or value not in {
        "epsilon",
        "x0",
        "v",
        "score",
    }:
        raise ValueError(
            "Gaussian prediction_type must be epsilon, x0, v, or score"
        )
    return cast(PredictionType, value)


def _validate_extension_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            "Gaussian simple-loss weighting registry name must be a non-empty "
            "string"
        )
    if value != value.strip():
        raise ValueError(
            "Gaussian simple-loss weighting registry name must not contain "
            "leading or trailing whitespace"
        )
    segments = value.split(".")
    if len(segments) < 2 or any(not segment for segment in segments):
        raise ValueError(
            "third-party Gaussian simple-loss weighting names must be "
            "namespaced, for example 'my-extension.policy'"
        )
    return value


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


__all__ = [
    "ConstantGaussianSimpleLossWeighting",
    "GaussianSimpleLossContext",
    "GaussianSimpleLossWeighting",
    "P2GaussianSimpleLossWeighting",
    "build_gaussian_simple_loss_weighting",
    "compute_gaussian_simple_loss_weights",
    "register_gaussian_simple_loss_weighting",
]
