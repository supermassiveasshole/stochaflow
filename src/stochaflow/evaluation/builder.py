"""Evaluation composition boundary and plan validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from packaging.utils import InvalidName, canonicalize_name
from torch import nn

from stochaflow.evaluation.config import (
    EvaluationProtocol,
    _freeze_evaluation_mapping,
    _snapshot_metric_specs,
)
from stochaflow.evaluation.contracts import Evaluator
from stochaflow.evaluation.predictions import EvaluationArtifactSink
from stochaflow.evaluation.sampling import EvaluationSamplingCapability
from stochaflow.metrics.config import MetricSpec
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog


@dataclass(frozen=True, slots=True)
class EvaluationProtocolIdentity:
    """Task-declared provider and preprocessing facts that affect scores."""

    providers: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    metric_providers: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        providers = _freeze_evaluation_mapping(
            self.providers,
            path="evaluation protocol identity providers",
        )
        if not providers:
            raise ValueError(
                "evaluation protocol identity providers must be non-empty"
            )
        preprocessing = _freeze_evaluation_mapping(
            self.preprocessing,
            path="evaluation protocol identity preprocessing",
        )
        if not preprocessing:
            raise ValueError(
                "evaluation protocol identity preprocessing must be non-empty"
            )
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "preprocessing", preprocessing)
        declared_metric_providers = cast(object, self.metric_providers)
        if type(declared_metric_providers) is not tuple:
            raise TypeError(
                "evaluation protocol identity metric_providers must be an "
                "exact tuple"
            )
        metric_providers: list[str] = []
        seen_metric_providers: set[str] = set()
        for index, declared_name in enumerate(self.metric_providers):
            name = _non_empty_string(
                declared_name,
                path=(
                    "evaluation protocol identity "
                    f"metric_providers[{index}]"
                ),
            )
            if name in seen_metric_providers:
                raise ValueError(
                    "evaluation protocol identity contains duplicate metric "
                    f"provider {name!r}"
                )
            seen_metric_providers.add(name)
            metric_providers.append(name)
        object.__setattr__(self, "metric_providers", tuple(metric_providers))
        declared_dependencies = cast(object, self.dependencies)
        if type(declared_dependencies) is not tuple:
            raise TypeError(
                "evaluation protocol identity dependencies must be an exact tuple"
            )
        dependencies: list[str] = []
        seen: set[str] = set()
        for index, declared_name in enumerate(self.dependencies):
            raw_name = _non_empty_string(
                declared_name,
                path=f"evaluation protocol identity dependencies[{index}]",
            )
            try:
                name = canonicalize_name(raw_name, validate=True)
            except InvalidName as error:
                raise ValueError(
                    "evaluation protocol identity dependency must be a valid "
                    f"distribution name: {raw_name!r}"
                ) from error
            if name in seen:
                raise ValueError(
                    "evaluation protocol identity contains duplicate dependency "
                    f"{name!r}"
                )
            seen.add(name)
            dependencies.append(name)
        object.__setattr__(self, "dependencies", tuple(dependencies))


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Fully composed evaluator and core-managed evaluation dependencies."""

    evaluator: Evaluator
    data: Iterable[Any]
    metric_specs: tuple[MetricSpec, ...]
    protocol: EvaluationProtocol
    subject: object
    data_identity: Mapping[str, Any]
    protocol_identity: EvaluationProtocolIdentity
    artifact_sink: EvaluationArtifactSink | None = None
    modules: Mapping[str, nn.Module] = field(default_factory=dict)


class EvaluationBuilderContext:
    """Read-only injected dependencies available to an EvaluationBuilder."""

    def __init__(
        self,
        *,
        params: Mapping[str, Any],
        subject: object,
        data: Iterable[Any],
        data_identity: Mapping[str, Any],
        inference: object | None,
        metric_specs: Sequence[MetricSpec],
        protocol: EvaluationProtocol,
        artifact_root: Path | None = None,
        sampling: EvaluationSamplingCapability | None = None,
    ) -> None:
        self.params = _freeze_evaluation_mapping(
            params,
            path="evaluation builder params",
        )
        if subject is None:
            raise TypeError("evaluation builder subject must be resolved")
        _validate_reiterable(data, path="evaluation builder data")
        if not isinstance(cast(object, protocol), EvaluationProtocol):
            raise TypeError("evaluation builder protocol must be EvaluationProtocol")
        self.subject = subject
        self.data = data
        self.data_identity = _freeze_evaluation_mapping(
            data_identity,
            path="evaluation builder data_identity",
        )
        self.inference = inference
        artifact_root_value = cast(object, artifact_root)
        if artifact_root_value is not None:
            if not isinstance(artifact_root_value, Path):
                raise TypeError("evaluation builder artifact_root must be a Path")
            if not artifact_root_value.is_absolute():
                raise ValueError(
                    "evaluation builder artifact_root must be an absolute Path"
                )
            if not artifact_root_value.is_dir():
                raise NotADirectoryError(
                    "evaluation builder artifact_root must be an existing directory"
                )
        self.artifact_root = artifact_root
        sampling_value = cast(object, sampling)
        if sampling_value is not None and not isinstance(
            sampling_value,
            EvaluationSamplingCapability,
        ):
            raise TypeError(
                "evaluation builder sampling must implement "
                "EvaluationSamplingCapability"
            )
        self.sampling = sampling
        self.metric_specs = _snapshot_metric_specs(
            metric_specs,
            path="evaluation builder metric_specs",
        )
        self.protocol = protocol


class EvaluationBuilder(ABC):
    """Compose one task-specific evaluation from injected narrow capabilities."""

    def __init__(self, context: EvaluationBuilderContext) -> None:
        self.context = context

    @abstractmethod
    def build(self) -> EvaluationPlan:
        """Return one complete plan without starting evaluation runtime."""


REGISTRIES.evaluation_builders.require_base(EvaluationBuilder)


def build_evaluation_plan(
    declaration: ComponentConfig,
    *,
    subject: object,
    data: Iterable[Any],
    data_identity: Mapping[str, Any],
    inference: object | None,
    metric_specs: Sequence[MetricSpec],
    protocol: EvaluationProtocol,
    artifact_root: Path | None = None,
    sampling: EvaluationSamplingCapability | None = None,
    registries: RegistryCatalog = REGISTRIES,
) -> EvaluationPlan:
    """Construct and validate one registered EvaluationBuilder result."""

    if not isinstance(cast(object, declaration), ComponentConfig):
        raise TypeError("evaluation declaration must be ComponentConfig")
    context = EvaluationBuilderContext(
        params=declaration.params,
        subject=subject,
        data=data,
        data_identity=data_identity,
        inference=inference,
        metric_specs=metric_specs,
        protocol=protocol,
        artifact_root=artifact_root,
        sampling=sampling,
    )
    builder_value = registries.evaluation_builders.create(
        declaration.name,
        context,
    )
    if not isinstance(builder_value, EvaluationBuilder):
        raise TypeError("registered evaluation builder must inherit EvaluationBuilder")
    plan = validate_evaluation_plan(cast(object, builder_value.build()))
    if plan.subject is not context.subject:
        raise ValueError("EvaluationPlan.subject must preserve the injected subject")
    if plan.data is not context.data:
        raise ValueError("EvaluationPlan.data must preserve the injected data")
    if plan.protocol is not context.protocol:
        raise ValueError("EvaluationPlan.protocol must preserve the injected protocol")
    if plan.data_identity != context.data_identity:
        raise ValueError(
            "EvaluationPlan.data_identity must preserve the injected identity"
        )
    if plan.metric_specs != context.metric_specs:
        raise ValueError(
            "EvaluationPlan.metric_specs must preserve the injected declarations"
        )
    return plan


def validate_evaluation_plan(value: object) -> EvaluationPlan:
    """Validate and snapshot an extension-produced evaluation plan."""

    if not isinstance(value, EvaluationPlan):
        raise TypeError("EvaluationBuilder.build() must return EvaluationPlan")
    evaluator = cast(object, value.evaluator)
    if not isinstance(evaluator, Evaluator):
        raise TypeError("EvaluationPlan.evaluator must implement Evaluator")
    _validate_reiterable(value.data, path="EvaluationPlan.data")
    if value.subject is None:
        raise TypeError("EvaluationPlan.subject must be resolved")
    if not isinstance(cast(object, value.protocol), EvaluationProtocol):
        raise TypeError("EvaluationPlan.protocol must be EvaluationProtocol")
    if not isinstance(
        cast(object, value.protocol_identity),
        EvaluationProtocolIdentity,
    ):
        raise TypeError(
            "EvaluationPlan.protocol_identity must be EvaluationProtocolIdentity"
        )
    metric_specs = _snapshot_metric_specs(
        value.metric_specs,
        path="EvaluationPlan.metric_specs",
    )
    channels = _evaluator_channels(value.evaluator)
    missing_channels = sorted({spec.channel for spec in metric_specs} - channels)
    if missing_channels:
        raise ValueError(
            "EvaluationPlan.evaluator is missing metric channel(s): "
            + ", ".join(missing_channels)
        )
    data_identity = _freeze_evaluation_mapping(
        value.data_identity,
        path="EvaluationPlan.data_identity",
    )
    declared_modules = cast(object, value.modules)
    if not isinstance(declared_modules, Mapping):
        raise TypeError("EvaluationPlan.modules must be a mapping")
    modules: dict[str, nn.Module] = {}
    for declared_name, declared_module in cast(
        Mapping[object, object],
        declared_modules,
    ).items():
        name = _non_empty_string(
            declared_name,
            path="EvaluationPlan.modules name",
        )
        if not isinstance(cast(object, declared_module), nn.Module):
            raise TypeError(f"EvaluationPlan.modules[{name!r}] must be an nn.Module")
        modules[name] = cast(nn.Module, declared_module)
    sink_value = cast(object, value.artifact_sink)
    if sink_value is not None and not isinstance(
        sink_value,
        EvaluationArtifactSink,
    ):
        raise TypeError(
            "EvaluationPlan.artifact_sink must implement EvaluationArtifactSink"
        )
    return EvaluationPlan(
        evaluator=value.evaluator,
        data=value.data,
        metric_specs=metric_specs,
        protocol=value.protocol,
        subject=value.subject,
        data_identity=data_identity,
        protocol_identity=value.protocol_identity,
        artifact_sink=value.artifact_sink,
        modules=MappingProxyType(modules),
    )


def _validate_reiterable(value: object, *, path: str) -> None:
    if not isinstance(value, Iterable):
        raise TypeError(f"{path} must be iterable")
    if iter(value) is value:
        raise TypeError(f"{path} must be re-iterable, not a one-shot iterator")


def _evaluator_channels(evaluator: Evaluator) -> frozenset[str]:
    declared = cast(object, evaluator.metric_channels)
    if isinstance(declared, (str, bytes)) or not isinstance(
        declared,
        Collection,
    ):
        raise TypeError("EvaluationPlan.evaluator.metric_channels must be a collection")
    channels: set[str] = set()
    for index, declared_channel in enumerate(declared):
        channels.add(
            _non_empty_string(
                declared_channel,
                path=f"EvaluationPlan.evaluator.metric_channels[{index}]",
            )
        )
    return frozenset(channels)


def _non_empty_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result:
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


__all__ = [
    "EvaluationBuilder",
    "EvaluationBuilderContext",
    "EvaluationPlan",
    "EvaluationProtocolIdentity",
    "build_evaluation_plan",
    "validate_evaluation_plan",
]
