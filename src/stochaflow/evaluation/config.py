"""Strict standalone configuration for formal evaluation runs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from stochaflow.metrics.config import MetricSpec, validate_metric_spec
from stochaflow.utils.config import ComponentConfig, ConfigError

EvaluationPurpose = Literal["selection_candidate", "final_test", "benchmark"]
EvaluationSplit = Literal["validation", "test"]
EvaluationDataSource = Literal["checkpoint", "prediction_artifact"]
CheckpointWeightVariant = Literal["raw", "ema"]

EVALUATION_PURPOSES = frozenset({"selection_candidate", "final_test", "benchmark"})
EVALUATION_SPLITS = frozenset({"validation", "test"})
EVALUATION_DATA_SOURCES = frozenset({"checkpoint", "prediction_artifact"})
CHECKPOINT_WEIGHT_VARIANTS = frozenset({"raw", "ema"})
_JSON_SCALAR_TYPES = (type(None), bool, int, float, str)


def _non_empty_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result:
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


def _freeze_evaluation_value(value: Any, *, path: str) -> Any:
    """Snapshot a JSON-shaped value into immutable builtin containers."""

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for declared_key, item in value.items():
            key = _non_empty_string(declared_key, path=f"{path} key")
            normalized[key] = _freeze_evaluation_value(
                item,
                path=f"{path}[{key!r}]",
            )
        return MappingProxyType(normalized)
    if type(value) in {list, tuple}:
        return tuple(
            _freeze_evaluation_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(cast(Sequence[Any], value))
        )
    if type(value) not in _JSON_SCALAR_TYPES:
        raise TypeError(
            f"{path} contains unsupported value type {type(value).__name__!r}; "
            "evaluation metadata supports JSON scalar, mapping, and sequence values"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _freeze_evaluation_mapping(
    value: object,
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    frozen = _freeze_evaluation_value(value, path=path)
    return cast(Mapping[str, Any], frozen)


def _thaw_evaluation_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_evaluation_value(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_thaw_evaluation_value(item) for item in cast(Sequence[Any], value)]
    return value


def _snapshot_component(
    value: object,
    *,
    path: str,
) -> ComponentConfig:
    if not isinstance(value, ComponentConfig):
        raise TypeError(f"{path} must be ComponentConfig")
    name = _non_empty_string(value.name, path=f"{path}.name")
    params = _freeze_evaluation_mapping(value.params, path=f"{path}.params")
    return ComponentConfig(
        name=name,
        params=cast(dict[str, Any], params),
    )


def _snapshot_metric_specs(
    value: object,
    *,
    path: str,
) -> tuple[MetricSpec, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of MetricSpec values")
    result: list[MetricSpec] = []
    seen_ids: set[str] = set()
    for index, declared_spec in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(declared_spec, MetricSpec):
            raise TypeError(f"{item_path} must be a MetricSpec")
        mutable_params = cast(
            dict[str, Any],
            _thaw_evaluation_value(declared_spec.params),
        )
        candidate = MetricSpec(
            id=declared_spec.id,
            name=declared_spec.name,
            channel=declared_spec.channel,
            params=mutable_params,
        )
        try:
            validate_metric_spec(candidate, path=item_path)
        except (TypeError, ValueError) as error:
            raise type(error)(str(error)) from error
        if candidate.id in seen_ids:
            raise ValueError(f"{path} contains duplicate metric id {candidate.id!r}")
        seen_ids.add(candidate.id)
        frozen_params = _freeze_evaluation_mapping(
            candidate.params,
            path=f"{item_path}.params",
        )
        result.append(
            MetricSpec(
                id=candidate.id,
                name=candidate.name,
                channel=candidate.channel,
                params=cast(dict[str, Any], frozen_params),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvaluationExtensionsConfig:
    """Extension plugin selection for one standalone evaluation authority."""

    plugins: tuple[str, ...] | None = ()

    def __post_init__(self) -> None:
        plugins = self.plugins
        if plugins is None:
            return
        if type(plugins) is not tuple:
            raise TypeError("extensions.plugins must be an exact tuple or None")
        normalized: list[str] = []
        seen: set[str] = set()
        for index, declared_name in enumerate(plugins):
            name = _non_empty_string(
                declared_name,
                path=f"extensions.plugins[{index}]",
            )
            if name in seen:
                raise ValueError(
                    f"extensions.plugins contains duplicate plugin {name!r}"
                )
            seen.add(name)
            normalized.append(name)
        object.__setattr__(self, "plugins", tuple(normalized))


@dataclass(frozen=True, slots=True)
class CheckpointSubjectConfig:
    """Explicit checkpoint-backed evaluation subject declaration."""

    kind: Literal["checkpoint"]
    path: Path
    weights: CheckpointWeightVariant

    def __post_init__(self) -> None:
        if self.kind != "checkpoint":
            raise ValueError("subject.kind must be 'checkpoint'")
        path_value = cast(object, self.path)
        if not isinstance(path_value, Path):
            raise TypeError("subject.path must be a Path")
        if not str(path_value):
            raise ValueError("subject.path must be non-empty")
        if self.weights not in CHECKPOINT_WEIGHT_VARIANTS:
            raise ValueError("subject.weights must be 'raw' or 'ema'")


@dataclass(frozen=True, slots=True)
class PredictionArtifactSubjectConfig:
    """Safe manifest-backed subject declaration for offline scoring."""

    kind: Literal["prediction_artifact"]
    path: Path

    def __post_init__(self) -> None:
        if self.kind != "prediction_artifact":
            raise ValueError("subject.kind must be 'prediction_artifact'")
        path_value = cast(object, self.path)
        if not isinstance(path_value, Path):
            raise TypeError("subject.path must be a Path")
        if not str(path_value):
            raise ValueError("subject.path must be non-empty")


EvaluationSubjectConfig = (
    CheckpointSubjectConfig | PredictionArtifactSubjectConfig
)


@dataclass(frozen=True, slots=True)
class EvaluationDataConfig:
    """Selected data authority and governance split for evaluation."""

    source: EvaluationDataSource
    split: EvaluationSplit

    def __post_init__(self) -> None:
        if self.source not in EVALUATION_DATA_SOURCES:
            raise ValueError(
                "data.source must be 'checkpoint' or 'prediction_artifact'"
            )
        if self.split not in EVALUATION_SPLITS:
            raise ValueError("data.split must be 'validation' or 'test'")


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Core-visible completeness protocol for one evaluation case."""

    id: str
    expected_examples: int
    strict_complete: bool = True

    def __post_init__(self) -> None:
        _non_empty_string(self.id, path="protocol.id")
        expected = cast(object, self.expected_examples)
        if type(expected) is not int or cast(int, expected) <= 0:
            raise ValueError("protocol.expected_examples must be a positive integer")
        if type(cast(object, self.strict_complete)) is not bool:
            raise TypeError("protocol.strict_complete must be a bool")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Complete standalone authority for one formal evaluation run."""

    version: int
    name: str
    purpose: EvaluationPurpose
    subject: EvaluationSubjectConfig
    data: EvaluationDataConfig
    evaluation: ComponentConfig
    metrics: tuple[MetricSpec, ...]
    protocol: EvaluationProtocol
    extensions: EvaluationExtensionsConfig = field(
        default_factory=EvaluationExtensionsConfig
    )

    def __post_init__(self) -> None:
        if type(cast(object, self.version)) is not int or self.version != 1:
            raise ValueError("version must be integer 1")
        _non_empty_string(self.name, path="name")
        if self.purpose not in EVALUATION_PURPOSES:
            raise ValueError(
                "purpose must be 'selection_candidate', 'final_test', or 'benchmark'"
            )
        if not isinstance(cast(object, self.extensions), EvaluationExtensionsConfig):
            raise TypeError("extensions must be EvaluationExtensionsConfig")
        if not isinstance(
            cast(object, self.subject),
            (CheckpointSubjectConfig, PredictionArtifactSubjectConfig),
        ):
            raise TypeError(
                "subject must be CheckpointSubjectConfig or "
                "PredictionArtifactSubjectConfig"
            )
        if not isinstance(cast(object, self.data), EvaluationDataConfig):
            raise TypeError("data must be EvaluationDataConfig")
        if not isinstance(cast(object, self.protocol), EvaluationProtocol):
            raise TypeError("protocol must be EvaluationProtocol")
        if self.subject.kind != self.data.source:
            raise ValueError(
                "subject.kind and data.source must select the same authority"
            )
        _validate_purpose_split(self.purpose, self.data.split)
        object.__setattr__(
            self,
            "evaluation",
            _snapshot_component(self.evaluation, path="evaluation"),
        )
        object.__setattr__(
            self,
            "metrics",
            _snapshot_metric_specs(self.metrics, path="metrics"),
        )


def _validate_purpose_split(
    purpose: EvaluationPurpose,
    split: EvaluationSplit,
) -> None:
    required = {
        "selection_candidate": "validation",
        "final_test": "test",
    }.get(purpose)
    if required is not None and split != required:
        raise ValueError(
            f"purpose {purpose!r} requires data.split {required!r}, got {split!r}"
        )


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ConfigError(f"{path} must be a mapping")
    mapping = cast(dict[object, Any], value)
    for key in mapping:
        if type(key) is not str:
            raise ConfigError(f"{path} field names must be strings")
    return cast(dict[str, Any], mapping)


def _fields(
    value: object,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    mapping = _mapping(value, path=path)
    unknown = sorted(set(mapping) - required - optional)
    if unknown:
        raise ConfigError(f"{path} contains unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(mapping))
    if missing:
        raise ConfigError(f"{path} is missing required field(s): {', '.join(missing)}")
    return mapping


def _parse_extensions(value: object) -> EvaluationExtensionsConfig:
    raw = _fields(
        value,
        path="extensions",
        required=frozenset({"plugins"}),
    )
    plugins = raw["plugins"]
    if plugins is None:
        return EvaluationExtensionsConfig(None)
    if type(plugins) is not list:
        raise ConfigError("extensions.plugins must be a list or null")
    try:
        return EvaluationExtensionsConfig(
            cast(tuple[str, ...], tuple(cast(list[object], plugins)))
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def _parse_subject(value: object) -> EvaluationSubjectConfig:
    declared = _mapping(value, path="subject")
    kind = declared.get("kind")
    if kind == "checkpoint":
        raw = _fields(
            value,
            path="subject",
            required=frozenset({"kind", "path", "weights"}),
        )
    elif kind == "prediction_artifact":
        raw = _fields(
            value,
            path="subject",
            required=frozenset({"kind", "path"}),
        )
        try:
            path = Path(_non_empty_string(raw["path"], path="subject.path"))
            return PredictionArtifactSubjectConfig(
                kind="prediction_artifact",
                path=path,
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(str(error)) from error
    else:
        raise ConfigError(
            "subject.kind must be 'checkpoint' or 'prediction_artifact'"
        )
    try:
        path = Path(_non_empty_string(raw["path"], path="subject.path"))
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    weights = raw["weights"]
    if weights not in CHECKPOINT_WEIGHT_VARIANTS:
        raise ConfigError("subject.weights must be 'raw' or 'ema'")
    return CheckpointSubjectConfig(
        kind="checkpoint",
        path=path,
        weights=cast(CheckpointWeightVariant, weights),
    )


def _parse_data(value: object) -> EvaluationDataConfig:
    raw = _fields(
        value,
        path="data",
        required=frozenset({"source", "split"}),
    )
    source = raw["source"]
    if source not in EVALUATION_DATA_SOURCES:
        raise ConfigError(
            "data.source must be 'checkpoint' or 'prediction_artifact'"
        )
    split = raw["split"]
    if split not in EVALUATION_SPLITS:
        raise ConfigError("data.split must be 'validation' or 'test'")
    return EvaluationDataConfig(
        source=cast(EvaluationDataSource, source),
        split=cast(EvaluationSplit, split),
    )


def _parse_component(value: object) -> ComponentConfig:
    raw = _fields(
        value,
        path="evaluation",
        required=frozenset({"name"}),
        optional=frozenset({"params"}),
    )
    try:
        name = _non_empty_string(raw["name"], path="evaluation.name")
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error
    params = raw.get("params", {})
    if type(params) is not dict:
        raise ConfigError("evaluation.params must be a mapping")
    try:
        return _snapshot_component(
            ComponentConfig(name=name, params=cast(dict[str, Any], params)),
            path="evaluation",
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def _parse_metrics(value: object) -> tuple[MetricSpec, ...]:
    if type(value) is not list:
        raise ConfigError("metrics must be a list")
    declarations: list[MetricSpec] = []
    for index, item in enumerate(cast(list[object], value)):
        path = f"metrics[{index}]"
        raw = _fields(
            item,
            path=path,
            required=frozenset({"id", "name", "channel"}),
            optional=frozenset({"params"}),
        )
        params = raw.get("params", {})
        if type(params) is not dict:
            raise ConfigError(f"{path}.params must be a mapping")
        declarations.append(
            MetricSpec(
                id=cast(str, raw["id"]),
                name=cast(str, raw["name"]),
                channel=cast(str, raw["channel"]),
                params=cast(dict[str, Any], params),
            )
        )
    try:
        return _snapshot_metric_specs(declarations, path="metrics")
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def _parse_protocol(value: object) -> EvaluationProtocol:
    raw = _fields(
        value,
        path="protocol",
        required=frozenset({"id", "expected_examples"}),
        optional=frozenset({"strict_complete"}),
    )
    try:
        return EvaluationProtocol(
            id=cast(str, raw["id"]),
            expected_examples=cast(int, raw["expected_examples"]),
            strict_complete=cast(bool, raw.get("strict_complete", True)),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def load_evaluation_config_dict(raw: object) -> EvaluationConfig:
    """Parse and validate a standalone evaluation mapping."""

    values = _fields(
        raw,
        path="evaluation config",
        required=frozenset(
            {
                "version",
                "name",
                "purpose",
                "subject",
                "data",
                "evaluation",
                "metrics",
                "protocol",
            }
        ),
        optional=frozenset({"extensions"}),
    )
    extensions = (
        _parse_extensions(values["extensions"])
        if "extensions" in values
        else EvaluationExtensionsConfig()
    )
    try:
        return EvaluationConfig(
            version=cast(int, values["version"]),
            name=cast(str, values["name"]),
            purpose=cast(EvaluationPurpose, values["purpose"]),
            extensions=extensions,
            subject=_parse_subject(values["subject"]),
            data=_parse_data(values["data"]),
            evaluation=_parse_component(values["evaluation"]),
            metrics=_parse_metrics(values["metrics"]),
            protocol=_parse_protocol(values["protocol"]),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def evaluation_config_to_dict(config: EvaluationConfig) -> dict[str, Any]:
    """Serialize one resolved standalone authority into JSON-shaped values."""

    config_value = cast(object, config)
    if not isinstance(config_value, EvaluationConfig):
        raise TypeError("evaluation config must be EvaluationConfig")
    return {
        "version": config.version,
        "name": config.name,
        "purpose": config.purpose,
        "extensions": {
            "plugins": (
                list(config.extensions.plugins)
                if config.extensions.plugins is not None
                else None
            )
        },
        "subject": (
            {
                "kind": config.subject.kind,
                "path": str(config.subject.path),
                "weights": config.subject.weights,
            }
            if isinstance(config.subject, CheckpointSubjectConfig)
            else {
                "kind": config.subject.kind,
                "path": str(config.subject.path),
            }
        ),
        "data": {
            "source": config.data.source,
            "split": config.data.split,
        },
        "evaluation": {
            "name": config.evaluation.name,
            "params": _thaw_evaluation_value(config.evaluation.params),
        },
        "metrics": [
            {
                "id": spec.id,
                "name": spec.name,
                "channel": spec.channel,
                "params": _thaw_evaluation_value(spec.params),
            }
            for spec in config.metrics
        ],
        "protocol": {
            "id": config.protocol.id,
            "expected_examples": config.protocol.expected_examples,
            "strict_complete": config.protocol.strict_complete,
        },
    }


class EvaluationConfigYamlLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping fields."""


def _construct_unique_mapping(
    loader: EvaluationConfigYamlLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str:
            raise ConstructorError(
                None,
                None,
                "evaluation config field names must be strings",
                key_node.start_mark,
            )
        if key in result:
            raise ConstructorError(
                None,
                None,
                f"duplicate field {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


EvaluationConfigYamlLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_evaluation_config_bytes(encoded: bytes) -> EvaluationConfig:
    """Parse one immutable UTF-8 snapshot of a standalone evaluation YAML."""

    if type(encoded) is not bytes:
        raise TypeError("evaluation config snapshot must be exact bytes")
    try:
        document = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigError(f"invalid evaluation config UTF-8: {error}") from error
    try:
        raw = yaml.load(
            document,
            Loader=EvaluationConfigYamlLoader,
        )
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid evaluation config YAML: {error}") from error
    return load_evaluation_config_dict(raw)


def load_evaluation_config_snapshot(
    path: str | Path,
) -> tuple[EvaluationConfig, bytes]:
    """Read, stability-check, and parse one exact evaluation authority snapshot."""

    config_path = Path(path)
    with config_path.open("rb") as source:
        before = os.fstat(source.fileno())
        encoded = source.read()
        after = os.fstat(source.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError("evaluation config changed while its snapshot was read")
    if len(encoded) != before.st_size:
        raise RuntimeError("evaluation config byte count changed during snapshot")
    return load_evaluation_config_bytes(encoded), encoded


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load a strict standalone evaluation YAML file."""

    return load_evaluation_config_snapshot(path)[0]


__all__ = [
    "CHECKPOINT_WEIGHT_VARIANTS",
    "EVALUATION_DATA_SOURCES",
    "EVALUATION_PURPOSES",
    "EVALUATION_SPLITS",
    "CheckpointSubjectConfig",
    "CheckpointWeightVariant",
    "EvaluationConfig",
    "EvaluationDataConfig",
    "EvaluationDataSource",
    "EvaluationExtensionsConfig",
    "EvaluationProtocol",
    "EvaluationPurpose",
    "EvaluationSplit",
    "EvaluationSubjectConfig",
    "PredictionArtifactSubjectConfig",
    "evaluation_config_to_dict",
    "load_evaluation_config",
    "load_evaluation_config_bytes",
    "load_evaluation_config_dict",
    "load_evaluation_config_snapshot",
]
