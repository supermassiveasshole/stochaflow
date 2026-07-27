"""Strict private configuration for AFHQ-v2 quality evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

SCHEMA_VERSION = 1
SAMPLING_RECIPE_NAME = "class_conditional_denoising"
AFHQ_V2_CLASS_MAPPING = {"cat": 0, "dog": 1, "wild": 2}


@dataclass(frozen=True, slots=True)
class AFHQV2MetricSpec:
    """One strict ReferenceMetricProvider declaration."""

    name: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AFHQV2EvaluationProtocol:
    """Frozen class and metric protocol for one evaluation invocation."""

    split: str
    class_mapping: dict[str, int]
    real_per_class: int
    fake_per_class: int
    metric_batch_size: int
    metric_seed: int
    metrics: tuple[AFHQV2MetricSpec, ...]


@dataclass(frozen=True, slots=True)
class AFHQV2EvaluationDocument:
    """Validated evaluation protocol plus a core-compatible sample request."""

    protocol: AFHQV2EvaluationProtocol
    sample_request: dict[str, Any]
    source_path: Path
    source_sha256: str


def _strict_mapping(
    value: object,
    *,
    path: str,
    allowed: set[str],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{path} field names must be strings")
    names = cast(set[str], set(raw))
    missing = sorted(required - names)
    unknown = sorted(names - allowed)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")
    return cast(dict[str, Any], value)


def _positive_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _metric_specs(value: object) -> tuple[AFHQV2MetricSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation.metrics must be a non-empty list")
    specs: list[AFHQV2MetricSpec] = []
    for index, item in enumerate(value):
        path = f"evaluation.metrics[{index}]"
        raw = _strict_mapping(
            item,
            path=path,
            allowed={"name", "params"},
            required={"name", "params"},
        )
        params = raw["params"]
        if not isinstance(params, dict) or any(
            not isinstance(key, str) for key in params
        ):
            raise TypeError(f"{path}.params must be a string-keyed mapping")
        specs.append(
            AFHQV2MetricSpec(
                name=_non_empty_string(raw["name"], path=f"{path}.name"),
                params=cast(dict[str, Any], dict(params)),
            )
        )
    names = tuple(spec.name for spec in specs)
    if len(names) != len(set(names)):
        raise ValueError("evaluation metric names must be unique")
    return tuple(specs)


def _class_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("evaluation.class_mapping must be a non-empty mapping")
    mapping: dict[str, int] = {}
    for name_value, label_value in value.items():
        name = _non_empty_string(
            name_value,
            path="evaluation.class_mapping class name",
        )
        label = _nonnegative_integer(
            label_value,
            path=f"evaluation.class_mapping.{name}",
        )
        mapping[name] = label
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("evaluation.class_mapping labels must be unique")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(
            "evaluation.class_mapping labels must be contiguous from zero"
        )
    return dict(sorted(mapping.items(), key=lambda item: item[1]))


def _evaluation_protocol(value: object) -> AFHQV2EvaluationProtocol:
    raw = _strict_mapping(
        value,
        path="evaluation",
        allowed={
            "schema_version",
            "split",
            "class_mapping",
            "real_per_class",
            "fake_per_class",
            "metric_batch_size",
            "metric_seed",
            "metrics",
        },
        required={
            "schema_version",
            "split",
            "class_mapping",
            "real_per_class",
            "fake_per_class",
            "metric_batch_size",
            "metric_seed",
            "metrics",
        },
    )
    if (
        isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("evaluation.schema_version must be 1")
    split = _non_empty_string(raw["split"], path="evaluation.split")
    if split != "test":
        raise ValueError("AFHQ-v2 quality evaluation requires split: test")
    return AFHQV2EvaluationProtocol(
        split=split,
        class_mapping=_class_mapping(raw["class_mapping"]),
        real_per_class=_positive_integer(
            raw["real_per_class"],
            path="evaluation.real_per_class",
        ),
        fake_per_class=_positive_integer(
            raw["fake_per_class"],
            path="evaluation.fake_per_class",
        ),
        metric_batch_size=_positive_integer(
            raw["metric_batch_size"],
            path="evaluation.metric_batch_size",
        ),
        metric_seed=_nonnegative_integer(
            raw["metric_seed"],
            path="evaluation.metric_seed",
        ),
        metrics=_metric_specs(raw["metrics"]),
    )


def _sampling_protocol(
    request: dict[str, Any],
    protocol: AFHQV2EvaluationProtocol,
) -> None:
    sampling_value = request.get("sampling")
    if isinstance(sampling_value, dict) and "builder" in sampling_value:
        raise ValueError(
            "legacy sampling.builder is unsupported; use sampling.sampler "
            "and sampling.options"
        )
    sampling = _strict_mapping(
        sampling_value,
        path="sampling",
        allowed={
            "sampler",
            "options",
            "shape",
            "num_samples",
            "batch_size",
            "seed",
            "writers",
        },
        required={
            "sampler",
            "options",
            "shape",
            "num_samples",
            "batch_size",
            "seed",
            "writers",
        },
    )
    if sampling["shape"] != [3, 128, 128]:
        raise ValueError("AFHQ-v2 evaluation sampling.shape must be [3, 128, 128]")
    _nonnegative_integer(sampling["seed"], path="sampling.seed")
    expected_total = protocol.fake_per_class * len(protocol.class_mapping)
    if _positive_integer(
        sampling["num_samples"],
        path="sampling.num_samples",
    ) != expected_total:
        raise ValueError(
            "sampling.num_samples must equal fake_per_class times class count"
        )
    _positive_integer(sampling["batch_size"], path="sampling.batch_size")
    options = _strict_mapping(
        sampling["options"],
        path="sampling.options",
        allowed={
            "weights",
            "clip_denoised",
            "guidance_scale",
            "conditions",
            "trajectory",
        },
        required={
            "weights",
            "clip_denoised",
            "guidance_scale",
            "conditions",
            "trajectory",
        },
    )
    if options["weights"] not in {"raw", "ema"}:
        raise ValueError(
            "evaluation sampling weights must explicitly select raw or ema"
        )
    guidance = options["guidance_scale"]
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(float(guidance))
        or float(guidance) < 0.0
    ):
        raise ValueError(
            "evaluation guidance_scale must be finite and non-negative"
        )
    conditions = options["conditions"]
    expected_conditions = [
        {"class_label": label, "count": protocol.fake_per_class}
        for label in protocol.class_mapping.values()
    ]
    if conditions != expected_conditions:
        raise ValueError(
            "evaluation sampling conditions must be balanced and ordered by "
            "class label"
        )
    sampler = _strict_mapping(
        sampling["sampler"],
        path="sampling.sampler",
        allowed={"name", "params"},
        required={"name", "params"},
    )
    _non_empty_string(sampler["name"], path="sampling.sampler.name")
    if not isinstance(sampler["params"], dict):
        raise TypeError("sampling.sampler.params must be a mapping")
    trajectory = _strict_mapping(
        options["trajectory"],
        path="sampling.options.trajectory",
        allowed={"enabled", "every_steps"},
        required={"enabled", "every_steps"},
    )
    if trajectory["enabled"] is not False:
        raise ValueError(
            "formal AFHQ-v2 evaluation must disable full-sample trajectory"
        )
    _positive_integer(
        trajectory["every_steps"],
        path="sampling.options.trajectory.every_steps",
    )
    writers = sampling["writers"]
    if writers != [{"name": "tensor", "params": {}}]:
        raise ValueError(
            "AFHQ-v2 evaluation requires exactly one parameter-free tensor writer"
        )


def load_evaluation_document(path: str | Path) -> AFHQV2EvaluationDocument:
    """Load and strictly validate one example-private evaluation document."""

    source = Path(path)
    encoded = source.read_bytes()
    raw = _strict_mapping(
        yaml.safe_load(encoded),
        path="evaluation config",
        allowed={"extensions", "sampling", "evaluation"},
        required={"extensions", "sampling", "evaluation"},
    )
    extensions = _strict_mapping(
        raw["extensions"],
        path="extensions",
        allowed={"plugins"},
        required={"plugins"},
    )
    if extensions["plugins"] != ["stochaflow-afhq-v2"]:
        raise ValueError(
            "AFHQ-v2 evaluation must select only stochaflow-afhq-v2"
        )
    protocol = _evaluation_protocol(raw["evaluation"])
    if protocol.class_mapping != AFHQ_V2_CLASS_MAPPING:
        raise ValueError(
            "AFHQ-v2 evaluation class_mapping must be "
            "{cat: 0, dog: 1, wild: 2}"
        )
    request = {
        "extensions": cast(dict[str, Any], raw["extensions"]),
        "sampling": cast(dict[str, Any], raw["sampling"]),
    }
    _sampling_protocol(request, protocol)
    return AFHQV2EvaluationDocument(
        protocol=protocol,
        sample_request=request,
        source_path=source.resolve(),
        source_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def sample_request_bytes(document: AFHQV2EvaluationDocument) -> bytes:
    """Encode the exact core-compatible sample request."""

    return yaml.safe_dump(
        document.sample_request,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def sampling_parameters(document: AFHQV2EvaluationDocument) -> dict[str, Any]:
    """Return the validated class-conditional options and sampler declaration."""

    sampling = document.sample_request["sampling"]
    options = cast(
        dict[str, Any],
        sampling["options"],
    )
    return {**options, "sampler": cast(dict[str, Any], sampling["sampler"])}


__all__ = [
    "AFHQ_V2_CLASS_MAPPING",
    "SAMPLING_RECIPE_NAME",
    "SCHEMA_VERSION",
    "AFHQV2EvaluationDocument",
    "AFHQV2EvaluationProtocol",
    "AFHQV2MetricSpec",
    "load_evaluation_document",
    "sample_request_bytes",
    "sampling_parameters",
]
