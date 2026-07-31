"""Composition-time verification for diagnostic metric sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, cast

from stochaflow.metrics import MetricDataRole, MetricSource
from stochaflow.training.diagnostics.contracts import (
    BoundTrainingDiagnostic,
    DiagnosticSourceProvider,
    DiagnosticSourceRequest,
    TrainingDiagnostic,
    VerifiedMetricSource,
)

_PROTOCOL_PROVENANCE_FIELDS = {
    "schema_version",
    "data_config",
    "data_artifacts",
    "extension_plugins",
}
_DATA_CONFIG_FIELDS = {"name", "params"}
_DATA_ARTIFACT_FIELDS = {"schema_version", "bindings"}
_DATA_ARTIFACT_BINDING_FIELDS = {"id", "identity"}
_DATA_ARTIFACT_IDENTITY_FIELDS = {
    "schema_version",
    "kind",
    "artifact_type",
    "source_name",
    "source_digest",
    "materializer_name",
    "materialization_digest",
    "content_digest",
    "artifact_digest",
    "manifest_sha256",
}
_EXTENSION_PROVENANCE_FIELDS = {
    "name",
    "distribution",
    "version",
    "target",
}
_DATA_ARTIFACT_DIGEST_FIELDS = {
    "source_digest",
    "materialization_digest",
    "content_digest",
    "artifact_digest",
    "manifest_sha256",
}


def _json_value(value: object, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"{path} must contain only JSON-compatible protocol values, "
        f"got {type(value).__name__}"
    )


def _strict_fields(
    value: Mapping[object, object],
    *,
    expected: set[str],
    path: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} field names must be strings")
    names = cast(set[str], set(value))
    if names != expected:
        missing = sorted(expected - names)
        unknown = sorted(names - expected)
        raise ValueError(
            f"{path} has invalid fields: missing={missing or '<none>'}, "
            f"unknown={unknown or '<none>'}"
        )


def _validate_data_artifact_bindings(value: Mapping[object, object]) -> None:
    _strict_fields(
        value,
        expected=_DATA_ARTIFACT_FIELDS,
        path="diagnostic protocol_provenance.data_artifacts",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
    ):
        raise ValueError(
            "diagnostic protocol_provenance.data_artifacts.schema_version "
            "must be 2"
        )
    bindings = value["bindings"]
    if not isinstance(bindings, list):
        raise TypeError(
            "diagnostic protocol_provenance.data_artifacts.bindings must "
            "be a list"
        )
    binding_ids: list[str] = []
    for index, binding in enumerate(bindings):
        path = (
            "diagnostic protocol_provenance.data_artifacts.bindings"
            f"[{index}]"
        )
        if not isinstance(binding, Mapping):
            raise TypeError(f"{path} must be a mapping")
        _strict_fields(
            cast(Mapping[object, object], binding),
            expected=_DATA_ARTIFACT_BINDING_FIELDS,
            path=path,
        )
        binding_id = binding["id"]
        if not isinstance(binding_id, str) or not binding_id:
            raise ValueError(f"{path}.id must be a non-empty string")
        binding_ids.append(binding_id)
        identity = binding["identity"]
        if not isinstance(identity, Mapping):
            raise TypeError(f"{path}.identity must be a mapping")
        _strict_fields(
            cast(Mapping[object, object], identity),
            expected=_DATA_ARTIFACT_IDENTITY_FIELDS,
            path=f"{path}.identity",
        )
        if (
            type(identity["schema_version"]) is not int
            or identity["schema_version"] != 2
        ):
            raise ValueError(f"{path}.identity.schema_version must be 2")
        if identity["kind"] not in {"managed", "referenced"}:
            raise ValueError(
                f"{path}.identity.kind must be managed or referenced"
            )
        for name in ("artifact_type", "source_name", "materializer_name"):
            item = identity[name]
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"{path}.identity.{name} must be a non-empty string"
                )
        for name in _DATA_ARTIFACT_DIGEST_FIELDS:
            digest = identity[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or digest != digest.lower()
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"{path}.identity.{name} must be a lowercase SHA-256 digest"
                )
    if binding_ids != sorted(binding_ids):
        raise ValueError(
            "diagnostic protocol_provenance.data_artifacts.bindings must be "
            "sorted by id"
        )
    if len(binding_ids) != len(set(binding_ids)):
        raise ValueError(
            "diagnostic protocol_provenance.data_artifacts.bindings ids must "
            "be unique"
        )


def _validated_protocol_provenance(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(cast(object, value), Mapping):
        raise TypeError(
            "diagnostic protocol_provenance must be a composition mapping"
        )
    raw = cast(Mapping[object, object], value)
    _strict_fields(
        raw,
        expected=_PROTOCOL_PROVENANCE_FIELDS,
        path="diagnostic protocol_provenance",
    )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError(
            "diagnostic protocol_provenance.schema_version must be 1"
        )

    data_config = raw["data_config"]
    if not isinstance(data_config, Mapping):
        raise TypeError(
            "diagnostic protocol_provenance.data_config must be a mapping"
        )
    _strict_fields(
        cast(Mapping[object, object], data_config),
        expected=_DATA_CONFIG_FIELDS,
        path="diagnostic protocol_provenance.data_config",
    )
    data_name = data_config["name"]
    if not isinstance(data_name, str) or not data_name:
        raise ValueError(
            "diagnostic protocol_provenance.data_config.name must be a "
            "non-empty string"
        )
    if not isinstance(data_config["params"], Mapping):
        raise TypeError(
            "diagnostic protocol_provenance.data_config.params must be a mapping"
        )

    data_artifacts = raw["data_artifacts"]
    if data_artifacts is not None:
        if not isinstance(data_artifacts, Mapping):
            raise TypeError(
                "diagnostic protocol_provenance.data_artifacts must be a "
                "mapping or null"
            )
        _validate_data_artifact_bindings(
            cast(Mapping[object, object], data_artifacts)
        )

    extension_plugins = raw["extension_plugins"]
    if not isinstance(extension_plugins, (list, tuple)):
        raise TypeError(
            "diagnostic protocol_provenance.extension_plugins must be a list"
        )
    normalized_plugins: list[dict[str, str]] = []
    seen_plugin_names: set[str] = set()
    for index, item in enumerate(extension_plugins):
        path = f"diagnostic protocol_provenance.extension_plugins[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{path} must be a mapping")
        _strict_fields(
            cast(Mapping[object, object], item),
            expected=_EXTENSION_PROVENANCE_FIELDS,
            path=path,
        )
        normalized_item: dict[str, str] = {}
        for name in _EXTENSION_PROVENANCE_FIELDS:
            field_value = item[name]
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(f"{path}.{name} must be a non-empty string")
            normalized_item[name] = field_value
        if normalized_item["name"] in seen_plugin_names:
            raise ValueError(
                "diagnostic protocol_provenance.extension_plugins contains "
                f"duplicate name {normalized_item['name']!r}"
            )
        seen_plugin_names.add(normalized_item["name"])
        normalized_plugins.append(normalized_item)

    normalized = {
        "schema_version": 1,
        "data_config": _json_value(
            data_config,
            path="diagnostic protocol_provenance.data_config",
        ),
        "data_artifacts": _json_value(
            data_artifacts,
            path="diagnostic protocol_provenance.data_artifacts",
        ),
        "extension_plugins": sorted(
            normalized_plugins,
            key=lambda item: (
                item["name"],
                item["distribution"],
                item["version"],
                item["target"],
            ),
        ),
    }
    return cast(dict[str, Any], normalized)


def _validated_data_iterables(
    value: Mapping[MetricDataRole, Iterable[Any]] | None,
) -> dict[MetricDataRole, Iterable[Any]]:
    if value is None:
        return {}
    if not isinstance(cast(object, value), Mapping):
        raise TypeError("diagnostic data_iterables must be a mapping or null")
    normalized: dict[MetricDataRole, Iterable[Any]] = {}
    for role, source_iterable in value.items():
        if role not in {"train", "validation"}:
            raise ValueError(
                "diagnostic data_iterables supports only train and validation"
            )
        if not isinstance(cast(object, source_iterable), Iterable):
            raise TypeError(
                f"diagnostic data_iterables[{role!r}] must be iterable"
            )
        if isinstance(cast(object, source_iterable), Iterator):
            raise TypeError(
                f"diagnostic data_iterables[{role!r}] must be re-iterable"
            )
        normalized[role] = source_iterable
    return normalized


def _protocol_digest(
    *,
    diagnostic_id: str,
    request: DiagnosticSourceRequest,
    provenance: Mapping[str, Any],
) -> str:
    descriptor = {
        "schema_version": 1,
        "diagnostic_id": diagnostic_id,
        "source_id": request.id,
        "data_role": request.data_role,
        "protocol": _json_value(
            request.protocol,
            path=f"diagnostic {diagnostic_id!r} source {request.id!r} protocol",
        ),
        "provenance": _json_value(
            provenance,
            path="diagnostic protocol provenance",
        ),
    }
    encoded = json.dumps(
        descriptor,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_training_diagnostic(
    diagnostic_id: str,
    diagnostic: TrainingDiagnostic,
    *,
    protocol_provenance: Mapping[str, Any] | None = None,
    data_iterables: Mapping[MetricDataRole, Iterable[Any]] | None = None,
) -> BoundTrainingDiagnostic:
    """Bind one diagnostic's requested sources to verified metadata."""

    requests: tuple[DiagnosticSourceRequest, ...] = ()
    diagnostic_value = cast(object, diagnostic)
    if isinstance(diagnostic_value, DiagnosticSourceProvider):
        requests_value = cast(object, diagnostic_value.metric_source_requests)
        if not isinstance(requests_value, tuple):
            raise TypeError(
                "DiagnosticSourceProvider.metric_source_requests must be a tuple"
            )
        requests = cast(tuple[DiagnosticSourceRequest, ...], requests_value)
    provenance = (
        _validated_protocol_provenance(protocol_provenance)
        if requests
        else {}
    )
    role_iterables = _validated_data_iterables(data_iterables)
    sources: dict[str, VerifiedMetricSource] = {}
    source_iterables: dict[str, Iterable[Any]] = {}
    for index, request in enumerate(requests):
        if not isinstance(cast(object, request), DiagnosticSourceRequest):
            raise TypeError(
                "DiagnosticSourceProvider.metric_source_requests"
                f"[{index}] must be a DiagnosticSourceRequest"
            )
        if request.id in sources:
            raise ValueError(
                f"diagnostic {diagnostic_id!r} requested duplicate source "
                f"id {request.id!r}"
            )
        if request.data_role == "test":
            raise ValueError(
                f"diagnostic {diagnostic_id!r} source {request.id!r} cannot "
                "bind data_role='test': TrainingDiagnostic FitStartEvent does "
                "not inject a test iterable"
            )
        if request.data_role in {"train", "validation"}:
            source_iterable = role_iterables.get(request.data_role)
            if source_iterable is None:
                raise ValueError(
                    f"diagnostic {diagnostic_id!r} source {request.id!r} "
                    f"requires the actual {request.data_role} fit iterable"
                )
            source_iterables[request.id] = source_iterable
        digest = _protocol_digest(
            diagnostic_id=diagnostic_id,
            request=request,
            provenance=provenance,
        )
        metadata = MetricSource(
            origin="diagnostic",
            data_role=request.data_role,
            protocol_id=f"sha256:{digest}",
            selection_eligible=request.data_role == "validation",
        )
        sources[request.id] = VerifiedMetricSource(
            id=request.id,
            metadata=metadata,
            protocol_digest=digest,
        )
    return BoundTrainingDiagnostic(
        id=diagnostic_id,
        diagnostic=diagnostic,
        sources=sources,
        source_iterables=source_iterables,
    )


def bind_training_diagnostics(
    diagnostic_ids: Sequence[str],
    diagnostics: Sequence[TrainingDiagnostic],
    *,
    protocol_provenance: Mapping[str, Any] | None = None,
    data_iterables: Mapping[MetricDataRole, Iterable[Any]] | None = None,
) -> list[BoundTrainingDiagnostic]:
    """Bind diagnostics in declaration order and reject ambiguous identities."""

    if len(diagnostic_ids) != len(diagnostics):
        raise ValueError("diagnostic ids and instances must have the same length")
    bindings = [
        bind_training_diagnostic(
            diagnostic_id,
            diagnostic,
            protocol_provenance=protocol_provenance,
            data_iterables=data_iterables,
        )
        for diagnostic_id, diagnostic in zip(
            diagnostic_ids,
            diagnostics,
            strict=True,
        )
    ]
    source_provider_ids = [
        binding.id
        for binding in bindings
        if binding.sources
    ]
    if len(source_provider_ids) != len(set(source_provider_ids)):
        raise ValueError(
            "diagnostics that emit epoch metrics require unique configured names"
        )
    return bindings


__all__ = [
    "bind_training_diagnostic",
    "bind_training_diagnostics",
]
