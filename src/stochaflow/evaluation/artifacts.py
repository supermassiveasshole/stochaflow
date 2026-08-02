"""Portable evaluation serialization and atomic bundle publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, cast

import yaml

from stochaflow.data.artifact_io import (
    canonical_directory,
    publish_cache_directory,
    remove_cache_directory,
)
from stochaflow.evaluation.contracts import EvaluationResult


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode one JSON-shaped value for stable identity hashing."""

    return json.dumps(
        _json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return a stable lowercase digest for one JSON-shaped declaration."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def evaluation_result_to_dict(result: EvaluationResult) -> dict[str, Any]:
    """Serialize immutable result facts without local absolute paths."""

    result_value = cast(object, result)
    if not isinstance(result_value, EvaluationResult):
        raise TypeError("evaluation result must be EvaluationResult")
    return {
        "schema_version": result.schema_version,
        "evaluation_id": result.evaluation_id,
        "protocol_id": result.protocol_id,
        "protocol_digest": result.protocol_digest,
        "status": result.status,
        "subject": _json_value(result.subject),
        "data": _json_value(result.data),
        "metrics": dict(result.metrics),
        "measurements": dict(result.measurements),
        "artifacts": _json_value(result.artifacts),
        "completeness": _json_value(result.completeness),
        "provenance": _json_value(result.provenance),
    }


@dataclass(frozen=True, slots=True)
class PublishedEvaluationBundle:
    """Paths atomically exposed by one successful evaluation publication."""

    output_dir: Path
    resolved_config_path: Path
    result_path: Path
    manifest_path: Path
    result_sha256: str
    artifacts: Mapping[str, Path]


def _write_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def publish_evaluation_bundle(
    output_dir: str | Path,
    *,
    result: EvaluationResult,
    resolved_config: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    prepared_artifacts: Mapping[str, Path] | None = None,
) -> PublishedEvaluationBundle:
    """Stage one complete result bundle and atomically publish it once."""

    declared_destination = Path(output_dir)
    if not declared_destination.name:
        raise ValueError("evaluation output directory must have a final name")
    declared_destination.parent.mkdir(parents=True, exist_ok=True)
    parent = canonical_directory(
        declared_destination.parent.resolve(),
        label="evaluation output parent",
    )
    destination = parent / declared_destination.name
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            f"evaluation output directory already exists: {destination}"
        )
    reserved = {
        "schema_version",
        "kind",
        "status",
        "evaluation_id",
        "protocol_id",
        "protocol_digest",
        "resolved_config",
        "result",
    }
    collisions = sorted(reserved.intersection(manifest_metadata))
    if collisions:
        raise ValueError(
            "evaluation manifest metadata contains reserved field(s): "
            + ", ".join(collisions)
        )
    artifact_sources = _validate_prepared_artifacts(prepared_artifacts)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.evaluation-",
            dir=parent,
        )
    ).resolve()
    try:
        result_document = evaluation_result_to_dict(result)
        result_bytes = json.dumps(
            result_document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        result_digest = hashlib.sha256(result_bytes).hexdigest()
        config_document = _json_value(resolved_config)
        config_bytes = yaml.safe_dump(
            config_document,
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "kind": "evaluation",
            "status": result.status,
            "evaluation_id": result.evaluation_id,
            "protocol_id": result.protocol_id,
            "protocol_digest": result.protocol_digest,
            **_json_value(manifest_metadata),
            "resolved_config": "resolved_evaluation.yaml",
            "result": {
                "path": "result.json",
                "sha256": result_digest,
            },
        }
        manifest_bytes = yaml.safe_dump(
            manifest,
            allow_unicode=True,
            sort_keys=False,
        ).encode("utf-8")

        staged_config = staging / "resolved_evaluation.yaml"
        staged_result = staging / "result.json"
        staged_manifest = staging / "evaluation_manifest.yaml"
        _write_bytes(staged_config, config_bytes)
        _write_bytes(staged_result, result_bytes)

        staged_artifacts: dict[str, str] = {}
        for relative_name, source in artifact_sources.items():
            target = staging / Path(*PurePosixPath(relative_name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            staged_artifacts[relative_name] = relative_name
        _write_bytes(staged_manifest, manifest_bytes)
        published = publish_cache_directory(
            parent,
            staging,
            destination,
            label="evaluation bundle publication",
        )
        resolved_config_path = published / staged_config.name
        result_path = published / staged_result.name
        manifest_path = published / staged_manifest.name
        published_artifacts = {
            relative_name: published / Path(*PurePosixPath(relative).parts)
            for relative_name, relative in staged_artifacts.items()
        }
        return PublishedEvaluationBundle(
            output_dir=published,
            resolved_config_path=resolved_config_path,
            result_path=result_path,
            manifest_path=manifest_path,
            result_sha256=result_digest,
            artifacts=MappingProxyType(published_artifacts),
        )
    except BaseException:
        try:
            staging.lstat()
        except FileNotFoundError:
            pass
        else:
            remove_cache_directory(
                parent,
                staging,
                label="evaluation bundle staging cleanup",
            )
        raise


def _validate_prepared_artifacts(
    value: Mapping[str, Path] | None,
) -> dict[str, Path]:
    if value is None:
        return {}
    if not isinstance(cast(object, value), Mapping):
        raise TypeError("prepared evaluation artifacts must be a mapping")
    reserved = {
        "resolved_evaluation.yaml",
        "result.json",
        "evaluation_manifest.yaml",
    }
    normalized: dict[str, Path] = {}
    for declared_name, declared_source in value.items():
        if type(declared_name) is not str or not declared_name:
            raise ValueError(
                "prepared evaluation artifact names must be non-empty strings"
            )
        posix = PurePosixPath(declared_name)
        windows = PureWindowsPath(declared_name)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or ".." in posix.parts
            or ".." in windows.parts
            or not posix.parts
            or declared_name != posix.as_posix()
            or "\\" in declared_name
        ):
            raise ValueError(
                "prepared evaluation artifact names must be portable relative paths"
            )
        if declared_name in reserved or posix.parts[0].startswith(".staging-"):
            raise ValueError(
                f"prepared evaluation artifact name {declared_name!r} is reserved"
            )
        source_value = cast(object, declared_source)
        if not isinstance(source_value, Path):
            raise TypeError(
                f"prepared evaluation artifact {declared_name!r} must be a Path"
            )
        source = declared_source.resolve(strict=True)
        normalized[declared_name] = source
    names = tuple(normalized)
    for index, name in enumerate(names):
        prefix = PurePosixPath(name)
        for other in names[index + 1 :]:
            other_path = PurePosixPath(other)
            if prefix in other_path.parents or other_path in prefix.parents:
                raise ValueError(
                    "prepared evaluation artifact names must not overlap"
                )
    return normalized


__all__ = [
    "PublishedEvaluationBundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "evaluation_result_to_dict",
    "publish_evaluation_bundle",
]
