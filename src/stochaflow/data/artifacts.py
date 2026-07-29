"""Materialized-data contracts and stable checkpoint identities."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from stochaflow.data.artifact_io import (
    MAX_ARTIFACT_VERIFICATION_WORKERS,
    canonical_directory,
    lexical_absolute_path,
    read_regular_file,
)
from stochaflow.utils.config import ConfigError

_IDENTITY_FIELDS = frozenset(
    {
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
)
_BINDING_FIELDS = frozenset({"id", "identity"})
_COLLECTION_FIELDS = frozenset({"schema_version", "bindings"})
_SHA256_LENGTH = 64

type ArtifactVerificationPhase = Literal["validate"]


@dataclass(frozen=True, slots=True)
class ArtifactVerificationEvent:
    """Progress for one complete artifact-content verification pass."""

    artifact_type: str
    source_name: str
    materializer_name: str
    phase: ArtifactVerificationPhase
    completed: int
    total: int

    def __post_init__(self) -> None:
        for name in ("artifact_type", "source_name", "materializer_name"):
            _non_empty_string(
                getattr(self, name),
                path=f"artifact verification event.{name}",
            )
        if self.phase != "validate":
            raise ValueError(
                "artifact verification event.phase must be validate"
            )
        for name in ("completed", "total"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"artifact verification event.{name} must be non-negative"
                )
        if self.completed > self.total:
            raise ValueError(
                "artifact verification event.completed must not exceed total"
            )


class ArtifactVerificationObserver(Protocol):
    """Receive ordered progress events for full artifact verification."""

    def __call__(self, event: ArtifactVerificationEvent, /) -> None:
        """Observe one artifact verification progress update."""


def _non_empty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _sha256(value: object, *, path: str) -> str:
    digest = _non_empty_string(value, path=path)
    if (
        len(digest) != _SHA256_LENGTH
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 hex digest")
    return digest


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{path} field names must be strings")
    names = cast(set[str], set(raw))
    missing = sorted(fields - names)
    unknown = sorted(names - fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class DataArtifactIdentity:
    """Location-independent schema-v2 identity for one data artifact."""

    kind: Literal["managed", "referenced"]
    artifact_type: str
    source_name: str
    source_digest: str
    materializer_name: str
    materialization_digest: str
    content_digest: str
    artifact_digest: str
    manifest_sha256: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("data artifact identity schema_version must be 2")
        if self.kind not in {"managed", "referenced"}:
            raise ValueError(
                "data artifact identity.kind must be managed or referenced"
            )
        for name in ("artifact_type", "source_name", "materializer_name"):
            _non_empty_string(
                getattr(self, name),
                path=f"data artifact identity.{name}",
            )
        for name in (
            "source_digest",
            "materialization_digest",
            "content_digest",
            "artifact_digest",
            "manifest_sha256",
        ):
            _sha256(
                getattr(self, name),
                path=f"data artifact identity.{name}",
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this identity using its strict stable schema."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "artifact_type": self.artifact_type,
            "source_name": self.source_name,
            "source_digest": self.source_digest,
            "materializer_name": self.materializer_name,
            "materialization_digest": self.materialization_digest,
            "content_digest": self.content_digest,
            "artifact_digest": self.artifact_digest,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "data artifact identity",
    ) -> DataArtifactIdentity:
        """Parse a strict schema-v2 identity."""

        raw = _strict_mapping(value, fields=_IDENTITY_FIELDS, path=path)
        return cls(
            schema_version=raw["schema_version"],
            kind=raw["kind"],
            artifact_type=raw["artifact_type"],
            source_name=raw["source_name"],
            source_digest=raw["source_digest"],
            materializer_name=raw["materializer_name"],
            materialization_digest=raw["materialization_digest"],
            content_digest=raw["content_digest"],
            artifact_digest=raw["artifact_digest"],
            manifest_sha256=raw["manifest_sha256"],
        )


@dataclass(frozen=True, slots=True)
class DataArtifact[ArtifactPayloadT]:
    """Verified runtime handle for managed or referenced content."""

    root: Path
    identity: DataArtifactIdentity
    payload: ArtifactPayloadT

    @property
    def kind(self) -> Literal["managed", "referenced"]:
        """Return the artifact ownership strategy."""

        return self.identity.kind

    @property
    def manifest_path(self) -> Path:
        """Return the framework-owned manifest path."""

        return self.root / "manifest.json"

    def __post_init__(self) -> None:
        identity = cast(object, self.identity)
        if not isinstance(identity, DataArtifactIdentity):
            raise TypeError("data artifact identity must be DataArtifactIdentity")
        root = canonical_directory(Path(self.root), label="data artifact root")
        encoded, _ = read_regular_file(
            root,
            "manifest.json",
            label="data artifact manifest",
        )
        if hashlib.sha256(encoded).hexdigest() != self.identity.manifest_sha256:
            raise ValueError(
                "data artifact manifest SHA-256 does not match its identity"
            )
        object.__setattr__(self, "root", root)


class DataSource[ArtifactPayloadT](ABC):
    """Artifact-producing source that never constructs runtime data loaders."""

    @abstractmethod
    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ArtifactPayloadT]:
        """Return one verified artifact for the requested policy."""


@dataclass(frozen=True, slots=True)
class DataArtifactBinding:
    """Stable role-to-artifact binding persisted with a training run."""

    id: str
    identity: DataArtifactIdentity

    def __post_init__(self) -> None:
        _non_empty_string(self.id, path="data artifact binding.id")
        if not isinstance(cast(object, self.identity), DataArtifactIdentity):
            raise TypeError(
                "data artifact binding.identity must be DataArtifactIdentity"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this binding without runtime cache paths."""

        return {"id": self.id, "identity": self.identity.to_dict()}

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "data artifact binding",
    ) -> DataArtifactBinding:
        """Parse one strict serialized binding."""

        raw = _strict_mapping(value, fields=_BINDING_FIELDS, path=path)
        return cls(
            id=_non_empty_string(raw["id"], path=f"{path}.id"),
            identity=DataArtifactIdentity.from_dict(
                raw["identity"],
                path=f"{path}.identity",
            ),
        )


@dataclass(frozen=True, slots=True)
class DataArtifactBindings:
    """Canonical schema-v2 collection of artifact bindings."""

    bindings: tuple[DataArtifactBinding, ...] = ()

    def __post_init__(self) -> None:
        bindings_value = cast(object, self.bindings)
        if not isinstance(bindings_value, tuple):
            raise TypeError("data artifact bindings must be a tuple")
        if any(
            not isinstance(binding, DataArtifactBinding)
            for binding in cast(tuple[object, ...], bindings_value)
        ):
            raise TypeError(
                "data artifact bindings must contain DataArtifactBinding values"
            )
        normalized = tuple(sorted(self.bindings, key=lambda binding: binding.id))
        ids = tuple(binding.id for binding in normalized)
        if len(ids) != len(set(ids)):
            raise ValueError("data artifact binding ids must be unique")
        object.__setattr__(self, "bindings", normalized)

    @property
    def ids(self) -> tuple[str, ...]:
        """Return binding identifiers in canonical order."""

        return tuple(binding.id for binding in self.bindings)

    def __len__(self) -> int:
        return len(self.bindings)

    def __iter__(self) -> Iterator[DataArtifactBinding]:
        return iter(self.bindings)

    def identity_for(self, binding_id: str) -> DataArtifactIdentity:
        """Return the identity for one required binding."""

        for binding in self.bindings:
            if binding.id == binding_id:
                return binding.identity
        raise KeyError(f"missing data artifact binding '{binding_id}'")

    def assert_ids(self, expected_ids: tuple[str, ...]) -> None:
        """Require an exact set of binding identifiers."""

        normalized = tuple(sorted(expected_ids))
        if len(normalized) != len(set(normalized)):
            raise ValueError("expected data artifact binding ids must be unique")
        if self.ids != normalized:
            raise ValueError(
                "strict resume data artifact binding ids do not match; "
                f"expected {normalized}, found {self.ids}"
            )

    def assert_exact(self, actual: DataArtifactBindings) -> None:
        """Require the exact identities selected by a strict resume."""

        if self != actual:
            raise ValueError(
                "strict resume data artifacts do not match the selected checkpoint"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this collection using canonical binding order."""

        return {
            "schema_version": 2,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "metadata.data_artifacts",
    ) -> DataArtifactBindings:
        """Parse a strict, canonically ordered schema-v2 collection."""

        raw = _strict_mapping(value, fields=_COLLECTION_FIELDS, path=path)
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
            raise ValueError(f"{path}.schema_version must be 2")
        serialized = raw["bindings"]
        if not isinstance(serialized, list):
            raise TypeError(f"{path}.bindings must be a list")
        bindings = tuple(
            DataArtifactBinding.from_dict(
                item,
                path=f"{path}.bindings[{index}]",
            )
            for index, item in enumerate(serialized)
        )
        parsed = cls(bindings)
        if tuple(binding.id for binding in bindings) != parsed.ids:
            raise ValueError(f"{path}.bindings must be sorted by id")
        return parsed


@dataclass(frozen=True, slots=True)
class DataSourceContext:
    """Materialization policy supplied by an artifact-aware DataBuilder."""

    cache_root: Path
    policy: Literal["require", "ensure"]
    verification: Literal["manifest", "full"]
    expected_identity: DataArtifactIdentity | None = None
    verification_observer: ArtifactVerificationObserver | None = None
    verification_workers: int | None = None

    def __post_init__(self) -> None:
        policy = cast(object, self.policy)
        if not isinstance(policy, str) or policy not in {"require", "ensure"}:
            raise ConfigError(
                "data source materialization.policy must be require or ensure"
            )
        verification = cast(object, self.verification)
        if not isinstance(verification, str) or verification not in {
            "manifest",
            "full",
        }:
            raise ConfigError(
                "data source materialization.verification must be manifest or full"
            )
        expected_identity = cast(object, self.expected_identity)
        if expected_identity is not None and not isinstance(
            expected_identity, DataArtifactIdentity
        ):
            raise TypeError(
                "data source expected_identity must be DataArtifactIdentity or None"
            )
        observer = cast(object, self.verification_observer)
        if observer is not None and not callable(observer):
            raise TypeError(
                "data source verification_observer must be callable or None"
            )
        verification_workers = cast(object, self.verification_workers)
        if verification_workers is not None and (
            not isinstance(verification_workers, int)
            or isinstance(verification_workers, bool)
            or verification_workers <= 0
            or verification_workers > MAX_ARTIFACT_VERIFICATION_WORKERS
        ):
            raise ConfigError(
                "data source materialization.verification_workers must be "
                "an integer between 1 and "
                f"{MAX_ARTIFACT_VERIFICATION_WORKERS}, or null"
            )
        object.__setattr__(
            self,
            "cache_root",
            lexical_absolute_path(Path(self.cache_root)),
        )
        if self.expected_identity is not None:
            object.__setattr__(self, "verification", "full")


__all__ = [
    "ArtifactVerificationEvent",
    "ArtifactVerificationObserver",
    "ArtifactVerificationPhase",
    "DataArtifact",
    "DataArtifactBinding",
    "DataArtifactBindings",
    "DataArtifactIdentity",
    "DataSource",
    "DataSourceContext",
]
