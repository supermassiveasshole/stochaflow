"""Materialized-data contracts and stable checkpoint identities."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from stochaflow.data.artifact_io import (
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
        "artifact_digest",
        "manifest_sha256",
    }
)
_BINDING_FIELDS = frozenset({"id", "identity"})
_COLLECTION_FIELDS = frozenset({"schema_version", "bindings"})
_SHA256_LENGTH = 64


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
class DataArtifactIdentity(ABC):
    """Location-independent identity shared by every materialized artifact."""

    artifact_type: str
    source_name: str
    source_digest: str
    materializer_name: str
    materialization_digest: str
    artifact_digest: str
    manifest_sha256: str
    schema_version: int = 1

    @property
    @abstractmethod
    def kind(self) -> Literal["managed", "referenced"]:
        """Return the persisted artifact ownership kind."""

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("data artifact identity schema_version must be 1")
        for name in ("artifact_type", "source_name", "materializer_name"):
            _non_empty_string(
                getattr(self, name),
                path=f"data artifact identity.{name}",
            )
        for name in (
            "source_digest",
            "materialization_digest",
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
        """Parse and dispatch an identity while rejecting unknown fields."""

        raw = _strict_mapping(value, fields=_IDENTITY_FIELDS, path=path)
        kind = raw["kind"]
        identity_type: type[DataArtifactIdentity]
        if kind == "managed":
            identity_type = ManagedDataArtifactIdentity
        elif kind == "referenced":
            identity_type = ReferencedDataArtifactIdentity
        else:
            raise ValueError(f"{path}.kind must be managed or referenced")
        if cls is not DataArtifactIdentity and cls is not identity_type:
            raise ValueError(f"{path}.kind is incompatible with {cls.__name__}")
        return identity_type(
            schema_version=raw["schema_version"],
            artifact_type=raw["artifact_type"],
            source_name=raw["source_name"],
            source_digest=raw["source_digest"],
            materializer_name=raw["materializer_name"],
            materialization_digest=raw["materialization_digest"],
            artifact_digest=raw["artifact_digest"],
            manifest_sha256=raw["manifest_sha256"],
        )


class ManagedDataArtifactIdentity(DataArtifactIdentity):
    """Identity of content owned by a Stochaflow-managed artifact."""

    __slots__ = ()

    @property
    def kind(self) -> Literal["managed"]:
        """Return the managed ownership discriminator."""

        return "managed"


class ReferencedDataArtifactIdentity(DataArtifactIdentity):
    """Identity of externally owned content indexed without copying it."""

    __slots__ = ()

    @property
    def kind(self) -> Literal["referenced"]:
        """Return the referenced ownership discriminator."""

        return "referenced"


class DataArtifact[ArtifactPayloadT](ABC):
    """Semantic root for a verified runtime artifact handle."""

    identity: DataArtifactIdentity
    payload: ArtifactPayloadT

    @property
    @abstractmethod
    def kind(self) -> Literal["managed", "referenced"]:
        """Return the runtime artifact ownership kind."""


def _verified_manifest(
    root: Path,
    manifest_path: Path,
    identity: DataArtifactIdentity,
    *,
    root_label: str,
) -> tuple[Path, Path]:
    canonical_root = canonical_directory(
        Path(root),
        label=f"data artifact {root_label}",
    )
    canonical_manifest = lexical_absolute_path(Path(manifest_path))
    try:
        relative_manifest = canonical_manifest.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(
            f"data artifact manifest must be inside its {root_label}"
        ) from exc
    encoded, _ = read_regular_file(
        canonical_root,
        relative_manifest.as_posix(),
        label="data artifact manifest",
    )
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != identity.manifest_sha256:
        raise ValueError(
            "data artifact manifest SHA-256 does not match its identity"
        )
    return canonical_root, canonical_manifest


@dataclass(frozen=True, slots=True)
class ManagedDataArtifact[ArtifactPayloadT](DataArtifact[ArtifactPayloadT]):
    """Runtime handle for content published inside a managed artifact root."""

    artifact_root: Path
    manifest_path: Path
    identity: ManagedDataArtifactIdentity
    payload: ArtifactPayloadT

    @property
    def kind(self) -> Literal["managed"]:
        """Return the managed ownership discriminator."""

        return "managed"

    def __post_init__(self) -> None:
        identity = cast(object, self.identity)
        if not isinstance(identity, ManagedDataArtifactIdentity):
            raise TypeError(
                "managed data artifact requires ManagedDataArtifactIdentity"
            )
        root, manifest = _verified_manifest(
            self.artifact_root,
            self.manifest_path,
            self.identity,
            root_label="artifact_root",
        )
        object.__setattr__(self, "artifact_root", root)
        object.__setattr__(self, "manifest_path", manifest)


@dataclass(frozen=True, slots=True)
class ReferencedDataArtifact[ArtifactPayloadT](DataArtifact[ArtifactPayloadT]):
    """Runtime handle for external content described by a cached index."""

    index_root: Path
    manifest_path: Path
    identity: ReferencedDataArtifactIdentity
    payload: ArtifactPayloadT

    @property
    def kind(self) -> Literal["referenced"]:
        """Return the referenced ownership discriminator."""

        return "referenced"

    def __post_init__(self) -> None:
        identity = cast(object, self.identity)
        if not isinstance(identity, ReferencedDataArtifactIdentity):
            raise TypeError(
                "referenced data artifact requires ReferencedDataArtifactIdentity"
            )
        root, manifest = _verified_manifest(
            self.index_root,
            self.manifest_path,
            self.identity,
            root_label="index_root",
        )
        object.__setattr__(self, "index_root", root)
        object.__setattr__(self, "manifest_path", manifest)


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
        identity = cast(object, self.identity)
        if not isinstance(identity, DataArtifactIdentity):
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
    """Canonical, strictly serializable collection of artifact bindings."""

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
            "schema_version": 1,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "metadata.data_artifacts",
    ) -> DataArtifactBindings:
        """Parse a strict, canonically ordered binding collection."""

        raw = _strict_mapping(value, fields=_COLLECTION_FIELDS, path=path)
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError(f"{path}.schema_version must be 1")
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
        object.__setattr__(
            self,
            "cache_root",
            lexical_absolute_path(Path(self.cache_root)),
        )
        if self.expected_identity is not None:
            object.__setattr__(self, "verification", "full")


__all__ = [
    "DataArtifact",
    "DataArtifactBinding",
    "DataArtifactBindings",
    "DataArtifactIdentity",
    "DataSource",
    "DataSourceContext",
    "ManagedDataArtifact",
    "ManagedDataArtifactIdentity",
    "ReferencedDataArtifact",
    "ReferencedDataArtifactIdentity",
]
