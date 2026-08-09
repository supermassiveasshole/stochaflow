"""Materialized-data contracts and stable checkpoint identities."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, ClassVar, Literal, NoReturn, Protocol, cast, final
from weakref import ReferenceType, ref

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
_DATA_ARTIFACT_STORE_AUTHORITY = object()

type ArtifactVerificationPhase = Literal["validate"]


@dataclass(frozen=True, slots=True)
class DataArtifactStoreReceipt:
    """Ephemeral evidence that one Store request verified an artifact."""

    identity: DataArtifactIdentity
    root: Path
    artifact_ref: ReferenceType[DataArtifact[Any]] = field(
        repr=False,
        compare=False,
    )
    request_token: object = field(repr=False, compare=False)
    verification: Literal["manifest", "full"]
    authority: object = field(repr=False, compare=False)

    @classmethod
    def _issue(
        cls,
        *,
        authority: object,
        artifact: DataArtifact[Any],
        identity: DataArtifactIdentity,
        root: Path,
        request_token: object,
        verification: Literal["manifest", "full"],
    ) -> DataArtifactStoreReceipt:
        if authority is not _DATA_ARTIFACT_STORE_AUTHORITY:
            raise TypeError("only DataArtifactStore may issue verification receipts")
        return cls(
            identity=identity,
            root=root,
            artifact_ref=ref(artifact),
            request_token=request_token,
            verification=verification,
            authority=_DATA_ARTIFACT_STORE_AUTHORITY,
        )

    def _matches(
        self,
        *,
        artifact: DataArtifact[Any],
        identity: DataArtifactIdentity,
        root: Path,
    ) -> bool:
        return (
            self.authority is _DATA_ARTIFACT_STORE_AUTHORITY
            and self.artifact_ref() is artifact
            and self.identity == identity
            and self.root == root
        )


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


class DataArtifactSelectionObserver(Protocol):
    """Observe formal source-request lifecycle and accepted artifacts."""

    def request_started(
        self,
        *,
        boundary: Literal["source", "store"],
    ) -> None:
        """Mark one outer runtime request as active."""

    def request_finished(self) -> None:
        """Mark one outer runtime request as finished."""

    def record(
        self,
        artifact: DataArtifact[Any],
        source_name: str | None,
    ) -> None:
        """Record one source-accepted handle."""


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


@final
@dataclass(frozen=True, init=False, repr=False, eq=False)
class DataArtifact[ArtifactPayloadT]:
    """Store-issued runtime handle for managed or referenced content."""

    # Keep the weak-reference slot explicit so exact-handle receipts work on
    # every supported Python 3.12 patch release as well as newer runtimes.
    __slots__ = (
        "__weakref__",
        "_store_receipt",
        "identity",
        "payload",
        "root",
    )

    root: Path
    identity: DataArtifactIdentity
    payload: ArtifactPayloadT
    _store_receipt: DataArtifactStoreReceipt

    def __repr__(self) -> str:
        """Represent stable fields without exposing runtime Store evidence."""

        return (
            f"DataArtifact(root={self.root!r}, identity={self.identity!r}, "
            f"payload={self.payload!r})"
        )

    @property
    def kind(self) -> Literal["managed", "referenced"]:
        """Return the artifact ownership strategy."""

        return self.identity.kind

    @property
    def manifest_path(self) -> Path:
        """Return the framework-owned manifest path."""

        return self.root / "manifest.json"

    def __init__(
        self,
        root: Path,
        identity: DataArtifactIdentity,
        payload: ArtifactPayloadT,
    ) -> None:
        del root, identity, payload
        raise TypeError(
            "DataArtifact is issued by DataArtifactStore and cannot be "
            "constructed directly"
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        del cls, kwargs
        raise TypeError("DataArtifact is final and cannot be subclassed")

    def __copy__(self) -> NoReturn:
        raise TypeError(
            "DataArtifact is a Store-issued runtime handle and cannot be copied"
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError(
            "DataArtifact is a Store-issued runtime handle and cannot be copied"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "DataArtifact is a Store-issued runtime handle and cannot be serialized"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError(
            "DataArtifact is a Store-issued runtime handle and cannot be serialized"
        )

    @classmethod
    def _from_store(
        cls,
        authority: object,
        context: DataSourceContext,
        *,
        root: Path,
        identity: DataArtifactIdentity,
        payload: ArtifactPayloadT,
        verification: Literal["manifest", "full"],
    ) -> DataArtifact[ArtifactPayloadT]:
        if authority is not _DATA_ARTIFACT_STORE_AUTHORITY:
            raise TypeError("only DataArtifactStore may issue DataArtifact handles")
        if cls is not DataArtifact:
            raise TypeError("DataArtifact is final and cannot be subclassed")
        if not isinstance(cast(object, context), DataSourceContext):
            raise TypeError("context must be DataSourceContext")
        context._require_active_parent()
        identity_value = cast(object, identity)
        if not isinstance(identity_value, DataArtifactIdentity):
            raise TypeError("data artifact identity must be DataArtifactIdentity")
        artifact_root = canonical_directory(Path(root), label="data artifact root")
        encoded, _ = read_regular_file(
            artifact_root,
            "manifest.json",
            label="data artifact manifest",
        )
        if hashlib.sha256(encoded).hexdigest() != identity.manifest_sha256:
            raise ValueError(
                "data artifact manifest SHA-256 does not match its identity"
            )
        artifact = object.__new__(cls)
        object.__setattr__(artifact, "root", artifact_root)
        object.__setattr__(artifact, "identity", identity)
        object.__setattr__(artifact, "payload", payload)
        receipt = DataArtifactStoreReceipt._issue(
            authority=authority,
            artifact=artifact,
            identity=identity,
            root=artifact_root,
            request_token=context._request_token,
            verification=verification,
        )
        object.__setattr__(artifact, "_store_receipt", receipt)
        return cast(DataArtifact[ArtifactPayloadT], artifact)

def data_artifact_store_receipt(value: object) -> DataArtifactStoreReceipt:
    """Return valid Store evidence for an exact DataArtifact handle."""

    if type(value) is not DataArtifact:
        raise TypeError("data artifact must be an exact DataArtifactStore handle")
    artifact = cast(DataArtifact[Any], value)
    receipt = artifact._store_receipt
    if not receipt._matches(
        artifact=artifact,
        identity=artifact.identity,
        root=artifact.root,
    ):
        raise TypeError("data artifact is missing valid DataArtifactStore evidence")
    return receipt


class DataSource[ArtifactPayloadT](ABC):
    """Artifact-producing source that never constructs runtime data loaders."""

    __stochaflow_data_source_lifecycle_wrapper__: ClassVar[
        Callable[[DataSource[Any], DataSourceContext], DataArtifact[Any]]
    ]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap each source implementation in the common acceptance boundary."""

        super().__init_subclass__(**kwargs)
        implementation = cls.__dict__.get("materialize")
        if implementation is None:
            return
        if not callable(implementation):
            raise TypeError("DataSource.materialize must be callable")

        @wraps(implementation)
        def checked_materialize(
            self: DataSource[Any],
            context: DataSourceContext,
        ) -> DataArtifact[Any]:
            if not isinstance(cast(object, context), DataSourceContext):
                raise TypeError("context must be DataSourceContext")
            lease = context._begin_materialization()
            selection_started = False
            body_succeeded = False
            end_succeeded = False
            try:
                if lease.context_outermost:
                    context._selection_request_started(boundary="source")
                    selection_started = True
                artifact = cast(
                    Callable[[DataSource[Any], DataSourceContext], object],
                    implementation,
                )(self, context)
                accepted = context._accept_artifact(
                    cast(DataArtifact[Any], artifact),
                    enforce_expectations=lease.context_outermost,
                )
                body_succeeded = True
            finally:
                try:
                    context._end_materialization(lease)
                    end_succeeded = True
                finally:
                    if selection_started and (
                        not body_succeeded or not end_succeeded
                    ):
                        context._selection_request_finished()
            try:
                if lease.logical_outermost:
                    context._record_accepted_artifact(accepted)
            finally:
                if selection_started:
                    context._selection_request_finished()
            return accepted

        cls.materialize = checked_materialize
        cls.__stochaflow_data_source_lifecycle_wrapper__ = checked_materialize

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
    def from_artifact(
        cls,
        binding_id: str,
        artifact: DataArtifact[Any],
    ) -> DataArtifactBinding:
        """Bind one exact Store-issued artifact to a runtime role."""

        data_artifact_store_receipt(artifact)
        return cls(id=binding_id, identity=artifact.identity)

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
    def from_artifacts(
        cls,
        artifacts: Sequence[tuple[str, DataArtifact[Any]]],
    ) -> DataArtifactBindings:
        """Build canonical runtime bindings from Store-issued handles."""

        return cls(
            tuple(
                DataArtifactBinding.from_artifact(binding_id, artifact)
                for binding_id, artifact in artifacts
            )
        )

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
class DataSourceRequestLease:
    """One balanced call in a logical and context-local source request."""

    logical_outermost: bool
    context_outermost: bool
    context_token: object = field(repr=False)


@dataclass(slots=True)
class DataSourceRequestState:
    """Thread-safe nesting state for one logical source request context."""

    active_calls: int = 0
    root_active: bool = False
    owner_thread_id: int | None = None
    completed: bool = False
    context_depths: dict[object, int] = field(default_factory=dict)
    context_owner_thread_ids: dict[object, int] = field(default_factory=dict)
    context_parent_tokens: dict[object, object | None] = field(
        default_factory=dict
    )
    active_descendant_counts: dict[object, int] = field(default_factory=dict)
    completed_contexts: set[object] = field(default_factory=set)
    lock: Lock = field(default_factory=Lock, repr=False)

    def begin(
        self,
        *,
        context_token: object,
        parent_context_token: object | None,
    ) -> DataSourceRequestLease:
        """Enter one source call and return its balanced request lease."""

        with self.lock:
            requires_active_parent = parent_context_token is not None
            if requires_active_parent and (
                not self.root_active
                or self.context_depths.get(parent_context_token, 0) <= 0
            ):
                raise RuntimeError(
                    "nested data source context may be used only during its "
                    "outer source request"
                )
            if not requires_active_parent and self.active_calls == 0:
                if self.completed:
                    raise RuntimeError(
                        "data source context cannot be reused for another request"
                    )
                self.owner_thread_id = get_ident()
                self.root_active = True
            elif (
                not requires_active_parent
                and self.owner_thread_id != get_ident()
            ):
                raise RuntimeError(
                    "data source context cannot be reused by independent "
                    "concurrent selections; derive a nested source context"
                )
            context_depth = self.context_depths.get(context_token, 0)
            if context_depth == 0:
                if context_token in self.completed_contexts:
                    raise RuntimeError(
                        "data source context cannot be reused for another request"
                    )
                if (
                    context_token in self.context_parent_tokens
                    and self.context_parent_tokens[context_token]
                    is not parent_context_token
                ):
                    raise RuntimeError(
                        "data source context parent does not match its request"
                    )
            elif self.context_owner_thread_ids[context_token] != get_ident():
                raise RuntimeError(
                    "data source context cannot be reused by independent "
                    "concurrent selections; derive a nested source context"
                )
            logical_outermost = self.active_calls == 0
            context_outermost = context_depth == 0
            if context_outermost:
                self.context_owner_thread_ids[context_token] = get_ident()
                self.context_parent_tokens[context_token] = parent_context_token
                ancestor = parent_context_token
                visited: set[object] = set()
                while ancestor is not None:
                    if ancestor in visited:
                        raise RuntimeError("data source context parent cycle detected")
                    visited.add(ancestor)
                    self.active_descendant_counts[ancestor] = (
                        self.active_descendant_counts.get(ancestor, 0) + 1
                    )
                    ancestor = self.context_parent_tokens.get(ancestor)
            self.context_depths[context_token] = context_depth + 1
            self.active_calls += 1
            return DataSourceRequestLease(
                logical_outermost=logical_outermost,
                context_outermost=context_outermost,
                context_token=context_token,
            )

    def end(self, lease: DataSourceRequestLease) -> None:
        """Leave one source call."""

        with self.lock:
            if self.active_calls <= 0:
                raise RuntimeError("data source request call state is unbalanced")
            context_depth = self.context_depths.get(lease.context_token, 0)
            if context_depth <= 0:
                raise RuntimeError("data source context call state is unbalanced")
            unfinished_logical_work = (
                lease.logical_outermost and self.active_calls != 1
            )
            unfinished_context_work = (
                lease.context_outermost
                and (
                    context_depth != 1
                    or self.active_descendant_counts.get(
                        lease.context_token,
                        0,
                    )
                    > 0
                )
            )
            self.active_calls -= 1
            context_depth -= 1
            if context_depth:
                self.context_depths[lease.context_token] = context_depth
            else:
                self.context_depths.pop(lease.context_token)
                self.context_owner_thread_ids.pop(lease.context_token)
            if lease.context_outermost:
                ancestor = self.context_parent_tokens.get(lease.context_token)
                while ancestor is not None:
                    descendant_count = self.active_descendant_counts[ancestor] - 1
                    if descendant_count:
                        self.active_descendant_counts[ancestor] = descendant_count
                    else:
                        self.active_descendant_counts.pop(ancestor)
                    ancestor = self.context_parent_tokens.get(ancestor)
                self.completed_contexts.add(lease.context_token)
            if lease.logical_outermost:
                self.root_active = False
                self.completed = True
                self.owner_thread_id = None
            if unfinished_logical_work or unfinished_context_work:
                raise RuntimeError(
                    "outer data source returned before nested source work completed"
                )

    def is_root_active(self) -> bool:
        """Return whether an outer source call is still active."""

        with self.lock:
            return self.root_active

    def is_context_active(self, context_token: object) -> bool:
        """Return whether one direct parent context is still active."""

        with self.lock:
            return (
                self.root_active
                and self.context_depths.get(context_token, 0) > 0
            )


@dataclass(frozen=True, slots=True, eq=False)
class DataSourceContext:
    """Materialization policy supplied by an artifact-aware DataBuilder."""

    cache_root: Path
    policy: Literal["require", "ensure"]
    verification: Literal["manifest", "full"]
    expected_identity: DataArtifactIdentity | None = None
    expected_source_name: str | None = None
    verification_observer: ArtifactVerificationObserver | None = None
    verification_workers: int | None = None
    _selection_session: DataArtifactSelectionObserver | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _request_token: object = field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )
    _context_token: object = field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )
    _request_state: DataSourceRequestState = field(
        default_factory=DataSourceRequestState,
        init=False,
        repr=False,
        compare=False,
    )
    _parent_context_token: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

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
        if self.expected_source_name is not None:
            _non_empty_string(
                self.expected_source_name,
                path="data source context.expected_source_name",
            )
        if (
            self.expected_identity is not None
            and self.expected_source_name is not None
            and self.expected_identity.source_name != self.expected_source_name
        ):
            raise ValueError(
                "data source expected identity belongs to a different source"
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

    def __copy__(self) -> DataSourceContext:
        raise TypeError(
            "DataSourceContext is a one-shot runtime request and cannot be "
            "copied; create a new materialization context"
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> DataSourceContext:
        del memo
        raise TypeError(
            "DataSourceContext is a one-shot runtime request and cannot be "
            "copied; create a new materialization context"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "DataSourceContext is a one-shot runtime request and cannot be "
            "serialized; persist configuration instead"
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError(
            "DataSourceContext is a one-shot runtime request and cannot be "
            "serialized; persist configuration instead"
        )

    def _accept_artifact[PayloadT](
        self,
        artifact: DataArtifact[PayloadT],
        *,
        enforce_expectations: bool,
    ) -> DataArtifact[PayloadT]:
        """Require a handle issued by this logical materialization request."""

        self._require_active_parent()
        receipt = data_artifact_store_receipt(artifact)
        if receipt.request_token is not self._request_token:
            raise ValueError(
                "data source returned an artifact issued for a different request"
            )
        strength = {"manifest": 0, "full": 1}
        if strength[receipt.verification] < strength[self.verification]:
            raise ValueError(
                "data source returned an artifact with insufficient verification"
            )
        if enforce_expectations and (
            self.expected_identity is not None
            and artifact.identity != self.expected_identity
        ):
            raise ValueError(
                "data source returned an artifact that does not match the "
                "expected identity"
            )
        if enforce_expectations and (
            self.expected_source_name is not None
            and artifact.identity.source_name != self.expected_source_name
        ):
            raise ValueError(
                "data source returned an artifact for a different registered "
                "source"
            )
        return artifact

    def _begin_materialization(self) -> DataSourceRequestLease:
        return self._request_state.begin(
            context_token=self._context_token,
            parent_context_token=self._parent_context_token,
        )

    def _require_active_parent(self) -> None:
        if self._parent_context_token is not None and not (
            self._request_state.is_context_active(self._parent_context_token)
        ):
            raise RuntimeError(
                "nested data source context may be used only during its outer "
                "source request"
            )

    def _end_materialization(self, lease: DataSourceRequestLease) -> None:
        self._request_state.end(lease)

    def _record_accepted_artifact(self, artifact: DataArtifact[Any]) -> None:
        session = self._selection_session
        if session is not None:
            session.record(artifact, self.expected_source_name)
            return
        from stochaflow.data.artifact_selection import (  # noqa: PLC0415
            record_accepted_data_artifact,
        )

        record_accepted_data_artifact(
            artifact,
            source_name=self.expected_source_name,
        )

    def _selection_request_started(
        self,
        *,
        boundary: Literal["source", "store"],
    ) -> None:
        session = self._selection_session
        if session is not None:
            session.request_started(boundary=boundary)
            return
        from stochaflow.data.artifact_selection import (  # noqa: PLC0415
            start_data_artifact_request,
        )

        start_data_artifact_request(boundary=boundary)

    def _selection_request_finished(self) -> None:
        session = self._selection_session
        if session is not None:
            session.request_finished()
            return
        from stochaflow.data.artifact_selection import (  # noqa: PLC0415
            finish_data_artifact_request,
        )

        finish_data_artifact_request()

    def nested_source_context(
        self,
        *,
        expected_source_name: str | None = None,
    ) -> DataSourceContext:
        """Derive an internal context belonging to this logical source request."""

        if not self._request_state.is_context_active(self._context_token):
            raise RuntimeError(
                "nested data source context may be derived only during its "
                "direct parent source request"
            )
        nested = DataSourceContext(
            cache_root=self.cache_root,
            policy=self.policy,
            verification=self.verification,
            expected_source_name=expected_source_name,
            verification_observer=self.verification_observer,
            verification_workers=self.verification_workers,
        )
        object.__setattr__(nested, "_request_token", self._request_token)
        object.__setattr__(nested, "_request_state", self._request_state)
        object.__setattr__(nested, "_parent_context_token", self._context_token)
        object.__setattr__(nested, "_selection_session", self._selection_session)
        return nested


def materialize_data_source[PayloadT](
    source: DataSource[PayloadT],
    context: DataSourceContext,
) -> DataArtifact[PayloadT]:
    """Materialize any DataSource and validate this request's Store receipt."""

    if not isinstance(cast(object, source), DataSource):
        raise TypeError("source must be a DataSource")
    implementation = getattr(type(source), "materialize", None)
    lifecycle_wrapper = getattr(
        type(source),
        "__stochaflow_data_source_lifecycle_wrapper__",
        None,
    )
    if implementation is not lifecycle_wrapper:
        raise TypeError(
            "source must nominally inherit DataSource and use its lifecycle "
            "wrapper"
        )
    return cast(
        Callable[[DataSource[PayloadT], DataSourceContext], DataArtifact[PayloadT]],
        implementation,
    )(source, context)


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
    "materialize_data_source",
]
