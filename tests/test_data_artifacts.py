"""Tests for schema-v2 data artifact identities and bindings."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from stochaflow.data.artifacts import (
    ArtifactVerificationEvent,
    DataArtifact,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataSource,
    DataSourceContext,
)
from stochaflow.utils.config import ConfigError


def identity(
    *,
    kind: str = "managed",
    manifest_sha256: str = "e" * 64,
) -> DataArtifactIdentity:
    return DataArtifactIdentity(
        kind=kind,  # type: ignore[arg-type]
        artifact_type="tests.payload.v1",
        source_name="tests-source",
        source_digest="a" * 64,
        materializer_name="tests-materializer",
        materialization_digest="b" * 64,
        content_digest="c" * 64,
        artifact_digest="d" * 64,
        manifest_sha256=manifest_sha256,
    )


def binding(binding_id: str) -> DataArtifactBinding:
    return DataArtifactBinding(id=binding_id, identity=identity())


def test_only_data_source_remains_abstract() -> None:
    assert not inspect.isabstract(DataArtifactIdentity)
    assert not inspect.isabstract(DataArtifact)
    assert inspect.isabstract(DataSource)


@pytest.mark.parametrize("kind", ["managed", "referenced"])
def test_identity_schema_v2_strict_round_trip(kind: str) -> None:
    value = identity(kind=kind)

    assert value.schema_version == 2
    assert DataArtifactIdentity.from_dict(value.to_dict()) == value


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([], "must be a mapping"),
        ({"schema_version": 2}, "missing"),
        ({**identity().to_dict(), "unexpected": True}, "unknown unexpected"),
        ({**identity().to_dict(), "schema_version": 1}, "must be 2"),
        ({**identity().to_dict(), "schema_version": True}, "must be 2"),
        ({**identity().to_dict(), "kind": "remote"}, "managed or referenced"),
        (
            {**identity().to_dict(), "content_digest": "not-a-digest"},
            "content_digest must be a lowercase SHA-256",
        ),
    ],
)
def test_identity_rejects_v1_and_malformed_values(
    value: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DataArtifactIdentity.from_dict(value)


def test_data_artifact_rejects_direct_construction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "object"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_bytes(b"{}\n")
    with pytest.raises(TypeError, match="issued by DataArtifactStore"):
        DataArtifact(
            root=root,
            identity=identity(
                kind="referenced",
                manifest_sha256=hashlib.sha256(b"{}\n").hexdigest(),
            ),
            payload={"value": 1},
        )


def test_data_artifact_rejects_subclass_stand_ins() -> None:
    with pytest.raises(TypeError, match="final and cannot be subclassed"):
        type("ForeignDataArtifact", (DataArtifact,), {})


def test_binding_collection_is_schema_v2_canonical_and_strict() -> None:
    collection = DataArtifactBindings(
        (binding("validation"), binding("primary"))
    )

    assert collection.ids == ("primary", "validation")
    assert collection.to_dict()["schema_version"] == 2
    assert DataArtifactBindings.from_dict(collection.to_dict()) == collection

    with pytest.raises(ValueError, match="ids must be unique"):
        DataArtifactBindings((binding("primary"), binding("primary")))
    with pytest.raises(KeyError, match="missing data artifact binding"):
        collection.identity_for("missing")

    v1 = collection.to_dict()
    v1["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version must be 2"):
        DataArtifactBindings.from_dict(v1)


def test_binding_collection_requires_canonical_order() -> None:
    serialized = DataArtifactBindings(
        (binding("primary"), binding("validation"))
    ).to_dict()
    serialized["bindings"].reverse()

    with pytest.raises(ValueError, match="must be sorted"):
        DataArtifactBindings.from_dict(serialized)


def test_expected_identity_forces_full_verification(tmp_path: Path) -> None:
    context = DataSourceContext(
        cache_root=tmp_path,
        policy="require",
        verification="manifest",
        expected_identity=identity(kind="referenced"),
    )

    assert context.verification == "full"


def test_data_source_context_accepts_only_callable_verification_observer(
    tmp_path: Path,
) -> None:
    def observer(event: object) -> None:
        del event

    context = DataSourceContext(
        cache_root=tmp_path,
        policy="require",
        verification="full",
        verification_observer=observer,
    )

    assert context.verification_observer is observer
    with pytest.raises(TypeError, match="verification_observer"):
        DataSourceContext(
            cache_root=tmp_path,
            policy="require",
            verification="full",
            verification_observer=object(),  # type: ignore[arg-type]
        )


def test_artifact_verification_event_accepts_only_validate_phase() -> None:
    event = ArtifactVerificationEvent(
        artifact_type="tests.payload.v1",
        source_name="tests-source",
        materializer_name="tests-materializer",
        phase="validate",
        completed=0,
        total=1,
    )

    assert event.phase == "validate"
    with pytest.raises(ValueError, match="phase must be validate"):
        ArtifactVerificationEvent(
            artifact_type="tests.payload.v1",
            source_name="tests-source",
            materializer_name="tests-materializer",
            phase="post_load",  # type: ignore[arg-type]
            completed=1,
            total=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", []),
        ("policy", {}),
        ("verification", []),
        ("verification", {}),
    ],
)
def test_data_source_context_rejects_non_string_lifecycle_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "cache_root": tmp_path,
        "policy": "ensure",
        "verification": "full",
    }
    arguments[field] = value

    with pytest.raises(ConfigError, match=rf"materialization\.{field}"):
        DataSourceContext(**arguments)  # type: ignore[arg-type]
