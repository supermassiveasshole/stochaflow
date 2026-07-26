"""Tests for materialized-data identity and binding contracts."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from stochaflow.data import (
    DataArtifact,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataSource,
    DataSourceContext,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
    ReferencedDataArtifact,
    ReferencedDataArtifactIdentity,
)
from stochaflow.utils.config import ConfigError


def managed_identity(
    *,
    source_name: str = "example-source",
    manifest_sha256: str = "e" * 64,
) -> ManagedDataArtifactIdentity:
    return ManagedDataArtifactIdentity(
        artifact_type="image-folder",
        source_name=source_name,
        source_digest="a" * 64,
        materializer_name="example-materializer",
        materialization_digest="b" * 64,
        artifact_digest="c" * 64,
        manifest_sha256=manifest_sha256,
    )


def referenced_identity(
    *,
    source_name: str = "example-reference",
    manifest_sha256: str = "e" * 64,
) -> ReferencedDataArtifactIdentity:
    return ReferencedDataArtifactIdentity(
        artifact_type="image-folder-reference",
        source_name=source_name,
        source_digest="1" * 64,
        materializer_name="example-indexer",
        materialization_digest="2" * 64,
        artifact_digest="3" * 64,
        manifest_sha256=manifest_sha256,
    )


def binding(binding_id: str) -> DataArtifactBinding:
    return DataArtifactBinding(
        id=binding_id,
        identity=managed_identity(source_name=f"source-{binding_id}"),
    )


def test_semantic_roots_are_abstract() -> None:
    assert inspect.isabstract(DataArtifactIdentity)
    assert inspect.isabstract(DataArtifact)
    assert inspect.isabstract(DataSource)


@pytest.mark.parametrize(
    "identity",
    [managed_identity(), referenced_identity()],
)
def test_identity_strict_round_trip(identity: DataArtifactIdentity) -> None:
    serialized = identity.to_dict()

    assert serialized["kind"] == identity.kind
    assert DataArtifactIdentity.from_dict(serialized) == identity


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([], "must be a mapping"),
        ({"schema_version": 1}, "missing"),
        (
            {
                **managed_identity().to_dict(),
                "unexpected": True,
            },
            "unknown unexpected",
        ),
        (
            {
                **managed_identity().to_dict(),
                "kind": "remote",
            },
            "kind must be managed or referenced",
        ),
        (
            {
                **managed_identity().to_dict(),
                "artifact_digest": "NOT-A-SHA256",
            },
            "artifact_digest must be a lowercase SHA-256",
        ),
    ],
)
def test_identity_rejects_malformed_values(
    value: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DataArtifactIdentity.from_dict(value)


def test_managed_and_referenced_artifacts_have_distinct_roots(
    tmp_path: Path,
) -> None:
    managed_root = tmp_path / "managed"
    managed_root.mkdir()
    managed_manifest = managed_root / "manifest.json"
    managed_manifest.write_bytes(b"{}\n")
    managed = ManagedDataArtifact(
        artifact_root=managed_root,
        manifest_path=managed_manifest,
        identity=managed_identity(
            manifest_sha256=hashlib.sha256(b"{}\n").hexdigest()
        ),
        payload={"split": "train"},
    )

    external_root = tmp_path / "external"
    external_root.mkdir()
    index_root = tmp_path / "index"
    index_root.mkdir()
    reference_manifest = index_root / "manifest.json"
    reference_manifest.write_bytes(b"{}\n")
    referenced = ReferencedDataArtifact(
        index_root=index_root,
        manifest_path=reference_manifest,
        identity=referenced_identity(
            manifest_sha256=hashlib.sha256(b"{}\n").hexdigest()
        ),
        payload={"external_root": external_root},
    )

    assert managed.artifact_root == managed_root.resolve()
    assert referenced.index_root == index_root.resolve()
    assert referenced.payload["external_root"] == external_root

    with pytest.raises(ValueError, match="inside its artifact_root"):
        ManagedDataArtifact(
            artifact_root=managed_root,
            manifest_path=reference_manifest,
            identity=managed.identity,
            payload=None,
        )


def test_artifact_constructor_rejects_linked_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    target = tmp_path / "manifest-target.json"
    target.write_bytes(b"{}\n")
    manifest = root / "manifest.json"
    try:
        manifest.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    identity = managed_identity(
        manifest_sha256=hashlib.sha256(b"{}\n").hexdigest()
    )

    with pytest.raises((OSError, ValueError), match=r"link|reparse"):
        ManagedDataArtifact(
            artifact_root=root,
            manifest_path=manifest,
            identity=identity,
            payload=None,
        )


def test_artifact_constructor_rejects_linked_root_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    root = real_parent / "artifact"
    root.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_bytes(b"{}\n")
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    identity = managed_identity(
        manifest_sha256=hashlib.sha256(b"{}\n").hexdigest()
    )

    with pytest.raises((OSError, ValueError), match=r"link|reparse"):
        ManagedDataArtifact(
            artifact_root=linked_parent / "artifact",
            manifest_path=linked_parent / "artifact" / "manifest.json",
            identity=identity,
            payload=None,
        )


def test_artifact_kind_and_identity_kind_must_match(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_bytes(b"{}\n")
    wrong_identity = referenced_identity(
        manifest_sha256=hashlib.sha256(b"{}\n").hexdigest()
    )

    with pytest.raises(TypeError, match="ManagedDataArtifactIdentity"):
        ManagedDataArtifact(
            artifact_root=root,
            manifest_path=manifest,
            identity=wrong_identity,  # type: ignore[arg-type]
            payload=None,
        )


def test_binding_collection_is_canonical_and_strict() -> None:
    collection = DataArtifactBindings(
        (binding("validation"), binding("primary"))
    )

    assert collection.ids == ("primary", "validation")
    assert collection.identity_for("primary").source_name == "source-primary"
    assert DataArtifactBindings.from_dict(collection.to_dict()) == collection

    with pytest.raises(ValueError, match="ids must be unique"):
        DataArtifactBindings((binding("primary"), binding("primary")))
    with pytest.raises(KeyError, match="missing data artifact binding"):
        collection.identity_for("missing")

    serialized = collection.to_dict()
    serialized["bindings"].reverse()
    with pytest.raises(ValueError, match="must be sorted"):
        DataArtifactBindings.from_dict(serialized)


def test_binding_collection_rejects_unknown_wire_fields() -> None:
    serialized = DataArtifactBindings((binding("source"),)).to_dict()
    serialized["unexpected"] = True

    with pytest.raises(ValueError, match="unknown unexpected"):
        DataArtifactBindings.from_dict(serialized)


def test_expected_identity_forces_full_verification(tmp_path: Path) -> None:
    context = DataSourceContext(
        cache_root=tmp_path,
        policy="require",
        verification="manifest",
        expected_identity=referenced_identity(),
    )

    assert context.verification == "full"


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
