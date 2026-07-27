"""Contract tests for the unified schema-v2 data artifact store."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

import stochaflow.data.artifact_store as artifact_store_module
from stochaflow.data.artifact_store import (
    DataArtifactLoadContext,
    DataArtifactStore,
    DataArtifactValidationError,
    ManagedDataArtifactBuild,
    ReferencedDataArtifactBuild,
    canonical_artifact_digest,
    canonical_artifact_json_bytes,
)
from stochaflow.data.artifacts import DataArtifact, DataSourceContext


def context(
    cache_root: Path,
    *,
    policy: str = "ensure",
    verification: str = "full",
    expected_identity: object = None,
) -> DataSourceContext:
    return DataSourceContext(
        cache_root=cache_root,
        policy=policy,  # type: ignore[arg-type]
        verification=verification,  # type: ignore[arg-type]
        expected_identity=expected_identity,  # type: ignore[arg-type]
    )


def managed_callbacks(
    calls: list[tuple[str, Path, str | None]],
) -> tuple[
    Callable[[Path], ManagedDataArtifactBuild],
    Callable[[DataArtifactLoadContext], bytes],
]:
    def build(data_root: Path) -> ManagedDataArtifactBuild:
        calls.append(("build", data_root, None))
        (data_root / "payload.bin").write_bytes(b"managed-content")
        return ManagedDataArtifactBuild(
            source_digest="a" * 64,
            materialization_digest="b" * 64,
            domain={"schema_version": 1, "filename": "payload.bin"},
        )

    def load(load_context: DataArtifactLoadContext) -> bytes:
        calls.append(
            (
                "load",
                load_context.data_root,
                load_context.verification,
            )
        )
        return (load_context.data_root / "payload.bin").read_bytes()

    return build, load


def materialize_managed(
    store_context: DataSourceContext,
    calls: list[tuple[str, Path, str | None]],
) -> DataArtifact[bytes]:
    build, load = managed_callbacks(calls)
    return DataArtifactStore(store_context).materialize_managed(
        artifact_type="tests.bytes.v1",
        source_name="tests-source",
        materializer_name="tests-writer",
        locator_key={"edition": 1},
        build=build,
        load=load,
    )


def materialize_referenced_file(
    cache_root: Path,
    external_path: Path,
    *,
    locator_slot: str,
    policy: str = "ensure",
    expected_identity: object = None,
    build_barrier: Barrier | None = None,
) -> DataArtifact[Path]:
    def build(data_root: Path) -> ReferencedDataArtifactBuild:
        encoded = external_path.read_bytes()
        content_digest = hashlib.sha256(encoded).hexdigest()
        (data_root / "sidecar.json").write_bytes(b"{}\n")
        if build_barrier is not None:
            build_barrier.wait(timeout=5)
        return ReferencedDataArtifactBuild(
            source_digest=content_digest,
            materialization_digest="b" * 64,
            content_digest=content_digest,
            domain={"schema_version": 1},
        )

    def load(load_context: DataArtifactLoadContext) -> Path:
        if load_context.verification == "full":
            observed = hashlib.sha256(external_path.read_bytes()).hexdigest()
            if observed != load_context.identity.content_digest:
                raise DataArtifactValidationError(
                    "represented external content changed"
                )
        return external_path

    return DataArtifactStore(
        context(
            cache_root,
            policy=policy,
            expected_identity=expected_identity,
        )
    ).materialize_referenced(
        artifact_type="tests.referenced-file.v1",
        source_name="tests-referenced-file",
        materializer_name="tests-reference-indexer",
        locator_key={"slot": locator_slot},
        referenced_roots={"content": external_path},
        build=build,
        load=load,
    )


def test_canonical_artifact_json_is_strict() -> None:
    assert canonical_artifact_json_bytes({"é": [2, 1]}) == (
        '{"é":[2,1]}\n'.encode()
    )
    assert canonical_artifact_digest({"a": 1}) == (
        "e346432021b04179518d9614f3560ccd71354a4ee101ddcb893d6959a9d6301c"
    )

    with pytest.raises(TypeError, match="field names must be strings"):
        canonical_artifact_json_bytes({1: "value"})
    with pytest.raises(ValueError, match="NaN"):
        canonical_artifact_json_bytes({"value": float("nan")})
    with pytest.raises(TypeError, match="JSON-safe"):
        canonical_artifact_json_bytes({"value": Path("not-json")})


def test_managed_ensure_builds_publishes_and_loads_final_root(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path, str | None]] = []
    artifact = materialize_managed(context(tmp_path / "cache"), calls)

    assert artifact.kind == "managed"
    assert artifact.payload == b"managed-content"
    assert artifact.root.parent.name == "objects"
    assert artifact.root.name == artifact.identity.artifact_digest
    assert artifact.manifest_path.is_file()
    assert calls[0][0] == "build"
    assert calls[-1] == ("load", artifact.root / "data", "full")
    assert all("staging" not in str(path) for path in [calls[-1][1]])
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["stored_files"]["record_count"] == 1
    assert artifact.identity.content_digest == manifest["stored_files"]["digest"]
    namespace = artifact.root.parents[1]
    assert not tuple((namespace / "staging").iterdir())


def test_ensure_cache_hit_does_not_build(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    first_calls: list[tuple[str, Path, str | None]] = []
    first = materialize_managed(context(cache_root), first_calls)
    second_calls: list[tuple[str, Path, str | None]] = []
    second = materialize_managed(context(cache_root), second_calls)

    assert second.identity == first.identity
    assert [name for name, _, _ in second_calls] == ["load"]


def test_require_miss_is_completely_read_only(tmp_path: Path) -> None:
    cache_root = tmp_path / "missing-cache"
    calls: list[tuple[str, Path, str | None]] = []

    with pytest.raises(FileNotFoundError, match="required data artifact locator"):
        materialize_managed(
            context(cache_root, policy="require", verification="manifest"),
            calls,
        )

    assert calls == []
    assert not cache_root.exists()


def test_manifest_and_full_have_different_owned_content_strength(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    payload_path = artifact.root / "data" / "payload.bin"
    payload_path.write_bytes(b"MANAGED-CONTENT")

    manifest_calls: list[tuple[str, Path, str | None]] = []
    manifest = materialize_managed(
        context(cache_root, policy="require", verification="manifest"),
        manifest_calls,
    )
    assert manifest.payload == b"MANAGED-CONTENT"

    with pytest.raises(
        DataArtifactValidationError,
        match="does not match inventory",
    ):
        materialize_managed(
            context(cache_root, policy="require", verification="full"),
            [],
        )


def test_ensure_repairs_corrupt_owned_content(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    (artifact.root / "data" / "payload.bin").write_bytes(b"broken")
    calls: list[tuple[str, Path, str | None]] = []

    repaired = materialize_managed(context(cache_root), calls)

    assert repaired.payload == b"managed-content"
    assert [name for name, _, _ in calls].count("build") == 1
    quarantine = artifact.root.parents[1] / "quarantine" / "objects"
    assert len(tuple(quarantine.iterdir())) == 1


@pytest.mark.parametrize("corrupt_target", ["manifest", "inventory"])
def test_require_is_read_only_and_ensure_repairs_framework_metadata(
    tmp_path: Path,
    corrupt_target: str,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    target = (
        artifact.manifest_path
        if corrupt_target == "manifest"
        else next((artifact.root / "inventory").iterdir())
    )
    target.write_bytes(b"{}\n")
    corrupted = target.read_bytes()
    namespace = artifact.root.parents[1]
    quarantine = namespace / "quarantine" / "objects"

    with pytest.raises(DataArtifactValidationError):
        materialize_managed(context(cache_root, policy="require"), [])

    assert target.read_bytes() == corrupted
    assert not tuple(quarantine.iterdir())

    repaired = materialize_managed(context(cache_root), [])

    assert repaired.payload == b"managed-content"
    assert len(tuple(quarantine.iterdir())) == 1


def test_ensure_quarantines_object_while_holding_its_digest_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    (artifact.root / "data" / "payload.bin").write_bytes(b"broken")
    held_locks: set[Path] = set()
    object_quarantine_checked = False
    original_lock = artifact_store_module.ArtifactMaterializationLock
    original_quarantine = artifact_store_module._quarantine_entry

    @contextmanager
    def tracking_lock(
        path: Path,
        *,
        cache_root: Path,
    ) -> Iterator[None]:
        with original_lock(path, cache_root=cache_root):
            held_locks.add(path)
            try:
                yield
            finally:
                held_locks.remove(path)

    def checking_quarantine(
        paths: artifact_store_module.ArtifactStorePaths,
        path: Path,
        *,
        destination_root: Path,
        is_directory: bool,
    ) -> None:
        nonlocal object_quarantine_checked
        if path == artifact.root:
            expected_lock = (
                paths.object_locks
                / f"{artifact.identity.artifact_digest}.lock"
            )
            assert expected_lock in held_locks
            object_quarantine_checked = True
        original_quarantine(
            paths,
            path,
            destination_root=destination_root,
            is_directory=is_directory,
        )

    monkeypatch.setattr(
        artifact_store_module,
        "ArtifactMaterializationLock",
        tracking_lock,
    )
    monkeypatch.setattr(
        artifact_store_module,
        "_quarantine_entry",
        checking_quarantine,
    )

    repaired = materialize_managed(context(cache_root), [])

    assert repaired.payload == b"managed-content"
    assert object_quarantine_checked


def test_expected_identity_bypasses_locator_and_forces_full(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    locator = next((artifact.root.parents[1] / "locators").iterdir())
    locator.write_text("not JSON")
    calls: list[tuple[str, Path, str | None]] = []

    resumed = materialize_managed(
        context(
            cache_root,
            policy="require",
            verification="manifest",
            expected_identity=artifact.identity,
        ),
        calls,
    )

    assert resumed.identity == artifact.identity
    assert calls == [("load", artifact.root / "data", "full")]
    assert locator.read_text() == "not JSON"


def test_expected_identity_mismatch_fails_before_cache_mutation(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    expected = replace(artifact.identity, source_digest="c" * 64)
    before = {
        path.relative_to(cache_root): path.read_bytes()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    calls: list[str] = []

    def unexpected_build(data_root: Path) -> ManagedDataArtifactBuild:
        calls.append(f"build:{data_root}")
        raise AssertionError("identity mismatch must not build")

    def unexpected_load(load_context: DataArtifactLoadContext) -> bytes:
        calls.append(f"load:{load_context.data_root}")
        raise DataArtifactValidationError(
            "identity mismatch must not load represented content"
        )

    with pytest.raises(
        DataArtifactValidationError,
        match="does not match expected identity",
    ):
        DataArtifactStore(
            context(cache_root, expected_identity=expected)
        ).materialize_managed(
            artifact_type="tests.bytes.v1",
            source_name="tests-source",
            materializer_name="tests-writer",
            locator_key={"edition": 1},
            build=unexpected_build,
            load=unexpected_load,
        )

    after = {
        path.relative_to(cache_root): path.read_bytes()
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert calls == []


def test_referenced_artifact_indexes_without_copying_external_file(
    tmp_path: Path,
) -> None:
    external = tmp_path / "trajectory.npy"
    external.write_bytes(b"external")
    cache_root = tmp_path / "cache"
    digest = canonical_artifact_digest(
        [{"path": external.name, "size_bytes": 8, "sha256": "c" * 64}]
    )

    def build(data_root: Path) -> ReferencedDataArtifactBuild:
        (data_root / "sidecar.json").write_text("{}\n")
        return ReferencedDataArtifactBuild(
            source_digest="a" * 64,
            materialization_digest="b" * 64,
            content_digest=digest,
            domain={"schema_version": 1},
        )

    def load(load_context: DataArtifactLoadContext) -> Path:
        return external

    artifact = DataArtifactStore(context(cache_root)).materialize_referenced(
        artifact_type="tests.reference.v1",
        source_name="tests-reference",
        materializer_name="tests-indexer",
        locator_key={"array": "trajectory"},
        referenced_roots={"trajectory": external},
        build=build,
        load=load,
    )

    assert artifact.kind == "referenced"
    assert artifact.payload == external
    assert list((artifact.root / "data").iterdir()) == [
        artifact.root / "data" / "sidecar.json"
    ]
    assert str(external.resolve()) not in artifact.manifest_path.read_text()


def test_referenced_external_change_preserves_shared_content_addressed_object(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(b"shared-content")
    second_path.write_bytes(b"shared-content")
    first = materialize_referenced_file(
        cache_root,
        first_path,
        locator_slot="first",
    )
    second = materialize_referenced_file(
        cache_root,
        second_path,
        locator_slot="second",
    )
    shared_manifest = first.manifest_path.read_bytes()

    assert second.root == first.root

    second_path.write_bytes(b"changed-content")
    rebuilt = materialize_referenced_file(
        cache_root,
        second_path,
        locator_slot="second",
    )

    assert rebuilt.root != first.root
    assert first.root.is_dir()
    assert first.manifest_path.read_bytes() == shared_manifest
    assert not tuple(
        (first.root.parents[1] / "quarantine" / "objects").iterdir()
    )
    required_first = materialize_referenced_file(
        cache_root,
        first_path,
        locator_slot="first",
        policy="require",
    )
    required_second = materialize_referenced_file(
        cache_root,
        second_path,
        locator_slot="second",
        policy="require",
    )
    assert required_first.identity == first.identity
    assert required_first.root == first.root
    assert required_second.identity == rebuilt.identity
    assert required_second.root == rebuilt.root


def test_concurrent_referenced_locators_share_the_publication_winner(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(b"shared-content")
    second_path.write_bytes(b"shared-content")
    build_barrier = Barrier(2)

    def materialize(request: tuple[Path, str]) -> DataArtifact[Path]:
        external_path, locator_slot = request
        return materialize_referenced_file(
            cache_root,
            external_path,
            locator_slot=locator_slot,
            build_barrier=build_barrier,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(
            materialize,
            ((first_path, "first"), (second_path, "second")),
        )

    assert first.root == second.root
    assert first.payload == first_path
    assert second.payload == second_path
    namespace = first.root.parents[1]
    objects = tuple((namespace / "objects").iterdir())
    locators = tuple((namespace / "locators").iterdir())
    assert objects == (first.root,)
    assert len(locators) == 2
    assert {
        json.loads(locator.read_text())["artifact_digest"]
        for locator in locators
    } == {first.identity.artifact_digest}
    assert not tuple((namespace / "quarantine" / "objects").iterdir())
    assert not tuple((namespace / "quarantine" / "locators").iterdir())


def test_expected_referenced_content_failure_does_not_move_framework_object(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    external_path = tmp_path / "external.bin"
    external_path.write_bytes(b"expected-content")
    artifact = materialize_referenced_file(
        cache_root,
        external_path,
        locator_slot="expected",
    )
    locator = next((artifact.root.parents[1] / "locators").iterdir())
    locator_bytes = locator.read_bytes()
    manifest_bytes = artifact.manifest_path.read_bytes()

    external_path.write_bytes(b"unexpected-content")
    with pytest.raises(
        DataArtifactValidationError,
        match="built data artifact identity does not match expected identity",
    ):
        materialize_referenced_file(
            cache_root,
            external_path,
            locator_slot="expected",
            expected_identity=artifact.identity,
        )

    assert artifact.root.is_dir()
    assert artifact.manifest_path.read_bytes() == manifest_bytes
    assert locator.read_bytes() == locator_bytes
    assert not tuple(
        (artifact.root.parents[1] / "quarantine" / "objects").iterdir()
    )

    external_path.write_bytes(b"expected-content")
    resumed = materialize_referenced_file(
        cache_root,
        external_path,
        locator_slot="expected",
        policy="require",
        expected_identity=artifact.identity,
    )
    assert resumed.identity == artifact.identity
    assert resumed.root == artifact.root


def test_referenced_root_must_not_overlap_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    external = cache_root / "external"
    external.mkdir(parents=True)
    calls = 0

    def build(data_root: Path) -> ReferencedDataArtifactBuild:
        nonlocal calls
        calls += 1
        raise AssertionError

    with pytest.raises(ValueError, match="overlaps"):
        DataArtifactStore(context(cache_root)).materialize_referenced(
            artifact_type="tests.reference.v1",
            source_name="tests-reference",
            materializer_name="tests-indexer",
            locator_key={},
            referenced_roots={"external": external},
            build=build,
            load=lambda _: None,
        )

    assert calls == 0


def test_producer_bug_is_not_classified_as_cache_corruption(
    tmp_path: Path,
) -> None:
    def build(data_root: Path) -> ManagedDataArtifactBuild:
        raise TypeError("producer bug")

    with pytest.raises(TypeError, match="producer bug"):
        DataArtifactStore(context(tmp_path / "cache")).materialize_managed(
            artifact_type="tests.bytes.v1",
            source_name="tests-source",
            materializer_name="tests-writer",
            locator_key={},
            build=build,
            load=lambda _: None,
        )

    namespace = (
        tmp_path
        / "cache"
        / "data-artifacts"
        / "v2"
        / "managed"
        / canonical_artifact_digest("tests.bytes.v1")
    )
    assert not tuple((namespace / "staging").iterdir())


@pytest.mark.parametrize("use_expected_identity", [False, True])
def test_load_oserror_is_not_classified_as_cache_corruption(
    tmp_path: Path,
    use_expected_identity: bool,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    build_calls = 0

    def unexpected_build(data_root: Path) -> ManagedDataArtifactBuild:
        nonlocal build_calls
        build_calls += 1
        raise AssertionError(f"producer load error must not rebuild {data_root}")

    def broken_load(load_context: DataArtifactLoadContext) -> bytes:
        raise FileNotFoundError(
            f"producer load bug under {load_context.data_root}"
        )

    with pytest.raises(FileNotFoundError, match="producer load bug"):
        DataArtifactStore(
            context(
                cache_root,
                verification="manifest",
                expected_identity=(
                    artifact.identity if use_expected_identity else None
                ),
            )
        ).materialize_managed(
            artifact_type="tests.bytes.v1",
            source_name="tests-source",
            materializer_name="tests-writer",
            locator_key={"edition": 1},
            build=unexpected_build,
            load=broken_load,
        )

    assert build_calls == 0
    assert artifact.root.is_dir()
    assert (artifact.root / "data" / "payload.bin").read_bytes() == (
        b"managed-content"
    )
    quarantine = artifact.root.parents[1] / "quarantine" / "objects"
    assert not tuple(quarantine.iterdir())


def test_manifest_load_detects_same_size_callback_mutation(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    payload_path = artifact.root / "data" / "payload.bin"

    def mutating_load(load_context: DataArtifactLoadContext) -> bytes:
        path = load_context.data_root / "payload.bin"
        metadata = path.stat()
        path.write_bytes(b"MANAGED-CONTENT")
        os.utime(
            path,
            ns=(
                metadata.st_atime_ns,
                metadata.st_mtime_ns + 1_000_000_000,
            ),
        )
        return path.read_bytes()

    build, _ = managed_callbacks([])
    with pytest.raises(
        RuntimeError,
        match="load callback mutated its artifact",
    ):
        DataArtifactStore(
            context(
                cache_root,
                policy="require",
                verification="manifest",
            )
        ).materialize_managed(
            artifact_type="tests.bytes.v1",
            source_name="tests-source",
            materializer_name="tests-writer",
            locator_key={"edition": 1},
            build=build,
            load=mutating_load,
        )

    assert payload_path.read_bytes() == b"MANAGED-CONTENT"
    quarantine = artifact.root.parents[1] / "quarantine" / "objects"
    assert not tuple(quarantine.iterdir())


def test_build_cannot_replace_framework_created_data_root(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    displaced = tmp_path / "displaced-data"

    def build(data_root: Path) -> ManagedDataArtifactBuild:
        data_root.rename(displaced)
        data_root.mkdir()
        return ManagedDataArtifactBuild(
            source_digest="a" * 64,
            materialization_digest="b" * 64,
            domain={"schema_version": 1},
        )

    with pytest.raises(RuntimeError, match="replaced its supplied data root"):
        DataArtifactStore(context(cache_root)).materialize_managed(
            artifact_type="tests.bytes.v1",
            source_name="tests-source",
            materializer_name="tests-writer",
            locator_key={},
            build=build,
            load=lambda _: None,
        )

    namespace = (
        cache_root
        / "data-artifacts"
        / "v2"
        / "managed"
        / canonical_artifact_digest("tests.bytes.v1")
    )
    assert displaced.is_dir()
    assert not tuple((namespace / "staging").iterdir())


def test_same_digest_different_valid_identity_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    collision_digest = "d" * 64
    monkeypatch.setattr(
        artifact_store_module,
        "_artifact_digest",
        lambda **_: collision_digest,
    )

    def materialize(source_name: str) -> DataArtifact[bytes]:
        build, load = managed_callbacks([])
        return DataArtifactStore(context(cache_root)).materialize_managed(
            artifact_type="tests.collision.v1",
            source_name=source_name,
            materializer_name="tests-writer",
            locator_key={"source": source_name},
            build=build,
            load=load,
        )

    winner = materialize("tests-source-a")
    manifest_bytes = winner.manifest_path.read_bytes()

    with pytest.raises(RuntimeError, match="digest collision"):
        materialize("tests-source-b")

    assert winner.manifest_path.read_bytes() == manifest_bytes
    quarantine = winner.root.parents[1] / "quarantine" / "objects"
    assert not tuple(quarantine.iterdir())


def test_incompatible_locator_is_repaired_without_quarantining_valid_object(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"

    def materialize(
        source_name: str,
        *,
        policy: str = "ensure",
    ) -> DataArtifact[bytes]:
        build, load = managed_callbacks([])
        return DataArtifactStore(
            context(cache_root, policy=policy)
        ).materialize_managed(
            artifact_type="tests.locator-compatibility.v1",
            source_name=source_name,
            materializer_name="tests-writer",
            locator_key={"source": source_name},
            build=build,
            load=load,
        )

    first = materialize("tests-source-a")
    second = materialize("tests-source-b")
    namespace = first.root.parents[1]
    second_locator = next(
        path
        for path in (namespace / "locators").iterdir()
        if json.loads(path.read_text())["artifact_digest"]
        == second.identity.artifact_digest
    )
    second_locator.write_bytes(
        canonical_artifact_json_bytes(
            {
                "schema_version": 2,
                "artifact_digest": first.identity.artifact_digest,
            }
        )
    )
    first_manifest = first.manifest_path.read_bytes()

    with pytest.raises(
        DataArtifactValidationError,
        match="incompatible producer",
    ):
        materialize("tests-source-b", policy="require")

    assert first.manifest_path.read_bytes() == first_manifest
    assert json.loads(second_locator.read_text())["artifact_digest"] == (
        first.identity.artifact_digest
    )
    assert not tuple((namespace / "quarantine" / "objects").iterdir())
    assert not tuple((namespace / "quarantine" / "locators").iterdir())

    repaired = materialize("tests-source-b")

    assert repaired.identity == second.identity
    assert first.manifest_path.read_bytes() == first_manifest
    assert not tuple((namespace / "quarantine" / "objects").iterdir())
    assert len(tuple((namespace / "quarantine" / "locators").iterdir())) == 1


@pytest.mark.parametrize(
    "unexpected_entry",
    [
        "root_file",
        "root_directory",
        "inventory_file",
        "data_file",
        "data_directory",
        "root_symlink",
        "root_fifo",
    ],
)
def test_object_layout_violation_is_read_only_or_repaired_by_policy(
    tmp_path: Path,
    unexpected_entry: str,
) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    if unexpected_entry == "root_file":
        extra = artifact.root / "unexpected.bin"
        extra.write_bytes(b"unexpected")
    elif unexpected_entry == "root_directory":
        extra = artifact.root / "unexpected"
        extra.mkdir()
    elif unexpected_entry == "inventory_file":
        extra = artifact.root / "inventory" / "unexpected.jsonl"
        extra.write_bytes(b"{}\n")
    elif unexpected_entry == "data_file":
        extra = artifact.root / "data" / "unexpected.bin"
        extra.write_bytes(b"unexpected")
    elif unexpected_entry == "data_directory":
        extra = artifact.root / "data" / "unexpected"
        extra.mkdir()
    elif unexpected_entry == "root_symlink":
        extra = artifact.root / "unexpected-link"
        try:
            extra.symlink_to(tmp_path, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks are unavailable: {exc}")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO files are unavailable")
        extra = artifact.root / "unexpected-fifo"
        os.mkfifo(extra)
    namespace = artifact.root.parents[1]

    with pytest.raises(DataArtifactValidationError):
        materialize_managed(context(cache_root, policy="require"), [])

    assert extra.exists() or extra.is_symlink()
    assert not tuple((namespace / "quarantine" / "objects").iterdir())

    repaired = materialize_managed(context(cache_root), [])

    assert repaired.payload == b"managed-content"
    assert len(tuple((namespace / "quarantine" / "objects").iterdir())) == 1


@pytest.mark.parametrize(
    "credential_key",
    ["Authorization", "apiKey", "access_key"],
)
def test_locator_key_rejects_common_credential_spellings(
    tmp_path: Path,
    credential_key: str,
) -> None:
    cache_root = tmp_path / "cache"

    with pytest.raises(ValueError, match="must not contain credentials"):
        DataArtifactStore(context(cache_root)).materialize_managed(
            artifact_type="tests.bytes.v1",
            source_name="tests-source",
            materializer_name="tests-writer",
            locator_key={credential_key: "sensitive"},
            build=lambda _: pytest.fail("credential preflight must not build"),
            load=lambda _: None,
        )

    assert not cache_root.exists()


def test_locator_rejects_v1_and_ensure_repairs_it(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    artifact = materialize_managed(context(cache_root), [])
    locator = next((artifact.root.parents[1] / "locators").iterdir())
    locator.write_bytes(
        canonical_artifact_json_bytes(
            {"schema_version": 1, "artifact_digest": artifact.root.name}
        )
    )

    with pytest.raises(DataArtifactValidationError, match="schema_version"):
        materialize_managed(context(cache_root, policy="require"), [])

    repaired = materialize_managed(context(cache_root), [])
    assert repaired.identity == artifact.identity
    assert json.loads(locator.read_text())["schema_version"] == 2
