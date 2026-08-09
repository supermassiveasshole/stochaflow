"""Contract tests for modality-neutral DataSource lifecycle enforcement."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import Event
from typing import Any, ClassVar, cast
from weakref import ref

import pytest

from stochaflow.data import build_data_loaders
from stochaflow.extensions import (
    ComponentConfig,
    DataArtifact,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataArtifactLoadContext,
    DataArtifactStore,
    DataArtifactValidationError,
    DataBuilder,
    DataLoaders,
    DataSource,
    DataSourceContext,
    DataSourceMaterializationConfig,
    ManagedDataArtifactBuild,
    ReferencedDataArtifactBuild,
    Registry,
    materialize_data_source,
)
from stochaflow.utils.registry import RegistryCatalog


@dataclass(frozen=True, slots=True)
class RecordArtifactPayload:
    """Project-owned runtime payload with no image or Dataset assumptions."""

    root: Path
    records: tuple[str, ...]


RECORD_SOURCES: Registry[type[DataSource[Any]]] = Registry(
    "test record source",
    expected_type=DataSource,
)


@RECORD_SOURCES.register("tests.records")
class RecordDataSource(DataSource[RecordArtifactPayload]):
    """Independent non-image source using only public extension contracts."""

    def __init__(self, params: Mapping[str, object]) -> None:
        root = params.get("root")
        if not isinstance(root, str) or not root:
            raise ValueError("record source root must be a non-empty string")
        self.root = Path(root)

    artifact_source_name = "tests.records"

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        return materialize_record_artifact(
            context,
            self.root,
            source_name=self.artifact_source_name,
        )


@RECORD_SOURCES.register("tests.wrong-name")
class WrongNameRecordDataSource(RecordDataSource):
    """Incorrect source that publishes another producer's identity."""

    artifact_source_name = "tests.different-records"


@RECORD_SOURCES.register("tests.composed-records")
class ComposedRecordDataSource(RecordDataSource):
    """Source that composes an internal source before publishing its result."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        RecordDataSource({"root": str(self.root)}).materialize(nested)
        return materialize_record_artifact(
            context,
            self.root,
            source_name="tests.composed-records",
        )


@RECORD_SOURCES.register("tests.derived-records")
class DerivedRecordDataSource(RecordDataSource):
    """Use a parent artifact as an intermediate before final publication."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        parent_context = context.nested_source_context(
            expected_source_name="tests.records"
        )
        parent = super().materialize(parent_context)
        if parent.payload.records != ("alpha", "beta"):
            raise ValueError("parent record payload is incompatible")
        return materialize_record_artifact(
            context,
            self.root,
            source_name="tests.derived-records",
        )


@RECORD_SOURCES.register("tests.nested-store-records")
class NestedStoreRecordDataSource(RecordDataSource):
    """Publish an internal Store artifact before the final source result."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        materialize_record_artifact(
            nested,
            self.root,
            source_name="tests.records",
        )
        return materialize_record_artifact(
            context,
            self.root,
            source_name="tests.nested-store-records",
        )


@RECORD_SOURCES.register("tests.delegating-records")
class DelegatingRecordDataSource(RecordDataSource):
    """Validate and return the parent implementation's final artifact."""

    artifact_source_name = "tests.delegating-records"

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        artifact = super().materialize(context)
        if not artifact.payload.records:
            raise ValueError("delegated record payload is empty")
        return artifact


@RECORD_SOURCES.register("tests.thread-composed-records")
class ThreadComposedRecordDataSource(RecordDataSource):
    """Compose an internal source on a worker before publishing one result."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(
                RecordDataSource({"root": str(self.root)}).materialize,
                nested,
            ).result()
        return materialize_record_artifact(
            context,
            self.root,
            source_name="tests.thread-composed-records",
        )


class WrongNestedNameRecordDataSource(RecordDataSource):
    """Incorrect composite whose internal producer violates its selection."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        return WrongNameRecordDataSource(
            {"root": str(self.root)}
        ).materialize(nested)


class WrongNestedStoreNameRecordDataSource(RecordDataSource):
    """Incorrect composite whose nested Store publishes another producer."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        return materialize_record_artifact(
            nested,
            self.root,
            source_name="tests.different-records",
        )


def materialize_record_artifact(
    context: DataSourceContext,
    root: Path,
    *,
    source_name: str,
) -> DataArtifact[RecordArtifactPayload]:
    """Materialize the test record payload through the public Store contract."""

    source_file = root / "records.txt"

    def build(data_root: Path) -> ManagedDataArtifactBuild:
        encoded = source_file.read_bytes()
        (data_root / "records.txt").write_bytes(encoded)
        return ManagedDataArtifactBuild(
            source_digest=hashlib.sha256(encoded).hexdigest(),
            materialization_digest="b" * 64,
            domain={"schema_version": 1, "format": "text-lines"},
        )

    def load(load_context: DataArtifactLoadContext) -> RecordArtifactPayload:
        path = load_context.data_root / "records.txt"
        return RecordArtifactPayload(
            root=load_context.data_root,
            records=tuple(path.read_text(encoding="utf-8").splitlines()),
        )

    return DataArtifactStore(context).materialize_managed(
        artifact_type="tests.records.v1",
        source_name=source_name,
        materializer_name="tests.text-lines",
        locator_key={"root": str(root.resolve())},
        build=build,
        load=load,
    )


def materialize_blocking_record_artifact(
    context: DataSourceContext,
    root: Path,
    started: Event,
    release: Event,
) -> DataArtifact[RecordArtifactPayload]:
    """Hold a direct nested Store request open inside its build callback."""

    source_file = root / "records.txt"

    def build(data_root: Path) -> ManagedDataArtifactBuild:
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("nested Store test request was not released")
        encoded = source_file.read_bytes()
        (data_root / "records.txt").write_bytes(encoded)
        return ManagedDataArtifactBuild(
            source_digest=hashlib.sha256(encoded).hexdigest(),
            materialization_digest="b" * 64,
            domain={"schema_version": 1, "format": "text-lines"},
        )

    def load(load_context: DataArtifactLoadContext) -> RecordArtifactPayload:
        path = load_context.data_root / "records.txt"
        return RecordArtifactPayload(
            root=load_context.data_root,
            records=tuple(path.read_text(encoding="utf-8").splitlines()),
        )

    return DataArtifactStore(context).materialize_managed(
        artifact_type="tests.records.v1",
        source_name="tests.records",
        materializer_name="tests.text-lines",
        locator_key={"root": str(root.resolve())},
        build=build,
        load=load,
    )


class CachedRecordDataSource(DataSource[RecordArtifactPayload]):
    """Incorrect source that reuses a handle from an earlier request."""

    def __init__(self, artifact: DataArtifact[RecordArtifactPayload]) -> None:
        self.artifact = artifact

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        del context
        return self.artifact


class BlockingRecordDataSource(RecordDataSource):
    """Hold one top-level source request open for concurrency checks."""

    def __init__(self, root: Path, started: Event, release: Event) -> None:
        super().__init__({"root": str(root)})
        self.started = started
        self.release = release

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test source was not released")
        return super().materialize(context)


class UnjoinedNestedRecordDataSource(RecordDataSource):
    """Incorrect composite that returns before its internal worker completes."""

    def __init__(self, root: Path) -> None:
        super().__init__({"root": str(root)})
        self.started = Event()
        self.release = Event()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Future[DataArtifact[RecordArtifactPayload]] | None = None

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        source = BlockingRecordDataSource(
            self.root,
            self.started,
            self.release,
        )
        self.future = self.executor.submit(
            materialize_data_source,
            source,
            nested,
        )
        if not self.started.wait(timeout=5):
            raise TimeoutError("nested test source did not start")
        return materialize_record_artifact(
            context,
            self.root,
            source_name="tests.unjoined-nested",
        )

    def close(self) -> None:
        """Release and join the intentionally unfinished worker."""

        self.release.set()
        assert self.future is not None
        with pytest.raises(RuntimeError, match="only during its outer"):
            self.future.result(timeout=5)
        self.executor.shutdown(wait=True)


class UnjoinedNestedStoreRecordDataSource(RecordDataSource):
    """Incorrect composite whose worker directly uses Store after return."""

    def __init__(self, root: Path) -> None:
        super().__init__({"root": str(root)})
        self.started = Event()
        self.release = Event()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future: Future[DataArtifact[RecordArtifactPayload]] | None = None

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        self.future = self.executor.submit(
            materialize_blocking_record_artifact,
            nested,
            self.root,
            self.started,
            self.release,
        )
        if not self.started.wait(timeout=5):
            raise TimeoutError("nested Store test request did not start")
        return materialize_record_artifact(
            context,
            self.root,
            source_name="tests.unjoined-nested-store",
        )

    def close(self) -> None:
        """Release and join the intentionally unfinished Store worker."""

        self.release.set()
        assert self.future is not None
        with pytest.raises(RuntimeError, match="only during its outer"):
            self.future.result(timeout=5)
        self.executor.shutdown(wait=True)


class WrappedOverrideRecordDataSource(RecordDataSource):
    """Incorrect override that copies metadata from the inherited wrapper."""

    @wraps(RecordDataSource.materialize)
    def materialize(
        self,
        context: DataSourceContext,
    ) -> Any:
        del context
        return object()


class RecordDataBuilder(DataBuilder):
    """Compose arbitrary record batches from a family-local source registry."""

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("record builder requires root and cache_root")
        source_name = self.context.params.get("source_name", "tests.records")
        if not isinstance(source_name, str):
            raise TypeError("record builder source_name must be a string")
        self.context.require_artifact_ids(("records",))
        source = cast(
            DataSource[RecordArtifactPayload],
            RECORD_SOURCES.create(source_name, {"root": root}),
        )
        source_context = self.context.data_source_context(
            DataSourceMaterializationConfig(
                cache_root=cache_root,
                verification="manifest",
            ),
            binding_id="records",
            source_name=source_name,
            path="data.params.source.materialization",
        )
        artifact = materialize_data_source(source, source_context)
        if not isinstance(cast(object, artifact.payload), RecordArtifactPayload):
            raise TypeError("record source returned an incompatible payload")
        if self.context.params.get("omit_binding") is True:
            return DataLoaders(train=[artifact.payload.records])
        bindings = DataArtifactBindings.from_artifacts((("records", artifact),))
        self.context.verify_artifacts(bindings)
        return DataLoaders(
            train=[artifact.payload.records],
            artifact_bindings=bindings,
        )


class ForgedBindingDataBuilder(DataBuilder):
    """Incorrect Builder that declares identity without materializing data."""

    def build(self) -> DataLoaders:
        identity = DataArtifactIdentity(
            kind="managed",
            artifact_type="tests.forged.v1",
            source_name="tests.forged",
            source_digest="a" * 64,
            materializer_name="tests.forged",
            materialization_digest="b" * 64,
            content_digest="c" * 64,
            artifact_digest="d" * 64,
            manifest_sha256="e" * 64,
        )
        return DataLoaders(
            train=[1],
            artifact_bindings=DataArtifactBindings(
                (DataArtifactBinding(id="source", identity=identity),)
            ),
        )


class EchoExpectedBindingsDataBuilder(DataBuilder):
    """Incorrect Builder that echoes checkpoint identity without verification."""

    def build(self) -> DataLoaders:
        if self.context.expected_artifacts is None:
            raise ValueError("test requires expected artifacts")
        return DataLoaders(
            train=[1],
            artifact_bindings=self.context.expected_artifacts,
        )


class DirectStoreDataBuilder(DataBuilder):
    """Incorrect Builder that skips the DataSource acceptance boundary."""

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("direct Store builder requires roots")
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.records",
            path="data.params.source.materialization",
        )
        artifact = materialize_record_artifact(
            context,
            Path(root),
            source_name="tests.records",
        )
        return DataLoaders(
            train=[artifact.payload.records],
            artifact_bindings=DataArtifactBindings.from_artifacts(
                (("records", artifact),)
            ),
        )


class DirectSourceRecordDataBuilder(RecordDataBuilder):
    """Use the supported direct DataSource call retained for compatibility."""

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("direct source builder requires roots")
        self.context.require_artifact_ids(("records",))
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.records",
            path="data.params.source.materialization",
        )
        artifact = RecordDataSource({"root": root}).materialize(context)
        return DataLoaders(
            train=[artifact.payload.records],
            artifact_bindings=DataArtifactBindings.from_artifacts(
                (("records", artifact),)
            ),
        )


class EquivalentBindingAfterMaterializeDataBuilder(RecordDataBuilder):
    """Return an equal serialized binding after current source verification."""

    def build(self) -> DataLoaders:
        loaders = super().build()
        if self.context.expected_artifacts is None:
            raise ValueError("test requires expected artifacts")
        return DataLoaders(
            train=loaders.train,
            artifact_bindings=self.context.expected_artifacts,
        )


class ManifestOnlyStrictDataBuilder(DataBuilder):
    """Incorrect strict Builder that avoids the required full verification."""

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("manifest-only strict builder requires roots")
        artifact = materialize_data_source(
            RecordDataSource({"root": root}),
            DataSourceContext(
                cache_root=Path(cache_root),
                policy="require",
                verification="manifest",
                expected_source_name="tests.records",
            ),
        )
        return DataLoaders(
            train=[artifact.payload.records],
            artifact_bindings=DataArtifactBindings.from_artifacts(
                (("records", artifact),)
            ),
        )


class IncompleteNestedRecordDataSource(DataSource[RecordArtifactPayload]):
    """Incorrect source that returns while a nested request remains active."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifact: DataArtifact[RecordArtifactPayload] | None = None
        self.nested: DataSourceContext | None = None
        self.nested_lease: Any = None

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        self.nested = context.nested_source_context(
            expected_source_name="tests.records"
        )
        self.nested_lease = self.nested._begin_materialization()
        assert not self.nested_lease.logical_outermost
        self.artifact = materialize_record_artifact(
            context,
            self.root,
            source_name="tests.incomplete-nested",
        )
        return self.artifact

    def close_nested_request(self) -> None:
        assert self.nested is not None
        self.nested._end_materialization(self.nested_lease)


class CaughtIncompleteSourceDataBuilder(DataBuilder):
    """Incorrect Builder that binds a source result after request-close failure."""

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("incomplete source builder requires roots")
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.incomplete-nested",
            path="data.params.source.materialization",
        )
        source = IncompleteNestedRecordDataSource(Path(root))
        with pytest.raises(RuntimeError, match="nested source work completed"):
            materialize_data_source(source, context)
        source.close_nested_request()
        assert source.artifact is not None
        return DataLoaders(
            train=[source.artifact.payload.records],
            artifact_bindings=DataArtifactBindings.from_artifacts(
                (("records", source.artifact),)
            ),
        )


class ThreadedRecordDataBuilder(RecordDataBuilder):
    """Materialize a source in a worker while retaining build evidence."""

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("threaded record builder requires roots")
        self.context.require_artifact_ids(("records",))
        source = RecordDataSource({"root": root})
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.records",
            path="data.params.source.materialization",
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            artifact = executor.submit(
                materialize_data_source,
                source,
                context,
            ).result()
        return DataLoaders(
            train=[artifact.payload.records],
            artifact_bindings=DataArtifactBindings.from_artifacts(
                (("records", artifact),)
            ),
        )


class UnjoinedSourceDataBuilder(DataBuilder):
    """Incorrect Builder that returns while a source request is in flight."""

    executor: ClassVar[ThreadPoolExecutor | None] = None
    future: ClassVar[Future[DataArtifact[RecordArtifactPayload]] | None] = None
    release: ClassVar[Event | None] = None

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("unjoined source builder requires roots")
        started = Event()
        release = Event()
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.records",
            path="data.params.source.materialization",
        )
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            materialize_data_source,
            BlockingRecordDataSource(Path(root), started, release),
            context,
        )
        UnjoinedSourceDataBuilder.executor = executor
        UnjoinedSourceDataBuilder.future = future
        UnjoinedSourceDataBuilder.release = release
        if not started.wait(timeout=5):
            raise TimeoutError("formal test source did not start")
        return DataLoaders(train=[1])

    @classmethod
    def close_worker(cls) -> None:
        """Release the intentionally unfinished formal-build request."""

        assert cls.executor is not None
        assert cls.future is not None
        assert cls.release is not None
        cls.release.set()
        with pytest.raises(RuntimeError, match="selection is closed"):
            cls.future.result(timeout=5)
        cls.executor.shutdown(wait=True)
        cls.executor = None
        cls.future = None
        cls.release = None


class CaughtUnjoinedNestedDataBuilder(DataBuilder):
    """Incorrect Builder that swallows a parent source lifecycle failure."""

    source: ClassVar[UnjoinedNestedRecordDataSource | None] = None

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("caught nested builder requires roots")
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.unjoined-nested",
            path="data.params.source.materialization",
        )
        source = UnjoinedNestedRecordDataSource(Path(root))
        CaughtUnjoinedNestedDataBuilder.source = source
        with pytest.raises(
            RuntimeError,
            match="outer data source returned before nested source work completed",
        ):
            materialize_data_source(source, context)
        return DataLoaders(train=[1])

    @classmethod
    def close_worker(cls) -> None:
        """Release the intentionally unfinished nested request."""

        assert cls.source is not None
        cls.source.close()
        cls.source = None


class CaughtUnjoinedNestedStoreDataBuilder(DataBuilder):
    """Incorrect Builder that swallows a nested Store lifecycle failure."""

    source: ClassVar[UnjoinedNestedStoreRecordDataSource | None] = None

    def build(self) -> DataLoaders:
        root = self.context.params.get("root")
        cache_root = self.context.params.get("cache_root")
        if not isinstance(root, str) or not isinstance(cache_root, str):
            raise TypeError("caught nested Store builder requires roots")
        context = self.context.data_source_context(
            DataSourceMaterializationConfig(cache_root=cache_root),
            binding_id="records",
            source_name="tests.unjoined-nested-store",
            path="data.params.source.materialization",
        )
        source = UnjoinedNestedStoreRecordDataSource(Path(root))
        CaughtUnjoinedNestedStoreDataBuilder.source = source
        with pytest.raises(
            RuntimeError,
            match="outer data source returned before nested source work completed",
        ):
            materialize_data_source(source, context)
        return DataLoaders(train=[1])

    @classmethod
    def close_worker(cls) -> None:
        """Release the intentionally unfinished nested Store request."""

        assert cls.source is not None
        cls.source.close()
        cls.source = None


class AliasedRoleRecordDataBuilder(RecordDataBuilder):
    """Bind one accepted artifact to two Builder-owned semantic roles."""

    def build(self) -> DataLoaders:
        loaders = super().build()
        assert loaders.artifact_bindings is not None
        identity = loaders.artifact_bindings.identity_for("records")
        return DataLoaders(
            train=loaders.train,
            artifact_bindings=DataArtifactBindings(
                (
                    DataArtifactBinding(id="records", identity=identity),
                    DataArtifactBinding(id="validation-records", identity=identity),
                )
            ),
        )


class ReferencedRecordDataSource(DataSource[RecordArtifactPayload]):
    """Independent non-image source whose represented payload stays external."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[RecordArtifactPayload]:
        source_file = self.root / "records.txt"

        def build(data_root: Path) -> ReferencedDataArtifactBuild:
            encoded = source_file.read_bytes()
            (data_root / "index.txt").write_text(
                "records.txt\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(encoded).hexdigest()
            return ReferencedDataArtifactBuild(
                source_digest=digest,
                materialization_digest="c" * 64,
                content_digest=digest,
                domain={"schema_version": 1, "format": "external-text-lines"},
            )

        def load(load_context: DataArtifactLoadContext) -> RecordArtifactPayload:
            encoded = source_file.read_bytes()
            if (
                load_context.verification == "full"
                and hashlib.sha256(encoded).hexdigest()
                != load_context.identity.content_digest
            ):
                raise DataArtifactValidationError(
                    "referenced record content does not match its identity"
                )
            return RecordArtifactPayload(
                root=self.root,
                records=tuple(encoded.decode("utf-8").splitlines()),
            )

        return DataArtifactStore(context).materialize_referenced(
            artifact_type="tests.referenced-records.v1",
            source_name="tests.referenced-records",
            materializer_name="tests.external-text-lines",
            locator_key={"root": str(self.root.resolve())},
            referenced_roots={"records": source_file},
            build=build,
            load=load,
        )


def data_builder_catalog(
    name: str,
    builder: type[DataBuilder],
) -> RegistryCatalog:
    catalog = RegistryCatalog()
    catalog.data_builders.add(name, builder)
    return catalog


def write_records(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "records.txt").write_text("alpha\nbeta\n", encoding="utf-8")


def test_non_image_family_local_source_is_store_verified_and_resumable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    params = {
        "root": str(root),
        "cache_root": str(tmp_path / "cache"),
    }
    catalog = data_builder_catalog("tests.record-builder", RecordDataBuilder)

    first = build_data_loaders(
        ComponentConfig(name="tests.record-builder", params=params),
        seed=7,
        registries=catalog,
    )
    assert list(first.train) == [("alpha", "beta")]
    assert first.artifact_bindings is not None
    assert first.artifact_bindings.ids == ("records",)

    resumed = build_data_loaders(
        ComponentConfig(name="tests.record-builder", params=params),
        seed=7,
        strict_resume=True,
        expected_artifacts=first.artifact_bindings,
        registries=catalog,
    )
    assert resumed.artifact_bindings == first.artifact_bindings


def test_composed_source_records_only_its_final_artifact(tmp_path: Path) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog("tests.record-builder", RecordDataBuilder)

    for source_name in (
        "tests.composed-records",
        "tests.delegating-records",
        "tests.derived-records",
        "tests.nested-store-records",
        "tests.thread-composed-records",
    ):
        loaders = build_data_loaders(
            ComponentConfig(
                name="tests.record-builder",
                params={
                    "root": str(root),
                    "cache_root": str(tmp_path / "cache"),
                    "source_name": source_name,
                },
            ),
            seed=7,
            registries=catalog,
        )

        assert loaders.artifact_bindings is not None
        assert (
            loaders.artifact_bindings.identity_for("records").source_name
            == source_name
        )
        resumed = build_data_loaders(
            ComponentConfig(
                name="tests.record-builder",
                params={
                    "root": str(root),
                    "cache_root": str(tmp_path / "cache"),
                    "source_name": source_name,
                },
            ),
            seed=7,
            strict_resume=True,
            expected_artifacts=loaders.artifact_bindings,
            registries=catalog,
        )
        assert resumed.artifact_bindings == loaders.artifact_bindings


def test_non_image_referenced_source_preserves_project_owned_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    source = ReferencedRecordDataSource(root)
    context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
        expected_source_name="tests.referenced-records",
    )

    artifact = materialize_data_source(source, context)

    assert artifact.kind == "referenced"
    assert artifact.payload == RecordArtifactPayload(
        root=root,
        records=("alpha", "beta"),
    )
    assert artifact.identity.source_name == "tests.referenced-records"
    assert deepcopy(artifact.identity) == artifact.identity

    (root / "records.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        DataArtifactValidationError,
        match="does not match its identity",
    ):
        materialize_data_source(
            source,
            DataSourceContext(
                cache_root=tmp_path / "cache",
                policy="require",
                verification="full",
                expected_identity=artifact.identity,
                expected_source_name="tests.referenced-records",
            ),
        )


def test_data_artifact_handle_rejects_copy_serialization_and_receipt_reuse(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    source = RecordDataSource({"root": str(root)})
    first = materialize_data_source(
        source,
        DataSourceContext(
            cache_root=tmp_path / "cache",
            policy="ensure",
            verification="full",
        ),
    )
    second = materialize_data_source(
        source,
        DataSourceContext(
            cache_root=tmp_path / "cache",
            policy="require",
            verification="full",
        ),
    )

    assert first.identity == second.identity
    assert first is not second
    assert first != second
    assert ref(first)() is first
    with pytest.raises(TypeError, match="Store-issued runtime handle"):
        copy(first)
    with pytest.raises(TypeError, match="Store-issued runtime handle"):
        deepcopy(first)
    with pytest.raises(TypeError, match="Store-issued runtime handle"):
        pickle.dumps(first)

    fabricated = object.__new__(DataArtifact)
    for name in ("root", "identity", "payload", "_store_receipt"):
        object.__setattr__(
            fabricated,
            name,
            object.__getattribute__(first, name),
        )
    with pytest.raises(TypeError, match="valid DataArtifactStore evidence"):
        DataArtifactBindings.from_artifacts(
            (("records", cast(DataArtifact[Any], fabricated)),)
        )


def test_source_request_rejects_cached_handle_from_another_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    source = RecordDataSource({"root": str(root)})
    first_context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
    )
    first = materialize_data_source(source, first_context)
    stale_context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="require",
        verification="manifest",
        expected_identity=first.identity,
    )

    with pytest.raises(ValueError, match="different request"):
        materialize_data_source(CachedRecordDataSource(first), stale_context)

    resume_context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="require",
        verification="manifest",
        expected_identity=first.identity,
    )
    resumed = materialize_data_source(source, resume_context)
    assert resumed.identity == first.identity

    with pytest.raises(RuntimeError, match="direct parent source request"):
        resume_context.nested_source_context(
            expected_source_name="tests.records"
        )


def test_inactive_nested_context_rejects_source_and_store(tmp_path: Path) -> None:
    root = tmp_path / "records"
    write_records(root)
    parent = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
    )
    parent_lease = parent._begin_materialization()
    nested = parent.nested_source_context(
        expected_source_name="tests.records"
    )
    parent._end_materialization(parent_lease)

    with pytest.raises(RuntimeError, match="only during its outer source request"):
        materialize_data_source(RecordDataSource({"root": str(root)}), nested)
    with pytest.raises(RuntimeError, match="only during its outer source request"):
        materialize_record_artifact(
            nested,
            root,
            source_name="tests.records",
        )


def test_source_context_rejects_independent_concurrent_reuse(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    started = Event()
    release = Event()
    source = BlockingRecordDataSource(root, started, release)
    context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
        expected_source_name="tests.records",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(materialize_data_source, source, context)
        assert started.wait(timeout=5)
        second = executor.submit(materialize_data_source, source, context)
        with pytest.raises(RuntimeError, match="independent concurrent"):
            second.result(timeout=5)
        release.set()
        assert first.result(timeout=5).identity.source_name == "tests.records"

    with pytest.raises(RuntimeError, match="cannot be reused"):
        materialize_data_source(source, context)


def test_source_context_rejects_copy_and_serialization(tmp_path: Path) -> None:
    context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
    )
    equivalent_values = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
    )

    assert context is not equivalent_values
    assert context != equivalent_values
    with pytest.raises(TypeError, match="one-shot runtime request"):
        copy(context)
    with pytest.raises(TypeError, match="one-shot runtime request"):
        deepcopy(context)
    with pytest.raises(TypeError, match="one-shot runtime request"):
        pickle.dumps(context)


def test_composite_source_rejects_wrong_nested_producer_and_unjoined_worker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    with pytest.raises(ValueError, match="does not match the selected source"):
        materialize_data_source(
            WrongNestedNameRecordDataSource({"root": str(root)}),
            DataSourceContext(
                cache_root=tmp_path / "wrong-cache",
                policy="ensure",
                verification="full",
            ),
        )

    unjoined = UnjoinedNestedRecordDataSource(root)
    try:
        with pytest.raises(
            RuntimeError,
            match="outer data source returned before nested source work completed",
        ):
            materialize_data_source(
                unjoined,
                DataSourceContext(
                    cache_root=tmp_path / "unjoined-cache",
                    policy="ensure",
                    verification="full",
                    expected_source_name="tests.unjoined-nested",
                ),
            )
    finally:
        unjoined.close()


def test_nested_store_rejects_wrong_selected_source(tmp_path: Path) -> None:
    root = tmp_path / "records"
    write_records(root)

    with pytest.raises(ValueError, match="does not match the selected source"):
        materialize_data_source(
            WrongNestedStoreNameRecordDataSource({"root": str(root)}),
            DataSourceContext(
                cache_root=tmp_path / "cache",
                policy="ensure",
                verification="full",
            ),
        )


def test_nested_source_must_finish_before_its_direct_parent_returns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    root_context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
    )
    root_lease = root_context._begin_materialization()
    middle_context = root_context.nested_source_context(
        expected_source_name="tests.unjoined-nested"
    )
    unjoined = UnjoinedNestedRecordDataSource(root)
    try:
        with pytest.raises(
            RuntimeError,
            match="outer data source returned before nested source work completed",
        ):
            materialize_data_source(unjoined, middle_context)
    finally:
        unjoined.close()
        root_context._end_materialization(root_lease)


def test_completed_nested_context_cannot_start_another_child(
    tmp_path: Path,
) -> None:
    root_context = DataSourceContext(
        cache_root=tmp_path / "cache",
        policy="ensure",
        verification="full",
    )
    root_lease = root_context._begin_materialization()
    parent = root_context.nested_source_context()
    parent_lease = parent._begin_materialization()
    parent._end_materialization(parent_lease)
    with pytest.raises(RuntimeError, match="direct parent source request"):
        parent.nested_source_context()
    root_context._end_materialization(root_lease)


def test_source_acceptance_occurs_inside_request_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    original = DataSourceContext._accept_artifact

    def assert_active_request(
        context: DataSourceContext,
        artifact: DataArtifact[Any],
        *,
        enforce_expectations: bool,
    ) -> DataArtifact[Any]:
        assert context._request_state.is_root_active()
        return original(
            context,
            artifact,
            enforce_expectations=enforce_expectations,
        )

    monkeypatch.setattr(
        DataSourceContext,
        "_accept_artifact",
        assert_active_request,
    )
    artifact = materialize_data_source(
        RecordDataSource({"root": str(root)}),
        DataSourceContext(
            cache_root=tmp_path / "cache",
            policy="ensure",
            verification="full",
            expected_source_name="tests.records",
        ),
    )

    assert artifact.payload.records == ("alpha", "beta")


def test_materialize_helper_rejects_abc_virtual_subclass(tmp_path: Path) -> None:
    class VirtualRecordDataSource:
        def materialize(self, context: DataSourceContext) -> object:
            del context
            return object()

    DataSource.register(VirtualRecordDataSource)
    virtual = cast(DataSource[Any], VirtualRecordDataSource())

    with pytest.raises(TypeError, match="nominally inherit DataSource"):
        materialize_data_source(
            virtual,
            DataSourceContext(
                cache_root=tmp_path / "cache",
                policy="ensure",
                verification="full",
            ),
        )

    with pytest.raises(TypeError, match="exact DataArtifactStore handle"):
        materialize_data_source(
            WrappedOverrideRecordDataSource({"root": str(tmp_path)}),
            DataSourceContext(
                cache_root=tmp_path / "wrapped-cache",
                policy="ensure",
                verification="full",
            ),
        )


def test_formal_builder_rejects_identity_without_current_store_evidence() -> None:
    forged_catalog = data_builder_catalog(
        "tests.forged-builder",
        ForgedBindingDataBuilder,
    )
    with pytest.raises(ValueError, match="was not accepted through its DataSource"):
        build_data_loaders(
            ComponentConfig(name="tests.forged-builder"),
            seed=1,
            registries=forged_catalog,
        )

    expected = DataArtifactBindings(
        (
            DataArtifactBinding(
                id="source",
                identity=DataArtifactIdentity(
                    kind="managed",
                    artifact_type="tests.expected.v1",
                    source_name="tests.expected",
                    source_digest="a" * 64,
                    materializer_name="tests.expected",
                    materialization_digest="b" * 64,
                    content_digest="c" * 64,
                    artifact_digest="d" * 64,
                    manifest_sha256="e" * 64,
                ),
            ),
        )
    )
    echo_catalog = data_builder_catalog(
        "tests.echo-builder",
        EchoExpectedBindingsDataBuilder,
    )
    with pytest.raises(ValueError, match="was not accepted through its DataSource"):
        build_data_loaders(
            ComponentConfig(name="tests.echo-builder"),
            seed=1,
            strict_resume=True,
            expected_artifacts=expected,
            registries=echo_catalog,
        )


def test_formal_builder_rejects_materialized_but_unbound_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog("tests.record-builder", RecordDataBuilder)

    with pytest.raises(ValueError, match="returned no artifact bindings"):
        build_data_loaders(
            ComponentConfig(
                name="tests.record-builder",
                params={
                    "root": str(root),
                    "cache_root": str(tmp_path / "cache"),
                    "omit_binding": True,
                },
            ),
            seed=1,
            registries=catalog,
        )


def test_formal_builder_rejects_wrong_source_role(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    params = {"root": str(root), "cache_root": str(tmp_path / "cache")}

    wrong_source_catalog = data_builder_catalog(
        "tests.record-builder",
        RecordDataBuilder,
    )
    with pytest.raises(ValueError, match="does not match the selected source"):
        build_data_loaders(
            ComponentConfig(
                name="tests.record-builder",
                params={**params, "source_name": "tests.wrong-name"},
            ),
            seed=1,
            registries=wrong_source_catalog,
        )


def test_formal_builder_rejects_direct_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    params = {"root": str(root), "cache_root": str(tmp_path / "cache")}

    direct_store_catalog = data_builder_catalog(
        "tests.direct-store-builder",
        DirectStoreDataBuilder,
    )
    with pytest.raises(
        RuntimeError,
        match="must materialize artifacts through DataSource",
    ):
        build_data_loaders(
            ComponentConfig(name="tests.direct-store-builder", params=params),
            seed=1,
            registries=direct_store_catalog,
        )


def test_formal_builder_allows_one_artifact_for_multiple_semantic_roles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog(
        "tests.aliased-role-builder",
        AliasedRoleRecordDataBuilder,
    )

    loaders = build_data_loaders(
        ComponentConfig(
            name="tests.aliased-role-builder",
            params={"root": str(root), "cache_root": str(tmp_path / "cache")},
        ),
        seed=1,
        registries=catalog,
    )

    assert loaders.artifact_bindings is not None
    assert loaders.artifact_bindings.ids == ("records", "validation-records")


def test_formal_builder_records_worker_thread_source_acceptance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog(
        "tests.threaded-record-builder",
        ThreadedRecordDataBuilder,
    )

    loaders = build_data_loaders(
        ComponentConfig(
            name="tests.threaded-record-builder",
            params={
                "root": str(root),
                "cache_root": str(tmp_path / "cache"),
            },
        ),
        seed=1,
        registries=catalog,
    )

    assert list(loaders.train) == [("alpha", "beta")]
    assert loaders.artifact_bindings is not None
    assert loaders.artifact_bindings.ids == ("records",)


def test_formal_builder_rejects_unjoined_source_request(tmp_path: Path) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog(
        "tests.unjoined-source-builder",
        UnjoinedSourceDataBuilder,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="before all source requests completed",
        ):
            build_data_loaders(
                ComponentConfig(
                    name="tests.unjoined-source-builder",
                    params={
                        "root": str(root),
                        "cache_root": str(tmp_path / "cache"),
                    },
                ),
                seed=1,
                registries=catalog,
            )
    finally:
        UnjoinedSourceDataBuilder.close_worker()


def test_formal_builder_rejects_swallowed_parent_error_with_nested_worker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog(
        "tests.caught-unjoined-nested-builder",
        CaughtUnjoinedNestedDataBuilder,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="before all source requests completed",
        ):
            build_data_loaders(
                ComponentConfig(
                    name="tests.caught-unjoined-nested-builder",
                    params={
                        "root": str(root),
                        "cache_root": str(tmp_path / "cache"),
                    },
                ),
                seed=1,
                registries=catalog,
            )
    finally:
        CaughtUnjoinedNestedDataBuilder.close_worker()


def test_formal_builder_rejects_swallowed_parent_error_with_nested_store_worker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog(
        "tests.caught-unjoined-nested-store-builder",
        CaughtUnjoinedNestedStoreDataBuilder,
    )

    try:
        with pytest.raises(
            RuntimeError,
            match="before all source requests completed",
        ):
            build_data_loaders(
                ComponentConfig(
                    name="tests.caught-unjoined-nested-store-builder",
                    params={
                        "root": str(root),
                        "cache_root": str(tmp_path / "cache"),
                    },
                ),
                seed=1,
                registries=catalog,
            )
    finally:
        CaughtUnjoinedNestedStoreDataBuilder.close_worker()


def test_formal_builder_accepts_equal_binding_after_current_verification(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    params = {"root": str(root), "cache_root": str(tmp_path / "cache")}
    direct_catalog = data_builder_catalog(
        "tests.direct-source-builder",
        DirectSourceRecordDataBuilder,
    )
    first = build_data_loaders(
        ComponentConfig(name="tests.direct-source-builder", params=params),
        seed=1,
        registries=direct_catalog,
    )
    assert first.artifact_bindings is not None

    serialized = DataArtifactBindings.from_dict(first.artifact_bindings.to_dict())
    equivalent_catalog = data_builder_catalog(
        "tests.equivalent-binding-builder",
        EquivalentBindingAfterMaterializeDataBuilder,
    )
    resumed = build_data_loaders(
        ComponentConfig(name="tests.equivalent-binding-builder", params=params),
        seed=1,
        strict_resume=True,
        expected_artifacts=serialized,
        registries=equivalent_catalog,
    )
    assert resumed.artifact_bindings == serialized


def test_strict_resume_rejects_manifest_only_current_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    params = {"root": str(root), "cache_root": str(tmp_path / "cache")}
    first_catalog = data_builder_catalog("tests.record-builder", RecordDataBuilder)
    first = build_data_loaders(
        ComponentConfig(name="tests.record-builder", params=params),
        seed=1,
        registries=first_catalog,
    )
    assert first.artifact_bindings is not None
    manifest_catalog = data_builder_catalog(
        "tests.manifest-only-builder",
        ManifestOnlyStrictDataBuilder,
    )

    with pytest.raises(ValueError, match="was not fully verified"):
        build_data_loaders(
            ComponentConfig(name="tests.manifest-only-builder", params=params),
            seed=1,
            strict_resume=True,
            expected_artifacts=first.artifact_bindings,
            registries=manifest_catalog,
        )


def test_failed_outer_request_does_not_publish_selection_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    write_records(root)
    catalog = data_builder_catalog(
        "tests.caught-incomplete-source-builder",
        CaughtIncompleteSourceDataBuilder,
    )

    with pytest.raises(ValueError, match="not accepted through its DataSource"):
        build_data_loaders(
            ComponentConfig(
                name="tests.caught-incomplete-source-builder",
                params={
                    "root": str(root),
                    "cache_root": str(tmp_path / "cache"),
                },
            ),
            seed=1,
            registries=catalog,
        )
