"""Runtime proof that declared data bindings were verified in this build."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

from stochaflow.data.artifacts import (
    DataArtifact,
    DataArtifactBindings,
    DataArtifactStoreReceipt,
    data_artifact_store_receipt,
)


@dataclass(slots=True)
class AcceptedDataArtifact:
    """One Store handle accepted through a DataSource request boundary."""

    artifact: DataArtifact[Any]
    receipt: DataArtifactStoreReceipt
    source_name: str | None


@dataclass(slots=True)
class DataArtifactSelectionSession:
    """Source artifacts accepted while one DataBuilder is being resolved."""

    accepted: list[AcceptedDataArtifact] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _in_flight: int = 0
    _direct_store_requests: int = 0
    _closed: bool = False

    def request_started(
        self,
        *,
        boundary: Literal["source", "store"],
    ) -> None:
        """Track one outer source or direct-Store request."""

        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "data Builder already returned; no new source request may start"
                )
            self._in_flight += 1
            if boundary == "store":
                self._direct_store_requests += 1

    def request_finished(self) -> None:
        """Close one previously started outer request."""

        with self._lock:
            if self._in_flight <= 0:
                raise RuntimeError("data artifact request tracking is unbalanced")
            self._in_flight -= 1

    def close(self) -> None:
        """Close the Builder selection window and reject unfinished work."""

        with self._lock:
            self._closed = True
            in_flight = self._in_flight
            direct_store_requests = self._direct_store_requests
        if in_flight:
            raise RuntimeError(
                "data Builder returned before all source requests completed"
            )
        if direct_store_requests:
            raise RuntimeError(
                "formal DataBuilder must materialize artifacts through DataSource, "
                "not DataArtifactStore directly"
            )

    def record(
        self,
        artifact: DataArtifact[Any],
        source_name: str | None,
    ) -> None:
        """Record one exact handle accepted by a source request."""

        receipt = data_artifact_store_receipt(artifact)
        accepted = AcceptedDataArtifact(
            artifact=artifact,
            receipt=receipt,
            source_name=source_name,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "data Builder already returned; artifact selection is closed"
                )
            if any(item.receipt is receipt for item in self.accepted):
                return
            self.accepted.append(accepted)

    def validate(
        self,
        bindings: DataArtifactBindings | None,
        *,
        strict_resume: bool,
    ) -> None:
        """Require every binding to come from this build's accepted sources."""

        with self._lock:
            accepted = tuple(self.accepted)

        if bindings is None or len(bindings) == 0:
            if accepted:
                raise ValueError(
                    "data builder accepted source artifacts but returned no "
                    "artifact bindings"
                )
            return

        used: set[object] = set()
        for binding in bindings:
            candidates = [
                item
                for item in accepted
                if item.artifact.identity == binding.identity
            ]
            if not candidates:
                raise ValueError(
                    f"data artifact binding '{binding.id}' was not accepted "
                    "through its DataSource request during this data build"
                )
            receipts: list[DataArtifactStoreReceipt] = []
            for selected in candidates:
                receipt = data_artifact_store_receipt(selected.artifact)
                if (
                    receipt is not selected.receipt
                    or receipt.identity != binding.identity
                ):
                    raise ValueError(
                        f"data artifact binding '{binding.id}' has invalid Store "
                        "evidence"
                    )
                if (
                    selected.source_name is not None
                    and binding.identity.source_name != selected.source_name
                ):
                    raise ValueError(
                        f"data artifact binding '{binding.id}' belongs to a "
                        "different registered source"
                    )
                receipts.append(receipt)
            if strict_resume and not any(
                receipt.verification == "full" for receipt in receipts
            ):
                raise ValueError(
                    f"strict resume data artifact binding '{binding.id}' was not "
                    "fully verified during this data build"
                )
            used.add(binding.identity)

        accepted_identities = {item.artifact.identity for item in accepted}
        if not accepted_identities.issubset(used):
            raise ValueError(
                "data builder accepted a source artifact without returning its "
                "binding"
            )


ACTIVE_DATA_ARTIFACT_SELECTION: ContextVar[
    DataArtifactSelectionSession | None
] = ContextVar("active_data_artifact_selection", default=None)


@contextmanager
def capture_data_artifact_selections() -> Generator[
    DataArtifactSelectionSession,
    None,
    None,
]:
    """Capture source-accepted artifacts for one formal DataBuilder execution."""

    session = DataArtifactSelectionSession()
    token = ACTIVE_DATA_ARTIFACT_SELECTION.set(session)
    try:
        yield session
    finally:
        try:
            session.close()
        finally:
            ACTIVE_DATA_ARTIFACT_SELECTION.reset(token)


def record_accepted_data_artifact(
    artifact: DataArtifact[Any],
    *,
    source_name: str | None,
) -> None:
    """Record a source-accepted artifact when a formal build is active."""

    session = ACTIVE_DATA_ARTIFACT_SELECTION.get()
    if session is not None:
        session.record(artifact, source_name)


def start_data_artifact_request(
    *,
    boundary: Literal["source", "store"],
) -> None:
    """Track one outer request through the active formal build, if any."""

    session = ACTIVE_DATA_ARTIFACT_SELECTION.get()
    if session is not None:
        session.request_started(boundary=boundary)


def finish_data_artifact_request() -> None:
    """Finish one outer request tracked through the active formal build."""

    session = ACTIVE_DATA_ARTIFACT_SELECTION.get()
    if session is not None:
        session.request_finished()
