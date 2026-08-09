"""Repository-wide pytest lifecycle guards."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Collection, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

import pytest

import stochaflow.utils.plugins as plugin_runtime


@pytest.fixture
def isolated_extension_activation_state() -> Generator[None, None, None]:
    """Isolate one test from process-wide extension activation state."""

    def reset() -> None:
        with plugin_runtime._activation_lock:
            plugin_runtime._activation_runtime.state = (
                plugin_runtime.PluginActivationState.UNACTIVATED
            )
            plugin_runtime._activation_runtime.selection = None
            plugin_runtime._activation_runtime.failure = None

    reset()
    try:
        yield
    finally:
        reset()


class MultiprocessingChild(Protocol):
    """Process operations used by the test-session leak guard."""

    name: str
    pid: int | None
    daemon: bool
    exitcode: int | None

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MultiprocessingChildLeakReport:
    """Identity and bounded-cleanup outcome for one leaked child."""

    name: str
    pid: int | None
    daemon: bool
    process_type: str
    cleanup: str
    exitcode: int | None
    errors: tuple[str, ...]

    def describe(self) -> str:
        """Render one actionable process identity for pytest output."""

        details = (
            f"name={self.name!r}, pid={self.pid}, daemon={self.daemon}, "
            f"type={self.process_type}, cleanup={self.cleanup}, "
            f"exitcode={self.exitcode}"
        )
        if self.errors:
            details += f", cleanup_errors={list(self.errors)!r}"
        return details


@dataclass(slots=True)
class MultiprocessingChildCleanupState:
    """Mutable cleanup bookkeeping retained independently of Process state."""

    process: MultiprocessingChild
    name: str
    pid: int | None
    daemon: bool
    process_type: str
    required_kill: bool = False
    errors: list[str] = field(default_factory=list)


def _record_cleanup_error(
    state: MultiprocessingChildCleanupState,
    action: str,
    error: Exception,
) -> None:
    state.errors.append(f"{action}: {type(error).__name__}: {error}")


def _is_alive(
    state: MultiprocessingChildCleanupState,
    *,
    action: str,
) -> bool:
    try:
        return state.process.is_alive()
    except Exception as error:  # noqa: BLE001
        # Teardown must continue through every child and report all failures.
        _record_cleanup_error(state, action, error)
        return True


def _join_until_deadline(
    states: Sequence[MultiprocessingChildCleanupState],
    *,
    wait_seconds: float,
    action: str,
) -> None:
    deadline = time.monotonic() + wait_seconds
    for state in states:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            state.process.join(timeout=remaining)
        except Exception as error:  # noqa: BLE001
            # A broken child must not prevent attempts to reap its siblings.
            _record_cleanup_error(state, action, error)


def cleanup_multiprocessing_children(
    baseline_object_ids: Collection[int],
    *,
    terminate_wait_seconds: float = 2.0,
    kill_wait_seconds: float = 2.0,
    active_children: Callable[
        [], Sequence[MultiprocessingChild]
    ]
    | None = None,
) -> tuple[MultiprocessingChildLeakReport, ...]:
    """Boundedly terminate, kill, and reap children created by this session."""

    children = (
        cast(
            Sequence[MultiprocessingChild],
            multiprocessing.active_children(),
        )
        if active_children is None
        else active_children()
    )
    states = tuple(
        MultiprocessingChildCleanupState(
            process=process,
            name=process.name,
            pid=process.pid,
            daemon=process.daemon,
            process_type=(
                f"{type(process).__module__}.{type(process).__qualname__}"
            ),
        )
        for process in children
        if id(process) not in baseline_object_ids
    )
    if not states:
        return ()

    for state in states:
        try:
            state.process.terminate()
        except Exception as error:  # noqa: BLE001
            _record_cleanup_error(state, "terminate", error)
    _join_until_deadline(
        states,
        wait_seconds=terminate_wait_seconds,
        action="join-after-terminate",
    )

    survivors = tuple(
        state
        for state in states
        if _is_alive(state, action="is-alive-after-terminate")
    )
    for state in survivors:
        state.required_kill = True
        try:
            state.process.kill()
        except Exception as error:  # noqa: BLE001
            _record_cleanup_error(state, "kill", error)
    _join_until_deadline(
        survivors,
        wait_seconds=kill_wait_seconds,
        action="join-after-kill",
    )

    reports: list[MultiprocessingChildLeakReport] = []
    for state in states:
        alive = _is_alive(state, action="is-alive-after-kill")
        if alive:
            cleanup = "still-alive"
        elif state.required_kill:
            cleanup = "killed-and-reaped"
        else:
            cleanup = "terminated-and-reaped"
        try:
            exitcode = state.process.exitcode
        except Exception as error:  # noqa: BLE001
            _record_cleanup_error(state, "read-exitcode", error)
            exitcode = None
        reports.append(
            MultiprocessingChildLeakReport(
                name=state.name,
                pid=state.pid,
                daemon=state.daemon,
                process_type=state.process_type,
                cleanup=cleanup,
                exitcode=exitcode,
                errors=tuple(state.errors),
            )
        )
    return tuple(reports)


def format_multiprocessing_leak_failure(
    reports: Sequence[MultiprocessingChildLeakReport],
) -> str:
    """Explain a session leak without attributing it to the final test."""

    identities = "\n".join(
        f"  - {report.describe()}" for report in reports
    )
    return (
        "pytest session leaked multiprocessing children; the last test shown "
        "is not necessarily their owner. Cleanup was attempted with bounded "
        f"waits:\n{identities}"
    )


@pytest.fixture(scope="session", autouse=True)
def multiprocessing_child_leak_guard() -> Iterator[None]:
    """Turn session-owned child-process leaks into bounded pytest failures."""

    baseline_children = tuple(multiprocessing.active_children())
    baseline_object_ids = frozenset(id(process) for process in baseline_children)
    yield
    reports = cleanup_multiprocessing_children(baseline_object_ids)
    if reports:
        pytest.fail(
            format_multiprocessing_leak_failure(reports),
            pytrace=False,
        )


@pytest.fixture
def multiprocessing_child_cleanup() -> Callable[
    ..., tuple[MultiprocessingChildLeakReport, ...]
]:
    """Expose the cleanup state machine for focused self-cleaning tests."""

    return cleanup_multiprocessing_children


@pytest.fixture
def multiprocessing_leak_failure_formatter() -> Callable[
    [Sequence[MultiprocessingChildLeakReport]], str
]:
    """Expose the actionable failure formatter to focused tests."""

    return format_multiprocessing_leak_failure
