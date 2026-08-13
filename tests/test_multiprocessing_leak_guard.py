"""Tests for the pytest-session multiprocessing leak guard."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TerminateRequiredProcess:
    """Process test double that exits when terminate is requested.

    A real ``spawn`` process also starts a resource-tracker subprocess that is
    intentionally absent from ``multiprocessing.active_children()``. Starting
    that infrastructure here would make the guard's own unit test capable of
    leaking an unobservable process during interpreter shutdown.
    """

    def __init__(self) -> None:
        self.name = "terminate-required-worker"
        self.pid = 3131
        self.daemon = False
        self.exitcode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_timeouts: list[float | None] = []
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False
        self.exitcode = -9


class KillRequiredProcess:
    """Process test double that ignores terminate and exits after kill."""

    def __init__(self) -> None:
        self.name = "kill-required-worker"
        self.pid = 4242
        self.daemon = True
        self.exitcode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_timeouts: list[float | None] = []
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False
        self.exitcode = -9


def test_github_failure_annotation_escapes_workflow_commands(
    github_failure_annotation_formatter: Callable[..., str],
) -> None:
    annotation = github_failure_annotation_formatter(
        nodeid="tests/test_example.py::test_value[param,other]",
        summary="tests/test_example.py:12: bad: value, 50%\nnext line",
    )

    assert annotation == (
        "::error title=tests/test_example.py%3A%3Atest_value[param%2Cother]::"
        "tests/test_example.py:12: bad: value, 50%25%0Anext line"
    )


def test_cleanup_terminates_and_reaps_child(
    multiprocessing_child_cleanup: Callable[..., tuple[Any, ...]],
) -> None:
    process = TerminateRequiredProcess()

    reports = multiprocessing_child_cleanup(
        frozenset(),
        terminate_wait_seconds=0.0,
        kill_wait_seconds=0.0,
        active_children=lambda: (process,),
    )

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.join_timeouts == [0.0]
    assert len(reports) == 1
    report = reports[0]
    assert report.name == "terminate-required-worker"
    assert report.pid == 3131
    assert report.cleanup == "terminated-and-reaped"
    assert report.exitcode == -15
    description = report.describe()
    assert "pid=3131" in description
    assert "cleanup=terminated-and-reaped" in description


def test_cleanup_escalates_to_kill_with_shared_bounded_waits(
    multiprocessing_child_cleanup: Callable[..., tuple[Any, ...]],
    multiprocessing_leak_failure_formatter: Callable[[Any], str],
) -> None:
    process = KillRequiredProcess()

    reports = multiprocessing_child_cleanup(
        frozenset(),
        terminate_wait_seconds=0.0,
        kill_wait_seconds=0.0,
        active_children=lambda: (process,),
    )

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_timeouts == [0.0, 0.0]
    assert len(reports) == 1
    report = reports[0]
    assert report.cleanup == "killed-and-reaped"
    assert report.exitcode == -9
    assert "name='kill-required-worker'" in report.describe()
    failure = multiprocessing_leak_failure_formatter(reports)
    assert "pytest session leaked multiprocessing children" in failure
    assert "last test shown is not necessarily their owner" in failure
    assert "name='kill-required-worker', pid=4242" in failure
