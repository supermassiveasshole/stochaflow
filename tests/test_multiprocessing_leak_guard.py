"""Tests for the pytest-session multiprocessing leak guard."""

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable
from typing import Any


def wait_until_terminated() -> None:
    """Remain alive until the parent exercises bounded process cleanup."""

    time.sleep(60.0)


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


def test_cleanup_terminates_and_reaps_real_spawned_child(
    multiprocessing_child_cleanup: Callable[..., tuple[Any, ...]],
) -> None:
    baseline_object_ids = frozenset(
        id(process) for process in multiprocessing.active_children()
    )
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=wait_until_terminated,
        name="intentional-leak-guard-worker",
    )
    process.start()
    try:
        reports = multiprocessing_child_cleanup(
            baseline_object_ids,
            terminate_wait_seconds=2.0,
            kill_wait_seconds=2.0,
        )

        assert len(reports) == 1
        report = reports[0]
        assert report.name == "intentional-leak-guard-worker"
        assert report.pid == process.pid
        assert report.cleanup == "terminated-and-reaped"
        assert report.exitcode is not None
        assert not process.is_alive()
        description = report.describe()
        assert f"pid={process.pid}" in description
        assert "cleanup=terminated-and-reaped" in description
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=2.0)


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
