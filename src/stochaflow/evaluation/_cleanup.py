"""Internal best-effort cleanup support for Evaluation lifecycles."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from torch import nn

type CleanupAction = tuple[str, Callable[[], None]]
type CleanupFailure = tuple[str, BaseException]


def restore_module_mode(module: nn.Module, training: bool) -> None:
    """Restore one module mode while presenting a cleanup-only signature."""

    module.train(training)


def run_cleanup_actions(
    actions: Iterable[CleanupAction],
) -> tuple[CleanupFailure, ...]:
    """Run every cleanup action and retain failures in execution order."""

    failures: list[CleanupFailure] = []
    for label, action in actions:
        try:
            action()
        except BaseException as error:  # noqa: BLE001
            failures.append((label, error))
    return tuple(failures)


def add_cleanup_failure_notes(
    primary: BaseException,
    failures: Iterable[CleanupFailure],
) -> None:
    """Attach cleanup failures without replacing the primary exception."""

    for label, error in failures:
        try:
            detail = str(error)
        except BaseException:  # noqa: BLE001
            detail = "<exception text unavailable>"
        BaseException.add_note(
            primary,
            f"{label}: {type(error).__name__}: {detail}",
        )


def first_cleanup_failure(
    failures: tuple[CleanupFailure, ...],
) -> BaseException | None:
    """Return the first failure after attaching every later failure to it."""

    if not failures:
        return None
    first_label, primary = failures[0]
    BaseException.add_note(primary, f"primary cleanup action: {first_label}")
    add_cleanup_failure_notes(primary, failures[1:])
    return primary
