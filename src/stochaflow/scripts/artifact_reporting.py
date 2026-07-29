"""Terminal progress for framework-owned data artifact verification."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from stochaflow.data import ArtifactVerificationEvent


class RichArtifactVerificationReporter:
    """Render ordered artifact verification events without entering data I/O."""

    def __init__(self, *, console: Console | None = None) -> None:
        self.console = console or Console(stderr=True)
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._operation: tuple[str, str] | None = None

    def observe(self, event: ArtifactVerificationEvent) -> None:
        """Render one verification event."""

        operation = (
            event.source_name,
            event.materializer_name,
        )
        if event.completed == 0 or operation != self._operation:
            self.close()
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"Verifying artifact files ({escape(event.source_name)})",
                total=event.total,
            )
            self._operation = operation
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(
            self._task_id,
            completed=event.completed,
            total=event.total,
        )
        if event.completed == event.total:
            self.close()

    def close(self) -> None:
        """Stop any active transient display."""

        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task_id = None
        self._operation = None


__all__ = ["RichArtifactVerificationReporter"]
