"""Terminal reporters for human-facing training progress."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
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
from rich.table import Table


def _format_loss(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _format_path(path: str | Path | None) -> str:
    if path is None:
        return "-"
    return str(path)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.0f}s"


@dataclass(slots=True)
class RunSummary:
    """High-level metadata shown when a training run starts."""

    experiment_name: str
    exp_id: str | None
    device: str
    output_dir: str | Path
    train_size: int | None
    valid_size: int | None
    test_size: int | None
    batch_size: int | None


@dataclass(slots=True)
class FinalSummary:
    """High-level metadata shown when a training run finishes."""

    best_epoch: int | None
    best_metric_name: str | None
    best_metric_value: float | None
    test_loss: float | None
    stopped_early: bool
    best_checkpoint: str | Path | None
    selected_checkpoint: str | Path | None
    selected_checkpoint_kind: str | None
    output_dir: str | Path
    metrics_path: str | Path
    log_path: str | Path
    artifacts: Mapping[str, str | Path] | None = None


class RichTrainingReporter:
    """Rich terminal UI for training progress and summaries."""

    def __init__(self, *, console: Console | None = None) -> None:
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def on_run_start(self, summary: RunSummary) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        table.add_row("Experiment", summary.experiment_name)
        table.add_row("exp_id", summary.exp_id or "-")
        table.add_row("Device", summary.device)
        table.add_row("Output", str(summary.output_dir))
        dataset_parts = [
            f"train={summary.train_size if summary.train_size is not None else '-'}"
        ]
        if summary.valid_size is not None:
            dataset_parts.append(f"valid={summary.valid_size}")
        if summary.test_size is not None:
            dataset_parts.append(f"test={summary.test_size}")
        table.add_row("Dataset", " ".join(dataset_parts))
        table.add_row(
            "Batch size",
            str(summary.batch_size) if summary.batch_size is not None else "-",
        )
        self.console.print(Panel(table, title="Stochaflow Training", border_style="cyan"))

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.console.rule(f"[bold]Epoch {epoch}/{total_epochs}")

    def on_phase_start(
        self,
        *,
        phase: str,
        epoch: int | None,
        total_batches: int | None,
        enabled: bool = True,
    ) -> None:
        self._stop_progress()
        if not enabled:
            return
        description = phase.capitalize()
        if epoch is not None:
            description = f"{description} {epoch}"
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]}"),
            TextColumn("avg={task.fields[avg_loss]}"),
            TextColumn("lr={task.fields[lr]}"),
            TextColumn("step={task.fields[step]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            description,
            total=total_batches,
            loss="-",
            avg_loss="-",
            lr="-",
            step="-",
        )

    def on_batch_end(
        self,
        *,
        phase: str,
        loss: float,
        avg_loss: float,
        lr: float | None,
        global_step: int,
    ) -> None:
        del phase
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(
            self._task_id,
            advance=1,
            loss=f"{loss:.4f}",
            avg_loss=f"{avg_loss:.4f}",
            lr="-" if lr is None else f"{lr:.2e}",
            step=str(global_step),
        )

    def on_phase_end(self) -> None:
        self._stop_progress()

    def on_epoch_end(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        valid_loss: float | None,
        best_metric_value: float | None,
        lr: float | None,
        train_batches: int,
        valid_batches: int | None,
        epoch_time: float,
        status: str,
    ) -> None:
        table = self._build_epoch_table()
        table.add_row(
            f"{epoch}/{total_epochs}",
            _format_loss(train_loss),
            _format_loss(valid_loss),
            _format_loss(best_metric_value),
            "-" if lr is None else f"{lr:.2e}",
            str(train_batches),
            "-" if valid_batches is None else str(valid_batches),
            _format_seconds(epoch_time),
            status,
        )
        self.console.print(table)

    def on_early_stopping(self, text: str) -> None:
        self.console.print(Panel(text, title="Early Stopping", border_style="yellow"))

    def on_run_end(self, summary: FinalSummary) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(ratio=1, overflow="fold")
        table.add_row(
            "Best epoch",
            "-" if summary.best_epoch is None else str(summary.best_epoch),
        )
        table.add_row("Best metric", summary.best_metric_name or "-")
        table.add_row("Best value", _format_loss(summary.best_metric_value))
        table.add_row("Test loss", _format_loss(summary.test_loss))
        table.add_row("Stopped early", str(summary.stopped_early))
        table.add_row("Best checkpoint", _format_path(summary.best_checkpoint))
        if summary.selected_checkpoint is not None:
            table.add_row(
                "Selected checkpoint",
                _format_path(summary.selected_checkpoint),
            )
            table.add_row(
                "Selection kind",
                summary.selected_checkpoint_kind or "-",
            )
        table.add_row("Output", _format_path(summary.output_dir))
        table.add_row("Metrics", _format_path(summary.metrics_path))
        table.add_row("Log", _format_path(summary.log_path))
        if summary.artifacts:
            for name, path in summary.artifacts.items():
                table.add_row(name, _format_path(path))
        self.console.print(Panel(table, title="Run Summary", border_style="green"))

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task_id = None

    @staticmethod
    def _build_epoch_table() -> Table:
        table = Table(title="Epoch Summary", show_lines=False)
        table.add_column("Ep", justify="right", style="bold", no_wrap=True)
        table.add_column("TrLoss", justify="right", no_wrap=True)
        table.add_column("VaLoss", justify="right", no_wrap=True)
        table.add_column("Best", justify="right", no_wrap=True)
        table.add_column("LR", justify="right", no_wrap=True)
        table.add_column("TrB", justify="right", no_wrap=True)
        table.add_column("VaB", justify="right", no_wrap=True)
        table.add_column("Time", justify="right", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        return table
