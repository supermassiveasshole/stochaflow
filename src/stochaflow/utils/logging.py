"""Experiment logging backends and runtime logging helpers."""

import json
import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from stochaflow.utils.registry import REGISTRIES


def _normalize_scalar(value: Any) -> int | float | str | bool:
    """Convert common numeric scalar types into JSON/logging friendly values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if hasattr(value, "item") and callable(value.item):
        item = value.item()
        if isinstance(item, (bool, int, float, str)):
            return item
    return str(value)


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, int | float | str | bool]:
    """Normalize a metrics payload for backend fan-out."""

    return {name: _normalize_scalar(value) for name, value in metrics.items()}


def _sanitize_wandb_key(name: str) -> str:
    """Convert metric names into a W&B-safe identifier."""

    sanitized = []
    for char in name:
        if char.isalnum() or char == "_":
            sanitized.append(char)
        else:
            sanitized.append("_")
    key = "".join(sanitized).strip("_")
    if not key:
        key = "metric"
    if key[0].isdigit():
        key = f"metric_{key}"
    return key


class ExperimentLogger(ABC):
    """Backend-agnostic interface for experiment metric/event logging."""

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Record a resolved experiment configuration."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        """Record a flat metrics payload at a given global step."""

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        """Record textual information when the backend supports it."""

        del tag, text, step

    @abstractmethod
    def close(self) -> None:
        """Flush and close backend resources."""


class NullLogger(ExperimentLogger):
    """No-op logger used when logging is intentionally disabled."""

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del metrics, step

    def close(self) -> None:
        return None


class CompositeLogger(ExperimentLogger):
    """Fan-out logger that forwards events to multiple backends."""

    def __init__(self, backends: list[ExperimentLogger]) -> None:
        self.backends = backends

    def log_config(self, config: dict[str, Any]) -> None:
        for backend in self.backends:
            backend.log_config(config)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        for backend in self.backends:
            backend.log_metrics(metrics, step=step)

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        for backend in self.backends:
            backend.log_text(tag, text, step=step)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


@REGISTRIES.loggers.register("local")
class LocalLogger(ExperimentLogger):
    """Structured local logger with human-readable text logs and JSONL metrics."""

    def __init__(
        self,
        *,
        output_dir: str,
        run_name: str,
        console: bool = True,
        text_filename: str = "train.log",
        metrics_filename: str = "metrics.jsonl",
        append: bool = False,
    ) -> None:
        self.run_dir = Path(output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / metrics_filename
        file_mode = "a" if append else "w"
        self.metrics_handle = self.metrics_path.open(file_mode, encoding="utf-8")

        logger_name = f"stochaflow.local.{run_name}.{id(self)}"
        self.text_logger = logging.getLogger(logger_name)
        self.text_logger.setLevel(logging.INFO)
        self.text_logger.propagate = False
        self.text_logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(
            self.run_dir / text_filename,
            mode=file_mode,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        self.text_logger.addHandler(file_handler)

        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.text_logger.addHandler(stream_handler)

    def log_config(self, config: dict[str, Any]) -> None:
        self.log_text("config", json.dumps(config, indent=2, sort_keys=True))

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        normalized = _normalize_metrics(metrics)
        payload = {"step": step, "metrics": normalized}
        self.metrics_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.metrics_handle.flush()
        metric_text = ", ".join(f"{name}={value}" for name, value in normalized.items())
        self.text_logger.info("step=%s | %s", step, metric_text)

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        prefix = f"{tag}"
        if step is not None:
            prefix = f"{prefix} step={step}"
        self.text_logger.info("%s | %s", prefix, text)

    def close(self) -> None:
        self.metrics_handle.close()
        for handler in list(self.text_logger.handlers):
            handler.close()
            self.text_logger.removeHandler(handler)


@REGISTRIES.loggers.register("tensorboard")
class TensorBoardLogger(ExperimentLogger):
    """TensorBoard backend for scalar and text summaries."""

    def __init__(
        self,
        *,
        output_dir: str,
        run_name: str,
        subdir: str = "tensorboard",
    ) -> None:
        from torch.utils.tensorboard.writer import SummaryWriter

        log_dir = Path(output_dir) / subdir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_config(self, config: dict[str, Any]) -> None:
        self.writer.add_text("config", json.dumps(config, indent=2, sort_keys=True))

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        normalized = _normalize_metrics(metrics)
        for name, value in normalized.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(name, value, step)
            else:
                self.writer.add_text(name, str(value), step)

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        global_step = 0 if step is None else step
        self.writer.add_text(tag, text, global_step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


@REGISTRIES.loggers.register("wandb")
class WandbLogger(ExperimentLogger):
    """Weights & Biases backend for experiment tracking."""

    def __init__(
        self,
        *,
        output_dir: str,
        run_name: str,
        project: str = "stochaflow",
        entity: str | None = None,
        mode: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError(
                "wandb backend requested but the 'wandb' package is not installed"
            ) from exc

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            dir=output_dir,
            mode=mode,
            tags=tags,
        )

    def log_config(self, config: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        normalized = _normalize_metrics(metrics)
        sanitized = {
            _sanitize_wandb_key(name): value for name, value in normalized.items()
        }
        self._wandb.log(sanitized, step=step)

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        payload = {tag: text}
        if step is None:
            self._wandb.log(payload)
        else:
            self._wandb.log(payload, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


def configure_torch_logging(settings: dict[str, Any] | None) -> None:
    """Apply optional ``torch._logging`` runtime diagnostics configuration."""

    if not settings:
        return

    try:
        torch_logging = importlib.import_module("torch._logging")
    except ImportError as exc:
        raise RuntimeError("torch._logging is not available in this PyTorch build") from exc

    set_logs = getattr(torch_logging, "set_logs", None)
    if not callable(set_logs):
        raise RuntimeError("torch._logging.set_logs is not available in this PyTorch build")

    converted: dict[str, Any] = {}
    for name, value in settings.items():
        if isinstance(value, str):
            upper = value.upper()
            if hasattr(logging, upper):
                converted[name] = getattr(logging, upper)
                continue
        converted[name] = value
    set_logs(**converted)
