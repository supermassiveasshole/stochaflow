"""Lightweight logging contract without built-in backend registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from stochaflow.utils.registry import REGISTRIES


class ExperimentLogger(ABC):
    """Observation-only interface for experiment metric and artifact logging.

    Logger instances, open files, and backend writers are runtime resources, not
    checkpoint state. Every training invocation constructs and closes its own
    logger resources in that invocation's output directory.
    """

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """Record a resolved experiment configuration."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        """Record a flat metrics payload at a given global step."""

    def log_text(self, tag: str, text: str, *, step: int | None = None) -> None:
        """Record textual information when the backend supports it."""

        del tag, text, step

    def log_image(
        self,
        tag: str,
        path: str | Path,
        *,
        step: int,
        caption: str | None = None,
    ) -> None:
        """Record an image artifact when the backend supports it."""

        del tag, path, step, caption

    @abstractmethod
    def close(self) -> None:
        """Flush and close backend resources."""


class NullLogger(ExperimentLogger):
    """No-op logger used when logging is intentionally disabled."""

    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        del metrics, step

    def log_image(
        self,
        tag: str,
        path: str | Path,
        *,
        step: int,
        caption: str | None = None,
    ) -> None:
        del tag, path, step, caption

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

    def log_image(
        self,
        tag: str,
        path: str | Path,
        *,
        step: int,
        caption: str | None = None,
    ) -> None:
        for backend in self.backends:
            backend.log_image(tag, path, step=step, caption=caption)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


# The base requirement is part of the public extension contract. Importing the
# lightweight contract installs that requirement without registering a backend.
REGISTRIES.loggers.require_base(ExperimentLogger)


# Preserve the established public class identities while keeping registration
# out of contract-only import paths.
ExperimentLogger.__module__ = "stochaflow.utils.logging"
NullLogger.__module__ = "stochaflow.utils.logging"
CompositeLogger.__module__ = "stochaflow.utils.logging"


__all__ = ["CompositeLogger", "ExperimentLogger", "NullLogger"]
