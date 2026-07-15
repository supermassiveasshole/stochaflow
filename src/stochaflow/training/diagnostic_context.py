"""Construction context for diagnostics with optional runtime dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stochaflow.utils.logging import ExperimentLogger


@dataclass(frozen=True, slots=True)
class DiagnosticBuildContext:
    """Runtime values available to context-aware diagnostic classes."""

    logger: ExperimentLogger
    output_dir: str | Path
    sample_shape: tuple[int, int, int]


class ContextAwareDiagnostic(Protocol):
    """Optional class-level hook for requesting diagnostic constructor values."""

    @classmethod
    def context_parameters(
        cls,
        context: DiagnosticBuildContext,
    ) -> Mapping[str, Any]:
        """Return constructor parameters derived from the runtime context."""

        ...


__all__ = ["ContextAwareDiagnostic", "DiagnosticBuildContext"]
