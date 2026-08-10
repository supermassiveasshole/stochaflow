"""Torch diagnostic logging configuration without backend registration."""

import importlib
import logging
from typing import Any


def configure_torch_logging(settings: dict[str, Any] | None) -> None:
    """Apply optional ``torch._logging`` runtime diagnostics configuration."""

    if not settings:
        return

    try:
        torch_logging = importlib.import_module("torch._logging")
    except ImportError as exc:
        raise RuntimeError(
            "torch._logging is not available in this PyTorch build"
        ) from exc

    set_logs = getattr(torch_logging, "set_logs", None)
    if set_logs is None:
        raise RuntimeError(
            "torch._logging.set_logs is not available in this PyTorch build"
        )
    if not callable(set_logs):
        raise TypeError("torch._logging.set_logs must be callable")

    converted: dict[str, Any] = {}
    for name, value in settings.items():
        if isinstance(value, str):
            upper = value.upper()
            if hasattr(logging, upper):
                converted[name] = getattr(logging, upper)
                continue
        converted[name] = value
    set_logs(**converted)


__all__ = ["configure_torch_logging"]
