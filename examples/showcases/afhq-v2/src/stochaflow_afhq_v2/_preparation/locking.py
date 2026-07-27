"""AFHQ-v2 error adaptation for the framework materialization lock."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Self

from stochaflow.extensions import ArtifactMaterializationLock

from .contracts import PreparationError


class ArtifactPreparationLock(AbstractContextManager["ArtifactPreparationLock"]):
    """Translate framework materialization-lock failures for the preparer."""

    def __init__(
        self,
        path: Path,
        *,
        cache_root: Path | None = None,
        timeout_seconds: float = 120.0,
        poll_seconds: float = 0.1,
    ) -> None:
        self._lock = ArtifactMaterializationLock(
            path,
            cache_root=cache_root,
            wait_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )

    def __enter__(self) -> Self:
        try:
            self._lock.__enter__()
        except (OSError, RuntimeError, ValueError) as error:
            raise PreparationError(str(error)) from error
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock.__exit__(*exc_info)
