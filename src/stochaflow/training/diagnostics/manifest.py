"""Collision-safe artifact allocation and epoch manifest persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from stochaflow.training.diagnostics.contracts import ArtifactRecord


class EpochArtifactStore:
    """Own artifact paths and manifest records for one diagnostic epoch."""

    def __init__(self, root: Path, epoch_index: int) -> None:
        self.epoch_dir = root / f"epoch_{epoch_index:04d}"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        self._reserved: set[Path] = set()
        self._artifacts: list[dict[str, Any]] = []
        self._errors: list[dict[str, str]] = []

    def reserve(self, relative_path: str | Path) -> Path:
        """Reserve a safe relative path and reject provider collisions."""

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative == Path():
            raise ValueError(f"artifact path must be a safe relative path: {relative}")
        if relative in self._reserved:
            raise ValueError(f"diagnostic artifact path collision: {relative}")
        self._reserved.add(relative)
        target = self.epoch_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def record(self, provider: str, records: Sequence[ArtifactRecord]) -> None:
        """Add provider artifact metadata to the pending manifest."""

        for record in records:
            try:
                relative = record.path.relative_to(self.epoch_dir)
            except ValueError as exc:
                raise ValueError(
                    f"provider '{provider}' returned an artifact outside "
                    f"the epoch directory: {record.path}"
                ) from exc
            if relative not in self._reserved:
                raise ValueError(
                    f"provider '{provider}' returned an unreserved artifact: {relative}"
                )
            self._artifacts.append(
                {
                    "provider": provider,
                    "kind": record.kind,
                    "path": str(relative),
                    "image_tag": record.image_tag,
                    "caption": record.caption,
                }
            )

    def record_error(self, *, phase: str, provider: str, error: Exception) -> None:
        """Record one isolated runtime failure for ``warn`` policy."""

        self._errors.append(
            {
                "phase": phase,
                "provider": provider,
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    def write_manifest(self, payload: Mapping[str, Any]) -> Path:
        """Write the final manifest after all provider work completes."""

        path = self.reserve("manifest.yaml")
        document = {
            **dict(payload),
            "artifacts": self._artifacts,
            "errors": self._errors,
        }
        path.write_text(
            yaml.safe_dump(document, sort_keys=False),
            encoding="utf-8",
        )
        return path


__all__ = ["EpochArtifactStore"]
