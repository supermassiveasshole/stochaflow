"""No-replace bundle publication for completed sampling runs."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from stochaflow.data.artifact_io import (
    canonical_directory,
    publish_cache_directory,
    remove_cache_directory,
)


@dataclass(frozen=True, slots=True)
class SamplingPublicationStaging:
    """One private sibling directory awaiting atomic publication."""

    parent: Path
    staging: Path
    destination: Path


def create_sampling_publication_staging(
    output_dir: str | Path,
) -> SamplingPublicationStaging:
    """Create a private sibling directory for one absent final destination."""

    declared = Path(output_dir)
    if not declared.name:
        raise ValueError("sampling output directory must have a final name")
    declared.parent.mkdir(parents=True, exist_ok=True)
    parent = canonical_directory(
        declared.parent.resolve(),
        label="sampling output parent",
    )
    destination = parent / declared.name
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            f"sampling output directory already exists: {destination}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.sampling-",
            dir=parent,
        )
    ).resolve()
    return SamplingPublicationStaging(parent, staging, destination)


def publish_sampling_staging(value: SamplingPublicationStaging) -> Path:
    """Atomically publish a completed sibling staging directory once."""

    return publish_cache_directory(
        value.parent,
        value.staging,
        value.destination,
        label="sampling output publication",
    )


def abort_sampling_staging(value: SamplingPublicationStaging) -> None:
    """Remove an unpublished staging directory owned by this run."""

    try:
        value.staging.lstat()
    except FileNotFoundError:
        return
    remove_cache_directory(
        value.parent,
        value.staging,
        label="sampling output staging cleanup",
    )


__all__ = [
    "SamplingPublicationStaging",
    "abort_sampling_staging",
    "create_sampling_publication_staging",
    "publish_sampling_staging",
]
