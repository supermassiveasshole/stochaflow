"""AFHQ-v2 source registrations and private parameter parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from stochaflow.extensions import (
    IMAGE_DATA_SOURCES,
    ClassLabeledImageFolderArtifactPayload,
    ConfigError,
    DataArtifact,
    DataSourceContext,
    ImageDataSource,
    ImageFolderArtifactPayload,
)
from stochaflow_afhq_v2.artifact import (
    AFHQV2_SOURCE_NAME,
    materialize_afhq_v2_artifact,
)
from stochaflow_afhq_v2.dog_artifact import (
    AFHQV2_DOG_SOURCE_NAME,
    materialize_afhq_v2_dog_artifact,
)

_PARAM_FIELDS = {"archive", "downloader", "lock_file", "resolution"}


def _params(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ConfigError(f"{path} field names must be strings")
    unknown = sorted(cast(set[str], set(raw)) - _PARAM_FIELDS)
    if unknown:
        fields = ", ".join(f"{path}.{name}" for name in unknown)
        raise ConfigError(f"unknown config field(s): {fields}")
    return cast(dict[str, Any], dict(value))


def _optional_path(value: object, *, path: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty path string")
    return Path(value)


def _downloader(
    value: object,
    *,
    path: str,
) -> Literal["auto", "curl", "python"]:
    if not isinstance(value, str) or value not in {
        "auto",
        "curl",
        "python",
    }:
        raise ConfigError(f"{path} must be auto, curl, or python")
    return cast(Literal["auto", "curl", "python"], value)


def _resolution(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")
    return value


@IMAGE_DATA_SOURCES.register(AFHQV2_SOURCE_NAME)
class AFHQV2ImageDataSource(ImageDataSource):
    """Materialize the source-locked official AFHQ-v2 image artifact."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ClassLabeledImageFolderArtifactPayload]:
        """Acquire or verify the official source and expose official splits."""

        path = f"{self.config_path}.params"
        raw = _params(self.params, path=path)
        return materialize_afhq_v2_artifact(
            context,
            archive=_optional_path(
                raw.get("archive"),
                path=f"{path}.archive",
            ),
            downloader=_downloader(
                raw.get("downloader", "auto"),
                path=f"{path}.downloader",
            ),
            lock_file=_optional_path(
                raw.get("lock_file"),
                path=f"{path}.lock_file",
            ),
            resolution=_resolution(
                raw.get("resolution", 128),
                path=f"{path}.resolution",
            ),
        )


@IMAGE_DATA_SOURCES.register(AFHQV2_DOG_SOURCE_NAME)
class AFHQV2DogImageDataSource(ImageDataSource):
    """Materialize the unlabeled 256px AFHQ-v2 train/dog benchmark."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ImageFolderArtifactPayload]:
        """Acquire or verify the train/dog-only benchmark artifact."""

        path = f"{self.config_path}.params"
        raw = _params(self.params, path=path)
        return materialize_afhq_v2_dog_artifact(
            context,
            archive=_optional_path(
                raw.get("archive"),
                path=f"{path}.archive",
            ),
            downloader=_downloader(
                raw.get("downloader", "auto"),
                path=f"{path}.downloader",
            ),
            lock_file=_optional_path(
                raw.get("lock_file"),
                path=f"{path}.lock_file",
            ),
            resolution=_resolution(
                raw.get("resolution", 256),
                path=f"{path}.resolution",
            ),
        )


__all__ = ["AFHQV2DogImageDataSource", "AFHQV2ImageDataSource"]
