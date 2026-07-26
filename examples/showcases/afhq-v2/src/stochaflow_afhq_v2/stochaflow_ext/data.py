"""Official, source-locked AFHQ-v2 image artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from stochaflow.extensions import (
    IMAGE_DATA_SOURCES,
    DataSourceContext,
    ImageDataSource,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
)
from stochaflow_afhq_v2.preparation import (
    PreparationError,
    PreparedArtifact,
    acquire_official_archive,
    build_preparation_plan,
    load_source_lock,
    prepare_archive,
    require_prepared_artifact,
)

_SOURCE_NAME = "afhq-v2.official"
_ARTIFACT_TYPE = "stochaflow.image-folder.v1"
_DEFAULT_LOCK = (
    Path(__file__).resolve().parents[1] / "resources" / "afhq-v2.lock.yaml"
)
_DEFAULT_VALIDATION_SEED = "stochaflow-afhq-v2-validation-v1"


@dataclass(frozen=True, slots=True)
class AFHQV2DataSourceConfig:
    """Strict private configuration for the official AFHQ-v2 source."""

    archive: Path | None = None
    downloader: str = "auto"
    lock_file: Path | None = None
    resolution: int = 128
    validation_per_class: int = 300
    validation_seed: str = _DEFAULT_VALIDATION_SEED

    @classmethod
    def from_params(
        cls,
        raw: dict[str, Any],
        *,
        path: str,
    ) -> AFHQV2DataSourceConfig:
        """Parse source parameters and reject every undeclared field."""

        allowed = {
            "archive",
            "downloader",
            "lock_file",
            "resolution",
            "validation_per_class",
            "validation_seed",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            fields = ", ".join(f"{path}.{name}" for name in unknown)
            raise ValueError(f"unknown AFHQ-v2 source field(s): {fields}")

        def optional_path(name: str) -> Path | None:
            value = raw.get(name)
            if value is None:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}.{name} must be a non-empty path string")
            return Path(value).resolve()

        config = cls(
            archive=optional_path("archive"),
            downloader=cast(str, raw.get("downloader", "auto")),
            lock_file=optional_path("lock_file"),
            resolution=cast(int, raw.get("resolution", 128)),
            validation_per_class=cast(
                int,
                raw.get("validation_per_class", 300),
            ),
            validation_seed=cast(
                str,
                raw.get("validation_seed", _DEFAULT_VALIDATION_SEED),
            ),
        )
        config.validate(path=path)
        return config

    def validate(self, *, path: str) -> None:
        """Validate value constraints with precise configuration paths."""

        for name in ("resolution", "validation_per_class"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{path}.{name} must be a positive integer")
        if type(self.validation_seed) is not str or not self.validation_seed:
            raise ValueError(f"{path}.validation_seed must be a non-empty string")
        downloader = cast(object, self.downloader)
        if not isinstance(downloader, str) or downloader not in {
            "auto",
            "curl",
            "python",
        }:
            raise ValueError(
                f"{path}.downloader must be auto, curl, or python"
            )


def _image_folder_payload(
    prepared: PreparedArtifact,
) -> ImageFolderArtifactPayload:
    roots = {
        role: prepared.root / role
        for role in ("train", "validation", "test")
    }
    partitions: dict[str, list[ImageFileRecord]] = {
        role: [] for role in roots
    }
    for record in prepared.image_records:
        parts = PurePosixPath(record.relative_path).parts
        if len(parts) != 3 or parts[0] not in partitions:
            raise PreparationError(
                f"prepared inventory has an invalid image path: "
                f"{record.relative_path!r}"
            )
        role = parts[0]
        partitions[role].append(
            ImageFileRecord(
                tree=role,
                path=PurePosixPath(*parts[1:]).as_posix(),
                size_bytes=record.size_bytes,
                sha256=record.sha256,
            )
        )
    return ImageFolderArtifactPayload(
        roots=roots,
        train=tuple(partitions["train"]),
        validation=tuple(partitions["validation"]),
        test=tuple(partitions["test"]),
    )


def _expected_identity(
    context: DataSourceContext,
) -> ManagedDataArtifactIdentity | None:
    expected = context.expected_identity
    if expected is None:
        return None
    if not isinstance(expected, ManagedDataArtifactIdentity):
        raise TypeError(
            "AFHQ-v2 strict resume requires a managed data artifact identity"
        )
    if expected.source_name != _SOURCE_NAME:
        raise ValueError(
            "AFHQ-v2 strict resume expected a different data source"
        )
    if expected.artifact_type != _ARTIFACT_TYPE:
        raise ValueError(
            "AFHQ-v2 strict resume expected an incompatible artifact type"
        )
    return expected


@IMAGE_DATA_SOURCES.register(_SOURCE_NAME)
class AFHQV2ImageDataSource(ImageDataSource):
    """Materialize the complete official AFHQ-v2 archive."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ManagedDataArtifact[ImageFolderArtifactPayload]:
        expected_identity = _expected_identity(context)
        config = AFHQV2DataSourceConfig.from_params(
            self.params,
            path=f"{self.config_path}.params",
        )
        lock = load_source_lock(config.lock_file or _DEFAULT_LOCK)
        plan = build_preparation_plan(
            lock=lock,
            resolution=config.resolution,
            validation_per_class=config.validation_per_class,
            validation_seed=config.validation_seed,
        )
        if lock.expected_sha256 is None:
            raise PreparationError("AFHQ-v2 source lock is missing SHA-256")
        materializer_name = cast(str, plan.recipe["id"])
        if expected_identity is not None:
            if expected_identity.source_digest != lock.expected_sha256:
                raise ValueError(
                    "AFHQ-v2 strict resume expected a different source digest"
                )
            if expected_identity.materializer_name != materializer_name:
                raise ValueError(
                    "AFHQ-v2 strict resume expected a different materializer"
                )
            if (
                expected_identity.materialization_digest
                != plan.recipe_sha256
            ):
                raise ValueError(
                    "AFHQ-v2 strict resume expected a different preparation "
                    "recipe"
                )
        if context.policy == "require":
            prepared = require_prepared_artifact(
                lock=lock,
                cache_root=context.cache_root,
                resolution=config.resolution,
                validation_per_class=config.validation_per_class,
                validation_seed=config.validation_seed,
                full=context.verification == "full",
            )
        else:
            try:
                prepared = require_prepared_artifact(
                    lock=lock,
                    cache_root=context.cache_root,
                    resolution=config.resolution,
                    validation_per_class=config.validation_per_class,
                    validation_seed=config.validation_seed,
                    full=context.verification == "full",
                )
            except PreparationError:
                prepared = None
        if prepared is None:
            source = acquire_official_archive(
                lock=lock,
                cache_root=context.cache_root,
                proxy=None,
                archive_override=config.archive,
                downloader=config.downloader,
            )
            prepared = prepare_archive(
                source=source,
                lock=lock,
                cache_root=context.cache_root,
                resolution=config.resolution,
                validation_per_class=config.validation_per_class,
                validation_seed=config.validation_seed,
                repair_invalid=True,
            )
        identity = ManagedDataArtifactIdentity(
            artifact_type=_ARTIFACT_TYPE,
            source_name=_SOURCE_NAME,
            source_digest=lock.expected_sha256,
            materializer_name=materializer_name,
            materialization_digest=plan.recipe_sha256,
            artifact_digest=prepared.artifact_digest,
            manifest_sha256=prepared.manifest_sha256,
        )
        if expected_identity is not None and identity != expected_identity:
            raise ValueError(
                "AFHQ-v2 strict resume data artifact identity does not match"
            )
        return ManagedDataArtifact(
            artifact_root=prepared.root,
            manifest_path=prepared.manifest_path,
            identity=identity,
            payload=_image_folder_payload(prepared),
        )


__all__ = ["AFHQV2DataSourceConfig", "AFHQV2ImageDataSource"]
