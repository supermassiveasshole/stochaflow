"""Image source registry composition and strict artifact bindings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from stochaflow.data.artifacts import (
    DataArtifact,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
)
from stochaflow.data.folder_sources import (
    ImageFolderDataSource,
    PairedImageFolderDataSource,
)
from stochaflow.data.image_contracts import (
    IMAGE_DATA_SOURCES,
    ClassLabeledImageFolderArtifactPayload,
    ImageArtifactPayload,
    ImageDataSource,
    ImageFolderArtifactPayload,
    PairedImageFolderArtifactPayload,
    TorchvisionImageArtifactPayload,
)
from stochaflow.data.recipe_config import ImageSourceConfig
from stochaflow.data.torchvision_source import TorchvisionImageDataSource
from stochaflow.utils.registry import Registry

if TYPE_CHECKING:
    from stochaflow.data.builder import DataBuilderContext

_BUILTIN_IMAGE_DATA_SOURCE_TYPES = (
    TorchvisionImageDataSource,
    ImageFolderDataSource,
    PairedImageFolderDataSource,
)


class ImageSourceFactory:
    """Materialize sources and validate the image-family artifact envelope.

    Recipe-specific payload and lifecycle compatibility remains the consuming
    DataBuilder's responsibility.
    """

    def __init__(
        self,
        registry: Registry[type[ImageDataSource]] = IMAGE_DATA_SOURCES,
    ) -> None:
        self.registry = registry

    def materialize(
        self,
        config: ImageSourceConfig,
        *,
        binding_id: str,
        builder_context: DataBuilderContext,
        path: str,
    ) -> DataArtifact[ImageArtifactPayload]:
        """Materialize one source and enforce any strict-resume identity."""

        config.validate(path=path)
        expected: DataArtifactIdentity | None = None
        if builder_context.strict_resume:
            if builder_context.expected_artifacts is None:
                raise ValueError(
                    "strict resume checkpoint is missing data artifact identities"
                )
            expected = builder_context.expected_artifacts.identity_for(binding_id)
            if expected.source_name != config.name:
                raise ValueError(
                    "strict resume expected a different registered data source"
                )
        source = self.registry.create(
            config.name,
            config.params,
            config_path=path,
        )
        context = config.materialization.context(
            expected_identity=expected,
            path=f"{path}.materialization",
        )
        artifact = source.materialize(context)
        if not isinstance(artifact, DataArtifact):
            raise TypeError(
                f"image data source '{config.name}' must return DataArtifact"
            )
        if not isinstance(
            artifact.payload,
            (
                ClassLabeledImageFolderArtifactPayload,
                TorchvisionImageArtifactPayload,
                ImageFolderArtifactPayload,
                PairedImageFolderArtifactPayload,
            ),
        ):
            raise TypeError(
                f"image data source '{config.name}' returned an incompatible payload"
            )
        if artifact.identity.source_name != config.name:
            raise ValueError(
                f"image source '{config.name}' returned identity for "
                f"'{artifact.identity.source_name}'"
            )
        if expected is not None and artifact.identity != expected:
            raise ValueError(
                "strict resume data artifact identity does not match"
            )
        return artifact


def artifact_bindings(
    artifacts: Sequence[tuple[str, DataArtifact[Any]]],
) -> DataArtifactBindings:
    """Build the canonical identities selected by an image recipe."""

    return DataArtifactBindings(
        tuple(
            DataArtifactBinding(id=binding_id, identity=artifact.identity)
            for binding_id, artifact in artifacts
        )
    )


__all__ = [
    "ImageSourceFactory",
    "artifact_bindings",
]
