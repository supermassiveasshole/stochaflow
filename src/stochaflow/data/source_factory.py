"""Image source registry composition and strict artifact bindings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from stochaflow._builtin_activation import activate_data_builtins
from stochaflow.data.artifacts import (
    DataArtifact,
    DataArtifactBindings,
    materialize_data_source,
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
from stochaflow.utils.registry import Registry

if TYPE_CHECKING:
    from stochaflow.data.builder import DataBuilderContext


class ImageSourceFactory:
    """Materialize sources and validate the image-family artifact envelope.

    Recipe-specific payload and lifecycle compatibility remains the consuming
    DataBuilder's responsibility.
    """

    def __init__(
        self,
        registry: Registry[type[ImageDataSource]] = IMAGE_DATA_SOURCES,
    ) -> None:
        if registry is IMAGE_DATA_SOURCES:
            activate_data_builtins()
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
        source = self.registry.create(
            config.name,
            config.params,
            config_path=path,
        )
        context = builder_context.data_source_context(
            config.materialization,
            binding_id=binding_id,
            source_name=config.name,
            path=f"{path}.materialization",
        )
        artifact = materialize_data_source(source, context)
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
        return artifact


def artifact_bindings(
    artifacts: Sequence[tuple[str, DataArtifact[Any]]],
) -> DataArtifactBindings:
    """Build the canonical identities selected by an image recipe."""

    return DataArtifactBindings.from_artifacts(artifacts)


__all__ = [
    "ImageSourceFactory",
    "artifact_bindings",
]
