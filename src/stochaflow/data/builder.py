"""DataBuilder contract and built-in image recipe composition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset

from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.data.dataloaders import (
    DataLoaders,
    build_class_labeled_image_data_loader,
    build_map_data_loader,
    build_multi_resolution_data_loader,
    collate_image_batch,
    collate_super_resolution_batch,
    configured_steps_per_epoch,
)
from stochaflow.data.datasets import (
    ClassLabeledImageDataset,
    GeneratedSuperResolutionDataset,
    ImageDatasetFactory,
    ImageDatasetPartitions,
    ImageRecipeDataset,
    MultiResolutionDataset,
    PairedSuperResolutionDataset,
    combine_image_datasets,
)
from stochaflow.data.image_contracts import (
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    PairedImageFolderArtifactPayload,
)
from stochaflow.data.partition import (
    partition_class_labeled_records,
    partition_datasets,
)
from stochaflow.data.recipe_config import (
    ClassLabeledImageDataBuilderConfig,
    ImageDataBuilderConfig,
    MultiResolutionDataBuilderConfig,
    SuperResolutionDataBuilderConfig,
)
from stochaflow.data.samplers import ResolutionBucketPolicy
from stochaflow.data.source_factory import (
    ImageSourceFactory,
    artifact_bindings,
)
from stochaflow.data.transforms import (
    GeneratedSuperResolutionTransform,
    ImageTransform,
    PairedSuperResolutionTransform,
)
from stochaflow.utils.config import ComponentConfig, coerce_config_section
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog


@dataclass(frozen=True, slots=True)
class DataBuilderContext:
    """Copied recipe parameters and strict data-resume expectations."""

    params: dict[str, Any]
    seed: int
    strict_resume: bool = False
    expected_artifacts: DataArtifactBindings | None = None

    def __post_init__(self) -> None:
        params = cast(object, self.params)
        if not isinstance(params, dict):
            raise TypeError("data builder params must be a mapping")
        strict_resume = cast(object, self.strict_resume)
        if not isinstance(strict_resume, bool):
            raise TypeError("strict_resume must be boolean")
        expected_artifacts = cast(object, self.expected_artifacts)
        if expected_artifacts is not None and not isinstance(
            expected_artifacts, DataArtifactBindings
        ):
            raise TypeError(
                "expected_artifacts must be DataArtifactBindings or None"
            )
        object.__setattr__(self, "params", deepcopy(self.params))

    def require_artifact_ids(self, binding_ids: tuple[str, ...]) -> None:
        """Fail before materialization when strict history is unavailable."""

        if not self.strict_resume:
            return
        if self.expected_artifacts is None:
            raise ValueError(
                "strict resume checkpoint is missing data artifact identities"
            )
        self.expected_artifacts.assert_ids(binding_ids)

    def verify_artifacts(self, actual: DataArtifactBindings) -> None:
        """Compare all selected identities before constructing Datasets."""

        if self.strict_resume:
            assert self.expected_artifacts is not None
            self.expected_artifacts.assert_exact(actual)


class DataBuilder(ABC):
    """Extension point assembling one run's complete data-loading stack."""

    def __init__(self, context: DataBuilderContext) -> None:
        self.context = context

    @abstractmethod
    def build(self) -> DataLoaders:
        """Return ready train, validation, and test iterables."""


REGISTRIES.data_builders.require_base(DataBuilder)


def build_data_loaders(
    config: ComponentConfig,
    *,
    seed: int,
    strict_resume: bool = False,
    expected_artifacts: DataArtifactBindings | None = None,
    registries: RegistryCatalog = REGISTRIES,
) -> DataLoaders:
    """Construct and validate one registered DataBuilder."""

    registries.data_builders.require_base(DataBuilder)
    builder = cast(
        DataBuilder,
        registries.data_builders.create(
            config.name,
            DataBuilderContext(
                params=config.params,
                seed=seed,
                strict_resume=strict_resume,
                expected_artifacts=expected_artifacts,
            ),
        ),
    )
    loaders_value = cast(object, builder.build())
    if not isinstance(loaders_value, DataLoaders):
        raise TypeError(
            f"data builder '{config.name}' must return DataLoaders"
        )
    return loaders_value


def _image_transform(
    config: ImageDataBuilderConfig,
    *,
    role: str,
) -> ImageTransform:
    return ImageTransform(
        config.image.resolved_size,
        role=role,
        channels=config.image.channels,
        normalize=config.image.normalize,
        random_horizontal_flip=config.image.random_horizontal_flip,
    )


def _wrap_image_partitions(
    partitions: Any,
    config: ImageDataBuilderConfig,
) -> tuple[Dataset[Any], Dataset[Any] | None, Dataset[Any] | None]:
    train = ImageRecipeDataset(
        partitions.train,
        _image_transform(config, role="train"),
    )
    validation = (
        ImageRecipeDataset(
            partitions.validation,
            _image_transform(config, role="eval"),
        )
        if partitions.validation is not None
        else None
    )
    test = (
        ImageRecipeDataset(
            partitions.test,
            _image_transform(config, role="eval"),
        )
        if partitions.test is not None
        else None
    )
    return train, validation, test


@REGISTRIES.data_builders.register("image")
class ImageDataBuilder(DataBuilder):
    """Standard single-source image training recipe."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            ImageDataBuilderConfig,
            coerce_config_section(
                ImageDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        self.config.validate()
        self.source_factory = ImageSourceFactory()
        self.dataset_factory = ImageDatasetFactory()

    def build(self) -> DataLoaders:
        self.context.require_artifact_ids(("source",))
        artifact = self.source_factory.materialize(
            self.config.source,
            binding_id="source",
            builder_context=self.context,
            path="data.params.source",
        )
        if isinstance(artifact.payload, PairedImageFolderArtifactPayload):
            raise TypeError(
                "image DataBuilder requires a single-image artifact payload"
            )
        bindings = artifact_bindings((("source", artifact),))
        self.context.verify_artifacts(bindings)
        source = self.dataset_factory.build(artifact)
        partitions = partition_datasets(
            source,
            self.config.partition,
            seed=self.context.seed,
        )
        train, validation, test = _wrap_image_partitions(
            partitions,
            self.config,
        )
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                build_map_data_loader(
                    train,
                    self.config.loader,
                    training=True,
                    seed=self.context.seed,
                    collate_fn=collate_image_batch,
                ),
            ),
            validation=build_map_data_loader(
                validation,
                self.config.loader,
                training=False,
                seed=self.context.seed + 1,
                collate_fn=collate_image_batch,
            ),
            test=build_map_data_loader(
                test,
                self.config.loader,
                training=False,
                seed=self.context.seed + 2,
                collate_fn=collate_image_batch,
            ),
            steps_per_epoch=configured_steps_per_epoch(self.config.loader),
            artifact_bindings=bindings,
        )


@REGISTRIES.data_builders.register("class_labeled_image")
class ClassLabeledImageDataBuilder(DataBuilder):
    """Class-conditioned image recipe for labeled folder artifacts."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            ClassLabeledImageDataBuilderConfig,
            coerce_config_section(
                ClassLabeledImageDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        self.config.validate()
        self.source_factory = ImageSourceFactory()

    def _dataset(
        self,
        roots: Mapping[str, Path],
        records: Sequence[ClassLabeledImageFileRecord],
        *,
        training: bool,
    ) -> ClassLabeledImageDataset:
        role = "train" if training else "eval"
        return ClassLabeledImageDataset(
            roots=roots,
            records=records,
            transform=ImageTransform(
                self.config.image.resolved_size,
                role=role,
                channels=self.config.image.channels,
                normalize=self.config.image.normalize,
                random_horizontal_flip=(
                    self.config.image.random_horizontal_flip
                    if training
                    else False
                ),
            ),
            seed=self.context.seed,
        )

    def build(self) -> DataLoaders:
        """Materialize one labeled artifact and assemble class-aware loaders."""

        self.context.require_artifact_ids(("source",))
        artifact = self.source_factory.materialize(
            self.config.source,
            binding_id="source",
            builder_context=self.context,
            path="data.params.source",
        )
        payload = artifact.payload
        if not isinstance(payload, ClassLabeledImageFolderArtifactPayload):
            raise TypeError(
                "class_labeled_image DataBuilder requires a "
                "ClassLabeledImageFolderArtifactPayload"
            )
        bindings = artifact_bindings((("source", artifact),))
        self.context.verify_artifacts(bindings)
        if payload.validation is not None:
            raise ValueError(
                "class_labeled_image DataBuilder requires a source without "
                "native validation records"
            )
        train_records, validation_records = partition_class_labeled_records(
            payload.train,
            self.config.partition,
            seed=self.context.seed,
        )
        train = self._dataset(payload.roots, train_records, training=True)
        validation = self._dataset(
            payload.roots,
            validation_records,
            training=False,
        )
        test = (
            self._dataset(payload.roots, payload.test, training=False)
            if payload.test is not None
            else None
        )
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                build_class_labeled_image_data_loader(
                    train,
                    self.config.loader,
                    training=True,
                    seed=self.context.seed,
                ),
            ),
            validation=build_class_labeled_image_data_loader(
                validation,
                self.config.loader,
                training=False,
                seed=self.context.seed + 1,
            ),
            test=build_class_labeled_image_data_loader(
                test,
                self.config.loader,
                training=False,
                seed=self.context.seed + 2,
            ),
            steps_per_epoch=configured_steps_per_epoch(self.config.loader),
            artifact_bindings=bindings,
        )


@REGISTRIES.data_builders.register("super_resolution")
class SuperResolutionDataBuilder(DataBuilder):
    """Paired or on-the-fly bicubic image super-resolution recipe."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            SuperResolutionDataBuilderConfig,
            coerce_config_section(
                SuperResolutionDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        self.config.validate()
        self.source_factory = ImageSourceFactory()
        self.dataset_factory = ImageDatasetFactory()

    def _wrap(
        self,
        dataset: Dataset[Any] | None,
        *,
        role: str,
    ) -> Dataset[Any] | None:
        if dataset is None:
            return None
        image = self.config.image
        if self.config.low_resolution.kind == "paired":
            return PairedSuperResolutionDataset(
                dataset,
                PairedSuperResolutionTransform(
                    image.resolved_high_resolution,
                    image.resolved_low_resolution,
                    role=role,
                    channels=image.channels,
                    normalize=image.normalize,
                    random_horizontal_flip=image.random_horizontal_flip,
                ),
            )
        return GeneratedSuperResolutionDataset(
            dataset,
            GeneratedSuperResolutionTransform(
                image.resolved_high_resolution,
                image.resolved_low_resolution,
                role=role,
                channels=image.channels,
                normalize=image.normalize,
                random_horizontal_flip=image.random_horizontal_flip,
            ),
        )

    def build(self) -> DataLoaders:
        self.context.require_artifact_ids(("source",))
        artifact = self.source_factory.materialize(
            self.config.source,
            binding_id="source",
            builder_context=self.context,
            path="data.params.source",
        )
        paired_payload = isinstance(
            artifact.payload,
            PairedImageFolderArtifactPayload,
        )
        if paired_payload != (self.config.low_resolution.kind == "paired"):
            raise TypeError(
                "super-resolution source payload does not match "
                "low_resolution.kind"
            )
        bindings = artifact_bindings((("source", artifact),))
        self.context.verify_artifacts(bindings)
        source = self.dataset_factory.build(artifact)
        partitions = partition_datasets(
            source,
            self.config.partition,
            seed=self.context.seed,
        )
        train = cast(
            Dataset[Any],
            self._wrap(partitions.train, role="train"),
        )
        validation = self._wrap(partitions.validation, role="eval")
        test = self._wrap(partitions.test, role="eval")
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                build_map_data_loader(
                    train,
                    self.config.loader,
                    training=True,
                    seed=self.context.seed,
                    collate_fn=collate_super_resolution_batch,
                ),
            ),
            validation=build_map_data_loader(
                validation,
                self.config.loader,
                training=False,
                seed=self.context.seed + 1,
                collate_fn=collate_super_resolution_batch,
            ),
            test=build_map_data_loader(
                test,
                self.config.loader,
                training=False,
                seed=self.context.seed + 2,
                collate_fn=collate_super_resolution_batch,
            ),
            steps_per_epoch=configured_steps_per_epoch(self.config.loader),
            artifact_bindings=bindings,
        )


def _source_weights(
    config: MultiResolutionDataBuilderConfig,
) -> Mapping[str, float] | None:
    if all(source.sampling_weight is None for source in config.sources):
        return None
    return {
        source.id: cast(float, source.sampling_weight)
        for source in config.sources
    }


def _wrap_multi_resolution(
    dataset: Dataset[Any] | None,
    policy: ResolutionBucketPolicy,
    config: MultiResolutionDataBuilderConfig,
    *,
    role: str,
) -> MultiResolutionDataset | None:
    if dataset is None:
        return None
    return MultiResolutionDataset(
        dataset,
        policy,
        role=role,
        channels=config.image.channels,
        normalize=config.image.normalize,
        random_horizontal_flip=config.image.random_horizontal_flip,
    )


@REGISTRIES.data_builders.register("multi_resolution_image")
class MultiResolutionImageDataBuilder(DataBuilder):
    """Multi-source image recipe with bucket-homogeneous dynamic batches."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = cast(
            MultiResolutionDataBuilderConfig,
            coerce_config_section(
                MultiResolutionDataBuilderConfig,
                context.params,
                "data.params",
            ),
        )
        self.config.validate()
        self.source_factory = ImageSourceFactory()
        self.dataset_factory = ImageDatasetFactory()

    def build(self) -> DataLoaders:
        source_ids = tuple(source.id for source in self.config.sources)
        self.context.require_artifact_ids(source_ids)
        artifacts = [
            (
                source.id,
                self.source_factory.materialize(
                    source.source,
                    binding_id=source.id,
                    builder_context=self.context,
                    path=f"data.params.sources[{index}].source",
                ),
            )
            for index, source in enumerate(self.config.sources)
        ]
        if any(
            isinstance(artifact.payload, PairedImageFolderArtifactPayload)
            for _, artifact in artifacts
        ):
            raise TypeError(
                "multi-resolution image builder requires single-image payloads"
            )
        bindings = artifact_bindings(
            cast(
                list[tuple[str, Any]],
                artifacts,
            )
        )
        self.context.verify_artifacts(bindings)
        sources = [
            (source_id, self.dataset_factory.build(artifact))
            for source_id, artifact in artifacts
        ]
        combined = ImageDatasetPartitions(
            train=cast(
                Dataset[Any],
                combine_image_datasets(sources, "train"),
            ),
            validation=(
                combine_image_datasets(sources, "validation")
                if self.config.partition.mode == "official"
                else None
            ),
            test=combine_image_datasets(sources, "test"),
        )
        partitions = partition_datasets(
            combined,
            self.config.partition,
            seed=self.context.seed,
        )
        policy = ResolutionBucketPolicy(
            self.config.batching.buckets,
            base_bucket=self.config.batching.base_bucket,
            dynamic_batch_size=self.config.batching.dynamic_batch_size,
        )
        train = _wrap_multi_resolution(
            partitions.train,
            policy,
            self.config,
            role="train",
        )
        assert train is not None
        source_weights = _source_weights(self.config)
        return DataLoaders(
            train=cast(
                DataLoader[Any],
                build_multi_resolution_data_loader(
                    train,
                    policy,
                    self.config.loader,
                    training=True,
                    seed=self.context.seed,
                    source_weights=source_weights,
                    collate_fn=collate_image_batch,
                ),
            ),
            validation=build_multi_resolution_data_loader(
                _wrap_multi_resolution(
                    partitions.validation,
                    policy,
                    self.config,
                    role="eval",
                ),
                policy,
                self.config.loader,
                training=False,
                seed=self.context.seed + 1,
                source_weights=None,
                collate_fn=collate_image_batch,
            ),
            test=build_multi_resolution_data_loader(
                _wrap_multi_resolution(
                    partitions.test,
                    policy,
                    self.config,
                    role="eval",
                ),
                policy,
                self.config.loader,
                training=False,
                seed=self.context.seed + 2,
                source_weights=None,
                collate_fn=collate_image_batch,
            ),
            steps_per_epoch=configured_steps_per_epoch(self.config.loader),
            artifact_bindings=bindings,
        )


__all__ = [
    "ClassLabeledImageDataBuilder",
    "DataBuilder",
    "DataBuilderContext",
    "ImageDataBuilder",
    "MultiResolutionImageDataBuilder",
    "SuperResolutionDataBuilder",
    "build_data_loaders",
]
