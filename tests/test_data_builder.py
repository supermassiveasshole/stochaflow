"""Tests for the minimal public data-builder contract."""

from __future__ import annotations

import ast
import inspect
from copy import deepcopy
from dataclasses import is_dataclass
from textwrap import dedent

import pytest
from torch.utils.data import DataLoader, IterableDataset

from stochaflow import data
from stochaflow.data import (
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataBuilder,
    DataBuilderContext,
    DataLoaders,
    DataRankContext,
    ImageDataBuilder,
    MultiResolutionImageDataBuilder,
    SuperResolutionDataBuilder,
    build_data_loaders,
    recipe_config,
)
from stochaflow.data.dataloaders import (
    collate_image_batch,
    collate_super_resolution_batch,
)
from stochaflow.data.datasets import (
    GeneratedSuperResolutionDataset,
    ImageDatasetFactory,
    ImageDatasetPartitions,
    ImageRecipeDataset,
    PairedSuperResolutionDataset,
)
from stochaflow.data.samplers import (
    EpochRandomSampler,
    MixtureBatchSampler,
    ResolutionBucketPolicy,
)
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog, RegistryError


class ReiterableStream:
    def __iter__(self):
        while True:
            yield {"state": 1, "condition": {"value": 2}}


class FiniteIterable:
    def __iter__(self):
        yield from (1, 2)


class StreamingIntegerDataset(IterableDataset[int]):
    def __iter__(self):
        yield from range(3)


@REGISTRIES.data_builders.register("test_streaming_builder")
class StreamingBuilder(DataBuilder):
    def build(self) -> DataLoaders:
        return DataLoaders(train=ReiterableStream(), steps_per_epoch=3)


@REGISTRIES.data_builders.register("test_wrong_result_builder")
class WrongResultBuilder(DataBuilder):
    def build(self):
        return [1, 2, 3]


def test_context_copies_params() -> None:
    params = {"nested": {"values": [1, 2]}}
    original = deepcopy(params)

    context = DataBuilderContext(params=params, seed=9)
    context.params["nested"]["values"].append(3)

    assert params == original


def test_context_rejects_non_mapping_params() -> None:
    with pytest.raises(TypeError, match="params must be a mapping"):
        DataBuilderContext(params=[], seed=9)  # type: ignore[arg-type]


def test_rank_context_is_runtime_injected_and_requires_ranked_support() -> None:
    rank_context = DataRankContext(rank=0, world_size=1)
    context = DataBuilderContext(
        params={"rank": 99, "world_size": 100},
        seed=9,
        rank_context=rank_context,
    )

    assert context.rank_context is rank_context
    ordinary = build_data_loaders(
        ComponentConfig(
            name="test_streaming_builder",
            params={"rank": 99, "world_size": 100},
        ),
        seed=7,
    )
    assert ordinary.ranked_execution is None

    with pytest.raises(ValueError, match="does not support ranked execution"):
        build_data_loaders(
            ComponentConfig(name="test_streaming_builder"),
            seed=7,
            rank_context=rank_context,
        )
    with pytest.raises(TypeError, match="rank_context"):
        DataBuilderContext(
            params={},
            seed=9,
            rank_context=object(),  # type: ignore[arg-type]
        )


def test_sized_and_streaming_loaders_validate_epoch_length() -> None:
    sized = DataLoaders(train=[1, 2])
    finite_iterable = DataLoaders(
        train=FiniteIterable(),
        steps_per_epoch=2,
    )
    streaming = build_data_loaders(
        ComponentConfig(name="test_streaming_builder"),
        seed=7,
    )

    assert sized.steps_per_epoch is None
    assert list(finite_iterable.train) == [1, 2]
    assert list(finite_iterable.train) == [1, 2]
    assert streaming.steps_per_epoch == 3
    assert next(iter(streaming.train))["condition"] == {"value": 2}


def test_pytorch_streaming_loader_uses_explicit_epoch_length() -> None:
    train_loader = DataLoader(
        StreamingIntegerDataset(),
        batch_size=None,
    )

    loaders = DataLoaders(train=train_loader, steps_per_epoch=2)

    assert loaders.train is train_loader
    assert loaders.steps_per_epoch == 2
    with pytest.raises(ValueError, match="expose len"):
        DataLoaders(train=train_loader)


def test_data_runtime_types_reside_in_responsibility_modules() -> None:
    assert DataBuilder.__module__ == "stochaflow.data.builder"
    assert DataBuilderContext.__module__ == "stochaflow.data.builder"
    assert DataLoaders.__module__ == "stochaflow.data.dataloaders"
    assert ImageRecipeDataset.__module__ == "stochaflow.data.datasets"
    assert (
        GeneratedSuperResolutionDataset.__module__
        == "stochaflow.data.datasets"
    )
    assert (
        PairedSuperResolutionDataset.__module__
        == "stochaflow.data.datasets"
    )
    assert ImageDatasetFactory.__module__ == "stochaflow.data.datasets"
    assert ImageDatasetPartitions.__module__ == "stochaflow.data.datasets"
    assert collate_image_batch.__module__ == "stochaflow.data.dataloaders"
    assert (
        collate_super_resolution_batch.__module__
        == "stochaflow.data.dataloaders"
    )
    assert EpochRandomSampler.__module__ == "stochaflow.data.samplers"
    assert MixtureBatchSampler.__module__ == "stochaflow.data.samplers"
    assert ResolutionBucketPolicy.__module__ == "stochaflow.data.samplers"


def test_every_recipe_config_exposes_one_public_validation_entrypoint() -> None:
    config_types = [
        value
        for value in vars(recipe_config).values()
        if (
            isinstance(value, type)
            and value.__module__ == recipe_config.__name__
            and is_dataclass(value)
        )
    ]

    assert config_types
    assert all("validate" in config_type.__dict__ for config_type in config_types)
    assert not any(
        name.startswith("_validate_") or name in {"validate_partition", "validate_size"}
        for name in vars(recipe_config)
    )


@pytest.mark.parametrize(
    "builder_type",
    [
        ImageDataBuilder,
        SuperResolutionDataBuilder,
        MultiResolutionImageDataBuilder,
    ],
)
def test_builtin_builder_constructor_calls_only_top_level_validate_once(
    builder_type: type[DataBuilder],
) -> None:
    tree = ast.parse(dedent(inspect.getsource(builder_type.__init__)))
    validate_calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "validate"
        )
    ]

    assert len(validate_calls) == 1
    validate_call = validate_calls[0]
    assert isinstance(validate_call.func, ast.Attribute)
    target = validate_call.func.value
    assert isinstance(target, ast.Attribute)
    assert target.attr == "config"
    assert isinstance(target.value, ast.Name)
    assert target.value.id == "self"


def test_data_loaders_preserve_optional_artifact_bindings() -> None:
    identity = DataArtifactIdentity(
        kind="managed",
        artifact_type="image-folder",
        source_name="test-source",
        source_digest="a" * 64,
        materializer_name="test-materializer",
        materialization_digest="b" * 64,
        content_digest="e" * 64,
        artifact_digest="c" * 64,
        manifest_sha256="d" * 64,
    )
    binding = DataArtifactBinding(id="primary", identity=identity)

    untracked = DataLoaders(train=[1])
    bindings = DataArtifactBindings((binding,))
    loaders = DataLoaders(train=[1], artifact_bindings=bindings)

    assert untracked.artifact_bindings is None
    assert loaders.artifact_bindings == bindings

    with pytest.raises(TypeError, match="must be DataArtifactBindings"):
        DataLoaders(train=[1], artifact_bindings=[binding])  # type: ignore[arg-type]


def test_data_loaders_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="train loader must be iterable"):
        DataLoaders(train=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expose len"):
        DataLoaders(train=FiniteIterable())
    with pytest.raises(TypeError, match="re-iterable"):
        DataLoaders(train=iter([1]), steps_per_epoch=1)
    with pytest.raises(ValueError, match="positive"):
        DataLoaders(train=[1], steps_per_epoch=0)
    with pytest.raises(ValueError, match="positive integer"):
        DataLoaders(train=[1, 2], steps_per_epoch=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not exceed"):
        DataLoaders(train=[1], steps_per_epoch=2)


def test_builder_errors_report_registry_and_return_contract() -> None:
    with pytest.raises(TypeError, match="must return DataLoaders"):
        build_data_loaders(
            ComponentConfig(name="test_wrong_result_builder"),
            seed=1,
        )
    with pytest.raises(RegistryError, match="unknown data builder"):
        build_data_loaders(ComponentConfig(name="missing_builder"), seed=1)

    catalog = RegistryCatalog()
    catalog.data_builders.require_base(DataBuilder)
    with pytest.raises(RegistryError, match=r"must inherit .*DataBuilder"):
        catalog.data_builders.add("wrong_base", object)


def test_only_new_data_contract_is_public() -> None:
    assert set(data.__all__) == {
        "ArtifactVerificationEvent",
        "ArtifactVerificationObserver",
        "ArtifactVerificationPhase",
        "ClassLabeledImageDataBuilder",
        "ClassLabeledImageFileRecord",
        "ClassLabeledImageFolderArtifactPayload",
        "DataArtifact",
        "DataArtifactBinding",
        "DataArtifactBindings",
        "DataArtifactIdentity",
        "DataArtifactLoadContext",
        "DataArtifactStore",
        "DataArtifactValidationError",
        "DataBuilder",
        "DataBuilderContext",
        "DataLoaders",
        "DataRankContext",
        "DataSource",
        "DataSourceContext",
        "DataSourceMaterializationConfig",
        "ExactCoverageReceipt",
        "ExactCoverageSpan",
        "ExactValidationBatch",
        "ExactValidationEpochPlan",
        "ExactValidationEpochReader",
        "ExactValidationExecution",
        "IMAGE_DATA_SOURCES",
        "ImageArtifactPayload",
        "ImageDataBuilder",
        "ImageDataSource",
        "ImageDimensionTable",
        "ImageDimensions",
        "ImageFilePair",
        "ImageFileRecord",
        "ImageFolderArtifactPayload",
        "ManagedDataArtifactBuild",
        "MultiResolutionImageDataBuilder",
        "PairedImageFolderArtifactPayload",
        "RankedBatchFacts",
        "RankedDataExecution",
        "RankedEpochCompletion",
        "RankedEpochDataIdentity",
        "RankedTrainEpochPlan",
        "RankedTrainEpochReader",
        "RankedTrainExecution",
        "RankedTrainWindow",
        "ReferencedDataArtifactBuild",
        "SuperResolutionDataBuilder",
        "TorchvisionImageArtifactPayload",
        "build_data_loaders",
        "canonical_artifact_digest",
        "canonical_artifact_json_bytes",
        "materialize_data_source",
    }
    for removed in (
        "DataPipeline",
        "DataBundle",
        "SplitData",
        "DatasetFactory",
        "DatasetView",
    ):
        assert not hasattr(data, removed)
