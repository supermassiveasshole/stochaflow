"""Tests for the minimal public data-builder contract."""

from __future__ import annotations

from copy import deepcopy

import pytest

from stochaflow.data import (
    DataBuilder,
    DataBuilderContext,
    DataLoaders,
    build_data_loaders,
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


def test_data_loaders_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="train loader must be iterable"):
        DataLoaders(train=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expose len"):
        DataLoaders(train=FiniteIterable())
    with pytest.raises(TypeError, match="re-iterable"):
        DataLoaders(train=iter([1]), steps_per_epoch=1)
    with pytest.raises(ValueError, match="positive"):
        DataLoaders(train=[1], steps_per_epoch=0)
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
    import stochaflow.data as data

    assert set(data.__all__) == {
        "DataBuilder",
        "DataBuilderContext",
        "DataLoaders",
        "build_data_loaders",
    }
    for removed in (
        "DataPipeline",
        "DataBundle",
        "SplitData",
        "DatasetFactory",
        "DatasetView",
    ):
        assert not hasattr(data, removed)
