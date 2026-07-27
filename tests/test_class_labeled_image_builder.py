"""Contract tests for the built-in class-labeled image DataBuilder."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Sized
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from stochaflow import data, extensions
from stochaflow.data import (
    ClassLabeledImageDataBuilder,
    build_data_loaders,
)
from stochaflow.extensions import (
    IMAGE_DATA_SOURCES,
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    ComponentConfig,
    DataArtifact,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactLoadContext,
    DataArtifactStore,
    DataSourceContext,
    ImageDataSource,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    ManagedDataArtifactBuild,
    canonical_artifact_digest,
)

_SOURCE_NAME = "tests.class-labeled-image"
_PLAIN_SOURCE_NAME = "tests.plain-image-for-class-builder"


def _image_record(root: Path, relative_path: str, *, label: int) -> (
    ClassLabeledImageFileRecord
):
    path = root / relative_path
    encoded = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
    return ClassLabeledImageFileRecord(
        image=ImageFileRecord(
            tree="images",
            path=relative_path,
            size_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
            width=width,
            height=height,
        ),
        class_label=label,
    )


def _fixture_digest(root: Path) -> str:
    return canonical_artifact_digest(
        [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(root.rglob("*.png"))
        ]
    )


def _build_fixture_artifact(
    data_root: Path,
    *,
    source_root: Path,
    native_validation: bool,
) -> ManagedDataArtifactBuild:
    for role in ("train", "test"):
        shutil.copytree(source_root / role, data_root / role)
    return ManagedDataArtifactBuild(
        source_digest=_fixture_digest(source_root),
        materialization_digest=canonical_artifact_digest(
            {"name": "tests.fixture", "version": 2}
        ),
        domain={
            "schema_version": 1,
            "native_validation": native_validation,
        },
    )


def _load_labeled_fixture(
    context: DataArtifactLoadContext,
) -> ClassLabeledImageFolderArtifactPayload:
    native_validation = context.domain.get("native_validation", False)
    train = tuple(
        _image_record(
            context.data_root,
            f"train/{class_name}/sample_{index}.png",
            label=label,
        )
        for class_name, label in (("alpha", 0), ("beta", 1))
        for index in range(3)
    )
    test = tuple(
        _image_record(
            context.data_root,
            f"test/{class_name}/sample.png",
            label=label,
        )
        for class_name, label in (("alpha", 0), ("beta", 1))
    )
    validation = (train[0], train[3]) if native_validation else None
    return ClassLabeledImageFolderArtifactPayload(
        roots={"images": context.data_root},
        class_mapping={"alpha": 0, "beta": 1},
        train=train,
        validation=validation,
        test=test,
    )


@IMAGE_DATA_SOURCES.register(_SOURCE_NAME)
class IndependentClassLabeledImageSource(ImageDataSource):
    """Independent extension exercising only the public labeled payload."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ClassLabeledImageFolderArtifactPayload]:
        root = Path(self.params["root"]).resolve()
        native_validation = bool(self.params.get("native_validation", False))
        return DataArtifactStore(context).materialize_managed(
            artifact_type="tests.class-labeled-image.v1",
            source_name=_SOURCE_NAME,
            materializer_name="tests.fixture",
            locator_key={
                "root": str(root),
                "native_validation": native_validation,
            },
            build=lambda data_root: _build_fixture_artifact(
                data_root,
                source_root=root,
                native_validation=native_validation,
            ),
            load=_load_labeled_fixture,
        )


@IMAGE_DATA_SOURCES.register(_PLAIN_SOURCE_NAME)
class IndependentPlainImageSource(ImageDataSource):
    """Independent source returning the ordinary, unlabeled payload."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ImageFolderArtifactPayload]:
        root = Path(self.params["root"]).resolve()
        return DataArtifactStore(context).materialize_managed(
            artifact_type="tests.plain-image.v1",
            source_name=_PLAIN_SOURCE_NAME,
            materializer_name="tests.fixture",
            locator_key={"root": str(root)},
            build=lambda data_root: _build_fixture_artifact(
                data_root,
                source_root=root,
                native_validation=False,
            ),
            load=lambda load_context: ImageFolderArtifactPayload(
                roots={"images": load_context.data_root},
                train=(
                    _image_record(
                        load_context.data_root,
                        "train/alpha/sample_0.png",
                        label=0,
                    ).image,
                ),
            ),
        )


def _write_fixture(root: Path) -> None:
    for class_index, class_name in enumerate(("alpha", "beta")):
        for index in range(3):
            path = root / f"train/{class_name}/sample_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (8, 8),
                (class_index * 80, index * 20, 120),
            ).save(path)
        test_path = root / f"test/{class_name}/sample.png"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (8, 8),
            (class_index * 80, 160, 120),
        ).save(test_path)


def _component(
    root: Path,
    *,
    source_name: str = _SOURCE_NAME,
    native_validation: bool = False,
) -> ComponentConfig:
    return ComponentConfig(
        name="class_labeled_image",
        params={
            "source": {
                "name": source_name,
                "params": {
                    "root": str(root),
                    "native_validation": native_validation,
                },
                "materialization": {
                    "cache_root": str(root / "cache"),
                    "policy": "ensure",
                    "verification": "full",
                },
            },
            "partition": {
                "validation_per_class": 1,
                "seed": "tests-class-partition-v1",
            },
            "image": {
                "size": [8, 8],
                "channels": 3,
                "normalize": False,
                "random_horizontal_flip": False,
            },
            "loader": {
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "drop_last": False,
                "pin_memory": False,
                "persistent_workers": False,
                "steps_per_epoch": "auto",
            },
        },
    )


def test_class_labeled_builder_uses_an_independent_registered_source(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    loaders = build_data_loaders(_component(tmp_path), seed=17)

    assert isinstance(loaders.train, DataLoader)
    assert isinstance(loaders.validation, DataLoader)
    assert isinstance(loaders.test, DataLoader)
    assert len(cast(Sized, loaders.train.dataset)) == 4
    assert len(cast(Sized, loaders.validation.dataset)) == 2
    assert len(cast(Sized, loaders.test.dataset)) == 2
    images, conditions = next(iter(loaders.train))
    assert images.shape == (4, 3, 8, 8)
    assert set(conditions) == {"class_label"}
    assert conditions["class_label"].dtype == torch.long
    assert sorted(conditions["class_label"].tolist()) == [0, 0, 1, 1]
    assert loaders.artifact_bindings is not None
    assert loaders.artifact_bindings.ids == ("source",)
    assert (
        loaders.artifact_bindings.identity_for("source").source_name
        == _SOURCE_NAME
    )

    resumed = build_data_loaders(
        _component(tmp_path),
        seed=17,
        strict_resume=True,
        expected_artifacts=loaders.artifact_bindings,
    )
    assert resumed.artifact_bindings == loaders.artifact_bindings


def test_class_labeled_builder_rejects_native_validation(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="without native validation"):
        build_data_loaders(
            _component(tmp_path, native_validation=True),
            seed=17,
        )


def test_class_labeled_builder_rejects_an_unlabeled_payload(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path)

    with pytest.raises(
        TypeError,
        match="ClassLabeledImageFolderArtifactPayload",
    ):
        build_data_loaders(
            _component(tmp_path, source_name=_PLAIN_SOURCE_NAME),
            seed=17,
        )


def test_strict_artifact_mismatch_precedes_dataset_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_fixture(tmp_path)
    initial = build_data_loaders(_component(tmp_path), seed=17)
    assert initial.artifact_bindings is not None
    identity = initial.artifact_bindings.identity_for("source")
    expected = DataArtifactBindings(
        (
            DataArtifactBinding(
                id="source",
                identity=replace(identity, artifact_digest="d" * 64),
            ),
        )
    )
    constructed = False

    def record_dataset_construction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal constructed
        constructed = True
        raise AssertionError("Dataset construction must not be reached")

    monkeypatch.setattr(
        ClassLabeledImageDataBuilder,
        "_dataset",
        record_dataset_construction,
    )

    with pytest.raises(ValueError, match="artifact identity does not match"):
        build_data_loaders(
            _component(tmp_path),
            seed=17,
            strict_resume=True,
            expected_artifacts=expected,
        )

    assert not constructed


def test_class_labeled_contracts_are_stably_exported() -> None:
    assert data.ClassLabeledImageDataBuilder is ClassLabeledImageDataBuilder
    assert (
        data.ClassLabeledImageFileRecord
        is extensions.ClassLabeledImageFileRecord
        is ClassLabeledImageFileRecord
    )
    assert (
        data.ClassLabeledImageFolderArtifactPayload
        is extensions.ClassLabeledImageFolderArtifactPayload
        is ClassLabeledImageFolderArtifactPayload
    )
