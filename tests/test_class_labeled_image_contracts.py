"""Contracts for class-labeled image artifacts and stratified partitioning."""

from __future__ import annotations

import hashlib
import operator
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pytest

from stochaflow.data.image_contracts import (
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    ImageFileRecord,
)
from stochaflow.data.partition import partition_class_labeled_records
from stochaflow.data.recipe_config import (
    ClassLabeledImageDataBuilderConfig,
    ClassStratifiedPartitionRecipeConfig,
    DataSourceMaterializationConfig,
    ImageRecipeConfig,
    ImageSourceConfig,
    LoaderRecipeConfig,
)
from stochaflow.utils.config import ConfigError


def _image_record(
    class_name: str,
    index: int,
    *,
    tree: str = "train",
) -> ImageFileRecord:
    identity = f"{tree}/{class_name}/{index}"
    return ImageFileRecord(
        tree=tree,
        path=f"{class_name}/{index}.png",
        size_bytes=index + 1,
        sha256=hashlib.sha256(identity.encode()).hexdigest(),
        width=8,
        height=8,
    )


def _labeled_records(
    *,
    per_class: int,
    tree: str = "train",
) -> tuple[ClassLabeledImageFileRecord, ...]:
    return tuple(
        ClassLabeledImageFileRecord(
            image=_image_record(class_name, index, tree=tree),
            class_label=class_label,
        )
        for class_label, class_name in enumerate(("cat", "dog", "wild"))
        for index in range(per_class)
    )


def _identity(
    record: ClassLabeledImageFileRecord,
) -> tuple[str, str, str]:
    image = record.image
    return image.tree, image.path, image.sha256


def test_class_labeled_record_requires_an_image_and_non_negative_integer() -> None:
    image = _image_record("cat", 0)
    record = ClassLabeledImageFileRecord(image=image, class_label=0)

    assert record.image is image
    assert record.class_label == 0
    with pytest.raises(TypeError, match="must be ImageFileRecord"):
        ClassLabeledImageFileRecord(
            image=object(),  # type: ignore[arg-type]
            class_label=0,
        )
    for value in (-1, True, 1.5, "1"):
        with pytest.raises(ValueError, match="non-negative integer"):
            ClassLabeledImageFileRecord(
                image=image,
                class_label=value,  # type: ignore[arg-type]
            )


def test_class_labeled_payload_freezes_mapping_and_inventories(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    train_root.mkdir()
    test_root.mkdir()
    roots = {"train": train_root, "test": test_root}
    class_mapping = {"cat": 0, "dog": 1, "wild": 2}
    train = list(_labeled_records(per_class=2))
    test = list(_labeled_records(per_class=1, tree="test"))

    payload = ClassLabeledImageFolderArtifactPayload(
        roots=roots,
        class_mapping=class_mapping,
        train=train,  # type: ignore[arg-type]
        test=test,  # type: ignore[arg-type]
    )
    roots.clear()
    class_mapping.clear()
    train.clear()
    test.clear()

    assert set(payload.roots) == {"train", "test"}
    assert dict(payload.class_mapping) == {"cat": 0, "dog": 1, "wild": 2}
    assert len(payload.train) == 6
    assert len(payload.test or ()) == 3
    with pytest.raises(TypeError):
        operator.setitem(
            cast(dict[str, int], payload.class_mapping),
            "cat",
            2,
        )


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"cat": 0, "dog": 0},
        {"cat": 0, "dog": 2},
        {"": 0},
        {"cat": True},
    ],
)
def test_class_labeled_payload_requires_contiguous_unique_class_mapping(
    tmp_path: Path,
    mapping: dict[str, Any],
) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()

    with pytest.raises((TypeError, ValueError)):
        ClassLabeledImageFolderArtifactPayload(
            roots={"train": train_root},
            class_mapping=mapping,
            train=(
                ClassLabeledImageFileRecord(
                    image=_image_record("cat", 0),
                    class_label=0,
                ),
            ),
        )


def test_class_labeled_payload_validates_trees_labels_and_train_coverage(
    tmp_path: Path,
) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    roots = {"train": train_root}
    mapping = {"cat": 0, "dog": 1}
    cat = ClassLabeledImageFileRecord(_image_record("cat", 0), 0)
    dog = ClassLabeledImageFileRecord(_image_record("dog", 0), 1)

    with pytest.raises(ValueError, match="missing class labels: 1"):
        ClassLabeledImageFolderArtifactPayload(
            roots=roots,
            class_mapping=mapping,
            train=(cat,),
        )
    with pytest.raises(ValueError, match="unknown class label"):
        ClassLabeledImageFolderArtifactPayload(
            roots=roots,
            class_mapping=mapping,
            train=(cat, dog),
            test=(
                ClassLabeledImageFileRecord(
                    _image_record("wild", 0),
                    2,
                ),
            ),
        )
    with pytest.raises(ValueError, match="unknown tree"):
        ClassLabeledImageFolderArtifactPayload(
            roots=roots,
            class_mapping=mapping,
            train=(
                cat,
                ClassLabeledImageFileRecord(
                    _image_record("dog", 0, tree="missing"),
                    1,
                ),
            ),
        )


def test_class_labeled_builder_config_validates_nested_recipe() -> None:
    config = ClassLabeledImageDataBuilderConfig(
        source=ImageSourceConfig(
            name="fixture",
            materialization=DataSourceMaterializationConfig(
                cache_root="./data",
                policy="require",
                verification="full",
            ),
        ),
        image=ImageRecipeConfig(size=[8, 8]),
        partition=ClassStratifiedPartitionRecipeConfig(
            validation_per_class=2,
            seed="fixture-partition",
        ),
        loader=LoaderRecipeConfig(
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
        ),
    )

    config.validate()
    for value in (0, -1, True, 1.5):
        invalid = ClassStratifiedPartitionRecipeConfig(
            validation_per_class=value,  # type: ignore[arg-type]
        )
        with pytest.raises(ConfigError, match="must be a positive integer"):
            invalid.validate(path="data.params.partition")
    for value in (True, 1.5, "", "   "):
        invalid_seed = ClassStratifiedPartitionRecipeConfig(
            validation_per_class=1,
            seed=value,  # type: ignore[arg-type]
        )
        with pytest.raises(ConfigError, match=r"\.seed must be"):
            invalid_seed.validate(path="data.params.partition")


def test_stratified_partition_is_identity_stable_and_preserves_input_order() -> None:
    records = _labeled_records(per_class=6)
    config = ClassStratifiedPartitionRecipeConfig(
        validation_per_class=2,
        seed="stable-fixture",
    )

    train, validation = partition_class_labeled_records(
        records,
        config,
        seed=91,
    )
    reversed_train, reversed_validation = partition_class_labeled_records(
        tuple(reversed(records)),
        config,
        seed=91,
    )
    selected = {_identity(record) for record in validation}

    assert selected == {
        _identity(record) for record in reversed_validation
    }
    assert Counter(record.class_label for record in validation) == {
        0: 2,
        1: 2,
        2: 2,
    }
    assert Counter(record.class_label for record in train) == {
        0: 4,
        1: 4,
        2: 4,
    }
    assert train == tuple(
        record for record in records if _identity(record) not in selected
    )
    assert validation == tuple(
        record for record in records if _identity(record) in selected
    )
    assert reversed_train == tuple(
        record
        for record in reversed(records)
        if _identity(record) not in selected
    )


def test_stratified_partition_uses_run_seed_only_without_override() -> None:
    records = _labeled_records(per_class=6)
    inherited = ClassStratifiedPartitionRecipeConfig(
        validation_per_class=2,
    )
    explicit = ClassStratifiedPartitionRecipeConfig(
        validation_per_class=2,
        seed=47,
    )

    inherited_result = partition_class_labeled_records(
        records,
        inherited,
        seed=47,
    )
    explicit_result = partition_class_labeled_records(
        records,
        explicit,
        seed=999,
    )

    assert inherited_result == explicit_result


def test_stratified_partition_must_leave_one_training_record_per_class() -> None:
    records = _labeled_records(per_class=2)

    with pytest.raises(ValueError, match="leave at least one training record"):
        partition_class_labeled_records(
            records,
            ClassStratifiedPartitionRecipeConfig(validation_per_class=2),
            seed=1,
        )
