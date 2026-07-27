"""Load and validate the pinned AFHQ-v2 source contract."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from .contracts import DatasetContract, PreparationError, SourceLock

_DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "afhq-v2.lock.yaml"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PreparationError(f"{label} must be a positive integer")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise PreparationError(f"{label} has unknown fields: {unknown}")
    if missing:
        raise PreparationError(f"{label} is missing fields: {missing}")


def load_source_lock(path: Path = _DEFAULT_LOCK_PATH) -> SourceLock:
    """Load and strictly validate the checked-in AFHQ-v2 source lock."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PreparationError(f"source lock does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise PreparationError(f"source lock is invalid YAML: {path}") from error

    root = _require_mapping(raw, label="source lock")
    _require_exact_keys(
        root,
        expected={
            "schema_version",
            "dataset",
            "source",
            "license",
            "homepage",
            "citation",
            "dataset_contract",
        },
        label="source lock",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise PreparationError("source lock schema_version must be 1")
    dataset = _require_string(root["dataset"], label="dataset")
    if dataset != "afhq-v2":
        raise PreparationError("source lock dataset must be 'afhq-v2'")

    source = _require_mapping(root["source"], label="source")
    _require_exact_keys(
        source,
        expected={"type", "url", "archive_name", "bytes", "sha256"},
        label="source",
    )
    if source["type"] != "official_archive":
        raise PreparationError("source.type must be 'official_archive'")
    expected_sha256 = source["sha256"]
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or _SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise PreparationError("source.sha256 must be null or lowercase SHA-256")

    license_value = _require_mapping(root["license"], label="license")
    _require_exact_keys(
        license_value,
        expected={"name", "url"},
        label="license",
    )
    contract_value = _require_mapping(
        root["dataset_contract"],
        label="dataset_contract",
    )
    _require_exact_keys(
        contract_value,
        expected={
            "classes",
            "class_mapping",
            "source_splits",
            "source_class_counts",
            "total_count",
            "input_resolution",
            "image_mode",
            "image_format",
        },
        label="dataset_contract",
    )

    raw_classes = contract_value["classes"]
    if not isinstance(raw_classes, Sequence) or isinstance(raw_classes, str):
        raise PreparationError("dataset_contract.classes must be a list")
    classes = tuple(
        _require_string(value, label="dataset_contract.classes item")
        for value in raw_classes
    )
    if classes != ("cat", "dog", "wild"):
        raise PreparationError(
            "dataset_contract.classes must be ['cat', 'dog', 'wild']"
        )

    raw_mapping = _require_mapping(
        contract_value["class_mapping"],
        label="dataset_contract.class_mapping",
    )
    class_mapping: dict[str, int] = {}
    for class_name in classes:
        value = raw_mapping.get(class_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise PreparationError(
                f"dataset_contract.class_mapping.{class_name} must be an integer"
            )
        class_mapping[class_name] = value
    if set(raw_mapping) != set(classes) or class_mapping != {
        "cat": 0,
        "dog": 1,
        "wild": 2,
    }:
        raise PreparationError(
            "dataset_contract.class_mapping must be cat: 0, dog: 1, wild: 2"
        )

    source_splits = _require_mapping(
        contract_value["source_splits"],
        label="dataset_contract.source_splits",
    )
    _require_exact_keys(
        source_splits,
        expected={"train", "test"},
        label="dataset_contract.source_splits",
    )
    train_count = _require_positive_int(
        source_splits["train"],
        label="dataset_contract.source_splits.train",
    )
    test_count = _require_positive_int(
        source_splits["test"],
        label="dataset_contract.source_splits.test",
    )
    raw_source_class_counts = _require_mapping(
        contract_value["source_class_counts"],
        label="dataset_contract.source_class_counts",
    )
    _require_exact_keys(
        raw_source_class_counts,
        expected={"train", "test"},
        label="dataset_contract.source_class_counts",
    )
    source_class_counts: dict[str, dict[str, int]] = {}
    for split, expected_split_count in (
        ("train", train_count),
        ("test", test_count),
    ):
        raw_split_counts = _require_mapping(
            raw_source_class_counts[split],
            label=f"dataset_contract.source_class_counts.{split}",
        )
        _require_exact_keys(
            raw_split_counts,
            expected=set(classes),
            label=f"dataset_contract.source_class_counts.{split}",
        )
        split_counts = {
            class_name: _require_positive_int(
                raw_split_counts[class_name],
                label=(
                    f"dataset_contract.source_class_counts."
                    f"{split}.{class_name}"
                ),
            )
            for class_name in classes
        }
        if sum(split_counts.values()) != expected_split_count:
            raise PreparationError(
                f"dataset_contract source class counts for {split} do not "
                "sum to its split count"
            )
        source_class_counts[split] = split_counts
    total_count = _require_positive_int(
        contract_value["total_count"],
        label="dataset_contract.total_count",
    )
    if train_count + test_count != total_count:
        raise PreparationError("source split counts do not sum to total_count")

    archive_name = _require_string(
        source["archive_name"],
        label="source.archive_name",
    )
    if archive_name != "afhq_v2.zip":
        raise PreparationError("source.archive_name must be exactly 'afhq_v2.zip'")
    input_resolution = _require_positive_int(
        contract_value["input_resolution"],
        label="dataset_contract.input_resolution",
    )
    if input_resolution != 512:
        raise PreparationError("dataset_contract.input_resolution must be 512")
    image_mode = _require_string(
        contract_value["image_mode"],
        label="dataset_contract.image_mode",
    )
    if image_mode != "RGB":
        raise PreparationError("dataset_contract.image_mode must be 'RGB'")
    image_format = _require_string(
        contract_value["image_format"],
        label="dataset_contract.image_format",
    )
    if image_format != "PNG":
        raise PreparationError("dataset_contract.image_format must be 'PNG'")

    return SourceLock(
        dataset=dataset,
        url=_require_string(source["url"], label="source.url"),
        archive_name=archive_name,
        expected_bytes=_require_positive_int(source["bytes"], label="source.bytes"),
        expected_sha256=expected_sha256,
        license_name=_require_string(license_value["name"], label="license.name"),
        license_url=_require_string(license_value["url"], label="license.url"),
        homepage=_require_string(root["homepage"], label="homepage"),
        citation=_require_string(root["citation"], label="citation"),
        contract=DatasetContract(
            classes=classes,
            class_mapping=class_mapping,
            train_count=train_count,
            test_count=test_count,
            total_count=total_count,
            input_resolution=input_resolution,
            image_mode=image_mode,
            image_format=image_format,
            source_class_counts=source_class_counts,
        ),
    )
