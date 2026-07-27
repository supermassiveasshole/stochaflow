"""Managed torchvision producer using the unified artifact lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image
from torchvision import datasets

from stochaflow.data.artifact_io import read_regular_file
from stochaflow.data.artifact_store import (
    DataArtifactLoadContext,
    DataArtifactStore,
    DataArtifactValidationError,
    ManagedDataArtifactBuild,
    canonical_artifact_digest,
    canonical_artifact_json_bytes,
)
from stochaflow.data.artifacts import DataArtifact, DataSourceContext
from stochaflow.data.image_contracts import (
    IMAGE_DATA_SOURCES,
    ImageDataSource,
    ImageDimensionTable,
    TorchvisionImageArtifactPayload,
)
from stochaflow.utils.config import ConfigError, coerce_config_section

_ARTIFACT_TYPE = "stochaflow.torchvision-image.v2"
_MATERIALIZER_NAME = "stochaflow.torchvision-download"
_DOMAIN_FIELDS = frozenset({"schema_version", "dataset", "dimensions"})
_DIMENSION_DESCRIPTOR_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_DIMENSION_SIDECAR = "image_dimensions.json"


@dataclass(slots=True)
class TorchvisionImageSourceConfig:
    """Provider parameters for the built-in torchvision source."""

    dataset: str

    def validate(self, *, path: str) -> None:
        """Validate the allowlisted torchvision dataset."""

        dataset = cast(object, self.dataset)
        if not isinstance(dataset, str):
            raise ConfigError(f"{path}.dataset must be a string")
        self.dataset = dataset.lower()
        if self.dataset not in {"mnist", "cifar10", "flowers102"}:
            raise ConfigError(
                f"{path}.dataset must be MNIST, CIFAR10, or Flowers102"
            )


def torchvision_datasets(
    dataset: str,
    root: Path,
    *,
    download: bool,
) -> dict[str, Any]:
    """Construct every native partition for one allowlisted dataset."""

    root_value = str(root)
    if dataset == "mnist":
        return {
            "train": datasets.MNIST(
                root_value,
                train=True,
                transform=None,
                download=download,
            ),
            "test": datasets.MNIST(
                root_value,
                train=False,
                transform=None,
                download=download,
            ),
        }
    if dataset == "cifar10":
        return {
            "train": datasets.CIFAR10(
                root_value,
                train=True,
                transform=None,
                download=download,
            ),
            "test": datasets.CIFAR10(
                root_value,
                train=False,
                transform=None,
                download=download,
            ),
        }
    return {
        role: datasets.Flowers102(
            root_value,
            split="val" if role == "validation" else role,
            transform=None,
            download=download,
        )
        for role in ("train", "validation", "test")
    }


def torchvision_dimension_table(
    dataset_name: str,
    dataset: Any,
) -> ImageDimensionTable:
    """Collect dimensions once while producing the managed artifact."""

    if dataset_name == "mnist":
        return ImageDimensionTable((28, 28) for _ in range(len(dataset)))
    if dataset_name == "cifar10":
        return ImageDimensionTable((32, 32) for _ in range(len(dataset)))

    dimensions: list[tuple[int, int]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        if not isinstance(image, Image.Image):
            raise TypeError(
                "torchvision image dataset returned an incompatible sample"
            )
        dimensions.append(image.size)
    return ImageDimensionTable(dimensions)


def write_torchvision_dimensions(
    data_root: Path,
    *,
    dataset_name: str,
    partitions: Mapping[str, Any],
) -> dict[str, object]:
    """Write an identity-bound canonical dimension sidecar."""

    serialized = {
        role: torchvision_dimension_table(
            dataset_name,
            partition,
        ).to_pairs()
        for role, partition in sorted(partitions.items())
    }
    encoded = canonical_artifact_json_bytes(
        {
            "schema_version": 1,
            "dataset": dataset_name,
            "partitions": serialized,
        }
    )
    (data_root / _DIMENSION_SIDECAR).write_bytes(encoded)
    return {
        "path": _DIMENSION_SIDECAR,
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _strict_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    raw = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise TypeError(f"{label} field names must be strings")
    if set(raw) != fields:
        raise ValueError(f"{label} must contain exactly {sorted(fields)}")
    return cast(dict[str, Any], value)


def load_torchvision_dimensions(
    data_root: Path,
    *,
    dataset_name: str,
    expected_record: Mapping[str, object],
) -> dict[str, ImageDimensionTable]:
    """Load a sidecar after verifying its domain descriptor."""

    record = _strict_mapping(
        dict(expected_record),
        fields=_DIMENSION_DESCRIPTOR_FIELDS,
        label="torchvision dimensions descriptor",
    )
    if record["path"] != _DIMENSION_SIDECAR:
        raise ValueError("torchvision dimensions path is invalid")
    size_bytes = record["size_bytes"]
    digest = record["sha256"]
    if (
        type(size_bytes) is not int
        or size_bytes <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("torchvision dimensions descriptor is invalid")
    encoded, metadata = read_regular_file(
        data_root,
        _DIMENSION_SIDECAR,
        label="torchvision image dimensions",
    )
    if (
        metadata.st_size != size_bytes
        or len(encoded) != size_bytes
        or hashlib.sha256(encoded).hexdigest() != digest
    ):
        raise ValueError(
            "torchvision image dimensions do not match their descriptor"
        )
    try:
        raw_value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "torchvision image dimensions contain invalid JSON"
        ) from exc
    if (
        not isinstance(raw_value, dict)
        or encoded != canonical_artifact_json_bytes(raw_value)
    ):
        raise ValueError(
            "torchvision image dimensions are not canonical JSON"
        )
    raw = cast(dict[str, Any], raw_value)
    if set(raw) != {"schema_version", "dataset", "partitions"}:
        raise ValueError("torchvision image dimensions have invalid fields")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["dataset"] != dataset_name
        or not isinstance(raw["partitions"], dict)
    ):
        raise ValueError("torchvision image dimensions are incompatible")
    expected_roles = (
        {"train", "test"}
        if dataset_name in {"mnist", "cifar10"}
        else {"train", "validation", "test"}
    )
    partitions = cast(dict[object, object], raw["partitions"])
    if set(partitions) != expected_roles:
        raise ValueError(
            "torchvision image dimensions have invalid partitions"
        )
    result: dict[str, ImageDimensionTable] = {}
    for role in sorted(expected_roles):
        serialized = partitions[role]
        if not isinstance(serialized, list) or not serialized:
            raise ValueError(
                f"torchvision {role} image dimensions must be a non-empty list"
            )
        dimensions: list[tuple[int, int]] = []
        for index, value in enumerate(serialized):
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    not isinstance(item, int) or isinstance(item, bool)
                    for item in value
                )
            ):
                raise ValueError(
                    f"torchvision {role} image dimensions[{index}] "
                    "must be [width, height]"
                )
            dimensions.append((value[0], value[1]))
        result[role] = ImageDimensionTable(dimensions)
    return result


def _source_digest(dataset: str) -> str:
    return canonical_artifact_digest(
        {
            "provider": "torchvision",
            "dataset": dataset,
            "selection_schema_version": 1,
        }
    )


def _materialization_digest(dataset: str) -> str:
    return canonical_artifact_digest(
        {
            "name": _MATERIALIZER_NAME,
            "version": 3,
            "dataset": dataset,
        }
    )


def _build_torchvision(
    data_root: Path,
    *,
    dataset: str,
) -> ManagedDataArtifactBuild:
    partitions = torchvision_datasets(dataset, data_root, download=True)
    dimensions = write_torchvision_dimensions(
        data_root,
        dataset_name=dataset,
        partitions=partitions,
    )
    return ManagedDataArtifactBuild(
        source_digest=_source_digest(dataset),
        materialization_digest=_materialization_digest(dataset),
        domain={
            "schema_version": 1,
            "dataset": dataset,
            "dimensions": dimensions,
        },
    )


def _load_torchvision(
    context: DataArtifactLoadContext,
    *,
    dataset: str,
) -> TorchvisionImageArtifactPayload:
    try:
        domain = _strict_mapping(
            dict(context.domain),
            fields=_DOMAIN_FIELDS,
            label="torchvision artifact domain",
        )
        if (
            type(domain["schema_version"]) is not int
            or domain["schema_version"] != 1
            or domain["dataset"] != dataset
        ):
            raise ValueError("torchvision artifact domain is incompatible")
        if context.identity.source_digest != _source_digest(dataset):
            raise ValueError("torchvision source identity is invalid")
        if (
            context.identity.materialization_digest
            != _materialization_digest(dataset)
        ):
            raise ValueError("torchvision materialization identity is invalid")
        dimensions = load_torchvision_dimensions(
            context.data_root,
            dataset_name=dataset,
            expected_record=cast(Mapping[str, object], domain["dimensions"]),
        )
        partitions = torchvision_datasets(
            dataset,
            context.data_root,
            download=False,
        )
        if set(partitions) != set(dimensions):
            raise ValueError("torchvision artifact partitions are incompatible")
        for role, partition in partitions.items():
            if len(partition) != len(dimensions[role]):
                raise ValueError(
                    f"torchvision {role} dimensions do not match its dataset"
                )
        return TorchvisionImageArtifactPayload(
            dataset=cast(Any, dataset),
            root=context.data_root,
            train_dimensions=dimensions["train"],
            validation_dimensions=dimensions.get("validation"),
            test_dimensions=dimensions.get("test"),
        )
    except DataArtifactValidationError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DataArtifactValidationError(str(exc)) from exc


@IMAGE_DATA_SOURCES.register("torchvision")
class TorchvisionImageDataSource(ImageDataSource):
    """Acquire one allowlisted dataset through the managed artifact store."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[TorchvisionImageArtifactPayload]:
        config = cast(
            TorchvisionImageSourceConfig,
            coerce_config_section(
                TorchvisionImageSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        return DataArtifactStore(context).materialize_managed(
            artifact_type=_ARTIFACT_TYPE,
            source_name="torchvision",
            materializer_name=_MATERIALIZER_NAME,
            locator_key={"dataset": config.dataset},
            build=lambda data_root: _build_torchvision(
                data_root,
                dataset=config.dataset,
            ),
            load=lambda load_context: _load_torchvision(
                load_context,
                dataset=config.dataset,
            ),
        )


__all__ = [
    "TorchvisionImageDataSource",
    "TorchvisionImageSourceConfig",
    "load_torchvision_dimensions",
    "torchvision_datasets",
    "torchvision_dimension_table",
    "write_torchvision_dimensions",
]
