"""Managed torchvision acquisition and authenticated image metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from PIL import Image
from torchvision import datasets

from stochaflow.data.artifact_io import (
    create_cache_directory,
    publish_cache_directory,
    read_regular_file,
    remove_cache_directory,
    scan_regular_files,
    write_cache_file,
)
from stochaflow.data.artifact_store import (
    ArtifactMaterializationLock,
    canonical_digest,
    canonical_json_bytes,
    load_canonical_json,
    path_exists_without_following,
    quarantine_path,
    read_locator_for_policy,
    sha256_bytes,
    strict_mapping,
    write_locator,
)
from stochaflow.data.artifacts import (
    DataSourceContext,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
)
from stochaflow.data.image_contracts import (
    IMAGE_DATA_SOURCES,
    ImageDataSource,
    ImageDimensionTable,
    TorchvisionImageArtifactPayload,
    validate_relative_image_path,
)
from stochaflow.utils.config import ConfigError, coerce_config_section

MANAGED_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
MANAGED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "artifact_type",
        "source_name",
        "source_digest",
        "materializer_name",
        "materialization_digest",
        "artifact_digest",
        "files",
    }
)


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
    cache_root: Path,
    artifact_root: Path,
    *,
    dataset_name: str,
    partitions: Mapping[str, Any],
) -> None:
    """Write an identity-bound canonical dimension sidecar."""

    serialized = {
        role: torchvision_dimension_table(
            dataset_name,
            partition,
        ).to_pairs()
        for role, partition in sorted(partitions.items())
    }
    write_cache_file(
        cache_root,
        artifact_root / "image_dimensions.json",
        canonical_json_bytes(
            {
                "schema_version": 1,
                "dataset": dataset_name,
                "partitions": serialized,
            }
        ),
        label="torchvision image dimensions",
    )


def load_torchvision_dimensions(
    artifact_root: Path,
    *,
    dataset_name: str,
    expected_record: Mapping[str, Any],
) -> dict[str, ImageDimensionTable]:
    """Load a sidecar after always verifying its manifest digest."""

    encoded, metadata = read_regular_file(
        artifact_root,
        "image_dimensions.json",
        label="torchvision image dimensions",
    )
    if (
        metadata.st_size != expected_record["size_bytes"]
        or len(encoded) != expected_record["size_bytes"]
        or hashlib.sha256(encoded).hexdigest() != expected_record["sha256"]
    ):
        raise ValueError(
            "torchvision image dimensions do not match the managed manifest"
        )
    try:
        raw_value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "torchvision image dimensions contain invalid JSON"
        ) from exc
    if (
        not isinstance(raw_value, dict)
        or encoded != canonical_json_bytes(raw_value)
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


def managed_files(
    root: Path,
    *,
    hash_contents: bool,
) -> tuple[dict[str, Any], ...]:
    """Inventory all managed files except the root manifest."""

    files: list[dict[str, Any]] = []
    snapshots = scan_regular_files(
        root,
        hash_contents=hash_contents,
        label="managed artifact",
        path_filter=lambda relative: relative != "manifest.json",
    )
    for snapshot in snapshots:
        digest = snapshot.sha256 if hash_contents else "0" * 64
        if digest is None:
            raise RuntimeError("hashed managed scan did not return a digest")
        files.append(
            {
                "path": snapshot.relative_path,
                "size_bytes": snapshot.size_bytes,
                "sha256": digest,
            }
        )
    files.sort(key=lambda record: cast(str, record["path"]))
    if not files:
        raise ValueError("managed torchvision artifact contains no files")
    return tuple(files)


def managed_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> ManagedDataArtifactIdentity:
    """Construct the strict identity represented by a managed manifest."""

    return ManagedDataArtifactIdentity(
        artifact_type=manifest["artifact_type"],
        source_name=manifest["source_name"],
        source_digest=manifest["source_digest"],
        materializer_name=manifest["materializer_name"],
        materialization_digest=manifest["materialization_digest"],
        artifact_digest=manifest["artifact_digest"],
        manifest_sha256=manifest_sha256,
    )


def load_managed_torchvision(
    artifact_root: Path,
    *,
    dataset: str,
    verification: Literal["manifest", "full"],
) -> ManagedDataArtifact[TorchvisionImageArtifactPayload]:
    """Load and verify a content-addressed managed torchvision artifact."""

    manifest_path = artifact_root / "manifest.json"
    raw, manifest_bytes = load_canonical_json(
        manifest_path,
        label="managed manifest",
    )
    manifest = strict_mapping(
        raw,
        fields=MANAGED_MANIFEST_FIELDS,
        path="managed manifest",
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["kind"] != "managed"
        or manifest["source_name"] != "torchvision"
        or manifest["artifact_type"] != "stochaflow.torchvision-image.v2"
    ):
        raise ValueError("managed torchvision manifest is incompatible")
    identity = managed_identity(
        manifest,
        manifest_sha256=sha256_bytes(manifest_bytes),
    )
    if artifact_root.name != identity.artifact_digest:
        raise ValueError(
            "managed artifact directory does not match its artifact digest"
        )
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("managed manifest files must be a non-empty list")
    normalized_files: list[dict[str, Any]] = []
    for index, value in enumerate(files):
        record = strict_mapping(
            value,
            fields=MANAGED_FILE_FIELDS,
            path=f"managed manifest.files[{index}]",
        )
        relative_path = record["path"]
        if not isinstance(relative_path, str):
            raise TypeError("managed manifest file path must be a string")
        validate_relative_image_path(
            relative_path,
            path=f"managed manifest.files[{index}].path",
        )
        size_bytes = record["size_bytes"]
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ValueError(
                "managed manifest file size_bytes must be non-negative"
            )
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("managed manifest file sha256 is invalid")
        normalized_files.append(record)
    recorded_paths = [record["path"] for record in normalized_files]
    if recorded_paths != sorted(recorded_paths) or len(recorded_paths) != len(
        set(recorded_paths)
    ):
        raise ValueError("managed manifest files must be sorted and unique")
    live = managed_files(
        artifact_root,
        hash_contents=verification == "full",
    )
    expected_without_hash = tuple(
        (record["path"], record["size_bytes"])
        for record in normalized_files
    )
    live_without_hash = tuple(
        (record["path"], record["size_bytes"]) for record in live
    )
    if live_without_hash != expected_without_hash:
        raise ValueError("managed artifact paths or sizes changed")
    if verification == "full" and tuple(normalized_files) != live:
        raise ValueError("managed artifact content digest changed")
    if identity.source_digest != canonical_digest(normalized_files):
        raise ValueError("managed manifest source digest is invalid")
    materializer_name = "stochaflow.torchvision-download"
    expected_materialization_digest = canonical_digest(
        {
            "name": materializer_name,
            "version": 2,
            "dataset": dataset,
        }
    )
    if (
        identity.materializer_name != materializer_name
        or identity.materialization_digest != expected_materialization_digest
    ):
        raise ValueError(
            "managed manifest materialization identity is invalid"
        )
    expected_artifact_digest = canonical_digest(
        {
            "kind": "managed",
            "artifact_type": identity.artifact_type,
            "source_name": identity.source_name,
            "source_digest": identity.source_digest,
            "materializer_name": identity.materializer_name,
            "materialization_digest": identity.materialization_digest,
        }
    )
    if identity.artifact_digest != expected_artifact_digest:
        raise ValueError("managed manifest artifact digest is invalid")
    dimension_records = [
        record
        for record in normalized_files
        if record["path"] == "image_dimensions.json"
    ]
    if len(dimension_records) != 1:
        raise ValueError(
            "managed torchvision artifact requires image_dimensions.json"
        )
    dimensions = load_torchvision_dimensions(
        artifact_root,
        dataset_name=dataset,
        expected_record=dimension_records[0],
    )
    return ManagedDataArtifact(
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        identity=identity,
        payload=TorchvisionImageArtifactPayload(
            dataset=cast(Any, dataset),
            root=artifact_root / "data",
            train_dimensions=dimensions["train"],
            validation_dimensions=dimensions.get("validation"),
            test_dimensions=dimensions.get("test"),
        ),
    )


@IMAGE_DATA_SOURCES.register("torchvision")
class TorchvisionImageDataSource(ImageDataSource):
    """Acquire and content-address one allowlisted torchvision dataset."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ManagedDataArtifact[TorchvisionImageArtifactPayload]:
        config = cast(
            TorchvisionImageSourceConfig,
            coerce_config_section(
                TorchvisionImageSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        expected = context.expected_identity
        pointer = (
            context.cache_root
            / "managed"
            / "torchvision"
            / config.dataset
            / "current.json"
        )
        if expected is not None:
            if not isinstance(expected, ManagedDataArtifactIdentity):
                raise ValueError(
                    "strict resume expected a different data artifact kind"
                )
            if expected.source_name != "torchvision":
                raise ValueError(
                    "strict resume expected a different managed data source"
                )
            artifact_digest: str | None = expected.artifact_digest
        else:
            artifact_digest = read_locator_for_policy(
                context.cache_root,
                pointer,
                policy=context.policy,
                quarantine_on_error=False,
            )
        artifact_parent = (
            context.cache_root
            / "managed"
            / "torchvision"
            / config.dataset
            / "artifacts"
        )
        if artifact_digest is not None:
            try:
                artifact = load_managed_torchvision(
                    artifact_parent / artifact_digest,
                    dataset=config.dataset,
                    verification=context.verification,
                )
                if expected is not None and artifact.identity != expected:
                    raise ValueError(
                        "strict resume managed data identity does not match"
                    )
                return artifact
            except (FileNotFoundError, OSError, TypeError, ValueError):
                if context.policy == "require":
                    raise
        if context.policy == "require":
            raise FileNotFoundError(
                f"required torchvision artifact is missing: {config.dataset}"
            )
        lock = (
            context.cache_root
            / "managed"
            / "torchvision"
            / config.dataset
            / "materialize.lock"
        )
        with ArtifactMaterializationLock(
            lock,
            cache_root=context.cache_root,
        ):
            winner_digest = (
                expected.artifact_digest
                if expected is not None
                else read_locator_for_policy(
                    context.cache_root,
                    pointer,
                    policy="ensure",
                )
            )
            if winner_digest is not None:
                winner_root = artifact_parent / winner_digest
                try:
                    winner = load_managed_torchvision(
                        winner_root,
                        dataset=config.dataset,
                        verification=context.verification,
                    )
                    if expected is not None and winner.identity != expected:
                        raise ValueError(
                            "strict resume managed data identity does not match"
                        )
                    return winner
                except (FileNotFoundError, OSError, TypeError, ValueError):
                    if path_exists_without_following(
                        context.cache_root,
                        winner_root,
                    ):
                        quarantine_path(context.cache_root, winner_root)
            staging = artifact_parent / f".staging.{uuid4().hex}"
            data_root = staging / "data"
            create_cache_directory(
                context.cache_root,
                staging,
                label="managed artifact staging directory",
            )
            create_cache_directory(
                context.cache_root,
                data_root,
                label="managed artifact data directory",
            )
            try:
                partitions = torchvision_datasets(
                    config.dataset,
                    data_root,
                    download=True,
                )
                write_torchvision_dimensions(
                    context.cache_root,
                    staging,
                    dataset_name=config.dataset,
                    partitions=partitions,
                )
                files = managed_files(staging, hash_contents=True)
                source_digest = canonical_digest(files)
                materializer_name = "stochaflow.torchvision-download"
                materialization_digest = canonical_digest(
                    {
                        "name": materializer_name,
                        "version": 2,
                        "dataset": config.dataset,
                    }
                )
                artifact_digest = canonical_digest(
                    {
                        "kind": "managed",
                        "artifact_type": "stochaflow.torchvision-image.v2",
                        "source_name": "torchvision",
                        "source_digest": source_digest,
                        "materializer_name": materializer_name,
                        "materialization_digest": materialization_digest,
                    }
                )
                manifest = {
                    "schema_version": 1,
                    "kind": "managed",
                    "artifact_type": "stochaflow.torchvision-image.v2",
                    "source_name": "torchvision",
                    "source_digest": source_digest,
                    "materializer_name": materializer_name,
                    "materialization_digest": materialization_digest,
                    "artifact_digest": artifact_digest,
                    "files": list(files),
                }
                manifest_bytes = canonical_json_bytes(manifest)
                write_cache_file(
                    context.cache_root,
                    staging / "manifest.json",
                    manifest_bytes,
                    label="managed artifact manifest",
                )
                staging_identity = managed_identity(
                    manifest,
                    manifest_sha256=sha256_bytes(manifest_bytes),
                )
                if expected is not None and staging_identity != expected:
                    raise ValueError(
                        "strict resume managed data identity does not match"
                    )
                final = artifact_parent / artifact_digest
                if path_exists_without_following(context.cache_root, final):
                    try:
                        winner = load_managed_torchvision(
                            final,
                            dataset=config.dataset,
                            verification="full",
                        )
                    except (FileNotFoundError, OSError, TypeError, ValueError):
                        quarantine_path(context.cache_root, final)
                    else:
                        remove_cache_directory(
                            context.cache_root,
                            staging,
                            label="managed artifact staging directory",
                        )
                        if expected is not None and winner.identity != expected:
                            raise ValueError(
                                "strict resume managed data identity "
                                "does not match"
                            )
                        if expected is None:
                            write_locator(
                                context.cache_root,
                                context.cache_root
                                / "managed"
                                / "torchvision"
                                / config.dataset
                                / "current.json",
                                winner.identity.artifact_digest,
                            )
                        return winner
                publish_cache_directory(
                    context.cache_root,
                    staging,
                    final,
                    label="managed artifact",
                )
                artifact = load_managed_torchvision(
                    final,
                    dataset=config.dataset,
                    verification=context.verification,
                )
                if expected is not None and artifact.identity != expected:
                    raise ValueError(
                        "strict resume managed data identity does not match"
                    )
                if expected is None:
                    write_locator(
                        context.cache_root,
                        context.cache_root
                        / "managed"
                        / "torchvision"
                        / config.dataset
                        / "current.json",
                        artifact.identity.artifact_digest,
                    )
                return artifact
            except BaseException:
                if path_exists_without_following(
                    context.cache_root,
                    staging,
                ):
                    remove_cache_directory(
                        context.cache_root,
                        staging,
                        label="managed artifact staging directory",
                    )
                raise


__all__ = [
    "TorchvisionImageDataSource",
    "TorchvisionImageSourceConfig",
    "load_managed_torchvision",
    "load_torchvision_dimensions",
    "managed_files",
    "managed_identity",
    "torchvision_datasets",
    "torchvision_dimension_table",
    "write_torchvision_dimensions",
]
