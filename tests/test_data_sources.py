"""Tests for artifact-backed image sources and Dataset adaptation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any, cast
from uuid import UUID

import pytest
from PIL import Image

from stochaflow.data import (
    DataArtifact,
    DataArtifactBindings,
    DataBuilderContext,
    DataSourceContext,
    ImageDataSource,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
    ReferencedDataArtifact,
    artifact_io,
    artifact_store,
    folder_sources,
    reference_artifacts,
    torchvision_source,
)
from stochaflow.data.datasets import ImageDatasetFactory
from stochaflow.data.folder_sources import (
    ImageFolderDataSource,
    PairedImageFolderDataSource,
)
from stochaflow.data.image_contracts import (
    PairedImageFolderArtifactPayload,
)
from stochaflow.data.recipe_config import (
    DataSourceMaterializationConfig,
    ImageSourceConfig,
)
from stochaflow.data.source_factory import ImageSourceFactory
from stochaflow.data.torchvision_source import TorchvisionImageDataSource
from stochaflow.utils.registry import Registry

TEST_IMAGE_SOURCES: Registry[type[ImageDataSource]] = Registry(
    "test image data source",
    expected_type=ImageDataSource,
)


def create_directory_link(path: Path, target: Path) -> None:
    """Create a directory symlink or Windows junction for security tests."""

    try:
        path.symlink_to(target, target_is_directory=True)
        return
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory links are unavailable: {error}")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(path), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory junctions are unavailable: {result.stderr}")


@TEST_IMAGE_SOURCES.register("tests.fake-image")
class FakeImageDataSource(ImageDataSource):
    """Test extension returning the public folder payload contract."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ManagedDataArtifact[ImageFolderArtifactPayload]:
        del context
        root = Path(self.params["root"]).resolve()
        image_path = root / "sample.png"
        encoded = image_path.read_bytes()
        with Image.open(image_path) as image:
            width, height = image.size
        manifest = root / "manifest.json"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        identity = ManagedDataArtifactIdentity(
            artifact_type="tests.image-folder.v1",
            source_name="tests.fake-image",
            source_digest="a" * 64,
            materializer_name="tests.fixture",
            materialization_digest="b" * 64,
            artifact_digest="c" * 64,
            manifest_sha256=digest,
        )
        return ManagedDataArtifact(
            artifact_root=root,
            manifest_path=manifest,
            identity=identity,
            payload=ImageFolderArtifactPayload(
                roots={"train": root},
                train=(
                    ImageFileRecord(
                        tree="train",
                        path="sample.png",
                        size_bytes=len(encoded),
                        sha256=hashlib.sha256(encoded).hexdigest(),
                        width=width,
                        height=height,
                    ),
                ),
            ),
        )


@TEST_IMAGE_SOURCES.register("tests.incompatible-image")
class IncompatibleImageDataSource(ImageDataSource):
    """Test extension deliberately violating the image payload contract."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[Any]:
        del context
        root = Path(self.params["root"]).resolve()
        manifest = root / "manifest.json"
        identity = ManagedDataArtifactIdentity(
            artifact_type="tests.incompatible.v1",
            source_name="tests.incompatible-image",
            source_digest="a" * 64,
            materializer_name="tests.fixture",
            materialization_digest="b" * 64,
            artifact_digest="c" * 64,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
        return ManagedDataArtifact(
            artifact_root=root,
            manifest_path=manifest,
            identity=identity,
            payload=object(),
        )


def write_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (20, 40, 60),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def source_context(
    cache_root: Path,
    *,
    policy: str = "ensure",
    verification: str = "full",
) -> DataSourceContext:
    return DataSourceContext(
        cache_root=cache_root,
        policy=cast(Any, policy),
        verification=cast(Any, verification),
    )


class FixtureManagedVisionDataset:
    """Acquisition-compatible torchvision double writing deterministic bytes."""

    acquisitions = 0

    def __init__(
        self,
        root: str,
        *,
        train: bool,
        download: bool,
        **kwargs: Any,
    ) -> None:
        del kwargs
        data_root = Path(root)
        marker = "train" if train else "test"
        if download:
            type(self).acquisitions += 1
            data_root.mkdir(parents=True, exist_ok=True)
            (data_root / f"{marker}.bin").write_bytes(marker.encode("ascii"))
        if not (data_root / f"{marker}.bin").is_file():
            raise RuntimeError("fixture managed dataset is unavailable")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        if index != 0:
            raise IndexError(index)
        return Image.new("L", (28, 28)), 0


def test_reference_source_indexes_without_copy_and_reads_manifest_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "z.png", color=(255, 0, 0))
    write_image(root / "nested" / "a.jpg", color=(0, 255, 0))
    source = ImageFolderDataSource(
        {"root": str(root), "layout": "flat"},
        config_path="data.params.source",
    )

    artifact = source.materialize(
        source_context(tmp_path / "cache")
    )
    dataset = ImageDatasetFactory().build(artifact).train

    assert isinstance(artifact, ReferencedDataArtifact)
    assert [record.path for record in artifact.payload.train] == [
        "nested/a.jpg",
        "z.png",
    ]
    assert len(cast(Sized, dataset)) == 2
    assert all(
        path.suffix in {".json", ".jsonl"}
        for path in artifact.index_root.rglob("*")
        if path.is_file()
    )
    assert not any(
        path.suffix in {".png", ".jpg"}
        for path in (tmp_path / "cache").rglob("*")
    )


@pytest.mark.parametrize("cache_inside_source", [False, True])
def test_reference_source_rejects_cache_and_source_tree_overlap(
    tmp_path: Path,
    cache_inside_source: bool,
) -> None:
    if cache_inside_source:
        root = tmp_path / "images"
        cache_root = root / "cache"
    else:
        cache_root = tmp_path / "cache"
        root = cache_root / "images"
    write_image(root / "sample.png")
    source = ImageFolderDataSource(
        {"root": str(root), "layout": "flat"},
        config_path="data.params.source",
    )

    with pytest.raises(ValueError, match="must not overlap"):
        source.materialize(source_context(cache_root))


def test_reference_dataset_detects_same_size_tampering_on_every_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    image_path = root / "sample.bmp"
    write_image(image_path, color=(1, 2, 3))
    artifact = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    dataset = ImageDatasetFactory().build(artifact).train

    first = dataset[0]
    original_size = image_path.stat().st_size
    write_image(image_path, color=(4, 5, 6))

    assert first.size == (8, 8)
    assert image_path.stat().st_size == original_size
    with pytest.raises(ValueError, match="content changed"):
        dataset[0]


def test_reference_dataset_rejects_link_substitution_after_materialization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    image_path = root / "sample.png"
    target_path = tmp_path / "same-content.png"
    write_image(image_path)
    target_path.write_bytes(image_path.read_bytes())
    artifact = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    dataset = ImageDatasetFactory().build(artifact).train
    image_path.unlink()
    try:
        image_path.symlink_to(target_path)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="link or reparse"):
        dataset[0]


def test_manifest_verification_enumerates_paths_and_sizes_without_hashing_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    source.materialize(source_context(cache))
    scan_modes: list[bool] = []
    original = reference_artifacts._scan_regular_file_snapshots

    def record_scan(
        root: Path,
        *,
        hash_contents: bool,
        label: str,
        path_filter: Any = None,
    ) -> tuple[Any, ...]:
        if label == "referenced data":
            scan_modes.append(hash_contents)
        return original(
            root,
            hash_contents=hash_contents,
            label=label,
            path_filter=path_filter,
        )

    monkeypatch.setattr(
        reference_artifacts,
        "_scan_regular_file_snapshots",
        record_scan,
    )
    source.materialize(
        source_context(cache, policy="require", verification="manifest")
    )

    assert scan_modes == [False]

    write_image(root / "added.png")
    with pytest.raises(ValueError, match="paths or sizes changed"):
        source.materialize(
            source_context(cache, policy="require", verification="manifest")
        )


def test_reference_scan_hashes_only_inventory_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    image = root / "sample.png"
    write_image(image)
    ignored = root / "large-sidecar.txt"
    ignored.write_bytes(b"x" * 4096)
    read_paths: list[str] = []
    original = reference_artifacts.read_regular_file

    def record_read(
        selected_root: Path,
        relative_path: str,
        *,
        label: str,
    ) -> tuple[bytes, os.stat_result]:
        if label == "referenced image":
            read_paths.append(relative_path)
        return original(selected_root, relative_path, label=label)

    monkeypatch.setattr(
        reference_artifacts,
        "read_regular_file",
        record_read,
    )

    ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))

    assert read_paths == ["sample.png"]
    assert ignored.name not in read_paths


def test_reference_scan_rejects_special_files(tmp_path: Path) -> None:
    make_fifo = getattr(os, "mkfifo", None)
    if not callable(make_fifo):
        pytest.skip("FIFO creation is unavailable")
    root = tmp_path / "images"
    write_image(root / "sample.png")
    try:
        make_fifo(root / "named-pipe")
    except OSError:
        pytest.skip("FIFO creation is unavailable")

    with pytest.raises(ValueError, match="unsupported filesystem entry"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "cache"))


def test_anchored_regular_file_open_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    make_fifo = getattr(os, "mkfifo", None)
    if not callable(make_fifo):
        pytest.skip("FIFO creation is unavailable")
    root = tmp_path / "files"
    root.mkdir()
    fifo = root / "named-pipe"
    try:
        make_fifo(fifo)
    except OSError:
        pytest.skip("FIFO creation is unavailable")

    with pytest.raises(ValueError, match="not a regular file"):
        artifact_io.read_regular_file(
            root,
            "named-pipe",
            label="FIFO regression",
        )


def test_open_cache_file_concurrent_creation_converges(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    lock_path = cache_root / ".locks" / "materialize.lock"

    def open_lock() -> int:
        return artifact_io.open_cache_file(
            cache_root,
            lock_path,
            label="concurrent lock",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(open_lock) for _ in range(2)]
        descriptors = [future.result() for future in futures]
    try:
        assert lock_path.is_file()
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_require_miss_is_read_only(tmp_path: Path) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "missing-cache"

    with pytest.raises(FileNotFoundError, match="not indexed"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(cache, policy="require"))

    assert not cache.exists()


def test_reference_cache_cannot_be_nested_in_source(tmp_path: Path) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")

    with pytest.raises(ValueError, match="must not overlap"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(root / ".cache"))


def test_reference_rejects_symlinks_and_unicode_collisions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "real.png")
    link = root / "linked.png"
    try:
        link.symlink_to(root / "real.png")
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="links or reparse"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "cache"))


def test_reference_rejects_non_nfc_disk_paths(tmp_path: Path) -> None:
    root = tmp_path / "images"
    decomposed_name = "e\u0301.png"
    image_path = root / decomposed_name
    write_image(image_path)
    if image_path.name != decomposed_name:
        pytest.skip("filesystem normalizes names before storage")

    with pytest.raises(ValueError, match="NFC normalization"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "cache"))


def test_configured_root_rejects_intermediate_directory_symlink(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    write_image(real_parent / "images" / "sample.png")
    linked_parent = tmp_path / "linked"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(ValueError, match="contains a symlink or reparse"):
        ImageFolderDataSource(
            {"root": str(linked_parent / "images")},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "cache"))


def test_cache_mutations_reject_linked_parent_without_touching_target(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = cache / "linked"
    create_directory_link(linked, outside)
    (outside / "staging").mkdir()
    corrupt = outside / "corrupt.json"
    corrupt.write_bytes(b"preserve")

    with pytest.raises(ValueError, match=r"symlink|reparse|invalid"):
        artifact_io.write_cache_file(
            cache,
            linked / "locator.json",
            b"unsafe",
            label="fixture locator",
        )
    with pytest.raises(ValueError, match=r"symlink|reparse|invalid"):
        artifact_io.create_cache_file_exclusive(
            cache,
            linked / "exclusive.tmp",
            label="fixture exclusive staging",
        )
    with pytest.raises(ValueError, match=r"symlink|reparse|invalid"):
        artifact_io.publish_cache_directory(
            cache,
            linked / "staging",
            linked / "published",
            label="fixture publication",
        )
    with pytest.raises(ValueError, match=r"symlink|reparse|invalid"):
        artifact_io.quarantine_cache_entry(
            cache,
            linked / "corrupt.json",
            suffix="corrupt",
            label="fixture quarantine",
        )

    assert not (outside / "locator.json").exists()
    assert not (outside / "exclusive.tmp").exists()
    assert (outside / "staging").is_dir()
    assert not (outside / "published").exists()
    assert corrupt.read_bytes() == b"preserve"
    assert not list(outside.glob("*.corrupt"))


def test_cache_publication_and_quarantine_never_replace_existing_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    staging = artifact_io.create_cache_directory(
        cache,
        cache / "entries" / "staging",
        label="fixture staging",
    )
    destination = artifact_io.create_cache_directory(
        cache,
        cache / "entries" / "published",
        label="fixture destination",
    )

    with pytest.raises(FileExistsError):
        artifact_io.publish_cache_directory(
            cache,
            staging,
            destination,
            label="fixture publication",
        )
    assert staging.is_dir()
    assert destination.is_dir()

    source = cache / "entries" / "corrupt.json"
    source.write_bytes(b"source")
    fixed_uuid = UUID(int=0)
    collision = source.with_name(f"{source.name}.{fixed_uuid.hex}.corrupt")
    collision.write_bytes(b"winner")
    monkeypatch.setattr(artifact_io, "uuid4", lambda: fixed_uuid)

    with pytest.raises(FileExistsError):
        artifact_io.quarantine_cache_entry(
            cache,
            source,
            suffix="corrupt",
            label="fixture quarantine",
        )
    assert source.read_bytes() == b"source"
    assert collision.read_bytes() == b"winner"


def test_reference_materialization_rejects_linked_index_cache_ancestor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"
    index_parent = reference_artifacts._reference_index_path(
        cache,
        "image_folder",
        "0" * 64,
    ).parent
    index_parent.parent.mkdir(parents=True)
    outside = tmp_path / "outside-reference-index"
    outside.mkdir()
    create_directory_link(index_parent, outside)

    with pytest.raises(ValueError, match=r"symlink|reparse|invalid"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(cache))

    assert not list(outside.iterdir())


def test_managed_materialization_rejects_linked_artifact_cache_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FixtureManagedVisionDataset.acquisitions = 0
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    cache = tmp_path / "cache"
    artifact_parent = (
        cache / "managed" / "torchvision" / "mnist" / "artifacts"
    )
    artifact_parent.parent.mkdir(parents=True)
    outside = tmp_path / "outside-managed-artifacts"
    outside.mkdir()
    create_directory_link(artifact_parent, outside)

    with pytest.raises(ValueError, match=r"symlink|reparse|invalid"):
        TorchvisionImageDataSource(
            {"dataset": "MNIST"},
            config_path="data.params.source",
        ).materialize(source_context(cache))

    assert FixtureManagedVisionDataset.acquisitions == 0
    assert not list(outside.iterdir())


def test_reference_scan_rejects_root_link_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    probe = tmp_path / "link-probe"
    try:
        probe.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    probe.unlink()
    original = folder_sources.partition_roots

    def substitute_root(selected_root: Path, layout: str) -> dict[str, Path]:
        roots = original(selected_root, layout)
        preserved = selected_root.with_name("preserved-images")
        selected_root.rename(preserved)
        selected_root.symlink_to(preserved, target_is_directory=True)
        return roots

    monkeypatch.setattr(folder_sources, "partition_roots", substitute_root)

    with pytest.raises(ValueError, match="link or reparse"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "cache"))


def test_reference_inventory_shards_at_one_hundred_thousand(
    tmp_path: Path,
) -> None:
    records = tuple(
        ImageFileRecord(
            tree="train",
            path=f"{index:06d}.png",
            size_bytes=1,
            sha256="a" * 64,
            width=1,
            height=1,
        )
        for index in range(100_001)
    )
    index_root = tmp_path / "index"
    index_root.mkdir()

    inventory = reference_artifacts._write_inventory(
        tmp_path,
        index_root,
        records,
    )

    assert inventory["record_count"] == 100_001
    assert [shard["record_count"] for shard in inventory["shards"]] == [
        100_000,
        1,
    ]
    assert reference_artifacts._read_inventory(
        index_root,
        inventory,
    ) == records


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("record_limit", True, "record_limit"),
        ("record_limit", 100_000.0, "record_limit"),
        ("record_count", True, "record_count"),
        ("record_count", 1.0, "record_count"),
    ],
)
def test_reference_inventory_rejects_non_integer_counts(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    record = ImageFileRecord("train", "sample.png", 1, "a" * 64, 1, 1)
    index_root = tmp_path / "index"
    index_root.mkdir()
    inventory = reference_artifacts._write_inventory(
        tmp_path,
        index_root,
        (record,),
    )
    inventory[field] = value

    with pytest.raises(ValueError, match=message):
        reference_artifacts._read_inventory(index_root, inventory)


def test_reference_inventory_requires_canonical_shard_shape(
    tmp_path: Path,
) -> None:
    record = ImageFileRecord("train", "sample.png", 1, "a" * 64, 1, 1)
    index_root = tmp_path / "index"
    index_root.mkdir()
    inventory = reference_artifacts._write_inventory(
        tmp_path,
        index_root,
        (record,),
    )

    non_integer = cast(dict[str, Any], inventory["shards"][0]).copy()
    non_integer["record_count"] = True
    malformed = {**inventory, "shards": [non_integer]}
    with pytest.raises(ValueError, match="record count is not canonical"):
        reference_artifacts._read_inventory(index_root, malformed)

    extra = cast(dict[str, Any], inventory["shards"][0]).copy()
    extra["path"] = "inventory/000001.jsonl"
    malformed = {**inventory, "shards": [inventory["shards"][0], extra]}
    with pytest.raises(ValueError, match="shard count is not canonical"):
        reference_artifacts._read_inventory(index_root, malformed)

    malformed = {
        **inventory,
        "record_count": 100_001,
        "shards": [inventory["shards"][0]],
    }
    with pytest.raises(ValueError, match="shard count is not canonical"):
        reference_artifacts._read_inventory(index_root, malformed)


def test_paired_source_matches_relative_stems_and_detects_missing_pairs(
    tmp_path: Path,
) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    write_image(high / "nested" / "sample.png")
    write_image(low / "nested" / "sample.jpg")
    source = PairedImageFolderDataSource(
        {
            "high_resolution_root": str(high),
            "low_resolution_root": str(low),
        },
        config_path="data.params.source",
    )

    artifact = source.materialize(source_context(tmp_path / "cache"))

    assert isinstance(artifact.payload, PairedImageFolderArtifactPayload)
    assert len(artifact.payload.train) == 1

    (low / "nested" / "sample.jpg").unlink()
    write_image(low / "nested" / "other.jpg")
    with pytest.raises(ValueError, match="missing LR"):
        source.materialize(
            source_context(
                tmp_path / "other-cache",
                verification="full",
            )
        )


def test_source_factory_enforces_strict_binding_before_materialization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    config = ImageSourceConfig(
        name="image_folder",
        params={"root": str(root)},
        materialization=DataSourceMaterializationConfig(
            cache_root=str(tmp_path / "cache")
        ),
    )
    context = DataBuilderContext(
        params={},
        seed=1,
        strict_resume=True,
        expected_artifacts=DataArtifactBindings(),
    )

    with pytest.raises(KeyError, match="missing data artifact binding"):
        ImageSourceFactory().materialize(
            config,
            binding_id="source",
            builder_context=context,
            path="data.params.source",
        )
    assert not (tmp_path / "cache").exists()


def test_reference_identity_is_location_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_image(first / "sample.png")
    second.mkdir()
    (second / "sample.png").write_bytes((first / "sample.png").read_bytes())

    first_artifact = ImageFolderDataSource(
        {"root": str(first)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache-one"))
    second_artifact = ImageFolderDataSource(
        {"root": str(second)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache-two"))

    assert first_artifact.identity == second_artifact.identity
    encoded = first_artifact.manifest_path.read_text(encoding="utf-8")
    assert str(first.resolve()) not in encoded
    assert str(second.resolve()) not in encoded


def test_concurrent_reference_roots_converge_on_content_address(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_image(first / "sample.png")
    second.mkdir()
    (second / "sample.png").write_bytes((first / "sample.png").read_bytes())
    cache = tmp_path / "cache"

    def materialize(root: Path) -> ReferencedDataArtifact[Any]:
        return ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(cache))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(materialize, first)
        second_future = executor.submit(materialize, second)
        first_artifact = first_future.result(timeout=10)
        second_artifact = second_future.result(timeout=10)

    assert first_artifact.identity == second_artifact.identity
    assert first_artifact.index_root == second_artifact.index_root


def test_corrupt_reference_index_is_quarantined_and_rebuilt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    first = source.materialize(source_context(cache))
    first.manifest_path.write_text("corrupt", encoding="utf-8")

    rebuilt = source.materialize(source_context(cache))

    assert rebuilt.identity == first.identity
    quarantined = list(
        first.index_root.parent.glob(f"{first.index_root.name}.*.corrupt")
    )
    assert len(quarantined) == 1
    assert rebuilt.manifest_path.is_file()


def test_orphan_reference_index_with_type_error_is_quarantined_and_rebuilt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    first = source.materialize(source_context(cache))
    locator = next((cache / "references" / "locators").rglob("*.json"))
    locator.unlink()
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_type"] = 7
    first.manifest_path.write_bytes(
        artifact_store.canonical_json_bytes(manifest)
    )

    rebuilt = source.materialize(source_context(cache))

    assert rebuilt.identity == first.identity
    quarantined = first.index_root.parent.glob(
        f"{first.index_root.name}.*.corrupt"
    )
    assert len(list(quarantined)) == 1


def test_reference_locator_recovery_is_policy_aware(tmp_path: Path) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    first = source.materialize(source_context(cache))
    locator = next(
        (cache / "references" / "locators").rglob("*.json")
    )
    malformed = b'{"artifact_digest":true}\n'
    locator.write_bytes(malformed)

    with pytest.raises(ValueError, match="artifact_digest is invalid"):
        source.materialize(
            source_context(cache, policy="require", verification="full")
        )
    assert locator.read_bytes() == malformed
    assert not list(locator.parent.glob("*.corrupt"))

    rebuilt = source.materialize(source_context(cache))

    assert rebuilt.identity == first.identity
    assert artifact_store.read_locator(
        locator
    ) == first.identity.artifact_digest
    assert len(list(locator.parent.glob("*.corrupt"))) == 1


def test_pointer_digest_is_strict_lowercase_sha256(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.json"
    pointer.write_bytes(b'{"artifact_digest":"XYZ"}\n')

    with pytest.raises(ValueError, match="artifact_digest is invalid"):
        artifact_store.read_locator(pointer)


def test_reference_index_directory_must_match_artifact_digest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    artifact = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    wrong_root = artifact.index_root.with_name("f" * 64)
    artifact.index_root.rename(wrong_root)

    with pytest.raises(ValueError, match=r"directory.*artifact digest"):
        reference_artifacts._load_reference_index(
            wrong_root,
            source_name="image_folder",
            artifact_type="stochaflow.image-folder-reference.v2",
            roots=artifact.payload.roots,
            layout={
                "type": "image_folder",
                "mode": "flat",
                "trees": ["train"],
            },
            verification="full",
        )


def test_reference_manifest_rejects_boolean_schema_version(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    artifact = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    manifest = json.loads(
        artifact.manifest_path.read_text(encoding="utf-8")
    )
    manifest["schema_version"] = True
    artifact.manifest_path.write_bytes(
        artifact_store.canonical_json_bytes(manifest)
    )

    with pytest.raises(ValueError, match="unsupported schema"):
        reference_artifacts._load_reference_index(
            artifact.index_root,
            source_name="image_folder",
            artifact_type="stochaflow.image-folder-reference.v2",
            roots=artifact.payload.roots,
            layout={
                "type": "image_folder",
                "mode": "flat",
                "trees": ["train"],
            },
            verification="full",
        )


def test_reference_lock_identity_includes_external_roots(
    tmp_path: Path,
) -> None:
    layout = {
        "type": "image_folder",
        "mode": "flat",
        "trees": ["train"],
    }

    first = reference_artifacts._reference_lock_path(
        tmp_path / "cache",
        "image_folder",
        {"train": tmp_path / "first"},
        layout,
    )
    second = reference_artifacts._reference_lock_path(
        tmp_path / "cache",
        "image_folder",
        {"train": tmp_path / "second"},
        layout,
    )

    assert first != second


def test_materialization_lock_wait_is_bounded_and_preserves_owner(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "materialize.lock"

    with artifact_store.ArtifactMaterializationLock(lock_path):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["hostname"]
        assert owner["pid"] > 0
        assert owner["created_at_ns"] > 0
        assert len(owner["nonce"]) == 32

        with (
            pytest.raises(
                RuntimeError,
                match=r"timed out.*observed owner",
            ),
            artifact_store.ArtifactMaterializationLock(
                lock_path,
                wait_seconds=0.02,
                poll_seconds=0.005,
            ),
        ):
            pass
        assert lock_path.is_file()
        assert json.loads(lock_path.read_text(encoding="utf-8")) == owner

    assert lock_path.is_file()
    assert json.loads(lock_path.read_text(encoding="utf-8")) == owner


def test_materialization_lock_reuses_persistent_unlocked_file(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "materialize.lock"
    stale = b'{"hostname":"other-host","pid":123}\n'
    lock_path.write_bytes(stale)

    with artifact_store.ArtifactMaterializationLock(lock_path):
        current = json.loads(lock_path.read_text(encoding="utf-8"))
        assert current["hostname"]
        assert current["nonce"]

    assert lock_path.is_file()
    assert lock_path.read_bytes() != stale


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX permits replacing a locked pathname",
)
def test_materialization_lock_release_preserves_replacement_owner(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "materialize.lock"
    displaced = tmp_path / "displaced.lock"
    replacement = b'{"hostname":"replacement","pid":999}\n'

    with artifact_store.ArtifactMaterializationLock(lock_path):
        lock_path.rename(displaced)
        lock_path.write_bytes(replacement)

    assert lock_path.read_bytes() == replacement
    assert displaced.is_file()


@pytest.mark.skipif(
    os.name != "nt",
    reason="Windows handle sharing regression",
)
def test_windows_materialization_lock_blocks_path_replacement(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "materialize.lock"

    with (
        artifact_store.ArtifactMaterializationLock(lock_path),
        pytest.raises(PermissionError),
    ):
        lock_path.rename(tmp_path / "replacement.lock")

    assert lock_path.is_file()


def test_managed_torchvision_require_miss_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    cache = tmp_path / "cache"

    with pytest.raises(FileNotFoundError, match="required torchvision"):
        TorchvisionImageDataSource(
            {"dataset": "MNIST"},
            config_path="data.params.source",
        ).materialize(source_context(cache, policy="require"))

    assert not cache.exists()


def test_managed_torchvision_cache_hit_and_verification_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FixtureManagedVisionDataset.acquisitions = 0
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    cache = tmp_path / "cache"
    source = TorchvisionImageDataSource(
        {"dataset": "MNIST"},
        config_path="data.params.source",
    )

    first = source.materialize(source_context(cache))
    second = source.materialize(source_context(cache))

    assert second.identity == first.identity
    assert FixtureManagedVisionDataset.acquisitions == 2

    train_file = first.artifact_root / "data" / "train.bin"
    train_file.write_bytes(b"xxxxx")
    manifest_only = source.materialize(
        source_context(cache, policy="require", verification="manifest")
    )
    assert manifest_only.identity == first.identity
    with pytest.raises(ValueError, match="content digest changed"):
        source.materialize(
            source_context(cache, policy="require", verification="full")
        )


def test_managed_locator_recovery_is_policy_aware(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FixtureManagedVisionDataset.acquisitions = 0
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    cache = tmp_path / "cache"
    source = TorchvisionImageDataSource(
        {"dataset": "MNIST"},
        config_path="data.params.source",
    )
    first = source.materialize(source_context(cache))
    pointer = cache / "managed" / "torchvision" / "mnist" / "current.json"
    malformed = b'{"artifact_digest":1.0}\n'
    pointer.write_bytes(malformed)

    with pytest.raises(ValueError, match="artifact_digest is invalid"):
        source.materialize(
            source_context(cache, policy="require", verification="full")
        )
    assert pointer.read_bytes() == malformed

    rebuilt = source.materialize(source_context(cache))

    assert rebuilt.identity == first.identity
    assert artifact_store.read_locator(
        pointer
    ) == first.identity.artifact_digest
    assert len(list(pointer.parent.glob("*.corrupt"))) == 1


def test_managed_artifact_directory_must_match_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    artifact = TorchvisionImageDataSource(
        {"dataset": "MNIST"},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    wrong_root = artifact.artifact_root.with_name("f" * 64)
    artifact.artifact_root.rename(wrong_root)

    with pytest.raises(ValueError, match=r"directory.*artifact digest"):
        torchvision_source.load_managed_torchvision(
            wrong_root,
            dataset="mnist",
            verification="full",
        )


def test_managed_manifest_rejects_boolean_schema_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    artifact = TorchvisionImageDataSource(
        {"dataset": "MNIST"},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    manifest = json.loads(
        artifact.manifest_path.read_text(encoding="utf-8")
    )
    manifest["schema_version"] = True
    artifact.manifest_path.write_bytes(
        artifact_store.canonical_json_bytes(manifest)
    )

    with pytest.raises(ValueError, match="incompatible"):
        torchvision_source.load_managed_torchvision(
            artifact.artifact_root,
            dataset="mnist",
            verification="full",
        )


def test_concurrent_managed_ensure_rechecks_published_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FixtureManagedVisionDataset.acquisitions = 0
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    original = torchvision_source.torchvision_datasets
    entered = Event()
    release = Event()
    call_lock = Lock()
    calls = 0

    def blocking_acquisition(
        dataset: str,
        root: Path,
        *,
        download: bool,
    ) -> dict[str, Any]:
        nonlocal calls
        with call_lock:
            calls += 1
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test acquisition release timed out")
        return original(dataset, root, download=download)

    monkeypatch.setattr(
        torchvision_source,
        "torchvision_datasets",
        blocking_acquisition,
    )
    cache = tmp_path / "cache"

    def materialize() -> ManagedDataArtifact[Any]:
        return TorchvisionImageDataSource(
            {"dataset": "MNIST"},
            config_path="data.params.source",
        ).materialize(source_context(cache))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(materialize)
        assert entered.wait(timeout=5)
        second_future = executor.submit(materialize)
        time.sleep(0.05)
        release.set()
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first.identity == second.identity
    assert calls == 1
    assert FixtureManagedVisionDataset.acquisitions == 2


def test_managed_corruption_is_quarantined_and_strict_ensure_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FixtureManagedVisionDataset.acquisitions = 0
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    cache = tmp_path / "cache"
    source = TorchvisionImageDataSource(
        {"dataset": "MNIST"},
        config_path="data.params.source",
    )
    first = source.materialize(source_context(cache))
    first.manifest_path.write_bytes(b"corrupt")

    rebuilt = source.materialize(source_context(cache))

    assert rebuilt.identity == first.identity
    assert list(
        first.artifact_root.parent.glob(
            f"{first.artifact_root.name}.*.corrupt"
        )
    )

    shutil.rmtree(rebuilt.artifact_root)
    strict = source.materialize(
        DataSourceContext(
            cache_root=cache,
            policy="ensure",
            verification="manifest",
            expected_identity=rebuilt.identity,
        )
    )

    assert strict.identity == rebuilt.identity
    assert strict.manifest_path.is_file()


def test_extension_source_reuses_public_payload_without_name_dispatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    write_image(root / "sample.png")
    (root / "manifest.json").write_bytes(b"{}\n")
    factory = ImageSourceFactory(TEST_IMAGE_SOURCES)
    context = DataBuilderContext(params={}, seed=1)
    config = ImageSourceConfig(
        name="tests.fake-image",
        params={"root": str(root)},
        materialization=DataSourceMaterializationConfig(
            cache_root=str(tmp_path / "cache")
        ),
    )

    artifact = factory.materialize(
        config,
        binding_id="source",
        builder_context=context,
        path="data.params.source",
    )
    dataset = ImageDatasetFactory().build(artifact).train

    assert len(cast(Sized, dataset)) == 1

    incompatible = ImageSourceConfig(
        name="tests.incompatible-image",
        params={"root": str(root)},
        materialization=config.materialization,
    )
    with pytest.raises(TypeError, match="incompatible payload"):
        factory.materialize(
            incompatible,
            binding_id="source",
            builder_context=context,
            path="data.params.source",
        )

    incompatible_artifact = IncompatibleImageDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    with pytest.raises(TypeError, match="public image artifact payload"):
        ImageDatasetFactory().build(cast(Any, incompatible_artifact))


@pytest.mark.parametrize(
    "unsafe_path",
    ["/absolute.png", "../escape.png", "folder/../escape.png"],
)
def test_image_file_record_rejects_unsafe_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError, match=r"relative|unsafe"):
        ImageFileRecord(
            tree="train",
            path=unsafe_path,
            size_bytes=1,
            sha256="a" * 64,
            width=1,
            height=1,
        )


def test_record_collection_rejects_casefold_collision() -> None:
    records = (
        ImageFileRecord("train", "A.png", 1, "a" * 64, 1, 1),
        ImageFileRecord("train", "a.png", 1, "b" * 64, 1, 1),
    )

    with pytest.raises(ValueError, match="duplicate paths"):
        reference_artifacts._assert_unique_records(records)


def test_image_file_record_rejects_non_nfc_path() -> None:
    with pytest.raises(ValueError, match="NFC"):
        ImageFileRecord(
            tree="train",
            path="e\u0301.png",
            size_bytes=1,
            sha256="a" * 64,
            width=1,
            height=1,
        )
