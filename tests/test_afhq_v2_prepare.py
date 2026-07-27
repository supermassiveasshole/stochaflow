from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from stochaflow.data import (
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    DataSourceContext,
    ImageFileRecord,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
    build_data_loaders,
)
from stochaflow.data import artifact_store as _ARTIFACT_STORE
from stochaflow.sampling.runtime import run_sampling
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    parse_rng_state,
    restore_rng_state,
)
from stochaflow.utils.config import ComponentConfig, load_config_dict
from stochaflow.utils.factory import build_training_components
from stochaflow.utils.seed import set_seed

_REPOSITORY = Path(__file__).resolve().parents[1]
_EXAMPLE_ROOT = _REPOSITORY / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _EXAMPLE_ROOT / "src"
_PACKAGE_ROOT = _EXAMPLE_SRC / "stochaflow_afhq_v2"
_PACKAGED_LOCK_PATH = (
    _PACKAGE_ROOT / "resources" / "afhq-v2.lock.yaml"
)


def _load_prepare_module() -> ModuleType:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.invalidate_caches()
    return importlib.import_module("stochaflow_afhq_v2.preparation")


_PREPARE = _load_prepare_module()
_DOWNLOADING = importlib.import_module(
    "stochaflow_afhq_v2._preparation.downloading"
)
_PREPARED_ARTIFACT = importlib.import_module(
    "stochaflow_afhq_v2._preparation.prepared_artifact"
)
_PUBLICATION = importlib.import_module(
    "stochaflow_afhq_v2._preparation.publication"
)
_SAFE_TREE = importlib.import_module(
    "stochaflow_afhq_v2._preparation.safe_tree"
)
_SOURCE_ACQUISITION = importlib.import_module(
    "stochaflow_afhq_v2._preparation.source_acquisition"
)
def test_checked_in_source_lock_is_fully_pinned() -> None:
    raw_lock = yaml.safe_load(_PACKAGED_LOCK_PATH.read_text(encoding="utf-8"))
    lock = _PREPARE.load_source_lock(_PACKAGED_LOCK_PATH)

    assert raw_lock["source"]["type"] == "official_archive"
    assert "kind" not in raw_lock["source"]
    assert lock.expected_bytes == 6_955_288_636
    assert (
        lock.expected_sha256
        == "6f2540f22c6d8ebb8879a2bc0227666dd4fc765cc355cb073b63a835d679e4e3"
    )
    assert lock.contract.total_count == 15_803
    assert lock.contract.source_class_counts == {
        "train": {"cat": 5_065, "dog": 4_678, "wild": 4_593},
        "test": {"cat": 493, "dog": 491, "wild": 483},
    }


def _png_bytes(
    *,
    size: int = 8,
    mode: str = "RGB",
    value: int = 32,
) -> bytes:
    output = BytesIO()
    color: int | tuple[int, int, int] = (
        (value, (value * 3) % 256, (value * 7) % 256)
        if mode == "RGB"
        else value
    )
    image = Image.new(mode, (size, size), color=color)
    marker: int | tuple[int, int, int] = (
        ((value + 97) % 256, (value + 53) % 256, (value + 11) % 256)
        if mode == "RGB"
        else (value + 97) % 256
    )
    image.putpixel((0, 0), marker)
    image.save(output, format="PNG")
    return output.getvalue()


def _write_tiny_archive(path: Path, *, image_size: int = 8) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        index = 0
        for class_name in ("cat", "dog", "wild"):
            for item in range(3):
                archive.writestr(
                    f"afhq_v2/train/{class_name}/{class_name}_{item:03d}.png",
                    _png_bytes(size=image_size, value=32 + index),
                )
                index += 1
            archive.writestr(
                f"afhq_v2/test/{class_name}/{class_name}_test.png",
                _png_bytes(size=image_size, value=32 + index),
            )
            index += 1


def _tiny_contract(*, input_resolution: int = 8) -> Any:
    return _PREPARE.DatasetContract(
        classes=("cat", "dog", "wild"),
        class_mapping={"cat": 0, "dog": 1, "wild": 2},
        train_count=9,
        test_count=3,
        total_count=12,
        input_resolution=input_resolution,
        image_mode="RGB",
        image_format="PNG",
        source_class_counts={
            "train": {"cat": 3, "dog": 3, "wild": 3},
            "test": {"cat": 1, "dog": 1, "wild": 1},
        },
    )


def _tiny_lock(
    archive_path: Path,
    *,
    input_resolution: int = 8,
) -> Any:
    return _PREPARE.SourceLock(
        dataset="afhq-v2",
        url="https://example.invalid/afhq_v2.zip",
        archive_name="afhq_v2.zip",
        expected_bytes=archive_path.stat().st_size,
        expected_sha256=_PREPARE.sha256_file(archive_path),
        license_name="CC BY-NC 4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        homepage="https://example.invalid/afhq",
        citation="Test citation.",
        contract=_tiny_contract(input_resolution=input_resolution),
    )


def _tiny_source(archive_path: Path) -> Any:
    return _PREPARE.SourceArchive(
        path=archive_path,
        sha256=_PREPARE.sha256_file(archive_path),
        size_bytes=archive_path.stat().st_size,
    )


def _write_tiny_source_lock(archive_path: Path, lock_path: Path) -> Path:
    lock = _tiny_lock(archive_path, input_resolution=512)
    lock_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": "afhq-v2",
                "source": {
                    "type": "official_archive",
                    "url": lock.url,
                    "archive_name": lock.archive_name,
                    "bytes": lock.expected_bytes,
                    "sha256": lock.expected_sha256,
                },
                "license": {
                    "name": lock.license_name,
                    "url": lock.license_url,
                },
                "homepage": lock.homepage,
                "citation": lock.citation,
                "dataset_contract": {
                    "classes": ["cat", "dog", "wild"],
                    "class_mapping": {"cat": 0, "dog": 1, "wild": 2},
                    "source_splits": {"train": 9, "test": 3},
                    "source_class_counts": {
                        "train": {"cat": 3, "dog": 3, "wild": 3},
                        "test": {"cat": 1, "dog": 1, "wild": 1},
                    },
                    "total_count": 12,
                    "input_resolution": 512,
                    "image_mode": "RGB",
                    "image_format": "PNG",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return lock_path


def _load_artifact_module() -> ModuleType:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.invalidate_caches()
    return importlib.import_module("stochaflow_afhq_v2.artifact")


def _activate_showcase_extension() -> ModuleType:
    _load_artifact_module()
    return importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")


def test_prepare_tool_uses_registered_data_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = importlib.import_module("stochaflow_afhq_v2.tools.prepare")
    observed: dict[str, Any] = {}
    train_root = tmp_path / "prepared" / "train"
    test_root = tmp_path / "prepared" / "test"
    train_root.mkdir(parents=True)
    test_root.mkdir(parents=True)
    train_record = ImageFileRecord(
        tree="train",
        path="cat/train.png",
        size_bytes=1,
        sha256="0" * 64,
        width=128,
        height=128,
    )
    test_record = ImageFileRecord(
        tree="test",
        path="cat/test.png",
        size_bytes=1,
        sha256="1" * 64,
        width=128,
        height=128,
    )
    prepared_payload = ClassLabeledImageFolderArtifactPayload(
        roots={"train": train_root, "test": test_root},
        class_mapping={"cat": 0},
        train=(
            ClassLabeledImageFileRecord(
                image=train_record,
                class_label=0,
            ),
        ),
        test=(
            ClassLabeledImageFileRecord(
                image=test_record,
                class_label=0,
            ),
        ),
    )

    class PreparedIdentity:
        def to_dict(self) -> dict[str, Any]:
            return {"source_name": "afhq-v2.official"}

    class PreparedArtifact:
        artifact_root = tmp_path / "artifact"
        manifest_path = artifact_root / "manifest.yaml"
        identity = PreparedIdentity()
        payload = prepared_payload

    class PreparedSource:
        def materialize(
            self,
            context: DataSourceContext,
        ) -> PreparedArtifact:
            observed["context"] = context
            return PreparedArtifact()

    def create(
        name: str,
        params: dict[str, Any],
        *,
        config_path: str,
    ) -> PreparedSource:
        observed["name"] = name
        observed["params"] = params
        observed["config_path"] = config_path
        return PreparedSource()

    monkeypatch.setattr(tool.IMAGE_DATA_SOURCES, "create", create)

    summary = tool.prepare_artifact(
        cache_root=tmp_path / "cache",
        archive=tmp_path / "afhq_v2.zip",
        lock_file=tmp_path / "lock.yaml",
        resolution=128,
        downloader="python",
        policy="require",
        verification="full",
    )

    assert observed["params"] == {
        "archive": str(tmp_path / "afhq_v2.zip"),
        "lock_file": str(tmp_path / "lock.yaml"),
        "downloader": "python",
        "resolution": 128,
    }
    assert observed["name"] == "afhq-v2.official"
    assert observed["config_path"] == "prepare"
    context = observed["context"]
    assert isinstance(context, DataSourceContext)
    assert context.policy == "require"
    assert context.verification == "full"
    assert summary["counts"] == {
        "train": 1,
        "test": 1,
    }


def _class_data_config(
    *,
    source_params: dict[str, Any],
    cache_root: Path,
    validation_per_class: int = 1,
    partition_seed: int | str = "fixture-seed",
    image_size: int = 4,
    policy: str = "require",
    num_workers: int = 0,
    shuffle: bool = True,
    random_horizontal_flip: bool = True,
) -> ComponentConfig:
    _activate_showcase_extension()
    return ComponentConfig(
        name="class_labeled_image",
        params={
            "source": {
                "name": "afhq-v2.official",
                "params": {
                    "archive": source_params.get("archive"),
                    "lock_file": source_params.get("lock_file"),
                    "downloader": source_params.get("downloader", "auto"),
                    "resolution": image_size,
                },
                "materialization": {
                    "cache_root": str(cache_root),
                    "policy": policy,
                    "verification": "full",
                },
            },
            "partition": {
                "validation_per_class": validation_per_class,
                "seed": partition_seed,
            },
            "image": {
                "size": [image_size, image_size],
                "channels": 3,
                "normalize": True,
                "random_horizontal_flip": random_horizontal_flip,
            },
            "loader": {
                "batch_size": 2,
                "num_workers": num_workers,
                "shuffle": shuffle,
                "drop_last": False,
                "pin_memory": False,
                "persistent_workers": num_workers > 0,
                "prefetch_factor": 2 if num_workers > 0 else None,
                "steps_per_epoch": "auto",
            },
        },
    )


def test_prepare_archive_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)

    first = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    assert first.cache_hit is False
    assert first.file_count == 12
    assert len(first.image_records) == 12
    manifest_bytes = first.manifest_path.read_bytes()
    inventory_bytes = (first.root / "files.sha256").read_bytes()

    manifest = yaml.safe_load(manifest_bytes)
    assert manifest["counts"] == {
        "test": {
            "classes": {"cat": 1, "dog": 1, "wild": 1},
            "total": 3,
        },
        "train": {
            "classes": {"cat": 3, "dog": 3, "wild": 3},
            "total": 9,
        },
    }
    assert manifest["source"]["archive"]["sha256"] == source.sha256
    assert manifest["inventory"]["sha256"] == hashlib.sha256(
        inventory_bytes
    ).hexdigest()
    assert "proxy" not in first.manifest_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in first.manifest_path.read_text(encoding="utf-8")

    prepared_images = sorted(
        path
        for split in ("train", "test")
        for path in (first.root / split).rglob("*.png")
    )
    assert len(prepared_images) == 12
    for path in prepared_images:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (4, 4)
    for record in first.image_records:
        path = first.root / record.relative_path
        assert path.stat().st_size == record.size_bytes
        assert _PREPARE.sha256_file(path) == record.sha256

    second = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    assert second.cache_hit is True
    assert second.root == first.root
    assert second.manifest_path.read_bytes() == manifest_bytes
    assert (second.root / "files.sha256").read_bytes() == inventory_bytes
    assert second.manifest_sha256 == first.manifest_sha256

    independent = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "independent-cache",
        resolution=4,
    )
    assert independent.cache_hit is False
    assert independent.artifact_digest == first.artifact_digest
    assert independent.manifest_path.read_bytes() == manifest_bytes
    assert (independent.root / "files.sha256").read_bytes() == inventory_bytes


def test_afhq_regular_file_open_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    make_fifo = getattr(os, "mkfifo", None)
    if not callable(make_fifo):
        pytest.skip("FIFO creation is unavailable")
    fifo = tmp_path / "archive.pipe"
    try:
        make_fifo(fifo)
    except OSError:
        pytest.skip("FIFO creation is unavailable")

    with pytest.raises(_PREPARE.PreparationError, match="regular file"):
        _PREPARE.sha256_file(fifo)


def test_afhq_registered_source_feeds_the_builtin_builder(
    tmp_path: Path,
) -> None:
    artifact_service = _load_artifact_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    source_params = {
        "archive": str(archive_path),
        "lock_file": str(lock_path),
        "resolution": 4,
    }
    artifact = artifact_service.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=4,
    )

    assert isinstance(artifact, ManagedDataArtifact)
    assert not isinstance(artifact, Dataset)
    assert artifact.identity.source_name == "afhq-v2.official"
    assert set(artifact.payload.roots) == {"train", "test"}
    assert len(artifact.payload.train) == 9
    assert artifact.payload.validation is None
    assert len(artifact.payload.test or ()) == 3
    assert all(
        entry.image.tree == "train"
        and not entry.image.path.startswith("train/")
        for entry in artifact.payload.train
    )
    assert dict(artifact.payload.class_mapping) == {
        "cat": 0,
        "dog": 1,
        "wild": 2,
    }
    with pytest.raises(ValueError, match="expected a different data source"):
        artifact_service.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=replace(
                    artifact.identity,
                    source_name="another-source",
                ),
            ),
            lock_file=lock_path,
            resolution=4,
        )
    with pytest.raises(ValueError, match="identity does not match"):
        artifact_service.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=replace(
                    artifact.identity,
                    artifact_digest="0" * 64,
                ),
            ),
            lock_file=lock_path,
            resolution=4,
        )

    cached_archive = (
        cache_root
        / "raw"
        / "afhq-v2"
        / artifact.identity.source_digest
        / "afhq_v2.zip"
    )
    assert cached_archive.is_file()
    cached_archive.unlink()
    archive_path.unlink()

    required = artifact_service.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="require",
            verification="full",
        ),
        lock_file=lock_path,
        resolution=4,
    )
    assert isinstance(required, ManagedDataArtifact)
    assert required.identity == artifact.identity
    assert required.artifact_root == artifact.artifact_root

    class_loaders = build_data_loaders(
        _class_data_config(
            source_params=source_params,
            cache_root=cache_root,
            shuffle=False,
            random_horizontal_flip=False,
        ),
        seed=7,
    )
    class_images, class_conditions = next(iter(class_loaders.train))
    assert class_images.shape == (2, 3, 4, 4)
    assert class_conditions["class_label"].dtype == torch.long
    assert class_conditions["class_label"].tolist() == [0, 0]
    assert class_loaders.artifact_bindings is not None
    assert (
        class_loaders.artifact_bindings.identity_for("source")
        == artifact.identity
    )
    alternate_loaders = build_data_loaders(
        _class_data_config(
            source_params=source_params,
            cache_root=cache_root,
            validation_per_class=2,
            partition_seed="alternate-split",
            shuffle=False,
            random_horizontal_flip=False,
        ),
        seed=7,
    )
    assert alternate_loaders.artifact_bindings == (
        class_loaders.artifact_bindings
    )
    assert class_loaders.validation is not None
    assert alternate_loaders.validation is not None
    validation_loader = cast(DataLoader[Any], class_loaders.validation)
    alternate_validation_loader = cast(
        DataLoader[Any],
        alternate_loaders.validation,
    )
    assert len(cast(Sized, validation_loader.dataset)) == 3
    assert len(cast(Sized, alternate_validation_loader.dataset)) == 6

    train_loader = cast(DataLoader[Any], class_loaders.train)
    dataset_records = cast(Any, train_loader.dataset).records
    changed_record = dataset_records[0]
    image_path = (
        artifact.payload.roots[changed_record.image.tree]
        / changed_record.image.path
    )
    image_path.write_bytes(b"\0" * image_path.stat().st_size)
    manifest_only = artifact_service.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="require",
            verification="manifest",
        ),
        lock_file=lock_path,
        resolution=4,
    )
    assert manifest_only.identity == artifact.identity
    with pytest.raises(ValueError, match="artifact image content changed"):
        next(iter(class_loaders.train))
    with pytest.raises(
        _PREPARE.PreparationError,
        match="prepared file digest mismatch",
    ):
        artifact_service.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="full",
            ),
            lock_file=lock_path,
            resolution=4,
        )


def test_builtin_class_builder_derives_resolution_and_enforces_strict_identity(
    tmp_path: Path,
) -> None:
    artifact_service = _load_artifact_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    source_params = {
        "archive": str(archive_path),
        "lock_file": str(lock_path),
        "resolution": 4,
    }
    artifact_service.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=4,
    )
    assert dict(artifact_service.AFHQV2_CLASS_MAPPING) == {
        "cat": 0,
        "dog": 1,
        "wild": 2,
    }
    immutable_mapping: Any = artifact_service.AFHQV2_CLASS_MAPPING
    with pytest.raises(TypeError):
        immutable_mapping["cat"] = 2

    config = _class_data_config(
        source_params=source_params,
        cache_root=cache_root,
        random_horizontal_flip=False,
    )
    loaders = build_data_loaders(config, seed=11)
    assert loaders.artifact_bindings is not None
    resumed = build_data_loaders(
        config,
        seed=11,
        strict_resume=True,
        expected_artifacts=loaders.artifact_bindings,
    )
    assert resumed.artifact_bindings == loaders.artifact_bindings
    with pytest.raises(
        ValueError,
        match="missing data artifact identities",
    ):
        build_data_loaders(
            config,
            seed=11,
            strict_resume=True,
            expected_artifacts=None,
        )

    resized = _class_data_config(
        source_params=source_params,
        cache_root=cache_root,
        image_size=8,
        policy="ensure",
    )
    resized_loaders = build_data_loaders(resized, seed=11)
    resized_images, _ = next(iter(resized_loaders.train))
    assert resized_images.shape[1:] == (3, 8, 8)
    assert resized_loaders.artifact_bindings != loaders.artifact_bindings


def _collect_epoch(
    loader: Any,
    *,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampler = loader.sampler
    sampler.set_epoch(epoch)
    batches = list(loader)
    return (
        torch.cat([batch[0] for batch in batches]),
        torch.cat([batch[1]["class_label"] for batch in batches]),
    )


def test_afhq_class_builder_is_worker_and_resume_deterministic(
    tmp_path: Path,
) -> None:
    artifact_service = _load_artifact_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    source_params = {
        "archive": str(archive_path),
        "lock_file": str(lock_path),
        "resolution": 4,
    }
    artifact_service.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=4,
    )
    single_worker = build_data_loaders(
        _class_data_config(
            source_params=source_params,
            cache_root=cache_root,
            num_workers=0,
        ),
        seed=29,
    )
    persistent_workers = build_data_loaders(
        _class_data_config(
            source_params=source_params,
            cache_root=cache_root,
            num_workers=1,
        ),
        seed=29,
    )
    resumed_loader = build_data_loaders(
        _class_data_config(
            source_params=source_params,
            cache_root=cache_root,
            num_workers=0,
        ),
        seed=29,
        strict_resume=True,
        expected_artifacts=single_worker.artifact_bindings,
    )

    single_images, single_labels = _collect_epoch(
        single_worker.train,
        epoch=3,
    )
    worker_images, worker_labels = _collect_epoch(
        persistent_workers.train,
        epoch=3,
    )
    resumed_images, resumed_labels = _collect_epoch(
        resumed_loader.train,
        epoch=3,
    )
    assert torch.equal(worker_labels, single_labels)
    assert torch.equal(resumed_labels, single_labels)
    assert torch.equal(worker_images, single_images)
    assert torch.equal(resumed_images, single_images)
    assert set(single_labels.tolist()) == {0, 1, 2}

    next_images, next_labels = _collect_epoch(single_worker.train, epoch=4)
    worker_next_images, worker_next_labels = _collect_epoch(
        persistent_workers.train,
        epoch=4,
    )
    assert torch.equal(worker_next_labels, next_labels)
    assert torch.equal(worker_next_images, next_images)
    assert not (
        torch.equal(next_labels, single_labels)
        and torch.equal(next_images, single_images)
    )
    del persistent_workers

    assert single_worker.validation is not None
    validation_images, validation_conditions = next(iter(single_worker.validation))
    repeated_images, repeated_conditions = next(iter(single_worker.validation))
    assert torch.equal(validation_images, repeated_images)
    assert torch.equal(
        validation_conditions["class_label"],
        repeated_conditions["class_label"],
    )
    assert set(validation_conditions) == {"class_label"}


def test_afhq_real_data_pipeline_trains_resumes_validates_and_samples(
    tmp_path: Path,
) -> None:
    artifact_service = _load_artifact_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    source_params = {
        "archive": str(archive_path),
        "lock_file": str(lock_path),
        "resolution": 8,
    }
    artifact_service.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=8,
    )
    data_config = _class_data_config(
        source_params=source_params,
        cache_root=cache_root,
        partition_seed="pipeline-fixture-seed",
        image_size=8,
    )
    raw_config = {
        "experiment": {
            "name": "afhq-v2-real-data-pipeline",
            "seed": 431,
            "output_dir": str(tmp_path / "uninterrupted"),
        },
        "extensions": {"plugins": []},
        "data": {
            "name": data_config.name,
            "params": data_config.params,
        },
        "model": {
            "name": "adm_unet",
            "params": {
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": 8,
                "channel_multipliers": [1, 2],
                "num_res_blocks": 1,
                "transformer_depths": [0, 0],
                "middle_transformer_depth": 0,
                "attention_head_dim": 8,
                "time_embedding_dim": 32,
                "num_classes": 3,
                "dropout": 0.0,
            },
        },
        "process": {
            "name": "discrete_gaussian",
            "params": {
                "schedule": {
                    "name": "linear_beta",
                    "params": {
                        "num_timesteps": 4,
                        "beta_start": 0.0001,
                        "beta_end": 0.02,
                    },
                }
            },
        },
        "training": {
            "name": "class_conditional_gaussian_denoising",
            "params": {
                "prediction_type": "v",
                "condition_dropout": 0.25,
            },
        },
        "objective": {
            "name": "mse",
            "params": {"reduction": "mean"},
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "params": {"lr": 0.001, "weight_decay": 0.0},
        },
        "lr_scheduler": {
            "name": "warmup_cosine",
            "interval": "step",
            "params": {
                "warmup_steps": 1,
                "total_steps": 4,
                "min_lr_ratio": 0.1,
            },
        },
        "ema": {
            "enabled": True,
            "decay": 0.9,
            "update_after_step": 0,
            "update_every": 1,
            "use_for_sampling": True,
        },
        "sampling": {
            "shape": [3, 8, 8],
            "num_samples": 3,
            "batch_size": 3,
            "seed": 431,
            "builder": {
                "name": "class_conditional_denoising",
                "params": {
                    "weights": "auto",
                    "prediction_type": "v",
                    "clip_denoised": True,
                    "guidance_scale": 2.0,
                    "conditions": [
                        {"class_label": 0, "count": 1},
                        {"class_label": 1, "count": 1},
                        {"class_label": 2, "count": 1},
                    ],
                    "sampler": {
                        "name": "ddim",
                        "params": {
                            "num_inference_steps": 2,
                            "eta": 0.0,
                        },
                    },
                    "trajectory": {
                        "enabled": False,
                        "every_steps": 1,
                    },
                },
            },
            "writers": [{"name": "tensor", "params": {}}],
        },
        "trainer": {
            "num_epochs": 2,
            "device": "cpu",
            "precision": "fp32",
            "accumulate_grad_batches": 2,
            "show_progress": False,
        },
        "logging": {
            "log_every": 1,
            "backends": [{"name": "local", "params": {"console": False}}],
            "torch_logs": {},
        },
        "artifacts": {"checkpoint_every": 1},
    }
    config = load_config_dict(raw_config)
    loaders = build_data_loaders(
        config.data,
        seed=config.experiment.seed,
    )
    assert loaders.artifact_bindings is not None

    set_seed(config.experiment.seed)
    uninterrupted = build_training_components(config)
    resumed = None
    try:
        uninterrupted.trainer.train_epoch(
            loaders.train,
            epoch_index=1,
            show_progress=False,
        )
        assert loaders.validation is not None
        validation = uninterrupted.trainer.evaluate_epoch(
            loaders.validation,
            epoch_index=1,
            show_progress=False,
        )
        assert validation["num_batches"] > 0
        assert loaders.test is not None
        test_metrics = uninterrupted.trainer.evaluate_epoch(
            loaders.test,
            epoch_index=1,
            show_progress=False,
            metric_prefix="test",
        )
        assert test_metrics["num_batches"] > 0
        checkpoint = uninterrupted.checkpoint_manager.save(
            tmp_path / "epoch-1.pt",
            epoch=1,
            global_step=uninterrupted.trainer.global_step,
            config=config.to_dict(),
            metadata={
                "data_artifacts": loaders.artifact_bindings.to_dict(),
            },
        )
        uninterrupted.trainer.train_epoch(
            loaders.train,
            epoch_index=2,
            show_progress=False,
        )
        expected = uninterrupted.checkpoint_manager.build_state()

        resumed_raw = dict(raw_config)
        resumed_raw["experiment"] = {
            **raw_config["experiment"],
            "output_dir": str(tmp_path / "resumed"),
        }
        resumed_config = load_config_dict(resumed_raw)
        resumed_loaders = build_data_loaders(
            resumed_config.data,
            seed=resumed_config.experiment.seed,
            strict_resume=True,
            expected_artifacts=loaders.artifact_bindings,
        )
        resumed = build_training_components(resumed_config)
        payload = CheckpointManager.load_payload(
            checkpoint,
            map_location="cpu",
        )
        loaded = resumed.checkpoint_manager.restore_payload(
            payload,
            path=checkpoint,
        )
        assert loaded.global_step is not None
        resumed.trainer.global_step = loaded.global_step
        rng_state = payload.get("rng_state")
        assert rng_state is not None
        restore_rng_state(
            parse_rng_state(rng_state),
            restore_cuda=False,
            restore_mps=False,
        )
        resumed.trainer.train_epoch(
            resumed_loaders.train,
            epoch_index=2,
            show_progress=False,
        )
        actual = resumed.checkpoint_manager.build_state()
        for key in (
            "model_state_dict",
            "optimizer_state_dict",
            "lr_scheduler_state_dict",
            "ema_state_dict",
        ):
            torch.testing.assert_close(
                actual.get(key),
                expected.get(key),
                rtol=0.0,
                atol=0.0,
            )

        assert resumed_loaders.artifact_bindings is not None
        final_checkpoint = resumed.checkpoint_manager.save(
            tmp_path / "epoch-2.pt",
            epoch=2,
            global_step=resumed.trainer.global_step,
            config=resumed_config.to_dict(),
            metadata={
                "data_artifacts": resumed_loaders.artifact_bindings.to_dict(),
            },
        )
        result = run_sampling(
            checkpoint=final_checkpoint,
            output_dir=tmp_path / "samples",
            device_name="cpu",
        )
        samples = torch.load(
            result.artifacts["samples"],
            map_location="cpu",
            weights_only=True,
        )
        assert samples.shape == (3, 3, 8, 8)
        assert result.metadata["conditions"] == [
            {"class_label": 0, "count": 1},
            {"class_label": 1, "count": 1},
            {"class_label": 2, "count": 1},
        ]
        assert result.metadata["conditional_branch_evaluation_count"] > 0
        assert (
            result.metadata["unconditional_branch_evaluation_count"]
            == result.metadata["conditional_branch_evaluation_count"]
        )
        assert result.artifacts["config"].is_file()
    finally:
        uninterrupted.logger.close()
        if resumed is not None:
            resumed.logger.close()


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    [
        ("source_digest", "0" * 64, "different source digest"),
        ("materializer_name", "another-materializer", "different materializer"),
        (
            "materialization_digest",
            "0" * 64,
            "different preparation recipe",
        ),
    ],
)
def test_afhq_strict_resume_preflights_identity_before_cache_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
    changed_value: str,
    message: str,
) -> None:
    artifact_service = _load_artifact_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    lock = _PREPARE.load_source_lock(lock_path)
    plan = _PREPARE.build_preparation_plan(
        lock=lock,
        resolution=4,
    )
    assert lock.expected_sha256 is not None
    expected = ManagedDataArtifactIdentity(
        artifact_type="stochaflow.class-labeled-image-folder.v1",
        source_name="afhq-v2.official",
        source_digest=lock.expected_sha256,
        materializer_name=str(plan.recipe["id"]),
        materialization_digest=plan.recipe_sha256,
        artifact_digest="1" * 64,
        manifest_sha256="2" * 64,
    )
    expected = replace(expected, **{changed_field: changed_value})
    calls: list[str] = []

    def unexpected_cache_io(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        calls.append("cache")
        pytest.fail("strict identity mismatch reached cache I/O")

    monkeypatch.setattr(
        artifact_service.preparation,
        "require_prepared_artifact",
        unexpected_cache_io,
    )
    monkeypatch.setattr(
        artifact_service.preparation,
        "acquire_official_archive",
        unexpected_cache_io,
    )
    cache_root = tmp_path / "cache"

    with pytest.raises(ValueError, match=message):
        artifact_service.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="ensure",
                verification="full",
                expected_identity=expected,
            ),
            archive=archive_path,
            lock_file=lock_path,
            resolution=4,
        )

    assert calls == []
    assert not cache_root.exists()


def test_builtin_builder_rejects_unknown_and_source_private_fields(
    tmp_path: Path,
) -> None:
    base = _class_data_config(
        source_params={
            "archive": str(tmp_path / "archive.zip"),
            "lock_file": str(tmp_path / "lock.yaml"),
        },
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(ValueError, match="unknown config field"):
        build_data_loaders(
            ComponentConfig(
                name=base.name,
                params={**base.params, "unexpected": True},
            ),
            seed=1,
        )
    source = base.params["source"]
    assert isinstance(source, dict)
    source_params = source["params"]
    assert isinstance(source_params, dict)
    with pytest.raises(ValueError, match="downloader must be"):
        build_data_loaders(
            ComponentConfig(
                name=base.name,
                params={
                    **base.params,
                    "source": {
                        **source,
                        "params": {
                            **source_params,
                            "downloader": "mirror",
                        },
                    },
                },
            ),
            seed=1,
        )
    for value in ([], {}):
        with pytest.raises(ValueError, match="downloader must be"):
            build_data_loaders(
                ComponentConfig(
                    name=base.name,
                    params={
                        **base.params,
                        "source": {
                            **source,
                            "params": {
                                **source_params,
                                "downloader": value,
                            },
                        },
                    },
                ),
                seed=1,
            )
    for field, value in (
        ("validation_per_class", 1),
        ("validation_seed", "source-must-not-own-splits"),
    ):
        with pytest.raises(
            ValueError,
            match=rf"unknown config field.*source\.params\.{field}",
        ):
            build_data_loaders(
                ComponentConfig(
                    name=base.name,
                    params={
                        **base.params,
                        "source": {
                            **source,
                            "params": {
                                **source_params,
                                field: value,
                            },
                        },
                    },
                ),
                seed=1,
            )


def test_cache_hit_rejects_modified_prepared_image(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    image_path = next((artifact.root / "train").rglob("*.png"))
    image_path.write_bytes(b"\0" * image_path.stat().st_size)

    with pytest.raises(
        _PREPARE.PreparationError,
        match="prepared file digest mismatch",
    ):
        _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
        )


def test_prepare_rejects_archive_path_substitution_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    original_process = _PUBLICATION._process_images
    substituted = False

    def substitute_archive(**kwargs: Any) -> Any:
        nonlocal substituted
        backup = archive_path.with_name("opened-afhq_v2.zip")
        try:
            archive_path.replace(backup)
            shutil.copyfile(backup, archive_path)
        except PermissionError as error:
            pytest.skip(f"platform prevents open-file substitution: {error}")
        substituted = True
        return original_process(**kwargs)

    monkeypatch.setattr(_PUBLICATION, "_process_images", substitute_archive)
    cache_root = tmp_path / "cache"

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"path no longer names the opened file",
    ):
        _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=cache_root,
            resolution=4,
        )

    assert substituted
    prepared_base = cache_root / "prepared" / "afhq-v2" / "4"
    assert not list(prepared_base.glob(".*.tmp-*"))
    assert not any(
        path.is_dir() and not path.name.startswith(".")
        for path in prepared_base.iterdir()
    )


def test_cache_hit_rejects_modified_manifest_recipe(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    manifest = yaml.safe_load(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["preparation"]["recipe"]["encoding"]["compress_level"] = 1
    artifact.manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(_PREPARE.PreparationError, match="wrong recipe digest"):
        _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
        )


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_prepared_manifest_requires_exact_integer_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    manifest = yaml.safe_load(artifact.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    artifact.manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match="schema_version must be 1",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
        )


def test_cache_hit_rejects_missing_inventory(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    (artifact.root / "files.sha256").unlink()

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"invalid layout.*files\.sha256",
    ):
        _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
        )


def _replace_path_with_symlink(
    path: Path,
    target: Path,
    *,
    target_is_directory: bool,
) -> None:
    if path.is_dir():
        shutil.move(str(path), str(target))
    else:
        path.replace(target)
    try:
        path.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        if target_is_directory and os.name == "nt":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(path), str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
        if target_is_directory:
            shutil.move(str(target), str(path))
        else:
            target.replace(path)
        pytest.skip(f"symbolic links are unavailable: {error}")


def _create_directory_link(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name == "nt":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(path), str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
        pytest.skip(f"directory links are unavailable: {error}")


@pytest.mark.parametrize("metadata_name", ["dataset_manifest.yaml", "files.sha256"])
def test_prepared_metadata_verification_does_not_follow_symlinks(
    tmp_path: Path,
    metadata_name: str,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    metadata_path = artifact.root / metadata_name
    outside = tmp_path / f"outside-{metadata_name}"
    _replace_path_with_symlink(
        metadata_path,
        outside,
        target_is_directory=False,
    )
    plan = _PREPARE.build_preparation_plan(
        lock=lock,
        resolution=4,
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"symlink|reparse|missing",
    ):
        _PREPARE.verify_prepared_artifact(
            artifact.root,
            expected_preparation_key=artifact.preparation_key,
            expected_recipe=plan.recipe,
            source_archive=source,
            source_lock=lock,
            expected_counts=plan.counts,
        )


def test_full_verification_rejects_symlinked_split_ancestor(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    class_path = artifact.root / "train" / "cat"
    outside = tmp_path / "outside-cat"
    _replace_path_with_symlink(
        class_path,
        outside,
        target_is_directory=True,
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"symlink|reparse|unsafe",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            full=True,
        )


def test_prepared_root_must_not_be_a_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    outside = tmp_path / "outside-artifact"
    _replace_path_with_symlink(
        artifact.root,
        outside,
        target_is_directory=True,
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"symlink|reparse",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            full=True,
        )


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_full_verification_rejects_unexpected_root_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    unexpected = artifact.root / "unexpected"
    if entry_kind == "file":
        unexpected.write_bytes(b"unexpected")
    else:
        unexpected.mkdir()

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"invalid layout.*unexpected",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            full=True,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_full_verification_rejects_windows_junction_ancestor(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    class_path = artifact.root / "train" / "cat"
    outside = tmp_path / "outside-cat-junction"
    shutil.move(str(class_path), str(outside))
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(class_path), str(outside)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.move(str(outside), str(class_path))
        pytest.skip(f"cannot create a Windows junction: {result.stderr}")

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"symlink|reparse",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            full=True,
        )


def test_directory_enumeration_detects_split_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    split_path = artifact.root / "train"
    outside = tmp_path / "outside-train"
    backup = artifact.root / "original-train"
    shutil.copytree(split_path, outside)
    original_scandir = os.scandir
    substituted = False

    def substitute_before_scan(path: Any) -> Any:
        nonlocal substituted
        if not substituted and (
            isinstance(path, int) or Path(path) == split_path
        ):
            substituted = True
            split_path.replace(backup)
            _create_directory_link(split_path, outside)
        return original_scandir(path)

    monkeypatch.setattr(_SAFE_TREE.os, "scandir", substitute_before_scan)

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"invalid layout.*unexpected=.*original-train",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            full=True,
        )
    assert substituted


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX descriptor-anchoring regression",
)
def test_verification_rejects_intermediate_root_link_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    container = tmp_path / "container"
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=container / "cache",
        resolution=4,
    )
    plan = _PREPARE.build_preparation_plan(
        lock=lock,
        resolution=4,
    )
    preserved = tmp_path / "preserved-container"
    original = _PREPARED_ARTIFACT.canonical_directory
    substituted = False

    def substitute_after_check(path: Path, *, label: str) -> Path:
        nonlocal substituted
        result = original(path, label=label)
        if not substituted and label == "prepared artifact root":
            substituted = True
            container.replace(preserved)
            container.symlink_to(preserved, target_is_directory=True)
        return result

    monkeypatch.setattr(
        _PREPARED_ARTIFACT,
        "canonical_directory",
        substitute_after_check,
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"unsafe|symlink|linked|cannot inspect",
    ):
        _PREPARE.verify_prepared_artifact(
            artifact.root,
            expected_preparation_key=artifact.preparation_key,
            expected_recipe=plan.recipe,
            source_archive=source,
            source_lock=lock,
            expected_counts=plan.counts,
            full=True,
        )
    assert substituted


def test_quarantine_rejects_linked_cache_ancestor(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    cache_root = tmp_path / "cache"
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=cache_root,
        resolution=4,
    )
    prepared_base = artifact.root.parent
    outside = tmp_path / "outside-prepared-base"
    _replace_path_with_symlink(
        prepared_base,
        outside,
        target_is_directory=True,
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"quarantine|symlink|reparse|unsafe",
    ):
        _PUBLICATION._quarantine_invalid_prepared_artifact(
            artifact.root,
            cache_root=cache_root,
        )

    assert (outside / artifact.root.name).is_dir()
    assert not list(outside.glob(f"{artifact.root.name}.*.invalid"))


def test_prepare_rejects_linked_prepared_cache_before_writing_outside(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside-prepared"
    outside.mkdir()
    _create_directory_link(cache_root / "prepared", outside)

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"symlink|reparse|artifact directory",
    ):
        _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=cache_root,
            resolution=4,
        )

    assert list(outside.iterdir()) == []


def test_ensure_repairs_invalid_prepared_artifact_under_lock(
    tmp_path: Path,
) -> None:
    artifact_service = _load_artifact_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    context = DataSourceContext(
        cache_root=cache_root,
        policy="ensure",
        verification="full",
    )
    def materialize() -> Any:
        return artifact_service.materialize_afhq_v2_artifact(
            context,
            archive=archive_path,
            lock_file=lock_path,
            resolution=4,
        )

    first = materialize()
    record = first.payload.train[0]
    image_path = (
        first.payload.roots[record.image.tree] / record.image.path
    )
    original = image_path.read_bytes()
    image_path.write_bytes(b"\0" * len(original))

    repaired = materialize()

    assert repaired.identity == first.identity
    repaired_path = (
        repaired.payload.roots[record.image.tree] / record.image.path
    )
    assert repaired_path.read_bytes() == original
    quarantined = list(
        first.artifact_root.parent.glob(
            f"{first.artifact_root.name}.*.invalid"
        )
    )
    assert len(quarantined) == 1
    quarantined_image = (
        quarantined[0] / record.image.tree / record.image.path
    )
    assert quarantined_image.read_bytes() == b"\0" * len(original)


def test_require_failure_never_quarantines_or_repairs(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    artifact = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
    )
    image_path = next((artifact.root / "train").rglob("*.png"))
    corrupt = b"\0" * image_path.stat().st_size
    image_path.write_bytes(corrupt)

    with pytest.raises(
        _PREPARE.PreparationError,
        match="prepared file digest mismatch",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            full=True,
        )

    assert image_path.read_bytes() == corrupt
    assert not list(artifact.root.parent.glob(f"{artifact.root.name}.*.invalid"))


def test_concurrent_preparation_waits_and_converges(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    cache_root = tmp_path / "cache"

    def prepare() -> Any:
        return _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=cache_root,
            resolution=4,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(prepare), executor.submit(prepare))
        results = tuple(future.result() for future in futures)

    assert {result.cache_hit for result in results} == {False, True}
    assert results[0].root == results[1].root
    assert results[0].artifact_digest == results[1].artifact_digest
    locks = list((cache_root / ".locks").glob("*.lock"))
    assert len(locks) == 1
    assert locks[0].is_file()


def test_preparation_lock_times_out_without_deleting_owner(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "prepare.lock"

    with _PREPARE.ArtifactPreparationLock(lock_path):
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        assert set(metadata) == {
            "created_at_ns",
            "hostname",
            "nonce",
            "pid",
            "schema_version",
        }
        assert metadata["schema_version"] == 1
        with pytest.raises(
            _PREPARE.PreparationError,
            match="timed out waiting",
        ), _PREPARE.ArtifactPreparationLock(
            lock_path,
            timeout_seconds=0.01,
            poll_seconds=0.001,
        ):
            pytest.fail("a second owner acquired the active lock")
        assert lock_path.is_file()
        assert json.loads(lock_path.read_text(encoding="utf-8")) == metadata

    assert lock_path.is_file()
    with _PREPARE.ArtifactPreparationLock(lock_path):
        assert lock_path.is_file()


@pytest.mark.parametrize(
    "error",
    [
        OSError("cannot open fixture lock"),
        ValueError("unsafe fixture lock"),
        RuntimeError("fixture lock timeout"),
    ],
)
def test_preparation_lock_translates_framework_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def fail_to_enter(
        lock: _ARTIFACT_STORE.ArtifactMaterializationLock,
    ) -> None:
        del lock
        raise error

    monkeypatch.setattr(
        _ARTIFACT_STORE.ArtifactMaterializationLock,
        "__enter__",
        fail_to_enter,
    )

    with (
        pytest.raises(_PREPARE.PreparationError, match=str(error)),
        _PREPARE.ArtifactPreparationLock(tmp_path / "prepare.lock"),
    ):
        pytest.fail("framework lock failure was not translated")


def test_preparation_lock_metadata_handles_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = _ARTIFACT_STORE.os.write

    def partial_write(descriptor: int, payload: Any) -> int:
        size = max(1, len(payload) // 2)
        return original_write(descriptor, payload[:size])

    monkeypatch.setattr(_ARTIFACT_STORE.os, "write", partial_write)
    lock_path = tmp_path / "partial-write.lock"

    with _PREPARE.ArtifactPreparationLock(lock_path):
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))

    assert metadata["pid"] == os.getpid()
    assert set(metadata) == {
        "created_at_ns",
        "hostname",
        "nonce",
        "pid",
        "schema_version",
    }
    assert metadata["schema_version"] == 1


@pytest.mark.parametrize(
    "member_name",
    [
        "../evil.png",
        "/afhq_v2/train/cat/evil.png",
        "C:/afhq_v2/train/cat/evil.png",
        r"afhq_v2\train\cat\evil.png",
        "afhq_v2/train//cat/evil.png",
        "afhq_v2/train/cat/CON.png",
        "afhq_v2/train/cat/CON .png",
        "afhq_v2/train/cat/evil .png",
        "afhq_v2/train/cat/evil:.png",
        "afhq_v2/train/cat/evil<.png",
        "afhq_v2/train/cat/evil>.png",
        'afhq_v2/train/cat/evil".png',
        "afhq_v2/train/cat/evil|.png",
        "afhq_v2/train/cat/evil?.png",
        "afhq_v2/train/cat/evil*.png",
        "afhq_v2/train/cat/control\u0001.png",
    ],
)
def test_inspect_archive_rejects_unsafe_member(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(member_name, _png_bytes())

    with pytest.raises(_PREPARE.PreparationError):
        _PREPARE.inspect_archive(
            archive_path,
            contract=_PREPARE.DatasetContract(
                classes=("cat", "dog", "wild"),
                class_mapping={"cat": 0, "dog": 1, "wild": 2},
                train_count=1,
                test_count=1,
                total_count=2,
                input_resolution=8,
                image_mode="RGB",
                image_format="PNG",
            ),
        )


def test_inspect_archive_rejects_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    info = ZipInfo("afhq_v2/train/cat/link.png")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(info, b"target")

    with pytest.raises(_PREPARE.PreparationError, match="symbolic link"):
        _PREPARE.inspect_archive(
            archive_path,
            contract=_PREPARE.DatasetContract(
                classes=("cat", "dog", "wild"),
                class_mapping={"cat": 0, "dog": 1, "wild": 2},
                train_count=1,
                test_count=1,
                total_count=2,
                input_resolution=8,
                image_mode="RGB",
                image_format="PNG",
            ),
        )


def test_inspect_archive_rejects_case_collision(tmp_path: Path) -> None:
    archive_path = tmp_path / "collision.zip"
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "afhq_v2/train/cat/CAT.png",
            _png_bytes(),
        )
        archive.writestr(
            "afhq_v2/train/cat/cat.png",
            _png_bytes(value=64),
        )

    with pytest.raises(_PREPARE.PreparationError, match="case-insensitive"):
        _PREPARE.inspect_archive(
            archive_path,
            contract=_PREPARE.DatasetContract(
                classes=("cat", "dog", "wild"),
                class_mapping={"cat": 0, "dog": 1, "wild": 2},
                train_count=1,
                test_count=1,
                total_count=2,
                input_resolution=8,
                image_mode="RGB",
                image_format="PNG",
            ),
        )


def test_prepare_rejects_non_rgb_source(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad-mode.zip"
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "afhq_v2/train/cat/cat_000.png",
            _png_bytes(mode="L"),
        )
        archive.writestr(
            "afhq_v2/train/cat/cat_001.png",
            _png_bytes(value=64),
        )
        archive.writestr(
            "afhq_v2/test/cat/cat_test.png",
            _png_bytes(value=96),
        )
    contract = _PREPARE.DatasetContract(
        classes=("cat",),
        class_mapping={"cat": 0},
        train_count=2,
        test_count=1,
        total_count=3,
        input_resolution=8,
        image_mode="RGB",
        image_format="PNG",
        source_class_counts={"train": {"cat": 2}, "test": {"cat": 1}},
    )
    lock = _PREPARE.SourceLock(
        dataset="afhq-v2",
        url="https://example.invalid/afhq_v2.zip",
        archive_name="afhq_v2.zip",
        expected_bytes=archive_path.stat().st_size,
        expected_sha256=_PREPARE.sha256_file(archive_path),
        license_name="test",
        license_url="https://example.invalid/license",
        homepage="https://example.invalid",
        citation="test",
        contract=contract,
    )

    with pytest.raises(_PREPARE.PreparationError, match="expected RGB"):
        _PREPARE.prepare_archive(
            source=_tiny_source(archive_path),
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
        )


def test_load_source_lock_rejects_unknown_fields(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": "afhq-v2",
                "source": {
                    "type": "official_archive",
                    "url": "https://example.invalid/archive.zip",
                    "archive_name": "archive.zip",
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
                "license": {
                    "name": "test",
                    "url": "https://example.invalid/license",
                },
                "homepage": "https://example.invalid",
                "citation": "test",
                "dataset_contract": {
                    "classes": ["cat", "dog", "wild"],
                    "class_mapping": {"cat": 0, "dog": 1, "wild": 2},
                    "source_splits": {"train": 2, "test": 1},
                    "total_count": 3,
                    "input_resolution": 8,
                    "image_mode": "RGB",
                    "image_format": "PNG",
                },
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(_PREPARE.PreparationError, match="unknown fields"):
        _PREPARE.load_source_lock(lock_path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_source_lock_requires_exact_integer_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    lock_data = yaml.safe_load(
        _PACKAGED_LOCK_PATH.read_text(encoding="utf-8")
    )
    lock_data["schema_version"] = schema_version
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(lock_data, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match="schema_version must be 1",
    ):
        _PREPARE.load_source_lock(lock_path)


def test_source_lock_rejects_archive_path_escape(tmp_path: Path) -> None:
    checked_in = yaml.safe_load(
        _PACKAGED_LOCK_PATH.read_text(encoding="utf-8")
    )
    checked_in["source"]["archive_name"] = "../escape.zip"
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(checked_in, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        _PREPARE.PreparationError,
        match="archive_name must be exactly",
    ):
        _PREPARE.load_source_lock(lock_path)


def test_wrong_hash_completed_download_is_quarantined_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_payload = b"right"
    invalid_payload = b"wrong"
    expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
    lock = _PREPARE.SourceLock(
        dataset="afhq-v2",
        url="https://example.invalid/afhq_v2.zip",
        archive_name="afhq_v2.zip",
        expected_bytes=len(expected_payload),
        expected_sha256=expected_sha256,
        license_name="test",
        license_url="https://example.invalid/license",
        homepage="https://example.invalid",
        citation="test",
        contract=_tiny_contract(),
    )
    cache_root = tmp_path / "cache"
    completed_download = (
        cache_root / "raw" / "afhq-v2" / ".downloads" / "afhq_v2.zip"
    )
    completed_download.parent.mkdir(parents=True)
    completed_download.write_bytes(invalid_payload)
    downloads: list[Path] = []

    def fake_download(**kwargs: Any) -> Path:
        destination = kwargs["destination"]
        destination.write_bytes(expected_payload)
        downloads.append(destination)
        return destination

    monkeypatch.setattr(
        _SOURCE_ACQUISITION,
        "download_official_archive",
        fake_download,
    )
    source = _PREPARE.acquire_official_archive(
        lock=lock,
        cache_root=cache_root,
        proxy=None,
    )

    assert len(downloads) == 1
    assert source.sha256 == expected_sha256
    assert source.path.read_bytes() == expected_payload
    quarantined = list(
        completed_download.parent.glob("afhq_v2.zip.*.invalid")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == invalid_payload


@pytest.mark.parametrize(
    ("header", "start", "expected"),
    [
        ("bytes 100-199/200", 100, 200),
        ("bytes 0-99/200", 0, 200),
    ],
)
def test_parse_content_range(
    header: str,
    start: int,
    expected: int,
) -> None:
    assert _DOWNLOADING._parse_content_range(
        header,
        expected_start=start,
    ) == expected


@pytest.mark.parametrize(
    ("header", "start"),
    [
        (None, 0),
        ("bytes 10-19/20", 0),
        ("bytes */20", 0),
        ("bytes 0-20/20", 0),
    ],
)
def test_parse_content_range_rejects_invalid_values(
    header: str | None,
    start: int,
) -> None:
    with pytest.raises(_PREPARE.SourceIntegrityError):
        _DOWNLOADING._parse_content_range(header, expected_start=start)
