from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
import yaml
from PIL import Image
from torch.utils.data import Dataset

from stochaflow.data import (
    DataSource,
    DataSourceContext,
    ManagedDataArtifact,
    ManagedDataArtifactIdentity,
    build_data_loaders,
)
from stochaflow.utils.config import ComponentConfig

_REPOSITORY = Path(__file__).resolve().parents[1]
_EXAMPLE_ROOT = _REPOSITORY / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _EXAMPLE_ROOT / "src"
_PACKAGE_ROOT = _EXAMPLE_SRC / "stochaflow_afhq_v2"
_PREPARE_PATH = _PACKAGE_ROOT / "preparation.py"
_PACKAGED_LOCK_PATH = (
    _PACKAGE_ROOT / "resources" / "afhq-v2.lock.yaml"
)


def _load_prepare_module() -> ModuleType:
    name = "stochaflow_afhq_v2_prepare_test_module"
    spec = importlib.util.spec_from_file_location(name, _PREPARE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_PREPARE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PREPARE = _load_prepare_module()


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
    Image.new(mode, (size, size), color=color).save(output, format="PNG")
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


def _load_extension_data_module() -> ModuleType:
    example_src = str(_EXAMPLE_SRC)
    if example_src not in sys.path:
        sys.path.insert(0, example_src)
    importlib.invalidate_caches()
    return importlib.import_module(
        "stochaflow_afhq_v2.stochaflow_ext.data"
    )


def test_prepare_archive_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)

    images = _PREPARE.inspect_archive(
        archive_path,
        contract=lock.contract,
    )
    first_selection = _PREPARE.select_validation_members(
        images,
        classes=lock.contract.classes,
        per_class=1,
        seed="fixture-seed",
    )
    second_selection = _PREPARE.select_validation_members(
        tuple(reversed(images)),
        classes=lock.contract.classes,
        per_class=1,
        seed="fixture-seed",
    )
    assert first_selection == second_selection
    assert len(first_selection) == 3

    first = _PREPARE.prepare_archive(
        source=source,
        lock=lock,
        cache_root=tmp_path / "cache",
        resolution=4,
        validation_per_class=1,
        validation_seed="fixture-seed",
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
            "classes": {"cat": 2, "dog": 2, "wild": 2},
            "total": 6,
        },
        "validation": {
            "classes": {"cat": 1, "dog": 1, "wild": 1},
            "total": 3,
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
        for split in ("train", "validation", "test")
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
        validation_per_class=1,
        validation_seed="fixture-seed",
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
        validation_per_class=1,
        validation_seed="fixture-seed",
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


def test_afhq_extension_materializes_and_feeds_image_builder(
    tmp_path: Path,
) -> None:
    extension = _load_extension_data_module()
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
        "validation_per_class": 1,
        "validation_seed": "fixture-seed",
    }
    source = extension.AFHQV2ImageDataSource(
        source_params,
        config_path="data.params.source",
    )
    assert isinstance(source, DataSource)

    artifact = source.materialize(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        )
    )

    assert isinstance(artifact, ManagedDataArtifact)
    assert not isinstance(artifact, Dataset)
    assert artifact.identity.source_name == "afhq-v2.official"
    assert set(artifact.payload.roots) == {"train", "validation", "test"}
    assert len(artifact.payload.train) == 6
    assert len(artifact.payload.validation or ()) == 3
    assert len(artifact.payload.test or ()) == 3
    assert all(
        record.tree == "train"
        and not record.path.startswith("train/")
        for record in artifact.payload.train
    )
    with pytest.raises(ValueError, match="expected a different data source"):
        source.materialize(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=replace(
                    artifact.identity,
                    source_name="another-source",
                ),
            )
        )
    with pytest.raises(ValueError, match="identity does not match"):
        source.materialize(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=replace(
                    artifact.identity,
                    artifact_digest="0" * 64,
                ),
            )
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

    required = source.materialize(
        DataSourceContext(
            cache_root=cache_root,
            policy="require",
            verification="full",
        )
    )
    assert isinstance(required, ManagedDataArtifact)
    assert required.identity == artifact.identity
    assert required.artifact_root == artifact.artifact_root

    loaders = build_data_loaders(
        ComponentConfig(
            name="image",
            params={
                "source": {
                    "name": "afhq-v2.official",
                    "params": source_params,
                    "materialization": {
                        "cache_root": str(cache_root),
                        "policy": "require",
                        "verification": "full",
                    },
                },
                "image": {
                    "size": [4, 4],
                    "channels": 3,
                    "normalize": True,
                    "random_horizontal_flip": False,
                },
                "loader": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "shuffle": False,
                    "drop_last": False,
                    "pin_memory": False,
                    "persistent_workers": False,
                    "prefetch_factor": None,
                    "steps_per_epoch": "auto",
                },
                "partition": {"mode": "official"},
            },
        ),
        seed=7,
    )
    images, conditions = next(iter(loaders.train))
    assert images.shape == (2, 3, 4, 4)
    assert conditions == {}
    assert loaders.artifact_bindings is not None
    assert (
        loaders.artifact_bindings.identity_for("source")
        == artifact.identity
    )

    changed_record = artifact.payload.train[0]
    image_path = (
        artifact.payload.roots[changed_record.tree]
        / changed_record.path
    )
    image_path.write_bytes(b"\0" * image_path.stat().st_size)
    manifest_only = source.materialize(
        DataSourceContext(
            cache_root=cache_root,
            policy="require",
            verification="manifest",
        )
    )
    assert manifest_only.identity == artifact.identity
    with pytest.raises(ValueError, match="image content changed"):
        next(iter(loaders.train))
    with pytest.raises(
        extension.PreparationError,
        match="prepared file digest mismatch",
    ):
        source.materialize(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="full",
            )
        )


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
    extension = _load_extension_data_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    lock = extension.load_source_lock(lock_path)
    plan = extension.build_preparation_plan(
        lock=lock,
        resolution=4,
        validation_per_class=1,
        validation_seed="fixture-seed",
    )
    assert lock.expected_sha256 is not None
    expected = ManagedDataArtifactIdentity(
        artifact_type="stochaflow.image-folder.v1",
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
        extension,
        "require_prepared_artifact",
        unexpected_cache_io,
    )
    monkeypatch.setattr(
        extension,
        "acquire_official_archive",
        unexpected_cache_io,
    )
    source = extension.AFHQV2ImageDataSource(
        {
            "archive": str(archive_path),
            "lock_file": str(lock_path),
            "resolution": 4,
            "validation_per_class": 1,
            "validation_seed": "fixture-seed",
        },
        config_path="data.params.source",
    )
    cache_root = tmp_path / "cache"

    with pytest.raises(ValueError, match=message):
        source.materialize(
            DataSourceContext(
                cache_root=cache_root,
                policy="ensure",
                verification="full",
                expected_identity=expected,
            )
        )

    assert calls == []
    assert not cache_root.exists()


def test_afhq_source_config_has_one_public_validation_entrypoint() -> None:
    extension = _load_extension_data_module()
    config = extension.AFHQV2DataSourceConfig.from_params(
        {"resolution": 64, "validation_per_class": 100},
        path="data.params.source.params",
    )

    config.validate(path="data.params.source.params")
    with pytest.raises(ValueError, match="unknown AFHQ-v2 source field"):
        extension.AFHQV2DataSourceConfig.from_params(
            {"unexpected": True},
            path="data.params.source.params",
        )
    with pytest.raises(ValueError, match="downloader must be"):
        extension.AFHQV2DataSourceConfig.from_params(
            {"downloader": "mirror"},
            path="data.params.source.params",
        )
    for value in ([], {}):
        with pytest.raises(ValueError, match="downloader must be"):
            extension.AFHQV2DataSourceConfig.from_params(
                {"downloader": value},
                path="data.params.source.params",
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
        validation_per_class=1,
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
            validation_per_class=1,
        )


def test_prepare_rejects_archive_path_substitution_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    source = _tiny_source(archive_path)
    lock = _tiny_lock(archive_path)
    original_process = _PREPARE._process_images
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

    monkeypatch.setattr(_PREPARE, "_process_images", substitute_archive)
    cache_root = tmp_path / "cache"

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"changed while it was in use",
    ):
        _PREPARE.prepare_archive(
            source=source,
            lock=lock,
            cache_root=cache_root,
            resolution=4,
            validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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
        validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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
            validation_per_class=1,
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
        validation_per_class=1,
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

    monkeypatch.setattr(_PREPARE.os, "scandir", substitute_before_scan)

    with pytest.raises(
        _PREPARE.PreparationError,
        match=r"unsafe|symlink|reparse|changed",
    ):
        _PREPARE.require_prepared_artifact(
            lock=lock,
            cache_root=tmp_path / "cache",
            resolution=4,
            validation_per_class=1,
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
        validation_per_class=1,
    )
    plan = _PREPARE.build_preparation_plan(
        lock=lock,
        resolution=4,
        validation_per_class=1,
    )
    preserved = tmp_path / "preserved-container"
    original = _PREPARE.canonical_directory
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
        _PREPARE,
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
        validation_per_class=1,
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
        _PREPARE._quarantine_invalid_prepared_artifact(
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
            validation_per_class=1,
        )

    assert list(outside.iterdir()) == []


def test_ensure_repairs_invalid_prepared_artifact_under_lock(
    tmp_path: Path,
) -> None:
    extension = _load_extension_data_module()
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, image_size=512)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    source = extension.AFHQV2ImageDataSource(
        {
            "archive": str(archive_path),
            "lock_file": str(lock_path),
            "resolution": 4,
            "validation_per_class": 1,
            "validation_seed": "fixture-seed",
        },
        config_path="data.params.source",
    )
    context = DataSourceContext(
        cache_root=cache_root,
        policy="ensure",
        verification="full",
    )
    first = source.materialize(context)
    record = first.payload.train[0]
    image_path = first.payload.roots[record.tree] / record.path
    original = image_path.read_bytes()
    image_path.write_bytes(b"\0" * len(original))

    repaired = source.materialize(context)

    assert repaired.identity == first.identity
    repaired_path = repaired.payload.roots[record.tree] / record.path
    assert repaired_path.read_bytes() == original
    quarantined = list(
        first.artifact_root.parent.glob(
            f"{first.artifact_root.name}.*.invalid"
        )
    )
    assert len(quarantined) == 1
    quarantined_image = quarantined[0] / record.tree / record.path
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
        validation_per_class=1,
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
            validation_per_class=1,
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
            validation_per_class=1,
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
            "created_unix",
            "hostname",
            "nonce",
            "pid",
        }
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


def test_preparation_lock_metadata_handles_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = _PREPARE.os.write

    def partial_write(descriptor: int, payload: Any) -> int:
        size = max(1, len(payload) // 2)
        return original_write(descriptor, payload[:size])

    monkeypatch.setattr(_PREPARE.os, "write", partial_write)
    lock_path = tmp_path / "partial-write.lock"

    with _PREPARE.ArtifactPreparationLock(lock_path):
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))

    assert metadata["pid"] == os.getpid()
    assert set(metadata) == {
        "created_unix",
        "hostname",
        "nonce",
        "pid",
    }


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
            validation_per_class=1,
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
        _PREPARE,
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
    assert _PREPARE._parse_content_range(header, expected_start=start) == expected


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
        _PREPARE._parse_content_range(header, expected_start=start)
