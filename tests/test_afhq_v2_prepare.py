from __future__ import annotations

import hashlib
import importlib
import json
import stat
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
import torch
import yaml
from PIL import Image

from stochaflow.data import (
    ClassLabeledImageFileRecord,
    ClassLabeledImageFolderArtifactPayload,
    DataArtifact,
    DataSourceContext,
    ImageFileRecord,
    build_data_loaders,
)
from stochaflow.utils.config import ComponentConfig

_REPOSITORY = Path(__file__).resolve().parents[1]
_EXAMPLE_ROOT = _REPOSITORY / "examples" / "showcases" / "afhq-v2"
_EXAMPLE_SRC = _EXAMPLE_ROOT / "src"
_PACKAGE_ROOT = _EXAMPLE_SRC / "stochaflow_afhq_v2"
_PACKAGED_LOCK_PATH = (
    _PACKAGE_ROOT / "resources" / "afhq-v2.lock.yaml"
)

if str(_EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_SRC))

archive_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.archive"
)
contracts_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.contracts"
)
downloading_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.downloading"
)
source_acquisition_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.source_acquisition"
)
source_lock_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.source_lock"
)
materialization_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.materialization"
)
artifact_module = importlib.import_module("stochaflow_afhq_v2.artifact")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    image.putpixel(
        (0, 0),
        (
            ((value + 97) % 256, (value + 53) % 256, (value + 11) % 256)
            if mode == "RGB"
            else (value + 97) % 256
        ),
    )
    image.save(output, format="PNG")
    return output.getvalue()


def _write_tiny_archive(
    path: Path,
    *,
    image_size: int = 512,
    mode: str = "RGB",
) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        index = 0
        for class_name in ("cat", "dog", "wild"):
            for item in range(3):
                archive.writestr(
                    f"afhq_v2/train/{class_name}/{class_name}_{item:03d}.png",
                    _png_bytes(
                        size=image_size,
                        mode=mode,
                        value=32 + index,
                    ),
                )
                index += 1
            archive.writestr(
                f"afhq_v2/test/{class_name}/{class_name}_test.png",
                _png_bytes(
                    size=image_size,
                    mode=mode,
                    value=32 + index,
                ),
            )
            index += 1


def _tiny_contract(*, input_resolution: int = 512) -> Any:
    return contracts_module.DatasetContract(
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
    input_resolution: int = 512,
) -> Any:
    return contracts_module.SourceLock(
        dataset="afhq-v2",
        url="https://example.invalid/afhq_v2.zip",
        archive_name="afhq_v2.zip",
        expected_bytes=archive_path.stat().st_size,
        expected_sha256=_sha256_file(archive_path),
        license_name="CC BY-NC 4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        homepage="https://example.invalid/afhq",
        citation="Test citation.",
        contract=_tiny_contract(input_resolution=input_resolution),
    )


def _write_tiny_source_lock(archive_path: Path, lock_path: Path) -> Path:
    lock = _tiny_lock(archive_path)
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


def _class_data_config(
    *,
    archive: Path,
    lock_file: Path,
    cache_root: Path,
    policy: str,
) -> ComponentConfig:
    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")
    return ComponentConfig(
        name="class_labeled_image",
        params={
            "source": {
                "name": "afhq-v2.official",
                "params": {
                    "archive": str(archive),
                    "lock_file": str(lock_file),
                    "resolution": 4,
                },
                "materialization": {
                    "cache_root": str(cache_root),
                    "policy": policy,
                    "verification": "full",
                },
            },
            "partition": {
                "validation_per_class": 1,
                "seed": "fixture-seed",
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
        },
    )


def test_checked_in_source_lock_is_fully_pinned() -> None:
    raw_lock = yaml.safe_load(_PACKAGED_LOCK_PATH.read_text(encoding="utf-8"))
    lock = source_lock_module.load_source_lock(_PACKAGED_LOCK_PATH)

    assert raw_lock["source"]["type"] == "official_archive"
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


def test_source_and_materialization_identities_are_orthogonal(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    archive_path.write_bytes(b"fixture source")
    lock = _tiny_lock(archive_path)
    source_digest = artifact_module._validate_lock(lock)
    recipe_digest = materialization_module.build_materialization_spec(
        lock=lock,
        resolution=128,
    ).digest

    relocated = replace(lock, url="https://mirror.invalid/afhq_v2.zip")
    assert artifact_module._validate_lock(relocated) == source_digest
    assert (
        materialization_module.build_materialization_spec(
            lock=relocated,
            resolution=128,
        ).digest
        == recipe_digest
    )

    changed_source = replace(lock, expected_sha256="f" * 64)
    assert artifact_module._validate_lock(changed_source) == "f" * 64
    assert (
        materialization_module.build_materialization_spec(
            lock=changed_source,
            resolution=128,
        ).digest
        == recipe_digest
    )

    assert (
        materialization_module.build_materialization_spec(
            lock=lock,
            resolution=256,
        ).digest
        != recipe_digest
    )


def test_prepare_tool_uses_the_registered_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = importlib.import_module("stochaflow_afhq_v2.tools.prepare")
    observed: dict[str, Any] = {}
    (tmp_path / "train/cat").mkdir(parents=True)
    (tmp_path / "test/cat").mkdir(parents=True)
    payload = ClassLabeledImageFolderArtifactPayload(
        roots={"train": tmp_path / "train", "test": tmp_path / "test"},
        class_mapping={"cat": 0},
        train=(
            ClassLabeledImageFileRecord(
                image=ImageFileRecord(
                    tree="train",
                    path="cat/train.png",
                    size_bytes=1,
                    sha256="0" * 64,
                    width=128,
                    height=128,
                ),
                class_label=0,
            ),
        ),
        test=(
            ClassLabeledImageFileRecord(
                image=ImageFileRecord(
                    tree="test",
                    path="cat/test.png",
                    size_bytes=1,
                    sha256="1" * 64,
                    width=128,
                    height=128,
                ),
                class_label=0,
            ),
        ),
    )

    class PreparedSource:
        def materialize(self, context: DataSourceContext) -> Any:
            observed["context"] = context
            return SimpleNamespace(
                root=tmp_path / "artifact",
                manifest_path=tmp_path / "artifact/manifest.json",
                identity=SimpleNamespace(
                    to_dict=lambda: {"source_name": "afhq-v2.official"}
                ),
                payload=payload,
            )

    def create(
        name: str,
        params: dict[str, Any],
        *,
        config_path: str,
    ) -> PreparedSource:
        observed.update(
            name=name,
            params=params,
            config_path=config_path,
        )
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

    assert observed["name"] == "afhq-v2.official"
    assert observed["config_path"] == "prepare"
    assert observed["params"]["archive"] == str(tmp_path / "afhq_v2.zip")
    assert observed["context"].policy == "require"
    assert summary["root"] == str(tmp_path / "artifact")
    assert summary["counts"] == {"train": 1, "test": 1}


def test_managed_source_cache_hit_and_builder_do_not_need_raw_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    artifact = artifact_module.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=4,
    )
    changed_source_lock = replace(
        source_lock_module.load_source_lock(lock_path),
        expected_sha256="f" * 64,
    )

    assert isinstance(artifact, DataArtifact)
    assert artifact.kind == "managed"
    assert artifact.identity.schema_version == 2
    assert artifact.identity.source_name == "afhq-v2.official"
    assert artifact.manifest_path == artifact.root / "manifest.json"
    assert len(artifact.payload.train) == 9
    assert artifact.payload.validation is None
    assert len(artifact.payload.test or ()) == 3
    assert (artifact.root / "data/_index/images.json").is_file()
    assert (
        cache_root
        / "source-acquisition/afhq-v2/raw/afhq-v2"
        / artifact.identity.source_digest
        / "afhq_v2.zip"
    ).is_file()

    archive_path.unlink()

    def fail_acquisition(**_: Any) -> Any:
        raise AssertionError("cache hit must not acquire the source archive")

    monkeypatch.setattr(
        artifact_module,
        "acquire_official_archive",
        fail_acquisition,
    )
    required = artifact_module.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="require",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=4,
    )
    assert required.identity == artifact.identity
    assert required.root == artifact.root
    assert all(
        record.image.tree == "train"
        and not record.image.path.startswith("train/")
        for record in required.payload.train
    )

    loaders = build_data_loaders(
        _class_data_config(
            archive=archive_path,
            lock_file=lock_path,
            cache_root=cache_root,
            policy="require",
        ),
        seed=7,
    )
    images, conditions = next(iter(loaders.train))
    assert images.shape == (2, 3, 4, 4)
    assert conditions["class_label"].dtype == torch.long
    assert loaders.artifact_bindings is not None
    assert (
        loaders.artifact_bindings.identity_for("source")
        == artifact.identity
    )
    resumed = build_data_loaders(
        _class_data_config(
            archive=archive_path,
            lock_file=lock_path,
            cache_root=cache_root,
            policy="require",
        ),
        seed=7,
        strict_resume=True,
        expected_artifacts=loaders.artifact_bindings,
    )
    assert resumed.artifact_bindings == loaders.artifact_bindings

    with pytest.raises(
        ValueError,
        match="strict resume AFHQ-v2 materialization identity",
    ):
        artifact_module.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=artifact.identity,
            ),
            archive=None,
            lock_file=lock_path,
            resolution=6,
        )

    monkeypatch.setattr(
        artifact_module,
        "load_source_lock",
        lambda _: changed_source_lock,
    )
    with pytest.raises(
        ValueError,
        match="strict resume AFHQ-v2 source identity",
    ):
        artifact_module.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=artifact.identity,
            ),
            archive=None,
            lock_file=lock_path,
            resolution=4,
        )


def test_managed_source_rejects_locator_for_different_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    resolution_four = artifact_module.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=4,
    )
    resolution_six = artifact_module.materialize_afhq_v2_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
        resolution=6,
    )
    locators = tuple(
        (resolution_four.root.parents[1] / "locators").glob("*.json")
    )
    by_artifact_digest = {
        json.loads(path.read_text(encoding="utf-8"))["artifact_digest"]: path
        for path in locators
    }
    by_artifact_digest[resolution_four.identity.artifact_digest].write_bytes(
        by_artifact_digest[resolution_six.identity.artifact_digest].read_bytes()
    )

    def fail_acquisition(**_: Any) -> Any:
        raise AssertionError("cache candidate validation must not acquire raw data")

    monkeypatch.setattr(
        artifact_module,
        "acquire_official_archive",
        fail_acquisition,
    )
    with pytest.raises(
        ValueError,
        match="materialization identity does not match the current recipe",
    ):
        artifact_module.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
            ),
            archive=None,
            lock_file=lock_path,
            resolution=4,
        )


def test_non_rgb_source_is_rejected_during_managed_build(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path, mode="L")
    lock_path = _write_tiny_source_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )

    with pytest.raises(
        contracts_module.PreparationError,
        match="expected RGB",
    ):
        artifact_module.materialize_afhq_v2_artifact(
            DataSourceContext(
                cache_root=tmp_path / "cache",
                policy="ensure",
                verification="full",
            ),
            archive=archive_path,
            lock_file=lock_path,
            resolution=4,
        )


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.png",
        "/absolute.png",
        "afhq_v2/train/cat/../../escape.png",
        "afhq_v2/train/cat/CON.png",
        "afhq_v2/train/cat/evil?.png",
    ],
)
def test_archive_audit_rejects_unsafe_members(
    tmp_path: Path,
    member_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(member_name, _png_bytes())

    with pytest.raises(contracts_module.PreparationError):
        archive_module.inspect_archive(
            archive_path,
            contract=_tiny_contract(input_resolution=8),
        )


def test_archive_audit_rejects_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    info = ZipInfo("afhq_v2/train/cat/link.png")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive_path, mode="w") as archive:
        archive.writestr(info, b"target")

    with pytest.raises(
        contracts_module.PreparationError,
        match="symbolic link",
    ):
        archive_module.inspect_archive(
            archive_path,
            contract=_tiny_contract(input_resolution=8),
        )


def test_source_lock_is_strict(tmp_path: Path) -> None:
    raw = yaml.safe_load(_PACKAGED_LOCK_PATH.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(
        contracts_module.PreparationError,
        match="unknown fields",
    ):
        source_lock_module.load_source_lock(lock_path)


def test_wrong_completed_download_is_quarantined_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_payload = b"right"
    invalid_payload = b"wrong"
    expected_sha256 = hashlib.sha256(expected_payload).hexdigest()
    lock = contracts_module.SourceLock(
        dataset="afhq-v2",
        url="https://example.invalid/afhq_v2.zip",
        archive_name="afhq_v2.zip",
        expected_bytes=len(expected_payload),
        expected_sha256=expected_sha256,
        license_name="test",
        license_url="https://example.invalid/license",
        homepage="https://example.invalid",
        citation="test",
        contract=_tiny_contract(input_resolution=8),
    )
    cache_root = tmp_path / "cache"
    completed = (
        cache_root / "raw/afhq-v2/.downloads/afhq_v2.zip"
    )
    completed.parent.mkdir(parents=True)
    completed.write_bytes(invalid_payload)
    downloads: list[Path] = []

    def fake_download(**kwargs: Any) -> Path:
        destination = kwargs["destination"]
        destination.write_bytes(expected_payload)
        downloads.append(destination)
        return destination

    monkeypatch.setattr(
        source_acquisition_module,
        "download_official_archive",
        fake_download,
    )
    source = source_acquisition_module.acquire_official_archive(
        lock=lock,
        cache_root=cache_root,
        proxy=None,
    )

    assert downloads == [completed]
    assert source.sha256 == expected_sha256
    assert len(list(completed.parent.glob("afhq_v2.zip.*.invalid"))) == 1


@pytest.mark.parametrize(
    ("header", "start", "expected"),
    [
        ("bytes 100-199/200", 100, 200),
        ("bytes 0-99/200", 0, 200),
    ],
)
def test_download_content_range_parser(
    header: str,
    start: int,
    expected: int,
) -> None:
    assert downloading_module._parse_content_range(
        header,
        expected_start=start,
    ) == expected
