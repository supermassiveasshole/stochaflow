from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from stochaflow.data import (
    IMAGE_DATA_SOURCES,
    DataArtifact,
    DataSourceContext,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    build_data_loaders,
)
from stochaflow.utils.config import ComponentConfig, ConfigError

_REPOSITORY = Path(__file__).resolve().parents[1]
_EXAMPLE_SRC = (
    _REPOSITORY / "examples" / "showcases" / "afhq-v2" / "src"
)

if str(_EXAMPLE_SRC) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_SRC))

contracts_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.contracts"
)
dog_artifact_module = importlib.import_module(
    "stochaflow_afhq_v2.dog_artifact"
)
dog_materialization_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.dog_materialization"
)
dog_transform_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.dog_image_transform"
)
materialization_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.materialization"
)
source_lock_module = importlib.import_module(
    "stochaflow_afhq_v2._preparation.source_lock"
)


def _asymmetric_png_bytes(*, size: int, value: int) -> bytes:
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    midpoint = size // 2
    pixels[:, :midpoint] = (
        value,
        (value * 3) % 256,
        (value * 5) % 256,
    )
    pixels[:, midpoint:] = (
        (value + 101) % 256,
        (value + 61) % 256,
        (value + 17) % 256,
    )
    pixels[: size // 4, :, 2] = (value + 211) % 256
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def _write_tiny_archive(path: Path) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        index = 0
        for class_name in ("cat", "dog", "wild"):
            for item in range(3):
                archive.writestr(
                    f"afhq_v2/train/{class_name}/{class_name}_{item:03d}.png",
                    _asymmetric_png_bytes(size=512, value=23 + index),
                )
                index += 1
            archive.writestr(
                f"afhq_v2/test/{class_name}/{class_name}_test.png",
                _asymmetric_png_bytes(size=512, value=23 + index),
            )
            index += 1


def _tiny_lock(archive_path: Path) -> Any:
    return contracts_module.SourceLock(
        dataset="afhq-v2",
        url="https://example.invalid/afhq_v2.zip",
        archive_name="afhq_v2.zip",
        expected_bytes=archive_path.stat().st_size,
        expected_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        license_name="CC BY-NC 4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        homepage="https://example.invalid/afhq",
        citation="Test citation.",
        contract=contracts_module.DatasetContract(
            classes=("cat", "dog", "wild"),
            class_mapping={"cat": 0, "dog": 1, "wild": 2},
            train_count=9,
            test_count=3,
            total_count=12,
            input_resolution=512,
            image_mode="RGB",
            image_format="PNG",
            source_class_counts={
                "train": {"cat": 3, "dog": 3, "wild": 3},
                "test": {"cat": 1, "dog": 1, "wild": 1},
            },
        ),
    )


def _write_tiny_lock(archive_path: Path, lock_path: Path) -> Path:
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


def _dog_data_config(
    *,
    archive: Path,
    lock_file: Path,
    cache_root: Path,
    random_horizontal_flip: bool,
) -> ComponentConfig:
    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")
    return ComponentConfig(
        name="image",
        params={
            "source": {
                "name": "afhq-v2.dog",
                "params": {
                    "archive": str(archive),
                    "lock_file": str(lock_file),
                },
                "materialization": {
                    "cache_root": str(cache_root),
                    "policy": "require",
                    "verification": "full",
                },
            },
            "partition": {"mode": "none"},
            "image": {
                "size": [256, 256],
                "channels": 3,
                "normalize": True,
                "random_horizontal_flip": random_horizontal_flip,
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


def _materialize_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, DataArtifact[ImageFolderArtifactPayload]]:
    archive_path = tmp_path / "afhq_v2.zip"
    _write_tiny_archive(archive_path)
    lock_path = _write_tiny_lock(
        archive_path,
        tmp_path / "afhq-v2.lock.yaml",
    )
    cache_root = tmp_path / "cache"
    artifact = dog_artifact_module.materialize_afhq_v2_dog_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="ensure",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
    )
    return archive_path, lock_path, cache_root, artifact


def test_dog_source_materializes_unlabeled_train_subset_and_generic_batch(
    tmp_path: Path,
) -> None:
    archive_path, lock_path, cache_root, artifact = _materialize_fixture(
        tmp_path
    )
    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext")

    assert "afhq-v2.dog" in IMAGE_DATA_SOURCES.names()
    assert isinstance(artifact, DataArtifact)
    assert artifact.kind == "managed"
    assert artifact.identity.source_name == "afhq-v2.dog"
    assert artifact.identity.artifact_type == (
        "stochaflow.afhq-v2-dog-image-folder.v1"
    )
    assert type(artifact.payload) is ImageFolderArtifactPayload
    assert artifact.payload.validation is None
    assert artifact.payload.test is None
    assert len(artifact.payload.train) == 3
    assert {
        (record.tree, record.path, record.width, record.height)
        for record in artifact.payload.train
    } == {
        ("train", f"dog/dog_{index:03d}.png", 256, 256)
        for index in range(3)
    }
    assert all(
        not hasattr(record, "class_label")
        for record in artifact.payload.train
    )
    prepared_files = {
        path.relative_to(artifact.root / "data").as_posix()
        for path in (artifact.root / "data").rglob("*.png")
    }
    assert prepared_files == {
        f"train/dog/dog_{index:03d}.png" for index in range(3)
    }

    loaders = build_data_loaders(
        _dog_data_config(
            archive=archive_path,
            lock_file=lock_path,
            cache_root=cache_root,
            random_horizontal_flip=False,
        ),
        seed=7,
    )
    images, conditions = next(iter(loaders.train))
    assert images.shape == (2, 3, 256, 256)
    assert conditions == {}
    assert loaders.validation is None
    assert loaders.test is None
    assert loaders.artifact_bindings is not None
    assert loaders.artifact_bindings.identity_for("source") == artifact.identity


def test_dog_cache_hit_and_strict_source_identity_need_no_raw_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path, lock_path, cache_root, artifact = _materialize_fixture(
        tmp_path
    )
    lock = source_lock_module.load_source_lock(lock_path)
    archive_path.unlink()

    def fail_acquisition(**_: Any) -> Any:
        raise AssertionError("cache hit must not acquire the source archive")

    monkeypatch.setattr(
        dog_artifact_module,
        "acquire_official_archive",
        fail_acquisition,
    )
    required = dog_artifact_module.materialize_afhq_v2_dog_artifact(
        DataSourceContext(
            cache_root=cache_root,
            policy="require",
            verification="full",
        ),
        archive=archive_path,
        lock_file=lock_path,
    )
    assert required.identity == artifact.identity
    assert required.root == artifact.root

    monkeypatch.setattr(
        dog_artifact_module,
        "load_source_lock",
        lambda _: replace(lock, expected_sha256="f" * 64),
    )
    with pytest.raises(
        ValueError,
        match="strict resume AFHQ-v2 Dog source identity",
    ):
        dog_artifact_module.materialize_afhq_v2_dog_artifact(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
                expected_identity=artifact.identity,
            ),
            archive=None,
            lock_file=lock_path,
        )


def test_dog_recipe_is_pinned_and_pixel_golden_is_exact(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "afhq_v2.zip"
    archive_path.write_bytes(b"fixture source")
    lock = _tiny_lock(archive_path)
    dog_spec = dog_materialization_module.build_dog_materialization_spec(
        lock=lock
    )
    official_spec = materialization_module.build_materialization_spec(
        lock=lock,
        resolution=256,
    )
    upstream = dog_spec.recipe["transform"]["upstream"]

    assert dog_spec.digest != official_spec.digest
    assert dog_spec.recipe["recipe"]["name"] != (
        official_spec.recipe["recipe"]["name"]
    )
    assert upstream == {
        "repository": "openai/guided-diffusion",
        "commit": "8fb3ad9197f16bbc40620447b2742e13458d2831",
        "file": "guided_diffusion/image_datasets.py",
        "function": "center_crop_arr",
    }
    assert dog_spec.recipe["augmentation"] == {
        "horizontal_flip": "runtime DataBuilder policy"
    }

    width, height = 37, 25
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x * 11 + y * 5 + 17) % 256,
            (x * 13 + y * 19 + 29) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    encoded = BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG")
    actual = dog_transform_module.decode_and_center_crop(
        encoded.getvalue(),
        member_name="gradient.png",
        expected_input_size=(width, height),
        output_resolution=8,
    )

    assert actual.size == (8, 8)
    assert actual.mode == "RGB"
    assert hashlib.sha256(np.asarray(actual).tobytes()).hexdigest() == (
        "749f843203eed043522e496b69eef4d239e3b472faea3281fb3752acf790e773"
    )


def test_horizontal_flip_is_runtime_policy_not_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path, lock_path, cache_root, artifact = _materialize_fixture(
        tmp_path
    )
    no_flip = build_data_loaders(
        _dog_data_config(
            archive=archive_path,
            lock_file=lock_path,
            cache_root=cache_root,
            random_horizontal_flip=False,
        ),
        seed=11,
    )
    transforms_module = importlib.import_module("stochaflow.data.transforms")
    monkeypatch.setattr(transforms_module.random, "random", lambda: 0.0)
    flip = build_data_loaders(
        _dog_data_config(
            archive=archive_path,
            lock_file=lock_path,
            cache_root=cache_root,
            random_horizontal_flip=True,
        ),
        seed=11,
    )

    no_flip_images, _ = next(iter(no_flip.train))
    flip_images, _ = next(iter(flip.train))
    assert torch.equal(flip_images, torch.flip(no_flip_images, dims=(-1,)))
    assert no_flip.artifact_bindings is not None
    assert flip.artifact_bindings is not None
    assert no_flip.artifact_bindings.identity_for("source") == artifact.identity
    assert flip.artifact_bindings.identity_for("source") == artifact.identity

    source = IMAGE_DATA_SOURCES.create(
        "afhq-v2.dog",
        {"random_horizontal_flip": True},
        config_path="data.params.source",
    )
    with pytest.raises(
        ConfigError,
        match="random_horizontal_flip",
    ):
        source.materialize(
            DataSourceContext(
                cache_root=cache_root,
                policy="require",
                verification="manifest",
            )
        )


def test_prepare_tool_selects_dog_profile_and_unlabeled_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = importlib.import_module("stochaflow_afhq_v2.tools.prepare")
    observed: dict[str, Any] = {}
    (tmp_path / "train/dog").mkdir(parents=True)
    payload = ImageFolderArtifactPayload(
        roots={"train": tmp_path / "train"},
        train=(
            ImageFileRecord(
                tree="train",
                path="dog/train.png",
                size_bytes=1,
                sha256="0" * 64,
                width=256,
                height=256,
            ),
        ),
    )

    class PreparedDogSource:
        def materialize(self, context: DataSourceContext) -> Any:
            observed["context"] = context
            return SimpleNamespace(
                root=tmp_path / "artifact",
                manifest_path=tmp_path / "artifact/manifest.json",
                identity=SimpleNamespace(
                    to_dict=lambda: {"source_name": "afhq-v2.dog"}
                ),
                payload=payload,
            )

    def create(
        name: str,
        params: dict[str, Any],
        *,
        config_path: str,
    ) -> PreparedDogSource:
        observed.update(
            name=name,
            params=params,
            config_path=config_path,
        )
        return PreparedDogSource()

    monkeypatch.setattr(tool.IMAGE_DATA_SOURCES, "create", create)
    summary = tool.prepare_artifact(
        cache_root=tmp_path / "cache",
        archive=tmp_path / "afhq_v2.zip",
        lock_file=tmp_path / "lock.yaml",
        resolution=256,
        downloader="python",
        policy="require",
        verification="full",
        profile="dog",
    )

    assert observed["name"] == "afhq-v2.dog"
    assert observed["params"]["resolution"] == 256
    assert summary["identity"]["source_name"] == "afhq-v2.dog"
    assert summary["counts"] == {"train": 1}
    assert "class_mapping" not in summary

    observed.clear()

    def fake_prepare_artifact(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"root": "artifact", "counts": {"train": 1}}

    monkeypatch.setattr(tool, "prepare_artifact", fake_prepare_artifact)
    tool.main(["--profile", "dog", "--no-progress"])
    assert observed["profile"] == "dog"
    assert observed["resolution"] == 256
