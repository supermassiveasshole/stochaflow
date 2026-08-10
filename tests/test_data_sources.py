"""Integration tests for artifact-backed image sources."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import unicodedata
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from stochaflow._builtin_activation import activate_data_builtins
from stochaflow.data import (
    DataArtifact,
    DataArtifactBindings,
    DataArtifactLoadContext,
    DataArtifactStore,
    DataBuilderContext,
    DataSourceContext,
    ImageDataSource,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    ManagedDataArtifactBuild,
    torchvision_source,
)
from stochaflow.data.datasets import ImageDatasetFactory
from stochaflow.data.folder_sources import (
    ImageFolderDataSource,
    PairedImageFolderDataSource,
)
from stochaflow.data.image_contracts import (
    IMAGE_DATA_SOURCES,
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


def test_source_factory_registers_builtin_image_sources() -> None:
    """Keep image sources in the explicit operation activation lifecycle."""

    activate_data_builtins()
    assert IMAGE_DATA_SOURCES.resolve("torchvision") is TorchvisionImageDataSource
    assert IMAGE_DATA_SOURCES.resolve("image_folder") is ImageFolderDataSource
    assert (
        IMAGE_DATA_SOURCES.resolve("paired_image_folders")
        is PairedImageFolderDataSource
    )


def test_builtin_image_sources_exist_in_a_fresh_process() -> None:
    """Prove explicit Data activation works without prior test imports."""

    script = (
        "from stochaflow._builtin_activation import activate_data_builtins; "
        "activate_data_builtins(); "
        "from stochaflow.data import IMAGE_DATA_SOURCES; "
        "assert IMAGE_DATA_SOURCES.names() == "
        "('image_folder', 'paired_image_folders', 'torchvision')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def create_directory_link(path: Path, target: Path) -> None:
    """Create a directory symlink or Windows junction."""

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
    """Acquisition-compatible torchvision double."""

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
            (data_root / f"{marker}.bin").write_bytes(marker.encode())
        if not (data_root / f"{marker}.bin").is_file():
            raise RuntimeError("fixture managed dataset is unavailable")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        if index != 0:
            raise IndexError(index)
        return Image.new("L", (28, 28)), 0


@TEST_IMAGE_SOURCES.register("tests.fake-image")
class FakeImageDataSource(ImageDataSource):
    """Independent extension using the same public managed store."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ImageFolderArtifactPayload]:
        source_root = Path(self.params["root"]).resolve()

        def build(data_root: Path) -> ManagedDataArtifactBuild:
            encoded = (source_root / "sample.png").read_bytes()
            (data_root / "sample.png").write_bytes(encoded)
            return ManagedDataArtifactBuild(
                source_digest=hashlib.sha256(encoded).hexdigest(),
                materialization_digest="b" * 64,
                domain={"schema_version": 1},
            )

        def load(
            load_context: DataArtifactLoadContext,
        ) -> ImageFolderArtifactPayload:
            image_path = load_context.data_root / "sample.png"
            encoded = image_path.read_bytes()
            with Image.open(image_path) as image:
                width, height = image.size
            return ImageFolderArtifactPayload(
                roots={"train": load_context.data_root},
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
            )

        return DataArtifactStore(context).materialize_managed(
            artifact_type="tests.image-folder.v1",
            source_name="tests.fake-image",
            materializer_name="tests.fixture",
            locator_key={},
            build=build,
            load=load,
        )


def test_reference_source_indexes_without_copy_in_canonical_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "z.png", color=(255, 0, 0))
    write_image(root / "nested" / "a.jpg", color=(0, 255, 0))
    artifact = ImageFolderDataSource(
        {"root": str(root), "layout": "flat"},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))

    assert artifact.kind == "referenced"
    assert [record.path for record in artifact.payload.train] == [
        "nested/a.jpg",
        "z.png",
    ]
    assert artifact.payload.roots["train"] == root.resolve()
    assert not (artifact.root / "data" / "nested" / "a.jpg").exists()
    assert len(cast(Sized, ImageDatasetFactory().build(artifact).train)) == 2
    assert str(root.resolve()) not in artifact.manifest_path.read_text()


@pytest.mark.parametrize("cache_inside_source", [True, False])
def test_reference_root_and_cache_must_not_overlap(
    tmp_path: Path,
    cache_inside_source: bool,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = root / "cache" if cache_inside_source else tmp_path / "cache"
    selected_root = root if cache_inside_source else cache / "external"
    if not cache_inside_source:
        write_image(selected_root / "sample.png")

    with pytest.raises(ValueError, match="overlaps"):
        ImageFolderDataSource(
            {"root": str(selected_root)},
            config_path="data.params.source",
        ).materialize(source_context(cache))


def test_manifest_verification_is_cheap_but_dataset_reads_are_strict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    image = root / "sample.bmp"
    write_image(image)
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    source.materialize(source_context(tmp_path / "cache"))
    encoded = bytearray(image.read_bytes())
    encoded[-1] ^= 1
    image.write_bytes(encoded)

    cached = source.materialize(
        source_context(
            tmp_path / "cache",
            policy="require",
            verification="manifest",
        )
    )
    with pytest.raises(ValueError, match=r"digest|content changed"):
        ImageDatasetFactory().build(cached).train[0]
    with pytest.raises(ValueError, match="content digest"):
        source.materialize(
            source_context(
                tmp_path / "cache",
                policy="require",
                verification="full",
            )
        )


def test_reference_rejects_link_substitution_and_linked_ancestors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    target = tmp_path / "target.png"
    write_image(target)
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    source.materialize(source_context(tmp_path / "cache"))
    (root / "sample.png").unlink()
    try:
        (root / "sample.png").symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    with pytest.raises((OSError, ValueError), match=r"link|reparse"):
        source.materialize(
            source_context(tmp_path / "cache", policy="require")
        )

    real_parent = tmp_path / "real"
    write_image(real_parent / "folder" / "sample.png")
    linked_parent = tmp_path / "linked"
    create_directory_link(linked_parent, real_parent)
    with pytest.raises((OSError, ValueError), match=r"link|reparse"):
        ImageFolderDataSource(
            {"root": str(linked_parent / "folder")},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "other-cache"))


def test_reference_rejects_special_files_and_path_collisions(
    tmp_path: Path,
) -> None:
    fifo_root = tmp_path / "fifo"
    fifo_root.mkdir()
    fifo = fifo_root / "sample.png"
    mkfifo = getattr(os, "mkfifo", None)
    if mkfifo is None:
        pytest.skip("FIFO creation is unavailable")
    mkfifo(fifo)
    with pytest.raises(
        ValueError,
        match=r"regular file|unsupported filesystem entry",
    ):
        ImageFolderDataSource(
            {"root": str(fifo_root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "fifo-cache"))

    collision_root = tmp_path / "collision"
    write_image(collision_root / "A.png")
    write_image(collision_root / "a.png")
    if len(tuple(collision_root.iterdir())) < 2:
        return
    with pytest.raises(ValueError, match="collision"):
        ImageFolderDataSource(
            {"root": str(collision_root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "collision-cache"))


def test_reference_rejects_non_nfc_paths(tmp_path: Path) -> None:
    root = tmp_path / "images"
    decomposed = unicodedata.normalize("NFD", "é") + ".png"
    if decomposed == unicodedata.normalize("NFC", decomposed):
        pytest.skip("filesystem normalizes names to NFC")
    write_image(root / decomposed)

    with pytest.raises(ValueError, match="NFC"):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(tmp_path / "cache"))


def test_require_miss_is_read_only_for_referenced_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"

    with pytest.raises(FileNotFoundError):
        ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(cache, policy="require"))

    assert not cache.exists()


def test_paired_source_matches_stems_and_detects_missing_pairs(
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

    assert artifact.kind == "referenced"
    assert isinstance(artifact.payload, PairedImageFolderArtifactPayload)
    assert len(artifact.payload.train) == 1

    (low / "nested" / "sample.jpg").unlink()
    write_image(low / "nested" / "other.jpg")
    with pytest.raises(ValueError, match="missing LR"):
        source.materialize(source_context(tmp_path / "other-cache"))


def test_reference_identity_is_location_independent(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_image(first / "sample.png")
    write_image(second / "sample.png")
    first_artifact = ImageFolderDataSource(
        {"root": str(first)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    second_artifact = ImageFolderDataSource(
        {"root": str(second)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))

    assert first_artifact.identity == second_artifact.identity
    assert first_artifact.root == second_artifact.root


def test_concurrent_reference_locators_converge_on_one_object(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_image(first / "sample.png")
    write_image(second / "sample.png")
    cache = tmp_path / "cache"

    def materialize(root: Path) -> DataArtifact[ImageFolderArtifactPayload]:
        return ImageFolderDataSource(
            {"root": str(root)},
            config_path="data.params.source",
        ).materialize(source_context(cache))

    with ThreadPoolExecutor(max_workers=2) as pool:
        artifacts = tuple(pool.map(materialize, (first, second)))

    assert artifacts[0].root == artifacts[1].root
    assert len(tuple(artifacts[0].root.parent.iterdir())) == 1


def test_corrupt_reference_sidecar_is_repaired_by_ensure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    cache = tmp_path / "cache"
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    artifact = source.materialize(source_context(cache))
    shard = next((artifact.root / "data" / "image-inventory").iterdir())
    shard.write_bytes(shard.read_bytes().replace(b'"width":8', b'"width":9'))

    with pytest.raises(ValueError, match=r"digest|inventory"):
        source.materialize(source_context(cache, policy="require"))
    repaired = source.materialize(source_context(cache))

    assert repaired.payload.train[0].width == 8
    quarantine = artifact.root.parents[1] / "quarantine" / "objects"
    assert len(tuple(quarantine.iterdir())) == 1


def test_source_factory_checks_binding_before_materialization(
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
    builder_context = DataBuilderContext(
        params={},
        seed=1,
        strict_resume=True,
        expected_artifacts=DataArtifactBindings(),
    )

    with pytest.raises(KeyError, match="missing data artifact binding"):
        ImageSourceFactory().materialize(
            config,
            binding_id="source",
            builder_context=builder_context,
            path="data.params.source",
        )
    assert not (tmp_path / "cache").exists()


def test_materialization_verification_workers_support_config_and_override(
    tmp_path: Path,
) -> None:
    config = DataSourceMaterializationConfig(
        cache_root=str(tmp_path / "cache"),
        verification_workers=5,
    )

    assert config.context().verification_workers == 5
    assert config.context(verification_workers=2).verification_workers == 2


@pytest.mark.parametrize("workers", [0, -1, True, 9, 10_000])
def test_materialization_rejects_invalid_verification_workers(
    tmp_path: Path,
    workers: object,
) -> None:
    config = DataSourceMaterializationConfig(
        cache_root=str(tmp_path / "cache"),
        verification_workers=workers,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="verification_workers"):
        config.validate(path="data.params.source.materialization")


def test_torchvision_managed_cache_hit_require_and_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    with pytest.raises(FileNotFoundError):
        source.materialize(source_context(cache, policy="require"))
    assert not cache.exists()

    artifact = source.materialize(source_context(cache))
    acquisition_count = FixtureManagedVisionDataset.acquisitions
    cached = source.materialize(
        source_context(cache, policy="require", verification="manifest")
    )
    assert cached.identity == artifact.identity
    assert FixtureManagedVisionDataset.acquisitions == acquisition_count
    assert artifact.payload.root == artifact.root / "data"

    sidecar = artifact.root / "data" / "image_dimensions.json"
    sidecar.write_bytes(
        sidecar.read_bytes().replace(b"[28,28]", b"[29,28]", 1)
    )
    with pytest.raises(ValueError, match=r"dimensions|inventory"):
        source.materialize(
            source_context(cache, policy="require", verification="manifest")
        )
    repaired = source.materialize(source_context(cache))
    assert repaired.payload.train_dimensions[0].width == 28
    assert FixtureManagedVisionDataset.acquisitions > acquisition_count


def test_extension_source_uses_unified_handle_without_name_dispatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png")
    config = ImageSourceConfig(
        name="tests.fake-image",
        params={"root": str(root)},
        materialization=DataSourceMaterializationConfig(
            cache_root=str(tmp_path / "cache")
        ),
    )
    artifact = ImageSourceFactory(TEST_IMAGE_SOURCES).materialize(
        config,
        binding_id="source",
        builder_context=DataBuilderContext(
            params={},
            seed=1,
            strict_resume=False,
            expected_artifacts=None,
        ),
        path="data.params.source",
    )

    assert isinstance(artifact, DataArtifact)
    assert artifact.kind == "managed"
    assert artifact.identity.source_name == "tests.fake-image"
    assert len(cast(Sized, ImageDatasetFactory().build(artifact).train)) == 1


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape.png",
        "/absolute.png",
        "nested\\windows.png",
        "C:/drive.png",
        "folder/../escape.png",
        "CON.png",
        "name. ",
    ],
)
def test_image_file_record_rejects_unsafe_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError, match=r"path|component|character|reserved"):
        ImageFileRecord(
            tree="train",
            path=unsafe_path,
            size_bytes=1,
            sha256="a" * 64,
            width=1,
            height=1,
        )
