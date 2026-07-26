"""Scalability and integrity contracts for multi-resolution metadata."""

from __future__ import annotations

import hashlib
import json
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image
from torch.utils.data import Dataset

from stochaflow.data import (
    DataSourceContext,
    ImageDimensions,
    ImageDimensionTable,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    torchvision_source,
)
from stochaflow.data.artifact_store import canonical_json_bytes
from stochaflow.data.datasets import (
    CodedLabelSequence,
    CompactIndexSequence,
    ImageDatasetFactory,
    MultiResolutionDataset,
    SourceConcatDataset,
)
from stochaflow.data.folder_sources import ImageFolderDataSource
from stochaflow.data.recipe_config import ResolutionBucketRecipeConfig
from stochaflow.data.samplers import (
    CompactSamplerIndex,
    MixtureBatchSampler,
    ResolutionBucketPolicy,
)
from stochaflow.data.torchvision_source import TorchvisionImageDataSource


class FixtureMetadataDataset(Dataset[Any]):
    """Map-style fixture exposing dimensions without sample reads."""

    def __init__(self, dimensions: Sequence[tuple[int, int]]) -> None:
        self.dimensions = ImageDimensionTable(dimensions)
        self.sample_reads = 0

    def __len__(self) -> int:
        return len(self.dimensions)

    def __getitem__(self, index: int) -> Image.Image:
        self.sample_reads += 1
        dimensions = self.dimensions[index]
        return Image.new("RGB", (dimensions.width, dimensions.height))

    def image_dimensions(self, index: int) -> ImageDimensions:
        """Return precomputed fixture dimensions."""

        return self.dimensions[index]


class FixtureManagedVisionDataset:
    """Minimal acquisition-compatible torchvision dataset."""

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
            data_root.mkdir(parents=True, exist_ok=True)
            (data_root / f"{marker}.bin").write_bytes(marker.encode("ascii"))
        if not (data_root / f"{marker}.bin").is_file():
            raise RuntimeError("fixture managed dataset is unavailable")

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        if index not in {0, 1}:
            raise IndexError(index)
        return Image.new("L", (28, 28)), index


class FixtureCompactBucketDataset:
    """Large bucket metadata fixture without Python integer lists."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.source_names = ("source",)
        self.bucket_names = ("square", "landscape")
        self.source_codes = CompactIndexSequence(
            (0 for _ in range(size)),
            maximum=0,
        )
        self.bucket_codes = CompactIndexSequence(
            (0 for _ in range(size)),
            maximum=1,
        )

    def __len__(self) -> int:
        return self.size


def write_image(path: Path, *, size: tuple[int, int]) -> None:
    """Write one deterministic RGB fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def source_context(
    cache_root: Path,
    *,
    policy: str = "ensure",
    verification: str = "full",
) -> DataSourceContext:
    """Build one typed source context for fixture materialization."""

    return DataSourceContext(
        cache_root=cache_root,
        policy=cast(Any, policy),
        verification=cast(Any, verification),
    )


def bucket_policy() -> ResolutionBucketPolicy:
    """Build the two-bucket policy shared by focused tests."""

    return ResolutionBucketPolicy(
        [
            ResolutionBucketRecipeConfig(
                name="square",
                height=16,
                width=16,
            ),
            ResolutionBucketRecipeConfig(
                name="landscape",
                height=16,
                width=32,
            ),
        ],
        base_bucket="square",
        dynamic_batch_size=False,
    )


def multi_resolution_dataset() -> MultiResolutionDataset:
    """Build a two-source fixture without reading any samples."""

    square = FixtureMetadataDataset([(16, 16)] * 6)
    landscape = FixtureMetadataDataset([(32, 16)] * 6)
    combined = SourceConcatDataset(
        [("square-source", square), ("landscape-source", landscape)]
    )
    dataset = MultiResolutionDataset(
        combined,
        bucket_policy(),
        role="eval",
        channels=3,
        normalize=False,
        random_horizontal_flip=False,
    )
    assert square.sample_reads == 0
    assert landscape.sample_reads == 0
    return dataset


def test_multi_resolution_construction_never_reads_or_decodes_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png", size=(31, 17))
    artifact = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    ).materialize(source_context(tmp_path / "cache"))
    native = ImageDatasetFactory().build(artifact).train
    combined = SourceConcatDataset([("referenced", native)])

    def reject_sample_read(self: object, index: int) -> None:
        del self, index
        raise AssertionError("dataset construction read an image sample")

    monkeypatch.setattr(type(native), "__getitem__", reject_sample_read)
    dataset = MultiResolutionDataset(
        combined,
        bucket_policy(),
        role="eval",
        channels=3,
        normalize=False,
        random_horizontal_flip=False,
    )

    assert dataset.bucket_ids[0] == "landscape"
    assert dataset.source_ids[0] == "referenced"


def test_reference_dimensions_are_canonical_and_integrity_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / "images"
    write_image(root / "sample.png", size=(31, 17))
    source = ImageFolderDataSource(
        {"root": str(root)},
        config_path="data.params.source",
    )
    artifact = source.materialize(source_context(tmp_path / "cache"))
    record = artifact.payload.train[0]
    assert (record.width, record.height) == (31, 17)

    manifest = json.loads(
        artifact.manifest_path.read_text(encoding="utf-8")
    )
    shard_path = artifact.index_root / manifest["inventory"]["shards"][0][
        "path"
    ]
    serialized = json.loads(shard_path.read_text(encoding="utf-8"))
    assert serialized["width"] == 31
    assert serialized["height"] == 17

    serialized["width"] = 32
    tampered = canonical_json_bytes(serialized)
    assert len(tampered) == shard_path.stat().st_size
    shard_path.write_bytes(tampered)

    with pytest.raises(ValueError, match="shard digest mismatch"):
        source.materialize(
            source_context(
                tmp_path / "cache",
                policy="require",
                verification="manifest",
            )
        )

    manifest["inventory"]["shards"][0]["sha256"] = hashlib.sha256(
        tampered
    ).hexdigest()
    artifact.manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="source digest is invalid"):
        source.materialize(
            source_context(
                tmp_path / "cache",
                policy="require",
                verification="manifest",
            )
        )


def test_managed_dimension_sidecar_is_hashed_in_manifest_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torchvision_source.datasets,
        "MNIST",
        FixtureManagedVisionDataset,
    )
    source = TorchvisionImageDataSource(
        {"dataset": "MNIST"},
        config_path="data.params.source",
    )
    artifact = source.materialize(source_context(tmp_path / "cache"))
    sidecar = artifact.artifact_root / "image_dimensions.json"
    encoded = sidecar.read_bytes()
    tampered = encoded.replace(b"[28,28]", b"[29,28]", 1)
    assert len(tampered) == len(encoded)
    assert tampered != encoded
    sidecar.write_bytes(tampered)

    with pytest.raises(
        ValueError,
        match="dimensions do not match the managed manifest",
    ):
        source.materialize(
            source_context(
                tmp_path / "cache",
                policy="require",
                verification="manifest",
            )
        )


def test_managed_dimension_sidecar_rejects_boolean_schema_version(
    tmp_path: Path,
) -> None:
    encoded = canonical_json_bytes(
        {
            "schema_version": True,
            "dataset": "mnist",
            "partitions": {
                "train": [[28, 28]],
                "test": [[28, 28]],
            },
        }
    )
    sidecar = tmp_path / "image_dimensions.json"
    sidecar.write_bytes(encoded)

    with pytest.raises(ValueError, match="incompatible"):
        torchvision_source.load_torchvision_dimensions(
            tmp_path,
            dataset_name="mnist",
            expected_record={
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
        )


def test_payload_metadata_is_immutable_and_validated(tmp_path: Path) -> None:
    source_dimensions = [(10, 20)]
    table = ImageDimensionTable(source_dimensions)
    source_dimensions[0] = (30, 40)
    serialized = table.to_pairs()
    serialized[0][0] = 99

    assert table[0] == ImageDimensions(10, 20)
    assert not hasattr(table, "widths")
    with pytest.raises(ValueError, match="32-bit"):
        ImageDimensionTable([(2**32, 1)])

    root = tmp_path / "images"
    root.mkdir()
    record = ImageFileRecord(
        tree="train",
        path="sample.png",
        size_bytes=1,
        sha256="a" * 64,
        width=10,
        height=20,
    )
    mutable_roots = {"train": root}
    mutable_records = [record]
    payload = ImageFolderArtifactPayload(
        roots=mutable_roots,
        train=cast(Any, mutable_records),
    )
    mutable_roots["other"] = root
    mutable_records.clear()

    assert set(payload.roots) == {"train"}
    assert payload.train == (record,)
    with pytest.raises(TypeError):
        cast(Any, payload.roots)["other"] = root
    with pytest.raises(TypeError, match="ImageFileRecord"):
        ImageFolderArtifactPayload(
            roots={"train": root},
            train=cast(Any, ["not-a-record"]),
        )


@pytest.mark.parametrize(
    ("weighted", "steps_per_epoch", "drop_last", "shuffle"),
    [
        (False, "auto", False, False),
        (False, "auto", True, True),
        (False, 7, False, True),
        (True, "auto", False, False),
        (True, 7, True, True),
    ],
)
def test_sampler_builds_only_active_compact_indexes(
    weighted: bool,
    steps_per_epoch: int | str,
    drop_last: bool,
    shuffle: bool,
) -> None:
    dataset = multi_resolution_dataset()
    source_weights = (
        {"square-source": 0.25, "landscape-source": 0.75}
        if weighted
        else None
    )
    sampler = MixtureBatchSampler(
        dataset,
        bucket_policy(),
        base_batch_size=4,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=17,
        source_weights=source_weights,
        steps_per_epoch=steps_per_epoch,
    )

    assert isinstance(dataset.source_codes, CompactIndexSequence)
    assert isinstance(dataset.bucket_codes, CompactIndexSequence)
    assert isinstance(dataset.source_ids, CodedLabelSequence)
    assert isinstance(dataset.bucket_ids, CodedLabelSequence)
    assert dataset.source_codes.storage_bytes == len(dataset)
    assert dataset.bucket_codes.storage_bytes == len(dataset)
    if weighted:
        assert sampler._bucket_indices is None
        assert sampler._source_bucket_indices is not None
        indexes = sampler._source_bucket_indices.values()
    else:
        assert sampler._bucket_indices is not None
        assert sampler._source_bucket_indices is None
        indexes = sampler._bucket_indices.values()
    assert all(isinstance(index, CompactSamplerIndex) for index in indexes)
    assert sum(index.storage_bytes for index in indexes) <= 4 * len(dataset)

    batches = list(sampler)
    assert len(batches) == len(sampler)
    for batch in batches:
        assert len({dataset.bucket_codes[index] for index in batch}) == 1
        if weighted:
            assert len({dataset.source_codes[index] for index in batch}) == 1


def test_unweighted_sampler_preserves_deterministic_batch_order() -> None:
    dataset = multi_resolution_dataset()
    sampler = MixtureBatchSampler(
        dataset,
        bucket_policy(),
        base_batch_size=4,
        drop_last=False,
        shuffle=True,
        seed=17,
        source_weights=None,
    )

    assert list(sampler) == [
        [0, 5, 1, 2],
        [11, 9, 6, 8],
        [10, 7],
        [3, 4],
    ]


def test_unweighted_sampler_does_not_expand_the_full_epoch() -> None:
    sample_count = 100_000
    dataset = FixtureCompactBucketDataset(sample_count)
    sampler = MixtureBatchSampler(
        dataset,
        bucket_policy(),
        base_batch_size=64,
        drop_last=False,
        shuffle=True,
        seed=17,
        source_weights=None,
    )

    tracemalloc.start()
    iterator = iter(sampler)
    first_batch = next(iterator)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(first_batch) == 64
    assert peak_bytes < sample_count * 12


@pytest.mark.parametrize(
    ("weight", "error_type"),
    [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-float("inf"), ValueError),
        (10**1000, ValueError),
        (True, TypeError),
        ("1.0", TypeError),
    ],
)
def test_sampler_rejects_invalid_weights_at_direct_boundary(
    weight: object,
    error_type: type[Exception],
) -> None:
    dataset = multi_resolution_dataset()

    with pytest.raises(error_type, match="finite positive"):
        MixtureBatchSampler(
            dataset,
            bucket_policy(),
            base_batch_size=2,
            drop_last=False,
            shuffle=False,
            seed=1,
            source_weights=cast(
                Any,
                {
                    "square-source": weight,
                    "landscape-source": 1.0,
                },
            ),
        )
