"""Tests for registered sampling artifact writers."""

from pathlib import Path

import pytest
import torch

from stochaflow.sampling import (
    SamplingArtifactContext,
    SamplingArtifactWriter,
    SamplingBatch,
    SamplingObservation,
    write_sampling_artifacts,
)
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES


def _context(tmp_path: Path, *, rank: int = 4, trajectory: bool = False):
    shape = (2, 1, 4, 4) if rank == 4 else (2, 5)
    samples = torch.zeros(shape)
    return SamplingArtifactContext(
        output_dir=tmp_path,
        batches=(
            SamplingBatch(
                samples=samples,
                num_samples=2,
                trajectory=(
                    (
                        SamplingObservation(0, 2, samples + 1, False, {}),
                        SamplingObservation(1, 0, samples, True, {}),
                    )
                    if trajectory
                    else None
                ),
            ),
        ),
        metadata={"domain": "physics"},
    )


def test_tensor_writer_supports_non_image_tensors(tmp_path: Path) -> None:
    artifacts = write_sampling_artifacts(
        [ComponentConfig(name="tensor")],
        _context(tmp_path, rank=2),
    )

    assert artifacts == {"samples": tmp_path / "samples.pt"}
    assert torch.load(artifacts["samples"], weights_only=False).shape == (2, 5)


def test_image_writer_writes_grid_and_trajectory(tmp_path: Path) -> None:
    artifacts = write_sampling_artifacts(
        [
            ComponentConfig(name="tensor"),
            ComponentConfig(
                name="image",
                params={"grid_nrow": 2, "gif_fps": 4, "denormalize": False},
            ),
        ],
        _context(tmp_path, trajectory=True),
    )

    assert set(artifacts) == {
        "samples",
        "trajectory",
        "image_grid",
        "trajectory_grid",
        "trajectory_gif",
    }
    assert all(path.is_file() for path in artifacts.values())


def test_image_writer_rejects_wrong_rank(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NCHW"):
        write_sampling_artifacts(
            [ComponentConfig(name="image")],
            _context(tmp_path, rank=2),
        )


@REGISTRIES.sampling_artifact_writers.register("stage2_duplicate_writer")
class DuplicateWriter(SamplingArtifactWriter):
    def __init__(self, *, key: str) -> None:
        self.key = key

    def write(self, context: SamplingArtifactContext):
        path = context.output_dir / f"{self.key}.txt"
        path.write_text("ok", encoding="utf-8")
        return {self.key: path}


@REGISTRIES.sampling_artifact_writers.register("stage2_missing_writer")
class MissingWriter(SamplingArtifactWriter):
    def write(self, context: SamplingArtifactContext):
        return {"missing": context.output_dir / "missing.txt"}


@REGISTRIES.sampling_artifact_writers.register("stage3_metadata_writer")
class MetadataWriter(SamplingArtifactWriter):
    def write(self, context: SamplingArtifactContext):
        path = context.output_dir / "metadata.txt"
        path.write_text(str(context.metadata["domain"]), encoding="utf-8")
        return {"metadata": path}


@REGISTRIES.sampling_artifact_writers.register("stage4_outside_writer")
class OutsideWriter(SamplingArtifactWriter):
    def write(self, context: SamplingArtifactContext):
        path = context.output_dir.parent / "outside.txt"
        path.write_text("outside", encoding="utf-8")
        return {"outside": path}


@REGISTRIES.sampling_artifact_writers.register("stage4_alias_writer")
class AliasWriter(SamplingArtifactWriter):
    def write(self, context: SamplingArtifactContext):
        path = context.output_dir / "shared.txt"
        path.write_text("shared", encoding="utf-8")
        return {"first": path, "second": path}


def test_custom_writer_receives_sampling_output_metadata(tmp_path: Path) -> None:
    artifacts = write_sampling_artifacts(
        [ComponentConfig(name="stage3_metadata_writer")],
        _context(tmp_path),
    )

    assert artifacts["metadata"].read_text(encoding="utf-8") == "physics"


def test_writer_contract_rejects_duplicate_keys_and_missing_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="duplicate sampling artifact key"):
        write_sampling_artifacts(
            [
                ComponentConfig(
                    name="stage2_duplicate_writer",
                    params={"key": "same"},
                ),
                ComponentConfig(
                    name="stage2_duplicate_writer",
                    params={"key": "same"},
                ),
            ],
            _context(tmp_path),
        )
    with pytest.raises(FileNotFoundError, match="returned missing path"):
        write_sampling_artifacts(
            [ComponentConfig(name="stage2_missing_writer")],
            _context(tmp_path / "missing"),
        )


def test_writer_contract_rejects_outside_and_aliased_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside its output directory"):
        write_sampling_artifacts(
            [ComponentConfig(name="stage4_outside_writer")],
            _context(tmp_path / "outside-root"),
        )
    with pytest.raises(ValueError, match="duplicate artifact path"):
        write_sampling_artifacts(
            [ComponentConfig(name="stage4_alias_writer")],
            _context(tmp_path / "alias-root"),
        )
