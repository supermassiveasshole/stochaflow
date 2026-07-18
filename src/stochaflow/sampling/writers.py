"""Registered writers for modality-specific sampling artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from stochaflow.sampling.grid import (
    save_image_grid,
    save_trajectory_gif,
    save_trajectory_grid,
)
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog


@dataclass(frozen=True, slots=True)
class SamplingBatch:
    """One generated batch and its optional reverse trajectory."""

    samples: Any
    trajectory: Mapping[int, Any] | None = None


@dataclass(frozen=True, slots=True)
class SamplingArtifactContext:
    """Generated batches and target directory supplied to artifact writers."""

    output_dir: Path
    batches: tuple[SamplingBatch, ...]

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("sampling artifact context requires generated batches")


class SamplingArtifactWriter(ABC):
    """Extension point for materializing generated samples."""

    @abstractmethod
    def write(self, context: SamplingArtifactContext) -> Mapping[str, Path]:
        """Write artifacts and return non-empty unique keys to existing paths."""


REGISTRIES.sampling_artifact_writers.require_base(SamplingArtifactWriter)


def _tensor_samples(batches: Sequence[SamplingBatch]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for index, batch in enumerate(batches):
        if not isinstance(batch.samples, torch.Tensor):
            raise TypeError(
                f"sampling batch {index} is not a tensor; tensor/image writer "
                "cannot serialize it"
            )
        if batch.samples.ndim == 0:
            raise ValueError("sampling tensors must include a batch dimension")
        parts.append(batch.samples.detach().cpu())
    try:
        return torch.cat(parts, dim=0)
    except RuntimeError as exc:
        raise ValueError("sampling batches must share compatible shapes") from exc


def _tensor_trajectory(
    batches: Sequence[SamplingBatch],
) -> dict[int, torch.Tensor] | None:
    if all(batch.trajectory is None for batch in batches):
        return None
    if any(batch.trajectory is None for batch in batches):
        raise ValueError("every sampling batch must provide the same trajectory")
    state_times = tuple(cast_trajectory(batches[0]).keys())
    parts: dict[int, list[torch.Tensor]] = {state_time: [] for state_time in state_times}
    for batch_index, batch in enumerate(batches):
        trajectory = cast_trajectory(batch)
        if tuple(trajectory.keys()) != state_times:
            raise ValueError("trajectory state times changed between sampling batches")
        for state_time, samples in trajectory.items():
            if not isinstance(samples, torch.Tensor):
                raise TypeError(
                    f"trajectory batch {batch_index} state {state_time} is not a tensor"
                )
            parts[state_time].append(samples.detach().cpu())
    try:
        return {
            state_time: torch.cat(state_parts, dim=0)
            for state_time, state_parts in parts.items()
        }
    except RuntimeError as exc:
        raise ValueError("trajectory batches must share compatible shapes") from exc


def cast_trajectory(batch: SamplingBatch) -> Mapping[int, Any]:
    """Return a trajectory after its non-null state is established."""

    assert batch.trajectory is not None
    return batch.trajectory


@REGISTRIES.sampling_artifact_writers.register("tensor")
class TensorSamplingArtifactWriter(SamplingArtifactWriter):
    """Save generated tensors and optional trajectories with ``torch.save``."""

    def write(self, context: SamplingArtifactContext) -> Mapping[str, Path]:
        samples_path = context.output_dir / "samples.pt"
        torch.save(_tensor_samples(context.batches), samples_path)
        artifacts = {"samples": samples_path}
        trajectory = _tensor_trajectory(context.batches)
        if trajectory is not None:
            trajectory_path = context.output_dir / "trajectory.pt"
            torch.save(trajectory, trajectory_path)
            artifacts["trajectory"] = trajectory_path
        return artifacts


@REGISTRIES.sampling_artifact_writers.register("image")
class ImageSamplingArtifactWriter(SamplingArtifactWriter):
    """Validate NCHW samples and write image grids and trajectory animation."""

    def __init__(
        self,
        *,
        grid_nrow: int = 4,
        gif_fps: int = 8,
        denormalize: bool = True,
    ) -> None:
        if grid_nrow <= 0:
            raise ValueError("image writer grid_nrow must be positive")
        if gif_fps <= 0:
            raise ValueError("image writer gif_fps must be positive")
        self.grid_nrow = grid_nrow
        self.gif_fps = gif_fps
        self.denormalize = denormalize

    @staticmethod
    def _validate_images(samples: torch.Tensor, *, label: str) -> None:
        if samples.ndim != 4:
            raise ValueError(f"image writer {label} must be NCHW tensors")
        if samples.shape[1] not in {1, 3}:
            raise ValueError(f"image writer {label} must have 1 or 3 channels")

    def write(self, context: SamplingArtifactContext) -> Mapping[str, Path]:
        samples = _tensor_samples(context.batches)
        self._validate_images(samples, label="samples")
        image_path = save_image_grid(
            samples,
            context.output_dir / "samples.png",
            nrow=self.grid_nrow,
            denormalize=self.denormalize,
        )
        artifacts = {"image_grid": image_path}
        trajectory = _tensor_trajectory(context.batches)
        if trajectory is None:
            return artifacts
        for state_time, snapshot in trajectory.items():
            self._validate_images(snapshot, label=f"trajectory[{state_time}]")
        grid_path = save_trajectory_grid(
            trajectory,
            context.output_dir / "trajectory.png",
            denormalize=self.denormalize,
        )
        gif_path = save_trajectory_gif(
            trajectory,
            context.output_dir / "trajectory.gif",
            nrow=self.grid_nrow,
            fps=self.gif_fps,
            denormalize=self.denormalize,
        )
        artifacts.update(
            {
                "trajectory_grid": grid_path,
                "trajectory_gif": gif_path,
            }
        )
        return artifacts


def write_sampling_artifacts(
    configs: Sequence[ComponentConfig],
    context: SamplingArtifactContext,
    *,
    registries: RegistryCatalog = REGISTRIES,
) -> dict[str, Path]:
    """Run declared writers and enforce their artifact-result contract."""

    context.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for config in configs:
        writer = registries.sampling_artifact_writers.create(
            config.name,
            **config.params,
        )
        if not isinstance(writer, SamplingArtifactWriter):
            raise TypeError(
                f"registered sampling artifact writer '{config.name}' did not "
                "produce SamplingArtifactWriter"
            )
        result = writer.write(context)
        if not isinstance(result, Mapping) or not result:
            raise ValueError(
                f"sampling artifact writer '{config.name}' must return artifacts"
            )
        for key, path_value in result.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"sampling artifact writer '{config.name}' returned an empty key"
                )
            if key in artifacts:
                raise ValueError(f"duplicate sampling artifact key '{key}'")
            path = Path(path_value)
            if not path.exists():
                raise FileNotFoundError(
                    f"sampling artifact writer '{config.name}' returned missing "
                    f"path: {path}"
                )
            artifacts[key] = path
    return artifacts


__all__ = [
    "ImageSamplingArtifactWriter",
    "SamplingArtifactContext",
    "SamplingArtifactWriter",
    "SamplingBatch",
    "TensorSamplingArtifactWriter",
    "write_sampling_artifacts",
]
