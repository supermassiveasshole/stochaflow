"""Utilities for dumping generated sample grids."""

from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def denormalize_samples(samples: torch.Tensor) -> torch.Tensor:
    """Map samples from the common DDPM ``[-1, 1]`` image range to ``[0, 1]``."""

    return (samples.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_image_grid(
    samples: torch.Tensor,
    path: str | Path,
    *,
    nrow: int = 8,
    denormalize: bool = True,
) -> Path:
    """Save a batch of image-like samples as a grid.

    The function only owns artifact formatting. It does not run the diffusion
    process; callers should pass samples already produced by a model-specific
    reverse/sampling API.
    """

    if samples.ndim != 4:
        raise ValueError(
            "save_image_grid expects image batches shaped as (batch, channels, height, width)"
        )
    if nrow <= 0:
        raise ValueError("nrow must be positive")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_batch = samples.detach().cpu()
    if denormalize:
        image_batch = denormalize_samples(image_batch)

    grid = make_grid(image_batch, nrow=nrow)
    save_image(grid, output_path)

    return output_path


def save_trajectory_grid(
    trajectory: Mapping[int, torch.Tensor],
    path: str | Path,
    *,
    denormalize: bool = True,
) -> Path:
    """Save reverse-process snapshots as a time-by-sample image grid.

    Rows are ordered from high-noise timesteps to low-noise timesteps. Columns
    correspond to samples from the same batch, so each column shows one sample's
    denoising path.
    """

    if not trajectory:
        raise ValueError("trajectory must contain at least one timestep snapshot")

    ordered_timesteps = sorted(trajectory, reverse=True)
    snapshots = [trajectory[timestep] for timestep in ordered_timesteps]
    first_shape = snapshots[0].shape
    if len(first_shape) != 4:
        raise ValueError(
            "save_trajectory_grid expects snapshots shaped as "
            "(batch, channels, height, width)"
        )
    for snapshot in snapshots:
        if snapshot.shape != first_shape:
            raise ValueError("all trajectory snapshots must share the same shape")

    image_batch = torch.cat([snapshot.detach().cpu() for snapshot in snapshots], dim=0)
    if denormalize:
        image_batch = denormalize_samples(image_batch)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_grid(image_batch, nrow=first_shape[0])
    save_image(grid, output_path)

    return output_path


def save_trajectory_gif(
    trajectory: Mapping[int, torch.Tensor],
    path: str | Path,
    *,
    nrow: int = 8,
    fps: int = 8,
    denormalize: bool = True,
) -> Path:
    """Save ordered reverse-process snapshots as one looping grid GIF."""

    if not trajectory:
        raise ValueError("trajectory must contain at least one timestep snapshot")
    if nrow <= 0:
        raise ValueError("nrow must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    frames: list[Image.Image] = []
    for timestep in sorted(trajectory, reverse=True):
        snapshot = trajectory[timestep]
        if snapshot.ndim != 4:
            raise ValueError(
                "save_trajectory_gif expects snapshots shaped as "
                "(batch, channels, height, width)"
            )
        image_batch = snapshot.detach().cpu()
        if denormalize:
            image_batch = denormalize_samples(image_batch)
        grid = make_grid(image_batch, nrow=nrow).clamp(0.0, 1.0)
        array = (
            grid.mul(255)
            .round()
            .to(dtype=torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        if array.shape[2] == 1:
            array = array[:, :, 0]
        frames.append(Image.fromarray(array))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path
