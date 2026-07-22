"""Convert sparse PhysicsNeMo input into aligned mmap-ready Stochaflow data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F

from stochaflow_physics_reconstruction.stochaflow_ext._alignment import (
    write_alignment,
)


def _trajectory_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 4:
        raise ValueError("reference must have shape [trajectory, time, height, width]")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("reference must be floating-point")
    return value


def _training_stats(reference: np.ndarray, stop: int) -> tuple[float, float]:
    count = 0
    total = 0.0
    squared = 0.0
    for index in range(stop):
        values = np.asarray(reference[index], dtype=np.float64)
        count += values.size
        total += float(values.sum(dtype=np.float64))
        squared += float(np.square(values).sum(dtype=np.float64))
    mean = total / count
    variance = max(squared / count - mean * mean, 0.0)
    scale = variance**0.5
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference training statistics are not finite and non-zero")
    return mean, scale


def _resize_and_smooth(
    sparse: np.ndarray,
    *,
    spatial_shape: tuple[int, int],
    smoothing_kernel: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.array(sparse, dtype=np.float32, copy=True))
    flat = tensor.reshape(-1, 1, *tensor.shape[-2:])
    if tuple(flat.shape[-2:]) != spatial_shape:
        flat = F.interpolate(
            flat,
            size=spatial_shape,
            mode="bicubic",
            align_corners=False,
        )
    if smoothing_kernel:
        if smoothing_kernel <= 0 or smoothing_kernel % 2 == 0:
            raise ValueError("smoothing_kernel must be zero or a positive odd integer")
        radius = smoothing_kernel // 2
        coordinate = torch.arange(-radius, radius + 1, dtype=flat.dtype)
        gaussian = torch.exp(-0.5 * (coordinate / smoothing_kernel).square())
        gaussian = gaussian / gaussian.sum()
        kernel = torch.outer(gaussian, gaussian).reshape(1, 1, smoothing_kernel, smoothing_kernel)
        flat = F.conv2d(F.pad(flat, (radius,) * 4, mode="circular"), kernel)
    return flat.reshape(*tensor.shape[:-2], *spatial_shape).numpy()


def prepare(
    *,
    reference_path: Path,
    sparse_path: Path,
    sparse_key: str,
    output_dir: Path,
    held_out_trajectories: int,
    smoothing_kernel: int,
) -> dict[str, Path]:
    """Prepare paired observation data and an audited alignment sidecar."""

    reference = _trajectory_array(reference_path)
    if held_out_trajectories <= 0 or held_out_trajectories >= reference.shape[0]:
        raise ValueError("held_out_trajectories must leave at least one training row")
    with np.load(sparse_path, allow_pickle=False) as archive:
        if sparse_key not in archive:
            raise KeyError(f"sparse archive has no key {sparse_key!r}")
        sparse = np.asarray(archive[sparse_key])
    if sparse.ndim != 4 or not np.issubdtype(sparse.dtype, np.floating):
        raise ValueError("sparse data must be a floating [trajectory, time, H, W] array")
    selected = sparse[-held_out_trajectories:]
    reference_selected = reference[-held_out_trajectories:]
    if selected.shape[:2] != reference_selected.shape[:2]:
        raise ValueError("sparse and reference trajectory/time axes do not align")
    processed = _resize_and_smooth(
        selected,
        spatial_shape=(reference.shape[2], reference.shape[3]),
        smoothing_kernel=smoothing_kernel,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = output_dir / "kolmogorov_observations.npy"
    temporary = output_dir / f".{observation_path.name}.{uuid4().hex}.tmp"
    try:
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=processed.shape,
        )
        mapped[:] = processed
        mapped.flush()
        del mapped
        os.replace(temporary, observation_path)
    finally:
        temporary.unlink(missing_ok=True)
    reference_range = (
        reference.shape[0] - held_out_trajectories,
        reference.shape[0],
    )
    observation_range = (0, held_out_trajectories)
    sample_count = held_out_trajectories * (reference.shape[1] - 2)
    alignment_path = output_dir / "kolmogorov-alignment.json"
    write_alignment(
        alignment_path,
        observation_path=observation_path,
        observation_range=observation_range,
        observation_shape=processed.shape,
        reference_path=reference_path,
        reference_range=reference_range,
        reference_shape=reference.shape,
        sample_count=sample_count,
    )
    mean, scale = _training_stats(reference, reference_range[0])
    stats_path = output_dir / "kolmogorov-stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "normalization_mean": mean,
                "normalization_scale": scale,
                "training_trajectories": [0, reference_range[0]],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "observations": observation_path,
        "alignment": alignment_path,
        "statistics": stats_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--sparse-key", default="u3232")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--held-out-trajectories", type=int, default=4)
    parser.add_argument("--smoothing-kernel", type=int, default=7)
    args = parser.parse_args()
    outputs = prepare(
        reference_path=args.reference,
        sparse_path=args.sparse,
        sparse_key=args.sparse_key,
        output_dir=args.output_dir,
        held_out_trajectories=args.held_out_trajectories,
        smoothing_kernel=args.smoothing_kernel,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
