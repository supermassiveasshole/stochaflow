"""Create deterministic tiny trajectory, observation, and reference arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stochaflow_physics_reconstruction.stochaflow_ext._alignment import (
    write_alignment,
)


def build_tiny_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return six small trajectories and one held-out degraded trajectory."""

    size = 8
    time_steps = 6
    coordinate = np.arange(size, dtype=np.float32) * (2.0 * np.pi / size)
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    trajectories = np.empty((6, time_steps, size, size), dtype=np.float32)
    for trajectory in range(6):
        phase = np.float32(trajectory * 0.17)
        for time in range(time_steps):
            t = np.float32(time * 0.11)
            trajectories[trajectory, time] = (
                np.sin(x + phase + t)
                + np.float32(0.5) * np.cos(2.0 * y - t)
                + np.float32(0.1) * np.sin(x + y + phase)
            )
    references = trajectories[5:6].copy()
    coarse = references.reshape(1, time_steps, size // 2, 2, size // 2, 2).mean(
        axis=(3, 5)
    )
    observations = np.repeat(np.repeat(coarse, 2, axis=2), 2, axis=3).astype(
        np.float32,
        copy=False,
    )
    return trajectories, observations, references


def write_tiny_data(output_dir: Path) -> dict[str, Path]:
    """Write deterministic `.npy` fixtures and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories, observations, references = build_tiny_arrays()
    paths = {
        "trajectories": output_dir / "trajectories.npy",
        "observations": output_dir / "observations.npy",
        "references": output_dir / "references.npy",
        "alignment": output_dir / "alignment.json",
    }
    np.save(paths["trajectories"], trajectories, allow_pickle=False)
    np.save(paths["observations"], observations, allow_pickle=False)
    np.save(paths["references"], references, allow_pickle=False)
    write_alignment(
        paths["alignment"],
        observation_path=paths["observations"],
        observation_range=(0, 1),
        observation_shape=observations.shape,
        reference_path=paths["references"],
        reference_range=(0, 1),
        reference_shape=references.shape,
        sample_count=observations.shape[0] * (observations.shape[1] - 2),
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tiny"),
        help="Target directory relative to the current working directory.",
    )
    args = parser.parse_args()
    paths = write_tiny_data(args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
