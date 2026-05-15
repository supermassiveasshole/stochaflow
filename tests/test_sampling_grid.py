import torch

from stochaflow.sampling import (
    denormalize_samples,
    save_image_grid,
    save_trajectory_grid,
)


def test_denormalize_samples_maps_minus_one_one_to_zero_one() -> None:
    samples = torch.tensor([-1.0, 0.0, 1.0])

    denormalized = denormalize_samples(samples)

    assert torch.allclose(denormalized, torch.tensor([0.0, 0.5, 1.0]))


def test_save_image_grid_writes_png(tmp_path) -> None:
    samples = torch.randn(4, 1, 8, 8)
    output_path = tmp_path / "grid.png"

    saved_path = save_image_grid(samples, output_path, nrow=2)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_trajectory_grid_writes_png(tmp_path) -> None:
    trajectory = {
        9: torch.randn(2, 1, 8, 8),
        5: torch.randn(2, 1, 8, 8),
        0: torch.randn(2, 1, 8, 8),
    }
    output_path = tmp_path / "trajectory.png"

    saved_path = save_trajectory_grid(trajectory, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
