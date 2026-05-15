"""Shape tests for DDPM process outputs."""

import torch
import torch.nn as nn

from stochaflow.diffusion import DDPM, LinearDDPMScheduler


class ToyDenoiser(nn.Module):
    """Minimal denoiser used to verify DDPM tensor plumbing."""

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        assert timesteps.shape == (xt.shape[0],)
        return torch.zeros_like(xt)


def test_ddpm_forward_returns_predicted_noise() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)
    ddpm = DDPM(scheduler=scheduler, model=ToyDenoiser())
    x0 = torch.randn(4, 1, 8, 8)
    timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    output = ddpm(x0, timesteps=timesteps)

    assert output.timesteps.shape == (4,)
    assert output.xt.shape == x0.shape
    assert output.noise.shape == x0.shape
    assert output.predicted_noise.shape == x0.shape


def test_ddpm_sample_trajectory_captures_initial_intermediate_and_final() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)
    ddpm = DDPM(scheduler=scheduler, model=ToyDenoiser())
    sample_shape = torch.Size((2, 1, 8, 8))

    trajectory = ddpm.sample_trajectory(
        sample_shape,
        device=torch.device("cpu"),
        capture_every=3,
    )

    assert set(trajectory) == {9, 6, 3, 0}
    for snapshot in trajectory.values():
        assert snapshot.shape == sample_shape
        assert snapshot.device == torch.device("cpu")
