"""Shape tests for DDPM process outputs."""

import torch
import torch.nn as nn

from stochaflow.diffusion import DDPM, LinearDDPMScheduler
from stochaflow.scripts.ddpm_runner import sample_reverse_trajectory


class ToyDenoiser(nn.Module):
    """Minimal denoiser used to verify DDPM tensor plumbing."""

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        assert timesteps.shape == (xt.shape[0],)
        return torch.zeros_like(xt)


class LoudNoiseDDPM(DDPM):
    """DDPM variant with obvious reverse noise for t=0 masking tests."""

    def _sample_noise(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.full_like(reference, 999.0)


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


def test_ddpm_estimate_x0_clips_denoised_reconstruction() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)
    ddpm = DDPM(scheduler=scheduler, model=ToyDenoiser(), clip_denoised=True)
    xt = torch.full((2, 1, 8, 8), 10.0)
    timesteps = torch.tensor([1, 2], dtype=torch.long)

    x0_hat = ddpm._estimate_x0_from_epsilon(
        xt,
        timesteps,
        predicted_noise=torch.zeros_like(xt),
        clip_denoised=True,
    )

    assert x0_hat.shape == xt.shape
    assert torch.all(x0_hat <= 1.0)
    assert torch.all(x0_hat >= -1.0)


def test_ddpm_reverse_step_at_timestep_zero_masks_noise_term() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)
    ddpm = LoudNoiseDDPM(scheduler=scheduler, model=ToyDenoiser())
    xt = torch.randn(2, 1, 8, 8)
    timesteps = torch.zeros(2, dtype=torch.long)

    x_prev = ddpm.reverse_step(xt, timesteps)
    predicted_noise = ddpm._predict_noise(xt, timesteps)
    x0 = ddpm._estimate_x0_from_epsilon(
        xt,
        timesteps,
        predicted_noise=predicted_noise,
        clip_denoised=True,
    )
    posterior_mean_coef1 = scheduler.coefficients_at(
        "posterior_mean_coef1", timesteps, xt.size()
    )
    posterior_mean_coef2 = scheduler.coefficients_at(
        "posterior_mean_coef2", timesteps, xt.size()
    )
    expected = posterior_mean_coef1 * x0 + posterior_mean_coef2 * xt

    assert torch.allclose(x_prev, expected)


def test_ddpm_reverse_method_epsilon_can_sample() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)
    ddpm = DDPM(
        scheduler=scheduler,
        model=ToyDenoiser(),
        reverse_method="epsilon",
    )

    samples = ddpm.sample(torch.Size((2, 1, 8, 8)), device=torch.device("cpu"))

    assert samples.shape == (2, 1, 8, 8)


def test_ddpm_rejects_unknown_reverse_method() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)

    try:
        DDPM(scheduler=scheduler, model=ToyDenoiser(), reverse_method="missing")
    except ValueError as exc:
        assert "reverse_method" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown reverse_method")


def test_script_reverse_trajectory_captures_initial_intermediate_and_final() -> None:
    scheduler = LinearDDPMScheduler(num_timesteps=10)
    ddpm = DDPM(scheduler=scheduler, model=ToyDenoiser())
    sample_shape = torch.Size((2, 1, 8, 8))

    trajectory = sample_reverse_trajectory(
        ddpm,
        sample_shape,
        device=torch.device("cpu"),
        capture_every=3,
    )

    assert set(trajectory) == {9, 6, 3, 0}
    for snapshot in trajectory.values():
        assert snapshot.shape == sample_shape
        assert snapshot.device == torch.device("cpu")
