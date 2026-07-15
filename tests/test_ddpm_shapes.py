"""Shape tests for DDPM process outputs."""

import pytest
import torch
import torch.nn as nn

from stochaflow.diffusion import DDPM, LinearBetaSchedule


class ToyDenoiser(nn.Module):
    """Minimal denoiser used to verify DDPM tensor plumbing."""

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        assert timesteps.shape == (xt.shape[0],)
        return torch.zeros_like(xt)


class RecordingDenoiser(ToyDenoiser):
    """Denoiser that records the zero-based model conditioning indices."""

    def __init__(self) -> None:
        super().__init__()
        self.timesteps: torch.Tensor | None = None

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        self.timesteps = timesteps.detach().clone()
        return super().forward(xt, timesteps)


class LoudNoiseDDPM(DDPM):
    """DDPM variant with obvious reverse noise for the final-step mask test."""

    def _sample_noise(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.full_like(reference, 999.0)


class RecordingReverseDDPM(DDPM):
    """DDPM variant that records public source-state times."""

    def __init__(self, noise_schedule: LinearBetaSchedule, model: nn.Module) -> None:
        super().__init__(noise_schedule=noise_schedule, model=model)
        self.reverse_step_indices: list[int] = []

    def reverse_step(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        clip_denoised: bool | None = None,
    ) -> torch.Tensor:
        self.reverse_step_indices.append(int(timesteps[0]))
        return super().reverse_step(xt, timesteps, clip_denoised=clip_denoised)


def test_ddpm_forward_returns_predicted_noise() -> None:
    noise_schedule = LinearBetaSchedule(num_timesteps=10)
    denoiser = RecordingDenoiser()
    ddpm = DDPM(noise_schedule=noise_schedule, model=denoiser)
    x0 = torch.randn(4, 1, 8, 8)
    timesteps = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    output = ddpm(x0, timesteps=timesteps)

    assert output.timesteps.shape == (4,)
    assert output.xt.shape == x0.shape
    assert output.noise.shape == x0.shape
    assert output.predicted_noise.shape == x0.shape
    assert torch.equal(output.timesteps, timesteps)
    assert denoiser.timesteps is not None
    assert torch.equal(denoiser.timesteps, torch.tensor([0, 1, 2, 3]))


def test_ddpm_add_noise_exposes_clean_state_time_zero() -> None:
    ddpm = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )
    x0 = torch.randn(2, 1, 8, 8)
    noise = torch.randn_like(x0)

    xt, returned_noise = ddpm.add_noise(
        x0,
        torch.zeros(2, dtype=torch.long),
        noise=noise,
    )

    assert torch.equal(xt, x0)
    assert torch.equal(returned_noise, noise)


def test_ddpm_estimate_x0_clips_denoised_reconstruction() -> None:
    noise_schedule = LinearBetaSchedule(num_timesteps=10)
    ddpm = DDPM(
        noise_schedule=noise_schedule,
        model=ToyDenoiser(),
        clip_denoised=True,
    )
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


def test_ddpm_reverse_step_at_state_time_one_masks_noise_term() -> None:
    noise_schedule = LinearBetaSchedule(num_timesteps=10)
    ddpm = LoudNoiseDDPM(noise_schedule=noise_schedule, model=ToyDenoiser())
    xt = torch.randn(2, 1, 8, 8)
    timesteps = torch.ones(2, dtype=torch.long)

    x_prev = ddpm.reverse_step(xt, timesteps)
    predicted_noise = ddpm._predict_noise(xt, timesteps)
    x0 = ddpm._estimate_x0_from_epsilon(
        xt,
        timesteps,
        predicted_noise=predicted_noise,
        clip_denoised=True,
    )
    posterior_mean_coef1 = ddpm._posterior_coefficients_at(
        ddpm.posterior_mean_coef1, timesteps, xt.size()
    )
    posterior_mean_coef2 = ddpm._posterior_coefficients_at(
        ddpm.posterior_mean_coef2, timesteps, xt.size()
    )
    expected = posterior_mean_coef1 * x0 + posterior_mean_coef2 * xt

    assert torch.allclose(x_prev, expected)


def test_ddpm_reverse_uses_all_transitions_to_reach_clean_time_zero() -> None:
    ddpm = RecordingReverseDDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    result = ddpm.reverse(torch.randn(2, 1, 8, 8), timestep_from=10)

    assert result.shape == (2, 1, 8, 8)
    assert ddpm.reverse_step_indices == list(range(10, 0, -1))


def test_ddpm_public_reverse_uses_zero_as_the_clean_endpoint() -> None:
    ddpm = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )
    x0 = torch.randn(2, 1, 8, 8)

    assert ddpm.reverse(x0, timestep_from=0) is x0


def test_ddpm_public_reverse_rejects_negative_state_times() -> None:
    ddpm = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(ValueError, match="0 <= timestep_to"):
        ddpm.reverse(torch.randn(2, 1, 8, 8), timestep_from=10, timestep_to=-1)


def test_ddpm_can_sample() -> None:
    noise_schedule = LinearBetaSchedule(num_timesteps=10)
    ddpm = DDPM(noise_schedule=noise_schedule, model=ToyDenoiser())

    samples = ddpm.sample(torch.Size((2, 1, 8, 8)), device=torch.device("cpu"))

    assert samples.shape == (2, 1, 8, 8)


def test_ddpm_reverse_step_clips_x0_before_posterior_mean() -> None:
    noise_schedule = LinearBetaSchedule(num_timesteps=10)
    ddpm = DDPM(
        noise_schedule=noise_schedule,
        model=ToyDenoiser(),
        clip_denoised=True,
    )
    xt = torch.full((2, 1, 8, 8), 10.0)
    timesteps = torch.ones(2, dtype=torch.long)

    x_prev = ddpm.reverse_step(xt, timesteps)
    x0_hat = ddpm._estimate_x0_from_epsilon(
        xt,
        timesteps,
        predicted_noise=torch.zeros_like(xt),
        clip_denoised=True,
    )
    expected = (
        ddpm._posterior_coefficients_at(
            ddpm.posterior_mean_coef1, timesteps, xt.size()
        )
        * x0_hat
        + ddpm._posterior_coefficients_at(
            ddpm.posterior_mean_coef2, timesteps, xt.size()
        )
        * xt
    )

    assert torch.allclose(x_prev, expected)


def test_ddpm_loads_legacy_scheduler_owned_buffers() -> None:
    original = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )
    posterior_names = {
        "sqrt_posterior_variance_t",
        "posterior_mean_coef1",
        "posterior_mean_coef2",
    }
    legacy_state: dict[str, torch.Tensor] = {}
    for name, value in original.state_dict().items():
        if name.startswith("noise_schedule."):
            legacy_state[f"scheduler.{name.removeprefix('noise_schedule.')}"] = value
        elif name in posterior_names:
            legacy_state[f"scheduler.{name}"] = value
        else:
            legacy_state[name] = value
    legacy_state["scheduler.sqrt_recip_alpha_t"] = torch.ones(10)

    restored = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )
    restored.load_state_dict(legacy_state)

    assert torch.equal(restored.noise_schedule.beta_t, original.noise_schedule.beta_t)
    assert torch.equal(restored.posterior_mean_coef1, original.posterior_mean_coef1)


def test_ddpm_trajectory_captures_initial_intermediate_and_final() -> None:
    noise_schedule = LinearBetaSchedule(num_timesteps=10)
    ddpm = DDPM(noise_schedule=noise_schedule, model=ToyDenoiser())
    sample_shape = torch.Size((2, 1, 8, 8))

    trace = ddpm.sample_trajectory(
        sample_shape,
        device=torch.device("cpu"),
        state_interval=3,
    )

    assert [frame.state_time for frame in trace.frames] == [10, 7, 4, 1, 0]
    for frame in trace.frames:
        assert frame.samples.shape == sample_shape
        assert frame.samples.device == torch.device("cpu")
