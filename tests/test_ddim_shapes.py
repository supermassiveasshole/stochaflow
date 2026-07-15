"""Contract and formula tests for DDIM."""

import inspect
from typing import cast

import pytest
import torch
import torch.nn as nn

from stochaflow.diffusion import (
    DDIM,
    DDPM,
    DDPMEpsilonObjective,
    DiffusionForwardOutput,
    LinearBetaSchedule,
)
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.registry import REGISTRIES


class ToyDenoiser(nn.Module):
    """Minimal denoiser used to verify shared training tensor plumbing."""

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        assert timesteps.shape == (xt.shape[0],)
        return torch.zeros_like(xt)


class RecordingZeroDenoiser(ToyDenoiser):
    """Zero-epsilon denoiser that records received model-conditioning indices."""

    def __init__(self) -> None:
        super().__init__()
        self.timesteps: torch.Tensor | None = None
        self.all_timesteps: list[torch.Tensor] = []

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        recorded_timesteps = timesteps.detach().clone()
        self.timesteps = recorded_timesteps
        self.all_timesteps.append(recorded_timesteps)
        return super().forward(xt, timesteps)


def test_ddim_reuses_ddpm_forward_training_contract() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=4,
    )
    x0 = torch.randn(3, 1, 8, 8)
    timesteps = torch.tensor([1, 5, 10], dtype=torch.long)

    output = ddim(x0, timesteps=timesteps)

    assert isinstance(output, DiffusionForwardOutput)
    assert output.xt.shape == x0.shape
    assert output.noise.shape == x0.shape
    assert output.predicted_noise.shape == x0.shape


def test_ddim_resolves_uniform_descending_sampling_timesteps() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=4,
    )

    timesteps = ddim.sampling_timesteps(device=torch.device("cpu"))

    assert torch.equal(timesteps, torch.tensor([10, 8, 5, 2, 0]))
    assert timesteps.dtype == torch.long


def test_ddim_one_step_schedule_contains_both_state_endpoints() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=1,
    )

    timesteps = ddim.sampling_timesteps()

    assert torch.equal(timesteps, torch.tensor([10, 0]))


def test_ddim_accepts_an_explicit_non_uniform_sampling_schedule() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    timesteps = ddim.sampling_timesteps(timesteps=[10, 7, 2, 0])

    assert torch.equal(timesteps, torch.tensor([10, 7, 2, 0]))
    assert timesteps.dtype == torch.long


def test_ddim_rejects_competing_sampling_schedule_inputs() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        ddim.sampling_timesteps(num_inference_steps=4, timesteps=[10, 7, 3, 0])


@pytest.mark.parametrize(
    ("timesteps", "exception", "match"),
    [
        ([], ValueError, "at least two"),
        ([10, 10, 0], ValueError, "strictly descending"),
        ([10, 11, 0], ValueError, "state times in \\[0, T\\]"),
        ([10, -1], ValueError, "state times in \\[0, T\\]"),
        ([10.0, 0.0], TypeError, "integer values"),
        ([10, 7, 2], ValueError, "start at T and end at 0"),
        ([9, 0], ValueError, "start at T and end at 0"),
    ],
)
def test_ddim_rejects_invalid_explicit_sampling_timesteps(
    timesteps: list[int] | list[float],
    exception: type[Exception],
    match: str,
) -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(exception, match=match):
        ddim.sampling_timesteps(timesteps=cast(list[int], timesteps))


def test_ddim_is_registered_and_uses_epsilon_training_contract() -> None:
    assert REGISTRIES.diffusions.resolve("ddim") is DDIM

    diffusion = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=5,
        eta=0.25,
    )
    train_output = ddpm_epsilon_train_step(
        diffusion,
        DDPMEpsilonObjective(),
        torch.randn(2, 1, 8, 8),
        torch.device("cpu"),
    )

    assert diffusion.num_inference_steps == 5
    assert diffusion.eta == 0.25
    assert not hasattr(diffusion, "posterior_mean_coef1")
    assert train_output.loss.ndim == 0


@pytest.mark.parametrize("num_inference_steps", [0, 11])
def test_ddim_rejects_invalid_num_inference_steps(num_inference_steps: int) -> None:
    with pytest.raises(ValueError, match="num_inference_steps"):
        DDIM(
            noise_schedule=LinearBetaSchedule(num_timesteps=10),
            model=ToyDenoiser(),
            num_inference_steps=num_inference_steps,
        )


@pytest.mark.parametrize("eta", [-0.1, 1.1])
def test_ddim_rejects_eta_outside_the_standard_range(eta: float) -> None:
    with pytest.raises(ValueError, match="eta"):
        DDIM(
            noise_schedule=LinearBetaSchedule(num_timesteps=10),
            model=ToyDenoiser(),
            eta=eta,
        )


def test_ddim_reverse_step_requires_a_previous_timestep() -> None:
    parameter = inspect.signature(DDIM.reverse_step).parameters["previous_timesteps"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_ddim_reverse_step_supports_non_uniform_batch_transitions() -> None:
    denoiser = RecordingZeroDenoiser()
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=denoiser,
        clip_denoised=False,
    )
    xt = torch.randn(2, 1, 4, 4)

    xs = ddim.reverse_step(
        xt,
        torch.tensor([10, 6]),
        previous_timesteps=torch.tensor([7, 2]),
        eta=0.0,
    )

    assert xs.shape == xt.shape
    assert denoiser.timesteps is not None
    assert torch.equal(denoiser.timesteps, torch.tensor([9, 5]))


def test_ddim_reverse_step_at_clean_target_returns_x0_hat() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        clip_denoised=False,
    )
    xt = torch.randn(2, 1, 4, 4)
    timesteps = torch.tensor([10, 5])
    expected = ddim._estimate_x0_from_epsilon(
        xt,
        timesteps,
        predicted_noise=torch.zeros_like(xt),
        clip_denoised=False,
    )

    xs = ddim.reverse_step(
        xt,
        timesteps,
        previous_timesteps=torch.zeros_like(timesteps),
        eta=0.0,
        clip_denoised=False,
    )

    assert torch.allclose(xs, expected)


def test_ddim_deterministic_step_does_not_advance_rng_state() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )
    xt = torch.randn(2, 1, 4, 4)
    timesteps = torch.tensor([10, 5])

    torch.manual_seed(123)
    rng_state_before = torch.random.get_rng_state()
    ddim.reverse_step(
        xt,
        timesteps,
        previous_timesteps=timesteps - 1,
        eta=0.0,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_state_before)


def test_ddim_eta_one_adjacent_step_matches_clipped_ddpm() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        eta=1.0,
        clip_denoised=True,
    )
    ddpm = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        clip_denoised=True,
    )
    xt = torch.full((2, 1, 4, 4), 10.0)
    timesteps = torch.tensor([10, 5])

    torch.manual_seed(456)
    xs_ddim = ddim.reverse_step(
        xt,
        timesteps,
        previous_timesteps=timesteps - 1,
        eta=1.0,
    )
    torch.manual_seed(456)
    xs_ddpm = ddpm.reverse_step(xt, timesteps)

    assert torch.allclose(xs_ddim, xs_ddpm)


def test_ddim_recomputes_direction_residual_after_clipping() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
        clip_denoised=True,
    )
    xt = torch.full((2, 1, 4, 4), 10.0)
    timesteps = torch.tensor([10, 5])
    previous_timesteps = torch.tensor([7, 2])
    x0_hat = ddim._estimate_x0_from_epsilon(
        xt,
        timesteps,
        predicted_noise=torch.zeros_like(xt),
        clip_denoised=True,
    )
    signal_scale_t, noise_scale_t = ddim.noise_schedule.marginal_scales(
        timesteps,
        xt.size(),
    )
    signal_scale_s, noise_scale_s = ddim.noise_schedule.marginal_scales(
        previous_timesteps,
        xt.size(),
    )
    corrected_eps = (xt - signal_scale_t * x0_hat) / noise_scale_t
    expected = signal_scale_s * x0_hat + noise_scale_s * corrected_eps

    xs = ddim.reverse_step(
        xt,
        timesteps,
        previous_timesteps=previous_timesteps,
        eta=0.0,
    )

    assert torch.allclose(xs, expected)


@pytest.mark.parametrize(
    ("eta", "exception"),
    [
        (-0.1, ValueError),
        (1.1, ValueError),
        (True, TypeError),
        ("invalid", TypeError),
    ],
)
def test_ddim_reverse_step_rejects_invalid_eta(
    eta: float | bool | str,
    exception: type[Exception],
) -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(exception, match="eta"):
        ddim.reverse_step(
            torch.randn(2, 1, 4, 4),
            torch.tensor([10, 5]),
            previous_timesteps=torch.tensor([7, 2]),
            eta=cast(float | None, eta),
        )


def test_ddim_reverse_step_rejects_non_descending_state_pairs() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(ValueError, match="smaller"):
        ddim.reverse_step(
            torch.randn(2, 1, 4, 4),
            torch.tensor([10, 5]),
            previous_timesteps=torch.tensor([10, 2]),
        )


def test_ddim_reverse_rejects_competing_schedule_inputs() -> None:
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        ddim.reverse(
            torch.randn(2, 1, 8, 8),
            num_inference_steps=4,
            timesteps=[10, 7, 2, 0],
        )


def test_ddim_sample_uses_k_transitions_and_returns_clean_shape() -> None:
    denoiser = RecordingZeroDenoiser()
    ddim = DDIM(
        noise_schedule=LinearBetaSchedule(num_timesteps=10),
        model=denoiser,
        num_inference_steps=4,
    )

    samples = ddim.sample(torch.Size((2, 1, 8, 8)))

    assert samples.shape == (2, 1, 8, 8)
    assert [int(timestep[0]) for timestep in denoiser.all_timesteps] == [9, 7, 4, 1]
