"""Contract tests for the DDIM process skeleton."""

import inspect
from typing import cast

import pytest
import torch
import torch.nn as nn

from stochaflow.diffusion import (
    DDIM,
    DDPMEpsilonObjective,
    DDPMForwardOutput,
    LinearDDPMScheduler,
)
from stochaflow.training.losses import ddpm_epsilon_train_step
from stochaflow.utils.registry import DIFFUSION_REGISTRY


class ToyDenoiser(nn.Module):
    """Minimal denoiser used to verify shared training tensor plumbing."""

    def forward(self, xt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        assert timesteps.shape == (xt.shape[0],)
        return torch.zeros_like(xt)


def test_ddim_reuses_ddpm_forward_training_contract() -> None:
    ddim = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=4,
    )
    x0 = torch.randn(3, 1, 8, 8)
    timesteps = torch.tensor([1, 5, 10], dtype=torch.long)

    output = ddim(x0, timesteps=timesteps)

    assert isinstance(output, DDPMForwardOutput)
    assert output.xt.shape == x0.shape
    assert output.noise.shape == x0.shape
    assert output.predicted_noise.shape == x0.shape


def test_ddim_resolves_uniform_descending_sampling_timesteps() -> None:
    ddim = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=4,
    )

    timesteps = ddim.sampling_timesteps(device=torch.device("cpu"))

    assert torch.equal(timesteps, torch.tensor([10, 8, 5, 2, 0]))
    assert timesteps.dtype == torch.long


def test_ddim_one_step_schedule_contains_both_state_endpoints() -> None:
    ddim = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ToyDenoiser(),
        num_inference_steps=1,
    )

    timesteps = ddim.sampling_timesteps()

    assert torch.equal(timesteps, torch.tensor([10, 0]))


def test_ddim_accepts_an_explicit_non_uniform_sampling_schedule() -> None:
    ddim = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ToyDenoiser(),
    )

    timesteps = ddim.sampling_timesteps(timesteps=[10, 7, 2, 0])

    assert torch.equal(timesteps, torch.tensor([10, 7, 2, 0]))
    assert timesteps.dtype == torch.long


def test_ddim_rejects_competing_sampling_schedule_inputs() -> None:
    ddim = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
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
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(exception, match=match):
        ddim.sampling_timesteps(timesteps=cast(list[int], timesteps))


def test_ddim_is_registered_and_uses_epsilon_training_contract() -> None:
    assert DIFFUSION_REGISTRY["ddim"] is DDIM

    diffusion = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
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
    assert train_output.loss.ndim == 0


@pytest.mark.parametrize("num_inference_steps", [0, 11])
def test_ddim_rejects_invalid_num_inference_steps(num_inference_steps: int) -> None:
    with pytest.raises(ValueError, match="num_inference_steps"):
        DDIM(
            scheduler=LinearDDPMScheduler(num_timesteps=10),
            model=ToyDenoiser(),
            num_inference_steps=num_inference_steps,
        )


@pytest.mark.parametrize("eta", [-0.1, 1.1])
def test_ddim_rejects_eta_outside_the_standard_range(eta: float) -> None:
    with pytest.raises(ValueError, match="eta"):
        DDIM(
            scheduler=LinearDDPMScheduler(num_timesteps=10),
            model=ToyDenoiser(),
            eta=eta,
        )


def test_ddim_reverse_step_requires_a_previous_timestep() -> None:
    parameter = inspect.signature(DDIM.reverse_step).parameters["previous_timesteps"]

    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_ddim_sampling_fails_explicitly_until_reverse_equation_is_added() -> None:
    ddim = DDIM(
        scheduler=LinearDDPMScheduler(num_timesteps=10),
        model=ToyDenoiser(),
    )

    with pytest.raises(NotImplementedError, match="DDIM sampling"):
        ddim.sample(torch.Size((2, 1, 8, 8)))
