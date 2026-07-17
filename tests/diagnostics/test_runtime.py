"""State, RNG, and sampling runtime service tests."""

import pytest
import torch

from stochaflow.diffusion import DDPM, LinearBetaSchedule
from stochaflow.training.diagnostics.runtime import (
    EvaluationGuard,
    SeedPolicy,
    prepare_reference_images,
)
from stochaflow.training.ema import ExponentialMovingAverage

from .helpers import TinyDenoiser, trainer


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_evaluation_guard_restores_weights_mode_and_rng_on_success_and_error(
    device_name,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    model = DDPM(
        noise_schedule=LinearBetaSchedule(num_timesteps=2),
        model=TinyDenoiser(),
    ).to(device)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    runtime = trainer(model, ema=ema)
    runtime.device = device
    ema.to(device)
    parameters_before = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    model.train()
    torch.manual_seed(987)
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )

    with EvaluationGuard(runtime, seed=123, use_ema=True):
        assert not model.training
        torch.rand(4, device=device)

    assert model.training
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters_before[name])

    with pytest.raises(RuntimeError, match="guard failure"):
        with EvaluationGuard(runtime, seed=456, use_ema=True):
            torch.rand(4, device=device)
            raise RuntimeError("guard failure")

    assert model.training
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters_before[name])


def test_seed_policy_is_stable_and_uses_common_initial_noise() -> None:
    policy = SeedPolicy(123)

    first = policy.initial_noise(3, (1, 4, 4), torch.device("cpu"))
    second = policy.initial_noise(3, (1, 4, 4), torch.device("cpu"))

    assert torch.equal(first, second)
    assert policy.profile_seed("ddpm") == policy.profile_seed("ddpm")
    assert policy.profile_seed("ddpm") != policy.profile_seed("ddim")


def test_prepare_reference_images_expands_grayscale_and_normalizes() -> None:
    prepared = prepare_reference_images(
        torch.tensor([[[[-1.0, 1.0], [0.0, 0.5]]]])
    )

    assert prepared.shape == (1, 3, 2, 2)
    assert prepared.min() == 0.0
    assert prepared.max() == 1.0
