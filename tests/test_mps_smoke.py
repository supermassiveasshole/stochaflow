"""Minimal end-to-end coverage for the native MPS execution path."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import (
    DDIMSampler,
    DDPMAncestralSampler,
    GaussianModelDynamics,
    Sampler,
)
from stochaflow.training import (
    GaussianDenoisingTrainingStrategy,
    MSEObjective,
    Trainer,
    TrainingPlan,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)


class _TinyGaussianModel(nn.Module):
    def __init__(self, initial_scale: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        return self.scale * state


def _process() -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 4,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )


def test_default_discrete_gaussian_process_migrates_to_mps_in_float32() -> None:
    process = _process()

    assert process.marginal_signal_t.dtype == torch.float32

    process.to("mps")

    assert all(buffer.device.type == "mps" for buffer in process.buffers())
    assert all(buffer.dtype == torch.float32 for buffer in process.buffers())


def test_trainer_runs_forward_backward_and_optimizer_step_on_mps() -> None:
    process = _process()
    model = _TinyGaussianModel()
    objective = MSEObjective()
    strategy = GaussianDenoisingTrainingStrategy(model, process, objective)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        plan=TrainingPlan(
            strategy=strategy,
            primary_model=model,
            process=process,
            objective=objective,
        ),
        optimizer=optimizer,
        device="mps",
    )
    torch.mps.manual_seed(0)

    history = trainer.fit(
        [torch.zeros(2, 1, 2, 2)],
        num_epochs=1,
        show_progress=False,
        track_best=False,
    )
    torch.mps.synchronize()

    assert trainer.global_step == 1
    assert history[0]["num_batches"] == 1
    assert model.scale.device.type == "mps"
    assert model.scale.detach().cpu().item() != 0.0


@pytest.mark.parametrize(
    ("sampler_factory", "expected_steps"),
    [
        (DDPMAncestralSampler, 4),
        (lambda: DDIMSampler(num_inference_steps=2), 2),
    ],
    ids=["ddpm", "ddim"],
)
def test_gaussian_samplers_run_complete_paths_on_mps(
    sampler_factory: Callable[[], Sampler],
    expected_steps: int,
) -> None:
    device = torch.device("mps")
    process = _process().to(device)
    model = _TinyGaussianModel(initial_scale=0.05).to(device)
    dynamics = GaussianModelDynamics(
        process,
        model,
        clip_denoised=False,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    initial_state = process.sample_terminal_prior(
        (2, 1, 2, 2),
        device=device,
        generator=generator,
    )

    result = sampler_factory().sample(
        dynamics,
        initial_state,
        generator=generator,
    )
    torch.mps.synchronize()
    final_state = result.final_state.detach().cpu()

    assert result.num_steps == expected_steps
    assert result.final_state.device.type == "mps"
    assert result.final_state.dtype == torch.float32
    assert final_state.shape == initial_state.shape
    assert torch.isfinite(final_state).all()
