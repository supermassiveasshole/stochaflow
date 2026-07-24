"""State, RNG, and sampling runtime service tests."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from stochaflow.sampling import PredictionType
from stochaflow.training import TrainStepOutput, TrainingStrategy
from stochaflow.training.diagnostics import runtime as diagnostic_runtime
from stochaflow.training.diagnostics.runtime import (
    EvaluationGuard,
    GaussianTrainingRuntime,
    SamplerPool,
    SamplerRunner,
    SeedPolicy,
    gaussian_training_runtime,
    prepare_reference_images,
)
from stochaflow.training.diagnostics.config import (
    SamplerProfileConfig,
    TrajectoryProviderConfig,
)
from stochaflow.training.ema import ExponentialMovingAverage

from .helpers import TinyDenoiser, gaussian_system, trainer


class MappingSignatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        self.calls += 1
        return torch.zeros_like(inputs["state"]) + self.offset


class MappingGaussianStrategy(TrainingStrategy):
    def __init__(self, model: MappingSignatureModel) -> None:
        self.model = model

    @property
    def prediction_type(self) -> PredictionType:
        return "epsilon"

    def predict_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        return self.model({"state": state, "time": model_time})

    def training_step(self, batch) -> TrainStepOutput:
        del batch
        return TrainStepOutput(self.model.offset.square())


class PredictionOnlyGaussianStrategy(TrainingStrategy):
    @property
    def prediction_type(self) -> PredictionType:
        return "epsilon"

    def training_step(self, batch) -> TrainStepOutput:
        del batch
        return TrainStepOutput(torch.zeros((), requires_grad=True))


@pytest.mark.parametrize("device_name", ["cpu", "cuda", "mps"])
def test_evaluation_guard_restores_weights_mode_and_rng_on_success_and_error(
    device_name,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device(device_name)
    model = gaussian_system(TinyDenoiser(), num_timesteps=2).to(device)
    ema = ExponentialMovingAverage(model.inference_model, decay=0.5)
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
    random.seed(789)
    np.random.seed(876)
    torch.manual_seed(987)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    )
    mps_rng_before = (
        torch.mps.get_rng_state().clone() if device.type == "mps" else None
    )

    with EvaluationGuard(runtime, seed=123, use_ema=True):
        assert not model.inference_model.training
        random.random()
        np.random.random()
        torch.rand(4, device=device)

    assert model.training
    assert random.getstate() == python_rng_before
    np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    if mps_rng_before is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_rng_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters_before[name])

    with pytest.raises(RuntimeError, match="guard failure"):
        with EvaluationGuard(runtime, seed=456, use_ema=True):
            random.random()
            np.random.random()
            torch.rand(4, device=device)
            raise RuntimeError("guard failure")

    assert model.training
    assert random.getstate() == python_rng_before
    np.testing.assert_equal(np.random.get_state(), numpy_rng_before)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng_before)
    if mps_rng_before is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_rng_before)
    for name, value in model.named_parameters():
        assert torch.equal(value, parameters_before[name])


def test_seed_policy_is_stable_and_uses_common_initial_noise() -> None:
    policy = SeedPolicy(123)

    first = policy.initial_noise(3, (1, 4, 4), torch.device("cpu"))
    second = policy.initial_noise(3, (1, 4, 4), torch.device("cpu"))

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    first_outer_state = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )
    with policy.fork_rng(torch.device("cpu")):
        first_draws = (random.random(), np.random.random(), torch.rand(()).item())
    assert random.getstate() == first_outer_state[0]
    np.testing.assert_equal(np.random.get_state(), first_outer_state[1])
    assert torch.equal(torch.random.get_rng_state(), first_outer_state[2])

    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    second_outer_state = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )
    with policy.fork_rng(torch.device("cpu")):
        second_draws = (random.random(), np.random.random(), torch.rand(()).item())

    assert torch.equal(first, second)
    assert policy.profile_seed("ddpm") == policy.profile_seed("ddpm")
    assert policy.profile_seed("ddpm") != policy.profile_seed("ddim")
    assert second_draws == first_draws
    assert random.getstate() == second_outer_state[0]
    np.testing.assert_equal(np.random.get_state(), second_outer_state[1])
    assert torch.equal(torch.random.get_rng_state(), second_outer_state[2])


def test_prepare_reference_images_expands_grayscale_and_normalizes() -> None:
    prepared = prepare_reference_images(
        torch.tensor([[[[-1.0, 1.0], [0.0, 0.5]]]])
    )

    assert prepared.shape == (1, 3, 2, 2)
    assert prepared.min() == 0.0
    assert prepared.max() == 1.0


def test_synchronize_dispatches_to_mps_without_requiring_mps(
    monkeypatch,
) -> None:
    calls = 0

    def synchronize() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(torch.mps, "synchronize", synchronize)

    diagnostic_runtime._synchronize(torch.device("mps"))

    assert calls == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_synchronize_waits_for_real_mps_work() -> None:
    value = torch.ones(4, device="mps").square()

    diagnostic_runtime._synchronize(value.device)

    assert value.cpu().tolist() == [1.0, 1.0, 1.0, 1.0]


def test_diagnostic_sampler_rejects_nonterminal_start() -> None:
    profile = SamplerProfileConfig(
        id="partial",
        name="ddpm",
        params={"start_time": 2},
        trajectory=TrajectoryProviderConfig(),
    )
    system = gaussian_system(num_timesteps=4)
    runtime = GaussianTrainingRuntime(
        system.process,
        system.prediction_type,
        system.strategy.predict_gaussian_model,
    )

    pool = SamplerPool(
        runtime,
        [profile],
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="start at process terminal time"):
        SamplerRunner(batch_size=1).run(
            pool.get("partial"),
            profile,
            torch.randn(1, 1, 4, 4),
        )


def test_diagnostic_sampler_rejects_nonclean_end() -> None:
    profile = SamplerProfileConfig(
        id="partial",
        name="ddpm",
        params={"end_time": 2},
        trajectory=TrajectoryProviderConfig(),
    )
    system = gaussian_system(num_timesteps=4)
    runtime = GaussianTrainingRuntime(
        system.process,
        system.prediction_type,
        system.strategy.predict_gaussian_model,
    )

    pool = SamplerPool(
        runtime,
        [profile],
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="end at process clean time"):
        SamplerRunner(batch_size=1).run(
            pool.get("partial"),
            profile,
            torch.randn(1, 1, 4, 4),
        )


def test_gaussian_diagnostic_uses_strategy_model_adapter() -> None:
    assets = gaussian_system(num_timesteps=2)
    model = MappingSignatureModel()
    strategy = MappingGaussianStrategy(model)
    resolved = gaussian_training_runtime(
        SimpleNamespace(model=model, process=assets.process, strategy=strategy)
    )
    profile = SamplerProfileConfig(
        id="adapted",
        name="ddpm",
        params={},
        trajectory=TrajectoryProviderConfig(),
    )
    pool = SamplerPool(resolved, [profile], device=torch.device("cpu"))

    SamplerRunner(batch_size=1).run(
        pool.get("adapted"),
        profile,
        torch.randn(1, 1, 4, 4),
    )

    assert model.calls == assets.process.num_timesteps


def test_gaussian_diagnostic_rejects_prediction_type_only_strategy() -> None:
    assets = gaussian_system(num_timesteps=2)

    with pytest.raises(TypeError, match="GaussianDiagnosticSemantics"):
        gaussian_training_runtime(
            SimpleNamespace(
                model=assets.inference_model,
                process=assets.process,
                strategy=PredictionOnlyGaussianStrategy(),
            )
        )
