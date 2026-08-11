"""State, RNG, and sampling runtime service tests."""

import random
import threading

import numpy as np
import pytest
import torch
from torch import nn

from stochaflow.sampling import PredictionType, VarianceMode
from stochaflow.training import TrainingStrategy, TrainStepOutput
from stochaflow.training.diagnostics import runtime as diagnostic_runtime
from stochaflow.training.diagnostics.config import (
    SamplerProfileConfig,
    TrajectoryProviderConfig,
)
from stochaflow.training.diagnostics.runtime import (
    DiagnosticModelAccessCleanupError,
    GaussianTrainingRuntime,
    SamplerPool,
    SamplerRunner,
    SeedPolicy,
    TrainingDiagnosticModelAccess,
    gaussian_training_runtime,
    prepare_reference_images,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.seed import preserve_global_rng_state

from .helpers import TinyDenoiser, gaussian_system, trainer


def _model_access(runtime, *, ema=None) -> TrainingDiagnosticModelAccess:
    return TrainingDiagnosticModelAccess(
        device=runtime.device,
        model=runtime.model,
        ema=ema,
        managed_modules=tuple(
            (name, asset.module)
            for name, asset in runtime.managed_modules.items()
            if name != "primary_model"
        ),
    )


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

    @property
    def variance_mode(self) -> VarianceMode:
        return "fixed"

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
def test_model_access_restores_weights_mode_and_rng_on_success_and_error(
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

    access = _model_access(runtime, ema=ema)
    with access.evaluation(seed=123, prefer_ema=True):
        assert not model.inference_model.training
        assert torch.is_inference_mode_enabled()
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

    def fail_inside_guard() -> None:
        with access.evaluation(seed=456, prefer_ema=True):
            random.random()
            np.random.random()
            torch.rand(4, device=device)
            raise RuntimeError("guard failure")

    with pytest.raises(RuntimeError, match="guard failure"):
        fail_inside_guard()

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


def test_model_access_raw_selection_fallback_and_mixed_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = gaussian_system(TinyDenoiser(), num_timesteps=2)
    ema = ExponentialMovingAverage(assets.inference_model, decay=0.5)
    runtime = trainer(assets, ema=ema)
    access = _model_access(runtime, ema=ema)
    for hidden_name in (
        "model",
        "trainer",
        "optimizer",
        "scheduler",
        "checkpoint_manager",
        "ema",
    ):
        assert not hasattr(access, hidden_name)
    assets.objective.eval()
    raw_parameters = {
        name: value.detach().clone()
        for name, value in assets.inference_model.named_parameters()
    }

    def unexpected_ema_call(module: nn.Module) -> None:
        del module
        raise AssertionError("raw Diagnostic evaluation must not touch EMA")

    monkeypatch.setattr(ema, "store", unexpected_ema_call)
    monkeypatch.setattr(ema, "copy_to", unexpected_ema_call)
    monkeypatch.setattr(ema, "restore", unexpected_ema_call)
    draws: list[torch.Tensor] = []
    for _ in range(2):
        with access.evaluation(seed=123, prefer_ema=False):
            assert torch.is_inference_mode_enabled()
            assert not assets.inference_model.training
            assert not assets.objective.training
            draws.append(torch.rand(3))

    assert access.ema_available
    assert torch.equal(draws[0], draws[1])
    assert assets.inference_model.training
    assert not assets.objective.training
    for name, value in assets.inference_model.named_parameters():
        assert torch.equal(value, raw_parameters[name])

    no_ema_access = _model_access(trainer(assets))
    assert not no_ema_access.ema_available
    with no_ema_access.evaluation(seed=3, prefer_ema=True):
        assert torch.is_inference_mode_enabled()


@pytest.mark.parametrize("failure_stage", ["success", "body", "entry"])
def test_model_access_restores_nested_mixed_module_modes(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    nested = nn.Sequential(nn.ReLU(), nn.Linear(2, 2))
    model = nn.Sequential(nn.Linear(2, 2), nested)
    model.train()
    model[0].eval()
    nested[0].eval()
    original_modes = {
        name: module.training for name, module in model.named_modules()
    }
    access = TrainingDiagnosticModelAccess(
        device=torch.device("cpu"),
        model=model,
        ema=None,
        managed_modules=(("shared_nested", nested),),
    )
    entry_error = RuntimeError("injected nested mode entry failure")
    if failure_stage == "entry":
        original_train = model[0].train
        entry_failed = False

        def fail_first_eval(mode: bool = True) -> nn.Module:
            nonlocal entry_failed
            result = original_train(mode)
            if not mode and not entry_failed:
                entry_failed = True
                raise entry_error
            return result

        monkeypatch.setattr(model[0], "train", fail_first_eval)

    body_error = RuntimeError("injected nested mode body failure")
    if failure_stage == "success":
        with access.evaluation(seed=17, prefer_ema=False):
            assert all(not module.training for module in model.modules())
    elif failure_stage == "body":
        def fail_inside_evaluation() -> None:
            with access.evaluation(seed=17, prefer_ema=False):
                assert all(not module.training for module in model.modules())
                raise body_error

        with pytest.raises(RuntimeError) as caught:
            fail_inside_evaluation()
        assert caught.value is body_error
    else:
        with (
            pytest.raises(RuntimeError) as caught,
            access.evaluation(seed=17, prefer_ema=False),
        ):
            pytest.fail("entry failure must prevent the context body")
        assert caught.value is entry_error

    restored_modes = {
        name: module.training for name, module in model.named_modules()
    }
    assert restored_modes == original_modes


@pytest.mark.parametrize("failure_stage", ["success", "body", "entry"])
def test_model_access_restores_shared_descendant_modes(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    shared = nn.ReLU()
    left = nn.Sequential(shared)
    right = nn.Sequential(shared)
    model = nn.ModuleDict({"left": left, "right": right})
    model.train()
    left.eval()
    right.train()
    shared.eval()
    modules = (model, left, right, shared)
    original_modes = tuple(module.training for module in modules)
    access = TrainingDiagnosticModelAccess(
        device=torch.device("cpu"),
        model=model,
        ema=None,
        managed_modules=(),
    )
    entry_error = RuntimeError("injected shared mode entry failure")
    if failure_stage == "entry":
        original_train = right.train
        entry_failed = False

        def fail_right_eval(mode: bool = True) -> nn.Module:
            nonlocal entry_failed
            result = original_train(mode)
            if not mode and not entry_failed:
                entry_failed = True
                raise entry_error
            return result

        monkeypatch.setattr(right, "train", fail_right_eval)

    body_error = RuntimeError("injected shared mode body failure")
    if failure_stage == "success":
        with access.evaluation(seed=19, prefer_ema=False):
            assert all(not module.training for module in modules)
    elif failure_stage == "body":
        def fail_inside_evaluation() -> None:
            with access.evaluation(seed=19, prefer_ema=False):
                assert all(not module.training for module in modules)
                raise body_error

        with pytest.raises(RuntimeError) as caught:
            fail_inside_evaluation()
        assert caught.value is body_error
    else:
        with (
            pytest.raises(RuntimeError) as caught,
            access.evaluation(seed=19, prefer_ema=False),
        ):
            pytest.fail("entry failure must prevent the context body")
        assert caught.value is entry_error

    assert tuple(module.training for module in modules) == original_modes


def test_model_access_does_not_seed_non_target_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    non_target_seed = 999

    def broadcast_seed(seed: int) -> torch.Generator:
        nonlocal non_target_seed
        non_target_seed = seed
        return torch.random.default_generator.manual_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", broadcast_seed)
    access = TrainingDiagnosticModelAccess(
        device=torch.device("cpu"),
        model=nn.Linear(2, 2),
        ema=None,
        managed_modules=(),
    )

    with access.evaluation(seed=123, prefer_ema=False):
        torch.rand(1)

    assert non_target_seed == 999


def test_model_access_attempts_every_rng_restore_after_one_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access = TrainingDiagnosticModelAccess(
        device=torch.device("cpu"),
        model=nn.Linear(2, 2),
        ema=None,
        managed_modules=(),
    )
    random.seed(101)
    np.random.seed(202)
    torch.random.default_generator.manual_seed(303)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    original_setstate = random.setstate
    restore_error = RuntimeError("injected Python RNG restore failure")

    def fail_python_restore(state: object) -> None:
        del state
        raise restore_error

    def consume_diagnostic_rng() -> None:
        with access.evaluation(seed=31, prefer_ema=False):
            random.random()
            np.random.random()
            torch.rand(1)

    monkeypatch.setattr(random, "setstate", fail_python_restore)
    with pytest.raises(DiagnosticModelAccessCleanupError) as caught:
        consume_diagnostic_rng()
    original_setstate(python_state)

    assert caught.value.__cause__ is restore_error
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_rng_restore_never_masks_body_error_with_invalid_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_error = RuntimeError("diagnostic callback failed")
    object.__setattr__(body_error, "__notes__", "invalid notes storage")
    python_state = random.getstate()
    original_setstate = random.setstate

    def fail_python_restore(state: object) -> None:
        del state
        raise RuntimeError("injected Python RNG restore failure")

    def fail_inside_scope() -> None:
        with preserve_global_rng_state(torch.device("cpu")):
            raise body_error

    monkeypatch.setattr(random, "setstate", fail_python_restore)
    with pytest.raises(RuntimeError) as caught:
        fail_inside_scope()
    original_setstate(python_state)

    assert caught.value is body_error


def test_model_access_restores_raw_weights_when_ema_copy_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = gaussian_system(TinyDenoiser(), num_timesteps=2)
    ema = ExponentialMovingAverage(assets.inference_model, decay=0.5)
    with torch.no_grad():
        for parameter in assets.inference_model.parameters():
            parameter.add_(1.0)
    runtime = trainer(assets, ema=ema)
    access = _model_access(runtime, ema=ema)
    raw = {
        name: value.detach().clone()
        for name, value in assets.inference_model.named_parameters()
    }
    original_copy_to = ema.copy_to
    copy_error = RuntimeError("injected diagnostic EMA copy failure")

    def copy_to_and_fail(module: nn.Module) -> None:
        original_copy_to(module)
        raise copy_error

    monkeypatch.setattr(ema, "copy_to", copy_to_and_fail)

    with (
        pytest.raises(RuntimeError) as caught,
        access.evaluation(seed=7, prefer_ema=True),
    ):
        pass

    assert caught.value is copy_error
    for name, value in assets.inference_model.named_parameters():
        assert torch.equal(value, raw[name])
    assert assets.inference_model.training


def test_model_access_restores_all_state_when_module_eval_mutates_then_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = gaussian_system(TinyDenoiser(), num_timesteps=2)
    runtime = trainer(assets)
    access = _model_access(runtime)
    objective = assets.objective
    original_train = objective.train
    entry_error = RuntimeError("injected diagnostic mode entry failure")
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()

    def enter_eval_and_fail(mode: bool = True) -> nn.Module:
        result = original_train(mode)
        if not mode:
            raise entry_error
        return result

    monkeypatch.setattr(objective, "train", enter_eval_and_fail)

    with (
        pytest.raises(RuntimeError) as caught,
        access.evaluation(seed=9, prefer_ema=False),
    ):
        pass

    assert caught.value is entry_error
    assert assets.inference_model.training
    assert assets.process.training
    assert objective.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_model_access_cleanup_failure_is_fatal_and_attempts_every_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = gaussian_system(TinyDenoiser(), num_timesteps=2)
    ema = ExponentialMovingAverage(assets.inference_model, decay=0.5)
    runtime = trainer(assets, ema=ema)
    access = _model_access(runtime, ema=ema)
    model_train = assets.inference_model.train
    objective_train = assets.objective.train
    restore_calls: list[str] = []
    body_error = RuntimeError("diagnostic body failed")
    ema_restore = ema.restore
    torch.manual_seed(4321)
    rng_before = torch.random.get_rng_state().clone()

    def failing_model_restore(mode: bool = True) -> nn.Module:
        result = model_train(mode)
        if mode:
            restore_calls.append("model")
            raise RuntimeError("model mode restore failed")
        return result

    def failing_objective_restore(mode: bool = True) -> nn.Module:
        result = objective_train(mode)
        if mode:
            restore_calls.append("objective")
            raise RuntimeError("objective mode restore failed")
        return result

    def failing_ema_restore(module: nn.Module) -> None:
        ema_restore(module)
        restore_calls.append("ema")
        raise RuntimeError("EMA restore failed")

    monkeypatch.setattr(assets.inference_model, "train", failing_model_restore)
    monkeypatch.setattr(assets.objective, "train", failing_objective_restore)
    monkeypatch.setattr(ema, "restore", failing_ema_restore)

    with (
        pytest.raises(DiagnosticModelAccessCleanupError) as caught,
        access.evaluation(seed=11, prefer_ema=True),
    ):
        raise body_error

    assert caught.value.__cause__ is body_error
    assert restore_calls == ["objective", "model", "ema"]
    notes = "\n".join(caught.value.__notes__)
    assert "objective mode restore failed" in notes
    assert "model mode restore failed" in notes
    assert "EMA restore failed" in notes
    assert assets.inference_model.training
    assert assets.objective.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_model_access_serializes_threads_and_rejects_same_thread_reentry() -> None:
    assets = gaussian_system(TinyDenoiser(), num_timesteps=2)
    runtime = trainer(assets)
    access = _model_access(runtime)
    attempting = threading.Event()
    entered = threading.Event()

    def worker() -> None:
        attempting.set()
        with access.evaluation(seed=5, prefer_ema=False):
            entered.set()

    with access.evaluation(seed=5, prefer_ema=False):
        with (
            pytest.raises(RuntimeError, match="cannot be nested"),
            access.evaluation(seed=5, prefer_ema=False),
        ):
            pass
        thread = threading.Thread(target=worker)
        thread.start()
        assert attempting.wait(timeout=2.0)
        assert not entered.is_set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert entered.is_set()


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
        assets.process,
        strategy,
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
            assets.process,
            PredictionOnlyGaussianStrategy(),
        )
