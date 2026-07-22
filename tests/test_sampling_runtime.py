"""Tests for SamplingBuilder orchestration and checkpoint-only sampling."""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import yaml

from stochaflow.processes import DiscreteGaussianProcess, Process
from stochaflow.sampling import (
    DDPMAncestralSampler,
    GaussianModelDynamics,
    GenerativeDynamics,
    Sampler,
    SamplerResult,
    SamplingBatch,
    SamplingBuilder,
    SamplingBuilderContext,
    SamplingObservation,
    SamplingObserver,
    SamplingOutput,
)
from stochaflow.sampling.runtime import run_sampling, validate_sampling_output
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION
from stochaflow.utils.config import load_config_dict
from stochaflow.utils.registry import REGISTRIES


class TinyDenoiser(nn.Module):
    """Parameter-bearing model with the built-in denoiser signature."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(state) + self.scale


REGISTRIES.models.add("stage3_tiny_model", TinyDenoiser)


class NoShapeBuilder(SamplingBuilder):
    calls = 0

    def run(self) -> SamplingOutput:
        type(self).calls += 1
        assert self.context.shape is None
        return SamplingOutput(
            (SamplingBatch(torch.tensor([[self.context.seed]], dtype=torch.float32)),),
            {"kind": "custom"},
        )


REGISTRIES.sampling_builders.add("stage3_no_shape", NoShapeBuilder)


@REGISTRIES.sampling_builders.register("stage3_direct_transform")
class DirectTransformBuilder(SamplingBuilder):
    """Direct model transform with no Process, Dynamics, or Sampler."""

    calls = 0

    def run(self) -> SamplingOutput:
        type(self).calls += 1
        if self.context.process is not None:
            raise TypeError("direct transform requires no Process")
        model = self.context.model_provider.get("raw")
        state = torch.ones(
            (self.context.num_samples, 1), device=self.context.device
        )
        times = torch.zeros(
            self.context.num_samples,
            device=self.context.device,
            dtype=torch.long,
        )
        transformed = model(state, times) + float(self.context.params["offset"])
        return SamplingOutput(
            (SamplingBatch(transformed.detach().cpu()),),
            {"kind": "direct_transform"},
        )


@REGISTRIES.processes.register("stage3_toy_flow")
class ToyFlowProcess(Process):
    """Test-only probability path for a second algorithm family."""

    rate: torch.Tensor

    def __init__(self, *, rate: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("rate", torch.tensor(float(rate)))


class ToyVectorFieldDynamics(GenerativeDynamics):
    """Test-only vector-field capability with no Gaussian API."""

    def __init__(self, process: ToyFlowProcess, predict: Any) -> None:
        self.process = process
        self._predict = predict

    def velocity(self, state: torch.Tensor, coordinate: float) -> torch.Tensor:
        times = torch.full(
            (state.shape[0],), coordinate, device=state.device, dtype=state.dtype
        )
        model_output = self._predict(state, times)
        if not isinstance(model_output, torch.Tensor):
            raise TypeError("toy vector field model must return a Tensor")
        return model_output + self.process.rate.to(state)


@REGISTRIES.samplers.register("stage3_toy_euler")
class ToyEulerSampler(Sampler):
    """Test-only solver for ToyVectorFieldDynamics."""

    def __init__(self, *, num_steps: int = 2) -> None:
        self.num_steps = num_steps

    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult:
        del generator
        if not isinstance(dynamics, ToyVectorFieldDynamics):
            raise TypeError("toy Euler sampler requires ToyVectorFieldDynamics")
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("toy Euler initial state must be a Tensor")
        state = initial_state
        if observer is not None:
            observer.observe(SamplingObservation(0, 1.0, state, False, {}))
        step_size = 1.0 / self.num_steps
        for step_index in range(1, self.num_steps + 1):
            coordinate = 1.0 - (step_index - 1) * step_size
            state = state - step_size * dynamics.velocity(state, coordinate)
            if observer is not None:
                observer.observe(
                    SamplingObservation(
                        step_index,
                        1.0 - step_index * step_size,
                        state,
                        step_index == self.num_steps,
                        {},
                    )
                )
        return SamplerResult(state, self.num_steps, {"family": "toy_flow"})


@REGISTRIES.sampling_builders.register("stage3_toy_flow")
class ToyFlowSamplingBuilder(SamplingBuilder):
    """Test-only task composition for the toy flow family."""

    def run(self) -> SamplingOutput:
        process = self.context.process
        if not isinstance(process, ToyFlowProcess):
            raise TypeError("toy flow builder requires ToyFlowProcess")
        sampler_config = self.context.params["sampler"]
        assert isinstance(sampler_config, dict)
        sampler = cast(
            Sampler,
            REGISTRIES.samplers.create(
                sampler_config["name"], **sampler_config.get("params", {})
            ),
        )
        model = self.context.model_provider.get("raw")
        dynamics = ToyVectorFieldDynamics(
            process,
            lambda state, time: model(state, time),
        )
        initial = torch.zeros(
            (self.context.num_samples, 1), device=self.context.device
        )
        result = sampler.sample(dynamics, initial)
        return SamplingOutput(
            (SamplingBatch(result.final_state.detach().cpu()),),
            {"family": "toy_flow", "solver": sampler_config["name"]},
        )


class BadResultSampler(Sampler):
    def sample(self, dynamics, initial_state, **kwargs):
        observer = kwargs.get("observer")
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    0, dynamics.process.terminal_time, initial_state, False, {}
                )
            )
            observer.observe(
                SamplingObservation(
                    1, dynamics.process.clean_time, initial_state, True, {}
                )
            )
        return initial_state


REGISTRIES.samplers.add("stage3_bad_result", BadResultSampler)


class WrongShapeSampler(Sampler):
    def sample(self, dynamics, initial_state, **kwargs):
        observer = kwargs.get("observer")
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    0, dynamics.process.terminal_time, initial_state, False, {}
                )
            )
            observer.observe(
                SamplingObservation(
                    1, dynamics.process.clean_time, initial_state, True, {}
                )
            )
        return SamplerResult(initial_state[:1], 1, {})


REGISTRIES.samplers.add("stage3_wrong_shape", WrongShapeSampler)


class WrongTrajectoryShapeSampler(Sampler):
    def sample(self, dynamics, initial_state, **kwargs):
        observer = kwargs.get("observer")
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    0, dynamics.process.terminal_time, initial_state, False, {}
                )
            )
            observer.observe(
                SamplingObservation(
                    1,
                    dynamics.process.clean_time,
                    initial_state[..., :1],
                    True,
                    {},
                )
            )
        return SamplerResult(initial_state, 1, {})


REGISTRIES.samplers.add(
    "stage3_wrong_trajectory_shape", WrongTrajectoryShapeSampler
)


class MalformedLifecycleSampler(Sampler):
    def __init__(self, *, mode: str) -> None:
        self.mode = mode

    def sample(self, dynamics, initial_state, **kwargs):
        observer = kwargs.get("observer")
        assert observer is not None
        observer.observe(
            SamplingObservation(
                0, dynamics.process.terminal_time, initial_state, False, {}
            )
        )
        if self.mode == "missing_final":
            return SamplerResult(initial_state, 0, {})
        if self.mode == "non_increasing":
            observer.observe(
                SamplingObservation(
                    0, dynamics.process.clean_time, initial_state, True, {}
                )
            )
        else:
            observer.observe(
                SamplingObservation(
                    1, dynamics.process.clean_time, initial_state, True, {}
                )
            )
            observer.observe(
                SamplingObservation(
                    2, dynamics.process.clean_time, initial_state, True, {}
                )
            )
        return SamplerResult(initial_state, 1, {})


REGISTRIES.samplers.add("stage3_malformed_lifecycle", MalformedLifecycleSampler)


def _raw_config(*, builder: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "experiment": {"name": "test", "output_dir": "unused", "seed": 7},
        "extensions": {"modules": []},
        "data": {"name": "image", "params": {}},
        "model": {"name": "stage3_tiny_model", "params": {}},
        "process": {
            "name": "discrete_gaussian",
            "params": {
                "schedule": {
                    "name": "linear_beta",
                    "params": {
                        "num_timesteps": 4,
                        "beta_start": 0.0001,
                        "beta_end": 0.02,
                    },
                }
            },
        },
        "training": {"name": "gaussian_denoising", "params": {}},
        "objective": {"name": "mse", "params": {}},
        "sampling": {
            "builder": builder,
            "shape": None,
            "num_samples": 3,
            "batch_size": 2,
            "seed": 11,
            "writers": [{"name": "tensor", "params": {}}],
        },
    }


def _checkpoint(
    path: Path,
    raw: dict[str, Any],
    *,
    version: int = CHECKPOINT_FORMAT_VERSION,
) -> Path:
    model = TinyDenoiser()
    payload: dict[str, Any] = {
        "format_version": version,
        "model_state_dict": model.state_dict(),
        "config": raw,
    }
    process_config = raw.get("process")
    if process_config is not None:
        assert isinstance(process_config, dict)
        process = cast(
            Process,
            REGISTRIES.processes.create(
                process_config["name"], **process_config["params"]
            ),
        )
        payload["process_state_dict"] = process.state_dict()
    torch.save(payload, path)
    return path


def test_custom_builder_runs_once_without_shape(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    NoShapeBuilder.calls = 0

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
        device_name="cpu",
    )

    assert NoShapeBuilder.calls == 1
    assert result.builder_name == "stage3_no_shape"
    assert result.metadata == {"kind": "custom"}
    assert torch.equal(torch.load(result.artifacts["samples"]), torch.tensor([[11.0]]))


def test_direct_transform_runs_without_process_or_sampler(tmp_path: Path) -> None:
    raw = _raw_config(
        builder={"name": "stage3_direct_transform", "params": {"offset": 3.0}}
    )
    raw.pop("process")
    checkpoint = _checkpoint(tmp_path / "direct.pt", raw)
    DirectTransformBuilder.calls = 0

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "direct-samples",
        device_name="cpu",
    )

    manifest = yaml.safe_load(result.artifacts["config"].read_text())
    assert DirectTransformBuilder.calls == 1
    assert result.metadata == {"kind": "direct_transform"}
    assert manifest["process"] is None
    assert torch.equal(
        torch.load(result.artifacts["samples"]),
        torch.full((3, 1), 3.0),
    )


def test_complete_external_sampling_config_can_omit_process(tmp_path: Path) -> None:
    raw = _raw_config(
        builder={"name": "stage3_direct_transform", "params": {"offset": 2.0}}
    )
    raw.pop("process")
    checkpoint = _checkpoint(tmp_path / "direct.pt", raw)
    config_path = tmp_path / "direct.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_sampling(
        checkpoint=checkpoint,
        config_path=config_path,
        output_dir=tmp_path / "direct-samples",
    )

    assert result.metadata == {"kind": "direct_transform"}


def test_standard_builder_rejects_missing_process_at_its_boundary(
    tmp_path: Path,
) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {
                "sampler": {"name": "ddim", "params": {"num_inference_steps": 2}}
            },
        }
    )
    raw["process"] = None
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "missing-process.pt", raw)

    with pytest.raises(TypeError, match="DiscreteGaussianDenoisingProcess"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_second_algorithm_family_runs_without_core_dispatch_changes(
    tmp_path: Path,
) -> None:
    raw = _raw_config(
        builder={
            "name": "stage3_toy_flow",
            "params": {
                "sampler": {
                    "name": "stage3_toy_euler",
                    "params": {"num_steps": 2},
                }
            },
        }
    )
    raw["process"] = {"name": "stage3_toy_flow", "params": {"rate": 2.0}}
    checkpoint = _checkpoint(tmp_path / "flow.pt", raw)

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "flow-samples",
        device_name="cpu",
    )

    assert result.metadata == {
        "family": "toy_flow",
        "solver": "stage3_toy_euler",
    }
    assert torch.equal(
        torch.load(result.artifacts["samples"]),
        torch.full((3, 1), -2.0),
    )


def test_family_samplers_reject_cross_family_dynamics() -> None:
    flow_process = ToyFlowProcess()
    flow_dynamics = ToyVectorFieldDynamics(
        flow_process, lambda state, time: torch.zeros_like(state)
    )
    gaussian_process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {"num_timesteps": 2},
        }
    )
    gaussian_dynamics = GaussianModelDynamics(
        gaussian_process,
        lambda state, time: torch.zeros_like(state),
    )

    with pytest.raises(TypeError, match="GaussianDenoisingDynamics"):
        DDPMAncestralSampler().sample(flow_dynamics, torch.zeros(1, 1))
    with pytest.raises(TypeError, match="ToyVectorFieldDynamics"):
        ToyEulerSampler().sample(gaussian_dynamics, torch.zeros(1, 1))


def test_standard_builder_batches_and_records_resolved_metadata(tmp_path: Path) -> None:
    builder = {
        "name": "standard_denoising",
        "params": {
            "weights": "raw",
            "prediction_type": "epsilon",
            "clip_denoised": False,
            "sampler": {
                "name": "ddim",
                "params": {"num_inference_steps": 2, "eta": 0.0},
            },
            "trajectory": {"enabled": True, "every_steps": 1},
        },
    }
    raw = _raw_config(builder=builder)
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
        device_name="cpu",
    )

    samples = torch.load(result.artifacts["samples"])
    trajectory = torch.load(result.artifacts["trajectory"])
    manifest = yaml.safe_load(result.artifacts["config"].read_text())
    assert samples.shape == (3, 2)
    assert trajectory["step_indices"] == [0, 1, 2]
    assert trajectory["coordinates"] == [4, 2, 0]
    assert trajectory["states"].shape == (3, 3, 2)
    assert manifest["metadata"]["sampler"]["name"] == "ddim"
    assert result.metadata["weights"] == "raw"


def test_standard_builder_rejects_invalid_sampler_result(tmp_path: Path) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {
                "sampler": {"name": "stage3_bad_result", "params": {}},
            },
        }
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(TypeError, match="must return SamplerResult"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_standard_builder_rejects_wrong_sampler_shape(tmp_path: Path) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {
                "sampler": {"name": "stage3_wrong_shape", "params": {}},
            },
        }
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="final state has shape"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_standard_builder_rejects_wrong_trajectory_shape(tmp_path: Path) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {
                "sampler": {
                    "name": "stage3_wrong_trajectory_shape",
                    "params": {},
                },
                "trajectory": {"enabled": True},
            },
        }
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="trajectory step 1 has shape"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("missing_final", "did not emit a final observation"),
        ("non_increasing", "step indices must increase"),
        ("duplicate_final", "after its final state"),
    ],
)
def test_standard_builder_validates_actual_sampler_lifecycle(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {
                "sampler": {
                    "name": "stage3_malformed_lifecycle",
                    "params": {"mode": mode},
                }
            },
        }
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match=message):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("prediction_type", "invalid", "prediction_type"),
        ("clip_denoised", "yes", "clip_denoised"),
    ],
)
def test_standard_builder_validates_external_dynamics_configuration_once(
    tmp_path: Path,
    parameter: str,
    value: object,
    message: str,
) -> None:
    params = {
        "sampler": {"name": "ddim", "params": {"num_inference_steps": 2}},
        parameter: value,
    }
    raw = _raw_config(
        builder={"name": "standard_denoising", "params": params}
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises((TypeError, ValueError), match=message):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


@pytest.mark.parametrize(
    "sampler",
    [
        {"name": "ddpm", "params": {"start_time": 2}},
        {"name": "ddim", "params": {"schedule": [2, 0]}},
    ],
)
def test_standard_builder_rejects_nonterminal_sampler_start(
    tmp_path: Path,
    sampler: dict[str, Any],
) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {"sampler": sampler},
        }
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="initial observation at terminal time"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


@pytest.mark.parametrize(
    "sampler",
    [
        {"name": "ddpm", "params": {"end_time": 2}},
        {"name": "ddim", "params": {"schedule": [4, 2]}},
    ],
)
def test_standard_builder_rejects_nonclean_sampler_end(
    tmp_path: Path,
    sampler: dict[str, Any],
) -> None:
    raw = _raw_config(
        builder={
            "name": "standard_denoising",
            "params": {"sampler": sampler},
        }
    )
    raw["sampling"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="final observation at clean time"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_full_external_config_can_override_sampling(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    external = _raw_config(
        builder={"name": "stage3_no_shape", "params": {"external": True}}
    )
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(yaml.safe_dump(external), encoding="utf-8")

    result = run_sampling(
        checkpoint=checkpoint,
        config_path=config_path,
        output_dir=tmp_path / "samples",
    )

    assert result.builder_name == "stage3_no_shape"


def test_full_external_config_must_match_checkpoint_model(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    external = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    external["model"]["params"] = {"incompatible": True}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(external), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible.*model"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_full_external_config_must_match_optional_process(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    raw["process"] = None
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    external = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(external), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible.*process"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sampling_only_config_can_override_checkpoint(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    sampling = _raw_config(
        builder={"name": "stage3_no_shape", "params": {"external": True}}
    )["sampling"]
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(
        yaml.safe_dump({"sampling": sampling}),
        encoding="utf-8",
    )

    result = run_sampling(
        checkpoint=checkpoint,
        config_path=config_path,
        output_dir=tmp_path / "samples",
    )

    manifest = yaml.safe_load(result.artifacts["config"].read_text())
    assert result.builder_name == "stage3_no_shape"
    assert manifest["builder"]["params"] == {"external": True}


def test_complete_config_can_locate_checkpoint(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    output_root = tmp_path / "outputs"
    raw["experiment"]["output_dir"] = str(output_root)
    checkpoint = output_root / "run" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    _checkpoint(checkpoint, raw)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = run_sampling(
        config_path=config_path,
        output_dir=tmp_path / "samples",
    )

    assert result.checkpoint_path == checkpoint


def test_checkpoint_only_sampling_does_not_build_training_assets(
    tmp_path: Path,
) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    raw["training"] = {"name": "not_registered", "params": {}}
    raw["objective"] = {"name": "also_not_registered", "params": {}}
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
    )

    assert result.builder_name == "stage3_no_shape"


def test_sampling_only_config_requires_explicit_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(
        yaml.safe_dump({"sampling": _raw_config(builder=None)["sampling"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires an explicit --checkpoint"):
        run_sampling(config_path=config_path)


def test_checkpoint_v6_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw, version=6)

    with pytest.raises(ValueError, match="expected version 7"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_sampling_rejects_missing_configured_process_state(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=False)
    payload.pop("process_state_dict")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="missing required 'process_state_dict'"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_sampling_rejects_process_state_when_config_is_null(tmp_path: Path) -> None:
    raw = _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    raw["process"] = None
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=False)
    payload["process_state_dict"] = {}
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="config.process is null"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (object(), "must return SamplingOutput"),
        (SamplingOutput((), {}), "must not be empty"),
        (
            SamplingOutput(cast(tuple[SamplingBatch, ...], (object(),)), {}),
            "must be SamplingBatch",
        ),
        (
            SamplingOutput((SamplingBatch(torch.zeros(1)),), cast(Any, [])),
            "metadata must be a mapping",
        ),
        (
            SamplingOutput((SamplingBatch(torch.zeros(1)),), cast(Any, {1: "bad"})),
            "metadata keys must be strings",
        ),
        (SamplingOutput((SamplingBatch(torch.zeros(1)),), {"bad": object()}), "JSON"),
    ],
)
def test_sampling_output_contract_errors(output: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_sampling_output(output)


def test_sampling_output_rejects_non_increasing_observations() -> None:
    observations = (
        SamplingObservation(0, 4, torch.zeros(1), False, {}),
        SamplingObservation(0, 0, torch.zeros(1), True, {}),
    )
    with pytest.raises(ValueError, match="must increase"):
        validate_sampling_output(
            SamplingOutput((SamplingBatch(torch.zeros(1), observations),), {})
        )


def test_cli_has_no_sampler_specific_flags() -> None:
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sample", "--checkpoint", "x.pt", "--sampler", "ddim"])


def test_sampling_builder_context_copies_params() -> None:
    raw = {"nested": {"value": 1}}
    config = load_config_dict(
        _raw_config(builder={"name": "stage3_no_shape", "params": {}})
    )
    assert config.process is not None
    context = SamplingBuilderContext(
        raw,
        DiscreteGaussianProcess(config.process.params["schedule"]),
        object(),  # type: ignore[arg-type]
        torch.device("cpu"),
        1,
        None,
        1,
        1,
    )
    context.params["nested"]["value"] = 2
    assert raw == {"nested": {"value": 1}}
    assert CHECKPOINT_FORMAT_VERSION == 7
