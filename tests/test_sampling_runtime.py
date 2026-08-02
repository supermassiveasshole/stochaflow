"""Tests for SamplingBuilder orchestration and checkpoint-only sampling."""

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import pytest
import torch
import yaml
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess, Process
from stochaflow.sampling import (
    DDPMAncestralSampler,
    GaussianModelDynamics,
    GenerativeDynamics,
    Sampler,
    SamplerResult,
    SamplingArtifactContext,
    SamplingBatch,
    SamplingBuilder,
    SamplingBuilderContext,
    SamplingObservation,
    SamplingObserver,
    SamplingOutput,
)
from stochaflow.sampling import runtime as sampling_runtime
from stochaflow.sampling.runtime import (
    resolve_sampling_inputs,
    run_resolved_sampling,
    validate_sampling_output,
)
from stochaflow.sampling.runtime import (
    run_sampling as _run_sampling,
)
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
)
from stochaflow.utils.config import (
    ConfigError,
    load_config,
    load_config_dict,
)
from stochaflow.utils.plugins import (
    ExtensionIdentityError,
    activate_extension_plugins,
)
from stochaflow.utils.registry import REGISTRIES

BUILTIN_CONFIGS = Path("examples/built-in/image-generation/configs")
BUILTIN_MNIST_TRAIN_CONFIG = BUILTIN_CONFIGS / "train/mnist.yaml"


class TinyDenoiser(nn.Module):
    """Parameter-bearing model with the built-in denoiser signature."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return torch.zeros_like(state) + self.scale


REGISTRIES.models.add("stage3_tiny_model", TinyDenoiser)


@runtime_checkable
class CalibrationCapabilityFixture(Protocol):
    """Test-only narrow capability consumed by an asset-aware builder."""

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrate one batch of logits."""
        ...


class EmbeddedCalibrationAsset(nn.Module):
    """Registered test asset reconstructed from embedded checkpoint state."""

    constructor_calls = 0

    def __init__(self, *, width: int = 2) -> None:
        super().__init__()
        type(self).constructor_calls += 1
        self.bias = nn.Parameter(torch.zeros(width))

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.bias


REGISTRIES.models.add("stage8_embedded_calibrator", EmbeddedCalibrationAsset)


class NoShapeBuilder(SamplingBuilder):
    calls = 0

    def run(self) -> SamplingOutput:
        type(self).calls += 1
        assert self.context.shape is None
        return SamplingOutput(
            (
                SamplingBatch(
                    torch.full(
                        (self.context.num_samples, 1),
                        self.context.seed,
                        dtype=torch.float32,
                    ),
                    num_samples=self.context.num_samples,
                ),
            ),
            {"kind": "custom"},
        )


REGISTRIES.sampling_builders.add("stage3_no_shape", NoShapeBuilder)


@REGISTRIES.sampling_builders.register("stage8_inference_asset")
class InferenceAssetSamplingBuilder(SamplingBuilder):
    """Consume one declared inference asset through its narrow capability."""

    def run(self) -> SamplingOutput:
        asset = self.context.inference_assets.get(
            "calibrator",
            expected_capability_role="classification_logit_calibrator",
        )
        if not isinstance(asset, CalibrationCapabilityFixture):
            raise TypeError(
                "calibrator must implement CalibrationCapabilityFixture"
            )
        logits = torch.zeros(
            self.context.num_samples,
            2,
            device=self.context.device,
        )
        calibrated = asset.calibrate(logits)
        return SamplingOutput(
            (
                SamplingBatch(
                    calibrated.detach().cpu(),
                    num_samples=calibrated.shape[0],
                ),
            ),
            {"asset": "calibrator"},
        )


@REGISTRIES.sampling_builders.register("stage8_wrong_asset_role")
class WrongInferenceAssetRoleBuilder(SamplingBuilder):
    """Request a declared asset under the wrong semantic role."""

    def run(self) -> SamplingOutput:
        self.context.inference_assets.get(
            "calibrator",
            expected_capability_role="wrong_role",
        )
        raise AssertionError("wrong-role inference asset request must fail")


@REGISTRIES.sampling_builders.register("stage8_missing_asset_slot")
class MissingInferenceAssetSlotBuilder(SamplingBuilder):
    """Request an undeclared inference asset slot."""

    def run(self) -> SamplingOutput:
        self.context.inference_assets.get(
            "missing",
            expected_capability_role="classification_logit_calibrator",
        )
        raise AssertionError("missing inference asset request must fail")


@REGISTRIES.sampling_builders.register("stage8_wrong_asset_capability")
class WrongInferenceAssetCapabilityBuilder(SamplingBuilder):
    """Reject a reconstructed module without the builder-owned capability."""

    def run(self) -> SamplingOutput:
        asset = self.context.inference_assets.get(
            "calibrator",
            expected_capability_role="classification_logit_calibrator",
        )
        if not isinstance(asset, CalibrationCapabilityFixture):
            raise TypeError(
                "calibrator must implement CalibrationCapabilityFixture"
            )
        raise AssertionError("wrong-capability inference asset must fail")


@REGISTRIES.sampling_builders.register("stage6_auto_weights")
class AutoWeightsBuilder(SamplingBuilder):
    """Expose the provider's automatic raw/EMA choice as a tiny sample."""

    def run(self) -> SamplingOutput:
        model, label = self.context.model_provider.resolve("auto")
        assert isinstance(model, TinyDenoiser)
        return SamplingOutput(
            (
                SamplingBatch(
                    model.scale.detach()
                    .cpu()
                    .reshape(1, 1)
                    .expand(self.context.num_samples, 1)
                    .clone(),
                    num_samples=self.context.num_samples,
                ),
            ),
            {"weights": label},
        )


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
            (
                SamplingBatch(
                    transformed.detach().cpu(),
                    num_samples=transformed.shape[0],
                ),
            ),
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
            model,
        )
        initial = torch.zeros(
            (self.context.num_samples, 1), device=self.context.device
        )
        result = sampler.sample(dynamics, initial)
        return SamplingOutput(
            (
                SamplingBatch(
                    result.final_state.detach().cpu(),
                    num_samples=result.final_state.shape[0],
                ),
            ),
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


class ReusingFinalBufferSampler(Sampler):
    """Test sampler that mutates one final-state buffer across batch calls."""

    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None

    def sample(self, dynamics, initial_state, **kwargs):
        assert isinstance(initial_state, torch.Tensor)
        observer = kwargs.get("observer")
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    0, dynamics.process.terminal_time, initial_state, False, {}
                )
            )
        if self.buffer is None:
            self.buffer = torch.zeros_like(initial_state)
        else:
            self.buffer.add_(1)
        if observer is not None:
            observer.observe(
                SamplingObservation(
                    1, dynamics.process.clean_time, self.buffer, True, {}
                )
            )
        return SamplerResult(self.buffer, 1, {})


REGISTRIES.samplers.add("stage6_reusing_final_buffer", ReusingFinalBufferSampler)


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


def _raw_config(
    *,
    recipe_name: str | None,
    recipe_contract: dict[str, Any] | None = None,
    sampler: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recipe = (
        {
            "schema_version": 1,
            "name": recipe_name,
            "contract": deepcopy(recipe_contract or {}),
        }
        if recipe_name is not None
        else None
    )
    return {
        "_recipe": recipe,
        "experiment": {"name": "test", "output_dir": "unused", "seed": 7},
        "extensions": {"plugins": []},
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
        "_sample": {
            "sampler": deepcopy(sampler),
            "options": deepcopy(options or {}),
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
    config = deepcopy(raw)
    recipe = config.pop("_recipe")
    sample = config.pop("_sample", None)
    payload: dict[str, Any] = {
        "format_version": version,
        "epoch": 1,
        "global_step": 2,
        "model_state_dict": model.state_dict(),
        "rng_state": capture_rng_state(),
        "config": config,
        "metadata": {"extension_plugins": []},
    }
    if version == CHECKPOINT_FORMAT_VERSION:
        payload.update(
            {
                "precision_kind": "fp32",
                "inference_asset_descriptors": {},
                "inference_recipe": recipe,
            }
        )
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
    if sample is not None:
        path.with_suffix(".sample.yaml").write_text(
            yaml.safe_dump({"sample": sample}, sort_keys=False),
            encoding="utf-8",
        )
    return path


def run_sampling(
    *,
    checkpoint: str | Path | None = None,
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> sampling_runtime.SamplingRunResult:
    """Run the production API with a complete fixture config by default."""

    if config_path is None and checkpoint is not None:
        config_path = Path(checkpoint).with_suffix(".sample.yaml")
    return _run_sampling(
        checkpoint=checkpoint,
        config_path=config_path,
        **kwargs,
    )


def _write_sample_config(
    path: Path,
    sample: dict[str, Any],
    *,
    extensions: list[str] | None = None,
) -> Path:
    document: dict[str, Any] = {"sample": deepcopy(sample)}
    if extensions is not None:
        document["extensions"] = {"plugins": list(extensions)}
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("profile_name", "sampler_name"),
    [
        ("mnist-ddpm.yaml", "ddpm"),
        ("mnist-ddim-50.yaml", "ddim"),
    ],
)
def test_builtin_mnist_sample_profiles_complete_the_train_checkpoint_request(
    tmp_path: Path,
    profile_name: str,
    sampler_name: str,
) -> None:
    raw = load_config(BUILTIN_MNIST_TRAIN_CONFIG).to_dict()
    assert "sampling" not in raw
    expected_checkpoint_config = deepcopy(raw)
    raw["_recipe"] = {
        "schema_version": 1,
        "name": "standard_denoising",
        "contract": {"prediction_type": "v"},
    }
    checkpoint = _checkpoint(tmp_path / "mnist.pt", raw)

    inputs = resolve_sampling_inputs(
        config_path=BUILTIN_CONFIGS / "sample" / profile_name,
        checkpoint=checkpoint,
    )

    assert inputs.recipe.name == "standard_denoising"
    assert inputs.sample_config.sample.shape == [1, 32, 32]
    assert inputs.sample_config.sample.sampler is not None
    assert inputs.sample_config.sample.sampler.name == sampler_name
    assert [writer.name for writer in inputs.sample_config.sample.writers] == [
        "tensor",
        "image",
    ]
    assert inputs.checkpoint_config.to_dict() == expected_checkpoint_config


def test_resolved_sampling_rejects_swapped_activation_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(
        tmp_path / "checkpoint.pt",
        _raw_config(recipe_name="stage3_no_shape"),
    )
    config_path = checkpoint.with_suffix(".sample.yaml")
    requested = resolve_sampling_inputs(
        config_path=config_path,
        checkpoint=checkpoint,
    )
    other = resolve_sampling_inputs(
        config_path=config_path,
        checkpoint=checkpoint,
    )
    swapped_receipt = activate_extension_plugins(other.extension_plan)
    output_dir = tmp_path / "swapped-receipt"

    with pytest.raises(ValueError, match="different extension plan"):
        run_resolved_sampling(
            requested,
            swapped_receipt,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert not output_dir.exists()


def test_custom_builder_runs_once_without_shape(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    NoShapeBuilder.calls = 0

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
        device_name="cpu",
    )

    assert NoShapeBuilder.calls == 1
    assert result.recipe_name == "stage3_no_shape"
    assert result.metadata == {"kind": "custom"}
    assert torch.equal(
        torch.load(result.artifacts["samples"], weights_only=True),
        torch.full((3, 1), 11.0),
    )


def test_sampling_publication_preserves_an_existing_output_directory(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    output_dir = tmp_path / "samples"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_sampling(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_sampling_writer_failure_removes_private_staging_and_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    output_dir = tmp_path / "samples"

    def fail_writing(
        configs: Any,
        context: SamplingArtifactContext,
    ) -> dict[str, Path]:
        del configs
        (context.output_dir / "partial.txt").write_text(
            "partial",
            encoding="utf-8",
        )
        raise RuntimeError("writer failed")

    monkeypatch.setattr(
        sampling_runtime,
        "write_sampling_artifacts",
        fail_writing,
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        run_sampling(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert not output_dir.exists()
    assert not tuple(tmp_path.glob(".samples.sampling-*"))


def test_direct_transform_runs_without_process_or_sampler(tmp_path: Path) -> None:
    raw = _raw_config(
        recipe_name="stage3_direct_transform",
        options={"offset": 3.0},
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
    assert manifest["selected_components"]["process"] is None
    assert (
        manifest["selected_components"]["inference_recipe"]
        == "stage3_direct_transform"
    )
    assert manifest["selected_components"]["sampler"] is None
    assert manifest["selected_components"]["artifact_writers"] == [
        "tensor"
    ]
    assert manifest["checkpoint_identity"] == {
        "path": str(checkpoint.resolve()),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": 1,
        "global_step": 2,
    }
    assert manifest["lineage"]["checkpoint"] == manifest["checkpoint_identity"]
    assert manifest["artifacts"] == {"samples": "samples.pt"}
    assert torch.equal(
        torch.load(result.artifacts["samples"]),
        torch.full((3, 1), 3.0),
    )


def test_complete_sample_config_owns_direct_recipe_options(tmp_path: Path) -> None:
    raw = _raw_config(
        recipe_name="stage3_direct_transform",
        options={"offset": 1.0},
    )
    raw.pop("process")
    checkpoint = _checkpoint(tmp_path / "direct.pt", raw)
    config_path = tmp_path / "direct.yaml"
    sample = deepcopy(raw["_sample"])
    sample["options"] = {"offset": 2.0}
    config_path.write_text(
        yaml.safe_dump({"sample": sample}),
        encoding="utf-8",
    )

    result = run_sampling(
        checkpoint=checkpoint,
        config_path=config_path,
        output_dir=tmp_path / "direct-samples",
    )

    assert torch.equal(
        torch.load(result.artifacts["samples"], weights_only=True),
        torch.full((3, 1), 2.0),
    )


def test_standard_builder_rejects_missing_process_at_its_boundary(
    tmp_path: Path,
) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        sampler={"name": "ddim", "params": {"num_inference_steps": 2}},
    )
    raw["process"] = None
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "missing-process.pt", raw)

    with pytest.raises(TypeError, match="DiscreteGaussianDenoisingProcess"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_second_algorithm_family_runs_without_core_dispatch_changes(
    tmp_path: Path,
) -> None:
    raw = _raw_config(
        recipe_name="stage3_toy_flow",
        sampler={
            "name": "stage3_toy_euler",
            "params": {"num_steps": 2},
        },
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
    raw = _raw_config(
        recipe_name="standard_denoising",
        recipe_contract={"prediction_type": "epsilon"},
        sampler={
            "name": "ddim",
            "params": {"num_inference_steps": 2, "eta": 0.0},
        },
        options={
            "weights": "raw",
            "clip_denoised": False,
            "trajectory": {"enabled": True, "every_steps": 1},
        },
    )
    raw["_sample"]["shape"] = [2]
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


def test_standard_builder_owns_each_writer_ready_batch(tmp_path: Path) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        sampler={"name": "stage6_reusing_final_buffer", "params": {}},
        options={"trajectory": {"enabled": True, "every_steps": 1}},
    )
    raw["_sample"].update({"shape": [2], "num_samples": 4, "batch_size": 2})
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
        device_name="cpu",
    )

    samples = torch.load(result.artifacts["samples"], weights_only=True)
    trajectory = torch.load(result.artifacts["trajectory"], weights_only=True)
    assert torch.equal(samples[:2], torch.zeros(2, 2))
    assert torch.equal(samples[2:], torch.ones(2, 2))
    assert torch.equal(trajectory["states"][1, :2], torch.zeros(2, 2))
    assert torch.equal(trajectory["states"][1, 2:], torch.ones(2, 2))


def test_standard_builder_rejects_invalid_sampler_result(tmp_path: Path) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        sampler={"name": "stage3_bad_result", "params": {}},
    )
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(TypeError, match="must return SamplerResult"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_standard_builder_rejects_wrong_sampler_shape(tmp_path: Path) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        sampler={"name": "stage3_wrong_shape", "params": {}},
    )
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="final state has shape"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_standard_builder_rejects_wrong_trajectory_shape(tmp_path: Path) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        sampler={
            "name": "stage3_wrong_trajectory_shape",
            "params": {},
        },
        options={"trajectory": {"enabled": True}},
    )
    raw["_sample"]["shape"] = [2]
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
        recipe_name="standard_denoising",
        sampler={
            "name": "stage3_malformed_lifecycle",
            "params": {"mode": mode},
        },
    )
    raw["_sample"]["shape"] = [2]
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
    raw = _raw_config(
        recipe_name="standard_denoising",
        recipe_contract=(
            {parameter: value} if parameter == "prediction_type" else None
        ),
        sampler={"name": "ddim", "params": {"num_inference_steps": 2}},
        options={parameter: value} if parameter != "prediction_type" else None,
    )
    raw["_sample"]["shape"] = [2]
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
        recipe_name="standard_denoising",
        sampler=sampler,
    )
    raw["_sample"]["shape"] = [2]
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
        recipe_name="standard_denoising",
        sampler=sampler,
    )
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="final observation at clean time"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_full_external_config_is_rejected(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown config field"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_complete_sample_config_is_independent_from_checkpoint(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    sample = deepcopy(raw["_sample"])
    sample["options"] = {"external": True}
    _write_sample_config(config_path, sample)

    result = run_sampling(
        checkpoint=checkpoint,
        config_path=config_path,
        output_dir=tmp_path / "samples",
    )

    manifest = yaml.safe_load(result.artifacts["config"].read_text())
    assert result.recipe_name == "stage3_no_shape"
    assert manifest["sample"]["options"] == {"external": True}
    assert manifest["selected_components"]["inference_recipe"] == "stage3_no_shape"
    assert manifest["selected_components"]["sampler"] is None
    assert "config" not in manifest


def test_sample_config_cannot_select_a_recipe(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    sample = deepcopy(raw["_sample"])
    sample["builder"] = {"name": "stage3_direct_transform"}
    _write_sample_config(config_path, sample)

    with pytest.raises(ConfigError, match=r"config\.sample\.builder"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sample_config_contract_collision_fails_before_plugin_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        recipe_contract={"prediction_type": "v"},
        sampler={
            "name": "ddim",
            "params": {"num_inference_steps": 2},
        },
    )
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    sample = deepcopy(raw["_sample"])
    sample["options"] = {"prediction_type": "epsilon"}
    _write_sample_config(config_path, sample)

    def unexpected_plugin_preflight(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("plugin preflight must not run for an invalid request")

    monkeypatch.setattr(
        sampling_runtime,
        "prepare_extension_plugins",
        unexpected_plugin_preflight,
    )
    with pytest.raises(ConfigError, match="fixed inference contract"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sample_config_cannot_override_learned_variance_contract(
    tmp_path: Path,
) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        recipe_contract={
            "prediction_type": "epsilon",
            "variance": {"mode": "learned_range"},
        },
        sampler={
            "name": "ddpm",
            "params": {"num_inference_steps": 2},
        },
    )
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    sample = deepcopy(raw["_sample"])
    sample["options"] = {"variance": {"mode": "fixed"}}
    _write_sample_config(config_path, sample)

    with pytest.raises(ConfigError, match="fixed inference contract"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sample_config_rejects_partial_invocation(
    tmp_path: Path,
) -> None:
    raw = _raw_config(
        recipe_name="standard_denoising",
        sampler={
            "name": "ddim",
            "params": {"num_inference_steps": 2},
        },
        options={"weights": "raw"},
    )
    raw["_sample"]["shape"] = [2]
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(
        yaml.safe_dump({"sample": {"num_samples": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="missing required config field"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sample_config_rejects_unknown_extension_fields(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    document = {"sample": raw["_sample"], "extensions": {"plguins": []}}
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"config\.extensions\.plguins"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sample_config_rejects_null_extensions_section(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(
        yaml.safe_dump({"sample": raw["_sample"], "extensions": None}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=r"config\.extensions must not be null"):
        run_sampling(checkpoint=checkpoint, config_path=config_path)


def test_sample_plugins_are_additive_to_checkpoint_requirements() -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    raw.pop("_recipe")
    raw.pop("_sample")
    raw["extensions"] = {"plugins": None}
    checkpoint_config = load_config_dict(raw)

    resolved, added = sampling_runtime._prepare_sample_extensions(
        checkpoint_config,
        additions=("extra", "required", "extra"),
        expected_plugin_names=("required",),
    )

    assert added is True
    assert resolved.extensions.plugins == ["required", "extra"]


def test_sample_config_rejects_unproven_checkpoint_config_plugins() -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    raw.pop("_recipe")
    raw.pop("_sample")
    raw["extensions"] = {"plugins": ["config-only"]}
    checkpoint_config = load_config_dict(raw)

    with pytest.raises(
        ExtensionIdentityError,
        match=r"unproven config-only plugin.*config-only",
    ):
        sampling_runtime._prepare_sample_extensions(
            checkpoint_config,
            additions=("request-added",),
            expected_plugin_names=("provenance-required",),
        )


def test_sample_plugin_additions_do_not_mutate_checkpoint_config() -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    raw.pop("_recipe")
    raw.pop("_sample")
    checkpoint_config = load_config_dict(raw)
    before = checkpoint_config.to_dict()

    resolved, added = sampling_runtime._prepare_sample_extensions(
        checkpoint_config,
        additions=("extra",),
        expected_plugin_names=(),
    )

    assert added is True
    assert resolved.extensions.plugins == ["extra"]
    assert checkpoint_config.to_dict() == before


def test_checkpoint_only_sampling_does_not_build_training_assets(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    raw["training"] = {"name": "not_registered", "params": {}}
    raw["objective"] = {"name": "also_not_registered", "params": {}}
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
    )

    assert result.recipe_name == "stage3_no_shape"


def test_sampling_inputs_drop_training_only_checkpoint_state(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    ema_model = TinyDenoiser()
    ema_model.load_state_dict(payload["model_state_dict"])
    ema = ExponentialMovingAverage(ema_model)
    payload.update(
        {
            "ema_model_state_dict": ema_model.state_dict(),
            "objective_state_dict": {"scale": torch.ones(1)},
            "training_assets_state_dict": {"teacher": {"weight": torch.ones(1)}},
            "optimizer_state_dict": {"state": {0: {"moment": torch.ones(1)}}},
            "lr_scheduler_state_dict": {"last_epoch": 2},
            "ema_state_dict": ema.state_dict(),
            "rng_state": capture_rng_state(),
            "epoch": 2,
            "global_step": 4,
        }
    )
    torch.save(payload, checkpoint)

    inputs = resolve_sampling_inputs(
        config_path=checkpoint.with_suffix(".sample.yaml"),
        checkpoint=checkpoint,
    )

    assert set(inputs.checkpoint) == {
        "format_version",
        "config",
        "inference_recipe",
        "metadata",
        "model_state_dict",
        "ema_model_state_dict",
        "process_state_dict",
        "inference_asset_descriptors",
        "inference_asset_state_dicts",
    }
    assert inputs.checkpoint["inference_asset_descriptors"] == {}
    assert inputs.checkpoint["inference_asset_state_dicts"] == {}
    persisted = torch.load(checkpoint, weights_only=True)
    assert "optimizer_state_dict" in persisted
    assert "training_assets_state_dict" in persisted


def test_sampling_view_projects_only_declared_asset_state_without_copying_tensors(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage8_inference_asset")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    calibration_state = {
        "bias": torch.tensor([4.0, -2.0], dtype=torch.float32)
    }
    teacher_state = {"weight": torch.ones(3)}
    payload["inference_asset_descriptors"] = {
        "calibrator": {
            "training_asset_name": "calibrator_module",
            "declaration": {
                "name": "stage8_embedded_calibrator",
                "params": {"width": 2},
            },
            "capability_role": "classification_logit_calibrator",
            "persistence": "embedded_state",
        }
    }
    payload["training_assets_state_dict"] = {
        "calibrator_module": calibration_state,
        "teacher": teacher_state,
    }

    view = sampling_runtime._sampling_checkpoint_view(payload)

    assert view["inference_asset_descriptors"] == payload[
        "inference_asset_descriptors"
    ]
    assert set(view["inference_asset_state_dicts"]) == {"calibrator_module"}
    assert (
        view["inference_asset_state_dicts"]["calibrator_module"]
        is calibration_state
    )
    assert (
        view["inference_asset_state_dicts"]["calibrator_module"]["bias"]
        is calibration_state["bias"]
    )
    assert set(payload["training_assets_state_dict"]) == {
        "calibrator_module",
        "teacher",
    }


def test_sampling_uses_requested_embedded_asset_and_skips_other_training_state(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage8_inference_asset")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload["inference_asset_descriptors"] = {
        "calibrator": {
            "training_asset_name": "calibrator_module",
            "declaration": {
                "name": "stage8_embedded_calibrator",
                "params": {"width": 2},
            },
            "capability_role": "classification_logit_calibrator",
            "persistence": "embedded_state",
        },
        "unused": {
            "training_asset_name": "unused_module",
            "declaration": {
                "name": "stage8_embedded_calibrator",
                "params": {"width": 2},
            },
            "capability_role": "unused_test_role",
            "persistence": "embedded_state",
        },
    }
    payload["training_assets_state_dict"] = {
        "calibrator_module": {
            "bias": torch.tensor([4.0, -2.0], dtype=torch.float32)
        },
        "unused_module": {
            "bias": torch.tensor([99.0, 99.0], dtype=torch.float32)
        },
        "teacher": {"weight": torch.ones(3)},
    }
    torch.save(payload, checkpoint)
    EmbeddedCalibrationAsset.constructor_calls = 0

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
        device_name="cpu",
    )

    assert EmbeddedCalibrationAsset.constructor_calls == 1
    assert result.metadata == {"asset": "calibrator"}
    assert torch.equal(
        torch.load(result.artifacts["samples"], weights_only=True),
        torch.tensor([[4.0, -2.0]]).expand(3, 2),
    )


def test_wrong_inference_asset_role_fails_before_construction_or_output(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage8_wrong_asset_role")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload["inference_asset_descriptors"] = {
        "calibrator": {
            "training_asset_name": "calibrator_module",
            "declaration": {
                "name": "stage8_embedded_calibrator",
                "params": {"width": 2},
            },
            "capability_role": "classification_logit_calibrator",
            "persistence": "embedded_state",
        }
    }
    payload["training_assets_state_dict"] = {
        "calibrator_module": {"bias": torch.zeros(2)}
    }
    torch.save(payload, checkpoint)
    output_dir = tmp_path / "samples"
    EmbeddedCalibrationAsset.constructor_calls = 0

    with pytest.raises(ValueError, match=r"has capability role.*expected"):
        run_sampling(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert EmbeddedCalibrationAsset.constructor_calls == 0
    assert not output_dir.exists()


def test_missing_inference_asset_slot_fails_before_construction_or_output(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage8_missing_asset_slot")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload["inference_asset_descriptors"] = {
        "calibrator": {
            "training_asset_name": "calibrator_module",
            "declaration": {
                "name": "stage8_embedded_calibrator",
                "params": {"width": 2},
            },
            "capability_role": "classification_logit_calibrator",
            "persistence": "embedded_state",
        }
    }
    payload["training_assets_state_dict"] = {
        "calibrator_module": {"bias": torch.zeros(2)}
    }
    torch.save(payload, checkpoint)
    output_dir = tmp_path / "samples"
    EmbeddedCalibrationAsset.constructor_calls = 0

    with pytest.raises(KeyError, match="unknown inference asset slot"):
        run_sampling(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert EmbeddedCalibrationAsset.constructor_calls == 0
    assert not output_dir.exists()


def test_builder_rejects_wrong_inference_asset_capability_without_output(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage8_wrong_asset_capability")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload["inference_asset_descriptors"] = {
        "calibrator": {
            "training_asset_name": "calibrator_module",
            "declaration": {
                "name": "stage3_tiny_model",
                "params": {},
            },
            "capability_role": "classification_logit_calibrator",
            "persistence": "embedded_state",
        }
    }
    payload["training_assets_state_dict"] = {
        "calibrator_module": TinyDenoiser().state_dict()
    }
    torch.save(payload, checkpoint)
    output_dir = tmp_path / "samples"

    with pytest.raises(
        TypeError,
        match="must implement CalibrationCapabilityFixture",
    ):
        run_sampling(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device_name="cpu",
        )

    assert not output_dir.exists()


def test_checkpoint_sampling_view_retains_and_uses_ema_weights(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage6_auto_weights")
    raw["ema"] = {"enabled": True}
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    ema_model = TinyDenoiser()
    with torch.no_grad():
        ema_model.scale.fill_(5.0)
    raw_model = TinyDenoiser()
    raw_model.load_state_dict(payload["model_state_dict"])
    ema = ExponentialMovingAverage(raw_model)
    ema.shadow_params["scale"].fill_(5.0)
    payload["ema_model_state_dict"] = ema_model.state_dict()
    payload["ema_state_dict"] = ema.state_dict()
    torch.save(payload, checkpoint)

    result = run_sampling(
        checkpoint=checkpoint,
        output_dir=tmp_path / "samples",
        device_name="cpu",
    )

    samples = torch.load(result.artifacts["samples"], weights_only=True)
    assert result.metadata == {"weights": "ema"}
    assert torch.equal(samples, torch.full((3, 1), 5.0))


def test_sample_config_requires_explicit_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "sampling.yaml"
    config_path.write_text(
        yaml.safe_dump({"sample": _raw_config(recipe_name=None)["_sample"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires an explicit --checkpoint"):
        run_sampling(config_path=config_path)


def test_checkpoint_sampling_requires_explicit_sample_config(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="requires an explicit --config"):
        _run_sampling(checkpoint=checkpoint)


def test_checkpoint_without_inference_recipe_field_is_rejected(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload.pop("inference_recipe")
    torch.save(payload, checkpoint)

    with pytest.raises(TypeError, match="missing inference_recipe"):
        run_sampling(checkpoint=checkpoint)


def test_null_inference_recipe_disables_checkpoint_sampling(
    tmp_path: Path,
) -> None:
    raw = _raw_config(recipe_name=None)
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)

    with pytest.raises(ValueError, match="does not support sampling"):
        run_sampling(checkpoint=checkpoint)


@pytest.mark.parametrize("version", [8, 9, 10, 11])
def test_sampling_runtime_rejects_legacy_checkpoint_headers(
    tmp_path: Path,
    version: int,
) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw, version=version)

    with pytest.raises(ValueError, match=r"expected version 12"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_sampling_rejects_missing_configured_process_state(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload.pop("process_state_dict")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="missing required 'process_state_dict'"):
        run_sampling(checkpoint=checkpoint, output_dir=tmp_path / "samples")


def test_sampling_rejects_process_state_when_config_is_null(tmp_path: Path) -> None:
    raw = _raw_config(recipe_name="stage3_no_shape")
    raw["process"] = None
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", raw)
    payload = torch.load(checkpoint, weights_only=True)
    payload["process_state_dict"] = {}
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=r"config\.process is null"):
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
            SamplingOutput(
                (SamplingBatch(torch.zeros(1), num_samples=1),),
                cast(Any, []),
            ),
            "metadata must be a mapping",
        ),
        (
            SamplingOutput(
                (SamplingBatch(torch.zeros(1), num_samples=1),),
                cast(Any, {1: "bad"}),
            ),
            "metadata keys must be strings",
        ),
        (
            SamplingOutput(
                (SamplingBatch(torch.zeros(1), num_samples=1),),
                {"bad": object()},
            ),
            "JSON",
        ),
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
            SamplingOutput(
                (
                    SamplingBatch(
                        torch.zeros(1),
                        num_samples=1,
                        trajectory=observations,
                    ),
                ),
                {},
            )
        )


def test_cli_has_no_sampler_specific_flags() -> None:
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sample", "--checkpoint", "x.pt", "--sampler", "ddim"])


def test_sampling_builder_context_copies_params() -> None:
    raw = {"nested": {"value": 1}}
    raw_config = _raw_config(recipe_name="stage3_no_shape")
    raw_config.pop("_recipe")
    raw_config.pop("_sample")
    config = load_config_dict(raw_config)
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
    assert CHECKPOINT_FORMAT_VERSION == 12
