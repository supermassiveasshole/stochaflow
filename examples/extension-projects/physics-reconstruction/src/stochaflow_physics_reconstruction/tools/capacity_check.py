"""Measure train, baseline, and guided phases on one real or synthetic batch."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from stochaflow.extensions import DDIMSampler, DiscreteGaussianProcess, MSEObjective
from stochaflow_physics_reconstruction.stochaflow_ext.data import (
    TrajectoryTripletDataset,
)
from stochaflow_physics_reconstruction.stochaflow_ext.model import (
    ConditionalDenoiser,
)
from stochaflow_physics_reconstruction.stochaflow_ext.sampling import (
    GuidedDDIMSampler,
    PhysicsGaussianDynamics,
)
from stochaflow_physics_reconstruction.stochaflow_ext.training import (
    PhysicsDenoisingStrategy,
)


class _ResourceUsage(Protocol):
    ru_maxrss: int


class _ResourceModule(Protocol):
    RUSAGE_SELF: int

    def getrusage(self, who: int) -> _ResourceUsage: ...


def _load_resource_module() -> _ResourceModule | None:
    try:
        module = importlib.import_module("resource")
    except ImportError:  # pragma: no cover - Windows has no resource module.
        return None
    return cast(_ResourceModule, module)


_resource = _load_resource_module()


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _module_bytes(model: torch.nn.Module) -> tuple[int, int]:
    parameters = sum(_tensor_bytes(value) for value in model.parameters())
    buffers = sum(_tensor_bytes(value) for value in model.buffers())
    return parameters, buffers


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _host_peak_rss() -> int | None:
    if _resource is None:
        return None
    value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _start_phase(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    if device.type == "mps":
        torch.mps.empty_cache()
        return {
            "mps_current_allocated_before_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_allocated_before_bytes": torch.mps.driver_allocated_memory(),
        }
    return {}


def _finish_phase(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "cumulative_lifetime_peak_rss_bytes": _host_peak_rss()
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if device.type == "mps":
        torch.mps.synchronize()
        result["mps_current_allocated_after_bytes"] = (
            torch.mps.current_allocated_memory()
        )
        result["mps_driver_allocated_after_bytes"] = torch.mps.driver_allocated_memory()
    return result


def _batch(
    *,
    data: Path | None,
    trajectory_start: int,
    batch_size: int,
    resolution: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if data is None:
        value = torch.randn(batch_size, 3, resolution, resolution, device=device)
        return value, {"kind": "synthetic", "path": None, "memory_mapped": False}
    dataset = TrajectoryTripletDataset(data, (trajectory_start, trajectory_start + 1))
    if batch_size > len(dataset):
        raise ValueError("selected trajectory has fewer triplets than batch_size")
    value = torch.stack([dataset[index] for index in range(batch_size)]).to(device)
    if tuple(value.shape[-2:]) != (resolution, resolution):
        raise ValueError("--resolution does not match the real mmap data")
    return value, {
        "kind": "real-mmap",
        "path": str(data.resolve()),
        "memory_mapped": True,
        "trajectory_start": trajectory_start,
    }


def _schedule(partial_time: int, sample_steps: int) -> list[int]:
    if sample_steps <= 0 or sample_steps > partial_time:
        raise ValueError("sample_steps must lie in [1, partial_time]")
    states = (
        torch.linspace(partial_time, 0, sample_steps + 1, dtype=torch.float64)
        .round()
        .to(torch.long)
    )
    if not torch.all(states[:-1] > states[1:]):
        raise ValueError("sample_steps produced a non-unique schedule")
    return states.tolist()


def run_capacity_check(
    *,
    device: torch.device,
    data: Path | None,
    trajectory_start: int,
    batch_size: int,
    resolution: int,
    hidden_channels: int,
    num_blocks: int,
    time_embedding_dim: int,
    num_timesteps: int,
    partial_time: int,
    sample_steps: int,
) -> dict[str, Any]:
    """Exercise all three coexisting paths and return measured evidence."""

    if batch_size <= 0 or resolution <= 0 or resolution % 2:
        raise ValueError("batch_size and even resolution must be positive")
    if num_timesteps <= 0 or not 0 < partial_time <= num_timesteps:
        raise ValueError("partial_time must lie in [1, num_timesteps]")
    physical, source = _batch(
        data=data,
        trajectory_start=trajectory_start,
        batch_size=batch_size,
        resolution=resolution,
        device=device,
    )
    model = ConditionalDenoiser(
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        time_embedding_dim=time_embedding_dim,
        normalization_mean=-2.27e-8,
        normalization_scale=4.78695,
    ).to(device)
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": num_timesteps,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    ).to(device)
    parameter_bytes, buffer_bytes = _module_bytes(model)
    phases: dict[str, Any] = {}

    train_start = _start_phase(device)
    strategy = PhysicsDenoisingStrategy(
        model,
        process,
        MSEObjective(),
        prediction_type="epsilon",
        conditioning_strength=1.0,
    )
    train_output = strategy.training_step(physical)
    train_output.loss.backward()
    phases["train_forward_backward"] = {
        **train_start,
        **_finish_phase(device),
        "loss": float(train_output.loss.detach().cpu()),
        "gradients_present": any(
            parameter.grad is not None for parameter in model.parameters()
        ),
    }
    model.zero_grad(set_to_none=True)

    state_times = torch.full(
        (batch_size,),
        partial_time,
        device=device,
        dtype=torch.long,
    )
    generator = torch.Generator(device=device).manual_seed(42)
    initial, _ = process.sample_marginal(
        model.normalize(physical),
        state_times,
        generator=generator,
    )
    dynamics = PhysicsGaussianDynamics(
        process,
        model,
        prediction_type="epsilon",
        clip_denoised=False,
        conditioning_strength=1.0,
        correction_strength=1.0,
    )
    schedule = _schedule(partial_time, sample_steps)

    baseline_start = _start_phase(device)
    with torch.no_grad():
        baseline = DDIMSampler(schedule=schedule, eta=0.0).sample(
            dynamics,
            initial.clone(),
            generator=generator,
        )
    baseline_state = baseline.final_state
    if not isinstance(baseline_state, torch.Tensor):
        raise TypeError("baseline capacity sample must return a Tensor")
    phases["baseline_ddim"] = {
        **baseline_start,
        **_finish_phase(device),
        "num_steps": baseline.num_steps,
        "final_state_bytes": _tensor_bytes(baseline_state),
    }
    del baseline_state, baseline

    guided_start = _start_phase(device)
    with torch.no_grad():
        guided = GuidedDDIMSampler(schedule=schedule, eta=0.0).sample(
            dynamics,
            initial.clone(),
            generator=generator,
        )
    guided_state = guided.final_state
    if not isinstance(guided_state, torch.Tensor):
        raise TypeError("guided capacity sample must return a Tensor")
    phases["guided_ddim"] = {
        **guided_start,
        **_finish_phase(device),
        "num_steps": guided.num_steps,
        "final_state_bytes": _tensor_bytes(guided_state),
    }
    return {
        "evidence_kind": "measured-runtime-capacity",
        "device": str(device),
        "source": source,
        "batch_shape": list(physical.shape),
        "dtype": str(physical.dtype),
        "model_parameter_bytes": parameter_bytes,
        "model_buffer_bytes": buffer_bytes,
        "input_bytes": _tensor_bytes(physical),
        "partial_time": partial_time,
        "num_timesteps": num_timesteps,
        "sample_steps": sample_steps,
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--trajectory-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--time-embedding-dim", type=int, default=128)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--partial-time", type=int, default=240)
    parser.add_argument("--sample-steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("capacity-report.json"))
    args = parser.parse_args()
    report = run_capacity_check(
        device=_device(args.device),
        data=args.data,
        trajectory_start=args.trajectory_start,
        batch_size=args.batch_size,
        resolution=args.resolution,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        time_embedding_dim=args.time_embedding_dim,
        num_timesteps=args.num_timesteps,
        partial_time=args.partial_time,
        sample_steps=args.sample_steps,
    )
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(args.output)


if __name__ == "__main__":
    main()
