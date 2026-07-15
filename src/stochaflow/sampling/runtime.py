"""Config-driven checkpoint sampling orchestration."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn
import yaml

from stochaflow.sampling.grid import (
    save_image_grid,
    save_trajectory_gif,
    save_trajectory_grid,
)
from stochaflow.sampling.sampler import SamplingTrace
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    CheckpointState,
)
from stochaflow.utils.config import (
    ComponentConfig,
    StochaflowConfig,
    load_config,
    load_config_dict,
)
from stochaflow.utils.factory import (
    build_diffusion,
    build_model,
    build_noise_schedule,
    resolve_device,
)
from stochaflow.utils.seed import set_seed


_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ResolvedSamplingInputs:
    """Merged configuration and checkpoint payload for one sampling run."""

    config: StochaflowConfig
    checkpoint_path: Path
    checkpoint: CheckpointState
    sampler: ComponentConfig


@dataclass(frozen=True, slots=True)
class SamplingRunResult:
    """Paths and runtime choices produced by one sampling invocation."""

    checkpoint_path: Path
    output_dir: Path
    sampler_name: str
    device: torch.device
    seed: int
    used_ema: bool
    artifacts: dict[str, Path]


def parse_sampler_params(values: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` sampler overrides."""

    parsed: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(
                f"invalid --sampler-param '{item}'; expected KEY=VALUE"
            )
        key, raw_value = item.split("=", 1)
        if not _PARAMETER_NAME.fullmatch(key):
            raise ValueError(
                f"invalid sampler parameter name '{key}'; expected a Python identifier"
            )
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"invalid YAML value for sampler parameter '{key}': {exc}"
            ) from exc
        if isinstance(value, dict):
            raise ValueError(
                f"sampler parameter '{key}' must be a scalar or list, not a mapping"
            )
        parsed[key] = value
    return parsed


def image_sample_shape(config: StochaflowConfig, num_samples: int) -> torch.Size:
    """Build a batch-first sample shape from the configured sample bucket."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    bucket = next(
        bucket
        for bucket in config.data.batching.buckets
        if bucket.name == config.data.batching.sample_bucket
    )
    return torch.Size(
        (num_samples, config.data.image.channels, bucket.height, bucket.width)
    )


def resolve_sampling_inputs(
    *,
    config_path: str | Path | None,
    checkpoint: str | Path | None,
    sampler_name: str | None = None,
    sampler_params: dict[str, Any] | None = None,
) -> ResolvedSamplingInputs:
    """Resolve checkpoint config, optional external config, and CLI overrides."""

    if config_path is None and checkpoint is None:
        raise ValueError("sample requires --config, --checkpoint, or both")

    external_config = load_config(config_path) if config_path is not None else None
    checkpoint_path = _resolve_checkpoint_path(checkpoint, external_config)
    payload = CheckpointManager.load_payload(checkpoint_path, map_location="cpu")
    checkpoint_config = _load_checkpoint_config(payload)

    if external_config is not None:
        validate_sampling_compatibility(external_config, checkpoint_config)
        config = deepcopy(checkpoint_config)
        config.sampling = deepcopy(external_config.sampling)
    else:
        config = checkpoint_config

    resolved_config = deepcopy(config)
    configured_sampler = resolved_config.sampling.sampler
    if configured_sampler is None:
        configured_sampler = ComponentConfig(
            name=resolved_config.diffusion.name,
            params=dict(resolved_config.diffusion.params),
        )
    else:
        configured_sampler = deepcopy(configured_sampler)

    if sampler_name is not None and sampler_name != configured_sampler.name:
        configured_sampler = ComponentConfig(name=sampler_name)
    elif sampler_name is not None:
        configured_sampler.name = sampler_name
    configured_sampler.params.update(sampler_params or {})
    resolved_config.sampling.sampler = deepcopy(configured_sampler)

    return ResolvedSamplingInputs(
        config=resolved_config,
        checkpoint_path=checkpoint_path,
        checkpoint=payload,
        sampler=configured_sampler,
    )


def validate_sampling_compatibility(
    external: StochaflowConfig,
    checkpoint: StochaflowConfig,
) -> None:
    """Reject external configs that do not describe the trained denoiser."""

    mismatches: list[str] = []
    if external.model != checkpoint.model:
        mismatches.append("model")
    if external.diffusion != checkpoint.diffusion:
        mismatches.append("diffusion")
    if external.data.image.channels != checkpoint.data.image.channels:
        mismatches.append("data.image.channels")
    if mismatches:
        raise ValueError(
            "external config is incompatible with checkpoint training config: "
            + ", ".join(mismatches)
        )


def run_sampling(
    *,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    sampler_name: str | None = None,
    sampler_param_values: list[str] | None = None,
) -> SamplingRunResult:
    """Run standalone or post-training sampling and write all artifacts."""

    inputs = resolve_sampling_inputs(
        config_path=config_path,
        checkpoint=checkpoint,
        sampler_name=sampler_name,
        sampler_params=parse_sampler_params(sampler_param_values),
    )
    config = inputs.config
    sampling = config.sampling
    seed = config.experiment.seed if sampling.seed is None else sampling.seed
    set_seed(seed)
    device = resolve_device(device_name or config.trainer.device)
    denoiser, used_ema = _build_checkpointed_denoiser(
        config,
        inputs.checkpoint,
        device=device,
    )
    noise_schedule = build_noise_schedule(config.diffusion.noise_schedule)
    sampler = build_diffusion(
        inputs.sampler.name,
        model=denoiser,
        noise_schedule=noise_schedule,
        params=inputs.sampler.params,
    )
    sampler.to(device)
    sampler.eval()

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_sampling_output_dir(inputs.checkpoint_path)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    batch_size = sampling.batch_size or config.data.dataloader.batch_size

    with torch.no_grad():
        samples, trajectory = _sample_batches(
            sampler,
            config,
            device=device,
            batch_size=batch_size,
        )
    artifacts = _write_sampling_artifacts(
        samples,
        trajectory,
        output_dir=target_dir,
        grid_nrow=sampling.grid_nrow,
        gif_fps=sampling.debug.trajectory.gif_fps,
    )
    manifest_path = target_dir / "resolved_sampling.yaml"
    manifest = {
        "checkpoint": str(inputs.checkpoint_path),
        "checkpoint_format_version": inputs.checkpoint.get("format_version"),
        "sampler": asdict(inputs.sampler),
        "sampling": asdict(sampling),
        "seed": seed,
        "device": str(device),
        "weights": "ema" if used_ema else "raw",
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    artifacts["config"] = manifest_path

    return SamplingRunResult(
        checkpoint_path=inputs.checkpoint_path,
        output_dir=target_dir,
        sampler_name=inputs.sampler.name,
        device=device,
        seed=seed,
        used_ema=used_ema,
        artifacts=artifacts,
    )


def _resolve_checkpoint_path(
    checkpoint: str | Path | None,
    external_config: StochaflowConfig | None,
) -> Path:
    if checkpoint is None:
        assert external_config is not None
        return CheckpointManager.find_best(external_config.experiment.output_dir)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        return CheckpointManager.find_best(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def _load_checkpoint_config(payload: CheckpointState) -> StochaflowConfig:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint does not contain a Stochaflow config")
    return load_config_dict(raw_config)


def _build_checkpointed_denoiser(
    config: StochaflowConfig,
    payload: CheckpointState,
    *,
    device: torch.device,
) -> tuple[nn.Module, bool]:
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "checkpoint does not contain portable denoiser weights from format "
            f"version {CHECKPOINT_FORMAT_VERSION}"
        )
    use_ema = config.ema.enabled and config.ema.use_for_sampling
    state_key = "ema_denoiser_state_dict" if use_ema else "denoiser_state_dict"
    state = payload.get(state_key)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is missing required '{state_key}'")

    denoiser = build_model(config.model)
    denoiser.load_state_dict(state)
    denoiser.to(device)
    denoiser.eval()
    return denoiser, use_ema


def _sample_batches(
    sampler: nn.Module,
    config: StochaflowConfig,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor] | None]:
    sampling = config.sampling
    counts = _batched_sample_counts(sampling.num_samples, batch_size)
    trajectory_config = sampling.debug.trajectory
    sample_parts: list[torch.Tensor] = []
    trajectory_parts: dict[int, list[torch.Tensor]] = {}
    expected_state_times: list[int] | None = None

    for count in counts:
        shape = image_sample_shape(config, count)
        if trajectory_config.enabled:
            trajectory_fn = getattr(sampler, "sample_trajectory", None)
            if not callable(trajectory_fn):
                raise TypeError(
                    f"sampler '{type(sampler).__name__}' does not support "
                    "trajectory debug"
                )
            trace = trajectory_fn(
                shape,
                device=device,
                **trajectory_config.params,
            )
            if not isinstance(trace, SamplingTrace):
                raise TypeError("sample_trajectory must return SamplingTrace")
            state_times = [frame.state_time for frame in trace.frames]
            if expected_state_times is None:
                expected_state_times = state_times
            elif state_times != expected_state_times:
                raise ValueError("trajectory frame times changed between sample batches")
            for frame in trace.frames:
                trajectory_parts.setdefault(frame.state_time, []).append(
                    frame.samples.detach().cpu()
                )
            sample_parts.append(trace.samples.detach().cpu())
        else:
            sample_fn = getattr(sampler, "sample", None)
            if not callable(sample_fn):
                raise TypeError("configured sampler does not provide sample()")
            sampled = sample_fn(shape, device=device)
            if not isinstance(sampled, torch.Tensor):
                raise TypeError("sample() must return a torch.Tensor")
            sample_parts.append(sampled.detach().cpu())

    samples = torch.cat(sample_parts, dim=0)
    if not trajectory_config.enabled:
        return samples, None
    trajectory = {
        state_time: torch.cat(parts, dim=0)
        for state_time, parts in trajectory_parts.items()
    }
    return samples, trajectory


def _batched_sample_counts(num_samples: int, batch_size: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("sampling.num_samples must be positive")
    if batch_size <= 0:
        raise ValueError("sampling batch size must be positive")
    return [
        min(batch_size, num_samples - offset)
        for offset in range(0, num_samples, batch_size)
    ]


def _write_sampling_artifacts(
    samples: torch.Tensor,
    trajectory: dict[int, torch.Tensor] | None,
    *,
    output_dir: Path,
    grid_nrow: int,
    gif_fps: int,
) -> dict[str, Path]:
    artifacts = {
        "raw_samples": output_dir / "samples.pt",
        "samples": output_dir / "samples.png",
    }
    torch.save(samples, artifacts["raw_samples"])
    save_image_grid(samples, artifacts["samples"], nrow=grid_nrow)
    if trajectory is None:
        return artifacts

    artifacts.update(
        {
            "raw_trajectory": output_dir / "trajectory.pt",
            "trajectory": output_dir / "trajectory.png",
            "trajectory_gif": output_dir / "trajectory.gif",
        }
    )
    torch.save(trajectory, artifacts["raw_trajectory"])
    save_trajectory_grid(trajectory, artifacts["trajectory"])
    save_trajectory_gif(
        trajectory,
        artifacts["trajectory_gif"],
        nrow=grid_nrow,
        fps=gif_fps,
    )
    return artifacts


def _default_sampling_output_dir(checkpoint_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = checkpoint_path.parent.parent / "samples"
    output_dir = root / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return output_dir


__all__ = [
    "ResolvedSamplingInputs",
    "SamplingRunResult",
    "image_sample_shape",
    "parse_sampler_params",
    "resolve_sampling_inputs",
    "run_sampling",
    "validate_sampling_compatibility",
]
