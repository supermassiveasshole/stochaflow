"""Config-driven checkpoint sampling orchestration."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from stochaflow.processes import Process
from stochaflow.sampling.builder import (
    InferenceModelProvider,
    SamplingBuilder,
    SamplingBuilderContext,
    SamplingOutput,
)
from stochaflow.sampling.sampler import SamplingObservation
from stochaflow.sampling.writers import (
    SamplingArtifactContext,
    SamplingBatch,
    write_sampling_artifacts,
)
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    CheckpointState,
)
from stochaflow.utils.config import (
    ConfigError,
    ExtensionsConfig,
    StochaflowConfig,
    coerce_config_section,
    load_config_dict,
)
from stochaflow.utils.factory import build_model, build_process, resolve_device
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.seed import set_seed


@dataclass(frozen=True, slots=True)
class ResolvedSamplingInputs:
    """Merged configuration and checkpoint payload for one sampling run."""

    config: StochaflowConfig
    checkpoint_path: Path
    checkpoint: CheckpointState


@dataclass(frozen=True, slots=True)
class SamplingRunResult:
    """Paths and runtime choices produced by one sampling invocation."""

    checkpoint_path: Path
    output_dir: Path
    builder_name: str
    device: torch.device
    seed: int
    metadata: dict[str, Any]
    artifacts: dict[str, Path]


def resolve_sampling_inputs(
    *,
    config_path: str | Path | None,
    checkpoint: str | Path | None,
) -> ResolvedSamplingInputs:
    """Resolve checkpoint config and an optional sampling-section override."""

    if config_path is None and checkpoint is None:
        raise ValueError("sample requires --config, --checkpoint, or both")
    external: StochaflowConfig | None = None
    sampling_overlay: dict[str, Any] | None = None
    if config_path is not None:
        raw_external = _load_yaml_mapping(config_path)
        keys = set(raw_external)
        if "sampling" in keys and keys <= {"sampling", "extensions"}:
            if checkpoint is None:
                raise ConfigError(
                    "a sampling-only config requires an explicit --checkpoint"
                )
            sampling_overlay = raw_external
        elif {"experiment", "data", "model", "objective"} <= keys:
            external = load_config_dict(raw_external)
        else:
            raise ConfigError(
                "sampling config must be either a complete Stochaflow config or "
                "contain only 'sampling' and optional 'extensions'"
            )
    checkpoint_path = _resolve_checkpoint_path(checkpoint, external)
    payload = CheckpointManager.load_payload(checkpoint_path, map_location="cpu")
    checkpoint_config = _load_checkpoint_config(payload)
    if sampling_overlay is not None:
        config = _apply_sampling_overlay(checkpoint_config, sampling_overlay)
    elif external is not None:
        validate_sampling_compatibility(external, checkpoint_config)
        config = _merge_external_config(checkpoint_config, external)
    else:
        config = checkpoint_config
    return ResolvedSamplingInputs(config, checkpoint_path, payload)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config does not exist: {source}")
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return raw


def _apply_sampling_overlay(
    checkpoint: StochaflowConfig,
    overlay: dict[str, Any],
) -> StochaflowConfig:
    extensions = coerce_config_section(
        ExtensionsConfig,
        overlay.get("extensions", {}),
        "config.extensions",
    )
    merged = checkpoint.to_dict()
    merged["sampling"] = deepcopy(overlay["sampling"])
    merged["extensions"] = {
        "modules": list(
            dict.fromkeys(
                [
                    *checkpoint.extensions.modules,
                    *extensions.modules,
                ]
            )
        )
    }
    return load_config_dict(merged)


def _merge_external_config(
    checkpoint: StochaflowConfig,
    external: StochaflowConfig,
) -> StochaflowConfig:
    merged = checkpoint.to_dict()
    merged["sampling"] = deepcopy(external.to_dict()["sampling"])
    merged["extensions"] = {
        "modules": list(
            dict.fromkeys(
                [
                    *checkpoint.extensions.modules,
                    *external.extensions.modules,
                ]
            )
        )
    }
    return load_config_dict(merged)


def validate_sampling_compatibility(
    external: StochaflowConfig,
    checkpoint: StochaflowConfig,
) -> None:
    """Reject external configs that do not describe the trained model/process."""

    mismatches: list[str] = []
    if external.model != checkpoint.model:
        mismatches.append("model")
    if external.process != checkpoint.process:
        mismatches.append("process")
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
) -> SamplingRunResult:
    """Run one configured SamplingBuilder and materialize its output."""

    inputs = resolve_sampling_inputs(config_path=config_path, checkpoint=checkpoint)
    config = inputs.config
    declaration = config.sampling.builder
    if declaration is None:
        raise ValueError("sample requires sampling.builder to be configured")
    seed = config.experiment.seed if config.sampling.seed is None else config.sampling.seed
    set_seed(seed)
    device = resolve_device(device_name or config.trainer.device)
    process = _build_checkpointed_process(config, inputs.checkpoint, device=device)
    provider = _build_model_provider(config, inputs.checkpoint, device=device)
    context = SamplingBuilderContext(
        params=declaration.params,
        process=process,
        model_provider=provider,
        device=device,
        seed=seed,
        shape=(tuple(config.sampling.shape) if config.sampling.shape is not None else None),
        num_samples=config.sampling.num_samples,
        batch_size=config.sampling.batch_size,
    )
    builder = cast(
        SamplingBuilder,
        REGISTRIES.sampling_builders.create(declaration.name, context),
    )
    with torch.no_grad():
        output_value = cast(object, builder.run())
    output = validate_sampling_output(output_value)

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_sampling_output_dir(inputs.checkpoint_path)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_sampling_artifacts(
        config.sampling.writers,
        SamplingArtifactContext(target_dir, output.batches, dict(output.metadata)),
    )
    manifest_path = target_dir / "resolved_sampling.yaml"
    manifest = {
        "checkpoint": str(inputs.checkpoint_path),
        "checkpoint_format_version": inputs.checkpoint.get("format_version"),
        "process": asdict(config.process) if config.process is not None else None,
        "builder": asdict(declaration),
        "sampling": asdict(config.sampling),
        "seed": seed,
        "device": str(device),
        "metadata": dict(output.metadata),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    artifacts["config"] = manifest_path
    return SamplingRunResult(
        inputs.checkpoint_path,
        target_dir,
        declaration.name,
        device,
        seed,
        dict(output.metadata),
        artifacts,
    )


def validate_sampling_output(output: object) -> SamplingOutput:
    """Validate the modality-neutral result contract owned by core runtime."""

    if not isinstance(output, SamplingOutput):
        raise TypeError("SamplingBuilder.run() must return SamplingOutput")
    if not output.batches:
        raise ValueError("SamplingOutput.batches must not be empty")
    for batch_index, declared_batch in enumerate(output.batches):
        batch_value = cast(object, declared_batch)
        if not isinstance(batch_value, SamplingBatch):
            raise TypeError(
                f"SamplingOutput.batches[{batch_index}] must be SamplingBatch"
            )
        batch = batch_value
        if batch.trajectory is not None:
            previous = -1
            for observation_index, declared_observation in enumerate(
                batch.trajectory
            ):
                observation_value = cast(object, declared_observation)
                if not isinstance(observation_value, SamplingObservation):
                    raise TypeError(
                        f"trajectory[{observation_index}] must be "
                        "SamplingObservation"
                    )
                observation = observation_value
                if observation.step_index <= previous:
                    raise ValueError(
                        "trajectory observation step indices must increase"
                    )
                previous = observation.step_index
    metadata = cast(object, output.metadata)
    if not isinstance(metadata, Mapping):
        raise TypeError("SamplingOutput.metadata must be a mapping")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("SamplingOutput.metadata keys must be strings")
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise TypeError("SamplingOutput.metadata must be JSON serializable") from exc
    return output


def _resolve_checkpoint_path(
    checkpoint: str | Path | None,
    external_config: StochaflowConfig | None,
) -> Path:
    if checkpoint is None:
        assert external_config is not None
        return CheckpointManager.find_best(external_config.experiment.output_dir)
    path = Path(checkpoint)
    if path.is_dir():
        return CheckpointManager.find_best(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def _load_checkpoint_config(payload: CheckpointState) -> StochaflowConfig:
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version!r} is unsupported; "
            f"expected version {CHECKPOINT_FORMAT_VERSION}"
        )
    raw = payload.get("config")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint does not contain a Stochaflow config")
    return load_config_dict(raw)


def _build_checkpointed_process(
    config: StochaflowConfig,
    payload: CheckpointState,
    *,
    device: torch.device,
) -> Process | None:
    has_state = "process_state_dict" in payload
    if config.process is None:
        if has_state:
            raise ValueError(
                "checkpoint contains 'process_state_dict' but config.process is null"
            )
        return None
    if not has_state:
        raise ValueError("checkpoint is missing required 'process_state_dict'")
    state = payload.get("process_state_dict")
    if not isinstance(state, dict):
        raise TypeError("checkpoint process_state_dict must be a mapping")
    process = build_process(config.process)
    process.load_state_dict(state)
    process.to(device)
    process.eval()
    return process


def _build_model_provider(
    config: StochaflowConfig,
    payload: CheckpointState,
    *,
    device: torch.device,
) -> InferenceModelProvider:
    raw = cast(object, payload.get("model_state_dict"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is missing required 'model_state_dict'")
    ema = cast(object, payload.get("ema_model_state_dict"))
    if ema is not None and not isinstance(ema, dict):
        raise TypeError("checkpoint ema_model_state_dict must be a mapping")
    return InferenceModelProvider(
        model_factory=lambda: build_model(config.model),
        raw_state_dict=raw,
        ema_state_dict=ema,
        device=device,
        prefer_ema=config.ema.enabled and config.ema.use_for_sampling,
    )


def _default_sampling_output_dir(checkpoint_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = checkpoint_path.parent.parent / "samples"
    result = root / timestamp
    suffix = 1
    while result.exists():
        result = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return result


__all__ = [
    "ResolvedSamplingInputs",
    "SamplingRunResult",
    "resolve_sampling_inputs",
    "run_sampling",
    "validate_sampling_compatibility",
    "validate_sampling_output",
]
