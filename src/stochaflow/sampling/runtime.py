"""Config-driven checkpoint sampling orchestration."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Required, TypedDict, cast

import torch
import yaml

from stochaflow.processes.base import Process
from stochaflow.sampling.assets import InferenceAssetProvider
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
    InferenceAssetDescriptor,
    validate_inference_asset_descriptors,
)
from stochaflow.utils.config import (
    ConfigError,
    ExtensionsConfig,
    ParsedSampleRequest,
    StochaflowConfig,
    apply_sample_request,
    coerce_config_section,
    load_config_dict,
    parse_sample_request,
)
from stochaflow.utils.device import move_module_to_device
from stochaflow.utils.factory import build_model, build_process, resolve_device
from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionIdentityError,
    ExtensionSelectionPolicy,
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
    parse_extension_plugin_provenance,
    prepare_extension_plugins,
)
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.run_manifest import (
    extension_runtime_metadata,
    selected_component_identities,
    write_yaml_manifest,
)
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    resolve_sampling_recipe_params,
    sampling_recipe_from_dict,
    sampling_recipe_to_dict,
)
from stochaflow.utils.seed import set_seed

_MISSING_EXTENSIONS = object()


class SamplingCheckpointView(TypedDict, total=False):
    """Validated inference-only state retained from a complete checkpoint."""

    format_version: int
    config: dict[str, Any]
    inference_recipe: dict[str, Any] | None
    metadata: dict[str, Any]
    model_state_dict: dict[str, Any]
    ema_model_state_dict: dict[str, Any]
    process_state_dict: dict[str, Any]
    inference_asset_descriptors: Required[
        dict[str, InferenceAssetDescriptor]
    ]
    inference_asset_state_dicts: Required[dict[str, dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ResolvedSamplingInputs:
    """Merged config and validated inference-only state for one sampling run."""

    config: StochaflowConfig
    checkpoint_path: Path
    checkpoint: SamplingCheckpointView
    recipe: SamplingRecipe
    config_source: str
    extension_plan: ExtensionActivationPlan


@dataclass(frozen=True, slots=True)
class SamplingRunResult:
    """Paths and runtime choices produced by one sampling invocation."""

    checkpoint_path: Path
    output_dir: Path
    recipe_name: str
    device: torch.device
    seed: int
    metadata: dict[str, Any]
    artifacts: dict[str, Path]


def resolve_sampling_inputs(
    *,
    config_path: str | Path | None,
    checkpoint: str | Path | None,
) -> ResolvedSamplingInputs:
    """Resolve one checkpoint recipe and an optional partial sample request."""

    if checkpoint is None:
        raise ValueError("sample requires an explicit --checkpoint")
    request: ParsedSampleRequest | None = None
    raw_extensions: object = _MISSING_EXTENSIONS
    if config_path is not None:
        raw_external = _load_yaml_mapping(config_path)
        keys = set(raw_external)
        if not keys or not keys <= {"sampling", "extensions"}:
            raise ConfigError(
                "sample request config may contain only 'sampling' and "
                "optional 'extensions'"
            )
        request = parse_sample_request(raw_external.get("sampling", {}))
        if "extensions" in raw_external:
            raw_extensions = raw_external["extensions"]
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    payload = CheckpointManager.load_payload(checkpoint_path, map_location="cpu")
    checkpoint_config = _load_checkpoint_config(payload)
    recipe = _load_checkpoint_recipe(payload)
    metadata = cast(object, payload.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint is missing valid metadata")
    expected_provenance = parse_extension_plugin_provenance(
        metadata.get("extension_plugins")
    )
    selection_policy = ExtensionSelectionPolicy.EXACT
    parsed_request = request or parse_sample_request({})
    config, added_plugins = _apply_sample_request(
        checkpoint_config,
        parsed_request,
        raw_extensions=raw_extensions,
        expected_plugin_names=tuple(
            provenance.name for provenance in expected_provenance
        ),
    )
    _sampling_recipe_params(recipe, config)
    if request is not None:
        config_source = "sample-request"
        if added_plugins:
            selection_policy = ExtensionSelectionPolicy.INTERSECTION
    else:
        config_source = "checkpoint"
    extension_plan = prepare_extension_plugins(
        config,
        expected_provenance=expected_provenance,
        selection_policy=selection_policy,
    )
    selected_plugin_names = {
        provenance.name for provenance in extension_plan.provenance
    }
    missing_required_plugins = sorted(
        {
            provenance.name for provenance in expected_provenance
        }
        - selected_plugin_names
    )
    if missing_required_plugins:
        raise ExtensionIdentityError(
            "sample request is missing checkpoint-required extension plugin(s): "
            + ", ".join(missing_required_plugins)
        )
    inference_payload = _sampling_checkpoint_view(payload)
    return ResolvedSamplingInputs(
        extension_plan.config,
        checkpoint_path,
        inference_payload,
        recipe,
        config_source,
        extension_plan,
    )


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config does not exist: {source}")
    with source.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return raw


def _apply_sample_request(
    checkpoint: StochaflowConfig,
    request: ParsedSampleRequest,
    *,
    raw_extensions: object,
    expected_plugin_names: tuple[str, ...],
) -> tuple[StochaflowConfig, bool]:
    merged = checkpoint.to_dict()
    merged["sampling"] = asdict(
        apply_sample_request(checkpoint.sampling, request)
    )
    added_plugins = False
    resolved_plugins = list(expected_plugin_names)
    checkpoint_plugins = checkpoint.extensions.plugins
    if checkpoint_plugins is not None:
        missing = sorted(set(expected_plugin_names) - set(checkpoint_plugins))
        unexpected = sorted(set(checkpoint_plugins) - set(expected_plugin_names))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(
                    "missing provenance-required plugin(s): " + ", ".join(missing)
                )
            if unexpected:
                details.append(
                    "unproven config-only plugin(s): " + ", ".join(unexpected)
                )
            raise ExtensionIdentityError(
                "checkpoint config extension selection conflicts with checkpoint "
                "provenance: " + "; ".join(details)
            )
    if raw_extensions is not _MISSING_EXTENSIONS:
        if not isinstance(raw_extensions, dict):
            raise ConfigError("config.extensions must be a mapping")
        extensions = cast(
            ExtensionsConfig,
            coerce_config_section(
                ExtensionsConfig,
                raw_extensions,
                "config.extensions",
            ),
        )
        if "plugins" in raw_extensions:
            additions = extensions.plugins
            if additions is None:
                raise ConfigError(
                    "sample request extensions.plugins must be an explicit list"
                )
            for plugin in additions:
                if plugin not in resolved_plugins:
                    resolved_plugins.append(plugin)
                    added_plugins = True
    merged["extensions"] = {"plugins": resolved_plugins}
    config = load_config_dict(merged)
    selected = config.extensions.plugins
    if selected is not None:
        missing = sorted(set(expected_plugin_names) - set(selected))
        if missing:
            raise ConfigError(
                "sample request cannot remove checkpoint-required extension "
                "plugin(s): " + ", ".join(missing)
            )
    return config, added_plugins


def run_sampling(
    *,
    config_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    extension_version_policy: ExtensionVersionPolicy = ExtensionVersionPolicy.REJECT,
    extension_acceptance_method: str | None = None,
) -> SamplingRunResult:
    """Execute checkpoint-backed inference and materialize its output."""

    inputs = resolve_sampling_inputs(config_path=config_path, checkpoint=checkpoint)
    extensions = activate_extension_plugins(
        inputs.extension_plan,
        policy=extension_version_policy,
        acceptance_method=extension_acceptance_method,
    )
    return run_resolved_sampling(
        inputs,
        extensions,
        output_dir=output_dir,
        device_name=device_name,
    )


def run_resolved_sampling(
    inputs: ResolvedSamplingInputs,
    extensions: ResolvedExtensions,
    *,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    startup_cwd: str | Path | None = None,
) -> SamplingRunResult:
    """Execute sampling after the caller has explicitly activated extensions."""

    config = extensions.config
    seed = config.experiment.seed if config.sampling.seed is None else config.sampling.seed
    set_seed(seed)
    device = resolve_device(device_name or config.trainer.device)
    process = _build_checkpointed_process(
        config,
        inputs.checkpoint,
        device=device,
    )
    provider = _build_model_provider(config, inputs.checkpoint, device=device)
    inference_assets = _build_inference_asset_provider(
        inputs.checkpoint,
        device=device,
    )
    recipe_params = _sampling_recipe_params(inputs.recipe, config)
    context = SamplingBuilderContext(
        params=recipe_params,
        process=process,
        model_provider=provider,
        device=device,
        seed=seed,
        shape=(tuple(config.sampling.shape) if config.sampling.shape is not None else None),
        num_samples=config.sampling.num_samples,
        batch_size=config.sampling.batch_size,
        inference_assets=inference_assets,
    )
    builder = cast(
        SamplingBuilder,
        REGISTRIES.sampling_builders.create(inputs.recipe.name, context),
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
        "kind": "sampling",
        "config_source": inputs.config_source,
        "config": config.to_dict(),
        "checkpoint": str(inputs.checkpoint_path),
        "checkpoint_format_version": inputs.checkpoint.get("format_version"),
        **extension_runtime_metadata(extensions),
        "selected_components": selected_component_identities(
            config,
            sampling_recipe=inputs.recipe.name,
        ),
        "lineage": {"checkpoint": str(inputs.checkpoint_path)},
        "startup_cwd": str(
            Path.cwd().resolve() if startup_cwd is None else Path(startup_cwd).resolve()
        ),
        "runtime_options": {
            "device": device_name,
            "output_dir": str(output_dir) if output_dir is not None else None,
        },
        "process": asdict(config.process) if config.process is not None else None,
        "recipe": sampling_recipe_to_dict(inputs.recipe),
        "sampling": asdict(config.sampling),
        "seed": seed,
        "device": str(device),
        "metadata": dict(output.metadata),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    write_yaml_manifest(manifest_path, manifest)
    artifacts["config"] = manifest_path
    return SamplingRunResult(
        inputs.checkpoint_path,
        target_dir,
        inputs.recipe.name,
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


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
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
    raw = cast(object, payload.get("config"))
    if raw is None:
        raise ValueError("checkpoint does not contain a Stochaflow config")
    if not isinstance(raw, dict):
        raise TypeError("checkpoint config must be a mapping")
    return load_config_dict(raw)


def _load_checkpoint_recipe(payload: CheckpointState) -> SamplingRecipe:
    if "inference_recipe" not in payload:
        raise ValueError("checkpoint does not contain an inference recipe")
    raw = cast(object, payload["inference_recipe"])
    if raw is None:
        raise ValueError("checkpoint does not support sampling")
    return sampling_recipe_from_dict(raw)


def _sampling_recipe_params(
    recipe: SamplingRecipe,
    config: StochaflowConfig,
) -> dict[str, Any]:
    try:
        return resolve_sampling_recipe_params(
            recipe,
            options=config.sampling.options,
            sampler=(
                asdict(config.sampling.sampler)
                if config.sampling.sampler is not None
                else None
            ),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _sampling_checkpoint_view(payload: CheckpointState) -> SamplingCheckpointView:
    """Drop training-only state before generated outputs begin to accumulate."""

    retained_keys = (
        "format_version",
        "config",
        "inference_recipe",
        "metadata",
        "model_state_dict",
        "ema_model_state_dict",
        "process_state_dict",
    )
    raw_payload = cast(dict[str, Any], payload)
    descriptors = validate_inference_asset_descriptors(
        payload.get("inference_asset_descriptors"),
        path="checkpoint.inference_asset_descriptors",
    )
    projected_states: dict[str, dict[str, Any]] = {}
    if descriptors:
        asset_states_value = cast(
            object,
            payload.get("training_assets_state_dict"),
        )
        if type(asset_states_value) is not dict:
            raise TypeError(
                "checkpoint with inference assets requires an exact "
                "training_assets_state_dict"
            )
        asset_states = cast(dict[object, object], asset_states_value)
        for descriptor in descriptors.values():
            asset_name = descriptor["training_asset_name"]
            state_value = asset_states.get(asset_name)
            if not isinstance(state_value, dict):
                raise TypeError(
                    "checkpoint embedded inference asset state "
                    f"{asset_name!r} must be a state dictionary"
                )
            projected_states[asset_name] = cast(dict[str, Any], state_value)
    view = cast(
        SamplingCheckpointView,
        {key: raw_payload[key] for key in retained_keys if key in raw_payload},
    )
    view["inference_asset_descriptors"] = descriptors
    view["inference_asset_state_dicts"] = projected_states
    return view


def _build_checkpointed_process(
    config: StochaflowConfig,
    payload: SamplingCheckpointView,
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
    move_module_to_device(process, device, role="sampling process")
    process.eval()
    return process


def _build_model_provider(
    config: StochaflowConfig,
    payload: SamplingCheckpointView,
    *,
    device: torch.device,
) -> InferenceModelProvider:
    raw = cast(object, payload.get("model_state_dict"))
    if raw is None:
        raise ValueError("checkpoint is missing required 'model_state_dict'")
    if not isinstance(raw, dict):
        raise TypeError("checkpoint model_state_dict must be a mapping")
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


def _build_inference_asset_provider(
    payload: SamplingCheckpointView,
    *,
    device: torch.device,
) -> InferenceAssetProvider:
    descriptors = cast(
        Mapping[str, InferenceAssetDescriptor],
        payload.get("inference_asset_descriptors", {}),
    )
    state_dicts = cast(
        Mapping[str, Mapping[str, object]],
        payload.get("inference_asset_state_dicts", {}),
    )
    return InferenceAssetProvider(
        descriptors=descriptors,
        state_dicts=state_dicts,
        device=device,
        model_factory=build_model,
    )


def _default_sampling_output_dir(checkpoint_path: Path) -> Path:
    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
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
    "run_resolved_sampling",
    "run_sampling",
    "validate_sampling_output",
]
