"""Config-driven checkpoint sampling orchestration."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from stochaflow.inference.checkpoint import (
    InferenceCheckpointView,
)
from stochaflow.inference.checkpoint import (
    build_checkpointed_process as _build_checkpointed_process,
)
from stochaflow.inference.checkpoint import (
    build_inference_asset_provider as _build_inference_asset_provider,
)
from stochaflow.inference.checkpoint import (
    build_inference_model_provider as _shared_build_model_provider,
)
from stochaflow.inference.checkpoint import (
    checkpoint_epoch_and_step as _checkpoint_epoch_and_step,
)
from stochaflow.inference.checkpoint import (
    load_checkpoint_config as _load_checkpoint_config,
)
from stochaflow.inference.checkpoint import (
    load_checkpoint_recipe as _shared_load_checkpoint_recipe,
)
from stochaflow.inference.checkpoint import (
    load_stable_checkpoint_snapshot as _load_stable_checkpoint_snapshot,
)
from stochaflow.inference.checkpoint import (
    project_inference_checkpoint as _shared_checkpoint_view,
)
from stochaflow.inference.checkpoint import (
    resolve_checkpoint_path as _resolve_checkpoint_path,
)
from stochaflow.inference.extensions import prepare_checkpoint_extension_plan
from stochaflow.inference.model import InferenceModelProvider
from stochaflow.sampling.builder import (
    SamplingBuilderContext,
)
from stochaflow.sampling.execution import (
    execute_sampling_builder,
    validate_sampling_output,
)
from stochaflow.sampling.publication import (
    abort_sampling_staging,
    create_sampling_publication_staging,
    publish_sampling_staging,
)
from stochaflow.sampling.writers import (
    SamplingArtifactContext,
    write_sampling_artifacts,
)
from stochaflow.utils.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointState,
    validate_checkpoint_payload,
)
from stochaflow.utils.config import (
    ConfigError,
    SampleConfig,
    SampleInvocationConfig,
    StochaflowConfig,
    load_sample_config,
)
from stochaflow.utils.factory import build_model, resolve_device
from stochaflow.utils.plugins import (
    ExtensionActivationPlan,
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
    parse_extension_plugin_provenance,
    prepare_extension_plugins,
    require_resolved_extensions_for_plan,
)
from stochaflow.utils.run_manifest import (
    extension_runtime_metadata,
    selected_sampling_component_identities,
    write_yaml_manifest,
)
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    resolve_sampling_recipe_params,
    sampling_recipe_to_dict,
)
from stochaflow.utils.seed import set_seed

SamplingCheckpointView = InferenceCheckpointView


def _sampling_checkpoint_view(
    payload: CheckpointState,
) -> SamplingCheckpointView:
    """Project the shared inference view used by the sampling operation."""

    return _shared_checkpoint_view(payload)


def _load_checkpoint_recipe(payload: CheckpointState) -> SamplingRecipe:
    """Load a checkpoint recipe with sampling-specific error language."""

    try:
        return _shared_load_checkpoint_recipe(payload)
    except ValueError as exc:
        if str(exc) == "checkpoint does not support task inference":
            raise ValueError("checkpoint does not support sampling") from exc
        raise


def _build_model_provider(
    config: StochaflowConfig,
    payload: SamplingCheckpointView,
    *,
    device: torch.device,
) -> InferenceModelProvider:
    """Build the shared provider through sampling's patchable model factory."""

    return _shared_build_model_provider(
        config,
        payload,
        device=device,
        model_factory=build_model,
    )


@dataclass(frozen=True, slots=True)
class SamplingCheckpointIdentity:
    """Stable content and training-progress identity for one checkpoint."""

    path: Path
    sha256: str
    format_version: int
    epoch: int
    global_step: int

    def __post_init__(self) -> None:
        if (
            not isinstance(cast(object, self.path), Path)
            or not self.path.is_absolute()
        ):
            raise ValueError("sampling checkpoint identity path must be absolute")
        if type(self.sha256) is not str:
            raise ValueError(
                "sampling checkpoint identity sha256 must be a lowercase digest"
            )
        if (
            len(self.sha256) != 64
            or self.sha256 != self.sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError(
                "sampling checkpoint identity sha256 must be a lowercase digest"
            )
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "sampling checkpoint identity format_version is unsupported"
            )
        if type(self.epoch) is not int or self.epoch <= 0:
            raise ValueError(
                "sampling checkpoint identity epoch must be a positive integer"
            )
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError(
                "sampling checkpoint identity global_step must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a portable manifest projection of this identity."""

        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "format_version": self.format_version,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSamplingInputs:
    """Independent checkpoint and invocation authorities for one sampling run."""

    checkpoint_config: StochaflowConfig
    sample_config: SampleInvocationConfig
    checkpoint_identity: SamplingCheckpointIdentity
    checkpoint: SamplingCheckpointView
    recipe: SamplingRecipe
    sample_config_path: Path
    extension_plan: ExtensionActivationPlan

    @property
    def checkpoint_path(self) -> Path:
        """Return the canonical checkpoint path bound during preflight."""

        return self.checkpoint_identity.path


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
    """Resolve one checkpoint recipe and a complete sample invocation."""

    if checkpoint is None:
        raise ValueError("sample requires an explicit --checkpoint")
    if config_path is None:
        raise ValueError("sample requires an explicit --config")
    sample_config_path = Path(config_path).resolve()
    sample_config = load_sample_config(sample_config_path)
    checkpoint_path = _resolve_checkpoint_path(checkpoint).resolve(strict=True)
    payload, checkpoint_sha256 = _load_stable_checkpoint_snapshot(checkpoint_path)
    payload = validate_checkpoint_payload(payload, source=checkpoint_path)
    checkpoint_config = _load_checkpoint_config(payload)
    recipe = _load_checkpoint_recipe(payload)
    metadata = cast(object, payload.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint is missing valid metadata")
    expected_provenance = parse_extension_plugin_provenance(
        metadata.get("extension_plugins")
    )
    _sampling_recipe_params(recipe, sample_config.sample)
    extension_plan = prepare_checkpoint_extension_plan(
        checkpoint_config,
        additions=tuple(sample_config.extensions.plugins or ()),
        expected_provenance=expected_provenance,
        plan_factory=prepare_extension_plugins,
    )
    inference_payload = _sampling_checkpoint_view(payload)
    checkpoint_epoch, checkpoint_global_step = _checkpoint_epoch_and_step(payload)
    checkpoint_identity = SamplingCheckpointIdentity(
        path=checkpoint_path,
        sha256=checkpoint_sha256,
        format_version=CHECKPOINT_FORMAT_VERSION,
        epoch=checkpoint_epoch,
        global_step=checkpoint_global_step,
    )
    return ResolvedSamplingInputs(
        checkpoint_config,
        sample_config,
        checkpoint_identity,
        inference_payload,
        recipe,
        sample_config_path,
        extension_plan,
    )


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

    if not isinstance(cast(object, inputs), ResolvedSamplingInputs):
        raise TypeError("sampling inputs must be ResolvedSamplingInputs")
    if not isinstance(cast(object, extensions), ResolvedExtensions):
        raise TypeError("sampling extensions must be ResolvedExtensions")
    require_resolved_extensions_for_plan(inputs.extension_plan, extensions)
    checkpoint_config = inputs.checkpoint_config
    sample = inputs.sample_config.sample
    seed = sample.seed
    set_seed(seed)
    device = resolve_device(device_name or "auto")
    process = _build_checkpointed_process(
        checkpoint_config,
        inputs.checkpoint,
        device=device,
    )
    provider = _build_model_provider(
        checkpoint_config,
        inputs.checkpoint,
        device=device,
    )
    inference_assets = _build_inference_asset_provider(
        inputs.checkpoint,
        device=device,
    )
    recipe_params = _sampling_recipe_params(inputs.recipe, sample)
    context = SamplingBuilderContext(
        params=recipe_params,
        process=process,
        model_provider=provider,
        device=device,
        seed=seed,
        shape=(tuple(sample.shape) if sample.shape is not None else None),
        num_samples=sample.num_samples,
        batch_size=sample.batch_size,
        inference_assets=inference_assets,
    )
    output = execute_sampling_builder(inputs.recipe.name, context)

    requested_target_dir = (
        Path(output_dir)
        if output_dir is not None
        else _default_sampling_output_dir(inputs.checkpoint_path)
    )
    publication = create_sampling_publication_staging(requested_target_dir)
    try:
        staged_artifacts = write_sampling_artifacts(
            sample.writers,
            SamplingArtifactContext(
                publication.staging,
                output.batches,
                dict(output.metadata),
            ),
        )
        artifact_names = {
            name: path.relative_to(publication.staging).as_posix()
            for name, path in staged_artifacts.items()
        }
        manifest_path = publication.staging / "resolved_sampling.yaml"
        checkpoint_identity = inputs.checkpoint_identity.to_dict()
        manifest = {
            "kind": "sampling",
            "config_source": str(inputs.sample_config_path),
            "sample_config": asdict(inputs.sample_config),
            "checkpoint": str(inputs.checkpoint_path),
            "checkpoint_format_version": inputs.checkpoint.get("format_version"),
            "checkpoint_identity": checkpoint_identity,
            **extension_runtime_metadata(extensions),
            "selected_components": selected_sampling_component_identities(
                checkpoint_config,
                sample,
                inference_recipe=inputs.recipe.name,
            ),
            "lineage": {"checkpoint": checkpoint_identity},
            "startup_cwd": str(
                Path.cwd().resolve()
                if startup_cwd is None
                else Path(startup_cwd).resolve()
            ),
            "runtime_options": {
                "device": device_name,
                "output_dir": str(output_dir) if output_dir is not None else None,
            },
            "process": (
                asdict(checkpoint_config.process)
                if checkpoint_config.process is not None
                else None
            ),
            "recipe": sampling_recipe_to_dict(inputs.recipe),
            "sample": asdict(sample),
            "seed": seed,
            "device": str(device),
            "metadata": dict(output.metadata),
            "artifacts": artifact_names,
        }
        write_yaml_manifest(manifest_path, manifest)
        target_dir = publish_sampling_staging(publication)
    except BaseException:
        abort_sampling_staging(publication)
        raise
    artifacts = {
        name: target_dir / relative_path
        for name, relative_path in artifact_names.items()
    }
    artifacts["config"] = target_dir / "resolved_sampling.yaml"
    return SamplingRunResult(
        inputs.checkpoint_path,
        target_dir,
        inputs.recipe.name,
        device,
        seed,
        dict(output.metadata),
        artifacts,
    )


def _sampling_recipe_params(
    recipe: SamplingRecipe,
    sample: SampleConfig,
) -> dict[str, Any]:
    try:
        return resolve_sampling_recipe_params(
            recipe,
            options=sample.options,
            sampler=(
                asdict(sample.sampler)
                if sample.sampler is not None
                else None
            ),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


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
    "SamplingCheckpointIdentity",
    "SamplingRunResult",
    "resolve_sampling_inputs",
    "run_resolved_sampling",
    "run_sampling",
    "validate_sampling_output",
]
