"""Read-only resolution of checkpoint-backed evaluation subjects."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from stochaflow._component_factory import build_model
from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.evaluation.config import (
    CheckpointSubjectConfig,
    CheckpointWeightVariant,
    _freeze_evaluation_mapping,
    _thaw_evaluation_value,
)
from stochaflow.inference.checkpoint import (
    InferenceCheckpointView,
    build_inference_model_provider,
    checkpoint_epoch_and_step,
    load_checkpoint_config,
    load_stable_checkpoint_snapshot,
    project_inference_checkpoint,
    resolve_checkpoint_path,
)
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION, CheckpointState
from stochaflow.utils.config import ComponentConfig, StochaflowConfig, load_config_dict
from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    extension_plugin_provenance_to_dicts,
    parse_extension_plugin_provenance,
)


def _checkpoint_metadata(
    payload: CheckpointState,
) -> tuple[
    Mapping[str, Any],
    tuple[ExtensionPluginProvenance, ...],
    Mapping[str, Any],
    Mapping[str, Any],
    DataArtifactBindings | None,
]:
    raw_value = cast(object, payload.get("metadata"))
    if type(raw_value) is not dict:
        raise TypeError("checkpoint metadata must be an exact dictionary")
    raw = cast(dict[str, Any], raw_value)
    provenance = parse_extension_plugin_provenance(raw.get("extension_plugins"))
    selected_components = _optional_metadata_mapping(
        raw.get("selected_components"),
        path="checkpoint metadata.selected_components",
    )
    lineage = _optional_metadata_mapping(
        raw.get("lineage"),
        path="checkpoint metadata.lineage",
    )
    data_artifacts_value = raw.get("data_artifacts")
    data_artifacts = (
        None
        if data_artifacts_value is None
        else DataArtifactBindings.from_dict(
            data_artifacts_value,
            path="checkpoint metadata.data_artifacts",
        )
    )
    metadata = _freeze_evaluation_mapping(
        raw,
        path="checkpoint metadata",
    )
    return metadata, provenance, selected_components, lineage, data_artifacts


def _optional_metadata_mapping(
    value: object,
    *,
    path: str,
) -> Mapping[str, Any]:
    if value is None:
        return _freeze_evaluation_mapping({}, path=path)
    return _freeze_evaluation_mapping(value, path=path)


@dataclass(frozen=True, slots=True)
class CheckpointSubjectInputs:
    """Preflighted checkpoint identity and retained inference-only state."""

    path: Path
    content_digest: str
    format_version: int
    epoch: int
    global_step: int
    requested_weights: CheckpointWeightVariant
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    extension_provenance: tuple[ExtensionPluginProvenance, ...]
    selected_components: Mapping[str, Any]
    lineage: Mapping[str, Any]
    data_artifacts: DataArtifactBindings | None
    _checkpoint_view: InferenceCheckpointView = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        path_value = cast(object, self.path)
        if not isinstance(path_value, Path) or not path_value.is_absolute():
            raise ValueError("checkpoint subject path must be an absolute Path")
        digest = cast(object, self.content_digest)
        if (
            type(digest) is not str
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "checkpoint subject content_digest must be a lowercase SHA-256 digest"
            )
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "checkpoint subject format_version must match the supported version"
            )
        checkpoint_epoch_and_step(
            {"epoch": self.epoch, "global_step": self.global_step}
        )
        if self.requested_weights not in {"raw", "ema"}:
            raise ValueError("checkpoint subject requested_weights must be raw or ema")
        object.__setattr__(
            self,
            "config",
            _freeze_evaluation_mapping(
                self.config,
                path="checkpoint subject config",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_evaluation_mapping(
                self.metadata,
                path="checkpoint subject metadata",
            ),
        )
        object.__setattr__(
            self,
            "selected_components",
            _freeze_evaluation_mapping(
                self.selected_components,
                path="checkpoint subject selected_components",
            ),
        )
        object.__setattr__(
            self,
            "lineage",
            _freeze_evaluation_mapping(
                self.lineage,
                path="checkpoint subject lineage",
            ),
        )
        provenance_value = cast(object, self.extension_provenance)
        if type(provenance_value) is not tuple:
            raise TypeError("checkpoint subject extension_provenance must be a tuple")
        validated_provenance = parse_extension_plugin_provenance(
            extension_plugin_provenance_to_dicts(self.extension_provenance)
        )
        object.__setattr__(self, "extension_provenance", validated_provenance)
        artifacts_value = cast(object, self.data_artifacts)
        if artifacts_value is not None and not isinstance(
            artifacts_value,
            DataArtifactBindings,
        ):
            raise TypeError(
                "checkpoint subject data_artifacts must be DataArtifactBindings or None"
            )
        if not isinstance(cast(object, self._checkpoint_view), dict):
            raise TypeError("checkpoint subject inference view must be a mapping")

    def training_config_copy(self) -> StochaflowConfig:
        """Return a fresh mutable config for extension preflight and activation."""

        raw = _thaw_evaluation_value(self.config)
        if type(raw) is not dict:
            raise TypeError("checkpoint subject config snapshot must thaw to a mapping")
        return load_config_dict(cast(dict[str, Any], raw))


@dataclass(frozen=True, slots=True)
class ResolvedCheckpointSubject:
    """Frozen checkpoint identity paired with one explicitly selected model."""

    inputs: CheckpointSubjectInputs = field(repr=False)
    model: nn.Module = field(repr=False, compare=False)
    resolved_weights: CheckpointWeightVariant

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.inputs), CheckpointSubjectInputs):
            raise TypeError("resolved checkpoint subject inputs are invalid")
        if not isinstance(cast(object, self.model), nn.Module):
            raise TypeError("resolved checkpoint subject model must be an nn.Module")
        if self.resolved_weights not in {"raw", "ema"}:
            raise ValueError("resolved checkpoint subject weights must be raw or ema")
        if self.resolved_weights != self.inputs.requested_weights:
            raise ValueError(
                "resolved checkpoint subject weights must match the explicit request"
            )

    @property
    def kind(self) -> Literal["checkpoint"]:
        """Return the subject discriminator."""

        return "checkpoint"

    @property
    def path(self) -> Path:
        """Return the canonical checkpoint path."""

        return self.inputs.path

    @property
    def content_digest(self) -> str:
        """Return the digest of the exact checkpoint file."""

        return self.inputs.content_digest

    @property
    def format_version(self) -> int:
        """Return the portable checkpoint schema version."""

        return self.inputs.format_version

    @property
    def epoch(self) -> int:
        """Return the checkpoint epoch identity."""

        return self.inputs.epoch

    @property
    def global_step(self) -> int:
        """Return the checkpoint optimizer-step identity."""

        return self.inputs.global_step

    @property
    def requested_weights(self) -> CheckpointWeightVariant:
        """Return the explicit weight variant requested by evaluation config."""

        return self.inputs.requested_weights

    @property
    def config(self) -> Mapping[str, Any]:
        """Return the normalized, deeply immutable training configuration."""

        return self.inputs.config

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return the deeply immutable checkpoint metadata snapshot."""

        return self.inputs.metadata

    @property
    def extension_provenance(self) -> tuple[ExtensionPluginProvenance, ...]:
        """Return checkpoint-declared extension provenance."""

        return self.inputs.extension_provenance

    @property
    def selected_components(self) -> Mapping[str, Any]:
        """Return checkpoint-declared selected component identities."""

        return self.inputs.selected_components

    @property
    def lineage(self) -> Mapping[str, Any]:
        """Return checkpoint-declared training lineage."""

        return self.inputs.lineage

    @property
    def data_artifacts(self) -> DataArtifactBindings | None:
        """Return checkpoint-bound data artifact identities, if declared."""

        return self.inputs.data_artifacts

    @property
    def identity(self) -> Mapping[str, Any]:
        """Return the portable, deeply immutable subject identity."""

        return _freeze_evaluation_mapping(
            {
                "kind": self.kind,
                "path": str(self.path),
                "sha256": self.content_digest,
                "format_version": self.format_version,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "requested_weights": self.requested_weights,
                "resolved_weights": self.resolved_weights,
                "extension_plugins": extension_plugin_provenance_to_dicts(
                    self.extension_provenance
                ),
                "selected_components": self.selected_components,
                "lineage": self.lineage,
                "data_artifacts": (
                    self.data_artifacts.to_dict()
                    if self.data_artifacts is not None
                    else None
                ),
            },
            path="resolved checkpoint subject identity",
        )


def load_checkpoint_subject(
    subject: CheckpointSubjectConfig,
    *,
    base_dir: Path | None = None,
) -> CheckpointSubjectInputs:
    """Safely preflight one v12 checkpoint without constructing runtime assets."""

    if not isinstance(cast(object, subject), CheckpointSubjectConfig):
        raise TypeError("checkpoint subject config must be CheckpointSubjectConfig")
    declared_path = subject.path
    if not declared_path.is_absolute():
        if base_dir is None:
            raise ValueError(
                "relative checkpoint subject paths require an explicit base_dir"
            )
        resolved_base = Path(base_dir).resolve(strict=True)
        if not resolved_base.is_dir():
            raise NotADirectoryError(
                f"checkpoint subject base_dir is not a directory: {resolved_base}"
            )
        declared_path = resolved_base / declared_path
    path = resolve_checkpoint_path(declared_path).resolve(strict=True)
    payload, content_digest = load_stable_checkpoint_snapshot(path)
    epoch, global_step = checkpoint_epoch_and_step(payload)
    training_config = load_checkpoint_config(payload)
    checkpoint_view = project_inference_checkpoint(payload)
    metadata, provenance, selected_components, lineage, data_artifacts = (
        _checkpoint_metadata(payload)
    )
    return CheckpointSubjectInputs(
        path=path,
        content_digest=content_digest,
        format_version=CHECKPOINT_FORMAT_VERSION,
        epoch=epoch,
        global_step=global_step,
        requested_weights=subject.weights,
        config=_freeze_evaluation_mapping(
            training_config.to_dict(),
            path="checkpoint config",
        ),
        metadata=metadata,
        extension_provenance=provenance,
        selected_components=selected_components,
        lineage=lineage,
        data_artifacts=data_artifacts,
        _checkpoint_view=checkpoint_view,
    )


def resolve_checkpoint_subject(
    inputs: CheckpointSubjectInputs,
    *,
    device: str | torch.device,
    resolved_config: StochaflowConfig | None = None,
    model_factory: Callable[[ComponentConfig], nn.Module] = build_model,
) -> ResolvedCheckpointSubject:
    """Construct only the explicitly selected primary inference model."""

    if not isinstance(cast(object, inputs), CheckpointSubjectInputs):
        raise TypeError("checkpoint subject inputs must be CheckpointSubjectInputs")
    config = resolved_config or inputs.training_config_copy()
    if not isinstance(cast(object, config), StochaflowConfig):
        raise TypeError("resolved checkpoint training config must be StochaflowConfig")
    provider = build_inference_model_provider(
        config,
        inputs._checkpoint_view,
        device=torch.device(device),
        model_factory=model_factory,
    )
    model, resolved_label = provider.resolve(inputs.requested_weights)
    if resolved_label not in {"raw", "ema"}:
        raise ValueError("inference provider returned an unsupported weight variant")
    model.eval()
    model.requires_grad_(False)
    return ResolvedCheckpointSubject(
        inputs=inputs,
        model=model,
        resolved_weights=cast(CheckpointWeightVariant, resolved_label),
    )


__all__ = [
    "CheckpointSubjectInputs",
    "ResolvedCheckpointSubject",
    "load_checkpoint_subject",
    "resolve_checkpoint_subject",
]
