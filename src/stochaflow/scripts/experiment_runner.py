"""Shared command-line runner helpers for config-driven experiments."""

import argparse
import hashlib
import math
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import yaml

from stochaflow.data import (
    DataArtifactBindings,
    DataLoaders,
    build_data_loaders,
)
from stochaflow.data.artifact_io import MAX_ARTIFACT_VERIFICATION_WORKERS
from stochaflow.metrics.config import METRIC_TAG_SEGMENT_PATTERN
from stochaflow.scripts.artifact_reporting import (
    RichArtifactVerificationReporter,
)
from stochaflow.scripts.epoch_validation import (
    EvaluationBackedEpochValidator,
)
from stochaflow.scripts.extensions_cli import activate_extensions_for_cli
from stochaflow.training.outcome import (
    CheckpointSelectionKind,
    TrainingRunOutcome,
)
from stochaflow.training.precision import validate_precision_support
from stochaflow.training.reporting import (
    FinalSummary,
    RichTrainingReporter,
    RunSummary,
)
from stochaflow.training.trainer import TrainingFitState
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    CheckpointState,
    ParsedRNGState,
    parse_rng_state,
    restore_rng_state,
)
from stochaflow.utils.config import (
    ConfigError,
    StochaflowConfig,
    load_config,
    load_config_dict,
)
from stochaflow.utils.device import validate_execution_device
from stochaflow.utils.factory import (
    TrainingComponents,
    build_model,
    build_training_components,
    resolve_device,
)
from stochaflow.utils.iterables import try_length
from stochaflow.utils.logging import resolve_local_log_path
from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    ExtensionSelectionPolicy,
    ResolvedExtensions,
    parse_extension_plugin_provenance,
    prepare_extension_plugins,
)
from stochaflow.utils.run_manifest import (
    extension_runtime_metadata,
    selected_training_component_identities,
    write_yaml_manifest,
)
from stochaflow.utils.seed import set_seed

_OBSERVABILITY_SECTIONS = ("diagnostics", "logging")
_OBSERVABILITY_LOGGING_FIELDS = ("log_every", "backends", "torch_logs")
_CONFIG_OVERLAY_AUDIT_FIELDS = frozenset(
    {
        "kind",
        "source_path",
        "source_sha256",
        "sections",
        "logging_fields",
    }
)


def _positive_optional(value: int | None, *, option: str) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{option} must be positive when provided")
    return value


def _artifact_verification_workers(value: int | None) -> int | None:
    workers = _positive_optional(
        value,
        option="--artifact-verification-workers",
    )
    if (
        workers is not None
        and workers > MAX_ARTIFACT_VERIFICATION_WORKERS
    ):
        raise ValueError(
            "--artifact-verification-workers must not exceed "
            f"{MAX_ARTIFACT_VERIFICATION_WORKERS}"
        )
    return workers


@dataclass(frozen=True, slots=True)
class ExperimentRunOptions:
    """Validated runtime options for one config-driven invocation."""

    num_epochs: int
    max_train_batches: int | None
    max_validation_batches: int | None
    max_test_batches: int | None
    deterministic: bool
    show_progress: bool
    artifact_verification_workers: int | None
    resume_checkpoint: Path | None
    device: str | None

    @classmethod
    def from_namespace(
        cls,
        args: argparse.Namespace,
        *,
        configured_num_epochs: int,
        configured_show_progress: bool,
    ) -> "ExperimentRunOptions":
        num_epochs = configured_num_epochs if args.epochs is None else args.epochs
        if num_epochs <= 0:
            raise ValueError("--epochs must be positive when provided")
        force_progress = bool(getattr(args, "progress", False))
        suppress_progress = bool(args.no_progress)
        if force_progress and suppress_progress:
            raise ValueError("--progress and --no-progress are mutually exclusive")
        return cls(
            num_epochs=num_epochs,
            max_train_batches=_positive_optional(
                args.limit_batches,
                option="--limit-batches",
            ),
            max_validation_batches=_positive_optional(
                args.limit_validation_batches,
                option="--limit-validation-batches",
            ),
            max_test_batches=_positive_optional(
                args.limit_test_batches,
                option="--limit-test-batches",
            ),
            deterministic=args.deterministic,
            show_progress=(
                force_progress
                or (configured_show_progress and not suppress_progress)
            ),
            artifact_verification_workers=_artifact_verification_workers(
                getattr(args, "artifact_verification_workers", None),
            ),
            resume_checkpoint=args.resume,
            device=args.device,
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Selected model state and summary values after fitting."""

    final_epoch: int
    final_metrics: dict[str, float]
    latest_checkpoint: Path | None
    best_epoch: int | None
    best_metric_name: str | None
    best_metric_value: float | None
    best_checkpoint: Path | None
    selected_checkpoint: Path | None
    selected_checkpoint_kind: CheckpointSelectionKind | None
    stopped_early: bool


@dataclass(frozen=True, slots=True)
class ResolvedTrainingInputs:
    """Config, extension identity, and optional state for one training run."""

    config: StochaflowConfig
    extensions: ResolvedExtensions
    config_source: str
    checkpoint_path: Path | None
    checkpoint: CheckpointState | None
    startup_cwd: Path
    config_overlays: list[dict[str, Any]]


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common config-driven experiment options to a parser."""

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the experiment config file.",
    )
    input_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Strictly resume from a checkpoint file or run directory using its "
            "saved training config and state."
        ),
    )
    parser.add_argument(
        "--observability-config",
        type=Path,
        default=None,
        help=(
            "Apply a diagnostics/logging-only YAML overlay while strictly "
            "resuming from --resume."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override trainer.device for this run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override experiment.output_dir for this run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override trainer.num_epochs from the config.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Maximum number of training batches per epoch.",
    )
    parser.add_argument(
        "--limit-validation-batches",
        type=int,
        default=None,
        help="Maximum number of validation batches per epoch.",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=None,
        help="Maximum number of test batches for the final evaluation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic Torch behavior where supported.",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Enable Rich progress bars, overriding the saved config when "
            "resuming."
        ),
    )
    progress_group.add_argument(
        "--no-progress",
        action="store_true",
        help=(
            "Disable Rich progress bars, overriding the saved config when "
            "resuming."
        ),
    )
    parser.add_argument(
        "--artifact-verification-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override source.materialization.verification_workers for artifact "
            "hashing (1-8); defaults to the config or min(8, logical CPUs)."
        ),
    )
    parser.add_argument(
        "--force-extension-version-mismatch",
        action="store_true",
        help=(
            "Accept extension version differences after identity validation; "
            "does not bypass checkpoint state compatibility."
        ),
    )
    return parser


def _make_timestamped_output_dir(base_output_dir: str) -> tuple[str, Path]:
    """Create a unique timestamp-based experiment directory."""

    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_output_dir)
    output_dir = base_dir / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.name, output_dir


def _local_log_paths(
    config: StochaflowConfig,
) -> tuple[Path | None, Path | None]:
    """Resolve local logger text and metrics paths from the run configuration."""

    output_dir = Path(config.experiment.output_dir)

    for backend in config.logging.backends:
        if backend.name == "local":
            text_path = resolve_local_log_path(
                output_dir,
                backend.params.get("text_filename", "train.log"),
                field="text_filename",
            )
            metrics_path = resolve_local_log_path(
                output_dir,
                backend.params.get("metrics_filename", "metrics.jsonl"),
                field="metrics_filename",
            )
            return metrics_path, text_path
    return None, None


def _dataset_size(loader: object | None) -> int | None:
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", None)
    return try_length(dataset)


def _effective_steps_per_epoch(
    loaders: DataLoaders,
    *,
    max_batches: int | None,
) -> int:
    steps_per_epoch = loaders.steps_per_epoch
    if steps_per_epoch is None:
        train_length = try_length(loaders.train)
        if train_length is None:
            raise TypeError("training loader does not expose a finite epoch length")
        steps_per_epoch = train_length
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    if steps_per_epoch <= 0:
        raise ValueError("training dataloader must yield at least one batch")
    return steps_per_epoch


def _batch_size(loader: object) -> int | None:
    loader_batch_size = getattr(loader, "batch_size", None)
    if isinstance(loader_batch_size, int):
        return loader_batch_size
    sampler = getattr(loader, "batch_sampler", None)
    for attribute in ("batch_size", "base_batch_size"):
        value = getattr(sampler, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _resolve_monitor(config: StochaflowConfig) -> str:
    return config.trainer.early_stopping.monitor


def _resolve_resume_checkpoint(
    resume: Path | None,
) -> tuple[Path | None, CheckpointState | None]:
    if resume is None:
        return None, None
    if resume.is_file():
        return resume, CheckpointManager.load_payload(resume, map_location="cpu")
    if resume.is_dir():
        return _resolve_resume_directory_checkpoint(resume)
    raise FileNotFoundError(f"checkpoint does not exist: {resume}")


@dataclass(frozen=True, slots=True)
class ResumeCheckpointCandidate:
    """Small validated summary of one atomic directory-resume candidate."""

    path: Path
    epoch: int
    global_step: int
    kind: str
    priority: int
    config: object
    config_source: object
    lineage: object
    extension_plugins: tuple[ExtensionPluginProvenance, ...]
    config_overlays: list[dict[str, Any]]
    data_artifacts: DataArtifactBindings | None
    metrics: dict[str, float]
    fit_state: TrainingFitState
    inherited_from: object


_EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epoch_([0-9]{4,})\.pt$")


def _numbered_checkpoint_epoch(path: Path) -> int:
    """Parse the trainer's zero-padded positive epoch filename."""

    match = _EPOCH_CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"malformed numbered checkpoint filename: {path}")
    epoch = int(match.group(1))
    if epoch <= 0:
        raise ValueError(f"numbered checkpoint epoch must be positive: {path}")
    if path.name != f"epoch_{epoch:04d}.pt":
        raise ValueError(f"noncanonical numbered checkpoint filename: {path}")
    return epoch


def _checkpoint_candidate_directory(root: Path) -> Path:
    """Resolve one checkpoint directory without mixing sibling lineages."""

    root = root.resolve()
    direct = (
        root
        if root.name.casefold() == "checkpoints"
        else root / "checkpoints"
    )
    if direct.is_dir():
        return direct.resolve()
    directories: set[Path] = set()
    for path in root.rglob("*.pt"):
        if (
            not path.is_file()
            or path.parent.name.casefold() != "checkpoints"
        ):
            continue
        if path.name not in {"latest.pt", "best.pt"} and (
            _EPOCH_CHECKPOINT_PATTERN.fullmatch(path.name) is None
        ):
            continue
        directories.add(path.parent.resolve())
    if not directories:
        raise FileNotFoundError(
            "could not find a resumable checkpoint directory under: "
            f"{root}"
        )
    if len(directories) > 1:
        rendered = ", ".join(str(path) for path in sorted(directories))
        raise ValueError(
            "resume directory contains multiple nested checkpoint "
            f"directories; pass one exact run directory or checkpoint: {rendered}"
        )
    return next(iter(directories))


def _candidate_checkpoint_paths(checkpoint_dir: Path) -> tuple[Path, ...]:
    """Return canonical and highest numbered checkpoint paths in one run."""

    paths = [
        path
        for path in (
            checkpoint_dir / "latest.pt",
            checkpoint_dir / "best.pt",
        )
        if path.is_file()
    ]
    numbered: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        if not path.is_file():
            continue
        numbered.append((_numbered_checkpoint_epoch(path), path))
    if numbered:
        paths.append(max(numbered, key=lambda item: item[0])[1])
    if not paths:
        raise FileNotFoundError(
            f"checkpoint directory contains no resumable snapshots: {checkpoint_dir}"
        )
    return tuple(paths)


def _resume_candidate(
    path: Path,
    payload: CheckpointState,
) -> ResumeCheckpointCandidate:
    """Strictly validate and summarize one filename-bound candidate."""

    epoch, global_step, _ = _parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )
    metadata = cast(object, payload.get("metadata"))
    if type(metadata) is not dict:
        raise TypeError("resume candidate metadata must be an exact mapping")
    checkpoint_kind = cast(dict[str, Any], metadata).get("checkpoint_kind")
    if path.name == "latest.pt":
        expected_kind = "latest"
        priority = 3
    elif path.name == "best.pt":
        expected_kind = "best"
        priority = 1
    else:
        filename_epoch = _numbered_checkpoint_epoch(path)
        if filename_epoch != epoch:
            raise ValueError(
                f"numbered checkpoint filename epoch does not match payload: {path}"
            )
        expected_kind = None
        priority = 2
    if checkpoint_kind != expected_kind:
        raise ValueError(
            f"resume checkpoint '{path}' must declare checkpoint_kind="
            f"{expected_kind!r}"
        )
    metadata_mapping = cast(dict[str, Any], metadata)
    return ResumeCheckpointCandidate(
        path=path,
        epoch=epoch,
        global_step=global_step,
        kind="epoch" if expected_kind is None else expected_kind,
        priority=priority,
        config=payload.get("config"),
        config_source=metadata_mapping.get("config_source"),
        lineage=metadata_mapping.get("lineage"),
        extension_plugins=parse_extension_plugin_provenance(
            metadata_mapping.get("extension_plugins")
        ),
        config_overlays=_load_checkpoint_config_overlays(payload),
        data_artifacts=_checkpoint_data_artifacts(
            payload,
            path="resume candidate metadata.data_artifacts",
        ),
        metrics=_checkpoint_epoch_metrics(
            payload.get("metrics"),
            path="resume candidate checkpoint metrics",
        ),
        fit_state=_checkpoint_training_fit_state(payload),
        inherited_from=metadata_mapping.get("inherited_from"),
    )


def _validate_resume_candidate_lineage(
    reference: ResumeCheckpointCandidate,
    candidate: ResumeCheckpointCandidate,
) -> None:
    """Require directory candidates to describe one exact training lineage."""

    if candidate.config != reference.config:
        raise ValueError("resume checkpoint candidates have different configs")
    if candidate.config_source != reference.config_source:
        raise ValueError(
            "resume checkpoint candidates have different config source"
        )
    if candidate.lineage != reference.lineage:
        raise ValueError("resume checkpoint candidates have different lineage")
    if candidate.extension_plugins != reference.extension_plugins:
        raise ValueError(
            "resume checkpoint candidates have different extension provenance"
        )
    if candidate.config_overlays != reference.config_overlays:
        raise ValueError(
            "resume checkpoint candidates have different config overlay history"
        )
    if candidate.data_artifacts != reference.data_artifacts:
        raise ValueError(
            "resume checkpoint candidates have different data artifacts"
        )


def _validate_same_progress_candidates(
    candidates: tuple[ResumeCheckpointCandidate, ...],
) -> None:
    """Reject contradictory snapshots and regressing cross-epoch progress."""

    by_epoch: dict[int, list[ResumeCheckpointCandidate]] = {}
    for candidate in candidates:
        by_epoch.setdefault(candidate.epoch, []).append(candidate)
    for epoch, peers in by_epoch.items():
        global_steps = {candidate.global_step for candidate in peers}
        if len(global_steps) != 1:
            raise ValueError(
                "resume checkpoint candidates disagree on global_step at "
                f"epoch {epoch}"
            )
        if len(peers) < 2:
            continue
        reference_metrics = peers[0].metrics
        reference_state = peers[0].fit_state.to_dict()
        for candidate in peers[1:]:
            metrics = candidate.metrics
            state = candidate.fit_state.to_dict()
            if metrics != reference_metrics or state != reference_state:
                raise ValueError(
                    "resume checkpoint candidates disagree on metrics or "
                    f"training-loop state at epoch {epoch}"
                )
    ordered = sorted(
        (peers[0] for peers in by_epoch.values()),
        key=lambda candidate: candidate.epoch,
    )
    for previous, current in pairwise(ordered):
        if current.global_step < previous.global_step:
            raise ValueError(
                "resume checkpoint candidates regress global_step between "
                f"epochs {previous.epoch} and {current.epoch}"
            )
        _validate_fit_state_transition(previous, current)


def _validate_fit_state_transition(
    previous: ResumeCheckpointCandidate,
    current: ResumeCheckpointCandidate,
) -> None:
    """Require one later candidate to extend the persisted fit state."""

    before = previous.fit_state
    after = current.fit_state
    if (
        before.tracking_enabled != after.tracking_enabled
        or before.monitor_policy != after.monitor_policy
        or before.early_stopping_patience != after.early_stopping_patience
    ):
        raise ValueError(
            "resume checkpoint candidates change the tracking policy between "
            f"epochs {previous.epoch} and {current.epoch}"
        )
    before_validation = before.epoch_validation
    after_validation = after.epoch_validation
    if (before_validation is None) != (after_validation is None):
        raise ValueError(
            "resume checkpoint candidates change epoch validation state "
            f"between epochs {previous.epoch} and {current.epoch}"
        )
    if before_validation is not None and after_validation is not None:
        if before_validation.identity != after_validation.identity:
            raise ValueError(
                "resume checkpoint candidates change epoch validation identity"
            )
        before_results = before_validation.results
        after_results = after_validation.results
        if (
            len(after_results) < len(before_results)
            or after_results[: len(before_results)] != before_results
        ):
            raise ValueError(
                "later resume checkpoint epoch validation history must extend "
                "the earlier exact result prefix"
            )
    if before.stopped_early:
        raise ValueError(
            "resume checkpoint candidates continue after an early-stopping "
            "boundary"
        )


def _resolve_resume_directory_checkpoint(
    root: Path,
) -> tuple[Path, CheckpointState]:
    """Select the highest lineage-consistent atomic snapshot in one run."""

    checkpoint_dir = _checkpoint_candidate_directory(root)
    candidates_list: list[ResumeCheckpointCandidate] = []
    selected: ResumeCheckpointCandidate | None = None
    selected_payload: CheckpointState | None = None
    for path in _candidate_checkpoint_paths(checkpoint_dir):
        payload = CheckpointManager.load_payload(
            path,
            map_location="cpu",
            mmap=True,
        )
        candidate = _resume_candidate(path, payload)
        candidates_list.append(candidate)
        candidate_key = (
            candidate.epoch,
            candidate.global_step,
            candidate.priority,
        )
        selected_key = (
            (selected.epoch, selected.global_step, selected.priority)
            if selected is not None
            else None
        )
        if selected_key is None or candidate_key > selected_key:
            selected = candidate
            selected_payload = payload
        else:
            del payload
    candidates = tuple(candidates_list)
    reference = candidates[0]
    for candidate in candidates[1:]:
        _validate_resume_candidate_lineage(reference, candidate)
    _validate_same_progress_candidates(candidates)
    if (
        len(candidates) == 1
        and candidates[0].kind == "best"
        and candidates[0].inherited_from is not None
    ):
        raise ValueError(
            "run directory contains only an inherited best checkpoint and "
            "does not preserve the selected parent progress; resume the "
            "explicit parent checkpoint instead"
        )
    latest = next(
        (candidate for candidate in candidates if candidate.kind == "latest"),
        None,
    )
    if latest is not None:
        furthest_epoch = max(candidate.epoch for candidate in candidates)
        if furthest_epoch > latest.epoch + 1:
            raise ValueError(
                "resume checkpoint candidates advance more than one epoch "
                "beyond latest.pt"
            )
    assert selected is not None
    assert selected_payload is not None
    return selected.path, selected_payload


def _load_training_checkpoint_config(payload: CheckpointState) -> StochaflowConfig:
    raw_config = cast(object, payload.get("config"))
    if not isinstance(raw_config, dict):
        raise TypeError("checkpoint is missing a valid training config")
    return load_config_dict(raw_config)


def _load_checkpoint_extension_provenance(
    payload: CheckpointState,
) -> tuple[ExtensionPluginProvenance, ...]:
    metadata = cast(object, payload.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint is missing valid metadata")
    return parse_extension_plugin_provenance(metadata.get("extension_plugins"))


def _validate_observability_overlay(
    raw: object,
    *,
    source: Path,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must contain a top-level mapping")
    if not raw:
        raise ConfigError(f"{source} must declare diagnostics and/or logging")
    unknown = [key for key in raw if key not in _OBSERVABILITY_SECTIONS]
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ConfigError(
            "observability config may contain only 'diagnostics' and 'logging'; "
            f"found {rendered}"
        )

    diagnostics = raw.get("diagnostics")
    if "diagnostics" in raw and not isinstance(diagnostics, list):
        raise ConfigError("observability config diagnostics must be a list")
    logging = raw.get("logging")
    if "logging" in raw:
        if not isinstance(logging, dict):
            raise ConfigError("observability config logging must be a mapping")
        if not logging:
            raise ConfigError(
                "observability config logging must declare at least one field"
            )
        unknown_logging = [
            key for key in logging if key not in _OBSERVABILITY_LOGGING_FIELDS
        ]
        if unknown_logging:
            rendered = ", ".join(repr(key) for key in unknown_logging)
            raise ConfigError(
                "observability config logging may contain only 'log_every', "
                f"'backends', and 'torch_logs'; found {rendered}"
            )
    return raw


def _load_observability_overlay(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"observability config does not exist: {source}")
    encoded = source.read_bytes()
    try:
        document = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"observability config must be UTF-8 encoded: {source}"
        ) from exc
    try:
        loaded = yaml.safe_load(document)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid observability config YAML: {source}") from exc
    raw = _validate_observability_overlay(loaded, source=source)
    logging = cast(dict[str, Any], raw.get("logging", {}))
    audit = {
        "kind": "observability",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "sections": [
            name for name in _OBSERVABILITY_SECTIONS if name in raw
        ],
        "logging_fields": [
            name for name in _OBSERVABILITY_LOGGING_FIELDS if name in logging
        ],
    }
    return deepcopy(raw), audit


def _diagnostic_module_names(
    config: StochaflowConfig,
    *,
    source: str,
) -> set[str]:
    names: set[str] = set()
    for index, diagnostic in enumerate(config.diagnostics):
        if "modules" not in diagnostic.params:
            continue
        raw_modules = cast(object, diagnostic.params["modules"])
        if not isinstance(raw_modules, (list, tuple)):
            raise ConfigError(
                f"{source}.diagnostics[{index}].params.modules must be a sequence"
            )
        for module_index, raw_module in enumerate(raw_modules):
            if not isinstance(raw_module, str) or not raw_module.strip():
                raise ConfigError(
                    f"{source}.diagnostics[{index}].params.modules"
                    f"[{module_index}] must be a non-empty string"
                )
            names.add(raw_module)
    return names


def _apply_observability_overlay(
    checkpoint_config: StochaflowConfig,
    overlay: dict[str, Any],
) -> StochaflowConfig:
    raw_overlay = _validate_observability_overlay(
        overlay,
        source=Path("<observability-config>"),
    )
    merged = checkpoint_config.to_dict()
    if "diagnostics" in raw_overlay:
        merged["diagnostics"] = deepcopy(raw_overlay["diagnostics"])
    if "logging" in raw_overlay:
        checkpoint_logging = cast(object, merged.get("logging"))
        if not isinstance(checkpoint_logging, dict):
            raise TypeError("checkpoint config logging must be a mapping")
        overlay_logging = cast(dict[str, Any], raw_overlay["logging"])
        merged["logging"] = {
            **deepcopy(checkpoint_logging),
            **deepcopy(overlay_logging),
        }
    effective = load_config_dict(merged)
    if "diagnostics" in raw_overlay:
        checkpoint_modules = _diagnostic_module_names(
            checkpoint_config,
            source="checkpoint config",
        )
        effective_modules = _diagnostic_module_names(
            effective,
            source="observability config",
        )
        added_modules = sorted(effective_modules - checkpoint_modules)
        if added_modules:
            rendered = ", ".join(repr(name) for name in added_modules)
            raise ConfigError(
                "observability config cannot introduce diagnostics "
                f"params.modules entries: {rendered}"
            )
    return effective


def _canonical_audit_names(
    raw: object,
    *,
    field_name: str,
    allowed: tuple[str, ...],
    allow_empty: bool,
) -> list[str]:
    if not isinstance(raw, list):
        raise TypeError(f"checkpoint metadata.config_overlays {field_name} must be a list")
    expected = [name for name in allowed if name in raw]
    if raw != expected or (not allow_empty and not raw):
        choices = ", ".join(repr(name) for name in allowed)
        raise ValueError(
            f"checkpoint metadata.config_overlays {field_name} must contain "
            f"a canonical subset of {choices}"
        )
    return list(raw)


def _is_absolute_audit_path(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _load_checkpoint_config_overlays(
    payload: CheckpointState,
) -> list[dict[str, Any]]:
    metadata = cast(object, payload.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint is missing valid metadata")
    if "config_overlays" not in metadata:
        return []
    raw_overlays = cast(object, metadata["config_overlays"])
    if not isinstance(raw_overlays, list):
        raise TypeError("checkpoint metadata.config_overlays must be a list")

    overlays: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_overlays):
        if not isinstance(raw_entry, dict):
            raise TypeError(
                f"checkpoint metadata.config_overlays[{index}] must be a mapping"
            )
        fields = set(raw_entry)
        if fields != _CONFIG_OVERLAY_AUDIT_FIELDS:
            missing = sorted(_CONFIG_OVERLAY_AUDIT_FIELDS - fields)
            unknown = sorted(fields - _CONFIG_OVERLAY_AUDIT_FIELDS, key=str)
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(map(str, unknown)))
            raise ValueError(
                f"checkpoint metadata.config_overlays[{index}] has invalid fields: "
                + "; ".join(details)
            )
        if raw_entry["kind"] != "observability":
            raise ValueError(
                f"checkpoint metadata.config_overlays[{index}].kind must be "
                "'observability'"
            )
        source_path = cast(object, raw_entry["source_path"])
        if (
            not isinstance(source_path, str)
            or not source_path
            or not _is_absolute_audit_path(source_path)
        ):
            raise ValueError(
                f"checkpoint metadata.config_overlays[{index}].source_path "
                "must be an absolute path string"
            )
        source_sha256 = cast(object, raw_entry["source_sha256"])
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or source_sha256 != source_sha256.lower()
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError(
                f"checkpoint metadata.config_overlays[{index}].source_sha256 "
                "must be a lowercase SHA-256 hex digest"
            )
        sections = _canonical_audit_names(
            raw_entry["sections"],
            field_name=f"[{index}].sections",
            allowed=_OBSERVABILITY_SECTIONS,
            allow_empty=False,
        )
        logging_fields = _canonical_audit_names(
            raw_entry["logging_fields"],
            field_name=f"[{index}].logging_fields",
            allowed=_OBSERVABILITY_LOGGING_FIELDS,
            allow_empty=True,
        )
        if "logging" not in sections and logging_fields:
            raise ValueError(
                f"checkpoint metadata.config_overlays[{index}].logging_fields "
                "must be empty when logging is not an overlaid section"
            )
        overlays.append(
            deepcopy(
                {
                    "kind": "observability",
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                    "sections": sections,
                    "logging_fields": logging_fields,
                }
            )
        )
    return overlays


def _resolve_training_inputs(args: argparse.Namespace) -> ResolvedTrainingInputs:
    """Resolve one authoritative training config before constructing components."""

    config_path = cast(Path | None, args.config)
    requested_resume = cast(Path | None, args.resume)
    observability_path = cast(
        Path | None,
        getattr(args, "observability_config", None),
    )
    if (config_path is None) == (requested_resume is None):
        raise ValueError("train requires exactly one of --config or --resume")
    if observability_path is not None and requested_resume is None:
        raise ValueError("--observability-config requires --resume")

    startup_cwd = Path.cwd().resolve()
    checkpoint_path, checkpoint = _resolve_resume_checkpoint(requested_resume)
    config_overlays: list[dict[str, Any]] = []
    if checkpoint_path is None:
        assert config_path is not None
        unresolved_config = load_config(config_path)
        plan = prepare_extension_plugins(unresolved_config)
        config_source = "external"
    else:
        assert checkpoint is not None
        checkpoint_config = _load_training_checkpoint_config(checkpoint)
        config_overlays = _load_checkpoint_config_overlays(checkpoint)
        if observability_path is None:
            unresolved_config = checkpoint_config
        else:
            overlay, audit = _load_observability_overlay(observability_path)
            unresolved_config = _apply_observability_overlay(
                checkpoint_config,
                overlay,
            )
            config_overlays.append(audit)
        plan = prepare_extension_plugins(
            unresolved_config,
            expected_provenance=_load_checkpoint_extension_provenance(checkpoint),
            selection_policy=ExtensionSelectionPolicy.EXACT,
        )
        requested_epochs = cast(object, args.epochs)
        if requested_epochs is not None and (
            isinstance(requested_epochs, bool)
            or not isinstance(requested_epochs, int)
            or requested_epochs <= 0
        ):
            raise ValueError("--epochs must be positive when provided")
        target_epoch = (
            unresolved_config.trainer.num_epochs
            if requested_epochs is None
            else requested_epochs
        )
        _preflight_inherited_best(
            checkpoint_path,
            checkpoint,
            target_epoch=target_epoch,
        )
        config_source = "checkpoint"

    extensions = activate_extensions_for_cli(
        plan,
        force_version_mismatch=getattr(
            args,
            "force_extension_version_mismatch",
            False,
        ),
    )
    return ResolvedTrainingInputs(
        config=extensions.config,
        extensions=extensions,
        config_source=config_source,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        startup_cwd=startup_cwd,
        config_overlays=config_overlays,
    )


def _write_resolved_config(config: StochaflowConfig) -> Path:
    """Persist the effective run configuration after all CLI overrides."""

    output_path = Path(config.experiment.output_dir) / "resolved_config.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            config.to_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output_path


def _restore_training_state(
    training: TrainingComponents,
    checkpoint: Path | None,
    payload: CheckpointState | None,
    *,
    target_epoch: int,
) -> int:
    """Restore runtime state and return the first epoch still to run."""

    if checkpoint is None:
        if payload is not None:
            raise ValueError("checkpoint payload was provided without a checkpoint path")
        return 1
    if payload is None:
        raise ValueError("checkpoint payload is required when resuming training")
    restore_cuda_rng = training.trainer.device.type == "cuda"
    restore_mps_rng = training.trainer.device.type == "mps"
    selected_epoch, selected_global_step, rng_state = _parse_strict_resume_state(
        payload,
        require_cuda_compatibility=restore_cuda_rng,
        require_mps_compatibility=restore_mps_rng,
    )
    loaded = training.checkpoint_manager.restore_payload(payload, path=checkpoint)
    if training.ema is not None:
        training.ema.to(training.trainer.device)
    training_loop_state = loaded.metadata.get("training_loop")
    if training_loop_state is None:
        raise TypeError(
            "checkpoint metadata is missing training_loop state required for resume"
        )
    restored_state_payload = cast(
        CheckpointState,
        {**payload, "metadata": loaded.metadata},
    )
    fit_state = _checkpoint_training_fit_state(restored_state_payload)
    if loaded.epoch != selected_epoch or loaded.global_step != selected_global_step:
        raise RuntimeError("restored checkpoint progress differs from its payload")
    training.trainer.global_step = selected_global_step
    start_epoch = selected_epoch + 1
    if start_epoch > target_epoch:
        completed_epoch = start_epoch - 1
        raise ValueError(
            f"checkpoint already completed epoch {completed_epoch}, which meets or "
            f"exceeds the target of {target_epoch}; increase --epochs to continue"
        )
    checkpoint_kind = loaded.metadata.get("checkpoint_kind")
    if checkpoint_kind is not None and (
        not isinstance(checkpoint_kind, str)
        or checkpoint_kind not in {"best", "latest"}
    ):
        raise ValueError(
            "checkpoint metadata.checkpoint_kind must be 'best', 'latest', or null"
        )
    training.trainer.restore_fit_state(
        fit_state.to_dict(),
        best_checkpoint_path=None,
    )
    if training.trainer.stopped_early:
        raise ValueError(
            "checkpoint training already stopped early; strict resume cannot "
            "change the saved early-stopping policy"
        )
    previous_best = _materialize_inherited_best(
        training,
        checkpoint=checkpoint,
        checkpoint_payload=payload,
        checkpoint_kind=checkpoint_kind,
        fit_state=fit_state,
    )
    training.trainer.best_checkpoint_path = previous_best
    if restore_mps_rng and rng_state.torch_mps is None:
        warnings.warn(
            "checkpoint does not contain MPS RNG state; strict resume will "
            "continue without restoring the MPS random stream, so stochastic "
            "results may diverge from an uninterrupted run",
            RuntimeWarning,
            stacklevel=2,
        )
    restore_rng_state(
        rng_state,
        restore_cuda=restore_cuda_rng,
        restore_mps=restore_mps_rng,
    )
    return start_epoch


def _checkpoint_epoch_metrics(
    value: object,
    *,
    path: str,
) -> dict[str, float]:
    """Validate one plain, finite epoch-metric checkpoint mapping."""

    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dictionary")
    metrics: dict[str, float] = {}
    for key, raw_value in cast(dict[object, object], value).items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{path} keys must be non-empty strings")
        if isinstance(raw_value, bool) or not isinstance(
            raw_value,
            (int, float),
        ):
            raise TypeError(f"{path}[{key!r}] must be numeric")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError(f"{path}[{key!r}] must be finite")
        metrics[key] = number
    return metrics


def _checkpoint_configured_final_epoch(payload: CheckpointState) -> int:
    """Return the original final epoch frozen into one checkpoint config."""

    config = cast(object, payload.get("config"))
    if type(config) is not dict:
        raise TypeError(
            "strict resume with include_final epoch validation requires "
            "checkpoint config as an exact mapping"
        )
    trainer = cast(dict[object, object], config).get("trainer")
    if type(trainer) is not dict:
        raise TypeError(
            "strict resume with include_final epoch validation requires "
            "checkpoint config.trainer as an exact mapping"
        )
    final_epoch = cast(dict[object, object], trainer).get("num_epochs")
    if type(final_epoch) is not int or cast(int, final_epoch) <= 0:
        raise TypeError(
            "strict resume with include_final epoch validation requires "
            "checkpoint config.trainer.num_epochs as a positive integer"
        )
    return cast(int, final_epoch)


def _checkpoint_training_fit_state(
    payload: CheckpointState,
) -> TrainingFitState:
    """Parse the versioned strict-resume fit state."""

    metadata = cast(object, payload.get("metadata"))
    if type(metadata) is not dict:
        raise TypeError("strict resume requires checkpoint metadata as an exact mapping")
    training_loop = cast(dict[str, Any], metadata).get("training_loop")
    return TrainingFitState.from_mapping(training_loop)


def _validate_checkpoint_epoch_validation_metrics(
    fit_state: TrainingFitState,
    *,
    checkpoint_epoch: int,
    checkpoint_global_step: int,
    configured_final_epoch: int | None,
    metrics: dict[str, float],
) -> None:
    """Require current metrics to agree with sparse evaluator state."""

    state = fit_state.epoch_validation
    if state is None:
        return
    last_epoch = state.last_evaluated_epoch
    if last_epoch is not None and last_epoch > checkpoint_epoch:
        raise ValueError(
            "strict resume epoch validation state is ahead of the checkpoint "
            "epoch"
        )
    future_steps = [
        result.global_step
        for result in state.results
        if result.global_step > checkpoint_global_step
    ]
    if future_steps:
        raise ValueError(
            "strict resume epoch validation result global_step is ahead of "
            "the checkpoint global_step"
        )
    if last_epoch != state.latest_observation_through(checkpoint_epoch):
        raise ValueError(
            "strict resume epoch validation state is missing a declared "
            "interval or final observation"
        )
    evaluated_now = last_epoch == checkpoint_epoch
    cadence = state.identity.cadence
    scheduled_now = (
        cadence.is_interval_due(checkpoint_epoch)
        or (
            cadence.include_final
            and configured_final_epoch == checkpoint_epoch
        )
        or checkpoint_epoch in state.off_cadence_final_epochs
    )
    if scheduled_now and not evaluated_now:
        raise ValueError(
            "strict resume epoch validation state is missing the scheduled "
            f"observation at epoch {checkpoint_epoch}"
        )
    if evaluated_now and not scheduled_now:
        raise ValueError(
            "strict resume epoch validation state contains an unexpected "
            f"off-cadence observation at epoch {checkpoint_epoch}"
        )
    if (
        evaluated_now
        and state.last_result is not None
        and state.last_result.global_step != checkpoint_global_step
    ):
        raise ValueError(
            "strict resume current epoch validation result global_step must "
            "match the checkpoint global_step"
        )

    declared_keys = set(state.identity.metric_keys)
    current_keys = declared_keys & set(metrics)
    if not evaluated_now:
        if current_keys:
            raise ValueError(
                "strict resume non-evaluated checkpoint metrics contain stale "
                "epoch validation key(s): " + ", ".join(sorted(current_keys))
            )
        return

    missing = sorted(declared_keys - current_keys)
    if missing:
        raise ValueError(
            "strict resume evaluated checkpoint metrics are missing epoch "
            "validation key(s): " + ", ".join(missing)
        )
    mismatched = sorted(
        key for key in declared_keys if metrics[key] != state.last_metrics[key]
    )
    if mismatched:
        raise ValueError(
            "strict resume checkpoint metrics disagree with epoch validation "
            "last_metrics for key(s): " + ", ".join(mismatched)
        )


def _parse_strict_resume_state(
    payload: CheckpointState,
    *,
    require_cuda_compatibility: bool,
    require_mps_compatibility: bool = False,
) -> tuple[int, int, ParsedRNGState]:
    """Validate all resume-only state before mutating the runtime."""

    epoch = cast(object, payload.get("epoch"))
    if type(epoch) is not int or cast(int, epoch) <= 0:
        raise TypeError("strict resume requires checkpoint epoch as a positive integer")
    global_step = cast(object, payload.get("global_step"))
    if type(global_step) is not int or cast(int, global_step) < 0:
        raise TypeError(
            "strict resume requires checkpoint global_step as a non-negative integer"
        )
    rng_state = parse_rng_state(
        payload.get("rng_state"),
        require_cuda_compatibility=require_cuda_compatibility,
        require_mps_compatibility=require_mps_compatibility,
    )
    metadata = cast(object, payload.get("metadata"))
    if type(metadata) is not dict:
        raise TypeError("strict resume requires checkpoint metadata as an exact mapping")
    fit_state = _checkpoint_training_fit_state(payload)
    metrics = _checkpoint_epoch_metrics(
        payload.get("metrics"),
        path="strict resume checkpoint metrics",
    )
    epoch_validation_state = fit_state.epoch_validation
    configured_final_epoch = (
        _checkpoint_configured_final_epoch(payload)
        if epoch_validation_state is not None
        and epoch_validation_state.identity.cadence.include_final
        else None
    )
    _validate_checkpoint_epoch_validation_metrics(
        fit_state,
        checkpoint_epoch=cast(int, epoch),
        checkpoint_global_step=cast(int, global_step),
        configured_final_epoch=configured_final_epoch,
        metrics=metrics,
    )
    monitor = fit_state.monitor
    if monitor is not None and monitor not in metrics:
        state = fit_state.epoch_validation
        if state is None or monitor not in state.identity.metric_keys:
            raise ValueError(
                f"strict resume checkpoint metrics are missing monitor {monitor!r}"
            )
    return cast(int, epoch), cast(int, global_step), rng_state


def _same_metric(left: object, right: float) -> bool:
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        return False
    value = float(left)
    return value == right or (math.isnan(value) and math.isnan(right))


def _validate_best_snapshot_fit_state(
    fit_state: TrainingFitState,
    *,
    label: str,
) -> None:
    if fit_state.best_epoch is None or fit_state.best_metric_value is None:
        raise ValueError(
            f"{label} must record best_epoch and best_metric_value"
        )
    if fit_state.observations_without_improvement != 0:
        raise ValueError(
            f"{label} must record observations_without_improvement=0"
        )
    if fit_state.stopped_early:
        raise ValueError(f"{label} must record stopped_early=false")


def _validate_inherited_best(
    payload: CheckpointState,
    *,
    source: Path,
    selected_payload: CheckpointState,
    best_epoch: int,
    best_metric: float,
    monitor: str,
    mode: str,
) -> None:
    selected_epoch, selected_global_step, _ = _parse_strict_resume_state(
        selected_payload,
        require_cuda_compatibility=False,
    )
    candidate_epoch, candidate_global_step, _ = _parse_strict_resume_state(
        payload,
        require_cuda_compatibility=False,
    )
    candidate_config = cast(object, payload.get("config"))
    selected_config = cast(object, selected_payload.get("config"))
    if (
        not isinstance(candidate_config, dict)
        or not isinstance(selected_config, dict)
        or candidate_config != selected_config
    ):
        raise ValueError(
            f"inherited best checkpoint '{source}' config does not match the "
            "selected checkpoint"
        )
    selected_metadata = cast(object, selected_payload.get("metadata"))
    candidate_metadata = cast(object, payload.get("metadata"))
    if not isinstance(selected_metadata, dict) or not isinstance(
        candidate_metadata, dict
    ):
        raise TypeError("checkpoint metadata must be a mapping")
    selected_provenance = parse_extension_plugin_provenance(
        selected_metadata.get("extension_plugins")
    )
    candidate_provenance = parse_extension_plugin_provenance(
        candidate_metadata.get("extension_plugins")
    )
    if candidate_provenance != selected_provenance:
        raise ValueError(
            f"inherited best checkpoint '{source}' extension provenance does "
            "not match the selected checkpoint"
        )
    selected_overlays = _load_checkpoint_config_overlays(selected_payload)
    candidate_overlays = _load_checkpoint_config_overlays(payload)
    if candidate_overlays != selected_overlays:
        raise ValueError(
            f"inherited best checkpoint '{source}' config overlay history does "
            "not match the selected checkpoint"
        )
    selected_artifacts = _checkpoint_data_artifacts(
        selected_payload,
        path="selected checkpoint metadata.data_artifacts",
    )
    candidate_artifacts = _checkpoint_data_artifacts(
        payload,
        path="inherited best checkpoint metadata.data_artifacts",
    )
    if candidate_artifacts != selected_artifacts:
        raise ValueError(
            f"inherited best checkpoint '{source}' data artifacts do not "
            "match the selected checkpoint"
        )
    if (
        candidate_epoch != best_epoch
        or candidate_epoch > selected_epoch
    ):
        raise ValueError(
            f"inherited best checkpoint '{source}' does not belong to the "
            "selected checkpoint history: best epoch mismatch"
        )
    if candidate_global_step > selected_global_step:
        raise ValueError(
            f"inherited best checkpoint '{source}' does not belong to the "
            "selected checkpoint history: global_step mismatch"
        )
    metadata = candidate_metadata
    if metadata.get("checkpoint_kind") != "best":
        raise ValueError(
            f"inherited best checkpoint '{source}' must declare "
            "metadata.checkpoint_kind='best'"
        )
    candidate_fit_state = _checkpoint_training_fit_state(payload)
    _validate_best_snapshot_fit_state(
        candidate_fit_state,
        label=f"inherited best checkpoint '{source}'",
    )
    expected_identity = {
        "best_epoch": best_epoch,
        "best_metric_value": best_metric,
        "monitor": monitor,
        "mode": mode,
    }
    for field, expected in expected_identity.items():
        actual = getattr(candidate_fit_state, field)
        matches = (
            _same_metric(actual, best_metric)
            if field == "best_metric_value"
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"inherited best checkpoint '{source}' does not belong to the "
                f"selected checkpoint history: {field} mismatch"
            )
    metrics = _checkpoint_epoch_metrics(
        payload.get("metrics"),
        path=f"inherited best checkpoint '{source}' metric snapshot",
    )
    if not _same_metric(metrics.get(monitor), best_metric):
        raise ValueError(
            f"inherited best checkpoint '{source}' does not contain the "
            f"recorded best metric '{monitor}'"
        )


def _preflight_inherited_best(
    checkpoint: Path,
    checkpoint_payload: CheckpointState,
    *,
    target_epoch: int,
) -> None:
    """Validate all pure resume state before importing selected plugins."""

    selected_epoch, _, _ = _parse_strict_resume_state(
        checkpoint_payload,
        require_cuda_compatibility=False,
    )
    if selected_epoch >= target_epoch:
        raise ValueError(
            f"checkpoint already completed epoch {selected_epoch}, which meets or "
            f"exceeds the target of {target_epoch}; increase --epochs to continue"
        )
    metadata = cast(object, checkpoint_payload.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    fit_state = _checkpoint_training_fit_state(checkpoint_payload)
    if fit_state.stopped_early:
        raise ValueError(
            "checkpoint training already stopped early; strict resume cannot "
            "change the saved early-stopping policy"
        )
    checkpoint_kind = cast(object, metadata.get("checkpoint_kind"))
    if checkpoint_kind is not None and (
        not isinstance(checkpoint_kind, str)
        or checkpoint_kind not in {"best", "latest"}
    ):
        raise ValueError(
            "checkpoint metadata.checkpoint_kind must be 'best', 'latest', or null"
        )
    if checkpoint_kind == "best":
        _validate_best_snapshot_fit_state(
            fit_state,
            label="checkpoint metadata.checkpoint_kind='best'",
        )
    _checkpoint_data_artifacts(checkpoint_payload)
    if fit_state.best_epoch is None:
        return
    assert fit_state.best_metric_value is not None
    assert fit_state.monitor is not None
    assert fit_state.mode is not None
    source = (
        checkpoint
        if checkpoint_kind == "best"
        else checkpoint.parent / "best.pt"
    )
    if source == checkpoint:
        candidate_payload = checkpoint_payload
    else:
        if not source.is_file():
            raise FileNotFoundError(
                "strict resume requires the sibling checkpoints/best.pt artifact "
                "recorded by best-tracking state"
            )
        candidate_payload = CheckpointManager.load_payload(source, map_location="cpu")
    _validate_inherited_best(
        candidate_payload,
        source=source,
        selected_payload=checkpoint_payload,
        best_epoch=fit_state.best_epoch,
        best_metric=fit_state.best_metric_value,
        monitor=fit_state.monitor,
        mode=fit_state.mode,
    )


def _materialize_inherited_best(
    training: TrainingComponents,
    *,
    checkpoint: Path,
    checkpoint_payload: CheckpointState,
    checkpoint_kind: object,
    fit_state: TrainingFitState,
) -> Path | None:
    best_epoch = training.trainer.best_epoch
    best_metric = training.trainer.best_metric_value
    if best_epoch is None or best_metric is None:
        return None
    monitor = fit_state.monitor
    mode = fit_state.mode
    if monitor is None or mode is None:
        raise ValueError("checkpoint best-tracking monitor/mode are invalid")

    source = checkpoint if checkpoint_kind == "best" else checkpoint.parent / "best.pt"
    if source == checkpoint:
        candidate_payload = checkpoint_payload
    else:
        if not source.is_file():
            raise FileNotFoundError(
                "strict resume requires the sibling checkpoints/best.pt artifact "
                "recorded by best-tracking state"
            )
        candidate_payload = CheckpointManager.load_payload(source, map_location="cpu")
    _validate_inherited_best(
        candidate_payload,
        source=source,
        selected_payload=checkpoint_payload,
        best_epoch=best_epoch,
        best_metric=best_metric,
        monitor=monitor,
        mode=mode,
    )

    # Loading through the active manager verifies that the inherited state fits
    # the current asset topology.  Restore the selected checkpoint even if that
    # validation or publication fails.
    try:
        training.checkpoint_manager.restore_payload(candidate_payload, path=source)
        checkpoint_dir = training.trainer.checkpoint_dir
        if checkpoint_dir is None:
            raise RuntimeError("strict resume requires a checkpoint directory")
        destination = checkpoint_dir / "best.pt"
        candidate_metadata_value = cast(object, candidate_payload.get("metadata"))
        if not isinstance(candidate_metadata_value, dict):
            raise TypeError("inherited best checkpoint metadata must be a mapping")
        candidate_metadata = candidate_metadata_value
        candidate_fit_state = _checkpoint_training_fit_state(candidate_payload)
        materialized_payload = dict(candidate_payload)
        if training.trainer.checkpoint_config is not None:
            materialized_payload["config"] = training.trainer.checkpoint_config
        materialized_payload["metadata"] = {
            **candidate_metadata,
            **training.trainer.checkpoint_metadata,
            "checkpoint_kind": "best",
            "monitor": monitor,
            "mode": mode,
            "training_loop": candidate_fit_state.to_dict(),
            "inherited_from": str(source),
        }
        CheckpointManager.save_payload(materialized_payload, destination)
    finally:
        training.checkpoint_manager.restore_payload(payload=checkpoint_payload, path=checkpoint)
        if training.ema is not None:
            # Checkpoint payloads are loaded on CPU.  The topology preflight
            # above therefore replaces EMA shadow tensors with CPU clones when
            # it restores both the candidate best and the selected state.
            training.ema.to(training.trainer.device)
    return destination


def _fit_and_select_best(
    training: TrainingComponents,
    config: StochaflowConfig,
    loaders: DataLoaders,
    options: ExperimentRunOptions,
    *,
    start_epoch: int,
    reporter: RichTrainingReporter,
) -> TrainingResult:
    """Fit one run, restore its selected checkpoint, and summarize it."""

    early_stopping = config.trainer.early_stopping
    validation_evaluation = config.trainer.validation_evaluation
    monitor = _resolve_monitor(config)
    if early_stopping.enabled and loaders.validation is None:
        raise ValueError("early stopping requires a validation dataloader")
    if validation_evaluation.enabled and loaders.validation is None:
        raise ValueError(
            "validation Evaluation requires a validation dataloader"
        )
    epoch_validation_evaluator = None
    if validation_evaluation.enabled:
        assert loaders.validation is not None
        epoch_validation_evaluator = EvaluationBackedEpochValidator(
            trainer=training.trainer,
            config=validation_evaluation,
            validation_data=loaders.validation,
            data_identity={
                "source": "training",
                "split": "validation",
                "builder": {
                    "name": config.data.name,
                    "params": deepcopy(config.data.params),
                },
                "artifacts": (
                    loaders.artifact_bindings.to_dict()
                    if loaders.artifact_bindings is not None
                    else None
                ),
            },
            model_factory=build_model,
        )
    history = training.trainer.fit(
        loaders.train,
        num_epochs=options.num_epochs,
        show_progress=options.show_progress,
        max_batches_per_epoch=_effective_steps_per_epoch(
            loaders,
            max_batches=options.max_train_batches,
        ),
        validation_dataloader=loaders.validation,
        max_validation_batches=options.max_validation_batches,
        epoch_validation_evaluator=epoch_validation_evaluator,
        start_epoch=start_epoch,
        close_logger=False,
        early_stopping_patience=(
            early_stopping.patience if early_stopping.enabled else None
        ),
        early_stopping_monitor=monitor,
        early_stopping_mode=early_stopping.mode,
        early_stopping_min_delta=early_stopping.min_delta,
        reporter=reporter,
        track_best=loaders.validation is not None,
    )
    if not history:
        raise RuntimeError("trainer returned no epoch history")

    stopped_early = training.trainer.stopped_early
    final_epoch = start_epoch + len(history) - 1
    final_metrics = dict(history[-1])
    checkpoint_dir = training.trainer.checkpoint_dir
    latest_checkpoint = (
        Path(checkpoint_dir) / "latest.pt"
        if checkpoint_dir is not None
        and (Path(checkpoint_dir) / "latest.pt").is_file()
        else None
    )
    validation_available = loaders.validation is not None
    best_checkpoint = (
        training.trainer.best_checkpoint_path
        if validation_available
        else None
    )
    best_epoch = training.trainer.best_epoch if validation_available else None
    best_metric_name = monitor if validation_available else None
    best_metric_value = (
        training.trainer.best_metric_value
        if validation_available
        else None
    )
    selected_checkpoint = best_checkpoint
    selected_checkpoint_kind = "best" if best_checkpoint is not None else None
    if best_checkpoint is not None:
        training.checkpoint_manager.load(
            best_checkpoint,
            map_location=training.trainer.device,
        )
    elif not validation_available and latest_checkpoint is not None:
        selected_checkpoint = latest_checkpoint
        selected_checkpoint_kind = "final"
    return TrainingResult(
        final_epoch=final_epoch,
        final_metrics=final_metrics,
        latest_checkpoint=latest_checkpoint,
        best_epoch=best_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        best_checkpoint=best_checkpoint,
        selected_checkpoint=selected_checkpoint,
        selected_checkpoint_kind=selected_checkpoint_kind,
        stopped_early=stopped_early,
    )


def _evaluate_test_split(
    training: TrainingComponents,
    loaders: DataLoaders,
    options: ExperimentRunOptions,
    *,
    reporter: RichTrainingReporter,
) -> dict[str, float]:
    if loaders.test is None:
        return {}
    metrics = training.trainer.evaluate_epoch(
        loaders.test,
        show_progress=options.show_progress,
        max_batches=options.max_test_batches,
        metric_prefix="test",
        reporter=reporter,
    )
    validated = _checkpoint_epoch_metrics(
        metrics,
        path="test epoch metrics",
    )
    required = {"loss", "num_batches", "duration_seconds"}
    missing = sorted(required - set(validated))
    if missing:
        raise ValueError(
            "test epoch metrics are missing required field(s): "
            + ", ".join(missing)
        )
    def is_canonical_test_metric(name: str) -> bool:
        segments = name.split("/")
        return (
            len(segments) in {3, 4}
            and segments[:2] == ["test", "metrics"]
            and all(
                METRIC_TAG_SEGMENT_PATTERN.fullmatch(segment) is not None
                for segment in segments[2:]
            )
        )

    unexpected = sorted(
        name
        for name in validated
        if name not in required and not is_canonical_test_metric(name)
    )
    if unexpected:
        raise ValueError(
            "test epoch metrics contain non-canonical field(s): "
            + ", ".join(unexpected)
        )
    num_batches = validated["num_batches"]
    if num_batches <= 0.0 or not num_batches.is_integer():
        raise ValueError("test epoch metrics num_batches must be a positive integer")
    if validated["duration_seconds"] < 0.0:
        raise ValueError(
            "test epoch metrics duration_seconds must be non-negative"
        )
    return {
        "test/loss": validated["loss"],
        **{
            name: value
            for name, value in validated.items()
            if is_canonical_test_metric(name)
        },
        "system/test/num_batches": validated["num_batches"],
        "system/test/duration_seconds": validated["duration_seconds"],
    }


def _checkpoint_data_artifacts(
    checkpoint: CheckpointState,
    *,
    path: str = "checkpoint metadata.data_artifacts",
) -> DataArtifactBindings | None:
    """Parse the optional artifact set stored by one checkpoint."""

    metadata = cast(object, checkpoint.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    raw = metadata.get("data_artifacts")
    if raw is None:
        return None
    return DataArtifactBindings.from_dict(raw, path=path)


def _validate_resume_data_artifacts(
    expected: DataArtifactBindings | None,
    current: DataArtifactBindings | None,
    *,
    strict_resume: bool,
) -> None:
    """Defensively enforce exact artifact identity before run creation."""

    if strict_resume and current != expected:
        raise ValueError(
            "strict resume data artifacts do not match the selected checkpoint"
        )


def _run_single_run(
    config: StochaflowConfig,
    loaders: DataLoaders,
    options: ExperimentRunOptions,
    *,
    extensions: ResolvedExtensions,
    config_source: str,
    checkpoint_payload: CheckpointState | None,
    startup_cwd: Path,
    runtime_options: dict[str, Any],
    data_artifacts: DataArtifactBindings | None = None,
    config_overlays: list[dict[str, Any]] | None = None,
) -> TrainingRunOutcome:
    config.trainer.num_epochs = options.num_epochs
    config.trainer.show_progress = options.show_progress
    if options.device is not None:
        config.trainer.device = options.device
    _write_resolved_config(config)
    lineage = {
        "resumed_from": (
            str(options.resume_checkpoint)
            if options.resume_checkpoint is not None
            else None
        )
    }
    extension_metadata = extension_runtime_metadata(extensions)
    overlay_history = deepcopy(config_overlays or [])
    checkpoint_metadata = {
        **extension_metadata,
        "config_source": config_source,
        "config_overlays": deepcopy(overlay_history),
        "lineage": lineage,
        "startup_cwd": str(startup_cwd),
        "runtime_options": runtime_options,
        "data_artifacts": (
            data_artifacts.to_dict() if data_artifacts is not None else None
        ),
    }
    training = build_training_components(
        config,
        checkpoint_metadata=checkpoint_metadata,
    )
    selected_components = selected_training_component_identities(
        config,
        inference_recipe=(
            training.plan.inference_recipe.name
            if training.plan.inference_recipe is not None
            else None
        ),
    )
    checkpoint_metadata["selected_components"] = selected_components
    training.trainer.checkpoint_metadata = checkpoint_metadata
    manifest_path = Path(config.experiment.output_dir) / "run_manifest.yaml"
    manifest: dict[str, Any] = {
        "kind": "training",
        "status": "running",
        "config_source": config_source,
        "config": config.to_dict(),
        **extension_metadata,
        "selected_components": selected_components,
        "config_overlays": deepcopy(overlay_history),
        "lineage": lineage,
        "startup_cwd": str(startup_cwd),
        "runtime_options": runtime_options,
        "data_artifacts": (
            data_artifacts.to_dict() if data_artifacts is not None else None
        ),
    }
    write_yaml_manifest(manifest_path, manifest)
    reporter = RichTrainingReporter()
    logger = training.logger
    close_attempted = False
    try:
        reporter.on_run_start(
            RunSummary(
                experiment_name=config.experiment.name,
                exp_id=config.experiment.exp_id,
                device=str(training.trainer.device),
                output_dir=config.experiment.output_dir,
                train_size=_dataset_size(loaders.train),
                valid_size=_dataset_size(loaders.validation),
                test_size=(
                    _dataset_size(loaders.test)
                    if config.trainer.test_after_fit
                    else None
                ),
                batch_size=_batch_size(loaders.train),
            )
        )
        start_epoch = _restore_training_state(
            training,
            options.resume_checkpoint,
            checkpoint_payload,
            target_epoch=options.num_epochs,
        )
        result = _fit_and_select_best(
            training,
            config,
            loaders,
            options,
            start_epoch=start_epoch,
            reporter=reporter,
        )
        phase_test_metrics = (
            _evaluate_test_split(
                training,
                loaders,
                options,
                reporter=reporter,
            )
            if config.trainer.test_after_fit
            else {}
        )
        metrics_path, log_path = _local_log_paths(config)
        outcome = TrainingRunOutcome(
            output_dir=Path(config.experiment.output_dir),
            final_epoch=result.final_epoch,
            final_metrics=result.final_metrics,
            latest_checkpoint=result.latest_checkpoint,
            best_epoch=result.best_epoch,
            best_metric_name=result.best_metric_name,
            best_metric_value=result.best_metric_value,
            best_checkpoint=result.best_checkpoint,
            selected_checkpoint=result.selected_checkpoint,
            selected_checkpoint_kind=result.selected_checkpoint_kind,
            stopped_early=result.stopped_early,
            phase_test_metrics=phase_test_metrics,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            log_path=log_path,
        )
        reporter.on_run_end(
            FinalSummary(
                best_epoch=outcome.best_epoch,
                best_metric_name=outcome.best_metric_name,
                best_metric_value=outcome.best_metric_value,
                phase_test_metrics=outcome.phase_test_metrics,
                stopped_early=outcome.stopped_early,
                best_checkpoint=outcome.best_checkpoint,
                selected_checkpoint=outcome.selected_checkpoint,
                selected_checkpoint_kind=outcome.selected_checkpoint_kind,
                output_dir=outcome.output_dir,
                metrics_path=outcome.metrics_path,
                log_path=outcome.log_path,
                artifacts=None,
            )
        )
        close_attempted = True
        logger.close()
        manifest["status"] = "completed"
        manifest["outcome"] = outcome.to_manifest()
        write_yaml_manifest(manifest_path, manifest)
        return outcome
    finally:
        if not close_attempted:
            logger.close()


def run_experiment_from_args(args: argparse.Namespace) -> TrainingRunOutcome:
    """Run one registered data builder as one independent experiment."""

    inputs = _resolve_training_inputs(args)
    config = inputs.config
    options = ExperimentRunOptions.from_namespace(
        args,
        configured_num_epochs=config.trainer.num_epochs,
        configured_show_progress=config.trainer.show_progress,
    )
    configured_output_dir = Path(config.experiment.output_dir)
    output_root = (
        configured_output_dir.parent
        if inputs.checkpoint_path is not None
        else configured_output_dir
    )
    if args.output_dir is not None:
        output_root = args.output_dir
    options = replace(options, resume_checkpoint=inputs.checkpoint_path)
    execution_device = resolve_device(options.device or config.trainer.device)
    validate_execution_device(execution_device)
    validate_precision_support(
        config.trainer.precision,
        execution_device,
    )
    strict_resume = inputs.checkpoint is not None
    expected_artifacts = (
        _checkpoint_data_artifacts(inputs.checkpoint)
        if inputs.checkpoint is not None
        else None
    )
    set_seed(config.experiment.seed, deterministic=options.deterministic)
    artifact_reporter = (
        RichArtifactVerificationReporter()
        if options.show_progress
        else None
    )
    try:
        loaders = build_data_loaders(
            config.data,
            seed=config.experiment.seed,
            strict_resume=strict_resume,
            expected_artifacts=expected_artifacts,
            verification_observer=(
                artifact_reporter.observe
                if artifact_reporter is not None
                else None
            ),
            verification_workers=options.artifact_verification_workers,
        )
    finally:
        if artifact_reporter is not None:
            artifact_reporter.close()
    data_artifacts = loaders.artifact_bindings
    _validate_resume_data_artifacts(
        expected_artifacts,
        data_artifacts,
        strict_resume=strict_resume,
    )
    exp_id, output_dir = _make_timestamped_output_dir(str(output_root))
    config.experiment.exp_id = exp_id
    config.experiment.output_dir = str(output_dir)
    set_seed(config.experiment.seed, deterministic=options.deterministic)
    observability_config = cast(
        Path | None,
        getattr(args, "observability_config", None),
    )
    runtime_options = {
        "device": args.device,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "observability_config": (
            str(observability_config.resolve())
            if observability_config is not None
            else None
        ),
        "epochs": args.epochs,
        "limit_batches": args.limit_batches,
        "limit_validation_batches": args.limit_validation_batches,
        "limit_test_batches": args.limit_test_batches,
        "deterministic": args.deterministic,
        "progress": args.progress,
        "no_progress": args.no_progress,
        "artifact_verification_workers": (
            options.artifact_verification_workers
        ),
        "force_extension_version_mismatch": (
            getattr(args, "force_extension_version_mismatch", False)
        ),
    }
    return _run_single_run(
        config,
        loaders,
        options,
        extensions=inputs.extensions,
        config_source=inputs.config_source,
        checkpoint_payload=inputs.checkpoint,
        startup_cwd=inputs.startup_cwd,
        runtime_options=runtime_options,
        data_artifacts=data_artifacts,
        config_overlays=inputs.config_overlays,
    )
