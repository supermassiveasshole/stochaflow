"""Shared command-line runner helpers for config-driven experiments."""

import argparse
import gc
import math
import warnings
from collections.abc import Sized
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from stochaflow.data import DataLoaders, build_data_loaders
from stochaflow.sampling.runtime import run_sampling
from stochaflow.scripts.extensions_cli import activate_extensions_for_cli
from stochaflow.training.reporting import (
    FinalSummary,
    RichTrainingReporter,
    RunSummary,
)
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    CheckpointState,
    ParsedRNGState,
    parse_rng_state,
    restore_rng_state,
)
from stochaflow.utils.config import StochaflowConfig, load_config, load_config_dict
from stochaflow.utils.factory import TrainingComponents, build_training_components
from stochaflow.utils.plugins import (
    ExtensionPluginProvenance,
    ExtensionSelectionPolicy,
    ResolvedExtensions,
    parse_extension_plugin_provenance,
    prepare_extension_plugins,
)
from stochaflow.utils.run_manifest import (
    extension_runtime_metadata,
    selected_component_identities,
    write_yaml_manifest,
)
from stochaflow.utils.seed import set_seed


def _positive_optional(value: int | None, *, option: str) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{option} must be positive when provided")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentRunOptions:
    """Validated runtime options for one config-driven invocation."""

    num_epochs: int
    max_train_batches: int | None
    max_validation_batches: int | None
    max_test_batches: int | None
    deterministic: bool
    show_progress: bool
    resume_checkpoint: Path | None
    device: str | None
    sample_after_training: bool

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
            show_progress=configured_show_progress and not args.no_progress,
            resume_checkpoint=args.resume,
            device=args.device,
            sample_after_training=not args.skip_final_sample,
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Selected model state and summary values after fitting."""

    best_epoch: int
    best_loss: float | None
    best_checkpoint: Path | None


@dataclass(frozen=True, slots=True)
class ResolvedTrainingInputs:
    """Config, extension identity, and optional state for one training run."""

    config: StochaflowConfig
    extensions: ResolvedExtensions
    config_source: str
    checkpoint_path: Path | None
    checkpoint: CheckpointState | None
    startup_cwd: Path


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
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars.",
    )
    parser.add_argument(
        "--force-extension-version-mismatch",
        action="store_true",
        help=(
            "Accept extension version differences after identity validation; "
            "does not bypass checkpoint state compatibility."
        ),
    )
    parser.add_argument(
        "--skip-final-sample",
        action="store_true",
        help="Skip the best-checkpoint acceptance sample after training.",
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


def _local_log_paths(config: StochaflowConfig) -> tuple[Path, Path]:
    """Resolve local logger text and metrics paths from the run configuration."""

    output_dir = Path(config.experiment.output_dir)
    text_filename = "train.log"
    metrics_filename = "metrics.jsonl"
    for backend in config.logging.backends:
        if backend.name == "local":
            text_filename = str(backend.params.get("text_filename", text_filename))
            metrics_filename = str(
                backend.params.get("metrics_filename", metrics_filename)
            )
            break
    return output_dir / metrics_filename, output_dir / text_filename


def _dataset_size(loader: object | None) -> int | None:
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", None)
    if isinstance(dataset, Sized):
        return len(dataset)
    return None


def _effective_steps_per_epoch(
    loaders: DataLoaders,
    *,
    max_batches: int | None,
) -> int:
    steps_per_epoch = loaders.steps_per_epoch
    if steps_per_epoch is None:
        if not isinstance(loaders.train, Sized):
            raise TypeError("training loader does not expose a finite epoch length")
        steps_per_epoch = len(loaders.train)
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


def _resolve_monitor(config: StochaflowConfig, loaders: DataLoaders) -> str:
    if loaders.validation is None:
        return "train_loss"
    return config.trainer.early_stopping.monitor


def _resolve_resume_checkpoint(
    resume: Path | None,
) -> Path | None:
    if resume is None:
        return None
    if resume.is_file():
        return resume
    if resume.is_dir():
        return CheckpointManager.find_latest(resume)
    raise FileNotFoundError(f"checkpoint does not exist: {resume}")


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


def _resolve_training_inputs(args: argparse.Namespace) -> ResolvedTrainingInputs:
    """Resolve one authoritative training config before constructing components."""

    config_path = cast(Path | None, args.config)
    requested_resume = cast(Path | None, args.resume)
    if (config_path is None) == (requested_resume is None):
        raise ValueError("train requires exactly one of --config or --resume")

    startup_cwd = Path.cwd().resolve()
    checkpoint_path = _resolve_resume_checkpoint(requested_resume)
    checkpoint: CheckpointState | None = None
    if checkpoint_path is None:
        assert config_path is not None
        unresolved_config = load_config(config_path)
        plan = prepare_extension_plugins(unresolved_config)
        config_source = "external"
    else:
        checkpoint = CheckpointManager.load_payload(
            checkpoint_path,
            map_location="cpu",
        )
        unresolved_config = _load_training_checkpoint_config(checkpoint)
        plan = prepare_extension_plugins(
            unresolved_config,
            expected_provenance=_load_checkpoint_extension_provenance(checkpoint),
            selection_policy=ExtensionSelectionPolicy.EXACT,
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
    if checkpoint_kind not in {None, "best", "latest"}:
        raise ValueError(
            "checkpoint metadata.checkpoint_kind must be 'best', 'latest', or null"
        )
    training.trainer.restore_fit_state(
        training_loop_state,
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
        selected_epoch=selected_epoch,
        training_loop_state=training_loop_state,
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


def _parse_strict_resume_state(
    payload: CheckpointState,
    *,
    require_cuda_compatibility: bool,
    require_mps_compatibility: bool = False,
) -> tuple[int, int, ParsedRNGState]:
    """Validate resume-only progress and RNG fields before mutating runtime state."""

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
    return cast(int, epoch), cast(int, global_step), rng_state


def _same_metric(left: object, right: float) -> bool:
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        return False
    value = float(left)
    return value == right or (math.isnan(value) and math.isnan(right))


def _validate_inherited_best(
    payload: CheckpointState,
    *,
    source: Path,
    selected_payload: CheckpointState,
    selected_epoch: int | None,
    best_epoch: int,
    best_metric: float,
    monitor: str,
    mode: str,
) -> None:
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
    candidate_epoch = cast(object, payload.get("epoch"))
    if (
        isinstance(candidate_epoch, bool)
        or not isinstance(candidate_epoch, int)
        or candidate_epoch != best_epoch
        or selected_epoch is None
        or candidate_epoch > selected_epoch
    ):
        raise ValueError(
            f"inherited best checkpoint '{source}' does not belong to the "
            "selected checkpoint history: best epoch mismatch"
        )
    metadata = candidate_metadata
    if metadata.get("checkpoint_kind") != "best":
        raise ValueError(
            f"inherited best checkpoint '{source}' must declare "
            "metadata.checkpoint_kind='best'"
        )
    candidate_loop = cast(object, metadata.get("training_loop"))
    expected_identity = {
        "best_epoch": best_epoch,
        "best_metric_value": best_metric,
        "monitor": monitor,
        "mode": mode,
    }
    if not isinstance(candidate_loop, dict):
        raise TypeError(
            f"inherited best checkpoint '{source}' is missing training_loop state"
        )
    for field, expected in expected_identity.items():
        actual = candidate_loop.get(field)
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
    metrics = cast(object, payload.get("metrics"))
    if (
        not isinstance(metrics, dict)
        or not _same_metric(metrics.get(monitor), best_metric)
    ):
        raise ValueError(
            f"inherited best checkpoint '{source}' does not contain the "
            f"recorded best metric '{monitor}'"
        )


def _materialize_inherited_best(
    training: TrainingComponents,
    *,
    checkpoint: Path,
    checkpoint_payload: CheckpointState,
    checkpoint_kind: object,
    selected_epoch: int | None,
    training_loop_state: object,
) -> Path | None:
    best_epoch = training.trainer.best_epoch
    best_metric = training.trainer.best_metric_value
    if best_epoch is None or best_metric is None:
        return None
    if not isinstance(training_loop_state, dict):
        raise TypeError("checkpoint metadata.training_loop must be a mapping")
    monitor = training_loop_state.get("monitor")
    mode = training_loop_state.get("mode")
    if not isinstance(monitor, str) or mode not in {"min", "max"}:
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
        selected_epoch=selected_epoch,
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
        materialized_payload = dict(candidate_payload)
        if training.trainer.checkpoint_config is not None:
            materialized_payload["config"] = training.trainer.checkpoint_config
        materialized_payload["metadata"] = {
            **candidate_metadata,
            **training.trainer.checkpoint_metadata,
            "checkpoint_kind": "best",
            "monitor": monitor,
            "mode": mode,
            "training_loop": candidate_metadata["training_loop"],
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
        start_epoch=start_epoch,
        close_logger=False,
        early_stopping_patience=(
            early_stopping.patience if early_stopping.enabled else None
        ),
        early_stopping_monitor=_resolve_monitor(config, loaders),
        early_stopping_mode=early_stopping.mode,
        early_stopping_min_delta=early_stopping.min_delta,
        reporter=reporter,
        track_best=True,
    )
    if not history:
        raise RuntimeError("trainer returned no epoch history")

    final_epoch = start_epoch + len(history) - 1
    best_checkpoint = training.trainer.best_checkpoint_path
    best_epoch = training.trainer.best_epoch or final_epoch
    if best_checkpoint is not None:
        training.checkpoint_manager.load(
            best_checkpoint,
            map_location=training.trainer.device,
        )
    return TrainingResult(
        best_epoch=best_epoch,
        best_loss=training.trainer.best_metric_value,
        best_checkpoint=best_checkpoint,
    )


def _evaluate_test_split(
    training: TrainingComponents,
    loaders: DataLoaders,
    options: ExperimentRunOptions,
    *,
    reporter: RichTrainingReporter,
) -> float | None:
    if loaders.test is None:
        return None
    metrics = training.trainer.evaluate_epoch(
        loaders.test,
        show_progress=options.show_progress,
        max_batches=options.max_test_batches,
        metric_prefix="test",
        reporter=reporter,
    )
    return metrics["loss"]


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
) -> None:
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
    selected_components = selected_component_identities(config)
    checkpoint_metadata = {
        **extension_metadata,
        "selected_components": selected_components,
        "config_source": config_source,
        "lineage": lineage,
        "startup_cwd": str(startup_cwd),
        "runtime_options": runtime_options,
    }
    write_yaml_manifest(
        Path(config.experiment.output_dir) / "run_manifest.yaml",
        {
            "kind": "training",
            "config_source": config_source,
            "config": config.to_dict(),
            **extension_metadata,
            "selected_components": selected_components,
            "lineage": lineage,
            "startup_cwd": str(startup_cwd),
            "runtime_options": runtime_options,
        },
    )
    training = build_training_components(
        config,
        checkpoint_metadata=checkpoint_metadata,
    )
    reporter = RichTrainingReporter()
    logger = training.logger
    logger_closed = False
    try:
        reporter.on_run_start(
            RunSummary(
                experiment_name=config.experiment.name,
                exp_id=config.experiment.exp_id,
                device=str(training.trainer.device),
                output_dir=config.experiment.output_dir,
                train_size=_dataset_size(loaders.train),
                valid_size=_dataset_size(loaders.validation),
                test_size=_dataset_size(loaders.test),
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
        test_loss = _evaluate_test_split(
            training,
            loaders,
            options,
            reporter=reporter,
        )
        stopped_early = training.trainer.stopped_early

        artifact_paths: dict[str, Path] | None = None
        if (
            options.sample_after_training
            and config.sampling.builder is not None
        ):
            if result.best_checkpoint is None:
                raise RuntimeError(
                    "post-training sampling requires a selected best checkpoint"
                )
            sampling_device = str(training.trainer.device)
            logger.close()
            logger_closed = True
            del training
            gc.collect()
            if sampling_device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif sampling_device.startswith("mps"):
                torch.mps.empty_cache()
            sampling_result = run_sampling(
                checkpoint=result.best_checkpoint,
                output_dir=Path(config.experiment.output_dir) / "samples" / "final",
                device_name=sampling_device,
            )
            artifact_paths = sampling_result.artifacts

        metrics_path, log_path = _local_log_paths(config)
        reporter.on_run_end(
            FinalSummary(
                best_epoch=result.best_epoch,
                best_valid_loss=result.best_loss,
                test_loss=test_loss,
                stopped_early=stopped_early,
                best_checkpoint=result.best_checkpoint,
                output_dir=config.experiment.output_dir,
                metrics_path=metrics_path,
                log_path=log_path,
                artifacts=artifact_paths,
            )
        )
    finally:
        if not logger_closed:
            logger.close()


def run_experiment_from_args(args: argparse.Namespace) -> None:
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

    set_seed(config.experiment.seed, deterministic=options.deterministic)
    loaders = build_data_loaders(
        config.data,
        seed=config.experiment.seed,
    )
    exp_id, output_dir = _make_timestamped_output_dir(str(output_root))
    config.experiment.exp_id = exp_id
    config.experiment.output_dir = str(output_dir)
    set_seed(config.experiment.seed, deterministic=options.deterministic)
    runtime_options = {
        "device": args.device,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "epochs": args.epochs,
        "limit_batches": args.limit_batches,
        "limit_validation_batches": args.limit_validation_batches,
        "limit_test_batches": args.limit_test_batches,
        "deterministic": args.deterministic,
        "no_progress": args.no_progress,
        "skip_final_sample": args.skip_final_sample,
        "force_extension_version_mismatch": (
            getattr(args, "force_extension_version_mismatch", False)
        ),
    }
    _run_single_run(
        config,
        loaders,
        options,
        extensions=inputs.extensions,
        config_source=inputs.config_source,
        checkpoint_payload=inputs.checkpoint,
        startup_cwd=inputs.startup_cwd,
        runtime_options=runtime_options,
    )
