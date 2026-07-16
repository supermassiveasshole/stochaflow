"""Shared command-line runner helpers for config-driven experiments."""

import argparse
from collections.abc import Sized
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
import gc
from pathlib import Path
from typing import Any

import torch
import yaml

from stochaflow.data.pipeline import DataBundle, DataPipeline
from stochaflow.sampling.runtime import run_sampling
from stochaflow.training.reporting import (
    FinalSummary,
    RichTrainingReporter,
    RunSummary,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import StochaflowConfig, load_config
from stochaflow.utils.factory import TrainingComponents, build_training_components
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


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common config-driven experiment options to a parser."""

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment config file.",
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
        "--resume",
        type=Path,
        nargs="?",
        const=Path("latest"),
        default=None,
        help=(
            "Resume training. With no value, uses the newest checkpoints/latest.pt "
            "under the configured output root. Use --resume PATH to choose one."
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def _dataset_size(split: Any | None) -> int | None:
    if split is None:
        return None
    if not isinstance(split.dataset, Sized):
        raise TypeError("run summary requires sized datasets")
    return len(split.dataset)


def _effective_steps_per_epoch(
    dataloader: Any,
    *,
    max_batches: int | None,
) -> int:
    if not isinstance(dataloader, Sized):
        raise TypeError(
            "lr_scheduler total_steps='auto' requires a sized training dataloader"
        )
    steps_per_epoch = len(dataloader)
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    if steps_per_epoch <= 0:
        raise ValueError("training dataloader must yield at least one batch")
    return steps_per_epoch


def _resolve_monitor(config: StochaflowConfig, bundle: DataBundle) -> str:
    if bundle.valid is None:
        return "train_loss"
    return config.trainer.early_stopping.monitor


def _resolve_resume_checkpoint(
    resume: Path | None,
    *,
    output_root: str | Path,
    fold_index: int | None = None,
) -> Path | None:
    if resume is None:
        return None
    if resume.is_file():
        return resume
    search_root = Path(output_root) if resume == Path("latest") else resume
    if fold_index is not None:
        return _find_latest_fold_checkpoint(search_root, fold_index=fold_index)
    if resume == Path("latest"):
        return CheckpointManager.find_latest(output_root)
    if resume.is_dir():
        return CheckpointManager.find_latest(resume)
    return resume


def _find_latest_fold_checkpoint(root: str | Path, *, fold_index: int) -> Path:
    """Find the newest latest checkpoint belonging to one fold only."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"checkpoint search root does not exist: {root_path}")
    fold_name = f"fold_{fold_index:02d}"
    candidates = [
        path
        for path in root_path.rglob("latest.pt")
        if path.parent.name == "checkpoints" and fold_name in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"could not find 'latest.pt' for {fold_name} under: {root_path}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _validate_resume_scope(
    config: StochaflowConfig,
    resume: Path | None,
) -> None:
    """Reject one-checkpoint-to-many-fold resume requests before data I/O."""

    runs_all_folds = (
        config.data.splits.mode == "kfold"
        and config.data.splits.fold_index is None
    )
    if not runs_all_folds or resume is None:
        return
    if resume != Path("latest") and not resume.is_dir():
        raise ValueError(
            "resuming multiple folds requires --resume without a value or a "
            "directory containing fold-specific checkpoints"
        )


def _configure_run_output(
    config: StochaflowConfig,
    *,
    base_exp_id: str,
    base_output_dir: Path,
    bundle: DataBundle,
) -> None:
    if bundle.fold_index is not None:
        output_dir = base_output_dir / f"fold_{bundle.fold_index:02d}"
        output_dir.mkdir(parents=True, exist_ok=False)
        config.experiment.exp_id = f"{base_exp_id}_fold_{bundle.fold_index:02d}"
        config.experiment.output_dir = str(output_dir)
    else:
        config.experiment.exp_id = base_exp_id
        config.experiment.output_dir = str(base_output_dir)


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
    *,
    target_epoch: int,
) -> int:
    """Restore runtime state and return the first epoch still to run."""

    if checkpoint is None:
        return 1
    loaded = training.checkpoint_manager.load(
        checkpoint,
        map_location=training.trainer.device,
    )
    if loaded.global_step is not None:
        training.trainer.global_step = loaded.global_step
    start_epoch = 1 if loaded.epoch is None else loaded.epoch + 1
    if start_epoch > target_epoch:
        completed_epoch = start_epoch - 1
        raise ValueError(
            f"checkpoint already completed epoch {completed_epoch}, which meets or "
            f"exceeds the target of {target_epoch}; increase --epochs to continue"
        )
    return start_epoch


def _fit_and_select_best(
    training: TrainingComponents,
    config: StochaflowConfig,
    bundle: DataBundle,
    options: ExperimentRunOptions,
    *,
    start_epoch: int,
    reporter: RichTrainingReporter,
) -> TrainingResult:
    """Fit a bundle, restore its selected checkpoint, and summarize it."""

    early_stopping = config.trainer.early_stopping
    history = training.trainer.fit(
        bundle.train.dataloader,
        num_epochs=options.num_epochs,
        show_progress=options.show_progress,
        max_batches_per_epoch=options.max_train_batches,
        validation_dataloader=(
            bundle.valid.dataloader if bundle.valid is not None else None
        ),
        max_validation_batches=options.max_validation_batches,
        start_epoch=start_epoch,
        close_logger=False,
        early_stopping_patience=(
            early_stopping.patience if early_stopping.enabled else None
        ),
        early_stopping_monitor=_resolve_monitor(config, bundle),
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
    bundle: DataBundle,
    options: ExperimentRunOptions,
    *,
    reporter: RichTrainingReporter,
) -> float | None:
    if bundle.test is None:
        return None
    metrics = training.trainer.evaluate_epoch(
        bundle.test.dataloader,
        show_progress=options.show_progress,
        max_batches=options.max_test_batches,
        metric_prefix="test",
        reporter=reporter,
    )
    return metrics["loss"]


def _run_single_bundle(
    config: StochaflowConfig,
    bundle: DataBundle,
    options: ExperimentRunOptions,
) -> None:
    config.trainer.num_epochs = options.num_epochs
    config.trainer.show_progress = options.show_progress
    if options.device is not None:
        config.trainer.device = options.device
    _write_resolved_config(config)
    training = build_training_components(
        config,
        steps_per_epoch=_effective_steps_per_epoch(
            bundle.train.dataloader,
            max_batches=options.max_train_batches,
        ),
        num_epochs=options.num_epochs,
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
                train_size=_dataset_size(bundle.train) or 0,
                valid_size=_dataset_size(bundle.valid),
                test_size=_dataset_size(bundle.test),
                batch_size=config.data.dataloader.batch_size,
            )
        )
        start_epoch = _restore_training_state(
            training,
            options.resume_checkpoint,
            target_epoch=options.num_epochs,
        )
        result = _fit_and_select_best(
            training,
            config,
            bundle,
            options,
            start_epoch=start_epoch,
            reporter=reporter,
        )
        test_loss = _evaluate_test_split(
            training,
            bundle,
            options,
            reporter=reporter,
        )
        stopped_early = training.trainer.stopped_early

        artifact_paths: dict[str, Path] | None = None
        if options.sample_after_training:
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
    """Run one registered single- or multi-dataset experiment."""

    config = load_config(args.config)
    if args.output_dir is not None:
        config.experiment.output_dir = str(args.output_dir)

    options = ExperimentRunOptions.from_namespace(
        args,
        configured_num_epochs=config.trainer.num_epochs,
        configured_show_progress=config.trainer.show_progress,
    )
    _validate_resume_scope(config, options.resume_checkpoint)
    set_seed(config.experiment.seed, deterministic=options.deterministic)
    bundles = DataPipeline(
        config.data,
        seed=config.experiment.seed,
    ).build()
    resume_checkpoints = [
        _resolve_resume_checkpoint(
            options.resume_checkpoint,
            output_root=config.experiment.output_dir,
            fold_index=bundle.fold_index,
        )
        for bundle in bundles
    ]
    base_exp_id, base_output_dir = _make_timestamped_output_dir(
        config.experiment.output_dir
    )

    for bundle, resume_checkpoint in zip(
        bundles,
        resume_checkpoints,
        strict=True,
    ):
        run_config = deepcopy(config)
        _configure_run_output(
            run_config,
            base_exp_id=base_exp_id,
            base_output_dir=base_output_dir,
            bundle=bundle,
        )
        run_options = replace(
            options,
            resume_checkpoint=resume_checkpoint,
        )
        set_seed(run_config.experiment.seed, deterministic=options.deterministic)
        _run_single_bundle(
            run_config,
            bundle,
            run_options,
        )
