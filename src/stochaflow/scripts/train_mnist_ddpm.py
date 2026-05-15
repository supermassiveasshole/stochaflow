"""Task-specific smoke training script for MNIST DDPM."""

import argparse
from collections.abc import Sized
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from stochaflow.diffusion import DDPM
from stochaflow.sampling import save_image_grid, save_trajectory_grid
from stochaflow.training.reporting import (
    FinalSummary,
    RichTrainingReporter,
    RunSummary,
)
from stochaflow.utils.config import load_config
from stochaflow.utils.factory import (
    TrainingComponents,
    build_dataloader,
    build_dataset,
    build_training_components,
)
from stochaflow.utils.seed import set_seed


@dataclass(slots=True)
class MnistDataSplits:
    """Train/validation/test dataloaders for the MNIST DDPM script."""

    train_dataset: Dataset
    valid_dataset: Dataset
    test_dataset: Dataset
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    test_dataloader: DataLoader


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the MNIST DDPM smoke script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ddpm_mnist.yaml"),
        help="Path to the MNIST DDPM config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run for the smoke test.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Maximum number of batches per epoch for the smoke test.",
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
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of samples to generate after training.",
    )
    parser.add_argument(
        "--sample-grid-size",
        type=int,
        default=4,
        help="Number of images per row in the post-training sample grid.",
    )
    parser.add_argument(
        "--trajectory-interval",
        type=int,
        default=200,
        help="Reverse-process step interval for trajectory snapshots.",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip post-training reverse sampling and artifact dumping.",
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


def _validation_size(total_size: int, configured_size: int | float) -> int:
    """Resolve a configured validation size against the train dataset length."""

    if isinstance(configured_size, float):
        valid_size = int(round(total_size * configured_size))
    else:
        valid_size = configured_size
    if valid_size <= 0 or valid_size >= total_size:
        raise ValueError(
            "data.splits.validation_size must produce a non-empty validation "
            "split and leave at least one training sample"
        )
    return valid_size


def _eval_dataloader_config(config):
    """Use the training dataloader shape with deterministic eval semantics."""

    return replace(config, shuffle=False, drop_last=False)


def _dataset_length(dataset: Dataset) -> int:
    """Return the length of a map-style dataset."""

    if not isinstance(dataset, Sized):
        raise TypeError("MNIST train/validation splitting requires a sized dataset")
    return len(dataset)


def _build_mnist_data_splits(config) -> MnistDataSplits:
    """Build train/validation subsets from MNIST train and test from MNIST test."""

    train_dataset_config = deepcopy(config.data.dataset)
    train_dataset_config.params["split"] = "train"
    full_train_dataset = build_dataset(train_dataset_config)
    full_train_size = _dataset_length(full_train_dataset)

    valid_size = _validation_size(
        full_train_size,
        config.data.splits.validation_size,
    )
    train_size = full_train_size - valid_size
    split_generator = torch.Generator().manual_seed(config.experiment.seed)
    train_dataset, valid_dataset = random_split(
        full_train_dataset,
        [train_size, valid_size],
        generator=split_generator,
    )

    test_dataset_config = deepcopy(config.data.dataset)
    test_dataset_config.params["split"] = config.data.splits.test_split
    test_dataset = build_dataset(test_dataset_config)

    eval_config = _eval_dataloader_config(config.data.dataloader)
    return MnistDataSplits(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        train_dataloader=build_dataloader(
            train_dataset,
            config.data.dataloader,
            seed=config.experiment.seed,
        ),
        valid_dataloader=build_dataloader(
            valid_dataset,
            eval_config,
            seed=config.experiment.seed,
        ),
        test_dataloader=build_dataloader(
            test_dataset,
            eval_config,
            seed=config.experiment.seed,
        ),
    )


def _local_log_paths(config) -> tuple[Path, Path]:
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


def _mnist_sample_shape(
    config_channels: int, image_size: int, num_samples: int
) -> torch.Size:
    """Build the full batch-first MNIST sample shape."""

    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    return torch.Size((num_samples, config_channels, image_size, image_size))


def _sample_reverse_trajectory(
    diffusion: DDPM,
    sample_shape: torch.Size,
    *,
    device: torch.device,
    capture_every: int,
) -> dict[int, torch.Tensor]:
    """Sample a trajectory by repeatedly calling the public reverse traversal."""

    if capture_every <= 0:
        raise ValueError("--trajectory-interval must be positive")

    current_timestep = diffusion.num_timesteps - 1
    x_t = torch.randn(sample_shape, device=device)
    trajectory: dict[int, torch.Tensor] = {
        current_timestep: x_t.detach().cpu(),
    }

    while current_timestep > 0:
        target_timestep = max(0, current_timestep - capture_every)
        x_t = diffusion.reverse(x_t, current_timestep, target_timestep)
        current_timestep = target_timestep
        trajectory[current_timestep] = x_t.detach().cpu()

    return trajectory


def _dump_sampling_artifacts(
    training: TrainingComponents,
    *,
    sample_shape: torch.Size,
    output_dir: Path,
    epoch: int,
    grid_nrow: int,
    trajectory_interval: int,
) -> dict[str, Path]:
    """Generate DDPM samples and reverse-trajectory artifacts."""

    if grid_nrow <= 0:
        raise ValueError("--sample-grid-size must be positive")
    if trajectory_interval <= 0:
        raise ValueError("--trajectory-interval must be positive")

    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = sample_dir / f"epoch_{epoch:04d}.pt"
    grid_path = sample_dir / f"epoch_{epoch:04d}.png"
    trajectory_tensor_path = sample_dir / f"epoch_{epoch:04d}_trajectory.pt"
    trajectory_grid_path = sample_dir / f"epoch_{epoch:04d}_trajectory.png"

    if not isinstance(training.diffusion, DDPM):
        raise TypeError("train_mnist_ddpm.py expects the built diffusion to be DDPM")

    diffusion = training.diffusion
    device = training.trainer.device
    diffusion.eval()
    with torch.no_grad():
        trajectory = _sample_reverse_trajectory(
            diffusion,
            sample_shape,
            device=device,
            capture_every=trajectory_interval,
        )
    samples = trajectory[0]

    torch.save(samples.detach().cpu(), tensor_path)
    torch.save(trajectory, trajectory_tensor_path)
    save_image_grid(samples, grid_path, nrow=grid_nrow, denormalize=True)
    save_trajectory_grid(trajectory, trajectory_grid_path, denormalize=True)

    return {
        "samples": grid_path,
        "raw_samples": tensor_path,
        "trajectory": trajectory_grid_path,
        "raw_trajectory": trajectory_tensor_path,
    }


def main() -> None:
    """Run a minimal MNIST DDPM training loop against the current implementation."""

    args = _build_argument_parser().parse_args()

    config = load_config(args.config)
    if config.data.dataset.name != "mnist":
        raise ValueError(
            f"train_mnist_ddpm.py expects an MNIST config, got '{config.data.dataset.name}'"
        )
    if config.diffusion.name != "ddpm":
        raise ValueError(
            f"train_mnist_ddpm.py expects a DDPM config, got '{config.diffusion.name}'"
        )

    set_seed(config.experiment.seed, deterministic=args.deterministic)
    exp_id, output_dir = _make_timestamped_output_dir(config.experiment.output_dir)
    config.experiment.exp_id = exp_id
    config.experiment.output_dir = str(output_dir)

    data = _build_mnist_data_splits(config)
    training = build_training_components(config)
    reporter = RichTrainingReporter()
    reporter.on_run_start(
        RunSummary(
            experiment_name=config.experiment.name,
            exp_id=config.experiment.exp_id,
            device=str(training.trainer.device),
            output_dir=config.experiment.output_dir,
            train_size=_dataset_length(data.train_dataset),
            valid_size=_dataset_length(data.valid_dataset),
            test_size=_dataset_length(data.test_dataset),
            batch_size=config.data.dataloader.batch_size,
        )
    )

    start_epoch = 1
    if args.resume is not None:
        loaded = training.checkpoint_manager.load(
            args.resume,
            map_location=training.trainer.device,
        )
        if loaded.epoch is not None:
            start_epoch = loaded.epoch + 1
        if loaded.global_step is not None:
            training.trainer.global_step = loaded.global_step

    try:
        early_stopping = config.trainer.early_stopping
        show_progress = not args.no_progress
        history = training.trainer.fit(
            data.train_dataloader,
            num_epochs=args.epochs,
            show_progress=show_progress,
            max_batches_per_epoch=args.limit_batches,
            validation_dataloader=data.valid_dataloader,
            max_validation_batches=args.limit_validation_batches,
            start_epoch=start_epoch,
            close_logger=False,
            early_stopping_patience=(
                early_stopping.patience if early_stopping.enabled else None
            ),
            early_stopping_monitor=early_stopping.monitor,
            early_stopping_mode=early_stopping.mode,
            early_stopping_min_delta=early_stopping.min_delta,
            reporter=reporter,
        )
        if not history:
            raise RuntimeError(
                "no epochs were run; check --epochs and the resumed checkpoint epoch",
            )
        final_epoch = start_epoch + len(history) - 1
        best_checkpoint_path = training.trainer.best_checkpoint_path
        best_epoch = training.trainer.best_epoch or final_epoch
        best_valid_loss = training.trainer.best_metric_value
        if best_checkpoint_path is not None:
            training.checkpoint_manager.load(
                best_checkpoint_path,
                map_location=training.trainer.device,
            )
        test_metrics = training.trainer.evaluate_epoch(
            data.test_dataloader,
            show_progress=show_progress,
            max_batches=args.limit_test_batches,
            metric_prefix="test",
            reporter=reporter,
        )

        artifact_paths: dict[str, Path] | None = None
        if not args.skip_sampling:
            channels = int(
                config.data.dataset.params.get(
                    "channels", config.model.params["in_channels"]
                )
            )
            image_size = int(config.data.dataset.params["image_size"])
            sample_shape = _mnist_sample_shape(channels, image_size, args.num_samples)
            artifact_paths = _dump_sampling_artifacts(
                training,
                sample_shape=sample_shape,
                output_dir=Path(config.experiment.output_dir),
                epoch=best_epoch,
                grid_nrow=args.sample_grid_size,
                trajectory_interval=args.trajectory_interval,
            )

        metrics_path, log_path = _local_log_paths(config)
        reporter.on_run_end(
            FinalSummary(
                best_epoch=best_epoch,
                best_valid_loss=best_valid_loss,
                test_loss=test_metrics["loss"],
                stopped_early=training.trainer.stopped_early,
                best_checkpoint=best_checkpoint_path,
                output_dir=config.experiment.output_dir,
                metrics_path=metrics_path,
                log_path=log_path,
                artifacts=artifact_paths,
            )
        )
    finally:
        training.logger.close()


if __name__ == "__main__":
    main()
