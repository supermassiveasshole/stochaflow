"""Task-specific smoke training script for MNIST DDPM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stochaflow.diffusion import DDPM
from stochaflow.sampling import save_image_grid, save_trajectory_grid
from stochaflow.utils.config import load_config
from stochaflow.utils.factory import (
    TrainingComponents,
    build_data_components,
    build_training_components,
)
from stochaflow.utils.seed import set_seed


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
        default=10,
        help="Maximum number of batches per epoch for the smoke test.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic Torch behavior where supported.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
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


def _mnist_sample_shape(
    config_channels: int, image_size: int, num_samples: int
) -> torch.Size:
    """Build the full batch-first MNIST sample shape."""

    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    return torch.Size((num_samples, config_channels, image_size, image_size))


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
        trajectory = diffusion.sample_trajectory(
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

    data = build_data_components(config.data, seed=config.experiment.seed)
    training = build_training_components(config)

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

    history = training.trainer.fit(
        data.dataloader,
        num_epochs=args.epochs,
        show_progress=not args.no_progress,
        max_batches_per_epoch=args.limit_batches,
        start_epoch=start_epoch,
    )
    if history:
        final_metrics = history[-1]
    else:
        raise RuntimeError(
            "no epochs were run; check --epochs and the resumed checkpoint epoch",
        )
    final_epoch = start_epoch + len(history) - 1

    sample_message = ""
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
            epoch=final_epoch,
            grid_nrow=args.sample_grid_size,
            trajectory_interval=args.trajectory_interval,
        )
        sample_message = (
            f", samples={artifact_paths['samples']}, "
            f"raw_samples={artifact_paths['raw_samples']}, "
            f"trajectory={artifact_paths['trajectory']}, "
            f"raw_trajectory={artifact_paths['raw_trajectory']}"
        )

    print(
        "MNIST DDPM smoke run completed: "
        f"epochs={args.epochs}, "
        f"limit_batches={args.limit_batches}, "
        f"loss={final_metrics['loss']:.6f}"
        f"{sample_message}"
    )


if __name__ == "__main__":
    main()
