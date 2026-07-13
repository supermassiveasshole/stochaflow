"""Sample images from a saved DDPM checkpoint."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from stochaflow.diffusion import DDPM
from stochaflow.sampling import save_image_grid, save_trajectory_grid
from stochaflow.scripts.ddpm_runner import image_sample_shape, sample_reverse_trajectory
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import StochaflowConfig, load_config_dict
from stochaflow.utils.factory import build_diffusion, build_ema, build_model
from stochaflow.utils.factory import build_scheduler
from stochaflow.utils.factory import resolve_device
from stochaflow.utils.seed import set_seed


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The checkpoint stores the training config used to rebuild the model, "
            "image shape, diffusion schedule, and EMA state. If --checkpoint is "
            "omitted, the sampler uses the newest checkpoints/best.pt under "
            "--search-dir."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a DDPM checkpoint saved by Stochaflow.",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=Path("outputs"),
        help="Output root to search for checkpoints/best.pt when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Sampling device, e.g. auto, cpu, cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed. Defaults to the experiment seed in config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of images to sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Number of images to sample per reverse-process batch. Defaults to "
            "data.dataloader.batch_size from the checkpoint config."
        ),
    )
    parser.add_argument(
        "--sample-grid-size",
        type=int,
        default=8,
        help="Number of images per row in the sample grid.",
    )
    parser.add_argument(
        "--trajectory-interval",
        type=int,
        default=200,
        help="Reverse-process interval for trajectory snapshots.",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Only save the final samples and skip trajectory artifacts.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sample",
        help="Filename prefix for generated artifacts.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help=(
            "Disable EMA for diagnostic sampling. By default, EMA is used when "
            "it was enabled during training and the checkpoint contains EMA state."
        ),
    )
    return parser


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint at '{checkpoint_path}' must contain a dictionary")
    return payload


def _resolve_sampling_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint
    if checkpoint is None:
        return CheckpointManager.find_best(args.search_dir)
    if checkpoint.is_dir():
        return CheckpointManager.find_best(checkpoint)
    return checkpoint


def _load_sampling_config(checkpoint_path: Path) -> StochaflowConfig:
    payload = _load_checkpoint_payload(checkpoint_path)
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint does not contain a Stochaflow config")
    return load_config_dict(raw_config)


def _default_output_dir(checkpoint_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return checkpoint_path.parent.parent / "samples_from_checkpoint" / timestamp


def _build_checkpointed_ddpm(
    config: StochaflowConfig,
    checkpoint_path: Path,
    *,
    device: torch.device,
    disable_ema: bool = False,
) -> tuple[DDPM, int | None, bool]:
    model = build_model(config.model)
    scheduler = build_scheduler(config.diffusion.scheduler)
    diffusion = build_diffusion(
        config.diffusion.name,
        model=model,
        scheduler=scheduler,
        params=config.diffusion.params,
    )
    if not isinstance(diffusion, DDPM):
        raise TypeError("sample_ddpm.py expects the built diffusion to be DDPM")

    should_use_ema = (
        config.ema.enabled and config.ema.use_for_sampling and not disable_ema
    )
    ema = None
    if should_use_ema:
        ema = build_ema(config.ema, diffusion)
        if ema is None:
            raise ValueError("EMA was requested but EMA is disabled in the config")

    checkpoint_manager = CheckpointManager(model=diffusion, ema=ema)
    loaded = checkpoint_manager.load(checkpoint_path, map_location=device)
    diffusion.to(device)
    if ema is not None:
        ema.copy_to(diffusion)
    diffusion.eval()
    return diffusion, loaded.epoch, ema is not None


def _batched_sample_counts(num_samples: int, batch_size: int) -> list[int]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    counts: list[int] = []
    remaining = num_samples
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        counts.append(current_batch_size)
        remaining -= current_batch_size
    return counts


def _sample_without_trajectory(
    diffusion: DDPM,
    config: StochaflowConfig,
    args: argparse.Namespace,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    samples = [
        diffusion.sample(image_sample_shape(config, count), device=device)
        .detach()
        .cpu()
        for count in _batched_sample_counts(args.num_samples, batch_size)
    ]
    return torch.cat(samples, dim=0)


def _sample_with_trajectory(
    diffusion: DDPM,
    config: StochaflowConfig,
    args: argparse.Namespace,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    trajectory_parts: dict[int, list[torch.Tensor]] = {}
    for count in _batched_sample_counts(args.num_samples, batch_size):
        batch_trajectory = sample_reverse_trajectory(
            diffusion,
            image_sample_shape(config, count),
            device=device,
            capture_every=args.trajectory_interval,
        )
        for timestep, snapshot in batch_trajectory.items():
            trajectory_parts.setdefault(timestep, []).append(snapshot)

    return {
        timestep: torch.cat(snapshots, dim=0)
        for timestep, snapshots in trajectory_parts.items()
    }


def _save_samples(
    diffusion: DDPM,
    config: StochaflowConfig,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    device: torch.device,
    epoch: int | None,
) -> dict[str, Path]:
    if args.sample_grid_size <= 0:
        raise ValueError("--sample-grid-size must be positive")
    batch_size = (
        config.data.dataloader.batch_size
        if args.batch_size is None
        else args.batch_size
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "latest" if epoch is None else f"epoch_{epoch:04d}"
    prefix = f"{args.prefix}_{suffix}"
    artifacts: dict[str, Path] = {}

    with torch.no_grad():
        if args.no_trajectory:
            samples = _sample_without_trajectory(
                diffusion,
                config,
                args,
                device=device,
                batch_size=batch_size,
            )
        else:
            trajectory = _sample_with_trajectory(
                diffusion,
                config,
                args,
                device=device,
                batch_size=batch_size,
            )
            trajectory_tensor_path = output_dir / f"{prefix}_trajectory.pt"
            trajectory_grid_path = output_dir / f"{prefix}_trajectory.png"
            torch.save(trajectory, trajectory_tensor_path)
            save_trajectory_grid(
                trajectory,
                trajectory_grid_path,
                denormalize=True,
            )
            artifacts["trajectory"] = trajectory_grid_path
            artifacts["raw_trajectory"] = trajectory_tensor_path
            samples = trajectory[0].detach().cpu()

    tensor_path = output_dir / f"{prefix}.pt"
    grid_path = output_dir / f"{prefix}.png"
    torch.save(samples, tensor_path)
    save_image_grid(
        samples,
        grid_path,
        nrow=args.sample_grid_size,
        denormalize=True,
    )
    artifacts["samples"] = grid_path
    artifacts["raw_samples"] = tensor_path
    return artifacts


def main() -> None:
    args = _build_argument_parser().parse_args()
    checkpoint_path = _resolve_sampling_checkpoint(args)
    config = _load_sampling_config(checkpoint_path)
    if config.diffusion.name != "ddpm":
        raise ValueError(
            f"sample_ddpm.py expects a DDPM config, got '{config.diffusion.name}'"
        )

    seed = config.experiment.seed if args.seed is None else args.seed
    set_seed(seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir or _default_output_dir(checkpoint_path)

    diffusion, epoch, used_ema = _build_checkpointed_ddpm(
        config,
        checkpoint_path,
        device=device,
        disable_ema=args.no_ema,
    )
    artifacts = _save_samples(
        diffusion,
        config,
        args,
        output_dir=output_dir,
        device=device,
        epoch=epoch,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"EMA weights: {'yes' if used_ema else 'no'}")
    print(f"Clip denoised: {'yes' if diffusion.clip_denoised else 'no'}")
    print(f"Output: {output_dir}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
