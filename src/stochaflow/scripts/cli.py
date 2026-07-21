"""Unified Stochaflow command-line interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from stochaflow.sampling.runtime import run_sampling
from stochaflow.scripts.experiment_runner import (
    add_training_arguments,
    run_experiment_from_args,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level train/sample command parser."""

    parser = argparse.ArgumentParser(prog="stochaflow", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run a config-driven training experiment.",
    )
    add_training_arguments(train_parser)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Sample a portable Stochaflow checkpoint.",
    )
    sample_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment config whose sampling section overrides the checkpoint.",
    )
    sample_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file or run directory. Optional when --config is provided.",
    )
    sample_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the sampling device.",
    )
    sample_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the selected Stochaflow subcommand."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        run_experiment_from_args(args)
        return
    if args.config is None and args.checkpoint is None:
        parser.error("sample requires --config, --checkpoint, or both")

    result = run_sampling(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Builder: {result.builder_name}")
    print(f"Device: {result.device}")
    print(f"Seed: {result.seed}")
    print(f"Weights: {result.metadata.get('weights', 'builder-defined')}")
    print(f"Output: {result.output_dir}")
    for name, path in result.artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
