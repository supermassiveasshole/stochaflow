"""Lightweight parser declaration for the training operation."""

import argparse
from pathlib import Path


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
        metavar="CHECKPOINT_OR_BUNDLE",
        help=(
            "Strictly resume from a single-process checkpoint/run directory, "
            "or from an exact DDP bundle directory when --ddp is selected."
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
        "--ddp",
        action="store_true",
        help=(
            "Run fixed single-node distributed training under torchrun. "
            "The first maintained path uses one process per local CUDA device."
        ),
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


__all__ = ["add_training_arguments"]
