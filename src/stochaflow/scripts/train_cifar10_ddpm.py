"""Task-specific training script for CIFAR-10 DDPM."""

import argparse
from pathlib import Path

from stochaflow.scripts.ddpm_runner import (
    add_ddpm_training_arguments,
    image_sample_shape,
    run_ddpm_from_args,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the CIFAR-10 DDPM training script."""

    parser = argparse.ArgumentParser(description=__doc__)
    return add_ddpm_training_arguments(
        parser,
        default_config=Path("configs/ddpm_cifar10.yaml"),
        default_num_samples=16,
        default_sample_grid_size=4,
    )


def main() -> None:
    """Run CIFAR-10 DDPM training against the current implementation."""

    run_ddpm_from_args(
        _build_argument_parser().parse_args(),
        expected_dataset="cifar10",
        script_name="train_cifar10_ddpm.py",
        sample_shape_fn=image_sample_shape,
    )


if __name__ == "__main__":
    main()
