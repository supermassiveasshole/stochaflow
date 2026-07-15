"""Task-specific smoke training script for MNIST DDPM."""

import argparse
from pathlib import Path

from stochaflow.scripts.ddpm_runner import (
    add_ddpm_training_arguments,
    image_sample_shape,
    run_ddpm_from_args,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the MNIST DDPM smoke script."""

    parser = argparse.ArgumentParser(description=__doc__)
    return add_ddpm_training_arguments(
        parser,
        default_config=Path("configs/ddpm_mnist.yaml"),
        default_num_samples=16,
        default_sample_grid_size=4,
    )


def main() -> None:
    """Run a minimal MNIST DDPM training loop against the current implementation."""

    run_ddpm_from_args(
        _build_argument_parser().parse_args(),
        script_name="train_mnist_ddpm.py",
        sample_shape_fn=image_sample_shape,
    )


if __name__ == "__main__":
    main()
