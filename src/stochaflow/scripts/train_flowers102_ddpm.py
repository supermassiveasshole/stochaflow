"""Task-specific training script for Oxford Flowers 102 DDPM."""

import argparse
from pathlib import Path

from stochaflow.scripts.ddpm_runner import (
    add_ddpm_training_arguments,
    image_sample_shape,
    run_ddpm_from_args,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Flowers102 DDPM training script."""

    parser = argparse.ArgumentParser(description=__doc__)
    return add_ddpm_training_arguments(
        parser,
        default_config=Path("configs/ddpm_flowers102.yaml"),
        default_num_samples=64,
        default_sample_grid_size=8,
    )


def main() -> None:
    """Run Oxford Flowers 102 DDPM training against the current implementation."""

    run_ddpm_from_args(
        _build_argument_parser().parse_args(),
        script_name="train_flowers102_ddpm.py",
        sample_shape_fn=image_sample_shape,
    )


if __name__ == "__main__":
    main()
