"""Train a registered single- or multi-dataset DDPM experiment."""

import argparse
from pathlib import Path

from stochaflow.scripts.ddpm_runner import (
    add_ddpm_training_arguments,
    run_ddpm_from_args,
)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return add_ddpm_training_arguments(
        parser,
        default_config=Path("configs/ddpm_mnist.yaml"),
    )


def main() -> None:
    """Run a config-selected DDPM experiment."""

    run_ddpm_from_args(
        _build_argument_parser().parse_args(),
        script_name="train_ddpm.py",
    )


if __name__ == "__main__":
    main()
