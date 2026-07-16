"""Contract tests for generated configuration documentation."""

from pathlib import Path
import subprocess
import sys

from stochaflow.utils.config import StochaflowConfig, load_config


def test_generated_config_reference_is_current() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/generate_config_reference.py",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_all_builtin_yaml_configs_load() -> None:
    paths = sorted(Path("configs").glob("*.yaml"))

    assert [path.name for path in paths] == [
        "ddim_cifar10.yaml",
        "ddim_flowers102.yaml",
        "ddpm_cifar10.yaml",
        "ddpm_flowers102.yaml",
        "ddpm_mnist.yaml",
        "ddpm_mnist_flowers102.yaml",
    ]
    assert all(isinstance(load_config(path), StochaflowConfig) for path in paths)
