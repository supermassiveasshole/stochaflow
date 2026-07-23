"""Contract tests for generated configuration documentation."""

from pathlib import Path
import subprocess
import sys

from stochaflow.utils.config import StochaflowConfig, load_config


def _component_section(reference: str, anchor: str) -> str:
    start = reference.index(f"(config-component-{anchor})=")
    end = reference.find("\n(config-", start + 1)
    return reference[start:] if end == -1 else reference[start:end]


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


def test_context_built_component_private_parameters_are_documented() -> None:
    reference = Path("docs/configuration/reference.md").read_text(encoding="utf-8")
    expected_paths = {
        "data_builders-image": (
            "source.kind",
            "image.size",
            "partition.mode",
            "loader.steps_per_epoch",
        ),
        "data_builders-multi-resolution-image": (
            "sources[].sampling_weight",
            "batching.buckets[].name",
            "batching.base_bucket",
            "loader.steps_per_epoch",
        ),
        "data_builders-super-resolution": (
            "source.high_resolution_path",
            "image.high_resolution",
            "low_resolution.kind",
            "loader.steps_per_epoch",
        ),
        "sampling_builders-standard-denoising": (
            "weights",
            "sampler.name",
            "trajectory.every_steps",
        ),
        "training_builders-gaussian-denoising": ("prediction_type",),
    }

    for anchor, paths in expected_paths.items():
        section = _component_section(reference, anchor)
        assert "无组件级配置参数。" not in section
        for path in paths:
            assert f"| `{path}` |" in section

    supervised = _component_section(reference, "training_builders-supervised")
    assert "无组件级配置参数。" not in supervised
    assert "`params` 必须是空 mapping（默认 `{}`）" in supervised


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
