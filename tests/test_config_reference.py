"""Contract tests for generated configuration documentation."""

import subprocess
import sys
from pathlib import Path

from stochaflow.utils.config import (
    SampleInvocationConfig,
    StochaflowConfig,
    load_config,
    load_sample_config,
)


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


def test_reference_separates_complete_sample_invocation_from_training() -> None:
    reference = Path("docs/configuration/reference.md").read_text(encoding="utf-8")

    for field_path in (
        "sample",
        "sample-sampler",
        "sample-options",
        "sample-shape",
        "sample-num-samples",
        "sample-batch-size",
        "sample-seed",
        "sample-writers",
    ):
        assert f"(config-field-path-{field_path})=" in reference
    assert "(config-field-path-sampling)=" not in reference
    assert "(config-field-path-ema-use-for-sampling)=" not in reference
    assert "(config-field-path-sample-builder)=" not in reference
    assert "checkpoint.inference_recipe.name" in reference
    assert "partial sample request" not in reference
    assert "完整 sample invocation config" in reference


def test_context_built_component_private_parameters_are_documented() -> None:
    reference = Path("docs/configuration/reference.md").read_text(encoding="utf-8")
    expected_paths = {
        "data_builders-class-labeled-image": (
            "source.name",
            "source.materialization.verification",
            "partition.validation_per_class",
            "image.size",
            "loader.steps_per_epoch",
        ),
        "data_builders-image": (
            "source.name",
            "source.materialization.policy",
            "image.size",
            "partition.mode",
            "loader.steps_per_epoch",
        ),
        "data_builders-multi-resolution-image": (
            "sources[].sampling_weight",
            "sources[].source.name",
            "batching.buckets[].name",
            "batching.base_bucket",
            "loader.steps_per_epoch",
        ),
        "data_builders-super-resolution": (
            "source.params.high_resolution_root",
            "source.materialization.verification",
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

    class_labeled = _component_section(
        reference,
        "data_builders-class-labeled-image",
    )
    assert "validation 必须为 null" in class_labeled
    assert "相同 payload type 不自动代表 runtime recipe 兼容" in class_labeled

    supervised = _component_section(reference, "training_builders-supervised")
    assert "无组件级配置参数。" not in supervised
    assert "`params` 必须是空 mapping（默认 `{}`）" in supervised


def test_reference_indexes_builtin_reference_metrics() -> None:
    reference = Path("docs/configuration/reference.md").read_text(encoding="utf-8")

    fid = _component_section(reference, "metrics-fid")
    assert "optional quality" in fid
    assert "| `feature` |" in fid
    assert "| `antialias` |" in fid

    kid = _component_section(reference, "metrics-kid")
    assert "optional quality" in kid
    for parameter in ("feature", "subsets", "subset_size", "degree", "gamma", "coef"):
        assert f"| `{parameter}` |" in kid


def test_all_builtin_train_configs_load() -> None:
    paths = sorted(
        Path("examples/built-in/image-generation/configs/train").glob("*.yaml")
    )

    assert [path.name for path in paths] == ["mnist.yaml"]
    assert all(isinstance(load_config(path), StochaflowConfig) for path in paths)


def test_all_builtin_sample_configs_load() -> None:
    paths = sorted(
        Path("examples/built-in/image-generation/configs/sample").glob("*.yaml")
    )

    assert [path.name for path in paths] == ["mnist-ddim-50.yaml", "mnist-ddpm.yaml"]
    assert all(
        isinstance(load_sample_config(path), SampleInvocationConfig) for path in paths
    )
