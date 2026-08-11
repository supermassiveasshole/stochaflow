"""Installed-wheel acceptance tests for core composition and AFHQ Evaluation."""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

_REPOSITORY: Final = Path(__file__).resolve().parents[1]
_AFHQ_ROOT: Final = _REPOSITORY / "examples/showcases/afhq-v2"


@dataclass(frozen=True, slots=True)
class FixtureInstalledWheelEnvironment:
    """An isolated environment containing current core and AFHQ wheels."""

    root: Path
    python: Path
    environment: dict[str, str]
    core_source: Path
    afhq_source: Path
    core_wheel: Path
    afhq_wheel: Path


def _run(
    arguments: list[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(argument) for argument in arguments],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"command failed: {' '.join(map(str, arguments))}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _copy_stochaflow_source(destination: Path) -> Path:
    destination.mkdir(parents=True)
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(_REPOSITORY / filename, destination / filename)
    shutil.copytree(
        _REPOSITORY / "src/stochaflow",
        destination / "src/stochaflow",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    return destination


def _copy_installable_project(source: Path, destination: Path) -> Path:
    return Path(
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "*.egg-info",
                ".pytest_cache",
                ".ruff_cache",
                ".stochaflow-cache",
                "build",
                "data",
                "dist",
                "outputs",
            ),
        )
    )


def _build_wheel(source: Path, output: Path) -> Path:
    script = """
import os
import sys
from setuptools.build_meta import build_wheel

os.chdir(sys.argv[1])
print(build_wheel(sys.argv[2]))
"""
    output.mkdir(parents=True)
    build_environment = dict(os.environ)
    build_environment.pop("PYTHONPATH", None)
    build_environment.pop("PYTHONHOME", None)
    build_environment["PYTHONNOUSERSITE"] = "1"
    build_environment["PYTHONSAFEPATH"] = "1"
    _run(
        [sys.executable, "-c", script, source, output],
        cwd=output.parent,
        environment=build_environment,
    )
    wheels = tuple(output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _environment_executable(root: Path, name: str) -> Path:
    if os.name == "nt":
        return root / "Scripts" / f"{name}.exe"
    return root / "bin" / name


def _subprocess_environment(environment_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    ):
        environment.pop(name, None)
    environment["PATH"] = os.pathsep.join(
        (
            str(_environment_executable(environment_root, "python").parent),
            environment.get("PATH", ""),
        )
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    return environment


def _create_installed_environment(
    root: Path,
    wheels: tuple[Path, ...],
) -> tuple[Path, dict[str, str]]:
    """Install current-checkout wheels without resolving published dependencies."""

    uv = shutil.which("uv")
    assert uv is not None, "the repository test environment requires the uv CLI"
    harness = root / "harness"
    harness.mkdir()
    for filename in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(_REPOSITORY / filename, harness / filename)
    sync_environment = dict(os.environ)
    sync_environment.pop("UV_PROJECT_ENVIRONMENT", None)
    sync_environment.pop("VIRTUAL_ENV", None)
    _run(
        [
            uv,
            "sync",
            "--project",
            harness,
            "--python",
            sys.executable,
            "--frozen",
            "--extra",
            "dev",
            "--no-install-project",
            "--offline",
        ],
        cwd=harness,
        environment=sync_environment,
        timeout=300,
    )
    environment_root = harness / ".venv"
    environment_python = _environment_executable(environment_root, "python")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            environment_python,
            "--no-deps",
            "--reinstall",
            "--offline",
            *wheels,
        ],
        cwd=harness,
        environment=sync_environment,
        timeout=300,
    )
    return environment_python, _subprocess_environment(environment_root)


@pytest.fixture(scope="module")
def installed_wheel_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> FixtureInstalledWheelEnvironment:
    root = tmp_path_factory.mktemp("installed-wheel-acceptance")
    source_root = root / "sources"
    source_root.mkdir()
    core_source = _copy_stochaflow_source(source_root / "stochaflow")
    afhq_source = _copy_installable_project(_AFHQ_ROOT, source_root / "afhq-v2")
    wheel_root = root / "wheels"
    wheel_root.mkdir()
    core_wheel = _build_wheel(core_source, wheel_root / "stochaflow")
    afhq_wheel = _build_wheel(afhq_source, wheel_root / "afhq-v2")
    python, environment = _create_installed_environment(
        root,
        (core_wheel, afhq_wheel),
    )
    assert python.is_file()
    return FixtureInstalledWheelEnvironment(
        root=root,
        python=python,
        environment=environment,
        core_source=core_source,
        afhq_source=afhq_source,
        core_wheel=core_wheel,
        afhq_wheel=afhq_wheel,
    )


def _entry_points_from_wheel(wheel: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            name for name in archive.namelist() if name.endswith("entry_points.txt")
        ]
        assert len(matches) == 1
        parser.read_string(archive.read(matches[0]).decode("utf-8"))
    return parser


def test_installed_core_wheel_preserves_runtime_composition_boundaries(
    installed_wheel_environment: FixtureInstalledWheelEnvironment,
) -> None:
    """Exercise each built-in scope from the installed core wheel."""

    installed = installed_wheel_environment
    with zipfile.ZipFile(installed.core_wheel) as archive:
        members = set(archive.namelist())
    assert {
        "stochaflow/_builtin_activation.py",
        "stochaflow/_component_factory.py",
        "stochaflow/training/composition.py",
        "stochaflow/scripts/training_arguments.py",
        "stochaflow/utils/logging_contracts.py",
        "stochaflow/utils/logging_paths.py",
        "stochaflow/utils/torch_logging.py",
    } <= members
    script = r"""
from pathlib import Path
import sys

mode = sys.argv[1]
environment_root = Path(sys.argv[2]).resolve()

import stochaflow
from stochaflow import _builtin_activation as activation
from stochaflow.utils.registry import REGISTRIES

assert Path(stochaflow.__file__).resolve().is_relative_to(environment_root)

if mode == "parser":
    from stochaflow.scripts.cli import build_argument_parser
    build_argument_parser()
    assert activation._activation_runtime.completed_modules == set()
elif mode == "sampling":
    activation.activate_sampling_builtins()
    assert activation._activation_runtime.completed_modules == set(
        activation.SAMPLING_BUILTIN_MODULES
    )
    assert REGISTRIES.models.names() == ("adm_unet", "dit", "unet")
    assert REGISTRIES.processes.names() == ("discrete_gaussian",)
    assert REGISTRIES.samplers.names() == ("ddim", "ddpm")
    assert REGISTRIES.training_builders.names() == ()
    assert REGISTRIES.loggers.names() == ()
elif mode == "evaluation":
    activation.activate_evaluation_builtins()
    assert activation._activation_runtime.completed_modules == set(
        activation.EVALUATION_BUILTIN_MODULES
    )
    assert REGISTRIES.data_builders.names() == (
        "class_labeled_image", "image", "multi_resolution_image",
        "super_resolution",
    )
    assert REGISTRIES.metrics.names() == ("fid", "kid", "mae", "mean", "mse")
    assert REGISTRIES.training_builders.names() == ()
    assert REGISTRIES.evaluation_builders.names() == ()
    assert REGISTRIES.loggers.names() == ()
elif mode == "training":
    activation.activate_training_builtins()
    assert activation._activation_runtime.completed_modules == set(
        activation.TRAINING_BUILTIN_MODULES
    )
    assert REGISTRIES.training_builders.names() == (
        "class_conditional_gaussian_denoising", "gaussian_denoising",
        "supervised",
    )
    assert REGISTRIES.diagnostics.names() == (
        "class_conditional_diffusion_quality", "diffusion_quality",
    )
    assert REGISTRIES.loggers.names() == ("local", "tensorboard", "wandb")
else:
    raise AssertionError(f"unknown mode: {mode}")

if mode != "training":
    assert "stochaflow.training.trainer" not in sys.modules
    assert not any(
        name == "stochaflow.training.diagnostics"
        or name.startswith("stochaflow.training.diagnostics.")
        for name in sys.modules
    )
"""
    environment_root = installed.root / "harness/.venv"
    for mode in ("parser", "sampling", "evaluation", "training"):
        _run(
            [installed.python, "-I", "-c", script, mode, environment_root],
            cwd=installed.root / "harness",
            environment=installed.environment,
        )


def test_afhq_formal_evaluation_activates_with_current_core_wheel(
    installed_wheel_environment: FixtureInstalledWheelEnvironment,
) -> None:
    """Verify packaged AFHQ content and activation against the current core wheel."""

    installed = installed_wheel_environment
    entry_points = _entry_points_from_wheel(installed.afhq_wheel)
    assert dict(entry_points["stochaflow.extensions"]) == {
        "stochaflow-afhq-v2": "stochaflow_afhq_v2.stochaflow_ext"
    }
    with zipfile.ZipFile(installed.afhq_wheel) as archive:
        members = tuple(archive.namelist())
    assert "stochaflow_afhq_v2/stochaflow_ext/evaluation.py" in members

    script = r"""
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys

from torchmetrics import Metric

from stochaflow.evaluation import EvaluationBuilder, load_evaluation_config
from stochaflow.extensions import (
    REGISTRIES,
    activate_extension_plugins,
    prepare_extension_plugins,
)
from stochaflow.utils.config import load_config

arguments = json.loads(sys.argv[1])
environment_root = Path(arguments["environment_root"]).resolve()
for raw_path in sys.path:
    candidate = Path(raw_path or ".").resolve()
    assert all(
        candidate != Path(value).resolve()
        for value in arguments["forbidden"]
    )

import stochaflow
assert Path(stochaflow.__file__).resolve().is_relative_to(environment_root)
distribution = metadata.distribution("stochaflow-afhq-v2")
assert Path(distribution.locate_file("")).resolve().is_relative_to(environment_root)
matches = [
    item
    for item in distribution.entry_points
    if item.group == "stochaflow.extensions"
]
assert [(item.name, item.value) for item in matches] == [
    ("stochaflow-afhq-v2", "stochaflow_afhq_v2.stochaflow_ext")
]

builder_name = "afhq-v2.class-conditional-generation"
metric_name = "afhq-v2.class-aware-distribution"
assert builder_name not in REGISTRIES.evaluation_builders.names()
assert metric_name not in REGISTRIES.metrics.names()
assert "stochaflow_afhq_v2.stochaflow_ext.evaluation" not in sys.modules

profiles = [load_evaluation_config(path) for path in arguments["profiles"]]
assert {
    profile.evaluation.params["sampling"]["recipe"]["contract"][
        "prediction_type"
    ]
    for profile in profiles
} == {"v"}
assert {
    profile.name: (profile.purpose, profile.data.split, profile.protocol.expected_examples)
    for profile in profiles
} == {
    "afhq-v2-adm-ddim50-cfg2-official-test-v1": ("final_test", "test", 1467),
}
for profile in profiles:
    assert profile.extensions.plugins == ("stochaflow-afhq-v2",)
    assert profile.evaluation.name == builder_name
    assert len(profile.metrics) == 1
    assert profile.metrics[0].name == metric_name
    assert profile.protocol.strict_complete is True

training_config = load_config(arguments["training_config"])
assert training_config.experiment.name == "afhq_v2_adm_128"
assert training_config.training.name == "class_conditional_gaussian_denoising"
assert training_config.trainer.device == "auto"
plan = prepare_extension_plugins(training_config)
assert [
    (item.name, item.distribution, item.target)
    for item in plan.provenance
] == [
    (
        "stochaflow-afhq-v2",
        "stochaflow-afhq-v2",
        "stochaflow_afhq_v2.stochaflow_ext",
    )
]
activate_extension_plugins(plan)

formal_module = importlib.import_module(
    "stochaflow_afhq_v2.stochaflow_ext.evaluation"
)
assert Path(formal_module.__file__).resolve().is_relative_to(environment_root)
builder_type = REGISTRIES.evaluation_builders[builder_name]
metric_type = REGISTRIES.metrics[metric_name]
assert builder_type is formal_module.AFHQV2GenerationEvaluationBuilder
assert metric_type is formal_module.AFHQV2ClassAwareDistributionMetric
assert issubclass(builder_type, EvaluationBuilder)
assert issubclass(metric_type, Metric)
"""
    arguments = {
        "environment_root": str(installed.root / "harness/.venv"),
        "forbidden": [
            str(_REPOSITORY / "src"),
            str(_AFHQ_ROOT / "src"),
            str(installed.core_source / "src"),
            str(installed.afhq_source / "src"),
        ],
        "training_config": str(
            installed.afhq_source / "experiments/production/train-adm-128.yaml"
        ),
        "profiles": [
            str(
                installed.afhq_source
                / "experiments/evaluation/formal-ddim50-cfg2-official-test.yaml"
            ),
        ],
    }
    _run(
        [installed.python, "-c", script, json.dumps(arguments)],
        cwd=installed.root / "harness",
        environment=installed.environment,
    )
