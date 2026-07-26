"""Installed-wheel acceptance tests for the extension reference projects."""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch
import yaml
from packaging.utils import canonicalize_name
from packaging.version import Version

_REPOSITORY: Final = Path(__file__).resolve().parents[1]
_REFERENCE_ROOT: Final = _REPOSITORY / "examples/extension-projects"


@dataclass(frozen=True, slots=True)
class FixtureReferenceProject:
    directory: str
    distribution: str
    plugin: str
    package: str
    target: str
    train_config: str
    registry_names: tuple[tuple[str, str], ...]

    @property
    def source(self) -> Path:
        return _REFERENCE_ROOT / self.directory


_PHYSICS: Final = FixtureReferenceProject(
    directory="physics-reconstruction",
    distribution="stochaflow-physics-reconstruction",
    plugin="physics-reconstruction",
    package="stochaflow_physics_reconstruction",
    target="stochaflow_physics_reconstruction.stochaflow_ext",
    train_config="experiments/tiny/train.yaml",
    registry_names=(
        ("data_builders", "physics-reconstruction.kolmogorov-trajectories"),
        ("models", "physics-reconstruction.conditional-denoiser"),
        ("training_builders", "physics-reconstruction.gaussian-denoising"),
        ("sampling_builders", "physics-reconstruction.reconstruction"),
        ("samplers", "physics-reconstruction.guided-ddim"),
        (
            "sampling_artifact_writers",
            "physics-reconstruction.reconstruction-artifacts",
        ),
    ),
)

_DISTILLATION: Final = FixtureReferenceProject(
    directory="knowledge-distillation",
    distribution="stochaflow-knowledge-distillation",
    plugin="stochaflow-knowledge-distillation",
    package="stochaflow_knowledge_distillation",
    target="stochaflow_knowledge_distillation.stochaflow_ext",
    train_config="experiments/tiny/train.yaml",
    registry_names=(
        (
            "data_builders",
            "stochaflow-knowledge-distillation.classification",
        ),
        ("models", "stochaflow-knowledge-distillation.student"),
        ("models", "stochaflow-knowledge-distillation.teacher"),
        ("objectives", "stochaflow-knowledge-distillation.cross-entropy"),
        ("objectives", "stochaflow-knowledge-distillation.temperature-kl"),
        (
            "training_builders",
            "stochaflow-knowledge-distillation.training",
        ),
        (
            "sampling_builders",
            "stochaflow-knowledge-distillation.predictions",
        ),
    ),
)

_PROJECTS: Final = (_PHYSICS, _DISTILLATION)


@dataclass(frozen=True, slots=True)
class FixtureInstalledReferenceEnvironment:
    root: Path
    python: Path
    cli: Path
    environment: dict[str, str]
    projects: dict[str, Path]
    wheels: dict[str, Path]


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


def _copy_reference_project(project: FixtureReferenceProject, destination: Path) -> Path:
    return Path(
        shutil.copytree(
            project.source,
            destination,
            ignore=shutil.ignore_patterns(
                ".venv",
                "__pycache__",
                "*.pyc",
                "*.egg-info",
                ".pytest_cache",
                ".ruff_cache",
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
from pathlib import Path
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
) -> tuple[Path, Path, dict[str, str]]:
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
    return (
        environment_python,
        _environment_executable(environment_root, "stochaflow"),
        _subprocess_environment(environment_root),
    )


@pytest.fixture(scope="module")
def installed_reference_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> FixtureInstalledReferenceEnvironment:
    root = tmp_path_factory.mktemp("installed-reference-projects")
    source_root = root / "sources"
    source_root.mkdir()
    root_source = _copy_stochaflow_source(source_root / "stochaflow")
    project_copies = {
        project.directory: _copy_reference_project(
            project,
            source_root / project.directory,
        )
        for project in _PROJECTS
    }
    wheel_root = root / "wheels"
    wheel_root.mkdir()
    wheels = {
        "stochaflow": _build_wheel(root_source, wheel_root / "stochaflow"),
        **{
            project.directory: _build_wheel(
                project_copies[project.directory],
                wheel_root / project.directory,
            )
            for project in _PROJECTS
        },
    }
    python, cli, environment = _create_installed_environment(
        root,
        tuple(wheels.values()),
    )
    assert python.is_file()
    assert cli.is_file()
    return FixtureInstalledReferenceEnvironment(
        root=root,
        python=python,
        cli=cli,
        environment=environment,
        projects=project_copies,
        wheels=wheels,
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


def _only_run(output_root: Path) -> Path:
    runs = tuple(path for path in output_root.iterdir() if path.is_dir())
    assert len(runs) == 1
    return runs[0]


def _run_cli(
    installed: FixtureInstalledReferenceEnvironment,
    project_root: Path,
    *arguments: str | Path,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [installed.cli, *arguments],
        cwd=project_root,
        environment=installed.environment,
    )


def _expected_provenance(project: FixtureReferenceProject) -> list[dict[str, str]]:
    declaration = tomllib.loads(
        (project.source / "pyproject.toml").read_text(encoding="utf-8")
    )
    return [
        {
            "name": project.plugin,
            "distribution": project.distribution,
            "version": declaration["project"]["version"],
            "target": project.target,
        }
    ]


def _assert_manifest_provenance(
    manifest_path: Path,
    project: FixtureReferenceProject,
) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["extension_plugins"] == _expected_provenance(project)


def _assert_component_metadata_persisted(
    manifest_path: Path,
    checkpoint_path: Path,
) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload: object = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert isinstance(payload, dict)
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata["selected_components"] == manifest["selected_components"]


def _checkpoint_assets(path: Path) -> dict[str, dict[str, object]]:
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    assets = payload.get("training_assets_state_dict")
    assert isinstance(assets, dict)
    assert all(
        isinstance(name, str) and isinstance(state, dict)
        for name, state in assets.items()
    )
    return assets


def _assert_tensor_state_equal(
    expected: dict[str, object],
    actual: dict[str, object],
) -> None:
    assert set(actual) == set(expected)
    for name, expected_value in expected.items():
        actual_value = actual[name]
        assert isinstance(expected_value, torch.Tensor)
        assert isinstance(actual_value, torch.Tensor)
        assert torch.equal(actual_value, expected_value)


def _load_tensor_state(path: Path) -> dict[str, object]:
    value: object = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    assert all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in value.items()
    )
    return value


def _tensor_state_differs(
    first: dict[str, object],
    second: dict[str, object],
) -> bool:
    assert set(first) == set(second)
    differs = False
    for name, first_value in first.items():
        second_value = second[name]
        assert isinstance(first_value, torch.Tensor)
        assert isinstance(second_value, torch.Tensor)
        differs = differs or not torch.equal(first_value, second_value)
    return differs


def test_reference_projects_are_independent_installable_distributions() -> None:
    root_version = Version(
        tomllib.loads((_REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
    )
    for project in _PROJECTS:
        assert project.source.is_dir()
        declaration = tomllib.loads(
            (project.source / "pyproject.toml").read_text(encoding="utf-8")
        )
        assert canonicalize_name(declaration["project"]["name"]) == project.distribution
        dependencies = declaration["project"]["dependencies"]
        assert f"stochaflow=={root_version}" in dependencies
        assert all("file:" not in item and " @ " not in item for item in dependencies)
        assert declaration["project"]["entry-points"]["stochaflow.extensions"] == {
            project.plugin: project.target
        }
        package_root = project.source / "src" / project.package
        assert (package_root / "__init__.py").is_file()
        assert (
            package_root / "stochaflow_ext" / "__init__.py"
        ).is_file()
        assert (project.source / project.train_config).is_file()


@pytest.mark.parametrize("project", _PROJECTS, ids=lambda item: item.directory)
def test_reference_project_acceptance_copy_excludes_local_data(
    tmp_path: Path,
    project: FixtureReferenceProject,
) -> None:
    project_copy = _copy_reference_project(
        project,
        tmp_path / project.directory,
    )

    assert not (project_copy / "data").exists()
    assert not (project_copy / "outputs").exists()


@pytest.mark.parametrize(
    ("profile", "accepted_steps", "partial_noise_time", "sampler_name"),
    [
        ("sample-baseline-ddim.yaml", 30, 240, "ddim"),
        (
            "sample-guided-ddim.yaml",
            40,
            320,
            "physics-reconstruction.guided-ddim",
        ),
    ],
)
def test_physics_real_smoke_profiles_preserve_production_solver_math(
    profile: str,
    accepted_steps: int,
    partial_noise_time: int,
    sampler_name: str,
) -> None:
    production_document = yaml.safe_load(
        (_PHYSICS.source / "experiments/production" / profile).read_text(
            encoding="utf-8"
        )
    )
    smoke_document = yaml.safe_load(
        (_PHYSICS.source / "experiments/real-smoke" / profile).read_text(
            encoding="utf-8"
        )
    )
    normalized_smoke = deepcopy(smoke_document)
    normalized_sampling = normalized_smoke["sampling"]
    production_sampling = production_document["sampling"]
    normalized_sampling["num_samples"] = production_sampling["num_samples"]
    normalized_sampling["batch_size"] = production_sampling["batch_size"]
    normalized_sampling["builder"]["params"]["weights"] = production_sampling[
        "builder"
    ]["params"]["weights"]
    assert normalized_smoke == production_document

    smoke = smoke_document["sampling"]
    smoke_params = smoke["builder"]["params"]
    assert smoke["shape"] == [3, 256, 256]
    assert smoke["num_samples"] == smoke["batch_size"] == 1
    assert smoke_params["weights"] == "raw"
    schedule = smoke_params["sampler"]["params"]["schedule"]
    assert smoke_params["partial_noise_time"] == partial_noise_time
    assert smoke_params["sampler"]["name"] == sampler_name
    assert len(schedule) - 1 == accepted_steps
    assert schedule[0] == partial_noise_time
    assert schedule[-1] == 0


def test_reference_project_wheels_have_isolated_entry_points(
    installed_reference_environment: FixtureInstalledReferenceEnvironment,
) -> None:
    installed = installed_reference_environment
    with zipfile.ZipFile(installed.wheels["stochaflow"]) as archive:
        root_members = tuple(archive.namelist())
    assert not any(
        project.package in member
        for project in _PROJECTS
        for member in root_members
    )

    for project in _PROJECTS:
        wheel = installed.wheels[project.directory]
        with zipfile.ZipFile(wheel) as archive:
            members = tuple(archive.namelist())
        assert any(member.startswith(f"{project.package}/") for member in members)
        assert not any(
            member.startswith(("tests/", "tools/", "data/", "experiments/"))
            for member in members
        )
        entry_points = _entry_points_from_wheel(wheel)
        assert dict(entry_points["stochaflow.extensions"]) == {
            project.plugin: project.target
        }


@pytest.mark.parametrize("project", _PROJECTS, ids=lambda item: item.directory)
def test_installed_entry_point_is_the_only_activation_path(
    installed_reference_environment: FixtureInstalledReferenceEnvironment,
    project: FixtureReferenceProject,
) -> None:
    installed = installed_reference_environment
    other_registry_names = [
        registration
        for other in _PROJECTS
        if other is not project
        for registration in other.registry_names
    ]
    script = r"""
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys

from stochaflow.extensions import (
    REGISTRIES,
    activate_extension_plugins,
    prepare_extension_plugins,
)
from stochaflow.utils.config import load_config

arguments = json.loads(sys.argv[1])
for raw_path in sys.path:
    candidate = Path(raw_path or ".").resolve()
    assert all(candidate != Path(value).resolve() for value in arguments["forbidden"])

import stochaflow
assert Path(stochaflow.__file__).resolve().is_relative_to(
    Path(arguments["environment_root"]).resolve()
)
distribution = metadata.distribution(arguments["distribution"])
assert Path(distribution.locate_file("")).resolve().is_relative_to(
    Path(arguments["environment_root"]).resolve()
)
matches = [
    item
    for item in distribution.entry_points
    if item.group == "stochaflow.extensions"
]
assert [(item.name, item.value) for item in matches] == [
    (arguments["plugin"], arguments["target"])
]

config = load_config(arguments["config"])
plan = prepare_extension_plugins(config)
assert [
    (item.name, item.distribution, item.target)
    for item in plan.provenance
] == [
    (
        arguments["plugin"],
        arguments["distribution"],
        arguments["target"],
    )
]
activate_extension_plugins(plan)
module = importlib.import_module(arguments["target"])
assert Path(module.__file__).resolve().is_relative_to(
    Path(arguments["environment_root"]).resolve()
)
for registry_name, component_name in arguments["registry_names"]:
    assert component_name in getattr(REGISTRIES, registry_name).names()
for registry_name, component_name in arguments["other_registry_names"]:
    assert component_name not in getattr(REGISTRIES, registry_name).names()
"""
    arguments = {
        "environment_root": str(installed.root / "harness/.venv"),
        "forbidden": [
            str(_REPOSITORY / "src"),
            *(str(item.source / "src") for item in _PROJECTS),
            *(str(path / "src") for path in installed.projects.values()),
        ],
        "distribution": project.distribution,
        "plugin": project.plugin,
        "target": project.target,
        "config": str(installed.projects[project.directory] / project.train_config),
        "registry_names": project.registry_names,
        "other_registry_names": other_registry_names,
    }
    _run(
        [installed.python, "-c", script, json.dumps(arguments)],
        cwd=installed.root / "harness",
        environment=installed.environment,
    )


def test_physics_production_training_config_composes_installed_extension(
    installed_reference_environment: FixtureInstalledReferenceEnvironment,
) -> None:
    installed = installed_reference_environment
    script = r"""
import json
import sys

from stochaflow.extensions import (
    REGISTRIES,
    activate_extension_plugins,
    prepare_extension_plugins,
)
from stochaflow.training.builder import build_training_plan
from stochaflow.utils.config import load_config
from stochaflow.utils.factory import build_model, build_objective, build_process

arguments = json.loads(sys.argv[1])
config = load_config(arguments["config"])
plugin_plan = prepare_extension_plugins(config)
assert [
    (item.name, item.distribution, item.target)
    for item in plugin_plan.provenance
] == [
    (
        arguments["plugin"],
        arguments["distribution"],
        arguments["target"],
    )
]
activate_extension_plugins(plugin_plan)

for registry_name, component_name in arguments["registry_names"]:
    assert component_name in getattr(REGISTRIES, registry_name).names()

assert config.extensions.plugins == [arguments["plugin"]]
assert config.data.name == "physics-reconstruction.kolmogorov-trajectories"
assert config.data.params["path"] == "data/kolmogorov.npy"
assert config.data.params["loader"]["batch_size"] == 4
assert config.model.name == "physics-reconstruction.conditional-denoiser"
assert {
    name: config.model.params[name]
    for name in ("hidden_channels", "num_blocks", "time_embedding_dim")
} == {
    "hidden_channels": 64,
    "num_blocks": 6,
    "time_embedding_dim": 128,
}
assert config.training.name == "physics-reconstruction.gaussian-denoising"
assert config.process is not None
assert config.process.name == "discrete_gaussian"
assert config.process.params["schedule"]["params"]["num_timesteps"] == 1000
assert config.objective is not None
assert config.objective.name == "mse"
assert config.optimizer.name == "torch.optim.Adam"
assert config.lr_scheduler is not None
assert config.lr_scheduler.name == "torch.optim.lr_scheduler.CosineAnnealingLR"
assert config.sampling.builder is not None
assert config.sampling.builder.name == "physics-reconstruction.reconstruction"
assert [writer.name for writer in config.sampling.writers] == [
    "physics-reconstruction.reconstruction-artifacts"
]

model = build_model(config.model)
process = build_process(config.process)
objective = build_objective(config.objective)
training_plan = build_training_plan(
    config.training,
    primary_model=model,
    process=process,
    objective=objective,
    model_factory=build_model,
    objective_factory=build_objective,
)
assert training_plan.primary_model is model
assert training_plan.process is process
assert training_plan.objective is objective
assert type(model).__name__ == "ConditionalDenoiser"
assert type(training_plan.strategy).__name__ == "PhysicsDenoisingStrategy"
assert training_plan.auxiliary_modules == {}
"""
    arguments = {
        "config": str(
            installed.projects[_PHYSICS.directory]
            / "experiments/production/train.yaml"
        ),
        "plugin": _PHYSICS.plugin,
        "distribution": _PHYSICS.distribution,
        "target": _PHYSICS.target,
        "registry_names": _PHYSICS.registry_names,
    }
    _run(
        [installed.python, "-c", script, json.dumps(arguments)],
        cwd=installed.projects[_PHYSICS.directory],
        environment=installed.environment,
    )


@pytest.mark.parametrize("project", _PROJECTS, ids=lambda item: item.directory)
def test_reference_project_unit_suite_uses_installed_wheel(
    installed_reference_environment: FixtureInstalledReferenceEnvironment,
    project: FixtureReferenceProject,
) -> None:
    installed = installed_reference_environment
    project_copy = installed.projects[project.directory]
    _run(
        [
            installed.python,
            "-m",
            "pytest",
            "--import-mode=importlib",
            "-q",
            project_copy / "tests",
        ],
        cwd=project_copy,
        environment=installed.environment,
    )


def test_distillation_cli_train_resume_and_student_only_sample(
    installed_reference_environment: FixtureInstalledReferenceEnvironment,
) -> None:
    installed = installed_reference_environment
    project_root = installed.projects[_DISTILLATION.directory]
    teacher_state = project_root / "data/teacher.pt"
    _run(
        [
            installed.python,
            project_root / "tools/create_teacher_bootstrap.py",
            "--output",
            teacher_state,
        ],
        cwd=project_root,
        environment=installed.environment,
    )
    assert teacher_state.is_file()
    initial_bootstrap = _load_tensor_state(teacher_state)

    _run_cli(
        installed,
        project_root,
        "train",
        "--config",
        _DISTILLATION.train_config,
        "--epochs",
        "1",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--limit-test-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    first_run = _only_run(project_root / "outputs/tiny")
    first_checkpoint = first_run / "checkpoints/latest.pt"
    assert first_checkpoint.is_file()
    _assert_manifest_provenance(first_run / "run_manifest.yaml", _DISTILLATION)
    _assert_component_metadata_persisted(
        first_run / "run_manifest.yaml",
        first_checkpoint,
    )
    first_assets = _checkpoint_assets(first_checkpoint)
    assert set(first_assets) == {"distillation_objective", "teacher"}
    assert set(first_assets["distillation_objective"]) == {"temperature"}
    _assert_tensor_state_equal(initial_bootstrap, first_assets["teacher"])

    _run(
        [
            installed.python,
            project_root / "tools/create_teacher_bootstrap.py",
            "--output",
            teacher_state,
            "--seed",
            "314159",
        ],
        cwd=project_root,
        environment=installed.environment,
    )
    changed_bootstrap = _load_tensor_state(teacher_state)
    assert _tensor_state_differs(initial_bootstrap, changed_bootstrap)

    resumed_root = project_root / "outputs/resumed"
    _run_cli(
        installed,
        project_root,
        "train",
        "--resume",
        first_run,
        "--output-dir",
        resumed_root,
        "--epochs",
        "2",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--limit-test-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    resumed_run = _only_run(resumed_root)
    resumed_checkpoint = resumed_run / "checkpoints/latest.pt"
    resumed_assets = _checkpoint_assets(resumed_checkpoint)
    _assert_tensor_state_equal(first_assets["teacher"], resumed_assets["teacher"])
    _assert_tensor_state_equal(
        first_assets["distillation_objective"],
        resumed_assets["distillation_objective"],
    )
    _assert_manifest_provenance(
        resumed_run / "run_manifest.yaml",
        _DISTILLATION,
    )
    _assert_component_metadata_persisted(
        resumed_run / "run_manifest.yaml",
        resumed_checkpoint,
    )

    teacher_state.unlink()
    sample_root = project_root / "sample-output"
    _run_cli(
        installed,
        project_root,
        "sample",
        "--checkpoint",
        resumed_run,
        "--output-dir",
        sample_root,
        "--device",
        "cpu",
    )
    assert (sample_root / "samples.pt").is_file()
    samples: object = torch.load(
        sample_root / "samples.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert isinstance(samples, torch.Tensor)
    assert samples.shape == (8, 4)
    _assert_manifest_provenance(
        sample_root / "resolved_sampling.yaml",
        _DISTILLATION,
    )
    sample_manifest = yaml.safe_load(
        (sample_root / "resolved_sampling.yaml").read_text(encoding="utf-8")
    )
    assert sample_manifest["builder"]["name"] == (
        "stochaflow-knowledge-distillation.predictions"
    )


def test_physics_cli_train_resume_and_sample_variants(
    installed_reference_environment: FixtureInstalledReferenceEnvironment,
) -> None:
    installed = installed_reference_environment
    project_root = installed.projects[_PHYSICS.directory]
    _run(
        [
            installed.python,
            "-m",
            "stochaflow_physics_reconstruction.tools.prepare_tiny_data",
            "--output-dir",
            project_root / "data/tiny",
        ],
        cwd=project_root,
        environment=installed.environment,
    )
    for filename in ("trajectories.npy", "observations.npy", "references.npy"):
        assert (project_root / "data/tiny" / filename).is_file()

    first_output = project_root / "outputs/tiny"
    _run_cli(
        installed,
        project_root,
        "train",
        "--config",
        _PHYSICS.train_config,
        "--epochs",
        "1",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--limit-test-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    first_run = _only_run(first_output)
    first_checkpoint = first_run / "checkpoints/latest.pt"
    assert first_checkpoint.is_file()
    _assert_manifest_provenance(first_run / "run_manifest.yaml", _PHYSICS)
    _assert_component_metadata_persisted(
        first_run / "run_manifest.yaml",
        first_checkpoint,
    )

    resumed_root = project_root / "outputs/resumed"
    _run_cli(
        installed,
        project_root,
        "train",
        "--resume",
        first_run,
        "--output-dir",
        resumed_root,
        "--epochs",
        "2",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--limit-test-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    resumed_run = _only_run(resumed_root)
    resumed_checkpoint = resumed_run / "checkpoints/latest.pt"
    assert resumed_checkpoint.is_file()
    _assert_manifest_provenance(resumed_run / "run_manifest.yaml", _PHYSICS)
    _assert_component_metadata_persisted(
        resumed_run / "run_manifest.yaml",
        resumed_checkpoint,
    )

    variants = (
        ("sample-baseline-ddim.yaml", "ddim"),
        ("sample-baseline-ddpm.yaml", "ddpm"),
        ("sample-guided-ddim.yaml", "physics-reconstruction.guided-ddim"),
    )
    for overlay_name, sampler_name in variants:
        overlay = project_root / "experiments/tiny" / overlay_name
        sample_root = project_root / "sample-output" / overlay.stem
        _run_cli(
            installed,
            project_root,
            "sample",
            "--checkpoint",
            resumed_run,
            "--config",
            overlay,
            "--output-dir",
            sample_root,
            "--device",
            "cpu",
        )
        reconstruction = np.load(sample_root / "reconstructions.npy")
        overlay_config = yaml.safe_load(overlay.read_text(encoding="utf-8"))
        num_samples = overlay_config["sampling"]["num_samples"]
        assert reconstruction.shape == (num_samples, 3, 8, 8)
        assert reconstruction.dtype == np.float32
        metrics = json.loads(
            (sample_root / "metrics.json").read_text(encoding="utf-8")
        )
        assert metrics["num_samples"] == num_samples
        manifest_path = sample_root / "resolved_sampling.yaml"
        _assert_manifest_provenance(manifest_path, _PHYSICS)
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert manifest["builder"]["name"] == (
            "physics-reconstruction.reconstruction"
        )
        assert manifest["metadata"]["sampler"]["name"] == sampler_name
