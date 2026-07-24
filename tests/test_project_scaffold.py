"""Project scaffold and init CLI contract tests."""

from __future__ import annotations

from importlib import metadata, resources
import os
from pathlib import Path
from runpy import run_path
import shutil
import site
import subprocess
import sys
import tarfile
import tomllib
from typing import Any, TextIO, cast
import zipfile

from packaging.version import Version
import pytest
import yaml

from stochaflow.extensions import DataBuilderContext
from stochaflow.projects import (
    ProjectScaffoldError,
    create_project,
    validate_project_name,
)
from stochaflow.projects import scaffold
from stochaflow.scripts.cli import build_argument_parser, main


EXPECTED_FILES = {
    ".gitignore",
    "README.md",
    "data/.gitkeep",
    "experiments/example/train.yaml",
    "notebooks/.gitkeep",
    "pyproject.toml",
    "src/example_lab/__init__.py",
    "src/example_lab/stochaflow_ext/__init__.py",
    "src/example_lab/stochaflow_ext/data.py",
    "src/example_lab/stochaflow_ext/diagnostics.py",
    "src/example_lab/stochaflow_ext/model.py",
    "src/example_lab/stochaflow_ext/sampling.py",
    "src/example_lab/stochaflow_ext/training.py",
    "tests/test_extensions.py",
}


def _generated_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _build_distribution(
    source: Path,
    output: Path,
    *,
    include_sdist: bool,
) -> tuple[Path, Path | None]:
    script = """
import os
from pathlib import Path
import sys
from setuptools.build_meta import build_sdist, build_wheel

os.chdir(sys.argv[1])
output = sys.argv[2]
sdist = build_sdist(output) if sys.argv[3] == "true" else None
wheel = build_wheel(output)
print(wheel)
print(sdist or "")
"""
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(source),
            str(output),
            "true" if include_sdist else "false",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(output.glob("*.whl"))
    sdists = tuple(output.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == int(include_sdist)
    return wheels[0], sdists[0] if sdists else None


def _only_directory(parent: Path) -> Path:
    directories = tuple(path for path in parent.iterdir() if path.is_dir())
    assert len(directories) == 1
    return directories[0]


def _run_generated_cli(
    executable: Path,
    project: Path,
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [executable, *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_create_project_writes_deterministic_installable_distribution(
    tmp_path: Path,
) -> None:
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    first = create_project("example-lab", cwd=first_parent)
    second = create_project("example-lab", cwd=second_parent)

    first_files = _generated_files(first)
    assert set(first_files) == EXPECTED_FILES
    assert first_files == _generated_files(second)
    assert not (first / "outputs").exists()
    assert all(b"__STOCHAFLOW_" not in content for content in first_files.values())

    project = tomllib.loads((first / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "example-lab"
    assert project["project"]["dependencies"] == [
        f"stochaflow=={Version(metadata.version('stochaflow'))}",
        "torch>=2.2,<3",
    ]
    assert project["project"]["entry-points"]["stochaflow.extensions"] == {
        "example-lab": "example_lab.stochaflow_ext"
    }
    config = yaml.safe_load(
        (first / "experiments/example/train.yaml").read_text(encoding="utf-8")
    )
    assert config["extensions"] == {"plugins": ["example-lab"]}
    assert config["process"] is None
    assert config["lr_scheduler"] == {
        "name": "torch.optim.lr_scheduler.StepLR",
        "interval": "epoch",
        "params": {"step_size": 1, "gamma": 0.9},
    }
    assert config["ema"] == {
        "enabled": True,
        "decay": 0.9,
        "update_after_step": 0,
        "update_every": 1,
        "use_for_sampling": True,
    }
    assert config["diagnostics"] == [
        {
            "name": "example-lab.regression-quality",
            "params": {"every_steps": 1},
        }
    ]
    assert [backend["name"] for backend in config["logging"]["backends"]] == [
        "local",
        "tensorboard",
    ]
    assert config["sampling"]["writers"] == [{"name": "tensor", "params": {}}]
    for relative_path in sorted(EXPECTED_FILES):
        if relative_path.endswith(".py"):
            source = (first / relative_path).read_text(encoding="utf-8")
            compile(source, str(first / relative_path), "exec")


def test_generated_readme_documents_the_complete_user_path(tmp_path: Path) -> None:
    project = create_project("workflow-lab", cwd=tmp_path)
    readme = (project / "README.md").read_text(encoding="utf-8")

    for generated_path in (
        "pyproject.toml",
        "stochaflow_ext/__init__.py",
        "stochaflow_ext/data.py",
        "stochaflow_ext/diagnostics.py",
        "stochaflow_ext/model.py",
        "stochaflow_ext/training.py",
        "stochaflow_ext/sampling.py",
        "experiments/example/train.yaml",
        "tests/test_extensions.py",
    ):
        assert generated_path in readme
    for command in (
        'python -m pip install -e ".[test]"',
        "python -m pytest",
        "stochaflow train --config experiments/example/train.yaml",
        "stochaflow train \\",
        "--resume outputs/example/<run-id>",
        "stochaflow sample \\",
        "--checkpoint outputs/example/<run-id>/checkpoints/best.pt",
        "tensorboard --logdir outputs/example/<run-id>/tensorboard",
    ):
        assert command in readme
    assert "strict resume" in readme.lower()
    assert "outputs/example/<run-id>/" in readme
    assert "samples.pt" in readme
    assert "resolved_sampling.yaml" in readme


def test_generated_training_shuffle_is_rebuilt_from_seed_and_epoch(
    tmp_path: Path,
) -> None:
    project = create_project("epoch-aware-lab", cwd=tmp_path)
    namespace = run_path(
        str(project / "src/epoch_aware_lab/stochaflow_ext/data.py"),
        run_name="epoch_aware_lab_generated_data",
    )
    builder_type = cast(Any, namespace["SyntheticRegressionDataBuilder"])

    def order(epoch: int) -> list[int]:
        builder = builder_type(
            DataBuilderContext(
                params={
                    "train_samples": 32,
                    "validation_samples": 8,
                    "batch_size": 8,
                },
                seed=42,
            )
        )
        loader = cast(Any, builder.build().train)
        loader.sampler.set_epoch(epoch)
        return list(loader.sampler)

    epoch_one = order(1)
    epoch_two = order(2)

    assert epoch_one != epoch_two
    assert order(2) == epoch_two


def test_generated_wheel_is_discovered_and_activated_by_entry_point(
    tmp_path: Path,
) -> None:
    project = create_project("example-lab", cwd=tmp_path)
    wheel, _ = _build_distribution(
        project,
        tmp_path / "generated-dist",
        include_sdist=False,
    )
    installed = tmp_path / "installed"
    installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    script = """
import sys
from stochaflow.extensions import (
    REGISTRIES,
    activate_extension_plugins,
    prepare_extension_plugins,
)
from stochaflow.utils.config import load_config

config = load_config(sys.argv[1])
assert "example-lab.linear-regression" not in REGISTRIES.models.names()
plan = prepare_extension_plugins(config)
assert [(item.name, item.target) for item in plan.provenance] == [
    ("example-lab", "example_lab.stochaflow_ext")
]
activate_extension_plugins(plan)
assert "example-lab.synthetic-regression" in REGISTRIES.data_builders.names()
assert "example-lab.linear-regression" in REGISTRIES.models.names()
assert "example-lab.regression" in REGISTRIES.training_builders.names()
assert "example-lab.regression-predictions" in REGISTRIES.sampling_builders.names()
assert "example-lab.regression-quality" in REGISTRIES.diagnostics.names()
"""
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    generated_path = str(installed)
    environment["PYTHONPATH"] = (
        generated_path
        if not existing_pythonpath
        else os.pathsep.join((generated_path, existing_pythonpath))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(project / "experiments/example/train.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_wheel_runs_train_resume_and_checkpoint_sampling(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    stochaflow_source = tmp_path / "stochaflow-source"
    (stochaflow_source / "src").mkdir(parents=True)
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository / filename, stochaflow_source / filename)
    shutil.copytree(
        repository / "src/stochaflow",
        stochaflow_source / "src/stochaflow",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    stochaflow_wheel, _ = _build_distribution(
        stochaflow_source,
        tmp_path / "stochaflow-e2e-dist",
        include_sdist=False,
    )
    environment_dir = tmp_path / "environment"
    create_environment = subprocess.run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            str(environment_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert create_environment.returncode == 0, (
        create_environment.stdout + create_environment.stderr
    )
    environment_python = (
        environment_dir / "Scripts/python.exe"
        if os.name == "nt"
        else environment_dir / "bin/python"
    )
    nested_site_result = subprocess.run(
        [
            environment_python,
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert nested_site_result.returncode == 0, (
        nested_site_result.stdout + nested_site_result.stderr
    )
    nested_site = Path(nested_site_result.stdout.strip())
    # A stdlib venv created from inside another venv exposes the base interpreter's
    # system site, not the parent venv that holds our already-installed dependencies.
    # A plain path file reuses those dependencies without exposing the repository
    # source tree through PYTHONPATH; the wheels installed below remain authoritative.
    dependency_paths = tuple(
        path for path in map(Path, site.getsitepackages()) if path != nested_site
    )
    assert dependency_paths
    (nested_site / "stochaflow-test-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in dependency_paths),
        encoding="utf-8",
    )
    install_stochaflow = subprocess.run(
        [
            environment_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--force-reinstall",
            str(stochaflow_wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert install_stochaflow.returncode == 0, (
        install_stochaflow.stdout + install_stochaflow.stderr
    )
    installed_module = subprocess.run(
        [
            environment_python,
            "-c",
            "import stochaflow; print(stochaflow.__file__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert installed_module.returncode == 0, (
        installed_module.stdout + installed_module.stderr
    )
    assert Path(installed_module.stdout.strip()).is_relative_to(environment_dir)

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    stochaflow_executable = (
        environment_dir / "Scripts/stochaflow.exe"
        if os.name == "nt"
        else environment_dir / "bin/stochaflow"
    )

    _run_generated_cli(
        stochaflow_executable,
        tmp_path,
        environment,
        "init",
        "example-lab",
    )
    project = tmp_path / "example-lab"
    assert set(_generated_files(project)) == EXPECTED_FILES
    extension_wheel, _ = _build_distribution(
        project,
        tmp_path / "generated-e2e-dist",
        include_sdist=False,
    )
    install_extension = subprocess.run(
        [
            environment_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--force-reinstall",
            str(extension_wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert install_extension.returncode == 0, (
        install_extension.stdout + install_extension.stderr
    )

    _run_generated_cli(
        stochaflow_executable,
        project,
        environment,
        "train",
        "--config",
        "experiments/example/train.yaml",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    initial_run = _only_directory(project / "outputs/example")
    assert (initial_run / "checkpoints/latest.pt").is_file()
    assert (initial_run / "checkpoints/best.pt").is_file()
    assert (initial_run / "run_manifest.yaml").is_file()
    assert "diagnostics/regression/rmse" in (
        initial_run / "metrics.jsonl"
    ).read_text(encoding="utf-8")
    tensorboard_events = tuple(
        (
            initial_run
            / "tensorboard"
            / "example-lab-example"
        ).glob("events.out.tfevents.*")
    )
    assert len(tensorboard_events) == 1

    resumed_root = project / "outputs/resumed"
    _run_generated_cli(
        stochaflow_executable,
        project,
        environment,
        "train",
        "--resume",
        str(initial_run),
        "--output-dir",
        str(resumed_root),
        "--epochs",
        "3",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    resumed_run = _only_directory(resumed_root)
    assert (resumed_run / "checkpoints/latest.pt").is_file()
    assert (resumed_run / "checkpoints/best.pt").is_file()
    assert (resumed_run / "run_manifest.yaml").is_file()

    # A resumed run owns its inherited best snapshot.  Removing its parent run
    # must not break another strict resume or checkpoint-only sampling.
    shutil.rmtree(initial_run)
    second_resumed_root = project / "outputs/resumed-again"
    _run_generated_cli(
        stochaflow_executable,
        project,
        environment,
        "train",
        "--resume",
        str(resumed_run),
        "--output-dir",
        str(second_resumed_root),
        "--epochs",
        "4",
        "--limit-batches",
        "1",
        "--limit-validation-batches",
        "1",
        "--skip-final-sample",
        "--no-progress",
    )
    second_resumed_run = _only_directory(second_resumed_root)
    assert (second_resumed_run / "checkpoints/latest.pt").is_file()
    assert (second_resumed_run / "checkpoints/best.pt").is_file()
    shutil.rmtree(resumed_run)

    sample_output = project / "sample-output"
    _run_generated_cli(
        stochaflow_executable,
        project,
        environment,
        "sample",
        "--checkpoint",
        str(second_resumed_run),
        "--output-dir",
        str(sample_output),
        "--device",
        "cpu",
    )
    assert (sample_output / "samples.pt").is_file()
    assert (sample_output / "resolved_sampling.yaml").is_file()


def test_runtime_dependencies_exclude_development_and_research_tools() -> None:
    repository = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (repository / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]

    assert "tensorboard>=2.16" in dependencies
    assert not any(
        dependency.startswith(("matplotlib", "ruff", "scipy", "tqdm"))
        for dependency in dependencies
    )
    assert {"pyright>=1.1.400", "pytest>=8.3", "ruff==0.16.0"} <= set(
        extras["dev"]
    )
    assert {"matplotlib>=3.10.9", "scipy>=1.16.1"} <= set(extras["docs"])


@pytest.mark.parametrize(
    "name",
    [
        "MyProject",
        "a_b",
        "a--b",
        "a/b",
        r"a\\b",
        ".",
        "..",
        "-project",
        "project-",
        "class",
        "match",
        "stochaflow",
        "con",
        "lpt9",
        "éclair",
        "a" * 65,
    ],
)
def test_project_name_rejects_noncanonical_and_reserved_values(name: str) -> None:
    with pytest.raises(ProjectScaffoldError):
        validate_project_name(name)


@pytest.mark.parametrize("name", ["a", "research2", "physics-ai-lab"])
def test_project_name_accepts_canonical_ascii_slugs(name: str) -> None:
    assert validate_project_name(name) == name


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure pre-existing directory publication is unavailable",
)
def test_create_project_supports_preexisting_empty_real_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()

    assert create_project("example-lab", cwd=tmp_path) == target
    assert set(_generated_files(target)) == EXPECTED_FILES


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure pre-existing directory publication is unavailable",
)
def test_preexisting_directory_identity_and_mode_are_preserved(
    tmp_path: Path,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    if os.name == "posix":
        target.chmod(0o750)
    before = target.stat(follow_symlinks=False)

    create_project("example-lab", cwd=tmp_path)

    after = target.stat(follow_symlinks=False)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after.st_mode & 0o777 == before.st_mode & 0o777


@pytest.mark.parametrize("entry_name", ["occupied.txt", ".hidden"])
def test_create_project_rejects_nonempty_target_without_writes(
    tmp_path: Path,
    entry_name: str,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    marker = target / entry_name
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ProjectScaffoldError, match="not empty"):
        create_project("example-lab", cwd=tmp_path)

    assert _generated_files(target) == {entry_name: b"keep"}


def test_nonempty_preflight_precedes_platform_capability_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    marker = target / "occupied.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(scaffold, "_supports_anchored_publication", lambda: False)

    with pytest.raises(ProjectScaffoldError, match="not empty"):
        create_project("example-lab", cwd=tmp_path)

    assert _generated_files(target) == {"occupied.txt": b"keep"}


def test_existing_empty_target_requires_secure_platform_primitives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    monkeypatch.setattr(scaffold, "_supports_anchored_publication", lambda: False)

    with pytest.raises(ProjectScaffoldError, match="not supported on this platform"):
        create_project("example-lab", cwd=tmp_path)

    assert list(target.iterdir()) == []


def test_create_project_rejects_file_and_symlink_targets(tmp_path: Path) -> None:
    file_target = tmp_path / "example-lab"
    file_target.write_text("keep", encoding="utf-8")
    with pytest.raises(ProjectScaffoldError, match="empty real directory"):
        create_project("example-lab", cwd=tmp_path)
    assert file_target.read_text(encoding="utf-8") == "keep"

    file_target.unlink()
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    try:
        file_target.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is unavailable")
    with pytest.raises(ProjectScaffoldError, match="empty real directory"):
        create_project("example-lab", cwd=tmp_path)
    assert file_target.is_symlink()
    assert list(destination.iterdir()) == []


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure pre-existing directory publication is unavailable",
)
def test_existing_empty_target_rolls_back_only_created_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    original_write = scaffold._exclusive_write_at
    calls = 0

    def fail_on_second_file(
        parent_descriptor: int,
        name: str,
        content: str,
        *,
        parent: tuple[str, ...],
    ) -> scaffold._AnchoredEntry:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        return original_write(
            parent_descriptor,
            name,
            content,
            parent=parent,
        )

    monkeypatch.setattr(scaffold, "_exclusive_write_at", fail_on_second_file)

    with pytest.raises(ProjectScaffoldError, match="simulated publish failure"):
        create_project("example-lab", cwd=tmp_path)

    assert target.exists()
    assert list(target.iterdir()) == []


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure pre-existing directory publication is unavailable",
)
def test_existing_empty_target_removes_partial_file_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    original_fdopen = os.fdopen

    class FailingTextHandle:
        def __init__(self, handle: TextIO) -> None:
            self._handle = handle

        def __enter__(self) -> FailingTextHandle:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            del exc_type, exc, traceback
            self._handle.close()

        def write(self, content: str) -> int:
            del content
            self._handle.write("partial")
            self._handle.flush()
            raise OSError("simulated write failure")

    def fail_after_partial_write(
        descriptor: int,
        mode: str,
        *,
        encoding: str,
        newline: str,
    ) -> FailingTextHandle:
        handle = original_fdopen(
            descriptor,
            mode,
            encoding=encoding,
            newline=newline,
        )
        return FailingTextHandle(cast(TextIO, handle))

    monkeypatch.setattr(scaffold.os, "fdopen", fail_after_partial_write)

    with pytest.raises(ProjectScaffoldError, match="simulated write failure"):
        create_project("example-lab", cwd=tmp_path)

    assert list(target.iterdir()) == []


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure pre-existing directory publication is unavailable",
)
def test_concurrently_replaced_generated_file_fails_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    original_write = scaffold._exclusive_write_at
    replaced = False

    def replace_first_file(
        parent_descriptor: int,
        name: str,
        content: str,
        *,
        parent: tuple[str, ...],
    ) -> scaffold._AnchoredEntry:
        nonlocal replaced
        entry = original_write(
            parent_descriptor,
            name,
            content,
            parent=parent,
        )
        if not replaced:
            replaced = True
            os.unlink(name, dir_fd=parent_descriptor)
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write("foreign")
        return entry

    monkeypatch.setattr(scaffold, "_exclusive_write_at", replace_first_file)

    with pytest.raises(ProjectScaffoldError, match="target directory changed"):
        create_project("example-lab", cwd=tmp_path)

    assert _generated_files(target) == {"pyproject.toml": b"foreign"}


def test_missing_target_failure_leaves_no_target_or_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write(path: Path, content: str) -> None:
        del path, content
        raise OSError("simulated staging failure")

    monkeypatch.setattr(scaffold, "_exclusive_write_text", fail_write)

    with pytest.raises(ProjectScaffoldError, match="simulated staging failure"):
        create_project("example-lab", cwd=tmp_path)

    assert not (tmp_path / "example-lab").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure recursive staging cleanup is unavailable",
)
def test_mode_probe_failure_removes_unpublished_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_mode_probe(staging: Path) -> int:
        residue = staging / "residue/nested"
        residue.mkdir(parents=True)
        (residue / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("simulated mode probe failure")

    monkeypatch.setattr(scaffold, "_ordinary_directory_mode", fail_mode_probe)

    with pytest.raises(ProjectScaffoldError, match="simulated mode probe failure"):
        create_project("example-lab", cwd=tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure recursive staging cleanup is unavailable",
)
def test_mode_probe_cleanup_does_not_traverse_reused_staging_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement: Path | None = None
    original_staging: Path | None = None

    def replace_staging_before_failure(staging: Path) -> int:
        nonlocal original_staging, replacement
        original_staging = staging.with_name(f"{staging.name}-original")
        staging.rename(original_staging)
        staging.mkdir()
        replacement = staging
        (replacement / "foreign.txt").write_text("foreign", encoding="utf-8")
        raise OSError("simulated mode probe failure")

    monkeypatch.setattr(
        scaffold,
        "_ordinary_directory_mode",
        replace_staging_before_failure,
    )

    with pytest.raises(ProjectScaffoldError, match="simulated mode probe failure"):
        create_project("example-lab", cwd=tmp_path)

    assert original_staging is not None and original_staging.is_dir()
    assert replacement is not None
    assert (replacement / "foreign.txt").read_text(encoding="utf-8") == "foreign"


@pytest.mark.skipif(
    not scaffold._supports_anchored_publication(),
    reason="secure pre-existing directory publication is unavailable",
)
def test_existing_target_swap_to_symlink_never_redirects_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    target.mkdir()
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    original_open = scaffold._open_existing_directory
    swapped = False

    def swap_before_open(
        path: Path,
        expected: scaffold._DirectoryIdentity,
    ) -> int:
        nonlocal swapped
        if path == target and not swapped:
            swapped = True
            target.rmdir()
            try:
                target.symlink_to(redirect, target_is_directory=True)
            except OSError:
                pytest.skip("creating directory symlinks is unavailable")
        return original_open(path, expected)

    monkeypatch.setattr(scaffold, "_open_existing_directory", swap_before_open)

    with pytest.raises(ProjectScaffoldError, match="target changed"):
        create_project("example-lab", cwd=tmp_path)

    assert target.is_symlink()
    assert list(redirect.iterdir()) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "example-lab",
        "redirect",
    ]


def test_absent_target_publish_does_not_replace_concurrent_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "example-lab"
    original_rename = scaffold._rename_no_replace
    injected = False

    def create_target_before_publish(source: Path, destination: Path) -> None:
        nonlocal injected
        if destination == target and not injected:
            injected = True
            target.mkdir()
        original_rename(source, destination)

    monkeypatch.setattr(
        scaffold,
        "_rename_no_replace",
        create_target_before_publish,
    )

    with pytest.raises(ProjectScaffoldError, match="target appeared"):
        create_project("example-lab", cwd=tmp_path)

    assert target.is_dir()
    assert list(target.iterdir()) == []
    assert list(tmp_path.iterdir()) == [target]


def test_successful_publish_does_not_touch_reused_staging_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rename = scaffold._rename_no_replace
    replacement: Path | None = None

    def reuse_source_after_publish(source: Path, destination: Path) -> None:
        nonlocal replacement
        original_rename(source, destination)
        source.mkdir()
        replacement = source
        (source / "foreign.txt").write_text("foreign", encoding="utf-8")

    monkeypatch.setattr(
        scaffold,
        "_rename_no_replace",
        reuse_source_after_publish,
    )

    target = create_project("example-lab", cwd=tmp_path)

    assert set(_generated_files(target)) == EXPECTED_FILES
    assert replacement is not None
    assert (replacement / "foreign.txt").read_text(encoding="utf-8") == "foreign"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission mode contract")
def test_absent_target_uses_normal_mkdir_umask_mode(tmp_path: Path) -> None:
    probe = tmp_path / "probe"
    probe.mkdir()
    expected_mode = probe.stat().st_mode & 0o777
    probe.rmdir()

    target = create_project("example-lab", cwd=tmp_path)

    assert target.stat().st_mode & 0o777 == expected_mode


def test_template_manifest_matches_packaged_resources_and_package_data() -> None:
    resource_root = resources.files("stochaflow.projects").joinpath("templates")
    actual_resources = {
        entry.name
        for entry in resource_root.iterdir()
        if entry.name.endswith(".tmpl")
    }
    declared_resources = {
        template.resource_name for template in scaffold._TEMPLATE_MANIFEST
    }
    assert actual_resources == declared_resources
    assert all(not name.startswith(".") for name in actual_resources)

    root_project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert root_project["tool"]["setuptools"]["package-data"] == {
        "stochaflow.projects": ["templates/*.tmpl"]
    }


def test_wheel_and_sdist_include_every_template_resource(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src/stochaflow").mkdir(parents=True)
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(filename, source / filename)
    shutil.copy2("src/stochaflow/__init__.py", source / "src/stochaflow/__init__.py")
    shutil.copytree(
        "src/stochaflow/projects",
        source / "src/stochaflow/projects",
    )
    wheel, sdist = _build_distribution(
        source,
        tmp_path / "dist",
        include_sdist=True,
    )
    expected = {
        template.resource_name for template in scaffold._TEMPLATE_MANIFEST
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_templates = {
            Path(name).name
            for name in archive.namelist()
            if "/projects/templates/" in name
        }
    assert sdist is not None
    with tarfile.open(sdist) as archive:
        sdist_templates = {
            Path(name).name
            for name in archive.getnames()
            if "/projects/templates/" in name
        }
    assert wheel_templates == expected
    assert sdist_templates == expected


def test_init_cli_and_reference_expose_required_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(["init", "example-lab"])
    assert args.command == "init"
    assert args.name == "example-lab"

    monkeypatch.chdir(tmp_path)
    main(["init", "example-lab"])
    assert Path.cwd() == tmp_path
    assert (tmp_path / "example-lab/pyproject.toml").is_file()
    assert f"Created project: {tmp_path / 'example-lab'}" in capsys.readouterr().out


def test_init_cli_reports_invalid_name_without_creating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        main(["init", "Invalid"])
    assert exc_info.value.code == 2
    assert list(tmp_path.iterdir()) == []
