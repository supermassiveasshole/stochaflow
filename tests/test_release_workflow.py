"""GitHub Release package delivery contract tests."""

import re
import tomllib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_ROOT = PROJECT_ROOT / ".github" / "workflows"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
TEST_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"
PROJECT_METADATA_PATH = PROJECT_ROOT / "pyproject.toml"
PUBLIC_TUTORIAL_PATHS = (
    PROJECT_ROOT / "docs" / "tutorials" / "custom-generation-family.md",
    PROJECT_ROOT / "docs" / "tutorials" / "reuse-gaussian-components.md",
)
REFERENCE_PROJECT_METADATA_PATHS = (
    PROJECT_ROOT
    / "examples"
    / "extension-projects"
    / "knowledge-distillation"
    / "pyproject.toml",
    PROJECT_ROOT
    / "examples"
    / "extension-projects"
    / "physics-reconstruction"
    / "pyproject.toml",
    PROJECT_ROOT / "examples" / "showcases" / "afhq-v2" / "pyproject.toml",
)
AFHQ_METADATA_PATH = REFERENCE_PROJECT_METADATA_PATHS[-1]
ACTION_REFERENCE_PATTERN = re.compile(
    r"^\s*uses:\s+(?P<reference>\S+?@\S+)(?:\s+#.*)?$",
    flags=re.MULTILINE,
)
RELEASE_WHEEL_URL_PATTERN = re.compile(
    r"https://github\.com/supermassiveasshole/stochaflow/releases/"
    r"download/v(?P<tag_version>[^/]+)/"
    r"stochaflow-(?P<wheel_version>[^/]+)-py3-none-any\.whl"
)
TAGGED_EXAMPLE_URL_PATTERN = re.compile(
    r"https://raw\.githubusercontent\.com/supermassiveasshole/stochaflow/"
    r"v(?P<tag_version>[^/]+)/examples/"
)
PYPI_STYLE_REQUIREMENT_PATTERN = re.compile(
    r"stochaflow(?:\[quality\])?\s*==\s*[^\s,\"']+"
)


def _load_workflow() -> dict[str, object]:
    workflow = yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(workflow, dict)
    return workflow


def test_release_workflow_is_tag_bound_and_can_publish_attestations() -> None:
    """Keep release writes restricted to explicit version tags."""

    workflow = _load_workflow()

    trigger = workflow["on"]
    assert isinstance(trigger, dict)
    push = trigger["push"]
    assert isinstance(push, dict)
    assert push["tags"] == ["v*"]
    assert workflow["permissions"] == {"contents": "read"}
    concurrency = workflow["concurrency"]
    assert isinstance(concurrency, dict)
    assert concurrency["cancel-in-progress"] == "true"

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    publish = jobs["publish"]
    assert isinstance(publish, dict)
    assert publish["needs"] == "build"
    assert publish["permissions"] == {
        "contents": "write",
        "id-token": "write",
        "attestations": "write",
    }


def test_release_workflow_builds_verifies_and_publishes_both_archives() -> None:
    """Keep GitHub Release assets installable and independently verifiable."""

    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    for contract in (
        "uv sync --frozen --extra dev",
        "uv run --frozen pytest",
        "uv build",
        'git merge-base --is-ancestor "${GITHUB_SHA}" "origin/main"',
        'uv venv "${wheel_environment}" --python 3.12',
        'uv venv "${sdist_environment}" --python 3.12',
        "stochaflow-${package_version}-py3-none-any.whl",
        "stochaflow-${package_version}.tar.gz",
        "sha256sum --check SHA256SUMS",
        "digest-mismatch: error",
        "release-dists-${GITHUB_REF_NAME}-${GITHUB_RUN_ATTEMPT}",
        "name: ${{ needs.build.outputs.artifact_name }}",
        'gh api "repos/${GH_REPO}/commits/${GITHUB_REF_NAME}" --jq .sha',
        '"${tag_commit}" != "${GITHUB_SHA}"',
        "subject-path: dist/*",
        "GH_REPO: ${{ github.repository }}",
        'gh release create "${GITHUB_REF_NAME}"',
        "--draft",
        'gh release upload "${GITHUB_REF_NAME}"',
        "--clobber",
        'gh release download "${GITHUB_REF_NAME}"',
        "cmp --silent",
        "gh attestation verify",
        '--signer-workflow "${GH_REPO}/.github/workflows/release.yml"',
        '--source-digest "${GITHUB_SHA}"',
        '--source-ref "${GITHUB_REF}"',
        'gh release edit "${GITHUB_REF_NAME}"',
        "--draft=false",
        "--verify-tag",
    ):
        assert contract in workflow_text


def test_workflow_actions_are_pinned_to_immutable_commits() -> None:
    """Keep every third-party workflow dependency bound to reviewed bytes."""

    for workflow_path in sorted(WORKFLOWS_ROOT.glob("*.yml")):
        workflow_text = workflow_path.read_text(encoding="utf-8")
        references = ACTION_REFERENCE_PATTERN.findall(workflow_text)
        assert references, workflow_path
        for reference in references:
            revision = reference.rsplit("@", maxsplit=1)[1]
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                workflow_path,
                reference,
            )


def test_workflows_pin_the_uv_runtime_version() -> None:
    """Keep a tagged source reproducible across later workflow reruns."""

    for workflow_path in sorted(WORKFLOWS_ROOT.glob("*.yml")):
        workflow = yaml.load(
            workflow_path.read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        assert isinstance(workflow, dict)
        jobs = workflow["jobs"]
        assert isinstance(jobs, dict)
        setup_steps = [
            step
            for job in jobs.values()
            if isinstance(job, dict)
            for step in job.get("steps", [])
            if isinstance(step, dict)
            and str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
        ]
        assert setup_steps, workflow_path
        for step in setup_steps:
            inputs = step["with"]
            assert isinstance(inputs, dict)
            assert inputs["version"] == "0.11.26"


def test_supported_python_314_lanes_pin_the_stable_patch_baseline() -> None:
    """Avoid the incremental-GC Python 3.14 patch releases in training CI."""

    workflow = yaml.load(
        TEST_WORKFLOW_PATH.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(workflow, dict)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["test"]
    assert isinstance(test_job, dict)
    strategy = test_job["strategy"]
    assert isinstance(strategy, dict)
    matrix = strategy["matrix"]
    assert isinstance(matrix, dict)
    entries = matrix["include"]
    assert isinstance(entries, list)

    python_314_lanes = {
        (entry["os"], entry["python-version"])
        for entry in entries
        if isinstance(entry, dict)
        and str(entry.get("python-version", "")).startswith("3.14")
    }
    assert python_314_lanes == {
        ("ubuntu-latest", "3.14.6"),
        ("windows-latest", "3.14.6"),
        ("macos-latest", "3.14.6"),
    }


def test_public_release_references_match_project_version() -> None:
    """Keep public install references synchronized with package metadata."""

    metadata = tomllib.loads(PROJECT_METADATA_PATH.read_text(encoding="utf-8"))
    assert isinstance(metadata, dict)
    version = metadata["project"]["version"]
    wheel_url = (
        "https://github.com/supermassiveasshole/stochaflow/releases/"
        f"download/v{version}/stochaflow-{version}-py3-none-any.whl"
    )

    public_paths = (
        PROJECT_ROOT / "README.md",
        *(
            path
            for path in (PROJECT_ROOT / "docs").rglob("*.md")
            if "development" not in path.relative_to(PROJECT_ROOT / "docs").parts
        ),
        *REFERENCE_PROJECT_METADATA_PATHS,
    )
    referenced_paths: set[Path] = set()
    tagged_example_paths: set[Path] = set()
    for path in public_paths:
        text = path.read_text(encoding="utf-8")
        matches = tuple(RELEASE_WHEEL_URL_PATTERN.finditer(text))
        if matches:
            referenced_paths.add(path)
        for match in matches:
            assert match["tag_version"] == version, path
            assert match["wheel_version"] == version, path
        example_matches = tuple(TAGGED_EXAMPLE_URL_PATTERN.finditer(text))
        if example_matches:
            tagged_example_paths.add(path)
        for match in example_matches:
            assert match["tag_version"] == version, path
        assert PYPI_STYLE_REQUIREMENT_PATTERN.search(text) is None, path

    assert PROJECT_ROOT / "README.md" in referenced_paths
    assert PROJECT_ROOT / "docs" / "configuration" / "index.md" in referenced_paths
    assert set(PUBLIC_TUTORIAL_PATHS).issubset(referenced_paths)
    assert set(REFERENCE_PROJECT_METADATA_PATHS).issubset(referenced_paths)
    assert PROJECT_ROOT / "README.md" in tagged_example_paths
    assert PROJECT_ROOT / "docs" / "index.md" in tagged_example_paths
    assert (
        PROJECT_ROOT / "docs" / "configuration" / "index.md"
        in tagged_example_paths
    )
    assert wheel_url in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


def test_reference_projects_use_installable_release_requirements() -> None:
    """Keep example metadata usable without an unpublished PyPI package."""

    root_metadata = tomllib.loads(
        PROJECT_METADATA_PATH.read_text(encoding="utf-8")
    )
    version = root_metadata["project"]["version"]
    wheel_url = (
        "https://github.com/supermassiveasshole/stochaflow/releases/"
        f"download/v{version}/stochaflow-{version}-py3-none-any.whl"
    )
    core_requirement = f"stochaflow @ {wheel_url}"

    for path in REFERENCE_PROJECT_METADATA_PATHS:
        metadata = tomllib.loads(path.read_text(encoding="utf-8"))
        assert core_requirement in metadata["project"]["dependencies"], path

    afhq_metadata = tomllib.loads(
        AFHQ_METADATA_PATH.read_text(encoding="utf-8")
    )
    quality_requirement = f"stochaflow[quality] @ {wheel_url}"
    assert quality_requirement in (
        afhq_metadata["project"]["optional-dependencies"]["quality"]
    )
    assert afhq_metadata["tool"]["uv"]["sources"]["stochaflow"] == {
        "path": "../../..",
        "editable": True,
    }
