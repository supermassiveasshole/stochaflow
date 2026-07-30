"""Published documentation content contract tests."""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
MNIST_README = (
    PROJECT_ROOT
    / "examples"
    / "built-in"
    / "image-generation"
    / "README.md"
)
AFHQ_README = (
    PROJECT_ROOT / "examples" / "showcases" / "afhq-v2" / "README.md"
)


def test_homepage_quick_start_installs_the_release_wheel() -> None:
    """Keep the primary user path independent of a source checkout."""

    homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    quick_start = homepage.split("## 五分钟快速开始", 1)[1].split(
        "\n## ", 1
    )[0]
    bash_blocks = re.findall(r"```bash\n(.*?)\n```", quick_start, re.DOTALL)
    expected_fragments = (
        "python -m venv .venv",
        "releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl",
        "stochaflow init my-research-project",
        "python -m pip install -e .",
        "stochaflow train --config experiments/example/train.yaml",
    )
    mnist_fragments = (
        "raw.githubusercontent.com/supermassiveasshole/stochaflow/v0.1.0/",
        "--config mnist.yaml",
        "--epochs 1",
        "--limit-batches 10",
        "--limit-validation-batches 2",
        "--limit-test-batches 2",
    )

    assert all("uv sync" not in block for block in bash_blocks)
    assert any(
        all(fragment in block for fragment in expected_fragments)
        for block in bash_blocks
    )
    assert any(
        all(fragment in block for fragment in mnist_fragments)
        for block in bash_blocks
    )
    assert "my-research-project/outputs/example/<run>/" in quick_start
    assert (
        'href="https://github.com/supermassiveasshole/stochaflow"'
        in quick_start
    )
    assert "点 Star" in quick_start
    stylesheet = (DOCS_ROOT / "_static" / "custom.css").read_text(
        encoding="utf-8"
    )
    assert ".sf-star-link a" in stylesheet


def test_readme_quick_start_prefers_the_release_wheel() -> None:
    """Keep source synchronization as an explicit contributor alternative."""

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = readme.split("## Quick start", 1)[1].split(
        "\n## ", 1
    )[0]
    release_section, source_section = quick_start.split(
        "### Run the built-in MNIST example from source", 1
    )

    assert (
        "releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl"
        in release_section
    )
    assert "python -m pip install" in release_section
    assert "stochaflow init my-research-project" in release_section
    assert "python -m pip install -e ." in release_section
    assert (
        "raw.githubusercontent.com/supermassiveasshole/stochaflow/v0.1.0/"
        in release_section
    )
    assert "--limit-test-batches 2" in release_section
    assert (
        "https://github.com/supermassiveasshole/stochaflow"
        in release_section
    )
    assert "star the project" in release_section
    assert "uv sync" not in release_section
    assert "uv sync --extra dev" in source_section


def test_homepage_presents_mnist_before_afhq() -> None:
    """Keep the minimal built-in workflow ahead of the larger showcase."""

    homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    asset_names = (
        "mnist_ddim50_epoch_0183_samples.png",
        "afhq_v2_adm_ddim50_epoch_0170_samples.png",
    )

    assert homepage.index(asset_names[0]) < homepage.index(asset_names[1])
    for filename in asset_names:
        assert f"_static/{filename}" in homepage
        assert (PROJECT_ROOT / "assets" / "readme" / filename).is_file()


def test_homepage_cards_override_furo_container_padding_reset() -> None:
    """Keep custom content containers from rendering flush against their borders."""

    stylesheet = (DOCS_ROOT / "_static" / "custom.css").read_text(
        encoding="utf-8"
    )

    for selector in (
        '[role="main"] .sf-hero',
        '[role="main"] .sf-card',
        '[role="main"] .sf-result-card',
    ):
        assert selector in stylesheet


def test_wide_docs_layout_widens_content_and_preserves_centering() -> None:
    """Keep the wide article width paired with Furo's drawer calculation."""

    stylesheet = (DOCS_ROOT / "_static" / "custom.css").read_text(
        encoding="utf-8"
    )
    wide_layout_contract = """@media (min-width: 108em) {
  .content {
    width: 56em;
  }

  .sidebar-drawer {
    width: calc(50% - 31em);
  }
}"""

    assert wide_layout_contract in stylesheet


def test_platform_policy_documents_the_python_314_patch_baseline() -> None:
    """Keep the supported matrix aligned with the pinned training CI lanes."""

    policy = (DOCS_ROOT / "platform-support.md").read_text(encoding="utf-8")
    supported_rows = [
        line for line in policy.splitlines() if "| Supported |" in line
    ]
    fullwidth_comma = "\N{FULLWIDTH COMMA}"

    assert supported_rows == [
        (
            f"| Linux x86_64 | Supported | Ubuntu CI{fullwidth_comma}"
            "CPython 3.12 和 3.14.6 |"
        ),
        (
            f"| Windows x86_64 | Supported | Windows CI{fullwidth_comma}"
            "CPython 3.14.6 |"
        ),
        (
            f"| macOS arm64 | Supported | macOS CI{fullwidth_comma}"
            "CPython 3.14.6 |"
        ),
    ]
    assert "Python 3.14 用户应以 **3.14.6 或更新的兼容 patch release**" in policy
    assert "3.14.0" in policy
    assert "3.14.4" in policy
    assert "从 3.14.5 起恢复" in policy


def test_maintained_examples_publish_grounded_results() -> None:
    """Keep the example landing pages tied to existing result evidence."""

    mnist = MNIST_README.read_text(encoding="utf-8")
    afhq = AFHQ_README.read_text(encoding="utf-8")

    assert "Best validation loss | **0.07189**" in mnist
    assert "mnist_ddpm_epoch_0183_samples.png" in mnist
    assert "mnist_ddim50_epoch_0183_samples.png" in mnist
    assert "mnist_ddpm_epoch_0183_trajectory.gif" in mnist
    assert "mnist_ddim50_epoch_0183_trajectory.gif" in mnist
    assert "Aggregate FID | **30.240**" in afhq
    assert "Aggregate KID | **0.005310 ± 0.000701**" in afhq
    assert "afhq_v2_adm_ddim50_epoch_0170_samples.png" in afhq

    for source in (MNIST_README, AFHQ_README):
        content = source.read_text(encoding="utf-8")
        local_images = [
            path
            for path in re.findall(r'<img src="([^"]+)"', content)
            if "://" not in path
        ]
        assert all((source.parent / path).resolve().is_file() for path in local_images)
