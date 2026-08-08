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
AFHQ_TUTORIAL = DOCS_ROOT / "tutorials" / "afhq-v2.md"


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
    mnist_asset = "mnist_ddim50_epoch_0183_samples.png"
    afhq_asset = (
        "afhq_v2_adm_learned_range_v_best_ddpm100_cfg2_samples.png"
    )
    afhq_card = "AFHQ-v2 · class-conditional ADM"
    legacy_afhq_asset = "afhq_v2_adm_ddim50_epoch_0170_samples.png"

    assert homepage.index(mnist_asset) < homepage.index(afhq_card)
    assert f"_static/{mnist_asset}" in homepage
    assert (PROJECT_ROOT / "assets" / "readme" / mnist_asset).is_file()
    assert f"_static/{afhq_asset}" in homepage
    assert (PROJECT_ROOT / "assets" / "readme" / afhq_asset).is_file()
    assert legacy_afhq_asset not in homepage


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
    assert "Intel" in policy
    assert "不受支持" in policy
    assert "Deprecated / best effort" not in policy


def test_maintained_examples_publish_grounded_results() -> None:
    """Keep the example landing pages tied to existing result evidence."""

    mnist = MNIST_README.read_text(encoding="utf-8")
    afhq = AFHQ_README.read_text(encoding="utf-8")
    afhq_tutorial = AFHQ_TUTORIAL.read_text(encoding="utf-8")
    normalized_afhq = " ".join(afhq.split())

    assert "Best validation loss | **0.07189**" in mnist
    assert "mnist_ddpm_epoch_0183_samples.png" in mnist
    assert "mnist_ddim50_epoch_0183_samples.png" in mnist
    assert "mnist_ddpm_epoch_0183_trajectory.gif" in mnist
    assert "mnist_ddim50_epoch_0183_trajectory.gif" in mnist
    assert "exact parameter count" in normalized_afhq
    assert "105,197,187" in normalized_afhq
    assert "100,351,366" in normalized_afhq
    assert "45.17 images/s" in normalized_afhq
    assert "10.455 GiB" in normalized_afhq
    assert "Validation-selected E190" in normalized_afhq
    assert "train-adm-128-learned-range-v.yaml" in normalized_afhq
    assert (
        "formal-ddpm100-cfg2-official-test-learned-range-v.yaml"
        in normalized_afhq
    )
    assert "aggregate FID" in normalized_afhq
    assert "`best.pt`" in normalized_afhq
    assert "Validation-selected E190" in afhq
    assert "**25.7572**" in afhq
    assert "**20.2478**" in afhq
    assert (
        "afhq_v2_adm_learned_range_v_best_ddpm100_cfg2_samples.png"
        in afhq
    )
    assert "EvaluationBuilder" in normalized_afhq
    assert "Metrics" in normalized_afhq
    assert "Aggregate FID | **30.240**" not in afhq
    assert "Aggregate KID | **0.005310 ± 0.000701**" not in afhq
    assert "afhq_v2_adm_ddim50_epoch_0170_samples.png" not in afhq
    assert "**25.7572**" in afhq_tutorial
    assert "**20.2478**" in afhq_tutorial
    assert "subset size 200" in afhq_tutorial
    assert "subset size 300" not in afhq_tutorial

    for source in (MNIST_README, AFHQ_README):
        content = source.read_text(encoding="utf-8")
        local_images = [
            path
            for path in re.findall(r'<img src="([^"]+)"', content)
            if "://" not in path
        ]
        assert all((source.parent / path).resolve().is_file() for path in local_images)


def test_public_docs_do_not_advertise_retired_p2_support() -> None:
    """Keep the retired experiment out of the supported user surface."""

    published_docs = (
        path
        for path in DOCS_ROOT.rglob("*.md")
        if "development" not in path.relative_to(DOCS_ROOT).parts
    )
    example_surfaces = (
        path
        for pattern in ("*.md", "*.yaml", "*.yml")
        for path in (PROJECT_ROOT / "examples").rglob(pattern)
    )
    public_paths = sorted(
        {
            PROJECT_ROOT / "README.md",
            DOCS_ROOT / "configuration" / "_reference.yaml",
            *published_docs,
            *example_surfaces,
        }
    )
    retired_symbols = (
        "P2GaussianDenoisingTrainingBuilder",
        "P2GaussianDenoisingTrainingStrategy",
        "ClassConditionalP2GaussianDenoisingTrainingBuilder",
        "ClassConditionalP2GaussianDenoisingTrainingStrategy",
    )

    for path in public_paths:
        content = path.read_text(encoding="utf-8")
        assert (
            re.search(r"(?<![A-Za-z0-9])P2(?![A-Za-z0-9])", content, re.IGNORECASE)
            is None
        ), path
        assert all(symbol not in content for symbol in retired_symbols), path
