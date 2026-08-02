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
    afhq_card = "AFHQ-v2 · class-conditional ADM"
    legacy_afhq_asset = "afhq_v2_adm_ddim50_epoch_0170_samples.png"

    assert homepage.index(mnist_asset) < homepage.index(afhq_card)
    assert f"_static/{mnist_asset}" in homepage
    assert (PROJECT_ROOT / "assets" / "readme" / mnist_asset).is_file()
    assert legacy_afhq_asset not in homepage


def test_afhq_p2_evaluation_documents_source_checkout_install_contract() -> None:
    """Keep unreleased AFHQ Evaluation on the current-checkout install path."""

    required_fragments = (
        "正式 P2 Evaluation 当前必须从本仓库 checkout",
        (
            "uv sync --project examples/showcases/afhq-v2 --locked "
            "--extra quality"
        ),
        "`--no-deps --offline`",
        "只验证 wheel 内容、隔离后的 extension entry point 与当前 core wheel",
        "不验证 AFHQ wheel 的 released-core resolver",
        "core/AFHQ 0.2 release",
        "post-release、non-blocking 的 follow-up",
        "不是当前 source-checkout P2 readiness 的 merge blocker",
    )

    for path in (AFHQ_README, AFHQ_TUTORIAL):
        content = path.read_text(encoding="utf-8")
        normalized = " ".join(content.split())
        assert all(fragment in normalized for fragment in required_fragments), path
        assert "releases/download/v0.2" not in content


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
    normalized_afhq = " ".join(afhq.split())

    assert "Best validation loss | **0.07189**" in mnist
    assert "mnist_ddpm_epoch_0183_samples.png" in mnist
    assert "mnist_ddim50_epoch_0183_samples.png" in mnist
    assert "mnist_ddpm_epoch_0183_trajectory.gif" in mnist
    assert "mnist_ddim50_epoch_0183_trajectory.gif" in mnist
    assert "exact parameter count 是 105,197,187" in normalized_afhq
    assert "已从 current result surface 移除" in normalized_afhq
    assert (
        "本 README 不发布 corrected ADM 的 production long-run quality baseline"
        in normalized_afhq
    )
    assert (
        "单 epoch 受控 A/B 数值只属于 pipeline/protocol readiness evidence"
        in normalized_afhq
    )
    assert "Aggregate FID | **30.240**" not in afhq
    assert "Aggregate KID | **0.005310 ± 0.000701**" not in afhq
    assert "afhq_v2_adm_ddim50_epoch_0170_samples.png" not in afhq

    for source in (MNIST_README, AFHQ_README):
        content = source.read_text(encoding="utf-8")
        local_images = [
            path
            for path in re.findall(r'<img src="([^"]+)"', content)
            if "://" not in path
        ]
        assert all((source.parent / path).resolve().is_file() for path in local_images)
