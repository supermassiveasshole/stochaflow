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


def test_homepage_has_a_copyable_bounded_quick_start() -> None:
    """Keep the published first run bounded across every data phase."""

    homepage = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    bash_blocks = re.findall(r"```bash\n(.*?)\n```", homepage, re.DOTALL)
    expected_fragments = (
        "uv run stochaflow train",
        "examples/built-in/image-generation/configs/train/mnist.yaml",
        "--epochs 1",
        "--limit-batches 10",
        "--limit-validation-batches 2",
        "--limit-test-batches 2",
    )

    assert "uv sync --extra dev" in "\n".join(bash_blocks)
    assert any(
        all(fragment in block for fragment in expected_fragments)
        for block in bash_blocks
    )
    assert "outputs/mnist/<run>/" in homepage


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
