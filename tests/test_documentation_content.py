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
