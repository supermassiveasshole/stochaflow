"""Documentation configuration contract tests."""

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"


def _load_documentation_configuration() -> dict[str, object]:
    original_sys_path = sys.path.copy()
    try:
        return runpy.run_path(str(DOCS_ROOT / "conf.py"))
    finally:
        sys.path[:] = original_sys_path


def test_mathjax_loads_mathtools_for_documented_relation_macros() -> None:
    """Keep optional TeX commands renderable in every generated HTML page."""

    configuration = _load_documentation_configuration()
    mathjax_config = configuration["mathjax4_config"]
    assert isinstance(mathjax_config, dict)
    assert mathjax_config["loader"]["load"] == ["[tex]/mathtools"]
    assert mathjax_config["tex"]["packages"]["[+]"] == ["mathtools"]

    sources = [
        path
        for path in DOCS_ROOT.rglob("*.md")
        if path.relative_to(DOCS_ROOT).parts[:2]
        != ("development", "notes")
        and "\\coloneqq" in path.read_text(encoding="utf-8")
    ]
    assert sources


def test_sphinx_publishes_roadmap_and_reader_facing_development_plans() -> None:
    """Expose maintained plans without publishing research archives as guides."""

    configuration = _load_documentation_configuration()
    exclude_patterns = configuration["exclude_patterns"]
    assert isinstance(exclude_patterns, list)
    assert "development/**" not in exclude_patterns
    assert "development/notes/**" in exclude_patterns
    review_candidates = configuration["DEVELOPMENT_REVIEW_CANDIDATES"]
    assert isinstance(review_candidates, tuple)
    assert set(review_candidates) == {
        "development/afhq-v2-checkpoint-cleanup-20260806.md",
        "development/afhq-v2-learned-range-v-closeout.md",
        "development/legacy-intel-macos-pytorch-test-lifecycle.md",
        "development/p2-experiment-closeout.md",
    }
    assert set(review_candidates) <= set(exclude_patterns)

    rewrite_links = cast(
        Callable[..., str],
        configuration["_rewrite_repository_markdown_links"],
    )
    development_index = DOCS_ROOT / "development" / "README.md"
    rewritten_index = rewrite_links(
        "[roadmap](../../ROADMAP.md) "
        "[plan](development-priority-roadmap.md) "
        "[notes](notes/history/milestone-id-map.md)",
        source_path=development_index,
        published_path=development_index,
    )
    assert "[roadmap](../roadmap.md)" in rewritten_index
    assert "[plan](development-priority-roadmap.md)" in rewritten_index
    assert (
        "[notes](https://github.com/supermassiveasshole/stochaflow/"
        "blob/main/docs/development/notes/history/milestone-id-map.md)"
        in rewritten_index
    )

    root_roadmap = PROJECT_ROOT / "ROADMAP.md"
    rewritten_roadmap = rewrite_links(
        root_roadmap.read_text(encoding="utf-8"),
        source_path=root_roadmap,
        published_path=DOCS_ROOT / "roadmap.md",
    )
    assert "(development/development-priority-roadmap.md)" in rewritten_roadmap
    assert (
        "https://github.com/supermassiveasshole/stochaflow/blob/main/SPEC.md"
        in rewritten_roadmap
    )

    code_examples = (
        "```md\n[roadmap](../../ROADMAP.md)\n```\n\n"
        "`[roadmap](../../ROADMAP.md)`\n\n"
        "[roadmap](../../ROADMAP.md)"
    )
    rewritten_examples = rewrite_links(
        code_examples,
        source_path=development_index,
        published_path=development_index,
    )
    assert "```md\n[roadmap](../../ROADMAP.md)\n```" in rewritten_examples
    assert "`[roadmap](../../ROADMAP.md)`" in rewritten_examples
    assert rewritten_examples.endswith("[roadmap](../roadmap.md)")

    configure_page = cast(
        Callable[..., None],
        configuration["_use_canonical_roadmap_source_links"],
    )
    roadmap_context: dict[str, object] = {}
    configure_page(None, "roadmap", "page.html", roadmap_context, None)
    assert roadmap_context["theme_source_view_link"] == (
        "https://github.com/supermassiveasshole/stochaflow/"
        "blob/main/ROADMAP.md?plain=true"
    )
    assert roadmap_context["theme_source_edit_link"] == (
        "https://github.com/supermassiveasshole/stochaflow/"
        "edit/main/ROADMAP.md"
    )
    ordinary_context: dict[str, object] = {}
    configure_page(None, "index", "page.html", ordinary_context, None)
    assert ordinary_context == {}


def test_furo_landing_page_assets_are_declared_and_present() -> None:
    """Keep the restrained Furo customization self-contained."""

    configuration = _load_documentation_configuration()

    assert configuration["html_theme"] == "furo"
    assert configuration["html_static_path"] == [
        "_static",
        "../assets/readme",
    ]
    assert configuration["html_css_files"] == ["custom.css"]
    assert (DOCS_ROOT / "_static" / "custom.css").is_file()
    static_paths = configuration["html_static_path"]
    assert isinstance(static_paths, list)
    static_roots = {
        (DOCS_ROOT / path).resolve()
        for path in static_paths
        if isinstance(path, str)
    }
    assert PROJECT_ROOT / "assets" / "readme" in static_roots
