"""Sphinx configuration for the Stochaflow documentation site."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = Path(__file__).resolve().parent
ROOT_ROADMAP = PROJECT_ROOT / "ROADMAP.md"
DEVELOPMENT_NOTES_ROOT = DOCS_ROOT / "development" / "notes"
REPOSITORY_BLOB_BASE = (
    "https://github.com/supermassiveasshole/stochaflow/blob/main/"
)
MARKDOWN_LINK_PATTERN = re.compile(
    r"(?P<prefix>!?\[[^\]\n]*\]\(\s*)"
    r"(?P<target><[^>\n]+>|[^)\s]+)"
    r"(?P<suffix>(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\))"
)
MARKDOWN_CODE_PATTERN = re.compile(
    r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n"
    r".*?^[ ]{0,3}(?P=fence)[ \t]*(?:\n|$)"
    r"|(?P<inline>`+)(?!`)[^\r\n]*?(?P=inline)(?!`)",
    re.MULTILINE | re.DOTALL,
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "Stochaflow"
author = "Stochaflow contributors"
copyright = "2026, Stochaflow contributors"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
language = "zh_CN"
html_search_language = "zh"
exclude_patterns = [
    "_build",
    "development/notes/**",
    "Thumbs.db",
    ".DS_Store",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 4

# Research notes use relation symbols provided by the TeX mathtools package,
# including ``\coloneqq``. Sphinx's default MathJax bundle does not activate
# optional TeX packages, so load and register mathtools for every page.
mathjax4_config = {
    "loader": {"load": ["[tex]/mathtools"]},
    "tex": {"packages": {"[+]": ["mathtools"]}},
}

nitpicky = True
autosummary_generate = True
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}

html_theme = "furo"
html_title = "Stochaflow 文档"
html_baseurl = "https://supermassiveasshole.github.io/stochaflow/"
html_static_path = ["_static", "../assets/readme"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/supermassiveasshole/stochaflow/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "#4338ca",
        "color-brand-content": "#4338ca",
        "color-api-name": "#0f766e",
        "font-stack": (
            '"Inter", "Noto Sans SC", "PingFang SC", '
            '"Microsoft YaHei", system-ui, sans-serif'
        ),
        "font-stack--headings": (
            '"Inter", "Noto Sans SC", "PingFang SC", '
            '"Microsoft YaHei", system-ui, sans-serif'
        ),
    },
    "dark_css_variables": {
        "color-brand-primary": "#a5b4fc",
        "color-brand-content": "#c7d2fe",
        "color-api-name": "#5eead4",
    },
}


def _rewrite_repository_markdown_links(
    content: str,
    *,
    source_path: Path,
    published_path: Path,
) -> str:
    """Map repository Markdown links to Sphinx pages or canonical GitHub files."""

    protected_code_ranges = tuple(
        (match.start(), match.end())
        for match in MARKDOWN_CODE_PATTERN.finditer(content)
    )

    def replace(match: re.Match[str]) -> str:
        if any(
            start <= match.start() < end
            for start, end in protected_code_ranges
        ):
            return match.group(0)
        target_token = match.group("target")
        target = target_token.strip("<>")
        if (
            target.startswith(("#", "//"))
            or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target)
            or match.group("prefix").startswith("!")
        ):
            return match.group(0)

        path_and_query, anchor_separator, anchor = target.partition("#")
        path_value, query_separator, query = path_and_query.partition("?")
        if Path(unquote(path_value)).suffix.lower() != ".md":
            return match.group(0)
        resolved = (source_path.parent / unquote(path_value)).resolve()
        if not resolved.is_file():
            return match.group(0)

        if resolved == ROOT_ROADMAP:
            rewritten = os.path.relpath(
                DOCS_ROOT / "roadmap.md",
                published_path.parent,
            ).replace("\\", "/")
        elif resolved.is_relative_to(DOCS_ROOT) and not resolved.is_relative_to(
            DEVELOPMENT_NOTES_ROOT
        ):
            rewritten = os.path.relpath(
                resolved,
                published_path.parent,
            ).replace("\\", "/")
        elif resolved.is_relative_to(PROJECT_ROOT):
            repository_path = resolved.relative_to(PROJECT_ROOT).as_posix()
            rewritten = f"{REPOSITORY_BLOB_BASE}{quote(repository_path, safe='/')}"
        else:
            return match.group(0)

        if query_separator:
            rewritten = f"{rewritten}?{query}"
        if anchor_separator:
            rewritten = f"{rewritten}#{anchor}"
        return f"{match.group('prefix')}{rewritten}{match.group('suffix')}"

    return MARKDOWN_LINK_PATTERN.sub(replace, content)


def _publish_roadmap_and_development_links(
    app: Any,
    docname: str,
    source: list[str],
) -> None:
    """Publish the root roadmap and keep development links useful in HTML."""

    published_path = DOCS_ROOT / f"{docname}.md"
    if docname == "roadmap":
        app.env.note_dependency(str(ROOT_ROADMAP))
        roadmap = ROOT_ROADMAP.read_text(encoding="utf-8")
        canonical_source_note = (
            "> This page is generated from the canonical "
            f"[root `ROADMAP.md`]({REPOSITORY_BLOB_BASE}ROADMAP.md).\n\n"
        )
        roadmap = roadmap.replace(
            "# Stochaflow Roadmap\n",
            f"# Stochaflow Roadmap\n\n{canonical_source_note}",
            1,
        )
        source[0] = _rewrite_repository_markdown_links(
            roadmap,
            source_path=ROOT_ROADMAP,
            published_path=published_path,
        )
        return
    if docname == "development/README" or docname.startswith("development/"):
        source[0] = _rewrite_repository_markdown_links(
            source[0],
            source_path=published_path,
            published_path=published_path,
        )


def _use_canonical_roadmap_source_links(
    app: Any,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    """Point Furo's roadmap source actions at the canonical root document."""

    if pagename != "roadmap":
        return
    context["theme_source_view_link"] = (
        f"{REPOSITORY_BLOB_BASE}ROADMAP.md?plain=true"
    )
    context["theme_source_edit_link"] = (
        "https://github.com/supermassiveasshole/stochaflow/edit/main/ROADMAP.md"
    )


def setup(app: Any) -> dict[str, bool]:
    """Register documentation-source transformations."""

    app.connect("source-read", _publish_roadmap_and_development_links)
    app.connect("html-page-context", _use_canonical_roadmap_source_links)
    return {"parallel_read_safe": True}
