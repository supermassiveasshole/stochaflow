"""Sphinx configuration for the Stochaflow documentation site."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
exclude_patterns = ["_build", "development/**", "Thumbs.db", ".DS_Store"]

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
