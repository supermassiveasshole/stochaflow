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

nitpicky = True
autosummary_generate = True
intersphinx_mapping: dict[str, tuple[str, str | None]] = {}

html_theme = "furo"
html_title = "Stochaflow 文档"
html_baseurl = "https://supermassiveasshole.github.io/stochaflow/"
html_theme_options = {
    "source_repository": "https://github.com/supermassiveasshole/stochaflow/",
    "source_branch": "main",
    "source_directory": "docs/",
}
