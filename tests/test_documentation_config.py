"""Documentation configuration contract tests."""

import runpy
import sys
from pathlib import Path

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
        if "development" not in path.parts
        and "\\coloneqq" in path.read_text(encoding="utf-8")
    ]
    assert sources


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
