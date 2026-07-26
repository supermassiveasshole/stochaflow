"""Repository-wide structural coding convention checks."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def controlled_python_files() -> tuple[Path, ...]:
    """Return tracked and non-ignored Python source candidates."""

    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            ".",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    relative_paths = (
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    )
    return tuple(
        REPOSITORY_ROOT / relative_path
        for relative_path in sorted(relative_paths)
        if relative_path.endswith((".py", ".py.tmpl"))
        and (REPOSITORY_ROOT / relative_path).is_file()
    )


def test_class_guard_includes_python_templates() -> None:
    expected = (
        REPOSITORY_ROOT
        / "src"
        / "stochaflow"
        / "projects"
        / "templates"
        / "data.py.tmpl"
    )

    assert expected in controlled_python_files()


def test_python_class_names_are_formal() -> None:
    """Reject private-looking class declarations in every controlled Python file."""

    violations: list[str] = []
    for path in controlled_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("_"):
                relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
                violations.append(f"{relative_path}:{node.lineno}: {node.name}")

    assert not violations, (
        "Class names must be formal PascalCase identifiers without a leading "
        "underscore:\n" + "\n".join(violations)
    )
