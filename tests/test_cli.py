"""Tests for the unified Stochaflow command-line interface."""

from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from stochaflow.scripts import branding
from stochaflow.scripts.branding import ASCII_ART_LOGO, print_ascii_art_logo
from stochaflow.scripts.cli import build_argument_parser, main


def test_particle_field_transports_and_converges_noise() -> None:
    initial_field = branding._particle_field(0.0)
    final_field = branding._particle_field(1.0)
    particle_heads = {"o", "*", "+"}
    initial_heads = {
        (row, column)
        for row, line in enumerate(initial_field)
        for column, character in enumerate(line)
        if character in particle_heads
    }
    final_heads = {
        (row, column)
        for row, line in enumerate(final_field)
        for column, character in enumerate(line)
        if character in particle_heads
    }

    assert {row for row, _ in initial_heads} == {0, 1, 2}
    assert {row for row, _ in final_heads} == {2}
    assert min(column for _, column in final_heads) > max(
        column for _, column in initial_heads
    )
    assert any("~" in line for line in final_field)
    assert ASCII_ART_LOGO.splitlines()[:3] == list(final_field)


def test_logo_uses_truecolor_gradient_in_supported_terminal() -> None:
    stream = StringIO()
    console = Console(
        file=stream,
        color_system="truecolor",
        force_terminal=True,
        no_color=False,
        width=120,
    )

    print_ascii_art_logo(console=console, animate=False)

    rendered = stream.getvalue()
    assert "\x1b[38;2;" in rendered
    assert Text.from_ansi(rendered).plain == f"{ASCII_ART_LOGO}\n"


def test_logo_animates_in_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    stream = StringIO()
    console = Console(
        file=stream,
        color_system="truecolor",
        force_interactive=True,
        force_terminal=True,
        no_color=False,
        width=120,
    )
    delays: list[float] = []
    monkeypatch.setattr(branding, "sleep", delays.append)

    print_ascii_art_logo(console=console)

    rendered = stream.getvalue()
    plain_rendered = Text.from_ansi(rendered).plain
    assert delays
    assert all(delay > 0 for delay in delays)
    assert plain_rendered.count("STOCHASTIC PATHS INTO GENERATIVE FLOW") > 1
    assert "\x1b[?25l" in rendered
    assert "\x1b[?25h" in rendered


def test_main_prints_ascii_art_logo_before_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        branding,
        "sleep",
        lambda _: pytest.fail("non-interactive CLI output must not animate"),
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert ASCII_ART_LOGO.isascii()
    assert "\x1b[" not in captured.out
    assert captured.out.startswith(f"{ASCII_ART_LOGO}\nusage: stochaflow")
    assert captured.out.count(ASCII_ART_LOGO) == 1


def test_building_parser_has_no_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_argument_parser()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
