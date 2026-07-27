"""Terminal branding for the Stochaflow command-line interface."""

from math import pi, sin
from time import sleep

from rich.console import Console
from rich.live import Live
from rich.text import Text

_WORDMARK_LINES = (
    r"   _____ ______ ____  ________  _____    ________    ____ _       __",
    r"  / ___//_  __// __ \/ ____/ / / /   |  / ____/ /   / __ \ |     / /",
    r"  \__ \  / /  / / / / /   / /_/ / /| | / /_  / /   / / / / | /| / /",
    r" ___/ / / /  / /_/ / /___/ __  / ___ |/ __/ / /___/ /_/ /| |/ |/ /",
    r"/____/ /_/   \____/\____/_/ /_/_/  |_/_/   /_____/\____/ |__/|__/",
)
_TAGLINE = "       ~ ~ ~  STOCHASTIC PATHS INTO GENERATIVE FLOW  ~ ~ ~ >"
_PARTICLE_FIELD_HEIGHT = 3
_LOGO_WIDTH = max(len(_TAGLINE), *(len(line) for line in _WORDMARK_LINES))

type _ParticlePath = tuple[int, int, int, int, float, str]

_PARTICLE_PATHS: tuple[_ParticlePath, ...] = (
    (0, 0, 25, 2, 0.8, "o"),
    (2, 2, 32, 2, -1.4, "*"),
    (5, 1, 39, 2, 1.0, "+"),
    (7, 0, 46, 2, -3.2, "o"),
    (10, 2, 52, 2, 0.7, "*"),
    (12, 1, 58, 2, -3.0, "+"),
    (15, 0, 63, 2, 0.9, "o"),
    (18, 2, 67, 2, -2.8, "*"),
)
_PARTICLE_TRAIL_GLYPHS = ("~", "~", "-", "-", ".")
_PARTICLE_TRAIL_STEP = 0.045


def _smoothstep(progress: float) -> float:
    bounded = min(1.0, max(0.0, progress))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def _particle_coordinate(
    path: _ParticlePath,
    progress: float,
) -> tuple[int, int]:
    start_x, start_y, end_x, end_y, curvature, _ = path
    bounded = min(1.0, max(0.0, progress))
    eased = _smoothstep(bounded)
    x_position = round(start_x + (end_x - start_x) * eased)
    y_position = round(
        start_y
        + (end_y - start_y) * eased
        + curvature * sin(pi * bounded)
    )
    return (
        min(_LOGO_WIDTH - 1, max(0, x_position)),
        min(_PARTICLE_FIELD_HEIGHT - 1, max(0, y_position)),
    )


def _particle_field(progress: float) -> tuple[str, ...]:
    grid = [
        [" " for _ in range(_LOGO_WIDTH)]
        for _ in range(_PARTICLE_FIELD_HEIGHT)
    ]
    for path in _PARTICLE_PATHS:
        for trail_index in range(len(_PARTICLE_TRAIL_GLYPHS), -1, -1):
            trail_progress = progress - trail_index * _PARTICLE_TRAIL_STEP
            if trail_progress < 0.0:
                continue
            x_position, y_position = _particle_coordinate(path, trail_progress)
            character = (
                path[-1]
                if trail_index == 0
                else _PARTICLE_TRAIL_GLYPHS[trail_index - 1]
            )
            grid[y_position][x_position] = character
    return tuple("".join(row).rstrip() for row in grid)


ASCII_ART_LOGO = "\n".join(
    (*_particle_field(1.0), *_WORDMARK_LINES, _TAGLINE)
)

_GRADIENT_STOPS = (
    (24, 238, 255),
    (50, 132, 255),
    (151, 71, 255),
    (255, 72, 176),
)
_ANIMATION_FRAME_COUNT = 20
_ANIMATION_FRAME_DELAY_SECONDS = 1 / 30
_HIGHLIGHT_RADIUS = 0.18


def _gradient_color(position: int, width: int) -> tuple[int, int, int]:
    if width <= 1:
        return _GRADIENT_STOPS[0]

    scaled_position = position * (len(_GRADIENT_STOPS) - 1) / (width - 1)
    stop_index = min(int(scaled_position), len(_GRADIENT_STOPS) - 2)
    offset = scaled_position - stop_index
    start = _GRADIENT_STOPS[stop_index]
    end = _GRADIENT_STOPS[stop_index + 1]
    red, green, blue = (
        round(start[channel] + (end[channel] - start[channel]) * offset)
        for channel in range(3)
    )
    return red, green, blue


def _highlight_color(
    color: tuple[int, int, int],
    *,
    position: int,
    width: int,
    highlight_position: float | None,
) -> tuple[int, int, int]:
    if highlight_position is None or width <= 1:
        return color

    normalized_position = position / (width - 1)
    distance = abs(normalized_position - highlight_position)
    strength = max(0.0, 1.0 - distance / _HIGHLIGHT_RADIUS) * 0.72
    red, green, blue = (
        round(channel + (255 - channel) * strength) for channel in color
    )
    return red, green, blue


def _render_ascii_art_logo(
    *,
    lines: tuple[str, ...] | None = None,
    reveal_width: int | None = None,
    highlight_position: float | None = None,
) -> Text:
    logo_lines = tuple(ASCII_ART_LOGO.splitlines()) if lines is None else lines
    rendered = Text()
    for line_index, line in enumerate(logo_lines):
        for column, character in enumerate(line):
            if (
                reveal_width is not None
                and line_index >= _PARTICLE_FIELD_HEIGHT
                and column >= reveal_width
            ):
                rendered.append(" ")
                continue
            if character == " ":
                rendered.append(character)
                continue
            red, green, blue = _highlight_color(
                _gradient_color(column, _LOGO_WIDTH),
                position=column,
                width=_LOGO_WIDTH,
                highlight_position=highlight_position,
            )
            rendered.append(character, style=f"rgb({red},{green},{blue})")
        if line_index < len(logo_lines) - 1:
            rendered.append("\n")
    return rendered


def _animation_frame(frame_index: int) -> Text:
    progress = (frame_index + 1) / _ANIMATION_FRAME_COUNT
    reveal_progress = min(1.0, max(0.0, (progress - 0.08) / 0.68))
    reveal_width = round(_LOGO_WIDTH * reveal_progress)
    highlight_position = progress * 1.35 - 0.15
    lines = (*_particle_field(progress), *_WORDMARK_LINES, _TAGLINE)
    return _render_ascii_art_logo(
        lines=lines,
        reveal_width=reveal_width,
        highlight_position=highlight_position,
    )


def _animate_ascii_art_logo(console: Console) -> None:
    with Live(
        _animation_frame(0),
        console=console,
        auto_refresh=False,
        transient=True,
        vertical_overflow="visible",
    ) as live:
        for frame_index in range(1, _ANIMATION_FRAME_COUNT):
            sleep(_ANIMATION_FRAME_DELAY_SECONDS)
            live.update(_animation_frame(frame_index), refresh=True)
        sleep(_ANIMATION_FRAME_DELAY_SECONDS)


def print_ascii_art_logo(
    *,
    console: Console | None = None,
    animate: bool | None = None,
) -> None:
    """Animate and print the truecolor Stochaflow ASCII art logo."""

    output = console or Console(highlight=False, markup=False)
    should_animate = (
        output.is_terminal and output.is_interactive
        if animate is None
        else animate
    )
    if should_animate:
        _animate_ascii_art_logo(output)
    output.print(_render_ascii_art_logo(), soft_wrap=True)
