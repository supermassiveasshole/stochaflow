"""Readability and integrity contracts for internal development documents."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_INDEX = PROJECT_ROOT / "docs" / "index.md"
DEVELOPMENT_ROOT = PROJECT_ROOT / "docs" / "development"
NOTES_ROOT = DEVELOPMENT_ROOT / "notes"
ROADMAP = PROJECT_ROOT / "ROADMAP.md"
DEVELOPMENT_ROADMAP = DEVELOPMENT_ROOT / "development-priority-roadmap.md"
DEVELOPMENT_INDEX = DEVELOPMENT_ROOT / "README.md"
CONTENT_MAP = NOTES_ROOT / "document-restructure-content-map.md"
HISTORICAL_REVIEW_CANDIDATES_PATH = (
    DEVELOPMENT_ROOT / "historical-review-candidates.txt"
)

STATUS_TRANSLATIONS = {
    "已完成": "Done",
    "进行中": "In progress",
    "下一步": "Next",
    "候选": "Candidate",
    "暂停": "Parked",
}
ALLOWED_STATUSES = frozenset(STATUS_TRANSLATIONS)
FUTURE_STATUSES = frozenset({"进行中", "下一步", "候选", "暂停"})
FULLWIDTH_COLON = "\N{FULLWIDTH COLON}"
FULLWIDTH_LEFT_PARENTHESIS = "\N{FULLWIDTH LEFT PARENTHESIS}"
FULLWIDTH_RIGHT_PARENTHESIS = "\N{FULLWIDTH RIGHT PARENTHESIS}"
STATUS_PATTERN = re.compile(
    rf"^> 工作状态{FULLWIDTH_COLON}(.+)$", re.MULTILINE
)
FENCED_CODE_PATTERN = re.compile(
    r"^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
MARKDOWN_LINK_PATTERN = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>\n]+>|[^)\s]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
VISIBLE_LINK_PATTERN = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]+\)")
H2_PATTERN = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)
H3_PATTERN = re.compile(r"^### (?P<title>.+?)\s*$", re.MULTILINE)
TOCTREE_PATTERN = re.compile(
    r"^```\{toctree\}\s*$\n(?P<body>.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)
LEGACY_STAGE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"A[0-3]|B[01]|C[01]|H[0-4]|R[0-2]|SR[0-2]|LG0|"
    r"CM[01]|W0[AB]|T[0-5]|D[0-7]|E[0-5]|L[0-3]|"
    r"LD(?:[23]|4[ABC])|Q0|S[01]|SD(?:10|[0-24-9])|X0|P2"
    r")(?![A-Za-z0-9])"
)
# SD3 is deliberately absent: it is also the maintained Stable Diffusion 3 name.
BANNED_JARGON = (
    "vertical slice",
    "substrate",
    "seam",
    "closeout",
    "promotion gate",
    "decision-gated",
    "rebase",
    "task-owned",
    "milestone",
)
FUTURE_REQUIRED_HEADINGS = (
    "完成后用户能做什么",
    "当前仓库已经支持什么",
    "还没有支持什么",
    "要完成哪些工作",
    "如何证明已经完成",
    "明确不包含什么",
    "详细设计和研究资料在哪里",
)
COMPLETED_REQUIRED_HEADINGS = (
    "完成后用户能做什么",
    "当前仓库已经支持什么",
    "还没有支持什么",
    "什么时候需要重新讨论",
    "如何证明当前决策仍然成立",
    "明确不包含什么",
    "详细设计和研究资料在哪里",
)
START_OR_REVIEW_HEADINGS = frozenset(
    {
        "什么时候可以开始",
        "什么时候重新审查",
        "什么时候可以开始或重新审查",
    }
)
TASK_CARD_FIELDS = (
    "动作",
    "原因",
    "影响范围",
    "交付物",
    "验证方法",
    "完成条件",
)
TASK_CARD_FIELD_PATTERN = re.compile(
    rf"^\s*[-*]\s+(?:\*\*)?(?P<label>{'|'.join(TASK_CARD_FIELDS)})"
    rf"[:{FULLWIDTH_COLON}](?:\*\*)?\s*(?P<value>\S.*)$",
    re.MULTILINE,
)
INDEX_STATUS_MARKERS = {
    "已完成": f"已完成{FULLWIDTH_COLON}",
    "候选": f"候选{FULLWIDTH_COLON}",
    "暂停": f"暂停{FULLWIDTH_COLON}",
}
ROOT_SELECTION_PATTERNS = {
    "进行中": re.compile(r"^> In progress: (?P<value>.+?)\s*$", re.MULTILINE),
    "下一步": re.compile(r"^> Next: (?P<value>.+?)\s*$", re.MULTILINE),
}
DEVELOPMENT_SELECTION_PATTERNS = {
    "进行中": re.compile(
        rf"^- 进行中{FULLWIDTH_COLON}(?P<value>.+?)\s*$", re.MULTILINE
    ),
    "下一步": re.compile(
        rf"^- 下一步{FULLWIDTH_COLON}(?P<value>.+?)\s*$", re.MULTILINE
    ),
}


def _historical_review_candidate_names() -> frozenset[str]:
    """Return historical files awaiting a separate deletion decision."""

    return frozenset(
        line.strip()
        for line in HISTORICAL_REVIEW_CANDIDATES_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    )


def _main_development_documents() -> tuple[Path, ...]:
    """Return reader-facing development documents, excluding the index."""

    return tuple(
        path
        for path in sorted(DEVELOPMENT_ROOT.glob("*.md"))
        if path.name != "README.md"
        and path.name not in _historical_review_candidate_names()
    )


def _main_plans() -> tuple[Path, ...]:
    """Return status-bearing plans, excluding the portfolio roadmap."""

    return tuple(
        path
        for path in _main_development_documents()
        if path != DEVELOPMENT_ROADMAP
    )


def _toctree_targets(content: str) -> frozenset[str]:
    """Return document targets from every MyST toctree block."""

    targets: set[str] = set()
    for match in TOCTREE_PATTERN.finditer(content):
        for raw_line in match.group("body").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            explicit_target = re.search(r"<([^<>]+)>$", line)
            targets.add(explicit_target.group(1) if explicit_target else line)
    return frozenset(targets)


def _without_code_or_comments(content: str) -> str:
    """Remove text that Markdown does not present as document navigation."""

    content = FENCED_CODE_PATTERN.sub("", content)
    content = HTML_COMMENT_PATTERN.sub("", content)
    return INLINE_CODE_PATTERN.sub("", content)


def _reader_text(content: str) -> str:
    """Return narrative text while retaining inline-code identifiers."""

    content = FENCED_CODE_PATTERN.sub("", content)
    content = HTML_COMMENT_PATTERN.sub("", content)
    return VISIBLE_LINK_PATTERN.sub(r"\1", content)


def _status(path: Path, content: str) -> str:
    """Read the one required work-status declaration."""

    matches = STATUS_PATTERN.findall(content)
    assert len(matches) == 1, (
        f"{path}: expected one '> 工作状态{FULLWIDTH_COLON}…' line"
    )
    status = matches[0].strip()
    assert status in ALLOWED_STATUSES, f"{path}: unsupported status {status!r}"
    return status


def _local_link_targets(path: Path, content: str) -> tuple[Path, ...]:
    """Resolve inline local Markdown links relative to their document."""

    content = _without_code_or_comments(content)
    resolved: list[Path] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(content):
        target = match.group("target").strip().strip("<>")
        if not target or target.startswith(("#", "//")):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
            continue
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if target:
            resolved.append((path.parent / target).resolve())
    return tuple(resolved)


def _heading_sections(content: str, level: int) -> tuple[tuple[str, str], ...]:
    """Return exact ATX headings and the content up to the next peer heading."""

    content = FENCED_CODE_PATTERN.sub("", content)
    pattern = H2_PATTERN if level == 2 else H3_PATTERN
    matches = tuple(pattern.finditer(content))
    return tuple(
        (
            match.group("title").strip(),
            content[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            ],
        )
        for index, match in enumerate(matches)
    )


def _one_section(
    path: Path, sections: tuple[tuple[str, str], ...], title: str
) -> str:
    """Return one required section body."""

    bodies = [body for heading, body in sections if heading == title]
    assert len(bodies) == 1, f"{path}: expected one H2 {title!r}"
    return bodies[0]


def _roadmap_preamble(content: str) -> str:
    """Exclude repeated explanatory status text below the first H2."""

    first_h2 = H2_PATTERN.search(content)
    return content[: first_h2.start()] if first_h2 else content


def _roadmap_selection(
    path: Path,
    content: str,
    pattern: re.Pattern[str],
    none_value: str,
) -> frozenset[Path]:
    """Parse one roadmap selection as no item or one linked main plan."""

    matches = tuple(pattern.finditer(_roadmap_preamble(content)))
    assert len(matches) == 1, f"{path}: expected one {pattern.pattern!r} declaration"
    value = matches[0].group("value").strip()
    if value == none_value:
        return frozenset()
    targets = _local_link_targets(path, value)
    assert len(targets) == 1, f"{path}: selected work must be one Markdown link"
    assert targets[0] in _main_plans(), f"{path}: selection is not a main plan"
    return frozenset(targets)


def _status_table_values(content: str, marker: str) -> frozenset[str]:
    """Read the first cells from the uniquely identified status-model section."""

    sections = _heading_sections(content, 2)
    matching = [body for title, body in sections if marker in title]
    assert len(matching) == 1, f"expected one status section containing {marker!r}"
    cells = re.findall(r"^\|\s*([^|]+?)\s*\|", matching[0], re.MULTILINE)
    return frozenset(
        cell.strip()
        for cell in cells
        if cell.strip() not in {"Status", "状态"}
        and re.fullmatch(r"-+", cell.strip()) is None
    )


def test_status_vocabulary_and_roadmap_selections_are_consistent() -> None:
    """Keep the five-state model and the two scheduling authorities aligned."""

    root = ROADMAP.read_text(encoding="utf-8")
    development = DEVELOPMENT_ROADMAP.read_text(encoding="utf-8")
    index = DEVELOPMENT_INDEX.read_text(encoding="utf-8")

    root_cells = _status_table_values(root, "Status model")
    assert root_cells == frozenset(STATUS_TRANSLATIONS.values())
    expected_chinese_cells = frozenset(
        f"{status}{FULLWIDTH_LEFT_PARENTHESIS}{english}"
        f"{FULLWIDTH_RIGHT_PARENTHESIS}"
        for status, english in STATUS_TRANSLATIONS.items()
    )
    for content in (development, index):
        cells = _status_table_values(content, "五种状态")
        assert cells == expected_chinese_cells

    plan_statuses = {
        path: _status(path, path.read_text(encoding="utf-8"))
        for path in _main_plans()
    }
    for status in ("进行中", "下一步"):
        root_selected = _roadmap_selection(
            ROADMAP, root, ROOT_SELECTION_PATTERNS[status], "None"
        )
        development_selected = _roadmap_selection(
            DEVELOPMENT_ROADMAP,
            development,
            DEVELOPMENT_SELECTION_PATTERNS[status],
            "无",
        )
        marked = frozenset(
            path for path, plan_status in plan_statuses.items() if plan_status == status
        )
        assert len(marked) <= 1, f"at most one plan may be {status}"
        assert root_selected == development_selected == marked

    assert len(root.splitlines()) <= 160
    assert len(development.splitlines()) <= 250


def test_development_index_links_and_classifies_every_main_document() -> None:
    """Require exact index links and the expected primary status membership."""

    index = DEVELOPMENT_INDEX.read_text(encoding="utf-8")
    index_targets = frozenset(_local_link_targets(DEVELOPMENT_INDEX, index))
    for path in _main_development_documents():
        assert path in index_targets, f"development index does not link {path.name}"

    sections = _heading_sections(index, 2)
    for status, marker in INDEX_STATUS_MARKERS.items():
        matches = [body for title, body in sections if marker in title]
        assert len(matches) == 1, f"index must have one {status} section"
        category_targets = frozenset(
            _local_link_targets(DEVELOPMENT_INDEX, matches[0])
        )
        # A plan may also document a completed foundation or a differently parked
        # subgoal, so require primary membership without forbidding extra links.
        for path in _main_plans():
            if _status(path, path.read_text(encoding="utf-8")) == status:
                assert path in category_targets, (
                    f"{path}: missing from index category {status}"
                )


def test_historical_review_candidates_are_not_treated_as_main_plans() -> None:
    """Keep unapproved historical deletions outside the reader-facing plan set."""

    existing_names = {
        path.name for path in DEVELOPMENT_ROOT.glob("*.md")
    }
    main_names = {path.name for path in _main_development_documents()}
    assert existing_names <= {
        "README.md",
        *main_names,
        *_historical_review_candidate_names(),
    }

    deletion_boundary = CONTENT_MAP.read_text(encoding="utf-8").split(
        "## 删除边界",
        1,
    )[1]
    for name in _historical_review_candidate_names():
        assert f"`{name}`" in deletion_boundary


def test_sphinx_navigation_reaches_roadmap_and_every_main_plan() -> None:
    """Keep the published maintenance entry complete and directly navigable."""

    homepage_targets = _toctree_targets(
        DOCS_INDEX.read_text(encoding="utf-8")
    )
    assert "roadmap" in homepage_targets
    assert "development/README" in homepage_targets

    development_targets = _toctree_targets(
        DEVELOPMENT_INDEX.read_text(encoding="utf-8")
    )
    expected_targets = frozenset(
        path.stem for path in _main_development_documents()
    )
    assert development_targets == expected_targets


def test_main_plans_use_ordered_nonempty_sections_and_task_cards() -> None:
    """Make every plan and every task card answer its operational questions."""

    for path in _main_plans():
        content = path.read_text(encoding="utf-8")
        status = _status(path, content)
        sections = _heading_sections(content, 2)
        titles = [title for title, _ in sections]

        assert len(content.splitlines()) <= 400, f"{path}: main plan exceeds 400 lines"
        assert len(titles) == len(set(titles)), f"{path}: duplicate H2 heading"
        for title, body in sections:
            assert _reader_text(body).strip(), f"{path}: empty H2 section {title!r}"

        source_lines = [
            line
            for line in content.splitlines()
            if line.startswith(f"> 规范来源{FULLWIDTH_COLON}")
        ]
        assert len(source_lines) == 1, f"{path}: expected one normative-source line"
        assert _local_link_targets(path, source_lines[0]), (
            f"{path}: normative sources must be Markdown links"
        )

        if status in FUTURE_STATUSES:
            start_titles = [title for title in titles if title in START_OR_REVIEW_HEADINGS]
            assert len(start_titles) == 1, f"{path}: expected one start/review H2"
            required = (
                *FUTURE_REQUIRED_HEADINGS[:3],
                start_titles[0],
                *FUTURE_REQUIRED_HEADINGS[3:],
            )
            work = _one_section(path, sections, "要完成哪些工作")
            cards = _heading_sections(work, 3)
            assert cards, f"{path}: work section must contain task cards"
            for card_title, card_body in cards:
                fields = tuple(TASK_CARD_FIELD_PATTERN.finditer(card_body))
                labels = tuple(field.group("label") for field in fields)
                assert labels == TASK_CARD_FIELDS, (
                    f"{path}: task card {card_title!r} has fields {labels!r}"
                )
                assert all(
                    _reader_text(field.group("value")).strip() for field in fields
                ), (
                    f"{path}: task card {card_title!r} has an empty field"
                )
        else:
            required = COMPLETED_REQUIRED_HEADINGS

        assert tuple(titles) == required, (
            f"{path}: H2 headings must exactly match the required ordered template; "
            f"got {titles!r}"
        )


def test_main_documents_avoid_legacy_ids_and_planning_jargon() -> None:
    """Keep retired shorthand in archives rather than reader-facing plans."""

    for path in (DEVELOPMENT_INDEX, *_main_development_documents()):
        visible = _reader_text(path.read_text(encoding="utf-8"))
        legacy_match = LEGACY_STAGE_PATTERN.search(visible)
        assert legacy_match is None, (
            f"{path}: legacy stage ID {legacy_match.group(0)!r} belongs in the "
            "history mapping"
        )
        for term in BANNED_JARGON:
            pattern = rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])"
            assert re.search(pattern, visible, re.IGNORECASE) is None, (
                f"{path}: replace planning jargon {term!r}"
            )
        for line in visible.splitlines():
            if line.startswith("#"):
                assert re.search(
                    r"\b(?:stage|phase|milestone)\b", line, re.IGNORECASE
                ) is None, f"{path}: heading must name an action or result: {line!r}"


def test_development_notes_are_owned_and_reachable_from_the_index() -> None:
    """Keep the index-to-plan-to-notes ownership graph navigable."""

    main_documents = _main_development_documents()
    notes = tuple(sorted(NOTES_ROOT.rglob("*.md")))
    nodes = frozenset((DEVELOPMENT_INDEX, *main_documents, *notes))
    edges = {
        path: frozenset(
            target
            for target in _local_link_targets(
                path, path.read_text(encoding="utf-8")
            )
            if target in nodes
        )
        for path in nodes
    }

    reachable = {DEVELOPMENT_INDEX}
    pending = [DEVELOPMENT_INDEX]
    while pending:
        source = pending.pop()
        for target in edges[source] - reachable:
            reachable.add(target)
            pending.append(target)

    assert set(main_documents) <= reachable
    assert set(notes) <= reachable
    for note in notes:
        incoming = {source for source, targets in edges.items() if note in targets}
        assert incoming, f"{note}: note has no inbound Markdown link"

    ownership = CONTENT_MAP.read_text(encoding="utf-8").split("## 删除边界", 1)[0]
    ownership_targets = frozenset(
        target
        for line in ownership.splitlines()
        if line.startswith("|")
        for target in _local_link_targets(CONTENT_MAP, line.split("|", 2)[1])
    )
    for path in _main_plans():
        content = path.read_text(encoding="utf-8")
        if _status(path, content) not in FUTURE_STATUSES:
            continue
        assert path in ownership_targets, (
            f"{path}: future plan has no owner record in the content map"
        )
        sections = _heading_sections(content, 2)
        start_titles = [
            title for title, _ in sections if title in START_OR_REVIEW_HEADINGS
        ]
        assert len(start_titles) == 1, f"{path}: expected one start/review H2"
        assert _reader_text(_one_section(path, sections, start_titles[0])).strip(), (
            f"{path}: start/review condition is empty"
        )
        detail = _one_section(path, sections, "详细设计和研究资料在哪里")
        note_targets = {
            target
            for target in _local_link_targets(path, detail)
            if target.suffix == ".md" and target.is_relative_to(NOTES_ROOT)
        }
        assert note_targets, f"{path}: detail section does not link owned notes"


def test_development_markdown_links_resolve_inside_the_repository() -> None:
    """Reject missing or machine-local targets in development documentation."""

    paths = (
        ROADMAP,
        DEVELOPMENT_INDEX,
        *_main_development_documents(),
        *sorted(NOTES_ROOT.rglob("*.md")),
    )
    for path in paths:
        content = path.read_text(encoding="utf-8")
        for target in _local_link_targets(path, content):
            assert target.is_relative_to(PROJECT_ROOT), (
                f"{path}: local link escapes the repository: {target}"
            )
            assert target.exists(), f"{path}: missing local link target {target}"
