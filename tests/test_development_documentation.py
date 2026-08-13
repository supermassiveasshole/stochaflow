"""Semantic integrity checks for internal development documentation."""

from __future__ import annotations

import re
from collections import deque
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

FULLWIDTH_COLON = "\N{FULLWIDTH COLON}"
FULLWIDTH_LEFT_PARENTHESIS = "\N{FULLWIDTH LEFT PARENTHESIS}"
FULLWIDTH_RIGHT_PARENTHESIS = "\N{FULLWIDTH RIGHT PARENTHESIS}"
EM_DASH = "\N{EM DASH}"

STATUS_TRANSLATIONS = {
    "已完成": "Done",
    "进行中": "In progress",
    "下一步": "Next",
    "候选": "Candidate",
    "暂停": "Parked",
}

TYPE_PATTERN = re.compile(
    rf"^> 文档类型{FULLWIDTH_COLON}\s*(?P<value>.+?)\s*$", re.MULTILINE
)
STATUS_PATTERN = re.compile(
    rf"^> 工作状态{FULLWIDTH_COLON}\s*(?P<value>已完成|进行中|下一步|候选|暂停)"
    rf"(?:{FULLWIDTH_LEFT_PARENTHESIS}"
    rf"(?P<english>Done|In progress|Next|Candidate|Parked)"
    rf"{FULLWIDTH_RIGHT_PARENTHESIS})?\s*$",
    re.MULTILINE,
)
AVAILABILITY_PATTERN = re.compile(
    rf"^> 当前可用性{FULLWIDTH_COLON}\s*(?P<value>\S.*?)\s*$", re.MULTILINE
)
RESEARCH_SCHEDULE_PATTERN = re.compile(
    rf"^> 排期状态{FULLWIDTH_COLON}\s*不参与排期\s*$", re.MULTILINE
)
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
H2_PATTERN = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)

LEGACY_STAGE_HEADING_PATTERN = re.compile(
    r"^(?:(?:阶段|里程碑|Phase|Stage)\s*)?(?:"
    r"A[0-3]|B[01]|C[01]|H[0-4]|R[0-2]|SR[0-2]|LG0|"
    r"CM[01]|W0[AB]|T[0-5]|D[0-7]|E[0-5]|L[0-3]|"
    r"LD(?:[23]|4[ABC])|Q0|S[01]|X0"
    rf")(?:\s*[:{FULLWIDTH_COLON}{EM_DASH}-]|\s*$)",
    re.IGNORECASE,
)
# SD3 is deliberately absent because it is also a maintained model name.
BANNED_PLANNING_JARGON = (
    "vertical slice",
    "substrate",
    "seam",
    "closeout",
    "promotion gate",
    "decision-gated",
    "rebase",
    "task-owned",
)
HISTORICAL_BASELINE = "5c75a76de3d696a5b734ae4eefe88a30532bd2de"


def _content(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _reader_documents() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(DEVELOPMENT_ROOT.glob("*.md"))
        if path.name != "README.md"
    )


def _typed_reader_documents() -> tuple[Path, ...]:
    return tuple(path for path in _reader_documents() if path != DEVELOPMENT_ROADMAP)


def _single_match(path: Path, pattern: re.Pattern[str], label: str) -> str:
    matches = tuple(pattern.finditer(_content(path)))
    assert len(matches) == 1, f"{path}: expected one {label} declaration"
    return matches[0].group("value").strip()


def _document_type(path: Path) -> str:
    return _single_match(path, TYPE_PATTERN, "document type")


def _status_or_none(path: Path) -> str | None:
    matches = tuple(STATUS_PATTERN.finditer(_content(path)))
    assert len(matches) <= 1, f"{path}: expected at most one work status"
    if not matches:
        return None
    status = matches[0].group("value").strip()
    english = matches[0].group("english")
    if english is not None:
        assert english == STATUS_TRANSLATIONS[status], (
            f"{path}: {status!r} does not match {english!r}"
        )
    return status


def _status(path: Path) -> str:
    status = _status_or_none(path)
    assert status is not None, f"{path}: expected one work status"
    return status


def _scheduled_documents() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _typed_reader_documents()
        if _status_or_none(path) in {"进行中", "下一步", "候选", "暂停"}
    )


def _research_documents() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _typed_reader_documents()
        if RESEARCH_SCHEDULE_PATTERN.search(_content(path))
    )


def _closed_documents() -> tuple[Path, ...]:
    return tuple(
        path
        for path in _typed_reader_documents()
        if _status_or_none(path) == "已完成"
    )


def _without_code_or_comments(content: str) -> str:
    content = FENCED_CODE_PATTERN.sub("", content)
    content = HTML_COMMENT_PATTERN.sub("", content)
    return INLINE_CODE_PATTERN.sub("", content)


def _has_semantic_paragraph(
    content: str, *concept_groups: tuple[str, ...]
) -> bool:
    """Find related ideas together without requiring one heading or wording."""

    visible = FENCED_CODE_PATTERN.sub("", content)
    visible = HTML_COMMENT_PATTERN.sub("", visible)
    visible = INLINE_CODE_PATTERN.sub(
        lambda match: match.group(0).strip("`"), visible
    )
    paragraphs = (
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", visible)
    )
    return any(
        all(any(term in paragraph for term in group) for group in concept_groups)
        for paragraph in paragraphs
        if paragraph
    )


def _local_link_targets(path: Path, content: str) -> tuple[Path, ...]:
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


def _heading_sections(content: str) -> tuple[tuple[str, str], ...]:
    content = FENCED_CODE_PATTERN.sub("", content)
    matches = tuple(H2_PATTERN.finditer(content))
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


def _section_containing(content: str, phrase: str) -> str:
    matches = [body for title, body in _heading_sections(content) if phrase in title]
    assert len(matches) == 1, f"expected one H2 containing {phrase!r}"
    return matches[0]


def _preamble(content: str) -> str:
    first_h2 = H2_PATTERN.search(content)
    return content[: first_h2.start()] if first_h2 else content


def _selection_value(content: str, pattern: re.Pattern[str]) -> str:
    matches = tuple(pattern.finditer(_preamble(content)))
    assert len(matches) == 1, f"expected one {pattern.pattern!r} declaration"
    return matches[0].group("value").strip()


def _selection_target(
    *, value: str, source: Path, empty_value: str
) -> Path | None:
    if value == empty_value:
        return None
    targets = _local_link_targets(source, value)
    assert len(targets) == 1, (
        f"{source}: a selected item must link exactly one development plan"
    )
    return targets[0]


def test_reader_documents_use_only_the_shared_header_contract() -> None:
    """Share status metadata without imposing one visible prose template."""

    for path in _typed_reader_documents():
        document_type = _document_type(path)
        assert document_type
        availability = _single_match(
            path, AVAILABILITY_PATTERN, "current availability"
        )
        assert availability
        content = _content(path)
        assert "## 先看结论" not in content

        status = _status_or_none(path)
        unscheduled = RESEARCH_SCHEDULE_PATTERN.search(content)
        assert (status is None) != (unscheduled is None), (
            f"{path}: declare either one work status or '不参与排期'"
        )


def test_status_vocabulary_and_current_queue_are_consistent() -> None:
    """Keep one scheduling authority and the derived execution queue in sync."""

    root = _content(ROADMAP)
    development = _content(DEVELOPMENT_ROADMAP)
    root_statuses = {
        cell.strip()
        for cell in re.findall(
            r"^\|\s*(Done|In progress|Next|Candidate|Parked)\s*\|",
            root,
            re.MULTILINE,
        )
    }
    assert root_statuses == set(STATUS_TRANSLATIONS.values())

    for status in ("进行中", "下一步"):
        root_value = _selection_value(root, ROOT_SELECTION_PATTERNS[status])
        development_value = _selection_value(
            development, DEVELOPMENT_SELECTION_PATTERNS[status]
        )
        root_target = _selection_target(
            value=root_value, source=ROADMAP, empty_value="None"
        )
        development_target = _selection_target(
            value=development_value,
            source=DEVELOPMENT_ROADMAP,
            empty_value="无",
        )
        selected_documents = {
            path for path in _scheduled_documents() if _status(path) == status
        }
        assert len(selected_documents) <= 1
        expected_targets = selected_documents or {None}
        assert {root_target, development_target} == expected_targets


def test_roadmap_schedules_only_concrete_directions() -> None:
    """Research notes and closed records must not become roadmap work."""

    root = _content(ROADMAP)
    candidate_links = set(
        _local_link_targets(
            ROADMAP, _section_containing(root, "Candidate")
        )
    )
    parked_links = set(
        _local_link_targets(
            ROADMAP, _section_containing(root, "Parked")
        )
    )
    backlog = {
        path
        for path in _scheduled_documents()
        if _status(path) in {"候选", "暂停"}
    }
    candidate_targets = candidate_links & backlog
    parked_targets = parked_links & backlog
    assert candidate_targets | parked_targets == backlog
    assert candidate_targets.isdisjoint(parked_targets)
    reader_documents = set(_typed_reader_documents())
    assert all(
        _status(path) == "候选"
        for path in candidate_links & reader_documents
    )
    assert all(
        _status(path) == "暂停" for path in parked_links & reader_documents
    )
    assert (candidate_links | parked_links).isdisjoint(
        {*_research_documents(), *_closed_documents()}
    )


def test_key_product_meanings_survive_editorial_rewrites() -> None:
    """Protect the few product ideas whose loss would change the roadmap."""

    workflow = _content(
        DEVELOPMENT_ROOT / "default-workflow-pipeline-support-plan.md"
    )
    for phrase in ("训练后蒸馏", "生图后超分"):
        assert phrase in workflow

    artifact = _content(
        DEVELOPMENT_ROOT / "artifact-metadata-provenance-capacity-model-proposal.md"
    )
    for idea_words in (("看懂", "说明"), ("追查", "来源"), ("资源",)):
        assert any(word in artifact for word in idea_words)
    assert any(word in artifact for word in ("独立", "分开", "不要求一起"))

    extension = _content(
        DEVELOPMENT_ROOT / "extension-import-boundary-and-activation-latency-plan.md"
    )
    assert "不修改" in extension
    assert "性能" in extension

    evaluation = DEVELOPMENT_ROOT / "post-training-evaluation-support-plan.md"
    assert _status(evaluation) == "已完成"

    sampling = _content(DEVELOPMENT_ROOT / "sampling-request-config-refactor.md")
    assert "Hydra" in sampling
    assert any(
        conclusion in sampling
        for conclusion in ("保持现状", "无需改变", "不修改")
    )

    large_data = _content(
        DEVELOPMENT_ROOT / "hierarchical-data-pipeline-support-plan.md"
    )
    for concept_words in (
        ("数据准备中断", "journal"),
        ("数据复制中断", "传输工具"),
        ("训练中断", "checkpoint"),
        ("artifact identity", "digest"),
        ("execution profile", "资源"),
    ):
        assert all(word in large_data for word in concept_words)
    assert re.search(
        r"PC.{0,120}服务器.{0,120}(?:同一|同一个).{0,40}(?:digest|identity)",
        large_data,
        re.DOTALL,
    )
    assert all(
        boundary in large_data
        for boundary in (
            "managed artifact",
            "referenced folder",
            "adopt 不能伪造",
            "显式引用已 adopt 的 artifact",
            "每个输入时仍要按 snapshot",
            "full verification",
            "DataArtifactStore",
        )
    )
    assert _status(
        DEVELOPMENT_ROOT / "hierarchical-data-pipeline-support-plan.md"
    ) == "暂停"

    assert "| Data |" in _content(ROADMAP)
    for name in (
        "data-recipe-extension-ergonomics-plan.md",
        "data-storage-and-payload-adapter-support-plan.md",
        "streaming-data-lifecycle-support-plan.md",
    ):
        path = DEVELOPMENT_ROOT / name
        assert path in _research_documents()


def test_distributed_and_large_data_stages_keep_their_product_boundary() -> None:
    """Keep the first DDP result narrow without prescribing the plan's outline."""

    distributed_path = (
        DEVELOPMENT_ROOT / "distributed-training-and-inference-support-plan.md"
    )
    distributed = _content(distributed_path)
    assert _status(distributed_path) == "暂停"
    assert _has_semantic_paragraph(
        distributed,
        ("第一项", "第一版", "首版", "首轮", "第一阶段"),
        ("单机", "single-node"),
        ("固定", "fixed"),
        ("DDP",),
        ("训练",),
    )

    deferred_words = (
        "后续",
        "随后",
        "以后",
        "第二阶段",
        "首轮之后",
        "不在首轮",
        "后置",
        "再做",
        "再考虑",
        "不会实现",
        "等待",
    )
    for concept in (("Sampling", "采样"), ("FSDP2",), ("elastic", "弹性")):
        assert _has_semantic_paragraph(distributed, concept, deferred_words)

    assert _has_semantic_paragraph(
        distributed,
        ("per-rank batch",),
        ("effective global batch",),
        ("world size",),
        ("gradient accumulation", "梯度累积"),
    )
    assert _has_semantic_paragraph(
        distributed,
        ("Trainer",),
        ("DDPTrainer",),
        ("if distributed", "条件分支", "模式分支"),
    )
    assert _has_semantic_paragraph(
        distributed,
        ("DDPTrainer",),
        ("FSDPTrainer",),
        ("独立", "不同"),
    )
    assert _has_semantic_paragraph(
        distributed,
        ("gradient accumulation", "梯度累积"),
        ("单卡", "单设备"),
        ("吞吐", "训练时间", "wall-time"),
        ("Parked", "开始前"),
    )
    roadmap = _content(ROADMAP)
    assert _has_semantic_paragraph(
        roadmap,
        ("Fixed single-node distributed training",),
        ("effective global batch",),
        ("quality",),
        ("wall-time", "throughput"),
    )
    assert _has_semantic_paragraph(
        distributed,
        ("DataBuilder",),
        ("实际样本数", "loss 权重"),
        ("不透明 batch", "不查看", "不解释"),
        ("覆盖证据",),
    )

    large_data = _content(
        DEVELOPMENT_ROOT / "hierarchical-data-pipeline-support-plan.md"
    )
    development_roadmap = _content(DEVELOPMENT_ROADMAP)
    for content in (large_data, development_roadmap):
        assert _has_semantic_paragraph(
            content,
            ("PC", "个人电脑"),
            ("服务器",),
            ("单卡", "单设备"),
            ("不依赖", "无需", "不需要", "可以先"),
            ("DDP", "Distributed"),
        )
        assert _has_semantic_paragraph(
            content,
            ("八卡", "8 卡", "8卡"),
            ("依赖", "才由", "需要"),
            ("DDP", "Distributed", "多设备"),
        )


def test_retained_future_interface_sketches_are_clearly_non_executable() -> None:
    """Keep only useful sketches, without presenting them as current APIs."""

    for name in (
        "default-workflow-pipeline-support-plan.md",
        "hydra-configuration-composition-migration-plan.md",
        "distributed-training-and-inference-support-plan.md",
        "automated-model-tuning-plan.md",
    ):
        content = _content(DEVELOPMENT_ROOT / name)
        assert re.search(r"(?:不能|不可)\s*执行", content)
        assert re.search(r"(?:不是|不构成|非)\s*公共\s*API", content)


def test_development_index_reaches_reader_topics_without_fixing_its_layout() -> None:
    """Allow natural grouping while keeping every reader topic discoverable."""

    reader_documents = set(_reader_documents())
    edges: dict[Path, set[Path]] = {}
    for path in (DEVELOPMENT_INDEX, *_reader_documents()):
        edges[path] = {
            target
            for target in _local_link_targets(path, _content(path))
            if target in reader_documents
        }

    distance = {DEVELOPMENT_INDEX: 0}
    queue: deque[Path] = deque([DEVELOPMENT_INDEX])
    while queue:
        source = queue.popleft()
        if distance[source] == 2:
            continue
        for target in edges.get(source, set()):
            if target not in distance:
                distance[target] = distance[source] + 1
                queue.append(target)

    missing = reader_documents - distance.keys()
    assert not missing, f"development topics are more than two clicks away: {missing}"
    assert "development/README" in _content(DOCS_INDEX)


def test_historical_baseline_has_a_current_location_map() -> None:
    """Preserve maintainer ideas explicitly instead of relying on Git history."""

    content = _content(CONTENT_MAP)
    assert HISTORICAL_BASELINE in content
    mapped_rows = []
    for line in content.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 3, f"malformed historical mapping row: {line}"
        if not re.fullmatch(r"`[^`]+\.md`", cells[0]):
            continue
        name = cells[0].strip("`")
        assert cells[1], f"historical mapping does not say what {name} preserves"
        assert "](" in cells[2], f"historical mapping has no current location: {name}"
        mapped_rows.append(name)

    assert mapped_rows, "historical mapping must retain at least one source document"
    assert len(mapped_rows) == len(set(mapped_rows)), (
        "historical mapping must not assign one source document more than once"
    )


def test_reader_openings_and_headings_avoid_old_planning_shorthand() -> None:
    """Keep old IDs and unexplained planning jargon out of the reading path."""

    for path in (DEVELOPMENT_INDEX, *_reader_documents()):
        content = _content(path)
        visible = _without_code_or_comments(content)
        headings = [title for title, _ in _heading_sections(visible)]
        for title in headings:
            assert LEGACY_STAGE_HEADING_PATTERN.search(title) is None, (
                f"{path}: legacy stage heading {title!r} belongs in history"
            )
        opening_and_headings = _preamble(visible) + "\n" + "\n".join(headings)
        for term in BANNED_PLANNING_JARGON:
            pattern = rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])"
            assert re.search(pattern, opening_and_headings, re.IGNORECASE) is None, (
                f"{path}: explain planning jargon {term!r} in ordinary language"
            )


def test_development_markdown_links_resolve_inside_the_repository() -> None:
    """Reject missing or machine-local targets in development documentation."""

    paths = (
        ROADMAP,
        DEVELOPMENT_INDEX,
        *_reader_documents(),
        *sorted(NOTES_ROOT.rglob("*.md")),
    )
    for path in paths:
        for target in _local_link_targets(path, _content(path)):
            assert target.is_relative_to(PROJECT_ROOT), (
                f"{path}: local link escapes the repository: {target}"
            )
            assert target.exists(), f"{path}: missing local link target {target}"
