"""Generate the deterministic Stochaflow configuration reference."""

from __future__ import annotations

import argparse
import difflib
import inspect
import re
from collections.abc import Mapping
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from stochaflow.scripts.cli import build_argument_parser
from stochaflow.utils.config import SampleInvocationConfig, StochaflowConfig
from stochaflow.utils.factory import load_builtin_components
from stochaflow.utils.registry import REGISTRIES

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = REPO_ROOT / "docs" / "configuration" / "_reference.yaml"
OUTPUT_PATH = REPO_ROOT / "docs" / "configuration" / "reference.md"

REGISTRY_NAMES = (
    "models",
    "data_builders",
    "sampling_artifact_writers",
    "noise_schedules",
    "processes",
    "samplers",
    "sampling_builders",
    "training_builders",
    "objectives",
    "metrics",
    "optimizers",
    "lr_schedulers",
    "loggers",
    "diagnostics",
)


class ReferenceError(ValueError):
    """Raised when reference metadata no longer matches runtime interfaces."""


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin not in {Union, UnionType}:
        return annotation
    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    return args[0] if len(args) == 1 else annotation


def _field_records(
    cls: type[Any],
    prefix: str = "",
) -> list[tuple[str, Any, Any]]:
    records: list[tuple[str, Any, Any]] = []
    hints = get_type_hints(cls)
    for field_info in fields(cls):
        path = f"{prefix}.{field_info.name}" if prefix else field_info.name
        annotation = hints.get(field_info.name, field_info.type)
        records.append((path, annotation, field_info))
        nested = _unwrap_optional(annotation)
        origin = get_origin(nested)
        if isinstance(nested, type) and is_dataclass(nested):
            records.extend(_field_records(nested, path))
        elif origin is list:
            element_type = _unwrap_optional(get_args(nested)[0])
            if isinstance(element_type, type) and is_dataclass(element_type):
                records.extend(_field_records(element_type, f"{path}[]"))
    return records


def _schema_field_records() -> list[tuple[str, Any, Any]]:
    """Return the fields owned by the training and sample config authorities."""

    records: list[tuple[str, Any, Any]] = []
    seen: dict[str, tuple[Any, Any]] = {}
    for schema in (StochaflowConfig, SampleInvocationConfig):
        for path, annotation, field_info in _field_records(schema):
            previous = seen.get(path)
            if previous is None:
                seen[path] = (annotation, field_info)
                records.append((path, annotation, field_info))
                continue
            previous_annotation, previous_field = previous
            previous_required, previous_default = _default_value(previous_field)
            required, default = _default_value(field_info)
            if (
                _type_name(previous_annotation) != _type_name(annotation)
                or previous_required != required
                or _plain_value(previous_default) != _plain_value(default)
            ):
                raise ReferenceError(
                    f"shared config field '{path}' has incompatible schema definitions"
                )
    return records


def _type_name(annotation: Any) -> str:
    if annotation is Any:
        return "any"
    if annotation is type(None):
        return "null"
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        return " | ".join(_type_name(arg) for arg in get_args(annotation))
    if origin is list:
        return f"list[{_type_name(get_args(annotation)[0])}]"
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return f"mapping[{_type_name(key_type)}, {_type_name(value_type)}]"
    if isinstance(annotation, type) and is_dataclass(annotation):
        return "mapping"
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _default_value(field_info: Any) -> tuple[bool, Any]:
    if field_info.default is not MISSING:
        return False, field_info.default
    if field_info.default_factory is not MISSING:
        return False, field_info.default_factory()
    return True, None


def _plain_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _plain_value(item) for key, item in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain_value(item) for key, item in value.items()}
    return value


def _yaml_inline(value: Any) -> str:
    rendered = yaml.safe_dump(
        _plain_value(value),
        allow_unicode=True,
        default_flow_style=True,
        sort_keys=True,
        width=4096,
    ).strip()
    return rendered.removesuffix("\n...").replace("|", "\\|")


def _anchor(prefix: str, name: str) -> str:
    normalized = name.replace("[]", "-item")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return f"{prefix}-{normalized}"


def _required_text(value: Any, *, path: str) -> str:
    if not isinstance(value, Mapping):
        raise ReferenceError(f"metadata for '{path}' must be a mapping")
    description = value.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ReferenceError(f"metadata for '{path}' is missing description")
    return description.strip()


def _load_metadata() -> dict[str, Any]:
    try:
        raw = yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReferenceError(f"missing reference metadata: {METADATA_PATH}") from exc
    if not isinstance(raw, dict):
        raise ReferenceError("reference metadata must be a top-level mapping")
    return raw


def _validate_fields(metadata: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    records = _schema_field_records()
    expected = {path for path, _, _ in records}
    field_metadata = metadata.get("fields")
    if not isinstance(field_metadata, Mapping):
        raise ReferenceError("reference metadata must define a fields mapping")
    actual = set(field_metadata)
    missing = sorted(expected - actual)
    stale = sorted(actual - expected)
    if missing or stale:
        raise ReferenceError(
            "field metadata is out of date; "
            f"missing={missing or '<none>'}, stale={stale or '<none>'}"
        )
    for path in sorted(expected):
        _required_text(field_metadata[path], path=path)
    return records


def _signature_parameters(
    component: Any,
    *,
    runtime_parameters: set[str],
) -> set[str]:
    signature = inspect.signature(component)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name not in runtime_parameters
        and parameter.kind
        not in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
    }


def _validate_registries(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    load_builtin_components()
    registry_metadata = metadata.get("registries")
    if not isinstance(registry_metadata, Mapping):
        raise ReferenceError("reference metadata must define a registries mapping")
    if set(registry_metadata) != set(REGISTRY_NAMES):
        raise ReferenceError(
            "registry metadata categories are out of date; "
            f"expected={sorted(REGISTRY_NAMES)}, actual={sorted(registry_metadata)}"
        )

    for registry_name in REGISTRY_NAMES:
        section = registry_metadata[registry_name]
        _required_text(section, path=f"registries.{registry_name}")
        if not isinstance(section, Mapping):
            raise ReferenceError(f"registries.{registry_name} must be a mapping")
        entries = section.get("entries")
        if not isinstance(entries, Mapping):
            raise ReferenceError(f"registries.{registry_name}.entries is required")
        registry = getattr(REGISTRIES, registry_name)
        expected_names = set(registry.names())
        if set(entries) != expected_names:
            raise ReferenceError(
                f"registry '{registry_name}' metadata is out of date; "
                f"expected={sorted(expected_names)}, actual={sorted(entries)}"
            )
        for entry_name in registry.names():
            entry = entries[entry_name]
            _required_text(
                entry,
                path=f"registries.{registry_name}.entries.{entry_name}",
            )
            if not isinstance(entry, Mapping):
                raise ReferenceError(
                    f"registry entry '{registry_name}.{entry_name}' must be a mapping"
                )
            parameters = entry.get("parameters", {})
            if not isinstance(parameters, Mapping):
                raise ReferenceError(
                    f"registry entry '{registry_name}.{entry_name}' parameters "
                    "must be a mapping"
                )
            parameter_note = entry.get("parameter_note")
            if parameter_note is not None and (
                not isinstance(parameter_note, str) or not parameter_note.strip()
            ):
                raise ReferenceError(
                    f"registry entry '{registry_name}.{entry_name}' parameter_note "
                    "must be non-empty text"
                )
            for parameter_name, parameter in parameters.items():
                _required_text(
                    parameter,
                    path=(
                        f"registries.{registry_name}.entries.{entry_name}."
                        f"parameters.{parameter_name}"
                    ),
                )
            inspection = entry.get("inspect", "signature")
            if inspection not in {
                "signature",
                "config_parameters",
                "context_parameters",
            }:
                raise ReferenceError(
                    f"registry entry '{registry_name}.{entry_name}' has unknown "
                    f"inspection mode {inspection!r}"
                )
            if inspection == "config_parameters":
                component = registry.resolve(entry_name)
                config_parameters = getattr(component, "config_parameters", None)
                if not isinstance(config_parameters, frozenset):
                    raise ReferenceError(
                        f"'{registry_name}.{entry_name}' must declare a "
                        "config_parameters frozenset"
                    )
                if set(parameters) != set(config_parameters):
                    raise ReferenceError(
                        f"parameters for '{registry_name}.{entry_name}' are out of "
                        f"date; expected={sorted(config_parameters)}, "
                        f"actual={sorted(parameters)}"
                    )
                runtime_parameters = set(entry.get("runtime_parameters", []))
                constructor_parameters = set(inspect.signature(component).parameters)
                if runtime_parameters != constructor_parameters:
                    raise ReferenceError(
                        f"runtime parameters for '{registry_name}.{entry_name}' "
                        f"are out of date; expected={sorted(constructor_parameters)}, "
                        f"actual={sorted(runtime_parameters)}"
                    )
            elif inspection == "context_parameters":
                component = registry.resolve(entry_name)
                runtime_parameters = set(entry.get("runtime_parameters", []))
                constructor_parameters = set(inspect.signature(component).parameters)
                if runtime_parameters != constructor_parameters:
                    raise ReferenceError(
                        f"runtime parameters for '{registry_name}.{entry_name}' "
                        f"are out of date; expected={sorted(constructor_parameters)}, "
                        f"actual={sorted(runtime_parameters)}"
                    )
                if not parameters and parameter_note is None:
                    raise ReferenceError(
                        f"context-parameter registry entry "
                        f"'{registry_name}.{entry_name}' must document parameters "
                        "or a parameter_note"
                    )
            elif not entry.get("version_dependent", False):
                runtime_parameters = set(entry.get("runtime_parameters", []))
                actual_parameters = _signature_parameters(
                    registry.resolve(entry_name),
                    runtime_parameters=runtime_parameters,
                )
                if set(parameters) != actual_parameters:
                    raise ReferenceError(
                        f"parameters for '{registry_name}.{entry_name}' are out of "
                        f"date; expected={sorted(actual_parameters)}, "
                        f"actual={sorted(parameters)}"
                    )
    return registry_metadata


def _cli_actions() -> dict[str, dict[str, argparse.Action]]:
    parser = build_argument_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    commands: dict[str, dict[str, argparse.Action]] = {}
    for command, command_parser in subparsers.choices.items():
        command_actions: dict[str, argparse.Action] = {}
        for action in command_parser._actions:
            if "--help" in action.option_strings:
                continue
            if action.option_strings:
                label = action.option_strings[-1]
            else:
                label = (
                    action.metavar
                    if isinstance(action.metavar, str)
                    else action.dest
                )
            command_actions[label] = action
        commands[command] = command_actions
    return commands


def _validate_cli(metadata: Mapping[str, Any]) -> dict[str, dict[str, argparse.Action]]:
    actions = _cli_actions()
    cli_metadata = metadata.get("cli")
    if not isinstance(cli_metadata, Mapping):
        raise ReferenceError("reference metadata must define a cli mapping")
    expected = {
        f"{command}.{option}"
        for command, options in actions.items()
        for option in options
    }
    actual = set(cli_metadata)
    if expected != actual:
        raise ReferenceError(
            "CLI metadata is out of date; "
            f"missing={sorted(expected - actual) or '<none>'}, "
            f"stale={sorted(actual - expected) or '<none>'}"
        )
    for path in sorted(expected):
        _required_text(cli_metadata[path], path=f"cli.{path}")
    return actions


def _render_field(
    path: str,
    annotation: Any,
    field_info: Any,
    metadata: Mapping[str, Any],
) -> list[str]:
    description = _required_text(metadata, path=path)
    required, default = _default_value(field_info)
    lines = [
        f"(config-field-{_anchor('path', path)})=",
        f"### `{path}`",
        "",
        description,
        "",
        f"- 类型：`{_type_name(annotation)}`",
        f"- 必填：{'是' if required else '否'}",
    ]
    if not required:
        lines.append(f"- 默认值：`{_yaml_inline(default)}`")
    constraints = metadata.get("constraints")
    if constraints:
        lines.append(f"- 约束：{constraints}")
    interactions = metadata.get("interactions")
    if interactions:
        lines.append(f"- 关联：{interactions}")
    cli_override = metadata.get("cli")
    if cli_override:
        lines.append(f"- CLI 覆盖：`{cli_override}`")
    lines.append("")
    return lines


def _render_registries(registry_metadata: Mapping[str, Any]) -> list[str]:
    lines = ["## Registry 组件索引", ""]
    for registry_name in REGISTRY_NAMES:
        section = registry_metadata[registry_name]
        lines.extend(
            [
                f"(config-registry-{_anchor('name', registry_name)})=",
                f"### `{registry_name}`",
                "",
                _required_text(section, path=f"registries.{registry_name}"),
                "",
                f"- 基类/契约：`{section['base']}`",
                f"- 配置位置：`{section['config_path']}`",
                "",
            ]
        )
        for entry_name, entry in section["entries"].items():
            lines.extend(
                [
                    f"(config-component-{_anchor(registry_name, entry_name)})=",
                    f"#### `{entry_name}`",
                    "",
                    _required_text(
                        entry,
                        path=f"registries.{registry_name}.entries.{entry_name}",
                    ),
                    "",
                ]
            )
            runtime_parameters = entry.get("runtime_parameters", [])
            if runtime_parameters:
                lines.append(
                    "运行时注入（不得在 YAML 中覆盖）："
                    + ", ".join(f"`{name}`" for name in runtime_parameters)
                    + "。"
                )
                lines.append("")
            if entry.get("version_dependent", False):
                lines.extend(
                    [
                        (
                            "> 参数由当前 PyTorch 版本透传；下表只列项目承诺的"
                            "常用参数，最终以所安装 PyTorch 版本为准。"
                        ),
                        "",
                    ]
                )
            parameters = entry.get("parameters", {})
            if parameters:
                lines.extend(["| 参数 | 含义与约束 |", "| --- | --- |"])
                for parameter_name, parameter in parameters.items():
                    description = _required_text(
                        parameter,
                        path=f"{registry_name}.{entry_name}.{parameter_name}",
                    ).replace("|", "\\|")
                    lines.append(f"| `{parameter_name}` | {description} |")
                lines.append("")
            else:
                parameter_note = entry.get("parameter_note")
                if parameter_note is not None:
                    lines.extend([parameter_note.strip(), ""])
                else:
                    lines.extend(["无组件级配置参数。", ""])
    return lines


def _render_cli(
    metadata: Mapping[str, Any],
    actions: Mapping[str, Mapping[str, argparse.Action]],
) -> list[str]:
    lines = ["## CLI 参数索引", ""]
    cli_metadata = metadata["cli"]
    for command, options in actions.items():
        lines.extend(
            [
                f"(config-cli-{command})=",
                f"### `stochaflow {command}`",
                "",
                "| 参数 | 必填 | 默认值 | 含义 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for option, action in options.items():
            description = _required_text(
                cli_metadata[f"{command}.{option}"],
                path=f"cli.{command}.{option}",
            ).replace("|", "\\|")
            default = "—" if action.default is None else _yaml_inline(action.default)
            lines.append(
                f"| `{option}` | {'是' if action.required else '否'} | "
                f"`{default}` | {description} |"
            )
        lines.append("")
    return lines


def render_reference() -> str:
    """Validate all sources and render the complete Markdown reference."""

    metadata = _load_metadata()
    records = _validate_fields(metadata)
    registry_metadata = _validate_registries(metadata)
    cli_actions = _validate_cli(metadata)
    field_metadata = metadata["fields"]

    lines = [
        "<!-- Generated by tools/generate_config_reference.py; do not edit. -->",
        "# 配置字段参考",
        "",
        "本页由运行时 dataclass、Registry、argparse 与中文元数据确定性生成。",
        "修改配置接口后必须运行 `uv run python tools/generate_config_reference.py`。",
        "",
        "字段同时覆盖完整训练配置与独立的 `sample:` 调用配置；二者只共享",
        "`extensions` 插件选择，不合并为一份运行配置。",
        "",
        "字段路径中的 `[]` 表示列表中的每个元素。`params` 是构造关键字参数容器；",
        "框架自有和 Registry 组件的参数见本页后半部分的组件索引。原生依赖 target",
        "遵循当前安装版本的上游 API，不在本地复制完整签名和默认值。",
        "",
    ]
    current_section: str | None = None
    for path, annotation, field_info in records:
        section = path.split(".", 1)[0].removesuffix("[]")
        if section != current_section:
            current_section = section
            lines.extend([f"## `{section}`", ""])
        lines.extend(
            _render_field(path, annotation, field_info, field_metadata[path])
        )
    lines.extend(
        [
            "## 原生依赖 Provider",
            "",
            "标准 PyTorch optimizer 使用 `torch.optim.<Class>`，LR scheduler 使用",
            "`torch.optim.lr_scheduler.<Class>`。这两个受限 namespace 不是任意 Python",
            "class-path importer，也不会复制为 Stochaflow Registry entry；对应前缀保留给",
            "native provider，扩展 Registry 不能占用。",
            "",
            "核心为 optimizer 注入 trainable parameters，为 scheduler 注入 optimizer；",
            "配置 `params` 的其余内容原样传给当前 PyTorch 构造器。具体签名与默认值见",
            "[PyTorch optimizer API](https://docs.pytorch.org/docs/stable/optim.html) 和",
            (
                "[LR scheduler API](https://docs.pytorch.org/docs/stable/optim.html"
                "#how-to-adjust-learning-rate)。"
            ),
            "",
            "`lr_scheduler.interval` 是 Stochaflow 的 step/epoch 生命周期策略。",
            "`T_max`、`total_steps` 等具体构造参数必须显式给出确定值；框架不解释",
            "`auto`，也不根据 epoch、DataLoader 长度或 CLI batch limit 推断它们。",
            "",
        ]
    )
    lines.extend(_render_registries(registry_metadata))
    lines.extend(_render_cli(metadata, cli_actions))
    return "\n".join(lines).rstrip() + "\n"


def _check(expected: str) -> int:
    try:
        actual = OUTPUT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        actual = ""
    if actual == expected:
        print(f"up to date: {OUTPUT_PATH.relative_to(REPO_ROOT)}")
        return 0
    diff = difflib.unified_diff(
        actual.splitlines(),
        expected.splitlines(),
        fromfile=str(OUTPUT_PATH.relative_to(REPO_ROOT)),
        tofile="generated",
        lineterm="",
    )
    print("\n".join(diff))
    print("configuration reference is stale; run the generator")
    return 1


def main() -> int:
    """Generate the reference, or verify that the committed file is current."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed reference differs from generated output.",
    )
    args = parser.parse_args()
    try:
        rendered = render_reference()
    except ReferenceError as exc:
        parser.error(str(exc))
    if args.check:
        return _check(rendered)
    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
