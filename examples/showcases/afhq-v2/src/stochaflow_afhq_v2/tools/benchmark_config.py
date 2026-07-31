"""Resolve the frozen AFHQ-v2 Dog P2 benchmark training variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from stochaflow.utils.config import (
    ConfigError,
    StochaflowConfig,
    load_config_dict,
)

type BenchmarkWeighting = Literal["constant", "p2"]

SHOWCASE_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_CONFIG_ROOT = (
    SHOWCASE_ROOT
    / "experiments"
    / "research"
    / "p2-afhq-v2-dog-256"
)
DEFAULT_BENCHMARK_BASE_CONFIG = BENCHMARK_CONFIG_ROOT / "train-base.yaml"
DEFAULT_P2_OVERRIDE = BENCHMARK_CONFIG_ROOT / "p2-loss-weighting.yaml"

_CONSTANT_WEIGHTING = {"name": "constant"}
_P2_WEIGHTING = {"name": "p2", "k": 1.0, "gamma": 1.0}
_WEIGHTING_PATH = "training.params.loss_weighting"


@dataclass(frozen=True, slots=True)
class BenchmarkConfigSource:
    """Identity of one source file used to resolve a benchmark config."""

    role: Literal["base", "override"]
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class BenchmarkConfigProvenance:
    """Auditable source and effective identities for one benchmark variant."""

    schema_version: int
    variant: BenchmarkWeighting
    sources: tuple[BenchmarkConfigSource, ...]
    changed_paths: tuple[str, ...]
    resolved_config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance without exposing dataclass implementation details."""

        return {
            "schema_version": self.schema_version,
            "variant": self.variant,
            "sources": [
                {
                    "role": source.role,
                    "path": source.path,
                    "sha256": source.sha256,
                }
                for source in self.sources
            ],
            "changed_paths": list(self.changed_paths),
            "resolved_config_sha256": self.resolved_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfigResolution:
    """A validated training config together with its composition provenance."""

    config: StochaflowConfig
    provenance: BenchmarkConfigProvenance


def _load_mapping(path: Path, *, role: str) -> tuple[dict[str, Any], bytes]:
    try:
        source_bytes = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read benchmark {role} config: {path}") from error
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigError(
            f"benchmark {role} config must be valid UTF-8: {path}"
        ) from error
    try:
        raw = yaml.safe_load(source_text) or {}
    except yaml.YAMLError as error:
        raise ConfigError(
            f"benchmark {role} config must contain valid YAML: {path}"
        ) from error
    if not isinstance(raw, dict):
        raise ConfigError(f"benchmark {role} config must contain a mapping")
    if any(not isinstance(key, str) for key in raw):
        raise ConfigError(f"benchmark {role} config keys must be strings")
    return cast(dict[str, Any], raw), source_bytes


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_config_sha256(config: StochaflowConfig) -> str:
    """Return a stable SHA-256 identity for a fully resolved typed config."""

    encoded = json.dumps(
        config.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _source_identity(
    path: Path,
    source_bytes: bytes,
    *,
    role: Literal["base", "override"],
) -> BenchmarkConfigSource:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(SHOWCASE_ROOT.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return BenchmarkConfigSource(
        role=role,
        path=display_path,
        sha256=_sha256_bytes(source_bytes),
    )


def _training_params(raw: dict[str, Any], *, role: str) -> dict[str, Any]:
    training = raw.get("training")
    if not isinstance(training, dict):
        raise ConfigError(f"benchmark {role} must declare training as a mapping")
    params = training.get("params")
    if not isinstance(params, dict):
        raise ConfigError(
            f"benchmark {role} must declare training.params as a mapping"
        )
    return cast(dict[str, Any], params)


def _validate_base_weighting(raw: dict[str, Any]) -> None:
    weighting = _training_params(raw, role="base config").get("loss_weighting")
    if weighting != _CONSTANT_WEIGHTING:
        raise ConfigError(
            "benchmark base config must declare exactly "
            "training.params.loss_weighting: {name: constant}"
        )


def _validate_p2_weighting(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "k", "gamma"}:
        raise ConfigError(
            "benchmark P2 override must declare exactly name, k, and gamma"
        )
    raw = cast(dict[str, object], value)
    if raw["name"] != "p2":
        raise ConfigError("benchmark P2 override loss_weighting.name must be p2")
    for field in ("k", "gamma"):
        number = raw[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or float(number) != 1.0
        ):
            raise ConfigError(
                f"benchmark P2 override loss_weighting.{field} must be 1.0"
            )
    return deepcopy(_P2_WEIGHTING)


def _p2_weighting_from_override(raw: dict[str, Any]) -> dict[str, Any]:
    if set(raw) != {"training"}:
        raise ConfigError(
            f"benchmark override may change only {_WEIGHTING_PATH}"
        )
    training = raw["training"]
    if not isinstance(training, dict) or set(training) != {"params"}:
        raise ConfigError(
            f"benchmark override may change only {_WEIGHTING_PATH}"
        )
    params = training["params"]
    if not isinstance(params, dict) or set(params) != {"loss_weighting"}:
        raise ConfigError(
            f"benchmark override may change only {_WEIGHTING_PATH}"
        )
    return _validate_p2_weighting(params["loss_weighting"])


def resolve_benchmark_training_config(
    base_path: str | Path = DEFAULT_BENCHMARK_BASE_CONFIG,
    *,
    override_path: str | Path | None = None,
) -> BenchmarkConfigResolution:
    """Resolve constant or official-P2 training from one frozen base config.

    The optional override is intentionally not a generic deep merge. It may
    replace only ``training.params.loss_weighting`` with the official
    ``p2(k=1, gamma=1)`` declaration.
    """

    base = Path(base_path)
    base_raw, base_bytes = _load_mapping(base, role="base")
    _validate_base_weighting(base_raw)
    sources = [_source_identity(base, base_bytes, role="base")]
    changed_paths: tuple[str, ...] = ()
    variant: BenchmarkWeighting = "constant"
    resolved_raw = deepcopy(base_raw)

    if override_path is not None:
        override = Path(override_path)
        override_raw, override_bytes = _load_mapping(
            override,
            role="override",
        )
        weighting = _p2_weighting_from_override(override_raw)
        _training_params(resolved_raw, role="resolved config")[
            "loss_weighting"
        ] = weighting
        sources.append(
            _source_identity(override, override_bytes, role="override")
        )
        changed_paths = (_WEIGHTING_PATH,)
        variant = "p2"

    config = load_config_dict(resolved_raw)
    return BenchmarkConfigResolution(
        config=config,
        provenance=BenchmarkConfigProvenance(
            schema_version=1,
            variant=variant,
            sources=tuple(sources),
            changed_paths=changed_paths,
            resolved_config_sha256=canonical_config_sha256(config),
        ),
    )


def resolve_benchmark_variant(
    variant: BenchmarkWeighting,
    *,
    base_path: str | Path = DEFAULT_BENCHMARK_BASE_CONFIG,
    p2_override_path: str | Path = DEFAULT_P2_OVERRIDE,
) -> BenchmarkConfigResolution:
    """Resolve one named benchmark variant through the restricted policy."""

    if variant not in {"constant", "p2"}:
        raise ConfigError("benchmark variant must be constant or p2")
    return resolve_benchmark_training_config(
        base_path,
        override_path=p2_override_path if variant == "p2" else None,
    )


def _resolved_yaml_bytes(resolution: BenchmarkConfigResolution) -> bytes:
    rendered = yaml.safe_dump(
        resolution.config.to_dict(),
        allow_unicode=True,
        sort_keys=False,
    )
    return rendered.encode("utf-8")


def _provenance_json_bytes(resolution: BenchmarkConfigResolution) -> bytes:
    rendered = json.dumps(
        resolution.provenance.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{rendered}\n".encode()


def _stage_bytes(path: Path, value: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def write_benchmark_resolution(
    resolution: BenchmarkConfigResolution,
    *,
    output_path: str | Path,
    provenance_path: str | Path,
) -> None:
    """Publish a staged resolved YAML and provenance sidecar without clobbering.

    Existing targets are rejected so a failed or repeated invocation cannot
    silently replace a previously audited resolution.
    """

    output = Path(output_path).resolve()
    provenance = Path(provenance_path).resolve()
    if output == provenance:
        raise ValueError(
            "resolved config and provenance must use different output paths"
        )
    targets = (output, provenance)
    existing = [path for path in targets if path.exists()]
    if existing:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"benchmark output already exists: {listed}")
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        staged.append((_stage_bytes(output, _resolved_yaml_bytes(resolution)), output))
        staged.append(
            (
                _stage_bytes(
                    provenance,
                    _provenance_json_bytes(resolution),
                ),
                provenance,
            )
        )
        if any(target.exists() for _, target in staged):
            raise FileExistsError(
                "benchmark output appeared while artifacts were staged"
            )
        for temporary, target in staged:
            # A hard-link commit is atomic and fails when the destination
            # appears after the preflight check. Unlike replace(), it cannot
            # overwrite a concurrently created, previously audited artifact.
            os.link(temporary, target)
            committed.append(target)
            temporary.unlink()
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for target in committed:
            target.unlink(missing_ok=True)
        raise


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve one frozen AFHQ-v2 Dog benchmark variant into a plain "
            "Stochaflow train config and provenance sidecar."
        )
    )
    parser.add_argument(
        "--variant",
        choices=("constant", "p2"),
        required=True,
        help="Select the only permitted training variation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New path for the fully resolved training YAML.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="New path for the composition provenance JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve and write one directly trainable benchmark configuration."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        resolution = resolve_benchmark_variant(
            cast(BenchmarkWeighting, arguments.variant),
        )
        write_benchmark_resolution(
            resolution,
            output_path=cast(Path, arguments.output),
            provenance_path=cast(Path, arguments.provenance),
        )
    except (ConfigError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"resolved_config: {Path(arguments.output).resolve()}")
    print(f"provenance: {Path(arguments.provenance).resolve()}")
    print(
        "resolved_config_sha256: "
        f"{resolution.provenance.resolved_config_sha256}"
    )
    return 0


__all__ = [
    "BENCHMARK_CONFIG_ROOT",
    "DEFAULT_BENCHMARK_BASE_CONFIG",
    "DEFAULT_P2_OVERRIDE",
    "BenchmarkConfigProvenance",
    "BenchmarkConfigResolution",
    "BenchmarkConfigSource",
    "BenchmarkWeighting",
    "canonical_config_sha256",
    "main",
    "resolve_benchmark_training_config",
    "resolve_benchmark_variant",
    "write_benchmark_resolution",
]


if __name__ == "__main__":
    raise SystemExit(main())
