"""Immutable result artifacts for AFHQ-v2 quality evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from stochaflow.data import DataArtifactBindings
from stochaflow.sampling.runtime import (
    ResolvedSamplingInputs,
    SamplingRunResult,
)
from stochaflow.utils.plugins import ResolvedExtensions
from stochaflow.utils.run_manifest import extension_runtime_metadata
from stochaflow_afhq_v2.tools.evaluation_config import (
    SCHEMA_VERSION,
    AFHQV2EvaluationDocument,
    sampling_parameters,
)

RESULT_NAME = "evaluation-result.json"
RESULT_DIGEST_NAME = "evaluation-result.sha256"
MANIFEST_NAME = "evaluation-manifest.json"
SAMPLE_REQUEST_NAME = "sample-request.yaml"


@dataclass(frozen=True, slots=True)
class AFHQV2EvaluationResult:
    """Paths produced after an immutable evaluation result is committed."""

    output_dir: Path
    result_path: Path
    result_sha256: str
    digest_path: Path
    manifest_path: Path
    sampling: SamplingRunResult


def sha256_file(path: Path) -> str:
    """Hash one stable file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path) -> dict[str, Any]:
    """Return path, size, and digest for one result dependency."""

    resolved = path.resolve()
    relative = result_path_text(resolved, root=root)
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def result_path_text(path: Path, *, root: Path) -> str:
    """Prefer a publish-stable path relative to the evaluation root."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_exclusive(path: Path, encoded: bytes) -> None:
    """Create one immutable-by-convention file and refuse replacement."""

    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def default_output_dir(checkpoint_path: Path) -> Path:
    """Choose a collision-free sibling evaluation directory."""

    timestamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    root = checkpoint_path.parent.parent / "evaluations"
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dependency_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": _distribution_version("torchvision"),
        "stochaflow": _distribution_version("stochaflow"),
        "stochaflow-afhq-v2": _distribution_version("stochaflow-afhq-v2"),
        "torchmetrics": _distribution_version("torchmetrics"),
        "torch-fidelity": _distribution_version("torch-fidelity"),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def materialize_result(
    *,
    root: Path,
    document: AFHQV2EvaluationDocument,
    inputs: ResolvedSamplingInputs,
    extensions: ResolvedExtensions,
    sampling: SamplingRunResult,
    expected_bindings: DataArtifactBindings,
    real_counts: Mapping[str, int],
    fake_counts: Mapping[str, int],
    metrics: Mapping[str, Any],
    provider_identities: Mapping[str, str],
    checkpoint_sha256: str,
    checkpoint_progress: Mapping[str, int],
    request_path: Path,
) -> tuple[Path, str, Path, Path]:
    """Commit result JSON, digest sidecar, and immutable result manifest."""

    sampling_artifacts = {
        name: file_record(path, root=root)
        for name, path in sorted(sampling.artifacts.items())
    }
    protocol = document.protocol
    params = sampling_parameters(document)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "afhq-v2-class-aware-quality-evaluation",
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": {
            "path": str(inputs.checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "format_version": inputs.checkpoint.get("format_version"),
            "epoch": checkpoint_progress["epoch"],
            "global_step": checkpoint_progress["global_step"],
            "weights": sampling.metadata["weights"],
        },
        "config": {
            "source_path": str(document.source_path),
            "source_sha256": document.source_sha256,
            "sample_request": file_record(request_path, root=root),
        },
        "extensions": extension_runtime_metadata(extensions),
        "protocol": {
            "split": protocol.split,
            "class_mapping": dict(protocol.class_mapping),
            "allocation": {
                "real": dict(real_counts),
                "fake": dict(fake_counts),
            },
            "selection": {
                "real": "authenticated-manifest-order",
                "fake": "ordered-class-label-blocks",
            },
            "sampling_seed": sampling.seed,
            "metric_seed": protocol.metric_seed,
            "metric_batch_size": protocol.metric_batch_size,
            "guidance_scale": params["guidance_scale"],
            "sampler": params["sampler"],
            "metrics": [
                {"name": spec.name, "params": dict(spec.params)}
                for spec in protocol.metrics
            ],
            "scopes": ["aggregate", "per_class"],
        },
        "data": {
            "builder": extensions.config.data.name,
            "source": "afhq-v2.official",
            "artifact_bindings": expected_bindings.to_dict(),
        },
        "sampling": {
            "recipe": sampling.recipe_name,
            "device": str(sampling.device),
            "metadata": dict(sampling.metadata),
            "output_dir": result_path_text(sampling.output_dir, root=root),
            "artifacts": sampling_artifacts,
        },
        "metrics": dict(metrics),
        "metric_providers": dict(provider_identities),
        "dependencies": _dependency_versions(),
    }
    result_path = root / RESULT_NAME
    encoded = _json_bytes(result)
    digest = hashlib.sha256(encoded).hexdigest()
    write_exclusive(result_path, encoded)
    digest_path = root / RESULT_DIGEST_NAME
    write_exclusive(
        digest_path,
        f"{digest}  {RESULT_NAME}\n".encode("ascii"),
    )
    manifest_path = root / MANIFEST_NAME
    write_exclusive(
        manifest_path,
        _json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "immutable-result-manifest",
                "result": {
                    "path": RESULT_NAME,
                    "bytes": len(encoded),
                    "sha256": digest,
                },
                "digest_file": RESULT_DIGEST_NAME,
            }
        ),
    )
    return result_path, digest, digest_path, manifest_path


__all__ = [
    "MANIFEST_NAME",
    "RESULT_DIGEST_NAME",
    "RESULT_NAME",
    "SAMPLE_REQUEST_NAME",
    "AFHQV2EvaluationResult",
    "default_output_dir",
    "file_record",
    "materialize_result",
    "result_path_text",
    "sha256_file",
    "write_exclusive",
]
