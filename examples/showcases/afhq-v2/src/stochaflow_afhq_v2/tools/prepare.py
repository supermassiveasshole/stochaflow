"""Prepare or fully verify the source-locked AFHQ-v2 artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from stochaflow.extensions import IMAGE_DATA_SOURCES, DataSourceContext
from stochaflow_afhq_v2.stochaflow_ext.data import (
    AFHQV2ImageFolderArtifactPayload,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone preparation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data"))
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--validation-per-class", type=int, default=300)
    parser.add_argument(
        "--validation-seed",
        default="stochaflow-afhq-v2-validation-v1",
    )
    parser.add_argument(
        "--policy",
        choices=("ensure", "require"),
        default="ensure",
        help="'ensure' may acquire/prepare; 'require' only verifies the cache.",
    )
    parser.add_argument(
        "--verification",
        choices=("manifest", "full"),
        default="full",
    )
    return parser


def prepare_artifact(
    *,
    cache_root: Path,
    archive: Path | None,
    lock_file: Path | None,
    resolution: int,
    validation_per_class: int,
    validation_seed: str,
    policy: Literal["ensure", "require"],
    verification: Literal["manifest", "full"],
) -> dict[str, Any]:
    """Materialize through the registered source lifecycle and summarize it."""

    params: dict[str, Any] = {
        "resolution": resolution,
        "validation_per_class": validation_per_class,
        "validation_seed": validation_seed,
    }
    if archive is not None:
        params["archive"] = str(archive)
    if lock_file is not None:
        params["lock_file"] = str(lock_file)
    source = IMAGE_DATA_SOURCES.create(
        "afhq-v2.official",
        params,
        config_path="afhq-v2.prepare.source",
    )
    artifact = source.materialize(
        DataSourceContext(
            cache_root=cache_root,
            policy=policy,
            verification=verification,
        )
    )
    payload = artifact.payload
    if not isinstance(payload, AFHQV2ImageFolderArtifactPayload):
        raise TypeError("AFHQ-v2 source returned an incompatible payload")
    return {
        "artifact_root": str(artifact.artifact_root),
        "manifest_path": str(artifact.manifest_path),
        "identity": artifact.identity.to_dict(),
        "class_mapping": dict(payload.class_mapping),
        "counts": {
            "train": len(payload.train),
            "validation": len(payload.validation or ()),
            "test": len(payload.test or ()),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare/verify AFHQ-v2 and print a machine-readable summary."""

    args = build_argument_parser().parse_args(argv)
    summary = prepare_artifact(
        cache_root=args.cache_root,
        archive=args.archive,
        lock_file=args.lock_file,
        resolution=args.resolution,
        validation_per_class=args.validation_per_class,
        validation_seed=args.validation_seed,
        policy=args.policy,
        verification=args.verification,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_argument_parser", "main", "prepare_artifact"]
