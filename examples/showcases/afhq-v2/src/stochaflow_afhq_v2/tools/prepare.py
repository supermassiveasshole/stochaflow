"""Prepare or fully verify the source-locked AFHQ-v2 artifact."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from stochaflow.extensions import (
    IMAGE_DATA_SOURCES,
    ArtifactVerificationObserver,
    ClassLabeledImageFolderArtifactPayload,
    DataSourceContext,
    ImageFolderArtifactPayload,
)
from stochaflow.scripts.artifact_reporting import (
    RichArtifactVerificationReporter,
)
from stochaflow_afhq_v2.artifact import (
    AFHQV2_SOURCE_NAME,
)
from stochaflow_afhq_v2.dog_artifact import AFHQV2_DOG_SOURCE_NAME


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone preparation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".stochaflow-cache"),
    )
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--lock-file", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=("official", "dog"),
        default="official",
        help="Prepare the official labeled artifact or the 256px Dog benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Defaults to 128 for official and 256 for dog.",
    )
    parser.add_argument(
        "--downloader",
        choices=("auto", "curl", "python"),
        default="auto",
        help="Select the official archive downloader for policy=ensure.",
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
    parser.add_argument(
        "--artifact-verification-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Artifact hashing workers (1-8); defaults to min(8, logical CPUs)."
        ),
    )
    progress = parser.add_mutually_exclusive_group()
    progress.add_argument(
        "--progress",
        action="store_true",
        dest="progress",
        help="Force artifact verification progress on stderr.",
    )
    progress.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable artifact verification progress.",
    )
    parser.set_defaults(progress=None)
    return parser


def prepare_artifact(
    *,
    cache_root: Path,
    archive: Path | None,
    lock_file: Path | None,
    resolution: int,
    downloader: Literal["auto", "curl", "python"],
    policy: Literal["ensure", "require"],
    verification: Literal["manifest", "full"],
    verification_observer: ArtifactVerificationObserver | None = None,
    verification_workers: int | None = None,
    profile: Literal["official", "dog"] = "official",
) -> dict[str, Any]:
    """Materialize through the registered source used by the DataBuilder."""

    if profile not in {"official", "dog"}:
        raise ValueError("profile must be official or dog")
    source_name = (
        AFHQV2_SOURCE_NAME
        if profile == "official"
        else AFHQV2_DOG_SOURCE_NAME
    )
    source_params: dict[str, Any] = {
        "downloader": downloader,
        "resolution": resolution,
    }
    if archive is not None:
        source_params["archive"] = str(archive)
    if lock_file is not None:
        source_params["lock_file"] = str(lock_file)
    importlib.import_module("stochaflow_afhq_v2.stochaflow_ext.source")
    source = IMAGE_DATA_SOURCES.create(
        source_name,
        source_params,
        config_path="prepare",
    )
    artifact = source.materialize(
        DataSourceContext(
            cache_root=cache_root,
            policy=policy,
            verification=verification,
            verification_observer=verification_observer,
            verification_workers=verification_workers,
        )
    )
    payload = artifact.payload
    if profile == "official" and not isinstance(
        payload,
        ClassLabeledImageFolderArtifactPayload,
    ):
        raise TypeError(
            "registered AFHQ-v2 source must return a class-labeled "
            "image-folder payload"
        )
    if profile == "dog" and not isinstance(
        payload,
        ImageFolderArtifactPayload,
    ):
        raise TypeError(
            "registered AFHQ-v2 Dog source must return an unlabeled "
            "image-folder payload"
        )
    if not isinstance(
        payload,
        (ClassLabeledImageFolderArtifactPayload, ImageFolderArtifactPayload),
    ):
        raise TypeError("AFHQ-v2 payload validation was not exhaustive")
    if profile == "official" and (
        payload.validation is not None or payload.test is None
    ):
        raise ValueError(
            "registered AFHQ-v2 source must expose official train/test only"
        )
    if profile == "dog" and (
        payload.validation is not None or payload.test is not None
    ):
        raise ValueError(
            "registered AFHQ-v2 Dog source must expose train/dog only"
        )
    summary: dict[str, Any] = {
        "root": str(artifact.root),
        "manifest_path": str(artifact.manifest_path),
        "identity": artifact.identity.to_dict(),
    }
    if isinstance(payload, ClassLabeledImageFolderArtifactPayload):
        assert payload.test is not None
        summary["class_mapping"] = dict(payload.class_mapping)
        summary["counts"] = {
            "train": len(payload.train),
            "test": len(payload.test),
        }
    else:
        summary["counts"] = {"train": len(payload.train)}
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare/verify AFHQ-v2 and print a machine-readable summary."""

    args = build_argument_parser().parse_args(argv)
    resolution = args.resolution
    if resolution is None:
        resolution = 256 if args.profile == "dog" else 128
    show_progress = (
        sys.stderr.isatty()
        if args.progress is None
        else bool(args.progress)
    )
    reporter = RichArtifactVerificationReporter() if show_progress else None
    try:
        summary = prepare_artifact(
            cache_root=args.cache_root,
            archive=args.archive,
            lock_file=args.lock_file,
            resolution=resolution,
            downloader=args.downloader,
            policy=args.policy,
            verification=args.verification,
            verification_observer=(
                reporter.observe if reporter is not None else None
            ),
            verification_workers=args.artifact_verification_workers,
            profile=args.profile,
        )
    finally:
        if reporter is not None:
            reporter.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_argument_parser", "main", "prepare_artifact"]
