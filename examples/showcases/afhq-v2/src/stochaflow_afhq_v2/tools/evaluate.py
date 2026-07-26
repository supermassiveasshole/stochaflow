"""Evaluate an AFHQ-v2 checkpoint with a frozen class-aware quality protocol."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from stochaflow.utils.plugins import ExtensionVersionPolicy
from stochaflow_afhq_v2.tools.evaluation import evaluate_checkpoint


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the standalone AFHQ-v2 evaluation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--force-extension-version-mismatch",
        action="store_true",
        help=(
            "Accept an installed extension version mismatch after identity "
            "validation; checkpoint and data contracts remain strict."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the frozen evaluation protocol and print its immutable result."""

    args = build_argument_parser().parse_args(argv)
    force = args.force_extension_version_mismatch
    result = evaluate_checkpoint(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        extension_version_policy=(
            ExtensionVersionPolicy.ALLOW
            if force
            else ExtensionVersionPolicy.REJECT
        ),
        extension_acceptance_method=(
            "force-flag" if force else None
        ),
    )
    print(f"Output: {result.output_dir}")
    print(f"Result: {result.result_path}")
    print(f"Result SHA-256: {result.result_sha256}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()


__all__ = ["build_argument_parser", "main"]
