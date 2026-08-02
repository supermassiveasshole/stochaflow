"""Unified Stochaflow command-line interface."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from stochaflow.evaluation import run_evaluation
from stochaflow.projects import ProjectScaffoldError, create_project
from stochaflow.sampling.runtime import (
    resolve_sampling_inputs,
    run_resolved_sampling,
)
from stochaflow.scripts.branding import print_ascii_art_logo
from stochaflow.scripts.experiment_runner import (
    add_training_arguments,
    run_experiment_from_args,
)
from stochaflow.scripts.extensions_cli import activate_extensions_for_cli


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the top-level init/train/sample/evaluate command parser."""

    parser = argparse.ArgumentParser(prog="stochaflow", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Run a config-driven training experiment.",
    )
    add_training_arguments(train_parser)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Run checkpoint-backed inference from a Stochaflow checkpoint.",
    )
    sample_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Required complete sample invocation config and optional additive "
            "inference plugin selection."
        ),
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run one standalone checkpoint evaluation protocol.",
    )
    evaluate_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Required complete standalone evaluation authority.",
    )
    evaluate_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the evaluation device.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New directory for immutable evaluation results.",
    )
    evaluate_parser.add_argument(
        "--force-extension-version-mismatch",
        action="store_true",
        help=(
            "Accept extension version differences after identity validation; "
            "does not bypass checkpoint state compatibility."
        ),
    )
    sample_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Required checkpoint file or run directory.",
    )
    sample_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the sampling device.",
    )
    sample_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated artifacts.",
    )
    sample_parser.add_argument(
        "--force-extension-version-mismatch",
        action="store_true",
        help=(
            "Accept extension version differences after identity validation; "
            "does not bypass checkpoint state compatibility."
        ),
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Create an installable Stochaflow extension project.",
    )
    init_parser.add_argument(
        "name",
        metavar="NAME",
        help="Canonical project slug, for example my-research-project.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the selected Stochaflow subcommand."""

    print_ascii_art_logo()
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        run_experiment_from_args(args)
        return
    if args.command == "init":
        try:
            project_path = create_project(args.name)
        except ProjectScaffoldError as exc:
            parser.error(str(exc))
        print(f"Created project: {project_path}")
        return
    if args.command == "evaluate":
        outcome = run_evaluation(
            args.config,
            output_dir=args.output_dir,
            device_name=args.device,
            force_extension_version_mismatch=(
                args.force_extension_version_mismatch
            ),
        )
        print(f"Evaluation: {outcome.evaluation_id}")
        print(f"Protocol: {outcome.protocol_id}")
        print(f"Status: {outcome.status}")
        print(f"Output: {outcome.output_dir}")
        print(f"Result: {outcome.result_path}")
        print(f"Manifest: {outcome.manifest_path}")
        for name, value in outcome.metrics.items():
            print(f"{name}: {value}")
        for name, value in outcome.measurements.items():
            print(f"{name}: {value}")
        return
    if args.checkpoint is None:
        parser.error("sample requires --checkpoint")

    startup_cwd = Path.cwd()
    inputs = resolve_sampling_inputs(
        config_path=args.config,
        checkpoint=args.checkpoint,
    )
    extensions = activate_extensions_for_cli(
        inputs.extension_plan,
        force_version_mismatch=args.force_extension_version_mismatch,
    )
    result = run_resolved_sampling(
        inputs,
        extensions,
        output_dir=args.output_dir,
        device_name=args.device,
        startup_cwd=startup_cwd,
    )
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Recipe: {result.recipe_name}")
    print(f"Device: {result.device}")
    print(f"Seed: {result.seed}")
    print(f"Weights: {result.metadata.get('weights', 'recipe-defined')}")
    print(f"Output: {result.output_dir}")
    for name, path in result.artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
