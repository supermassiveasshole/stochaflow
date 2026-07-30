"""Create plain teacher and logit-calibrator bootstrap states."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stochaflow_knowledge_distillation.stochaflow_ext.models import (
    LogitCalibrator,
    TeacherClassifier,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-output",
        "--output",
        dest="teacher_output",
        type=Path,
        default=Path("data/teacher.pt"),
    )
    parser.add_argument(
        "--calibrator-output",
        type=Path,
        default=Path("data/calibrator.pt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-features", type=int, default=8)
    parser.add_argument("--hidden-features", type=int, default=24)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--calibrator-scale", type=float, default=0.0)
    parser.add_argument(
        "--calibrator-bias",
        type=float,
        nargs="+",
        default=None,
        help="Per-class bias; defaults to 0, 1, ..., num_classes - 1.",
    )
    return parser


def main() -> None:
    """Write deterministic plain teacher and calibrator state mappings."""

    parser = _parser()
    args = parser.parse_args()
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        teacher = TeacherClassifier(
            input_features=args.input_features,
            hidden_features=args.hidden_features,
            num_classes=args.num_classes,
        )
    calibrator = LogitCalibrator(num_classes=args.num_classes)
    bias = (
        list(map(float, range(args.num_classes)))
        if args.calibrator_bias is None
        else args.calibrator_bias
    )
    if len(bias) != args.num_classes:
        parser.error(
            "--calibrator-bias must provide exactly "
            f"{args.num_classes} values"
        )
    with torch.no_grad():
        calibrator.scale.fill_(args.calibrator_scale)
        calibrator.bias.copy_(
            torch.tensor(bias, dtype=calibrator.bias.dtype)
        )

    args.teacher_output.parent.mkdir(parents=True, exist_ok=True)
    args.calibrator_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(teacher.state_dict(), args.teacher_output)
    torch.save(calibrator.state_dict(), args.calibrator_output)
    print(args.teacher_output)
    print(args.calibrator_output)


if __name__ == "__main__":
    main()
