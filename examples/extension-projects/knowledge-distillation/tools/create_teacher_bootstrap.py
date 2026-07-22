"""Create the plain model state used to initialize a fresh teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stochaflow_knowledge_distillation.stochaflow_ext.models import (
    TeacherClassifier,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/teacher.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-features", type=int, default=8)
    parser.add_argument("--hidden-features", type=int, default=24)
    parser.add_argument("--num-classes", type=int, default=4)
    return parser


def main() -> None:
    """Write a deterministic, plain teacher ``state_dict``."""

    args = _parser().parse_args()
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        model = TeacherClassifier(
            input_features=args.input_features,
            hidden_features=args.hidden_features,
            num_classes=args.num_classes,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
