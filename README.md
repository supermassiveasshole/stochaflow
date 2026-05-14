# Stochaflow

Stochaflow is a research-oriented Python project scaffold for **stochastic flows**, with an initial repository layout that is convenient for diffusion-family methods such as DDPM and DDIM, but is not limited to diffusion models.

The current repository is intentionally a skeleton:

- the package layout is in place
- experiment configs and entry scripts are stubbed out
- implementation files are placeholders
- the core training, sampling, and evaluation logic is still to be written

## Scope

This project is meant to host work on stochastic flow models, including but not limited to:

- diffusion probabilistic models
- deterministic or stochastic samplers
- score-based generative modeling components
- flow-style training and sampling utilities

In other words, the existing `diffusion/` package reflects the first planned implementation track, not the full conceptual boundary of the repository.

## Status

This is an **initial scaffold commit**, not a finished library.

What is already included:

- `src`-layout Python package structure
- module boundaries for data, models, diffusion, training, sampling, and utilities
- starter configuration files under `configs/`
- placeholder scripts under `scripts/`
- placeholder tests and notebook slots
- output and asset directories tracked with `.gitkeep`

What is not included yet:

- working model implementations
- real dataset pipelines
- training loops
- checkpointing logic
- sampling and evaluation pipelines

## Repository Layout

```text
stochaflow/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── ddpm_mnist.yaml
│   ├── ddpm_cifar10.yaml
│   └── ddim_cifar10.yaml
├── scripts/
│   ├── train.py
│   ├── sample.py
│   └── eval.py
├── src/
│   └── stochaflow/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── datasets.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── unet.py
│       │   ├── blocks.py
│       │   └── embeddings.py
│       ├── diffusion/
│       │   ├── __init__.py
│       │   ├── schedules.py
│       │   ├── ddpm.py
│       │   ├── ddim.py
│       │   └── objectives.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── ema.py
│       ├── sampling/
│       │   ├── __init__.py
│       │   ├── sampler.py
│       │   └── grid.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── seed.py
│           ├── checkpoint.py
│           └── logging.py
├── tests/
│   ├── test_schedules.py
│   ├── test_ddpm_shapes.py
│   └── test_unet_shapes.py
├── notebooks/
│   └── ddpm_sanity_check.ipynb
├── outputs/
│   └── .gitkeep
└── assets/
    └── .gitkeep
```

## Package Structure

`src/stochaflow/data/`
: dataset definitions, transforms, dataloaders, and dataset-specific helpers.

`src/stochaflow/models/`
: model backbones and reusable building blocks.

`src/stochaflow/diffusion/`
: diffusion-process-specific schedules, objectives, and algorithms.

`src/stochaflow/training/`
: trainer orchestration, losses, optimization helpers, and EMA utilities.

`src/stochaflow/sampling/`
: sampling routines, sample post-processing, and grid/export helpers.

`src/stochaflow/utils/`
: shared infrastructure such as config loading, seeding, checkpoint paths, and logging.

## Configuration

The repository currently includes placeholder experiment configs:

- `configs/ddpm_mnist.yaml`
- `configs/ddpm_cifar10.yaml`
- `configs/ddim_cifar10.yaml`

These are starter files for future experiments. They should be treated as templates rather than final experiment definitions.

## Development Setup

The project uses a `src` layout and `pyproject.toml` packaging metadata.

Example local setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Notes:

- this repository currently targets Python `>=3.12`
- Intel macOS uses a constrained PyTorch dependency path for wheel compatibility
- newer Python versions may be used on other platforms as long as dependency resolution succeeds

## Roadmap

A sensible implementation order for this scaffold is:

1. Define config loading and experiment schema.
2. Implement dataset loading and preprocessing.
3. Implement model backbones and embeddings.
4. Implement schedules and stochastic flow / diffusion core logic.
5. Add training orchestration and checkpointing.
6. Add sampling, evaluation, and tests.

## License

This project is released under the [MIT License](LICENSE).
