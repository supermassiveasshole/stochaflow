# Repository Guidelines

## Project Structure & Module Organization

Stochaflow is a Python 3.12+ package using a `src` layout. Core code lives in
`src/stochaflow/`: diffusion processes in `diffusion/`, neural networks in
`models/`, training orchestration in `training/`, sampling helpers in
`sampling/`, shared infrastructure in `utils/`, and console entry points in
`scripts/`. Dataset builders are in `data/`. Experiment YAML files belong in
`configs/`; tests in `tests/` should mirror the feature they cover. Keep
documentation and research notes in `docs/` or `notebooks/`, and reserve
`assets/readme/` for curated documentation images. Generated runs belong under
`outputs/` and must not be committed.

## Build, Test, and Development Commands

- `uv sync --extra dev` creates or updates the locked development environment.
- `uv run pytest` runs the complete test suite; pass a file path for a focused run.
- `uv run ruff check .` checks formatting-independent style and common errors.
- `uv run pyright` performs basic static type checking across `src/` and `tests/`.
- `uv build` creates wheel and source distributions in `dist/`.
- `uv run stochaflow-train-mnist-ddpm --config configs/ddpm_mnist.yaml --epochs 1 --limit-batches 10` performs a short end-to-end smoke run.

## Coding Style & Naming Conventions

Use four-space indentation, Ruff’s default 88-character line length, and type
annotations for public APIs. Prefer small, single-purpose modules and explicit
imports. Name modules, functions, and variables `snake_case`; classes
`PascalCase`; constants `UPPER_SNAKE_CASE`. Add concise docstrings to public
classes and non-obvious methods. Register new configurable components through
the existing registry/factory pattern rather than adding command-specific
branches.

## Architecture & Open-Closed Principle

Treat the Open-Closed Principle as a repository-level design constraint. Core
runtime code coordinates stable lifecycles; task and algorithm variation belongs
in registered components. A compatible DataBuilder, Process, Sampler,
SamplingBuilder, model, Loss, or TrainingStrategy must be addable through a new
implementation, registration, configuration, and tests without editing core
dispatch or adding branches keyed by a registered name or concrete class.

Keep public contracts minimal and compositional. Do not widen universal base
classes with task-specific fields, large sets of optional methods, or mode enums.
Use narrow capability protocols and validate compatibility at the Strategy or
Builder boundary. In particular:

- DataBuilder is the only core data extension entrypoint. It directly assembles
  ordinary Dataset, split, transform, PyTorch sampler, collate, and DataLoader
  objects and returns the ready train/validation/test iterables. Core code treats
  batches as structured `Any` and must not impose image, condition, target, sample-key,
  bucket, or metadata fields. Do not create universal Dataset/Sampler/DataLoader
  registries or schemas; optional source adapters may exist only as helpers for
  a specific built-in recipe.
- Process describes a model-free probability path and its mathematical
  capabilities. It must not own a task model, interpret a training batch, or run
  a sampling loop.
- Generative dynamics are composed from a Process and model callable through
  narrow protocols; do not introduce a global dynamics registry without a
  demonstrated cross-component need.
- Sampler owns the complete numerical sampling algorithm and its ephemeral
  solver state. SamplingBuilder owns task composition, including model adapters,
  conditioning, guidance, initialization, and Process/Sampler compatibility.
- Built-in components use the same registry and construction paths as external
  extensions; do not add hidden core-only shortcuts.

When a feature appears to require task-specific branching in a runner, a common
config schema field used by only one modality, or changes to an existing
compatible component, stop and revise the extension boundary and implementation
plan before coding.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and functions named `test_*`. Add focused
unit tests for shapes, configuration validation, and error cases; add runner or
script tests when changing CLI behavior. No coverage threshold is configured,
so every behavior change should include a regression test. Avoid full training
runs in tests; use tiny tensors, fixtures, and limited batches.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, sentence-case subjects such as
`Patch UNet bugs` and `Improve MNIST DDPM training UX`. Keep each commit scoped
to one logical change. Pull requests should explain the motivation and user
impact, list verification commands, link relevant issues, and call out config
or dependency changes. Include sample images or metric excerpts when generated
output changes, but do not commit checkpoints, datasets, secrets, or routine
run artifacts.
