# Repository Guidelines

## Project Structure & Module Organization

Stochaflow is a Python 3.12+ package using a `src` layout. Core code lives in
`src/stochaflow/`: probability processes in `processes/`, neural networks in
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

## Architecture & SOLID Principles

Treat all five SOLID principles as repository-level design constraints, not
only as class-level style guidance:

- **Single Responsibility:** each public component owns one cohesive policy and
  has one primary reason to change. Keep probability math, model adaptation,
  numerical solving, task composition, persistence, and artifact I/O in their
  respective layers. A component may parse and validate its private parameters,
  but it must not absorb unrelated runtime orchestration.
- **Open-Closed:** core runtime code coordinates stable lifecycles; task and
  algorithm variation belongs in registered components. A compatible
  DataBuilder, Process, Sampler, SamplingBuilder, model, Loss, or
  TrainingStrategy must be addable through implementation, registration,
  configuration, and tests without editing core dispatch or adding branches
  keyed by a registered name or concrete class.
- **Liskov Substitution:** implementations of a public base class or capability
  must preserve its documented inputs, outputs, invariants, state semantics, and
  error guarantees. Do not imply support for mutable or learnable state,
  arbitrary batch types, devices, or time domains unless every consumer of the
  contract supports it. Test important contracts with an independent custom
  implementation, not only built-in subclasses.
- **Interface Segregation:** keep public contracts minimal and compositional.
  Prefer narrow capability protocols used by one collaboration over universal
  base classes with task-specific fields, large sets of optional methods, mode
  enums, or methods irrelevant to some implementations. A generative method is
  not required to use every framework role.
- **Dependency Inversion:** runners, trainers, strategies, and builders depend on
  public contracts and capability protocols rather than built-in algorithms,
  concrete storage classes, or registered names. Construct implementations at
  registry/factory composition boundaries and inject them into mathematical and
  runtime code. Concrete-class checks are allowed only when the concrete type is
  itself the explicitly documented contract; otherwise define or reuse a narrow
  capability.

Validate cross-component compatibility at the Strategy or Builder boundary,
where the complete task composition is known. In particular:

- DataBuilder is the only core data extension entrypoint. It directly assembles
  ordinary Dataset, split, transform, PyTorch sampler, collate, and DataLoader
  objects and returns the ready train/validation/test iterables. Core code treats
  batches as structured `Any` and must not impose image, condition, target, sample-key,
  bucket, or metadata fields. Do not create universal Dataset/Sampler/DataLoader
  registries or schemas; optional source adapters may exist only as helpers for
  a specific built-in recipe.
- Process describes a model-free probability path and its mathematical
  capabilities. It must not own a task model, interpret a training batch, or run
  a sampling loop. The `Process` root is a registry and lifecycle boundary, not
  a universal mathematical API; algorithm families define cohesive Process
  subclasses for their own needs. Do not force methods without a model-free
  probability path to invent a Process solely to satisfy core dispatch.
- A Gaussian noise schedule is a Process-owned, Gaussian-specific composition
  point. It must not become a universal schedule abstraction for interpolants,
  solver time grids, or learning-rate policies. Processes depend on schedule
  capabilities rather than a concrete coefficient-table implementation, and
  derived coefficients must remain consistent with any mutable schedule state.
- `GenerativeDynamics` is only a semantic root for an assembled generation
  direction. Algorithm families define narrow Dynamics contracts such as a
  Gaussian denoising prediction, vector field, reverse SDE, or denoiser
  function. Do not add universal `predict`, `step`, `drift`, `score`, or
  `denoise` methods to the root, and do not introduce a global Dynamics registry
  without a demonstrated cross-component need.
- Sampler owns the complete numerical sampling algorithm and its ephemeral
  solver state. SamplingBuilder owns task composition, including model adapters,
  conditioning, guidance, initialization, and family-level
  Process/Dynamics/Sampler compatibility. A family-specific Sampler may require
  its narrow Dynamics capability at the call boundary; core code must not
  maintain a global name-based compatibility matrix.
  Do not force methods with a direct exact generation transform to invent a
  numerical Sampler.
- Built-in components use the same registry and construction paths as external
  extensions; do not add hidden core-only shortcuts.

When a feature appears to require task-specific branching in a runner, a common
config schema field used by only one modality, a dependency on a concrete
implementation where a capability would suffice, or changes to an existing
compatible component, stop and revise the responsibility or extension boundary
before coding.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and functions named `test_*`. Add focused
unit tests for shapes, configuration validation, and error cases; add runner or
script tests when changing CLI behavior. No coverage threshold is configured,
so every behavior change should include a regression test. Avoid full training
runs in tests; use tiny tensors, fixtures, and limited batches.

For routine incremental changes, repository-wide validation requires only
`uv run ruff check .` and `uv run pyright`, plus focused tests for changed
behavior. Defer additional static analyzers and full acceptance checks until a
complete feature branch is ready to merge; before merging, run and fix the full
verification suite required by that feature and CI.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, sentence-case subjects such as
`Patch UNet bugs` and `Improve MNIST DDPM training UX`. Keep each commit scoped
to one logical change. Pull requests should explain the motivation and user
impact, list verification commands, link relevant issues, and call out config
or dependency changes. Include sample images or metric excerpts when generated
output changes, but do not commit checkpoints, datasets, secrets, or routine
run artifacts.
