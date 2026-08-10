# Repository Guidelines

## Workflow

1. Analyze
2. Plan
3. Implement
4. Test
5. Review
6. Fix
7. Repeat Test–Review–Fix until convergence

Do not treat a change as complete while relevant tests fail, documentation or
generated references are stale, or review findings remain unresolved. Preserve
unrelated worktree changes and keep the implementation diff scoped to the task.

## Fundamental Documents

Use the root-level documents as the project governance set:

| Document | Authority | Update when |
| --- | --- | --- |
| [`SPEC.md`](SPEC.md) | Normative product behavior, supported operations, invariants, and non-goals | User-visible behavior or compatibility changes |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Normative architecture, ownership, dependency direction, and composition boundaries | Responsibilities, layers, or public extension contracts change |
| [`ROADMAP.md`](ROADMAP.md) | Canonical high-level direction, priority, milestone status, and promotion gates | Product ordering, milestone state, or exit evidence changes |
| [`CHANGELOG.md`](CHANGELOG.md) | Released history and the source of truth for notable `Unreleased` changes | Every notable change and every release |

`ROADMAP.md` is a high-level project document. Detailed estimates and active
implementation notes may live under `docs/development/`, but those plans are
subordinate to the root roadmap and are not evidence that a feature exists.
Stable behavior belongs in `SPEC.md` and the public documentation. Stable
architecture belongs only in `ARCHITECTURE.md`; `docs/framework.md` is a
descriptive overview of current capabilities and workflows, not a second
architecture authority.

Before planning or implementation, read the sections of `SPEC.md`,
`ARCHITECTURE.md`, and `ROADMAP.md` relevant to the task. If code, tests, and
fundamental documents disagree, stop and reconcile them explicitly; do not
silently choose one source.

When a responsibility or document boundary remains ambiguous and these
authorities do not support a clear decision:

1. First consult established engineering standards and primary-source examples
   from mature projects.
2. Record the unresolved question under `docs/development/`. Include the
   context, competing options, evidence, current status or decision owner, and
   the condition for resolution.
3. Do not leave the issue only in chat, silently invent a convention, or create
   a speculative public abstraction merely to bypass the decision.
4. Once resolved, promote the durable rule to `SPEC.md`, `ARCHITECTURE.md`,
   `AGENTS.md`, or the public documentation as appropriate, then close or
   archive the intermediate record.

Keep documentation synchronized in the same change:

- behavior or compatibility change: update `SPEC.md` and `CHANGELOG.md`;
- architecture or extension-boundary change: update `ARCHITECTURE.md`;
- priority or milestone change: update `ROADMAP.md`;
- release: move accumulated `Unreleased` entries into a dated version section.

## Project Structure & Module Organization

Stochaflow is a Python 3.12+ package using a `src` layout. The main package map
under `src/stochaflow/` is:

| Package | Responsibility |
| --- | --- |
| `data` | DataSource/DataArtifact lifecycles and runtime DataBuilder composition |
| `models` | Reusable model implementations and narrow model capabilities |
| `families` | Process-free tensor semantics shared within one algorithm family |
| `processes` | Model-free probability paths, schedules, and mathematical capabilities |
| `training` | TrainingBuilder/Plan, Strategy, Trainer, optimization, metrics binding, diagnostics, and outcomes |
| `sampling` | SamplingBuilder, inference adaptation, Dynamics, Samplers, execution, observers, and writers |
| `inference` | Read-only checkpoint projection shared by sampling and evaluation |
| `evaluation` | Standalone subjects, protocols, Builders, prediction artifacts, runtime, and result bundles |
| `metrics` | Task-neutral metric declarations, payloads, providers, and state lifecycle |
| `extensions` | Stable public extension imports |
| `projects` | Installable extension-project scaffolding and templates |
| `scripts` | CLI parsing and thin operation entry points |
| `utils` | Configuration, registries, checkpoints, plugins, logging, devices, and manifests |

Repository-owned examples live under `examples/built-in/`,
`examples/extension-projects/`, or `examples/showcases/`; keep each experiment
YAML with its owning project. Tests in `tests/` should mirror the behavior they
cover, using a subdirectory only where a feature has a cohesive test surface.
Keep documentation and research notes in `docs/` or `notebooks/`, and reserve
`assets/readme/` for curated documentation images. Generated data, checkpoints,
and runs belong under ignored data/cache/output locations and must not be
committed.

Published user documentation must describe current framework behavior and user
workflows, not implementation-stage history. Sphinx also exposes the canonical
root roadmap plus the reader-facing development index and main plans in a
clearly labeled maintenance area. Their visibility does not make a candidate
feature implemented, scheduled, or part of the public API. Detailed research
and historical notes under `docs/development/notes/` remain outside the Sphinx
site. Before a feature branch merges, move stable architecture, feature,
configuration, and usage content into the normal documentation tree. Close or
archive implementation-only material, while preserving any still-unimplemented
future direction in its owning plan.

### Development Documentation Style

`ROADMAP.md` is the only scheduling authority. Use only `Done`, `In progress`,
`Next`, `Candidate`, and `Parked` for work status; at most one item may be
`In progress` and at most one item may be `Next`; either may explicitly be
`None`. A development plan must not create a second priority vocabulary or
present an unselected candidate as committed work. The development index and
execution roadmap are derived navigation views and must match the root roadmap.

Write root governance documents in English. Write development plans in Chinese
while preserving exact API, type, and configuration identifiers in English.
Assume the first reader has only learned how to install the library, edit a YAML
file, and run `train`, `sample`, or `evaluate`. The first job of a plan is to
explain the idea to that reader in ordinary, connected prose. Start from a real
problem, show how it leads to the proposed direction, explain an important
alternative that was not chosen, and then describe how the work could become a
usable result. The author chooses the order and headings that fit the idea; do
not expose a standard questionnaire, task-card template, or repeated “current
state / gap / work / acceptance” skeleton merely to make documents look uniform.

During review, confirm that the prose as a whole lets a reader work out what the
idea is, why it matters, what works now, what remains missing, what evidence or
decision would start the work, and what using the completed result would feel
like. These are review questions, not required headings or literal phrases. A
future command, Python call, configuration, or directory tree should appear
only when it materially clarifies the idea. Mark it as non-executable and not a
public API when it does not exist. Prefer an ordinary user story over an
invented interface whose names and shape have not been decided.

Use headings that advance the particular argument. A step title should explain
an observable action, result, or design decision. Keep core technical reasoning
in the main plan when it explains why the design has its shape; move long API
drafts, provider surveys, mathematical derivations, test matrices, and
historical logs to notes. Opaque stage identifiers belong only in a linked
history note.

Explain a technical term the first time it is needed. Prefer “运行产物
（artifact）” to a bare `artifact`, “结果清单（manifest）” to a bare `manifest`,
and an ordinary description of what a Builder, runtime, writer, registry,
identity, digest, or provider does before using its exact name. Keep class names
and architecture contracts in the maintainer section; do not place a chain of
internal types in the user flow. Use “继续中断的训练” for resume and reserve
“图像恢复” or “超分推理” for image restoration so that the two meanings are not
mixed.

Keep a main plan as short as its argument permits. Do not use a hard line budget
to cut one continuous explanation into a maze of files, but move bulky research
that does not support the reader's immediate reasoning into notes. Keep each
reader-facing topic discoverable from `docs/development/README.md`, normally
within one or two clicks. Create a main plan only for a concrete product function
that a maintainer has selected as Candidate, Parked, Next, or In progress. A
research question, possible helper, provider survey, or hypothetical workload
may remain in a clearly labelled research note without appearing in
`ROADMAP.md`. Preserving an idea does not promote it into the product schedule.

Keep readable main plans separate from detailed research. Move
version-sensitive provider research, candidate API drafts, mathematical
derivations, long test matrices, and historical implementation logs under
`docs/development/notes/`. A closed capability may have a short “what users can
do now” record; it must not be presented as Parked work. A conditional review
must say that it may conclude “no change”. Do not remove a unique future-support
idea until its current destination is recorded, but do not make Git history or
a maze of sibling links its only destination. Do not attach time estimates to
unselected work; estimate again only after it becomes `Next`. Mark external
library, hardware, and dataset research with its last review date and require
fresh verification before implementation.

Use `docs/development/README.md` as the only reader-facing development entry.
Group links by the function a user is trying to obtain, without copying each
plan's opening argument into the index. The subordinate development roadmap
shows only the current execution queue and direct product dependencies; it must
not duplicate the full plan catalog. Prefer direct prerequisites, stable user
documentation, and a clearly labelled technical appendix over chains of sibling
links. Explain why a link is useful.

Before a documentation restructure removes or shortens old material, compare it
with the maintainer-designated historical baseline. Record each unique feature
idea as one of: current behavior, concrete future function, research material,
or retired history. “It remains in Git” is not preservation. Human review must
confirm that every concrete future function still has a plain-language
explanation and a discoverable current location. As a final readability check,
hide the headings and verify that the paragraphs still connect through cause,
consequence, contrast, and the next decision.

Use ordinary language instead of unexplained planning jargon. In particular,
replace terms such as “vertical slice”, “substrate”, “seam”, “closeout”,
“promotion gate”, “decision-gated”, “rebase”, and “task-owned” with a concrete
description of the result, existing ability, interface, completed decision,
entry condition, current status, required review, or responsible component.

## Build, Test, and Development Commands

- `uv sync --extra dev` synchronizes the local development environment and
  lockfile.
- `uv sync --extra docs` installs the documentation toolchain.
- `uv run pytest` runs the complete test suite; pass a file path or `-k` filter
  for focused validation.
- `uv run ruff check .` checks formatting-independent style and common errors.
- `uv run pyright` runs the repository static type checks.
- `uv run python tools/generate_config_reference.py --check` verifies that the
  generated configuration reference matches the source declarations.
- `uv run sphinx-build -W --keep-going -b html docs docs/_build/html` performs
  the strict documentation build used by CI.
- `uv build` creates wheel and source distributions in `dist/`.
- `uv run stochaflow train --config examples/built-in/image-generation/configs/train/mnist.yaml --epochs 1 --limit-batches 10 --limit-validation-batches 2 --limit-test-batches 2`
  performs the maintained bounded MNIST smoke run.

CI synchronizes its environments with `--frozen`; quality and release gates
also use `uv run --frozen`. Do not claim parity with CI if the lockfile or
generated references were allowed to change during verification.

## Coding Style & Naming Conventions

Use four-space indentation, the configured 88-character line-length target, and
type annotations for public APIs. Prefer small, single-purpose modules and
explicit imports. Name modules, functions, and variables `snake_case`; classes
`PascalCase`; constants `UPPER_SNAKE_CASE`. Every class declaration, including
internal helpers, protocols, typed dictionaries, dataclasses, templates, and
test fixtures, must use a formal descriptive name without a leading underscore;
`tests/test_code_conventions.py` enforces this repository-wide. Add concise
docstrings to public classes and non-obvious methods. Add configurable
components through an existing registry/factory contract rather than a
command-specific branch, and do not create a new registry unless the shared
public contract passes the admission criteria in `SPEC.md`.

## Architecture and Extension Guardrails

Treat all five SOLID principles as repository-level design constraints, not
only as class-level style guidance:

- **Single Responsibility:** each public component owns one cohesive policy and
  has one primary reason to change. Keep probability math, model adaptation,
  numerical solving, task composition, persistence, and artifact I/O in their
  respective layers. A component may parse and validate its private parameters,
  but it must not absorb unrelated runtime orchestration.
- **Open-Closed:** core runtime code coordinates stable lifecycles; task and
  algorithm variation belongs in registered components. A compatible
  DataBuilder, Process, Sampler, SamplingBuilder, TrainingBuilder,
  EvaluationBuilder, model, Objective, or Metric must be addable through
  implementation, registration, configuration, and tests without editing core
  dispatch or adding branches keyed by a registered name or concrete class.
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
- **Dependency Inversion:** runners, trainers, strategies, evaluators, and
  builders depend on public contracts and capability protocols rather than
  built-in algorithms, concrete storage classes, or registered names. Construct
  implementations at registry/factory composition boundaries and inject them
  into mathematical and runtime code. Concrete-class checks are allowed only
  when the concrete type is itself the explicitly documented contract;
  otherwise define or reuse a narrow capability.

Validate local parameter shape near component construction, cross-component
compatibility where a Builder can see the complete collaboration, and persisted
state or artifact compatibility at the read boundary. Runtime coordinators own
stable lifecycles; they must not recover task semantics by inspecting names,
concrete classes, or modality-specific fields.

### Data boundary

`DataSource` is the artifact-producing data extension entrypoint. It acquires,
reads, validates, and transforms external data, then materializes it as a
verified `DataArtifact`. It must not construct runtime Dataset views, data
partitions, PyTorch data samplers, collate functions, or DataLoader objects.
Only `DataArtifactStore` may issue a runtime `DataArtifact`; extensions must use
the generic source-materialization boundary and must not construct, subclass,
cache across source requests, or fabricate artifact handles. A
`DataSourceContext` represents one logical request and must not be reused for
another selection. Formal Builders may bind only artifacts
accepted through a DataSource request during that same build, and strict resume
requires a current full-verification receipt in addition to identity equality.
These receipts are runtime-only and never belong in manifests or checkpoints;
they prove declared binding provenance, not how trusted Builder code uses a
payload.

`DataBuilder` is the runtime data-composition entrypoint. It selects compatible
sources, binds artifact identities, and assembles Dataset views, splits,
transforms, PyTorch data samplers, collate functions, and ready
train/validation/test iterables. Core code treats batches as structured `Any`
and must not impose image, condition, target, sample-key, bucket, or metadata
fields. Family-specific source registries are allowed; do not create universal
Dataset, DataLoader, or `torch.utils.data.Sampler` registries or schemas.

### Training boundary

`TrainingBuilder` receives the primary model, optional `Process` and Objective,
and narrow factories, then returns a validated `TrainingPlan` that preserves
those injected assets. The Builder owns complex Python composition: it may
construct, load, freeze, and declare managed auxiliary modules, inference asset
projections, and a fixed inference recipe. Do not replace this boundary with a
universal YAML graph for multi-model training.

`TrainingStrategy` owns only batch interpretation, model/objective calls, and
loss, scalar, and `MetricUpdate` production. It must not construct, move, freeze,
select parameters from, or serialize managed training assets. Core owns device
and mode management, one automatic optimizer lifecycle, precision and gradient
accumulation, backward/step ordering, scheduler advancement, primary-model EMA,
checkpointing, and run outcomes.

Frozen-teacher distillation follows the same split: the Builder prepares and
declares the teacher; the Strategy combines forward passes and Objectives.
Independent optimizers, alternating updates, closure-driven optimization, or
manual backward require a distinct supported training-loop family, not Strategy
flags. Diagnostics that invoke a task model must consume an explicit narrow
Strategy capability; they must not infer a model signature from a Process family
or prediction-type field.

### Native optimization providers

Do not mirror mature dependency namespaces into Stochaflow registries or
configuration metadata. Standard `torch.optim` optimizers and
`torch.optim.lr_scheduler` schedulers are constructed through the allowlisted
native-provider resolver and validated against their public PyTorch contracts.
Core injects the trainable parameter iterable or optimizer and forwards the
remaining parameter mapping as constructor keyword arguments; do not copy
upstream defaults or infer runtime injections from arbitrary constructor names.

The automatic loop requires optimizer and scheduler `step()` methods callable
without required arguments, and a scheduler must retain the injected optimizer.
Third-party registered subclasses remain valid when they preserve the same
construction and lifecycle contracts. Closure-required optimizers and
metric-driven schedulers need an explicit lifecycle rather than concrete-name
special cases.

### Metrics and observability

`MetricSpec` declares a provider and task-owned channel; `MetricUpdate` carries
task-interpreted opaque arguments; `MetricEngine` owns only construction,
device, update, compute, and reset. The metric layer must not infer a batch
schema, phase, checkpoint policy, model signature, or evaluation protocol.
Training and evaluation own their separate metric bindings and lifecycles.

Training checkpoint selection and early stopping may consume validation
observations only. Train/test/system scalars and Diagnostic observations must
not silently become selection evidence. A training Diagnostic is an
observation hook, not a substitute for standalone formal evaluation.

### Algorithm-family and Process boundary

`Process` describes an optional model-free probability path. It must not own a
task model, interpret a batch, or run a sampling loop. The root is a registry and
lifecycle boundary, not a universal mathematical API; each algorithm family
defines the cohesive capabilities it actually needs. A method without a
probability path must not invent a `Process` merely to satisfy core dispatch.

Process-free tensor semantics shared within one family belong in `families/`.
A Gaussian noise schedule remains a Gaussian Process-owned composition point,
not a universal abstraction for interpolants, solver grids, or learning-rate
policies. Consumers depend on schedule capabilities, and derived coefficients
must remain consistent with mutable schedule state.

`GenerativeDynamics` is only a semantic root for an assembled generation
direction. Family contracts define concrete prediction, vector-field, reverse
SDE, or denoiser capabilities. Do not add universal `predict`, `step`, `drift`,
`score`, or `denoise` methods to the root, or create a global Dynamics registry
without demonstrated cross-component need.

### Inference and sampling boundary

The `inference` package projects portable checkpoint state into read-only model,
Process, and explicitly declared inference-asset capabilities shared by sampling
and checkpoint evaluation. It must exclude optimizer, scheduler, scaler, RNG,
and other training-loop state, and it must not own task sampling or evaluation
policy.

`Sampler` owns a complete numerical algorithm and its ephemeral solver state.
`SamplingBuilder` owns and executes task composition, including model adapters,
conditioning, guidance, initialization, inference assets, family compatibility,
and writer-ready `SamplingOutput`. A family-specific Sampler may require its
narrow Dynamics capability at the call boundary; core must not maintain a
global name-based compatibility matrix.

A family may expose narrow model-free transition or schedule primitives needed
by multiple solvers. These primitives remain family-specific and must not become
requirements of the root `Sampler` or `GenerativeDynamics`; built-in complete
samplers delegate to the shared family primitives instead of duplicating the
mathematics. A direct exact transform may return a valid `SamplingOutput`
without inventing a numerical Sampler, Dynamics, or Process.

### Evaluation boundary

Evaluation is a standalone operation with its own strict configuration
authority. `EvaluationBuilder` receives an explicit subject, re-iterable data,
data identity, protocol, metric declarations, and optional narrow inference or
sampling capability, then returns a validated `EvaluationPlan`. The Builder may
compose task-specific modules and an optional prediction-artifact sink; it must
not start the runtime or reinterpret configuration authority.

`Evaluator` interprets each opaque batch or prediction record and returns
`EvaluationStepOutput` with an exact count, unique sample IDs, task-owned metric
updates, measurements, and optional typed records. Core owns inference-mode
execution, declared-module eval-mode handling, metric lifecycle, global sample
identity and completeness checks, immutable result publication, and atomic
artifact publication.

The current formal subject types are a v12 checkpoint with explicit raw/EMA
selection and a complete versioned prediction artifact for offline replay.
Offline replay must remain read-only and must not reconstruct or rerun the model;
its sampling capability is absent. Training metrics, Diagnostics, and ordinary
sampling outputs are not formal evaluation evidence.

### Registry and extension boundary

Stochaflow-owned built-ins use the same registry, Builder, and construction paths
as external extensions; do not add hidden core-only shortcuts. Installed
extensions activate explicitly through the `stochaflow.extensions` entry-point
group, with identity and version provenance validated before code import.
Allowlisted native dependency providers are a separate documented boundary, not
Stochaflow-owned built-ins.

When a feature appears to require task-specific branching in a runner, a common
config field used by one modality, a concrete implementation dependency where a
capability would suffice, or a change to an otherwise compatible component,
stop and revise the responsibility or extension boundary before coding.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and functions named `test_*`. Add focused
unit tests for shapes, configuration validation, and error cases; add runner or
script tests when changing CLI behavior. No coverage threshold is configured,
so every behavior change should include a regression test. Avoid full training
runs in tests; use tiny tensors, fixtures, and limited batches.

Match verification to the changed surface:

- routine code changes: focused tests, `uv run ruff check .`, and
  `uv run pyright`;
- configuration declarations or generated reference changes: also run
  `uv run python tools/generate_config_reference.py --check`;
- published documentation or Sphinx navigation changes: also run the strict
  Sphinx build;
- packaging or release changes: run `uv build` and the applicable release
  contract tests;
- a feature branch ready to merge: run the full test suite and every relevant
  CI gate, then fix failures rather than documenting them as accepted debt.

Repository-wide static checks are not a substitute for focused behavioral
tests. Conversely, defer unrelated expensive acceptance or hardware runs until
the feature's merge gate requires them.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, sentence-case subjects such as
`Patch UNet bugs` and `Improve MNIST DDPM training UX`. Keep each commit scoped
to one logical change. Pull requests should explain the motivation and user
impact, list verification commands, link relevant issues, and call out config
or dependency changes. Include sample images or metric excerpts when generated
output changes, but do not commit checkpoints, datasets, secrets, or routine
run artifacts. Review the final diff for unrelated worktree changes before
staging or handing off the change.
