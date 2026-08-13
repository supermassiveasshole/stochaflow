# Stochaflow Specification

> Status: Normative
>
> Applies to: the current source tree
>
> Last reviewed: 2026-08-13

This document defines the product-level contract of Stochaflow: what the
framework is responsible for, which observable guarantees supported workflows
must preserve, and which responsibilities remain outside the project. Detailed
configuration and API syntax live under [`docs/`](docs/index.md); architectural
ownership lives in [`ARCHITECTURE.md`](ARCHITECTURE.md).

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are
normative. A change that intentionally alters this contract must update this
file, its tests, and the [`CHANGELOG.md`](CHANGELOG.md) `Unreleased` section in
the same change.

## 1. Product Purpose

Stochaflow is a configuration-driven, extensible research framework for
generative modeling. It supplies the composition and execution layer between
algorithm implementations and external infrastructure. Its primary value is to
turn research components into workflows that are:

- explicit about configuration and component identity;
- validated at composition boundaries;
- resumable within declared compatibility limits;
- auditable through checkpoints, manifests, metrics, and artifacts;
- reusable by installed extension projects without core dispatch changes.

The maintained built-in algorithm surface currently centers on pixel-space
discrete Gaussian diffusion. This specification is the sole authority for
long-term product scope and explicit non-goals. Current user-visible capabilities
are described in [`README.md`](README.md) and
[`docs/framework.md`](docs/framework.md); a capability being compatible with
this specification does not mean that it is implemented or scheduled.

## 2. Supported Operations

The command-line product surface consists of four independent operations:

| Operation | Responsibility |
| --- | --- |
| `stochaflow init` | Scaffold an installable extension project with a runnable example. |
| `stochaflow train` | Compose data and managed training assets, execute a supported training loop, and publish checkpoints and run evidence. |
| `stochaflow sample` | Reconstruct checkpoint-backed inference assets and execute a complete sampling request. |
| `stochaflow evaluate` | Evaluate a declared checkpoint or prediction-artifact subject under an explicit protocol and publish an immutable result bundle. |

Operations MUST have distinct configuration authority. Training configuration
MUST NOT silently control a later sampling or evaluation invocation. A
convenience workflow MAY call another public operation, but it MUST reuse that
operation's public contract instead of copying its runtime logic.

## 3. Configuration and Composition

1. Every operation MUST have one authoritative base configuration.
2. Configuration MUST select stable registered components and provide validated
   parameters; it MUST NOT become a universal arbitrary-object graph.
3. Complex task composition MUST remain in a narrow Python Builder or Strategy.
4. Resolved configuration MUST describe the components and semantics actually
   used by the run.
5. Unknown fields, incompatible component combinations, and unsupported
   lifecycle requirements MUST fail closed with actionable errors.
6. Built-in components MUST use the same public registry and construction paths
   available to third-party extensions.
7. Mature dependency namespaces, such as `torch.optim`, SHOULD be consumed
   through allowlisted native-provider boundaries rather than mirrored into
   Stochaflow registries.

The canonical configuration reference is
[`docs/configuration/reference.md`](docs/configuration/reference.md).

## 4. Data Contract

- A `DataSource` MUST acquire, read, validate, transform, and materialize
  external data as a verified `DataArtifact`.
- `DataArtifactStore` MUST be the sole issuer of an accepted `DataArtifact`.
  Issuance MUST occur only after object validation, payload loading, and the
  post-load mutation check. Direct construction, subclass stand-ins, copied or
  serialized handles, and a receipt attached to a different handle MUST fail
  closed. Runtime handles have object-identity semantics; stable value
  comparison MUST use `DataArtifactIdentity`.
- Each source materialization request MUST accept only a Store receipt produced
  for the same logical request represented by its `DataSourceContext`, including
  explicitly derived nested contexts. One context represents one logical source
  request and MUST NOT be copied, serialized, or reused for another source
  selection. Runtime
  receipt evidence MUST remain
  ephemeral and MUST NOT enter artifact identity serialization, manifests,
  cache keys, or checkpoints.
- A composite source MAY derive nested contexts for internal producers. Its
  nested work MUST finish before the direct parent source call returns. A
  nested context MUST reject work after that parent context closes. The
  final artifact identity MUST bind every internal artifact fact that can alter
  the final payload or represented content; nested runtime receipts are not
  persisted as separate Builder bindings.
- A `DataSource` MUST NOT construct runtime dataset views, partitions, PyTorch
  samplers, collate functions, or data loaders.
- A `DataBuilder` MUST own runtime data composition, including source selection,
  artifact binding, partitions, transforms, ordinary dataset views, samplers,
  collation, and train/validation/test iterables.
- A distributed training runtime MAY inject only an immutable rank and world-size
  projection before `DataBuilder` construction. The Builder MUST continue to own
  task-specific assignment, worker, sampler, and coverage semantics; core MUST
  NOT replace or inspect an arbitrary task loader after construction.
- A `DataBuilder` used by the fixed distributed lifecycle MUST return a narrow
  ranked-execution capability bound to the exact train and validation iterables.
  That capability MUST plan complete equal optimizer windows, report
  Builder-authenticated batch and terminal facts, and prove the declared
  validation coverage without imposing a framework batch schema.
- Core runtime code MUST treat a batch as structured `Any`; modality-specific
  keys and semantics belong to the selected Strategy or project component.
- Inputs that participate in resume, inference, or evaluation MUST carry the
  artifact identity required to validate that use.
- Formal `DataBuilder` execution MUST reject declared artifact bindings that
  were not accepted through a `DataSource` request during the current build.
  A formal Builder MUST NOT call `DataArtifactStore` directly and MUST finish
  every source request before returning `DataLoaders`; direct Store use,
  in-flight requests, and late source results MUST fail closed. Every outermost
  call for a distinct nested context MUST remain counted until it finishes,
  even when its direct parent catches a lifecycle error.
  Strict resume MUST additionally require full verification and exact equality
  with checkpoint identities. Binding-role semantics remain Builder-owned, so
  one accepted artifact MAY be assigned to more than one role. A binding MAY
  carry an equal schema-round-tripped identity when the same identity has
  current accepted receipt evidence; Python object identity is not normative.
- Artifact payloads and batch semantics MAY be arbitrary project-owned Python
  values. Core MUST NOT require image types or a universal payload schema;
  source registries and payload validation remain family-local.
- A Builder MAY omit artifact bindings only when its inputs are fully determined
  by resolved configuration and the run seed. External files, services, or
  streams MUST use a governed artifact boundary before they can provide formal
  provenance or strict-resume evidence.
- Runtime receipt checks prove the origin and current verification of declared
  bindings. They MUST NOT be described as proving that trusted project Builder
  code actually used a bound payload when assembling its iterables.

## 5. Training Contract

- A `TrainingBuilder` MUST assemble a `TrainingPlan` from injected assets and
  private task parameters, and MUST validate cross-component compatibility.
- A `TrainingStrategy` MUST own only batch interpretation, model/objective
  invocation, and loss or metric production.
- The supported automatic Trainer lifecycle MUST own device placement, module
  mode, precision, optimization, gradient handling, EMA, checkpointing, and
  framework-owned run state.
- A Strategy MUST NOT construct, move, freeze, select parameters from, or
  serialize managed assets.
- Independent optimizers, alternating updates, closure-required optimization,
  or manual backward semantics require an explicitly supported training-loop
  family; they MUST NOT be inferred from optional Strategy flags.
- Checkpoint selection MUST use validation observations. Diagnostics and test
  observations MUST NOT silently influence model selection.
- A configured epoch-end validation Evaluation MAY evaluate the current raw or
  EMA training snapshot and publish canonical `valid/metrics/*` observations.
  Its profile, metric surface, cadence, complete interval/final observation
  history, and last completed result MUST be strict-resume state. Epochs outside
  the cadence MUST NOT reuse an older observation or advance early-stopping
  patience.
- A completed training run SHOULD expose a structured outcome containing final
  metrics, selected checkpoints, and published artifact references.

### 5.1 Fixed Single-Node DDP Contract

The current distributed training software path is an explicit
`stochaflow train --ddp` operation launched by `torchrun`. It has the following
fixed first-version contract:

- it MUST use one non-elastic process per local CUDA device, a single node, a
  fixed world size of at least two, and the NCCL backend; CPU/Gloo execution is
  a test surface, not a maintained user operation;
- `LOCAL_RANK` MUST select the process device, `world_size` MUST equal
  `local_world_size`, and elastic restarts, multi-node membership, and adoption
  of an existing process group MUST fail closed;
- the configured batch size is the per-rank batch. The effective global batch
  is `world size * per-rank batch * gradient accumulation`; Stochaflow MUST
  record this fact and MUST NOT scale the learning rate automatically;
- every committed optimizer window MUST contain complete, equal rank-local
  microbatch and loss-weight facts. Non-final accumulation microbatches MUST
  suppress DDP gradient synchronization across both forward and backward;
  any rank-local invalid loss, gradient, data fact, or lifecycle failure MUST
  prevent publication of that epoch as a resumable result;
- a compatible `TrainingBuilder` and its registered module classes MUST treat
  resolved configuration plus registered parameter, buffer, and checkpoint
  state as the complete authority for forward semantics. Undeclared mutable
  Python fields that change forward behavior are unsupported and MUST NOT be
  used to claim fixed-DDP or exact-resume compatibility;
- validation MUST use one exact, complete global view. The first version runs
  that view on rank zero while every other rank participates in the same
  bounded heartbeat, result broadcast, and completion checks; only the global
  result may drive checkpoint selection or early stopping;
- an exact training resume result MUST be a manifest-last epoch-boundary bundle
  containing one common checkpoint plus every rank's RNG and next ranked-data
  plan. Resume MUST require the same fixed topology, backend/device type,
  artifact identity, ranked-data identity, batch semantics, and freshly
  verified next plan before mutating runtime state;
- a separately exported checkpoint-v12 file is portable inference state for
  ordinary single-process `sample` or `evaluate`. It MUST NOT be accepted as an
  exact DDP resume bundle, and a bundle/common checkpoint MUST NOT be accepted
  as an ordinary single-process training checkpoint.

The first maintained built-in ranked-data implementation is
`class_labeled_image`. Other Builders enter this lifecycle only through the
same public ranked-data and primary-execution-binding contracts. Unsupported
first-version behavior, including fp16, training Diagnostics, epoch-end live
Evaluation, train/test phase Metrics, test-after-fit, epoch-interval schedulers,
partial accumulation windows, truncated validation, and checkpoint intervals
other than every epoch, MUST fail closed before training data is consumed.

The target Linux CUDA/NCCL and 8xH200 performance, capacity, and fault
acceptance remains an in-progress roadmap requirement. The existence of this
software path and CPU/Gloo correctness tests MUST NOT be represented as that
hardware acceptance or as a throughput guarantee.

## 6. Process and Algorithm-Family Contract

- `Process` is a registry and lifecycle root for model-free probability paths;
  it is not a universal mathematical API.
- Algorithm families MUST define only the narrow mathematical capabilities their
  consumers require.
- A method without a probability path MUST NOT invent a Process merely to fit
  the framework shape.
- Gaussian schedules and coefficients remain Gaussian-family concepts; they
  MUST NOT become universal learning-rate, ODE-grid, or interpolation schemas.
- Shared family tensor mathematics MUST remain independent of task models,
  batches, Objectives, and runtime orchestration.

## 7. Sampling and Inference Contract

- A `Sampler` MUST own a complete numerical sampling algorithm and its ephemeral
  solver state.
- A `SamplingBuilder` MUST own task composition: model adaptation,
  conditioning, guidance, initialization, family compatibility, and
  writer-ready output.
- Every `SamplingBatch` MUST declare an exact positive sample count, and the
  sum of one `SamplingOutput` MUST equal the invocation's requested count.
  Core MUST NOT infer this count from a task-private payload shape.
- Sampling requests MUST be complete invocations and MUST identify both the
  checkpoint subject and the sampling configuration.
- Checkpoint sampling MUST bind the loaded stable bytes to a SHA-256 digest,
  format version, epoch, and global step before execution; later path changes
  MUST NOT change the resolved subject.
- Checkpoint-backed inference MUST reconstruct only declared inference assets;
  it MUST NOT require optimizer, scheduler, or training RNG restoration.
- Prediction type, variance semantics, conditioning semantics, and other facts
  needed to interpret model output MUST be frozen or validated at the
  checkpoint/inference boundary.
- Samplers belonging to one family MAY share public family-specific transition
  primitives, but generic Sampler or Dynamics roots MUST NOT gain universal
  `predict`, `step`, `drift`, `score`, or `denoise` methods.

## 8. Evaluation Contract

- An Evaluation MUST receive an explicit subject, selected data, protocol, and
  metric set. A formal Evaluation additionally MUST be a standalone operation
  with an explicit output destination.
- Supported subjects MUST expose enough identity to bind results to exact
  checkpoint assets or versioned prediction artifacts.
- Checkpoint evaluation MUST make raw/EMA selection explicit and MUST reuse the
  same narrow inference composition used by sampling where applicable.
- Prediction production and offline replay MUST enforce exact sample-plan
  completeness and reject missing, duplicate, corrupt, or incompatible records.
- Metric implementations MUST consume task-interpreted updates; the metric
  runtime MUST NOT infer a universal batch schema.
- A successful evaluation MUST publish an immutable result bundle with subject,
  data, protocol, metric, provider, and artifact identity sufficient for later
  verification.
- Every EvaluationPlan MUST explicitly declare non-empty, JSON-shaped provider
  and preprocessing identity. Nested registered metric providers and additional
  distributions that affect results MUST be named by the Builder; core MUST
  bind those declarations to selected implementation and runtime versions and
  fail closed before executing the data loop. At publication, the protocol
  digest MUST include that bound implementation record and the observed ordered
  sample-ID digest. Device and hardware facts MUST remain execution provenance
  rather than task schema.
- The protocol digest MUST identify the evaluation method, governed data/sample
  plan, and implementation compatibility independently of the exact subject
  content. Exact checkpoint or prediction-artifact digests MUST remain in the
  result subject identity so compatible results from different subjects can be
  compared without losing provenance.
- An epoch-end training Evaluation MAY reuse the same EvaluationPlan/runtime on
  a live raw or EMA snapshot without publishing a formal result bundle. Its
  metrics are validation observations for checkpoint selection, not benchmark
  evidence.
- Ordinary training metrics and diagnostics MAY provide development feedback,
  but they MUST NOT be represented as formal benchmark evidence.

## 9. State, Artifact, and Compatibility Contract

1. Checkpoints, run manifests, prediction artifacts, and evaluation results MUST
   use explicit format or schema versions.
2. Resume MUST validate the complete state required by the selected lifecycle;
   it MUST NOT silently load a semantically incompatible checkpoint.
3. An explicit checkpoint-file resume MUST use that exact snapshot. A directory
   resume MUST select the highest complete, lineage-consistent atomic snapshot
   and MUST fail closed on contradictory or ambiguous candidates.
4. Unsupported old formats MUST fail closed unless an explicit, tested migration
   exists.
5. A distributed resume bundle and a portable checkpoint MUST retain distinct
   roles. Neither a common file extracted from a bundle nor its portable export
   MAY silently enter the wrong training-resume lifecycle.
6. Publication of a completed result or manifest MUST occur only after required
   files and reporters finish successfully.
7. Artifact publication SHOULD use atomic staging where partial output would be
   ambiguous or unsafe.
8. Sampling bundles MUST confine writer paths to a private sibling staging
   directory, record portable relative artifact paths, and atomically publish
   to an absent final directory. Failure MUST leave no final bundle.
9. Generated datasets, checkpoints, routine run outputs, credentials, and
   machine-specific caches MUST NOT be committed.
10. Maintained host-platform support MUST be limited to the matrix declared in
   [`docs/platform-support.md`](docs/platform-support.md). macOS x86_64 is
   explicitly unsupported; core, dependency metadata, CI, and tests MUST NOT
   retain dedicated compatibility paths for it without revising this contract.

Stochaflow is pre-1.0 software. Breaking changes are permitted, but they MUST be
intentional, documented in `Unreleased`, and accompanied by explicit rejection
or migration behavior. See
[`docs/configuration/compatibility-and-migration.md`](docs/configuration/compatibility-and-migration.md).

## 10. Extensibility Contract

A compatible external component MUST be addable through implementation,
registration, configuration, and tests without editing core name-based dispatch.
Public extension contracts MUST preserve documented inputs, outputs, invariants,
state semantics, and errors. Important substitutability claims SHOULD be tested
with an independent extension implementation rather than only built-in
subclasses.

Extension activation MUST be explicit and auditable. Preparing an activation
MUST validate and capture a private configuration snapshot before importing
extension code. Public access to the prepared configuration MUST return a
detached value, and activation MUST materialize its resolved configuration from
the same captured snapshot rather than from caller-mutable state. This boundary
MUST preserve valid programmatic configuration value and container types. An
activation receipt MUST remain bound to the plan that produced it.

Project-private batch, condition, artifact, or model semantics MUST remain
within the project unless a second demonstrated consumer justifies a shared
public contract.

## 11. Observability Contract

- Metrics MUST be task-neutral stateful computations driven by explicit Strategy
  channels or evaluation updates.
- Logging backends and artifact writers MUST be replaceable integrations; core
  MUST NOT absorb their external dashboard, query, retention, or access-control
  planes.
- Training Diagnostic events MUST carry observed facts rather than a Trainer or
  full step result. Strategy-owned Diagnostic observations MUST remain opaque to
  core and MUST NOT imply a cross-family observation schema.
- A Diagnostic that invokes the task model MUST receive protected model access
  and any task-specific Strategy or Process behavior through explicit narrow
  capabilities validated during construction. Protected access MUST isolate RNG,
  raw/EMA selection, inference mode, and managed-module modes, then attempt every
  restoration even when execution or another restoration fails.
- A Diagnostic MUST declare cadence, additional assets, or a degradable failure
  policy only when its concrete behavior owns that concern. The framework MUST
  NOT require empty universal Diagnostic declarations.
- Failures MUST be visible and attributable. Optional diagnostics MAY degrade
  according to an explicit policy, but core state publication MUST remain
  correct.

## 12. Explicit Non-Goals

Stochaflow is not intended to become:

- a universal ML platform covering arbitrary predictive tasks or workflows;
- a dataset catalog, annotation system, dataset versioning service, or general
  dataflow engine;
- an experiment-tracking dashboard, metadata warehouse, or organization-wide
  lineage and governance graph;
- a storage engine, content-distribution system, replication service, backup
  product, or retention platform;
- a compute scheduler, cluster resource manager, autoscaler, quota manager, or
  permissions system;
- a general distributed runtime that reimplements collectives, elastic
  membership, fault-tolerant scheduling, or cluster coordination;
- distributed sampling, formal distributed Evaluation, model/optimizer
  sharding, FSDP, multi-node execution, topology-changing resume, and
  mid-epoch distributed resume as consequences of the fixed DDP lifecycle;
- a production serving control plane, request router, model deployment system,
  or online feature platform;
- a universal mathematical API shared by unrelated generative methods;
- an arbitrary YAML object graph or general-purpose DAG language;
- a mirror of mature dependency namespaces, defaults, or constructor schemas;
- the global owner of task, modality, condition, target, sample, or domain
  semantics;
- an environment manager, source-code manager, package index, or
  organization-wide repository manager.

Integrations with systems that own these responsibilities SHOULD remain narrow
adapters at the workflow boundary.

## 13. External Responsibility Delegation

Stochaflow owns only the workflow semantics needed to compose and validate a
generative-modeling operation. Infrastructure control planes remain external:

| Capability | Stochaflow owns | External owner owns |
| --- | --- | --- |
| Generative algorithms | Family contracts, composition boundaries, supported execution semantics | Research implementations beyond those contracts |
| Data | Artifact identity, verification hooks, task-side runtime composition | Catalogs, annotation, distribution, retention, and organization-wide lineage |
| Observability | Structured events, metrics, reporter interfaces, and artifact references | Dashboards, query planes, alerting, access control, and long-term retention |
| Persistence | Versioned state and artifact schemas, validation, and publication semantics | Object stores, replication, backup, lifecycle policy, and content delivery |
| Distributed execution | Correctness requirements for a declared supported lifecycle | Collectives, elastic membership, cluster coordination, and fault-tolerant scheduling |
| Resources | Runtime requirements and launcher-facing configuration | Queues, quotas, provisioning, autoscaling, placement, and accounting |
| Serving | Reconstructable inference assets and stable consumer-facing contracts | Deployment, routing, batching, admission control, rollout, and online monitoring |
| User interfaces | Stable operations and Builder contracts that a UI may consume | Visual editors, collaboration state, permissions, and product-specific interaction models |

An adapter MAY translate between a Stochaflow contract and an external system,
but it MUST NOT move the external system's control plane into core.

## 14. Public Abstraction Admission Criteria

A new public framework abstraction MUST satisfy all of the following:

1. **Framework ownership:** the responsibility belongs to Stochaflow's
   generative-workflow composition or lifecycle problem.
2. **Reuse evidence:** at least two credible consumers or workflows require the
   same stable semantics; speculative reuse is insufficient.
3. **Semantic stability:** the contract can be described without embedding one
   project's private vocabulary or implementation stage.
4. **Narrowness:** consumers can depend on a minimal capability instead of a
   universal base class or large optional interface.
5. **Composition root:** one Builder, Strategy, family boundary, or runtime
   lifecycle clearly owns construction and compatibility validation.
6. **Integration test:** a narrow adapter to an existing external system cannot
   solve the problem more appropriately.
7. **Substitutability:** an independent implementation can preserve the stated
   inputs, outputs, invariants, state semantics, and error guarantees.
8. **Evolution cost:** versioning, migration, documentation, and failure behavior
   can be maintained without creating hidden compatibility matrices.

If any criterion is not met, the default placement is:

| Situation | Default placement |
| --- | --- |
| One project, task, or modality | Project Builder, Strategy, or extension component |
| One algorithm family's mathematics | Family-specific contract or implementation |
| Capability already owned by a mature platform | Adapter or orchestration integration |
| One consumer or unstable semantics | Private implementation until evidence accumulates |
| Multiple workflows with a stable shared lifecycle | Candidate for a public framework contract |

## 15. Typical Boundary Decisions

| Request | Correct default owner | Rejected default |
| --- | --- | --- |
| Add an ODE or SDE solver | Family-specific Dynamics and Sampler contracts | Universal methods on root Process, Dynamics, or Sampler types |
| Add class conditioning or classifier-free guidance | Task SamplingBuilder and model adapter | Generic sampler condition fields |
| Add a homogeneous image source | DataSource plus a compatible DataBuilder recipe | A core dispatch branch for each dataset name |
| Add sequence packing or windowing | Project DataBuilder | Universal batch or dataset schema |
| Add frozen-teacher distillation | Builder manages teacher assets; Strategy combines forward passes and losses | Strategy loads, moves, freezes, or serializes the teacher |
| Add independent optimizers or alternating updates | A distinct supported training-loop family | Optional flags on the automatic loop or Strategy |
| Add an experiment dashboard | Reporter or tracking adapter | Core-owned UI, query plane, or metadata warehouse |
| Add remote artifact storage | Storage adapter using the provider SDK | Reimplemented object storage or replication logic |
| Launch Kubernetes GPU jobs | External launcher or orchestrator | Core-owned queues, provisioning, and autoscaling |
| Add a visual workflow editor | Upper-layer UI consuming stable operations and Builders | UI node models becoming the core object model |

## 16. Future-Compatible Directions

The following directions are compatible with this specification but are not
implementation commitments or roadmap priorities:

- flow matching, probability-flow ODE, score SDE, rectified flow, stochastic
  interpolants, and other probability-transport families under their own narrow
  contracts;
- explicit lifecycle families for demonstrated multi-optimizer, manual
  optimization, or distributed-execution requirements;
- remote artifact-store, tracking-backend, and execution-backend integrations
  through adapters;
- visual editors, batch systems, and serving consumers built over stable public
  operation and Builder contracts.

Every direction must still pass the admission criteria above. A roadmap item,
development plan, or example does not by itself expand product scope.
Support for a new task or algorithm family is one vertical-slice change: it MUST
include appropriate training observability, checkpoint-backed inference, and
formal Evaluation evidence when those lifecycles apply. Core MUST NOT add
task-specific fields, branches, profiles, or abstractions merely to anticipate a
future consistency, super-resolution, latent, codec, or distillation workflow.

## 17. Governance, Acceptance, and Change Control

This specification decides whether a responsibility may enter Stochaflow core.
[`README.md`](README.md) and [`docs/framework.md`](docs/framework.md) describe
current behavior; plans under `docs/development/` do not constitute a public
commitment. If a proposed feature violates this specification, it MUST be moved
to an algorithm family, project extension, adapter, or external system unless
the specification is explicitly revised first or in the same change.

A scope revision MUST identify the blocked real-world use case, explain why a
narrow extension or integration is insufficient, and state how the new
responsibility will be contained. Public documentation MUST NOT describe a
future-compatible direction as an implemented capability.

A behavior change is complete only when:

1. the implementation preserves this specification and
   [`ARCHITECTURE.md`](ARCHITECTURE.md), or the same change explicitly updates
   them;
2. focused regression tests cover success, validation, and important failure
   paths;
3. `uv run ruff check .` and `uv run pyright` pass for routine changes;
4. public configuration or workflow changes update the relevant documentation;
5. notable user-visible changes are recorded under
   [`CHANGELOG.md`](CHANGELOG.md) `Unreleased`;
6. future priorities are updated in [`ROADMAP.md`](ROADMAP.md) only when their
   ordering or status changes.

Before a major refactor is complete, verification SHOULD also demonstrate that
core does not depend on task registration names or concrete built-ins, important
contracts work with an independent implementation, and configuration,
checkpoint, artifact, and migration documentation remain consistent.

The implementation workflow and repository contribution rules are defined in
[`AGENTS.md`](AGENTS.md).
