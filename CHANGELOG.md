# Changelog

All notable changes to Stochaflow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Because Stochaflow is pre-1.0, minor releases may intentionally contain breaking
changes. Such changes are called out explicitly and fail closed where migration
is unavailable.

## [Unreleased]

### Added

- Added learned-range Gaussian variance, hybrid variational-bound training,
  respaced ancestral DDPM, and learned-variance classifier-free guidance.
- Added configurable task-neutral training metrics, Strategy metric channels,
  phase-local metric state, and validation-only checkpoint monitoring.
- Added cadence-controlled epoch-end validation Evaluation over the current raw
  or EMA snapshot. Declared `valid/metrics/*` observations now feed the existing
  best-checkpoint and early-stopping lifecycle, with profile identity, cadence,
  complete interval/final observation history, and the last completed result
  persisted for strict resume.
- Added structured training outcomes and completed-run manifest publication.
- Added standalone checkpoint evaluation with explicit raw/EMA selection,
  registered Evaluation Builders, exact completeness, and immutable results.
- Added versioned prediction artifacts, streamed canonical JSONL records,
  deterministic gallery identity, and offline metric replay.
- Added core FID/KID providers and a public AFHQ-v2 full-official-test Evaluation
  profile with aggregate/per-class scoring and replayable predictions.
- Added the narrow public `ShareableImageFeatureMetric` capability so composite
  Metrics can reuse image features only across identical extractor identities;
  Evaluation continues to submit samples. KID now also accepts
  `antialias: bool = true` as part of that identity.
- Added exact modality-neutral `SamplingBatch` counts, stable checkpoint
  SHA-256/progress identity, and atomic no-replace sampling-bundle publication
  with portable relative artifact paths.
- Added a fresh AFHQ production quality candidate that keeps the canonical ADM
  graph while jointly using a `[1,2,3,4]` / 16x16 scale layout and a `2C`
  learned-range-v head. Its complete validation Evaluation samples 300 examples
  per class every 10 epochs from epoch 100 through 200, in batches of 15, and
  reports aggregate and per-class FID/KID for selecting `best.pt`.
- Completed that fresh candidate through epoch 200 with zero skipped optimizer
  updates. Eleven strict 900-example validation Evaluations selected epoch 190
  at aggregate FID 25.757212; one subsequent frozen 1,467-example official-test
  Evaluation reported aggregate FID 20.247791 and KID mean 0.002929 with zero
  missing sample IDs. Historical ADM comparisons retain their disclosed
  protocol differences and are not treated as isolated variance or topology
  ablations.
- Added a reproducible class-balanced AFHQ-v2 showcase panel sampled from that
  epoch-190 EMA checkpoint with DDPM-100, CFG 2.0, and the frozen evaluation
  seed, and published it across the root README and AFHQ documentation.
- Recorded a schema-v3 corrected-topology capacity sweep from an RTX 4090 with
  PyTorch 2.11/CUDA 12.8: four BF16 micro-batch trials, each with 5 warmup and
  25 measured updates, completed with zero non-finite observations.
- Recorded exact RTX 4090 evidence for the 100,351,366-parameter
  topology-and-variance candidate: micro batch 8 / accumulation 4 sustained
  45.17 images/s with 10.455 GiB peak reserved memory and zero non-finite
  observations across 25 measured updates.
- Added root-level [`SPEC.md`](SPEC.md),
  [`ARCHITECTURE.md`](ARCHITECTURE.md),
  [`ROADMAP.md`](ROADMAP.md), and `CHANGELOG.md` governance documents.

### Changed

- Hardened standalone and epoch-end Evaluation failure cleanup so the original
  failure remains primary while every Metric, module-mode, EMA, and unpublished
  prediction-sink restoration is still attempted. Prediction sinks now finalize
  only after framework-owned state has been restored successfully.
- Added a parked development plan for processing and training on datasets larger
  than RAM through bounded storage, host-memory, pinned-memory, and device
  stages. The incoming eight-H200 node is only the first validation environment;
  the capability preserves the completed Data lifecycle, works independently
  of single- or multi-device execution, prefers mature PyTorch, dataset-format,
  profiling, DALI, and GDS implementations over framework-owned replacements,
  and adds adapters only where lifecycle or recovery semantics require them.
- Added one current, user-facing guide for training Metrics, Diagnostics, and
  model selection. It documents the built-in Strategy channels, validation-only
  selection, checkpoint v12 state boundary, and the handoff to standalone
  Evaluation without restoring the superseded diagnostic-selection design.
- Reorganized the development documentation as short, feature-specific design
  narratives. Plans now start from the user problem and explain how that
  problem leads to the proposed design, its limits, and the evidence needed to
  build it; the former six-question form is retained only as an author review
  checklist. Future command examples are kept only when they clarify the core
  experience and are marked as non-executable. The root roadmap remains the
  only scheduling authority; completed Data and Evaluation work is not
  presented as open work; and research ideas are preserved without promoting
  them into the product schedule. The content map traces retained ideas back to
  the maintainer-designated `5c75a76` baseline.
- Closed the modality-neutral `DataArtifact` producer lifecycle. Artifacts are
  now sealed, Store-issued return values; source requests reject stale handles
  and requests with a selected or expected source reject wrong producer
  identities, while formal DataBuilder execution rejects
  direct-Store, fabricated, previous-build, or unbound declarations before
  checkpoint provenance is published. Runtime
  verification receipts remain outside manifests and checkpoint schemas, while
  arbitrary family-owned payloads and family-local source registries remain
  supported. Receipts are bound to exact non-copyable runtime handles, and
  nested worker contexts remain in flight even when a parent catches its own
  lifecycle error. An explicit weak-reference slot keeps exact-handle receipts
  compatible across supported Python 3.12+ patch releases. Promoted the durable
  lifecycle and composition boundary into the specification, architecture, and
  public Data documentation, then removed the two completed development
  decision records while preserving future review triggers under their
  long-term owners.
- Reorganized the root roadmap and internal development plans around plain
  current-state, user-outcome, action, and verification sections. Detailed
  future designs now remain in linked development notes, opaque milestone IDs
  are historical aliases only, and unselected work no longer carries stale
  schedule estimates.
- Added a Sphinx maintenance entry for the canonical root roadmap, development
  execution route, and every reader-facing main plan. Research archives remain
  outside site search, and the published pages state explicitly that visibility
  does not make candidate work implemented or scheduled.
- Made Evaluation protocol identity fail closed: Builders now declare explicit
  provider/preprocessing facts and nested provider dependencies, while formal
  and epoch-validation runtimes bind registered implementations and runtime
  versions before execution. Offline protocol compatibility no longer changes
  with exact prediction-subject bytes, while the ordered sample-ID digest keeps
  different sample plans distinct. Removed unused speculative request and
  Gate-path placeholders from the public contracts.
- Raised extension-project Torch declarations to the package-wide 2.11 minimum
  and migrated FP16 scaling/checkpoint identity to `torch.amp.GradScaler`.
- Moved checkpoint inference-asset reconstruction into the shared `inference`
  layer and the reusable epoch Evaluation adapter into `training`, preserving
  sampling as a downstream consumer and `scripts` as operation entry points.
- Versioned the nested epoch-validation strict-resume state independently from
  outer checkpoint v12. New checkpoints persist every completed result with its
  epoch, global step, and exact metric surface; unversioned summary-only state
  now fails strict resume while remaining valid for read-only inference and
  Evaluation projection.
- Closed ordinary pixel-space AFHQ learned-range-v training, validation
  selection, and final-test publication. This candidate jointly changes the
  scale layout and variance head;
  it is not an isolated learned-variance result or an exact epsilon-prediction
  IDDPM reproduction. Future tasks still require an explicit roadmap decision
  and must deliver observability, checkpoint inference, and formal Evaluation
  with their first implementation.
- Kept Metrics task-neutral: FID/KID providers consume Evaluation-owned image
  pair updates, while Evaluation owns sampling, sample identity, and strict
  completeness. Re-evaluating multiple checkpoints now means repeating that
  same Evaluation and comparing its metrics, not using a separate selector
  runtime.
- Established `ARCHITECTURE.md` as the sole normative architecture authority and
  recast `docs/framework.md` as a descriptive overview of current capabilities
  and workflows.
- Corrected the ADM U-Net to the canonical encoder/decoder skip topology and
  rejected checkpoints created by the incompatible earlier graph.
- Promoted AFHQ-v2 ADM production batching from provisional micro batch 1 /
  accumulation 32 to measured micro batch 8 / accumulation 4, preserving
  420 optimizer updates per epoch and 84,000 total updates. The default
  five-level ADM and the four-level learned-range quality candidate retain
  separate, exact capacity evidence.
- Separated training and sampling configuration authority; sampling is now an
  independent complete checkpoint-backed invocation.
- Reorganized Gaussian code by framework layer so family tensor semantics,
  Process mathematics, Training recipes, and Sampling algorithms have explicit
  dependency boundaries.
- Replaced generic Gaussian loss-weighting policy abstractions with concrete
  TrainingStrategy and TrainingBuilder recipes.
- Advanced the strict checkpoint contract to format version 12 for the new
  operation and state semantics.
- Sampling now requires current v12 progress identity and rejects incomplete or
  legacy checkpoint topology instead of providing compatibility fallbacks.
- Resolved sampling and evaluation calls now require the opaque activation
  receipt produced for their exact extension preflight plan.
- Made the release-wheel path the default quick-start workflow.
- Consolidated product scope, non-goals, external responsibility delegation,
  and public-abstraction admission rules into [`SPEC.md`](SPEC.md).

### Removed

- Removed the Intel macOS dependency pins, CI lane, and runtime test exception;
  macOS x86_64 is no longer a supported or best-effort target.
- Removed the legacy AFHQ-v2 evaluator executable, its private configuration and
  result runtime, and the superseded 900-sample profile. Maintained AFHQ
  benchmarks now use only the public formal Evaluation operation; historical
  removal facts remain in this changelog and Git history.
- Removed the experimental P2 TrainingBuilder/Strategy recipes and their AFHQ
  smoke, profiling, production, selection, and official-test profiles. The
  retirement rationale remains in `ROADMAP.md`, this changelog, and Git history;
  no runnable profile or intermediate development record remains authoritative.
- Removed automatic post-training final sampling and its skip flag.
- Removed the retired single-class AFHQ reproduction lane from the maintained
  example surface.
- Removed generic Gaussian weighting registries and loss-composer abstractions
  that violated task and layer ownership.
- Removed the duplicate `docs/design/scope.md` authority after consolidating its
  effective requirements into [`SPEC.md`](SPEC.md).
- Removed four retired development records covering AFHQ-v2 checkpoint cleanup,
  learned-range-v closeout, Intel macOS test retirement, and P2 experiment
  closeout after preserving their durable conclusions in current documentation.
  The temporary Sphinx and documentation-test exclusion list was removed with
  them.

### Fixed

- Aligned Pylance and repository Pyright diagnostics for deprecated overloads
  and unused declarations. Context-manager generators now use precise return
  types, cross-module helpers no longer masquerade as module-private dead code,
  built-in image-source registration is explicit, and extension activation test
  resets live in pytest support instead of the production plugin runtime.
- Documented the sealed extension preflight snapshot as a normative activation
  guarantee in `SPEC.md` and `ARCHITECTURE.md`.
- Sealed each extension activation preflight behind a private deep-copy
  configuration snapshot. `ExtensionActivationPlan.config` now returns a
  detached value, so callers cannot change the validated activation input
  between prepare and activate, while programmatic tuple and `torch.dtype`
  values retain their types.
- Excluded checkpoint metadata, including training-loop validation and monitor
  state, from the read-only inference projection.
- Preserved the Process-owned high-precision alpha-bar authority across strict
  state round trips so marginal, selected-pair, and learned-range coefficients
  cannot diverge after checkpoint loading.
- Kept nested Metric construction under the active injected registry authority
  instead of silently falling back to process-global registrations.
- Created validated nested prediction-shard parents and made tensor/image
  writers reject payload and trajectory batch counts that disagree with their
  declared `SamplingBatch.num_samples`.
- Preserved the filesystem's actual checkpoint-directory spelling during
  run-directory resume discovery, avoiding a path-identity mismatch on
  case-insensitive macOS filesystems.
- Made run-directory strict resume select the highest lineage-consistent atomic
  snapshot across `latest.pt`, `best.pt`, and the highest numbered checkpoint,
  while explicit-file resume remains exact. Contradictory, corrupt, regressing,
  mixed-lineage, inherited-best-only, and multiple-nested-directory candidate
  sets now fail closed, including impossible ordinary-validation observation,
  wait-counter, best-epoch, and best-metric transitions.
- Reordered epoch publication so an improved `best.pt` is followed by
  `latest.pt`, then an optional numbered checkpoint, before epoch diagnostics,
  logging, and reporting. Directory recovery now retains the furthest
  successfully published epoch boundary after any later publication failure.
- Persisted every interval and staged-final validation result, including its
  epoch, global step, and exact metrics. Strict resume now replays
  monitor/best/patience/stopping state from that full history instead of
  reconstructing observations from cadence summaries.
- Allowed strict resume from non-due epoch-validation checkpoints whose current
  epoch metrics intentionally omit sparse validation observations, while
  requiring scheduled/evaluated checkpoints to exactly match their persisted
  metric surface, rejecting stale observations, and rejecting claimed
  evaluations outside the frozen cadence or original final epoch.
- Propagated an injected RegistryCatalog through epoch validation Evaluation
  builders, MetricEngine construction, and writer-free SamplingBuilder
  execution instead of partially falling back to process-global registries.
- Restored the canonical ADM mixed-precision boundary before the output head,
  so the final normalization consumes the input dtype instead of a BF16/FP16
  decoder activation.
- Prevented Windows artifact scans from treating a lazily reported directory
  allocation size as a content mutation while retaining identity, timestamp,
  link, file-state, and manifest validation.
- Fixed discrete Gaussian coefficient state on MPS devices.
- Fixed validation metric monitoring and strict-resume behavior.
- Fixed evaluation config/checkpoint time-of-check/time-of-use gaps, strict data
  artifact binding, AFHQ offline generation-profile binding, validation-split
  identity, and deterministic KID subset sampling without mutating global RNG.

## [0.1.0] - 2026-07-30

### Added

- Initial public release of the configuration-driven Stochaflow research
  framework.
- Added installable CLI workflows, strict checkpoint-backed training and
  sampling, extension-project scaffolding, and the maintained MNIST example.

[Unreleased]: https://github.com/supermassiveasshole/stochaflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/supermassiveasshole/stochaflow/releases/tag/v0.1.0
