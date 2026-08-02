# Changelog

All notable changes to Stochaflow are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Because Stochaflow is pre-1.0, minor releases may intentionally contain breaking
changes. Such changes are called out explicitly and fail closed where migration
is unavailable.

## [Unreleased]

### Added

- Added exact unconditional and class-conditional P2 Gaussian training recipes.
- Added learned-range Gaussian variance, hybrid variational-bound training,
  respaced ancestral DDPM, and learned-variance classifier-free guidance.
- Added configurable task-neutral training metrics, Strategy metric channels,
  phase-local metric state, and validation-only checkpoint monitoring.
- Added structured training outcomes and completed-run manifest publication.
- Added standalone checkpoint evaluation with explicit raw/EMA selection,
  registered Evaluation Builders, exact completeness, and immutable results.
- Added versioned prediction artifacts, streamed canonical JSONL records,
  deterministic gallery identity, and offline metric replay.
- Added core FID/KID providers and a public AFHQ-v2 full-official-test Evaluation
  profile with aggregate/per-class scoring and replayable predictions.
- Added real-AFHQ P2 tiny-wiring and corrected-full-topology sanity profiles,
  plus a common epsilon/fixed equal-budget `latest.pt` EMA protocol for formal
  standard/P2 comparison.
- Added a maintained 200-epoch / 84,000-update AFHQ P2 production candidate and
  a versioned 300-per-class validation Evaluation profile for selecting one
  epoch EMA subject before a one-shot official-test run.
- Added a machine-readable AFHQ P2 closeout policy that freezes eligible epochs,
  validation FID/KID selection order, earliest-epoch final tie-break, exact
  official-test completeness, required quality thresholds, and the aspirational
  aggregate-FID target before the production run.
- Added exact modality-neutral `SamplingBatch` counts, stable checkpoint
  SHA-256/progress identity, and atomic no-replace sampling-bundle publication
  with portable relative artifact paths.
- Recorded a completed one-epoch controlled AFHQ P2 A/B under protocol
  `afhq-v2-adm-epsilon-ddim50-cfg2-official-test-v1`: `gamma: 0` control versus
  `gamma: 1` P2 on exact full-official-test samples. P2 was slightly worse for
  aggregate FID (369.621427 -> 371.250343), aggregate KID mean
  (0.476357937 -> 0.479742199), and every class; the single-seed result is
  readiness evidence, not a statistically significant or 200-epoch promotion
  claim.
- Recorded a schema-v3 corrected-topology P2 capacity sweep from an RTX 4090
  with PyTorch 2.11/CUDA 12.8: four BF16 micro-batch trials, each with 5 warmup and 25
  measured updates, completed with zero non-finite observations.
- Added root-level [`SPEC.md`](SPEC.md),
  [`ARCHITECTURE.md`](ARCHITECTURE.md),
  [`ROADMAP.md`](ROADMAP.md), and `CHANGELOG.md` governance documents.

### Changed

- Made single-arm pixel/P2 absolute-quality closeout the post-merge experiment
  priority; validation Evaluation is used to select one frozen subject and
  official test is reserved for its one-shot final acceptance. A standard/P2 superiority
  claim remains gated on a separately matched `gamma: 0` production control.
- Made the P2 production lane fail closed on CUDA, documented deterministic
  launch/resume commands, replaced reusable checkpoint aliases with explicit
  run/selected-epoch placeholders, and fixed KID seeds in formal AFHQ profiles.
- Clarified that the 900-example P2 validation protocol ranks checkpoints only,
  while absolute acceptance belongs exclusively to the one-shot 1,467-example
  epsilon/fixed P2 official-test subject. Marked its frozen FID/KID thresholds
  as internal project criteria that standard-v ADM and DiT do not inherit, and
  distinguished them from the paper's non-comparable AFHQ-Dog-256 benchmark.
- All other tasks and methods are outside the current P2 gate. Any future task
  requires an explicit roadmap decision and must deliver observability,
  checkpoint inference, and formal Evaluation with its first implementation.
- Established `ARCHITECTURE.md` as the sole normative architecture authority and
  recast `docs/framework.md` as a descriptive overview of current capabilities
  and workflows.
- Corrected the ADM U-Net to the canonical encoder/decoder skip topology and
  rejected checkpoints created by the incompatible earlier graph.
- Promoted AFHQ-v2 ADM production batching from provisional micro batch 1 /
  accumulation 32 to measured micro batch 8 / accumulation 4, preserving
  420 optimizer updates per epoch and 84,000 total updates.
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

- Removed automatic post-training final sampling and its skip flag.
- Removed the retired single-class AFHQ reproduction lane from the maintained
  example surface.
- Removed generic Gaussian weighting registries and loss-composer abstractions
  that violated task and layer ownership.
- Removed the duplicate `docs/design/scope.md` authority after consolidating its
  effective requirements into [`SPEC.md`](SPEC.md).

### Fixed

- Fixed discrete Gaussian coefficient state on MPS devices.
- Fixed metric monitoring and resume behavior after integrating P2 training.
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
