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
  and the last completed result persisted for strict resume.
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

- Made ordinary pixel-space AFHQ learned-range-v training the active quality
  closeout. This candidate jointly changes the scale layout and variance head;
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

- Removed the experimental P2 TrainingBuilder/Strategy recipes and their AFHQ
  smoke, profiling, production, selection, and official-test profiles. Their
  completed runs remain documented only as historical development evidence.
- Removed automatic post-training final sampling and its skip flag.
- Removed the retired single-class AFHQ reproduction lane from the maintained
  example surface.
- Removed generic Gaussian weighting registries and loss-composer abstractions
  that violated task and layer ownership.
- Removed the duplicate `docs/design/scope.md` authority after consolidating its
  effective requirements into [`SPEC.md`](SPEC.md).

### Fixed

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
