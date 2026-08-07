# Stochaflow Roadmap

> Status: Active
>
> Horizon: pre-1.0 product development
>
> Last reviewed: 2026-08-02

This is the canonical high-level product roadmap. It communicates direction,
priority, dependencies, and promotion evidence. It is not a statement that a
feature is implemented; current behavior is defined by [`SPEC.md`](SPEC.md),
[`README.md`](README.md), public documentation, and tests.

The detailed maintainer schedule, estimates, and cross-plan decisions live in
[`docs/development/development-priority-roadmap.md`](docs/development/development-priority-roadmap.md).
That document must remain consistent with this roadmap, but implementation-stage
plans do not override this high-level ordering.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| Done | Contract, implementation, tests, and stable documentation are closed. |
| Active | Currently receiving implementation or experiment effort. |
| Next | Directly follows the active milestone when its entry gate is satisfied. |
| Planned | Direction is accepted, but it is not on the immediate critical path. |
| Decision-gated | Work starts only after named evidence demonstrates the need. |
| Deferred | Intentionally outside the current execution horizon. |

Roadmap status is evidence-based. A design document or partial implementation
does not make a milestone Done.

## Product Direction

Stochaflow is building a trustworthy composition layer for generative-model
research workflows. The product sequence is:

```text
pixel-space Gaussian foundation
    -> formal evaluation foundation
    -> epoch-end validation Evaluation and metric-selected checkpoints
    -> pixel-space AFHQ learned-range-v quality closeout
    -> roadmap re-decision
```

The project deliberately develops one validated vertical slice before widening
the platform surface.

## Completed Foundation

The following foundations are considered closed and should receive maintenance,
not competing redesign:

- corrected canonical ADM topology and explicit checkpoint incompatibility;
- independent train and sample configuration authority with strict checkpoint
  semantics;
- discrete Gaussian fixed and learned-range variance, hybrid VB, respaced
  ancestral DDPM, DDIM, and learned-variance CFG handling;
- task-neutral Metrics with Strategy channels and validation-only selection;
- structured training outcomes and completed-run manifests;
- standalone checkpoint Evaluation with explicit raw/EMA selection;
- versioned prediction artifacts, exact sample completeness, streaming records,
  and offline metric replay;
- core FID/KID providers and a public class-aware AFHQ-v2 full-test profile;
- a maintained real-AFHQ class-conditional training, sampling, and formal
  Evaluation surface;
- schema-v3 RTX 4090 capacity evidence for the default five-level,
  105,197,187-parameter ADM and its measured production batch of 8 with
  accumulation 4;
- a maintained MNIST workflow and a class-aware AFHQ-v2 showcase contract.

The former P2 experiment surface was retired after its controlled runs showed
no verified benefit. It remains only as a development record and is not a
supported training recipe or active product gate.

## Completed Pixel-Space Closeout

### AFHQ learned-range quality evidence

The active branch closes ordinary pixel-space image generation with a coherent
training, validation, and final-test workflow. The production quality candidate
keeps the canonical ADM input/output-block graph, cosine Process, v-prediction, and
data/optimizer budget, but jointly changes the default five-level
`[1,1,2,3,4]` / 8x8 model to `[1,2,3,4]` / 16x16 and changes the variance head
to learned range. It is therefore a topology-and-variance candidate, not an
isolated learned-variance ablation or an exact epsilon-prediction IDDPM
reproduction. P2 is intentionally absent.

Repository merge evidence:

- canonical FID/KID provider, preprocessing, reference, and protocol identity;
- aggregate and class-aware AFHQ profiles with immutable result bundles;
- deterministic live/offline parity and completeness failure tests;
- target-device capacity evidence for the corrected ADM configuration;
- epoch-end live Evaluation with immutable profile identity, exact completeness,
  canonical validation metric keys, strict-resume state, and existing
  `best.pt` monitor integration.

Completed experiment evidence:

- the maintained learned-range-v AFHQ recipe completed 200 epochs from random
  initialization with zero skipped optimizer updates;
- all 11 scheduled 900-example EMA validation Evaluations from epochs 100
  through 200 completed with aggregate/per-class FID and KID, strict
  completeness, and a preserved interval/final observation history;
- validation aggregate FID selected epoch 190 (`best.pt`) at 25.757212, while
  KID and per-class observations remained evidence without independent
  selection authority;
- the frozen EMA/DDPM-100/CFG-2 official-test Evaluation ran exactly once on
  the selected checkpoint and all 1,467 official-test images, publishing
  aggregate FID 20.247791 and KID mean 0.002929 with zero missing sample IDs;
- the learned-range, fixed-variance current-ADM, and legacy-ADM results are
  recorded with their protocol differences. They establish disclosed
  end-to-end quality but not an isolated topology or variance effect.

The candidate has its own bounded RTX 4090 evidence: 100,351,366 parameters,
micro batch 8 / accumulation 4, 45.17 images/s, 10.455 GiB peak reserved memory,
and zero non-finite observations across 25 measured optimizer updates. It
retains 420 optimizer updates per epoch and 84,000 total updates. Training
diagnostics run every 10 epochs with DDPM-100 and DDIM-50, while periodic
checkpoints are retained every 50 epochs; those diagnostics remain observations
and have no selection authority.

## Current Decision Gate

No implementation milestone is currently selected after the completed
learned-range pixel-space evidence. Codec/latent diffusion, consistency methods,
super-resolution, distillation, Stable Diffusion integration, and new algorithm
families are neither `Next` nor `Planned`; they require a fresh roadmap decision
after the published result is reviewed.

If one of those tasks is later accepted, its first vertical slice must deliver
training observability, checkpoint-backed inference, and task-appropriate
formal Evaluation together. Core will not pre-implement task fields, branches,
profiles, or public abstractions in anticipation of that decision. Detailed
documents under `docs/development/` are parked design records, not active work
or evidence of commitment.

## Decision-Gated Work

The following areas are valuable but must not displace the current vertical
slice without their trigger evidence:

| Area | Trigger |
| --- | --- |
| Distributed training or inference | A validated single-device workload cannot meet measured capacity or wall-clock requirements. |
| Hyperparameter optimization | A stable objective, budget, and formal evaluation protocol exist. |
| General workflow orchestration | At least two real workflows require the same lifecycle beyond explicit operation composition. |
| Organization-wide provenance or capacity services | A second external consumer requires a shared contract beyond run-local evidence. |
| New algorithm families | A concrete end-to-end use case demonstrates the required narrow contracts. |

## Promotion Gates

A major milestone is not Done until it has:

1. a stable contract reflected in [`SPEC.md`](SPEC.md) when user-visible;
2. architecture ownership reflected in
   [`ARCHITECTURE.md`](ARCHITECTURE.md) when boundaries change;
3. implementation through public built-in and extension paths;
4. focused contract, failure, and independent-extension tests;
5. stable user documentation and a maintained example where appropriate;
6. an entry under [`CHANGELOG.md`](CHANGELOG.md) `Unreleased`;
7. measured evidence for claims about quality, performance, or capacity.

Hardware runs and long training are operational evidence. They must not be used
to justify architecture shortcuts, but architecture completion must not be
misrepresented as experimental proof.

## Updating This Roadmap

- Update this file when product ordering, milestone state, or promotion evidence
  changes.
- Keep implementation details, engineering estimates, and active decision logs
  in `docs/development/`.
- Move stable behavior into public docs when a milestone closes.
- Never use roadmap status as the sole evidence that code exists or works.
