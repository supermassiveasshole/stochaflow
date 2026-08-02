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
    -> pixel-space P2 production-quality closeout
    -> publish production P2 quality evidence
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
- discrete Gaussian fixed and learned-range variance, hybrid VB, concrete P2
  recipes, respaced ancestral DDPM, DDIM, and learned-variance CFG handling;
- task-neutral Metrics with Strategy channels and validation-only selection;
- structured training outcomes and completed-run manifests;
- standalone checkpoint Evaluation with explicit raw/EMA selection;
- versioned prediction artifacts, exact sample completeness, streaming records,
  and offline metric replay;
- core FID/KID providers and a public class-aware AFHQ-v2 full-test profile;
- maintained real-AFHQ P2 tiny-wiring, corrected-full-topology sanity, and
  epsilon/fixed equal-budget A/B protocol surfaces;
- schema-v3 RTX 4090 capacity evidence for the corrected 105,197,187-parameter
  topology under P2 training and a measured ADM production batch of 8 with
  accumulation 4;
- a completed one-epoch, single-seed controlled P2 A/B whose exact official-test
  result showed no P2 benefit and does not close the production long-run gate;
- a maintained MNIST workflow and a class-aware AFHQ-v2 showcase contract.

Completion of these contracts does not imply that corrected ADM or P2 has
published long-run quality evidence.

## Active Priority

### P2 — Pixel/P2 Production-Quality Closeout

The current branch closes only the pixel-space P2 and AFHQ Evaluation contract,
tests, documentation, and readiness evidence. After that branch merges, the
current experiment priority is a production P2 candidate long run. No other
task or method implementation belongs to this branch or the post-merge
experiment window.

Repository merge evidence:

- canonical FID/KID provider, preprocessing, reference, and protocol identity;
- aggregate and class-aware AFHQ profiles with immutable result bundles;
- deterministic live/offline parity and completeness failure tests;
- target-device capacity evidence for the corrected ADM configuration;
- a controlled one-epoch P2 comparison after checkpoint, data, sampler, and
  metric protocols are frozen.

Post-merge experiment exit evidence:

- execute the checked-in
  [`p2-production-closeout-policy.yaml`](examples/showcases/afhq-v2/experiments/evaluation/p2-production-closeout-policy.yaml),
  which freezes eligible epochs 20, 40, ..., 200, validation aggregate FID
  (lower) as primary, aggregate KID mean (lower) then earliest epoch as
  tie-breakers, and seed `20260726` before training or selection;
- run the maintained `train-adm-128-p2.yaml` P2 `gamma: 1` candidate for
  200 epochs / 84,000 optimizer updates with the measured batch 8 /
  accumulation 4 profile, fail-closed CUDA, and deterministic runtime;
- evaluate eligible epoch EMA checkpoints only with the versioned 300-per-class
  `selection-ddim50-cfg2-validation-epsilon.yaml` profile, record one uniquely
  selected subject, and never use `valid/loss` or official-test results for that
  choice;
- run the full 1,467-example official-test profile exactly once for the frozen
  subject and publish the complete immutable result bundle; acceptance requires
  aggregate FID <= 35, aggregate KID mean <= 0.01, and every class FID <= 65,
  while aggregate FID <= 30 is aspirational;
- do not claim that P2 is superior to standard from this single-arm closeout. A
  later superiority claim requires a matched production `gamma: 0` control that
  changes no other training or evaluation field.

The AFHQ public profile, epsilon/fixed A/B protocol, and bounded P2 readiness
runs are now available. The tiny CUDA lane completed two optimizer updates with
EMA/diagnostic/validation/test/checkpoint lifecycles; the corrected 105,197,187-
parameter topology completed eight BF16 updates plus validation/test/checkpoint
publication at 4.34 compute optimizer steps/s. Those runs remain wiring and
full-topology readiness evidence.

The target-device capacity requirement now has a valid schema-v3 RTX 4090
report for the corrected topology under P2 BF16 training. Micro batches 1, 4,
6, and 8 each
completed 5 warmup plus 25 measured optimizer updates with zero non-finite
observations; the selected 8 / 4 trial sustained 60.068 images/s at 8.260 GiB
peak allocated and 8.506 GiB peak reserved memory. The production ADM recipe
therefore uses micro batch 8 / accumulation 4 while retaining 420 updates per
epoch and 84,000 total updates. This closes operational capacity selection.

The controlled protocol run is also complete. It changed only P2 `gamma` from
0 for the strict-standard epsilon control to 1 for treatment, used one full
epoch (420 optimizer updates) and each arm's `latest.pt` EMA, then evaluated the
same 1,467 official-test sample IDs with deterministic DDIM-50, eta 0, and CFG
2.0. Aggregate FID moved 369.621427 -> 371.250343 and KID mean moved
0.476357937 -> 0.479742199; all class scopes were also slightly worse for P2.
The KID change is of the same order as its reported standard deviation, and one
seed/epoch is not statistical significance or 200-epoch promotion evidence.
Engineering readiness is closed; the production long-run quality gate remains
open and is the next experiment after merge.

No corrected-ADM production long-run absolute-quality baseline may be claimed
before that evidence gate closes. P2 superiority remains unavailable without
the separately matched `gamma: 0` production control described above.

## After P2

No implementation milestone is currently selected after the production P2
evidence. Codec/latent diffusion, consistency methods, super-resolution,
distillation, Stable Diffusion integration, and new algorithm families are
neither `Next` nor `Planned`; they require a fresh roadmap decision after the P2
result is published.

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
