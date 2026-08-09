# Stochaflow Roadmap

> Document status: Maintained
>
> Product horizon: pre-1.0
>
> In progress: None
>
> Next: None
>
> Last reviewed: 2026-08-09

This is the canonical product roadmap. It records what is finished, what may be
chosen next, and what is intentionally waiting. It does not define current
behavior. Current contracts live in [`SPEC.md`](SPEC.md),
[`ARCHITECTURE.md`](ARCHITECTURE.md), the public documentation, and tests. The
subordinate [development roadmap](docs/development/development-priority-roadmap.md)
explains dependencies and links to design plans, but cannot select work.

## Status model

| Status | Meaning |
| --- | --- |
| Done | Implemented, tested, documented, and maintained. |
| In progress | The one selected item being implemented now. |
| Next | The approved item that directly follows the work in progress. |
| Candidate | A realistic choice for the next product decision, with no schedule yet. |
| Parked | A preserved longer-term idea whose start condition has not been met. |

Every item has one status. A detailed plan, experiment, estimate, example, or
reusable building block does not make a complete user workflow Done.

## Product direction

Stochaflow is a trustworthy composition and execution layer for generative-model
research. It develops one small but complete user path at a time. A new task
must include the monitoring, checkpoint-backed inference, and formal Evaluation
needed to validate that task. Core does not add task branches or broad public
abstractions before a real use case needs them.

## Done: maintained capabilities

| Capability | What is available now |
| --- | --- |
| Operations | Independent `init`, `train`, `sample`, and `evaluate` commands with separate configuration ownership. |
| Pixel Gaussian | Training and sampling with fixed or learned-range variance, DDPM/DDIM, CFG, and maintained MNIST/AFHQ paths. |
| Training | Builder/Plan/Strategy composition, one automatic optimizer lifecycle, EMA, checkpoints, diagnostics, structured outcomes, and validation-only selection. |
| Sampling | Checkpoint-v12 inference loading, a complete sample configuration for each run, and atomic output publication. |
| Evaluation | Checkpoint and prediction-file subjects, live/offline execution, exact completeness, immutable results, FID/KID, and epoch-end checkpoint selection. |
| Data and extensions | Verified data artifacts, task-level DataBuilders, explicit extension activation, and stable public extension contracts. |

Some useful pieces are narrower than a complete workflow. The repository has a
super-resolution data recipe and example, but not a maintained built-in
super-resolution workflow. Frozen-teacher distillation can be expressed through
extension contracts, but there is no built-in path that trains a teacher and
hands its output to a distillation run. Structured training outcomes exist, but
a public library call and multi-step workflow description do not.

## Selected work

The header records the only scheduling values: no work is currently in progress
and no next item has been selected. The completed pixel-space work is waiting
for a maintainer product decision.
Maintenance, bug fixes, and documentation cleanup may continue without choosing
a new product direction.

## Candidate: realistic next choices

| Direction | First complete user result | Start when | Detail |
| --- | --- | --- | --- |
| Built-in recipes and explicit multi-step workflows | Discoverable task recipes plus one selected typed sequence; both `train -> export -> distill -> evaluate` and `generate -> super-resolution -> evaluate` remain preserved outcomes | One concrete task is selected | [Plan](docs/development/default-workflow-pipeline-support-plan.md) |
| Super-resolution | A tested deterministic restoration workflow before a conditional Gaussian version | Data, metrics, output files, and Evaluation rules are agreed | [Plan](docs/development/super-resolution-workflow-support-plan.md) |
| Consistency and distillation | One teacher/student task with correct target updates and formal results | A research case and Evaluation rules are selected | [Plan](docs/development/consistency-distillation-support-plan.md) |
| Codec and latent diffusion | A frozen codec followed by AFHQ latent training, sampling, and Evaluation | The latent route is selected and codec assumptions are checked again | [Plan](docs/development/latent-diffusion-support-plan.md) |
| Hydra training configuration | Readable reusable fresh-training configuration that still uses Stochaflow validation and execution | A selected product task needs composition beyond plain YAML and the shared single-run training call exists | [Plan](docs/development/hydra-configuration-composition-migration-plan.md) |

## Parked: longer-term directions

| Direction | Start when | Detail |
| --- | --- | --- |
| Stable Diffusion | Shared codec and latent production behavior is Done, then this direction is selected | [Plan](docs/development/stable-diffusion-component-native-support-plan.md) |
| Sampling configuration review | Hydra training composition is Done and real sampling-call problems are observed | [Review](docs/development/sampling-request-config-refactor.md) |
| Evaluation cache, speed, or comparison policy | Measurements show repeated reference cost, slow execution, or a real policy owner | [Decision record](docs/development/post-training-evaluation-support-plan.md) |
| Extension import performance | Independent measurements show a user-visible import or activation problem | [Plan](docs/development/extension-import-boundary-and-activation-latency-plan.md) |
| Distributed execution | A validated single-device workload misses measured capacity or time requirements | [Plan](docs/development/distributed-training-and-inference-support-plan.md) |
| Automated model tuning | A stable objective, budget, Evaluation protocol, and reusable single-run library call exist | [Plan](docs/development/automated-model-tuning-plan.md) |
| Broader artifact metadata, provenance, or capacity models | The metadata, provenance, or resource-evidence direction independently meets its own multi-producer or multi-consumer evidence requirement | [Proposal](docs/development/artifact-metadata-provenance-capacity-model-proposal.md) |
| Broader data abstractions | Multiple data sources or task recipes need the same new extension behavior | [Done boundary record with reopen conditions](docs/development/data-layer-composition-boundary-review.md) |
| New algorithm families | A complete use case proves that current family contracts cannot express it | [Specification](SPEC.md#16-future-compatible-directions) |
| General workflow orchestrator | At least two stable multi-step workflows share the same control behavior and manual composition is a maintenance problem | [Plan](docs/development/general-workflow-orchestrator-plan.md) |

## History

The completed main path was:

```text
pixel-space Gaussian foundation
    -> standalone Evaluation
    -> epoch-end Evaluation and metric-selected checkpoints
    -> AFHQ learned-range-v quality validation
    -> current product decision
```

Detailed experiment numbers belong in the AFHQ public documentation and
`CHANGELOG.md`, not in scheduling text. The retired **AFHQ-v2 Gaussian SNR
loss-weighting experiment (gamma 1 versus gamma 0)** showed no verified benefit
and is not a supported recipe. Any renewed SNR-weighting proposal needs a new
product decision, a namespaced task recipe, and matched formal validation. Old
plan identifiers are explained only in the
[historical plan map](docs/development/notes/history/milestone-id-map.md).

## When work becomes Done

A major item becomes Done only when it has:

1. a stable user-visible contract in `SPEC.md`;
2. clear ownership in `ARCHITECTURE.md` when responsibilities change;
3. implementation through the same public construction paths used by built-ins
   and extensions;
4. focused success, failure, and extension tests where relevant;
5. stable user documentation and a maintained example where appropriate;
6. a `CHANGELOG.md` `Unreleased` entry; and
7. measured evidence for quality, performance, or capacity claims.

## Updating this roadmap

- Change this file only when product selection, ordering, status, or completion
  evidence changes.
- Keep implementation details in the owning development plan.
- Keep exactly one In progress item and one Next item, or write `None`.
- Preserve Candidate and Parked ideas until a maintainer reviews them.
- Never use roadmap status as the sole evidence that code exists or works.
