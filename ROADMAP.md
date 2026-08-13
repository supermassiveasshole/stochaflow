# Stochaflow Roadmap

> Document status: Maintained
>
> Product horizon: pre-1.0
>
> In progress: None
>
> Next: None
>
> Last reviewed: 2026-08-13

This is the only document that schedules product work. It says what users can
do today, what could be selected next, and what is intentionally waiting.
Implementation details belong in development plans. Current behavior belongs in
[`SPEC.md`](SPEC.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), public documentation,
and tests.

## Status model

| Status | Meaning |
| --- | --- |
| Done | Users can use it through maintained code, tests, and documentation. |
| In progress | The one selected item being implemented now. |
| Next | The one approved item that will start next. |
| Candidate | A real product function that may be selected, but is not scheduled. |
| Parked | A real product function whose required evidence or prerequisite is missing. |

A research question, reusable building block, or detailed design is not a
scheduled product function. It does not appear here until a maintainer selects
a concrete user result.

## What users can do now

| Capability | Maintained user result |
| --- | --- |
| Operations | Run independent `init`, `train`, `sample`, and `evaluate` commands, each with its own complete configuration. |
| Pixel Gaussian models | Train and sample fixed- or learned-range-variance models with DDPM/DDIM and classifier-free guidance on the maintained MNIST/AFHQ paths. |
| Training | Compose a task through Builder/Plan/Strategy, train with one automatic optimizer lifecycle, save resumable checkpoints, track EMA, and receive a structured outcome. |
| Sampling | Load checkpoint-v12 inference state, run one complete sampling request, and publish output atomically. |
| Evaluation | Evaluate a checkpoint or saved prediction artifact, reject missing or duplicate samples, and publish an immutable result. |
| Data | Acquire and verify arbitrary task data through `DataSource`, then assemble training inputs through `DataBuilder`, including strict-resume identity checks. |
| Extensions | Activate installed extensions explicitly and use the same public construction paths as built-ins. |

These lower-level abilities do not yet create every product workflow. For
example, the repository has super-resolution data support but no maintained
built-in super-resolution operation. It can express a frozen teacher, but it
does not yet run “train the teacher, then distill the student” as one documented
workflow.

## Current selection

No product item is in progress and no next item has been selected. Maintenance,
bug fixes, and documentation work may continue without selecting a new product
direction.

## Candidate product functions

| Direction | What the user will do | Start when | Detail |
| --- | --- | --- | --- |
| Explicit multi-step workflows | Run existing operations in order and pass a typed result to the next step, first for either teacher training followed by distillation or image generation followed by super-resolution | The required distillation or super-resolution operation is Done, and its multi-step workflow is selected | [Plan](docs/development/default-workflow-pipeline-support-plan.md) |
| Super-resolution | Train with paired low/high-resolution images, feed in low-resolution images, and receive high-resolution artifacts plus a formal quality result | The first maintained dataset, model, output format, and Evaluation protocol are selected | [Plan](docs/development/super-resolution-workflow-support-plan.md) |
| Consistency and distillation | Give the run a teacher checkpoint and training data, then receive a faster student checkpoint with quality and speed results | One teacher/student task and its success criteria are selected | [Plan](docs/development/consistency-distillation-support-plan.md) |
| Codec and latent diffusion | Encode images with a frozen codec, train in latent space, then decode generated samples and evaluate them | This direction is selected and codec assumptions are checked again | [Plan](docs/development/latent-diffusion-support-plan.md) |
| Hydra training configuration | Choose reusable config fragments, preview the merged config, and launch the existing single training operation | A selected task has a real composition problem that plain YAML cannot solve | [Plan](docs/development/hydra-configuration-composition-migration-plan.md) |

## Parked product functions

| Direction | What the user would eventually do | Start when | Detail |
| --- | --- | --- | --- |
| Stable Diffusion | Supply a compatible component bundle and image-text data, fine-tune the UNet, then generate images from prompts | Latent/codec support is Done and this direction is selected | [Plan](docs/development/stable-diffusion-component-native-support-plan.md) |
| Extension import performance | Start commands and activate extensions with less import delay, without changing extension behavior | Measurements show a user-visible startup problem | [Plan](docs/development/extension-import-boundary-and-activation-latency-plan.md) |
| Fixed single-node distributed training | Run one maintained training workload on a fixed number of Linux GPUs and receive globally consistent validation, a fixed-topology resumable training bundle, and a portable checkpoint | At the required effective global batch and quality constraints, a maintained single-device workload still misses a measured wall-time or throughput requirement, and one exact Linux CUDA/NCCL topology is selected for acceptance | [Plan](docs/development/distributed-training-and-inference-support-plan.md) |
| Portable large-scale data preparation and training | Prepare a resumable, verified dataset once, copy it to another machine, and train from the same data identity using bounded machine-specific storage, host-memory, pinned-memory, and device queues | One finite large-data workload, its preparation semantics and storage format, and both PC and server acceptance environments are selected; later performance layers still require a measured bottleneck | [Plan](docs/development/hierarchical-data-pipeline-support-plan.md) |
| Automated model tuning | Give the system a base training config, parameter choices, and a budget; receive the best checkpoint and its formal Evaluation result | A stable objective, budget, Evaluation protocol, and reusable single-run call exist | [Plan](docs/development/automated-model-tuning-plan.md) |
| General workflow orchestrator | Submit a list of already supported operations, inspect each step, and retry or resume a failed step | At least two maintained workflows repeat the same recovery and state-hand-off code | [Plan](docs/development/general-workflow-orchestrator-plan.md) |

Data-source helpers, streaming data, storage adapters, post-Hydra sampling review,
optional Evaluation enhancements, and the three independent artifact questions
are preserved as research or review notes in the
[development index](docs/development/README.md). They are not product roadmap
items. Data and Evaluation remain Done unless a maintainer selects a new,
concrete user function.

## Completed and retired work

The maintained pixel-space path completed standalone Evaluation, epoch-end
Evaluation and metric-based checkpoint selection, then the AFHQ learned-range-v
quality validation. Detailed measurements live in the AFHQ documentation and
`CHANGELOG.md`.

The retired **AFHQ-v2 Gaussian SNR loss-weighting experiment (gamma 1 versus
gamma 0)** showed no verified benefit and is not a supported recipe. Renewing
that exact idea requires a new product decision, a separately named recipe, and
matched formal validation. The evidence and interpretation are retained in the
[historical decision](docs/development/notes/history/afhq-snr-weighting-decision.md).

Old plan identifiers are kept only in the
[historical plan map](docs/development/notes/history/milestone-id-map.md).

## When work becomes Done

A major item becomes Done only when users have a maintained way to run it, its
behavior and ownership are documented, built-ins and extensions use the same
public boundaries, success and failure tests pass, and any quality, performance,
or capacity claim has measured evidence. The same change must update
`SPEC.md`, `ARCHITECTURE.md`, public documentation, and `CHANGELOG.md` where
applicable.

## Updating this roadmap

- Change this file only when product selection, ordering, status, or completion
  evidence changes.
- Keep at most one In progress item and one Next item, or write `None`.
- Do not promote a research note into this file merely to preserve it.
- A development plan cannot select or schedule itself.
