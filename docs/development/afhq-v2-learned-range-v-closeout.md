# AFHQ-v2 Learned-Range-v Closeout

- Status: running; formal checkpoint-selection evidence has not started
- Scope: AFHQ-v2 128x128, ordinary pixel-space class-conditional generation
- Recorded: 2026-08-06

This record tracks the remaining quality gate for the maintained pixel-space
ADM workflow. The run manifest and published Evaluation bundles are the
authoritative evidence; training diagnostics in this document are observational
only.

## Frozen run identity

- run:
  `G:\stochaflow\outputs\afhq-v2\adm-128-learned-range-v\20260806_035926`;
- launch:
  `G:\stochaflow\outputs\afhq-v2\adm-128-learned-range-v-launch-20260806_035723-1ae02d0`;
- launch source: `1ae02d09d464001274a8c7147b8537b6cc84597d`;
- production YAML SHA256:
  `766DB805E9862C2CE24C8B2F90BC133ED0C23A055EBBC19BC79160C30D8D74D6`;
- pre-record branch documentation baseline:
  `7a81775a7f60f3225db793953f698b7e58d2e1d7`.

The later branch baseline contains documentation and registry-propagation
changes. It is not the training source and must not replace the launch source
in experiment provenance. The run is a fresh random initialization
(`config_source: external`, `lineage.resumed_from: null`) with seed `20260726`,
target epoch 200, and `deterministic: false`.

The data binding is `afhq-v2.official`, partition seed
`stochaflow-afhq-v2-validation-v1`, with 300 held-out validation images per
class. Its managed artifact identity is:

- artifact digest:
  `b7a3350bcae8e53a43aa046ed6e7df81db360fa9002501dd26d99e0be0a25cc9`;
- content digest:
  `8af25d20af76aded630e3e0f4f6d55919c433ec1f3fd7484400742b3399b7cd8`;
- manifest SHA256:
  `b95c5435d34175000ea8b9f5bed987a372558da90cea641adb10bf927ae5b820`.

## Frozen training recipe

- current canonical ADM input/output-block topology, base channels 128,
  channel multipliers `[1,2,3,4]`, two residual blocks, attention at 32 and 16,
  and six output channels;
- cosine alpha-bar Process with 1,000 timesteps, `s: 0.008`, and
  `max_beta: 0.999`;
- v-prediction with learned-range variance and condition dropout 0.1;
- AdamW at `1e-4`, batch 8, accumulation 4, BF16 mixed precision, EMA 0.9999,
  and 200 epochs / 84,000 optimizer steps;
- checkpoint cadence every 50 epochs.

Training observation runs every 10 epochs with EMA, CFG 2.0, seed 20260726,
DDPM 100 and deterministic DDIM 50 sample grids, plus reconstruction panels.
Diagnostic manifests must report `errors: []`. These artifacts may reveal gross
quality failure but have no checkpoint-selection or benchmark authority.

## Validation checkpoint selection

Epochs 100, 110, ..., 200 execute the same complete live Evaluation against
the current EMA snapshot. The frozen profile is:

- builder: `afhq-v2.class-conditional-generation`;
- protocol: `afhq-v2-adm-learned-range-v-ddpm100-validation-v1`;
- sampling: EMA, DDPM 100, CFG 2.0, clipping enabled, seed 20260726;
- sample plan: 900 images in batches of 15, exactly 300 for each of cat, dog,
  and wild;
- metrics: Inception-2048 FID and KID with 100 subsets of size 200;
- completeness: exactly 900 unique examples, strict failure on incomplete
  output.

The required validation observations are:

```text
valid/metrics/distribution/aggregate.fid
valid/metrics/distribution/aggregate.kid_mean
valid/metrics/distribution/aggregate.kid_std
valid/metrics/distribution/cat.fid
valid/metrics/distribution/cat.kid_mean
valid/metrics/distribution/cat.kid_std
valid/metrics/distribution/dog.fid
valid/metrics/distribution/dog.kid_mean
valid/metrics/distribution/dog.kid_std
valid/metrics/distribution/wild.fid
valid/metrics/distribution/wild.kid_mean
valid/metrics/distribution/wild.kid_std
```

`valid/metrics/distribution/aggregate.fid` is the sole `best.pt` monitor and is
minimized. KID and per-class values are recorded evidence but do not
independently select a checkpoint. Early stopping is disabled. An epoch outside
the configured cadence must not reuse a stale observation or update `best.pt`.

| Epoch | Complete examples | Aggregate FID | Aggregate KID mean | `best.pt` after epoch | Evidence status |
| ---: | ---: | ---: | ---: | :---: | --- |
| 100 | pending | pending | pending | pending | pending |
| 110 | pending | pending | pending | pending | pending |
| 120 | pending | pending | pending | pending | pending |
| 130 | pending | pending | pending | pending | pending |
| 140 | pending | pending | pending | pending | pending |
| 150 | pending | pending | pending | pending | pending |
| 160 | pending | pending | pending | pending | pending |
| 170 | pending | pending | pending | pending | pending |
| 180 | pending | pending | pending | pending | pending |
| 190 | pending | pending | pending | pending | pending |
| 200 | pending | pending | pending | pending | pending |

Do not populate this table from Diagnostics, ordinary validation loss, partial
sampling, or a different profile.

## Completion and final-test gate

Training closes only after the manifest reports `completed` with final epoch
200 and `latest.pt`, `epoch_0200.pt`, the validation-selected `best.pt`, and the
epoch-200 diagnostic artifacts are complete. The final validation Evaluation
must contain all 12 observations and strict 900-example completeness evidence.

Only then replace the placeholder subject in
`examples/showcases/afhq-v2/experiments/evaluation/formal-ddpm100-cfg2-official-test-learned-range-v.yaml`
with the exact selected checkpoint. Execute that frozen EMA/DDPM-100/CFG-2
profile once on all 1,467 official-test images (493 cat, 491 dog, 483 wild) and
publish its immutable result bundle. Official-test data and metrics must not be
read or used during checkpoint selection.

## Comparison boundary

The fixed-variance current-ADM record is
`G:\stochaflow\outputs\afhq-v2\adm-128-current-cosine-v\20260804_222432`.
It used the five-level `[1,1,2,3,4]` topology, attention at 32/16/8, three model
outputs, and fixed variance. The legacy-ADM record is
`G:\stochaflow\outputs\afhq-v2\adm-128\20260729_221804`; it used the legacy
four-level `[1,2,3,4]` graph, three model outputs, and fixed variance. Both used
cosine and v-prediction, but their historical diagnostic and post-training
Evaluation protocols are not the learned-range validation profile above.

Therefore the eventual three-way report must state topology, capacity,
variance, sample count, split, sampler, step count, CFG, checkpoint, and metric
provider differences beside every number. It may establish end-to-end product
quality under disclosed protocols; it must not claim an isolated learned-range,
topology, or capacity effect.

## Scope boundary

This closeout covers only ordinary pixel-space image generation. It does not
authorize latent, codec, consistency, super-resolution, or distillation work,
nor compatibility scaffolding for those future tasks.
