# AFHQ-v2 Learned-Range-v Closeout

- Status: running; epoch-130 production checkpoint-selection Evaluation verified
- Scope: AFHQ-v2 128x128, ordinary pixel-space class-conditional generation
- Recorded: 2026-08-06

This record tracks the remaining quality gate for the maintained pixel-space
ADM workflow. The run manifest, metrics log, and checkpoint selection state are
the authoritative live-validation evidence; the eventual standalone official
test publishes the immutable Evaluation bundle. Training diagnostics in this
document are observational only.

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

## Bounded runtime preflight

The schema-v3 capacity report
`G:\stochaflow\outputs\benchmarks\afhq-v2\learned-range-v-midres-four-level-4090.json`
matches the production candidate's capacity-critical path: `[1,2,3,4]`, minimum
resolution 16, attention at 32/16, six outputs, BF16, batch 8, and accumulation
4. It records 100,351,366 parameters and 25 successful measured optimizer
updates at 1.411495 updates/s (45.16785 images/s), with 10,759,796,224 bytes
peak allocated, 11,226,054,656 bytes (10.455 GiB) peak reserved, and zero
non-finite losses or gradients. The underlying trial metrics are at
`G:\stochaflow\outputs\afhq-v2\capacity-audit\midres-four-level\batch-8\bf16-mixed\metrics.jsonl`.
The capacity trial alters non-capacity cadence and Evaluation settings, so it is
not a byte-identical production recipe or quality result.

The one-step fresh smoke run
`G:\stochaflow\outputs\afhq-v2\adm-128-learned-range-v-midres-capacity-smoke\20260806_032129`
also exercised the complete 900-sample live validation Evaluation. It completed
with all 12 required aggregate/per-class FID/KID observations and emitted
`best/epoch: 1`; its checkpoint directory was subsequently removed by the
approved checkpoint cleanup. This proves the Evaluation-to-best-checkpoint
lifecycle can complete, not that a one-step random-initialized model has useful
quality or selection authority.

That smoke run took about 28 minutes end to end. Independently, the production
epoch-70 DDPM-100 diagnostic measured 0.53989 samples/s, which projects 900
samples to about 27.8 minutes before conservative publication overhead. Allow
roughly 28--35 minutes for each due production Evaluation; paused epoch metrics
or stdout while the runtime and GPU remain active during that window is not by
itself evidence of a hang.

## Validation checkpoint selection

Epochs 100, 110, ..., 200 execute the same complete live Evaluation against
the current EMA snapshot. The frozen profile is:

- builder: `afhq-v2.class-conditional-generation`;
- protocol: `afhq-v2-adm-learned-range-v-ddpm100-validation-v1`;
- persisted live profile digest:
  `bc9bf92cf2a479a3dcbdee614feb720d0d979d065a8a0ce14ee4d633e1bef42a`;
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
The digest and complete metric surface above were already persisted in the
format-v12 `latest.pt` strict-resume state before the first due Evaluation;
`last_evaluated_epoch: null`, empty last metrics, and zero monitor observations
are required until epoch 100 completes.

| Epoch | Complete examples | Aggregate FID | Aggregate KID mean | `best.pt` after epoch | Evidence status |
| ---: | ---: | ---: | ---: | :---: | --- |
| 100 | 900 | 31.802366 | 0.003904 | 100 | verified |
| 110 | 900 | 30.408995 | 0.003643 | 110 | verified |
| 120 | 900 | 29.181736 | 0.003361 | 120 | verified |
| 130 | 900 | 28.108370 | 0.003002 | 130 | verified |
| 140 | pending | pending | pending | pending | pending |
| 150 | pending | pending | pending | pending | pending |
| 160 | pending | pending | pending | pending | pending |
| 170 | pending | pending | pending | pending | pending |
| 180 | pending | pending | pending | pending | pending |
| 190 | pending | pending | pending | pending | pending |
| 200 | pending | pending | pending | pending | pending |

Do not populate this table from Diagnostics, ordinary validation loss, partial
sampling, or a different profile.

The epoch-100 Evaluation completed at global step 42,000. Its complete metric
surface was:

| Scope | FID | KID mean | KID std |
| --- | ---: | ---: | ---: |
| aggregate | 31.802366 | 0.003904 | 0.000854 |
| cat | 32.745491 | 0.008232 | 0.001078 |
| dog | 70.156059 | 0.025607 | 0.001863 |
| wild | 22.746731 | 0.004202 | 0.000919 |

The strict 900-example Evaluation returned successfully under the frozen
profile, after which `best.pt`, `epoch_0100.pt`, and `latest.pt` were published
in that order. All three are format-v12 epoch-100 / step-42,000 checkpoints,
contain identical epoch metrics, record the same 12 validation observations,
and pass the current strict-resume parser. `best.pt` records aggregate FID as a
lower-is-better monitor with one observation and epoch 100 as the current best.
The epoch-100 Diagnostic manifest reports `errors: []`; its EMA DDPM-100 and
DDIM-50 grids are class-correct and free of systematic color noise or visible
mode collapse. Dog remains the weakest class by a substantial FID margin, so
the result is an initial selection baseline rather than a final checkpoint.

The epoch-110 Evaluation completed at global step 46,200 with the same strict
900-example profile. Its complete metric surface was:

| Scope | FID | KID mean | KID std |
| --- | ---: | ---: | ---: |
| aggregate | 30.408995 | 0.003643 | 0.000860 |
| cat | 32.208645 | 0.007867 | 0.001006 |
| dog | 67.228462 | 0.023031 | 0.001784 |
| wild | 21.489483 | 0.003597 | 0.000850 |

All four FID scopes improved over epoch 100. Aggregate FID decreased by
1.393371 and aggregate KID mean decreased by 0.000261. `best.pt` therefore
advanced atomically to epoch 110 / step 46,200, with two monitor observations,
zero observations without improvement, and the same 12 metrics in its
top-level epoch record and strict-resume validation state. The subsequent
`latest.pt` snapshots through epoch 119 retained epoch 110 as the selected
best. The epoch-110 Diagnostic manifest reports `errors: []`; reconstruction,
DDPM-100, and DDIM-50 tensor/PNG pairs are complete. Dog remains the weakest
class, although its FID improved by 2.927597.

The epoch-120 Evaluation completed at global step 50,400 with the same strict
900-example profile. Its complete metric surface was:

| Scope | FID | KID mean | KID std |
| --- | ---: | ---: | ---: |
| aggregate | 29.181736 | 0.003361 | 0.000837 |
| cat | 32.085747 | 0.007348 | 0.000929 |
| dog | 64.840919 | 0.020436 | 0.001732 |
| wild | 20.049139 | 0.002766 | 0.000741 |

All four FID scopes improved again over epoch 110. Aggregate FID decreased by
1.227259 and aggregate KID mean decreased by 0.000282. `best.pt` therefore
advanced atomically to epoch 120 / step 50,400 with three complete monitor
observations and the same persisted profile digest and 12-key metric surface.
The epoch-120 Diagnostic manifest reports `errors: []`; reconstruction,
DDPM-100, and DDIM-50 tensor/PNG pairs are complete. Both sampling grids remain
class-correct and free of systematic color noise or visible mode collapse. Dog
remains the weakest class but improved by 2.387543 FID from epoch 110.

The epoch-130 Evaluation completed at global step 54,600 with the same strict
900-example profile. Its complete metric surface was:

| Scope | FID | KID mean | KID std |
| --- | ---: | ---: | ---: |
| aggregate | 28.108370 | 0.003002 | 0.000812 |
| cat | 31.392347 | 0.006449 | 0.000871 |
| dog | 62.419773 | 0.018274 | 0.001614 |
| wild | 19.828720 | 0.002645 | 0.000760 |

All four FID scopes improved for a third consecutive observation. Relative to
epoch 120, aggregate FID decreased by 1.073366 and aggregate KID mean decreased
by 0.000360; cat, dog, and wild FID decreased by 0.693399, 2.421146, and
0.220419. `best.pt` therefore advanced atomically to epoch 130 / step 54,600
with four complete monitor observations. The epoch-130 Diagnostic manifest
reports `errors: []`; its reconstruction, DDPM-100, and DDIM-50 artifacts are
complete. Both 12-sample grids remain class-correct and show no systematic hue
noise or visible mode collapse. Against the same-seed epoch-130 DDIM-50 tensors,
learned range differs from current fixed ADM by MSE 0.017298 / cosine 0.949176
and from legacy ADM by MSE 0.016585 / cosine 0.950886. The scenes correspond but
are not duplicate outputs. Its reconstruction MSE at panel timesteps 100, 500,
and 900 is 0.002005, 0.010548, and 0.064367; these four-example Diagnostic
measurements are observations only and have no checkpoint-selection authority.
Training continued normally after publication.

### Interim comparison through epoch 130

The closest retained fixed-variance current-ADM validation result is epoch 200
at
`G:\stochaflow\outputs\afhq-v2\evaluations\current-adm-cosine-v-20260805\validation\epoch_0200`.
It used the same validation data identity, 900-example allocation, EMA, CFG,
seed, and FID/KID provider parameters, but used fixed variance, DDIM-50, and a
sampling batch of 30 rather than learned range, DDPM-100, and a batch of 15. Its
aggregate FID/KID mean was 30.577572 / 0.005761; cat, dog, and wild FID were
34.594212, 62.956085, and 24.328765. At epoch 130, learned range is numerically
lower by 2.469202 aggregate FID and 0.002760 KID mean, and its cat, dog, and wild
FID are lower by 3.201865, 0.536312, and 4.500045. The sampler, topology,
variance, and optimizer-budget differences still prevent a strict ranking.

The strongest retained legacy-ADM result is epoch 170 at
`G:\stochaflow\outputs\afhq-v2\evaluations\formal\epoch_0170`, with aggregate
FID/KID mean 30.239530 / 0.005310. It instead used a 300-per-class subset of the
official test split, DDIM-50, and the older metric profile with KID subset size
300. It is historical context, not a directly comparable validation result.

The learned-range and current fixed-variance epoch-100 DDIM-50 Diagnostics do
share EMA, CFG 2.0, seed, step count, and 12-sample class allocation. Their
common random noise yields closely corresponding scenes, but the retained
tensors are distinct (MSE 0.015472, cosine similarity 0.950862). Both grids are
clean and class-correct; the learned-range grid is slightly sharper by visual
inspection, but 12 Diagnostic examples have no ranking authority. The nearest
retained legacy grid is epoch 130 and has 30% more optimizer steps, so it is not
an equal-budget comparison. Learned-range DDPM-100 looks stronger than its own
DDIM-50 observation, especially for dog, but neither historical baseline has a
matching DDPM-100 artifact.

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

### Published AFHQ benchmark boundary

The commonly cited StarGAN v2 AFHQ number is not a noise-to-image generation
benchmark. The accepted paper reports latent-guided FID 16.3 / LPIPS 0.451 and
reference-guided FID 19.7 / LPIPS 0.432; the official repository's ten-seed
repeat reports 16.18 +/- 0.15 / 0.4501 +/- 0.0007 and 19.78 +/- 0.01 /
0.4315 +/- 0.0002, respectively. These values measure 256x256 multi-domain
image translation, not 128x128 class-conditional generation from noise. See
the [accepted paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.pdf)
and [official reproduction table](https://github.com/clovaai/stargan-v2/blob/master/README.md).

For each of the six directed source-to-target domain pairs, the StarGAN v2
protocol translates 500 source-domain test images ten times, compares the
resulting 5,000 fakes with target-domain *training* images, and then averages
the six FIDs. It uses a project-specific torchvision Inception-v3 pipeline and
repeats the experiment across ten seeds. The current Stochaflow protocol will
instead evaluate one frozen EMA checkpoint once, generating 1,467 images from
noise against the AFHQ-v2 official-test split, reporting pooled and per-class
TorchMetrics/torch-fidelity FID/KID. The task, conditioning, real-reference
split, resolution, fake count, aggregation, feature pipeline, repetition, and
dataset version all differ. The exact translation protocol is documented in
the [supplement Appendix C](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Choi_StarGAN_v2_Diverse_CVPR_2020_supplemental.pdf),
[official evaluation loop](https://github.com/clovaai/stargan-v2/blob/master/metrics/eval.py),
and [FID implementation](https://github.com/clovaai/stargan-v2/blob/master/metrics/fid.py).

Consequently, StarGAN v2's FID 16.3 (or 16.18 repeat) is historical context,
not a target line or a numerically rankable comparator for this run. A strict
comparison would require a separate 256x256 source-conditioned translation
model and the original six-direction, target-train-reference, 5,000-fake per
pair, ten-seed evaluation protocol. The original paper used the pre-2021 AFHQ
release; the official repository notes that AFHQ-v2 changed resampling and file
format and reduced the dataset from 16,130 to 15,803 images.

The P2 paper is closer in algorithm family but still does not provide a
comparable benchmark. Its AFHQ experiment is 256x256 *AFHQ-Dogs* single-class
unconditional generation with a linear beta schedule, epsilon prediction,
learned-range variance, 2.4 million training images, and 50,000 generated
samples. Table 1 reports FID and KID (the displayed KID values are scaled by
1,000): 12.47 / 4.79 for the 1,000-step baseline and 11.55 / 4.10 for P2; at
250 sampling steps it reports 12.95 / 5.25 and 11.66 / 4.20. See the
[P2 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.pdf),
[supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Choi_Perception_Prioritized_Training_CVPR_2022_supplemental.pdf),
and [official implementation](https://github.com/jychoi118/P2-weighting).
The present run instead uses AFHQ-v2 cat/dog/wild class conditioning at 128x128,
cosine, v-prediction plus learned range, DDPM-100, and 900 validation or 1,467
official-test samples. Its FID therefore must not be ranked against the paper's
11.55. Those results remain recipe context only and do not reopen the retired
P2 implementation scope.

## Scope boundary

This closeout covers only ordinary pixel-space image generation. It does not
authorize latent, codec, consistency, super-resolution, or distillation work,
nor compatibility scaffolding for those future tasks.
