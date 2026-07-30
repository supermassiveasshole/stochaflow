# Stochaflow

Stochaflow is a configuration-driven, extensible research framework for
generative modeling. It provides the reusable workflow around an experiment:
data preparation, component composition, training, strict resume,
checkpoint-backed inference, diagnostics, logging, and output artifacts.

The maintained implementation currently focuses on **pixel-space discrete
Gaussian diffusion**:

- unconditional generation with a UNet;
- class-conditional generation with an ADM-UNet or pixel-space DiT;
- epsilon, x0, v, and score prediction targets;
- DDPM and DDIM sampling, EMA weights, trajectories, and classifier-free
  guidance.

Stochaflow is pre-1.0 research software and currently permits breaking changes.
Latent diffusion, pretrained autoencoder integration, Stable Diffusion
components, flow matching, and distributed training are development directions,
not implemented capabilities. See the [framework overview](docs/framework.md)
for current behavior and the [architecture scope](docs/design/scope.md) for the
long-term boundary.

## Quick start

Stochaflow requires Python 3.12 or newer. From a source checkout:

```bash
uv sync --extra dev
```

Run a bounded MNIST training job:

```bash
uv run stochaflow train \
  --config examples/built-in/image-generation/configs/train/mnist.yaml \
  --epochs 1 \
  --limit-batches 10 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

The run is written below `outputs/mnist/<run>/`. Sample from its best
checkpoint with either maintained profile:

```bash
uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml

uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddim-50.yaml
```

The training command may download and materialize MNIST under `./data`. The two
sample profiles reuse the same trained checkpoint; choosing DDPM or DDIM does
not require retraining.

## Current capability

| Area | Implemented today |
| --- | --- |
| Data | Verified managed and referenced data artifacts; image, class-labeled image, super-resolution data, and multi-resolution image recipes |
| Models | Unconditional UNet, class-conditional ADM-UNet, and class-conditional pixel-space DiT |
| Training | Unconditional and class-conditional Gaussian denoising, supervised training, mixed precision, gradient accumulation, EMA, and a single-optimizer automatic loop |
| Probability process | Discrete variance-preserving Gaussian process with linear-beta and cosine-alpha-bar schedules |
| Sampling | DDPM, DDIM, class allocation, classifier-free guidance, trajectory observation, and Tensor/PNG/GIF writers |
| Runtime | Registry-based composition, explicit extension activation, checkpoint v10 strict resume, checkpoint-backed inference, local/TensorBoard/W&B logging, and training diagnostics |
| CLI | `stochaflow init`, `stochaflow train`, and `stochaflow sample` |

The built-in `super_resolution` capability covers paired data and degradation
recipes. A complete conditional super-resolution model, training strategy, and
sampling composition remain project responsibilities.

Standard PyTorch optimizers and learning-rate schedulers are selected through
validated native `torch.optim.*` and `torch.optim.lr_scheduler.*` targets.
Stochaflow does not copy the upstream namespaces or every constructor parameter
into its own Registry.

## Maintained examples

The actively maintained example surface is deliberately small:

| Example | Role | Current status |
| --- | --- | --- |
| [MNIST](examples/built-in/image-generation/README.md) | Minimal built-in image-generation workflow | One train config, DDPM/DDIM sample profiles, and a resume observability overlay |
| [AFHQ-v2](examples/showcases/afhq-v2/README.md) | Installable class-conditional data-source showcase | ADM-UNet and pixel DiT production configs; published results below are from ADM-UNet |

Separate CIFAR-10, Flowers102, and multi-source training YAMLs are not maintained.
Their removal does not remove the underlying reusable data sources or recipes.
The Physics reconstruction and knowledge-distillation directories are legacy
reference implementations retained temporarily for cleanup; they are outside
the maintained example and compatibility surface.

### MNIST DDPM and DDIM

The reference MNIST run completed 200 epochs and 78,000 optimizer updates.
Checkpoint selection used validation denoising loss; the minimum occurred at
epoch 183.

| Evaluation | Result |
| --- | ---: |
| Selected checkpoint | `best.pt`, epoch 183 / step 71,370 |
| Best validation loss | **0.07189** |
| Test loss after restoring the best checkpoint | **0.07363** |

These are v-prediction denoising losses, not perceptual-quality metrics. The
panels use the same EMA checkpoint, seed, and terminal-noise batch. DDPM performs
1,000 model evaluations; deterministic DDIM uses 50.

| DDPM, 1,000 evaluations | DDIM, 50 evaluations |
| :---: | :---: |
| <img src="assets/readme/mnist_ddpm_epoch_0183_samples.png" width="300" alt="Thirty-six MNIST samples generated with DDPM from the epoch-183 checkpoint"> | <img src="assets/readme/mnist_ddim50_epoch_0183_samples.png" width="300" alt="Thirty-six MNIST samples generated with DDIM-50 from the epoch-183 checkpoint"> |

| DDPM trajectory | DDIM-50 trajectory |
| :---: | :---: |
| <img src="assets/readme/mnist_ddpm_epoch_0183_trajectory.gif" width="300" alt="Animated MNIST DDPM reverse-process trajectory"> | <img src="assets/readme/mnist_ddim50_epoch_0183_trajectory.gif" width="300" alt="Animated MNIST DDIM-50 reverse-process trajectory"> |

The corresponding sample profiles are
[`mnist-ddpm.yaml`](examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml)
and
[`mnist-ddim-50.yaml`](examples/built-in/image-generation/configs/sample/mnist-ddim-50.yaml).
The referenced trained checkpoint is not distributed.

### AFHQ-v2 class-conditional ADM

The published result below was produced by a completed 200-epoch, 84,000-update
ADM-UNet run. After training, periodic EMA snapshots were screened with one
fixed sampling protocol; among the three finalists, aggregate FID was the
primary selection metric and aggregate KID was secondary.

| Evaluation | Result |
| --- | ---: |
| Selected checkpoint | EMA weights from `epoch_0170.pt`, epoch 170 / step 71,400 |
| Aggregate FID | **30.240** |
| Aggregate KID | **0.005310 ± 0.000701** |
| Per-class FID (cat / dog / wild) | **37.965 / 58.565 / 24.352** |

The fixed protocol uses 300 official-test images and 300 generated images per
class, for 900 real and 900 generated images in aggregate. Sampling uses EMA,
deterministic DDIM-50 (`eta: 0`), classifier-free guidance 2.0, and seed
`20260726`. These are 900-sample protocol results, not 50,000-image benchmark
scores. The selected checkpoint SHA-256 is
`ea43404395d884c03fd7b130f407e5ace6c35b2336d2c5bd073f630828c2e4ce`.

The following panel is the complete 36-sample output, not a curated subset.
Rows 1–2 are cat, rows 3–4 are dog, and rows 5–6 are wild:

<p align="center">
  <img src="assets/readme/afhq_v2_adm_ddim50_epoch_0170_samples.png" width="720" alt="Thirty-six independently generated class-conditional AFHQ-v2 samples using EMA, DDIM-50, and classifier-free guidance 2.0">
</p>

The repository includes:

- the measured
  [ADM-UNet training config](examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml);
- a runnable
  [pixel DiT-B/8 training config](examples/showcases/afhq-v2/experiments/production/train-dit-128.yaml),
  for which this README does not claim a completed quality result;
- the exact
  [README sampling profile](examples/showcases/afhq-v2/experiments/sampling/ddim50-cfg2-readme.yaml);
- the
  [KID/FID evaluation profile](examples/showcases/afhq-v2/experiments/evaluation/ddim50-cfg2-kid-fid.yaml).

The trained checkpoint is not distributed in this repository. Reproducing the
sampling or evaluation therefore requires a compatible checkpoint produced by
training. The dataset source is approximately 6.96 GB and is licensed CC BY-NC
4.0; review the [AFHQ-v2 guide](docs/tutorials/afhq-v2.md) before downloading or
training.

## Composition boundaries

Stochaflow standardizes workflow lifecycles without turning YAML into an
arbitrary Python object graph.

### Data

```text
DataSource -> DataArtifact -> DataBuilder -> DataLoaders
```

- `DataSource` acquires, validates, transforms, and materializes external data.
- `DataArtifact` is the verified, identity-bearing result of that producer
  lifecycle.
- `DataBuilder` is the runtime data-recipe composition root. It owns artifact
  binding, partitioning, Dataset views, transforms, PyTorch samplers, collate
  behavior, and loaders.
- Core training code consumes structured batches without imposing image,
  condition, target, or metadata fields.

A new source can reuse an existing DataBuilder only when both its complete
artifact contract and the required runtime recipe semantics are compatible.
Different partition, streaming, sampler, resume, or batch semantics require a
new recipe-level DataBuilder—not a Builder for every Dataset class. See
[Data configuration and pipelines](docs/configuration/data-pipeline.md).

### Training and inference

```text
DataLoaders + configured components
  -> TrainingBuilder
  -> TrainingPlan
  -> Trainer
  -> checkpoint

checkpoint + sample profile
  -> checkpoint-owned SamplingBuilder recipe
  -> Sampler
  -> artifact writers
```

`TrainingBuilder` owns task composition. A thin `TrainingStrategy` interprets
batches and computes loss and metrics; the framework owns device, precision,
optimization, EMA, and checkpoint lifecycles.

`Sampler` owns a complete numerical sampling algorithm. `SamplingBuilder` owns
task concerns such as model adaptation, conditions, guidance, initialization,
and compatibility. The core runner does not maintain a global
model/process/sampler compatibility matrix.

## Configuration, sampling, and resume

The maintained built-in authoring tree is:

```text
examples/built-in/image-generation/configs/
├── train/mnist.yaml
├── sample/mnist-ddpm.yaml
├── sample/mnist-ddim-50.yaml
└── overlays/mnist-observability.yaml
```

The MNIST train config contains one training recipe. The two sample files are
checkpoint-backed profiles with the current top-level `sampling:` request
envelope; they choose request-time solver, output, and writer settings without
selecting an internal SamplingBuilder. Current checkpoint v10 remains
authoritative for the fixed inference recipe.

Strict resume restores the saved configuration and full training state:

```bash
uv run stochaflow train \
  --resume outputs/mnist/<run> \
  --epochs 200
```

`--config` and `--resume` are mutually exclusive. A resume run may replace only
its observation surface through `--observability-config`, for example:

```bash
uv run stochaflow train \
  --resume outputs/mnist/<run> \
  --observability-config \
    examples/built-in/image-generation/configs/overlays/mnist-observability.yaml
```

Only checkpoint format v10 is currently accepted; older formats are not
migrated. Strict resume guarantees only the state explicitly owned by the
framework and does not promise cross-device or cross-version bitwise equality.
See the [configuration handbook](docs/configuration/index.md) and
[checkpoint and portability guide](docs/configuration/compatibility-and-migration.md)
for field definitions, override authority, RNG boundaries, and artifact-binding
rules.

## Create an extension project

Generate a normal installable Python project:

```bash
uv run stochaflow init my-research-project
```

The scaffold includes an extension entry point, a small custom data/training/
sampling vertical slice, a train config, and tests. Stochaflow discovers only
installed entry points selected by configuration; it does not scan arbitrary
source directories. Complex task composition stays in Python Builders and
Strategies rather than a universal YAML graph.

Extension authors should use the stable imports documented in the
[public extension API](docs/api/extensions.md). See
[Extensions and registries](docs/configuration/extensions.md) for activation,
packaging, provenance checks, and the generated project workflow.

## Installation and platforms

The current GitHub Release provides a universal Python wheel, source
distribution, SHA-256 checksums, and GitHub build-provenance attestations. To
install the v0.1.0 wheel directly:

```bash
python -m pip install \
  https://github.com/supermassiveasshole/stochaflow/releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl
```

The assets and generated release notes are on the
[v0.1.0 release page](https://github.com/supermassiveasshole/stochaflow/releases/tag/v0.1.0).
After downloading an asset, its provenance can be checked with GitHub CLI:

```bash
gh attestation verify \
  stochaflow-0.1.0-py3-none-any.whl \
  --repo supermassiveasshole/stochaflow
```

For source development, `uv sync --extra dev` installs Pytest, Ruff, and
Pyright. Optional extras are independent:

| Extra | Purpose |
| --- | --- |
| `quality` | KID/FID dependencies |
| `wandb` | Weights & Biases logging |
| `docs` | Sphinx and documentation tooling |
| `dev` | Tests, linting, and type checking |

TensorBoard is part of the base installation. To build a local wheel and source
distribution:

```bash
uv build
```

Supported validation targets are Linux x86_64, Windows x86_64, and macOS arm64.
Intel macOS is **Deprecated / best effort** and retains a transitional pinned
dependency path; it must not be assumed to support every current feature. The
source checkout routes Windows GPU dependencies through PyTorch's CUDA 12.8
wheel index. See the [platform support policy](docs/platform-support.md) for the
current Python and CI matrix.

`trainer.device: auto` selects CUDA, then Apple MPS, then CPU. Apple MPS cannot
run managed modules with float64 or complex128 parameters or buffers, including
Process state.

## Documentation

- [Framework features and architecture](docs/framework.md)
- [Configuration and workflow handbook](docs/configuration/index.md)
- [Data configuration and pipelines](docs/configuration/data-pipeline.md)
- [Extension API](docs/api/extensions.md)
- [Checkpoint compatibility and migration](docs/configuration/compatibility-and-migration.md)
- [MNIST built-in image-generation workflow](examples/built-in/image-generation/README.md)
- [AFHQ-v2 data and training guide](docs/tutorials/afhq-v2.md)
- [TensorBoard guide](docs/tutorials/tensorboard.md)
- [Troubleshooting](docs/configuration/troubleshooting.md)
- [Architecture scope and non-goals](docs/design/scope.md)

## Development

For a routine change, run focused tests together with Ruff and Pyright:

```bash
uv run pytest tests/test_ddpm_shapes.py tests/test_sampling_runtime.py
uv run ruff check .
uv run pyright
```

Before merging a complete feature branch, run the full verification required by
that feature, including the full test suite when appropriate:

```bash
uv run pytest
```

Generated datasets, checkpoints, and ordinary run outputs belong under
`data/` or `outputs/` and must not be committed.

## License

Stochaflow is released under the [MIT License](LICENSE). Dataset and pretrained
asset licenses remain independent of the framework license.
