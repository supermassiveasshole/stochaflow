# Stochaflow

Stochaflow is a research-oriented Python framework for generative modeling
through probability paths, generative dynamics, and numerical samplers. The
current implementation focuses on config-driven DDPM and DDIM training and
sampling for MNIST, CIFAR-10, and Oxford Flowers 102.

The codebase is organized around registries and config-driven components: thin
data builders, models, optional probability processes, complete samplers, task sampling
builders, training builders, artifact writers, objectives, optimizers, diagnostics, and loggers
are selected from YAML.

## Scope

Stochaflow is intended to cover the probability-transport family of generative
methods. Its first-class scope includes diffusion and score-based models,
probability-flow ODEs, flow matching, rectified flows, stochastic interpolants,
continuous flows, and their ODE/SDE/discrete samplers. These methods may use
stochastic or deterministic trajectories, but all transport samples from a
simple reference distribution toward a data distribution.

The project is not intended to define one universal abstraction for every
generative model. Autoregressive models, GANs, VAEs, energy-based models, and
discrete invertible flows may be integrated through compatible extension
lifecycles, but the core runtime will not acquire task-specific branches or
artificial Process/Sampler requirements solely to encompass them.

The architectural boundary is deliberately compositional, but it is not one
universal mathematical interface. The framework standardizes registration,
configuration, checkpointing, and the complete Sampler lifecycle. Each
algorithm family defines the Process and Dynamics capabilities it needs, while a
SamplingBuilder assembles compatible family components with task-specific
models, conditions, guidance, and initial states. Not every method is required
to use every role.

| Layer | Shared responsibility | Deliberately not shared |
| --- | --- | --- |
| Framework | Registry, config, checkpoint, `Sampler.sample()` lifecycle | Mathematical compatibility |
| Algorithm family | Cohesive optional Process, Dynamics, and Sampler contracts | A universal `predict`, `drift`, or `step` API |
| Task | Model adaptation, condition, guidance, initialization, artifacts | Core runner branches |

The implemented Gaussian family uses `DiscreteGaussianDenoisingProcess`,
`GaussianDenoisingDynamics`, and DDPM/DDIM. A future flow-matching or score-SDE
family may define a vector field or reverse SDE with its own compatible solver;
it does not need to implement Gaussian behavior or change the core runtime.

## Status

Implemented:

- DDPM/DDIM epsilon-prediction training
- model-free discrete Gaussian Process with marginal and posterior mathematics
- Gaussian model Dynamics with epsilon/x0/v/score prediction conversion
- registered DDPM and DDIM Samplers with one complete `sample()` interface
- registered SamplingBuilder composition and observer-based trajectories
- beta-native linear and alpha-bar-native cosine noise schedules
- a thin, modality-neutral `DataBuilder -> DataLoaders` extension contract
- built-in image, super-resolution, and multi-resolution image recipes
- torchvision and stable local image-folder sources for built-in recipes
- recipe-local holdout/K-fold, multi-source mixtures, and dynamic pixel batches
- registered tensor, image, and user-defined sampling artifact writers
- registered TrainingBuilder composition with thin task-specific TrainingStrategy
- reusable scalar Objectives and managed auxiliary modules for frozen-teacher distillation
- UNet backbone with optional attention blocks
- EMA tracking and EMA sampling
- warmup-cosine and common PyTorch optimizer LR schedulers
- multi-sampler diffusion diagnostics with denoiser metrics and visual artifacts
- Rich terminal reporting, checkpointing, and local/TensorBoard/W&B logging
- one `stochaflow` CLI with `train` and `sample` subcommands
- pytest, ruff, and pyright configuration

Still evolving:

- faster samplers and broader learned/perceptual evaluation metrics
- multi-optimizer and manual-optimization training loop families
- probability-flow ODE, flow-matching, rectified-flow, and continuous-flow
  implementations beyond the current diffusion baseline

## Installation

This project uses a `src` layout and is designed to work with `uv`.

```bash
uv sync --extra dev
```

KID/FID diagnostics are optional because they add an Inception feature model:

```bash
uv sync --extra dev --extra quality
```

For an editable pip workflow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Notes:

- Python `>=3.12` is required.
- Intel macOS uses Python 3.12 with pinned PyTorch/Torchvision wheels.
- Windows GPU runs are resolved through the PyTorch CUDA 12.8 wheel index and
  can use Python 3.14.
- `trainer.device: auto` selects CUDA first, then Apple MPS when available,
  and otherwise falls back to CPU.

Windows GPU setup example:

```powershell
uv python install 3.14
uv sync --python 3.14 --extra dev
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Training

MNIST smoke run:

```bash
uv run stochaflow train \
  --config configs/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 10
```

CIFAR-10 smoke run:

```bash
uv run stochaflow train \
  --config configs/ddpm_cifar10.yaml \
  --epochs 1 \
  --limit-batches 10 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

Oxford Flowers 102 smoke run:

```bash
uv run stochaflow train \
  --config configs/ddpm_flowers102.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --skip-final-sample
```

Oxford Flowers 102 baseline run:

```bash
uv run stochaflow train \
  --config configs/ddpm_flowers102.yaml
```

Multi-source MNIST + Flowers102 run:

```bash
uv run stochaflow train \
  --config configs/ddpm_mnist_flowers102.yaml \
  --epochs 1 \
  --limit-batches 10
```

Custom data builders are registered as classes and imported through
`extensions.modules`; see [Extensions and registries](docs/configuration/extensions.md).

For the complete YAML schema, defaults, validation rules, built-in component
parameters, multi-source mixing, buckets, K-fold, logging, and CLI overrides,
see [Configuration handbook](docs/configuration/index.md).

If `--epochs` is omitted, the runner uses `trainer.num_epochs` from the YAML
config. Passing `--epochs` is an explicit run-time override.

Useful training options:

```bash
--resume [PATH]               Resume training. Without PATH, use latest.pt.
--device DEVICE               Override trainer.device for this run.
--output-dir PATH             Override experiment.output_dir for this run.
--skip-final-sample           Skip best-checkpoint acceptance sampling.
--no-progress                 Disable Rich terminal progress bars.
```

Each run writes a timestamped directory under the configured output root:

```text
outputs/<experiment>/<YYYYMMDD_HHMMSS>/
  checkpoints/
    best.pt
    latest.pt
    epoch_XXXX.pt
  metrics.jsonl
  train.log
  samples/
    final/
      samples.png
      samples.pt
      resolved_sampling.yaml
  diagnostics/
    diffusion_quality/
      epoch_XXXX/
        manifest.yaml
        denoiser/
        <sampler-profile>/
```

Example MNIST DDPM samples:

![MNIST DDPM generated samples](assets/readme/mnist_ddpm_epoch_0100.png)

MNIST reverse-process trajectory:

![MNIST DDPM reverse trajectory](assets/readme/mnist_ddpm_epoch_0100_trajectory.png)

Example Oxford Flowers 102 DDPM samples:

![Oxford Flowers 102 DDPM generated samples](assets/readme/flowers102_ddpm_epoch_0681.png)

Oxford Flowers 102 reverse-process trajectory:

![Oxford Flowers 102 DDPM reverse trajectory](assets/readme/flowers102_ddpm_epoch_0681_trajectory.png)

## Training diagnostics

The `diffusion_quality` diagnostic can compare multiple sampler profiles against
the same denoiser during training. Profiles receive identical fixed terminal
noise, use EMA weights when configured, and report under independent metric
namespaces. Step hooks record timestep-bucket loss, noise statistics, cosine
similarity, and fixed-timestep reconstruction MSE/PSNR. Epoch hooks write sample,
reconstruction, and trajectory panels and record sample statistics and latency.

The implementation is a provider pipeline under `training/diagnostics/`.
Step metrics, sampler metrics, denoiser artifacts, sampler artifacts, and
reference metrics have separate registries. External modules can register a new
provider and enable it from `diagnostics[].params.modules` without modifying the
`diffusion_quality` orchestrator; explicit empty provider lists disable a phase.

Images are saved locally and forwarded to every configured TensorBoard or W&B
logger. Optional KID/FID evaluation uses a fixed validation reference cache and
is enabled with the `quality` installation extra. See the Flowers102 configs for
a complete DDPM/DDIM comparison declaration.

## Sampling

Sample directly from a portable Stochaflow checkpoint. The checkpoint contains
the model and experiment config, so an extra YAML file is optional:

```bash
uv run stochaflow sample \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

Switch a DDPM-trained denoiser to DDIM by supplying an external sampling YAML:

```bash
uv run stochaflow sample \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --config path/to/sampling.yaml
```

Sampler selection and solver parameters belong to `sampling.builder.params`;
the CLI intentionally has no sampler-specific flags. A custom SamplingBuilder
may instead construct conditions, guidance, multiple Samplers, or an initial
state without requiring `sampling.shape`.

You can alternatively pass only `--config`; Stochaflow then finds the newest
`best.pt` under `experiment.output_dir`. With both inputs, the checkpoint
provides weights while a lightweight external file supplies the complete
`sampling` section and optional `extensions`. A complete external config is also
accepted, with model and Process compatibility checks. Checkpoint and external
`extensions.modules` are merged in stable order before Builder construction.

The standard builder uses EMA model weights when `ema.enabled` and
`ema.use_for_sampling` are both true. Declared `sampling.writers` decide the outputs: `tensor` writes PT
files, `image` validates NCHW data and writes PNG/GIF artifacts, and extensions
can write domain formats such as NetCDF. Every run also writes
`resolved_sampling.yaml`.

## Configuration

Experiment configuration lives under `configs/`.

```text
configs/ddpm_mnist.yaml
configs/ddpm_cifar10.yaml
configs/ddim_cifar10.yaml
configs/ddpm_flowers102.yaml
configs/ddim_flowers102.yaml
configs/ddpm_mnist_flowers102.yaml
```

Important sections:

- `experiment`: run name, seed, and output directory
- `data`: registered builder name plus builder-owned parameters
- `model`: registered model name and constructor parameters
- `training`: registered TrainingBuilder and builder-owned task parameters
- `process`: optional registered model-free probability process and its parameters
- `objective`: optional reusable scalar training objective
- `optimizer`: optimizer name and hyperparameters
- `lr_scheduler`: optional optimizer learning-rate scheduler
- `ema`: optional exponential moving average tracking and sampling policy
- `sampling`: optional task Builder, shape/batching, and artifact writers
- `diagnostics`: optional denoiser and multi-sampler training diagnostics
- `trainer`: loop, device, gradient, and early stopping policy
- `logging`: metric logging backends
- `artifacts`: checkpoint cadence

The built-in image recipes provide these private partition modes:

- `holdout`: deterministic train/validation split from a finite source
- `official`: source-provided train/validation/test roles
- `none`: build only a training split
- `kfold`: one deterministic fold per independent run

These modes are recipe capabilities, not requirements on custom DataBuilders.
K-fold requires both `num_folds` and `fold_index`; running all folds is left to
a project script or sweep.

The Flowers102 config trains on the official `train` split, validates on `val`,
and reserves `test` for final evaluation. It uses train-time random crops,
evaluation center crops, EMA sampling, a linear DDPM beta schedule, clipped
posterior DDPM sampling, and a warmup-cosine optimizer LR schedule.

## Architecture

`src/stochaflow/data/`
: the thin DataBuilder contract plus private image, super-resolution, source,
  partition, and bucket recipe implementations.

`src/stochaflow/models/`
: UNet, residual blocks, attention blocks, and timestep embeddings.

`src/stochaflow/processes/`
: model-free Process roots, Gaussian probability-path capabilities, concrete
  processes, and class-based `noise_schedules/` parameterizations.

`src/stochaflow/training/`
: TrainingBuilder/Plan/Strategy contracts, built-in supervised and Gaussian
  training composition, objectives, trainer lifecycle, diagnostics, reporting, and EMA.

`src/stochaflow/sampling/`
: unified Samplers and observers, task SamplingBuilders, checkpoint sampling
  runtime, and registered tensor/image artifact writers.

`src/stochaflow/utils/`
: config loading, registries, factories, checkpointing, logging, and seeding.

`src/stochaflow/scripts/`
: package entry points for training and sampling commands.

## Development

Run checks:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

Run targeted tests while iterating:

```bash
uv run pytest tests/test_ddpm_shapes.py tests/test_sampling_runtime.py
```

## Repository Layout

```text
stochaflow/
  configs/
  src/stochaflow/
    data/
    processes/
    models/
    sampling/
    scripts/
    training/
    utils/
  tests/
  assets/
```

## License

This project is released under the [MIT License](LICENSE).
