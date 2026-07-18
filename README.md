# Stochaflow

Stochaflow is a research-oriented Python project for stochastic flows. The
current implementation includes config-driven DDPM and DDIM training and
sampling paths for MNIST, CIFAR-10, and Oxford Flowers 102.

The codebase is organized around registries and config-driven components: whole
data pipelines, reusable dataset factories, sampling artifact writers, models,
diffusion processes, objectives, optimizers, diagnostics, and loggers are
constructed from YAML configuration.

## Status

Implemented:

- DDPM/DDIM epsilon-prediction training
- DDPM and DDIM reverse sampling with sampler-specific debug trajectories
- beta-native linear and alpha-bar-native cosine noise schedules
- registered modality-neutral data pipelines with structured-batch passthrough
- built-in fixed-batch `map` and multi-resolution image pipelines
- class-based MNIST, CIFAR-10, and Oxford Flowers 102 dataset factories
- multi-source image mixtures, global holdout/K-fold, and dynamic pixel batches
- registered tensor, image, and user-defined sampling artifact writers
- UNet backbone with optional attention blocks
- EMA tracking and EMA sampling
- warmup-cosine and common PyTorch optimizer LR schedulers
- multi-sampler diffusion diagnostics with denoiser metrics and visual artifacts
- Rich terminal reporting, checkpointing, and local/TensorBoard/W&B logging
- one `stochaflow` CLI with `train` and `sample` subcommands
- pytest, ruff, and pyright configuration

Still evolving:

- faster samplers and broader learned/perceptual evaluation metrics
- broader stochastic-flow abstractions beyond diffusion

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

Custom data pipelines and factories are registered as classes and imported through
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

Switch a DDPM-trained denoiser to DDIM sampling without writing another config:

```bash
uv run stochaflow sample \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --sampler ddim \
  --sampler-param num_inference_steps=100 \
  --sampler-param eta=0.0
```

`--sampler-param KEY=VALUE` is repeatable and parses YAML scalar/list values.
When `--sampler` changes the configured sampler name, its previous constructor
parameters are discarded before the CLI parameters are applied.

You can alternatively pass only `--config`; Stochaflow then finds the newest
`best.pt` under `experiment.output_dir`. With both inputs, the checkpoint
provides weights while the external config supplies the complete sampling section
after model, training diffusion, and noise-schedule compatibility checks.

EMA denoiser weights are used when `ema.enabled` and `ema.use_for_sampling` are
both true. Declared `sampling.writers` decide the outputs: `tensor` writes PT
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
- `data`: registered pipeline name plus pipeline-owned parameters
- `model`: registered model name and constructor parameters
- `diffusion`: process type, forward noise schedule, and sampler parameters
- `objective`: training objective
- `optimizer`: optimizer name and hyperparameters
- `lr_scheduler`: optional optimizer learning-rate scheduler
- `ema`: optional exponential moving average tracking and sampling policy
- `sampling`: sample shape/batching, sampler selection, writers, and debug trajectory
- `diagnostics`: optional denoiser and multi-sampler training diagnostics
- `trainer`: loop, device, gradient, and early stopping policy
- `logging`: metric logging backends
- `artifacts`: checkpoint cadence

The built-in `map` and `multi_resolution_image` pipelines support these split modes:

- `random_holdout`: deterministic train/validation split from one source split
- `official`: named dataset splits directly
- `none`: build only a training split
- `kfold`: deterministic cross-validation bundles

The Flowers102 config trains on the official `train` split, validates on `val`,
and reserves `test` for final evaluation. It uses train-time random crops,
evaluation center crops, EMA sampling, a linear DDPM beta schedule, clipped
posterior DDPM sampling, and a warmup-cosine optimizer LR schedule.

## Architecture

`src/stochaflow/data/`
: modality-neutral pipeline contracts, built-in pipelines, dataset factories,
  and image-specific bucket helpers.

`src/stochaflow/models/`
: UNet, residual blocks, attention blocks, and timestep embeddings.

`src/stochaflow/diffusion/`
: diffusion processes, objectives, and the class-based `noise_schedules/`
  package. Noise-path abstractions, discrete VP storage, and concrete
  parameterizations are kept in separate modules.

`src/stochaflow/training/`
: trainer, train-step adapters, diagnostics, reporting, and EMA.

`src/stochaflow/sampling/`
: checkpoint sampling runtime and registered tensor/image artifact writers.

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
    diffusion/
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
