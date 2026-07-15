# Stochaflow

Stochaflow is a research-oriented Python project for stochastic flows. The
current implementation track is DDPM-style diffusion, with end-to-end training
paths for MNIST, CIFAR-10, and Oxford Flowers 102.

The codebase is organized around registries and config-driven factories:
datasets, data splits, models, diffusion processes, objectives, optimizers,
learning-rate schedulers, diagnostics, loggers, checkpoints, and runners are
constructed from YAML configuration.

## Status

Implemented:

- DDPM epsilon-prediction training
- DDPM reverse sampling and trajectory capture
- beta-native linear and alpha-bar-native cosine noise schedules
- class-based MNIST, CIFAR-10, and Oxford Flowers 102 dataset factories
- multi-source mixtures with optional step weights and global holdout/K-fold splits
- per-sample resolution buckets with pixel-budget-scaled batch sizes
- UNet backbone with optional attention blocks
- EMA tracking and EMA sampling
- warmup-cosine and common PyTorch optimizer LR schedulers
- DDPM diagnostic logging for timestep-bucket losses and sample artifacts
- Rich terminal reporting, checkpointing, and local/TensorBoard/W&B logging
- package entry points for training and checkpoint sampling
- pytest, ruff, and pyright configuration

Still evolving:

- DDIM implementation
- faster samplers and evaluation metrics such as FID/KID
- broader stochastic-flow abstractions beyond diffusion

## Installation

This project uses a `src` layout and is designed to work with `uv`.

```bash
uv sync --extra dev
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
- `trainer.device: auto` selects CUDA when `torch.cuda.is_available()` is true.

Windows GPU setup example:

```powershell
uv python install 3.14
uv sync --python 3.14 --extra dev
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Training

MNIST smoke run:

```bash
uv run stochaflow-train-mnist-ddpm \
  --config configs/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 10
```

CIFAR-10 smoke run:

```bash
uv run stochaflow-train-cifar10-ddpm \
  --config configs/ddpm_cifar10.yaml \
  --epochs 1 \
  --limit-batches 10 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

Oxford Flowers 102 smoke run:

```bash
uv run stochaflow-train-flowers102-ddpm \
  --config configs/ddpm_flowers102.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --skip-sampling
```

Oxford Flowers 102 baseline run:

```bash
uv run stochaflow-train-flowers102-ddpm \
  --config configs/ddpm_flowers102.yaml
```

Multi-source MNIST + Flowers102 run:

```bash
uv run stochaflow-train-ddpm \
  --config configs/ddpm_mnist_flowers102.yaml \
  --epochs 1 \
  --limit-batches 10
```

Custom factories are registered as classes and imported through
`data.modules`; see [Custom datasets](docs/custom-datasets.md).

For the complete YAML schema, defaults, validation rules, built-in component
parameters, multi-source mixing, buckets, K-fold, logging, and CLI overrides,
see [Configuration reference](docs/configuration.md).

If `--epochs` is omitted, the runner uses `trainer.num_epochs` from the YAML
config. Passing `--epochs` is an explicit run-time override.

Useful training options:

```bash
--resume [PATH]               Resume training. Without PATH, use latest.pt.
--num-samples 16              Number of post-training samples to generate.
--sample-grid-size 4          Number of images per row for the final sample grid.
--trajectory-interval 200     Reverse-process interval for trajectory snapshots.
--skip-sampling               Train without dumping post-training samples.
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
    epoch_XXXX.png
    epoch_XXXX.pt
    epoch_XXXX_trajectory.png
    epoch_XXXX_trajectory.pt
```

Example MNIST DDPM samples:

![MNIST DDPM generated samples](assets/readme/mnist_ddpm_epoch_0100.png)

MNIST reverse-process trajectory:

![MNIST DDPM reverse trajectory](assets/readme/mnist_ddpm_epoch_0100_trajectory.png)

Example Oxford Flowers 102 DDPM samples:

![Oxford Flowers 102 DDPM generated samples](assets/readme/flowers102_ddpm_epoch_0681.png)

Oxford Flowers 102 reverse-process trajectory:

![Oxford Flowers 102 DDPM reverse trajectory](assets/readme/flowers102_ddpm_epoch_0681_trajectory.png)

## Sampling

Sample from a Stochaflow DDPM checkpoint. If `--checkpoint` is omitted, the
sampler uses the newest `checkpoints/best.pt` under `outputs/`.

```bash
uv run stochaflow-sample-ddpm \
  --num-samples 16 \
  --sample-grid-size 4
```

Use `--checkpoint PATH` for a specific checkpoint or pass a run directory to use
that run's `checkpoints/best.pt`.

The sampler rebuilds the model from the config stored in the checkpoint. EMA
weights are used automatically when training enabled EMA and the checkpoint
contains EMA state. Pass `--no-ema` to sample raw weights.

## Configuration

Experiment configuration lives under `configs/`.

```text
configs/ddpm_mnist.yaml
configs/ddpm_cifar10.yaml
configs/ddim_cifar10.yaml
configs/ddpm_flowers102.yaml
configs/ddpm_mnist_flowers102.yaml
```

Important sections:

- `experiment`: run name, seed, and output directory
- `data`: dataset declaration, dataloader policy, and split policy
- `model`: registered model name and constructor parameters
- `diffusion`: process type, forward noise schedule, and sampler parameters
- `objective`: training objective
- `optimizer`: optimizer name and hyperparameters
- `lr_scheduler`: optional optimizer learning-rate scheduler
- `ema`: optional exponential moving average tracking and sampling policy
- `diagnostics`: optional algorithm-specific training diagnostics
- `trainer`: loop, device, gradient, and early stopping policy
- `logging`: metric logging backends
- `artifacts`: checkpoint cadence

Data split modes:

- `random_holdout`: deterministic train/validation split from one source split
- `official`: named dataset splits directly
- `all`: concatenate named splits into one training set
- `none`: build only a training split
- `kfold`: deterministic cross-validation bundles

The Flowers102 config uses all official `train`, `val`, and `test` images for
generative training, center-crop preprocessing, EMA sampling, linear DDPM beta
schedule, clipped posterior DDPM sampling, and a warmup-cosine optimizer LR
schedule.

## Architecture

`src/stochaflow/data/`
: dataset builders, transforms, and the unified data bundle pipeline.

`src/stochaflow/models/`
: UNet, residual blocks, attention blocks, and timestep embeddings.

`src/stochaflow/diffusion/`
: diffusion processes, objectives, and the class-based `noise_schedules/`
  package. Noise-path abstractions, discrete VP storage, and concrete
  parameterizations are kept in separate modules.

`src/stochaflow/training/`
: trainer, train-step adapters, diagnostics, reporting, and EMA.

`src/stochaflow/sampling/`
: image-grid and trajectory artifact utilities.

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
uv run pytest tests/test_ddpm_shapes.py tests/test_sample_ddpm_script.py
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
