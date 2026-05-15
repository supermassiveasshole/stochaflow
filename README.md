# Stochaflow

Stochaflow is a research-oriented Python project for **stochastic flows**. The
first concrete implementation track is DDPM-style diffusion, but the repository
is intentionally structured around broader stochastic flow experiments rather
than diffusion models alone.

The current codebase includes an end-to-end MNIST DDPM smoke path: dataset
loading, a UNet denoiser, DDPM forward/reverse logic, an epsilon-prediction
objective, a generic trainer, checkpointing, local logging, and post-training
sample dumps.

简要说明：Stochaflow 的目标不是只做扩散模型，而是作为随机流模型研究代码库。当前优先实现 DDPM，用 MNIST 作为端到端验证路径。

## Current Status

Implemented:

- DDPM epsilon-prediction training forward path
- DDPM reverse sampling and reverse-trajectory capture
- linear DDPM scheduler with cached timestep coefficients
- MNIST dataset wrapper with normalization support
- compact UNet backbone for image diffusion smoke tests
- registry/factory-based component construction from YAML config
- generic trainer with checkpoint integration
- local experiment logging and optional TensorBoard/W&B backends
- EMA utility
- sample grid and reverse-trajectory artifact dumping
- ruff, pyright, and pytest configuration

Still evolving:

- DDIM implementation
- CIFAR-10 experiment path
- richer stochastic-flow abstractions beyond diffusion
- evaluation metrics and larger experiment scripts

## Installation

This project uses a `src` layout and is designed to work well with `uv`.

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
- Intel macOS uses Python 3.12 with pinned PyTorch/Torchvision wheels for
  compatibility.
- Other platforms can use newer Python versions if dependency resolution
  succeeds.

## MNIST DDPM Smoke Run

Run the task-specific training entry point:

```bash
uv run stochaflow-train-mnist-ddpm \
  --config configs/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 10
```

Equivalent compatibility wrapper:

```bash
uv run python scripts/train_mnist_ddpm.py --epochs 1 --limit-batches 10
```

Useful options:

```bash
--resume PATH                 Resume model and optimizer state from checkpoint.
--num-samples 16              Number of post-training samples to generate.
--sample-grid-size 4          Number of images per row for the final sample grid.
--trajectory-interval 200     Reverse-process interval for trajectory snapshots.
--skip-sampling               Train without dumping post-training samples.
--no-progress                 Disable tqdm progress bars.
```

After training, the script writes artifacts under the configured experiment
output directory, for example `outputs/ddpm_mnist/`:

```text
outputs/ddpm_mnist/
├── checkpoints/
│   └── epoch_0001.pt
├── metrics.jsonl
└── samples/
    ├── epoch_0001.png
    ├── epoch_0001.pt
    ├── epoch_0001_trajectory.png
    └── epoch_0001_trajectory.pt
```

`epoch_XXXX.png` shows final generated samples. `epoch_XXXX_trajectory.png`
shows the reverse process from high-noise states to lower-noise generated
images; rows are timesteps and columns are individual samples.

## Configuration

Experiment configuration lives in YAML files under `configs/`.

The main working config is:

```text
configs/ddpm_mnist.yaml
```

Config sections are intentionally component-oriented:

- `experiment`: run name, seed, and output directory
- `data`: dataset declaration and dataloader policy
- `model`: registered model name and constructor params
- `diffusion`: process type and scheduler config
- `objective`: training objective
- `optimizer`: optimizer name and hyperparameters
- `trainer`: loop/device/gradient settings
- `logging`: metric logging backends
- `artifacts`: checkpoint and sampling cadence

Components are built through registries and factories. New models, datasets,
schedulers, objectives, loggers, optimizers, and diffusion processes should be
registered close to their implementation rather than added through ad hoc
conditionals in scripts.

## Architecture

`src/stochaflow/data/`
: dataset wrappers, transforms, and dataloader-facing utilities.

`src/stochaflow/models/`
: UNet, residual blocks, attention blocks, and timestep embeddings.

`src/stochaflow/diffusion/`
: diffusion-specific algorithms, schedulers, and objectives. `DDPM` owns DDPM
training forward logic and reverse sampling logic.

`src/stochaflow/training/`
: generic optimization loop, train-step adapters, and EMA utilities.

`src/stochaflow/sampling/`
: artifact utilities for visualizing generated samples and reverse trajectories.
Algorithmic sampling belongs to model/process classes such as `DDPM`.

`src/stochaflow/utils/`
: config loading, registries, factories, checkpointing, logging, and seeding.

`src/stochaflow/scripts/`
: packaged task-specific entry points.

`scripts/`
: thin compatibility wrappers for direct script execution.

## Development

Run static checks:

```bash
uv run ruff check .
uv run pyright
```

Run tests:

```bash
uv run pytest
```

Run targeted tests while iterating:

```bash
uv run pytest tests/test_ddpm_shapes.py tests/test_sampling_grid.py
```

## Repository Layout

```text
stochaflow/
├── configs/
│   ├── ddpm_mnist.yaml
│   ├── ddpm_cifar10.yaml
│   └── ddim_cifar10.yaml
├── scripts/
│   └── train_mnist_ddpm.py
├── src/
│   └── stochaflow/
│       ├── data/
│       ├── diffusion/
│       ├── models/
│       ├── sampling/
│       ├── scripts/
│       ├── training/
│       └── utils/
├── tests/
├── outputs/
└── assets/
```

## License

This project is released under the [MIT License](LICENSE).
