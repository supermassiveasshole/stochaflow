# Stochaflow

Stochaflow is a research-oriented Python framework for generative modeling
through probability paths, generative dynamics, and numerical samplers. The
current implementation focuses on config-driven DDPM and DDIM training and
sampling for MNIST, CIFAR-10, and Oxford Flowers 102.

The codebase is organized around registries and config-driven components. Thin
data builders, models, optional probability processes, complete samplers, task
sampling builders, training builders, artifact writers, objectives, diagnostics,
and loggers are selected through registries. Standard PyTorch optimizers and LR
schedulers use allowlisted native target paths instead of copied Registry aliases.

Start with the [framework overview](docs/framework.md) or the
[configuration and workflow handbook](docs/configuration/index.md).

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

## Framework features

Implemented:

- DDPM/DDIM epsilon-prediction training
- model-free discrete Gaussian Process with marginal and posterior mathematics
- Gaussian model Dynamics with epsilon/x0/v/score prediction conversion
- registered DDPM and DDIM Samplers with one complete `sample()` interface and
  reusable discrete-Gaussian transition/schedule primitives
- registered SamplingBuilder composition and observer-based trajectories
- beta-native linear and alpha-bar-native cosine noise schedules
- a thin, modality-neutral `DataBuilder -> DataLoaders` extension contract
- built-in image, super-resolution-data, and multi-resolution image recipes;
  conditional super-resolution models and training/sampling composition remain
  project extensions
- torchvision and stable local image-folder sources for built-in recipes
- recipe-local holdout/K-fold, multi-source mixtures, and dynamic pixel batches
- registered tensor, image, and user-defined sampling artifact writers
- registered TrainingBuilder composition with thin task-specific TrainingStrategy
- reusable scalar Objectives and managed auxiliary modules for single-optimizer
  training compositions
- a fully exercised frozen-teacher distillation reference; offline and
  multi-teacher policies remain extension-defined rather than core modes
- UNet backbone with optional attention blocks
- EMA tracking and EMA sampling
- a project-owned warmup-cosine scheduler and direct native PyTorch optimizer/LR
  scheduler construction
- multi-sampler diffusion diagnostics with denoiser metrics and visual artifacts
- Rich terminal reporting, checkpointing, and local/TensorBoard/W&B logging
- one `stochaflow` CLI with `init`, `train`, and `sample` subcommands
- pytest, ruff, and pyright configuration

Still evolving:

- faster samplers and broader learned/perceptual evaluation metrics
- multi-optimizer and manual-optimization training loop families
- probability-flow ODE, flow-matching, rectified-flow, and continuous-flow
  implementations beyond the current diffusion baseline

## Installation

Install the released package into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install stochaflow
stochaflow --help
```

Windows PowerShell uses `.venv\Scripts\Activate.ps1`. A local wheel can be
installed with `python -m pip install path/to/stochaflow.whl`.

Optional dependency groups:

| Extra | Purpose |
| --- | --- |
| `wandb` | Weights & Biases logging |
| `quality` | KID/FID diagnostics and their feature model |
| `docs` | Sphinx documentation toolchain |
| `dev` | Pytest and Pyright for source contributors |

TensorBoard support is part of the base installation; it does not need a
separate extra.

For development from a source checkout:

```bash
uv sync --extra dev
```

Add `--extra quality`, `--extra wandb`, or `--extra docs` when working on those
features. An editable pip workflow is also supported:

```bash
python -m pip install -e ".[dev]"
```

Platform notes:

- Python `>=3.12` is required.
- Intel macOS uses Python 3.12 with pinned PyTorch/Torchvision wheels.
- The source checkout's `uv` configuration routes Windows GPU dependencies
  through the PyTorch CUDA 12.8 wheel index and can use Python 3.14.
- `trainer.device: auto` selects CUDA first, then Apple MPS when available,
  and otherwise falls back to CPU.

Windows GPU setup example:

```powershell
uv python install 3.14
uv sync --python 3.14 --extra dev
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Extension projects

Create a package-manager-neutral extension repository with:

```bash
stochaflow init my-research-project
cd my-research-project
```

`init` writes an installable Python distribution, a small end-to-end extension,
and a runnable experiment config. It does not create an environment, install
dependencies, overwrite a non-empty directory, or require `uv`. Install the
generated distribution into the same Python environment as the Stochaflow CLI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
stochaflow train --config experiments/example/train.yaml
```

When the platform provides secure descriptor-relative filesystem operations,
`init` can also populate an existing empty real directory. On other platforms,
remove that empty directory and let `init` create it.

An optional `uv` workflow works as well:

```bash
uv sync --extra test
uv run stochaflow train --config experiments/example/train.yaml
```

The generated repository is an ordinary single Python distribution with room
for multiple experiments and unrelated research code. Stochaflow only discovers
the installed entry point and runs explicitly selected components; it does not
scan or manage the repository. Relative paths in configs and opaque builder
parameters resolve from the process working directory, so run the generated
example from the project root.

The package declares its aggregate registration module through standard Python
metadata:

```toml
[project.entry-points."stochaflow.extensions"]
my-research-project = "my_research_project.stochaflow_ext"
```

The experiment selects that exact entry-point name:

```yaml
extensions:
  plugins: [my-research-project]
```

Omitting `extensions` (or writing `plugins: []`) loads no third-party code;
`plugins: null` explicitly opts into every Stochaflow extension installed in the
current environment. Config parsing itself is side-effect free. Discovery,
checkpoint provenance checks, and plugin activation happen before Registry
components are constructed.

## Training

The following built-in configs are part of the source repository. Package users
can start from the runnable config generated by `stochaflow init`.

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

Custom data builders are registered as classes and exposed through an installed
`stochaflow.extensions` entry point. A config activates them by plugin name; see
[Extensions and registries](docs/configuration/extensions.md).

Two installable vertical reference projects live under
`examples/extension-projects/`: a discrete-Gaussian Physics reconstruction
workflow and a frozen-teacher knowledge-distillation workflow. They exercise
data, training, checkpoint, and sampling extension boundaries without being
packaged into Stochaflow or adding task-specific runner branches. See the
[reference-project guide](docs/configuration/reference-projects.md).

For the complete YAML schema, framework-owned component parameters,
multi-source mixing, buckets, K-fold, logging, and CLI overrides, see the
[Configuration handbook](docs/configuration/index.md). Native optimizer and LR
scheduler constructor parameters follow the installed PyTorch version.

Extension authors can follow the independent tutorials for
[conditional Gaussian super-resolution](docs/tutorials/super-resolution.md),
[reusing the Gaussian family](docs/tutorials/reuse-gaussian-components.md), or
[adding a custom generation family/direct transform](docs/tutorials/custom-generation-family.md).
The [checkpoint and portability guide](docs/configuration/compatibility-and-migration.md)
documents config authority and checkpoint/environment boundaries, while the
[public extension API reference](docs/api/extensions.md) lists the supported
third-party import surface. The
[framework overview](docs/framework.md) records the stable architecture and
extension responsibilities.

If `--epochs` is omitted, the runner uses `trainer.num_epochs` from the YAML
config. Passing `--epochs` is an explicit run-time override.

Useful training options:

```bash
--resume CHECKPOINT           Strictly resume saved config and full training state.
--device DEVICE               Override trainer.device for this run.
--output-dir PATH             Override experiment.output_dir for this run.
--skip-final-sample           Skip best-checkpoint acceptance sampling.
--no-progress                 Disable Rich terminal progress bars.
--force-extension-version-mismatch
                              Accept a plugin version mismatch after identity checks.
```

`--config` and `--resume` are mutually exclusive. Resume always uses the config
stored in the checkpoint and creates a new sibling run directory; it is not a
weights-only warm start and does not reopen the previous output directory.
For strict best-selection continuity, resume a run directory or keep the matching
`best.pt` beside any `latest.pt`/epoch checkpoint you move. Its recorded best
resolved config, extension provenance, epoch, metric, monitor, and mode must
match the selected checkpoint; a mutable `best.pt` produced by a later epoch
cannot resume an older snapshot. A standalone
best checkpoint is self-sufficient because its metadata identifies it as the
selected best. Resume materializes the validated best in the new run before
training, so later resume and sampling do not depend on the parent run.
Strict resume also restores the checkpointed Python, NumPy, Torch CPU, and
applicable CUDA RNG streams after all selected/best state validation. A device
override remains allowed, but matching runtime and device topology is necessary,
not sufficient, for bitwise continuity. DataBuilder, Dataset, loader, worker,
and sampler runtime state is not checkpointed. The built-in image recipes
rebuild shuffled index order from the configured seed and epoch; worker-side
random crop/flip state is not restored and therefore is not guaranteed to match
an uninterrupted run. Custom stochastic loaders own the same boundary and must
derive each epoch from the configured seed and a duck-typed `set_epoch(epoch)`
contract when continuity matters.

Each run writes a timestamped directory under the configured output root:

```text
outputs/<experiment>/<YYYYMMDD_HHMMSS>/
  checkpoints/
    best.pt
    latest.pt
    epoch_XXXX.pt
  metrics.jsonl
  train.log
  resolved_config.yaml
  run_manifest.yaml
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
reference metrics have separate registries. Diagnostic-local modules can register a new
provider and enable it from `diagnostics[].params.modules` without modifying the
`diffusion_quality` orchestrator; explicit empty provider lists disable a phase.

Images are saved locally and forwarded to every configured TensorBoard or W&B
logger. Optional KID/FID evaluation uses a fixed validation reference cache and
is enabled with the `quality` installation extra. See the Flowers102 configs for
a complete DDPM/DDIM comparison declaration.

## Sampling

Sample directly from a Stochaflow checkpoint. The checkpoint contains model
state and the resolved experiment config, so an extra YAML file is optional;
extension-backed checkpoints also require their recorded distributions to be
installed in the CLI environment:

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
`best.pt` under `experiment.output_dir`. With both inputs, a lightweight YAML
containing `sampling` and optional `extensions` overlays the checkpoint config.
Its `sampling` section is the complete replacement, so it may change sample
count, batch size, shape, Builder/Sampler parameters, trajectory, writers, and
raw/EMA selection. A complete external config is authoritative as a whole;
the checkpoint supplies state, and normal state-loading contracts determine
whether the chosen components are compatible. Stochaflow does not compare or
merge two complete configs.

For a lightweight overlay, `extensions: {}` keeps the checkpoint plugin
selection. An explicitly present `extensions.plugins` replaces that selection
instead of appending to it. Plugin identity must still match any reused
checkpoint provenance; only a version difference can be accepted with an
interactive confirmation or `--force-extension-version-mismatch`.

The standard builder uses EMA model weights when `ema.enabled` and
`ema.use_for_sampling` are both true. Declared `sampling.writers` decide the outputs: `tensor` writes PT
files, `image` validates NCHW data and writes PNG/GIF artifacts, and extensions
can write domain formats such as NetCDF. Every run also writes
`resolved_sampling.yaml`.

Sampling artifacts currently use a materialized lifecycle: a Builder returns all
batches and retained trajectory states before any writer runs. The built-in
Standard Builder moves writer-ready tensors to CPU; the public contract does not
force custom outputs onto a particular device. This is not a streaming contract.
The conservative Physics AI profile is 1,272 float32 states
shaped `[3, 256, 256]` with trajectory disabled; a domain writer may persist only
the center field after per-batch metrics are computed. Full dense trajectories at
that scale are unsupported. Trajectory visualization must use a separate preview
run with at most 8 samples, no more than 40 accepted steps, and
`every_steps >= 10`. See the
[sampling capacity guide](docs/configuration/sampling-capacity.md) for the exact
payload formulas and the distinction between analytical bounds and host-specific
RSS measurements.

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
- `extensions`: installed entry-point plugins activated for this component graph
- `data`: registered builder name plus builder-owned parameters
- `model`: registered model name and constructor parameters
- `training`: registered TrainingBuilder and builder-owned task parameters
- `process`: optional registered model-free probability process and its parameters
- `objective`: optional reusable scalar training objective
- `optimizer`: an allowlisted `torch.optim.<Class>` target or extension name,
  plus constructor keyword arguments
- `lr_scheduler`: an optional allowlisted
  `torch.optim.lr_scheduler.<Class>` target or extension name, its constructor
  keyword arguments, and the Stochaflow step/epoch lifecycle interval
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
posterior DDPM sampling, and a project-owned warmup-cosine LR scheduler with an
explicit total step count.

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
  training composition, objectives, managed auxiliary assets, trainer lifecycle,
  diagnostics, reporting, and EMA. Builders own dependency composition; Strategies
  own only batch-to-loss/metrics computation.

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
