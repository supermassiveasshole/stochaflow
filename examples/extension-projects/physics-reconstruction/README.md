# Physics reconstruction extension

This is an independently installable Stochaflow reference project for
three-frame Kolmogorov-vorticity reconstruction. It demonstrates the extension
boundaries, not a pretrained model or a claim that the paper's reported accuracy
has been reproduced.

The package owns task-specific data, model, training, dynamics, guided solver,
and artifact policies. Stochaflow still owns the registry, CLI, automatic
training loop, checkpoint, EMA, and sampling runtime.

## What is composed

- `KolmogorovDataBuilder` memory-maps `[trajectory, time, H, W]` `.npy` data and
  yields raw consecutive `[3, H, W]` triplets. Explicit trajectory ranges keep
  train/test policy in this recipe; the production example deliberately has no
  validation split.
- `ConditionalDenoiser` checkpoints normalization statistics and PDE constants
  as model buffers. The included residual network is intentionally compact and
  replaceable; it is not PhysicsNeMo's SongUNet.
- `PhysicsDenoisingStrategy` normalizes raw fields, samples a Gaussian marginal,
  computes the detached `mean(R^2)` gradient condition, and reuses
  `gaussian_training_target()` plus the configured Objective.
- `PhysicsGaussianDynamics` combines the Process, conditional model, prediction
  semantics, and physics directions. Built-in DDPM/DDIM consume its ordinary
  Gaussian `predict()` behavior unchanged.
- `GuidedDDIMSampler` is needed only for the algorithmic variant that subtracts
  an energy-normalized PDE correction after each accepted transition. It reuses
  public `DDIMSampler.resolve_schedule()` and `DDIMSampler.transition()`; no
  DDIM equation is copied.
- `ReconstructionArtifactWriter` writes final physical-unit samples batch by
  batch to `reconstructions.npy`, publishes each file with same-directory
  replacement, removes a partial pair on failure, and refuses to overwrite an
  existing result.

The condition gradient and post-transition correction are separate policies.
No classifier-free-guidance weight is exposed. `clip_denoised: true` is rejected
because standardized vorticity is not constrained to `[-1, 1]`.

## Tiny end-to-end run

Run commands from this project root so `data/` and `outputs/` resolve here.
Neither `uv` nor editable installation is required; these are ordinary Python
packaging and console-entry-point workflows.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m stochaflow_physics_reconstruction.tools.prepare_tiny_data \
  --output-dir data/tiny
stochaflow train --config experiments/tiny/train.yaml --skip-final-sample
```

Use the emitted run directory or checkpoint for each independent sampling
policy:

```bash
stochaflow sample --checkpoint outputs/tiny/<run> \
  --config experiments/tiny/sample-baseline-ddim.yaml \
  --output-dir outputs/sample-baseline-ddim
stochaflow sample --checkpoint outputs/tiny/<run> \
  --config experiments/tiny/sample-baseline-ddpm.yaml \
  --output-dir outputs/sample-baseline-ddpm
stochaflow sample --checkpoint outputs/tiny/<run> \
  --config experiments/tiny/sample-guided-ddim.yaml \
  --output-dir outputs/sample-guided-ddim
```

Every overlay uses a mathematically aligned public state schedule: the marginal
is sampled at `partial_noise_time`, the first reverse source is that same state,
and the final target is clean state `0`. This intentionally avoids the legacy
off-by-one path in which initial noising and the first reverse source differed.

## Prepare the 40-trajectory dataset

The reference workflow expects a floating high-resolution `.npy` array with
shape `[40, 320, 256, 256]` and a sparse `.npz` member such as `u3232`. The tool
requires that member to be an ordinary C-order, uncompressed `ZIP_STORED` NumPy
array so it can memory-map only the selected tail instead of materializing the
multi-gigabyte member. It selects the last four sparse trajectories,
bicubic-resizes them when necessary, optionally applies periodic Gaussian
smoothing, writes a mmap-ready observation array, computes first-36
normalization statistics, and writes a strict positional-alignment sidecar.

```bash
python tools/prepare_kolmogorov.py \
  --reference data/kolmogorov.npy \
  --sparse data/kolmogorov_sparse.npz \
  --sparse-key u3232 \
  --output-dir data \
  --held-out-trajectories 4 \
  --smoothing-kernel 7
```

Copy the reported mean/scale from `data/kolmogorov-stats.json` into the model
configuration if they differ from the documented reference values. The
alignment sidecar is checked against both configured paths, ranges, source
shapes, and the expected 1272 trajectory-major triplets before sampling.

The production configs are executable declarations, but real scientific
training needs an appropriate denoiser architecture, compute budget, data
license review, and independent metric validation. The default final sampling
is final-state only and writes 1272 float32 samples of shape `[3, 256, 256]`.

## Real-batch capacity evidence

The capacity helper can consume the real mmap array rather than fabricating
scientific evidence. It exercises a train forward/backward, baseline DDIM, and
guided DDIM, and records model/input tensors plus host and CUDA peaks where
available. MPS has no resettable peak API, so the report labels its current and
driver allocations before/after each phase without calling them peaks:

The host `cumulative_lifetime_peak_rss_bytes` value is the process lifetime
high-water mark observed by that phase, not a resettable phase-local peak.

```bash
python -m stochaflow_physics_reconstruction.tools.capacity_check \
  --data data/kolmogorov.npy \
  --trajectory-start 0 \
  --batch-size 1 \
  --resolution 256 \
  --hidden-channels 64 \
  --num-blocks 6 \
  --time-embedding-dim 128 \
  --num-timesteps 1000 \
  --partial-time 240 \
  --sample-steps 2 \
  --device auto \
  --output capacity-report.json
```

Do not treat a report as completed evidence unless this command was actually
run on the target accelerator and real array. The measured maintainer-host
record is committed at
`../../../benchmarks/results/stage7-physics-macos-arm64.json`; it records source,
config, implementation, command, and ignored runtime-artifact hashes rather
than committing the datasets, checkpoint, or outputs.

## Real-data production-path smoke

The capacity helper is not a replacement for the configured runtime. After
preparing the real observation and alignment files, run one bounded optimizer
step through the production training config:

```bash
stochaflow train \
  --config experiments/production/train.yaml \
  --epochs 1 \
  --limit-batches 1 \
  --limit-test-batches 1 \
  --skip-final-sample \
  --no-progress
```

The `real-smoke` overlays reduce only the output count and batch size to one,
select the freshly updated raw weights, and retain the production solver math:
30 accepted transitions from state time 240 for baseline DDIM and 40 accepted
transitions from state time 320 for guided DDIM.

```bash
stochaflow sample \
  --checkpoint outputs/production/<run>/checkpoints/latest.pt \
  --config experiments/real-smoke/sample-baseline-ddim.yaml \
  --output-dir outputs/real-smoke-baseline
stochaflow sample \
  --checkpoint outputs/production/<run>/checkpoints/latest.pt \
  --config experiments/real-smoke/sample-guided-ddim.yaml \
  --output-dir outputs/real-smoke-guided
```

Together with the tiny deterministic E2E and two-step capacity helper, this is
the third validation tier. It is not converged training, a 1272-sample job, or
scientific accuracy evidence; those remain an explicit unexecuted fourth tier.

## Provenance

The task equations and data layout follow the PhysicsNeMo Flow Reconstruction
Diffusion example and the associated DFSR method. This project is an original,
small integration example and does not copy its model implementation or ship
its datasets/checkpoints.
