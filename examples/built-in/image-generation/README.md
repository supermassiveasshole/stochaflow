# Built-in image-generation examples

These repository-local examples exercise Stochaflow's built-in image data,
Gaussian process, training, sampling, diagnostic, and writer components. They do
not require an extension package.

The maintained runnable configuration tree is deliberately small:

```text
configs/
├── train/mnist.yaml
├── sample/mnist-ddpm.yaml
├── sample/mnist-ddim-50.yaml
└── overlays/mnist-observability.yaml
```

`train/mnist.yaml` is the single training recipe and owns its training
diagnostics. The two files under `sample/` are checkpoint-bound sampling
profiles: each keeps the current top-level `sampling:` request envelope and
selects only request-time sampler, output, and writer settings. The checkpoint
supplies the trained inference recipe, so users never select an internal
SamplingBuilder. The overlay can change only diagnostics and logging during
strict resume.

The repository no longer maintains separate runnable CIFAR-10, Flowers102, or
multi-source experiment YAMLs here. That is an example-maintenance boundary,
not a statement that the corresponding framework data sources or recipes are
unsupported.

Run a short MNIST smoke experiment from the repository root:

```bash
uv run stochaflow train \
  --config examples/built-in/image-generation/configs/train/mnist.yaml \
  --epochs 1 \
  --limit-batches 10 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

The run is created below `outputs/mnist/`. After training, compare DDPM and
DDIM-50 without retraining:

```bash
uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml

uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddim-50.yaml
```

## Reference result

The maintained MNIST result comes from one 200-epoch run with 78,000 optimizer
updates. Validation denoising loss selected `best.pt` at epoch 183; the reported
losses are v-prediction objectives, not perceptual-quality metrics.

| Evaluation | Result |
| --- | ---: |
| Selected checkpoint | `best.pt`, epoch 183 / step 71,370 |
| Best validation loss | **0.07189** |
| Test loss after restoring the best checkpoint | **0.07363** |

Both panels use that checkpoint's EMA weights, the same seed, and the same
terminal-noise batch:

| DDPM, 1,000 model evaluations | Deterministic DDIM, 50 model evaluations |
| :---: | :---: |
| <img src="../../../assets/readme/mnist_ddpm_epoch_0183_samples.png" width="300" alt="Thirty-six MNIST samples generated with DDPM from the epoch-183 checkpoint"> | <img src="../../../assets/readme/mnist_ddim50_epoch_0183_samples.png" width="300" alt="Thirty-six MNIST samples generated with DDIM-50 from the epoch-183 checkpoint"> |

| DDPM trajectory | DDIM-50 trajectory |
| :---: | :---: |
| <img src="../../../assets/readme/mnist_ddpm_epoch_0183_trajectory.gif" width="300" alt="Animated MNIST DDPM reverse-process trajectory"> | <img src="../../../assets/readme/mnist_ddim50_epoch_0183_trajectory.gif" width="300" alt="Animated MNIST DDIM-50 reverse-process trajectory"> |

The trained checkpoint is not distributed. The bounded command above is a
workflow smoke test and is not expected to reproduce this converged result.
