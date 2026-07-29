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
  --limit-batches 10
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
