# Built-in image-generation examples

These repository-local examples exercise Stochaflow's built-in image data,
Gaussian process, training, sampling, diagnostic, and writer components. They do
not require an extension package.

The `experiments/` directory contains complete training configurations.
`experiments/overlays/` contains strict-resume observability overlays, while
`experiments/sampling/` contains checkpoint-bound partial sampling requests.
Those requests select a sampler and output policy; the checkpoint supplies the
trained inference recipe, so users never select an internal SamplingBuilder.
Keeping
these files beside the example distinguishes runnable examples from framework
configuration schemas and generated configuration reference sources.

Run a short MNIST smoke experiment from the repository root:

```bash
uv run stochaflow train \
  --config examples/built-in/image-generation/experiments/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 10 \
  --skip-final-sample
```
