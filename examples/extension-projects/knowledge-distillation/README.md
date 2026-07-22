# Frozen-teacher knowledge distillation

This independent Python distribution demonstrates Stochaflow's training
composition boundary with a frozen teacher, a checkpointed auxiliary objective,
and student-only inference. It is an architecture reference, not a benchmark or
a claim about distillation accuracy.

The extension registers a deterministic synthetic classification `DataBuilder`,
student and teacher models, cross-entropy and temperature-KL objectives, a
`TrainingBuilder`/`TrainingStrategy`, and a direct prediction `SamplingBuilder`.
All names use the `stochaflow-knowledge-distillation` distribution namespace.

## Install and run

Run commands from this project root so relative `data/` and `outputs/` paths
resolve predictably. Any PEP 517-compatible installer works; uv is optional.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python tools/create_teacher_bootstrap.py --output data/teacher.pt
stochaflow train --config experiments/tiny/train.yaml
```

The bootstrap file is a plain PyTorch model `state_dict`, not a Stochaflow
checkpoint. A fresh run and a resumed run use it only to construct a compatible
teacher before core restores managed asset state. On resume, checkpointed
`training_assets_state_dict.teacher` and
`training_assets_state_dict.distillation_objective` are authoritative.

Strict resume uses the checkpoint's saved configuration:

```bash
stochaflow train --resume outputs/tiny/<run-id> --epochs 3
```

Checkpoint-only sampling constructs only the primary student and prediction
builder. It does not construct the training builder, teacher, or distillation
objective, so `data/teacher.pt` may be removed before this command:

```bash
stochaflow sample --checkpoint outputs/tiny/<run-id>
```

Sampling writes `samples.pt` containing student logits and a
`resolved_sampling.yaml` manifest. The optional `vision` dependency enables the
project-private torchvision source (`MNIST`, `FashionMNIST`, or `CIFAR10`); it
does not add a framework-level dataset registry.

## Responsibility boundary

- The Builder constructs, initializes, freezes, and declares the teacher.
- Core owns device, mode, optimizer selection, checkpoint, and resume lifecycle.
- The Strategy only interprets batches, performs forwards, and combines losses.
- The sampling builder resolves only the checkpointed student model.

The entry-point aggregation module only imports registration modules. No project
source path scanning or `PYTHONPATH` injection is required after installation.
