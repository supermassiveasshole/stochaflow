# Frozen-teacher knowledge distillation

This independent Python distribution demonstrates Stochaflow's training and
inference-asset composition boundaries with a frozen teacher, a checkpointed
auxiliary objective, and an embedded logit calibrator. It is an architecture
reference, not a benchmark or a claim about distillation accuracy.

The extension registers a deterministic in-memory synthetic classification
`DataBuilder`, student and teacher models, cross-entropy and temperature-KL
objectives, a `TrainingBuilder`/`TrainingStrategy`, a registered
`LogitCalibrator`, and a direct prediction `SamplingBuilder`. All names use the
`stochaflow-knowledge-distillation` distribution namespace.

## Install and run

Run commands from this project root so relative `data/` and `outputs/` paths
resolve predictably. Any PEP 517-compatible installer works; uv is optional.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python tools/create_teacher_bootstrap.py \
  --teacher-output data/teacher.pt \
  --calibrator-output data/calibrator.pt
stochaflow train --config experiments/tiny/train.yaml
```

Both bootstrap files are plain PyTorch model `state_dict` mappings, not
Stochaflow checkpoints. A fresh run uses them to initialize compatible modules.
A resumed run uses them only to construct those modules before core restores
managed asset state. On resume, checkpointed
`training_assets_state_dict.teacher`,
`training_assets_state_dict.distillation_objective`, and
`training_assets_state_dict.calibrator` are authoritative.

Strict resume uses the checkpoint's saved configuration:

```bash
stochaflow train --resume outputs/tiny/<run-id> --epochs 3
```

Checkpoint-only sampling follows the checkpoint's calibrated student-prediction
recipe. It constructs the primary student and requests only the declared
`calibrator` inference asset. It does not construct the training builder,
teacher, or distillation objective, so both bootstrap files may be removed
before this command:

```bash
stochaflow sample --checkpoint outputs/tiny/<run-id>
```

Sampling writes `samples.pt` containing calibrated student logits and a
`resolved_sampling.yaml` manifest. The example bootstrap initializes the
calibrator with zero scale and biases `0, 1, ..., num_classes - 1`, making the
embedded state visibly authoritative: every output row equals that bias vector.

## Accepted reference result

The installed-wheel, checkpoint-only acceptance run produces a tensor with
shape `[8, 4]`. Every row is exactly:

```text
[0.0, 1.0, 2.0, 3.0]
```

The run first resumes after replacing the local bootstrap, then deletes both
bootstrap files and samples offline from a non-project working directory. This
exact output proves that sampling consumed the checkpoint's embedded calibrator
state rather than either acquisition path; the sampling view contains neither
the teacher nor the distillation objective.

The inference declaration contains only the registered calibrator name and
`num_classes`; the bootstrap path is acquisition-only training configuration
and is not part of reconstruction. Core retains only the descriptor-referenced
calibrator state in its sampling view. The teacher and auxiliary objective
remain training-only.

The data recipe intentionally has no external input artifact: its `synthetic`
parameters and experiment seed generate all three in-memory splits.
Consequently `artifact_bindings` is always `None`, strict resume does not
request an artifact identity, and this recipe does not create
`.stochaflow-cache`. A real classification dataset must enter through an
extension-owned `DataSource`/`DataArtifact` pair or a compatible framework
recipe; acquisition and download do not belong in this Builder.

## Responsibility boundary

- The synthetic DataBuilder owns only deterministic split generation, Dataset,
  sampler, and loader composition; it performs no external acquisition.
- The TrainingBuilder constructs, initializes, freezes, and declares the
  teacher and calibrator.
- Core owns device, mode, optimizer selection, checkpoint, and resume lifecycle.
- The Strategy only interprets batches, performs forwards, and combines losses.
- The checkpoint records a reconstruction-only calibrator declaration plus its
  embedded state. Core lazily constructs it only when the SamplingBuilder
  requests the `classification_logit_calibrator` role.
- The SamplingBuilder validates the extension-owned
  `LogitCalibrationCapability` before applying it to student logits.

The entry-point aggregation module only imports registration modules. No project
source path scanning or `PYTHONPATH` injection is required after installation.
