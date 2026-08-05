# P2 Experiment Closeout

- Status: archived experiment record; not a supported framework capability
- Closed: 2026-08-06
- Scope: AFHQ-v2 128x128, canonical current ADM, single RTX 4090

P2 weighting was implemented and exercised as an experimental Gaussian training
recipe. The maintained surface was retired before merge because the completed
evidence did not demonstrate a reliable quality benefit, while the active
ordinary image-generation path does not need this additional recipe.

## Preserved evidence

- cosine + epsilon + gamma 1 negative control:
  `G:\stochaflow\outputs\afhq-v2\adm-128-p2\20260802_172503`;
- linear + epsilon + gamma 1 lineage: root E1-25, sibling E26-40, sibling
  E41-149 (E150 partial);
- matched linear + epsilon + gamma 0 run: complete through E100 before a host
  crash;
- one-epoch controlled gamma 0/gamma 1 formal comparison: P2 was slightly worse
  in aggregate and every class, without statistical significance.

The cosine run learned animal structure but retained systematic chroma noise.
The linear arms did not produce evidence strong enough to justify keeping a
second public TrainingBuilder family. Machine-local manifests, metrics,
diagnostics, and evaluation bundles remain experiment records; routine
checkpoints may be removed under the output-retention audit.

## Durable decision

- remove P2 TrainingBuilder/Strategy implementations, registrations, examples,
  configuration reference entries, and public documentation;
- do not resume these runs or use their diagnostics as formal Evaluation;
- keep Metrics task-neutral and use complete validation Evaluation for
  checkpoint selection;
- continue with fresh canonical ADM + cosine + v + learned-range variance.

This record is historical only. Reintroducing any SNR weighting method requires
a new roadmap decision, a namespaced concrete recipe, and matched validation
evidence; this document does not authorize that work.
