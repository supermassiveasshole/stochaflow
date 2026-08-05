# AFHQ-v2 checkpoint cleanup pre-deletion audit — 2026-08-06

Status: pending explicit approval for the exact destructive target set below.
No checkpoint listed in this record has been removed by this cleanup yet.

This record audits the proposed cleanup of superseded AFHQ-v2 training
attempts. The retained experiment evidence was verified before any removal:

- 19/19 runs retained `run_manifest.yaml`, `resolved_config.yaml`,
  `metrics.jsonl`, and `train.log`;
- all 7,904 metric records parsed successfully;
- 198 diagnostic manifests referenced 1,475 present artifacts and reported no
  errors;
- all retained Evaluation result hashes matched, and every v12 validation
  result reported strict 900/900 completeness;
- Evaluation manifests retain the evaluated checkpoint epoch, path, and
  SHA-256 identity.

The proposed cleanup targets 19 `checkpoints` directories containing 236
checkpoint files plus one zero-byte interrupted atomic-write temporary file.
Their logical size was 480,367,477,703 bytes (447.377 GiB). Metrics, manifests,
logs, diagnostics, samples, prediction artifacts, and Evaluation results would
be retained. After deletion, these runs could no longer be resumed or re-sampled.

Proposed directories:

```text
G:\stochaflow\outputs\afhq-v2\acceptance\20260726_214216\checkpoints
G:\stochaflow\outputs\afhq-v2\acceptance-final-20260726\20260726_233755\checkpoints
G:\stochaflow\outputs\afhq-v2\acceptance-final-deterministic-20260726\20260726_234233\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128\20260728_234059\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128\20260729_012316\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128\20260729_090710\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128\20260729_092900\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128\20260729_211544\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128\20260729_221804\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-current-cosine-v\20260804_222432\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-learned-range-v-capacity-smoke\20260806_023955\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-learned-range-v-midres-capacity-smoke\20260806_032129\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-p2\20260802_172503\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-p2-linear\20260803_084539\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-p2-linear\20260803_204030\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-p2-linear\20260803_223519\checkpoints
G:\stochaflow\outputs\afhq-v2\adm-128-p2-linear-gamma0\20260804_100404\checkpoints
G:\stochaflow\outputs\afhq-v2\dit-b8-128\20260728_214416\checkpoints
G:\stochaflow\outputs\afhq-v2\dit-b8-128\20260728_224020\checkpoints
```

The historical P2 run-local `.venv-quality` environment and an empty
Evaluation temporary directory were outside the requested checkpoint scope and
were left untouched.
