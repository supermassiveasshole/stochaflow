# P2 Weighting 与 ADM 拓扑重构计划

- 文档性质：开发计划；不属于当前公开 API 或正式文档导航
- 状态：Implemented（A0/A1 与范围修订后的 A2 class-aware example contract 已完成；
  单类别 reproduction lane 已取消）
- 制定日期：2026-07-30
- 最近范围修订：2026-07-31
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)
- 关联计划：
  [Metrics、训练诊断与模型选择正式契约](../metrics.md)、
  [训练后 Evaluation 支持计划](post-training-evaluation-support-plan.md)
- 名词说明：本文中的 **P2 weighting** 是论文算法名，不是 Roadmap 的 `P2`
  优先级

## 1. 结论

本计划关闭两个必须分开归因的问题：

1. **A0：ADM topology correctness。**
   `adm_unet` 已切换到 canonical input/output block graph、逐 block skip ledger 和
   QKV residual attention。旧拓扑 checkpoint 与新模型不兼容，必须 fresh train。
2. **A1：Gaussian learned-range、P2 与 respaced ancestral DDPM。**
   P2 只作为 Gaussian epsilon simple-loss policy；learned-range 使用 `2C` 输出与
   detached-mean hybrid variational bound；respaced DDPM 使用 selected-pair
   transition，不借用 DDIM 语义。

原计划中的单类别 AFHQ reproduction lane 不再属于仓库范围。不会保留专用
DataSource、materializer、benchmark config resolver、训练/采样 YAML 或对应测试。
AFHQ showcase 只维护 official cat/dog/wild class-conditional surface，并通过同一
evaluation protocol 报告 aggregate 和 per-class KID/FID。

目标 GPU 上的容量、吞吐、显存、长训练质量和恢复演练是运行层证据，不是 A0/A1
代码合并门槛。训练指标、诊断来源与 monitor 语义已经作为正式框架能力发布。

## 2. 已实施能力

### 2.1 Canonical ADM topology

`adm_unet` 的结构不变量为：

- initial projection 进入 skip ledger；
- 每个 encoder residual block 和 downsample 都保存一个 skip；
- decoder 每个 resolution 使用 `num_res_blocks + 1` 个 residual block；
- 每个 decoder block 恰好消费一个 skip；
- `attention_resolutions` 表示实际空间尺寸；
- attention 使用 GroupNorm、QKV attention、output projection 和 residual；
- middle block 固定为 `ResBlock -> Attention -> ResBlock`；
- up/downsample 均留在 residual topology 内。

维护的 AFHQ-128 production config 参数量为 `105,197,187`。用于 P2 reference
parity 的 256 unconditional topology 参数量为 `93,563,910`；这个 golden 只验证
模型构造，不引入单类别数据或实验配置。

旧 topology fields 被删除，不增加 legacy mode、partial state load、checkpoint
adapter 或自动 config conversion。旧 raw、EMA 与 optimizer state 均 fail closed。

### 2.2 Gaussian loss-policy boundary

Loss weighting 属于 Gaussian training family，不是通用 `Objective`：

```text
per_sample_simple_loss
  -> GaussianLossWeighting
  -> weighted simple term
  -> optional unweighted variance term
  -> batch reduction
```

首批内置 policy：

- `constant`：权重恒为 1；
- `p2`：仅支持 epsilon prediction。

P2 coefficient 使用 cumulative marginal SNR：

```text
snr(t) = alpha_bar_t / (1 - alpha_bar_t)
weight(t) = (k + snr(t)) ** (-gamma)
```

该权重：

- 逐 sample 应用；
- 不做 batch renormalization；
- 只乘 simple denoising term；
- 不乘 learned-variance variational term；
- 不进入 sampling config；
- 与 Metrics 的 `loss_aggregation_weight` 无关。

### 2.3 Learned-range variance

`variance.mode: learned_range` 要求模型输出 `2C`：

```text
raw model output = [prediction_head, variance_head]
                     C channels       C channels
```

prediction head 按 `prediction_type` 解释；variance head 在 selected transition 的
posterior lower bound 与 forward-process upper bound 之间插值。配置与 runtime 都
fail closed：

- 声明 `DenoiserChannelLayout` 的模型在 Builder 边界检查 `C`/`2C`；
- opaque extension model 在首批 forward 检查实际 channel；
- fixed variance 不接受多余 variance head；
- learned-range 不接受缺失 variance head。

训练 metric channel 只能消费拆分后的 `C` prediction head 或 clean reconstruction，
不能把 raw `2C` tensor泄漏给 prediction/target metric。

### 2.4 Hybrid objective

learned-range loss 为：

```text
simple loss + variance loss
```

其中：

- simple loss 使用配置的 constant/P2 policy；
- mean-prediction branch 在 variance term 中 detach；
- timestep 1 使用 decoder NLL；
- 其余 timestep 使用 posterior KL；
- `rescaled_variational_bound` 在 uniform single-timestep estimator 中使用
  `T / 1000` scaling；
- variance term 不受 P2 coefficient 影响。

`GaussianLossComputation` 是这一组合结果的单一事实来源：它携带最终 loss、
prediction、target、逐项 loss 与 diagnostic values，Strategy 不重新推导 target。

### 2.5 Respaced ancestral DDPM

完整与 respaced ancestral DDPM 共用 family-specific selected-pair primitives。
对于相邻或非相邻 `(current_t, previous_t)`：

- mean、posterior variance 与 learned-range bounds 使用同一 selected pair；
- terminal `previous_t = -1` 明确返回 clean endpoint；
- uniform-section respacing 保持严格递减并包含首尾；
- `num_inference_steps == T` 与完整 DDPM 一致；
- `num_inference_steps < T` 是 ancestral DDPM，不是 DDIM。

DDIM 保留独立 generalized transition，并明确忽略 variance head。Sampler root 不新增
通用 transition API，Process root 也不吸收 model 或 sampling-loop 责任。

### 2.6 Classifier-free guidance

class-conditional learned-range 输出只对 prediction half 做 guidance：

```text
guided_prediction =
    unconditional_prediction
    + scale * (conditional_prediction - unconditional_prediction)
```

- scale `0` 返回完整 unconditional branch；
- scale `1` 返回完整 conditional branch；
- 其他 scale 使用 guided prediction 与 conditional variance half；
- DDPM 消费 variance half；
- DDIM 忽略 variance half。

## 3. Config 与 checkpoint authority

Gaussian training defaults保持：

```yaml
training:
  params:
    variance:
      mode: fixed
    loss_weighting:
      name: constant
```

P2、learned-range 与 respacing 都通过既有 registry/factory 和 typed config path
组合，不在 runner 中增加按名称或具体类分支。

新建 Gaussian checkpoint 的 inference recipe 显式冻结：

- `prediction_type`；
- `variance.mode`；
- Process identity；
- SamplingBuilder 需要的 model/channel contract。

pre-change v10 checkpoint 缺少 `variance` 时，strict resume 可以拒绝；sampling
仅在其他 model/state contract 兼容时使用 fixed-compatible default。框架不改写旧
checkpoint recipe。

## 4. AFHQ scope

AFHQ showcase 保留一个数据入口：

- `AFHQV2ImageDataSource` 发布 authenticated official train/test
  `ClassLabeledImageFolderArtifactPayload`；
- core `class_labeled_image` Builder 负责 derived validation、Dataset、Sampler、
  collate、DataLoader 与 `class_label` batch semantics；
- cat、dog、wild 都属于同一 class mapping 和同一实验；
- 不存在单独的单类别 DataSource 或 unlabeled training lane。

正式 evaluation 使用 frozen checkpoint 和统一 class allocation，至少记录：

- aggregate KID/FID；
- cat、dog、wild per-class KID/FID；
- checkpoint/weights digest；
- data artifact identity；
- sampler、steps、CFG、seed；
- metric provider、参数与依赖版本；
- output artifact hashes。

按类结果是 aggregate 结果的细分诊断，不应选择性省略某个类别，也不建立独立类别
benchmark。

## 5. Breaking policy

### 5.1 Model state

旧 ADM topology 与 canonical topology 的参数名称、数量和 skip graph 不同。以下行为
均不支持：

- strict 或 non-strict partial load；
- raw/EMA state conversion；
- optimizer-state conversion；
- 旧 topology config 自动迁移；
- 用旧结果证明新 topology 或 P2。

### 5.2 Published results

canonical cutover 前的 AFHQ checkpoint、FID/KID 与 sample panel 不属于新模型结果。
在新长训练结果冻结前，公开文档只声明当前可执行能力和 evaluation protocol，不声明
新的质量数值。

## 6. 实施切片

### A0 — ADM correctness（完成）

- topology/parameter golden；
- tiny pinned forward 与 input-gradient fixture；
- attention placement/parity；
- skip ledger 与 shape tests；
- fixed-variance train/sample；
- old-checkpoint rejection；
- public config 与文档 cutover。

### A1 — Gaussian/P2 capability（完成）

- selected-pair Process coefficients；
- fixed/learned-range variance；
- detached-mean hybrid VB；
- exact P2 weighting；
- full/respaced ancestral DDPM；
- DDIM separation；
- CFG split；
- inference recipe/channel validation；
- reference-parity fixtures。

### A2 — Class-aware evaluation（范围修订）

- 删除单类别 reproduction substrate；
- 保留 official three-class AFHQ data path；
- 保留 aggregate/per-class evaluation config、实现与测试；
- Metrics 合并后保证 prediction metric 只消费 `C` head；
- monitor 在 epoch loop 前验证 metric id、phase 与 validation-loader compatibility。

该范围已由 AFHQ example 的 class-aware evaluation 与回归测试闭合；它不依赖
dog-specific lane，也不等待 4090/DGX 性能数据。通用、跨任务的 Evaluation Operation
仍由独立 Evaluation 计划负责，不能反向把 A2 标为硬件阻塞。

## 7. Test plan

### 7.1 ADM

- exact parameter counts；
- encoder/decoder block counts；
- skip ledger exhaustion；
- attention spatial placement；
- tiny reference forward/gradient；
- odd/even shapes 与 invalid config；
- raw/EMA/optimizer old-state rejection。

### 7.2 Gaussian and P2

- all supported prediction targets；
- constant exact compatibility；
- P2 formula、broadcast、dtype/device 与 epsilon-only rejection；
- learned-range `2C` validation；
- detached mean and variance gradients；
- decoder NLL/KL boundary；
- selected-pair posterior/reference parity；
- full versus respaced DDPM；
- DDIM learned-variance ignore；
- CFG prediction/variance split；
- MPS coefficient-state regression。

### 7.3 Metrics integration

- prediction metric payload 使用 `C` mean head，而不是 raw learned-range `2C`；
- target 与 objective 使用同一 `GaussianLossComputation.target`；
- clean reconstruction metric 使用 `C` channels；
- `loss_aggregation_weight == batch size`；
- P2 timestep weights 不改变 metric batch aggregation；
- unconditional 与 class-conditional Strategy 都覆盖；
- uneven batches 的 epoch aggregation 正确。

### 7.4 AFHQ evaluation

- official source 与 class mapping；
- aggregate + cat/dog/wild per-class KID/FID；
- class allocation 与 manifest ordering；
- checkpoint/data identity；
- immutable result workspace；
- 不存在单类别 source/config/tool/test。

## 8. Validation commands

Routine validation：

```bash
uv run ruff check .
uv run pyright
uv run pytest \
  tests/test_adm_unet.py \
  tests/test_gaussian_loss_weighting.py \
  tests/test_gaussian_learned_variance.py \
  tests/test_training_strategy.py \
  tests/test_class_conditional_gaussian.py \
  tests/test_denoiser_channel_layout.py \
  tests/test_ddpm_shapes.py \
  tests/test_ddim_shapes.py \
  tests/test_class_conditional_sampling.py \
  tests/test_afhq_v2_showcase.py \
  tests/test_afhq_v2_evaluation.py
```

合并前：

```bash
uv run pytest
uv build
```

## 9. Operational evidence

以下验证建议在目标硬件执行，但不阻塞代码合并：

- corrected AFHQ-128 ADM 的 4090 BF16 capacity sweep；
- DGX Spark smoke/resume/sample；
- throughput、peak allocated/reserved VRAM 与 data-wait/compute breakdown；
- frozen class-aware evaluation run；
- 长训练 checkpoint selection 与最终质量报告。

硬件报告只能支持运行容量与性能结论，不能替代 topology、loss、variance、sampler 或
Metric contract 的自动化测试。反过来，仓库测试通过也不构成任何 GPU 吞吐或显存
承诺。

## 10. Closeout

完成条件：

- A0/A1 tests、Ruff、Pyright、full Pytest 与 build 通过；
- P2 与 Metrics 的三个语义冲突按组合语义解决；
- monitor preflight 在训练迭代前 fail closed；
- AFHQ 只保留 class-aware evaluation；
- 文档不把旧 ADM 结果归因给新 topology；
- repository 与远端目标分支 clean、可追溯。
