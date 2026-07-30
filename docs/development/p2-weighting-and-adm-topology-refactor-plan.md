# P2 Weighting 复刻与 ADM 拓扑修复计划

- 文档性质：开发计划；不属于当前公开 API 或正式文档导航
- 状态：Planned
- 制定日期：2026-07-30
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)
- 参考实现：
  [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) 与
  [P2-weighting 官方仓库](https://github.com/jychoi118/P2-weighting)
- 兼容性：Breaking；不加载、转换或部分复用旧 `adm_unet` checkpoint
- 名词说明：本文中的 **P2 weighting** 是论文算法名，不是 Roadmap 的 `P2`
  优先级

## 1. 执行结论

本计划解决两个相关但必须分开归因的问题：

1. 当前 `ADMUNet` 只有“ADM-style”外形，skip ledger、decoder block 数、
   attention 类型与位置、resampling 位置均不符合 OpenAI ADM U-Net；
2. 当前 Gaussian training 只有 fixed variance 的普通 scalar Objective，不能忠实表达
   P2 官方使用的 learned-range variance、hybrid variational bound 和
   SNR-based simple-loss weighting。

因此实施固定拆成三条线：

```text
A0  canonical ADM topology correction
    |
    v
A1  learned-range Gaussian recipe + exact P2 loss semantics
    |
    v
A2  controlled AFHQ-Dog reproduction and AFHQ-v2 product experiment
```

关键决策：

- **A0 是 correctness cutover。** 保留 registry 名 `adm_unet`，直接替换错误实现；
  不增加 `legacy` mode、旧构造参数 alias 或 state-dict adapter。
- **A0 不同时引入 P2。** topology 修复先以 fixed variance、原有训练目标通过独立
  验收，避免无法判断结果变化来自哪里。
- **A1 才实现 paper-compatible Gaussian recipe。** P2 权重只作用于 epsilon
  regression；learned variance 的 variational-bound term 不受 P2 权重影响。
- **P2 不成为通用 Objective。** 它依赖 Gaussian Process、timestep、SNR 和 prediction
  parameterization，属于 concrete Gaussian TrainingBuilder/Strategy 的私有
  loss-policy composition。
- **不建立 weighting registry。** 首版只支持 `constant` 与 `p2` 两种
  Gaussian-local declaration；第二个真实、需要外部扩展的 weighting 方法出现后，再
  决定是否公开窄 capability。
- **正式比较只使用 corrected topology。** 最小 A/B 是相同模型、数据、训练预算和
  sampling protocol 下的 `constant` 与 `P2(k=1, gamma=1)`；旧 91.3M run 只能作为
  pre-correction historical evidence。
- **论文复刻与产品实验分开。** AFHQ-Dog 256 unconditional 是 reproduction lane；
  AFHQ-v2 三类 conditional/CFG 是 framework/product lane，后者不能冒充论文结果。
- **正式长训练等待配置与指标 authority 稳定。** A0 代码修复可以先完成；A2 的新
  checkpoint 和质量声明必须在 Train/Sample authority cutover、Metrics M0–M1 和
  所需 Evaluation protocol 后执行。

## 2. Motivation

### 2.1 当前模型名称承诺超过了实现

当前 `src/stochaflow/models/adm_unet.py` 使用：

```text
每个 stage 的多个 ResBlock
  -> 一个 stage-end SpatialTransformer
  -> 保存一个 skip
  -> stage-external downsample
```

decoder 每个 stage 只在第一块消费一个 skip。以 production 的四个 stage 计算，整个
U-Net 只有四个 skip。

OpenAI guided-diffusion 的 ADM U-Net 则是：

```text
initial convolution                    -> save skip
each encoder ResBlock + optional attn  -> save skip
each downsample block                  -> save skip
middle ResBlock -> Attention -> ResBlock
each decoder ResBlock:
    pop exactly one skip
    concatenate
    ResBlock + optional attention
last decoder block at a level:
    residual upsample
```

设 resolution level 数为 `L`，每级 encoder residual block 数为 `R`：

```text
encoder skips  = 1 + L * R + (L - 1) = L * (R + 1)
decoder blocks = L * (R + 1)
```

当前实现不是一个小的参数差异，而是 skip topology 与 block graph 不同。继续把它作为
ADM 质量基线，会让模型、配置、checkpoint 和文档的含义同时失真。

### 2.2 当前 AFHQ recipe 不能回答 P2 是否有效

当前 AFHQ-v2 production recipe 是：

```text
AFHQ-v2 all classes
128×128
class conditional + classifier-free guidance
v-prediction
fixed posterior variance
cosine alpha-bar schedule
DDIM-50
84k optimizer updates
```

P2 论文的 AFHQ-D recipe 是：

```text
AFHQ-Dog single domain
256×256
unconditional
epsilon prediction
learned-range variance
linear beta schedule, T=1000
uniform timestep sampling
P2-weighted simple MSE + unweighted VB
250/1000-step ancestral sampling
2.4M images seen
```

因此不能通过给当前 v-MSE 简单乘权重后，把结果称为 P2 reproduction。

### 2.3 质量问题不只是一条 loss 公式

P2 的公开 AFHQ-D 结果是：

| Protocol | Baseline | P2 |
| --- | ---: | ---: |
| FID-50k, 1000-step ancestral | 12.47 | 11.55 |
| FID-50k, 250-step ancestral | 12.95 | 11.66 |
| KID × 1000, 1000-step ancestral | 4.79 | 4.10 |
| KID × 1000, 250-step ancestral | 5.25 | 4.20 |

这些数值同时依赖正确的 ADM topology、learned covariance、训练预算、EMA、ancestral
respacing、50,000 fake samples 和完整 training reference。当前 README 的 900-sample
official-test DDIM-50 指标与其没有可比性。

## 3. Evidence Baseline

### 3.1 当前实现与 canonical ADM 的差异

| Concern | 当前实现 | canonical guided-diffusion | 本轮决定 |
| --- | --- | --- | --- |
| Skip 保存 | 每 stage 一个 | initial、每个 ResBlock、每个 downsample | 按 input block 保存 |
| Decoder | 每级 `R` blocks，仅第一块 concat | 每级 `R + 1` blocks，每块 concat | 完整重建 |
| Downsample channel | 直接切到下一 stage width | 保持当前 width | 下一 stage ResBlock 再改宽 |
| Attention 类型 | LayerNorm + MHSA + MLP 的 Spatial Transformer | GroupNorm + QKV self-attention + residual | 新建 ADM attention block |
| Attention 位置 | stage end，可配置 arbitrary depth | active resolution 的每个 ResBlock 后 | 采用 canonical placement |
| Middle | 可配置 Transformer depth | ResBlock → Attention → ResBlock | 固定 canonical middle |
| Attention init | 普通 output projection | zero-init output projection | zero-init |
| 128 resolution levels | `[1,2,3,4]`，最低 16×16 | `[1,1,2,3,4]`，最低 8×8 | 到达 8×8 |
| Time embedding | 512-d sinusoid → 512 → 512 | base-d sinusoid → 4×base → 4×base | 改为 base → 4×base |
| Current parameter count | 91,300,867 | corrected compact 约 105.2M | 重新冻结 exact count |

保留的 Stochaflow extension：

- class-conditioning；
- 为 classifier-free guidance 保留的 null class embedding；
- registry/factory 构造；
- runtime input validation；
- PyTorch SDPA 可以作为 QKV attention 的计算 backend，但不能改变 block semantics。

### 3.2 P2 官方 recipe 的可确认事实

| 项目 | AFHQ-D official recipe |
| --- | --- |
| Model | lightweight ADM, 约 94M |
| Resolution | 256×256 |
| Base channels | 128 |
| Channel multipliers | `(1, 1, 2, 2, 4, 4)` |
| ResBlocks | encoder 每级 1；decoder 每级 2 |
| Attention | 16×16 + 8×8 middle |
| Head width | 64 channels/head |
| Resampling | BigGAN residual up/down |
| Scale/shift norm | enabled |
| Dropout | 0.1 |
| Prediction | epsilon |
| Variance | learned range |
| Schedule | linear beta, 1000 steps |
| Timestep sampling | uniform |
| P2 | `k=1`, `gamma=1` |
| Optimizer | AdamW, LR `2e-5`, weight decay 0 |
| Batch | 8 |
| Precision | FP32 |
| EMA | 0.9999 per optimizer update |
| Budget | 2.4M images seen |
| Sampling | EMA, clip denoised, 1000/250 ancestral |
| Evaluation | 50k fake vs complete train reference |

`2.4M / 8 = 300k` optimizer updates 是由论文预算和 README batch 推导出的值；论文没有
直接写出 AFHQ 的 iteration 数。

### 3.3 一手资料未冻结的 reproduction gaps

以下事实无法从 P2 论文和官方仓库完整恢复：

- 使用 AFHQ v1 还是 AFHQ-v2；
- 精确 archive checksum 和 train file list；
- training/data/generation seed；
- KID implementation、版本和完整参数；
- checkpoint selection policy；
- 是否存在没有公开的 AFHQ-specific runtime override。

所以发布声明分两级：

1. **Algorithm parity**：topology、loss、variance、respacing 与 pinned reference
   数值一致；
2. **Benchmark reproduction**：数据 manifest、metric profile、seed bank、subject
   selection 和 sample count 全部冻结后，才允许声称复现历史数值。

首个可执行历史对照应准确命名为：

> P2-compatible AFHQ-v2 Dog reproduction

在数据版本差异关闭前，不命名为“exact P2 AFHQ-D reproduction”。

## 4. Design Goals

### 4.1 必须实现

- canonical ADM input/output block graph；
- per-block skip ledger 与逐 skip decoder consumption；
- ADM QKV residual attention；
- correct 128×128 and P2 256×256 topology declarations；
- learned-range Gaussian variance；
- epsilon simple loss + detached-mean variational-bound loss；
- exact P2 weighting；
- 250-step respaced ancestral DDPM；
- learned variance 下正确的 CFG channel handling；
- fixed config/checkpoint/evaluation semantics；
- topology、loss、sampling 与 benchmark protocol 的独立测试。

### 4.2 明确不实现

- 旧 ADM constructor/state/checkpoint compatibility；
- classifier guidance；
- arbitrary P2 application to `v`, `x0` 或 score prediction；
- loss-aware timestep sampler；
- Min-SNR、EDM weighting 或 universal weighting registry；
- generic learned-variance interface on every Process/Sampler root；
- pretrained ADM/P2 checkpoint import 作为 runtime capability；
- 为 benchmark 创建 dataset-name-specific Builder；
- 用当前 900-sample evaluator声称论文级 FID。

## 5. Proposed Architecture

### 5.1 A0：ADM backbone topology

`ADMUNet` 保留一个 registry identity：

```yaml
model:
  name: adm_unet
  params:
    input_size: 128
    in_channels: 3
    out_channels: 3
    base_channels: 128
    channel_multipliers: [1, 1, 2, 3, 4]
    num_res_blocks: 2
    attention_resolutions: [32, 16, 8]
    attention_head_channels: 64
    num_classes: 3
    dropout: 0.1
```

配置决策：

- `input_size` 是模型 topology fact，不从 DataLoader 或 sample shape 猜测；
- `attention_resolutions` 使用用户可见的 spatial resolution，而不是把
  downsample factor 错叫 resolution；
- 删除 `transformer_depths` 与 `middle_transformer_depth`；
- `time_embedding_dim` 固定为 `4 * base_channels`，不重复暴露论文不需要的 knob；
- scale-shift norm、BigGAN residual resampling、residual/output zero-init 是
  `adm_unet` 的定义，不再让 production YAML 重复声明；
- `ADMUNet` 不再 nominally 继承 `ClassConditionalDenoiser`；
- `num_classes: null` 表达 unconditional ADM：不创建 class embedding，
  `forward(state, model_time)` 可被现有 unconditional Strategy 直接调用，并拒绝
  意外传入的 label；
- 正整数表达 class-conditioned ADM：
  `forward(state, model_time, class_labels)` 和
  `predict_class_conditioned(...)` 均可用，并增加一个 null class id 供 CFG 使用；
- `num_classes`/`null_class_id` 在 concrete model 上允许为 `None`，conditional
  TrainingBuilder/SamplingBuilder 继续在组合边界要求二者是合法正整数关系后，才把
  model 视为 `ClassConditionalDenoiser`；不能仅凭 constructor type 猜 capability；
- 同一个 registry identity 支持两种构造，但无条件与有条件 recipe 仍通过各自
  TrainingBuilder/SamplingBuilder 验收，不在 runner 按 `num_classes` 分支。

模型内部结构可以使用 ADM-specific block container，但不把 upstream
`TimestepEmbedSequential` 提升为 framework-wide abstraction。

### 5.2 ADM attention contract

新增 model-private `ADMAttentionBlock`：

```text
GroupNorm
  -> 1×1 QKV projection
  -> multi-head spatial self-attention
  -> zero-initialized 1×1 output projection
  -> residual add
```

约束：

- 不含 MLP；
- 不使用 LayerNorm；
- 不含 attention/output dropout；
- head channel 数必须整除 feature channels；
- 可以使用 `torch.nn.functional.scaled_dot_product_attention`，但 golden test 以
  block semantics 与数值为准，不以 backend class 名为准。

现有 `SpatialTransformer`/`SpatialTransformerLayer` 仅由错误 ADM 路径使用，直接
删除，不留 alias。

### 5.3 Gaussian loss-policy boundary

P2 不是：

```yaml
objective:
  name: p2
```

首版 API 是 concrete Gaussian TrainingBuilder 的私有参数：

```yaml
training:
  name: class_conditional_gaussian_denoising
  params:
    prediction_type: epsilon
    condition_dropout: 0.1
    variance:
      mode: learned_range
      loss: rescaled_variational_bound
    loss_weighting:
      name: p2
      k: 1.0
      gamma: 1.0
```

baseline 只改变：

```yaml
loss_weighting:
  name: constant
```

边界：

- top-level Objective 继续定义 simple prediction loss；
- P2 official profile 固定使用 `MSEObjective(reduction="mean")`；
- weighting 启用时，Strategy 依赖现有 `PerSampleObjective` capability 取得每样本
  simple loss；
- Gaussian-local policy 根据 `process.marginal_scales()` 和 timestep 产生权重；
- Strategy 对 weighted per-sample loss 做 batch mean，再与 variance term 相加；
- 其他自定义 per-sample Objective 可以复用同一机制，但结果只能称
  “P2-style weighting”，不能称官方 P2 reproduction；
- 不把 process、timestep 或 SNR 参数加入所有 Objective 的通用签名。

若未来第二个真实算法需要 extension：

```python
class GaussianLossWeighting(Protocol):
    def weights(
        self,
        process: DiscreteGaussianDenoisingProcess,
        state_times: torch.Tensor,
    ) -> torch.Tensor: ...
```

届时再评估 public protocol/registry；本轮不提前发布。

### 5.4 Exact P2 weighting

定义：

\[
\operatorname{SNR}(t)
=
\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
=
\frac{s_t^2}{n_t^2}
\]

\[
w_t=(k+\operatorname{SNR}(t))^{-\gamma}
\]

\[
L_{\text{simple}}
=
\frac{1}{B}\sum_{i=1}^{B}
w_{t_i}
\operatorname{MSE}_i(\epsilon_\theta,\epsilon)
\]

实现约束：

- `state_times` 使用 Process 的 public state `1..T`；映射到模型 timestep `0..T-1`
  时不能 off-by-one；
- SNR 来自 cumulative marginal scales，不使用单步 alpha；
- 权重不做 batch normalization 或 mean renormalization；
- `k` 必须 finite 且大于 0；
- `gamma` 必须 finite 且非负；
- `gamma=0` 必须逐元素退化为 weight 1；
- `k=1, gamma=1` 在 VP schedule 下必须等于 `1 - alpha_bar_t`；
- P2 只在 `prediction_type: epsilon` 下称为官方支持；
- timestep sampling 仍是 uniform，不额外做 importance correction。

### 5.5 Learned-range variance

模型输出：

```text
first C channels  -> epsilon/model prediction
last C channels   -> variance interpolation values
```

learned-range mapping：

\[
f=(v_\theta+1)/2
\]

\[
\log \sigma_\theta^2
=
f\log\beta_t
+(1-f)\log\tilde{\beta}_t
\]

其中 lower/upper bounds 属于 model-free discrete Gaussian process facts。

新增窄 Gaussian-family capability，而不是给所有 Process 增加 optional 方法：

```python
@runtime_checkable
class LearnedRangeGaussianVarianceProcess(Protocol):
    def reverse_log_variance_bounds(
        self,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> GaussianLogVarianceBounds: ...
```

`DiscreteGaussianProcess` 实现该 capability。TrainingBuilder 和 SamplingBuilder 在
完整组合边界验证 `learned_range` 所需 capability。

sampling prediction 使用窄 value type：

```python
@dataclass(frozen=True, slots=True)
class LearnedVarianceGaussianPrediction(GaussianPrediction):
    log_variance: torch.Tensor
```

这样 fixed-variance consumer 继续只依赖 `GaussianPrediction`；只有 ancestral
transition 消费 subclass 的 variance，不在 root Dynamics 添加一组 nullable 方法。

### 5.6 Hybrid objective

官方语义：

\[
L_{\text{hybrid}}
=
L_{\text{simple}}^{P2}
+10^{-3}L_{\text{VLB}}
\]

在官方 uniform single-timestep estimator 中，`rescaled_variational_bound` 对 VB
样本项乘 `T / 1000`；`T=1000` 时数值因子为 1。

执行顺序固定：

```python
model_prediction, variance_values = split(raw_model_output)

variance_loss = variational_bound(
    model_prediction.detach(),
    variance_values,
)

simple_loss = objective.per_sample_loss(
    model_prediction,
    epsilon_target,
)

loss = mean(p2_weight * simple_loss + variance_loss)
```

约束：

- P2 不乘 variance loss；
- variance loss 中 mean/prediction branch 必须 detach；
- timestep 1 使用 discretized Gaussian decoder NLL，其余使用 posterior KL；
- fixed variance path 不计算 VB；
- variance mode、prediction type 和 raw output layout 在 TrainingBuilder 预检，并在
  首批 runtime shape validation 中 fail closed；
- 不把 VB 伪装成第二个通用 Objective 或独立 optimizer。

### 5.7 Respaced ancestral transition

当前 `DDPMAncestralSampler` 只能执行相邻 `t -> t-1`，不能复现 P2 的 250-step
ancestral protocol。A1 增加：

```yaml
sampler:
  name: ddpm
  params:
    num_inference_steps: 250
```

`num_inference_steps` 与 explicit schedule 互斥。1000-step 默认保持完整相邻 path。

Process family 提供 model-free selected-pair marginal coefficient snapshot：

```text
(alpha_bar_s, alpha_bar_t, alpha_bar_t / alpha_bar_s),
0 <= s < t <= T
```

DDPM 使用这些系数构造 `q(x_s | x_t, x0)` posterior；DDIM 使用相同 marginal
coefficients 构造自己的 implicit/generalized `eta` transition。二者不能共同委托一个
DDPM posterior mean，因为 `eta=0` 的 DDIM update 并不等于 posterior mean，
`eta>0` 的 generalized variance 也不是普通 DDPM posterior variance。

250-step schedule 与 improved-diffusion 的 uniform section respacing 对齐；它不是：

- DDIM-250；
- 把 1000-step loop 简单跳过 model evaluation；
- adjacent posterior variance 配 non-adjacent mean。

learned-range 的 lower/upper log-variance bounds 必须基于 respaced transition，而不是
错误复用原始相邻 beta。

### 5.8 CFG with `2C` output

class-conditioned learned-variance sampling 固定：

```text
conditional output   = [conditional mean head, conditional variance head]
unconditional output = [unconditional mean head, unconditional variance head]

when scale not in {0, 1}:
    guided mean head = uncond + scale * (cond - uncond)
    variance head    = conditional variance head
```

CFG 只作用于 denoising prediction channels。不能对 variance half 做同样的线性外推。
为保留用户可理解的 endpoint semantics：

- `guidance_scale == 0` 返回完整 unconditional `2C` output；
- `guidance_scale == 1` 返回完整 conditional `2C` output；
- 其他 scale 只外推 mean/prediction half，variance half 使用 conditional branch，并将
  该 policy 写入 inference recipe/evaluation protocol。

unconditional P2 reproduction 不经过 CFG。

### 5.9 Diagnostics and Metrics

Gaussian training diagnostics 明确输出：

```text
timesteps
snr
timestep_loss_weight
per_sample_simple_loss
per_sample_weighted_simple_loss
per_sample_variational_bound
per_sample_loss
```

其中 `per_sample_loss` 是实际优化的 composite term。

Metrics 计划中的 epoch loss 聚合权重必须命名为
`loss_aggregation_weight`，它表示不同 batch 对 epoch aggregate 的统计权重，不参与
autograd。P2 的 `timestep_loss_weight` 是 timestep-dependent 可微目标系数；两者不得
复用字段、配置或日志语义。

### 5.10 Data boundary for AFHQ-Dog

论文复刻需要 unconditional dog-only image batches，但当前
`class_labeled_image` Builder 正确地产生 class label batch。不能增加
`drop_labels: true` mode 让一个 Builder 同时承担两种 batch contract。

首选接入：

```text
AFHQV2DogImageDataSource
    -> generic ImageFolderArtifactPayload
    -> existing image DataBuilder
    -> unconditional Tensor batches
```

该 DataSource：

- 只复用 AFHQ extension 的 source acquisition lock、safe archive 和认证逻辑；
- 不复用当前一次性 PIL Lanczos resize helper；benchmark transform 单独对齐并 pin
  guided-diffusion loader 的逐级 BOX downsample、bicubic resize 与 center crop；
- 注册独立 source identity，例如 `afhq-v2.dog`；
- 只物化 authenticated train dog subset at 256×256；
- 返回通用无标签 image artifact contract；
- 不创建 Dataset、Sampler 或 DataLoader；
- 不增加 dataset-name-specific Builder。

它仍然是 AFHQ domain-specific source，而不是 framework 新抽象。

历史 P2 使用的数据版本未锁定，因此 artifact identity 必须准确记录 Stochaflow 使用的
AFHQ-v2 archive 与 file inventory，不伪造论文 identity。

### 5.11 Config and checkpoint authority

- production examples 仍只保留一个 canonical AFHQ ADM production train config；
- 不永久新增 `train-adm-baseline.yaml` 与 `train-adm-p2.yaml`；
- reproduction protocol 放在 AFHQ showcase 的 benchmark/research 区域，不进入
  maintained production train surface；
- A/B 通过同一 base config 和一个显式、受限的 training override 生成两个 resolved
  configs；
- P2 是训练变化，不能放入 sample profile；
- sample profile 只选择 DDPM/DDIM、steps、seed、conditions、weights 与 writers；
- `inference_recipe` 保存 `prediction_type` 和 `variance.mode`；
- P2 `k/gamma` 是 training/resume fact，不是 inference override；
- strict resume 使用完整 resolved config 固定 weighting 与 variance-loss policy。

## 6. Breaking Policy

### 6.1 Old ADM checkpoints

删除：

- `transformer_depths`；
- `middle_transformer_depth`；
- 旧 `SpatialTransformer` state；
- 旧 stage-level skip graph。

旧 checkpoint 的 raw model、EMA 和 optimizer state 全部不兼容：

- 不提供 partial state load；
- 不做 key mapping；
- 不保留 `adm_unet_legacy`；
- 不自动改写 resolved config；
- sampling 与 resume fail closed；
- 用户必须启动 fresh run。

全局 checkpoint schema 不需要只为 topology 单独 bump；现有 strict state shape/key
validation 已能拒绝旧 state。若 B1 同期一次性升级 checkpoint authority，则 breaking
说明合并进入该 schema release。

### 6.2 Existing AFHQ result

当前 README 的 epoch-170、FID/KID 与 sample panel属于旧 91.3M topology。A0 完成时：

- 从“当前 production config 的 measured result”中移除；
- 不用于证明 corrected ADM 或 P2；
- 若保留历史记录，必须带 pre-topology-fix 标记、原 commit/resolved config 和
  checkpoint digest；
- 新 production config 不得链接旧结果作为可复现输出；
- corrected topology 完成长训练后再发布新的 current result。

## 7. Implementation Phases

### Phase A0.0 — Reference freeze and characterization

1. pin guided-diffusion 与 P2 upstream commit；
2. 写 topology ledger、attention placement、parameter-count golden；
3. 冻结当前 91,300,867 参数和旧 config/checkpoint rejection characterization；
4. 生成独立 tiny reference 的 forward/gradient fixtures；
5. 不修改训练 recipe。

退出条件：测试能先在当前实现上准确失败，并指出 topology 差异，而不是只比较一个总
parameter count。

### Phase A0.1 — Canonical ADM topology cutover

1. 新增 ADM attention block；
2. 重建 encoder input-block ledger；
3. 重建 `R + 1` decoder blocks；
4. down/up resampling 保持当前 level channel；
5. 修正 time embedding；
6. 支持 unconditional/class-conditioned construction；
7. 删除旧 transformer types 与构造参数；
8. 更新 tiny/production config 和 fixtures。

退出条件：

- `L * (R + 1)` skips 全部且仅消费一次；
- 128 model 到达 8×8；
- attention placement、zero-init 和 shape parity 通过；
- fixed-variance train/sample smoke 通过；
- 旧 checkpoint fail closed。

### Phase A0.2 — Capacity and public-doc cutover

1. 对 corrected AFHQ 128 config 重新计算参数、FLOPs 和 activation peak；
2. 在 4090 上实测 BF16 microbatch 1/2/4/8；
3. 在 DGX Spark 上执行相同 smoke profile；
4. 根据测量决定 production microbatch/accumulation；
5. 不先验承诺当前 microbatch 8 仍可用；
6. 更新公开模型/config/tutorial 描述；
7. 处理旧 README result attribution。

activation checkpointing 只有在 profile 证明必要时才进入独立小切片；它不成为 ADM
topology 的隐藏默认值。

### Phase A1.0 — Gaussian selected-pair and variance primitives

1. 实现 selected-pair marginal coefficient snapshot；
2. 让 DDPM 基于 snapshot 构造 selected-pair posterior；
3. 让 DDIM 基于 snapshot 保留自己的 generalized `eta` transition；
4. 实现 learned-range log-variance bounds；
5. 增加 learned-variance prediction value type；
6. 不改变 fixed-variance默认行为。

### Phase A1.1 — Learned-range training and sampling

1. 支持 `2C` raw model output；
2. 实现 split、variance mapping、KL/decoder NLL；
3. 实现 detached mean branch；
4. DDPM 使用 learned transition variance；
5. DDIM 明确忽略 learned variance 但正确消费 mean head；
6. CFG 只 guide mean head；
7. 将 variance mode 写入 inference recipe；
8. 更新 diagnostics。

### Phase A1.2 — P2 weighting

1. 实现 `constant`/`p2` Gaussian-local parser；
2. 实现 SNR 与 weight；
3. 限制 official P2 declaration 到 epsilon prediction；
4. 复用 `PerSampleObjective`；
5. P2 只乘 simple term；
6. 增加 gamma-zero、closed-form 与 upstream numeric parity tests；
7. 不改变 default config，直到 experiment gate。

### Phase A1.3 — Respaced ancestral DDPM

1. `ddpm.num_inference_steps`；
2. official uniform section schedule parity；
3. non-adjacent learned-range transition；
4. 250/1000 deterministic seed and observer parity；
5. sampling profile 与 checkpoint contract tests。

### Phase A2.0 — AFHQ-v2 Dog benchmark substrate

依赖 Evaluation E0–E1 foundation；它只解锁不报告 FID/KID 的 operational pilot。
任何 metric pilot 和正式 50k benchmark 还必须先完成 E2 prediction
artifact/completeness 与 E3 FID/KID、reference cache、Gaussian generation profile
的所需子集。

1. 新增 dog-only DataSource；
2. 冻结 AFHQ-v2 archive/file inventory；
3. 用 pinned guided-diffusion BOX/bicubic/center-crop algorithm materialize
   256×256 RGB data，并以 golden fixture 对齐 upstream pixel output；
4. 使用 generic unlabelled image Builder；
5. 冻结 benchmark base config、seed bank、EMA 和 budget；
6. 完成 short overfit 与 4090/DGX capacity trial。

### Phase A2.1 — Controlled pilot

相同初始权重和顺序运行：

| Run | Topology | Mean | Variance | Weighting | Budget |
| --- | --- | --- | --- | --- | --- |
| baseline | corrected ADM | epsilon | learned range | constant | pilot |
| weighted | corrected ADM | epsilon | learned range | P2 `k=1, γ=1` | pilot |

第一段 operational pilot 在 E0–E1 后执行，只用于：

- loss/gradient/throughput 稳定性；
- checkpoint/sample/evaluation operational validation；
- 发现数据或 evaluator 偏差。

E2/E3 所需子集完成后，第二段 metric pilot 才运行 FID-10k/KID-10k 趋势。两段都使用
相同 frozen base protocol，不能因先看到 samples 再改变 A/B 唯一变量。

pilot 不发布论文级质量声明。

### Phase A2.2 — Formal P2-compatible reproduction

固定：

- 2.4M images seen；
- effective batch 8；
- constant LR `2e-5`；
- EMA 0.9999；
- FP32 reference；若 4090 需要 microbatch accumulation，完整记录为 hardware
  adaptation；
- fixed final-budget checkpoint，不用 test/FID 在多个 checkpoint 间挑选；
- 50k fake；
- 1000-step 和 250-step ancestral；
- full authenticated training reference；
- frozen FID/KID implementation and preprocessing；
- resumable sample shards 与 completeness manifest。

报告：

- baseline 与 P2 的 absolute score；
- relative delta；
- sampling NFE、throughput 和 peak memory；
- 与论文数值的差异及已知 reproduction gaps；
- 不只报告胜出的 run。

### Phase A2.3 — Three-class AFHQ-v2 product experiment

在 corrected 128 class-conditioned topology 上执行独立 A/B：

```text
same data partitions
same initialization
same update/image budget
same epsilon + learned variance recipe
same CFG search protocol
constant vs P2 only
```

该实验回答：

> P2 capability 是否改善 Stochaflow 当前三类 AFHQ workflow？

它不回答：

> 是否复现了 P2 论文的 AFHQ-D 数值？

只有 validation 选择一次 frozen subject/CFG 后，才对 official test 执行一次最终评估。

## 8. Experiment Promotion Gates

### 8.1 Algorithm parity gate

必须全部满足：

- corrected P2 256 unconditional topology 参数量为 93,563,910；
- topology ledger 与 pinned reference 一致；
- `gamma=0` 与 unweighted per-sample MSE 一致；
- `k=1, gamma=1` 与 `1-alpha_bar` 一致；
- P2 不改变 VB；
- VB 不向 mean branch 传播梯度；
- learned-range endpoints 正确；
- 250-step respacing 与 pinned reference schedule/transition 一致；
- fixed variance old-path regression 通过。

### 8.2 Benchmark evidence gate

“P2 improved this Stochaflow baseline”至少要求：

- 同一 frozen protocol；
- 两个 run 都 complete；
- 50k FID/KID 或预先声明的完整 protocol；
- P2 的 FID 与 KID 都不劣于 baseline；
- delta 大于 evaluator 重复运行的数值噪声；
- raw results、sample manifests 和 resolved configs 同时保存。

“Reproduced P2 AFHQ-D”还额外要求：

- 历史 dataset version/file list 关闭；
- metric implementation 可与论文实现对齐；
- checkpoint selection 与 seed uncertainty 已明确处理；
- 预先冻结 numeric tolerance，而不是看到结果后修改阈值。

若这些事实无法恢复，发布名称保持“P2-compatible AFHQ-v2 Dog reproduction”。

### 8.3 Production promotion gate

将 P2 升为唯一 canonical AFHQ ADM production train recipe 前要求：

- 三类 conditional experiment 在 validation protocol 上改善；
- official test 只运行一次；
- 4090 production memory/throughput 可接受；
- sample diversity/precision/recall 没有明显退化；
- no-progress local logger、resume、EMA 和 independent sample 全部通过；
- canonical config 仍可在一屏内解释核心 recipe。

未通过时 capability 保留为 experimental Gaussian training option，不修改 canonical
AFHQ train config。

## 9. File-level Change Map

### 9.1 Core model

- `src/stochaflow/models/adm_blocks.py`
  - 删除 Spatial Transformer；
  - 新增 ADM attention；
  - 保留/校正 residual resampling。
- `src/stochaflow/models/adm_unet.py`
  - corrected block graph；
  - two-argument unconditional forward 与显式 class-conditioned capability；
  - new readable topology config。

### 9.2 Gaussian process, training and sampling

- `src/stochaflow/processes/gaussian.py`
  - narrow selected-pair marginal coefficient/variance capabilities。
- `src/stochaflow/processes/discrete_gaussian.py`
  - marginal coefficient snapshot、DDPM posterior 与 log-variance math。
- `src/stochaflow/training/gaussian_loss.py`
  - family-specific P2 + learned-variance hybrid loss helper。
- `src/stochaflow/training/objectives.py`
  - 将现有 `PerSampleObjective` capability 的说明从 diagnostic-only 收紧为可供
    training policy 与 diagnostics 共同消费；不改变其签名。
- `src/stochaflow/training/gaussian.py`
  - unconditional builder/strategy composition。
- `src/stochaflow/training/class_conditional_gaussian.py`
  - conditional builder/strategy composition and diagnostics。
- `src/stochaflow/sampling/gaussian.py`
  - learned-variance prediction semantics。
- `src/stochaflow/sampling/ddpm.py`
  - respaced ancestral transition。
- `src/stochaflow/sampling/ddim.py`
  - shared marginal coefficients、DDIM-owned `eta` transition 与 explicit
    learned-variance ignore。
- `src/stochaflow/sampling/class_conditional.py`
  - `2C` CFG partition behavior。

不在 runner 中增加 `if model.name == "adm_unet"` 或 name-based compatibility table。

### 9.3 AFHQ extension and configs

- AFHQ extension新增 dog-only DataSource；
- production/smoke ADM config 改用 corrected topology fields；
- production train surface仍只有一个 canonical AFHQ ADM production config；
- benchmark/research protocol 与 production config 分目录；
- sample profile只增加 250/1000 DDPM 变体，不包含 training fields。

### 9.4 Tests

更新现有测试：

- `tests/test_adm_unet.py`
- `tests/test_class_conditional_integration.py`
- `tests/test_training_strategy.py`
- `tests/test_class_conditional_gaussian.py`
- `tests/test_ddpm_shapes.py`
- `tests/test_ddim_shapes.py`
- `tests/test_class_conditional_sampling.py`
- `tests/test_checkpoint.py`
- `tests/test_sampling_runtime.py`
- `tests/test_config.py`
- `tests/test_config_reference.py`
- `tests/test_data_sources.py`
- `tests/test_afhq_v2_showcase.py`
- `tests/test_afhq_v2_evaluation.py`

A1 新增：

- `tests/test_gaussian_loss_weighting.py`
- `tests/test_gaussian_learned_variance.py`

## 10. Test Plan

### 10.1 ADM topology

- skip count 精确等于 `L * (R + 1)`；
- decoder block 数与 skip 数相同；
- 每个 decoder block消费一个 skip；
- downsample保持 current-level channels；
- next-level first ResBlock改变 channels；
- attention只在声明 resolution，每个 ResBlock 后出现；
- middle始终只有一组 ADM attention；
- 128 production attention count：
  encoder 6 + decoder 9 + middle 1 = 16；
- attention使用 GroupNorm、无 MLP、zero output projection；
- time MLP 为 `base -> 4*base -> 4*base`；
- output projection zero-init；
- forward/backward、state round-trip、mixed precision；
- `num_classes: null` 经过 unconditional TrainingBuilder/Strategy 和 standard
  denoising SamplingBuilder 的集成路径；
- 正整数 `num_classes` 经过 conditional TrainingBuilder/Strategy 与 CFG
  SamplingBuilder 的集成路径；
- invalid input size/resolution/head divisibility fail closed；
- old config fields rejected；
- old checkpoint raw/EMA state rejected。

### 10.2 P2 and hybrid loss

- exact SNR values；
- state/model timestep off-by-one；
- constant and gamma-zero parity；
- `k=1,gamma=1` closed form；
- no weight renormalization；
- epsilon-only validation；
- per-sample reduction；
- P2 simple term only；
- detached mean gradient；
- KL and t=1 decoder NLL；
- `T/1000` VB rescaling；
- non-finite `k/gamma` rejection；
- independent custom `PerSampleObjective` path marked P2-style。

### 10.3 Learned variance and sampling

- raw C/2C layout validation；
- range endpoints and extrapolation semantics；
- conditional-only variance under CFG；
- CFG scale 0/1 完整分支 parity 与其他 scale 的 conditional-variance policy；
- adjacent learned transition；
- 250-step non-adjacent transition；
- uniform section schedule parity；
- DDIM mean parity and explicit variance ignore；
- observer coordinates/NFE；
- EMA/raw independent sampling；
- inference recipe rejects variance mismatch。

### 10.4 Data and benchmark

- dog-only source emits generic image payload；
- no class labels enter unconditional batch；
- source materialization owns no Dataset/loader；
- archive/file identity frozen；
- guided-diffusion BOX/bicubic/center-crop pixel fixture parity 与 horizontal flip
  policy；
- sample shard resume and exact 50k completeness；
- reference/fake protocol identity mismatch rejected；
- validation selection cannot consume official test；
- old 900-sample and new 50k results cannot share a comparison id。

## 11. Validation Commands

A0 topology cutover：

```bash
uv run pytest \
  tests/test_adm_unet.py \
  tests/test_class_conditional_integration.py \
  tests/test_class_conditional_gaussian.py \
  tests/test_class_conditional_sampling.py \
  tests/test_training_strategy.py \
  tests/test_checkpoint.py \
  tests/test_sampling_runtime.py \
  tests/test_config.py \
  tests/test_config_reference.py \
  tests/test_afhq_v2_showcase.py
```

A1 Gaussian/P2 capability（其中两个 `test_gaussian_*` 文件由 A1 新增）：

```bash
uv run pytest \
  tests/test_gaussian_loss_weighting.py \
  tests/test_gaussian_learned_variance.py \
  tests/test_training_strategy.py \
  tests/test_class_conditional_gaussian.py \
  tests/test_ddpm_shapes.py \
  tests/test_ddim_shapes.py \
  tests/test_class_conditional_sampling.py
```

A2 data/evaluation：

```bash
uv run pytest \
  tests/test_data_sources.py \
  tests/test_afhq_v2_showcase.py \
  tests/test_afhq_v2_evaluation.py
```

每个 phase 的静态检查：

```bash
uv run ruff check .
uv run pyright
```

Config/docs closeout:

```bash
uv run python tools/generate_config_reference.py
uv run python tools/generate_config_reference.py --check
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Feature-branch merge gate:

```bash
uv build
uv run pytest
```

GPU acceptance separately记录：

```text
4090: corrected ADM-128 BF16 forward/backward + short train/resume/sample
4090: P2-256 FP32 effective-batch-8 capacity trial
DGX Spark: same resolved configs and checkpoint continuation
```

## 12. Documentation Closeout

A0/A1 implementation完成后必须同步：

- `README.md`
- `docs/framework.md`
- `docs/tutorials/afhq-v2.md`
- `examples/showcases/afhq-v2/README.md`
- `docs/configuration/_reference.yaml`
- generated `docs/configuration/reference.md`
- `docs/configuration/compatibility-and-migration.md`
- `docs/configuration/index.md`
- `docs/configuration/extensions.md`

公开文档必须说明：

- `adm_unet` 使用 canonical ADM block graph；
- `num_res_blocks` 是 encoder 每 resolution 的 block 数，decoder 为 `R+1`；
- `attention_resolutions` 是 spatial resolution；
- old ADM checkpoint unsupported；
- fixed vs learned-range variance；
- P2 只支持 epsilon paper-compatible semantics；
- P2 与 Metrics aggregation weight 不同；
- DDPM-250 是 respaced ancestral，不是 DDIM；
- current 900-sample AFHQ protocol与 P2 FID-50k不可比较；
- P2 benchmark claim 的 dataset/version限制。

development docs 同步：

- 本计划状态与验收记录；
- [Development Priority Roadmap](development-priority-roadmap.md)；
- [Metrics 支持计划](metrics-support-plan.md)；
- [Hydra 配置迁移计划](hydra-configuration-composition-migration-plan.md)；
- [Post-training Evaluation 计划](post-training-evaluation-support-plan.md)；
- [Latent Diffusion 支持计划](latent-diffusion-support-plan.md)。

public docs 不链接本 development plan。

## 13. Risks and Open Questions

### 13.1 Historical data ambiguity

AFHQ version/file list无法从 P2 publication完全恢复，可能造成绝对 FID 偏差。解决方式是
准确发布 Stochaflow protocol identity，而不是用模糊“same dataset”掩盖。

### 13.2 Learned variance scope

learned variance横跨 Process、training、CFG、DDPM 和 diagnostics。A0 必须独立提交；
A1 不能只把 `out_channels` 改成 6 就宣称支持。

### 13.3 4090 memory

corrected ADM-128 约增加 15% parameters，并保留更多 skip activations。microbatch 8
可能失效；只能根据实测调整 accumulation 或决定 activation checkpointing。

### 13.4 P2 expected gain

论文 AFHQ-D FID 改善约 7%，不是数量级变化。若 topology、预算、evaluator 或数据版本
未对齐，P2 不会单独把当前 900-sample FID 变成论文数值。

### 13.5 Metric variance

小数据集对 FID/KID 和 checkpoint choice敏感。正式报告必须保存全部候选、预先声明
selection policy，并区分 evaluator noise 与训练 seed variance。

### 13.6 Latent reuse

P2 capability未来可以用于 latent epsilon recipe，但：

- 它不是 latent Phase 1 correctness前置；
- 不自动成为 DiT 或 Stable Diffusion默认；
- v-prediction需要重新推导 parameterization-correct weighting；
- official DiT learned sigma需要 A1 的 `2C`/variance capability，但首个 latent
  fixed-variance slice不等待它。

## 14. Source References

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [OpenAI guided-diffusion README and 128×128 flags](https://github.com/openai/guided-diffusion)
- [OpenAI ADM U-Net implementation](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py)
- [OpenAI Gaussian learned-variance implementation](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)
- [Perception Prioritized Training of Diffusion Models](https://arxiv.org/abs/2204.00227)
- [P2-weighting official implementation](https://github.com/jychoi118/P2-weighting)
- [P2 supplemental hyperparameter table](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Choi_Perception_Prioritized_Training_CVPR_2022_supplemental.pdf)
