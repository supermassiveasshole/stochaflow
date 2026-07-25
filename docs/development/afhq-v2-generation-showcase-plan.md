# AFHQ-v2 类条件生成展示实施计划

- 文档性质：开发草案；不属于当前公开 API 或正式文档导航
- 状态：提案，尚未进入实现
- 制定日期：2026-07-26
- 首版展示：AFHQ-v2、128×128、cat/dog/wild 类条件 Gaussian diffusion
- 目标环境：Windows、单张 CUDA GPU、连续训练约 24–48 小时
- 训练默认：200 epochs、micro batch 8、gradient accumulation 4、
  effective batch 32、`num_workers: 2`
- 关联计划：
  [Metrics 支持开发计划](metrics-support-plan.md)、
  [训练后 Evaluation 与 Benchmark 支持计划](post-training-evaluation-support-plan.md)、
  [默认工作流与推理 Pipeline 支持计划](default-workflow-pipeline-support-plan.md)、
  [Latent Diffusion、DiT 与 Stable Diffusion 支持计划](latent-diffusion-and-stable-diffusion-support-plan.md)

## 1. 目标与核心决策

本阶段不只增加一个 AFHQ 配置，而是完成一条可复用的高质量图像生成纵向切片：

1. Trainer 原生支持 AMP 和 Gradient Accumulation；
2. 框架提供安全、可复现的数据 artifact 与 provenance 基础能力；
3. 新增能读取已准备分类图像目录的内置 DataBuilder；
4. 新增类条件 Gaussian training、diagnostic 与 classifier-free guidance
   sampling 组合；
5. 新增一个带低分辨率 Transformer blocks 的 ADM 风格 UNet；
6. 用 AFHQ-v2 showcase 串起下载、预处理、训练、resume、TensorBoard、
   采样和最终质量评估。

首版展示模型选择新的 `adm_unet`，而不是直接把纯 pixel DiT 作为唯一基线。
AFHQ-v2 规模约 1.58 万张，动物毛发、眼睛和口鼻等局部结构对卷积归纳偏置很敏感；
128×128 pixel DiT 若采用 patch 4 会产生 1024 tokens，单卡成本较高，采用 patch 8
又更容易损失细节。`adm_unet` 在 32×32 和 16×16 feature maps 上加入
Transformer blocks，先获得全局表达能力，同时保留高分辨率卷积路径。

原生 `dit` 仍包含在本计划的后续阶段，用作 architecture comparison；它不阻塞
AFHQ 首个可展示结果。256×256 及以上的 latent DiT 需要 frozen codec、
latent scaling 和 codec provenance，应按
[独立 Latent Diffusion 计划](latent-diffusion-and-stable-diffusion-support-plan.md)
推进，不在本阶段暗中引入。AFHQ 的 class-aware data、条件 denoiser 与 CFG 是该计划的
前置能力；后续 latent 项目复用这些窄契约，而不是把 VAE、DiT 或 Stable Diffusion
反向塞进本计划的 pixel baseline。

AMP 与 Gradient Accumulation 是 Trainer 的通用自动优化能力，不属于 AFHQ
DataBuilder、model 或 example。现有配置默认仍是 `fp32` 和一次一更新，因此已有
训练行为和 checkpoint 恢复边界保持明确。

## 2. 成熟实现调研与采用结论

### 2.1 数据集与数据准备

[StarGAN v2 官方仓库](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)
说明 AFHQ-v2 包含 15,803 张 512×512 PNG，分为 cat、dog、wild 三个 domain；
新版使用 Lanczos 重建并采用 CC BY-NC 4.0。官方数据应是默认来源和计数、许可、
引用信息的权威依据。

[Hugging Face Hub 下载接口](https://huggingface.co/docs/huggingface_hub/en/guides/download)
支持按完整 commit revision 固定 snapshot。首版可提供显式的 HF 来源适配器，
但不能使用浮动的 `main`，也不能在官方来源失败后静默切换到社区镜像。
当前可评估的完整镜像是
[`ryushinn/AFHQv2`](https://huggingface.co/datasets/ryushinn/AFHQv2)；
[`huggan/AFHQv2`](https://huggingface.co/datasets/huggan/AFHQv2)
当前展示的样本规模不同，不能在未完成计数、split 和文件摘要核验前作为等价来源。

[EDM 数据准备流程](https://github.com/NVlabs/edm#preparing-datasets) 和
[StyleGAN3 的 AFHQ-v2 准备方式](https://github.com/NVlabs/stylegan3)
都把“原始下载”与“训练时读取的确定性 artifact”分开。本计划采用相同原则：
训练不临时下载、重采样或猜测 split；所有不可逆预处理先离线完成并写入 manifest。

### 2.2 模型与条件生成

[OpenAI guided-diffusion](https://github.com/openai/guided-diffusion)
验证了 scale-shift normalization、attention、residual resampling 和 class
conditioning 在 diffusion UNet 中的成熟组合。首版 `adm_unet` 采用这些结构原则，
但作为新注册模型实现，不修改已有 `unet` 的参数拓扑。

[DiT 官方实现](https://github.com/facebookresearch/DiT) 使用 adaLN-Zero、
固定二维位置编码、零初始化输出和 class dropout。它适合作为后续原生
Transformer baseline；首版混合模型的 attention 使用 PyTorch
[`scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)，
不复制第三方 attention kernel。

[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
通过训练期随机丢弃条件和采样期组合 conditional/unconditional prediction 实现引导。
因此 class dropout 属于 TrainingStrategy，CFG 属于 SamplingBuilder，二者都不属于
Process 或 Sampler。

### 2.3 AMP 与 Gradient Accumulation

[PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
明确要求在完整 effective batch 内保持同一 loss scale，仅在累积完成后
`unscale -> clip -> step -> update`。forward/loss 在 autocast 中，backward
在 autocast 外执行。

[Lightning mixed precision](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.precision.MixedPrecision.html)
把 forward context、backward、optimizer step 和可恢复 precision state 收敛在一个
窄的 precision collaborator 中；其 `accumulate_grad_batches` 也属于自动优化循环，
而不是 model。

[Hugging Face Accelerate](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)
同样把 scheduler 与 optimizer update 对齐，而不是按输入 micro-batch 推进。

因此本计划不会在每个 Strategy 中散落 `autocast`/`GradScaler`，也不会把
accumulation 做成 AFHQ 私有参数。

## 3. 范围与非目标

### 3.1 首版必须完成

- `fp32`、`bf16-mixed`、`fp16-mixed` 三种 Trainer precision；
- 单 optimizer 自动训练循环的固定 Gradient Accumulation；
- AMP state 的严格 checkpoint/resume；
- epoch 末不足一个 accumulation window 时的正确 flush；
- AMP、accumulation、overflow、scheduler、EMA、diagnostic 的一致 step 语义；
- 安全下载/缓存/解包、内容摘要、原子发布和数据 manifest 基础能力；
- 已准备分类图像目录的内置 DataBuilder 与严格 data provenance；
- class-conditional Gaussian TrainingBuilder/Strategy；
- class-conditional SamplingBuilder 与 CFG；
- class-conditional diagnostic capability；
- 新 `adm_unet`；
- AFHQ-v2 准备脚本、生产配置、smoke 配置、教程和固定质量产物规范。

### 3.2 明确不进入首版

- DDP/FSDP、多 optimizer、alternating updates、manual backward；
- 动态 accumulation schedule 或按 epoch 改变 accumulation；
- FP8、true-half 参数、自动 precision fallback；
- mid-epoch checkpoint、待累积 gradient 或 DataLoader cursor 恢复；
- 把任意 Hugging Face/Kaggle/URL provider 镜像成框架 registry；
- 通用 Dataset/Transform/Sampler/DataLoader YAML graph；
- silent source fallback、浮动 Hub revision、运行时在线 resize；
- `torch.compile`、activation checkpointing 和 accumulation 的自动联调；
- text conditioning、cross-attention、latent VAE 或 256×256 生产训练；
- 以 GPU utilization 百分比作为唯一性能验收。

多 optimizer 或 closure-required optimizer 仍定义新的训练循环 family；本计划不通过
给自动循环增加 mode flags 来伪装支持。

## 4. 责任边界

| 组件 | 本阶段职责 | 明确不负责 |
| --- | --- | --- |
| `Trainer` | micro-batch window、backward、clip、optimizer update、step 生命周期 | 解释 batch、创建模型、猜测样本数 |
| `PrecisionRuntime` | autocast、可选 GradScaler、AMP step 与 precision state | scheduler、EMA、checkpoint 路径、任务逻辑 |
| `TrainingStrategy` | 解释 class batch、生成 diffusion target、计算 loss | zero-grad、backward、step、class embedding 构造 |
| `TrainingBuilder` | 组装并验证条件训练能力 | 修改 Trainer 或按注册名分支 |
| `DataBuilder` | 构造完整 train/validation/test iterable 和 provenance | 训练期临时下载、通用 provider dispatch |
| data artifact helpers | verified bytes、安全解包、manifest、原子缓存 | 理解 AFHQ class/domain 语义 |
| AFHQ prepare recipe | source 选择、split、Lanczos resize、AFHQ 校验 | 成为新的 core dataset registry |
| `ADMUNet` | time/class-conditioned denoising prediction | class dropout、CFG、sampling loop |
| `SamplingBuilder` | label allocation、CFG adapter、family compatibility | 数值求解 |
| Gaussian `Sampler` | 完整数值采样算法 | class label、CFG、具体 model signature |
| `Diagnostic` | 只读采样、重建、metric/artifact | 恢复状态、修改 optimizer/model |

不新增通用 `Condition` 基类、condition registry、顶层 `conditioning:` schema，
也不向 `Process`、`Sampler` 或 `GenerativeDynamics` root 增加条件方法。

## 5. 公共配置

### 5.1 Trainer

在 `TrainerConfig` 增加：

```yaml
trainer:
  num_epochs: 200
  device: auto
  precision: bf16-mixed
  accumulate_grad_batches: 4
  max_grad_norm: 1.0
  show_progress: true
```

公共语义：

- `precision`: `fp32 | bf16-mixed | fp16-mixed`，默认 `fp32`；
- `accumulate_grad_batches`: 正整数且拒绝 `bool`，默认 `1`；
- 不提供 `auto`，防止换设备或 resume 后训练拓扑悄然变化；
- 框架不主动降低 model 参数或 optimizer state 精度，不调用 `.half()` 或
  `.bfloat16()`；内置标准 AdamW 在 FP32 参数下保持 FP32 state，第三方
  optimizer 自己声明并保持其公共 dtype contract；
- `precision` 与 `accumulate_grad_batches` 是训练状态边界，strict resume
  从 checkpoint config 恢复，不能通过 observability overlay 改写。

设备支持矩阵：

| precision | CPU | CUDA | MPS |
| --- | --- | --- | --- |
| `fp32` | 支持 | 支持 | 支持 |
| `bf16-mixed` | 支持 autocast | 需 `torch.cuda.is_bf16_supported()` | 首版拒绝 |
| `fp16-mixed` | 首版拒绝 | autocast + GradScaler | 首版拒绝 |

硬件不满足时在创建 run 目录前失败，不静默降级为 FP32。AFHQ 生产配置默认
`bf16-mixed`；如 GPU 不支持 BF16，用户应明确选择 `fp16-mixed`。

### 5.2 AFHQ 候选生产配置

以下是当前根据完整 HF candidate 元数据计算的生产候选。Phase B 必须先完成官方
archive source lock audit，测试确认 canonical inventory 与精确 split，再冻结和合并
production YAML；不能因为候选数值已经写在计划中就跳过该 gate。

最终 production YAML 应固化以下关键值：

```yaml
data:
  name: class_image_folder
  params:
    root: ./data/prepared/afhq-v2/128/<preparation-key>
    manifest: ./data/prepared/afhq-v2/128/<preparation-key>/dataset_manifest.yaml
    image:
      size: [128, 128]
      channels: 3
      require_exact_size: true
      normalize: true
      random_horizontal_flip: true
    loader:
      batch_size: 8
      num_workers: 2
      shuffle: true
      drop_last: true
      pin_memory: true
      persistent_workers: true
      prefetch_factor: 4
      steps_per_epoch: auto

model:
  name: adm_unet
  params:
    in_channels: 3
    out_channels: 3
    base_channels: 128
    channel_multipliers: [1, 2, 3, 4]
    num_res_blocks: 2
    transformer_depths: [0, 0, 1, 2]
    middle_transformer_depth: 1
    attention_head_dim: 64
    time_embedding_dim: 512
    num_classes: 3
    dropout: 0.1
    scale_shift_norm: true
    residual_resampling: true
    zero_init_residual: true
    zero_init_output: true

process:
  name: discrete_gaussian
  params:
    schedule:
      name: cosine_alpha_bar
      params:
        num_timesteps: 1000
        s: 0.008
        max_beta: 0.999

training:
  name: class_conditional_gaussian_denoising
  params:
    prediction_type: v
    condition_dropout: 0.1

objective:
  name: mse
  params: {reduction: mean}

optimizer:
  name: torch.optim.AdamW
  params:
    lr: 1.0e-4
    weight_decay: 0.0
    betas: [0.9, 0.999]
    eps: 1.0e-8

lr_scheduler:
  name: warmup_cosine
  interval: step
  params:
    warmup_steps: 1680
    total_steps: 84000
    min_lr_ratio: 0.05

ema:
  enabled: true
  decay: 0.9999
  update_after_step: 500
  update_every: 1
  use_for_sampling: true

trainer:
  num_epochs: 200
  device: auto
  precision: bf16-mixed
  accumulate_grad_batches: 4
  max_grad_norm: 1.0
  show_progress: true
  early_stopping: {enabled: false}

logging:
  log_every: 100
  backends:
    - {name: local, params: {console: false, append: false}}
    - {name: tensorboard, params: {}}
  torch_logs: {}

artifacts:
  checkpoint_every: 5
```

计划固定从 official train 中按 class 各取 300 张作为 validation，并原样保留
official test。2026-07-26 查询到的
[pinned-candidate HF 镜像元数据](https://datasets-server.huggingface.co/info?dataset=ryushinn%2FAFHQv2)
是
train/test `14,336/1,467`，合计 15,803；AFHQ-v2 清理过原始 split，因此不能继续
假设 test 恰好是旧版的每类 500 张。正式 source lock 必须用官方 archive 再次验证
这些精确数量。验证通过后，prepared train/validation/test 为
13,436/900/1,467。在 `drop_last: true` 下：
在 `drop_last: true` 下：

```text
microbatches_per_epoch = floor(13,436 / 8) = 1,679
optimizer_updates_per_epoch = ceil(1,679 / 4) = 420
total_steps = 200 × 420 = 84,000
warmup_steps = round(0.02 × 84,000) = 1,680
```

这些值在 source lock 审计后由测试重新计算并与 YAML 字面值比对。核心 scheduler
factory 不根据构造参数名猜测或重写 `total_steps`。若审计后的官方计数不同，先更新
lock、公式测试和 YAML，再发布配置。

显存不足时使用 `batch_size: 4` 与 `accumulate_grad_batches: 8`，仍保持 effective
batch 32；在相同 prepared train count 下 optimizer update 数和 scheduler 配置不变。
Gradient Accumulation 用于保持 effective batch，不是提高单次 GPU kernel 占用率；
GPU 有余量时优先提高 micro batch。

### 5.3 采样展示

默认展示使用 DDIM-50、固定 seed、每类 12 张和 CFG 2.0：

```yaml
sampling:
  shape: [3, 128, 128]
  num_samples: 36
  batch_size: 12
  seed: 20260726
  builder:
    name: class_conditional_denoising
    params:
      weights: auto
      prediction_type: v
      clip_denoised: true
      guidance_scale: 2.0
      conditions:
        - {class_label: 0, count: 12}
        - {class_label: 1, count: 12}
        - {class_label: 2, count: 12}
      sampler:
        name: ddim
        params: {num_inference_steps: 50, eta: 0.0}
      trajectory: {enabled: true, every_steps: 5}
```

训练完成后另跑 `1.0/1.5/2.0/3.0` guidance sweep。生产默认先选 2.0，
最终以固定 seed grid、diversity 和 KID/FID 共同决定，不凭单张最好看的图选择。

## 6. Phase A：Trainer AMP 与 Gradient Accumulation

### 6.1 PrecisionRuntime

新增 `stochaflow.training.precision.PrecisionRuntime`，只负责：

```python
device_type: str
autocast_dtype: torch.dtype | None
grad_scaler: torch.amp.GradScaler | None

autocast() -> ContextManager[None]
backward(loss: torch.Tensor) -> None
unscale_(optimizer: Optimizer) -> None
step(optimizer: Optimizer) -> bool
```

factory 在 device 解析后构造一次，并把同一个实例注入 Trainer；
CheckpointManager 接收明确的 `precision_kind`，并在 FP16 时持有同一个
`grad_scaler` 引用。Runtime 自身不定义另一份 state dict；唯一有状态资产是
FP16 GradScaler。
Trainer 初始化时验证对象身份，避免训练与 checkpoint 操作不同 scaler。

不为 precision 建 registry。它是自动训练循环的 lifecycle policy，不是可插拔算法族。

### 6.2 累积窗口算法

`train_epoch()` 在 iterator 层按最多
`K = accumulate_grad_batches` 个 micro-batches 分窗。每个窗口先得到实际长度
`r`，只把当前 micro-batch 搬到 GPU，不同时保留多个 GPU batch：

```text
optimizer.zero_grad(set_to_none=True)  # epoch/direct-call 初始边界

for window in windows(iterator, K):
    r = len(window)

    for microbatch in window:
        prepared = move_to_device(microbatch)
        with precision.autocast():
            output = strategy.training_step(prepared)
            normalized_loss = output.loss / r
        precision.backward(normalized_loss)  # autocast 外

    if clipping:
        precision.unscale_(optimizer)        # fp16 每窗口最多一次
        clip_grad_norm_(...)

    succeeded = precision.step(optimizer)    # scaler step/update 或普通 step
    optimizer.zero_grad(set_to_none=True)    # 成功/overflow 后的 window boundary

    if succeeded:
        update EMA
        step step-scheduler
        increment global_step
        emit step diagnostic/logging
```

最后不足 `K` 的窗口按实际 `r` 归一化，不能除以固定 `K`。框架只保证 Strategy
返回的 scalar losses 按 micro-batch 等权平均，避免 partial window 被错误缩小为
`r/K`。只有在 loss 是 per-sample mean、micro-batches 等长、模型没有 BatchNorm
等跨样本状态且计算无随机差异时，它才与一次物理大 batch update 等价。核心不猜测
任意 batch 中的 sample 数；不同大小 micro-batch 仍定义为“micro-batch mean 的
平均”。AFHQ production 使用 GroupNorm 和 `drop_last: true`，避免这部分歧义。

同时更新 `TrainStepOutput.loss` 文档契约：它是 Strategy 产生的 scalar optimization
loss；自动 accumulation 对这些 scalar 等权平均。希望获得通常所说的 effective
batch 语义时，Strategy 必须返回 logical samples 上 mean-reduced loss。返回 sum 或
自定义缩放仍有定义，但不保证等价于物理大 batch。内置 AFHQ Strategy 与
`objective.reduction: mean` 满足该前提。

window 观测聚合规则：

- `window_loss` 是未归一化 micro-batch losses 的等权平均；
- epoch loss 是所有 consumed micro-batch losses 的等权平均；
- `TrainStepOutput.metrics` 中每个 detached scalar 按该 key 在 window 内的出现次数
  平均，step log 使用 window mean；
- diagnostic event 仍携带最后一个 micro-batch 的 batch/output，不能把其 task
  tensors 冒充为 window aggregate；
- progress reporter 的 loss 使用当前 micro-batch loss，optimizer-step log 使用
  `window_loss`。

新增私有 `_run_accumulation_window()` 与 `_finish_optimizer_step()`；
后者是 optimizer update、EMA、step scheduler 和 successful `global_step`
增量的唯一 owner。`train_epoch()` 不再按 micro-batch 调用 `train_batch()`，也不再
自行递增 step，而是直接调用 window primitive，再负责 epoch-owned
diagnostic/logging/progress。

`train_batch()` 保留现有方法形态并定义为“一次完整 optimizer update attempt”，
内部用 window size 1 调用同一 primitive：成功时递增 `global_step` 并推进
EMA/scheduler，FP16 overflow 时不递增。成功 direct call 从“未计入
`global_step`”改为“计入一次 successful update”，这是有意的行为收敛，必须写入
compatibility note。为保持当前直接调用边界，它不发
diagnostic、regular logger 或 progress event。针对这一行为增加显式回归测试，
避免未来内外两层双增 step。

zero-grad lifecycle 采用“进入首个 window 前清一次，之后每个成功/overflow boundary
清一次”；window 内绝不清 gradient。若 forward/backward 抛错，exception cleanup
也必须清除已累积 gradient 后再传播异常。测试按这一边界计数，不再笼统声称
“每个 window 只能调用一次”。

progress reporter 仍在每个 consumed micro-batch 后推进一次；例如 epoch total 为
1,679 时必须收到 1,679 次 batch progress，而不是 420 次。前 `K-1` 个
micro-batches 可显示尚未变化的 successful `global_step`，window boundary 再显示
新的 step。optimizer-step diagnostic 和 regular step logging 仍只在成功 boundary
触发。

若需要 clipping 或读取未缩放 grad norm，FP16 每个 window 在全部 backward 完成后
显式 `unscale_` 且只调用一次；若既不 clip 也不检查 gradient，则不显式 unscale，
交给 `scaler.step()` 内部处理。不能为了逐 micro-batch 监控而提前 unscale。

### 6.3 FP16 overflow

`GradScaler.step()` 的 optimizer 返回值不能用来判断是否跳步。使用公开 scale 语义：

```python
old_scale = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
succeeded = scaler.get_scale() >= old_scale
```

scale 下降表示本次发现非有限 gradient 并跳过 update。无论成功或失败，进入下一窗口前
都清空 gradient。只有成功时才推进：

1. EMA；
2. step scheduler；
3. `global_step`；
4. step diagnostic；
5. 以 step 为 cadence 的普通 logging。

这种 success 判定只用于本计划的单 optimizer 和由 `PrecisionRuntime` 独占调用的
默认 `GradScaler.update()`；不开放手动 `new_scale` 或多 optimizer lifecycle。
测试覆盖 scale 保持、正常增长和 overflow 回退三条路径。

epoch scheduler 只在该 epoch 至少有一次成功 update 后推进一次。全 epoch overflow
时不推进 scheduler，记录 warning 和 epoch metric。

统一定义：

```text
global_step = 成功的 optimizer update 数
num_batches = 消费的 micro-batch 数
num_optimizer_steps = 成功的 optimizer update 数
num_skipped_optimizer_steps = FP16 overflow 跳过的窗口数
```

旧 checkpoint 在 accumulation=1 时原有 batch step 与 optimizer update 一一对应，
所以迁移后 `global_step` 数值含义不漂移。

### 6.4 Evaluation、diagnostic 与日志

- `evaluate_epoch()` 在 `torch.no_grad()` 内使用相同 autocast context，不使用 scaler；
- standalone sampling 和 diagnostic 内额外 sampling 的 precision 属于 sampling
  runtime，首版保持原有 FP32 行为，不能由 Trainer 隐式接管；
- `TrainBatchEndEvent` 现有文档语义已经是 successful optimizer step；累积后每个成功
  window 只发一次，携带该 window 最后一个 micro-batch/output；
- epoch loss 继续统计未除以 `r` 的原始 micro-batch loss 平均；
- 记录 `train/epoch_micro_batches`、`train/epoch_optimizer_steps`、
  `train/epoch_skipped_optimizer_steps`、`train/optimizer_steps_per_second`；
- FP16 额外记录当前 loss scale；失败窗口只计入 epoch skipped counter，避免在相同
  TensorBoard `global_step` 上覆盖成功 step；
- 增加 data-wait 与 compute 时间，使“GPU 72%”能区分 input stall 和 kernel
  不饱和；不新增 NVML 强依赖。

CUDA batch transfer 可在内部使用 `non_blocking=True`；未 pinned 的 tensor 会保留
正确性。AFHQ 配置配合 `pin_memory: true`、`persistent_workers: true`、
`prefetch_factor: 4` 和固定 `num_workers: 2`。

### 6.5 Checkpoint v9

AMP 是有状态训练资产，不可沿用 observability component 的无状态规则。
checkpoint 格式从 v8 升到 v9，并选定唯一 schema：

```text
precision_kind                     # v9 始终存在
grad_scaler_class                  # 仅 fp16-mixed
grad_scaler_state_dict             # 仅 fp16-mixed
data_provenance                    # v9 始终存在；null 或固定 schema mapping
inference_asset_descriptors        # v9 始终存在；空或固定 schema mapping
```

`fp32`/`bf16-mixed` 必须有 `precision_kind` 且不得出现两个 scaler 字段；
`fp16-mixed` 必须同时有三个字段。CheckpointManager 不序列化
`PrecisionRuntime` class identity 或第二份 `precision_state_dict`。
`data_provenance` 从 v9 首次发布即存在：普通/legacy DataBuilder 写 `null`，
提供 provenance 的 DataBuilder 写
`{schema_version, artifact_digest, manifest_sha256}`。checkpoint 不保存绝对 cache
path；run manifest 可记录本次复制到 run 内的 manifest 相对路径。Phase B 只能把
`null` 填成已定义 mapping，不能再次修改 v9 schema；若未来需要不兼容字段则升 v10。
`inference_asset_descriptors` 也从 v9 首次发布即存在：没有推理辅助资产时写 `{}`；
有辅助资产时，每个 entry 只投影 TrainingPlan 已声明的 training asset slot、
immutable declaration、capability role 与 persistence policy。它不保存第二份权重，
也不允许 sampling provider 重复发明 asset declaration。完整 typed contract 由
Latent Diffusion 计划定义；AFHQ 首版只需要固定空 mapping，从而避免之后为了 frozen
codec 再改 v9 schema。

加载顺序：

1. 校验 header/version/config；
2. 校验 model、Process、Objective、auxiliary、optimizer、scheduler、EMA 拓扑；
3. 校验 precision/scaler 拓扑和 class identity；
4. 用同 class 的临时 GradScaler 预加载 scaler state，验证内部字段可恢复；
5. 所有校验通过后才修改任何 managed runtime state；
6. 加载实际 scaler；
7. 最后恢复 RNG。

严格规则：

- runtime 有 scaler、checkpoint 无 scaler：拒绝；
- runtime 无 scaler、checkpoint 有 scaler：拒绝；
- scaler class 或 precision kind 不同：拒绝；
- checkpoint 只能在完成 `scaler.update()` 且没有 pending micro-batch gradient 时保存；
- epoch 末先 flush partial window，再 evaluate/save；
- 不序列化 pending gradients、micro-step counter 或 DataLoader cursor。

兼容策略：

- writer 只写 v9；
- v8 原始 payload/config 必须不含 `precision_kind`、`grad_scaler_class`、
  `grad_scaler_state_dict`、`data_provenance`、`inference_asset_descriptors`、
  `trainer.precision` 或 `trainer.accumulate_grad_batches`；夹带任一新字段即拒绝，
  不能采信或忽略；
- 通过上述 raw validation 的 v8 训练 checkpoint 才迁移为
  `fp32 + accumulate_grad_batches=1 + data_provenance=null +
  inference_asset_descriptors={}`；
- 不通过创建新 scaler 把 v8 伪装成严格 `fp16-mixed` resume；
- inference/sampling 的 shared header validator 同时接受 v8/v9，并执行同一迁移；
- 从 v8 恢复后产生的新 latest/best 使用 v9 和 effective default fields；
- sibling best 的验证仍先使用 source checkpoint 原始 config/provenance。

## 7. Phase B：数据 artifact、预处理与 provenance

### 7.1 内置可复用能力

核心只加入非注册、非 provider、data-semantic-free 的内部 utilities：

- verified download request：URL、expected bytes、SHA-256、cache key；
- `.partial` 下载和完成后的 digest 校验；
- safe ZIP extraction；
- 文件数量、总展开大小和单文件大小上限；
- staging directory + 原子 rename；
- 进程间 preparation lock 和幂等 cache hit；
- canonical dataset manifest 与 `files.sha256`；
- `DataProvenanceRef`：
  `manifest_path`、`manifest_sha256`、`artifact_digest`。

不增加 `DataSource` provider registry，也不新增
`stochaflow data prepare <arbitrary-provider>` 的通用 dispatch。
AFHQ source 语义留在 example recipe；第二个真实 recipe 出现前不把
Hugging Face adapter 提升为核心抽象。

### 7.2 安全边界

解包必须拒绝：

- `..` traversal、绝对路径、Windows drive/UNC path；
- 反斜杠伪装路径、NUL、空名；
- symlink、hardlink、device、FIFO 等非普通文件；
- 大小写冲突、重复 canonical path；
- 超出成员数、单文件大小、总展开大小或压缩比限制的 archive；
- truncated download、digest mismatch 和 source lock mismatch。

不得把 token、cookie、signed URL、绝对 cache root、hostname 或时间戳写入
artifact identity。

digest 层次固定如下，避免自引用：

```text
files.sha256
  = 只列 prepared image relative path + file SHA-256
  = 不列 files.sha256 自身，不列 dataset_manifest.yaml

inventory_digest
  = SHA-256(canonical serialization of sorted files.sha256 records)

source_provenance_digest
  = SHA-256(canonical tagged source provenance)

preparation_key
  = SHA-256(source_provenance_digest + transform_recipe_hash)
  = 用作本地 prepared cache path，避免不同来源互相覆盖 manifest

artifact_digest
  = SHA-256(transform recipe schema/version/hash + inventory_digest)
  = 有意排除 source locator/provenance

manifest_sha256
  = manifest 完成写入后的独立 SHA-256
  = manifest 内不嵌自己的 SHA
```

这样 official 与经过审计的 HF mirror 在产生完全相同 prepared contents 时共享
`artifact_digest`，但保留不同的 source provenance 和 `manifest_sha256`。
strict resume 比较内容身份 `artifact_digest`；manifest SHA 用于审计本次来源记录，
不能制造 recursive hash。

### 7.3 AFHQ-v2 prepare recipe

新增 example command：

```powershell
uv run python examples/showcases/afhq-v2/prepare.py `
  --source official `
  --cache-root .\data `
  --resolution 128
```

可选 HF：

```powershell
uv run --extra data-hub python examples/showcases/afhq-v2/prepare.py `
  --source hf `
  --revision <full-commit-sha> `
  --cache-root .\data `
  --resolution 128
```

HF 适配器通过新的可选 `data-hub` extra 懒加载 `datasets`、
`huggingface_hub` 和直接使用的 `pyarrow`；它们不进入基础训练依赖。
`pyproject.toml` 声明受支持范围，`uv.lock` 固定实际版本。缺少 extra 时在网络请求
或 run 目录创建前给出安装命令，不能退回另一个 source。

规则：

- `official` 是默认且使用仓库审计过的 archive digest；
- `hf` 必须使用 lock 中的 repo id 与完整 commit SHA；
- AFHQ recipe 私有 manifest 使用 source tagged union，而不是强迫所有来源都有
  archive 字段：
  - `official_archive`：URL、archive SHA-256、bytes；
  - `hf_snapshot`：repo id、full revision、config/split、parquet blob inventory
    digest；
- canonical sample identity 优先使用
  `(official split, class, normalized official relative path)`；HF
  `image_relpath` 必须经审计映射到这一 identity。若 provider path 不同，只能通过
  `(class, decoded RGB pixel digest)` 的一对一无冲突映射采用 official identity，
  不能自行重命名后声称等价；
- HF source inventory 记录 canonical sample identity、label 和解码后的 RGB
  pixel digest；
- 两种来源都必须还原成同一 canonical source inventory，并匹配官方
  15,803 总数、`14,336/1,467` split、三类名称和文件约束；
- 禁止 source 失败后的自动 fallback；
- 保留官方 test，每类从 official train 按稳定 relative-path hash 取 300 张
  validation；
- 输入必须为 512×512 RGB PNG；
- 一次性用 Lanczos 生成 128×128 RGB PNG，不 crop；
- 在线 train transform 只做 horizontal flip、tensor conversion 和 `[-1, 1]`
  normalization；
- validation/test 不做随机增强；
- class mapping 固定记录为 `cat: 0, dog: 1, wild: 2`。

“确定性预处理”固定为一个 versioned recipe，而不只写 `Lanczos`：

- 按 canonical POSIX relative path 的 Unicode NFC 序排序；
- `pyproject.toml` 把核心代码直接 import 的 Pillow 声明为直接依赖，
  `uv.lock` 固定实际版本；
- `Image.open -> verify -> reopen -> convert("RGB")`；
- 拒绝非 identity EXIF orientation，丢弃 EXIF/ICC/text 等 metadata；
- `PIL.Image.Resampling.LANCZOS`、精确目标尺寸、固定 resize 参数；
- 8-bit RGB PNG、`optimize=False`、固定 `compress_level`、空 `pnginfo`；
- decoder/Pillow version、颜色/EXIF策略、resize/save 参数、split algorithm 和
  path normalization version 全部进入 transform recipe version/hash。

更换 Pillow 或任一编码参数会产生新的 transform recipe hash，而不能继续写入旧
artifact。

artifact 布局：

```text
<cache-root>/
├── raw/afhq-v2/<source-artifact-sha256>/afhq_v2.zip
└── prepared/afhq-v2/128/<preparation-key>/
    ├── train/{cat,dog,wild}/
    ├── validation/{cat,dog,wild}/
    ├── test/{cat,dog,wild}/
    ├── files.sha256
    └── dataset_manifest.yaml
```

manifest 至少记录：

- schema/version、transform recipe id/version/hash 和 preparation key；
- source kind 与对应 tagged provenance；official 记录 archive digest/bytes，
  HF 记录 repo/revision/blob inventory；
- license、homepage、citation；
- input/output resolution、RGB、PNG、Lanczos；
- validation split algorithm/version/seed；
- 每个 split/class 的样本数；
- `files.sha256` 的 `inventory_digest`；
- prepared artifact digest。

### 7.4 内置 class image DataBuilder

新增 `class_image_folder` DataBuilder，消费已经 prepared 的 artifact：

- 在创建 DataLoader 前校验 manifest 和文件摘要；
- 严格要求 manifest class mapping 与目录一致；
- `require_exact_size: true` 时拒绝尺寸漂移，不在线 resize/crop；
- batch contract 为 `(images, {"class_label": labels})`；
- 返回 `DataLoaders(..., provenance=DataProvenanceRef(...))`；
- 不把 sample path、domain name 或任意 metadata 强塞给 core。

train loader 使用该 recipe 私有的 epoch-tagged sampler：

- `set_epoch(epoch)` 用 `(run seed, epoch)` 产生确定性 permutation；
- sampler 向 Dataset 传递 `(epoch, sample_index)`，而不是让 persistent worker
  保存可变 epoch/RNG state；
- horizontal flip 由
  `SHA-256(run seed, epoch, canonical sample identity, "hflip-v1")`
  的固定 bit 决定；
- worker 不调用全局 Python `random`，因此 Windows spawn、`num_workers: 2` 和
  `persistent_workers: true` 下，epoch-boundary resume 仍产生相同 sample order
  与 augmentation；
- validation/test 使用稳定顺序且无随机增强。

这个 sampler 是 `class_image_folder` 的私有实现，不创建通用 Sampler registry。
Trainer 继续只调用已有的 duck-typed `set_epoch()`，不理解 sample identity 或 flip。

runner 在创建 sibling run 前比较 source checkpoint 与当前
`DataProvenanceRef.artifact_digest`：

- checkpoint 有 provenance、当前缺失或 mismatch：拒绝；
- manifest 校验失败：拒绝；
- `artifact_digest` 是 strict training-data identity gate；若它相同但
  `manifest_sha256` 不同，可视为 content-equivalent source relocation，允许恢复，
  但新 run manifest 必须同时记录 source/current manifest SHA 和
  `source_provenance_changed: true`，新 checkpoint 保存当前 manifest SHA；
- provenance identity 不比较绝对 root，但 strict resume 仍使用 checkpoint-authoritative
  `data.params.root/manifest` 构造 loader；首版只保证 workspace relocation 后相同相对
  locator 仍可用。绝对 locator 改变时用户需在原 locator 提供 junction/symlink；
  不新增 data config overlay 或隐藏的 `--data-root`；
- legacy v8 无 provenance：允许一次 `legacy-unverified` 恢复，新 v9
  checkpoint 开始记录当前 provenance；
- observability overlay 不得覆盖 provenance。

## 8. Phase C：类条件 Gaussian 训练与采样

### 8.1 模型能力

定义窄协议，不增加新的 universal model base：

```python
@runtime_checkable
class ClassConditionalDenoiser(Protocol):
    @property
    def num_classes(self) -> int: ...

    @property
    def null_class_id(self) -> int: ...

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor: ...
```

`null_class_id == num_classes`，embedding size 为 `num_classes + 1`。
模型只完成条件预测，不拥有 class dropout 或 `forward_with_cfg()`。

### 8.2 TrainingBuilder/Strategy

新增注册项 `class_conditional_gaussian_denoising`：

- Builder 要求 primary model 满足 `ClassConditionalDenoiser`；
- Builder 要求 Process 满足离散 Gaussian denoising family capability；
- Builder 要求 Objective 存在；
- 所有兼容性在 Builder boundary 验证，不在 runner 按模型名分支；
- Strategy 校验 labels 是一维整数 Tensor、batch 对齐且范围为
  `[0, num_classes)`；
- `training_step()` 按样本以 `condition_dropout` 替换为 null class；
- `evaluation_step()` 永不 dropout；
- dropout 使用普通 PyTorch RNG，使已有 strict RNG resume 自然覆盖；
- Strategy 支持现有 Gaussian epsilon/x0/v/score target 归一化路径，
  production 选 `v`。

必须用一个独立 toy conditional model 测试 capability，不能只用 `ADMUNet`
自证兼容。

### 8.3 SamplingBuilder 与 CFG

新增 `class_conditional_denoising` SamplingBuilder。其私有配置拥有：

- condition label/count allocation；
- guidance scale；
- conditional model adapter；
- weights 选择；
- Gaussian Process/Dynamics/Sampler compatibility。

预测公式：

```text
prediction = uncond + guidance_scale × (cond - uncond)
```

优化规则：

- scale 0：只跑 null branch；
- scale 1：只跑 conditional branch；
- 其他有限非负 scale：cond/uncond 沿 batch 拼接后一次双 batch forward；
- condition counts 总和必须等于 `sampling.num_samples`；
- 跨 sampling batches 保持 label 顺序；
- raw/EMA weights resolve 后再次验证实际被调用的 module 满足
  `ClassConditionalDenoiser`，不能只验证 primary raw model；
- 将闭包交给现有 `GaussianModelDynamics`，DDPM/DDIM 不感知 class 或 CFG；
- manifest 记录展开后的 label counts、scale、weights、forward call count 和
  conditional/unconditional branch evaluation count。

无条件 `standard_denoising` 保持原实现。可把两者共用的 observer/runtime
抽到私有 `_denoising_runtime.py`，但不向无条件 Builder 增加 condition mode。

### 8.4 Conditional diagnostics

现有 `GaussianDiagnosticSemantics` 是无条件二参 predictor，不得伪造默认 label。
新增：

- `ClassConditionalGaussianDiagnosticSemantics`；
- `class_conditional_diffusion_quality`；
- 复用的私有 `gaussian_quality_engine.py`，只抽 cadence/provider/artifact
  orchestration。

规则：

- reconstruction 使用当前真实 batch 的原始 class labels；
- 固定 noise、固定 label allocation 和固定 guidance；
- sample grid 按 class balanced；
- step providers 通过 Strategy capability 调用模型，不猜测 signature；
- 首版 periodic quality 以 balanced overall KID 为主；
- classwise FID 和正式 test protocol 进入 Evaluation 计划。

## 9. Phase D：ADM-UNet

新增模型而不是就地扩展旧 `unet`，确保旧 checkpoint state topology 不变。

关键结构：

- sinusoidal timestep embedding + MLP；
- class embedding，包括独立 null class；
- ADM residual block，GroupNorm + SiLU；
- time/class embedding 通过 scale-shift normalization 注入；
- residual up/downsample；
- residual output projection zero-init；
- final output convolution zero-init；
- 四个 resolution stages 依次为 128/64/32/16；
- `transformer_depths[i]` 表示该 stage 在 down path 最后一个 ResBlock 后插入的
  Transformer depth，并在 mirrored up path 最后一个 ResBlock 后用独立参数再插入
  同样 depth；不是每个 ResBlock 后都重复插入；
- `middle_transformer_depth: 1` 表示 16×16 middle 的
  `ResBlock -> Transformer(depth=1) -> ResBlock`；
- 因此候选 `[0, 0, 1, 2]` 精确表示：32×32 down/up 各一层，
  16×16 down/up 各两层，另有一层 16×16 middle Transformer；
- pre-norm self-attention + MLP residual；
- attention 使用 PyTorch SDPA；
- attention head dimension 固定校验；
- 任意输入空间尺寸、channel multiplier 和 downsample 层数在构造期校验。

参数量先以约 90–120M 为设计区间，不在 prototype 前承诺精确值。冻结 production
配置前先用 meta/tiny prototype 得到 exact parameter count，再在目标 CUDA GPU
实测 BF16 forward/backward peak VRAM 和 SDPA kernel；测试固定窄参数量区间，防止
插入规则误解或配置误改导致容量突然翻倍。`8×4` 只有通过该 gate 后才成为主配置，
否则使用已经定义的 `4×8` fallback。

模型不得：

- 创建 Process、Objective、Sampler；
- 执行 class dropout 或 CFG；
- 根据 config name 切换训练算法；
- 把 activation checkpointing 与 gradient accumulation 混为同一能力。

若 24GB GPU 的 micro batch 8 仍有显存压力，先采用 4×8 fallback。只有 benchmark
证明仍无法达到目标后，才为 Transformer block 增加 model-owned activation
checkpointing 开关；该开关不进入 Trainer。

## 10. Phase E：Showcase、性能与质量验收

### 10.1 Example 布局

```text
examples/showcases/afhq-v2/
├── README.md
├── prepare.py
├── afhq-v2.lock.yaml
├── configs/
│   ├── smoke-128.yaml
│   └── train-128.yaml
└── tests/
    └── test_prepare.py
```

这是 built-in capabilities 的 showcase，不要求用户实现插件。AFHQ-specific source
逻辑留在 example；稳定的训练配置在 example 内，避免 root `configs/` 在没有准备数据
时看似可直接运行。功能稳定后将用户工作流移入 `docs/tutorials/afhq-v2.md`，
开发计划不加入公开 Sphinx index。

### 10.2 训练前容量预检

增加 example capacity command，对 micro batch `4/6/8` 分别运行 warmup 和至少
25 个 optimizer updates，报告：

- images/second 和 optimizer updates/second；
- peak allocated/reserved VRAM；
- data-wait/compute ratio；
- forward/backward/optimizer step 时间；
- FP32 与 BF16 mixed 的吞吐、显存差异；
- 是否出现 non-finite loss/gradient。

选择规则：

1. peak reserved VRAM 保留约 10% Windows/display headroom；
2. 在安全候选中选择 images/second 最高的 micro batch；
3. 调整 accumulation 保持 effective batch 32；
4. `num_workers` 固定为用户要求的 2，仅比较 pin/prefetch 和 prepared data；
5. 若 GPU utilization 仍约 72%，先用 data-wait 与 kernel timeline 定位原因，
   不盲目增加 accumulation。

性能数据写入 `benchmarks/results/` 的机器标识结果，不把它当跨平台 CI
硬阈值。建议目标是在目标 CUDA GPU 上 BF16 mixed 相对同 micro batch FP32
吞吐提升至少 25%，但真实交付以测量报告为准。

### 10.3 训练监控

TensorBoard 至少包含：

- raw train loss、EMA-smoothed loss；
- learning rate；
- epoch micro-batches、successful/skipped optimizer steps；
- images/s、optimizer steps/s、data-wait/compute；
- grad norm；
- FP16 loss scale（仅 FP16）；
- timestep bucket loss；
- noise alignment；
- x0 reconstruction panel；
- 每 5 epochs 的 balanced DDIM-50 class grid；
- 固定 seed trajectory；
- 每 20 epochs 的 optional balanced KID。

生产训练不启用 early stopping。epoch 200 前出现明显崩溃、持续 non-finite、
class collapse 或 loss/quality 同时长期退化时人工停止；不能仅凭训练 MSE 的微小波动
选择 checkpoint。

### 10.4 质量 gate

阶段性保留 epoch 1/10/25/50/100/150/200 的固定 seed grid。最终候选至少满足：

- 三个 class 都能稳定生成可辨认的动物脸；
- 眼睛、鼻口、耳朵和整体脸部布局无系统性破坏；
- 同 class 内有颜色、品种、视角和背景多样性；
- 没有明显训练样本复制；
- CFG 1.0/1.5/2.0/3.0 sweep 中有可解释的 fidelity/diversity 变化；
- EMA 权重优于或不劣于 raw 权重；
- final test KID/FID 由独立 Evaluation operation 计算并冻结 protocol identity。

不得为了 README 挑图而改变 seed 后不记录。最终 sample grid、trajectory GIF、
checkpoint identity、config、guidance、seed 和 evaluation result 必须能关联回同一
run manifest。

## 11. Phase F：原生 Pixel DiT 对照

AFHQ showcase 稳定后实现可选 built-in `dit`：

```yaml
model:
  name: dit
  params:
    input_size: [128, 128]
    patch_size: 8
    in_channels: 3
    out_channels: 3
    hidden_size: 384
    depth: 12
    num_heads: 6
    mlp_ratio: 4.0
    num_classes: 3
```

要求：

- adaLN-Zero；
- fixed 2D sin/cos position embedding；
- zero-init modulation/final projection；
- PyTorch SDPA；
- fixed-variance output，`out_channels == in_channels`；
- 实现相同 `ClassConditionalDenoiser`；
- class dropout/CFG 仍由 Strategy/Builder 拥有；
- 与 `adm_unet` 使用同一数据、Process、Objective、effective batch 和 evaluation
  protocol。

DiT-S/8 约 33M 参数、256 tokens，适合作为单卡 architecture comparison。
首版不做 patch 4 或 DiT-B。若它在相同 24–48 小时预算下明显欠拟合，不通过偷偷延长
训练或更换数据 protocol 美化对比，而是在 benchmark 中如实记录。

## 12. 文件级实施清单

建议修改/新增：

```text
src/stochaflow/utils/config.py
src/stochaflow/utils/factory.py
src/stochaflow/utils/checkpoint.py
src/stochaflow/utils/run_manifest.py
src/stochaflow/scripts/experiment_runner.py
src/stochaflow/sampling/runtime.py

src/stochaflow/training/precision.py
src/stochaflow/training/trainer.py
src/stochaflow/training/class_conditional_gaussian.py
src/stochaflow/training/diagnostics/contracts.py
src/stochaflow/training/diagnostics/class_conditional_diffusion_quality.py
src/stochaflow/training/diagnostics/gaussian_quality_engine.py

src/stochaflow/data/artifacts.py
src/stochaflow/data/provenance.py
src/stochaflow/data/class_image.py
src/stochaflow/data/builder.py

src/stochaflow/models/conditioning.py
src/stochaflow/models/adm_blocks.py
src/stochaflow/models/adm_unet.py

src/stochaflow/sampling/class_conditional.py
src/stochaflow/sampling/_denoising_runtime.py

examples/showcases/afhq-v2/**
docs/tutorials/afhq-v2.md
docs/configuration/reference.md
docs/configuration/compatibility-and-migration.md
pyproject.toml
uv.lock
```

Phase F 另增：

```text
src/stochaflow/models/dit.py
src/stochaflow/models/transformer_blocks.py
tests/test_dit.py
```

实现时先确认工作树中这些文件的已有修改，逐文件保留用户当前变更，不批量覆盖。

## 13. 测试矩阵

### 13.1 Config 与 precision

- 新字段默认值保持现有配置；
- precision 枚举、空值、unknown、错误类型；
- accumulation 为 0、负数、float、string、`bool`；
- CPU/CUDA/MPS precision capability matrix；
- BF16 CUDA capability 拒绝路径；
- model/optimizer state 仍为 FP32；
- evaluation autocast 且无 scaler。

### 13.2 Accumulation 生命周期

- `K=1` 保持现有 optimizer/scheduler/EMA 行为；
- `K=2`、5 个 micro-batches 得到 3 次 successful updates；
- 使用 deterministic linear model、per-sample mean、等长 batch 且无跨样本状态，
  验证 full/partial window 与对应物理大 batch 的参数更新一致；
- 使用不同 micro-batch 大小时只验证“micro-batch scalar 等权平均”契约，
  不声称 sample-weighted 等价；
- loss 除以实际 `r`；
- 首个 window 前一次 zero-grad，之后每个 successful/overflow boundary 一次，
  window 内没有 zero-grad；
- exception cleanup 清除 partial gradients；
- 需要 inspect/clip 时每窗口只 unscale 一次、clip 一次、step/update 一次；
- 严格验证 `unscale -> clip -> step -> update` 顺序；
- overflow 不推进 global step、scheduler、EMA、diagnostic；
- overflow 后 gradient 被清空，下个窗口可恢复；
- epoch scheduler 在有成功 update 时一次、全 overflow 时零次；
- `train_batch()` 与 window size 1 共用同一 global-step 终点，成功/overflow
  均不发 direct diagnostic/logger/progress event；
- max-batches 截断形成 partial window 时正确 flush；
- reporter 的 batches 与 optimizer steps 不混淆。
- window/epoch loss、per-key scalar metric 和 final-output diagnostic 按文档语义聚合。

### 13.3 Checkpoint/resume

- FP16 scaler strict round-trip；
- BF16/FP32 不保存 scaler；
- v9 `data_provenance` 始终存在且只能为 `null` 或固定 schema mapping；
- v9 `inference_asset_descriptors` 始终存在；AFHQ 是 `{}`，非空 entry 必须通过
  shared typed descriptor validation；
- missing/unexpected/mismatched scaler 在修改任何资产前失败；
- malformed scaler state 先在临时 scaler 上失败，不留下部分恢复的 managed state；
- v8 -> v9 的 FP32/accumulation=1 migration；
- v8 header 夹带任意 precision/accumulation/scaler 新字段时拒绝；
- v8 不可 strict-resume 为 FP16；
- epoch-boundary uninterrupted 与 resume 对比 model、optimizer、scheduler、
  EMA、scaler、global step、RNG；
- 强制最后一个 accumulation window overflow，随后 epoch checkpoint/resume；
  下一次成功 update 与 uninterrupted 路径的 model、optimizer、scheduler、EMA、
  scaler、global step 和 RNG 一致；
- observability overlay 不可改变 precision/accumulation/provenance；
- sampling runtime 共享 v8/v9 header migration；
- materialized best/latest 使用 effective v9 config。

### 13.4 Data

- archive traversal、Windows path、symlink、case collision、duplicate、bomb；
- interrupted download、checksum mismatch、cache hit、concurrent prepare；
- official/HF inventory canonical equality；
- official archive 与 HF snapshot tagged provenance 的 required/unknown fields；
- 不同 source、相同 prepared content：不同 preparation key/manifest SHA，
  相同 artifact digest；
- artifact 相同但 manifest SHA 改变时允许并审计 source transition；artifact
  mismatch 在新 run 前拒绝；
- provider path 不同只能通过无冲突 pixel mapping 采用 canonical identity；
- source revision 必须完整固定；
- 15,803 总数、split/class counts、RGB/PNG/512 输入；
- Lanczos 128 输出、无 crop、deterministic bytes；
- stable validation split；
- `files.sha256`、inventory、artifact、manifest 四层 digest 无自引用；
- manifest canonical hash 不受时间、host 或 absolute root 影响；
- Pillow/decoder/EXIF/resize/PNG 参数变化会改变 transform recipe hash；
- class mapping 和 `(image, {"class_label": label})` batch；
- 保持相对 locator 的 workspace relocation 仍匹配 provenance；绝对 locator
  不可通过 resume overlay 改写；
- `num_workers=2`、persistent workers、random flip 下 uninterrupted/resume 的
  sample order、augmentation 和下一 optimizer update 一致；
- modified/missing file在创建 run 前失败；
- legacy checkpoint provenance 标记。

### 13.5 Model、训练与采样

- `ADMUNet` 小尺寸 shape/dtype/device；
- timestep/label batch mismatch、label dtype/range/null class；
- zero-init residual/output；
- class/null embedding gradient；
- state_dict round-trip；
- 独立自定义 `ClassConditionalDenoiser` 被 Builder 接受；
- 不满足协议的普通 module 在 Builder boundary 拒绝；
- class dropout 0、1 和 seeded 中间概率；
- evaluation 无 dropout；
- epsilon/x0/v/score targets；
- label dropout 在 checkpoint resume 后逐步一致；
- CFG scale 0/1/2 的解析 toy-model 精确公式；
- concatenate double-batch 与两次 forward 一致；
- condition count、范围、跨 batch 顺序；
- DDPM/DDIM 无需修改；
- raw/EMA、fixed seed、trajectory；
- reconstruction 保持真实 class label；
- tiny class DataBuilder + toy model 完成 train/checkpoint/resume；
- AFHQ YAML 在 CI 只做 schema、formula 和 reference 测试，不下载或全训。

## 14. 实施顺序与提交边界

1. **Trainer contracts**：config、PrecisionRuntime、accumulation、logging；
2. **Checkpoint v9**：scaler topology、v8 migration、sampling shared validator；
3. **Data foundation**：artifact safety、manifest、provenance、class folder builder；
4. **Conditional family**：model protocol、TrainingBuilder、SamplingBuilder、
   diagnostic capability；
5. **ADM-UNet**：新模型和独立 contract tests；
6. **AFHQ showcase**：source lock、prepare、smoke/production configs、教程；
7. **Capacity and quality run**：选择 8×4 或 4×8，完成 200 epochs 和评估；
8. **Pixel DiT comparison**：在首个 showcase 稳定后单独提交。

每一步保持一个可审查的逻辑提交。Checkpoint v9 不与模型大改放在同一提交，
AFHQ 大文件、prepared data、checkpoints 和普通 run artifacts 不提交。

## 15. 验收命令

增量阶段运行 focused tests、Ruff 和 Pyright。完整 feature branch 合并前运行：

```powershell
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

另执行：

```powershell
# 无网络 tiny smoke
uv run stochaflow train `
  --config examples/showcases/afhq-v2/configs/smoke-128.yaml `
  --epochs 1 `
  --limit-batches 8

# 数据准备幂等性
uv run python examples/showcases/afhq-v2/prepare.py `
  --source official `
  --cache-root .\data `
  --resolution 128

# 生产训练
uv run stochaflow train `
  --config examples/showcases/afhq-v2/configs/train-128.yaml `
  --epochs 200
```

production config、source lock 和 prepared manifest 中的 count/digest 必须在启动
长训练前完成一次人工复核。

## 16. 完成定义

本计划仅在以下条件全部满足后完成：

- 旧配置默认行为、旧 `unet` checkpoint 和 strict resume 有明确兼容结果；
- AMP/accumulation 的 step、overflow、scheduler、EMA、diagnostic 和 checkpoint
  语义通过测试；
- v9 state 可严格恢复，v8 migration 不伪造 AMP 状态；
- v9 从首次发布即预留 typed `inference_asset_descriptors`；AFHQ 写空 mapping，
  v8 migration 也确定性地产生空 mapping；
- 数据 artifact 可离线、幂等、安全地准备并由 digest 识别；
- 官方与 HF 来源都不能绕过 source lock、license 和 inventory 校验；
- class conditioning 没有泄漏到 Process/Sampler root 或无条件 Builder；
- `adm_unet` 通过 capability、shape、初始化和 state tests；
- AFHQ production config 固化 200 epochs、`num_workers: 2`、AMP、accumulation、
  optimizer、scheduler、EMA、CFG 和 diagnostic 参数；
- 在目标 Windows CUDA 环境完成容量报告和一次可恢复的 200-epoch run；
- 固定 seed 的 class-balanced samples、trajectory、quality result 和 run manifest
  可互相追溯；
- 稳定架构、配置和用户工作流迁移到正常文档树，本开发计划在合并前删除或归档，
  且不从公开文档 index 链接。
