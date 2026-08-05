# Stochaflow 文档

<div class="sf-hero">
  <p class="sf-eyebrow">Configuration-driven generative modeling</p>
  <p class="sf-lead">
    Stochaflow 把数据准备、组件组合、自动训练、严格恢复、checkpoint-backed
    inference、独立 evaluation、诊断与结果 artifact 串成一条可扩展研究工作流。
  </p>
  <ul class="sf-pills" aria-label="当前核心能力">
    <li class="sf-pill">Python 3.12+</li>
    <li class="sf-pill">PyTorch</li>
    <li class="sf-pill">DDPM / DDIM</li>
    <li class="sf-pill">可安装扩展</li>
    <li class="sf-pill">Checkpoint v12</li>
    <li class="sf-pill">Standalone Evaluation</li>
  </ul>
</div>

当前内置实现聚焦 pixel-space 离散 Gaussian diffusion，包括无条件与类别条件训练、
fixed/learned-range variance、full/respaced ancestral
DDPM、DDIM、EMA、CFG、结果 writers 与独立 checkpoint evaluation runtime。项目可以
通过普通 Python distribution 接入自己的数据、训练策略、生成算法与 artifact；
latent diffusion、pretrained
autoencoder 和 distributed training 尚未实现。当前 evaluation core 已支持显式 raw/EMA、
validation/test completeness、versioned prediction artifact、按 exact sample plan 的 offline
scoring 与 immutable result bundle。core FID/KID providers 与 maintained AFHQ-v2
source-checkout
full-official-test public profile 已闭合普通 pixel-space 图像生成。通用 runtime 不自动使
SR、consistency、latent/codec 或 distillation 成为受支持任务；未来任务必须同步交付自己的
monitoring、checkpoint inference 与 formal Evaluation。reference cache、comparison/gate 是
可选增强。

```{toctree}
:maxdepth: 2
:caption: 框架与使用
:hidden:

framework
platform-support
configuration/index
tutorials/tensorboard
tutorials/class-metrics
tutorials/afhq-v2
tutorials/super-resolution
tutorials/reuse-gaussian-components
tutorials/custom-generation-family
api/extensions
```

```{toctree}
:maxdepth: 2
:caption: 学习与研究
:hidden:

ddpm-notes
research-notes/part-1-distribution-transport
research-notes/part-2-ddpm
research-notes/part-3-ddim
```

## 五分钟快速开始

默认用户路径直接安装已发布的 wheel，不需要 clone Stochaflow 源码，也不需要
`uv sync`：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install \
  https://github.com/supermassiveasshole/stochaflow/releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl
stochaflow init my-research-project
cd my-research-project
python -m pip install -e .
stochaflow train --config experiments/example/train.yaml
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。这个命令序列会
生成并安装一个普通 Python extension project，运行其中的两轮微型示例；结果位于
`my-research-project/outputs/example/<run>/`。源码贡献、完整 MNIST workflow、
严格恢复与 checkpoint sampling 步骤见
[配置手册的五分钟快速开始](configuration/index.md#五分钟快速开始)。

如果想直接试跑内置的生成 example，也不需要 clone 仓库。下载与 wheel 同属
`v0.1.0` tag 的独立 MNIST 配置，并限制所有数据阶段：

```bash
curl --fail --location --output mnist.yaml \
  https://raw.githubusercontent.com/supermassiveasshole/stochaflow/v0.1.0/examples/built-in/image-generation/configs/train/mnist.yaml
stochaflow train \
  --config mnist.yaml \
  --epochs 1 \
  --limit-batches 10 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

Windows PowerShell 请将下载命令中的 `curl` 写成 `curl.exe`。这个有界运行只验证
workflow，不等价于下方展示的收敛训练。

<p class="sf-star-link">
  <a href="https://github.com/supermassiveasshole/stochaflow">⭐ Example 跑通了？回到 GitHub 查看源码并为 Stochaflow 点 Star</a>
</p>

## 框架如何分工

::::{container} sf-grid
:::{container} sf-card
<h3 class="sf-card-title">组合，而非硬编码</h3>

TrainingBuilder 与 SamplingBuilder 在边界上组装任务；core runtime 不按模型名或
算法名维护分支矩阵。
:::

:::{container} sf-card
<h3 class="sf-card-title">严格、可追溯的运行</h3>

resolved config、extension provenance、data identity、checkpoint 与 sampling
manifest 共同固定一次运行的来源。
:::

:::{container} sf-card
<h3 class="sf-card-title">普通 Python 扩展</h3>

数据、模型、Objective、训练或采样 family 可以由独立安装的 distribution 注册，
无需修改 core dispatch。
:::
::::

先阅读[框架概览与当前能力](framework.md)了解当前行为与工作流；长期产品范围、
非目标与新公共抽象的准入规则记录在
[项目规范](https://github.com/supermassiveasshole/stochaflow/blob/main/SPEC.md)，
架构所有权与依赖方向记录在
[架构说明](https://github.com/supermassiveasshole/stochaflow/blob/main/ARCHITECTURE.md)。

## 结果一览

MNIST 卡片来自仓库记录的固定协议，不把短 smoke run 当作质量 benchmark。AFHQ-v2
卡片描述 canonical ADM 拓扑、learned-range-v production recipe、完整 validation
Evaluation 和 formal test surface；schema-v3 RTX 4090 sweep 提供 operational capacity
evidence。训练期 FID/KID 只用于 validation 选模，不冒充最终 benchmark。

::::{container} sf-results
:::{container} sf-result-card
<h3 class="sf-card-title">MNIST · DDIM-50</h3>

<img src="_static/mnist_ddim50_epoch_0183_samples.png" width="206" height="206" loading="lazy" decoding="async" alt="使用 epoch 183 EMA checkpoint 和 DDIM-50 生成的 36 张 MNIST 样本">

同一份 EMA checkpoint 可由 DDPM-1000 或 deterministic DDIM-50 消费。选中
checkpoint 的 validation v-prediction loss 为 **0.07189**。

<p class="sf-result-meta">200 epochs · 78,000 optimizer updates · best epoch 183</p>

[查看 MNIST 命令、DDPM/DDIM 面板与轨迹](https://github.com/supermassiveasshole/stochaflow/tree/main/examples/built-in/image-generation)
:::

:::{container} sf-result-card
<h3 class="sf-card-title">AFHQ-v2 · class-conditional ADM</h3>

`adm_unet` 使用 canonical input/output block graph、逐 block skip ledger 和 QKV
residual attention。旧拓扑 checkpoint 不兼容；先前展示的指标与样本不作为 corrected
模型的证据。learned-range-v recipe 在配置 cadence 上运行 300/class validation
Evaluation，并按 aggregate FID 维护 `best.pt`。

<p class="sf-result-meta">live validation FID/KID · public formal test · measured RTX 4090 capacity</p>

[查看 AFHQ-v2 数据、配置与按类评估流程](tutorials/afhq-v2.md)
:::
::::

## 按目标继续

- [项目规范](https://github.com/supermassiveasshole/stochaflow/blob/main/SPEC.md)：
  长期产品范围、明确非目标和新公共抽象的准入门槛。
- [架构说明](https://github.com/supermassiveasshole/stochaflow/blob/main/ARCHITECTURE.md)：
  稳定所有权、依赖方向与组合边界。
- [框架概览与当前能力](framework.md)：当前内置能力、工作流和 extension 心智模型。
- [平台支持政策](platform-support.md)：Supported、Deprecated / best effort 等级和
  当前 CI 验证矩阵。
- [配置手册](configuration/index.md)：从最小 YAML 到多源数据、K-fold、自定义组件、
  训练恢复、checkpoint evaluation 与排错。
- [完整字段参考](configuration/reference.md)：由 dataclass、Registry 和 CLI 自动生成。
- [TensorBoard 使用指南](tutorials/tensorboard.md)：启用日志、比较多次运行、解读
  loss/LR/diagnostic 面板并排查 event 路径。
- [按类别验证与自定义 Metric](tutorials/class-metrics.md)：用自定义 Strategy channel
  聚合逐类别与 macro validation 指标。
- [AFHQ-v2 数据准备与训练](tutorials/afhq-v2.md)：安全下载、确定性 managed artifact、
  离线验证、strict resume 与 128×128 showcase。
- [纵向扩展参考项目](configuration/reference-projects.md)：Physics reconstruction 与
  frozen-teacher distillation 的 legacy install/architecture fixtures，不是 maintained task。
- [复用 Gaussian family 教程](tutorials/reuse-gaussian-components.md)与
  [自定义生成 family 教程](tutorials/custom-generation-family.md)：两条独立最小扩展路径。
- [条件 Gaussian 超分辨率](tutorials/super-resolution.md)：从内置 data-only SR recipe
  说明项目自行实现 condition-aware task 时需要遵守的 extension 边界。
- [Checkpoint、配置权威与可移植性](configuration/compatibility-and-migration.md)：
  checkpoint v12、canonical epoch metric mapping、fixed inference recipe、完整独立
  sample invocation 和跨环境恢复边界。
- [扩展公共 API](api/extensions.md)：第三方 extension 的稳定 Python import surface。
- [Sampling artifact 容量](configuration/sampling-capacity.md)：整体物化生命周期、
  内存估算、trajectory 限制和参考主机证据。
- [DDPM 学习笔记](ddpm-notes.md)：结合代码理解训练与采样实现。
