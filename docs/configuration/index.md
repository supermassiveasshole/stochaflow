# Stochaflow 配置手册

本手册是 `stochaflow train`、`stochaflow sample`、`stochaflow evaluate`、内置组件和
自定义扩展的可查阅说明。training、sample 与 evaluation 使用各自独立的 strict schema；
`data.params` 的语义由所选 DataBuilder 拥有。

```{toctree}
:maxdepth: 2
:hidden:

reference
data-pipeline
extensions
compatibility-and-migration
reference-projects
sampling-capacity
workflows
troubleshooting
```

## 按任务查阅

| 任务 | 入口 |
| --- | --- |
| 先了解框架包含什么、各扩展角色如何协作 | [框架概览与当前能力](../framework.md) |
| 查某个 YAML 字段、默认值或 CLI 覆盖 | [完整字段参考](reference.md) |
| 配置图像、超分辨率或自定义数据构建 | [数据构建](data-pipeline.md) |
| 了解自行实现 conditional super-resolution 所需边界 | [条件 Gaussian 超分辨率教程](../tutorials/super-resolution.md) |
| 注册自定义 DataBuilder、writer 或其他组件 | [扩展与 Registry](extensions.md) |
| 理解 config/checkpoint 权威并跨环境移动实验 | [Checkpoint、配置权威与可移植性](compatibility-and-migration.md) |
| 查看 Physics reconstruction 与蒸馏的 legacy architecture fixtures | [纵向扩展参考项目](reference-projects.md) |
| 训练、smoke run、恢复、checkpoint 采样和独立评估 | [常用工作流](workflows.md) |
| 配置 learned-range variance、P2 training 或 respaced DDPM | {ref}`Gaussian variance、P2 与 respaced DDPM <gaussian-variance-p2-respaced-ddpm>` |
| 用 TensorBoard 查看 loss、学习率、样本网格并比较运行 | [TensorBoard 使用指南](../tutorials/tensorboard.md) |
| 估算大规模输出与 trajectory 内存 | [Sampling artifact 容量](sampling-capacity.md) |
| 根据错误信息定位问题 | [排错索引](troubleshooting.md) |

## 五分钟快速开始

### 使用发布包

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

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。`stochaflow init`
生成普通 Python distribution、一个最小 extension、测试和可运行配置；它不创建环境、
安装依赖或要求 `uv`。需要运行生成项目的测试时，可改用
`python -m pip install -e ".[test]"`。生成项目的 README 继续说明 resume 和
checkpoint sampling。

若要在 wheel 环境中快速试跑仓库维护的生成 example，而不 clone 源码，可下载
与当前 wheel 相同 tag 的独立 MNIST 配置：

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

Windows PowerShell 请使用 `curl.exe`。这个命令只做 workflow smoke test；完整
MNIST 结果和 DDPM/DDIM 采样协议见本页后文。Example 跑通后，可以回到
[Stochaflow GitHub 仓库](https://github.com/supermassiveasshole/stochaflow)
查看源码、提交 issue 或点 Star。

扩展 package 与 `stochaflow` CLI 必须安装在同一个 Python environment。若从 wheel
的本地副本安装，可将 URL 替换为下载路径：

```bash
python -m pip install ./dist/stochaflow-0.1.0-py3-none-any.whl
```

GitHub [v0.1.0 Release](https://github.com/supermassiveasshole/stochaflow/releases/tag/v0.1.0)
同时提供 source distribution、`SHA256SUMS` 和 build-provenance attestation。

可选依赖：

| 安装方式 | 能力 |
| --- | --- |
| `stochaflow` | 核心 runtime、phase metrics、TensorBoard logger 和 CLI |
| `stochaflow[wandb]` | W&B logger |
| `stochaflow[quality]` | KID/FID diagnostic 与 formal Evaluation providers |
| `stochaflow[docs]` | 本地构建文档与研究图表 |
| `stochaflow[dev]` | Pytest、Ruff 与 Pyright；面向源码贡献 |

从 GitHub wheel 安装 extra 时，使用 PEP 508 direct reference；例如：

```bash
python -m pip install \
  "stochaflow[quality] @ https://github.com/supermassiveasshole/stochaflow/releases/download/v0.1.0/stochaflow-0.1.0-py3-none-any.whl"
```

可将 `quality` 替换为上表中的其他 extra。源码贡献仍建议使用下一节的
`uv sync --extra dev`，以锁文件固定完整开发环境。

### 从源码贡献

源码 checkout 才包含仓库内的 `examples/`、tests 和文档：

```bash
uv sync --extra dev
uv run stochaflow train \
  --config examples/built-in/image-generation/configs/train/mnist.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

确认 smoke run 后，移除 CLI limit 运行 YAML 声明的完整实验：

```bash
uv run stochaflow train \
  --config examples/built-in/image-generation/configs/train/mnist.yaml
```

仓库只维护这一份 MNIST 训练配置。DDPM 与 DDIM-50 是复用同一 checkpoint 的采样
profile，而不是两份重复训练配置：

```bash
uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml

uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddim-50.yaml
```

仓库发布的 reference run 完成 200 epochs / 78,000 updates，在 epoch 183 以
validation v-prediction loss **0.07189** 选中 checkpoint；恢复该 checkpoint 后的
test loss 为 **0.07363**。DDPM-1000、DDIM-50 样本与完整轨迹见
[MNIST example result](https://github.com/supermassiveasshole/stochaflow/tree/main/examples/built-in/image-generation)。
上面的有界命令只验证工作流，不等价于这次收敛训练。

两个 sample profile 都是顶层 `sample:` 的完整独立调用配置，显式声明 sampler、
options、shape、数量、batch、seed 和 writers；checkpoint 只提供 state 与固化的
inference recipe。训练配置启用
逐 step warmup-cosine LR scheduler、EMA、TensorBoard 和固定 seed 的 DDIM-50
`diffusion_quality` 对照。diagnostics 属于训练配置，不会由 sample profile 改写。
顶层 `metrics` 则消费 TrainingStrategy 明确公开的 channel，在 train、validation 或
test phase 内聚合普通标量；每个 declaration、每个 phase 都使用独立 Metric 状态。
它不解释通用 batch schema，也不接管需要额外采样、参考数据或 artifact 的
`diagnostics`。`torchmetrics` 因此属于基础安装，`quality` extra 只补充 KID/FID
diagnostic/formal Evaluation 所需的 Inception dependencies。
该对照每 100 step 记录 timestep 分桶损失、噪声对齐统计，
并在 5 个固定噪声时刻记录 `x0` 重建 MSE/PSNR；每 10 轮用 EMA 生成固定的 32 张
样本网格。轨迹动画默认关闭，以减少 I/O 和视觉噪声。reference KID/FID 也默认关闭：
其 ImageNet 特征不适合直接衡量 MNIST 数字语义，因此不需要 `quality` extra。查看
训练与 diagnostic 指标：

当前 MNIST quality profile 使用 41.7M 参数的 attention UNet、cosine alpha-bar
噪声计划、v-prediction、batch 128 和 2 个 DataLoader worker。训练集每轮 390 个
optimizer step，因而 200 轮 scheduler 明确配置为 78,000 step，其中前 2,000 step
warmup；使用 `--epochs` 或 `--limit-batches` 做 smoke run 时，这条完整调度曲线只用于
组件验证，不代表缩短实验已经完成一次等比例 LR 周期。

```bash
tensorboard --logdir outputs/mnist/<run>/tensorboard
```

若已有 strict-resume checkpoint，只想启用同一套固定 DDIM-50、32 样本、seed 123
的 `x0` 重建监控和 local/TensorBoard 输出，可使用仓库提供的 observation-only 配置：

```bash
uv run stochaflow train \
  --resume outputs/mnist/<run>/checkpoints/latest.pt \
  --observability-config \
    examples/built-in/image-generation/configs/overlays/mnist-observability.yaml
```

该文件只允许 `diagnostics` 与 `logging`，不会放宽模型、数据、optimizer、scheduler、
EMA 或训练进度的严格恢复。详细替换、继承和审计语义见[恢复训练](workflows.md#恢复训练)。

从最佳 checkpoint 运行固化 inference recipe，并显式选择本次 DDPM profile：

```bash
uv run stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml
```

## 最小完整训练配置

下面只填写 schema 必填项以及内置 UNet/噪声 schedule 的必需参数；其余部分使用
[字段参考](reference.md)中的默认值。这个配置使用 MNIST 完整 train split，不创建
验证集，但会把原生 test split 构建为最终评估集。

```yaml
experiment:
  name: minimal_mnist

data:
  name: image
  params:
    source:
      name: torchvision
      params:
        dataset: MNIST
      materialization:
        cache_root: ./data
        policy: ensure
        verification: manifest
    partition: {mode: none}
    image: {size: [32, 32], channels: 1, normalize: true}

model:
  name: unet
  params:
    in_channels: 1
    out_channels: 1

process:
  name: discrete_gaussian
  params:
    schedule:
      name: linear_beta
      params:
        num_timesteps: 1000

training:
  name: gaussian_denoising
  params:
    prediction_type: epsilon

objective:
  name: mse
  params: {reduction: mean}
```

训练配置不含采样字段，也不会在训练结束时自动调用 `sample`。采样使用另一份完整配置：

```yaml
sample:
  sampler: {name: ddim, params: {num_inference_steps: 100, eta: 0.0}}
  options:
    weights: auto
    clip_denoised: true
    trajectory: {enabled: false, every_steps: 1}
  shape: [1, 32, 32]
  num_samples: 16
  batch_size: 16
  seed: 123
  writers:
    - {name: tensor, params: {}}
    - {name: image, params: {grid_nrow: 4}}
```

`load_config()` 和 `load_config_dict()` 只把 YAML/mapping 结构化为 dataclass 并校验
schema，不导入第三方代码。runner 随后发现并预检所选
`stochaflow.extensions` entry points，在任何插件导入前处理 checkpoint provenance 和
version policy，再激活插件、执行跨组件校验并构建组件。训练、resume 和
checkpoint-backed inference 因而使用同一套显式插件选择与审计结果。这里的
`prediction_type` 来自 TrainingBuilder 固化到 v12 checkpoint 的 recipe contract，
不会在 sample config 中重复声明。

(gaussian-variance-p2-respaced-ddpm)=
## Gaussian variance、P2 与 respaced DDPM

内置 Gaussian training 的兼容默认值是：

```yaml
training:
  name: gaussian_denoising
  params:
    prediction_type: epsilon
    variance: {mode: fixed}
```

`gaussian_denoising` 与 `class_conditional_gaussian_denoising` 是标准、未加权的
Gaussian TrainingBuilder。`prediction_type` 支持 `epsilon`、`x0`、`v` 和 `score`；
`variance` 是 Builder 的 private recipe fact，不是新的顶层 schema 或通用 Objective。
fixed variance 要求模型输出与 state 相同的 `C` channels，并且不计算
variational-bound term。

P2 不是标准 Builder 上的 weighting option，而是两个拥有完整训练语义的具体
TrainingBuilder：无条件使用 `p2_gaussian_denoising`，类条件使用
`class_conditional_p2_gaussian_denoising`。两者固定 epsilon prediction，不接受
`prediction_type`，并要求 `objective.name: mse`。

paper-compatible P2 + learned-range recipe 写作：

```yaml
model:
  name: adm_unet
  params:
    input_size: 256
    in_channels: 3
    out_channels: 6
    base_channels: 128
    channel_multipliers: [1, 1, 2, 2, 4, 4]
    num_res_blocks: 1
    attention_resolutions: [16]
    attention_head_channels: 64
    num_classes: null
    dropout: 0.1

training:
  name: p2_gaussian_denoising
  params:
    k: 1.0
    gamma: 1.0
    variance:
      mode: learned_range

objective:
  name: mse
  params: {reduction: mean}
```

模型的前 `C` channels 是 epsilon prediction，后 `C` channels 是 variance
interpolation values。共享的 process-free Gaussian family math 负责校验/拆分该 raw
output，并使用 Process 提供的 bounds 做插值；Training 与 Sampling 不各自复制这套
model-output 数学。hybrid loss 为 P2-weighted per-sample simple loss 加未被 P2
加权的 `0.001 ×` 完整 VLB；uniform single-timestep estimator 将它实现为
`T / 1000 ×` sampled VB term，而不是再额外乘一次 `0.001`。VB 的
mean/prediction branch 会 detach。
P2 权重精确为
`(k + alpha_bar_t / (1 - alpha_bar_t)) ** (-gamma)`，来自 cumulative marginal SNR，
不做 batch mean renormalization。`k` 必须 finite 且大于 0，`gamma` 必须 finite 且
非负；`gamma: 0` 逐元素退化为 constant weight，`k: 1, gamma: 1` 在 VP schedule
下逐元素等于 `1 - alpha_bar_t`。Process public state 使用 `1..T`，对应 model time
`0..T-1`；weight 与被采样 noisy state 使用同一个 cumulative marginal，不能错一位或
改用单步 alpha。

P2 Builder 固定 `prediction_type: epsilon`，并拒绝其他 prediction 参数。P2 与
learned-range 的内置实现都要求 `MSEObjective`；普通 fixed-variance Strategy 把 scalar
reduction 完整交给 Objective，而 P2/learned-range 的逐样本组合与 batch reduction 是
具体 Gaussian Strategy 对内置 MSE 的私有语义，不建立通用 Objective reducer 契约。
`MSEObjective` 的 `mean` 是 feature mean + batch mean，`sum` 是 feature sum + batch
sum；`gamma: 0` 直接使用标准 Strategy 路径，因此两种 reduction 下都严格等价于相应的
未加权 MSE。learned-range VB 仍按原语义相加且不被 P2 加权。

标准与 P2 Strategy 不把 SNR 或内部 timestep coefficient 发布为 diagnostic；可用时只
报告最终 `per_sample_loss`。`loss_aggregation_weight` 只控制 batch 的 epoch 统计聚合，
不参与 autograd，也不是 P2 coefficient。

类条件 P2 配置使用相同的 `k`、`gamma` 和 `variance`，并额外接受
`condition_dropout`：

```yaml
training:
  name: class_conditional_p2_gaussian_denoising
  params:
    condition_dropout: 0.1
    k: 1.0
    gamma: 1.0
    variance: {mode: fixed}
```

第三方 weighting 变体应实现并注册自己的 namespaced `TrainingBuilder` 和具体
`TrainingStrategy`，例如 `training.name: my_lab.min_snr_gaussian`。框架不提供
weighting policy registry、通用 Composer 或按 policy 名称分派的标准 Gaussian
Builder。任何开发期 `loss_weighting` 配置都不受支持，也没有 alias 或自动迁移。

在一份完整 `sample:` profile 中，ancestral 250-step sampling 的 Sampler 片段写作如下；
shape、数量、batch、seed、options 和 writers 仍须按完整 invocation contract 声明：

```yaml
sample:
  sampler:
    name: ddpm
    params: {num_inference_steps: 250}
```

这是 uniform-section selected-pair ancestral DDPM，不是 DDIM-250。`num_inference_steps`
与显式 `schedule` 互斥，使用任一 respaced 声明时不能再组合 `start_time/end_time`。
DDPM 从同一 selected-pair coefficient snapshot 构造 mean 和 variance；DDIM 保留自己的
generalized `eta` transition，并在 learned-range checkpoint 上明确忽略 variance half。

class-conditional learned-range CFG 只 guide prediction half。scale 0/1 返回完整
unconditional/conditional `2C` output；其他 scale 保留 conditional variance half。
`prediction_type` 与 `variance.mode` 写入 checkpoint inference recipe，独立 sample
config 不可覆盖。P2 Builder identity、`k/gamma` 与 `variance.mode` 是
training/resume facts，不应放进 sample profile。

## 配置层次

完整训练配置使用以下顶层段：

| 顶层段 | 作用 |
| --- | --- |
| `experiment` | 名称、seed、输出根目录和 run id |
| `extensions` | 本次运行激活的已安装 entry-point 插件；默认不激活第三方代码 |
| `data` | DataBuilder Registry 声明与 builder 专属参数 |
| `model` | 去噪模型 Registry 声明 |
| `training` | TrainingBuilder Registry 声明与任务组合参数 |
| `process` | 可选的 model-free probability process；`null` 表示所选算法不需要 |
| `objective` | 可选、可复用的标量训练目标；由 TrainingBuilder 决定是否需要 |
| `optimizer` / `lr_scheduler` | 原生 PyTorch target、构造参数与调度器推进周期 |
| `ema` | 模型参数指数移动平均 |
| `diagnostics` | 扩散特有的训练期诊断 |
| `trainer` | epoch、设备、梯度和提前停止 |
| `logging` | 日志频率与后端 |
| `artifacts` | checkpoint 周期 |

完整 sample invocation 使用另一份顶层 `sample` 配置，并可带独立的 `extensions` 插件追加
声明；它不是训练配置的一部分。

独立 evaluation config 又使用自己的 `version`、`name`、`purpose`、`subject`、`data`、
`evaluation`、`metrics` 与 `protocol` 顶层 schema；它显式引用 checkpoint 或 complete
prediction artifact，并由注册的 EvaluationBuilder 解释 live batch 或 offline record。
完整示例、CLI、prediction manifest、result/manifest 和当前 runtime contract 见
[独立 checkpoint Evaluation](workflows.md#独立-checkpoint-evaluation)。

## PyTorch optimizer 与 LR scheduler

标准 PyTorch 实现直接通过受限 target path 选择，不复制为 Stochaflow Registry alias：

```yaml
optimizer:
  name: torch.optim.AdamW
  params:
    lr: 0.0002
    weight_decay: 0.01

lr_scheduler:
  name: torch.optim.lr_scheduler.CosineAnnealingLR
  interval: epoch
  params:
    T_max: 100
```

核心分别注入模型的可训练 parameters 和已经构造的 optimizer；其余 `params` 原样作为
构造关键字参数传给当前安装的 PyTorch。省略的参数采用该 PyTorch 版本的默认值，完整
签名以 [PyTorch optimizer](https://docs.pytorch.org/docs/stable/optim.html) 和
[LR scheduler](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
文档为准。项目应锁定依赖版本，而不是依赖 Stochaflow 复制上游默认值。

`interval` 是 Stochaflow 的训练生命周期策略，不是 PyTorch 构造参数。当前自动训练循环
只支持能够无参数调用 `step()` 的 optimizer 与 scheduler；需要 closure 的 optimizer 或
validation metric 的 scheduler 暂不支持。`T_max`、`total_steps` 等具体参数必须写入确定
整数，不支持 `auto`，CLI 的 epoch 或 batch limit 覆盖也不会隐式重写这些参数。设
`lr_scheduler: null` 可禁用 scheduler。

## 术语表

source
: 内置图像 recipe 的原始数据入口，例如 torchvision 数据集或本地图像目录。

partition
: `image`、`super_resolution` 和 `multi_resolution_image` recipe 的私有划分功能。
  它不是所有 DataBuilder 都必须支持的通用策略。

bucket
: `multi_resolution_image` recipe 的命名尺寸；私有 sampler 保证一个 batch 内形状一致。

base bucket
: 只定义图像 recipe 动态 batch 的像素预算基准；采样输出由 `sample.shape` 独立声明。

Registry
: 名称到 Stochaflow 组件类/构造器的显式映射。`extensions.plugins` 选择已安装
  distribution 的 `stochaflow.extensions` entry point，聚合模块通过 decorator 注册通用
  组件；diagnostic 的 `params.modules` 是独立的局部 provider 机制。标准 PyTorch
  optimizer 和 LR scheduler 使用上面的受限原生 provider，不会复制到 Registry 中。

plugin selection
: 省略 `extensions` 或写 `plugins: []` 表示不加载第三方插件；`plugins: null` 是显式
  opt-in，加载当前环境发现的全部插件；非空列表按精确 entry-point name 选择。resolved
  config 总是保存排序后的确定列表，不保存 `null`。

## 内置示例

仓库维护四份职责明确的内置配置：

```text
examples/built-in/image-generation/configs/
├── train/mnist.yaml
├── sample/mnist-ddpm.yaml
├── sample/mnist-ddim-50.yaml
└── overlays/mnist-observability.yaml
```

其中只有 `train/mnist.yaml` 是完整训练配置；两个 `sample/` 文件是完整、独立的
checkpoint-bound sample invocation config，overlay 只用于 strict resume 的
diagnostics/logging。仓库不再维护
独立的 CIFAR-10、Flowers102 或 multi-source runnable YAML；这是示例收敛，不是对底层
数据来源或 recipe 能力的否定。
