# Stochaflow 配置手册

本手册是 `stochaflow train`、`stochaflow sample`、内置组件和自定义扩展的
可查阅说明。顶层配置采用严格 schema；`data.params` 的语义由所选 DataBuilder
拥有。

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
| 先了解框架包含什么、各扩展角色如何协作 | [框架特性与架构](../framework.md) |
| 查某个 YAML 字段、默认值或 CLI 覆盖 | [完整字段参考](reference.md) |
| 配置图像、超分辨率或自定义数据构建 | [数据构建](data-pipeline.md) |
| 实现端到端 conditional diffusion 超分辨率 | [条件 Gaussian 超分辨率教程](../tutorials/super-resolution.md) |
| 注册自定义 DataBuilder、writer 或其他组件 | [扩展与 Registry](extensions.md) |
| 理解 config/checkpoint 权威并跨环境移动实验 | [Checkpoint、配置权威与可移植性](compatibility-and-migration.md) |
| 查看 Physics reconstruction 与蒸馏的完整扩展 | [纵向扩展参考项目](reference-projects.md) |
| 训练、smoke run、恢复和 checkpoint 采样 | [常用工作流](workflows.md) |
| 估算大规模输出与 trajectory 内存 | [Sampling artifact 容量](sampling-capacity.md) |
| 根据错误信息定位问题 | [排错索引](troubleshooting.md) |

## 五分钟快速开始

### 使用发布包

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install stochaflow
stochaflow init my-research-project
cd my-research-project
python -m pip install -e ".[test]"
stochaflow train --config experiments/example/train.yaml
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。`stochaflow init`
生成普通 Python distribution、一个最小 extension、测试和可运行配置；它不创建环境、
安装依赖或要求 `uv`。生成项目的 README 继续说明 resume 和 checkpoint sampling。

扩展 package 与 `stochaflow` CLI 必须安装在同一个 Python environment。若从 wheel
安装，可将第一条安装命令替换为：

```bash
python -m pip install ./dist/stochaflow-0.1.0-py3-none-any.whl
```

可选依赖：

| 安装方式 | 能力 |
| --- | --- |
| `stochaflow` | 核心 runtime、TensorBoard logger 和 CLI |
| `stochaflow[wandb]` | W&B logger |
| `stochaflow[quality]` | KID/FID diagnostic |
| `stochaflow[docs]` | 本地构建文档与研究图表 |
| `stochaflow[dev]` | Pytest、Ruff 与 Pyright；面向源码贡献 |

### 从源码贡献

源码 checkout 才包含仓库内的 `configs/`、tests 和文档：

```bash
uv sync --extra dev
uv run stochaflow train \
  --config configs/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --skip-final-sample
```

确认 smoke run 后，移除 CLI limit 运行 YAML 声明的完整实验：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
```

该示例同时启用原生 PyTorch LR scheduler、EMA、TensorBoard 和轻量
`diffusion_quality` diagnostic；reference KID/FID 默认关闭，因此不需要
`quality` extra。查看训练与 diagnostic 指标：

```bash
tensorboard --logdir outputs/ddpm_mnist/<run>/tensorboard
```

从最佳 checkpoint 采样：

```bash
uv run stochaflow sample \
  --checkpoint outputs/ddpm_mnist/<run>/checkpoints/best.pt
```

## 最小完整配置

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
      kind: torchvision
      dataset: MNIST
      root: ./data
      download: true
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

sampling:
  shape: [1, 32, 32]
  builder:
    name: standard_denoising
    params:
      weights: auto
      prediction_type: epsilon
      clip_denoised: true
      sampler: {name: ddim, params: {num_inference_steps: 100, eta: 0.0}}
      trajectory: {enabled: false, every_steps: 1}
  writers:
    - {name: tensor, params: {}}
    - {name: image, params: {grid_nrow: 4}}
```

`load_config()` 和 `load_config_dict()` 只把 YAML/mapping 结构化为 dataclass 并校验
schema，不导入第三方代码。runner 随后发现并预检所选
`stochaflow.extensions` entry points，在任何插件导入前处理 checkpoint provenance 和
version policy，再激活插件、执行跨组件校验并构建组件。训练、resume 和 sampling 因而
使用同一套显式插件选择与审计结果。

## 配置层次

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
| `sampling` | 独立采样和训练后验收采样 |
| `diagnostics` | 扩散特有的训练期诊断 |
| `trainer` | epoch、设备、梯度和提前停止 |
| `logging` | 日志频率与后端 |
| `artifacts` | checkpoint 周期 |

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
: 只定义图像 recipe 动态 batch 的像素预算基准；采样输出由 `sampling.shape` 独立声明。

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

仓库提供六份可直接加载的配置：MNIST DDPM、CIFAR-10 DDPM/DDIM、Flowers102
DDPM/DDIM，以及 MNIST + Flowers102 多源 DDPM。它们位于 `configs/`，并在 CI 中
逐一执行 schema 加载。
