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
workflows
troubleshooting
```

## 按任务查阅

| 任务 | 入口 |
| --- | --- |
| 查某个 YAML 字段、默认值或 CLI 覆盖 | [完整字段参考](reference.md) |
| 配置图像、超分辨率或自定义数据构建 | [数据构建](data-pipeline.md) |
| 注册自定义 DataBuilder、writer 或其他组件 | [扩展与 Registry](extensions.md) |
| 训练、smoke run、恢复和 checkpoint 采样 | [常用工作流](workflows.md) |
| 根据错误信息定位问题 | [排错索引](troubleshooting.md) |

## 五分钟快速开始

安装开发环境后，先做两个 batch 的 MNIST smoke run：

```bash
uv sync --extra dev
uv run stochaflow train \
  --config configs/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --skip-final-sample
```

确认训练可用后，移除 CLI 限制运行 YAML 中声明的完整实验：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
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

配置加载顺序是：YAML 结构化为 dataclass → 导入
`extensions.modules` → 执行跨字段校验 → runner 应用 CLI 覆盖 → 构建数据和训练
组件。自定义模块因此既可用于训练，也可在只读取 checkpoint 的采样流程中注册组件。

## 配置层次

| 顶层段 | 作用 |
| --- | --- |
| `experiment` | 名称、seed、输出根目录和 run id |
| `extensions` | 训练、采样和 checkpoint 重建前导入的 Registry 扩展模块 |
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

`interval` 是 Stochaflow 的训练生命周期策略，不是 PyTorch 构造参数。首版自动训练循环
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
: 名称到 Stochaflow 组件类/构造器的显式映射。`extensions.modules` 导入通用组件，
  diagnostic 的 `params.modules` 导入可插拔 provider。标准 PyTorch optimizer 和 LR
  scheduler 使用上面的受限原生 provider，不会复制到 Registry 中。

## 内置示例

仓库提供六份可直接加载的配置：MNIST DDPM、CIFAR-10 DDPM/DDIM、Flowers102
DDPM/DDIM，以及 MNIST + Flowers102 多源 DDPM。它们位于 `configs/`，并在 CI 中
逐一执行 schema 加载。
