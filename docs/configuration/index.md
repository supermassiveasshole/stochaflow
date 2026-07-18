# Stochaflow 配置手册

本手册是 `stochaflow train`、`stochaflow sample`、内置组件和自定义扩展的
可查阅说明。顶层配置采用严格 schema；`data.params` 的语义由所选 DataPipeline
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
| 配置 map、图像、流式或自定义数据管线 | [数据管线](data-pipeline.md) |
| 注册自定义 DataPipeline、DatasetFactory、writer 或其他组件 | [扩展与 Registry](extensions.md) |
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
  name: multi_resolution_image
  params:
    datasets:
      - id: mnist
        factory: mnist
        splits: {train: train, test: test}
    image: {channels: 1, normalize: true}
    batching:
      buckets:
        - {name: square_32, height: 32, width: 32}
      base_bucket: square_32

model:
  name: unet
  params:
    in_channels: 1
    out_channels: 1

diffusion:
  name: ddpm
  noise_schedule:
    name: linear_beta
    params:
      num_timesteps: 1000

objective:
  name: ddpm_epsilon
  params: {}

sampling:
  shape: [1, 32, 32]
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
| `data` | DataPipeline Registry 声明与管线专属参数 |
| `model` | 去噪模型 Registry 声明 |
| `diffusion` | 训练扩散过程和前向噪声 schedule |
| `objective` | 训练损失 |
| `optimizer` / `lr_scheduler` | 参数更新与学习率策略 |
| `ema` | 模型参数指数移动平均 |
| `sampling` | 独立采样和训练后验收采样 |
| `diagnostics` | 扩散特有的训练期诊断 |
| `trainer` | epoch、设备、梯度和提前停止 |
| `logging` | 日志频率与后端 |
| `artifacts` | checkpoint 周期 |

## 术语表

source
: 内置图像管线 `data.params.datasets` 中的一项，由唯一 `id` 标识。

native split
: 数据集实现提供的物理分区名，例如 MNIST 的 `train`/`test`、Flowers102 的
  `train`/`val`/`test`。

logical split
: Stochaflow 统一使用的 `train`、`validation`、`test` 角色，由每个 source 的
  `splits` 映射到 native split。

role
: Factory 构建视图时收到的 `train` 或 `eval` 预处理角色。它与 native split
  分开：random holdout 会从同一个 native train split 构建 train-role 与
  eval-role 两个对齐视图。

bucket
: `multi_resolution_image` 管线的命名空间尺寸；图像 metadata 决定 bucket，batch
  sampler 保证一个 batch 内形状一致。

base bucket
: 只定义图像管线动态 batch 的像素预算基准；采样输出由 `sampling.shape` 独立声明。

Registry
: 名称到组件类/构造器的显式映射。配置只写名称和参数；`extensions.modules`
  导入通用组件，diagnostic 的 `params.modules` 导入可插拔 provider。

## 内置示例

仓库提供六份可直接加载的配置：MNIST DDPM、CIFAR-10 DDPM/DDIM、Flowers102
DDPM/DDIM，以及 MNIST + Flowers102 多源 DDPM。它们位于 `configs/`，并在 CI 中
逐一执行 schema 加载。
