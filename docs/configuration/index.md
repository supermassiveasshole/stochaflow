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
python -m pip install -e ".[test]"
stochaflow train --config experiments/example/train.yaml
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。`stochaflow init`
生成普通 Python distribution、一个最小 extension、测试和可运行配置；它不创建环境、
安装依赖或要求 `uv`。生成项目的 README 继续说明 resume 和 checkpoint sampling。

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
| `stochaflow` | 核心 runtime、TensorBoard logger 和 CLI |
| `stochaflow[wandb]` | W&B logger |
| `stochaflow[quality]` | KID/FID diagnostic |
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

两个 sample profile 都保留当前顶层 `sampling:` request envelope，只选择 sampler、
request options 和 writers；checkpoint 继续提供固化的 inference recipe。训练配置启用
逐 step warmup-cosine LR scheduler、EMA、TensorBoard 和固定 seed 的 DDIM-50
`diffusion_quality` 对照。diagnostics 属于训练配置，不会由 sample profile 改写。
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

sampling:
  run_after_training: true
  shape: [1, 32, 32]
  sampler: {name: ddim, params: {num_inference_steps: 100, eta: 0.0}}
  options:
    weights: auto
    clip_denoised: true
    trajectory: {enabled: false, every_steps: 1}
  writers:
    - {name: tensor, params: {}}
    - {name: image, params: {grid_nrow: 4}}
```

`load_config()` 和 `load_config_dict()` 只把 YAML/mapping 结构化为 dataclass 并校验
schema，不导入第三方代码。runner 随后发现并预检所选
`stochaflow.extensions` entry points，在任何插件导入前处理 checkpoint provenance 和
version policy，再激活插件、执行跨组件校验并构建组件。训练、resume 和
checkpoint-backed inference 因而使用同一套显式插件选择与审计结果。这里的
`prediction_type` 来自 TrainingBuilder 固化到 v10 checkpoint 的 recipe contract，
不会在 sampling request 中重复声明。

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
| `sampling` | 训练后 inference 开关，以及 checkpoint recipe 的可调 request defaults |
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

仓库维护四份职责明确的内置配置：

```text
examples/built-in/image-generation/configs/
├── train/mnist.yaml
├── sample/mnist-ddpm.yaml
├── sample/mnist-ddim-50.yaml
└── overlays/mnist-observability.yaml
```

其中只有 `train/mnist.yaml` 是完整训练配置；两个 `sample/` 文件是 checkpoint-bound
partial request，overlay 只用于 strict resume 的 diagnostics/logging。仓库不再维护
独立的 CIFAR-10、Flowers102 或 multi-source runnable YAML；这是示例收敛，不是对底层
数据来源或 recipe 能力的否定。
