# 配置使用与字段参考

Stochaflow 使用 YAML 描述一次实验。配置负责声明数据源、全局划分、
resolution bucket、模型、扩散过程、采样器、优化器、训练策略、日志与产物。
统一训练入口为：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
```

训练 runner 根据 `diffusion.name` 构建已注册的过程，因此同一入口可运行 DDPM、
DDIM 及后续兼容实现：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
uv run stochaflow train --config configs/ddim_cifar10.yaml
```

现成配置包括：

- [`ddpm_mnist.yaml`](../configs/ddpm_mnist.yaml)：单数据源、全局随机留出；
- [`ddpm_cifar10.yaml`](../configs/ddpm_cifar10.yaml)：单数据源、全局随机留出；
- [`ddpm_flowers102.yaml`](../configs/ddpm_flowers102.yaml)：官方划分、EMA、
  warmup cosine 与 DDPM diagnostics；
- [`ddpm_mnist_flowers102.yaml`](../configs/ddpm_mnist_flowers102.yaml)：多数据源、
  显式 step 权重和多分辨率 bucket；
- [`ddim_cifar10.yaml`](../configs/ddim_cifar10.yaml)：DDIM 训练与采样配置示例；
- [`ddim_flowers102.yaml`](../configs/ddim_flowers102.yaml)：Flowers102 DDIM 配置。

## Schema 规则

- 配置加载严格拒绝未知字段，拼写错误不会被静默忽略。
- 标记为“必填”的字段必须在 YAML 中出现；其余字段省略时使用表中的默认值。
- `model`、`objective`、`noise_schedule`、optimizer、LR scheduler、logger、
  diagnostic 和 dataset factory 都通过 registry 名称解析。
- 通用 component 结构为 `{name: <registry 名>, params: {...}}`；`params`
  会作为关键字参数传给对应类或 builder。
- 相对路径相对于启动命令时的当前工作目录，而不是 YAML 文件所在目录。
- 旧 `data.dataset` 和 `data.source` 已被删除并会产生明确迁移错误。新配置必须
  使用 `data.datasets`、`data.image` 和 `data.batching`。
- 旧 `diffusion.scheduler` 仍可被读取并迁移到 `noise_schedule`，但新配置应只写
  `diffusion.noise_schedule`。
- `sampling` 完全可选；即使 YAML 未声明，resolved config 与新 checkpoint 仍会记录
  展开后的默认 sampling 配置，便于复现。

## 顶层字段

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `experiment` | mapping | 必填 | 实验标识、随机种子与输出根目录。 |
| `data` | mapping | 必填 | 数据源、图像契约、bucket、loader 与划分策略。 |
| `model` | component | 必填 | 注册模型及其构造参数。 |
| `diffusion` | mapping | 必填 | 扩散过程与前向噪声 schedule。 |
| `objective` | component | 必填 | 训练目标。 |
| `optimizer` | mapping | 见下文 | 优化器。整段省略时使用项目的 Adam 默认配置。 |
| `lr_scheduler` | mapping | disabled | 优化器学习率 schedule。 |
| `ema` | mapping | disabled | 模型指数移动平均。 |
| `sampling` | mapping | 见下文 | 可选 sampler、采样批量与 debug 轨迹。整段可省略。 |
| `diagnostics` | list[component] | `[]` | 训练期 diagnostic 插件。 |
| `trainer` | mapping | 见下文 | epoch、设备、梯度裁剪与 early stopping。 |
| `logging` | mapping | local logger | 指标、文本日志与 Torch 内部日志。 |
| `artifacts` | mapping | 见下文 | checkpoint 保存频率。 |

## `experiment`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `name` | string | 必填 | 人类可读的实验名称，也作为 logger 的 run name。 |
| `seed` | int | `42` | 实验随机种子；用于全局划分、K-fold、batch sampler、DataLoader worker 与训练随机数。 |
| `output_dir` | string | `outputs/default` | 输出根目录。CLI 会在其下创建时间戳 run 目录。 |
| `exp_id` | string 或 null | `null` | 程序化调用时可使用的实验 id；训练 CLI 会用新建 run 目录名覆盖它，K-fold 会追加 `fold_XX`。 |

一次 CLI 运行通常生成：

```text
<output_dir>/<timestamp>/
  checkpoints/
    best.pt
    latest.pt
    epoch_XXXX.pt
  metrics.jsonl
  train.log
  diagnostics/          # 启用 diagnostic 时
  samples/final/        # 未传 --skip-final-sample 时，使用 best.pt 生成
    samples.pt
    samples.png
    resolved_sampling.yaml
```

## `data`

### `data` 本身

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `modules` | list[string] | `[]` | 在解析 factory 前幂等导入的 Python 模块。自定义模块通过 import side effect 注册类。 |
| `datasets` | list[dataset] | 必填、非空 | 一个或多个数据源，按 YAML 顺序参与全局合并。 |
| `image` | mapping | 必填 | 所有数据源共同遵守的通道数和数值范围。 |
| `batching` | mapping | 必填 | resolution bucket、采样 bucket 与 epoch 长度。 |
| `dataloader` | mapping | 见下文 | PyTorch DataLoader 策略。 |
| `splits` | mapping | `{mode: none}` | 在数据源合并后执行的全局划分策略。 |

### `data.modules`

模块必须能被当前 Python 环境 import。例如：

```yaml
data:
  modules:
    - my_project.datasets
```

模块内注册 factory：

```python
from stochaflow.data import DatasetFactory
from stochaflow.utils.registry import REGISTRIES

@REGISTRIES.dataset_factories.register("my_images")
class MyImagesFactory(DatasetFactory):
    ...
```

完整契约与实现示例见[自定义数据集](custom-datasets.md)。

### `data.datasets[]`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `id` | string | 必填 | 配置内唯一的数据源 id；同时进入 source metadata。 |
| `factory` | string | 必填 | `REGISTRIES.dataset_factories` 中的注册名。 |
| `params` | mapping | `{}` | 原样注入 `DatasetFactoryContext.params`；具体字段由 factory 定义。 |
| `splits` | mapping | `{train: train}` | 逻辑 train/validation/test 到原生 split 名称的映射。 |
| `sampling_weight` | float 或 null | `null` | 训练 step 的期望来源比例。必须为正数；所有 source 要么全部声明，要么全部省略。 |

`splits` 的字段为：

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `train` | string | `train` | 逻辑训练分区使用的原生 split；不可为空。 |
| `validation` | string 或 null | `null` | `official` 模式下使用的原生验证 split。 |
| `test` | string 或 null | `null` | 可选原生测试 split。多个 source 必须全部声明 test 或全部省略。 |

同一物理训练 split 的 train/eval view 必须返回完全一致且唯一的
`sample_keys`，这样 random holdout 与 K-fold 才能安全地把训练增强视图和确定性
验证视图对齐。

#### 内置 dataset factory 参数

| `factory` | `params` 字段 | 默认值 | 支持的原生 split |
| --- | --- | --- | --- |
| `mnist` | `root`、`download` | `./data`、`true` | `train` 为训练集；`test`、`eval`、`validation`、`val` 都映射到测试集。 |
| `cifar10` | `root`、`download`、`random_horizontal_flip` | `./data`、`true`、`true` | 与 MNIST 相同。 |
| `flowers102` | `root`、`download`、`random_horizontal_flip` | `./data`、`true`、`true` | `train`、`val`/`validation`、`test`。 |

内置 factory 拒绝未知参数，并只支持全局 `channels` 为 1 或 3。它们先根据原始
图像尺寸选择 bucket，再执行 resize-cover：train role 使用随机 crop，并在启用时
随机水平翻转；eval role 使用 center crop。通道转换、归一化和最终尺寸都在 factory
中完成，不在 collate 阶段隐式修改。

Dataset 可以返回图像 Tensor，或返回首元素为图像 Tensor 的 tuple/list。
`ImageBatchCollator` 会在 loader 边界只保留图像，因此不同 source 的标签或 metadata
结构不需要一致。

### 多数据源混合与 `sampling_weight`

不写权重时采用自然混合：所有 source 合并后按 bucket 组 batch，同一 bucket 可包含
多个 source。`steps_per_epoch: auto` 时，每个自然 batch 最多出现一次；
`drop_last: true` 会丢弃每个 bucket 的不足 batch 尾部。

写权重时，每个 batch 只来自一个 source。sampler 先按 `sampling_weight` 选择 source，
再按该 source 各 bucket 的自然 batch 数选择 bucket。权重是相对值，不要求和为 1：

```yaml
data:
  datasets:
    - id: digits
      factory: mnist
      sampling_weight: 0.4
      params: {root: ./data, download: true}
      splits: {train: train, test: test}
    - id: flowers
      factory: flowers102
      sampling_weight: 0.6
      params: {root: ./data, download: true}
      splits: {train: train, test: test}
```

小 source 会循环、重新 shuffle 并按需过采样。权重只影响 train；validation/test
始终不加权、不 shuffle、不 drop last。

### `data.image`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `channels` | int | `3` | 所有输出图像的通道数，必须为正；内置 factory 仅支持 1 或 3。UNet 的输入/输出通道必须与之相同。 |
| `normalize` | bool | `true` | 内置 factory 为 true 时把 `[0,1]` 映射到 `[-1,1]`；为 false 时保留 `[0,1]`。自定义 factory 必须自行遵守。 |

### `data.batching`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `buckets` | list[bucket] | 必填、非空 | 允许的输出分辨率，按声明顺序参与 tie-break。 |
| `sample_bucket` | string | 必填 | 必须指向已声明 bucket；决定采样、trajectory、diagnostic 的 `H×W`，也是基础 batch size 的像素基准。 |
| `dynamic_batch_size` | bool | `true` | 是否按像素预算为不同 bucket 缩放 batch size。 |
| `steps_per_epoch` | `auto` 或正 int | `auto` | `auto` 使用自然遍历产生的 batch 数；正整数强制每 epoch 的训练 step 数。 |

每个 `buckets[]` 元素包含：

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `name` | string | 必填 | bucket 唯一名称。 |
| `height` | int | 必填 | 输出高度，必须为正。 |
| `width` | int | 必填 | 输出宽度，必须为正。 |

Bucket 选择按以下顺序最小化距离：

1. 原图与 bucket 宽高比的对数距离；
2. 宽高比相同时，原图与 bucket 面积的对数距离；
3. 仍相同时，选择 YAML 中更早声明的 bucket。

动态 batch size 公式为：

```text
bucket_batch_size = max(
    1,
    floor(base_batch_size * sample_bucket_pixels / bucket_pixels),
)
```

其中 `base_batch_size` 是 `data.dataloader.batch_size`。因此该字段不是所有 bucket
的固定 batch size，而是 `sample_bucket` 的基准值。

`steps_per_epoch: auto` 等于各 bucket 自然 batch 数之和：`drop_last: true` 使用
floor，否则使用 ceil。显式整数会严格控制 step 数；无权重时必要时重复自然 batch，
有权重时通过 source/bucket 循环池持续取样。

### `data.dataloader`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `batch_size` | int | `128` | `sample_bucket` 的基础 batch size，必须为正。 |
| `num_workers` | int | `4` | DataLoader worker 数，必须非负；`0` 表示主进程加载。 |
| `shuffle` | bool | `true` | 是否打乱训练 bucket 内索引和 batch 顺序。eval 强制为 false。 |
| `drop_last` | bool | `true` | 是否丢弃训练 bucket 的不完整尾 batch。eval 强制为 false。 |
| `pin_memory` | bool | `true` | 传给 DataLoader 的 pinned-memory 选项。 |
| `persistent_workers` | bool | `true` | epoch 间保留 worker；只能在 `num_workers > 0` 时启用。 |
| `prefetch_factor` | int 或 null | `null` | 每个 worker 预取的 batch 数；非 null 时必须为正且要求 `num_workers > 0`。省略时由 PyTorch 使用其默认值。 |

### `data.splits`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `mode` | string | `none` | `none`、`official`、`random_holdout` 或 `kfold`。 |
| `validation_size` | int、float 或 null | `null` | random holdout 的验证集大小。int 表示样本数；float 必须在 `(0,1)` 内并表示比例。 |
| `num_folds` | int 或 null | `null` | K-fold 的 fold 数；`kfold` 时必须至少为 2，且不能超过合并训练集大小。 |
| `fold_index` | int 或 null | `null` | 只运行一个 fold，范围 `[0,num_folds)`；null 表示依次生成并训练全部 fold。 |

模式行为：

| mode | train | validation | test |
| --- | --- | --- | --- |
| `none` | 每个 source 的 `splits.train` 完整合并。 | 无。 | 若声明，合并 `splits.test`。 |
| `official` | 合并 `splits.train` 的 train role。 | 合并 `splits.validation` 的 eval role；所有 source 必须全部声明或全部省略。 | 若声明，合并 `splits.test`。 |
| `random_holdout` | 先合并所有 `splits.train`，再用全局随机索引留下训练部分。 | 从同一物理 train split 的 eval view 中选择全局 holdout；不使用 `splits.validation`。 | 若声明，合并 `splits.test`。 |
| `kfold` | 先合并所有 `splits.train`，每个 fold 使用其余全局索引。 | 当前 fold 在同一物理 train split 的 eval view 上取值；每个样本恰好作为一次验证样本。 | 若声明，合并 `splits.test`。 |

Random holdout 和 K-fold 使用 `experiment.seed`，并且是在所有 source 合并后全局执行，
不是每个 source 分别划分。

## `model`

```yaml
model:
  name: unet
  params: {...}
```

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `name` | string | 必填 | `REGISTRIES.models` 中的名称。内置值：`unet`。 |
| `params` | mapping | `{}` | 模型构造参数。 |

### 内置 `unet` 参数

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `in_channels` | int | `3` | 输入图像通道；必须等于 `data.image.channels`。 |
| `out_channels` | int | `3` | 预测输出通道；必须等于 `data.image.channels`。 |
| `base_channels` | int | `128` | 第一级特征宽度，应为正数。 |
| `channel_multipliers` | list[int] | `[1,2,4,8]` | 每级宽度为 `base_channels * multiplier`；必须非空。列表长度决定下采样次数。 |
| `num_res_blocks` | int | `2` | 每个 level 的 residual block 数，应为正数。 |
| `time_embedding_dim` | int | `128` | 正弦 timestep embedding 维度，至少为 2；MLP 输出宽度为 `base_channels * 4`。 |
| `dropout` | float | `0.0` | residual block 中的 dropout 概率。 |
| `attention_levels` | list[int] 或 null | `null` | 启用 self-attention 的零基 level 索引。最后一级被选中时 mid block 也启用 attention。 |
| `attention_heads` | int | `4` | attention head 数，必须为正，且对应 level 的通道数必须可被它整除。 |

若 UNet 有 `L = len(channel_multipliers)` 个 level，每个 bucket 的高宽都必须可被
`2 ** (L - 1)` 整除，否则 skip connection 的空间尺寸不能对齐。该约束在加载配置时
检查。

## `diffusion`

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `name` | string | 必填 | `REGISTRIES.diffusions` 中的过程名。内置：`ddpm`、`ddim`。 |
| `noise_schedule` | component | 必填 | 前向噪声路径；其 `params` 必须显式包含 `num_timesteps`。 |
| `params` | mapping | `{}` | 扩散过程构造参数；`model` 和 `noise_schedule` 由 runtime 注入，不应写入。 |

### `ddpm` 参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `clip_denoised` | bool | `true` | 反向过程中是否把估计的干净样本裁剪到 `[-1,1]`。 |

### `ddim` 参数

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `num_inference_steps` | int 或 null | `null` | 默认反向 transition 数；null 等于训练 `num_timesteps`，否则必须位于 `[1,num_timesteps]`。 |
| `eta` | float | `0.0` | DDIM 随机性，范围 `[0,1]`；0 为确定性 DDIM。 |
| `clip_denoised` | bool | `true` | 是否裁剪估计的干净样本。 |

## `diffusion.noise_schedule`

```yaml
noise_schedule:
  name: linear_beta
  params:
    num_timesteps: 1000
```

### `linear_beta`

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `num_timesteps` | int | 必填 | 前向 transition 数，必须为正。公共状态时间为 `0..T`。 |
| `beta_start` | float | `0.0001` | 第一条 transition 的 beta。 |
| `beta_end` | float | `0.02` | 最后一条 transition 的 beta；要求 `0 < beta_start < beta_end < 1`。 |
| `dtype` | `torch.dtype` | `torch.float32` | schedule buffer dtype。YAML 不能直接构造 `torch.dtype`，通常应省略；该参数主要供 Python API 使用。 |

### `cosine_alpha_bar`

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `num_timesteps` | int | 必填 | 前向 transition 数，必须为正。 |
| `s` | float | `0.008` | cosine alpha-bar 曲线的非负有限偏移。 |
| `max_beta` | float | `0.999` | 离散化后每个 beta 的上限，必须在 `(0,1)` 内。 |
| `dtype` | `torch.dtype` | `torch.float32` | 同上，YAML 中通常省略。 |

## `objective`

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `name` | string | 必填 | `REGISTRIES.objectives` 中的名称。内置：`ddpm_epsilon`。 |
| `params` | mapping | `{}` | objective 构造参数。 |

`ddpm_epsilon` 计算预测噪声与真实噪声之间的 MSE：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `reduction` | string | `mean` | MSE reduction。训练器需要标量 loss，推荐 `mean` 或 `sum`；`none` 会返回非标量并不适用于当前训练循环。 |

## `optimizer`

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `name` | string | `adam` | `REGISTRIES.optimizers` 中的名称。内置：`adam`、`adamw`。 |
| `params` | mapping | 项目默认值 | 直接传给 PyTorch optimizer；模型 parameters 由 runtime 注入。 |

整段 `optimizer` 或仅 `params` 省略时，项目默认参数为：

```yaml
optimizer:
  name: adam
  params:
    lr: 0.0002
    weight_decay: 0.0
    betas: [0.9, 0.999]
    eps: 1.0e-8
```

显式写 `params: {}` 则使用当前 PyTorch 类自己的默认值，而不是上述项目默认值。

### `adam` 参数

| 参数 | PyTorch 默认值 | 含义 |
| --- | --- | --- |
| `lr` | `0.001` | 学习率。 |
| `betas` | `[0.9,0.999]` | 一阶、二阶矩估计的指数衰减系数。YAML list 可传给构造器。 |
| `eps` | `1e-8` | 数值稳定项。 |
| `weight_decay` | `0` | 权重衰减系数。 |
| `amsgrad` | `false` | 是否使用 AMSGrad 变体。 |
| `foreach` | `null` | 是否使用 foreach 实现；null 由 PyTorch 选择。 |
| `maximize` | `false` | 是否最大化目标而非最小化。 |
| `capturable` | `false` | 是否允许在 CUDA graph 等可捕获环境中安全执行。 |
| `differentiable` | `false` | optimizer step 是否参与 autograd。 |
| `fused` | `null` | 是否使用 fused 实现；可用性依设备和 PyTorch 版本。 |
| `decoupled_weight_decay` | `false` | 是否使用解耦 weight decay；较旧的受支持 PyTorch 版本可能没有此参数。 |

### `adamw` 参数

参数与 Adam 相同，但 `weight_decay` 的 PyTorch 默认值为 `0.01` 且始终使用
AdamW 的解耦权重衰减；没有 `decoupled_weight_decay` 参数。高级执行参数
`foreach`、`maximize`、`capturable`、`differentiable`、`fused` 的支持受当前
PyTorch 版本和设备影响。项目不会修改这些值。

## `lr_scheduler`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `name` | string 或 null | `null` | null 禁用；内置：`warmup_cosine`、`cosine`、`step`、`multistep`、`exponential`、`linear`。 |
| `interval` | string | `step` | 启用 scheduler 时必须为 `step` 或 `epoch`；决定在每个 optimizer step 后还是每个 epoch 后调用一次。 |
| `params` | mapping | `{}` | scheduler 构造参数；optimizer 由 runtime 注入。 |

### `warmup_cosine`

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `warmup_steps` | int | 必填 | 线性 warmup step 数，必须为正。 |
| `total_steps` | 正 int 或 `auto` | 必填 | 总 schedule step 数，必须大于 warmup；`auto` 使用 runner 的有效 `steps_per_epoch * num_epochs`。 |
| `min_lr_ratio` | float | `0.0` | 最终学习率与初始学习率之比，范围 `[0,1]`。 |

通常将它与 `interval: step` 配合。`--limit-batches` 会参与有效
`steps_per_epoch` 计算，因此 `total_steps: auto` 与 smoke run 保持一致。

### 透传的 PyTorch schedulers

| `name` | `params` | 说明 |
| --- | --- | --- |
| `cosine` | `T_max` 必填；`eta_min=0.0`；`last_epoch=-1` | `CosineAnnealingLR`。`T_max` 的单位由 `interval` 决定。 |
| `step` | `step_size` 必填；`gamma=0.1`；`last_epoch=-1` | 每 `step_size` 次 scheduler 调用乘以 gamma。 |
| `multistep` | `milestones` 必填；`gamma=0.1`；`last_epoch=-1` | 在指定 scheduler 调用序号处衰减。 |
| `exponential` | `gamma` 必填；`last_epoch=-1` | 每次 scheduler 调用乘以 gamma。 |
| `linear` | `start_factor=1/3`；`end_factor=1.0`；`total_iters=5`；`last_epoch=-1` | 在指定调用次数内线性改变学习率。 |

这些 `params` 直接传给当前安装的 PyTorch。版本特有参数与精确语义以
[PyTorch optimizer/scheduler 文档](https://docs.pytorch.org/docs/stable/optim.html)
为准。

## `ema`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否维护模型参数与浮点 buffer 的 EMA shadow。 |
| `decay` | float | `0.9999` | EMA 衰减，必须满足 `0 <= decay < 1`。 |
| `update_after_step` | int | `0` | 前多少次 update 使用 decay 0 直接同步，必须非负。 |
| `update_every` | int | `1` | 每多少个 optimizer step 更新一次 EMA，必须为正。 |
| `use_for_sampling` | bool | `true` | 训练后采样、checkpoint 采样时是否临时应用 EMA 权重。 |

`enabled: false` 时其余 EMA 参数仍会被 schema 校验，但不会创建 EMA 对象。

## `sampling`

整段可省略。`sampler: null` 时继承顶层 `diffusion.name` 和
`diffusion.params`，始终复用训练 noise schedule。省略整段等价于：

```yaml
sampling:
  sampler: null
  num_samples: 16
  batch_size: null
  seed: null
  grid_nrow: 4
  debug:
    trajectory:
      enabled: false
      params: {}
      gif_fps: 8
```

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `sampler` | component 或 null | `null` | 独立反向 sampler；可与训练 diffusion 不同。其 `params` 只传给 sampler 构造器，不能覆盖 model 或 noise schedule。 |
| `num_samples` | int | `16` | 一次采样的总样本数，必须为正。 |
| `batch_size` | int 或 null | `null` | 每次反向过程的 batch；null 继承 `data.dataloader.batch_size`。总样本会按该值分批并在 CPU 合并。 |
| `seed` | int 或 null | `null` | 采样种子；null 继承 `experiment.seed`。 |
| `grid_nrow` | int | `4` | 最终样本网格每行数量，必须为正。 |
| `debug.trajectory.enabled` | bool | `false` | 是否执行 sampler-specific 轨迹 debug。 |
| `debug.trajectory.params` | mapping | `{}` | 直接传给 sampler 的 `sample_trajectory`。内置 DDPM 接受 `state_interval`，内置 DDIM 接受 `step_interval`。两者都必须为正。 |
| `debug.trajectory.gif_fps` | int | `8` | 循环 GIF 帧率，必须为正。 |

示例：DDPM 训练、DDIM 采样并保存每 10 个 DDIM inference step 的轨迹：

```yaml
sampling:
  sampler:
    name: ddim
    params:
      num_inference_steps: 100
      eta: 0.0
  num_samples: 64
  batch_size: 16
  seed: 123
  grid_nrow: 8
  debug:
    trajectory:
      enabled: true
      params:
        step_interval: 10
      gif_fps: 8
```

使用 DDPM sampler 时，轨迹参数改为数学状态时间间隔：

```yaml
sampling:
  sampler: null
  debug:
    trajectory:
      enabled: true
      params:
        state_interval: 100
      gif_fps: 8
```

采样固定使用训练 checkpoint 中的 model 和 noise schedule。权重选择由 checkpoint
内嵌配置的 `ema.enabled` 与 `ema.use_for_sampling` 共同决定：两者均为 true 时加载
`ema_denoiser_state_dict`，否则加载 `denoiser_state_dict`。开启轨迹后，除常规
`samples.pt` 和 `samples.png` 外还会生成 `trajectory.pt`、`trajectory.png` 和
`trajectory.gif`。

## `diagnostics[]`

每项都是 component：

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `name` | string | 必填 | `REGISTRIES.diagnostics` 中的注册名。内置：`ddpm`。 |
| `params` | mapping | `{}` | diagnostic 自身参数。`logger`、`output_dir` 与声明需要的 runtime context 参数由系统注入，不能在这里覆盖。 |

### `ddpm` diagnostic 参数

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `interval` | int | `100` | 每多少个 global step 记录 timestep/loss diagnostics，必须为正。 |
| `timestep_buckets` | int | `10` | 汇总 timestep loss 时的分桶数，必须为正。 |
| `sample_every_epochs` | int | `5` | 每多少个 epoch 保存生成样本，必须为正。 |
| `sample_num` | int | `16` | 每次 diagnostic 生成的样本数，必须为正。 |
| `sample_seed` | int | `123` | diagnostic 采样随机种子。 |
| `sample_grid_size` | int | `4` | 样本网格每行图片数，必须为正。 |
| `reconstruction_every_epochs` | int | `5` | 每多少个 epoch 保存加噪/重建图，必须为正。 |
| `reconstruction_timesteps` | list[int] | `[50,250,500,900]` | 重建时使用的数学状态时间；只有当前 schedule 有效范围内的值会执行，越界值会被跳过。 |
| `use_ema_for_artifacts` | bool | `true` | diagnostic 生成产物时是否临时应用 EMA。 |

`sample_shape` 不属于 YAML 参数；系统从 `data.image.channels` 与
`data.batching.sample_bucket` 注入 `(C,H,W)`。

## `trainer`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `num_epochs` | int | `1` | 目标总 epoch 数，必须为正；`--epochs` 可覆盖。resume 时它仍表示最终目标 epoch，而不是追加数量。 |
| `device` | string | `cpu` | `auto`、`cpu` 或 Torch 接受的设备字符串，如 `cuda`、`cuda:0`。`auto` 优先 CUDA，否则 CPU。 |
| `max_grad_norm` | float 或 null | `null` | 非 null 时启用全模型 gradient norm clipping，必须为正。 |
| `show_progress` | bool | `true` | 是否显示 Rich 训练进度；`--no-progress` 可在 CLI 上强制关闭。 |
| `early_stopping` | mapping | 见下文 | Early stopping 与 best checkpoint 的监控设置。 |

### `trainer.early_stopping`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | 是否在连续无提升后提前停止。best checkpoint 即使关闭 early stopping 仍会跟踪。 |
| `monitor` | string | `valid_loss` | 有验证集时监控的 epoch metric，通常为 `valid_loss` 或 `train_loss`；无验证集时 runner 强制使用 `train_loss`。 |
| `mode` | string | `min` | `min` 表示越小越好，`max` 表示越大越好。 |
| `patience` | int | `10` | 允许连续无提升的 epoch 数，必须为正。 |
| `min_delta` | float | `0.0` | 计为提升所需的最小变化量，必须非负。 |

## `logging`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `log_every` | int | `100` | 每多少个 global step 写一次 batch 级训练指标，必须为正。epoch 指标始终写入。 |
| `backends` | list[component] | `[{name: local}]` | 一个或多个 logger，不能为空；多个 backend 由 CompositeLogger 扇出。 |
| `torch_logs` | mapping | `{}` | 传给 `torch._logging.set_logs(**settings)` 的版本相关参数；字符串日志级别如 `INFO` 会先转换为 Python logging 常量。 |

Logger 的 `output_dir` 和 `run_name` 由系统注入，不写进 `params`。

### `local` logger 参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `console` | bool | `true` | 是否同时把文本日志输出到控制台。 |
| `text_filename` | string | `train.log` | 文本日志文件名。 |
| `metrics_filename` | string | `metrics.jsonl` | JSON Lines 指标文件名。 |
| `append` | bool | `false` | true 追加现有文件，false 覆盖。CLI 通常创建新的时间戳目录。 |

### `tensorboard` logger 参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `subdir` | string | `tensorboard` | run 输出目录下的 TensorBoard 子目录。最终路径为 `<output>/<subdir>/<experiment.name>`。 |

### `wandb` logger 参数

先安装可选依赖：

```bash
uv sync --extra wandb
```

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `project` | string | `stochaflow` | W&B project。 |
| `entity` | string 或 null | `null` | W&B entity/team。 |
| `mode` | string 或 null | `null` | W&B mode，例如 `online`、`offline` 或 `disabled`。 |
| `tags` | list[string] 或 null | `null` | Run tags。 |

## `artifacts`

| 字段 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `checkpoint_every` | int | `1` | 每多少个 epoch 保存一个 `epoch_XXXX.pt`，必须为正。`latest.pt` 每个已完成 epoch 都会更新，`best.pt` 在监控指标改善时更新。 |

Checkpoint 包含完整 diffusion model、optimizer、可选 LR scheduler、可选 EMA、epoch、
global step、resolved config、metrics 与 metadata。格式版本 2 另外保存
`denoiser_state_dict`，并在启用 EMA 时保存 `ema_denoiser_state_dict`；独立采样只加载
portable denoiser，因此 DDPM 训练权重可以交给 DDIM sampler。只保存新 data schema。

## K-fold 配置示例

运行全部 fold：

```yaml
data:
  # datasets/image/batching/dataloader 同常规配置
  splits:
    mode: kfold
    num_folds: 5
    fold_index: null
```

只运行 fold 2：

```yaml
data:
  splits:
    mode: kfold
    num_folds: 5
    fold_index: 2
```

每个 fold 写入独立的 `fold_XX` 子目录。多 fold resume 必须使用 `--resume`
自动查找或传入包含各 fold checkpoint 的目录，不能把单个 checkpoint 文件应用到
所有 fold。

## CLI 覆盖与 smoke run

常用训练参数：

| CLI 参数 | 作用 |
| --- | --- |
| `--config PATH` | 指定 YAML。 |
| `--epochs N` | 覆盖 `trainer.num_epochs`。 |
| `--limit-batches N` | 每 epoch 最多训练 N 个 batch；也影响 warmup cosine 的 auto total steps。 |
| `--limit-validation-batches N` | 每 epoch 最多验证 N 个 batch。 |
| `--limit-test-batches N` | 最终测试最多 N 个 batch。 |
| `--deterministic` | 请求 Torch 使用可用的确定性行为。 |
| `--no-progress` | 关闭 Rich 进度显示。 |
| `--resume [PATH]` | 无 PATH 时查找输出根目录下最新 `latest.pt`；也可传 checkpoint 或 run 目录。 |
| `--device DEVICE` | 覆盖 `trainer.device`。 |
| `--output-dir PATH` | 覆盖 `experiment.output_dir`。 |
| `--skip-final-sample` | 不执行最佳 checkpoint 的训练验收采样。 |

推荐先执行短 smoke run：

```bash
uv run stochaflow train \
  --config configs/ddpm_mnist.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --limit-validation-batches 1 \
  --limit-test-batches 1 \
  --skip-final-sample \
  --no-progress
```

## 独立 checkpoint 采样

`stochaflow sample` 支持 checkpoint-only、config-only 或两者同时提供。只给
checkpoint 时使用其中保存的配置：

```bash
uv run stochaflow sample \
  --checkpoint outputs/<run>/checkpoints/best.pt
```

只给 config 时，在 `experiment.output_dir` 下递归查找修改时间最新的 `best.pt`，并使用
该文件的权重：

```bash
uv run stochaflow sample --config configs/ddpm_mnist.yaml
```

两者同时给出时，checkpoint 提供训练结构、noise schedule、seed、EMA 选择和权重，
外部 config 只提供 `sampling` 段。加载前会比较 `model`、完整训练 `diffusion` 和
`data.image.channels`；任一不兼容都会报错。外部 config 的 data bucket、trainer、
experiment 或 EMA 字段不会覆盖 checkpoint 内嵌值。外部 config 省略 `sampling` 时，
使用本节列出的 sampling 默认值。

无需额外 YAML 即可切换 sampler：

```bash
uv run stochaflow sample \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --sampler ddim \
  --sampler-param num_inference_steps=100 \
  --sampler-param eta=0.0
```

CLI 合并顺序为 checkpoint 内嵌配置、外部 config 的 sampling 段、`--sampler`、
`--sampler-param`。切换 sampler 名称时不会继承原 sampler params；同名时则保留并
覆盖。没有显式 sampler 时，名称和参数继承训练 `diffusion`。

`--sampler-param` 在第一个 `=` 处分割。key 必须是合法参数名；value 使用 YAML
scalar/list 解析，因此整数、浮点数、布尔值、null 和列表都保留其类型。重复 key
最后一次生效，mapping 值不接受。例如：

```bash
uv run stochaflow sample \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --sampler ddim \
  --sampler-param num_inference_steps=100 \
  --sampler-param eta=0.0 \
  --sampler-param clip_denoised=false
```

若自定义 sampler 接受列表或含等号的字符串参数，应给完整的 `KEY=VALUE` 加引号，
例如 `--sampler-param "timesteps=[999, 499, 0]"` 或
`--sampler-param "label=a=b"`。这些名称只是解析语法示例，不是内置 DDPM/DDIM 参数。

每次运行都会保存 `resolved_sampling.yaml`，其中记录 checkpoint、格式版本、最终
sampler 与 params、展开后的 sampling 配置、seed、device、raw/EMA 权重选择和产物
路径。独立采样默认写入 `<checkpoint-run>/samples/<timestamp>/`；显式
`--output-dir` 时直接写入指定目录。

采样 CLI 字段：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--checkpoint PATH` | null | checkpoint 文件或目录；目录下递归查找最新 `best.pt`。与 config 至少提供一个。 |
| `--config PATH` | null | 外部实验配置；单独提供时在 `experiment.output_dir` 下查找最新 `best.pt`。 |
| `--sampler NAME` | 配置值 | 覆盖注册 sampler 名称。 |
| `--sampler-param KEY=VALUE` | 无 | 可重复的 sampler 构造参数覆盖；value 按 YAML scalar/list 解析，同名 key 最后一次生效。 |
| `--output-dir PATH` | checkpoint 旁的时间戳目录 | 生成产物目录。 |
| `--device DEVICE` | checkpoint 配置中的 trainer device | 覆盖采样设备。 |

新 checkpoint 同时保存完整训练状态、portable denoiser 和可选 EMA denoiser。
旧 checkpoint 缺少格式版本或 portable 字段时不能通过新采样入口加载，也不会自动
转换。

## 常见配置错误

- `model.params.in_channels/out_channels` 与 `data.image.channels` 不同；
- bucket 高宽不能被 UNet 的总下采样倍数整除；
- `sample_bucket` 没有在 `buckets` 中声明；
- 多 source 只给一部分 source 配置 `sampling_weight`、test split 或 official
  validation split；
- `persistent_workers: true` 但 `num_workers: 0`；
- `prefetch_factor` 非正，或在 `num_workers: 0` 时设置；
- random holdout 忘记 `validation_size`，或验证集大小没有同时留下至少一个训练和
  一个验证样本；
- K-fold 的 `num_folds` 大于合并训练集大小；
- `warmup_cosine.total_steps <= warmup_steps`；
- `sample` 同时缺少 `--config` 和 `--checkpoint`；
- 外部 config 与 checkpoint 的 model、训练 diffusion/noise schedule 或图像通道不兼容；
- sampler 切换后仍传入只属于旧 sampler 的参数，或 trajectory 参数与 sampler 不匹配；
- 请求 EMA 采样，但 checkpoint 没有 `ema_denoiser_state_dict`；
- 自定义 Factory 的 train/eval view 使用了不同的 `sample_keys` 或错误的 bucket id；
- Dataset 输出不是图像 Tensor，也不是首元素为图像 Tensor 的非空 tuple/list。
