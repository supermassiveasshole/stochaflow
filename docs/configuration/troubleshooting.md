# 配置与运行排错

本页按常见错误文字组织。先保留完整异常链；Registry 的最后一行通常是表象，前面的
导入或构造错误更能说明根因。

## Schema

### `unknown config field(s)`

YAML 含拼写错误、层级错误或已删除字段。到[字段参考](reference.md)搜索完整
dotted path。schema 不忽略未知字段。

### `must not be null` / `missing required positional argument`

前者表示非 Optional schema 字段写成了 `null`；后者通常表示组件或原生 PyTorch target
的 `params` 缺少构造参数。分别查[字段索引](reference.md)、框架组件索引和所安装版本的
上游 API 文档。

## 扩展模块与 Registry

### `failed to import registry module '...'`

确认 `extensions.modules` 使用完整 Python import path，当前环境已安装扩展包，而且
模块的依赖也能导入。不要把文件路径（如 `my_project/datasets.py`）写进配置，应写
`my_project.datasets`。

### `unknown ... 'name'. Available: ...`

普通名称不在对应 Registry。检查：

1. 装饰器是否注册到了正确 Registry；
2. 定义装饰器的模块是否列入 `extensions.modules`；
3. 名称大小写是否完全一致；
4. 扩展模块是否在训练与 checkpoint sampling 的 Python 环境中都可导入。

标准 PyTorch optimizer/scheduler 不使用 Registry alias，应写完整受限 target，例如
`torch.optim.AdamW` 或 `torch.optim.lr_scheduler.CosineAnnealingLR`。

### `already registered by ...`

两个不同组件使用了同一 Registry 名。改用项目命名空间前缀，例如
`acme_manifest_images`；不要覆盖内置名称。

### `registrations must inherit ...`

注册类继承了错误基类。数据入口继承 `DataBuilder`，采样输出继承
`SamplingArtifactWriter`，概率过程继承 `Process`，求解器继承 `Sampler`，任务采样器
继承 `SamplingBuilder`；模型/目标继承 `torch.nn.Module`，logger 继承
`ExperimentLogger`。复用内置 Gaussian 训练或采样还需分别实现
`DiscreteGaussianDenoisingProcess` 能力。

### `... sampler requires ... Dynamics`

SamplingBuilder 组合了不兼容的算法 family。内置 DDPM/DDIM 只消费
`GaussianDenoisingDynamics`；自定义 flow、SDE 或其他 Sampler 应检查自己所属 family 的
窄 Dynamics 契约。修复 Builder 的组合，不要在核心添加 Process/Sampler 名称兼容表，
也不要给 `GenerativeDynamics` 根类型补万能方法。

### `config cannot override runtime parameter(s)`

从组件 `params` 删除运行时注入参数，例如 diagnostic 的 `logger`、`output_dir`、
`sample_shape`。完整列表见[扩展手册](extensions.md#其他构造约定)。

optimizer 的 trainable parameter iterable 和 LR scheduler 的 optimizer 也由核心注入。
不要在 `optimizer.params` 中再次写 `params`，也不要在 `lr_scheduler.params` 中写
`optimizer`。

### `unknown native ... target` / `must name a direct class`

原生 provider 不是任意 Python importer。optimizer 只能使用单层
`torch.optim.<Class>`，LR scheduler 只能使用单层
`torch.optim.lr_scheduler.<Class>`；目标必须是满足对应 PyTorch 基类契约的类。其他实现
应由已安装 extension 注册到相应 Registry，再在配置中使用其注册名。

### `failed to initialize optimizer` / `failed to initialize lr scheduler`

Stochaflow 会将配置 `params` 原样交给所选构造器，并保留原始异常链。检查 target 对应的
当前 PyTorch 文档、拼写、类型和值范围。框架不会复制完整签名、补默认值或把旧短 alias
翻译成原生 target。

`T_max`、`total_steps` 等参数必须是构造器接受的确定值；`auto` 不再具有特殊含义。
CLI `--epochs` 或 `--limit-batches` 也不会自动更新它们。

### `lr scheduler step must be callable without arguments`

当前自动 Trainer 只支持在配置的 `step` 或 `epoch` interval 调用 `scheduler.step()`。
`ReduceLROnPlateau` 等需要 validation metric 的 scheduler 尚无明确 monitor lifecycle，
不能通过这个入口使用。不要仅为了绕过检查添加无意义默认 metric；应等待对应训练循环
契约，或在拥有完整生命周期的自定义训练 loop family 中处理。

同样地，当前 Trainer 直接调用 `optimizer.step()`，因此 `LBFGS` 等必须提供 closure 的
optimizer 会在构建边界收到 `optimizer ... step() must be callable without arguments`。
这不是构造参数错误，而是所选 optimizer 不满足当前自动训练 lifecycle。

## 数据与 partition

### `DataBuilder.build() must return DataLoaders`

自定义 builder 返回了 Dataset、DataLoader、list 或旧数据契约。将组装后的 train、
validation、test iterable 包装为单个 `DataLoaders`。

### `train loader has no length; steps_per_epoch is required`

生成器或流式 loader 没有 `len()`，核心无法判断 epoch 何时结束。由 builder 返回
`DataLoaders(train=loader, steps_per_epoch=<正整数>)`。

### `partition mode 'official' requires ...`

所选内置 source 没有提供 recipe 需要的原生 role。改用 `none`/`holdout`，或选择具有
对应 train/validation/test 分区的 source。partition 只属于支持它的内置 recipe；自定义
DataBuilder 可以采用完全不同的逻辑。

### `validation_size must leave at least one ... sample`

holdout 样本数为 0 或占满 train。浮点比例必须位于 0 与 1 之间；整数必须小于训练
样本数。

### `num_folds must not exceed the dataset size` / `fold_index is required`

降低 `num_folds`、增加训练样本，并为本次独立运行指定从 0 开始的 `fold_index`。

## bucket、batch 与 worker

### 一个 batch 出现不同形状

自定义 DataBuilder 应通过 transform、Sampler 或 collate 保证 batch 兼容。
`multi_resolution_image` 会在其私有实现中扫描图像尺寸、选择 bucket 并 resize/crop；
若扩展数据不满足该 recipe 的约束，应注册自己的 DataBuilder。

### batch size 与配置值不同

图像 recipe 的 `data.params.loader.batch_size` 只对应 `base_bucket`。启用
`dynamic_batch_size` 后，其他 bucket 按像素预算缩放；公式见
{ref}`数据构建 <multi-resolution-image-recipe>`。

### `persistent_workers requires num_workers > 0`

本地调试若设 `num_workers: 0`，同时设 `persistent_workers: false`，并删除或设
`prefetch_factor: null`。

### DataLoader worker 退出或卡住

先用 `num_workers: 0` 让真实 Dataset 异常出现在主进程，再检查 Dataset 是否可
pickle、文件句柄是否按 worker 打开、路径是否有效。确认后逐步恢复 worker 数量。

## 训练、checkpoint 与采样

### 训练时出现 state shape / channel 错误

核心不把模型 shape 与数据构建做任务专属交叉校验。确认所选 DataBuilder 的 batch、
TrainingBuilder 组装的 TrainingStrategy、模型输入输出契约彼此一致。内置图像 recipe
的通道配置位于 `data.params.image.channels`。

### resume 找不到 `latest.pt`

无参数 `--resume` 在配置的 `experiment.output_dir` 下寻找最新 run 的
`checkpoints/latest.pt`。若输出根改变，传入明确 checkpoint 路径。

### checkpoint 配置与外部采样配置不兼容

同时传 `--config` 与 `--checkpoint` 时，外部文件可以只包含 `sampling` 和可选
`extensions`，不能改变权重依赖的模型结构或可选训练 Process。若传入完整 Stochaflow
配置，model 和 Process（包括是否为 `null`）会与 checkpoint 配置做兼容性校验。只需切换
采样行为时，优先使用轻量 YAML，在 `sampling` 段改变 Builder、Sampler 和 solver 参数。

### 预期 EMA 采样但 manifest 显示 raw

同时检查 `ema.enabled`、`ema.use_for_sampling` 和 checkpoint 是否包含 EMA state。
旧 checkpoint 或训练期间禁用 EMA 时不能事后恢复 EMA 权重。

### trajectory 不生成

对 `standard_denoising` 设置
`sampling.builder.params.trajectory.enabled: true` 和正整数 `every_steps`。所有 Sampler
共用 observer 契约，不再提供 sampler-specific trajectory 方法。还需声明能保存
trajectory 的 writer，例如 `tensor` 或 `image`。

### `sampling.shape is required`

`standard_denoising` 需要固定 shape。把单样本形状（不含 batch 维）写入
`sampling.shape`。该值与 DataBuilder 独立，外部 sampling 配置可以覆盖它。自定义
SamplingBuilder 可以在 shape 为 null 时构造自己的 initial state。

### image writer 报 NCHW 或通道错误

只有 `image` writer 要求 rank 4 batch 且通道数为 1 或 3。物理场、序列或其他
Tensor 只声明 `tensor` writer，或注册领域 writer。

### `checkpoint format version ... is unsupported`

当前 checkpoint 格式为 v7，分别保存 primary model、可选 Process/Objective、可选
EMA model、带 concrete class identity 的 optimizer/scheduler，以及按稳定名称组织的
training assets。训练恢复和 checkpoint-only sampling 不读取 v6 及更早格式；
请用当前代码重新训练或重新生成 checkpoint。

## 文档生成与 CI

### `field metadata is out of date`

配置 dataclass 已变化，但 `docs/configuration/_reference.yaml` 未同步。为错误中列出的
每个 missing 字段写中文说明，删除 stale 字段，再运行：

```bash
uv run python tools/generate_config_reference.py
uv run python tools/generate_config_reference.py --check
```

### `registry ... metadata is out of date` / `CLI metadata is out of date`

新增或删除 Registry 组件/CLI 参数后，同步 `_reference.yaml`。项目自有组件还会校验
构造签名，避免参数表静默过期。

### Sphinx `-W` 构建失败

本地运行与 CI 相同的命令；第一个 warning 通常最接近根因：

```bash
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

常见原因是内部链接目标不存在、heading 改名后 anchor 过期、代码 fence 未闭合或
MyST directive 缩进错误。
