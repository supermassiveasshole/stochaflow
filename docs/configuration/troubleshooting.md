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

## Entry-point 插件与 Registry

### `secure publication into a pre-existing empty directory is not supported`

当前平台缺少安全的 descriptor-relative 文件系统原语。该限制只影响已存在的空目标目录；
确认目录为空后删除它，再运行同一条 `stochaflow init NAME`，让 CLI 原子创建目标目录。
非空目录始终会在写入前拒绝。

### 请求的 extension plugin 未安装

确认 `extensions.plugins` 使用 `[project.entry-points."stochaflow.extensions"]` 声明的
精确 entry-point name，而不是 Python module path、distribution 的任意别名或文件路径。
扩展 distribution 与 `stochaflow` CLI 必须安装在同一 Python environment；错误信息中的
Python executable 可以帮助确认实际运行环境。

错误上下文取决于 provenance 来源：

- fresh config 只知道请求的 entry-point name，因此 distribution/target 会明确显示为
  unavailable；entry-point name 不保证等于 distribution name；
- checkpoint-backed resume/sampling 已保存 expected provenance，因此还会显示 expected
  distribution、version 和 target。

使用任意 Python package manager，把“声明该
`stochaflow.extensions` entry point 的 distribution”安装到错误中所示 Python executable
对应的 environment，再重试。不要根据 entry-point name 猜包名，也不需要改用某个指定的
包管理器。

`load_config()`/`load_config_dict()` 不会导入插件，所以“配置可以解析”不代表当前环境已经
安装或成功激活 extension。先在 CLI 所在环境执行对应包管理器的包列表/metadata 检查，再
确认 entry-point target 的依赖均可导入。

### `failed to activate extension plugin ...`

插件已发现，但导入 pure-module target 或 decorator 注册失败。错误上下文会包含 entry-point
name、distribution、version 和 target；继续阅读保留的原始异常链，检查聚合模块的 import
依赖和 Registry 重名。聚合模块导入时只应定义并注册组件，不应访问数据、构建 runtime
资产或启动任务。发生 partial import 后必须重启进程再试。

### duplicate `stochaflow.extensions` entry-point name

两个 distribution 声明了相同插件名。Stochaflow 不采用“后加载覆盖”，即使配置只选择
该名称也无法确定身份；卸载冲突 distribution 或修改其中一个 entry-point name。若配置是
非空显式列表，未被选择的其他插件 metadata 不会影响本次运行；`plugins: null` 会检查全
环境，因此也会暴露所有冲突。

### `extension plugin version mismatch`

checkpoint 记录的插件 version 与当前安装 version 不同，但 name/distribution/target identity
仍一致。交互式 CLI 会在任何插件代码导入前汇总询问，默认 No；非交互式运行直接失败。
确认代码和 checkpoint state contract 后，可以传
`--force-extension-version-mismatch`。该 flag 只接受 version 差异，不能绕过 identity、
缺失插件或 state shape/asset topology 错误。

若 version 没变但 editable source 已修改，distribution metadata 无法检测。checkpoint 不
freeze extension class 或源码；请用项目 lockfile、发布版本或其他代码 provenance 管理环境。

### `extension activation previously failed; restart the Python process`

某个聚合模块导入、decorator 注册或 Registry 冲突已造成 partial activation。注册是进程
全局副作用，不能安全 unload/rollback；修复最初异常并重启进程。相同 selection 只在首次
activation 完整成功后才可幂等复用，一个进程也不能先后切换不同 plugin selections。

### `unknown ... 'name'. Available: ...`

普通名称不在对应 Registry。检查：

1. 装饰器是否注册到了正确 Registry；
2. 聚合模块是否从 entry point 暴露，且其精确 name 是否列入 `extensions.plugins`；
3. 名称大小写是否完全一致；
4. 扩展 distribution 是否在训练与 checkpoint sampling 的 Python 环境中都已安装。

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

### MPS 出现 pinned-memory 警告

内置 recipe 的 `loader.pin_memory` 默认是 `false`，这是 CPU、CUDA 与 MPS 间的可移植
默认值。MPS 通常不受益于 CUDA 风格的 pinned-memory DataLoader；保持关闭即可。CUDA
用户可以在测量输入吞吐后显式启用。

## 训练、checkpoint 与采样

### MPS 拒绝 `float64` module 或 Process

Apple MPS 不支持 `float64` module parameter/buffer。把模型或 Process coefficient
构造为 `float32`；若算法确实要求 double precision，则显式改用 CPU 或 CUDA。核心不会
静默降精度，因为这会改变扩展声明的数值语义。

### `--deterministic` 因不支持的算子失败

该 flag 启用 PyTorch 严格 deterministic-algorithm 模式；没有确定性实现的算子会直接
报错。移除或替换该算子，或在接受非确定性执行时不传此 flag。固定 seed 仍是必要条件，
但不能保证不同 backend 或 PyTorch 版本逐位一致。

### 训练时出现 state shape / channel 错误

核心不把模型 shape 与数据构建做任务专属交叉校验。确认所选 DataBuilder 的 batch、
TrainingBuilder 组装的 TrainingStrategy、模型输入输出契约彼此一致。内置图像 recipe
的通道配置位于 `data.params.image.channels`。

### `train` 要求 `--config` 或 `--resume CHECKPOINT`

新训练传 `--config`；严格恢复传明确 checkpoint 文件或 run directory。两者互斥，裸
`--resume` 不再搜索 config 输出根。resume 使用 checkpoint 保存的 config 与完整
optimizer/scheduler/EMA/progress state，只允许文档化的安全 runtime flags，并写入新的兄弟
run directory。需要更换组件配置并只加载旧权重不是 resume，当前没有通用 warm-start CLI。

如果单独复制了 `latest.pt` 或某个 epoch checkpoint，strict resume 还会要求原
`checkpoints/` 目录中的 sibling `best.pt`，用于延续 best-selection/early-stopping 状态。
优先复制完整 `checkpoints/` 目录或直接传 run directory。单独的 best checkpoint 会由
`metadata.checkpoint_kind` 识别，不依赖文件名。sibling best 还必须匹配所选 checkpoint
记录的 resolved config、extension provenance、best epoch、metric、monitor 和 mode；若旧
epoch 旁的 mutable `best.pt` 已被后续 epoch 覆盖，当前格式无法重建被覆盖的历史 best，
恢复会明确拒绝。成功恢复时会验证 state topology，再用当前 config/provenance 将 best 原子
物化到新 run，因此可以独立移动或删除父 run。

Strict resume 还要求合法的 `epoch`、`global_step` 和 RNG snapshot；缺失或损坏会在修改训练
资产前拒绝。只有 strict training resume 会从 checkpoint 恢复 RNG snapshot。普通
checkpoint 读取不修改全局 RNG；sampling 不恢复该 snapshot，但会按 `sampling.seed`（为
`null` 时使用 `experiment.seed`）重置 Python、NumPy 与 Torch 全局 RNG。strict resume
恢复适用的 CPU/CUDA/MPS 全局 RNG；早期 v8 checkpoint 缺少 MPS RNG 字段时会警告，并且
无法保证该随机流精确衔接。跨 backend 或不同 CUDA topology 不保证逐位复现。
DataBuilder/DataLoader/Sampler/worker 的运行态不进入 checkpoint。内置图像 recipe 会从
seed 与 epoch 重建 shuffle/batch 索引，但 worker 中的随机 crop/flip state 不恢复，因此
这类增强不保证逐位连续。自定义随机 loader 应由 seed 与 epoch 确定，并按需实现
`set_epoch(epoch)`。

### MPS 上启用 FID 后运行在 CPU

这是预期行为。FID 的距离计算需要 MPS 不支持的 double-precision linear algebra，因此
reference provider 会在 MPS 运行中把 FID feature accumulation 和 compute 放到 CPU。
KID 不使用这条 fallback，仍保留配置的 runtime device。CPU transfer 可能增加 diagnostic
耗时，但不会改变训练资产所在设备。

### 完整外部 sampling config 加载 checkpoint state 失败

同时传完整 `--config` 与 `--checkpoint` 时，外部 config 整体权威，checkpoint 只提供
state；核心不会先比较两份完整配置。可以自由改变 sampling 数量、shape、Builder、Sampler、
solver、trajectory、writers 和 raw/EMA 选择。primary model 始终必须满足 checkpoint
state contract；若外部 config 选择 Process，该 Process 也必须严格加载对应 state。
完整外部 config 还可以明确写 `process: null`，用兼容的 direct-transform Builder 丢弃
checkpoint 中未使用的 Process state。除此之外的 missing/unexpected key 或 shape mismatch
应通过修改外部组件配置解决，而不是期待自动 merge。

只想改变 sampling 时，可以提供仅含完整 `sampling` 段与可选 `extensions` 的 lightweight
overlay。`extensions: {}` 保留 checkpoint selection；只有明确写出 `extensions.plugins`
才完整替换列表，不与 checkpoint 插件追加。

### 预期 EMA 采样但 manifest 显示 raw

同时检查 `ema.enabled`、`ema.use_for_sampling` 和 checkpoint 是否包含 EMA state。
旧 checkpoint 或训练期间禁用 EMA 时不能事后恢复 EMA 权重。

### trajectory 不生成

对 `standard_denoising` 设置
`sampling.builder.params.trajectory.enabled: true` 和正整数 `every_steps`。所有 Sampler
共用 observer 契约，不再提供 sampler-specific trajectory 方法。还需声明能保存
trajectory 的 writer，例如 `tensor` 或 `image`。

### 大样本或 trajectory 采样时 CPU OOM

当前 SamplingBuilder 必须先返回完整 `SamplingOutput`，runtime 才会调用
writer。因此所有 sample batch 和已保留 trajectory 会在 writer 开始前同时存在。
内置 Standard Builder 将 writer-ready Tensor 转存到 CPU；自定义 Builder 若返回
device Tensor，则还会持续占用 accelerator。对内置路径，减小 `batch_size` 只能
减少单次 accelerator 工作集，不会减少固定 `num_samples` 对应的最终 CPU
output。

在当前受支持的 lifecycle 内，可以：

- 降低 `num_samples`；
- 对主运行关闭 trajectory，另建小样本 preview；
- 增大 `every_steps`，减少保留的 observation；
- 让任务 Builder 先完成指标计算，再只把 writer 真正需要的场或通道
  放入 `SamplingOutput`；
- 对 Physics AI 预览使用独立配置，限制 `num_samples <= 8`、
  `every_steps >= 10` 且 accepted steps 不超过 40。

仅注册一个新 writer 不会把当前 lifecycle 变成 streaming；writer 看到数据时，
Builder 的全部 output 已经物化。全量 dense trajectory 不在当前容量支持
边界内。若任务确实要求该能力，需要先设计新的增量 sampling/writer
lifecycle，不要让 Builder 绕过 writer 直接写 artifact。详细公式和 DFSR
边界见 [Sampling artifact 容量](sampling-capacity.md)。文档中的
RSS 和编码峰值只能视为对应主机/backend 的证据，不是跨平台保证。

### `sampling.shape is required`

`standard_denoising` 需要固定 shape。把单样本形状（不含 batch 维）写入
`sampling.shape`。该值与 DataBuilder 独立，外部 sampling 配置可以覆盖它。自定义
SamplingBuilder 可以在 shape 为 null 时构造自己的 initial state。

### image writer 报 NCHW 或通道错误

只有 `image` writer 要求 rank 4 batch 且通道数为 1 或 3。物理场、序列或其他
Tensor 只声明 `tensor` writer，或注册领域 writer。

### `checkpoint format version ... is unsupported`

当前 checkpoint writer 只生成 v9，分别保存 primary model、可选 Process/Objective、
可选 EMA model、带 concrete class identity 的 optimizer/scheduler，以及按稳定名称组织的
training assets。runtime 只额外接受满足
[受限迁移规则](compatibility-and-migration.md#v8-到-v9-的受限迁移)的 legacy v8；
v7 及更早格式会在修改 runtime state 前失败。请用当前代码重新训练或重新生成 checkpoint。

### 蒸馏 resume 找不到 teacher bootstrap state

TrainingBuilder 在核心加载 checkpoint 之前必须先构造结构兼容的完整 TrainingPlan。因此
strict resume 仍要求项目配置中的 teacher bootstrap 文件可读取；它只是构造资源，文件中
的值随后会被 checkpoint 的同名 `training_assets_state_dict` 覆盖。不要根据 output path
或文件是否存在猜测 fresh/resume。checkpoint-only sampling 不构造 TrainingBuilder；若它
仍读取 teacher 文件，说明项目 SamplingBuilder 错误地依赖了训练资产。

### Physics reconstruction 的 DDIM 首坐标不匹配

partial-noised initial state 使用 public state time `t` 时，显式 DDIM schedule 的第一个
坐标也必须是 `t`，最后一个坐标必须是 clean time `0`。例如 `t=240, r=30` 使用
`[240, 232, ..., 8, 0]`，而不是先在 `alpha_bar[239]` 加噪后从较小 source state 开始。
参考项目的 lifecycle observer 会在写 artifact 前拒绝这种错位。

### checkpoint state 不是 weights-only 安全值

当前 v9 保存前递归拒绝 extension/custom class、任意 pickle object、custom Tensor
subclass 和其他 `torch.load(..., weights_only=True)` 默认值域之外的对象；同一约束也适用
于可迁移的 legacy v8。自定义 module、optimizer 或 scheduler 的 extra state 应转换为
Tensor、primitive、list/tuple/dict 等普通 container；不要添加 `safe_globals` 或依赖导入
extension class 才能读取的 pickle 对象。异常中的 state path 会定位具体非法值。

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
