# Checkpoint、配置权威与可移植性

本页描述当前发布格式中 config、checkpoint、extension 代码和运行环境之间的边界。
Stochaflow 不把 checkpoint 当作源码或环境快照，也不会静默猜测其他 checkpoint 格式的
语义。

## Checkpoint v8

当前 runtime 只接受 `format_version: 8`。其他版本在构建或恢复 runtime state 前直接失败；
没有内置升级、降级或旧 payload 推断。若手中有其他格式，只能重新训练，或在项目外编写
一次性、经过任务所有者验证的转换工具；这种转换不属于 Stochaflow 的兼容保证。

v8 使用 `torch.load(..., weights_only=True)`，并递归限制 payload 为精确 Tensor/
Parameter、primitive 和普通 `dict`、`OrderedDict`、`list`、`tuple`。训练 checkpoint
保存：

- format version、epoch/global step、resolved config、metrics 与 metadata；
- primary model state；
- 存在时的 Process、Objective、EMA model 与 EMA runtime state；
- optimizer/scheduler 的 concrete class identity 与 state；
- 按名称保存的 managed auxiliary module state；
- Python、NumPy、Torch CPU 及可用 CUDA/MPS 的 epoch-boundary RNG snapshot；
- extension entry-point provenance、version acceptance、lineage 和
  `selected_components` 等审计信息。

v8 不保存：

- extension 的 Python class、源码、wheel、依赖环境或 lockfile；
- DataBuilder/TrainingBuilder/Strategy/SamplingBuilder 实例；
- Dataset、DataLoader、iterator、worker、PyTorch data sampler 或 partition runtime state；
- Sampler、Observer、solver history、sampling trajectory 等临时采样状态；
- 用户私有 generator 或未声明为 managed training asset 的对象；
- 数据集、网络资源、输出目录内容或相对路径所指向的文件。

当前 v8 写入 MPS RNG state。早期实现生成、仍标记为 v8 的 checkpoint 可能没有这个
字段；它们在 MPS strict resume 时会发出警告并继续加载，但无法保证 MPS 随机流与未中断
运行精确衔接。CPU/CUDA/MPS RNG snapshot 都只覆盖相应的全局 generator，不扩展
DataLoader worker 或用户私有 generator 的持久化边界。

在 `train --resume` 的完整训练恢复中，可选资产按“存在性 + state”严格配对：runtime
有 Process/Objective 时 checkpoint 必须有对应 state，runtime 没有时 payload 也不能含
该 key。辅助资产名称、optimizer/scheduler class 和可加载 state 同样必须匹配。因此
strict resume 是完整恢复，不是 weights-only warm start。

sampling 使用单独的 inference view：它只保留 model、可选 EMA、可选 Process 以及必要
metadata，不构建或加载 Objective、optimizer、scheduler 和 managed training assets。
checkpoint config 或 sampling-only overlay 会保留 checkpoint 的 Process 声明和 state
配对；完整外部 sampling config 可以声明 `process: null` 并忽略 checkpoint 中未使用的
Process state。无论哪条路径，实际构建的 model/Process 仍必须能严格加载所复用的 state。

## Config authority

每个 workflow 先确定唯一 base config，再应用该 workflow 明确允许的覆盖：

- 新训练以 `--config` 指向的完整配置为 base；
- strict resume 以 checkpoint 内的 config 为 base，`--config` 与 `--resume` 互斥；
- checkpoint-only sampling 默认使用 checkpoint config；
- sampling 可以改用一份完整外部 config，或在 checkpoint config 上应用
  sampling-only overlay；sampling CLI flags 最后覆盖本次调用允许的字段。

因此，改变采样数量、shape、Builder、Sampler 私有参数或 writer 是 sampling workflow 的
正常能力，不构成训练配置“冲突”。相反，resume 不接受任意模型、训练资产或 optimizer
替换；需要改变它们时应启动新的训练 workflow。

## `selected_components` 的含义

训练 manifest、checkpoint metadata 和 sampling manifest 使用同一
`selected_components` schema 与纯函数，分别从各自的最终 typed config 生成。一次训练的
run manifest 与其 checkpoint metadata 保存相同值；sampling overlay 或完整外部 sampling
config 可以合法改变 Builder/writers，因此 sampling manifest 的值可以不同。摘要显式保留
可选 role 的 `null` 和 writers/loggers/diagnostics 的声明顺序。

该字段仅用于快速审计：

- 不遍历 Builder、Process 或其他 component 的私有 `params`；
- 不记录 sampler、noise schedule、teacher、source、condition 等嵌套拓扑；
- 不冻结 class/source，也不证明当前 environment 与训练时逐位相同；
- 不参与 Registry dispatch、插件归属推断或 Process/Sampler compatibility 判断。

完整 checkpoint config 才是重建权威；`selected_components` 不能替代它。

## Family 与 capability 边界

Registry 只校验注册名称和声明的根类型。数学或任务兼容性在拥有完整组合信息的窄边界验证：

- DataBuilder 保证自己返回合法 `DataLoaders`；
- TrainingBuilder/Strategy 验证 batch、模型、Objective、Process 与辅助资产的协作；
- SamplingBuilder 验证模型适配、initial state 与算法 family；
- family-specific Sampler 在调用时要求自己的 Dynamics capability，例如 DDPM/DDIM 要求
  Gaussian denoising Dynamics。

核心没有 `process × sampler` 名称矩阵，`GenerativeDynamics` 也没有 universal
`predict/step/drift` API。新算法 family 应定义自己的窄 Process/Dynamics/Sampler 契约并由
自己的 Builder 组合；能直接完成生成变换的方法不需要虚构 Process 或 Sampler。

插件 provenance 能检测 entry-point name、distribution、version 和 target 的变化，但不能
检测“editable install 在版本号不变时修改源码”。版本不同可以由 CLI 警告、询问或显式
force；name/distribution/target identity 不匹配会失败。即使 provenance 相同，extension
实现变化仍可能因 state shape、资产拓扑或行为改变而不兼容，这不由核心自动迁移。

## 跨机器检查清单

移动 checkpoint 或 run directory 前后，逐项确认：

1. 在目标环境安装所需 Stochaflow 和 extension distributions；extension 必须向
   CLI 所用 Python 声明相同 entry point。缺失插件错误会显示当前 `sys.executable`。
2. 使用项目自己的 lockfile/environment specification 固定 Python、PyTorch、extension
   及其原生依赖。checkpoint 不承担环境快照职责。
3. strict resume 优先复制整个 run directory。latest/epoch checkpoint 还依赖同一
   `checkpoints/` 目录中的、经过 lineage 校验的 `best.pt`；单独的
   `metadata.checkpoint_kind: best` checkpoint 才可独立恢复。
4. 同步数据、模型外部资产和配置中引用的文件。相对 data/output 路径与 Builder 私有
   path 都以目标进程启动 cwd 解释；必要时从项目根运行并显式覆盖 output dir。
5. 核对 device/backend。strict resume 若恢复 CUDA/MPS RNG，目标 backend 必须可用；
   CUDA device count 也必须兼容。跨 device、PyTorch 版本或第三方 kernel 不保证逐位一致。
6. 先用小 batch 和有限 step 做 data/model/state-load smoke，再运行完整作业。sampling
   不恢复 checkpoint RNG，而是按 `sampling.seed`（或 experiment seed）重新初始化。
7. 在目标主机重新评估 RAM、accelerator memory、临时磁盘和 artifact 大小。当前 sampling
   是整体物化 contract；详细公式、基准方法和 trajectory 限制见
   [Sampling artifact 容量](sampling-capacity.md)。

checkpoint 可移植表示“在满足上述显式依赖与 capability contract 的环境中可以重建并加载
状态”，不表示 extension 源码、数据、硬件或数值执行已经被冻结。
