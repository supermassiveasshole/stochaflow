# Checkpoint、配置权威与可移植性

本页描述当前发布格式中 config、checkpoint、extension 代码和运行环境之间的边界。
Stochaflow 不把 checkpoint 当作源码或环境快照，也不会静默猜测其他 checkpoint 格式的
语义。

## Checkpoint v9

当前 writer 只生成 `format_version: 9`。runtime 接受 v9，也接受满足下述严格迁移规则的
v8；其他版本在修改任何 runtime state 前失败。payload 通过
`torch.load(..., weights_only=True)` 读取，并递归限制为 Tensor/Parameter、primitive 和
普通 `dict`、`OrderedDict`、`list`、`tuple` 等 data-only 值。

训练 checkpoint 保存：

- format version、epoch/global step、resolved config、metrics 与 metadata；
- primary model，以及存在时的 Process、Objective 和按名称声明的 managed auxiliary
  module state；
- optimizer/scheduler 的 concrete class identity 与 state；
- EMA runtime shadows 与可直接推理的 EMA model projection；
- `precision_kind`，以及仅在 `fp16-mixed` 时存在的 GradScaler class/state；
- 始终存在的 typed `inference_asset_descriptors` mapping；没有外部推理资产时为 `{}`；
- Python、NumPy、Torch CPU 及可用 CUDA/MPS 的 epoch-boundary RNG snapshot；
- extension provenance、lineage、`selected_components` 和可选 data artifact bindings。

v9 恢复是事务性的。manager 先验证完整 header、precision/scaler topology、inference
descriptors、module key/shape/dtype/layout、optimizer parameter groups、scheduler、
EMA 与可选资产拓扑，再加载 state；后期 load hook 或任一资产失败时，已触及的 runtime
对象会回滚到恢复前快照。strict resume 因而是完整恢复，不是 weights-only warm start。

可选资产按“存在性 + state”严格配对：runtime 有 Process/Objective 时 checkpoint 必须有
对应 state，runtime 没有时 payload 也不能含该 key。辅助资产名称、
optimizer/scheduler class、EMA topology、precision 和 accumulation 同样必须匹配。
data-aware 训练还会在构建 sibling run 和恢复任何训练资产前，重新 materialize 并比较
checkpoint 的完整 `DataArtifactBindings`。

### Precision 与 accumulation

`trainer.precision` 支持：

| precision | CPU | CUDA | MPS |
| --- | --- | --- | --- |
| `fp32` | 支持 | 支持 | 支持 |
| `bf16-mixed` | BF16 autocast | 需要可用 CUDA 与 BF16 capability | 不支持 |
| `fp16-mixed` | 不支持 | FP16 autocast + GradScaler | 不支持 |

不支持的组合在创建 run directory 前失败，不会自动 fallback。模型参数和标准 AdamW state
保持 FP32；mixed precision 只改变 forward/evaluation autocast 与必要的 gradient scaling。
FP32/BF16 checkpoint 不得含 scaler fields，FP16 checkpoint 必须同时含 scaler class 和
state。

`trainer.accumulate_grad_batches` 是正整数。Trainer 在每个固定窗口内累积 backward，
在实际窗口末尾执行 `unscale -> clip -> step/update`；epoch 末 partial window 按实际
micro-batch 数归一化。只有成功的 optimizer update 才推进 global step、step scheduler、
EMA 和 update-level diagnostics。precision、scaler state 和 accumulation 都属于 strict
resume config/state 边界，不能通过 observability overlay 改写。

### v8 到 v9 的受限迁移

v8 只在没有任何 v9-only header/config 字段时迁移。runtime 确定性补入：

```yaml
format_version: 9
precision_kind: fp32
inference_asset_descriptors: {}
config:
  trainer:
    precision: fp32
    accumulate_grad_batches: 1
```

v8 不能携带或推断 GradScaler state，也不能被解释为历史 FP16/BF16 或 accumulation
训练。若一个标记为 v8 的 payload 已含 precision、accumulation、scaler 或 descriptors
等 v9-only 字段，迁移直接拒绝，而不是猜测它们的来源。迁移只规范 header；其余资产、
class identity、RNG 和 data provenance 仍按当前 strict restore 规则验证。

### 不在 checkpoint 中的状态

v9 不保存：

- extension 的 Python class、源码、wheel、依赖环境或 lockfile；
- DataBuilder/TrainingBuilder/Strategy/SamplingBuilder 实例；
- Dataset、DataLoader、iterator、worker、PyTorch data sampler 或 partition runtime
  state；
- epoch 中间尚未 step 的 gradients、accumulation window 或 DataLoader cursor；
- Sampler、Observer、solver history、sampling trajectory 等临时采样状态；
- TrainingDiagnostic/ExperimentLogger 实例、diagnostic cache/counter、打开的日志文件或
  TensorBoard writer/event 文件；
- 用户私有 generator、数据集、网络资源或输出目录内容。

CPU/CUDA/MPS RNG snapshot 只覆盖相应全局 generator，不扩展 DataLoader worker 或用户
私有 generator 的持久化边界。需要 epoch-boundary 逐 batch 重建的 DataBuilder 应使用
epoch-aware sampler 和 stateless `(seed, epoch, sample identity)` augmentation。

sampling 使用单独的 inference view：它只保留 model、可选 EMA、可选 Process、声明的
inference assets 和必要 metadata，不构建或加载 Objective、optimizer、scheduler 或其他
training-only assets。checkpoint config 或 sampling-only overlay 会保留 checkpoint 的
Process 声明和 state 配对；完整外部 sampling config 可以声明 `process: null` 并忽略未
使用的 Process state。无论哪条路径，实际构建的 model/Process 仍必须严格加载被复用的
state。

## Config authority

每个 workflow 先确定唯一 base config，再应用该 workflow 明确允许的覆盖：

- 新训练以 `--config` 指向的完整配置为 base；
- strict resume 以 checkpoint 内的 config 为 base，`--config` 与 `--resume` 互斥；
  可选 `--observability-config` 只允许原子替换 `diagnostics`，以及逐字段替换显式声明的
  `logging` 字段；
- checkpoint-only sampling 默认使用 checkpoint config；
- sampling 可以改用一份完整外部 config，或在 checkpoint config 上应用
  sampling-only overlay；sampling CLI flags 最后覆盖本次调用允许的字段。

因此，改变采样数量、shape、Builder、Sampler 私有参数或 writer 是 sampling workflow 的
正常能力，不构成训练配置“冲突”。resume observability config 也只能改变没有恢复状态的
监控表面，不能改变 extension selection，也不能引入 checkpoint 未选择的 diagnostic
provider module。它的有效配置和 provenance 会写入新兄弟 run 及其 checkpoint，旧
logger/event 文件不会续写。除此之外，resume 不接受任意模型、训练资产或 optimizer
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
