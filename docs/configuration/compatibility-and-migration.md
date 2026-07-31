# Checkpoint、配置权威与可移植性

本页描述当前发布格式中 config、checkpoint、extension 代码和运行环境之间的边界。
Stochaflow 不把 checkpoint 当作源码或环境快照，也不会静默猜测其他 checkpoint 格式的
语义。

## Data artifact schema v2 是 breaking boundary

当前 `DataArtifactIdentity`、`DataArtifactBindings`、manifest、locator 和 cache layout
只接受 schema v2。框架有意不提供旧格式 reader、alias、dual lookup 或 cache/checkpoint
migration：

- managed/referenced 现在共用一个 `DataArtifact` runtime handle 与
  `DataArtifactStore` lifecycle；
- cache 位于 `<cache_root>/data-artifacts/v2/...`，旧目录不会被发现或自动删除；
- 升级后应以 `policy: ensure` 重新 materialize AFHQ、Torchvision、image-folder、
  paired-folder 与 Physics 输入；
- 保存旧 artifact binding 的 checkpoint 不能 strict resume，应启动新 run；
- `require` 不会将旧 cache 转换为新格式，也不会产生任何修复写入。

这是数据 artifact 格式的断代，不改变 `DataSource → DataArtifact → DataBuilder →
DataLoaders` 的职责边界。完全 synthetic、没有外部 artifact binding 的 recipe 不受该
格式迁移影响。

## Canonical ADM topology 是 breaking boundary

`adm_unet` registry identity 现在直接表示 canonical ADM input/output block graph：
initial projection、每个 encoder ResBlock 和每个 downsample 都保存 skip；decoder
每个 resolution 使用 `num_res_blocks + 1` 个 ResBlock，并逐 block 消费 skip。attention
是 GroupNorm/QKV/output projection/residual block，`attention_resolutions` 填写实际
spatial resolution。

旧实现的 stage-level skip graph 和 Spatial Transformer 没有 compatibility mode。以下
配置字段已删除并会被拒绝：

- `transformer_depths`、`middle_transformer_depth`、`attention_head_dim`；
- 可配置的 `time_embedding_dim`、`scale_shift_norm`、`residual_resampling`；
- `zero_init_residual` 与 `zero_init_output`。

使用新字段 `input_size`、`attention_resolutions` 和
`attention_head_channels`。scale-shift norm、residual up/down、residual/output
zero-init 以及 `4 * base_channels` time embedding 都是 `adm_unet` 定义的一部分，不再
作为 recipe 开关。

旧 ADM 的 raw、EMA 与 optimizer state 具有不同 key/shape/topology。框架不会 partial
load、映射 key、转换 state、保留 legacy model name 或自动改写 resolved config；strict
state validation 会 fail closed。升级时必须 fresh train，并从新 checkpoint 执行 resume
或 sampling。拓扑修复前发布的 AFHQ 指标与样本不能归因给 corrected ADM 或 P2。

## Checkpoint v11

当前 writer 和 runtime 只接受 `format_version: 11`；不读取或迁移 v8/v9/v10。旧
checkpoint 不能 strict resume，也不能作为 `stochaflow sample` 的输入，应由新运行重新
生成。手工改写 `format_version` 不能补齐所需 schema，也不是受支持的迁移。payload 通过
`torch.load(..., weights_only=True)` 读取，并递归限制为 Tensor/Parameter、primitive 和
普通 `dict`、`OrderedDict`、`list`、`tuple` 等 data-only 值。

训练 checkpoint 保存：

- format version、epoch/global step、resolved config、canonical epoch metrics 与
  `metadata.metric_sources`；
- primary model，以及存在时的 Process、Objective 和按名称声明的 managed auxiliary
  module state；
- optimizer/scheduler 的 concrete class identity 与 state；
- EMA runtime shadows 与可直接推理的 EMA model projection；
- `precision_kind`，以及仅在 `fp16-mixed` 时存在的 GradScaler class/state；
- 始终存在的 typed `inference_asset_descriptors` mapping；没有外部推理资产时为 `{}`；
- 始终存在的 `inference_recipe`；`null` 表示不支持 checkpoint-backed inference，
  否则严格保存 `{schema_version: 1, name, contract}`；
- Python、NumPy、Torch CPU 及可用 CUDA/MPS 的 epoch-boundary RNG snapshot；
- 完整 `metadata.training_loop`，包括是否启用 best tracking、monitor
  `metric/mode/missing/min_delta`、patience、最佳值，以及按 observation 计数的
  early-stopping state；
- extension provenance、lineage、`selected_components` 和可选 data artifact bindings。

每个 non-empty inference asset descriptor 将一个 slot 一对一绑定到
`training_assets_state_dict` 中的 managed auxiliary asset，并固定
`persistence: embedded_state`。descriptor 的 `declaration` 是自包含的
reconstruction-only component config；训练时下载地址、bootstrap path 或其他
acquisition identity 不能作为 sampling reconstruction 输入。多个 slot 指向同一 training
asset 会被拒绝。空 descriptor checkpoint 的 wire shape 和 pixel inference 行为不变。

v11 恢复是事务性的。manager 先验证完整 header、precision/scaler topology、inference
descriptors、fixed inference recipe、module key/shape/dtype/layout、optimizer parameter
groups、scheduler、EMA 与可选资产拓扑，再加载 state；后期 load hook 或任一资产失败时，
已触及的 runtime 对象会回滚到恢复前快照。strict resume 因而是完整恢复，不是
weights-only warm start。

strict resume 还会在修改 runtime 前解析完整 epoch metric snapshot、逐 key source
metadata 与 training-loop state。恢复后的 best-tracking policy、missing policy 和
patience 必须与继续运行时完全一致；不能通过 resume 把已保存的 observation counter
重解释为另一套 early-stopping 规则。

可选资产按“存在性 + state”严格配对：runtime 有 Process/Objective 时 checkpoint 必须有
对应 state，runtime 没有时 payload 也不能含该 key。辅助资产名称、
optimizer/scheduler class、EMA topology、precision 和 accumulation 同样必须匹配。
data-aware 训练还会在构建 sibling run 和恢复任何训练资产前，重新 materialize 并比较
checkpoint 的完整 `DataArtifactBindings`。

### Canonical metric key 是 breaking boundary

v11 checkpoint、logger、history 和 monitor 只使用以下 canonical epoch key：

- phase loss：`train/loss`、`valid/loss`、`test/loss`；
- stateful phase metric：
  `train|valid|test/metrics/<id>[/<subkey>]`；
- verified diagnostic metric：
  `diagnostics/<diagnostic-id>/<metric...>`；
- runtime observation：`system/<scope>/<metric...>`，不能用于模型选择。

旧 `train_loss`、`valid_loss` 及其他 underscore alias 没有 reader、双写或迁移路径。
配置加载和 checkpoint snapshot 解析都会拒绝非 canonical key；不要通过修改 JSONL、
YAML 或 checkpoint metadata 尝试混用旧名称。test-role metric 即使 key 合法，也不能控制
best checkpoint 或 early stopping。

### Diagnostic monitor 的 missing 语义

`trainer.early_stopping.missing` 支持 `error` 与 `skip`：

- `error` 是默认值；本 epoch 找不到 monitor 时立即失败；
- `skip` 只允许 `diagnostics/...` monitor，并且只在已验证 source 按 cadence **尚未到期**
  时跳过本次选择；
- source 已到期但 callback 没有返回被监控 key、返回失败、source id 不匹配或 metric
  非有限值时仍然失败，不能用上一次值、零或 `NaN` 代替；
- 整个 fit 没有产生任何 monitor observation 时失败，避免一次没有真正评估的训练被标记为
  成功；
- patience 按“已经观察到但没有改进”的次数累计；cadence skip 不增加 wait counter。

diagnostic 只有 composition 验证过的 validation source 才有 selection 资格。checkpoint
同时保存其 `data_role=validation`、`selection_eligible=true` 与
`protocol_id=sha256:<digest>`；external sampler statistics、耗时、artifact 以及任何
test-role 结果都不能成为 monitor。

### Gaussian inference recipe

新构建的所有 Gaussian training recipe 都在 checkpoint fixed contract 中同时冻结
`prediction_type` 与 `variance: {mode: fixed|learned_range}`。即使使用兼容默认值
`fixed`，该字段也会显式保存；sample request 不能覆盖它。

当前 v11 writer 总是显式保存该字段。所有 v10 Gaussian checkpoint 会先因 format
version 失败；即使其 model state 看似相同，也不能通过手工补字段或改版本升级。对 ADM
checkpoint 还同时存在上一节更强的 state key/shape/topology 不兼容。

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

### v10 到 v11 是 breaking migration

v11 把 canonical metric snapshot、逐结果 source metadata、完整 monitor policy 和
observation-based early-stopping counter 固化为 strict-resume schema。v10 没有这些完整
事实，框架无法可靠判断旧 best 值来自 train、validation、test 还是 external diagnostic，
也无法把 epoch wait counter 无歧义地转换成 observation counter。因此没有 v10 reader、
alias、dual write 或自动迁移；升级方式是以 v11 启动 fresh run。

更早的 v10 曾把训练时确定的 inference composition 从用户可改写的 sampling 配置中移出。
v11 保留该 authority boundary，但旧 sampling 文件仍不能直接使用：

旧 sampling 文件也不再有效：

| 旧字段/行为 | 当前 v11 替换 |
| --- | --- |
| `sampling.builder.name` | 由 `TrainingPlan.inference_recipe.name` 固化进 checkpoint |
| `sampling.builder.params.prediction_type` 等训练语义 | `inference_recipe.contract`，request 不可覆盖 |
| `sampling.builder.params.sampler` | `sampling.sampler` |
| 其他 Builder 可调参数 | `sampling.options` |
| sampling-only 整段 replacement | partial sample request；普通字段继承 checkpoint |
| 完整外部 config + checkpoint | 不支持；`sample` 必须以显式 checkpoint 为唯一 base |
| `extensions.plugins` 替换 selection | 只追加，不得删除 checkpoint-required plugin |

### 不在 checkpoint 中的状态

v11 不保存：

- extension 的 Python class、源码、wheel、依赖环境或 lockfile；
- DataBuilder/TrainingBuilder/Strategy/SamplingBuilder 实例；
- Dataset、DataLoader、iterator、worker、PyTorch data sampler 或 partition runtime
  state；
- epoch 中间尚未 step 的 gradients、accumulation window 或 DataLoader cursor；
- epoch 中间的 Metric state；只保存完成 epoch 的 canonical scalar snapshot 与 source
  metadata；
- Sampler、Observer、solver history、sampling trajectory 等临时采样状态；
- TrainingDiagnostic/ExperimentLogger 实例、diagnostic 自有 cache/counter、打开的日志
  文件或 TensorBoard writer/event 文件；best/early-stopping 的 monitor observation
  counters 则属于上述 `metadata.training_loop`；
- 用户私有 generator、数据集、网络资源或输出目录内容。

CPU/CUDA/MPS RNG snapshot 只覆盖相应全局 generator，不扩展 DataLoader worker 或用户
私有 generator 的持久化边界。需要 epoch-boundary 逐 batch 重建的 DataBuilder 应使用
epoch-aware sampler 和 stateless `(seed, epoch, sample identity)` augmentation。

`sample` 使用单独的 inference view：它只保留 model、可选 EMA、可选 Process、按 slot
保存的 inference asset descriptors、按 training asset name 保存且被 descriptor 引用的
embedded state、fixed recipe 和必要 metadata；tensor storage 不因 projection 再复制。
Objective、optimizer、scheduler、teacher 和其他 training-only assets 不进入 view。
checkpoint config 始终保留 Process 声明与 state
配对；sample request 不能删除或替换 model、Process、TrainingBuilder 或 recipe。实际构建
的 model/Process 仍必须严格加载被复用的 state。

extension plugin 激活后，SamplingBuilder 可通过 `InferenceAssetProvider.get(slot,
expected_capability_role=...)` 请求资产。provider 在 factory 调用前拒绝未知 slot 或错误
role，随后通过已激活的 model Registry 延迟构造，在 CPU 上校验 exact
key/shape/dtype/layout 并 strict-load，最后迁移到 sampling device、设为 eval 并缓存。
未请求的合法资产 constructor count 保持为零；失败不缓存。role 只证明语义身份，实际
窄行为 capability 必须由具体 SamplingBuilder 再验证。sample request 不能覆盖
descriptor、declaration、role 或 embedded state。

## Config authority

每个 workflow 先确定唯一 base config，再应用该 workflow 明确允许的覆盖：

- 新训练以 `--config` 指向的完整配置为 base；
- strict resume 以 checkpoint 内的 config 为 base，`--config` 与 `--resume` 互斥；
  可选 `--observability-config` 只允许原子替换 `diagnostics`，以及逐字段替换显式声明的
  `logging` 字段；
- checkpoint-backed inference 始终要求显式 `--checkpoint`，以其中的 config、state 和
  fixed `inference_recipe` 为 base；
- 可选 `--config` 只能是 partial sample request，顶层只允许 `sampling` 与 optional
  `extensions`；sampling CLI runtime flags 最后覆盖 device/output 等调用事实。

request 可改变 shape、数量、batch、seed、Sampler、writers 和 recipe 公开的 options。
`options` 浅合并，Sampler/writers 原子替换；Builder identity 和 fixed contract 不可修改。
Gaussian checkpoint 的 fixed contract 包含 `prediction_type` 和 `variance.mode`：
sample request 不能把 fixed checkpoint 改成 learned-range，反之亦然。P2 的
`loss_weighting` 与 learned-range 的 `variance.loss` 是 training/resume facts，保存在完整
resolved config 中；它们不是 sampling option。`ddpm.num_inference_steps` 或显式 schedule
只是 request-time solver protocol，不能改变 checkpoint 的 prediction/variance contract。
resume observability config 也只能改变没有恢复状态的监控表面，不能改变 extension
selection，也不能引入 checkpoint 未选择的 diagnostic provider module。它的有效配置和
provenance 会写入新兄弟 run 及其 checkpoint，旧 logger/event 文件不会续写。除此之外，
resume 不接受任意模型、训练资产或 optimizer 替换；需要改变它们时应启动新的训练
workflow。

### Gaussian weighting 的恢复与开发期配置切换

`GaussianSimpleLossWeighting` 是由配置构造的 family policy，不是 managed
`nn.Module`，核心不会为它保存或恢复独立 `state_dict`。policy 的 `name` 与 `params`
只保存在 checkpoint 的完整 resolved config；`selected_components` 不展开
`training.params`，不能代替该配置。

strict resume 先以 `EXACT` policy 验证 checkpoint 保存的 extension provenance，再激活
插件并通过 Gaussian family registry 重建 weighting。第三方 policy 的代码身份因此由
所选 entry point 的 name、distribution、version 和 target 约束；缺失插件或
name/distribution/target identity 不匹配会在训练组件构造前失败。sampling request 和
observability overlay 都不能替换 weighting。

开发期 flat P2 declaration：

```yaml
loss_weighting: {name: p2, k: 1.0, gamma: 1.0}
```

不受支持。新格式必须显式使用：

```yaml
loss_weighting:
  name: p2
  params:
    k: 1.0
    gamma: 1.0
```

框架不提供 alias、弃用期、config/checkpoint 转换器或参数猜测；旧 declaration 在 Builder
组合时 fail closed。这个配置切换不改变 checkpoint container format，当前 writer 仍是
v11。

终端进度显示不属于训练状态。strict resume 可用互斥的 `--progress` 与
`--no-progress` 覆盖 checkpoint 保存的 `trainer.show_progress`；两者都不指定时继承
checkpoint。最终生效值和 CLI 覆盖意图分别记录在新 run 的 config 与 runtime options
中，不改变模型、optimizer、RNG 或数据 artifact 恢复语义。

## `selected_components` 的含义

训练 manifest、checkpoint metadata 和 sampling manifest 使用同一
`selected_components` schema 与纯函数，分别从各自的最终 typed config 生成。一次训练的
run manifest 与其 checkpoint metadata 保存相同值；sample request 可以合法改变
Sampler/options/writers，但 `selected_components.sampling_recipe` 始终来自 checkpoint。
摘要显式保留可选 role 的 `null` 和 writers/loggers/diagnostics 的声明顺序。

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
6. 先用小 batch 和有限 step 做 data/model/state-load smoke，再运行完整作业。inference
   不恢复 checkpoint RNG，而是按 `sampling.seed`（或 experiment seed）重新初始化。
7. 在目标主机重新评估 RAM、accelerator memory、临时磁盘和 artifact 大小。当前 sampling
   是整体物化 contract；详细公式、基准方法和 trajectory 限制见
   [Sampling artifact 容量](sampling-capacity.md)。

checkpoint 可移植表示“在满足上述显式依赖与 capability contract 的环境中可以重建并加载
状态”，不表示 extension 源码、数据、硬件或数值执行已经被冻结。
