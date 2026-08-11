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
  paired-folder 与扩展自有的外部输入；
- 保存旧 artifact binding 的 checkpoint 不能 strict resume，应启动新 run；
- `require` 不会将旧 cache 转换为新格式，也不会产生任何修复写入。

这是数据 artifact 格式的断代，不改变 `DataSource → DataArtifactStore → sealed
DataArtifact → DataBuilder → DataLoaders` 的职责边界。`DataArtifact` 只能由 Store
签发；其临时 runtime receipt 不属于 schema v2，也不会进入 checkpoint。完全 synthetic、
没有外部 artifact binding 的 recipe 不受该格式迁移影响。

自定义多 source Builder 需要通过 `DataBuilderContext.data_source_context()` 创建每个
逻辑 source 请求，再调用 `materialize_data_source()`。旧的 identity-only binding、直接在
Builder 中调用 Store，或把同一个 context 复用于后续/独立并发选择，都不提供当前 build
的正式 provenance。

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
或 sampling。拓扑修复前发布的 AFHQ 指标与样本不能归因给 corrected ADM。

## Checkpoint v12

当前 writer 和 runtime 只接受 `format_version: 12`；不读取或迁移 v8-v11。旧
checkpoint 不能 strict resume，也不能作为 `stochaflow sample` 的输入，应由新运行重新
生成。手工改写 `format_version` 不能补齐所需 schema，也不是受支持的迁移。payload 通过
`torch.load(..., weights_only=True)` 读取，并递归限制为 Tensor/Parameter、primitive 和
普通 `dict`、`OrderedDict`、`list`、`tuple` 等 data-only 值。

训练 checkpoint 保存：

- format version、epoch/global step、resolved config 与一个普通的 canonical epoch
  metrics mapping；
- primary model，以及存在时的 Process、Objective 和按名称声明的 managed auxiliary
  module state；
- optimizer/scheduler 的 concrete class identity 与 state；
- EMA runtime shadows 与可直接推理的 EMA model projection；
- `precision_kind`，以及仅在 `fp16-mixed` 时存在的
  `torch.amp.GradScaler("cuda")` class/state；旧 `torch.cuda.amp` class identity
  不属于当前 v12 contract；
- 始终存在的 typed `inference_asset_descriptors` mapping；没有外部推理资产时为 `{}`；
- 始终存在的 `inference_recipe`；`null` 表示不支持 checkpoint-backed inference，
  否则严格保存 `{schema_version: 1, name, contract}`；
- Python、NumPy、Torch CPU 及可用 CUDA/MPS 的 epoch-boundary RNG snapshot；
- 完整 `metadata.training_loop`，包括是否启用 best tracking、monitor
  `metric/mode/min_delta`、patience、最佳值，以及 early-stopping wait state；
- extension provenance、lineage、`selected_components` 和可选 data artifact bindings。

每个 non-empty inference asset descriptor 将一个 slot 一对一绑定到
`training_assets_state_dict` 中的 managed auxiliary asset，并固定
`persistence: embedded_state`。descriptor 的 `declaration` 是自包含的
reconstruction-only component config；训练时下载地址、bootstrap path 或其他
acquisition identity 不能作为 sampling reconstruction 输入。多个 slot 指向同一 training
asset 会被拒绝。空 descriptor checkpoint 的 wire shape 和 pixel inference 行为不变。

v12 恢复是事务性的。manager 先验证完整 header、precision/scaler topology、inference
descriptors、fixed inference recipe、module key/shape/dtype/layout、optimizer parameter
groups、scheduler、EMA 与可选资产拓扑，再加载 state；后期 load hook 或任一资产失败时，
已触及的 runtime 对象会回滚到恢复前快照。strict resume 因而是完整恢复，不是
weights-only warm start。

strict resume 还会在修改 runtime 前解析完整 epoch metrics mapping 与 training-loop
state。恢复后的 best-tracking policy 和 patience 必须与继续运行时完全一致；不能通过
resume 把已保存的 wait counter 重解释为另一套 early-stopping 规则。

可选资产按“存在性 + state”严格配对：runtime 有 Process/Objective 时 checkpoint 必须有
对应 state，runtime 没有时 payload 也不能含该 key。辅助资产名称、
optimizer/scheduler class、EMA topology、precision 和 accumulation 同样必须匹配。
data-aware 训练还会在构建 sibling run 和恢复任何训练资产前，重新 materialize 并比较
checkpoint 的完整 `DataArtifactBindings`。

### Canonical metric key 是 breaking boundary

训练 fit 完成 epoch 后，history、checkpoint 与 epoch logger 共享同一个 plain metrics
mapping，使用以下 canonical key：

- `train/loss` 与 `train/metrics/<id>[/<subkey>]`；
- 有 validation DataLoader 时的 `valid/loss` 与
  `valid/metrics/<id>[/<subkey>]`；
- runtime observation：`system/<scope>/<metric...>`，不能用于模型选择。

fit 后的 test evaluation 另外向 logger 写入 `test/loss` 与
`test/metrics/<id>[/<subkey>]`，不回写已经完成的训练 checkpoint。

diagnostic 可以另外向 logger 写入 `diagnostics/<diagnostic-id>/<metric...>`，并生成图片、
manifest 等 artifact；这些 observation 不进入 epoch history 或 checkpoint metrics。

旧 `train_loss`、`valid_loss` 及其他 underscore alias 没有 reader、双写或迁移路径。
当前 writer/logger 不生成这些 alias，monitor 配置也会拒绝它们。checkpoint metrics 不是
用户可编辑的迁移表面；不要通过修改 JSONL、YAML 或 checkpoint metadata 尝试混用旧名称。

### Validation-only 模型选择

best checkpoint 与 early stopping 的 monitor 只允许：

- `valid/loss`；
- `valid/metrics/<id>[/<subkey>]`，由 validation phase Metric 或声明该 exact key 的
  epoch-end validation Evaluation 产生。

train、test、system 与 `diagnostics/...` 指标都不能控制模型选择。普通 phase monitor 每个
epoch 都必须存在且 finite。若 monitor 来自 `trainer.validation_evaluation.metric_keys`，
则只在该 evaluator 的 absolute epoch cadence 到期时要求并消费新值；非到期 epoch 不复用
旧值、不保存新的 `best.pt`，也不推进 patience。到期运行缺失、非 finite、sample ID 重复或
strict completeness 不满足时立即失败。

普通 phase monitor 的 strict-resume state 是逐 epoch 的 dense history：checkpoint epoch 必须
等于 monitor observation 数，`best_epoch` 不能晚于 checkpoint，wait counter 必须等于二者
之差。目录恢复还要求较晚 candidate 单调延伸 observation、best epoch 与满足 mode/min-delta
的 best metric；缺失、倒退或相互矛盾的 sibling 会 fail closed。

live evaluator 的 profile digest、exact metric keys、cadence 和完整 interval/final result
history 属于 checkpoint strict-resume state。`metadata.training_loop.epoch_validation` 使用独立
`schema_version: 1`，并按 epoch 顺序保存每个 completed result 的 `epoch`、`global_step` 和
全部声明 metrics。last result、last evaluated epoch、last metrics 和 staged off-cadence final
都从这一个 `results` authority 派生，不再重复保存 summary 字段。resume 必须提供相同
identity，不能跳过任何已到期 interval observation；result epoch 严格递增、global step
单调不下降，selection、patience 和 stopping state 必须能从完整 metrics history 精确重放。

`include_final: true` 可以要求每个 staged target 的最后一个 epoch 额外执行一次，即使它不落
在 interval 上。这样的 result 留在完整 history 中；扩展后续 target 不会删除或重解释它。
旧的 unversioned summary-only epoch-validation state 缺少逐 observation metrics/global step，
因此不能 strict resume，也不会由 counter、best state 或 sibling checkpoint 猜测补齐。外层
checkpoint 仍是 v12：read-only inference、sampling 和 formal Evaluation 会投影掉 training
lifecycle state，因而仍可使用这类 checkpoint；只有 strict training resume fail closed。

没有 validation DataLoader 时，Trainer 默认关闭 best tracking，不创建或伪造 best
metric/epoch/checkpoint；显式请求 best tracking 或 early stopping 会在训练循环开始前失败。
该路径仍逐 epoch 保存 `latest.pt`；后续显式 `sample` 调用可以使用这个 final/latest
checkpoint，但它不是 best checkpoint。

TrainingDiagnostic 是只观测回调：它可以写 logger 与 artifact，epoch callback 必须返回
`None`，也不能修改收到的只读 epoch metrics。diagnostic 不参与 history、checkpoint、best
tracking、early stopping 或 strict-resume selection state。

### Gaussian inference recipe

新构建的所有 Gaussian training recipe 都在 checkpoint fixed contract 中同时冻结
`prediction_type` 与 `variance: {mode: fixed|learned_range}`。即使使用兼容默认值
`fixed`，该字段也会显式保存；sample config 不能覆盖它。

当前 v12 writer 总是显式保存该字段。所有 v11 及更早的 Gaussian checkpoint 会先因 format
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

v11 把 canonical plain epoch metrics、validation-only monitor policy 和完整
early-stopping state 固化为 strict-resume schema。v10 没有这些完整事实，框架无法可靠
恢复当前 selection state。因此没有 v10 reader、alias、dual write 或自动迁移；升级方式
是以 v11 启动 fresh run。

这是历史格式说明；当前 runtime 不读取 v10 或 v11 checkpoint。

### v11 到 v12 是 breaking migration

v12 将 mutable sample invocation 从训练配置和 checkpoint config 中彻底分离。它没有 v11
reader、alias、dual write 或自动迁移；升级方式是以当前训练 schema 启动 fresh run，再为
新 checkpoint 编写完整、独立的 sample config。

| v11/更早字段或行为 | v12 替换 |
| --- | --- |
| 更早的 `sampling.builder.name` | 由 `TrainingPlan.inference_recipe.name` 固化进 checkpoint |
| 更早的 `sampling.builder.params.prediction_type` 等训练语义 | `inference_recipe.contract`，sample config 不可覆盖 |
| 训练配置中的顶层 `sampling` | 删除；mutable inference 字段只写入独立顶层 `sample` |
| `sampling.run_after_training` | 删除；训练结束不自动采样 |
| `ema.use_for_sampling` | 删除；`sample.options.weights: auto` 按 EMA state 是否存在选择 |
| 可选 partial sample request | `--config` 必填，且 sampler/options/shape/数量/batch/seed/writers 必须完整 |
| `sampling.sampler/options/...` | `sample.sampler/options/...` |
| checkpoint defaults 与 request merge | 禁止；checkpoint state/recipe 与完整 sample config 保持平行权威 |
| `train --skip-final-sample` | 删除；因为训练 workflow 不再运行 final sample |
| `extensions.plugins` 替换 selection | 只追加，不得删除 checkpoint-required plugin |

当前 checkpoint-backed sampling 还要求 v12 subject 具有精确的正整数 `epoch` 和非负
`global_step`；runtime 对同一稳定 bytes snapshot 计算 SHA-256，并把这组 identity 写入
`resolved_sampling.yaml`。缺少 progress identity 的手工或旧 checkpoint 直接失败。项目处于
pre-1.0，本次变更不提供 fallback、字段猜测或旧 checkpoint 转换器；请用当前代码重新训练。

`SamplingBatch` 公共扩展契约现在要求显式正整数 `num_samples`，一次 output 的声明总数必须
等于完整 sample request。core 不从 task-private Tensor、record 或其他 payload 形状猜数量。
sampling 最终目录也改为 immutable no-replace bundle：writers 在 sibling staging 中完成，
manifest 使用相对 artifact path，然后一次性原子发布；已有目录不会继续写入或合并。

### 不在 checkpoint 中的状态

v12 不保存：

- extension 的 Python class、源码、wheel、依赖环境或 lockfile；
- DataBuilder/TrainingBuilder/Strategy/SamplingBuilder 实例；
- Dataset、DataLoader、iterator、worker、PyTorch data sampler 或 partition runtime
  state；
- epoch 中间尚未 step 的 gradients、accumulation window 或 DataLoader cursor；
- epoch 中间的 Metric state；只保存完成 epoch 的 canonical scalar mapping；
- Sampler、Observer、solver history、sampling trajectory 等临时采样状态；
- TrainingDiagnostic/ExperimentLogger 实例、diagnostic 自有 cache/counter、打开的日志
  文件或 TensorBoard writer/event 文件；best/early-stopping state 则属于上述
  `metadata.training_loop`；
- 用户私有 generator、数据集、网络资源或输出目录内容。

CPU/CUDA/MPS RNG snapshot 只覆盖相应全局 generator，不扩展 DataLoader worker 或用户
私有 generator 的持久化边界。需要 epoch-boundary 逐 batch 重建的 DataBuilder 应使用
epoch-aware sampler 和 stateless `(seed, epoch, sample identity)` augmentation。

`sample` 使用单独的 inference view：它只保留 model、可选 EMA、可选 Process、按 slot
保存的 inference asset descriptors、按 training asset name 保存且被 descriptor 引用的
embedded state、fixed recipe 和必要 metadata；tensor storage 不因 projection 再复制。
Objective、optimizer、scheduler、teacher 和其他 training-only assets 不进入 view。
checkpoint config 始终保留 Process 声明与 state
配对；sample config 不能删除或替换 model、Process、TrainingBuilder 或 recipe。实际构建
的 model/Process 仍必须严格加载被复用的 state。

extension plugin 激活后，SamplingBuilder 可通过 `InferenceAssetProvider.get(slot,
expected_capability_role=...)` 请求资产。provider 在 factory 调用前拒绝未知 slot 或错误
role，随后通过已激活的 model Registry 延迟构造，在 CPU 上校验 exact
key/shape/dtype/layout 并 strict-load，最后迁移到 sampling device、设为 eval 并缓存。
未请求的合法资产 constructor count 保持为零；失败不缓存。role 只证明语义身份，实际
窄行为 capability 必须由具体 SamplingBuilder 再验证。sample config 不能覆盖
descriptor、declaration、role 或 embedded state。

`evaluate` 复用同一安全 checkpoint inference projection，并只构造配置显式选择的 raw
或 EMA primary model。需要生成的 Builder 另外收到绑定这一个 pinned model/variant 的窄
sampling capability；它通过 shared SamplingBuilder execution seam 恢复 Process/assets 并
产生 writer-free output，不能再次选择权重。subject preflight 固定 checkpoint SHA-256、
format、epoch/global step、config、extension provenance、lineage 与 data artifact bindings；
optimizer、scheduler、GradScaler 和 training RNG 缺失不影响 evaluation。它不调用 strict
training restore，也不构造 TrainingPlan，源 checkpoint 不会被修改。

## Config authority

训练与恢复先确定唯一训练 config；sample 明确保留 checkpoint 与 invocation config 两条
平行权威：

- 新训练以 `--config` 指向的完整配置为 base；
- strict resume 以 checkpoint 内的 config 为 base，`--config` 与 `--resume` 互斥；
  可选 `--observability-config` 只允许原子替换 `diagnostics`，以及逐字段替换显式声明的
  `logging` 字段；
- checkpoint-backed inference 始终同时要求显式 `--checkpoint` 与 `--config`；checkpoint
  的 config/state/fixed `inference_recipe` 只负责重建并加载推理资产；
- 独立 config 顶层只允许完整 `sample` 与 optional `extensions`，负责本次 sampler、
  options、shape、数量、batch、seed 和 writers；它不与 checkpoint config 合并；
- sample CLI runtime flags 最后决定 device/output 等调用事实；
- checkpoint evaluation 以完整独立 evaluation config 为 authority；其中 subject 显式引用
  v12 checkpoint 并选择 raw/EMA，data 选择 checkpoint DataBuilder 的 validation/test，
  evaluation/metrics/protocol 定义本次任务组合和 strict expected count；
- evaluation CLI 只覆盖 device/output 与 extension-version acceptance，不提供独立
  checkpoint flag 或 arbitrary patch。

sample config 必须显式声明全部 mutable invocation 字段。Builder identity 和 fixed contract
不可修改，也不存在从 checkpoint defaults 继承或浅合并 options 的路径。
Gaussian checkpoint 的 fixed contract 包含 `prediction_type` 和 `variance.mode`：
sample config 不能把 fixed checkpoint 改成 learned-range，反之亦然。TrainingBuilder
identity 与参数是 training/resume facts，保存在完整 resolved config 中；它们不是
sampling option。
`ddpm.num_inference_steps` 或显式 schedule 只是 invocation-time solver protocol，不能改变
checkpoint 的 prediction/variance contract。
resume observability config 也只能改变没有恢复状态的监控表面，不能改变 extension
selection，也不能引入 checkpoint 未选择的 diagnostic provider module。它的有效配置和
provenance 会写入新兄弟 run 及其 checkpoint，旧 logger/event 文件不会续写。除此之外，
resume 不接受任意模型、训练资产或 optimizer 替换；需要改变它们时应启动新的训练
workflow。

### TrainingBuilder 与 validation Evaluation 的恢复边界

strict resume 以 checkpoint 保存的 `training.name` 和完整参数重建同一个
TrainingBuilder。第三方 recipe 必须注册自己的 namespaced
TrainingBuilder/TrainingStrategy；其代码身份继续由所选 entry point 的 name、
distribution、version 和 target 约束。sample config 和 observability overlay 都不能替换
训练算法、prediction/variance contract 或 validation Evaluation identity。

`trainer.validation_evaluation` 是训练选择策略的一部分，不是可随意替换的 diagnostic。
更改 Builder、MetricSpec、protocol、raw/EMA variant、expected count、metric keys 或
cadence 会改变 profile identity，必须启动 fresh run。只对一组已有 checkpoints 做离线
选择时，不修改其训练 state：逐个运行同一 standalone validation Evaluation，再比较其
结果即可。

终端进度显示不属于训练状态。strict resume 可用互斥的 `--progress` 与
`--no-progress` 覆盖 checkpoint 保存的 `trainer.show_progress`；两者都不指定时继承
checkpoint。最终生效值和 CLI 覆盖意图分别记录在新 run 的 config 与 runtime options
中，不改变模型、optimizer、RNG 或数据 artifact 恢复语义。

## `selected_components` 的含义

训练 manifest/checkpoint metadata 与 sampling/evaluation manifest 使用分别的
component/provenance 投影。一次训练的 run manifest 与其 checkpoint metadata 保存相同
training-owned 摘要；sampling manifest 只组合 checkpoint-owned model/Process/recipe 与
本次 sample config 选择的 Sampler/writers；evaluation result/manifest 记录 checkpoint
selected components 与本次 EvaluationBuilder/Metric declarations。各 authority 不会互相
伪造字段。
摘要显式保留可选 role 的 `null` 和 writers/loggers/diagnostics 的声明顺序。

该字段仅用于快速审计：

- 不遍历 Builder、Process 或其他 component 的私有 `params`；
- 不记录 sampler、noise schedule、teacher、source、condition 等嵌套拓扑；
- 不冻结 class/source，也不证明当前 environment 与训练时逐位相同；
- 不参与 Registry dispatch、插件归属推断或 Process/Sampler compatibility 判断。

完整 checkpoint config 是训练资产重建权威，完整 sample config 是 mutable invocation
权威；`selected_components` 不能替代任何一侧。

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
3. strict resume 优先复制整个 run directory。启用了 best tracking 的 latest/epoch
   checkpoint 还依赖同一 `checkpoints/` 目录中经过 lineage 校验的 `best.pt`；没有
   validation、因而未启用 best tracking 的 run 不产生这项依赖。目录恢复会检查
   `latest.pt`、`best.pt` 与最大编号 checkpoint，并选择最高一致 snapshot；只剩 inherited
   best 而没有父进度 snapshot 时必须显式提供原父 checkpoint。
4. 同步数据、模型外部资产和配置中引用的文件。相对 data/output 路径与 Builder 私有
   path 都以目标进程启动 cwd 解释；必要时从项目根运行并显式覆盖 output dir。
5. 核对 device/backend。strict resume 若恢复 CUDA/MPS RNG，目标 backend 必须可用；
   CUDA device count 也必须兼容。跨 device、PyTorch 版本或第三方 kernel 不保证逐位一致。
6. 先用小 batch 和有限 step 做 data/model/state-load smoke，再运行完整作业。inference
   不恢复 checkpoint RNG，而是按完整 sample config 的 `sample.seed` 重新初始化。
7. 在目标主机重新评估 RAM、accelerator memory、临时磁盘和 artifact 大小。当前 sampling
   是整体物化 contract；详细公式、基准方法和 trajectory 限制见
   [Sampling artifact 容量](sampling-capacity.md)。

checkpoint 可移植表示“在满足上述显式依赖与 capability contract 的环境中可以重建并加载
状态”，不表示 extension 源码、数据、硬件或数值执行已经被冻结。
