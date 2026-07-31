# Metrics M0–M4 实现决策与维护者审查记录

- 文档性质：feature branch 的维护者审查记录；不是稳定公共 API 文档
- 审查基线：[Metrics 支持开发计划](metrics-support-plan.md)
- 实现范围：M0–M4
- 状态：实现、公开文档和本地 full branch verification 已收束；远端 CI 待推送确认
- 记录日期：2026-07-31

本文记录本次实现采用的责任边界、兼容性决策和已知限制。计划仍是需求依据；当计划中
列出 decision gate 时，本文记录该 gate 在本次实现中的最终结论。本文也保留早期
M0–M1 审查时的历史结论，但凡与后续实现冲突，以标注为“当前”的 M2–M4 决策为准。

## 1. 范围结论

| 阶段 | 当前状态 | 落地内容 |
| --- | --- | --- |
| M0 | 已实现 | canonical tag、旧 monitor alias 拒绝、`MetricUpdate`、`MetricSource`、`EpochMetricSnapshot`、显式 detach、loss reporting weight |
| M1 | 已实现 | `MetricConfig`/`MetricSpec`、metric registry/factory、task-neutral `MetricEngine`、training phase binding、`mean`/`mse`/`mae`、Strategy channel、logger/history/checkpoint 接入 |
| M2 | 已实现 | typed diagnostic result/source request、stable provenance + actual fit iterable verified source、checkpoint 前 diagnostic merge、cadence-aware missing policy、observation-based patience、严格恢复 |
| M3 | 已实现（单进程承诺） | stable extension exports、真实 plugin activation 下的 custom Strategy + Metric contract/provenance、built-in/extension DDP reduction declaration matrix；distributed Trainer 仍明确不在范围内 |
| M4 | 已实现 | config reference、公开 extension/config/migration 文档、custom metric 教程与开发计划归档已收束 |

本次完成不扩大到 post-training Evaluation、HPO、多目标选择、mid-epoch resume 或
distributed Trainer，也没有把 FID/KID 搬进普通 validation batch 循环。M3 的
“distributed readiness”只表示声明级契约已审计，并不表示多进程运行语义已被支持。

## 2. 依赖决策

### 2.1 TorchMetrics 是基础依赖

`torchmetrics>=1.9,<2` 从 `quality` extra 提升为基础依赖。原因不是当前三个内置
metric 难以自行实现，而是所有安装都需要同一个稳定的 state contract：

- registry 注册项必须继承 `torchmetrics.Metric`；
- `MetricEngine` 依赖其 `update()`、`compute()`、`reset()`、device state 和
  distributed-reduction 声明；
- metric config 在 extension 激活前保持纯数据验证，真正构造发生在 composition
  boundary；
- 若 TorchMetrics 仍是可选依赖，普通训练配置会直到 runtime composition 或首个
  validation 才失败，错误边界过晚。

这与 TorchMetrics 官方
[`Metric` contract](https://lightning.ai/docs/torchmetrics/stable/references/metric.html)
一致：自定义 metric 实现 `update()`/`compute()`，state 由 `add_state()` 声明，
base class 负责 reset、device movement 和可配置的 process synchronization。

### 2.2 `torch-fidelity` 继续属于 `quality` extra

`torch-fidelity>=0.3,<1` 没有进入基础依赖。当前普通 phase metrics
`mean`、`mse`、`mae` 不需要它；它服务于带 Inception feature extractor 的 FID/KID
质量路径。TorchMetrics 的
[FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)
与
[KID](https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html)
文档也明确说明默认 feature extractor 需要 `torch-fidelity`。

因此依赖层级表达真实责任：

```text
base install     -> ordinary stateful train/validation/test metrics
quality extra    -> FID/KID 等需要额外模型或特征依赖的质量诊断/评估
```

## 3. 构造决策：拒绝 native namespace resolver

本次对计划中的 native-provider gate 给出否定结论。配置只接受
`REGISTRIES.metrics` 中的稳定名字；不接受任意
`torchmetrics.*` target，不动态 import Python symbol，也不镜像 TorchMetrics 的完整
命名空间。

首批 allowlist 只有：

| Registry name | Implementation |
| --- | --- |
| `mean` | 固定 `nan_strategy="error"` 的 `MeanMetric` wrapper |
| `mse` | 固定 `squared=True`、`num_outputs=1` 的 `MeanSquaredError` |
| `mae` | 固定 `num_outputs=1` 的 `MeanAbsoluteError` |

拒绝 native resolver 的理由：

1. TorchMetrics 的不同 domain 有不同 update signature、constructor contract 和可选
   依赖，不能由一个 namespace prefix 推导共同生命周期；
2. 直接暴露完整 upstream namespace 会把其类名、默认值和版本变化变成
   Stochaflow 的隐式配置 API；
3. 任意 target import 扩大配置执行面，也削弱 unknown component 的可读错误；
4. 当前真实配置只需要三个稳定实现，没有足够重复证明 resolver 的维护成本合理；
5. 第三方仍可显式注册一个保持 `Metric` contract 的实现，不需要修改 core dispatch。

计划中的 `torchmetrics.classification.*` 示例因此也不是当前可用配置。需要该类指标
时，应先增加受测 wrapper 或由 extension 注册稳定名字；只有新的独立 decision gate
通过后才能引入受限 native provider。

## 4. Strategy channel 与 task-neutral core 边界

| 组件 | 拥有的语义 | 明确不拥有 |
| --- | --- | --- |
| `TrainingStrategy` | structured batch 解释、model/objective 调用、channel 名及其 payload contract、epoch loss reporting weight | metric state、跨 phase 生命周期、结果 tag |
| `MetricChannelProvider` | 可选地声明 Strategy 能产生的 channel 集合 | 通用 batch schema、signature introspection |
| `TrainingMetricRuntime` | 在完整 Strategy + declarations 边界检查 channel，组合 phase 到 engine 的映射 | batch/model/task 解释 |
| `MetricEngine` | 一个统计 scope 的实例、binding、detach、`no_grad` update、compute/reset、device、结果 flatten/collision | `preds`/`target`/image/label 等字段含义、额外 model forward、sampling |
| metric implementation | 自己的 `update(*args, **kwargs)` 和统计 state | 获取 batch、调用 Strategy、猜测 sample weight |

内置 channel 是公开的 task adapter contract，而不是全框架 batch schema：

```text
supervised.prediction_target
    -> (prediction, target)

gaussian.prediction_target
    -> (model_output, target)

gaussian.clean_reconstruction
    -> (predicted_clean, clean)
```

无条件和 class-conditional Gaussian Strategy 共享相同 Gaussian channel，因为
conditioning 已在 Strategy 的 model invocation 中完成；core 不需要增加 AFHQ、
class label 或 Gaussian concrete-class 分支。未配置 metric 的自定义 Strategy 无需实现
`MetricChannelProvider`，保持旧 extension 的最小接口。

所有 payload tensor 在分发前 detach，metric update 在 `torch.no_grad()` 下执行。
普通 `dict`、`OrderedDict`、`MappingProxyType`、list、tuple、namedtuple 与安全 scalar
leaf 保留各自容器语义；未知的有状态容器不会被反射遍历。自定义容器必须显式实现
`MetricPayloadDetachable.detach_metric_payload()`，否则 fail closed。这里没有依赖
PyTorch 私有的 `torch.utils._pytree`：私有 API 的注册表、支持类型和兼容性不属于
Stochaflow 可以承诺的公共边界，而一个窄的 opt-in protocol 可以让 extension 对
detach 后的类型与不变量负责。

`diagnostics` 不作为缺失 channel 的 fallback；否则 diagnostic key 拼写会成为隐藏 API。
内置 `mse` 和 `mae` 固定为 scalar-only、`num_outputs=1`；`mse` 还固定
`squared=True`。这避免一个名字看似 scalar、直到 epoch `compute()` 才返回向量并失败。
需要多输出统计时，extension 应注册一个命名与 flatten contract 都明确的 Metric。

## 5. 每个 phase 的独立生命周期

同一 declaration 绑定多个 phase 时，每个 phase 都通过 registry 构造独立 metric
实例和独立 `MetricEngine`。task-neutral engine 不知道 training phase；training
binding 才把 `validation` 映射为 tag prefix `valid`。

```text
train engine:       reset -> update successful optimizer windows -> compute -> reset
validation engine:  reset -> update evaluation batches           -> compute -> reset
test engine:        reset -> update evaluation batches           -> compute -> reset
```

具体政策：

- epoch-only metric 直接调用 `update()`，不调用会在每 batch 同时 compute 的
  `forward()`；TorchMetrics 的
  [Lightning integration 指南](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
  也建议 epoch-only metric 采用这种方式；
- train accumulation window 只有在 optimizer step 成功后才提交该 window 的 metric
  updates，GradScaler overflow/跳步不会伪装成成功训练 observation；
- validation/test 在 `torch.no_grad()` evaluation step 后立即更新；
- phase 异常、空 phase、全零 loss aggregation weight 或 compute 失败时执行 reset，
  不把部分 state 泄漏到下一次调用；
- 一个 update 在任意已绑定 metric 上失败时，整个 engine 立即 reset，再带 metric id
  与 channel 上下文抛错；不会保留“前几个 metric 已更新、后一个失败”的部分提交；
- `TrainingMetricRuntime` 在组合完成后显式迁移到 Trainer device，不从 payload 或
  metric 类名猜测 device；
- metric state 不属于 managed training assets，不进入 optimizer、mode lifecycle 或
  checkpoint。

## 6. Step report、epoch canonical key 与 W&B

### 6.1 命名

| 事件 | Key | 消费者 |
| --- | --- | --- |
| train optimizer step report | `train/step/loss` | logger；不进入 epoch history/checkpoint/monitor |
| completed train epoch | `train/loss` | logger、history、checkpoint、monitor |
| completed validation epoch | `valid/loss` | logger、history、checkpoint、monitor |
| completed test phase | `test/loss` | logger 和 post-training test result；永不参与训练选择 |
| stateful phase metric | `<phase>/metrics/<id>[/<subkey>]` | 对应 phase 的 canonical result |
| epoch diagnostic metric | `diagnostics/<diagnostic-id>/<metric...>` | logger、history、checkpoint；只有 verified validation source 可参与 monitor |
| runtime/throughput | `system/...` | observation only，source 不具 selection 资格 |

将 step loss 改为 `train/step/loss`，是为了避免一个 W&B/TensorBoard series 同时混入
batch 值和 epoch aggregate。`train/loss`、`valid/loss` 等 canonical epoch key 才能
被 monitor 使用；`train_loss`、`valid_loss` 不提供 alias、双写或 reader。

metric 返回 scalar 时 key 结束于 `<id>`；返回 flat mapping 时增加经过验证的
`<subkey>`。bool、list、tuple、non-scalar tensor、嵌套 mapping 和冲突 key 默认失败。

当前 monitor grammar 接受 `train/loss`、`valid/loss`、`test/loss`、对应
`<phase>/metrics/<id>[/<subkey>]`，以及
`diagnostics/<diagnostic-id>/<metric...>`。配置加载时拒绝 step、system、
不完整 diagnostic path、周围空白和旧 alias，避免长训练直到首个 epoch 结束才发现
monitor 不存在。无 validation loader 时，runner 只把默认的 `valid/loss` 回退为
`train/loss`；显式 `train/metrics/...` 保持不变，显式 `valid/metrics/...` 或可选择的
diagnostic 则在训练前失败。`test/*` 的 key grammar 合法，但 selection preflight 会因
test-role source 不具资格而拒绝。

当 best tracking 或 early stopping 实际启用时，`Trainer.fit()` 还会在 diagnostic、
loader iteration 和 epoch loop 之前预检 monitor 的完整静态依赖：`valid/*` 必须存在
validation loader，phase metric id 必须声明在对应 phase；diagnostic monitor 必须定位
到恰好一个 composition-verified、validation-role、selection-eligible source。flat
mapping 的 subkey 由 `Metric.compute()` 或 diagnostic callback 动态产生，因此
preflight 验证静态 binding，完整 key 继续在 due snapshot 中 fail closed。
`track_best: false` 时 monitor 未被消费，不会触发这项语义检查。

### 6.2 W&B 保留 canonical path

W&B backend 保留 `/`、`.`、`-` 和 `_`，只替换其他不支持字符，并在 sanitize 后 key
数量减少时报告 collision。它不把 `valid/loss` 改写成 `valid_loss`，因此 local
JSONL、TensorBoard、W&B、history、checkpoint 和 monitor 使用同一个名字。

W&B 官方说明 `/` 会把 metric 组织到 panel section，见
[`Run.log`/`Run.define_metric` reference](https://docs.wandb.ai/ref/python/experiments/run/)
和
[library integration guide](https://docs.wandb.ai/models/integrations/add-wandb-to-any-library)。
保留 canonical path 也意味着没有 backend-specific alias 可供 checkpoint 或 monitor
误用。

## 7. Sample-weighted loss reporting 不改变 backward

`TrainStepOutput.loss_aggregation_weight` 是 detached、有限、非负的 scalar reporting
contract。内置 Supervised/Gaussian Strategy 使用 logical batch size；一个 epoch 的
loss 为：

```text
epoch reported loss = sum(detach(batch_loss) * batch_weight)
                      / sum(batch_weight)
```

它不参与当前 accumulation window 的 autograd 表达式。backward 仍保持既有的
micro-batch scalar 等权语义：

```text
backward loss = sum(microbatch_loss) / accumulation_window_size
```

因此 variable batch size 会改变 epoch report 的样本权重，但不会暗中改变优化轨迹。
单个 step 可以声明零权重，完整 phase 的权重和必须大于零。若某个 metric 自己需要
sample weight，Strategy 必须在该 channel 的 `MetricUpdate.args/kwargs` 中显式提供；
core 不从 tensor shape 或 constructor 参数名猜测。

该字段也不承载 diffusion timestep/SNR/P2 weighting。后者属于 Strategy 内可微
objective 数学；两者共享字段会混淆训练目标和观测统计。

## 8. Diagnostic monitoring、source verification 与 checkpoint v11

### 8.1 Composition 绑定 source，而不是相信 callback

会产生 epoch metric 的 diagnostic 通过 `DiagnosticSourceRequest` 声明 source id、
`train|validation|test|external` data role 与 JSON-safe protocol descriptor。完整
Builder/factory composition 把以下事实组成 canonical descriptor，再计算 SHA-256：

- configured diagnostic id 与 source id；
- source role 与 diagnostic 自己的 versioned protocol/config descriptor；
- resolved data config；
- 实际 data artifact identities；
- 已选择 extension plugin provenance。

digest 同时保存在 `VerifiedMetricSource.protocol_digest` 和
`MetricSource.protocol_id="sha256:<digest>"`。callback 只返回 `DiagnosticResult` 的
`source_id` 与 scalar mapping，不能自行声明 data role 或
`selection_eligible=True`。未知 source、重复 source、错误 diagnostic-id 前缀、key
collision 或 binding mismatch 都会 fail closed。

request 中的 role 只是待验证约束，不是授权。composition 还把 train/validation
request 绑定到本次 DataBuilder 实际创建的 re-iterable；
`BoundTrainingDiagnostic.source_iterables` 只保留这一运行期对象绑定，Trainer 在任何
callback 或 loader iteration 前按对象 identity 与本次 `fit()` 参数核对。Python
identity 不进入 SHA-256 或 checkpoint，因此新进程可用相同 stable descriptor 和新建
loader 得到相同 protocol digest。缺少完整 provenance、缺少实际 iterable、当前 fit
对象错配，以及未经绑定但声明 source 的 raw diagnostic 都 fail closed。没有 source
的 observation-only raw diagnostic 保持兼容。

当前 `FitStartEvent` 只注入 train/validation iterable；因此 training diagnostic
binder 不接受 test-role request。正式 test 只属于冻结 subject 后的独立 Evaluation，
不能在训练期 diagnostic 中旁路这一治理边界。

Gaussian quality diagnostic 将结果分成两个 source：

- `observation` 是 `data_role="external"` 的 sampler statistics、样本计数与耗时；
- `validation_quality` 是使用已绑定 validation reference protocol 得到的 FID/KID。

这项拆分防止仅凭 `diagnostics/...` 名称或 provider 类型，把外部运行观测误当成
validation 质量。所有 key 使用
`diagnostics/<configured-diagnostic-id>/<metric...>`，因此同类 diagnostic 的多个
配置实例不会互相碰撞。首版每个 diagnostic 最多有一个 selection-eligible source；
否则一个 monitor 缺失时无法无歧义地归因 cadence。

callback 的时间语义也是 typed contract：

- `None` 表示本 epoch 没有任何 source 到 cadence；
- 非空 tuple 表示其中 source 已到期；
- `DiagnosticResult(metrics={})` 是明确的“已到期” marker，不是 cadence skip。

因此 `missing="skip"` 只跳过尚未到 cadence 的 observation。source 已到期却没有返回
monitor key、返回失败或拼错 key 时仍立即失败；整个 fit 一次有效 observation 都没有
也会失败。diagnostic 在 history、best/early-stopping 与 checkpoint 之前运行并合并到
同一个 `EpochMetricSnapshot`。

### 8.2 Observation-based monitor policy

当前 monitor policy 保存 canonical `metric`、`mode`、`missing` 与 `min_delta`。
`missing="skip"` 只允许 `diagnostics/...` monitor；普通 train/validation metric
缺席继续报错。patience 的计数单位是有效 observation 次数，而不是 wall-clock epoch：
未到 cadence 不增加 wait counter，也不 carry-forward 陈旧值。一个有效且未改善的
observation 才增加 `observations_without_improvement`。

strict resume 精确核对并恢复：

- tracking 是否启用；
- monitor 的 `metric/mode/missing/min_delta`；
- early-stopping patience；
- best epoch/value；
- `monitor_observations` 与 `observations_without_improvement`；
- stopped state。

resume 不能静默切换 policy、关闭已启用的 tracking，或把 epoch 数解释成 observation
数。无论是否继承 best checkpoint，strict reader 都解析并验证完整 training-loop
snapshot。

### 8.3 为什么 bump 到 v11

早期 M0–M1 审查曾决定保持 v10，因为当时只复用现有 metric values/source metadata
容器。M2 后这个结论已被明确取代：observation-based patience 与完整 monitor policy
成为 strict resume 所需状态，v10 不能无歧义表达它们，因此
`CHECKPOINT_FORMAT_VERSION` bump 到 v11。

v11 保存：

- resolved metric declarations 与 extension/data provenance；
- 本 epoch `EpochMetricSnapshot.values` 和逐 key `MetricSource`；
- 完整 monitor policy、best state、observation counters 与 patience。

source metadata 不放进 `metrics`，因为
`origin/data_role/protocol_id/selection_eligible` 是 provenance 和选择资格，不是 scalar
observation。`EpochMetricSnapshot` 同时严格验证 canonical namespace 与 source：
`train/*`、`valid/*`、`test/*` 分别匹配 phase role，`system/*` 使用 system source，
`diagnostics/*` 使用 diagnostic source；prefix 与 metadata 冲突时 fail closed。

Metric state 本身仍不保存：checkpoint 只在完整 epoch 后产生，下个 phase 总会先
reset；保存 list state 还可能把全量 prediction/target 写进 checkpoint。mid-epoch
resume 若未来进入范围，必须同时设计 loader cursor 和 persistent metric state。

post-training test 发生在训练 checkpoint 冻结之后。`test/metrics/*` 由 logger 发布，
不会回写 best/latest checkpoint；test-role source 始终不具 selection 资格。v11 writer
只写 canonical keys，也不为旧 `train_loss`/`valid_loss` 或 pre-cutover v10 snapshot
提供 alias/migration reader。

## 9. MNIST 与 AFHQ 指标选择

保留的 MNIST、AFHQ-v2 ADM、AFHQ-v2 DiT 和 AFHQ smoke 训练配置采用相同的低成本
Gaussian phase metrics：

| ID | Channel / payload | Phase | 解释 |
| --- | --- | --- | --- |
| `prediction_mae` | `gaussian.prediction_target(model_output, target)` | validation、test | 当前 prediction parameterization（现有配置为 `v`）与其监督 target 的元素级 MAE |
| `clean_reconstruction_mse` | `gaussian.clean_reconstruction(predicted_clean, clean)` | validation、test | 同一个随机 noisy marginal 反演出的 clean prediction 与 paired clean sample 的元素级 MSE |

选择理由：

- 两个 payload 已由 Gaussian Strategy 的一次 evaluation forward 产生，不增加 sampling
  loop 或额外 model invocation；
- MAE 与训练 objective 的 MSE 提供不同误差视角，clean reconstruction 又把
  parameterization-space 误差投影回 clean sample space；
- 相同 channel 可服务无条件 MNIST 和 class-conditional AFHQ，不向 Trainer 泄漏
  modality 或 condition schema；
- 只配置 validation/test，避免默认给训练 loop 增加不必要的 state update 开销；
- best checkpoint 继续监控 `valid/loss`，不把新增 proxy 在未经质量相关性验证前升级
  为 selection target。

这些数值不是生成质量指标。它们只衡量 teacher-forced noisy/clean pair 上的一步预测：

- 不运行 reverse sampling；
- 不衡量 sample diversity、mode coverage 或 class-conditional fidelity；
- 不使用独立真实样本分布特征；
- 受随机 timestep、noise schedule、prediction type 和 reconstruction conditioning
  影响，不能跨协议直接比较。

因此不得把 `clean_reconstruction_mse` 命名为 FID、KID、PSNR 或“生成质量”。AFHQ
正式质量结论仍来自冻结 checkpoint 上带固定 real/fake protocol 的独立 class-aware
KID/FID Evaluation；现有训练期 quality diagnostic 的 sampling、cache、cadence 和
failure policy 也保持原边界。

这里的 contract 验证按配置的数据 class/protocol 工作，不增加“狗”这一类别的专门
验收，也不为 Metrics 功能另行下载或生成狗样本。若使用仓库已有 AFHQ example，类别
范围由该 example 的 data/evaluation 配置决定；通用实现与测试不写入 dog-specific
分支、fixture 或输出。

## 10. Stable extension surface 与单进程承诺

### 10.1 公开 provider-level diagnostic contract

M3 的 decision gate 选择公开 provider-level diagnostic contract。原因是现有
diagnostic `modules` 配置已经允许 extension provider；若只公开完整
`TrainingDiagnostic`，配置能力与可依赖 API 会不一致。`stochaflow.extensions`
因此稳定导出：

- `MetricConfig`、`MetricSpec`、`MetricUpdate`、`MetricPayloadDetachable`、
  `MetricSource`、`EpochMetricSnapshot`、metric registry/factory/runtime contracts；
- `DiagnosticSourceRequest`、`VerifiedMetricSource`、`DiagnosticResult`、
  `BoundTrainingDiagnostic` 与 source provider capability；
- Gaussian diagnostic 的 step/sampler/reference provider bases、contexts 与局部
  `DIAGNOSTIC_PROVIDERS` catalog。

provider registry 只扩展 Gaussian diagnostic 内部 pipeline，不是 framework-global
Metric registry。reference provider 的质量结果由 verified validation source 承载；
sampler statistics 与耗时仍进入 external observation。独立 extension fixture 同时注册
custom Strategy 与 custom Metric，并经过 entry-point discovery 和真实模块激活运行
微型 train/validation：resolved config 保存 plugin name 与 component declarations，
run manifest/checkpoint 保存安装 distribution/version/target provenance。这避免手工
伪造 metadata 或只用仓库内置 subclass 自证公共 API。

### 10.2 单进程承诺与未来 DDP

TorchMetrics 提供 state reduction 和 compute-time synchronization，但这不等于
Stochaflow Trainer 已具备 DDP/FSDP 语义。TorchMetrics 的
[structure/DDP guide](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html)
还指出 distributed sampler 为补齐各 rank 可能复制样本并造成 evaluation bias。

M3 已加入声明级 contract matrix：逐项审计 built-in 与独立 extension fixture 的每个
TorchMetrics state 都有显式 reduction，并检查 compute-time synchronization policy。
这能在单进程测试中阻止“新增 state 却忘记声明 reduction”的退化，但不能证明
distributed Trainer 语义。

继续只承诺单进程 CPU/单 GPU 结果。分布式支持的阻塞是以下 correctness 与 publication
协议尚未定义，而不是缺少一张特定显卡上的速度曲线：

- distributed sampler 为补齐 uneven shard 可能复制 validation 样本；需要定义样本
  identity、计数、去重或明确接受 bias 的协议；
- rank-local metric update 失败、OOM、non-finite 或 GradScaler overflow 时，必须让
  所有 rank 对提交/reset/终止达成一致，避免某些 rank 进入 compute collective、另一些
  已退出；
- list-state metric 的 gather/cat、uneven state 与 process-group policy 需要受测；
- canonical snapshot、logger、artifact、best/latest checkpoint 必须由明确的 rank-zero
  publication policy 产生，同时保证其他 rank 的 loop state 与选择决策一致；
- diagnostic sampling 若每个 rank 独立运行，可能重复样本或重复 artifact；若只在
  rank zero 运行，又需要定义其他 rank 的同步与失败传播。

这些问题与类别无关，不是 dog-specific validation；它们也不只是“还没在实际硬件规格
上跑性能验证”。即使无限算力，若没有上述语义，结果仍可能重复计数、死锁或由多个 rank
竞写 checkpoint。真实多机/多卡性能与 soak test 应在语义 contract 落地后进行。

## 11. 审查中发现的问题与剩余风险

### 11.1 已关闭

- `validate_metric_configs` 曾是只验证不返回的 helper，而
  `TrainingMetricRuntime` 需要迭代其结果；现已明确返回已验证 declaration list。
- W&B 旧 sanitizer 会移除 `/`，破坏 canonical identity；现已保留 path 字符并增加
  sanitize collision 检查。
- 当前主要生图任务使用 class-conditional Gaussian Strategy；它现在与无条件
  Gaussian Strategy 声明相同 channel，没有通过 example-specific runner 分支补丁接入。
- diagnostic scalar 过去只直接写 logger 且在 checkpoint 之后运行；现在通过
  Builder-bound verified source 合并进 checkpoint 前的 canonical snapshot。
- sampler observation 与 reference FID/KID 过去共享一个无 data-role 区分的 mapping；
  现在分别绑定 external 与 validation source。
- 低频 monitor 过去只能把缺席当错误；现在只对“未到 cadence”提供显式
  `missing="skip"`，patience 与 strict resume 都按 observation 计数。
- 自定义 stateful payload 过去可能被不完整递归 detach 后保留 autograd graph；现在未知
  容器必须实现 `MetricPayloadDetachable`。
- metric update 过去可能在后一个 metric 失败后留下前面的部分 state；现在 update
  failure 原子 reset 整个 engine。
- MSE/MAE 的多输出 constructor 过去可能延迟到 epoch compute 才以 non-scalar 失败；
  内置名称现在固定 single-output scalar 语义。

### 11.2 剩余

- W&B 官方支持 `/` 进行 panel grouping，但其部分 GraphQL sort/filter UI 要求
  identifier-style 名称；见
  [W&B metric naming constraints](https://docs.wandb.ai/support/models/articles/why-cant-i-sort-or-filter-metrics-with-c)。
  当前决策优先保持跨 backend canonical identity，不新增 underscore alias；需要在真实
  W&B project 上验证 query/dashboard UX。
- post-training `test/metrics/*` 只进入 logger，不进入 `FinalSummary` 的结构化多指标
  字段，也不回写已冻结 checkpoint；这是本次明确的摘要范围，不应由调用者猜测。
- metric state 不支持 mid-epoch resume；OOM 风险仍由具体 list-state metric 与
  validation budget 负责，MetricEngine 不隐式截断样本。
- proxy metric 可能与最终 sample quality 弱相关，尤其是高噪声 timestep 的
  reconstructed-clean error；在建立实证相关性前，retained configs 继续用
  `valid/loss` 选择 checkpoint。
- train metric 为等待 optimizer window 是否成功，会先递归 detach 并暂存 payload；
  `MetricEngine` 在分发前仍进行一次防御性 detach。Tensor 的第二次 `detach()` 不复制
  storage，但会重建小型容器；`MetricPayloadDetachable` 只描述如何安全 detach 自定义
  容器，并不是“已经 detached”的免检标记。后续若优化需另设可验证的窄 contract。
- DDP declaration matrix 已完成，但 distributed sampler、rank-local failure/overflow、
  diagnostic duplication 与 rank-zero publication 仍未定义，所以产品承诺保持单进程。
- `torchmetrics>=1.9,<2` 成为基础依赖后，发布前仍需在仓库支持的平台矩阵执行安装与
  import smoke，尤其要保留 Intel macOS best-effort lane 的明确结果。

## 12. 维护者自审清单

- [x] core 不解包或猜测任意 batch schema。
- [x] Strategy 只负责 batch/model/objective 与 task-owned channel payload。
- [x] MetricEngine 保持 task-neutral，不调用模型、不拥有 sampling/diagnostic。
- [x] built-in 和 extension metric 共享 registry/factory 路径；没有 hidden core path。
- [x] config load 不导入任意 target；native TorchMetrics namespace resolver 已拒绝。
- [x] channel compatibility 在完整 Strategy + declarations composition boundary 失败。
- [x] metric payload 按内置容器或 `MetricPayloadDetachable` 明确 detach，并在
  `torch.no_grad()` 下 update。
- [x] train/validation/test 使用不同 metric instances，异常与部分 update 路径 reset。
- [x] variable batch 的 epoch loss report 使用显式 weight，backward loss 不乘该 weight。
- [x] step 与 epoch loss 使用不同 key；logger/history/checkpoint/monitor 共享 canonical
  epoch key。
- [x] test-role observation 不能参与 best/early-stopping selection。
- [x] 被消费的 monitor 在 loader iteration 前验证 validation loader、metric id 与
  phase；动态 mapping subkey 在 compute 后验证。
- [x] checkpoint 只保存完成 epoch 的 values/source metadata，不保存派生 metric state。
- [x] retained MNIST/AFHQ configs 提供低成本 prediction/reconstruction proxy，且
  monitor 已迁移为 `valid/loss`。
- [x] M2 typed/verified diagnostic result、cadence-aware missing policy 与 checkpoint
  前排序。
- [x] M2 validation source 同时绑定实际 fit iterable 与 stable provenance；raw
  source provider、缺失 provenance/iterable 和 fit 对象错配均 fail closed。
- [x] M2 只把 validation-role FID/KID 作为可选择 source；external sampler observation
  不可选择。
- [x] M2 observation-based patience、至少一次 observation guard 与 exact strict resume。
- [x] checkpoint 已 bump v11，并记录 pre-cutover v10 不兼容。
- [x] M3 独立 extension contract fixture 经 entry-point discovery、真实 import
  activation、factory/Trainer 和 checkpoint 验证 plugin provenance；并覆盖
  distributed reduction declaration matrix。
- [x] M3 provider-level diagnostic contract 已进入 stable extension exports；产品承诺
  仍为单进程。
- [x] M4 stable API/config/migration 文档、教程与公开导航收束。
- [x] 本计划已在 `docs/development/` 原地归档；该目录不进入 Sphinx 发布。
- [x] 本机完成全量 pytest、Ruff、Pyright、config reference、严格 Sphinx、package
  build 和短 Gaussian end-to-end run。
- [ ] 远端支持平台矩阵 CI 待本次提交推送后确认。

## 13. 一手资料

- [TorchMetrics overview](https://lightning.ai/docs/torchmetrics/stable/)
- [TorchMetrics `Metric` reference](https://lightning.ai/docs/torchmetrics/stable/references/metric.html)
- [Implementing a TorchMetrics metric](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html)
- [TorchMetrics with PyTorch Lightning](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
- [TorchMetrics structure and DDP behavior](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html)
- [TorchMetrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)
- [TorchMetrics KID](https://lightning.ai/docs/torchmetrics/stable/image/kernel_inception_distance.html)
- [W&B Run API](https://docs.wandb.ai/ref/python/experiments/run/)
- [W&B library integration guidance](https://docs.wandb.ai/models/integrations/add-wandb-to-any-library)
- [W&B metric naming constraints](https://docs.wandb.ai/support/models/articles/why-cant-i-sort-or-filter-metrics-with-c)

## 14. Feature branch 验证记录

### 14.1 历史 M0–M1 基线

2026-07-31 曾在 Windows/CUDA 开发环境对 M0–M1 基线完成：

| 验证 | 结果 |
| --- | --- |
| `uv run pytest -q` | 1243 passed、26 skipped；skip 为平台/权限条件 |
| `uv run pytest tests/test_training_metrics.py -q` | 12 passed；包含无外部数据的 Gaussian train/validation/test metric end-to-end smoke |
| `uv run ruff check .` | 通过 |
| `uv run pyright` | 0 errors、0 warnings |
| `uv run python tools/generate_config_reference.py --check` | 配置参考最新 |
| 根项目与 AFHQ `uv lock --check --offline` | 均通过 |
| `uv run sphinx-build -W --keep-going -b html docs docs/_build/html` | 严格构建通过 |
| `uv build` | wheel 与 source distribution 均构建成功，包内包含 `stochaflow.metrics` |

真实 MNIST CLI smoke 已尝试，但本机无完整数据缓存：数据物化下载
`train-labels-idx1-ubyte.gz` 时主源发生 SSL EOF，备用源返回 404，因此在 DataSource
阶段终止，尚未进入训练。这一外部获取失败不计为 metrics runtime 验证；无网络
Gaussian smoke 覆盖了同一 Strategy channel、phase engine、monitor、checkpoint 和
test metric 路径。

### 14.2 当前 M0–M4 本地收束结果

2026-07-31 在 macOS/Python 3.14.3 对当前工作树完成：

| 验证 | 结果 |
| --- | --- |
| `uv run pytest` | 1405 passed、14 skipped；skip 均为 CUDA/BF16 不可用 |
| diagnostic/source/plugin 聚焦组合 | 171 passed |
| built-in Diffusion quality end-to-end | 内存 Gaussian batch；FID/KID cadence、`warn` 隔离与 best checkpoint 通过 |
| `uv run ruff check .` | 通过 |
| `uv run pyright` | 0 errors、0 warnings |
| `uv run python tools/generate_config_reference.py --check` | 配置参考最新 |
| `uv run sphinx-build -W --keep-going -b html ...` | 严格构建通过 |
| `uv build` | wheel 与 source distribution 构建成功，包含 diagnostic binding |
| built wheel import smoke | 从 wheel 导入 `stochaflow.training.diagnostics.binding` 成功 |

这些结果不依赖 dog-specific 数据或外部下载。远端支持平台矩阵 CI 需要在本次提交推送后
单独确认；在其通过前不声称 branch ready to merge。
