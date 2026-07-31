# Metrics M0–M1 实现决策与维护者审查记录

- 文档性质：feature branch 的维护者审查记录；不是稳定公共 API 文档
- 审查基线：[Metrics 支持开发计划](metrics-support-plan.md)
- 实现范围：M0–M1
- 明确延期：M2 diagnostic monitoring、M3 extension/distributed readiness、M4
  正式文档与计划收束
- 记录日期：2026-07-31

本文记录本次实现采用的责任边界、兼容性决策和已知限制。计划仍是需求依据；当计划中
列出 decision gate 时，本文记录该 gate 在本次实现中的最终结论。

## 1. 范围结论

| 阶段 | 本次状态 | 落地内容或延期理由 |
| --- | --- | --- |
| M0 | 已实现 | canonical tag、旧 monitor alias 拒绝、`MetricUpdate`、`MetricSource`、`EpochMetricSnapshot`、递归 detach、loss reporting weight |
| M1 | 已实现 | `MetricConfig`/`MetricSpec`、metric registry/factory、task-neutral `MetricEngine`、training phase binding、`mean`/`mse`/`mae`、Strategy channel、logger/history/checkpoint 接入 |
| M2 | 延期 | Diagnostic 尚未返回带 verified source 的结果，调用仍发生在 checkpoint 选择之后；因此 FID/KID 不能作为本次实现的 monitor |
| M3 | 延期 | 已提供必要的 `MetricUpdate`、`MetricChannelProvider` 和 registry 基础，但独立第三方 extension、provenance 及 DDP contract matrix 尚未完成，不能宣称 M3 完成 |
| M4 | 延期 | 本记录仍位于 `docs/development/`；稳定 extension API 教程、完整公开文档迁移和计划归档留待后续 |

M0–M1 的完成不扩大到 post-training Evaluation、HPO、多目标选择、mid-epoch
resume 或 distributed Trainer。本次也没有把 FID/KID 搬进普通 validation batch
循环。

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
| `mse` | `MeanSquaredError` |
| `mae` | `MeanAbsoluteError` |

拒绝 native resolver 的理由：

1. TorchMetrics 的不同 domain 有不同 update signature、constructor contract 和可选
   依赖，不能由一个 namespace prefix 推导共同生命周期；
2. 直接暴露完整 upstream namespace 会把其类名、默认值和版本变化变成
   Stochaflow 的隐式配置 API；
3. 任意 target import 扩大配置执行面，也削弱 unknown component 的可读错误；
4. 当前真实配置只需要三个稳定实现，没有足够重复证明 resolver 的维护成本合理；
5. 第三方仍可显式注册一个保持 `Metric` contract 的实现，不需要修改 core dispatch。

计划中的 `torchmetrics.classification.*` 示例因此不是 M0–M1 可用配置。需要该类指标
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

所有 payload tensor 在分发前递归 detach，metric update 在 `torch.no_grad()` 下执行。
`diagnostics` 不作为缺失 channel 的 fallback；否则 diagnostic key 拼写会成为隐藏 API。

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
| runtime/throughput | `system/...` | observation only，source 不具 selection 资格 |

将 step loss 改为 `train/step/loss`，是为了避免一个 W&B/TensorBoard series 同时混入
batch 值和 epoch aggregate。`train/loss`、`valid/loss` 等 canonical epoch key 才能
被 monitor 使用；`train_loss`、`valid_loss` 不提供 alias、双写或 reader。

metric 返回 scalar 时 key 结束于 `<id>`；返回 flat mapping 时增加经过验证的
`<subkey>`。bool、list、tuple、non-scalar tensor、嵌套 mapping 和冲突 key 默认失败。

当前 monitor grammar 只接受 `train/loss`、`valid/loss` 和对应
`<phase>/metrics/<id>[/<subkey>]`。配置加载时即拒绝 step、system、test、diagnostic
namespace、周围空白和旧 alias，避免长训练直到首个 epoch 结束才发现 monitor 不存在。
无 validation loader 时，runner 只把默认的 `valid/loss` 回退为 `train/loss`；显式
`train/metrics/...` 保持不变，显式 `valid/metrics/...` 则在训练前失败。

当 best tracking 或 early stopping 实际启用时，`Trainer.fit()` 还会在 diagnostic、
loader iteration 和 epoch loop 之前预检 monitor 的完整静态依赖：`valid/*` 必须存在
validation loader，metric id 必须声明在对应 train/validation phase。flat mapping 的
subkey 由 `Metric.compute()` 动态产生，因此 preflight 只验证 base id；完整 subkey
继续在首个 epoch snapshot 中 fail closed。`track_best: false` 时 monitor 未被消费，
不会触发这项语义检查。

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

## 8. Checkpoint v10 与 source metadata

`CHECKPOINT_FORMAT_VERSION` 保持 v10，不 bump。理由是本次没有增加新的 required
top-level header、managed asset state、inference recipe 字段或 restore 顺序，而是复用
v10 已有的三个容器：

- resolved metric declarations 进入既有 `config`；
- `EpochMetricSnapshot.values` 进入既有 scalar `metrics` mapping；
- `EpochMetricSnapshot.sources` 序列化到既有 `metadata.metric_sources`。

source metadata 不放进 `metrics`，因为 `MetricSource` 的
`origin/data_role/protocol_id/selection_eligible` 是 provenance 和选择资格，不是 scalar
observation。保持分离后，logger 仍只消费标量，monitor 可用相同 canonical key 同时查
value 和 source，checkpoint reader 也不会把 metadata 错当成数值。

`EpochMetricSnapshot` 同时验证 key namespace 与 source：`train/*`、`valid/*`、
`test/*` 必须分别匹配对应 phase data role，`system/*` 必须使用 system source，
`diagnostics/*` 必须使用 diagnostic source。prefix 与 metadata 冲突时 fail closed；
其中 diagnostic prefix 只约束 origin，不推断其 validation/test role 或 selection
资格。

Metric state 本身不保存：当前 checkpoint 只在完整 epoch 后产生，下一个 phase 总会
reset；保存 list state 还可能把全量 prediction/target 写进 checkpoint。mid-epoch
resume 若未来进入范围，必须同时设计 loader cursor 和 persistent metric state，不能
从本实现推断已经支持。

post-training test 发生在训练 checkpoint 冻结之后。`test/metrics/*` 由 logger 发布，
终端 `FinalSummary` 仍只展示 test loss；测试结果不会回写 best/latest checkpoint。
test-role source 始终不具 selection 资格。

v10 writer 现在只写 canonical keys；旧 `train_loss`/`valid_loss` semantic snapshot
没有迁移层。这是计划已明确的 compatibility cutover，仍应在 release notes 中提醒
持有 pre-cutover v10 checkpoint 的用户。

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

## 10. 单进程承诺与未来 DDP

TorchMetrics 提供 state reduction 和 compute-time synchronization，但这不等于
Stochaflow Trainer 已具备 DDP/FSDP 语义。TorchMetrics 的
[structure/DDP guide](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html)
还指出 distributed sampler 为补齐各 rank 可能复制样本并造成 evaluation bias。

M0–M1 只承诺单进程 CPU/单 GPU 的正确结果。未来 M3 至少需要验证：

- 每个 built-in 与 extension metric 的 `dist_reduce_fx`；
- uneven/replicated validation samples 的计数与去重政策；
- rank-local update failure、compute synchronization 和 reset；
- train overflow/跳步在各 rank 上的一致提交；
- list-state memory、CPU offload 和 process-group policy；
- logger/checkpoint 只由明确 rank 发布，且 canonical snapshot 在各 rank 一致。

在这些 contract tests 完成前，不能因 `Metric` 默认支持 synchronization 就在文档中
宣称 distributed-ready。

## 11. 审查中发现的问题与剩余风险

### 11.1 已关闭

- `validate_metric_configs` 曾是只验证不返回的 helper，而
  `TrainingMetricRuntime` 需要迭代其结果；现已明确返回已验证 declaration list。
- W&B 旧 sanitizer 会移除 `/`，破坏 canonical identity；现已保留 path 字符并增加
  sanitize collision 检查。
- 当前主要生图任务使用 class-conditional Gaussian Strategy；它现在与无条件
  Gaussian Strategy 声明相同 channel，没有通过 example-specific runner 分支补丁接入。

### 11.2 剩余

- M2 未实现，diagnostic result 仍不能进入同一 snapshot、best selection 或 early
  stopping；FID/KID monitor 配置现在不应被宣称支持。
- M3 未完成，stable extension 文档、独立第三方实现/provenance 和 DDP contract tests
  仍是 merge 后续工作。
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
  storage，但会重建小型容器；这是当前正确性优先的轻微开销，后续若优化需引入能证明
  payload 已 detached 的窄 contract，不能用不受约束的布尔开关绕过安全边界。
- `torchmetrics>=1.9,<2` 成为基础依赖后，发布前仍需在仓库支持的平台矩阵执行安装与
  import smoke，尤其要保留 Intel macOS best-effort lane 的明确结果。

## 12. 维护者自审清单

- [x] core 不解包或猜测任意 batch schema。
- [x] Strategy 只负责 batch/model/objective 与 task-owned channel payload。
- [x] MetricEngine 保持 task-neutral，不调用模型、不拥有 sampling/diagnostic。
- [x] built-in 和 extension metric 共享 registry/factory 路径；没有 hidden core path。
- [x] config load 不导入任意 target；native TorchMetrics namespace resolver 已拒绝。
- [x] channel compatibility 在完整 Strategy + declarations composition boundary 失败。
- [x] metric payload 递归 detach，并在 `torch.no_grad()` 下 update。
- [x] train/validation/test 使用不同 metric instances，异常路径 reset。
- [x] variable batch 的 epoch loss report 使用显式 weight，backward loss 不乘该 weight。
- [x] step 与 epoch loss 使用不同 key；logger/history/checkpoint/monitor 共享 canonical
  epoch key。
- [x] test-role observation 不能参与 best/early-stopping selection。
- [x] 被消费的 monitor 在 loader iteration 前验证 validation loader、metric id 与
  phase；动态 mapping subkey 在 compute 后验证。
- [x] checkpoint 只保存完成 epoch 的 values/source metadata，不保存派生 metric state。
- [x] retained MNIST/AFHQ configs 提供低成本 prediction/reconstruction proxy，且
  monitor 已迁移为 `valid/loss`。
- [ ] M2 verified diagnostic result、cadence-aware missing policy 与 checkpoint 前排序。
- [ ] M3 独立 extension contract fixture、provenance 和 distributed reduction matrix。
- [ ] M4 稳定 API 文档、教程、计划归档与公开导航收束。
- [ ] 合并前在全部支持平台完成 dependency/import smoke、全量 pytest、Ruff、Pyright
  和短 Gaussian end-to-end run。

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

2026-07-31 在 Windows/CUDA 开发环境完成：

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
