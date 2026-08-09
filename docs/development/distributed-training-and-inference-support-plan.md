# Distributed Training 与 Inference 支持计划

> 工作状态：暂停
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## 完成后用户能做什么

SPMD 表示多个进程运行同一程序，DDP 表示 PyTorch 的分布式数据并行。只有单设备实测无法满足
容量或时间预算后，用户才可以用固定数量的进程运行受支持的 DDP 训练，并从完整的分布式恢复包
继续训练。

模型能够在每个设备上完整加载时，用户还可以把采样任务分给多个进程；每个进程写自己的结果，
最后验证并发布一个完整 artifact。FSDP2（PyTorch Fully Sharded Data Parallel v2，模型状态分片）、
多节点、精确分布式 Evaluation 和弹性伸缩保留在文末，不随“启用 distributed”自动获得。

## 当前仓库已经支持什么

- 单设备 Trainer 已集中 optimizer lifecycle、accumulation、precision、EMA、checkpoint 和
  outcome。
- TrainingBuilder/Strategy 和 DataSource/DataBuilder 已有明确的单次任务分工。
- SamplingBuilder 负责任务组合和 writer-ready output；普通 sampling runtime 负责
  writers、manifest 与原子发布。
- Data artifact 已有 lock、private staging、verification 和 atomic publication 基础。
- 可搬迁 checkpoint 与只读 inference projection 可作为后续导出和采样基础。

当前公开 training、sampling 和 Evaluation 仍是单进程、单设备。支持面以
[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md) 和公开文档为准。

## 还没有支持什么

- 没有 distributed config、launcher/session、process-group lifecycle 或不可变的进程编号（rank）
  context。
- 没有感知 rank 的 `DataBuilder` 约定、分片证据、等步数或精确 Evaluation 规则。
- 没有 parallel execution rebinding、global metric reduction、step consensus 或 rank-zero side-
  effect routing。
- 没有 distributed checkpoint、rank-local RNG、sharded optimizer/EMA 或 topology resume。
- 没有全局采样分配、进程分片合并、FSDP2、多节点或弹性伸缩支持。

## 什么时候可以开始或重新审查

只有以下条件全部成立，才由 [`ROADMAP.md`](../../ROADMAP.md) 选择最小实施切片：

- 一个已通过正确性测试与正式 Evaluation 的工作负载在目标硬件上违反明确容量或
  wall-clock 预算；
- 测量证明瓶颈可由分布式执行改善，且更窄的 precision、batch、compile 或
  activation-checkpointing 调整不足；
- 指定负责人、平台、进程布局、backend、资源和验收环境；
- 用当前 lockfile 重验 PyTorch distributed、DDP 和 launcher API；
- DataBuilder sharding、execution binding、global reduction、checkpoint artifact 和 failure
  ordering 草案获得审阅；
- CPU multi-process contract tests 可进 CI，CUDA 有真实验收资源。

条件缺失时保持暂停；开始条件只授权证据指向的能力，不激活全文所有未来构想。

## 要完成哪些工作

### 建立固定进程会话并保持单进程行为一致

- 动作：冻结 workload profile，提供 immutable run context、session lifecycle、窄 collectives、
  topology/config preflight 和 rank-zero output identity。
- 原因：launcher 只创建进程；operation 需要确定的 init、routing、failure 和 teardown 语义。
- 影响范围：library entry point、device resolution、process group、logging 和 output publication。
- 交付物：world-size-one context、fixed-world session、collective adapter 和 failure taxonomy。
- 验证方法：单进程行为一致性、CPU multi-process success/failure、invalid rank env、config mismatch、
  timeout、partial failure 和 cleanup tests。
- 完成条件：未启用 distributed 时行为不变，任一进程失败都不会发布成功 run。

### 完成数据分片、全局观测与 DDP 训练

- 动作：向 DataBuilder 注入 context 并返回 sharding evidence；由 core parallelize primary model，
  再通过 Builder 窄能力重绑 execution module、构建 optimizer 并运行 DDP。
- 原因：core 不能事后替换未知 loader sampler，Strategy 也不能绕过 parallel wrapper。
- 影响范围：DataBuilderContext、TrainingPlan binding、accumulation sync、Metric、selection、EMA 和
  precision。
- 交付物：equal-step policy、exact validation coverage、sum/weight reduction、gradient-sync/
  step-consensus capabilities 和 DDP-compatible automatic loop。
- 验证方法：map-style/custom batch sampler 无重无漏；单进程/DDP 数值对照、accumulation、
  non-finite/skip consensus、best/early-stop 一致性和 external Builder substitution tests。
- 完成条件：一个 optimizer、自动 backward、唯一 trainable primary root 的范围在目标平台
  稳定；未声明可训练 root 或出现依赖 rank 的 forward 时会被明确拒绝。

### 提供分布式 checkpoint 与固定进程数恢复

- 动作：引入显式 checkpoint backend，保存 all-rank state、rank-local RNG 和 global progress，
  并从 distributed state 显式导出 portable full-state inference checkpoint；首轮只从完整 epoch
  边界恢复，不保存运行中的 DataLoader iterator 或 data cursor。
- 原因：sharded training resume 与单文件推理 artifact 的 state、I/O 和兼容语义不同。
- 影响范围：checkpoint staging/commit、optimizer/scheduler/scaler/EMA、RNG 和 restore transaction。
- 交付物：distributed resume bundle、fixed-topology restore、完整 inventory/digest 和 portable
  export。
- 验证方法：中断、损坏/缺失 shard、publish failure、事务式 restore、epoch-boundary trajectory
  一致性和 portable inference load tests。
- 完成条件：两类 artifact 名称、schema、reader 和用途清楚分离，单设备 checkpoint 不被静默
  改写。

### 增加多进程采样

- 动作：按稳定 global sample ID 分配 request，派生 rank-local random stream，让各 rank 原子
  发布 shard，由 rank zero 验证并合并 manifest。
- 原因：每 rank 重复完整 request 或 gather 大 tensor 都会破坏 identity、内存和 artifact
  completeness。
- 影响范围：sampling assignment、seed protocol、writers、shard manifest 和 failure publication。
- 交付物：replicated inference mode、rank-local shards 和可选 writer-owned aggregation。
- 验证方法：不同 world size/batch partition 下 sample IDs 无重无漏；partial rank failure 不发布
  完整 bundle；大 tensor 不经 object gather。
- 完成条件：并行只改变执行划分，不改变 SamplingBuilder contract 或正式 sample identity。

## 如何证明已经完成

- 单进程与当前 config、artifact 和 operation 行为一致。
- built-in 与外部 Builder 走同一窄 capability，没有 task-name 或 concrete-class dispatch。
- data、forward、optimizer step、metric、selection、checkpoint 和 publication 的 all-rank/
  rank-zero 顺序由 failure tests 证明。
- 支持的平台和进程布局有明确矩阵；未验证组合会被明确拒绝。
- portable inference artifact 与 distributed resume bundle 可分别验证和恢复。
- 对外支持变化同步 SPEC、ARCHITECTURE、ROADMAP、CHANGELOG、配置 reference 和用户文档。

## 明确不包含什么

- 不预建 HTTP/gRPC serving、dynamic batching、autoscaling、多租户或长驻服务。
- 首轮不包含 HSDP、tensor/pipeline/3D parallel、ZeRO/DeepSpeed/Megatron/FairScale adapter、
  自动 learning-rate scaling 或 distributed mid-epoch iterator resume；它们的独立开始条件保留在
  下节。
- 不预建 parameter server、remote actor 或通用 parallel mode enum。
- 不按模型类名、参数数量或 registry name 猜 FSDP partition。
- 不用 `all_gather_object` 搬运模型 state、大样本或大 manifest。
- 不因计划存在而承诺 Windows/NCCL、MPS、FSDP2、multi-node 或 elastic。
- 不触碰 Physics/KD；它们不是 trigger 或验收 fixture。

## 详细设计和研究资料在哪里

以下能力不属于首版 DDP 完成条件。每项必须单独满足触发证据，并由列出的负责人重新制定任务。

| 后续方向 | 重审触发 | Owner |
| --- | --- | --- |
| FSDP2 与分片状态 | 首版 DDP 已稳定，但目标工作负载仍因单设备模型状态容量失败，测量明确指向状态分片 | Distributed training、checkpoint 与模型 Builder 负责人 |
| 精确分布式 Evaluation | 一个已选择的正式协议无法在单设备预算内完成，并且需要全局样本完整性与 Metric 汇总 | Evaluation runtime 与 distributed 执行负责人 |
| 多节点、改变进程布局的恢复与 elastic | 单节点仍不满足已测量预算，并且存在真实网络环境、验收资源和进程布局变化规则 | Distributed runtime、checkpoint 与基础设施负责人 |
| 其他并行 provider 与自动学习率调整 | 原生 DDP/FSDP2 仍不能满足已证明的工作负载；每种 provider 或训练策略都有独立证据 | 对应 provider adapter 或新 training-loop family 负责人 |
| Epoch 中途恢复 | 从 epoch 边界重算违反恢复预算，并且 `DataBuilder` 能提供可版本化、可验证的 cursor | DataBuilder、training runtime 与 checkpoint 负责人 |

### FSDP2 与分片状态

- 保留范围：显式 FSDP2 sharding plan、分片后构造 optimizer、sharded EMA、PyTorch
  Distributed Checkpoint（DCP）和可搬迁 full-state inference export。
- 验证要求：覆盖 DTensor optimizer/EMA/clip/checkpoint，并保持单设备 checkpoint 语义不变。

### 精确分布式 Evaluation

- 保留范围：精确样本覆盖、全局 Metric 相等性、失败一致性和不可变结果发布。

### 多节点与改变进程布局的恢复

- 保留范围：multi-node、network failure、topology-changing resume 和 elastic；进程布局变化时的
  RNG 与数据身份必须另行定义。

### 其他并行 provider 与训练策略

- 保留范围：HSDP、TP、PP、3D parallel、ZeRO/DeepSpeed/Megatron/FairScale adapter，以及自动
  learning-rate scaling；学习率变化还需受控 batch/LR 实验。

### Epoch 中途恢复

- 保留范围：mid-epoch iterator resume、cursor 身份和序列一致性；不保存不可恢复的任意
  DataLoader iterator。

### 相关资料

- [PyTorch/provider 调研、配置/API 草案与历史映射](notes/distributed-training-and-inference-support-plan/research-and-api-draft.md)
- [根级路线图的 distributed 触发条件](../../ROADMAP.md)
- [当前架构边界](../../ARCHITECTURE.md)

附录内容必须在启动时重验。本文作为未来支持构想保留，未经维护者明确审阅不得删除。
