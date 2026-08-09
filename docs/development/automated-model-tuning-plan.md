# 自动化模型调优计划

> 工作状态：暂停
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## 完成后用户能做什么

HPO 指超参数优化。根路线图选择本计划后，用户可以固定一份完整训练配置，声明允许尝试的参数
范围、只使用 validation 的目标和总预算。系统会依次运行彼此隔离的尝试，记录每次输入、结果和
停止原因，并能继续同一次搜索。

首版只支持一个目标、单机顺序执行、Grid/Random 搜索、每个 epoch 报告和基础剪枝。最终候选
仍需通过独立的正式 Evaluation；测试集结果不会反馈给搜索。自适应搜索、并行和外部调度保留在
文末，不属于首版完成条件。

## 当前仓库已经支持什么

- 单次 training operation 会发布 resolved config、checkpoint、manifest 和不可变 outcome。
- TrainingBuilder、Trainer、Metrics、checkpoint、extension provenance 和 standalone
  Evaluation 已有明确所有权。
- canonical validation observation 可以作为未来 objective 的基础；test、Diagnostic 和普通
  sampling output 都不是 selection/pruning 证据。
- extension selection 是进程级固定状态，可作为后续进程隔离的约束。

这些基础不等于仓库已有 HPO runtime。当前行为以 [`SPEC.md`](../../SPEC.md)、
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) 和公开文档为准。

## 还没有支持什么

- 没有公开 tuning config、study controller、search-space contract 或 provider adapter。
- 没有脱离 CLI 的稳定训练调用入口与窄的 epoch observation/control 接口。训练库调用入口由
  [内置操作与工作流组合计划](default-workflow-pipeline-support-plan.md)负责；HPO 只消费该入口，
  不另建训练调用面。
- 没有 pruning、study storage、resume fingerprint、trial failure taxonomy 或候选选择操作。
- 没有冻结的目标、预算和正式 Evaluation 协议作为真实 HPO 首个完整结果。
- 没有 Grid/Random correctness oracle，也没有自适应搜索、并行或统计确认支持。

## 什么时候可以开始或重新审查

只有以下条件全部成立，才由 [`ROADMAP.md`](../../ROADMAP.md) 把本记录提升为实施计划：

- 指定真实任务、负责人、支持平台、资源预算和首个交付结果；
- [内置操作与工作流组合计划](default-workflow-pipeline-support-plan.md)已交付并验证普通 CLI 与
  Python 调用共用的单次训练入口；HPO 只在其上接入 observation/control adapter；
- 搜索目标、优化方向、预算和正式 Evaluation 协议已固定；
- 使用当前 installed environment 重新完成 provider/version/license/optional-dependency 调研；
- study config、resume fingerprint、failure taxonomy 和 artifact layout 获得审阅；
- 先批准 Grid/Random correctness baseline，再讨论自适应搜索或并行。

条件不完整时保持暂停，不以循环调用 CLI、解析日志或复制 `Trainer.fit()` 代替这些门槛。

## 要完成哪些工作

### 消费公共训练入口并固定目标与预算

- 动作：消费[内置操作与工作流组合计划](default-workflow-pipeline-support-plan.md)交付的不可变
  单次训练入口；HPO 只增加读取规范 validation 结果的 observation/control adapter，并固定
  搜索目标、优化方向、每个 epoch 的资源计数和单次/整体预算。
- 原因：provider 必须消费稳定 evidence，不能从日志、test 或 Diagnostic 恢复训练语义。
- 影响范围：HPO trial adapter、epoch observation、outcome 和 failure taxonomy；不改变训练入口
  的所有权。
- 交付物：对公共训练入口的窄调用 adapter、继续/剪枝决定，以及
  completed/pruned/failed/cancelled 语义。
- 验证方法：普通 CLI 与 fake HPO adapter 走同一单次训练调用；若 Hydra 已实现，它也只能消费
  同一入口，但不是 HPO 的前置。非有限、缺失、重复 report 和 publication failure 均有回归测试。
- 完成条件：工作流计划拥有并验证公共训练入口，HPO 作为另一个 Python 使用方可安全调用它，
  且没有 HPO 专用训练分支。

### 建立独立搜索配置与顺序 Grid/Random

- 动作：新增独立 study config，冻结 base config，以受控 JSON Pointer patch 生成 trial config，
  并通过窄 adapter 接入重新选定的成熟 provider。
- 原因：搜索配置与训练配置必须分开；Grid/Random 为参数范围、seed 和恢复行为提供明确对照。
- 影响范围：study parser、fingerprint、trial identity、provider adapter 和 artifact layout。
- 交付物：float/int/categorical/finite-grid domains、单并发 controller、study resume 和 trial
  lineage。
- 验证方法：未知字段、重复或无权修改的目标、类型错误、extension/output/seed 修改、provider
  失败和研究问题变化都会被明确拒绝。
- 完成条件：相同 frozen study 与 seed 可重建 suggestion 序列，或明确记录 provider 的
  非确定性边界。

### 选择最终候选并运行正式 Evaluation

- 动作：由搜索规则发布最佳尝试记录，显式选择 checkpoint，再运行一次独立正式 Evaluation；
  若需要合并 train+validation 重新训练，由具体 `DataBuilder` 另行提供明确 recipe/config，HPO
  只绑定并记录它。
- 原因：Evaluation runtime 不拥有 study selection/acceptance；test 不得反馈给 search。
- 影响范围：搜索结果、artifact pointers、候选选择命令、可选 DataBuilder recipe identity
  和 Evaluation invocation。
- 交付物：不可变 best-trial record、非覆盖式 checkpoint pointer 和 protocol-compatible final
  result；启用 refit 时另含具体 DataBuilder 拥有的 resolved recipe/config。
- 验证方法：不同 protocol identity 不会被静默排序；test result 不产生 suggestion；候选选择
  不覆盖原 trial artifact；未声明 recipe 时不自动合并 train/validation。
- 完成条件：搜索证据、每次尝试的 artifacts、候选选择规则和最终 Evaluation 可分别恢复与
  审计；可选 refit 的数据组合由具体 DataBuilder 负责并可独立复现。

## 如何证明已经完成

- plain training 在未安装 HPO extra 时完全可用。
- built-in 与外部 adapter 走同一 contract，没有 provider-name core dispatch。
- trial 不共享 model、optimizer、scheduler、EMA、Metric、Diagnostic 或 RNG mutable state。
- resume 不改变 base config、search space、objective、protocol 或 extension identity。
- validation-only selection、pruning、failure、候选选择和最终 Evaluation 的职责由测试证明。
- 对外变化同步 SPEC、ARCHITECTURE、ROADMAP、CHANGELOG、配置 reference 和用户文档。

## 明确不包含什么

- 不承诺完整 AutoML、自动清洗、特征工程、NAS、ensemble 或 deployment。
- 不支持 population-based training、manual backward、多 optimizer 或新 training-loop family。
- 不自动合并 train/validation，不自动选择 test，不修改 DataBuilder 的 task semantics。
- 不把 runtime resource、study/replication seed 伪装成普通 model hyperparameter。
- 不把 provider API 草案、版本或 artifact layout 当作当前兼容承诺。

## 详细设计和研究资料在哪里

以下能力不属于首版完成条件。每项只能在自己的触发证据成立后重开。

### 自适应搜索、统计重复、约束和多目标

- 触发证据：顺序 Grid/Random、参数范围、预算、噪声和恢复语义已经稳定，且真实任务证明固定
  搜索不足。
- 负责人：HPO provider adapter 与 `TuningBuilder` 维护者；通用 black-box optimizer 仍由独立
  BO 仓库拥有。
- 保留范围：TPE、GP-based Bayesian Optimization、provider scheduler/pruner、统计重复、约束、
  多目标、各类 seed、pruned checkpoint 保留和缺失/非有限/部分失败规则。
- 验证要求：TPE 不冒充 GP-BO；sampler 与 pruner 职责分开；与 trial 内 early stopping 冲突时
  明确拒绝；原始参数分配和运行来源不被汇总覆盖。

### 本地多进程

- 触发证据：顺序执行已经稳定，实际 wall-clock 测量证明并行收益足以承担隔离成本。
- 负责人：HPO controller 与资源分配维护者。
- 保留范围：进程 worker、并发上限、CPU/GPU 分配、重试、孤儿进程清理和确定的 artifact 路由。
- 验证要求：覆盖设备冲突、worker 崩溃、controller 重启、SIGINT 和部分发布；并行只改变吞吐，
  不改变搜索目标、尝试身份或 artifact 身份。

### Cluster 或 remote launcher

- 触发证据：本地多进程仍不能满足已测量预算，并且存在真实集群使用方、资源与验收环境。
- 负责人：独立 launcher/infrastructure 集成负责人；HPO 只提交隔离的单次尝试。
- 保留范围：cluster launcher、remote launcher 和外部执行状态绑定；不隐式启用 distributed
  training。

### 相关资料

- [Provider 调研、配置/API 草案与历史映射](notes/automated-model-tuning-plan/research-and-api-draft.md)
- [Post-training Evaluation 后续决策](post-training-evaluation-support-plan.md)
- [根级路线图的 HPO 触发条件](../../ROADMAP.md)

附录中的 provider、版本、schema 和 API 必须在启动时重验。本文作为未来支持构想保留，
未经维护者明确审阅不得删除。
