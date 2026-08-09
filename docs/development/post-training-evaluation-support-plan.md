# Post-training Evaluation 后续决策记录

> 工作状态：暂停
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## 完成后用户能做什么

当前仓库已经具备正式 Evaluation、离线回放和训练期 Evaluation。本记录保留五类可能的后续
改进：可验证的参考特征缓存、正式的质量/速度测量、可复用的结果比较与报告工具、任务自己的
随机或成对评估方案，以及外部报告服务集成。

五类改进彼此独立；任何一项未进入路线图，都不影响当前 Evaluation 的可用性。

## 当前仓库已经支持什么

- 独立且严格的 Evaluation 配置、checkpoint subject 的 raw/EMA 选择，以及只读 prediction
  artifact 的离线回放。
- `EvaluationBuilder -> EvaluationPlan -> Evaluator` 任务组合；core 负责 inference mode、Metric
  生命周期、精确身份与完整性、不可变结果和原子发布。
- 带版本、完整且通过身份验证的 prediction artifact；离线回放不会重跑模型。
- 训练期 validation Evaluation、冻结的 profile 身份，以及只使用 validation 结果选择 checkpoint。
- 在执行前绑定 provider、backbone、weights、preprocessing 和必要的实现/运行身份，并在结果中
  记录来源。
- FID/KID provider 与持续维护的 AFHQ-v2 分类评估方案。

当前结果对象不保留没有运行时使用方的请求、比较、通过条件、套件或报告器占位。行为与架构以
[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md) 和
[configuration workflows](../configuration/workflows.md) 为准。

## 还没有支持什么

- 没有通用的参考特征缓存。
- 没有正式的 latency/throughput/memory/NFE 测量协议或质量/速度证据包。
- 没有 core 级结果比较、通过条件、评估套件、报告规则或多 checkpoint 选择器。
- 没有为未选中的 SR、latent、consistency、codec 等任务预建评估方案。
- 没有把 W&B、MLflow 等外部报告服务作为 core 前置集成。

这些是暂停的后续改进，不是当前 Evaluation 的缺口。已经结束的实验和退役结论只属于根路线
图、变更历史与 Git，不在本文重写为待办。

## 什么时候可以开始或重新审查

某一后续改进只有在以下条件全部成立时才重开：

- [`ROADMAP.md`](../../ROADMAP.md)指定真实使用方、负责人、预期结果和成功标准；
- 当前 Evaluation 的公开文档、测试和实现彼此一致；
- 需求不能由任务自己的评估方案或应用层组合更简单地解决；
- 提案不恢复空的运行时占位，也不改变现有 subject 和完整性约定；
- 行为、架构、兼容性和发布影响都有同步计划。

没有满足开始条件时，所有后续改进继续暂停。拒绝一项不会删除其他未来构想。

## 要完成哪些工作

### 在真实成本下评审参考特征缓存

- 动作：当持续维护的评估方案重复计算参考特征成为主要成本时，设计由任务或 provider 负责、
  按内容寻址的缓存。
- 原因：缓存身份依赖 provider、weights、preprocessing 以及数据和样本事实，不能下沉到与任务
  无关的 Metric 根类型。
- 影响范围：任务的 `EvaluationBuilder`/provider、artifact 存储、协议来源和失败处理。
- 交付物：覆盖 provider/version、backbone/weights、preprocessing、data artifact/split/sample IDs
  与特征 dtype 的缓存约定。
- 验证方法：命中缓存与重新计算数值等价；缓存损坏、缺失、版本变化或只写入一部分时会被明确
  拒绝。
- 完成条件：真实评估方案获得可测收益，core Metric 不承担数据或缓存职责。

### 在真实工作流下评审性能测量

- 动作：固定 warmup、同步、重复次数、硬件、dtype、batch 和汇总方式，发布与质量结果分开的
  性能测量。
- 原因：latency/throughput/memory/NFE 只有在测量方法固定后才是正式证据，不能直接采用开发
  日志中的数字。
- 影响范围：任务评估方案、执行来源记录、结果呈现和基准文档。
- 交付物：可复现的质量/速度证据包，以及不确定值和失败值的处理规则。
- 验证方法：覆盖重复测量、非有限值、硬件变化和汇总测试；硬件身份不改变质量协议的兼容性。
- 完成条件：路线图任务能按明确协议使用这些证据，而不会把性能测量混入质量 Metric。

### 在真实发布选择需求下评审结果比较和报告工具

- 动作：先明确规则负责人、结果兼容性、结果不完整或非有限时的默认行为、schema 迁移和呈现
  边界，再决定功能属于 core、应用层还是 extension。
- 原因：如果直接比较不可变结果已经足够，就不应预建选择器运行时或基类。
- 影响范围：发布选择流程、结果读取方、规则序列化和报告输出。
- 交付物：若获批，交付最窄的可复用规则；否则记录继续使用应用层比较的结论。
- 验证方法：不同协议身份默认不可比较；通过条件不重算 Metric；报告工具不修改完整性或补造记录。
- 完成条件：至少一个真实发布流程复用该规则，结果对象不恢复空占位。

### 随未来任务交付随机或成对评估方案

- 动作：由获选任务定义冻结的 `SamplePlan`、样本身份、配对、汇总、prediction artifact/gallery
  和正式协议。
- 原因：成对失真/感知评估或重复采样属于任务语义，不是通用字段。
- 影响范围：该任务的 Training/Sampling/Evaluation Builder、数据方案和公开工作流。
- 交付物：与该任务首个完整功能一起发布的任务专用正式评估方案。
- 验证方法：training、checkpoint inference、训练期/离线 Evaluation 与 artifact 完整性同时通过，
  且不增加模态专用的 core schema。
- 完成条件：该任务能在相同协议身份下评估冻结 subject；其他任务不受影响。

### 通过可选 extension 连接外部报告服务

- 动作：让集成读取不可变 result/artifact，并向 W&B、MLflow 或其他系统发布事实。
- 原因：网络和供应商服务不能成为 core 执行 subject 或 Metric 生命周期的前置依赖。
- 影响范围：可选 extension、凭据处理、重试和呈现。
- 交付物：不决定结果是否通过的只读报告集成。
- 验证方法：未安装集成时 core 完全可用；网络失败不修改已发布的 Evaluation 证据。
- 完成条件：外部系统只呈现事实，不重算 Metric、不改变 subject/result identity。

## 如何证明已经完成

- 每项重开的改进都有真实使用方、独立测试和明确的协议与身份。
- 缓存与重新计算、训练期与离线、单次与重复测量都有明确对照。
- 不同协议摘要的结果不会被悄悄比较或判定通过。
- final test 只接受冻结 subject，不参与 checkpoint selection。
- 稳定变化同步 SPEC、ARCHITECTURE、ROADMAP、CHANGELOG 和公开文档。

## 明确不包含什么

- 不把新任务的评估方案写成通用 Evaluation 路线图的剩余阶段。
- 不为未来的结果比较预建基类、配置、结果对象或 CLI。
- 不把普通 sampling、Diagnostic 或 train/test scalar 升格为正式证据。
- 不把 device/hardware identity 误作 task batch schema 或 quality protocol identity。
- 不重复已经结束的 AFHQ-v2 实验历史。

## 详细设计和研究资料在哪里

- [Evaluation 后续设计笔记](notes/post-training-evaluation-support-plan/design-notes.md)
- [Evaluation 规范行为](../../SPEC.md)
- [Evaluation 架构边界](../../ARCHITECTURE.md)
- [公开 configuration workflows](../configuration/workflows.md)
- [根路线图中的候选与暂停工作](../../ROADMAP.md)

本文只保留未来改进构想；未经维护者明确审阅不得删除。
