# 为真实 streaming workload 建立可恢复的数据生命周期

> 工作状态：暂停
>
> 当前结论：project-private iterable 可以用于普通 Python 组合；在没有真实 workload
> 证明 replay、resume 和 completeness 需求前，不向 core 增加通用 streaming runtime。
>
> 规范来源：[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 排期权威：[`ROADMAP.md`](../../ROADMAP.md)

## 完成后用户能做什么

用户可以运行一个明确获批的 streaming recipe，并知道数据 identity、重放范围、分片、
epoch、恢复 cursor 和正式 Evaluation 的样本完整性分别由谁负责。

## 当前仓库已经支持什么

- `DataLoaders` 接受普通 reiterable，不要求具体 Dataset 基类。
- Project `DataBuilder` 可以组合自己的 `IterableDataset`、worker 和 batch 结构。
- Managed/referenced `DataArtifact` 可以认证有限 snapshot 或外部文件表示。
- Distributed 与 Evaluation 已各自保留未来的 sharding 和 completeness 启动条件。

## 还没有支持什么

- snapshot-free stream 的稳定 identity；
- shuffle buffer、shard cursor、worker/rank repartition 和 mid-epoch resume；
- elastic membership 下的数据恢复；
- streaming formal Evaluation 的全局 sample completeness。

## 什么时候可以开始

必须先有一个可运行 workload，证明 project-private `IterableDataset`/`DataBuilder` 与现有
artifact snapshot 无法满足要求，并记录 replay、sharding、epoch、sample identity、失败恢复
和容量约束。预想中的 LLM 或远程流名称不足以启动。

## 要完成哪些工作

### 固定一个真实 stream 的身份和重放规则

- 动作：定义 source snapshot、record identity、epoch 边界、ordering 和允许重放的范围。
- 原因：没有这些事实就无法区分恢复、重新读取和数据漂移。
- 影响范围：DataSource、DataBuilder、artifact evidence 和 run outcome。
- 交付物：一个可运行 workload、不可变 identity 和明确失败案例。
- 验证方法：相同 identity 重放得到相同 record 集，变化的 source 明确失败或产生新 identity。
- 完成条件：不依赖任务名称即可判断 stream 是否兼容和可恢复。

### 连接 workload 实际需要的 cursor、分片和 Evaluation

- 动作：先接入选中 workload 必需的 iterator state 和 checkpoint；只有 workload 使用多 worker、
  distributed execution 或正式 Evaluation 时，才与对应 owner 增加 ownership 和 completeness。
- 原因：单进程 stream 不应被迫同时实现尚未选择的 distributed 或 Evaluation 产品范围。
- 影响范围：DataBuilder 和 checkpoint；按需涉及 distributed execution 与 Evaluation runtime。
- 交付物：最小恢复 schema 和 failure tests；按选中范围补分片证据或 completeness 规则。
- 验证方法：中断恢复与连续运行等价；若启用多 rank 则无重复/遗漏，若发布正式结果则样本集合可审计。
- 完成条件：选中 workload 的所有输入和状态 fail closed，且 portable inference artifact 不携带 iterator state。

## 如何证明已经完成

- 同一输入身份的 replay、resume 和 worker-count tests。
- 选中 workload 使用多 worker 或多 rank 时，互斥、完整性和失败恢复 tests 通过。
- 选中 workload发布正式 Evaluation 时，对 sample IDs 和缺失记录 fail closed。
- 一个独立 project recipe 无需 core 的 task-name 分支即可替换。

## 明确不包含什么

- 不承诺任意网络 stream、消息队列或数据库 provider。
- 不把所有 `IterableDataset` 自动升级为 formal artifact-backed execution。
- 不把 distributed launcher 或 collective lifecycle 放进 DataBuilder。
- 不在没有 workload 的情况下预建 universal cursor schema。

## 详细设计和研究资料在哪里

- [原始 streaming、iterable resume 与 ownership 研究](notes/data-layer-composition-boundary-review/research-archive.md)
- [Distributed 计划](distributed-training-and-inference-support-plan.md)
- [当前 Data pipeline](../configuration/data-pipeline.md)
