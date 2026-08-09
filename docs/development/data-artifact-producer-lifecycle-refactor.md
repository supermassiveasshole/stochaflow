# DataArtifact producer lifecycle 决策记录

> 工作状态：已完成
>
> 当前结论：仓库只维护一条高层 producer lifecycle。旧 artifact 格式和旧 API
> 不构成兼容性承诺；当前行为以规范、公开文档和实现为准。
>
> 规范来源：[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 最后核对：2026-08-09

## 完成后用户能做什么

Extension 可以实现一个 `DataSource[P]`，把外部数据物化为经过验证、具有稳定
identity 的 `DataArtifact[P]`。训练、采样或 Evaluation 的 `DataBuilder` 随后绑定
该 artifact，并组合本次运行需要的数据视图。

```text
DataSource[P]
    -> DataArtifactStore
    -> DataArtifact[P]
    -> DataBuilder
    -> DataLoaders
```

## 当前仓库已经支持什么

- `DataSource` 是 producer entrypoint；它负责 acquire/read/validate/materialize。
- `DataArtifactStore` 负责受治理的物化目录、manifest 发布和 cache reuse。
- `DataArtifact` 提供稳定 identity、payload contract 和验证后的内容边界。
- Cache 命中必须重新验证 manifest、producer identity、schema 和内容要求；不能只看
  目录是否存在。
- `DataBuilder` 消费 artifact 并构造 runtime Dataset、split、sampler、collate 和
  loaders；producer 不拥有这些对象。
- Built-in 和 extension 走相同 lifecycle，没有 core-only shortcut。
- 当前 schema 与旧 artifact/API 是 breaking transition；旧格式需要显式迁移或重建。

完整使用方式见 [Data pipeline](../configuration/data-pipeline.md) 和
[Extension API](../api/extensions.md)。

## 还没有支持什么

- 组织级 artifact catalog、metadata warehouse 或容量调度；
- remote object store 的通用实现；
- 任意 payload 的 universal schema；
- 从 artifact 自动推断 Dataset、task 或 DataBuilder；
- 对旧 schema、旧路径布局或旧 cache key 的永久兼容。

这些不是当前 lifecycle 的未完成步骤。共享 metadata、provenance 或 capacity 的未来支持由
[独立暂停提案](artifact-metadata-provenance-capacity-model-proposal.md)负责。

## 什么时候需要重新讨论

只有现有 lifecycle 无法表达新的持久化需求时才重开本记录，例如：

- 第二个独立消费者需要跨 run 查询相同的共享 metadata contract；
- remote store adapter 需要稳定、provider-neutral 的窄接口；
- 新 payload family 证明当前 semantic payload contract 过于 storage-specific；
- cache 或 manifest failure 无法通过现有 read-boundary validation 安全处理。

重开时必须提供可运行消费者和失败案例，不能只根据预想字段扩展 schema。

## 如何证明当前决策仍然成立

- producer identity、manifest schema、atomic publication 和 cache validation tests；
- corrupted/missing/stale artifact failure tests；
- built-in 与独立 extension 的生命周期一致性测试；
- compatible DataSource/DataBuilder substitution tests；
- `SPEC.md`、`ARCHITECTURE.md`、公开文档和配置参考保持一致。

## 明确不包含什么

- 不在开发记录中复制完整公共 API reference。
- 不维护会随测试增长而过期的用例数量。
- 不根据一个项目的私有 metadata 扩大 core schema。
- 不在本轮决定任何具体 example project 的保留或支持级别。

## 详细设计和研究资料在哪里

- [原始 API、manifest、迁移和验收明细](notes/data-artifact-producer-lifecycle-refactor/implementation-archive.md)
- [Data 层组合边界决策](data-layer-composition-boundary-review.md)
- [统一 metadata/provenance/capacity 候选](artifact-metadata-provenance-capacity-model-proposal.md)
