# Data 层组合边界决策记录

> 工作状态：已完成
>
> 当前结论：`DataSource` 负责产生经过验证的 `DataArtifact`；`DataBuilder`
> 负责把 artifact、Dataset view、split、sampler、collate 和 loader 组合成一次
> runtime 使用的数据。这个边界已经进入规范、公开文档、实现和测试。
>
> 规范来源：[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 最后核对：2026-08-09

## 完成后用户能做什么

Extension 作者可以先判断自己真正要改变的是哪一层：

- 新的数据获取或存储表示：实现 `DataSource`，产出兼容的 `DataArtifact`。
- 一次实验中的 Dataset、split、transform、sampler、collate 或 loader 组合：使用
  Python 组合，或者在需要配置选择和复用时实现 `DataBuilder`。
- 新的 batch 结构：留在具体任务的 DataBuilder、Strategy、SamplingBuilder 或
  Evaluator 内，不把字段加入 core 通用 schema。

用户可以从配置路径或 direct Python 路径使用这些边界。两条路径的差别已经在
[Data pipeline 文档](../configuration/data-pipeline.md)中说明。

## 当前仓库已经支持什么

```text
external data
    -> DataSource
    -> verified DataArtifact
    -> DataBuilder
    -> Dataset views / partitions / samplers / collate / loaders
```

- `DataSource` 获取、读取、验证、转换并物化外部数据；它不构造 runtime Dataset
  或 DataLoader。
- `DataArtifact` 保存验证后的内容、稳定 identity、manifest 和语义 payload。
- `DataBuilder` 在看到完整任务配置后选择 source、绑定 artifact identity，并组合
  runtime data objects。
- Core 把 batch 当作结构化 `Any`，不要求图像、condition、target、sample key 或
  metadata 字段。
- 同一语义 artifact 可以被不同兼容 DataBuilder 使用；同一个 DataBuilder 也可以
  接受满足其契约的其他 DataSource。
- Source-only extension 的兼容范围、direct Python 使用方式和配置路径都已经进入
  [Extension API](../api/extensions.md)、[Data pipeline](../configuration/data-pipeline.md)
  和 [Troubleshooting](../configuration/troubleshooting.md)。

## 还没有支持什么

以下项目不是当前公共能力，也不是这个已完成决策的遗漏：

- 任意 Dataset、transform、sampler、collate 或 DataLoader 的全局 registry；
- 用 YAML 描述任意 Python 数据图；
- core 自动检查 Dataset 具体类并推断任务语义；
- 通用 artifact-to-Dataset adapter registry；
- 无边界 streaming data runtime；
- distributed sharding、elastic membership 或集群调度；
- 组织级 metadata、provenance 或 capacity service。

## 什么时候需要重新讨论

当前决策保持关闭。只有出现下面的真实证据时才建立新的候选计划。

### 多个项目重复同一种函数注册需求

- 动作：比较至少两个独立 extension 的调用签名、identity、序列化和失败语义。
- 原因：单个项目的 helper 不足以证明需要公共 function registry。
- 影响范围：可能新增窄的 DataBuilder construction helper；不自动注册任意函数。
- 交付物：两个消费者的 contract tests 和一份独立 API decision。
- 验证方法：第三方实现无需 core 分支即可替换。
- 完成条件：公共字段具有相同稳定含义，并能保存 provenance。

### 真实 streaming workload 无法使用现有 materialization

- 动作：记录数据可重放性、分片、epoch 边界、sample identity、失败恢复和容量约束。
- 原因：streaming 会同时影响 data、distributed、Evaluation completeness 和恢复语义。
- 影响范围：先由具体项目拥有；需要共享 lifecycle 时再决定 core 边界。
- 交付物：可运行 workload、失败案例和 owner 分析。
- 验证方法：相同输入身份下的重放、分片互斥和样本完整性测试。
- 完成条件：需求不能由 project-private IterableDataset/DataBuilder 满足。

### Distributed 训练要求统一 sharding 行为

- 动作：与 distributed 支持计划共同定义 rank、epoch、sampler state 和 resume。
- 原因：data 层不能脱离 execution lifecycle 单独承诺 distributed correctness。
- 影响范围：DataBuilder 只声明或组合必要 policy；launcher/collective 仍在外部。
- 交付物：单独批准的 distributed data contract。
- 验证方法：多 rank 无重复/遗漏、resume parity 和 Evaluation completeness tests。
- 完成条件：已有测量证明单设备或 project-private 组合不能满足需求。

### 多种存储表示重复需要相同 adapter

- 动作：核对至少两个 storage representation 是否共享稳定的语义 artifact contract。
- 原因：按存储格式建立全局 adapter registry 会把 storage 偶然性变成 core API。
- 影响范围：优先 common helper；只有两个独立消费者稳定后才考虑公共 contract。
- 交付物：复用证据、兼容矩阵和拒绝路径。
- 验证方法：独立 implementation substitution tests。
- 完成条件：adapter 不需要读取任务名称或 modality-specific config。

## 如何证明当前决策仍然成立

- `DataSource` isolation 和 artifact validation tests。
- compatible/incompatible source substitution tests。
- independent `DataBuilder` contract tests。
- split、sampler、collate、loader 和 epoch state tests。
- direct Python 与 config-driven 文档示例。
- `SPEC.md`、`ARCHITECTURE.md`、公开文档和实现的职责描述一致。

## 明确不包含什么

- 本记录不决定任何具体 example project 的保留或支持级别。
- 不因未来可能出现 streaming、LLM 或 distributed workload 而预先扩大 core。
- 不把 `DataBuilder` 解释成“每一种 Dataset 都必须有一个 Builder”。
- 不把 `DataArtifact` 解释成 runtime Dataset 或 loader cache。
- 不让开发记录成为现行 API 的唯一来源。

## 详细设计和研究资料在哪里

- [原始仓库审查、成熟框架比较、候选 API 和迁移记录](notes/data-layer-composition-boundary-review/research-archive.md)
- [DataArtifact producer lifecycle 决策](data-artifact-producer-lifecycle-refactor.md)
- [统一 metadata/provenance/capacity 候选](artifact-metadata-provenance-capacity-model-proposal.md)
- [Distributed 支持计划](distributed-training-and-inference-support-plan.md)
