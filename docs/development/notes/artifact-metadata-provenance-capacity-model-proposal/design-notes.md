# Artifact metadata、provenance 与 capacity 设计附录

> 当前 run-local identity、manifest 和 artifact evidence 已足够支持维护中的操作。
> 本文保存更广共享模型的边界问题；状态以
> [`artifact-metadata-provenance-capacity-model-proposal.md`](../../artifact-metadata-provenance-capacity-model-proposal.md)
> 为准。
>
> 最后核对：2026-08-09。

## 三类问题不能绑成一个项目

### Metadata

描述 artifact 本身的稳定事实，例如 schema、logical type、shape、codec 或 task-neutral
properties。只有两个独立消费者需要相同字段且当前 manifest 无法表达时才考虑共享 contract。

### Provenance

描述 artifact 怎样产生，例如输入 identity、producer implementation/version、config digest
和上游 artifact references。当前 run manifest 已保存操作所需来源；只有跨 run 查询或外部
消费者证明重复语义时才扩大。

### Capacity

描述执行资源和测量，例如设备、memory、throughput、batch/accumulation 和限制。Capacity
会随硬件、依赖和 workload 变化，不应混入 artifact semantic identity。只有两个真实容量
规划消费者需要同一查询模型时才考虑共享服务。

## 共同约束

- 新字段有明确 owner、producer 和 consumer；
- 稳定语义不能从文件名或目录布局推断；
- schema/version/migration 和敏感信息规则先确定；
- remote storage、tracking backend 或 scheduler 继续由外部系统拥有；
- 一个 showcase 或历史项目不能单独证明 organization-wide abstraction。

## 候选实现顺序

先扩展现有 manifest 或提供 read-only adapter，验证消费者。只有重复查询、索引、权限和
retention lifecycle 已经真实出现，才讨论 repository/service。Capacity 如需独立模型，应与
metadata/provenance 分开批准和实施。

## 需要拒绝的设计

- 把所有可能字段提前加入一个 universal artifact base class；
- 为一个 provider 复制对象存储、tracking 或 scheduler API；
- 用设备测量改变 artifact 内容 identity；
- 让可选 organization service 成为本地训练、采样或 Evaluation 的必需依赖。
