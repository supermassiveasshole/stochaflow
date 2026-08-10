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

旧提案还保存过 source citation/license、输入引用、materialization transformation sequence，
以及 code/config/environment snapshot 等候选事实。它们的用途不同：citation/license 帮助
用户核对来源和使用条件；transformation sequence 解释数据经历了什么处理；code、config 和
environment snapshot 帮助重现实验。任何一项都需要明确 writer、reader、敏感信息规则和
版本边界，不能一起塞进一个任意字典，也不能因为有记录就自动改变 artifact identity。

### Capacity

Capacity 必须继续区分三类信息：

- **artifact footprint：**样本数、存储字节数、原始尺寸等 artifact 自身占用；
- **workload estimate：**在运行前估计的 host/GPU memory、I/O 或 compute；
- **runtime observation：**某次运行在明确机器和配置上的实测 peak、throughput 和耗时。

后两类会随模型、硬件、依赖和 batch/accumulation 变化，不应混入 artifact semantic
identity。估算不能冒充测量，同一个 artifact 也不能因为换了机器就获得新的内容 identity。
只有真实容量规划消费者需要同一查询模型时，才考虑共享 contract 或服务。

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
