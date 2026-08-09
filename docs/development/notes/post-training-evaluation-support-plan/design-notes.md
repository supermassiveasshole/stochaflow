# Evaluation 后续设计附录

> 当前 standalone checkpoint Evaluation、prediction-artifact offline replay 和 epoch-end
> validation 已经实现。本文只保存暂停项的设计理由；状态与开始条件以
> [`post-training-evaluation-support-plan.md`](../../post-training-evaluation-support-plan.md)
> 为准。
>
> 最后核对：2026-08-09。

## Reference feature cache

只有同一 reference data/protocol 重复计算造成可测成本时才考虑 cache。Key 至少要绑定 data
identity、ordered sample plan、preprocessing、provider/backbone/weights、dependency version
和 metric-specific parameters。读取时必须验证完整性；cache failure 不能改变正式 protocol。

Cache 是性能实现，不是新 Evaluation subject，也不能成为结果正确性的唯一来源。

## 质量与速度证据

若某个正式 workflow 需要报告质量—速度关系，应先固定 warmup、measurement window、
sample plan、device/runtime provenance 和 uncertainty。不同硬件可以保留相同 protocol digest，
但性能结果必须带 execution provenance，不能混成任务 protocol 字段。

## Comparison、Gate、Suite 与 Reporter

普通调用方已经可以读取两个不可变结果并比较 metric。只有真实发布或模型晋升流程反复需要
同一规则时，才考虑新增窄 policy：

- Comparison 只接收兼容 protocol 的结果；
- Gate 读取结果但不重新执行 Metric；
- Suite 只组织明确列出的 Evaluation invocations；
- Reporter 只发布事实，不决定 acceptance。

任何类型都必须有真实 owner 和消费者；不恢复空 outcome 字段或无 runtime consumer 的
placeholder。

## 任务特定 profile

Paired distortion/perception、随机多次采样、latent reconstruction、text-image alignment、
超分辨率恢复或 consistency quality-speed 都由具体任务声明 data、preprocessing、provider、
sample identity 和正式 profile。通用 Evaluation schema 不预先加入这些字段。

## 外部 reporting integration

W&B、MLflow 等 adapter 只读取已发布 result/artifact。网络服务不参与 subject execution、
Metric lifecycle、result identity 或 atomic publication，也不能成为 core 的必需依赖。

## 重新讨论时必须回答

- 谁消费结果，谁拥有通过/失败规则；
- 当前应用层组合为什么不足；
- 不完整、非有限值、协议不兼容和 provider 变化时如何明确拒绝；
- schema/version/migration 怎样保持历史结果可读；
- 哪些测试证明它不是一个任务的私有字段集合。
