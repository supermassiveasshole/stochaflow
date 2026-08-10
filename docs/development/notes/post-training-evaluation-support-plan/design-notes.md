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

旧计划还提出过独立的 `SelectionPolicy` / `SelectionRecord`。这个方向后来被明确替代：

- 训练中的 checkpoint 选择只读取正式 validation observation；
- HPO 由自己的 study policy 选择 trial；
- 发布流程由调用 Evaluation 的应用读取不可变结果后作决定。

Evaluation 只负责产生可核对的事实，不再建立第二套“替用户选模型”的 runtime。因此旧
Selection 不是尚待补做的 Evaluation 功能，也不应因为保留历史编号而重新进入排期。

## 任务特定 profile

Paired distortion/perception、随机多次采样、latent reconstruction、text-image alignment、
超分辨率恢复或 consistency quality-speed 都由具体任务声明 data、preprocessing、provider、
sample identity 和正式 profile。通用 Evaluation schema 不预先加入这些字段。

## 外部 reporting integration

W&B、MLflow 等 adapter 只读取已发布 result/artifact。网络服务不参与 subject execution、
Metric lifecycle、result identity 或 atomic publication，也不能成为 core 的必需依赖。

## 旧计划中其他未排期的集成构想

`5c75a76` 的最后一组设想还包括下面几项。它们只是研究入口，不表示 Evaluation 基础功能
没有完成，也不表示这些名称已经得到兼容承诺：

- **HEIM、GenEval、T2I-CompBench 等 benchmark extension：**只有对应文本生成任务被
  维护、数据与 provider 许可可核对，并且具体 extension 愿意拥有预处理和结果解释时才
  重新调研。通用 Evaluation 不复制这些项目的全部 schema。
- **人工评价的 artifact 与导出：**将来若真实评审流程需要，可以把已冻结的样本、问题、
  匿名回答和汇总结果保存成可审计 artifact。收集平台、人员管理、隐私和伦理审查不由
  core Evaluation 自动承担。
- **精确的 distributed Evaluation：**由多设备计划负责样本分配、全局统计和完整性；只有
  该执行方式被单独选择并验收后，Evaluation 才消费它提供的窄能力。
- **把 inference bundle 作为 Evaluation subject：**先要有稳定、可移植的 inference bundle
  及真实评估调用方，再单独决定 subject identity 和加载规则。当前正式 subject 仍只有
  checkpoint 与完整 prediction artifact。
- **更丰富的置信区间和结果比较 provider：**只有一个真实发布流程反复需要同一统计规则，
  并能冻结重复采样、缺失值和协议兼容性时才考虑；普通调用方现在仍可直接读取结果。

这些构想分别依赖具体任务、多设备执行或工作流资产。保存于此只是为了不让原始研究方向
只剩 Git 历史，不会重新打开一个总括性的“Evaluation 后续计划”。

## 重新讨论时必须回答

- 谁消费结果，谁拥有通过/失败规则；
- 当前应用层组合为什么不足；
- 不完整、非有限值、协议不兼容和 provider 变化时如何明确拒绝；
- schema/version/migration 怎样保持历史结果可读；
- 哪些测试证明它不是一个任务的私有字段集合。
