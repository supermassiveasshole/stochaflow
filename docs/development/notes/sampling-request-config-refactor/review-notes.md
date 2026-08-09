# Sampling 调用方式复审附录

> 本文保存 Hydra 完成后可能需要回答的设计问题。当前 sampling v12 行为已经稳定，
> 以 [`sampling-request-config-refactor.md`](../../sampling-request-config-refactor.md)
> 和规范文档为准。
>
> 最后核对：2026-08-09。

## 当前不可回退的事实

- Training 和 sampling 使用两份完整配置；sample seed、count、batch size、sampler 和 writers
  不从 training config 继承。
- Checkpoint 固定 SamplingBuilder identity、任务语义 contract 和显式 inference assets。
- Runtime 绑定 checkpoint digest/progress identity，验证总样本数，在 private staging 中
  运行 writers，并原子发布完整 bundle。
- SamplingBuilder 解释任务输入、初始化并按 count/batch size 分批，返回 writer-ready output。
- Direct transform 可以返回合法 output，不需要伪造 Process、Dynamics 或 Sampler。

旧 partial-request 设计不能复活为隐式 merge 规则。

## Hydra 完成后先收集什么证据

- 用户是否因为训练和采样字段重复而真正出错，而不是只觉得文件较长；
- library 调用方是否需要一个不可变 invocation value；
- 第二个任务是否需要相同的 options 可发现性；
- writer-free execution 是否已有 sampling 之外的第二个消费者；
- 多 inference asset 是否已经不能由现有 checkpoint recipe 表达。

没有这些证据时，合法结论是维持完整 sample config。

## 候选问题

### 是否需要稳定的 library invocation

若 public facade 需要 `run_sampling` 等稳定入口，必须明确 request/result、destination、
错误、raw/EMA 选择、extension plan 和并发规则。CLI 应只解析参数并调用该入口。

### 如何让任务 options 可发现

可以考虑由具体 SamplingBuilder 提供 JSON-safe reference/schema，但不能把 prompt、
conditioning、codec、tile 或其他任务字段加入 core sampling config。

### 是否需要独立不可变 invocation

只有多个调用方需要保存、比较或重放“同一个采样调用”时才增加类型。否则完整 config
已经是可持久化输入，不需要第二份重复对象。

### 如何复用 writer-free execution

Evaluation 已能在自己的计划内调用不发布普通 sampling bundle 的执行路径。若新增第二个
消费者，共享边界仍应只覆盖执行和输出验证，不能把 Evaluation policy 带回 sampling。

### 多资产推理如何演进

新的 inference asset 必须由 checkpoint recipe 显式声明和版本化。不能通过检查模型具体类、
任务名称或目录内容恢复资产语义。

## 复审输出

复审必须明确选择：维持现状、只改善文档/authoring、提升 public library entry，或增加一个
有第二消费者证据的窄 contract。每个 public change 都要有 migration、independent extension
test 和完整任务交付，不能只发布类型定义。
