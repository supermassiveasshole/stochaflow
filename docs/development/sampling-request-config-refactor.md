# Sampling 调用：Hydra 迁移后的复审

> 工作状态：暂停
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## 完成后用户能做什么

如果 Hydra 迁移后的证据表明当前完整 sample config 难以编写，或 Python 调用接口不足，用户
可以获得最窄的配置编写、选项参考或调用能力。sample 配置仍保持独立、完整且可审计。复审也
可以得出“维持现状”的结论。

Hydra 配置迁移完成后不会自动产生这些能力；本复审必须由根路线图另行选择。

## 当前仓库已经支持什么

- `train` 与 `sample` 是两个独立操作，各自使用完整配置。
- `sample` 显式接收 checkpoint 与完整 sample config；二者不合并，sample seed 不继承
  training seed。
- checkpoint 提供可搬迁的只读 inference projection、extension 来源信息和任务专用
  fixed sampling recipe。
- sample config 声明 sampler、任务专用 options、shape、count、batch size、seed 和 writers，
  但不能覆盖 checkpoint 固定的 SamplingBuilder identity/contract。
- core 校验每批与总 exact count；runtime 绑定 checkpoint identity，在 private staging 运行
  writers，并只发布完整 bundle。
- direct transform 可由 SamplingBuilder 直接返回 output，不必伪造 Sampler、Dynamics 或 Process。

当前行为以 [`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md)、
[configuration workflows](../configuration/workflows.md) 和
[compatibility and migration](../configuration/compatibility-and-migration.md) 为准。

## 还没有支持什么

- 没有证据证明 sample 配置需要类似 Hydra 的组合方式。
- 没有统一的任务专用选项参考或 schema 能力。
- 没有为未来多资产 inference 结构扩展 recipe envelope。
- 没有第二个真实使用方证明需要独立的不可变调用对象或不带 writer 的接口。
- 没有获批的公共 request/checkpoint 兼容性变化。

这些问题尚未被证明确实需要解决；它们是复审问题，不是当前待办。

## 什么时候可以开始或重新审查

必须同时满足：

- [Hydra 全新训练配置迁移](hydra-configuration-composition-migration-plan.md)已经完成配置组合、
  单次训练行为对照和易用性验收；
- [`ROADMAP.md`](../../ROADMAP.md)另行指定本复审、负责人和问题陈述；
- 当前 sampling 公开文档、测试和实现彼此一致；
- 至少两个独立使用方提出同一种配置编写、inference 结构或 Python 调用缺口；
- 提案继续分开 checkpoint、sample、Builder、writer 和 extension 的职责；
- 公共变化有完整用户结果、版本和迁移方案。

任一条件缺失时继续暂停。Hydra 完成只满足一个前置，不会自动重开。

## 要完成哪些工作

### 收集并分类复审证据

- 动作：从仓库维护的配置和真实外部项目中收集重复配置、难懂错误、任务选项、inference 结构
  和 Python 调用问题，同时评估维持现状是否更合适。
- 原因：文件复制、文档发现性和真正的协议缺口需要不同解法。
- 影响范围：复审记录、sample configs、外部 extension 反馈和维护测试。
- 交付物：按问题类型整理的证据，以及修改或不修改的建议。
- 验证方法：至少两个独立使用方证明同一缺口；单一未来任务不足以改变 core 约定。
- 完成条件：能够结束没有证据的问题，只为反复出现的问题建立后续工作。

### 评审 sample 配置是否需要组合

- 动作：若结构性 YAML 重复被证明造成实际问题，评估只组合配置片段、最终仍生成完整基础类型
  sample config 的方案。
- 原因：checkpoint 不能参与合并；简化配置编写也不能恢复不完整的 request。
- 影响范围：sample config 编写、最终配置预览和组合来源审计。
- 交付物：若获批，交付最小配置组合方案；否则记录继续使用完整普通 YAML 配置的结论。
- 验证方法：最终 request 完整可审计，不继承 training/checkpoint 的可变默认值，不开放任意
  Python 对象构造。
- 完成条件：至少两个使用方的维护成本下降，且运行时 request 不产生两套事实来源。

### 评审任务专用选项是否容易发现

- 动作：若外部 `SamplingBuilder` 的校验信息和选项参考确实不足，评估 recipe 自带 schema 或
  自动生成选项参考的能力。
- 原因：文档问题不能通过把模态专用字段提升为通用 core schema 来解决。
- 影响范围：`SamplingBuilder` 约定、extension 文档、错误路径和配置参考。
- 交付物：可选的任务专用参考能力，不增加 conditioning/prompt/codec core 字段。
- 验证方法：独立 extension 可新增选项而不修改 core dispatch；与固定约定冲突时会被明确拒绝。
- 完成条件：选项更容易发现，且 extension 不需要修改 core dispatch。

### 评审新的 inference 资产结构

- 动作：只有被选中的完整任务证明现有 asset projection/recipe 不足时，才设计最窄的版本化
  asset/recipe extension。
- 原因：不能为未选中的 latent、SR 或多资产任务预建通用字段。
- 影响范围：checkpoint projection、sampling recipe、Builder、artifact 和正式 Evaluation。
- 交付物：由真实任务驱动的版本化 envelope，或维持现状的结论。
- 验证方法：内置实现与真实 extension 同时通过 checkpoint、sampling、artifact 和 Evaluation；
  shape 等通用字段也重新验证。
- 完成条件：完整任务可用，core 不按任务或 registry name 分支。

### 向操作入口 owner 提交 sampling 专用需求

- 动作：若第二个使用方需要 sampling 专用的不可变 request 或不带 writer 的窄能力，记录需求并
  交给[显式顺序工作流计划](default-workflow-pipeline-support-plan.md)拥有的 operation library
  入口；本文只复审 sampling-specific request、capability 和兼容性。
- 原因：公共 Python operation 入口和 train-to-sample 组合只有一个 owner；sampling 复审不能
  建立第二套入口，也不能把执行、失败处理或输出职责塞回 train。
- 影响范围：sampling request identity、writer-free capability 建议和 checkpoint/config
  兼容性；operation request/result 与组合示例仍归工作流计划。
- 交付物：提交给 operation owner 的 sampling 专用需求，或“当前入口已经足够”的结论。
- 验证方法：若需求获批，CLI 与 workflow-owned Python API 的身份、失败顺序和 artifact 发布一致；
  旧输入不会被静默猜测，也不维护两套事实来源。
- 完成条件：sampling 复审不导出自己的公共 operation entry；显式 train-to-sample 组合只在工作流
  计划中定义，任何公共变化同步 SPEC/ARCHITECTURE/ROADMAP/CHANGELOG 与公开文档。

## 如何证明已经完成

- 证据明确支持修改，或明确支持不修改并结束当次复审。
- checkpoint mutable defaults、partial request、train/sample merge 和 auto final sample 不回归。
- public Builder selection、arbitrary `_target_`/class/import path 继续被阻断。
- 内置实现、真实 extension、checkpoint inference、artifact 和正式 Evaluation 同时通过。
- compatibility、version、migration 和 provenance 对任何 public change 都完整可审计。

## 明确不包含什么

- 不把 Hydra fresh-training 配置直接扩展到 sample。
- 不把当前维护问题包装成新的架构工作项。
- 不按 registry name、concrete class 或 payload shape 推断任务语义。
- 不为一个未来任务增加通用 `Process`、`Dynamics`、`Sampler` 或 batch API。
- 不改动 Evaluation、training、Physics/KD 或未选中的未来任务。

## 详细设计和研究资料在哪里

- [Hydra 完成后的 sampling 重审笔记](notes/sampling-request-config-refactor/review-notes.md)
- [显式顺序工作流与唯一 operation library owner](default-workflow-pipeline-support-plan.md)
- [Hydra Fresh-Training Composition Migration](hydra-configuration-composition-migration-plan.md)
- [当前 configuration workflows](../configuration/workflows.md)
- [当前 compatibility and migration](../configuration/compatibility-and-migration.md)
- [Sampling 与 inference 架构权威](../../ARCHITECTURE.md)

旧 request 迁移细节保存在公开兼容文档、`CHANGELOG.md` 和 Git 历史。本文保留未来支持构想，
未经维护者明确审阅不得删除。
