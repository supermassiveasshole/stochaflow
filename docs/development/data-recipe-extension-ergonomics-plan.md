# 简化 Data recipe 扩展的重复代码

> 工作状态：暂停
>
> 当前结论：Python 组合和 recipe-level `DataBuilder` 是当前稳定入口。只有多个独立
> extension 重复完全相同的构造与状态规则时，才提炼更窄的 helper 或注册形式。
>
> 规范来源：[`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 排期权威：[`ROADMAP.md`](../../ROADMAP.md)

## 完成后用户能做什么

Extension 作者可以复用经过真实项目验证的窄 helper，减少重复的 recipe 构造、artifact
binding 或 deterministic sampler 代码，同时继续由 Python 明确表达任务数据组合。

## 当前仓库已经支持什么

- `DataBuilder` 是配置驱动的 recipe composition root，并通过 Registry 构造。
- Direct Python 可以直接组合 Dataset、sampler、collate 和 iterable。
- `DataBuilderContext` 提供 seed、strict-resume expectations 和通用 source context。
- Family-local Registry 可以选择 project-owned `DataSource`，不需要全局数据 schema。

## 还没有支持什么

- function-based recipe registration；
- 面向 extension 的公共 deterministic sampler、binding 或 config helper；
- 跨 run 的轻量 data-view identity；
- 对 `DataBuilder` 或 `DataLoaders` 的 major-version 命名迁移。

## 什么时候可以开始

至少两个独立 extension 必须重复相同的调用签名、ordering、identity、序列化、失败和
resume 语义，并且提炼后能够删除真实重复代码。单个项目的 convenience wrapper 不足以启动。

命名重审使用另一条触发条件：只能在已批准的 major-version compatibility window 中进行，
并且持续的真实误用、迁移收益和替代名称已经有文档或支持请求证据。它不会单独授权新增 helper。

## 要完成哪些工作

### 比较真实重复代码

- 动作：对照两个独立消费者的输入、输出、状态、错误和恢复规则。
- 原因：相似函数名不能证明它们共享稳定 contract。
- 影响范围：Extension recipe、DataBuilder construction 和 project helper。
- 交付物：兼容矩阵、拒绝路径和可删除的重复代码清单。
- 验证方法：第三个独立实现无需 core 分支即可替换。
- 完成条件：所有公共字段和状态转换对两个消费者具有相同含义。

### 提炼最窄的公共能力

- 动作：只为已证明重复的 construction、binding、sampler 或 view identity 提供 helper。
- 原因：避免把任意 Python 数据图或 PyTorch primitive 镜像进 Stochaflow Registry。
- 影响范围：`stochaflow.extensions`、对应 Registry 和迁移文档。
- 交付物：窄 API、独立 extension contract tests 和兼容说明。
- 验证方法：built-in 与 external extension 使用同一路径，runner 不检查具体类型或名称。
- 完成条件：helper 删除重复代码，且未给不需要它的 recipe 增加必填方法或字段。

### 只在 major version 重新核对公共名称

- 动作：根据真实误用核对 `DataBuilder`、`DataLoaders` 与候选名称表达的责任边界。
- 原因：术语偏好不足以承担 extension 迁移和兼容成本。
- 影响范围：公共类型、配置文档、extension scaffold 和迁移说明。
- 交付物：保留现名或执行 rename 的独立兼容决定，以及完整引用和迁移清单。
- 验证方法：现有 extension 可按文档迁移，旧名称的兼容行为与移除窗口明确。
- 完成条件：major-version 条件已满足，且新名称可量化减少真实误用；否则明确保留现名。

## 如何证明已经完成

- 两个真实消费者和一个独立替代实现通过同一 contract tests。
- strict resume、ordering、serialization 和 failure tests 保持一致。
- API 文档明确 direct Python、完整 `DataBuilder` 与新 helper 的选择条件。
- 没有新增 universal Dataset、Sampler、DataLoader Registry 或 YAML graph。

## 明确不包含什么

- 不为单个项目提升 convenience helper。
- 不重新注册 PyTorch 已提供的所有 Dataset、Sampler 或 DataLoader primitive。
- 不因术语偏好在 pre-1.0 小改动中重命名公共类型。
- 不把 task batch 字段加入 core schema。

## 详细设计和研究资料在哪里

- [原始 Data layer 比较、API 草案和开放问题](notes/data-layer-composition-boundary-review/research-archive.md)
- [当前 Extension API](../api/extensions.md)
- [当前 Data pipeline](../configuration/data-pipeline.md)
