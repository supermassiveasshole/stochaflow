# Extension 导入边界与激活延迟记录

> 工作状态：暂停
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## 完成后用户能做什么

只有测量证明“启动一个新的 Python 进程”确实过慢，而且主要时间花在 Stochaflow 自己的导入
路径上，才开始性能改进。改进可能只针对公共导入、Registry 启动、plugin 激活或 CLI 启动中的
一项，同时保持当前 extension 行为、公共对象身份和固定激活顺序不变。

本记录不承诺全面延迟导入，也不承诺重写整个模块依赖关系。

## 当前仓库已经支持什么

- 读取配置时只解析和验证内容，不导入 extension 代码。
- 准备阶段只读取已安装包的 metadata，验证选择与来源，并保存不可被调用方改写的配置副本。
- 激活计划每次返回隔离的配置副本；调用方不能在准备和激活之间改写已验证事实。
- 只有真正激活时才按预检顺序导入选中的聚合模块，并从同一份配置副本构造对象。
- 身份、版本、验收记录或回执不合法时会被明确拒绝；首次成功激活后，进程内选择固定。
- `stochaflow.extensions` 导出的对象身份、`__all__` 和 API 文档由测试约束。

正确性由当前实现、`tests/test_extension_plugins.py`、`tests/test_extensions_api.py`、
`tests/test_extensions_cli.py` 和公开文档保证；本文没有剩余正确性待办。

## 还没有支持什么

- 没有基于当前 wheel 和 lockfile 的新进程导入时间、内存与模块数量基线。
- 没有用户场景预算或测量证明主要成本来自 Stochaflow 自身导入。
- 没有按能力拆分的公共导入面、延迟导入方案或明确的第一方启动入口。
- 没有分别约束公共导入、Registry 启动、plugin 激活和 CLI 的结构检查。
- 没有已审阅的最小改动、wheel 安装验收和回退方案。

这些是暂停的性能问题，不是已确认的产品缺陷或 SLA。

## 什么时候可以开始或重新审查

只有以下条件全部成立才重开：

- [`ROADMAP.md`](../../ROADMAP.md)指定性能负责人、要改善的用户场景、目标和预算；
- 使用当前已安装 wheel、lockfile 和多个新进程建立可复现基线；
- 测量区分 Python 解释器/原生依赖、公共导入、契约访问、内置注册、选中 extension 激活和 CLI，
  并定位由 Stochaflow 负责的瓶颈；
- 提案明确保护公共对象身份、`__all__`/文档一致性、Registry 即时校验、来源记录、固定聚合激活
  和失败终止语义；
- 结构检查、同平台 benchmark、wheel 安装测试和回退方案在实现前获得审阅。

任一条件缺失时保持暂停。warm process、测试顺序或宽松 wall-clock threshold 不构成证据。

## 要完成哪些工作

### 建立新进程启动基线

- 动作：分别测量公共导入、契约访问、内置注册、选中 extension 激活，以及 CLI root/help/
  subcommand 的新进程启动时间、峰值内存和模块数量。
- 原因：必须先区分 Python/原生依赖成本与 Stochaflow 自己负责的导入成本。
- 影响范围：benchmark 工具、已安装 wheel、lockfile 和目标平台。
- 交付物：可复现的时间分布、导入时间分析、内存/模块清单和用户场景预算。
- 验证方法：在同一平台用多个新进程重复测量，记录测试工具自身成本与环境；用隔离 prototype
  验证瓶颈原因。
- 完成条件：能把主要成本定位到具体负责人；否则结束复审，不重构。

### 确定每段导入成本由谁负责

- 动作：把模块依赖分为 metadata、公共契约、注册和真正执行四类，列出每个公共使用方实际必须
  导入的模块。
- 原因：公共导入、Registry 启动、激活和 CLI 的成本与正确性约束不同。
- 影响范围：模块依赖、公共 imports、注册时机和 extension 激活。
- 交付物：责任关系图、候选接口、必须保护的规则和逐项回退点。
- 验证方法：时间测量和结构分析能解释同一项成本；不能只按模块数量或名称猜瓶颈。
- 完成条件：只保留有测量收益且不破坏正确性的候选改动。

### 只实施测量指向的最小改动

- 动作：只实现基线测量指向且通过架构审查的最小接口，并同步结构检查、benchmark、wheel 安装
  验收和文档一致性检查。
- 原因：一次重写整个 import graph 会混淆收益、责任和回退。
- 影响范围：被选中模块、公共导入或 CLI dispatch；其他部分保持不变。
- 交付物：独立可回退实现、新进程测量证据和迁移/兼容性结论。
- 验证方法：public identity、`__all__`、Registry immediate validation、provenance、deterministic
  activation 与 failure behavior 回归；真实场景达到预算。
- 完成条件：收益可复现，正确性基线和 external extension substitution 全部保持。

## 如何证明已经完成

- 同一 wheel 和平台的新进程测量达到预先定义的预算。
- 测量证明收益来自被修改的 Stochaflow 导入路径。
- 公共对象身份、`__all__`/文档一致性、Registry 校验和来源记录不变。
- plugin selection、receipt、activation ordering、partial failure 和 new-process requirement 不变。
- 结构检查、功能测试、wheel 安装验收与回退演练通过。

## 明确不包含什么

- 当前不实现 lazy/capability facade、`__getattr__` map 或 component-level deferred registration。
- 当前不移动 owner modules、不拆 Registry/bootstrap、不改变 built-in registration timing。
- 当前不改变 entry-point schema、selection、provenance、version/receipt 或 aggregate activation。
- 当前不重构 CLI imports/help dispatch，也不把本记录链接进公开文档导航。
- 不用 proxy class、implicit discovery 或 deferred validation 换取未经测量的速度。

## 详细设计和研究资料在哪里

### 更细粒度的延迟激活

- 触发证据：首个最小改动完成后，仍有测量明确的 extension 激活瓶颈，且收益足以承担额外
  并发和失败恢复复杂度。
- 负责人：extension activation 与被延迟执行模块的共同维护者。
- 保留范围：延迟导入真正执行代码；不引入按组件隐式发现、proxy class 或宽松校验。
- 验证要求：覆盖并发激活、部分导入、重复进入、对象身份、失败终止和外部 extension 替换；
  如果证据不足，继续使用当前固定顺序的聚合激活。

### 相关资料

- [Extension 导入边界与性能设计笔记](notes/extension-import-boundary-and-activation-latency-plan/design-notes.md)
- [扩展与 Registry 公开说明](../configuration/extensions.md)
- [扩展公共 API](../api/extensions.md)
- [Extension 架构边界](../../ARCHITECTURE.md)
- [根路线图](../../ROADMAP.md)

旧性能阶段、阈值和模块布局只存在于 Git 历史。本文作为未来性能构想保留，未经维护者明确审阅
不得删除。
