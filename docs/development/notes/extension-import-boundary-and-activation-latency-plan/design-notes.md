# Extension import 与 activation 性能设计附录

> Activation 正确性已经实现；import 性能仍然暂停。本文保存候选分析，不授权
> lazy import 或性能重构。状态以
> [`extension-import-boundary-and-activation-latency-plan.md`](../../extension-import-boundary-and-activation-latency-plan.md)
> 为准。
>
> 最后核对：2026-08-09。以下旧测量只能作为重新测量的起点。

## 已完成的正确性边界

- Discovery/provenance preflight 不导入 extension code。
- Plan 保存私有 deep-copy config snapshot；公开 `config` 返回隔离副本。
- Activation 从同一 snapshot 构造 resolved config，并返回与 plan 绑定的 receipt。
- 插件排序、version policy、重复 activation 和失败后必须重启的行为都有测试。
- Public facade 的 object identity、`__all__` 和文档覆盖受回归测试保护。

这些正确性要求不能为降低 import time 而放宽。

## 需要重新测量的旧现象

一次旧的 fresh-process 审查观察到 bare extension facade 会加载 torch、torchvision、PIL、
Rich、Trainer 和 diagnostics，约 2.37 秒/2737 modules；`utils.plugins` 约 1.29 秒，CLI
入口也会 eager 导入多个 operation。环境、依赖和实现会变化，启动任何性能工作前必须在
当前 installed wheel、lockfile、平台和 fresh process 中复测。

## 候选拆分方向

`5c75a76` 中的旧计划提出过下面的候选结构。它们是用来解释“如果测量确认瓶颈，可以从
哪里下手”，不是已经批准的目标架构，也不能在没有测量时一起实施。

### 四类导入工作

- **metadata：**读取 distribution 和 entry-point metadata，不导入 extension code；
- **contract：**只让调用者取得稳定 public types 和 protocols；
- **registration：**按明确顺序注册 built-ins，并激活用户明确选择的 extension；
- **execution：**只有真正运行 train、sample、evaluate 等 operation 时才导入重型实现。

如果测量证明当前模块把这四件事绑在一起，才考虑按实际依赖拆开。拆分后 built-in 仍必须
经过与外部 extension 相同的 Registry 校验，不能获得隐藏捷径。

### Public facade import closure

检查稳定导出是否能延迟重型 implementation import，同时保持 object identity、`__all__`、
typing、文档和异常行为。只有测量证明用户可见收益时才考虑 `__getattr__` 或模块拆分。

旧计划还保留了两种更具体的候选做法：按 training、sampling、evaluation 等能力提供较小的
public facade；或者使用明确的 lazy-export map 延迟导入实现。两者都必须保证同一个 public
对象无论从哪个支持入口导入，Python object identity 不变，类型检查和 Sphinx 也能看到同一
API。proxy class 或隐式兜底查找不能替代真实对象。

### Contract 与 implementation 分开

若导入分析证明稳定 contract 因为和具体实现放在同一模块而被迫加载 torch、torchvision、
Trainer 或 task code，可以把 contract 移到窄模块，具体实现留在原 responsibility package。
这不是为了制造一套新的镜像 package；只有能删除已测量的重型依赖闭包时才做。

### Plugin discovery 与 Registry bootstrap

区分读取 entry-point metadata、导入选中 extension、注册 built-ins 和构造具体组件。Built-in
不能获得绕过公开 Registry contract 的隐藏路径；排序必须确定。

旧计划提出的“显式 first-party bootstrap”是指：由一个明确函数按固定顺序注册 Stochaflow
built-ins，外部 aggregate activation 仍只导入用户选择的 entry points。它不表示把注册推迟
到第一次 lookup，也不允许插件 discovery 自动导入所有已安装 package。

注册入口是否应从当前跨领域组装代码中移出，是[运行组装与 Diagnostic 边界备忘](../runtime-composition-and-diagnostic-boundary-refactor-plan.md)
记录的架构问题。那份备忘只要求入口可定位、顺序确定、幂等且失败行为明确。本附录只在真实
启动时间测量指向这里时，判断 facade、catalog、CLI operation 或注册动作是否需要延迟；显式
初始化不自动等于 lazy initialization，也不自动证明启动更快。

旧计划还提出过一个更窄的 Registry 拆分候选。当前 `Registry` 已支持
`expected_type_resolver`：某个 Registry 可以在第一次真正注册组件时解析并缓存自己需要的
基类，同时仍在注册边界立即拒绝错误类型。这不是延迟加载组件，也不能把错误拖到运行期。

如果重新测量证明创建全局 Registry catalog 会因为 Torch、optimizer 或其他不相关 contract
产生明显导入成本，才继续评审下面的拆分：

- 把通用 `Registry` / `RegistryError` primitive 放进不依赖 Torch 的窄模块；
- 把 Torch-aware、Stochaflow 全量 inventory 的 `RegistryCatalog` 留在应用层；
- 让每个 capability 只解析自己的 expected type，不在激活一个数据 extension 前绑定所有
  model、optimizer、Metric 和 Evaluation contract；
- built-in 与外部 extension 仍通过同一注册和类型检查路径。

这部分拆分尚未作为性能工作获批。只有 fresh-process import closure 指向这里，并且收益足以
覆盖模块迁移和循环依赖风险时才实施；否则保留现状。

### CLI import closure

CLI 顶层只应解析子命令所需内容。若 train/sample/evaluate 相互 eager 导入造成可测延迟，
可把 operation import 移到选中分支，但必须保持 error/help 行为和 installed-wheel tests。

候选 dispatcher 只解析根命令并把控制交给选中的 subcommand；它不能改变参数、错误码、
help 文本或 activation 时机。若 CLI 时间主要来自 Python 或原生依赖初始化，而不是未选中的
operation import，则不做这项拆分。

### API inventory

Facade 导出必须被 API 文档覆盖；Evaluation 自己的 `__all__` 仍由 Evaluation 管理。文档
覆盖不应错误承诺页面内容与两个不同 facade 的并集一一对应。

## 决策要求

- 建立 interpreter/native dependency/framework/activation/CLI 分层测量；
- 指定用户场景和预算，不用单次开发机数字定义 SLA；
- 每个 lazy boundary 有 rollback point 和 structural test；
- installed-wheel scaffold、plugin activation、public import identity 和失败路径保持通过；
- 若收益不足以覆盖复杂度，明确记录继续使用 eager import。

如果测量最终支持 lazy facade，还要逐项核对普通 import、具名 import、`import *`、`__all__`、
`dir()` 和 `help()`；Pyright 与 Sphinx 仍须看见相同公共对象。延迟读取必须返回原始对象，保持
`isinstance`、`issubclass`、`__module__` 和 pickle 行为；并发首次访问、循环导入和导入失败不能
缓存半成品。导入公共符号不能激活外部 extension，源码目录与安装 wheel 必须一致，每个延迟
边界也必须能单独撤回。仅为性能新增 training、sampling 或 Evaluation 公共 facade 需要另外的
真实 API 使用证据，不能由一次启动优化自动获得授权。

除 wall-clock benchmark 外，旧计划还保留结构检查：例如 public contract import 不应意外
加载某组 execution modules，CLI help 不应导入所有 operation，registration 必须有一个明确
入口。结构检查只约束责任边界；实际性能是否改善仍必须由同平台 fresh-process 测量证明。
