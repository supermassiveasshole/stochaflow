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

### Public facade import closure

检查稳定导出是否能延迟重型 implementation import，同时保持 object identity、`__all__`、
typing、文档和异常行为。只有测量证明用户可见收益时才考虑 `__getattr__` 或模块拆分。

### Plugin discovery 与 Registry bootstrap

区分读取 entry-point metadata、导入选中 extension、注册 built-ins 和构造具体组件。Built-in
不能获得绕过公开 Registry contract 的隐藏路径；排序必须确定。

### CLI import closure

CLI 顶层只应解析子命令所需内容。若 train/sample/evaluate 相互 eager 导入造成可测延迟，
可把 operation import 移到选中分支，但必须保持 error/help 行为和 installed-wheel tests。

### API inventory

Facade 导出必须被 API 文档覆盖；Evaluation 自己的 `__all__` 仍由 Evaluation 管理。文档
覆盖不应错误承诺页面内容与两个不同 facade 的并集一一对应。

## 决策要求

- 建立 interpreter/native dependency/framework/activation/CLI 分层测量；
- 指定用户场景和预算，不用单次开发机数字定义 SLA；
- 每个 lazy boundary 有 rollback point 和 structural test；
- installed-wheel scaffold、plugin activation、public import identity 和失败路径保持通过；
- 若收益不足以覆盖复杂度，明确记录继续使用 eager import。
