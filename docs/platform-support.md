# 平台支持政策

Stochaflow 使用以下支持等级描述操作系统与架构组合。Python package metadata 中的
`requires-python` 只表示解释器版本范围，不等于对每个 Python、操作系统、架构和
PyTorch 组合都作出支持承诺。

## 支持等级

**Supported**

: 纳入当前发布验证矩阵。框架自身引入的回归应在发布前修复；具体硬件加速能力仍受
  PyTorch 等上游依赖约束。

未列出的环境不属于当前发布验证矩阵。它们可能可以运行，但项目不据此作出兼容性
保证。

## 当前矩阵

| 环境 | 等级 | 当前验证基线 |
| --- | --- | --- |
| Linux x86_64 | Supported | Ubuntu CI，CPython 3.12 和 3.14.6 |
| Windows x86_64 | Supported | Windows CI，CPython 3.14.6 |
| macOS arm64 | Supported | macOS CI，CPython 3.14.6 |

Python 3.14 用户应以 **3.14.6 或更新的兼容 patch release** 作为基线。CPython
3.14.0–3.14.4 使用 incremental garbage collector；上游因生产环境的显著内存压力，
从 3.14.5 起恢复为此前的 generational garbage collector。3.14.6 继承了该回退并包含
后续修复，因此当前 CI 明确固定到 3.14.6，而不是浮动的 `3.14`。背景见
[Python 3.14 的 garbage collector 变更](https://docs.python.org/3/whatsnew/3.14.html#garbage-collection)。

macOS x86_64（Intel）不受支持。项目不再为该平台保留专用依赖约束、CI lane、运行时
例外或修复承诺；源码在该平台上偶然可安装或运行，不构成兼容性保证。
