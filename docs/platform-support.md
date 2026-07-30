# 平台支持政策

Stochaflow 使用以下支持等级描述操作系统与架构组合。Python package metadata 中的
`requires-python` 只表示解释器版本范围，不等于对每个 Python、操作系统、架构和
PyTorch 组合都作出支持承诺。

## 支持等级

**Supported**

: 纳入当前发布验证矩阵。框架自身引入的回归应在发布前修复；具体硬件加速能力仍受
  PyTorch 等上游依赖约束。

**Deprecated / best effort**

: 现有安装路径和部分 CI 验证会在迁移期内保留，但不再承诺完整功能、性能或问题修复
  时限。已知不可靠的上游能力可以不纳入运行时契约，新功能也不需要为该平台增加专用
  兼容层。后续版本可以另行移除对应依赖约束和 CI lane。

未列出的环境不属于当前发布验证矩阵。它们可能可以运行，但项目不据此作出兼容性
保证。

## 当前矩阵

| 环境 | 等级 | 当前验证基线 |
| --- | --- | --- |
| Linux x86_64 | Supported | Ubuntu CI，CPython 3.12 和 3.14.6 |
| Windows x86_64 | Supported | Windows CI，CPython 3.14.6 |
| macOS arm64 | Supported | macOS CI，CPython 3.14.6 |
| macOS x86_64（Intel） | **Deprecated / best effort** | 过渡期 CI，CPython 3.12、PyTorch 2.2.2、Torchvision 0.17.2 |

Python 3.14 用户应以 **3.14.6 或更新的兼容 patch release** 作为基线。CPython
3.14.0–3.14.4 使用 incremental garbage collector；上游因生产环境的显著内存压力，
从 3.14.5 起恢复为此前的 generational garbage collector。3.14.6 继承了该回退并包含
后续修复，因此当前 CI 明确固定到 3.14.6，而不是浮动的 `3.14`。背景见
[Python 3.14 的 garbage collector 变更](https://docs.python.org/3/whatsnew/3.14.html#garbage-collection)。

Intel macOS 的 PyPI wheel 供应已经停留在旧版 PyTorch 组合。该组合的
multi-worker DataLoader helper process 可能在迭代完成后阻塞解释器退出，因此它不在
Stochaflow 承诺的运行时契约内。在这一平台上应使用：

```yaml
loader:
  num_workers: 0
  persistent_workers: false
  prefetch_factor: null
```

保留 `macos-15-intel` CI lane 只是迁移期的兼容性信号，不会把 Intel macOS 重新定义为
完整支持平台，也不保证所有 Supported 平台测试都会在该 lane 上执行。若该 lane 与
框架演进发生冲突，维护者可以选择跳过不受上游支持的能力，或在后续独立变更中移除
该 lane 和专用依赖 pins。

安装或运行时问题可先查阅[排错索引](configuration/troubleshooting.md)。Intel macOS
问题会按影响范围和修复成本进行 best-effort 评估；项目不承诺为其引入新的
platform-specific framework abstraction。
