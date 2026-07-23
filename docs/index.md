# Stochaflow 文档

Stochaflow 是一个围绕随机流、DDPM 和 DDIM 的配置驱动研究项目。这里包含可执行的
配置手册、实现学习笔记和扩散模型研究笔记。

```{toctree}
:maxdepth: 2
:caption: 使用手册

configuration/index
tutorials/reuse-gaussian-components
tutorials/custom-generation-family
api/extensions
macos-terminal-setup
```

```{toctree}
:maxdepth: 2
:caption: 学习与研究

ddpm-notes
research-notes/part-1-distribution-transport
research-notes/part-2-ddpm
research-notes/part-3-ddim
```

```{toctree}
:maxdepth: 1
:caption: 设计与开发

custom-code-extension-support-plan
development/extension-refactor-decisions
development/sampling-capacity
development/extension-refactor-report
```

## 快速入口

- [配置手册](configuration/index.md)：从最小 YAML 到多源数据、K-fold、自定义组件、
  训练恢复与排错。
- [完整字段参考](configuration/reference.md)：由 dataclass、Registry 和 CLI 自动生成。
- [纵向扩展参考项目](configuration/reference-projects.md)：Physics reconstruction 与
  frozen-teacher distillation 的独立可安装实现。
- [复用 Gaussian family 教程](tutorials/reuse-gaussian-components.md)与
  [自定义生成 family 教程](tutorials/custom-generation-family.md)：两条独立最小扩展路径。
- [兼容性、迁移与可移植性](configuration/compatibility-and-migration.md)：breaking
  changes、checkpoint 内容和跨环境恢复边界。
- [扩展公共 API](api/extensions.md)：第三方 extension 的稳定 Python import surface。
- [DDPM 学习笔记](ddpm-notes.md)：结合代码理解训练与采样实现。
- [Sampling artifact 容量边界](development/sampling-capacity.md)：整体物化
  lifecycle、DFSR 保守 profile 和 trajectory preview 限制。
- [Extension 重构报告](development/extension-refactor-report.md)：Stage 1–8 的架构结果、
  OCP 验收与真实 Physics 证据。
- [macOS 环境指南](macos-terminal-setup.md)：Intel/Apple Silicon 环境准备。
