# Stochaflow 文档

Stochaflow 是一个围绕随机流、DDPM 和 DDIM 的配置驱动研究项目。这里包含可执行的
配置手册、实现学习笔记和扩散模型研究笔记。

```{toctree}
:maxdepth: 2
:caption: 使用手册

configuration/index
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

## 快速入口

- [配置手册](configuration/index.md)：从最小 YAML 到多源数据、K-fold、自定义组件、
  训练恢复与排错。
- [完整字段参考](configuration/reference.md)：由 dataclass、Registry 和 CLI 自动生成。
- [DDPM 学习笔记](ddpm-notes.md)：结合代码理解训练与采样实现。
- [macOS 环境指南](macos-terminal-setup.md)：Intel/Apple Silicon 环境准备。
