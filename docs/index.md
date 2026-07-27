# Stochaflow 文档

Stochaflow 是一个配置驱动、面向扩展的生成建模研究框架。当前内置实现聚焦离散
Gaussian diffusion、DDPM/DDIM、图像数据 recipe 和自动训练生命周期；项目可以通过
普通 Python distribution 接入自己的数据、训练、生成算法和 artifact。

```{toctree}
:maxdepth: 2
:caption: 框架与使用

design/scope
framework
configuration/index
tutorials/tensorboard
tutorials/afhq-v2
tutorials/super-resolution
tutorials/reuse-gaussian-components
tutorials/custom-generation-family
api/extensions
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

- [架构范围与非目标](design/scope.md)：长期职责边界、明确拒绝的复杂度和新公共抽象的
  准入门槛。
- [框架特性与架构](framework.md)：稳定职责边界、当前内置能力和 extension 心智模型。
- [配置手册](configuration/index.md)：从最小 YAML 到多源数据、K-fold、自定义组件、
  训练恢复与排错。
- [完整字段参考](configuration/reference.md)：由 dataclass、Registry 和 CLI 自动生成。
- [TensorBoard 使用指南](tutorials/tensorboard.md)：启用日志、比较多次运行、解读
  loss/LR/diagnostic 面板并排查 event 路径。
- [AFHQ-v2 数据准备与训练](tutorials/afhq-v2.md)：安全下载、确定性 managed artifact、
  离线验证、strict resume 与 128×128 showcase。
- [纵向扩展参考项目](configuration/reference-projects.md)：Physics reconstruction 与
  frozen-teacher distillation 的独立可安装实现。
- [复用 Gaussian family 教程](tutorials/reuse-gaussian-components.md)与
  [自定义生成 family 教程](tutorials/custom-generation-family.md)：两条独立最小扩展路径。
- [条件 Gaussian 超分辨率](tutorials/super-resolution.md)：从内置 SR 数据 recipe 到
  condition-aware 训练和复用 DDPM/DDIM 的完整组合。
- [Checkpoint、配置权威与可移植性](configuration/compatibility-and-migration.md)：
  checkpoint v10、fixed inference recipe、partial sample request 和跨环境恢复边界。
- [扩展公共 API](api/extensions.md)：第三方 extension 的稳定 Python import surface。
- [Sampling artifact 容量](configuration/sampling-capacity.md)：整体物化生命周期、
  内存估算、trajectory 限制和参考主机证据。
- [DDPM 学习笔记](ddpm-notes.md)：结合代码理解训练与采样实现。
