# 开发文档重构内容转移清单

> 用途：在缩短主计划前，记录每类内容的新归属，防止 future-support
> 构想因改写而消失。
>
> Last reviewed: 2026-08-09

## 处理规则

- 当前已经实现的行为只在主计划中保留结论和权威链接，详细契约以
  `SPEC.md`、`ARCHITECTURE.md` 和公开文档为准。
- 尚未实现的用户结果、接口构想、失败边界和重审条件必须留在主计划或其
  `notes/` 附录中，不能只依赖 Git 历史。
- 外部库、硬件、数据集和性能调研移入附录，标明实施前必须重新验证。
- 已完成的阶段编号只移入历史映射；主计划改用描述动作或结果的标题。
- 陈旧工期估算不迁移。它们不是产品要求，候选被选为 `Next` 后重新估算。

## 逐文档转移

| 原主文件 | 当前事实的新归属 | Future-support 的新归属 | 研究与历史的新归属 |
| --- | --- | --- | --- |
| [开发执行路线图](../development-priority-roadmap.md) | 精简后的当前能力表和根路线图链接 | 精简后的候选/暂停队列及各 owner plan 链接 | [历史计划 ID 映射](history/milestone-id-map.md) |
| [显式顺序工作流](../default-workflow-pipeline-support-plan.md) | 精简后的独立 operation 与 runtime/Builder 职责 | 原路径只负责显式顺序组合；超分辨率与通用编排分别移入独立主计划 | [原始设计归档](default-workflow-pipeline-support-plan/design-archive.md) |
| [通用工作流编排器（新 owner）](../general-workflow-orchestrator-plan.md) | 链接当前独立 operation 与显式组合基础 | 独立保留恢复、重试、分支和通用状态传递构想及重审条件 | 复用[原始工作流设计归档](default-workflow-pipeline-support-plan/design-archive.md) |
| [内置超分辨率工作流（新 owner）](../super-resolution-workflow-support-plan.md) | 链接当前 DataBuilder、教程、Sampling 和 Evaluation 基础 | 独立保留确定性 baseline、Gaussian SR 及高级恢复方向 | 复用[原始工作流设计归档](default-workflow-pipeline-support-plan/design-archive.md)中的 SR 研究 |
| [DataArtifact lifecycle](../data-artifact-producer-lifecycle-refactor.md) | 原路径改为简短的已完成决策记录 | 统一 metadata 构想继续由独立 proposal 负责 | [实施归档](data-artifact-producer-lifecycle-refactor/implementation-archive.md) |
| [Data layer 边界](../data-layer-composition-boundary-review.md) | 原路径改为简短的已完成边界记录 | function registration、streaming、distributed ownership 和复用条件保留在原路径的重审部分 | [研究归档](data-layer-composition-boundary-review/research-archive.md) |
| [Latent diffusion](../latent-diffusion-support-plan.md) | 主计划链接现有 checkpoint/inference 基础 | 原路径保留完整 latent 用户结果、启动条件和候选任务 | [Codec、provider、hardware 与数据调研](latent-diffusion-support-plan/design-and-research-notes.md) |
| [Stable Diffusion](../stable-diffusion-component-native-support-plan.md) | 主计划只写当前未提供完整产品支持 | 原路径保留 text encoder、UNet、CFG、模型兼容及后续模型家族 | [Provider/API 调研](stable-diffusion-component-native-support-plan/design-and-research-notes.md) |
| [Consistency 与蒸馏](../consistency-distillation-support-plan.md) | 主计划区分 frozen-teacher 基础和内置 workflow 缺口 | 原路径保留 teacher/target lifecycle、算法支持和未来工作流 | [数学与设计依据](consistency-distillation-support-plan/design-and-research-notes.md) |
| [自动化模型调优](../automated-model-tuning-plan.md) | 主计划链接当前 operation、metrics 和 Evaluation 基础 | 原路径保留 HPO 用户结果、启动条件和失败边界 | [Provider 选型与 API 草案](automated-model-tuning-plan/research-and-api-draft.md) |
| [Distributed 支持](../distributed-training-and-inference-support-plan.md) | 主计划链接当前单设备生命周期 | 原路径保留 distributed correctness、checkpoint、sampling 与 Evaluation 目标 | [PyTorch/provider/version 调研](distributed-training-and-inference-support-plan/research-and-api-draft.md) |
| [Hydra 配置迁移](../hydra-configuration-composition-migration-plan.md) | 主计划说明当前 plain-YAML 基线 | 原路径保留 fresh-training composition 和启动迁移目标 | [详细设计](hydra-configuration-composition-migration-plan/design-notes.md) |
| [Sampling 调用重审](../sampling-request-config-refactor.md) | 主计划说明当前 sampling v12 已完成 | 原路径保留 Hydra 完成后的调用方式重审问题 | [历史 sampling-request 设计](sampling-request-config-refactor/review-notes.md) |
| [Evaluation 后续](../post-training-evaluation-support-plan.md) | 主计划链接当前 standalone/offline Evaluation | 原路径保留 reference cache、性能证据和 comparison policy 重审条件 | [已完成设计背景](post-training-evaluation-support-plan/design-notes.md) |
| [Extension 导入性能](../extension-import-boundary-and-activation-latency-plan.md) | 主计划链接当前 activation 正确性 | 原路径保留 import 性能目标和重开条件 | [性能设计笔记](extension-import-boundary-and-activation-latency-plan/design-notes.md) |
| [Artifact metadata/provenance/capacity](../artifact-metadata-provenance-capacity-model-proposal.md) | 主计划链接当前 run-local artifact 证据 | 原路径分别保留三类信息各自的多使用方触发条件 | [设计与容量调研](artifact-metadata-provenance-capacity-model-proposal/design-notes.md) |

## 新增的清晰 owner

- `super-resolution-workflow-support-plan.md` 接管原工作流文档中的超分辨率数据、
  训练、推理、writer、tiling、metric 和正式 Evaluation 构想。
- `default-workflow-pipeline-support-plan.md` 不再复制 latent、Stable Diffusion 或
  consistency 的内部实现；它只说明 operation 之间如何显式传递 typed result
  和 artifact。
- `general-workflow-orchestrator-plan.md` 接管启动条件不同的通用恢复、重试、分支和
  状态传递构想，避免与候选的显式顺序组合共用状态。

## 删除边界

本轮不新增整份文档删除。工作树中已有的四份历史删除候选保持独立审阅；
它们不属于本清单，也不得被当成 future-support 转移的目标。
