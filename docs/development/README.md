# Stochaflow 开发文档导览

`docs/development/` 保存工程决策、未来设计和实施记录，不是公开 API 或当前行为的权威。
不要按文件长度或阶段数量猜优先级；先看状态，再决定是否需要读专项计划。

:::{warning}
Sphinx 公开这些页面是为了让路线和未来构想可查阅，不表示候选功能已经承诺或实现。
根 `ROADMAP.md` 仍是唯一排期权威；当前行为仍以规范、架构和用户文档为准。
:::

## 1. 阅读顺序

1. [`SPEC.md`](../../SPEC.md)：产品负责什么、支持什么、明确不做什么。
2. [`ARCHITECTURE.md`](../../ARCHITECTURE.md)：当前各模块由谁负责、怎样协作。
3. [`ROADMAP.md`](../../ROADMAP.md)：哪些已完成、哪些可能下一步做、哪些仍需等待。
4. [开发方向与执行顺序](development-priority-roadmap.md)：已有基础、候选关系和全部入口。
5. 只有要实现或复审某项能力时，才进入对应专项计划。

公开用法以 `README.md`、`docs/framework.md`、配置文档和教程为准。专项计划里的 API 草案、
阶段名称和示例不表示功能已经实现。

## 2. 五种状态

| 状态 | 怎样理解 |
| --- | --- |
| 已完成（Done） | 已实现、已测试、已写入稳定文档，并继续维护。 |
| 进行中（In progress） | 唯一一个已经选中、正在实施的工作项。当前为无。 |
| 下一步（Next） | 已批准、直接排在进行中工作之后的工作项。当前为无。 |
| 候选（Candidate） | 可以成为下一项的现实选择，但尚未排期。 |
| 暂停（Parked） | 保留的更长期构想，开始条件尚未满足。 |

每份专项计划顶部只给它尚待交付目标一个状态。“当前仓库已经支持什么”记录可复用的
已完成基础，不是第二个工作项；如果同一文件包含两个启动条件不同的未来目标，应拆成
两个 owner，而不是写复合状态。

## 3. 已完成：当前维护的基础

| 已完成内容 | 文档入口 | 它不代表什么 |
| --- | --- | --- |
| Data artifact lifecycle 与 DataSource/DataBuilder 分工 | [Data pipeline](../configuration/data-pipeline.md)、[Extension API](../api/extensions.md) | 不表示任意新 producer 都应进入 core，也不表示要建设通用 Dataset/YAML graph。 |
| Sampling v12 配置归属 | [Sampling 复审记录](sampling-request-config-refactor.md) | 不表示 Hydra 后复审已经开始。 |
| 正式、offline 和 live Evaluation | [Evaluation 决策记录](post-training-evaluation-support-plan.md) | 不表示所有未来任务自动拥有质量评估。 |
| Extension activation correctness | [Extension 计划](extension-import-boundary-and-activation-latency-plan.md) | 不表示性能重构已经获批。 |

已有基础只说明可复用的代码和规则存在，不等于上层完整任务已经可直接使用。

## 4. 候选：可以成为下一项

| 方向 | 文档入口 | 开始前要确认 |
| --- | --- | --- |
| 显式顺序工作流与内置操作配方 | [工作流计划](default-workflow-pipeline-support-plan.md) | 从训练后蒸馏或生图后超分中先选一条完整交付；另一条继续保留。 |
| Super-resolution workflow | [SR 计划](super-resolution-workflow-support-plan.md) | 区分已有 data/tutorial 与完整内置工作流。 |
| Consistency 与 distillation | [Consistency 计划](consistency-distillation-support-plan.md) | 明确 teacher、student、target 更新和正式 Evaluation。 |
| Codec 与 latent diffusion | [Latent 计划](latent-diffusion-support-plan.md) | 重新核对 codec provider、文件身份和生产训练要求。 |
| Hydra training configuration | [Hydra 计划](hydra-configuration-composition-migration-plan.md) | 已选产品功能需要超出普通 YAML 的组合，并且共用的单次训练入口已经存在。 |

这些方向保留 `train -> export -> distill -> evaluate`、
`generate -> super-resolution -> evaluate`、latent 和新训练方法等未来支持构想。
目前没有任何候选被选为进行中或下一步。

## 5. 暂停：开始条件尚未满足

| 方向 | 文档入口 | 等待什么 |
| --- | --- | --- |
| Stable Diffusion component support | [Stable Diffusion 计划](stable-diffusion-component-native-support-plan.md) | codec/latent 生产能力先达到已完成。 |
| Sampling 的 Hydra 后复审 | [Sampling 复审记录](sampling-request-config-refactor.md) | Hydra configuration 达到已完成，并出现真实调用问题。 |
| Evaluation cache、速度和比较政策 | [Evaluation 决策记录](post-training-evaluation-support-plan.md) | 有测量结果和明确负责人。 |
| Extension import 性能 | [Extension 计划](extension-import-boundary-and-activation-latency-plan.md) | benchmark 证明用户可见瓶颈。 |
| Distributed training/inference | [Distributed 计划](distributed-training-and-inference-support-plan.md) | 单设备不能满足实际 workload。 |
| Automated model tuning | [Tuning 计划](automated-model-tuning-plan.md) | objective、budget、Evaluation 规则和库调用稳定。 |
| Artifact metadata/provenance/capacity | [暂停提案](artifact-metadata-provenance-capacity-model-proposal.md) | 任一子方向满足提案内各自的多产出方或多使用方证据条件。 |
| Data recipe helper 与注册体验 | [Recipe extension 计划](data-recipe-extension-ergonomics-plan.md) | 至少两个独立 extension 重复同一构造、状态和失败语义。 |
| Streaming data lifecycle | [Streaming 计划](streaming-data-lifecycle-support-plan.md) | 真实 workload 无法由 project-private iterable 和现有 artifact 边界满足。 |
| Data storage 与 payload adapter | [Storage/payload 计划](data-storage-and-payload-adapter-support-plan.md) | 多种真实表示重复同一 contract，或出现现有 read boundary 无法拒绝的失败。 |
| 通用工作流编排器 | [独立暂停计划](general-workflow-orchestrator-plan.md) | 至少两个稳定流程重复相同控制逻辑，手工组合已成为问题。 |

## 6. 怎样阅读一份专项计划

主计划应按下面的问题排列；长篇调研和旧设计可以放入同目录的 `notes/`：

1. **状态是什么？** 哪些已经完成，哪些是候选或暂停。
2. **用户得到什么？** 完成后用户能运行、组合或验证什么。
3. **仓库已经有什么？** 给出代码、测试或公开文档证据。
4. **决定了什么、不做什么？** 让实现者不必重新选择架构。
5. **每一步做什么？** 写成“开始条件 → 动作 → 交付物 → 验证 → 完成条件”。
6. **还有什么没决定？** 未决问题不能伪装成执行步骤。
7. **旧名称在哪里？** 只在历史说明中保留，不能重新进入当前排期。

计划只有在根 Roadmap 选择该方向并重新核对当前仓库后，才成为可执行顺序。

## 7. 文档维护规则

- 当前行为变化：同步 `SPEC.md`、公开文档和 `CHANGELOG.md`。
- 模块责任变化：同步 `ARCHITECTURE.md`。
- 产品选择或状态变化：同步根 `ROADMAP.md` 与开发执行顺序。
- 候选和暂停计划可以保留详细构想，但开头必须区分已完成内容和未来内容。
- 稳定结论只保留一个权威位置；开发文档用链接，不复制第二份规范。
- 本轮主文档与附录的逐项去向见
  [内容转移清单](notes/document-restructure-content-map.md)。
- 含 future-support 构想的文档删除前，必须由维护者逐文件审阅。
- 已退休名称与旧链接只进入[历史计划 ID 映射](notes/history/milestone-id-map.md)、
  `CHANGELOG.md` 或 Git，不重新进入当前排期。

写作方法参考 [Microsoft 内容快速指南](https://learn.microsoft.com/en-us/contribute/content/style-quick-start)、
[Google procedure 规范](https://developers.google.com/style/procedures) 和
[Microsoft 分步说明规范](https://learn.microsoft.com/en-us/style-guide/procedures-instructions/writing-step-by-step-instructions)、
[Diátaxis 信息分类](https://diataxis.fr/start-here/)：先说明读者目标，用动作命名步骤，
并把操作说明与研究背景分开。

```{toctree}
:maxdepth: 1
:caption: 开发路线与主计划
:hidden:

development-priority-roadmap
default-workflow-pipeline-support-plan
super-resolution-workflow-support-plan
consistency-distillation-support-plan
latent-diffusion-support-plan
hydra-configuration-composition-migration-plan
stable-diffusion-component-native-support-plan
sampling-request-config-refactor
post-training-evaluation-support-plan
extension-import-boundary-and-activation-latency-plan
distributed-training-and-inference-support-plan
automated-model-tuning-plan
artifact-metadata-provenance-capacity-model-proposal
data-recipe-extension-ergonomics-plan
streaming-data-lifecycle-support-plan
data-storage-and-payload-adapter-support-plan
general-workflow-orchestrator-plan
```
