# 开发文档内容保全清单

> 核对基线：`5c75a76de3d696a5b734ae4eefe88a30532bd2de`
>
> 最后核对：2026-08-10

本清单回答一个问题：旧计划被缩短或改名以后，里面仍有价值的功能构想现在在哪里。
“还在 Git 历史里”不算保留；每项内容必须在当前文档、规范、公开用法或研究附录中有
明确落点。

## 分类规则

- 已实现并继续维护的内容写入规范、架构和公开使用文档，不再伪装成开放计划；
- 有清楚输入、操作和产物的未来功能，保留为功能计划，并由 `ROADMAP.md` 决定状态；
- 没有真实使用者、数据源或 provider 的假设，保留为研究备忘，不进入排期；
- 数学推导、API 草案、provider 调研、长测试矩阵和旧阶段编号放在 `notes/`；
- 已退休实验和平台只保留结论、证据位置和重新讨论的条件。

## 历史基线逐项映射

| `5c75a76` 中的文件 | 需要保留的内容 | 当前落点与状态 |
| --- | --- | --- |
| `artifact-metadata-provenance-capacity-model-proposal.md` | metadata、来源追踪和资源证据是三个不同功能；citation/license、转换记录、code/config/environment snapshot；artifact footprint、运行前估算和运行后实测不能混用 | [研究备忘](../artifact-metadata-provenance-capacity-model-proposal.md)逐项解释使用场景；[设计附录](artifact-metadata-provenance-capacity-model-proposal/design-notes.md)保留候选事实、三类 capacity 和失败边界；三个问题分别等待真实写入方、读取方和用户操作，不参与排期 |
| `automated-model-tuning-plan.md` | 固定训练配置、搜索范围、预算、validation 目标、trial 隔离/恢复、最终正式 Evaluation；provider、剪枝、并行和统计候选 | [HPO 功能计划](../automated-model-tuning-plan.md)保留用户流程；[调研与 API 草案](automated-model-tuning-plan/research-and-api-draft.md)保留 provider、配置、seed、失败、泄漏和并行设计；当前暂停 |
| `consistency-distillation-support-plan.md` | teacher/student 数学、teacher bundle、target model 更新、一步/少步采样、只依赖学生的推理产物、未来算法 | [蒸馏功能计划](../consistency-distillation-support-plan.md)解释完整使用流程；[数学与设计附录](consistency-distillation-support-plan/design-and-research-notes.md)保留公式和候选；当前候选 |
| `data-artifact-producer-lifecycle-refactor.md` | producer、manifest、identity、cache、strict resume 和任意 payload 的生命周期 | 已闭合到 [`SPEC.md`](../../../SPEC.md)、[`ARCHITECTURE.md`](../../../ARCHITECTURE.md)、[Data pipeline](../../configuration/data-pipeline.md)和测试；[实施记录](data-artifact-producer-lifecycle-refactor/implementation-archive.md)保留迁移历史；不是开放计划 |
| `data-layer-composition-boundary-review.md` | Artifact 不是 Dataset；`DataBuilder` 是一次运行的数据组合；source-only、direct Python 和 recipe 的选择；LMDB、streaming、helper 等研究问题 | 当前边界在 [`ARCHITECTURE.md`](../../../ARCHITECTURE.md)和[Data pipeline](../../configuration/data-pipeline.md)；完整方案比较在[研究归档](data-layer-composition-boundary-review/research-archive.md)；helper、streaming 和新存储分别保留为研究备忘，不参与排期 |
| `default-workflow-pipeline-support-plan.md` | 可发现的任务 recipe、独立 operation、typed artifact 传递、训练后蒸馏、生图后超分、推理 bundle 和是否需要通用编排器 | [显式工作流计划](../default-workflow-pipeline-support-plan.md)保留两条具体流程；[超分计划](../super-resolution-workflow-support-plan.md)和[通用编排器计划](../general-workflow-orchestrator-plan.md)分别承接独有功能；[原始设计](default-workflow-pipeline-support-plan/design-archive.md)保留接口草案和完整矩阵 |
| `development-priority-roadmap.md` | 产品先后关系、AFHQ 后的 latent/codec/DiT、Stable Diffusion 原生采样和微调，以及更晚的规模化方向 | 根 [`ROADMAP.md`](../../../ROADMAP.md)是唯一排期；[当前开发队列](../development-priority-roadmap.md)只保留直接依赖；Latent、Stable Diffusion 和 Distributed 计划保留具体功能，旧编号只在[历史编号表](history/milestone-id-map.md) |
| `distributed-training-and-inference-support-plan.md` | 多进程训练、数据分片、全局指标、分布式 checkpoint、合并采样结果；FSDP2、多节点和弹性后续 | [Distributed 功能计划](../distributed-training-and-inference-support-plan.md)保留用户结果和首版；[调研与 API 草案](distributed-training-and-inference-support-plan/research-and-api-draft.md)保留 PyTorch/version、配置、checkpoint 和后续候选；当前暂停 |
| `extension-import-boundary-and-activation-latency-plan.md` | 先测公共导入、Registry、plugin activation 和 CLI；保持对象身份、激活顺序和安装 wheel 行为；保留 capability-scoped expected-type resolver 与轻量 Registry/应用级 catalog 拆分候选 | [Extension 性能计划](../extension-import-boundary-and-activation-latency-plan.md)只保留测量驱动的性能功能；[设计附录](extension-import-boundary-and-activation-latency-plan/design-notes.md)保留导入和 Registry 拆分；当前暂停，正确性已闭合 |
| `hydra-configuration-composition-migration-plan.md` | 用 Hydra 组合全新训练配置，但仍由 Stochaflow 校验和执行；preview、受信任配置根、普通 YAML 对照；不接管 resume/sample | [Hydra 功能计划](../hydra-configuration-composition-migration-plan.md)保留用户流程；[设计附录](hydra-configuration-composition-migration-plan/design-notes.md)保留配置组、override、失败和 multirun 问题；当前候选 |
| `latent-diffusion-support-plan.md` | 冻结 codec、在线编码与 prepared posterior 两条数据路、AFHQ bring-up、生产资产、开放数据 DiT、独立非 DiT 替换、未来 codec/VAE | [Latent 功能计划](../latent-diffusion-support-plan.md)解释训练和生成流程；[设计与研究附录](latent-diffusion-support-plan/design-and-research-notes.md)保留 codec/provider、The Met、ImageNet-100、DomainNet、硬件和未来算法；当前候选 |
| `legacy-intel-macos-pytorch-test-lifecycle.md` | Intel macOS 退休决定和不再维护专用依赖/CI 的理由 | 当前支持矩阵在[平台支持](../../platform-support.md)、根 README 和 `CHANGELOG.md`；历史实施记录仍作为独立删除候选，不含 future 功能 |
| `metrics-support-plan.md` | task-neutral `MetricSpec`/`MetricUpdate`/`MetricEngine`，training 与 Evaluation 分离，selection 只使用 validation | 已闭合到 [`SPEC.md`](../../../SPEC.md)、[`ARCHITECTURE.md`](../../../ARCHITECTURE.md)、[公开工作流](../../configuration/workflows.md)和测试；不再有 Metrics 开放计划 |
| `p2-weighting-and-adm-topology-refactor-plan.md` | canonical ADM 拓扑、learned-range variance、hybrid loss、respaced sampling、CFG 2C、AFHQ 对照；拓扑与 weighting 分开归因、论文复刻与产品实验分线、谨慎命名复刻结果、Gaussian-local weighting 边界；SNR weighting 实验结论 | ADM/learned-range 能力和证据在 [`CHANGELOG.md`](../../../CHANGELOG.md)、[AFHQ 教程](../../tutorials/afhq-v2.md)、规范和测试；[SNR weighting 历史决策](history/afhq-snr-weighting-decision.md)保留研究理由、gamma 0/1 对照、色彩噪声、联合候选边界和重新讨论条件；P2 支持已退休 |
| `post-training-evaluation-support-plan.md` | checkpoint Evaluation、prediction artifact offline replay、完整性、结果发布；reference cache、性能、比较、任务 profile、外部报告、benchmark extension、人工评价 artifact、distributed Evaluation、inference-bundle subject 和置信区间设想 | 基础功能已闭合到规范和[公开工作流](../../configuration/workflows.md)；[已完成说明](../post-training-evaluation-support-plan.md)解释当前用法；[设计备忘](post-training-evaluation-support-plan/design-notes.md)保存未排期想法，并说明旧 Selection runtime 已被 training/HPO/调用方各自决策替代；不再把 Evaluation 列为开放工作 |
| `sampling-request-config-refactor.md` | training/sample 配置分离、完整 sample request、CLI/plugin/output 约定；Hydra 后是否需要稳定 Python 调用的复查 | 当前 v12 行为在规范和[公开工作流](../../configuration/workflows.md)；[条件性复查](../sampling-request-config-refactor.md)明确可能保持现状；[复查资料](sampling-request-config-refactor/review-notes.md)保留旧设计 |
| `stable-diffusion-component-native-support-plan.md` | black-box 参考路径与逐组件路径、codec/text encoder/UNet/CFG、图文数据、256/512 验证、LoRA/SDXL/SD3/Flux 后续 | [Stable Diffusion 功能计划](../stable-diffusion-component-native-support-plan.md)解释微调和生成流程；[设计与研究附录](stable-diffusion-component-native-support-plan/design-and-research-notes.md)保留 provider、数据、parity 和未来模型；当前暂停 |

## 后来的决定覆盖了哪些旧想法

保全历史不表示永远服从旧计划。以下内容经过后续实现或维护者决定，已经明确被替换：

- `metrics-support-plan.md` 曾讨论让带 validation metadata 的 Diagnostic 参与模型选择；当前
  规范只允许正式 validation observations 参与 checkpoint selection 和 early stopping，
  Diagnostic 只是观察工具。旧想法作为历史背景保留，但不能写回当前功能；
- `hydra-configuration-composition-migration-plan.md` 曾把清理 Physics/KD reference projects
  放进迁移范围；维护者后来明确要求本轮不修改 Physics/KD。当前 Hydra 计划只讨论配置
  组合，不授权删除、移动或改写这些项目；
- Evaluation 旧计划把 comparison、Selection、gate、suite 和可选 integration 放在后续
  阶段。当前基础 Evaluation 已闭合；独立 Selection runtime 已被 validation-only training
  selection、HPO study policy 和发布调用方各自决策替代。其余 benchmark extension、人工
  评价 artifact、distributed Evaluation、inference-bundle subject、置信区间和外部 reporting
  只作为彼此独立的未排期研究入口保存；
- Sampling 旧文件记录的 partial-request 重构已经完成。当前文件只保留 Hydra 之后的条件性
  复查，并允许得出“无需改变”的结论。
- Workflow 旧草案曾保留训练结束后自动 sampling、把 sampling/Evaluation reference 写回
  `TrainingRunOutcome`，以及让符合条件的 Diagnostic 参与 selection。后来决定改为显式、
  独立的 train/sample/evaluate stage，跨步骤引用由 workflow manifest 保存；模型选择只读取
  validation observation。这些旧草案不再是 future 功能，训练后蒸馏和生图后超分本身仍由
  当前工作流计划保留。

## 基线以后新增但必须继续保留的功能

- [内置超分辨率](../super-resolution-workflow-support-plan.md)：低清/高清配对训练、低清输入、
  高清 artifact、正式 Evaluation，以及以后可能的条件 Gaussian、tiling 和更广恢复任务；
- [通用工作流编排器](../general-workflow-orchestrator-plan.md)：至少两条稳定工作流重复以后，
  才提供步骤状态、失败重试和恢复；
- [大规模数据集的层级处理与训练](../hierarchical-data-pipeline-support-plan.md)：让大于内存的数据
  通过有预算、可回压、可测量的持久存储、RAM、pinned memory 和设备内存路径；八卡 H200 只
  作为首个压力测试环境，DALI、GPUDirect Storage 和远程存储仅在实测指向时复查；
- Data 的三份研究备忘分别保存公共 helper、连续数据流和新存储形式，但不重新打开已经
  闭合的 Data lifecycle。

## 未合入主线的远端审查草案

2026-07-31 的 Clean Architecture 与内置 Evaluation 草案后来提交在未合入主线的远端审查
分支 `codex/metrics-system`（提交 `af13619`）。它从未进入 `main` 或当前 `ROADMAP.md`，
因此不能伪装成已经执行或已经排期的仓库计划。独立 Evaluation 后来已经完成；草案中仍需
重新判断的运行组装责任和 Diagnostic 窄能力问题，保存在
[运行组装与 Diagnostic 边界备忘](runtime-composition-and-diagnostic-boundary-refactor-plan.md)。
该备忘不参与排期，不重新打开 Evaluation，也不授权全仓 package 改名或机械拆分 Trainer。

## 删除边界

Data lifecycle 和 Data composition 的当前行为、future 构想与研究材料已经完成迁移，维护者
此前批准删除以下两份已完成记录；其当前内容和研究归档仍由上表提供入口：

- `data-artifact-producer-lifecycle-refactor.md`
- `data-layer-composition-boundary-review.md`

工作树中已有的四份历史删除候选仍保持独立审阅；它们不属于本次批准范围，也不得被
当成 future-support 转移的目标：

- `afhq-v2-checkpoint-cleanup-20260806.md`
- `afhq-v2-learned-range-v-closeout.md`
- `legacy-intel-macos-pytorch-test-lifecycle.md`
- `p2-experiment-closeout.md`

在维护者逐文件批准删除前，Sphinx 和主计划结构检查必须忽略这些文件；忽略不表示
删除已经获批。
