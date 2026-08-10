# 历史计划 ID 映射

> 文档性质：历史索引，不参与排期
>
> 重点基线：`5c75a76de3d696a5b734ae4eefe88a30532bd2de`

这些短 ID 只用于读旧提交和设计归档。它们不表示当前顺序，也不能把研究材料重新变成
路线图工作。相同 ID 曾被不同文件复用；本表会写明所属文件和时期。

## `5c75a76` 开发优先级路线图中的 ID

| 旧 ID | 当时的完整含义 | 当前落点 |
| --- | --- | --- |
| `B0` | 可运行、可验证的 baseline | 已完成；规范、测试和公开文档 |
| `A0` | 修正 canonical ADM topology 与 checkpoint compatibility | 已完成；当前 Gaussian 架构和 AFHQ 文档 |
| `B1` | 切分 train/sample configuration authority | 已完成；当前配置与 checkpoint-v12 规则 |
| `K0` | Metrics foundation | 已完成；task-neutral Metrics 与 validation-only selection |
| `A1` | learned-range Gaussian 与 P2 capability | learned-range 已完成；SNR weighting recipe 已退休 |
| `A2` | Controlled AFHQ P2 Evidence | 实验已结束且未验证出可靠收益；见 [SNR weighting 决策](afhq-snr-weighting-decision.md) |
| `L0` | Pretrained codec ready | Latent Diffusion 候选功能的一部分 |
| `L1` | AFHQ Latent Diffusion 首条完整流程 | Latent Diffusion 候选功能的一部分 |
| `L2` | 可长期训练的 latent 数据、预算和资产保存 | Latent Diffusion 候选功能的一部分 |
| `Q0` | Configuration and Latent Evaluation closeout | 后来拆开：Hydra 是独立候选；评估由具体 latent 任务交付 |
| `L3` | 开放数据上的 DiT baseline | 保留在 Latent Diffusion 计划和研究附录 |
| `S0` | Stable Diffusion 1.x native sampling | 保留在 Stable Diffusion 暂停计划 |
| `S1` | Stable Diffusion 1.x full fine-tuning | 保留在 Stable Diffusion 暂停计划 |
| `X0` | Scale and New Families 总括 | 已拆为 Distributed、HPO、workflow 等独立方向，不再作为一个工作项 |

后来有文档把 `A2` 改写成 class-aware Evaluation readiness，并新增 `A3` 表示 AFHQ
learned-range-v quality validation。那是较晚时期的复用，不能覆盖 `5c75a76` 中 `A2` 的
Controlled AFHQ P2 Evidence 原意。

## Evaluation 计划中的 ID

| 旧 ID | `5c75a76` 中的含义 | 当前理解 |
| --- | --- | --- |
| `E0` | structured training outcome 与 phase metric 基础 | 已完成 |
| `E1` | standalone checkpoint Evaluation | 已完成 |
| `E2` | prediction artifact 与 offline scoring | 已完成 |
| `E3` | generation 与 super-resolution quality profiles | AFHQ generation 已完成；其他 profile 随对应任务交付 |
| `E4` | comparison、selection、gate 与 suite | Comparison/Gate/Suite 只保留为未排期研究；独立 Selection runtime 已被 validation-only training selection 和调用方/HPO/发布流程各自决策替代 |
| `E5` | 可选 extension/integration | 未实施；benchmark extension、人工评价 artifact、distributed Evaluation、inference-bundle subject 和置信区间等入口保存在 [Evaluation 设计附录](../post-training-evaluation-support-plan/design-notes.md) |

一次较晚的文档重构把 reference cache、性能和比较政策误写成 Evaluation 的 `D1`–`D3`。
基线计划没有这些编号，所以它们不是可追溯的历史 ID。

## Metrics、HPO、Extension 和 Consistency 计划中的 ID

| 所属计划 | 旧 ID | 当时含义 |
| --- | --- | --- |
| Metrics | `M0`–`M4` | 契约/命名、validation MetricEngine、Diagnostic monitoring、extension/distributed readiness、文档收束；当前 Metrics 已完成，Diagnostic 不参与模型选择 |
| HPO | `T0`–`T5` | single-run call、顺序 Grid/Random、adaptive search、local workers、统计/multi-objective、外部 launcher/provider；当前整个 HPO 功能暂停 |
| Extension performance | `Phase 0`–`Phase 5` | 冻结证据、拆 contract/implementation、Registry/bootstrap、lazy facade、迁移 CLI/examples、再次评估更细延迟激活；当前只保留测量驱动的性能候选 |
| Consistency | `Stage 0`–`Stage 6` | 固定算法、teacher bundle、pair construction、训练/target、student sampling、稳定性政策、质量文档；当前改用具体功能描述 |

## Latent 与 Stable Diffusion 计划中的 ID

| 旧 ID | 当时含义 | 当前理解 |
| --- | --- | --- |
| `LD2` / `LD3` | 更早版本中的 codec 与 AFHQ latent 阶段名 | 已由完整功能名称替代 |
| `LD4A` | prepared posterior moments artifact | Latent 生产数据功能 |
| `LD4B` | optimizer-step production loop | Latent 长训练功能 |
| `LD4C` | 可搬迁 codec asset bundle | Latent 资产保存功能 |
| Latent `Phase 0`–`Phase 8` | 边界、codec、AFHQ、prepared data、长训练、开放数据 DiT、ImageNet/DomainNet、非 DiT 替换 | 当前正文按用户流程说明；细节在设计附录 |
| Stable Diffusion `SD0`–`SD8` | 共享前置、reference backend、text assets、逐组件 sampling、图文 data、256/512 fine-tuning、random-init 与优化 | 当前暂停计划和设计附录；基线没有 `SD9`/`SD10` |

## Hydra 与 Sampling 文档中的 ID

| 旧 ID | `5c75a76` 中的含义 | 当前理解 |
| --- | --- | --- |
| `C0` / `C1` | plain-YAML train/sample authority cutover | 已完成 |
| `H0` | Hydra composition kernel | Hydra 候选功能的一部分 |
| `H1` | fresh single-run launcher | Hydra 候选功能的一部分 |
| `H2` | MNIST/AFHQ plain-YAML parity | Hydra 候选功能的一部分 |
| `H3` | scaffold、configuration reference 与文档 | Hydra 候选功能的一部分 |
| `H4` | 有限 multirun | 不属于第一版，只保存在研究附录 |

基线 `sampling-request-config-refactor.md` 没有 `R0`–`R2` 或 `R0`–`R5`。这些编号来自较晚的
post-Hydra 复查改写，不是 `5c75a76` 的历史阶段；当前读者文档已经删除它们。

## Workflow 与 Super-resolution 文档中的 ID

| 旧 ID | `5c75a76` 中的含义 | 当前理解 |
| --- | --- | --- |
| Workflow `R0` | 术语与 library operation API | 显式工作流候选中的稳定 Python operation |
| Workflow `R1` | recipe manifest 与 first-party catalog | 显式工作流候选中的配方目录 |
| `SR0` | deterministic super-resolution baseline | Super-resolution 候选的第一版 |
| `SR1` | SR Metrics 与正式 Evaluation | Super-resolution 候选的一部分 |
| `SR2` | conditional Gaussian super-resolution | 确定性版本完成后的保留方向 |
| `LG0` | latent generation recipe | 归 Latent Diffusion 候选 |
| Workflow `SD0` | Stable Diffusion recipe publication | 归 Stable Diffusion 暂停计划 |
| `CM0` / `CM1` | 重新核对 consistency 计划并交付首条完整流程 | 归 Consistency/Distillation 候选 |
| Workflow `R2` | inference bundle 与重复调用 Pipeline | 显式工作流的后续构想 |
| `W0` | 判断是否需要通用 workflow orchestrator | 只有两条稳定流程重复控制逻辑后才重开 |

`W0A` / `W0B` 是较晚的文档重构为了区分“显式顺序组合”和“通用编排器”而新增的标签，
不在 `5c75a76` 基线中。当前正文已用完整功能名称替代。

## Distributed 计划中的 ID

`D0`–`D7` 在基线 Distributed 计划中依次表示：固定行为与架构约定、进程会话/rank-zero I/O、
DDP training、distributed checkpoint、replicated sampling、FSDP2 training、FSDP2 inference
与多节点强化、精确 distributed Evaluation 与 elastic。当前计划先解释固定进程 DDP 和合并
sampling 的用户结果；其余保留在研究附录，不会因一个编号自动开始。

## 查阅规则

- 当前路线图、开发入口和功能计划不使用这些短 ID；
- 设计附录可保留 ID，以便对应旧提交，但必须同时写所属计划；
- 遇到同名 ID，先看文件和提交时期，不能只看字母数字；
- 新工作使用完整功能名称；只有真正进入实施后才考虑机器可读标识。
