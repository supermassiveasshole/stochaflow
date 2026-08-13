# Stochaflow 开发构想与计划

这里是开发文档的统一入口。根级 [`ROADMAP.md`](../../ROADMAP.md) 决定什么工作进入排期；
[当前开发队列](development-priority-roadmap.md)只记录已经选中的工作和跨计划前置关系。本页只帮
读者找到相关构想，不替路线图作决定。

- 正在实施：无
- 已经选定的下一项：无

“候选（Candidate）”表示一个完整用户功能可以被选择，但尚未排期；“暂停（Parked）”表示
想法继续保留，但启动它所需的前置能力或真实证据还不存在。两者都不表示功能已经实现。

## 已经能用的功能

这些能力已有维护代码、测试和使用文档，不需要再从开发计划里猜测完成度。

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 准备训练、验证和测试数据 | [Data pipeline](../configuration/data-pipeline.md) |
| 分别运行训练、生成和正式评估 | [配置与工作流](../configuration/workflows.md) |
| 继续中断的训练，或从 checkpoint 独立推理 | [Checkpoint 与迁移说明](../configuration/compatibility-and-migration.md) |
| 安装自己的数据、模型或任务扩展 | [扩展使用说明](../configuration/extensions.md)与[扩展 API](../api/extensions.md) |

## 新模型和新任务

- **[内置超分辨率](super-resolution-workflow-support-plan.md)（候选）**：训练一个 x4 超分模型，
  放大普通图片或生成结果，并用固定方法评估质量。
- **[Consistency 与蒸馏](consistency-distillation-support-plan.md)（候选）**：用较慢的教师模型训练
  一个生成更快、推理时不再需要教师的学生模型。
- **[Latent Diffusion](latent-diffusion-support-plan.md)（候选）**：先用固定的图片压缩模型跑通训练、
  继续训练、生成、解码和评估，再决定是否扩展到更复杂表示。
- **[Stable Diffusion 1.x](stable-diffusion-component-native-support-plan.md)（暂停）**：在 Latent 基础
  完成后，组合固定版本组件完成可核对的文字生图微调。

## 把已有操作接起来

- **[训练后蒸馏或生图后超分](default-workflow-pipeline-support-plan.md)（候选）**：先让对应操作能够
  单独运行，再把上一步正式结果可靠地交给下一步。
- **[通用工作流编排器](general-workflow-orchestrator-plan.md)（暂停）**：只有多条稳定流程反复编写
  相同的状态记录、失败后继续和重试逻辑时，才考虑公共编排能力。

## 配置、运行规模和自动化

- **[用 Hydra 组合训练配置](hydra-configuration-composition-migration-plan.md)（候选）**：在重复 YAML
  已成为真实问题时，先预览合并结果，再交给现有训练入口。
- **[固定单机的多设备训练](distributed-training-and-inference-support-plan.md)（暂停）**：在所需有效 batch
  和训练质量下，单设备任务仍无法满足吞吐或训练时间预算时，用固定数量的 Linux GPU 闭合一场
  DDP 训练；多进程采样随后单独验收。
- **[一次准备、跨机器运行的大规模数据管线](hierarchical-data-pipeline-support-plan.md)（暂停）**：把本地
  TB 级原始数据可恢复地准备成便携版本，复制后只更换机器资源 profile，再用有界管线训练。
- **[自动寻找训练参数](automated-model-tuning-plan.md)（暂停）**：隔离运行多次普通训练，只按
  validation 选择最佳候选，再独立运行正式 Evaluation。

## 只有证据出现才处理的问题

- **[Extension 启动速度复查](extension-import-boundary-and-activation-latency-plan.md)（暂停）**：Extension
  当前可以正确使用；只有可重复测量证明启动过慢时才定位瓶颈，也允许结论是“不修改”。

## 已完成说明与研究备忘

下面的页面保存当前用法、历史理由或以后可能重看的问题。它们不参与当前产品排期。

- [Evaluation 已完成：当前用法和历史想法](post-training-evaluation-support-plan.md)
- [Hydra 完成后的 Sampling 复查](sampling-request-config-refactor.md)
- [什么时候值得提取公共 Data 辅助函数](data-recipe-extension-ergonomics-plan.md)
- [以后怎样处理连续数据流](streaming-data-lifecycle-support-plan.md)
- [以后怎样接入新的数据存储形式](data-storage-and-payload-adapter-support-plan.md)
- [运行产物的说明、来源和资源记录为什么是三个独立问题](artifact-metadata-provenance-capacity-model-proposal.md)

更长的接口草案、公式、调研和历史记录位于 `docs/development/notes/`。主计划会在这些资料真正
支撑一个设计决定时说明为什么值得继续阅读；普通使用者不需要逐份浏览归档。
