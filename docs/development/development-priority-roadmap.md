:orphan:

# Stochaflow 当前开发队列

- 文档性质：根级 [`ROADMAP.md`](../../ROADMAP.md) 的当前执行说明
- 进行中：[固定单机的多设备训练](distributed-training-and-inference-support-plan.md)
- 下一步：无
- 当前决定：先闭合 Linux 单机固定进程数的 DDP 训练，不把 FSDP2、弹性运行或多进程采样塞进首轮
- 全部计划入口：[按使用目的查找开发计划](README.md)

## 现在要做什么

当前实施[固定单机的多设备训练](distributed-training-and-inference-support-plan.md)。独立的
`DDPTrainer`、固定单机运行会话、任务自有的数据分工、全体进程共同更新、rank 0 完整
validation、固定拓扑继续训练 bundle 和普通 checkpoint-v12 导出已经进入源码，并由 CPU/Gloo
测试证明协调逻辑。当前工作是完成指定 Linux CUDA/NCCL 与 8×H200 的数值、性能、容量和故障
证据；这些证据闭合以前，本项仍不能标为 Done。

下一项尚未选择。本页不复制全部候选和暂停材料。

## 选择下一项以前要看清的前置关系

这些关系连接不同计划或复查记录，不表示其中任何一项已经排期。每份计划内部先做什么，
由该计划自己说明，不在这里重复。

- **[Latent Diffusion](latent-diffusion-support-plan.md) 再到
  [Stable Diffusion](stable-diffusion-component-native-support-plan.md)。** 先证明在压缩后的图像数据上可以训练、继续训练、
  生成并评估结果；这些基础稳定后，才开始依赖它们的 Stable Diffusion 微调和文字生图。
- **先完成单独操作，再接成[顺序工作流](default-workflow-pipeline-support-plan.md)，最后才考虑
  [通用编排](general-workflow-orchestrator-plan.md)。** [蒸馏操作](consistency-distillation-support-plan.md)
  可单独运行后，才能做“训练教师模型后蒸馏学生模型”；
  [超分操作](super-resolution-workflow-support-plan.md)可单独运行后，才能做“生成图像后做超分辨率”。
  首次从两条流程中选择一条接通，另一条继续保留。只有两条以上稳定流程反复需要相同的失败后
  继续、重试、结果交接或状态记录时，才讨论通用编排器。
- **[Hydra](hydra-configuration-composition-migration-plan.md) 完成后，
  [Sampling](sampling-request-config-refactor.md) 也不自动重构。** Hydra 先复用一个普通 Python 训练入口，解决
  全新训练配置的组合问题。只有迁移完成后仍出现真实的采样配置或重复推理问题，才重审
  Sampling 调用方式。
- **[大规模数据集管线](hierarchical-data-pipeline-support-plan.md)不等于
  [多设备执行](distributed-training-and-inference-support-plan.md)。** 可恢复的数据准备、跨机器复制、验证采用
  以及 PC/服务器单卡的有界读取都不依赖 DDP。只有固定八卡训练验收时，才依赖多设备计划的首个
  固定单机 DDP 交付，由它提供稳定的 rank/world size、共同更新和失败语义；并不依赖后续的多进程
  采样、FSDP2 或弹性运行。多设备运行不应猜测任务的数据布局，数据管线也不负责启动 DDP。

排期发生变化时，先更新根 `ROADMAP.md`，再把被选中工作的实际步骤写到本页。

```{toctree}
:maxdepth: 1
:caption: 开发文档页面
:hidden:

super-resolution-workflow-support-plan
consistency-distillation-support-plan
latent-diffusion-support-plan
stable-diffusion-component-native-support-plan
default-workflow-pipeline-support-plan
general-workflow-orchestrator-plan
hydra-configuration-composition-migration-plan
distributed-training-and-inference-support-plan
hierarchical-data-pipeline-support-plan
automated-model-tuning-plan
artifact-metadata-provenance-capacity-model-proposal
extension-import-boundary-and-activation-latency-plan
post-training-evaluation-support-plan
sampling-request-config-refactor
data-recipe-extension-ergonomics-plan
streaming-data-lifecycle-support-plan
data-storage-and-payload-adapter-support-plan
```
