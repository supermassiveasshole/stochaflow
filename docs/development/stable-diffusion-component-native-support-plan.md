# Stable Diffusion 1.x Component-Native 支持计划

> 工作状态：暂停
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

- 文档类型：未来能力计划；不属于公开 API 或用户文档
- 最近复核：2026-08-09
- 受众：model、training、sampling、data、Evaluation 与 extension 维护者
- 必需前置：
  [Latent Diffusion 支持计划](latent-diffusion-support-plan.md)完成并验收共享的
  codec、normalized latent、prepared posterior、asset persistence 与 decoded
  sampling/Evaluation 基础能力
- 研究资料：
  [完整设计与研究附录](notes/stable-diffusion-component-native-support-plan/design-and-research-notes.md)

本文只保存一个尚未获批的 Stable Diffusion 1.x 首个完整功能候选。实现、优先级和
产品声明只能由根路线图重新选择；附录中的 provider API、模型、数据集、硬件、配置草案
和任务分解都须在启动时重新核验，当前不构成承诺。

## 完成后用户能做什么

若本计划获批并通过全部验收，用户可以：

1. 从固定版本的 Hub snapshot、本地 Diffusers 目录或可搬迁的 Stochaflow run bundle 读取一套
   身份完整的 Stable Diffusion 1.x 组件；
2. 选择 Diffusers 参考后端，或选择由 Stochaflow 自己组合组件和采样步骤的原生后端；
3. 冻结图像 codec 与 text encoder，对预训练 conditional UNet 做 512×512 full-parameter
   fine-tuning，并支持严格恢复、checkpoint 采样和正式 Evaluation；
4. 用固定的数据、prompt、seed 和协议发布可重放结果，并清楚说明验证到哪一层兼容性。

首个候选范围是与 Stable Diffusion 1.x 兼容的 512×512 UNet 全参数微调。256×256 只用于早期
调试；Diffusers 参考后端只用于对照，二者都不能单独证明 Stochaflow 原生训练支持。

## 当前仓库已经支持什么

当前仓库只有可复用的通用基础，没有 Stable Diffusion 产品支持：

| 已有基础 | 当前保证 | 不代表什么 |
| --- | --- | --- |
| Gaussian Process、Dynamics 与 DDPM/DDIM Sampler | 对任务 tensor state 执行现有数学和数值流程 | 没有 Stable Diffusion Process，也未验证 SD 1.x schedule 或 CFG 一致性 |
| TrainingBuilder、managed auxiliaries 与 inference-asset projection | task Builder 可声明冻结资产并投影到 sampling/Evaluation | 没有 `AutoencoderKL`、text encoder、tokenizer 或 SD component bundle provider |
| checkpoint v12 与完整 sampling invocation | checkpoint 固定训练语义，sample config 提供单次 request policy | checkpoint 尚不包含 SD component、prompt 或 text-conditioning contract |
| `SamplingBuilder` 与 standalone/live Evaluation 基础能力 | task 可拥有 conditioning、guidance、writer-ready output 和正式 protocol | 没有 SD SamplingBuilder、fixed prompt suite 或 text-image Evaluation profile |
| schema-v2 `DataArtifact` 与 DataSource/DataBuilder 边界 | 可物化和严格绑定有身份的数据 | 没有 curated image-text、caption、prepared posterior 或 prepared text-embedding artifact |
| extension activation 与注册构造路径 | 外部 task 可在不改 core dispatch 的前提下接入 | 没有 Stable Diffusion extension project、registered model 或独立 provider contract test |

Latent Diffusion 计划本身也仍是候选。它列出的 codec/provider、prepared
posterior 与 decoded generation 不能被当作当前实现，更不能据此宣称 Stable Diffusion
sampling、fine-tuning 或质量支持已经存在。

## 还没有支持什么

| 缺口 | 所有者 | 必须避免的错误边界 |
| --- | --- | --- |
| 已验收的 frozen codec、latent transform、prepared posterior 与 asset persistence | Latent Diffusion 前置计划 | 在本计划复制 VAE loader、normalization、posterior schema、writer 或 checkpoint store |
| pinned component source、role、digest、license/safety 与离线恢复 | family-private provider + asset publication | 用 repo name 或 floating `main` 代替完整身份，或混合 revision |
| tokenizer、frozen text encoder、caption/tokenization 与 condition-dropout contract | task Builder/Strategy | 建立 universal tokenizer/condition registry，或让 DataBuilder 永久改写 caption |
| conditional UNet provider 与 pretrained/random-init identity | model provider + TrainingBuilder | Builder 替换注入的 primary，或把 fine-tuning 冒充 from-scratch training |
| black-box reference backend | 独立 inference backend | 把 `DiffusionPipeline` 包装成 native Sampler 或建立 dummy Process/checkpoint |
| component-native text conditioning、CFG、decode 与分层对照 | `SamplingBuilder` + family adapter | 让 `Sampler` 理解 text、UNet、VAE 或 Stable Diffusion |
| image-text source、caption artifact、runtime recipes 与 sample identity | DataSource/DataBuilder | 向 core batch schema 增加 nullable prompt 或 task-specific 字段 |
| 512 fine-tuning、strict resume、observability 与 checkpoint-only sampling | task TrainingBuilder/Strategy | 用 256 smoke、显存容量或短跑吞吐冒充正式能力 |
| fixed prompt suite、text-image metrics 与 immutable result identity | task EvaluationBuilder | 用人工挑选图片或单一指标替代正式证据 |
| 当前 provider、数据许可、安全策略和硬件证据 | 本计划负责人 | 沿用附录中的旧版本、估算或设备假设而不重测 |

## 什么时候可以开始或重新审查

只有同时满足以下条件，根路线图才能把本记录从暂停改为候选；真正实施还需要再
选为进行中：

1. 产品决策先选择并验收 Latent Diffusion 的首个完整功能；其 codec 约定、prepared
   posterior、run-level asset bundle、checkpoint-only decode 与正式 Evaluation 可由独立
   denoiser 复用；
2. 根路线图再明确选择 Stable Diffusion 1.x 的首个用户结果、支持声明和预算；
3. 维护者重新核验 optional Diffusers 版本、`AutoencoderKL`/conditional UNet/text
   encoder/tokenizer 来源、immutable revision、标准权重格式、许可、安全和离线获取策略；
4. 首个 image-text snapshot、caption materialization、数据许可、去重规则、sample plan、
   fixed prompt suite 与正式 Evaluation protocol 均有可执行草案；
5. component-native 路径所需 schedule、prediction、CFG 与 codec 一致性有明确对照和
   容差定义；数学不同时预先限制声明范围；
6. 单设备 profiling 先给出 batch、吞吐、activation memory、data wait、checkpoint 和
   sample-time 瓶颈；不得因预计模型规模预先启动 distributed；
7. 若需要改变公开 extension 约定、checkpoint schema、training loop 或配置解释规则，同一
   变更先更新对应根级规范；
8. black-box prototype 即使可运行，也只可作为参考后端证据，不能提前触发
   component-native 或 training 支持声明。

## 要完成哪些工作

### 任务卡：确定共享前置和首个完整功能

- **动作：** 引用 Latent Diffusion 已验收 contract，冻结 1.x-compatible 512×512
  full-UNet fine-tuning 结果及 black-box/native/fine-tuning/random-init 声明层级。
- **原因：** 防止复制 codec、latent、artifact、Sampler 或 writer，并避免模糊“支持”。
- **影响范围：** 两份计划的依赖、支持矩阵、数据与 Evaluation scope。
- **交付物：** 依赖清单、固定版本的参考 bundle、正式数据候选和对照 profile。
- **验证方法：** 复核 revision、许可、safety、离线恢复和每个声明的独立证据要求。
- **完成条件：** 共享前置只由 Latent Diffusion 计划负责，Stable Diffusion 独有结果可单独验收。

### 任务卡：建立隔离的参考后端

- **动作：** 加载 pinned、可离线的 Diffusers pipeline snapshot，统一 request、seed、
  writer、prompt suite 和 result manifest。
- **原因：** 需要隔离的对照基准，而不是伪装成 native `Sampler`。
- **影响范围：** reference backend、artifact manifest 和对照测试。
- **交付物：** Diffusers-owned scheduler/offload/inference loop 的独立 backend。
- **验证方法：** 离线重放相同 request/seed，并核对完整 result identity。
- **完成条件：** 不注入 `Process`/`Sampler`、不创建虚假状态，只作为后端对照证据。

### 任务卡：绑定组件与文本资产

- **动作：** 由 family-private provider 解析 bundle，构造 conditional UNet，并让 Builder
  验证 codec、tokenizer、text encoder、prediction 与 geometry identity。
- **原因：** component 与 text preprocessing 必须作为一个可恢复、可审计协作冻结。
- **影响范围：** provider、TrainingBuilder、checkpoint/run bundle 与 inference projection。
- **交付物：** immutable tokenizer、冻结 auxiliary assets 及 token/dropout/RNG contract。
- **验证方法：** 独立 fake provider、错误 identity、dtype/device 与 strict-resume tests。
- **完成条件：** sampling/Evaluation 从 bundle 恢复资产，用户不重复声明或拼装组件。

### 任务卡：完成 component-native sampling 并与参考后端对照

- **动作：** 由 task SamplingBuilder 组合 prompt、negative prompt、CFG、initialization、
  seed、decode 与 output，并复用 Gaussian Sampler。
- **原因：** 任务组合与数值算法必须继续由各自组件负责。
- **影响范围：** SamplingBuilder、family adapter、sample config 与 writer artifacts。
- **交付物：** 由 checkpoint 恢复的 component-native sampling 和分层对照报告。
- **验证方法：** 对 pinned oracle 比较 component、schedule、trajectory、decoded output 与分布。
- **完成条件：** Sampler 不解释 text/UNet/VAE；数学不同时只声明已证明的兼容层级。

### 任务卡：交付可审计的 image-text data

- **动作：** profiling The Met 候选并物化 versioned caption artifact；COCO 仅作多 caption
  reference，VLM recaption 必须形成新 artifact。
- **原因：** image-text 训练需要许可、来源、过滤、模板和 sample identity 可重放。
- **影响范围：** DataSource、DataArtifact、两条 DataBuilder recipe 与数据文档。
- **交付物：** snapshot/filter/template identity、stable sample key 和 frozen inventory。
- **验证方法：** 下载重放、去重、metadata、分辨率、caption 与 strict binding tests。
- **完成条件：** image-backed/prepared-backed 在 Dataset 前绑定，旧 captions 不被覆盖。

### 任务卡：完成 full-parameter fine-tuning

- **动作：** 先做 256 tiny overfit/短跑，再执行 512、64×64×4 latent 的预训练 UNet
  full fine-tuning。
- **原因：** 先验证冻结资产、dropout、resume 和 sampling，再承担正式训练成本。
- **影响范围：** TrainingBuilder/Strategy、optimizer-step lifecycle、EMA 与 bundle。
- **交付物：** bounded config、正式 config、Evaluation cadence、stop/resume contract。
- **验证方法：** online/prepared 数据路径一致性、raw/EMA sampling 和目标设备当前 1k-step 实测。
- **完成条件：** batch/accumulation/checkpointing 来自实测，256 路径不冒充 native result。

### 任务卡：发布正式 Evaluation 与支持声明

- **动作：** 版本化 prompt suite，并分离 reconstruction、distribution、alignment、coverage、
  memorization、performance 与 safety evidence。
- **原因：** 单一质量数值不能证明 component-native 语义或完整支持层级。
- **影响范围：** EvaluationBuilder、protocol、result bundle、报告与公开支持声明。
- **交付物：** immutable results 和完整 subject/component/data/prompt/sampler/seed identity。
- **验证方法：** 分布内、held-out、空/负/长 prompt、稀有组合、seed replay 与 CFG sweep。
- **完成条件：** 首个完整功能通过验收后，才评审 random-init、prepared text 或性能改进。

## 如何证明已经完成

首个正式能力必须同时满足：

- Latent Diffusion 的共享能力已由独立 denoiser 验收，本计划没有第二套 codec/latent
  lifecycle；
- pinned Hub/local/run-bundle 三种允许来源按已声明范围工作，混合 revision、digest mismatch、
  missing/corrupt asset、floating source 和 sampling overlay replacement 都会被明确拒绝；
- black-box 与 component-native backend ownership 在配置、执行和 manifest 中不混淆；
- tokenizer/text encoder/provider 有独立 fake implementation contract test，training 与
  sampling 不依赖未固定的远程状态；
- component-native sampling 的对照层级、基准、输入、timesteps、RNG 与容差可复核；
- 512 full-parameter UNet fine-tuning 可 strict pause/resume，并从 checkpoint-only raw/EMA
  subject 离线重放 sampling；
- sampling config 不重复或替换 checkpoint-owned component identity；
- curated image-text/caption artifact 可重建、严格绑定并拒绝身份或内容漂移；
- prompt suite 与正式 Evaluation 具备 exact counts、stable IDs、完整 protocol/provider/
  preprocessing identity、atomic publication 和非挑选式结果；
- 性能、容量和质量声明具有当前 hardware、software、data、protocol 和 artifact evidence；
- focused tests、independent-extension tests、ruff、pyright、配置 reference 检查和严格文档
  构建通过，SPEC/ARCHITECTURE/ROADMAP/CHANGELOG 与公开文档按实际行为同步；
- 支持声明严格区分 black-box inference、native sampling、pretrained full fine-tuning、
  random-init training 和后续 family，不用较窄能力暗示较宽能力。

## 明确不包含什么

- Stochaflow-native VAE、VAE training、joint VAE+UNet training 或任意 external VAE trainer；
- trainable text encoder、LoRA/PEFT、ControlNet、IP-Adapter、image-to-image 或 inpainting；
- SDXL、SD3、Flux-style transformer 或 arbitrary Diffusers Pipeline component graph；
- arbitrary `.ckpt` conversion、floating Hub revision、未声明的混合 component source；
- universal tokenizer、condition、pretrained-model graph、model-name compatibility matrix 或
  task-specific core batch schema；
- undocumented scraped dataset、未经审计的 community mirror 或数据许可背书；
- 1024 from-scratch production 或 web-scale foundation-model quality claim；
- 在 profiling 前承诺 prepared text embeddings、特定 attention、`torch.compile`、gradient
  checkpointing、sharded access、distributed training 或 bundle deduplication。

这些项目不是被删除的构想。可训练 text encoder、LoRA/PEFT、SDXL、SD3、Flux-style
transformer、其他 condition、prepared embeddings、VLM recaption、random-init 与 production
optimization 的重审触发和负责人见下表；首个完整功能不通过 nullable fields 提前实现它们。

## 详细设计和研究资料在哪里

| 未来方向 | 重审触发 | Owner |
| --- | --- | --- |
| 可训练 text encoder | 冻结 text encoder 的首版已经验收，正式消融证明它限制目标质量，并且额外训练状态、恢复和 Evaluation 规则已明确 | Stable Diffusion task `TrainingBuilder` 与 training runtime 负责人 |
| LoRA/PEFT | 全参数微调基线已验收，真实使用方需要更低显存或可搬迁 adapter，并能定义 base/adapter 身份与合并规则 | Stable Diffusion extension、checkpoint 与 adapter provider 负责人 |
| ControlNet、IP-Adapter、image-to-image、inpainting | 其中一个具体任务被路线图选择，输入 artifact、conditioning、数据和正式 Evaluation 已确定 | 对应任务的 `SamplingBuilder`/`TrainingBuilder`/`EvaluationBuilder` 负责人 |
| Prepared text embeddings | profiling 证明 tokenizer/text encoder 重复计算是主要成本，caption、provider 和 embedding 身份可以版本化 | Stable Diffusion DataSource/DataBuilder 与 text-asset provider 负责人 |
| SDXL | Stable Diffusion 1.x 原生路径已经验收，SDXL 被单独选择，双 text encoder、codec、UNet 和分辨率约定已重新核验 | 独立 SDXL task extension 负责人 |
| SD3 或 Flux-style transformer | 一个完整用例证明现有 Gaussian/UNet family 约定不能表达其数学与模型调用，并已冻结 provider、数据和 Evaluation | 新算法 family 与对应 text-to-image task 负责人 |
| 任意 Diffusers Pipeline component graph | 至少两个真实 Pipeline 重复同一稳定组件图语义，现有 component-native Builder 组合不足 | extension 架构维护者与两个消费任务负责人 |
| Random-init training | 预训练微调首版已验收，并有足够数据、计算预算、停止规则和正式质量目标 | Stable Diffusion training task 负责人 |
| VLM recaption | 选中的数据集确实需要重新生成 caption，并能固定 VLM、prompt、许可与新 artifact 身份 | image-text `DataSource` 与数据治理负责人 |
| 性能和生产优化 | 当前 profiling 指向明确瓶颈，且每项收益能独立验收 | 对应 attention/compile/checkpoint/distributed/bundle 负责人；不得打包启动 |

### 相关资料

- [完整设计与研究附录](notes/stable-diffusion-component-native-support-plan/design-and-research-notes.md)：
  保存组件职责、provider/bundle API、文本约定、数据集、caption、对照方法、training profile、
  硬件测量、配置草案、详细测试矩阵、风险和外部参考。附录用于未来重查，不是当前实现、排期
  或兼容性依据；重审触发与 Owner 以上表为准。
- [Latent Diffusion 支持计划](latent-diffusion-support-plan.md)：唯一负责共享 codec/latent 能力；
  未通过其验收前不得启动 component-native Stable Diffusion。
- [Evaluation 后续决策记录](post-training-evaluation-support-plan.md)：通用 Evaluation 基础能力
  已经实现；Stable Diffusion 的任务专用协议与证据仍由本计划交付。
