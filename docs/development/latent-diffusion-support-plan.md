# Latent Diffusion 支持计划

> 工作状态：候选
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

- 文档类型：未来能力计划；不属于公开 API 或用户文档
- 最近复核：2026-08-09
- 受众：codec、data、training、sampling 与 Evaluation 维护者
- 下游依赖：
  [Stable Diffusion Component-Native](stable-diffusion-component-native-support-plan.md)
  必须复用本计划验收后的 codec/latent 基础能力；它不能反向把 text、UNet 或
  Diffusers Pipeline 语义引入本计划
- 研究资料：
  [设计与研究附录](notes/latent-diffusion-support-plan/design-and-research-notes.md)

本文只保存一个尚未获批的首个完整功能候选。实现、优先级和产品声明只能由根路线图
重新选择；附录中的旧阶段、API 草案、数据集和硬件结论均不是当前承诺。

## 完成后用户能做什么

若本计划获批并通过全部验收，用户可以：

1. 在训练配置中只声明一次冻结的预训练图像 codec，也就是负责图像编码和解码的模型；
2. 直接从图像或已经验证的预计算 latent 数据训练生成模型；
3. 只用 checkpoint 恢复训练出的模型、Gaussian `Process` 和 codec，完成采样、图像解码与
   artifact 发布；
4. 分开评估 codec 重建质量与最终生成质量，不让训练日志冒充正式结果；
5. 用一个非 DiT 模型走通同一流程，证明这套能力不依赖某个特定 backbone。

首个候选范围是“冻结的预训练图像 codec + conditional latent Gaussian diffusion”。DiT-S/2 与
DiT-B/2 只是用于验证的 denoiser，不是公共抽象。

## 当前仓库已经支持什么

当前仓库只具备可复用基础，不具备 Latent Diffusion 产品能力：

| 已有基础 | 当前保证 | 不代表什么 |
| --- | --- | --- |
| Gaussian Process、Dynamics 与 Sampler | 对 tensor state 执行既有训练和采样语义 | 不存在 `LatentProcess`，也未验证 latent recipe |
| `TrainingPlan.inference_assets` 与只读 inference projection | 显式资产可随 checkpoint 投影到 sampling/Evaluation | 没有内置 image codec provider、codec bundle 或 latent diagnostic |
| schema-v2 `DataArtifact` lifecycle | 可表达有身份、可验证的物化数据 | 尚无 posterior moments payload、source 或 prepared-latent recipe |
| 与任务无关的 Metrics 和 standalone/live Evaluation | 可由未来任务 Builder 声明正式 protocol | 没有 latent reconstruction 或 decoded-generation profile |
| checkpoint v12 与完整 sampling invocation | 固定训练语义并接受独立 sample request | 当前 checkpoint 不包含可恢复的 codec contract |

通用 inference asset 投影已经实现，但它只证明资产投影机制；不能据此宣称
Diffusers codec、latent training、prepared posterior、decoded sampling 或 latent quality
支持已经实现。

## 还没有支持什么

| 缺口 | 所有者 | 必须避免的错误边界 |
| --- | --- | --- |
| 冻结 codec capability 与 concrete provider | task adapter / extension | core 导入 Diffusers 或建立万能 VAE hierarchy |
| codec source、revision、weights/config digest 与离线恢复 | provider + checkpoint asset | sampling 重新填写 source，或 Hub `main` 静默漂移 |
| range、geometry、posterior、normalization 与 precision contract | codec adapter | 硬编码 scaling、重复 normalization、混搭 encoder/decoder |
| image-backed latent `TrainingBuilder`/`TrainingStrategy` | 任务自己的组合逻辑 | `Process`、`Sampler` 或 core runtime 解释 image/codec |
| prepared posterior artifact 与两条完整 data recipe | DataSource/DataBuilder | 用 `latent: true` 合并不同 lifecycle，或把模型权重当数据 artifact |
| optimizer-step 长训练流程与可观察性 | training runtime 决策 | 用零散 nullable flags 偷渡新的 loop family |
| run-level codec persistence | checkpoint/asset publication | 每个 checkpoint 重复大型权重，或依赖原 Hub/cache/path |
| decoded sampling 与正式 Evaluation profile | `SamplingBuilder`/`EvaluationBuilder` | 把 latent MSE 命名为图像重建质量 |
| 数据、硬件与质量证据 | task profile | 用 AFHQ smoke、显存容量或短跑吞吐冒充规模/质量结论 |

## 什么时候可以开始或重新审查

只有同时满足以下条件，根路线图才能把本记录选为进行中：

1. 产品决策明确选择 codec/latent 的首个完整功能，并固定首个用户结果；
2. 维护者重新核验 optional Diffusers 版本、首个 `AutoencoderKL` 权重来源、许可、
   immutable revision 与离线获取策略；
3. AFHQ-v2 只用于 correctness/smoke；正式目标的数据许可、原始分辨率、taxonomy、样本身份
   和预算需要另有可执行计划；
4. 正式 Evaluation 同时有 codec reconstruction 和 decoded generation 的任务专用
   protocol 草案；
5. 单设备 profiling 先给出真实瓶颈，不能因预计模型规模预先启动 distributed；
6. 若需要改变 training loop、checkpoint schema、公开 extension 约定或配置解释规则，
   同一变更先更新相应根级权威；
7. Stable Diffusion 仍保持下游：它不能成为提前扩大本计划 scope 的理由。

已完成的 pixel-space、Metrics、Evaluation、configuration 或 inference-asset 工作只是技术
前置，不自动授权启动。

## 要完成哪些工作

### 任务卡：确定 codec 约定与 provider

- **动作：** 定义窄 codec capability，并以可选 Diffusers `AutoencoderKL` adapter 验证
  geometry、posterior、transform、range、precision/upcast 与 identity。
- **原因：** codec 语义必须显式且可恢复，不能由上游 Pipeline 或硬编码 scaling 隐式拥有。
- **影响范围：** task adapter、TrainingPlan inference assets、checkpoint 与 reconstruction profile。
- **交付物：** 不可拆分的 encoder/decoder asset、固定版本的 provider 和独立 fake codec。
- **验证方法：** 替换性、错误 geometry/range/digest、离线恢复和无网络 forward tests。
- **完成条件：** core 不导入 Diffusers；训练只声明一次 codec；reconstruction 验收通过。

### 任务卡：交付最小但完整的 image-backed 功能

- **动作：** 用任务专用 Builder 组合 frozen codec、denoiser、Gaussian Process、Objective
  与 Strategy，完成 bounded train/resume/sample/evaluate。
- **原因：** 先证明完整流程，再承担 prepared data 或规模训练复杂度。
- **影响范围：** latent task extension、配置、checkpoint、sampling writer 与 Evaluation profile。
- **交付物：** AFHQ-v2 correctness config、小型 reference denoiser 和 raw/EMA result bundles。
- **验证方法：** fresh/resume、checkpoint-only decode、count/identity、range 与 clipping tests。
- **完成条件：** sample config 不重复 codec，报告只声明 functional correctness。

### 任务卡：物化 prepared posterior

- **动作：** 定义 posterior moments artifact，并分别交付 image-backed 与 prepared-backed
  `DataBuilder` recipe。
- **原因：** production training 需去除重复 encode，同时保持数据与 codec lineage。
- **影响范围：** DataSource/DataArtifactStore、payload、storage、DataBuilder 与 runtime RNG。
- **交付物：** stable keys、shards、inventory、codec/preprocessing/dtype identity 和原子发布。
- **验证方法：** online/offline 数据一致性、mutation、错误 revision、interrupt 与 strict binding
  tests。
- **完成条件：** sample/mode 与 fixed-view policy 明确，不虚构 online augmentation。

### 任务卡：完成生产训练和资产保存

- **动作：** 固定 optimizer-step 流程，并在测量重复成本后决定 run-level codec bundle。
- **原因：** 长训练、resume 与多 checkpoint 不能依赖 epoch 偶然长度或重复大型权重。
- **影响范围：** Trainer policy、checkpoint/asset publication、retention 与 local logging。
- **交付物：** 预算、频率、停止和完成约定，以及可搬迁的 offline bundle。
- **验证方法：** mid-epoch resume、controlled stop、missing/corrupt asset 与 relocation tests。
- **完成条件：** 若需新 loop family 则另行决策，不用 Strategy flags 绕过边界。

### 任务卡：建立开放数据与质量证据

- **动作：** profiling 并冻结开放数据、condition/sample plan，再按当前硬件实测推进
  DiT-S/2 与候选 DiT-B/2。
- **原因：** 数据许可、原始分辨率、codec ceiling 和吞吐决定正式质量声明是否成立。
- **影响范围：** The Met 候选、LHQ probe、ImageNet-100/DomainNet 验收与硬件配置。
- **交付物：** versioned snapshot/profile、reconstruction report、training/Evaluation results。
- **验证方法：** 许可与下载审计、原图检查、condition completeness、4090/Spark 1k-step 重测。
- **完成条件：** 不用 160-pixel 镜像冒充 256 source，并分离 codec 与 denoiser quality。

### 任务卡：证明可替换性并交接下游

- **动作：** 用独立 extension 提供的非 DiT denoiser 复用该能力，并把共享能力交给
  Stable Diffusion 计划。
- **原因：** 证明 Latent Diffusion 是工作流约定，而不是 DiT 或 SD 的别名。
- **影响范围：** extension compatibility、Builder validation 与两份计划的依赖边界。
- **交付物：** 非 DiT denoiser 的完整运行结果，以及共享 codec/latent/artifact/asset 约定清单。
- **验证方法：** substitution、checkpoint/resume/sample/evaluate 与无 core diff 审查。
- **完成条件：** 无 concrete/name/topology 分支；text、CFG 和 SD training 仍由下游拥有。

## 如何证明已经完成

功能完成需要同时具备：

- 一个公开权重 codec 和一个独立 fake codec 通过同一 capability、identity、offline 与失败
  contract；
- bounded fresh/resumed image-backed 训练以及 checkpoint-only raw/EMA sampling；
- prepared artifact 的完整性、确定性、身份漂移和中断恢复测试；
- relocation 后不依赖原 Hub cache/local source 的 run bundle；
- decoded output 的 exact sample count、stable IDs、atomic writers 与 replayable provenance；
- 任务专用 reconstruction 和 generation Evaluation，provider/preprocessing identity 非空，
  completeness 精确，training diagnostics 不冒充 benchmark；
- 独立非 DiT denoiser 在不修改共享 runtime 的情况下完成相同流程；
- focused tests、independent-extension tests、ruff、pyright、配置 reference 检查和严格文档构建；
- SPEC/ARCHITECTURE/ROADMAP/CHANGELOG 与公开使用文档按实际行为同步；
- 所有性能、容量和质量声明都有当前硬件、数据、protocol 与 artifact 身份证据。

## 明确不包含什么

- Stochaflow-native VAE、VQ-VAE、VQGAN 或通用 external VAE trainer；
- VAE 与 denoiser joint training、adversarial/multi-optimizer autoencoder loop；
- LPIPS/discriminator 公共抽象、万能 representation 或 pretrained-model registry；
- arbitrary checkpoint conversion、floating Hub revision、encoder/decoder 混搭；
- 首个 recipe 默认启用 learned variance；它需要独立的 latent 一致性和 Evaluation 验收；
- Stable Diffusion text/UNet/512 profile、完整 Diffusers Pipeline、SDXL、SD3 或 LoRA；
- ImageNet-1K、DiT-XL/2、video/VQ/RAE 等新 codec family；
- universal condition/batch schema 或 Dataset/Sampler/DataLoader registry；
- 通用 metadata/provenance/capacity service；
- DataLoader worker 写 cache、persistent read-through hybrid latent cache。

这些项目不是被删除的构想；它们的重审触发与负责人见下表。

## 详细设计和研究资料在哪里

| 未来方向 | 重审触发 | Owner |
| --- | --- | --- |
| Stochaflow-native VAE training | 一个被选择的完整任务确实需要在 Stochaflow 内训练 codec，冻结的外部 codec 无法满足；数据、重建 Evaluation 和训练预算已经明确 | 独立 VAE task extension 与对应 training-loop 负责人 |
| External VAE trainer provider | 至少两个外部训练工作流需要相同的可恢复 codec bundle、身份和导入失败语义 | codec provider/extension 负责人 |
| VAE 与 denoiser joint training | 正式证据表明冻结 codec 是主要质量瓶颈，并且多 optimizer 或交替更新的 training-loop family 已单独获批 | latent task `TrainingBuilder` 与独立 training-loop 负责人 |
| 新 codec family（VQ-VAE、VQGAN、RAE 或 video codec） | 被选择的任务需要不同 posterior、离散表示或时序 geometry，现有 codec 约定无法表达 | 对应任务 extension 与 codec capability 负责人 |
| Learned variance | 固定 variance 的 latent 首版已经验收，消融证明 learned variance 有质量收益，并有独立 latent Evaluation | Gaussian family 与 latent task 负责人 |
| Full Diffusers Pipeline | 下游完整产品明确需要 Pipeline 级兼容，而 component-native 组合不足；不得作为本计划首版捷径 | 对应下游产品计划负责人，首个 owner 为 Stable Diffusion 计划 |
| Text conditioning | Latent 基础已验收，具体 text-to-image 任务已选择 tokenizer、text encoder、数据和正式 Evaluation | Stable Diffusion 或其他 text-to-image task extension 负责人 |
| 通用 pretrained asset abstraction | 至少两个独立任务重复相同的资产身份、离线恢复与版本规则，现有 inference asset 约定不足 | inference/extension 架构维护者与两个消费任务负责人 |

### 相关资料

- [完整设计与研究附录](notes/latent-diffusion-support-plan/design-and-research-notes.md)：
  保存 codec API、posterior/precision policy、Diffusers provider、external VAE workflow、
  asset/checkpoint、prepared artifact、配置草案、数据集、硬件、测试矩阵和风险。附录用于未来
  重查，不是当前实现或排期依据；重审触发与 Owner 以上表为准。
- [Stable Diffusion Component-Native 计划](stable-diffusion-component-native-support-plan.md)：
  只在本计划共享能力通过验收并被路线图选择后启动。
- [Evaluation 后续决策记录](post-training-evaluation-support-plan.md)：通用 Evaluation
  已经实现；latent 的正式 profile 由本计划的首个完整功能交付。
