# Stable Diffusion 1.x 逐组件支持计划

> 文档类型：功能计划
>
> 工作状态：暂停（Parked）
>
> 当前可用性：不可用。必须先完成 Latent Diffusion 的 codec、推理文件保存和图片解码能力，
> 才能开始 Stable Diffusion 1.x 支持。

## Stable Diffusion 不是一个可以直接塞进框架的模型文件

Stable Diffusion 1.x 由一组必须彼此配合的组件组成。codec 负责压缩和还原图片，tokenizer 把
提示词切成模型能读取的片段，text encoder 把这些片段变成数字特征，条件 UNet 再根据文字特征
预测怎样去噪。噪声变化规则、模型输出含义和 classifier-free guidance（CFG，无分类器引导）
也会改变最后的图片。

因此，“逐组件支持”不是把整个 Diffusers Pipeline 当作一个黑盒导入。Stochaflow 要分别加载、
检查和组合这些组件，让用户知道每一部分来自哪个固定版本，断网后仍能重新建立同一套推理，
并且能够解释 Stochaflow 与参考实现从哪一步开始出现差异。

仓库已有 Gaussian 训练、DDPM/DDIM 采样、可附带只读推理文件的 checkpoint 和独立 Evaluation，
扩展也可以在不修改核心分发代码的情况下接入。它们只能提供运行外壳：目前没有 Stable Diffusion
组件读取、文字处理、UNet 微调、逐组件采样或受维护的评估配置。

## 为什么这项工作必须等 Latent Diffusion

Stable Diffusion 的 UNet 并不直接处理图片，而是处理 codec 产生的 latent。如果框架还不能固定
codec 的版本和数值规则、保存可搬迁的 codec 文件、验证预计算 latent，并把生成结果可靠地解码
成图片，那么文字与 UNet 的问题会和底层图片压缩问题混在一起。此时即使能生成一张图，也无法
判断错误来自哪一层。

所以这项计划保持暂停。开始前，[Latent Diffusion 计划](latent-diffusion-support-plan.md)必须先用
另一个生成模型证明 codec、预计算的分布参数、运行目录搬迁和解码评估都可用。之后还要由根
[`ROADMAP.md`](../../ROADMAP.md) 把 Stable Diffusion 1.x 选为 `Next`，确定许可清楚的图文数据、
计算预算和明确支持范围。模型来源、Diffusers 版本、固定版本号（revision）、权重格式、许可、
安全与离线获取方式也要在实施前重新核对。

## 第一版只承诺一层明确的支持

“支持 Stable Diffusion”可能指四件范围完全不同的事：用原 Pipeline 黑盒推理、由 Stochaflow
逐组件采样、微调预训练 UNet，或者从随机权重训练整套模型。第一版交付的是 Stable Diffusion
1.x、512×512、预训练 UNet 的全参数微调，以及与之配套的逐组件采样。codec、tokenizer 和
text encoder 全部冻结。完成这层不能暗示已经支持从头训练或任意 Pipeline 组合。

256×256 小配置只用于尽快检查组件是否冻结、文字条件是否正确、中断后能否继续训练以及采样
能否运行。正式支持结果来自 512×512 图片和 64×64×4 latent 上的微调。训练参数、保存频率和
硬件范围依据当前机器的实测决定；不能因为模型看起来很大，就提前把分布式执行塞进首版。

用户会准备一套固定版本的组件，以及带许可说明、稳定图片编号和固定 caption（图片文字说明）的
训练数据。训练得到的 UNet checkpoint 可以继续训练，也能与随运行保存的冻结组件一起搬到另一台
机器。采样读取固定提示词，发布带 prompt、seed 和来源记录的图片；正式 Evaluation 再对完整
提示词集合给出质量、文字匹配、样本覆盖、记忆训练图片的风险、速度、显存和安全结果。

## 先保留一个独立的 Diffusers 参考答案

实施时会固定一份可离线读取的 Diffusers Pipeline 快照，作为隔离的参考后端。它继续使用
Diffusers 自己的噪声步调度器和生成循环，不伪装成 Stochaflow 的 `Process` 或 `Sampler`。
参考路径与逐组件路径共用 prompt、seed、输出格式和记录方式，这样差异才可以逐层定位。

比较顺序先从组件版本和噪声 schedule 开始，再检查中间 latent 轨迹与最终图片。如果两套数学
存在已知差异，文档只声明已经证明一致的层级，不能为了得到相同图片而隐藏不同算法，也不能用
“看起来差不多”代替可重放的对照。

## 文字输入和组件组合必须在运行前说清楚

组件读取器会支持固定模型仓库快照、本地目录和可搬迁文件包，并检查 codec、tokenizer、
text encoder、conditional UNet、模型输出含义和图片尺寸是否兼容。浮动的远程最新版、混合
revision、缺失文件以及未经明确设计的任意 `.ckpt` 转换都会被拒绝。

任务还要固定 caption 如何清理、怎样切成 token、过长文字怎样截断，以及训练时何时丢弃文字
条件（condition dropout）。逐组件采样由任务自己的组合代码处理 prompt、negative prompt
（不希望出现的内容）、CFG、初始 latent、随机种子、图片解码和输出；通用 Gaussian sampler
只负责数值步骤。checkpoint 已固定的组件不应要求用户在采样时重新填写一遍。

## 图文数据本身也是结果的一部分

The Met Open Access 可以作为首个公开数据候选，COCO 2017 只作为多 caption 对照。实施时要固定
下载快照、许可、过滤、去重、caption 模板和稳定样本编号。训练、采样与评估必须能证明使用的是
同一版本的数据，而不是仅记录一个目录路径。

如果以后用视觉语言模型（VLM）重写 caption，新文字必须保存为一份新的数据结果，并记录模型、
prompt、许可和安全审查，不能覆盖原 caption。这样，质量变化才能与文字数据的变化明确对应。

## 正式结果要覆盖完整提示词集合

评估会固定 prompt 集合和 seed，分别报告 codec 重建、生成图片分布、图文匹配、样本覆盖、
记忆风险、性能和安全。集合中要包括空 prompt、negative prompt、长 prompt、少见组合和不同
CFG 设置。不能靠人工挑图或一个总分宣布支持，也不能把训练过程中的日志当成正式 Evaluation。

当前独立评估机制见
[独立 checkpoint Evaluation](../configuration/workflows.md#独立-checkpoint-evaluation)。同一组
prompt 与 seed 还要能在 Diffusers 参考后端和 Stochaflow 逐组件后端中重放，并说明差异属于
组件、schedule、数值轨迹还是最终解码。

## “已支持”必须能够搬走、重放并解释

- 模型仓库快照、本地目录和可搬迁文件包按声明范围工作，错误版本或损坏组件在运行前报错；
- 参考后端与逐组件后端职责分开，相同 prompt/seed 的对照可以重放并逐层解释差异；
- tokenizer、text encoder 和组件读取器有独立替代实现测试，不依赖未固定的远程状态；
- 512×512 全参数微调可以中断后继续，只靠 checkpoint 文件包就能使用 raw/EMA 权重采样；
- 图文数据可以重建，caption、过滤或图片内容变化不会被静默接受；
- 正式结果覆盖完整 prompt 集合，不挑选样本，并记录硬件、软件、数据和采样设置；
- 测试、静态检查、配置参考和严格文档构建通过，公开说明准确区分支持层级。

## 第一版之后仍然保留的方向

| 未来方向 | 重新考虑的条件 |
| --- | --- |
| 同时训练 text encoder | 冻结 text encoder 的版本完成，对照证明它限制质量，并已设计学习率、EMA、保存和继续训练 |
| 用 LoRA/PEFT 降低显存和文件大小 | 全参数基线完成，用户确实需要更低显存或可搬迁 adapter，并能说明基础权重、合并与兼容规则 |
| ControlNet、IP-Adapter、image-to-image 或 inpainting | 一个具体控制任务被选择，输入、条件、数据和评估已经确定 |
| 预先保存文字特征（prepared text embeddings） | 实测证明文字编码是主要重复成本，caption 与特征版本可以固定 |
| 更高分辨率的 SDXL | 1.x 逐组件路径完成后，单独处理两个 text encoder、额外文字特征、尺寸信息和 1024 分辨率数据 |
| SD3 或 Flux 类模型 | 单独评审 transformer、多个 text encoder 和 flow 数学，不塞进 1.x UNet 任务 |
| 让用户任意组合 Diffusers 组件 | 至少两个真实 Pipeline 重复同一种稳定组合，现有任务组合确实不够 |
| 从随机权重训练 | 预训练微调完成，并具备足够数据、计算预算、停止规则和正式质量目标 |
| 用 VLM 重新生成图片说明 | 固定 VLM、prompt、许可和安全审查，生成新 caption 数据而不覆盖原文 |
| 注意力优化、编译、节省显存、多设备和文件去重 | 当前实测指出具体瓶颈，每项收益能够单独验证 |

第一版不训练 VAE，也不联合训练 VAE 与 UNet；不支持任意 Pipeline、未审计爬取数据、1024
从头训练或“达到互联网规模”的质量声明；也不建立通用 tokenizer、条件或模型图。

## 维护者资料

- [组件职责、文字输入规则、数据调研、对照方法、配置草案、测试矩阵和历史记录](notes/stable-diffusion-component-native-support-plan/design-and-research-notes.md)
  供实施时核对，普通使用者无需阅读。
- 行为和架构边界以 [`SPEC.md`](../../SPEC.md) 和 [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
  为准；它们不表示本功能已经实现。
