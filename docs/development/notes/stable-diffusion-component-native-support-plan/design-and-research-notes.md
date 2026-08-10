# Stable Diffusion 1.x Component-Native 设计与研究附录

- 性质：
  [主计划](../../stable-diffusion-component-native-support-plan.md)的非规范设计、调研和
  future-gate 保存记录
- 来源：2026-07-28 初稿及 2026-08-09 路线图复核时保留的候选内容
- 当前状态：Parked；不是实现事实、公开 API、排期承诺或 provider/license endorsement
- 规范权威：[`SPEC.md`](../../../../SPEC.md)、
  [`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) 与
  [`ROADMAP.md`](../../../../ROADMAP.md)
- 共享前置：
  [Latent Diffusion 支持计划](../../latent-diffusion-support-plan.md)

本附录有意完整保存原计划中不适合放入短主计划的细节。所有外部 API、模型格式、权重、
数据规模、许可、安全策略、硬件结论和配置形状都必须在路线图真正选择本能力后重新核验；
它们不能反向成为当前仓库支持的证据。

## 候选结果与能力声明边界

保留的首个候选路线是 Stable Diffusion 1.x-compatible composition：

- Diffusers `AutoencoderKL`；
- tokenizer；
- frozen text encoder；
- conditional `UNet2DConditionModel`；
- epsilon prediction；
- classifier-free guidance；
- 512×512 image / 64×64 latent profile；
- frozen VAE + frozen text encoder 下的 conditional UNet full-parameter fine-tuning；
- checkpoint-backed native sampling 与 task-owned formal Evaluation。

Stochaflow 候选 ownership 包括 training、resume、sampling、Evaluation、asset binding、
Gaussian schedule/Sampler 和结果 publication。Diffusers 只提供 component model、config、
标准权重格式，以及在 black-box 路径中由它完全拥有的 pipeline lifecycle。

下列能力必须分开声明：

| 能力 | 可证明的声明 | 不能推导的声明 |
| --- | --- | --- |
| 独立 pretrained codec | 共享 substrate 可固定并离线恢复 codec | SD sampling/training |
| black-box Diffusers pipeline | pinned pipeline 可重放且 ownership 清楚 | native Sampler/training |
| component bundle resolution | roles、revision、digests 可验证 | schedule/trajectory parity |
| component-native sampling | task Builder 与 Gaussian Sampler 通过已声明 parity | fine-tuning |
| pretrained full fine-tuning | frozen codec/text encoder 下的 strict-resume lifecycle | from-scratch |
| random-init UNet training | 相同 task contract 的 framework-ownership 验收 | 预训练质量预期 |
| prepared text embeddings | profiling 驱动的 recipe-specific artifact | universal cache |
| trainable text encoder | 新 optimizer/EMA/checkpoint lifecycle | 布尔 flag 增量 |
| LoRA/PEFT | 独立 adapter parameterization/checkpoint lifecycle | full-module 替代验收 |
| SDXL | 双 text encoder、micro-conditioning、1024 数据的新 family | SD 1.x nullable fields |
| SD3/Flux-style family | transformer/flow 与多 text asset 的新 family | SD 1.x UNet family |

fine-tuning 与 random-init 共享 codec/text encoder/tokenizer contract，但 checkpoint/result
identity、optimizer protocol、质量预期和能力声明必须不同。

## 与 Latent Diffusion 的责任边界

### 唯一共享 substrate

Latent Diffusion 计划先交付并验收：

- 窄 `ImageCodec` capability 与 Diffusers `AutoencoderKL` adapter；
- image range、latent affine transform、posterior sample/mode policy；
- codec source pinning、config/weights digest 与 immutable identity；
- prepared posterior moments artifact；
- model asset checkpoint/run bundle；
- Gaussian Process、Dynamics、DDPM/DDIM；
- step-based training、resume、local logging 与 `--no-progress`；
- reconstruction ceiling 与 decoded-generation Evaluation substrate。

Stable Diffusion 不建立第二套 VAE loader、latent normalization、posterior shard format、codec
binding、checkpoint asset store 或 sampling writer。只有上述 substrate 通过独立 denoiser
复用和离线搬迁验收后，本计划才可开始。

### Stable Diffusion 独有责任

- tokenizer/text encoder source resolution；
- caption/tokenization contract；
- conditional UNet model provider；
- text condition dropout 与 cross-attention condition adapter；
- Stable Diffusion CFG composition；
- SD 1.x component/schedule/trajectory parity；
- positive/negative prompt、prompt manifest 与 fixed prompt suite；
- image-text training/Evaluation protocol；
- pretrained UNet import、full-parameter fine-tuning 与独立 random-init profile。

DiT 与 Stable Diffusion UNet 是不同 model/backbone family，但都不产生新的 Process 或
Sampler root。不得增加 `StableDiffusionProcess`、universal `Condition` base、universal
tokenizer registry、universal pretrained-model graph、model-name compatibility matrix，或
同时容纳 class/domain/text/ControlNet/SDXL micro-conditioning 的公共 YAML 字段。兼容性由
具体 `TrainingBuilder` / `SamplingBuilder` composition root 验证。

## Component ownership 细节

### Training composition

```text
StableDiffusionTrainingBuilder
  ├── primary model: conditional UNet
  ├── frozen auxiliaries: image codec, text encoder
  ├── non-module asset: tokenizer
  ├── Gaussian Process and epsilon Objective
  ├── StableDiffusionTrainingStrategy
  └── optimizer / scheduler
```

- registered model provider 在进入 Builder 前构造 primary conditional UNet；
- Builder 不替换 primary，只验证 bundle、condition、prediction 和 geometry contract；
- Builder 解析 bundle，加载、冻结和声明 codec/text encoder；
- Strategy 只解释 batch 并执行 encode、text conditioning、model 和 Objective；
- Strategy 不下载、构造、移动、冻结或序列化模型资产；
- core 只管理声明 modules、一个 optimizer lifecycle、primary EMA、checkpoint 与 mode；
- tokenizer 是 immutable preprocessing asset，不伪装成 `nn.Module`；
- Process 不解释 prompt 或 cross-attention states。

component source 只在 `model` declaration 出现一次。provider 解析 source、构造 UNet，并让
primary 暴露 family-private、immutable bundle descriptor capability；Builder 从中取得同一
resolved identity，再加载其他 roles。这保持 primary 在 Builder 前构造，也避免在
model/training 两处重复 repo/revision；descriptor 不提升为通用 pretrained registry。

### Sampling composition

```text
checkpoint / component bundle
  -> tokenizer -> text encoder -> positive/negative hidden states
  -> StableDiffusionSamplingBuilder -> conditional/unconditional UNet dynamics
  -> CFG -> Gaussian Sampler -> normalized latent -> shared codec decode
  -> writer-ready image
```

SamplingBuilder 拥有 prompt batching、negative-prompt policy、truncation/reporting、
condition/uncondition concatenation、CFG scale、latent shape、generator/seed plan、decode 和
writer-ready image。Sampler 不知道 text、UNet、VAE 或 Stable Diffusion。

### Black-box pipeline backend

完整 `DiffusionPipeline` 可用于运行既有上游 pipeline、提供 parity oracle、复用 request/
seed/writer/Evaluation publication，以及验证 offline bundle 与 pinned revision。Diffusers
拥有 scheduler、offload 与 inference loop；Stochaflow 不注入 Process/Sampler，不创建 dummy
checkpoint、Process 或 Dynamics。manifest 必须记录 backend ownership；其输出不能证明
native training/sampling compatibility。

## Provider、bundle 与资产身份研究

### 候选 component source

首版只考虑 pinned Hugging Face Hub snapshot、local Diffusers-format directory 与 Stochaflow
run asset bundle。production profile 需要保存：

- immutable revision；
- component config digests 与 weight-file digests；
- tokenizer-file digests；
- component class/provider 与 expected subfolder layout；
- acquisition/offline-recovery facts；
- safety 与 license decision record。

禁止 floating `main`、未声明的混合 revision、sampling overlay 替换 checkpoint component、
只用 repo name 代表完整 identity、把 credentials 写入 manifest，或依赖原 remote cache/path。

### Family-private bundle roles

```text
codec
tokenizer
text_encoder
denoiser
```

这些 role 是 SD 1.x family 私有 schema，不提升为 universal model graph。native path 的 noise
schedule 由 Stochaflow Gaussian Process 配置与 checkpoint state 拥有；black-box pipeline 的
scheduler 由 Diffusers 拥有。两者不得形成双重 authority。

### Provider/API 重核要求

启动时重新核验：

- `AutoencoderKL`、`UNet2DConditionModel` 的构造、config 与权重布局；
- `DiffusionPipeline` component resolution、scheduler、offload 与 generator 行为；
- tokenizer/text encoder provider、hidden-state 与 attention-mask contract；
- Hub snapshot pinning、本地离线加载、license metadata 和 safety component；
- 上游 text-to-image example 的 optimizer、precision、EMA 与 validation 习惯。

针对锁定版本做 code-level probe、离线加载、精确 digest、错误 component、混合 revision 和
fake-provider tests；不能复制本文 API 假设直接实现。

## Text condition contract 研究

### Task-private batch 与 recipe facts

```python
{
    "image" | "posterior_moments": ...,
    "caption": str,
    "sample_key": str,
}
```

这是 recipe 私有 batch contract，不进入 core schema。resolved recipe 固定 tokenizer/text
encoder identity、maximum length、truncation、empty-caption probability、caption normalization、
Unicode/language、attention mask、hidden-state selection 与 dtype/device policy。

### Classifier-free condition dropout

dropout 属于 Strategy policy：使用 checkpointed run RNG；区分真实空 caption 与 dropout
empty；记录 probability；resume 后保持 batch-level replay；不让 DataBuilder 永久改写 caption。

### Prepared text embeddings gate

首版 on-the-fly tokenize/encode。只有 profiling 证明 text encoder 是显著 bottleneck 后才引入
prepared embeddings，其 identity 包含 caption artifact digest、tokenizer digest/config、text
encoder weights/config digest、max length、normalization/truncation、hidden-state selection、
output dtype/shape 与 shard inventory。它是 recipe-specific `DataArtifact`，不是 universal
embedding cache；任一输入或 policy 变化都必须使其失效。

## 数据与 caption 研究

### The Met Open Access 候选

首个开放候选 `met-open-curated-v1` 保存以下方向：public-domain image、CC0 metadata、
native-resolution filtering、deterministic snapshot、待 profiling 的 150k–300k 目标范围、
metadata-derived captions，以及 department/medium/period/culture/object-type 分层诊断。

确定性模板示意：

```text
{title}. A {object_name} from {culture_or_country},
dated {object_date}, made of {medium}.
```

模板是 source/recipe 私有 policy。正式 snapshot 需审计 metadata 缺失、Unicode、极端 aspect
ratio、重复 view、下载内容变化、caption/image mismatch 与 taxonomy freeze。

VLM recaption 是后续独立 artifact：固定 model/revision 与 generation parameters；保留 raw
output 和 normalized caption；审计 hallucination、图中文字、人物和敏感内容；不覆盖
deterministic captions。

### COCO 2017 reference

COCO 2017 保留为约 118k train images、每图多人工 caption 的 reference，用于 prompt/caption
variation 和 text-conditioned Evaluation；不把多对象图片压成单 class。image license/
provenance 仍按 snapshot 记录，不能因 annotation 开放而假定所有 pixel 许可一致。

### Research inventory

- LHQ：高质量 landscape，但 text 缺失，mirror/license 需审计；
- WikiArt：style/genre/caption 有研究价值，但版权和镜像一致性不足；
- Danbooru/SFW derivatives：tag 丰富，但版权、NSFW、重复和过滤成本高；
- `photo-concept-bucket`：高分辨率/caption 有吸引力，但 provenance 不清；
- PD12M：公开领域路线有价值，但 12M 规模与筛选超出首轮；
- Smithsonian Open Access：可评估未来 250k–500k thematic snapshot。

这些条目不构成 compatibility 或 license endorsement。正式使用前需冻结 source、许可、
snapshot digest、过滤、去重、sample/caption identity、预算与 Evaluation reference。

## Parity、training profile 与硬件研究

### 分层 parity

与 pinned Diffusers SD 1.x pipeline 的受控比较固定相同 component weights、positive/negative
prompt、tokenizer output、text hidden states、initial latent、generator state、timesteps、
prediction type、CFG algebra、scheduler coefficients 与 VAE transform/decode，并定义浮点容差
和 safety-policy 差异。

```text
component parity
schedule parity
trajectory parity
decoded-output tolerance
distribution-level compatibility
```

若 sampler 数学不同，只能声明“相同 components 上受支持的 Stochaflow sampler”，不能声明
step-by-step parity。

### Training profiles

256×256 只用于 contract smoke、memory/throughput profiling、tiny overfit、resume、caption
dropout 与 prepared-latent parity，不是 native-resolution 或质量证据。

512×512 候选正式 profile 使用 f8d4 codec 到 64×64×4 latent、frozen text encoder、epsilon
prediction、fixed schedule、full UNet fine-tuning、明确 EMA、prepared posterior、step-based
training、fixed prompt suite 与 reconstruction/generation Evaluation。

random-init 只有 fine-tuning 稳定后才评审：使用相同 dataset/codec/text encoder、固定随机
初始化和 step budget；只报告 dataset-domain quality，不与 web-scale model 虚假等价。

### 硬件候选与重测

原计划保留 RTX 4090 作为 single-device baseline/短期 fine-tuning 候选；DGX Spark 是否用于
production 取决于实测吞吐，不能由 unified memory 容量推断。启动后两个目标设备各跑固定
1k optimizer steps，报告 samples/s、data wait、activation memory、checkpoint time、sample
time、batch/accumulation/activation-checkpointing，以及 software/precision/data/protocol identity。

设备选择与 1k-step 方案均须重核。distributed、sharded access 或特殊 attention 只能由实际
瓶颈触发。

## Sampling 与 Evaluation 研究

fixed prompt suite 至少覆盖 in-distribution metadata、held-out composition、empty、negative、
long/truncated、rare metadata combinations、seed replay 与 CFG sweep；必须版本化，不能只展示
人工挑选图片。

首版至少评估 codec reconstruction ceiling、适合规模的 KID/FID、CLIP-style alignment、
condition bucket coverage、nearest-neighbor/memorization、duplicate-aware split、throughput/
VRAM/NFE 与 safety/manual review。单一指标不能决定 promotion；还须保存固定样本、failure
taxonomy、provider/preprocessing identity 与 exact completeness。

每个正式 result 绑定 checkpoint/component bundle、dataset/caption、prompt suite、sampler/
schedule、CFG、seed plan、codec、generated-file inventory 与 Evaluation protocol，并服从
portable paths、private staging 与 atomic publication。

## 非规范配置草案

具体语法可能被 Hydra 或后续配置工作改变；以下只表达 recipe 私有参数与 authority。

```yaml
model:
  name: stable_diffusion_1x_unet
  params:
    components:
      provider: diffusers
      source: <pinned-hub-or-local-bundle>
      revision: <immutable-revision>
    initialization: pretrained

training:
  name: stable_diffusion_text_to_image
  params:
    text:
      max_length: 77
      empty_probability: 0.1
      truncation: report
    latent:
      source: prepared_posterior
    prediction:
      type: epsilon
```

```yaml
sample:
  sampler:
    name: ddpm
    params: {}
  options:
    weights: ema
    prompts: prompts.yaml
    negative_prompt: ""
    guidance_scale: 7.5
    height: 512
    width: 512
  num_samples: 16
  batch_size: 4
  seed: 42
  writers:
    - name: image
      params:
        grid_nrow: 4
        denormalize: true
```

sampling 从 checkpoint/run bundle 恢复 component identity；overlay 只能修改允许的 request
policy。

## 候选执行分解与退出证据

这些是原计划工作顺序的保存记录，不是已批准阶段。

### 验收共享 substrate

- frozen codec 可随 checkpoint 离线恢复；
- prepared posterior 通过 parity、身份、损坏和中断测试；
- step-based resume、日志、controlled stop 与 completion semantics 可用；
- run-level codec bundle 通过 relocation/offline 验收；
- 512 reconstruction profile 和独立 denoiser复用通过。

black-box prototype 可在后段隔离进行，但不是 latent 前置，也不触发 native 声明。

### 建立 reference backend

pinned pipeline load、offline snapshot、request/seed/writer、fixed prompts、result manifest 与
backend ownership 全部闭合；只声明 parity oracle/独立 backend。

### 建立 text assets

tokenizer/frozen text encoder provider、digest/pinning、tokenization/truncation、两种 empty
caption、run-bundle projection 与 fake provider tests 通过；training/sampling 不依赖 floating
Hub state。

### 建立 component-native sampling

conditional UNet adapter、text-conditioned Dynamics adapter、CFG、schedule/prediction validation、
shared decode 与分层 parity report 完成；声明仅到实际证明层级。

### 建立 image-text runtime data

Met profiling、curated snapshot、deterministic caption artifact、image/prepared recipes、Dataset
前 strict binding 与 COCO reference 完成；recipe 不污染 core schema。

### 完成 bounded fine-tuning bring-up

pretrained full-parameter UNet、frozen auxiliaries、strict resume、local logger/no-progress、tiny
overfit、fixed prompts、online/prepared parity 与 checkpoint-only raw/EMA sampling完成；不从
256 结果声明正式质量。

### 完成 512 formal fine-tuning

当前硬件 1k-step benchmark、batch/accumulation/activation-checkpointing、fixed step budget、EMA、
Evaluation cadence、offline bundle 与 curated-data report 同时通过 reproducibility、quality 和
operational gates。

### 评审 random-init 与优化

random-init 使用相同 dataset/codec/text assets、固定初始化和独立报告，只证明 training
lifecycle。prepared embeddings、attention、`torch.compile`、gradient checkpointing、sharded
data、distributed 与 bundle deduplication 只按 profiling 选择，且不改变 caption、latent、
schedule、identity、resume 或 Evaluation contract。

## 详细测试矩阵

### Component source 与资产

- pinned Hub/local/run bundle、moving revision、missing/wrong component；
- mixed revision、digest mismatch、offline load、tokenizer files；
- config/weights collision、corrupt asset、overlay replacement rejected；
- relocation without original cache/source、credentials absent from manifests。

### Text

- Unicode/language、real empty 与 dropout empty、truncation/report、maximum length；
- attention mask、deterministic tokenization、hidden-state shape/dtype/selection；
- frozen text encoder、resume RNG、fake tokenizer/text encoder provider。

### UNet 与 training

- input/output channels、cross-attention dimension、prediction type、timestep dtype；
- optimizer excludes frozen modules、EMA primary-only、image/prepared parity；
- full-fine-tuning checkpoint、pretrained/random-init identity、strict resume/controlled stop；
- primary/bundle descriptor mismatch。

### Sampling 与 parity

- prompt/negative batching、CFG 1.0 和大于 1、initial latent replay；
- timestep equality、trajectory checkpoints、decode range、black-box ownership；
- component/schedule/trajectory declarations、decoded tolerance、256/512 identity；
- exact count、stable sample IDs 与 atomic publication。

### Data 与 caption

- Met API snapshot、public-domain/image filter、retry/content mutation；
- caption template、taxonomy freeze、duplicate views、extreme aspect ratio；
- caption/image mismatch、VLM identity、COCO multi-caption；
- snapshot/license/filter/digest drift、prepared binding before Dataset construction。

## 风险清单

| 风险 | 后果 | 候选缓解 |
| --- | --- | --- |
| 用“Diffusers 可加载”替代 parity | 输出漂移 | 分层 parity manifest |
| 把 black-box pipeline 包成 Sampler | 双重 lifecycle | ownership 分离 |
| 混合 component revision | components 不匹配 | roles + exact digests |
| sampling 重填 components | checkpoint 被 overlay | checkpoint-owned bundle |
| caption 现场生成 | run 不可重放 | versioned caption artifact |
| core 增加 nullable prompt | modality 污染 | recipe contract |
| 256 smoke 冒充 parity | 声明失真 | 512 formal profile |
| fine-tuning 冒充 from scratch | 结果误导 | init-profile identity |
| community mirror 直接发布 | 许可/NSFW/重复风险 | audit gate |
| 提前实现 SDXL/LoRA | 首个 lifecycle 不闭合 | 先关 SD 1.x vertical slice |
| 设备容量当作吞吐 | production 计划失真 | current benchmark |
| 上游 API/格式变化 | provider 假设失效 | 锁版本并重跑 probes |

## 保留的 future capability gates

### Trainable text encoder

需决定单 optimizer 或新 loop family、learning-rate groups、EMA、checkpoint role、prepared
embedding invalidation 与 bundle export；不能通过 `train_text_encoder: true` 偷渡。

### LoRA/PEFT

需定义 target modules、base-weight identity、adapter checkpoint、merge/unmerge、optimizer
ownership、sampling bundle 与 compatibility validation；不替代 full-module 首轮验证。

### SDXL

双 text encoders/tokenizers、pooled embeddings、time IDs/micro-conditioning、不同 codec/UNet、
1024/bucketed data 与更大 compute profile 需要新 concrete family plan。

### SD3 与 Flux-style family

transformer、multiple text encoders 与 flow formulation 需要独立评审 TrainingBuilder、数学、
Dynamics/Sampler、assets、data、Evaluation 与预算。

### 其他 conditions

ControlNet、image-to-image、inpainting、IP-Adapter 分别需要窄 capability、asset role、
training/sampling composition 与 Evaluation；第二个真实组合前不抽象 universal graph。

### VLM recaption

需先关闭 deterministic caption artifact，再固定 model/revision、generation parameters、raw/
normalized outputs 与 safety/hallucination audit；不覆盖原 captions。

### Prepared embeddings、random-init 与 production optimization

prepared embeddings 只由 bottleneck 触发；random-init 只在 fine-tuning 后启动；1024、web-scale
quality、distributed、attention、compile、sharding 与 deduplication 各需独立证据，不能从首个
512 fine-tuning 结果推导。

## 外部研究参考

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
- [Diffusers Stable Diffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- [Diffusers training overview](https://huggingface.co/docs/diffusers/training/overview)
- [Diffusers text-to-image training example](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
- [Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
- [The Met Open Access](https://www.metmuseum.org/hubs/open-access)
- [The Met Collection API](https://metmuseum.github.io/)
- [COCO](https://cocodataset.org/)
- [LHQ](https://arxiv.org/abs/2104.06954)
- [Public Domain 12M](https://huggingface.co/datasets/Spawning/PD12M)

## 维护说明

- 主计划用完整的设计理由说明何时启动、首版范围和完成证据；本附录不形成第二套路线图。
- 候选被选中后先重核 provider/data/hardware，再写最小 active implementation plan；不要把
  本附录整体当作承诺。
- 关闭后把稳定行为提升到根级权威、公开文档和 CHANGELOG，再处置一次性研究材料。
- future gate 若由新 owner 接管，应先留下链接再去重，不能因尚未实现而静默删除。
