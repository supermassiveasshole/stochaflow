# Stable Diffusion Component-Native 支持计划

- 文档性质：开发计划；不属于当前公开 API 或正式用户文档
- 状态：提案，尚未进入实现
- 初始制定日期：2026-07-28
- 当前主线：Stable Diffusion 1.x-compatible component import、text
  conditioning、UNet training/fine-tuning 与 512×512 sampling
- 共享前置：
  [Latent Diffusion 支持计划](latent-diffusion-support-plan.md)
- 首个开放数据候选：冻结的 The Met Open Access curated image-text snapshot
- 对照数据候选：COCO 2017 captions
- 暂不等同于：任意 Diffusers Pipeline、SDXL、SD3、LoRA 或 VAE training

## 1. 结论

Stable Diffusion 是正式计划内能力，不是 Latent Diffusion 完成后再决定是否支持的
模糊 future item。

本计划冻结以下路线：

1. 先复用 Latent Diffusion 计划闭合的 frozen codec、normalized latent、
   prepared posterior artifact、checkpoint asset 和 Gaussian sampling
   lifecycle。
2. 首个 component-native family 聚焦 Stable Diffusion 1.x-compatible
   composition：
   - `AutoencoderKL`；
   - tokenizer；
   - frozen text encoder；
   - conditional `UNet2DConditionModel`；
   - epsilon prediction；
   - classifier-free guidance；
   - 512×512 image / 64×64 latent profile。
3. Stochaflow 拥有 training、resume、sampling、evaluation 和 asset binding；
   Diffusers 提供 component model implementations、config 和标准权重格式。
4. 完整 `DiffusionPipeline` black-box inference 是计划内独立 backend；
   它由 Diffusers 拥有 scheduler/offload/inference loop，不能伪装成
   Stochaflow native `Sampler`。
5. component-native interop 不能简单拆出上游 modules 后宣称兼容；必须验证
   codec transform、tokenization、text hidden states、prediction type、
   timestep/schedule、CFG、decode 和 safety policy 的 parity。
6. 首个训练能力是 frozen VAE + frozen text encoder 下的 conditional UNet
   full-parameter fine-tuning/training。
7. 预训练 full-parameter fine-tuning 和 random-init training 是两个明确
   profile；前者先落地，后者在同一 contract 上作为规模验收，不混用结果声明。
8. Stable Diffusion 不要求 Stochaflow 实现或训练 VAE。外部 VAE training
   仍服从共享 codec import contract。
9. SDXL、SD3、Flux-style transformer、LoRA/PEFT 和可训练 text encoder 各自
   需要新的 family/lifecycle gate，不通过 nullable fields 偷渡。

## 2. 与 Latent Diffusion 计划的边界

### 2.1 共享能力

由 Latent Diffusion 计划先提供：

- `ImageCodec` capability；
- Diffusers `AutoencoderKL` adapter；
- image range 与 latent affine transform；
- posterior sample/mode；
- codec source pinning 和 digest；
- prepared posterior moments artifact；
- model asset checkpoint/bundle；
- Gaussian Process、Dynamics 和 DDPM/DDIM；
- step-based training、resume、local logging 和 `--no-progress`；
- reconstruction ceiling 和 generation evaluation infrastructure。

Stable Diffusion 不复制这些实现，也不建立第二套：

- VAE loader；
- latent normalization；
- posterior shard format；
- codec binding；
- checkpoint asset store；
- sampling writer。

### 2.2 本计划独有能力

本计划拥有：

- tokenizer/text encoder source resolution；
- caption/tokenization contract；
- conditional UNet model provider；
- text condition dropout；
- cross-attention condition adapter；
- Stable Diffusion CFG composition；
- SD 1.x component parity；
- text-to-image prompts、negative prompts 和 prompt manifests；
- image-text training/evaluation protocol；
- pretrained UNet import 与 full-parameter fine-tuning。

### 2.3 明确不共享的抽象

DiT 和 Stable Diffusion UNet 是两个 model/backbone family，但都不成为新的
Process 或 Sampler root。

不增加：

- `StableDiffusionProcess`；
- universal `Condition` base class；
- universal tokenizer registry；
- universal pretrained-model graph；
- model-name compatibility matrix；
- common YAML field 同时容纳 class、domain、text、ControlNet 和 SDXL
  micro-conditioning。

兼容性在具体 `TrainingBuilder` / `SamplingBuilder` composition root 验证。

## 3. 支持层级

| 层级 | 能力 | 计划状态 |
| --- | --- | --- |
| SD0 | 独立 pretrained `AutoencoderKL` codec | 共享前置 |
| SD1 | 完整 Diffusers Pipeline black-box inference | 计划内独立 backend |
| SD2 | SD 1.x component bundle resolution 和 parity validation | 首个 component milestone |
| SD3 | Stochaflow native SD 1.x sampling | 首个 component milestone |
| SD4 | frozen text encoder + full UNet fine-tuning | 首个 training milestone |
| SD5 | random-init UNet text-conditioned training | 规模验收 |
| SD6 | prepared text embeddings | production optimization gate |
| SD7 | trainable text encoder | 新 optimizer/asset lifecycle gate |
| SD8 | LoRA/PEFT | 独立 parameterization/checkpoint gate |
| SD9 | SDXL | 双 text encoder/micro-conditioning family gate |
| SD10 | SD3 | transformer/flow family gate |

每个层级分别声明；实现 SD1 不能声称支持 SD4，实现 SD4 不能声称支持 SDXL。

## 4. Component ownership

### 4.1 Training composition

```text
StableDiffusionTrainingBuilder
  ├── primary model
  │     └── conditional UNet
  ├── frozen auxiliaries
  │     ├── image codec
  │     └── text encoder
  ├── non-module assets
  │     └── tokenizer
  ├── Gaussian Process
  ├── epsilon Objective
  ├── StableDiffusionTrainingStrategy
  └── optimizer / scheduler
```

责任约束：

- Builder 解析并验证 component bundle；
- Builder 加载、冻结和声明 codec/text encoder；
- Builder 构造 primary UNet；
- Strategy 只解释 batch、执行 encode/text conditioning/model/objective；
- Strategy 不下载或构造模型资产；
- core 只管理声明的 modules、optimizer、EMA、checkpoint 和 device/mode；
- tokenizer 是 immutable preprocessing asset，不伪装成 `nn.Module`；
- Process 不解释 prompt 或 cross-attention states。

### 4.2 Sampling composition

```text
checkpoint / component bundle
  -> tokenizer
  -> text encoder
  -> positive/negative hidden states
  -> StableDiffusionSamplingBuilder
  -> conditional/unconditional UNet dynamics
  -> CFG
  -> Gaussian Sampler
  -> normalized latent
  -> shared codec decode
  -> writer
```

SamplingBuilder 拥有：

- prompt batching；
- negative prompt policy；
- tokenizer truncation/reporting；
- condition/uncondition concatenation policy；
- CFG scale；
- latent shape；
- generator/seed plan；
- decode；
- writer-ready image。

Sampler 不知道 text、UNet、VAE 或 Stable Diffusion。

### 4.3 Black-box pipeline backend

black-box `DiffusionPipeline` 路径用于：

- 运行已存在的上游 pipeline；
- 对 component-native sampling 做 parity/reference；
- 统一 request、seed、writer 和 Evaluation；
- 测试 offline bundle 和 pinned revision。

该路径中：

- Diffusers 拥有 scheduler 和 inference loop；
- Stochaflow 不注入自己的 Process/Sampler；
- 不创建 dummy checkpoint、Process 或 Dynamics；
- result manifest 必须记录 backend ownership；
- black-box 输出不能证明 native training/sampling compatibility。

## 5. Component bundle 与 pretrained support

### 5.1 首版 source

支持：

1. pinned Hugging Face Hub snapshot；
2. local Diffusers-format model directory；
3. Stochaflow run asset bundle。

production profile 必须解析为：

- immutable revision；
- component config digests；
- weight file digests；
- tokenizer file digests；
- component class/provider；
- expected subfolder layout；
- safety/license decision record。

不允许：

- production 中使用 floating `main`；
- VAE、UNet、text encoder 来自未经声明的混合 revision；
- sampling overlay 静默替换 checkpoint-owned component；
- 仅用 repo name 代表完整 component identity；
- 把 credentials 写入 manifest。

### 5.2 Bundle roles

首版固定角色：

```text
codec
tokenizer
text_encoder
denoiser
noise_schedule
```

这些 role 是 SD 1.x family 的私有 composition schema，不提升为 universal
framework model graph。

### 5.3 Pretrained 和 random-init profile

必须区分：

| profile | denoiser init | 用途 |
| --- | --- | --- |
| `sd1_full_finetune` | pinned pretrained UNet | 首个可见质量和 resume 验收 |
| `sd1_random_init` | config-pinned random initialization | 从零训练和 framework ownership 验收 |

两者共享 codec/text encoder/tokenizer contract，但：

- checkpoint/result identity 不同；
- optimizer/learning-rate protocol 不同；
- 质量预期不同；
- 不能把 fine-tuning 结果描述为 from-scratch training。

## 6. Text condition contract

### 6.1 Caption facts

正式 training batch 至少提供：

```python
{
    "image" | "posterior_moments": ...,
    "caption": str,
    "sample_key": str,
}
```

这是具体 Stable Diffusion recipe 的 batch contract，不进入 core universal schema。

resolved recipe 固定：

- tokenizer identity；
- text encoder identity；
- maximum token length；
- truncation policy；
- empty-caption probability；
- caption normalization；
- Unicode/language policy；
- attention-mask policy；
- hidden-state selection；
- dtype/device policy。

### 6.2 Classifier-free condition dropout

training dropout 属于 Strategy policy，必须：

- 使用 checkpointed run RNG 语义；
- 与 dataset 中真实空 caption 区分；
- 记录 probability；
- 在 resume 后保持 batch-level replay contract；
- 不通过 DataBuilder 永久改写 caption。

### 6.3 Prepared text embeddings

首版先 on-the-fly tokenize/encode。只有 profiling 证明 text encoder 成为显著
bottleneck 后，才引入 prepared embeddings artifact。

若引入，identity 必须包含：

- caption artifact digest；
- tokenizer digest/config；
- text encoder weights/config digest；
- max length；
- normalization/truncation；
- hidden-state selection；
- output dtype/shape；
- shard inventory。

它是一个 recipe-specific DataArtifact，不是 universal embedding cache。

## 7. Dataset 路线

### 7.1 The Met Open Access

首个开放正式候选是共享计划中的 `met-open-curated-v1`：

- public-domain image；
- CC0 metadata；
- native-resolution filtering；
- deterministic snapshot；
- 150k–300k 目标范围，而非预先强制数量；
- metadata-derived captions；
- department/medium/period/culture/object-type 可用于诊断和分层评估。

第一版 caption 使用确定性模板，示意：

```text
{title}. A {object_name} from {culture_or_country},
dated {object_date}, made of {medium}.
```

模板只是 materialization recipe，不进入 framework 公共 API。

VLM recaption 是后续独立 artifact：

- 固定 model/revision；
- 固定 generation parameters；
- 保存 raw output 和 normalized caption；
- 进行 hallucination、文字、人物和敏感内容审计；
- 不覆盖 deterministic captions。

### 7.2 COCO 2017

COCO 作为多对象自然语言 reference profile：

- 约 118k train images；
- 每图多条人工 caption；
- 验证 prompt/caption variation；
- 验证 text-conditioned evaluation；
- 不把多对象图片压成单 class condition。

COCO 的 image license/provenance 仍需按具体 snapshot 记录，不因 annotation
开放而假定所有 pixel 具有统一许可。

### 7.3 Research inventory

以下候选保留在 research inventory，不作为首个官方 source：

- LHQ：高质量 landscape，但 text 缺失且 mirror/license 需审计；
- WikiArt：style/genre/caption 很有趣，但版权和镜像一致性不足；
- Danbooru/SFW derivatives：tag 丰富，但版权、NSFW、重复和质量过滤成本高；
- `photo-concept-bucket`：高分辨率/caption 有吸引力，但 provenance 不清；
- PD12M：公开领域路线有价值，但 12M 规模和质量筛选超出首轮；
- Smithsonian Open Access：适合未来 250k–500k thematic snapshot。

研究 inventory 中的存在不构成 framework compatibility 或 license endorsement。

## 8. Stable Diffusion parity

component-native 路径晋升前，必须与一个 pinned Diffusers SD 1.x pipeline 做
受控 parity：

- 同一 component weights；
- 同一 prompt/negative prompt；
- 同一 tokenizer output；
- 同一 text hidden states；
- 同一 initial latent 和 generator state；
- 同一 timesteps；
- 同一 prediction type；
- 同一 CFG algebra；
- 同一 scheduler coefficients；
- 同一 VAE transform/decode；
- 明确允许的 floating-point tolerance。

若 Stochaflow sampler 与上游 scheduler 数学不同，只能声明“同一 model
components 上的受支持 Stochaflow sampler”，不能声明 step-by-step parity。

parity manifest 必须区分：

```text
component parity
schedule parity
trajectory parity
decoded-output tolerance
distribution-level compatibility
```

## 9. Training profiles

### 9.1 256 bring-up

256×256 只用于：

- contract smoke；
- memory/throughput profiling；
- tiny overfit；
- checkpoint/resume；
- caption dropout；
- prepared latent parity。

它不能被描述为预训练 SD 1.x 的 native-resolution parity。

### 9.2 512 formal profile

正式 SD 1.x-compatible profile：

- 512×512 image；
- f8d4 codec -> 64×64×4 latent；
- frozen text encoder；
- epsilon prediction；
- fixed noise schedule；
- full UNet fine-tuning；
- EMA policy 明确；
- prepared posterior production path；
- step-based training；
- fixed prompt suite；
- reconstruction/generation evaluation。

硬件决策：

- RTX 4090 用于 single-device baseline 和短期 fine-tuning；
- DGX Spark 是否用于 production 取决于实测吞吐，不由 unified memory 决定；
- 两端各跑固定 1k optimizer steps；
- 比较 samples/s、data wait、activation memory、checkpoint time 和 sample time。

### 9.3 Random-init profile

只有 full fine-tuning 路径稳定后才进入：

- 同一 curated dataset；
- 同一 codec/text encoder；
- random-init conditional UNet；
- 固定 training-step budget；
- 不承诺达到通用 Stable Diffusion 能力；
- 报告 dataset-domain quality，而不是与 web-scale foundation model 做虚假等价。

## 10. Sampling 与 Evaluation

### 10.1 Fixed prompt suites

至少包含：

- in-distribution metadata prompts；
- held-out composition prompts；
- empty prompt；
- negative prompt；
- long/truncated prompt；
- rare metadata combinations；
- seed replay；
- CFG scale sweep。

prompt suite 必须版本化，不能只展示人工挑选图片。

### 10.2 Metrics

第一版至少报告：

- codec reconstruction ceiling；
- KID/FID 或适合数据规模的 distribution metric；
- CLIP-style prompt-image alignment；
- condition bucket coverage；
- nearest-neighbor/memorization audit；
- duplicate-aware train/reference split；
- generation throughput、VRAM 和 NFE；
- safety/manual review report。

指标不能单独决定 promotion。必须同时保留固定样本和 failure taxonomy。

### 10.3 Writer artifacts

每个正式 generation result 绑定：

- checkpoint/component bundle identity；
- dataset/caption artifact identity；
- prompt suite identity；
- sampler/schedule；
- CFG；
- seed plan；
- codec；
- generated-file inventory；
- evaluation protocol。

## 11. Configuration sketch

以下仅表示 concrete recipe 私有参数，不是最终公共 schema：

```yaml
training:
  builder:
    name: stable_diffusion_text_to_image
    params:
      components:
        provider: diffusers
        source: <pinned-hub-or-local-bundle>
        revision: <immutable-revision>
      initialization:
        denoiser: pretrained
      text:
        max_length: 77
        empty_probability: 0.1
        truncation: report
      latent:
        source: prepared_posterior
      prediction:
        type: epsilon
```

sampling config 不重复 component source：

```yaml
sampling:
  builder:
    name: stable_diffusion_text_to_image
    params:
      prompts: prompts.yaml
      negative_prompt: ""
      guidance_scale: 7.5
      height: 512
      width: 512
```

sampling 从 checkpoint/run bundle 恢复 component identity；用户 overlay 只能修改
明确允许的 request policy。

## 12. 实施阶段

### Phase SD0：共享前置验收

- Latent Diffusion 计划 Phase 1–4 完成；
- frozen codec 可以 checkpoint/sample 恢复；
- prepared posterior artifact 已验证；
- step-based resume 和本地日志可用；
- 512 codec reconstruction profile 通过。

退出条件：Stable Diffusion 不需要复制 latent lifecycle。

### Phase SD1：black-box reference backend

- pinned Diffusers pipeline load；
- offline snapshot；
- request/seed/writer；
- fixed prompt suite；
- result manifest；
- backend ownership 明确。

退出条件：可作为 component-native parity oracle，但不声称 native support。

### Phase SD2：text assets

- tokenizer provider；
- frozen text encoder provider；
- digest/pinning；
- tokenization/truncation；
- empty-caption dropout；
- checkpoint/run-bundle projection；
- independent fake provider contract test。

退出条件：training/sampling 不依赖 floating Hub state。

### Phase SD3：component-native sampling parity

- conditional UNet adapter；
- text-conditioned Dynamics；
- CFG composition；
- SD 1.x schedule/prediction validation；
- shared codec decode；
- black-box/component-native parity report。

退出条件：支持级别只能声明到实际通过的 parity 层。

### Phase SD4：image-text DataSource/DataBuilder

- Met Open profiling；
- curated snapshot；
- deterministic caption artifact；
- image-backed/prepared-posterior batch recipes；
- strict binding before Dataset construction；
- COCO reference profile。

退出条件：caption/data recipe 不污染 core batch schema。

### Phase SD5：256 full fine-tuning bring-up

- pretrained UNet full-parameter fine-tuning；
- frozen codec/text encoder；
- checkpoint/resume；
- local logger/no-progress；
- tiny overfit；
- fixed prompts；
- online/prepared parity。

退出条件：完整训练和独立 sampling 可重放。

### Phase SD6：512 formal fine-tuning

- 4090/Spark 1k-step benchmark；
- batch/accumulation/activation-checkpointing 决策；
- fixed optimizer-step budget；
- EMA；
- evaluation cadence；
- offline run bundle；
- Met curated formal report。

退出条件：达到正式 profile 的 reproducibility、quality 和 operational gates。

### Phase SD7：random-init UNet gate

- fixed random initialization；
- same dataset/codec/text assets；
- from-scratch training report；
- 与 full fine-tuning 严格区分；
- dataset-domain capability statement。

退出条件：证明 Stochaflow 拥有训练 lifecycle，不声称复现 web-scale foundation
model。

### Phase SD8：production optimizations

按 profiling 选择：

- prepared text embeddings；
- attention implementation；
- `torch.compile`；
- gradient checkpointing；
- sharded data access；
- distributed training；
- component bundle deduplication。

优化不得改变 caption、latent、schedule 或 resume contract。

## 13. 测试矩阵

### 13.1 Component source

- pinned Hub/local/bundle；
- moving revision；
- missing/wrong component；
- mixed revision；
- digest mismatch；
- offline load；
- tokenizer files；
- config/weights collision；
- corrupt asset；
- sampling overlay replacement rejected。

### 13.2 Text

- Unicode；
- empty caption；
- real empty vs dropout empty；
- truncation；
- maximum length；
- attention mask；
- deterministic tokenization；
- hidden-state shape/dtype；
- frozen text encoder；
- resume RNG；
- fake tokenizer/text encoder provider。

### 13.3 UNet/training

- input/output channels；
- cross-attention dimension；
- prediction type；
- timestep dtype；
- optimizer excludes frozen modules；
- EMA primary-only；
- image-backed/prepared-backed parity；
- full fine-tuning checkpoint；
- random-init identity；
- strict resume。

### 13.4 Sampling/parity

- prompt/negative prompt batching；
- CFG 1.0 and greater-than-1；
- initial latent replay；
- timestep equality；
- trajectory checkpoints；
- decode range；
- black-box ownership；
- component/schedule/trajectory parity declarations；
- 256 vs 512 profile identity。

### 13.5 Data

- Met API snapshot；
- public-domain/image filter；
- download retry/content mutation；
- metadata caption template；
- taxonomy freeze；
- duplicate views；
- extreme aspect ratio；
- caption/image mismatch；
- VLM recaption distinct identity；
- COCO multi-caption selection。

## 14. 完成标准

Stable Diffusion 首个正式 milestone 只有同时满足以下条件才完成：

- Stable Diffusion 仍复用共享 codec/latent lifecycle；
- black-box 和 component-native ownership 不混淆；
- pinned SD 1.x bundle 可离线 sampling；
- component-native sampling parity 有明确报告；
- 512 full UNet fine-tuning 可 pause/resume；
- sampling config 不重复 component identity；
- Met curated image-text artifact 可重建并严格绑定；
- fixed prompt suite 和非挑选式 metrics 完整；
- 4090/Spark profile 有实测数据；
- 文档只声明实际达到的 SD support level。

## 15. Future gates

### 15.1 Trainable text encoder

需要决定：

- 单/多 optimizer；
- learning-rate groups；
- EMA；
- checkpoint role；
- condition embedding artifact invalidation；
- pretrained bundle export。

不能通过 `train_text_encoder: true` 偷渡。

### 15.2 LoRA/PEFT

需要独立定义：

- target module selection；
- base-weight identity；
- adapter checkpoint；
- merge/unmerge；
- optimizer parameter ownership；
- sampling bundle；
- compatibility validation。

它是后续正式能力，不用来替代 full-module lifecycle 的首轮验证。

### 15.3 SDXL

SDXL 至少引入：

- two text encoders/tokenizers；
- pooled embeddings；
- time IDs/micro-conditioning；
- different codec/UNet configs；
- 1024/bucketed data；
- larger asset/compute profile。

因此它是新的 concrete family plan，不给 SD 1.x Builder 堆 nullable fields。

### 15.4 SD3

SD3 的 transformer、multiple text encoders 和 flow-style formulation 不满足
SD 1.x conditional UNet family；需要独立 TrainingBuilder、Dynamics/Sampler
兼容评审。

### 15.5 Other conditions

ControlNet、image-to-image、inpainting、IP-Adapter 等分别需要窄 condition
capability 和 asset role。没有第二个真实组合前不抽象 universal condition graph。

## 16. 风险与缓解

| 风险 | 后果 | 缓解 |
| --- | --- | --- |
| 用“Diffusers 可加载”代替 parity | 输出漂移且难定位 | 分层 parity manifest |
| 把 black-box pipeline 包成 Sampler | 双重 scheduler/lifecycle | backend ownership 分离 |
| 混合 component revision | codec/text/UNet 不匹配 | bundle role + digest |
| sampling 重复填写 components | checkpoint 被 overlay 漂移 | checkpoint-owned bundle |
| caption 现场生成 | run 不可重放 | versioned caption artifact |
| class batch schema 加 nullable prompt | core 被 modality 污染 | concrete recipe contract |
| 256 smoke 冒充 SD 1.x parity | 质量和分辨率声明失真 | 512 formal profile |
| fine-tuning 冒充 from scratch | 结果误导 | initialization profile identity |
| scraped community mirror 直接发布 | 许可/NSFW/重复风险 | research inventory + audit gate |
| 先实现 SDXL/LoRA | 首个 lifecycle 永远不闭合 | SD 1.x full-module vertical slice |
| Spark 容量被当作吞吐 | production 计划失真 | 1k-step cross-device benchmark |

## 17. 明确不进入首个 Stable Diffusion milestone

- VAE training；
- joint VAE + UNet training；
- trainable text encoder；
- LoRA/PEFT；
- ControlNet/IP-Adapter/inpainting；
- SDXL；
- SD3；
- arbitrary Diffusers Pipeline component graph；
- floating Hub revision；
- arbitrary `.ckpt` conversion；
- universal tokenizer/condition/model registry；
- undocumented scraped dataset；
- 1024 from-scratch production；
- web-scale foundation-model quality claim。

## 18. 调研与实现参考

- [Latent Diffusion 支持计划](latent-diffusion-support-plan.md)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
- [Diffusers Stable Diffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- [Diffusers training overview](https://huggingface.co/docs/diffusers/training/overview)
- [Diffusers text-to-image training](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
- [Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
- [The Met Open Access](https://www.metmuseum.org/hubs/open-access)
- [The Met Collection API](https://metmuseum.github.io/)
- [COCO](https://cocodataset.org/)
- [LHQ](https://arxiv.org/abs/2104.06954)
- [Public Domain 12M](https://huggingface.co/datasets/Spawning/PD12M)
