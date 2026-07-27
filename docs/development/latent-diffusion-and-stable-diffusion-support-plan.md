# Latent Diffusion、DiT 与 Stable Diffusion 支持计划

- 文档性质：开发计划；不属于当前公开 API 或正式用户文档
- 状态：提案，尚未进入实现
- 制定日期：2026-07-26
- 适用范围：冻结图像 codec 的 latent generative training/sampling/evaluation，
  以及分层的 Diffusers/Stable Diffusion 互操作
- 首个原生纵向切片候选：
  Flowers102、256×256、类条件 frozen-codec latent generation、CFG、
  conditional UNet/DiT 对照
- 前置项目：
  [AFHQ-v2 类条件生成 showcase](../tutorials/afhq-v2.md)中的
  class-aware data、条件 denoiser、CFG、AMP 与 gradient accumulation
- 关联计划：
  [默认工作流与推理 Pipeline 支持计划](default-workflow-pipeline-support-plan.md)、
  [Metrics 支持开发计划](metrics-support-plan.md)、
  [训练后 Evaluation 与 Benchmark 支持计划](post-training-evaluation-support-plan.md)

## 1. 核心判断

当前 Stochaflow 的总体抽象仍在正确道路上。Latent Diffusion 不要求推翻
`Process`、`Sampler`、`TrainingBuilder`、`TrainingStrategy` 或
`SamplingBuilder` 的边界；它主要暴露了三个尚未闭合的工程契约：

1. **表示空间契约**：像素怎样被 frozen codec 转成 diffusion-normalized latent，
   以及 latent transform 怎样严格逆变换后再由 codec 解码；
2. **冻结资产契约**：VAE、未来的 text encoder 等预训练资产怎样加载、冻结、
   固定 revision、参与 device/mode/checkpoint/resume 和 provenance；
3. **推理资产契约**：训练 checkpoint 已经保存 auxiliary state，但当前 sampling
   runtime 只恢复 primary model、EMA 与 Process，不能恢复 codec。

本计划作出以下决策：

1. **不新增 `LatentProcess`。**
   Gaussian Process 作用于 tensor state；该 state 是 pixel 还是 normalized latent
   不改变其数学契约。codec 属于任务组合，不属于概率路径。
2. **不把 DiT 定义成新 diffusion family。**
   DiT 是 denoiser backbone；conditional UNet 与 DiT 应能在同一 latent Process、
   Objective、CFG 和 Sampler 组合中替换。
3. **新增 image-specific 的窄 latent codec capability。**
   codec adapter 独占图像范围、posterior policy、有损 encode/decode、精度策略，
   以及 scaling/shift/mean/std 的严格逆变换；其他层只接收
   diffusion-normalized latent。
4. **primary model 继续是 denoiser。**
   frozen codec 是具名 managed auxiliary；它不进入 denoiser EMA，也不进入 optimizer。
5. **SamplingBuilder 负责完整 latent inference。**
   它拥有 latent shape、condition、CFG、Dynamics、Sampler、decode 和 writer-ready
   output；数值 Sampler 不理解 VAE、class label 或 prompt。
6. **Stable Diffusion 支持采用分层互操作，不采用一个模糊开关。**
   原生 class-conditional LDM、SD 1.x component interop、完整 Diffusers pipeline
   backend、SDXL、SD3 和 LoRA 是不同 scope，不能用 `stable_diffusion: true` 混合。
7. **Flowers102 是第二阶段 capability benchmark，不替代 AFHQ 首个视觉展示。**
   AFHQ 先验证高质量 pixel conditional generation；Flowers 再压力测试 frozen
   codec、细粒度条件、latent state、DiT 可替换性和正式 Evaluation。

一句话目标是：

> 证明同一个 model-free Process 和数值 Sampler，可以通过任务 Builder 注入不同的
> 表示 codec、denoiser backbone 与 conditioning composition，在 pixel DDPM、
> Flowers latent generation 和后续 Stable Diffusion interop 之间复用。

## 2. 先校正 Flowers102 调研结论

### 2.1 数据属性属实，但官方 split 对生成任务很特殊

[Oxford 官方页面](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
确认 Flowers102 有 102 类、每类 40–258 张，存在明显尺度、姿态、光照变化，
同时有类内大变化和相似类别。

[TFDS 数据目录](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)
给出的官方 split 是：

| split | 数量 | 每类约束 |
| --- | ---: | --- |
| train | 1,020 | 10 张/类 |
| validation | 1,020 | 10 张/类 |
| test | 6,149 | 至少 20 张/类 |
| total | 8,189 | 40–258 张/类 |

这意味着“直接加载官方 train split 训练生成模型”实际只有 1,020 张图。
另一方面，把全部 8,189 张用于训练虽然更容易得到漂亮样图，却不再有 held-out test。
因此 Flowers recipe 必须显式声明自己是 showcase 还是 held-out transfer protocol，
不能含糊地写成“在 Flowers102 上训练”。

Oxford 数据页面没有给出清晰、统一的 Flowers102 数据许可声明，且图片主要来自多个
网站。实现前必须单独完成数据和展示产物的许可审计；不能把其他 Oxford/VGG 数据集
条款或代码仓库许可证自动套到 Flowers102 图片上。

### 2.2 用户列出的 FID 表不能当完整 Flowers102 基准

`166.83 / 134.62 / 91.07 / 89.64 / 103.64` 这组数字可以追溯到
[EDG-CDM 论文](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70018)
的一个内部对照表，但该实验只从 Flowers102 随机选择了 13 类、共 2,067 张图。

此外，该结果没有充分固定：

- 被选中的 13 个具体 class IDs、随机抽样 seed 与选择规则；
- 训练/验证/测试 split；
- 生成分辨率；
- real/generated sample count；
- resize、crop、antialias 与 feature extractor；
- 实现版本、checkpoint 与随机种子。

DDIM 本身还是采样过程而不是独立 backbone，把“DDPM”和“DDIM”并列为 model
也说明这张表更适合论文内部相对比较，而不是公共 benchmark 定义。

因此本计划只把它当作“该论文重实现/比较的 baselines 在其有限数据、13 类内部协议下
数值较弱”的观察证据，不把任何一个数字写进 Stochaflow quality gate。

### 2.3 Anthos 证明可行性，不证明泛化或量化质量

[Anthos model card](https://huggingface.co/Glint-Research/Anthos-1)
确实给出了一个很贴近本计划的组合：

- 983,808 参数的 DiT-Nano/2；
- 256×256 image、32×32×4 latent；
- frozen `stabilityai/sd-vae-ft-ema`；
- 102 classes + null token；
- class dropout 0.1；
- CFG 4；
- Heun 50 steps。

但需要作四个重要校正：

1. 它训练的是 **rectified flow matching velocity field**，不是普通 DDPM，
   也不是把 Gaussian diffusion 的 `v_prediction` 换个名字；
2. 它使用 Heun flow sampler，不是 DDIM；
3. 它把 train、validation、test 全部 8,189 张都用于训练，并通过水平翻转得到
   16,378 个 latent，因此没有 held-out test；
4. model card 明确说明没有 FID/IS，只做视觉检查，并将项目标为 research prototype。

固定 revision 的
[仓库文件](https://huggingface.co/Glint-Research/Anthos-1/tree/56ff6df2849f4c4ad99a5cc5e804da6b4177dfda)
也没有包含完整训练脚本。因此 Anthos 能支持的结论只有：

> frozen SD VAE + tiny class-conditioned DiT/flow 在全量 Flowers102 上形成了可运行的
> class-conditioned sampling artifact，并展示了作者报告的可辨识样本。

它不能证明：

- 约 1M 参数已经达到良好 FID；
- tiny DiT 优于同预算 UNet；
- 该结果对 held-out 花朵有泛化能力；
- 18 分钟训练时间能迁移到普通消费级 GPU；
- Stochaflow 应把 flow matching 偷偷实现为 Gaussian training 的一种模式。

### 2.4 “DiT 更适合 Flowers”应作为假设而不是框架结论

[DiT 官方实现](https://github.com/facebookresearch/DiT)证明 Transformer 可以在
class-conditional latent ImageNet 上作为扩展性良好的 denoiser。DiT-S/2 的
`depth=12`、`hidden=384`、`heads=6` 也确实是官方配置。

但原始结果没有证明在 8,189 张、102 个细粒度类的 Flowers 数据上，DiT 天然优于 UNet。
“全局花瓣布局需要 Transformer”“纹理丰富更适合 attention”是合理研究假设，
不是可以写进 core dispatch 的事实。

首个 Flowers 纵向切片应：

1. 先用已有 class-conditional ADM-style UNet 验证 latent plumbing；
2. 再加入 DiT；
3. 在相同 codec、数据、condition、prediction type、训练预算、EMA、CFG、
   sampler 和 sample plan 下做等预算 ablation。

## 3. 成熟方案给出的稳定边界

### 3.1 Latent Diffusion 是两阶段表示组合

[Latent Diffusion 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
把任务拆成：

```text
image x
  -> pretrained perceptual autoencoder
  -> latent z
  -> diffusion/score model in latent space
  -> decoded image
```

其关键价值不是“任何 VAE 都让维度下降一个数量级”，而是选择感知压缩率与细节保真度
之间的工作点。codec 的 representational support、decoder artifacts 与重建质量会对
最终生成形成强约束和实际质量上限：denoiser 不能消除 decoder 的系统性伪影，也不能
可靠表达 codec 支持范围之外的花瓣、花蕊或边缘结构。decoder 仍可能合成统计上合理的
高频细节，所以这里不是声称每个重建像素都是生成结果的严格逐像素上限。

因此 latent training 前必须先通过 codec reconstruction gate，不能看到 latent shape
是 32×32×4 就直接开始长训练。

### 3.2 latent normalization 是 codec contract，不是全局常数

[Diffusers `AutoencoderKL` 文档](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
说明经典 SD-style latent 会在进入 diffusion model 前乘 `scaling_factor`，decode 前
再执行严格逆变换。常见的 `0.18215` 是特定模型配置，不是 Latent Diffusion 公理。

不同模型还可能具有：

- `shift_factor`；
- `latents_mean` / `latents_std`；
- 不同 latent channel count；
- 不同空间下采样倍数；
- VAE `force_upcast` 或其他精度政策；
- posterior sample 与 posterior mode；
- VQ 或确定性 encoder。

因此不能让 TrainingStrategy、SamplingBuilder 和 config 各自读取
`vae.config.scaling_factor`。稳定语义应是：

> codec adapter 输出和接收 diffusion-normalized latent，并独占所有正反变换。

### 3.3 DiT 是 backbone substitution

[DiT 论文和官方实现](https://github.com/facebookresearch/DiT)在 autoencoder latent
patch 上用 Transformer 替换常见 UNet，并使用 class embedding、class dropout、
adaLN-Zero 与 CFG。

对 Stochaflow 的启示是：

- DiT 注册为新 model；
- AdaLN 属于该 model 的内部实现；
- class dropout 属于具体 TrainingStrategy；
- CFG 属于具体 SamplingBuilder；
- patch size、token geometry 和 class count 由 Builder 在完整组合处验证；
- Process、Sampler 和 `GenerativeDynamics` root 不增加 DiT 特殊方法。

官方 DiT 默认 `learn_sigma=True`，即对 C-channel latent 输出 2C channels，
同时包含 denoising prediction 与 learned variance。本计划首版只复用
DiT-S/2/Nano 的 depth、width、heads、patch 与 adaLN architecture，显式采用
`learned_variance=false`，只输出 C 个 prediction channels。若后续支持 2C 输出，
必须增加 prediction/variance partition、Objective 和 Sampler contract；不能把 2C
tensor 直接交给当前只接受 C-channel prediction 的 Gaussian Dynamics。

本计划把 class dropout 交给 Strategy，所以 native DiT model 内部的随机 label dropout
必须关闭，避免双重 dropout。上游实现的复现特例不自动成为 Stochaflow 默认。

### 3.4 Diffusers Pipeline 是推理组合，不是训练抽象

[Diffusers Pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview)
把 pipeline 定义为包含 models、scheduler 和 processors 的完整推理对象，并明确
training 应操作独立组件。

经典
[StableDiffusionPipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
组合：

- `AutoencoderKL`；
- frozen `CLIPTextModel`；
- `CLIPTokenizer`；
- `UNet2DConditionModel`；
- diffusion scheduler；
- reference SD1 bundle 通常还有 safety checker 与 feature extractor；具体 bundle
  允许显式缺失，因此 presence/enabled state 必须进入 manifest。

这支持两条都合理但不能混合的互操作路径：

1. **完整 pipeline backend**：Diffusers 独占全部推理 lifecycle，Stochaflow 只负责
   operation、输入、结果、artifact 和 Evaluation；
2. **component-native integration**：独立加载 VAE、denoiser、text encoder、
   tokenizer 与 scheduler metadata，由 Stochaflow 的 Builder/Process/Sampler
   独占 lifecycle。

不能把 `DiffusionPipeline` 伪装成 Stochaflow `Sampler`，否则 models、scheduler、
device、offload 和 inference loop 会出现两个 owner。

## 4. 准确术语与 scope ladder

### 4.1 名称不可混用

| 名称 | 本计划中的精确定义 | 不等于 |
| --- | --- | --- |
| Latent Diffusion | 在预训练表示 codec 的 normalized latent 上运行 diffusion | Stable Diffusion、DiT、文本生成 |
| DiT | 处理 latent/image patches 的 diffusion denoiser backbone | Process、Sampler、文本条件 |
| SD 1.x | f=8 VAE、单 CLIP、conditional UNet 等组成的具体 text-to-image LDM family | 所有 latent diffusion |
| SD 2.x | f=8 VAE、OpenCLIP ViT-H、conditional UNet；具体 checkpoint 固定 epsilon/v 语义 | SD1 component parity |
| SDXL | 双 text encoder、更大 UNet、额外尺寸/裁剪条件；base 可独立运行，也可把 latent 交给单独加载的 refiner img2img pipeline | “更大的 SD1 配置” |
| SD3 | rectified-flow training formulation、MMDiT 与不同 latent/condition semantics；reference inference pipeline 使用 flow-matching scheduler | Gaussian DDPM mode |
| Diffusers pipeline | 第三方完整推理 composition | Stochaflow Sampler 或 TrainingPlan |
| LoRA | 依附于特定 base topology/weights 的低秩 adapter artifact；本计划采用冻结 base、只训练 adapter 的政策 | 完整 model checkpoint |

[SDXL model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
明确说明它使用两个固定 text encoders。
[Diffusers SD3 API](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
的 SD3 Medium reference bundle 则组合 MMDiT、
`FlowMatchEulerDiscreteScheduler`、VAE、两个 CLIP 和一个 T5；特定推理配置可以显式
省略 T5 并接受质量损失，所以实际组件拓扑仍须按 bundle/revision 验证。
[Stable Diffusion v2 model card](https://huggingface.co/stabilityai/stable-diffusion-2)
也说明 SD2 不能因为复用某个 pipeline class 就自动继承 SD1 parity。
这些差异足以说明“支持 Stable Diffusion”必须版本化、family-specific。

### 4.2 支持分层

| 层级 | 能力 | Stochaflow 的 owner | 本计划判断 |
| --- | --- | --- | --- |
| L0 | pixel class-conditional Gaussian + CFG | 原生 Builder/Process/Sampler | AFHQ 前置 |
| L1 | frozen-codec class-conditional latent Gaussian | 原生组件 | 首版核心 |
| L2 | latent conditional UNet/DiT substitution | 原生 model/capability | 首版核心 |
| L3 | 完整 Diffusers pipeline inference backend | optional integration | 可并行早做 |
| L4 | SD 1.x component import、原生 text-conditioned inference | optional integration + 原生 lifecycle | 后续 |
| L5 | SD 2.x component compatibility case | 独立 parity suite/adapter | 不由 SD1 自动获得 |
| L6 | SD 1.x/SDXL LoRA fine-tuning | family-specific TrainingBuilder | 后续独立项目 |
| L7 | SDXL native composition | 独立 family Builder | 不进入首版 |
| L8 | SD3/flow native composition | 新 flow family + family Builder | 不进入 Gaussian 首版 |

“支持 Stable Diffusion”对外必须写成具体层级，例如：

- `diffusers-pipeline inference backend supports SD 1.5`;
- `SD 1.x components can be imported into native Stochaflow sampling`;
- `SDXL LoRA training is supported`.

不能只写“Stable Diffusion compatible”。

## 5. 当前仓库审计

### 5.1 已经适合 Latent Diffusion 的边界

| 当前能力 | 为什么正确 |
| --- | --- |
| `Process` model-free、task-free | 同一 Gaussian path 可作用于 pixel 或 latent tensor |
| `TrainingPlan` 有 primary、Process、Objective、具名 auxiliaries | denoiser 可保持 primary，codec 可保持 frozen auxiliary |
| `ManagedTrainingModule(mode="eval")` | core 可维持 codec eval mode |
| trainable parameters 按 `requires_grad` 收集 | Builder freeze 后 codec 不进入 optimizer |
| EMA 只跟踪 primary model | 不会给 frozen VAE 建无意义 EMA |
| Strategy 解释结构化 `Any` batch | 可以消费 `(images, {"class_label": labels})` |
| `epsilon/x0/v/score` prediction types | 覆盖首批 native latent Gaussian 参数化 |
| `SamplingBuilder` 拥有完整任务采样 | 可组合 condition、CFG、latent prior、decode |
| Gaussian Dynamics 接受 task adapter | Sampler 不需要理解具体 model signature |
| checkpoint 保存 auxiliary state | 训练和 strict resume 已有正确性基础 |

特别要保留：

- 现有 unconditional Gaussian Strategy 对非空 condition 明确失败；
- conditional latent training 使用独立 Builder/Strategy；
- `standard_denoising` 不增加 `latent=true`、`conditional=true`、
  `text=true` 等模式堆叠；
- 现有 `unet` 不增加大量 optional condition 参数。

### 5.2 当前最大阻塞点：sampling 丢失 auxiliary assets

checkpoint v10 已把具名 auxiliary modules 写入
`training_assets_state_dict`，并保存 fixed `inference_recipe`，但
`SamplingCheckpointView` 只保留：

- primary raw state；
- primary EMA state；
- Process state；
- config、recipe 与 metadata。

`InferenceModelProvider` 也只能构造 primary model。

因此当前系统可以严格 resume 一个带 frozen codec 的训练，却无法从该 checkpoint
独立执行 latent sampling。这个断点必须在 Flowers 实现前修复。

### 5.3 当前 Flowers config 的准确定位

当前 `examples/built-in/image-generation/experiments/ddpm_flowers102.yaml` 是：

- 64×64 pixel space；
- ordinary unconditional UNet；
- `image` DataBuilder 丢弃 Flowers class label；
- epsilon prediction；
- unconditional DDPM sampling；
- 训练使用官方 partition。

它适合作为 tutorial/smoke 或“为什么简单 recipe 不足”的 baseline，不应继续被描述为
Flowers 高质量 showcase，也不应作为 latent/conditional 计划的隐式基础。

## 6. 表示空间与 codec capability

### 6.1 三个状态必须区分

```text
observation image x
  -- codec preprocessing + encode -->
codec-native latent z_native
  -- scaling / shift / normalization -->
diffusion state z_model
```

latent normalization 的反向必须严格对应：

```text
z_model
  -- inverse normalization -->
z_native
  -- decode + output postprocessing -->
image x_hat
```

Process、Objective、denoiser 和 Sampler 只看到 `z_model`。
DataBuilder 默认产生 observation image；codec adapter 负责从 observation space
跨到 model state space。VAE encode/decode 本身是有损映射，不被称为严格逆；
必须精确可逆的是 scaling、shift、mean/std 等 latent transform。

### 6.2 不建立万能 `RepresentationModel`

首版只定义 image latent generation 所需的窄 capability。候选形态：

```python
@dataclass(frozen=True, slots=True)
class ImageLatentSpec:
    latent_channels: int
    input_channels: int
    input_value_range: tuple[float, float]
    geometry_identity: str
    transform_identity: str


@runtime_checkable
class ImageLatentGeometry(Protocol):
    def latent_shape_for(
        self,
        image_shape: tuple[int, int, int],
    ) -> tuple[int, int, int]: ...


@runtime_checkable
class DiffusionImageEncoder(Protocol):
    @property
    def latent_spec(self) -> ImageLatentSpec: ...

    def encode_for_diffusion(
        self,
        images: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor: ...


@runtime_checkable
class DiffusionImageDecoder(Protocol):
    @property
    def latent_spec(self) -> ImageLatentSpec: ...

    def decode_from_diffusion(self, latents: Tensor) -> Tensor: ...
```

首版 built-in adapter 通常同时实现 encoder 和 decoder，但拆成两个 capability 的原因是：

- 某些训练路径只消费 prepared latents，不需要 encoder；
- 某些 inference bundle 只需 decoder；
- reconstruction evaluation 才需要两者；
- 不强迫所有未来表示模型暴露 VAE posterior。

如果实际实现发现这两个 capability 永远成对消费，可以由一个
`ImageLatentCodec` facade 同时组合它们，但不把 `encode/decode` 加到通用 model root。

### 6.3 adapter 必须独占的语义

`DiffusersAutoencoderKLAdapter` 或其他 codec adapter 必须独占：

- codec-specific image channel/range validation 和到 encoder-native range 的转换；
- spatial divisibility 和 latent geometry；
- posterior sample 或 posterior mode；
- generator 的使用和可重放语义；
- scaling factor；
- shift factor；
- latent mean/std；
- latent normalization 的严格逆变换；
- decoder output extraction；
- VAE dtype、`force_upcast`、slicing/tiling 等明确策略；
- upstream component identity 和 provenance。

Strategy、SamplingBuilder 和 EvaluationBuilder 不能再次执行这些变换。

DataBuilder 仍独占 dataset-specific crop、resize、interpolation、antialias 和 augmentation，
并声明其输出 tensor range。具体 TrainingBuilder 在完整组合处验证 DataBuilder recipe
的 output descriptor 与 codec expected input；codec 只执行一次必要的 native-range
转换，不能再次 crop/resize/normalize。配置中的 `normalize: true` 必须展开成明确
output range，不能同时让 VAE image processor 再做一遍相同归一化。

### 6.4 posterior policy 是 recipe identity

VAE training 常用 posterior sample；reconstruction reporting 常用 deterministic mode。
二者结果不同，必须进入 resolved config 和 manifest。

推荐让 adapter 在构造时固定 encoding policy，而不是每次调用传一个随意字符串：

```yaml
codec:
  name: diffusers_autoencoder_kl
  params:
    source:
      repo_id: stabilityai/sd-vae-ft-ema
      revision: <immutable-commit>
      subfolder: null
    encoding_policy: posterior_sample
```

这样训练的 codec capability 本身就具有稳定语义。另一个 deterministic adapter
实例用于 reconstruction evaluation。

### 6.5 首个 SD-style normalized latent recipe 必须关闭 pixel clipping

pixel Gaussian sampling 常把预测的 `x0` clip 到 `[-1, 1]`。
SD-style diffusion-normalized latent 不满足这个 pixel 范围，本计划首个 Flowers
recipe 必须验证：

```yaml
clip_denoised: false
```

该值应成为该 family 的 recipe/checkpoint compatibility 不变量，不能由 sampling
overlay 静默改成 pixel 默认值。未来 bounded/discrete latent family 是否 clipping
由其自己的 contract 决定。

## 7. frozen asset 生命周期与 provenance

### 7.1 TrainingBuilder 的责任

具体 latent TrainingBuilder 负责：

1. 从显式 declaration 构造 codec；
2. 固定外部 repo revision/variant/subfolder；
3. 校验 codec capability 和 latent geometry；
4. 调用 `requires_grad_(False)`；
5. 以 `ManagedTrainingModule(mode="eval")` 声明为具名 auxiliary；
6. 断言 codec 全部参数均为 `requires_grad=False`；
7. 把可调用的 encoder/clean-state capability 注入 Strategy；
8. 在 TrainingPlan 中返回 codec 的 typed inference-asset projection，包括构造
   declaration、capability role 与 persistence policy；
9. 由 core 把 projection descriptor 写入 checkpoint/run provenance。

`mode="eval"` 只管理 train/eval mode，不代表 freeze。若 Builder 忘记
`requires_grad_(False)`，当前 core 会把 auxiliary 的 trainable parameters 纳入统一
optimizer。这一行为应保留，并用 Builder contract test 防止遗漏。optimizer 在
TrainingPlan 之后构造，因此 Builder 不可能检查一个尚不存在的 optimizer；core 先按
Plan 的 `requires_grad` 精确选择参数，Trainer 再验证 optimizer parameter set 与
Plan-selected parameters 完全一致。

### 7.2 TrainingPlan 必须产生 inference asset identity

训练配置把 codec declaration 放在具体 `training.params` 中，而 sampling 只应请求一个
稳定 slot；二者之间不能靠重复手写 config 或 auxiliary 名称约定连接。

建议给 `TrainingPlan` 增加独立 projection mapping：

```python
@dataclass(frozen=True, slots=True)
class ManagedInferenceAsset:
    training_asset_name: str
    declaration: ComponentConfig
    capability_role: str
    persistence: EmbeddedState | ImmutableReference


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    # existing fields...
    inference_assets: Mapping[str, ManagedInferenceAsset]
```

mapping key 是 sampling/bundle 请求的稳定 slot，例如 `codec`。projection 引用已经存在的
managed training auxiliary，不再次持有 module，避免同一 state root 被声明两次。
core 校验：

- slot 非空、唯一且不使用保留名；
- `training_asset_name` 精确存在于 `auxiliary_modules`；
- declaration 是该 module 的可重构造身份；
- persistence 与 checkpoint state 一致；
- capability role 是 provenance/compatibility identity，不替代运行时 Protocol 验证。

core 将 projection 数据化写入 `inference_asset_descriptors`；TrainingBuilder 不直接写
checkpoint metadata。SamplingBuilder 只请求 slot，provider 从 checkpoint descriptor
取得 declaration 并构造 module，避免训练/采样配置漂移。没有被 projection 的
distillation teacher 等 training-only asset 不可被 inference 自动发现。

当前 checkpoint v10 已经固定 strict envelope，且只有 `inference_recipe`，没有
`inference_asset_descriptors`。本计划若引入该字段，必须发布新的 checkpoint schema
version；无推理 auxiliary 的 run 写空 mapping。不得原地改变 v10，也不为 v10
checkpoint 猜测或迁移 inference auxiliary identity。

### 7.3 首版采用 embedded state，随后增加 immutable reference

当前 checkpoint 会在每个 checkpoint 中复制完整 frozen VAE state。
这浪费空间，但语义正确、离线可恢复。

建议分两阶段：

#### Phase A：embedded state

- 每个 checkpoint 保存完整 codec state；
- sampling runtime 只投影被具体 Builder 请求的 codec；
- strict resume 完整校验 state topology；
- 优先获得正确闭环。

#### Phase B：tagged persistence

给 managed asset 增加明确 tagged policy：

```text
embedded_state
immutable_reference
```

`immutable_reference` 至少记录：

- provider kind；
- repo/model ID；
- immutable revision；
- component subfolder；
- variant；
- config digest；
- weights digest；
- library/version requirements；
- license identifier 与需要用户接受的条款；
- codec transform descriptor。

resume/sampling 时必须解析同一 revision 并核验 digest；缺失或不匹配立即失败，
不能下载浮动 `main`，也不能静默选择“兼容 VAE”。

### 7.4 training checkpoint 与 inference bundle 不混用

training checkpoint 保存：

- optimizer/scheduler；
- EMA lifecycle；
- RNG；
- epoch/global step；
- primary 与 managed assets；
- strict resume metadata。

inference bundle 保存：

- 任务 capability；
- denoiser weights variant；
- codec/text assets；
- preprocessing/postprocessing；
- sampler compatibility；
- condition vocabulary；
- provenance、license 与 runtime requirements。

可移植 export 可以把 immutable referenced assets vendor 进 bundle；普通训练 checkpoint
不必永远重复复制几百 MB 的 frozen VAE。

## 8. inference asset provider

### 8.1 扩展 sampling checkpoint projection

sampling runtime 必须保留并验证：

```text
training_assets_state_dict
inference_asset_descriptors
```

但不能把原始 checkpoint mapping 直接暴露给 SamplingBuilder。

### 8.2 新增窄 provider

候选 API：

```python
class InferenceAssetProvider:
    def resolve_primary(
        self,
        weights: WeightSelection,
    ) -> ResolvedInferenceModule: ...

    def resolve_asset(
        self,
        slot: str,
    ) -> ResolvedInferenceModule: ...
```

provider 负责：

- 通过现有模型构造路径创建 module；
- 从 checkpoint 的 typed descriptor 取得构造 declaration，不接受 sampling config
  重复提供另一份 declaration；
- 严格加载具名 embedded state 或解析 immutable reference；
- move to device；
- `eval()`；
- capability/type validation 前的基础 `nn.Module` 保证；
- 返回 resolved provenance。

具体 latent SamplingBuilder 调用 `resolve_asset("codec")`，再验证该 module 是否实现
`DiffusionImageDecoder`。未来 text Builder 可以请求 `text_encoder`，但不会自动加载
checkpoint 中所有 training auxiliaries；例如 distillation teacher 不应出现在
student-only inference。

### 8.3 不新增全局 arbitrary-asset registry

codec、text encoder 和 denoiser 都可通过已有 registered model/extension 构造路径
生成。首版不建立：

- 任意 Python object registry；
- 通用多模型 YAML graph；
- 自动按 state dict key 猜 module class；
- 自动按 auxiliary 名称猜 capability。

具名 slot、declaration 和 capability 都由具体 Builder 显式拥有。

## 9. 数据与 prepared latent

### 9.1 class label 仍是具体 DataBuilder 私有语义

Flowers DataBuilder 应输出：

```python
(images, {"class_label": labels})
```

并固定：

- stable class ID ↔ class name mapping；
- split/sample IDs；
- source inventory；
- RGB/crop/resize/interpolation/antialias；
- augmentation；
- dataset license audit 状态；
- manifest digest。

这不意味着 core 增加通用 `Condition`、`Target` 或 `SampleMetadata` schema。

### 9.2 on-the-fly encode 是正确性 baseline

首个实现应支持：

```text
image augmentation
-> frozen codec encode
-> normalized latent
-> diffusion training
```

优点是 augmentation 在 encode 前发生，语义清晰；缺点是训练反复运行 VAE encoder。
该路径是 correctness baseline，即使后续 production recipe 默认 prepared latents，
也应保留 tiny contract test。

### 9.3 prepared latent 是版本化数据 artifact，不是隐式 cache

为提升 Flowers/DiT 训练吞吐，可以增加离线 prepared-latent preparation utility：

```text
verified image artifact
-> fixed preprocessing/augmentation enumeration
-> fixed codec revision + policy
-> normalized latent shards
-> latent dataset manifest
```

manifest 至少绑定：

- source sample ID；
- source image digest；
- preprocessing identity；
- augmentation identity；
- codec repo/revision/config/weights digest；
- posterior policy；
- latent transform identity；
- latent shape/dtype；
- shard inventory/digest；
- producer library/version/device precision。

训练时 DataBuilder 只读取这个已验证 artifact，不能在 loader 内发现 cache miss 后临时
编码并写盘。

### 9.4 不把 prepared latent 设为唯一入口

必须同时允许 extension 选择：

- on-the-fly image encode；
- prepared normalized latents；
- 自定义 task batch adapter。

具体 Strategy 通过 Builder 注入的私有 clean-state provider 解释 batch；core 不判断
batch 里是 image 还是 latent。

## 10. latent training composition

### 10.1 独立 Builder/Strategy

新增类似：

```text
class_conditional_latent_gaussian
```

而不是修改现有 unconditional `gaussian_denoising`。

Builder 组装：

- primary conditional denoiser；
- frozen codec encoder；
- Gaussian Process；
- Objective；
- class conditioning policy；
- class dropout policy；
- clean-state provider；
- managed auxiliary declaration。

Strategy 只执行：

```text
interpret batch
-> encode or read normalized clean latent z0
-> sample timestep/noise
-> Process.sample_marginal(z0)
-> apply condition dropout
-> invoke narrow denoiser capability
-> build epsilon/x0/v/score target
-> Objective
-> scalar loss + MetricUpdate channels
```

Strategy 不构造、下载、freeze、move、serialize codec，也不执行 backward/optimizer step。

### 10.2 prediction type 是跨阶段兼容不变量

Latent Diffusion 不天然等于 `v_prediction`：

- 官方 SD 1.x reference checkpoints 使用 epsilon prediction；community finetune
  仍必须从固定 bundle/config 验证；
- 其他 family/recipe 可能使用 v prediction；
- flow matching 又是不同 family。

`prediction_type` 必须贯穿：

```text
training target
-> denoiser output semantics
-> Dynamics conversion
-> Sampler compatibility
-> checkpoint/bundle manifest
-> Evaluation protocol
```

strict partial sample request 不能覆盖 checkpoint recipe 固化的训练语义。

### 10.3 Flowers 首版先做 Gaussian，不偷渡 flow

Anthos 的 flow matching 很有研究价值，但当前仓库已有成熟 Gaussian Process 与
DDPM/DDIM family。首个 latent vertical slice 应先证明：

- pixel/latent state independence；
- codec lifecycle；
- condition/CFG；
- conditional UNet/DiT substitution；
- evaluation/provenance。

Rectified flow 应在单独计划中定义自己的 Process/Dynamics/Sampler capability，
不能为了复现 Anthos 给 Gaussian Strategy 增加 `objective: flow` 模式。

### 10.4 VAE 联合训练不进入首版

以下都定义新的训练问题，部分还需要新 loop family：

- 自训练 KL autoencoder；
- VQ/VQGAN；
- adversarial reconstruction；
- VAE + diffusion joint training；
- 多 optimizer；
- discriminator alternating updates。

它们不能通过给首版 latent Builder 增加 optional flags 伪装成同一 lifecycle。

## 11. condition 与 CFG

### 11.1 三种责任必须拆开

| 责任 | owner |
| --- | --- |
| batch 中怎样取得 class/prompt | concrete Strategy |
| condition 怎样进入 denoiser | model-specific narrow capability/adapter |
| 训练时怎样丢 condition | concrete Strategy policy |
| sampling 时怎样组合 cond/uncond prediction | concrete SamplingBuilder |

不增加全局 `conditioning:` schema 或 `conditional: true`。

### 11.2 Flowers class condition

复用 AFHQ 计划中的窄 `ClassConditionalDenoiser` capability：

```python
class ClassConditionalDenoiser(Protocol):
    @property
    def num_classes(self) -> int: ...

    @property
    def null_class_id(self) -> int: ...

    def predict_class_conditioned(
        self,
        state: Tensor,
        model_time: Tensor,
        class_labels: Tensor,
    ) -> Tensor: ...
```

ADM UNet 与 DiT 都实现同一行为契约，但内部可分别使用 scale-shift norm 或
adaLN-Zero。Builder 校验：

- class count = 102；
- null class ID 不与真实 class 冲突；
- latent channels/shape；
- output shape；
- model time domain；
- prediction semantics。

### 11.3 Stable Diffusion text condition 是另一个 family

SD 1.x text conditioning 使用 tokenizer、CLIP sequence embeddings 与 UNet
cross-attention；SDXL 还有第二 text encoder、pooled embeddings 和尺寸/裁剪信息；
SD3 Medium reference bundle 使用 MMDiT、两个 CLIP 与一个 T5；具体 bundle/revision
可能在显式质量权衡下省略 T5，因此组件拓扑仍须逐 family 验证。

因此后续分别定义：

- `SD1TextConditionedDenoiser`；
- `SDXLConditionedDenoiser`；
- `SD3FlowDenoiser`；

或等价的更窄 adapter capability。不要把它们折叠成一个带大量 optional tensor 的
`ConditionalDenoiser.forward(...)`。

## 12. latent sampling composition

### 12.1 具体 SamplingBuilder 的完整流程

`class_conditional_latent_denoising` 负责：

```text
resolve denoiser weights
-> resolve frozen codec
-> validate requested output size against codec geometry
-> allocate class labels and null labels
-> sample latent terminal prior
-> build conditional/unconditional model adapter
-> apply CFG
-> construct Gaussian Dynamics
-> run DDPM/DDIM Sampler
-> decode normalized latent in bounded batches
-> return writer-ready image batches
```

Process/Sampler 看不到 class label 和 codec。

### 12.2 output size 由 concrete Builder 推导

当前通用 `sampling.shape` 是 raw tensor state shape。对 latent task，用户更关心
256×256 image，而不是 `[4, 32, 32]`。

首版不修改所有 sampling schema 去假设 image。具体 latent Builder 可以在私有 params
接受：

```yaml
output_image_size: [256, 256]
```

再用 codec geometry 推导并验证 latent shape。若同时提供公共 `sampling.shape`，
二者必须精确一致，否则 fail closed。

### 12.3 trajectory 的 observation space 必须明确

数值 Sampler 产生的是 latent trajectory，普通 image writer 不能把它直接当
`[-1, 1]` RGB：

- 默认只 decode final state；
- 若请求可视化 trajectory，由 latent SamplingBuilder 选择少量 observation decode；
- manifest 同时记录 solver latent steps 与 decoded artifact steps；
- 不在每个数值 step 强制运行 VAE decoder；
- raw latent trajectory 可以由 tensor writer 单独保存。

### 12.4 CFG 不进入 Sampler

CFG adapter 可以批量拼接 cond/uncond forward，再把组合后的 prediction 提供给
Gaussian Dynamics。Sampler 仍只消费 family-specific Dynamics capability。

首版 fixed-variance Flowers DiT 对全部 4 个 prediction channels 应用 CFG。
官方 DiT `forward_with_cfg()` 为历史复现实验保留了只 guidance 前三个通道的特例，
该行为只能作为显式 upstream-parity policy，不能成为四通道 latent 默认。未来若支持
learned variance，只对 prediction partition 应用 guidance；variance partition 的
组合政策必须单独声明和测试。

CFG scale、label prior、negative/null condition、batching policy 必须进入
resolved sampling manifest。

## 13. Stable Diffusion / Diffusers 互操作

### 13.1 路径 A：完整 Diffusers pipeline backend

当前 `run_sampling()` 是 checkpoint-bound：strict partial sample request 必须配合
显式 checkpoint，且 runtime 会在构造内部 recipe Builder 前恢复 Process 与 primary
model provider。
因此完整 Diffusers pipeline 不能通过一个 dummy checkpoint 或仅含 `sampling:` 的
现有 config 接入。

Stage I0 应先给计划中的 library-level sampling request 增加显式 subject tagged union：

```python
SamplingSubject = (
    CheckpointSamplingSubject
    | InferenceBundleSamplingSubject
    | ExternalPipelineSamplingSubject
)
```

- checkpoint subject 保留当前 `SamplingBuilder` 路径；
- inference bundle subject 解析 Stochaflow task bundle；
- external pipeline subject 由 optional integration 构造 task-specific
  Diffusers inference method，不创建 Stochaflow primary provider 或 Process；
- 三者归一化成同一个 sampling outcome/artifact contract。

subject dispatch 是 operation 的显式 public contract，不按注册名、state-dict key 或
`model_index.json` 猜测。若暂不扩展 sampling request，则完整 pipeline backend 必须是
独立 external-pipeline operation；两种方案都不能绕过当前 runtime 约束。

在 external subject 内，由 Diffusers 完整拥有：

- components；
- scheduler；
- prompt encoding；
- CFG；
- preprocessing/postprocessing；
- device/offload；
- inference loop。

Stochaflow 只拥有：

- operation request；
- explicit model ID/revision；
- prompts/seed/sample IDs；
- budget；
- result normalization；
- artifact sink；
- Evaluation protocol 和 provenance。

该路径最快获得 SD 1.x、SDXL 或 SD3 的推理能力，但它证明的是互操作，不证明
Stochaflow 原生 Process/Sampler 可复现上游 scheduler。

### 13.2 路径 B：component-native integration

明确支持一个 family 后，分别加载：

- codec；
- denoiser；
- tokenizer/text encoders；
- scheduler/noise metadata；
- condition adapter。

随后由 Stochaflow Builder 组合 TrainingPlan 或 sampling method，且 Stochaflow 是
唯一 lifecycle owner。

首个 component-native target 选择 SD 1.x，而不是一次支持所有 SD：

- 组件拓扑相对简单；
- 单 CLIP text encoder；
- conditional UNet；
- 与已有 Gaussian family 更接近；
- LoRA 生态成熟。

### 13.3 Diffusers scheduler 不能直接等同于一个本地组件

Diffusers scheduler 往往同时包含：

- noise/timestep schedule；
- model input scaling；
- prediction conversion；
- inference timestep construction；
- solver transition。

它既不只等于 Stochaflow Gaussian schedule，也不天然只等于 Sampler。

策略是：

1. 完整 pipeline backend 中让 Diffusers scheduler 保持完整 owner；
2. 原生 recipe 继续使用 Stochaflow Process + family Sampler；
3. component-native interop 只为明确支持的 scheduler family 写完整 adapter；
4. 不镜像 Diffusers 全部 scheduler 名称、参数和默认值；
5. 不因为两个对象都有 `step()` 就宣称兼容。

### 13.4 LoRA 使用成熟实现，不自研注入引擎

[Diffusers LoRA 文档](https://huggingface.co/docs/diffusers/training/lora)和
[adapter loading API](https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters)
展示了 Diffusers/PEFT 的 adapter 注入、加载、组合、fuse/unfuse 与保存；官方
family-specific example scripts 提供训练参考。它不是对所有 pipeline 通用的高层
LoRA trainer。

Stochaflow 后续 LoRA scope 应聚焦：

- base asset immutable provenance；
- 哪些 module 拥有 adapter；
- adapter-only trainable parameter selection；
- optimizer/checkpoint/resume；
- adapter-only export；
- base + adapter inference bundle；
- family-specific capability validation；
- Evaluation。

不重写 LoRA 矩阵注入、fuse/unfuse 或 safetensors 序列化。

### 13.5 许可和 safety 属于 bundle provenance

不同 SD/codec assets 具有不同许可证和访问条款。bundle/manifest 必须记录并在
分发前检查：

- model license；
- gated access 或用户接受状态；
- base 与 adapter 的组合条款；
- safety checker 是否存在、启用或被显式省略；
- upstream model card 与已知限制。

Stochaflow 首版不自研 safety classifier，也不能因为 optional backend 没加载
safety checker 就把该事实从 manifest 中省略。

## 14. Flowers102 两种正式 profile

### 14.1 `flowers102-full-showcase-v1`

用途：得到可辨识、适合展示的 102 类 flower samples，并验证端到端工程能力。

允许：

- 使用全部 8,189 张；
- 固定水平翻转枚举或显式 augmentation；
- prepared latents；
- class-balanced sampling grid；
- 多个模型/CFG 的视觉 ablation。

必须声明：

```yaml
data_governance:
  profile: flowers102-full-showcase-v1
  uses_official_test_for_training: true
  heldout_test_claim: false
publication:
  seed_bank: <pre-registered>
  class_order: <stable-class-mapping>
  samples_per_class: <fixed-positive-integer>
  retain_failed_samples: true
```

禁止：

- 把训练分布上的 FID 写成 held-out benchmark；
- 与官方 classification split 的方法直接横向排名；
- 通过挑 seed/挑 class grid 宣称整体质量；
- 隐藏 pretrained codec 的 transfer 信息。

### 14.2 `flowers102-heldout-transfer-v1`

用途：评估在固定 pretrained codec 表示下，conditional latent generator 的
held-out transfer 能力。

协议：

1. official train 用于训练；
2. official validation 用于 recipe、checkpoint、CFG、sampler 和 early stopping；
3. `finalization` 必须选择并冻结一个 tagged variant：
   - `selected-checkpoint`：只对 validation 已选中的唯一 checkpoint 做一次 test；
   - `retrain-trainval`：在全部 recipe/hyperparameters 冻结后创建一个新 subject，
     用 train+validation 重训；预注册固定 epoch/step、seed/replicate policy 与最终
     checkpoint rule，不再使用 early stopping 或 validation selection；
4. 两个 finalization variants 具有不同 subject provenance、protocol digest 与
   result namespace，不能只把它们当同一 run 的实现细节；
5. official test 不能进入 HPO、checkpoint selection、early stopping 或可视化挑选；
6. 使用 pretrained VAE 时明确称为 transfer benchmark，不能声称严格 from scratch；
7. 记录无法排除 codec 预训练语料包含 Flowers 图片的限制。

由于 official train 只有每类 10 张，该 profile 很可能质量有限，但它的协议含义明确。

### 14.3 可选自定义 generative split

如果官方 split 对生成训练过小，可以从全部 8,189 张生成一个固定、按类分层的
generative split。它必须：

- 使用稳定 sample IDs 与 seed；
- 版本化 split manifest；
- 明确“不是官方 classification split”；
- held-out split 永不用于训练/选择；
- 结果只和相同 split identity 的 run 比较。

该方案不能与 `full-showcase-v1` 或 `heldout-transfer-v1` 共用 result namespace。

## 15. Flowers 模型与 recipe 决策

### 15.1 推荐实验顺序

| 阶段 | model state | backbone | 目的 |
| --- | --- | --- | --- |
| F0 | 64×64 pixel | 现有 UNet | 保留 smoke/失败对照 |
| F1 | 256×256 -> f8 latent | ADM-style conditional UNet | 验证 codec/condition/CFG/eval |
| F2 | 同 F1 | DiT-Nano/2 或小型 DiT | 验证 backbone substitution |
| F3 | 同 F1 | 等参数/等算力 UNet vs DiT | 形成研究结论 |
| F4 | pretrained SD family | LoRA | 视觉上限/interop showcase |

F1 是首个 promotion gate。不能因为 Anthos 使用 DiT 就让 F2 阻塞 latent plumbing。

### 15.2 初始候选，不是冻结 production 配置

```yaml
image_size: [256, 256]
latent_geometry: derived_from_codec
prediction_type: v
class_dropout_probability: 0.1
ema: true
sampling:
  sampler: ddim
  num_inference_steps: 50
  cfg_scale_sweep: [1.0, 2.0, 3.0, 4.0, 5.0]
```

这些值需要实验选择，尤其：

- epsilon vs v；
- UNet vs DiT；
- posterior sample vs prepared posterior sample；
- CFG scale；
- class prior；
- 训练时长；
- latent normalization 与 precision。

`v` 不能作为“MNIST 有效，所以 latent 必须有效”的框架默认。

### 15.3 Stable Diffusion fine-tune 的定位

LoRA/SD fine-tune 最容易产生漂亮花朵，因为 base model 已学习花、摄影、纹理、
光照和语言语义。但它主要证明：

- pretrained asset interop；
- adapter lifecycle；
- prompt data；
- parameter-efficient tuning；
- bundle/export/evaluation。

它不单独证明 Stochaflow 原生 Process/Sampler 或 from-scratch generative recipe。
因此它是 F4 interop showcase，不替代 F1/F2。

## 16. Evaluation 设计

### 16.1 先评 codec reconstruction

在训练 denoiser 前，对 frozen codec 执行正式 reconstruction Evaluation：

- deterministic posterior mode encode/decode，作为 optimistic reconstruction profile；
- 与实际训练完全相同的 configured posterior policy，使用固定 seed/multiple
  replicates，作为 operational codec profile；
- 两种 profile 使用不同 protocol/result identity；prepared latent recipe 还必须复用
  实际 artifact 的 posterior policy、latent transform 与存储 dtype；
- exact profile-declared evaluation sample IDs；heldout profile 必须是未参与训练/选择的
  test，full-showcase 只能形成 reconstruction sanity result；
- PSNR；
- SSIM；
- LPIPS；
- reconstruction FID/rFID 或固定 feature distance；
- 花瓣、花蕊、叶片边缘与背景的固定 visual panel；
- encode/decode latency、peak memory；
- non-finite 与 range audit。

若 codec 已明显破坏关键结构，必须：

- 更换 codec；
- 降低压缩率；
- 调整输入预处理；
- 或降低 recipe 的质量目标。

不能用更大的 DiT 掩盖 codec 的 representational limit。

### 16.2 generation 不是一个 FID

Flowers 正式生成 Evaluation 至少包括：

1. distribution quality；
2. intended-class fidelity；
3. diversity/coverage；
4. memorization audit；
5. performance；
6. provenance/completeness。

推荐 held-out profile：

- 按 official test label histogram 生成相同 6,149 张；
- 预注册 3–5 个 seed；每个 seed 都生成一个完整的 6,149-sample replicate，
  逐 replicate 报告结果，再汇总 arithmetic mean 与 `between_seed_std`，不能拼成一个
  样本池只报一个分数；v1 不计算未定义 estimator 的置信区间；
- 因 KID 有简单无偏估计，本 protocol 选择它作为小数据 primary distribution
  metric；固定 provider、feature extractor/version、subset size/count 与 metric RNG。
  KID 单个 replicate 的 subset resampling 结果命名为 `kid_within_subset_mean/std`，
  跨 generation seed 的汇总命名为 `kid_between_seed_mean/std`，不得混成同一个
  `kid_std`；
- FID 作为 secondary，并由固定版本的 Clean-FID `clean` protocol 计算；固定
  input quantization/PNG、resize、feature extractor 和 result key；
- precision/recall 固定具体定义、实现、`k` 与 feature extractor；
- uniform-class quantitative plan 固定每类样本数，另附预注册 seed、class ordering、
  每类样本数且保留失败样本的 balanced grid；
- frozen Flowers classifier 的 intended-class macro accuracy/confusion；其 artifact
  manifest 必须冻结 model/weights digest、training dataset/split IDs、preprocessing、
  HPO/selection history，且 official test 与 generated images 不得参与 classifier
  fit、HPO 或 checkpoint selection；无法审计 lineage 的外部 classifier 只能标为
  non-independent supporting evidence；
- classifier 在真实 test 上的 accuracy、per-class recall、calibration 与 confusion
  同时报告，作为 measurement reliability/reference baseline，而不是 generated
  accuracy 的数学上限；只有分类器通过预注册可靠性 gate 时才解释 generated score，
  否则结果标为 inconclusive；
- nearest-neighbor memorization 与 exact/near-duplicate audit 默认只对实际训练语料
  比较，并固定 feature/threshold；test-neighbor 仅作 reference/leakage investigation，
  不与 memorization result 混名；
- raw/EMA、CFG、Sampler、NFE、seed 和 sample count 全部冻结。

固定同一 official test reference 时，`between_seed_std` 只描述给定数据集下的
generator seed 与 metric RNG 条件变异，不代表完整 dataset uncertainty。置信区间待
Evaluation 计划的 richer comparison/CI provider 明确 estimator、confidence level、
重采样单位与 result naming 后再加入。

[KID 论文](https://arxiv.org/abs/1801.01401)、
[有限样本 FID 偏差研究](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)
与 [Clean-FID](https://github.com/GaParmar/clean-fid)共同说明小数据、预处理和
sample count 会显著影响结论。

### 16.3 不把 per-class FID 设为主指标

Flowers 单类 held-out real images 很少，102 个 per-class FID 的 covariance estimate
不稳定。首版使用：

- aggregate KID/FID；
- intended-class macro accuracy；
- confusion matrix；
- class-balanced coverage summary；
- worst-class visual panel；
- 每类有效样本数。

若未来引入 per-class distribution metric，必须先通过最小 sample size 与置信区间 gate。

### 16.4 label prior 是协议的一部分

至少区分：

- **empirical-prior quality**：按 reference split 的 class histogram 生成；
- **uniform-class capability**：每类生成相同数量，测 class coverage/fidelity。

二者不能合并成一个 FID，也不能让 sample allocation 根据运行中失败自动变化。

### 16.5 Evaluation 复用推理 capability

EvaluationBuilder 不重新实现：

- latent shape；
- CFG；
- condition adapter；
- Dynamics/Sampler compatibility；
- codec decode；
- preprocessing。

它消费由 sampling/task 层已验证的窄 inference capability，在外层组合：

- selected subject；
- data/reference plan；
- metric bindings；
- artifact sink；
- completeness；
- immutable EvaluationResult。

## 17. 配置草案

以下只表达 responsibility，不是冻结 schema。

### 17.1 on-the-fly latent training

```yaml
data:
  name: flowers102_class_image
  params:
    manifest: ./data/prepared/flowers102/<dataset-key>/manifest.yaml
    split_profile: flowers102-heldout-transfer-v1
    image:
      size: [256, 256]
      crop: center_square
      interpolation: bicubic
      antialias: true
      output_value_range: [-1.0, 1.0]
      random_horizontal_flip: true

model:
  name: class_conditional_dit
  params:
    input_channels: 4
    output_channels: 4
    patch_size: 2
    hidden_size: 384
    depth: 12
    num_heads: 6
    num_classes: 102
    learned_variance: false
    internal_class_dropout: false

training:
  name: class_conditional_latent_gaussian
  params:
    prediction_type: v
    class_dropout_probability: 0.1
    codec:
      name: diffusers_autoencoder_kl
      params:
        source:
          repo_id: stabilityai/sd-vae-ft-ema
          revision: <immutable-commit>
        encoding_policy: posterior_sample

sampling:
  run_after_training: true
  sampler:
    name: ddim
    params:
      num_inference_steps: 50
      eta: 0.0
  options:
    output_image_size: [256, 256]
    clip_denoised: false
    labels:
      policy: balanced
    guidance:
      scale: 4.0
```

TrainingBuilder 将 `class_conditional_latent_denoising` 写入
`TrainingPlan.inference_recipe.name`，并把 `prediction_type: v` 与
`codec_asset: codec` 放入 fixed contract。codec config 留在具体 TrainingBuilder 私有
params；不新增顶层通用 `codec:` 或 `conditioning:`，除非第二个稳定 task family
证明存在真正跨任务公共语义。

### 17.2 prepared latent training

```yaml
data:
  name: prepared_class_latents
  params:
    manifest: ./data/prepared/flowers102-latents/<latent-key>/manifest.yaml
    split: train

training:
  name: class_conditional_prepared_latent_gaussian
  params:
    prediction_type: v
    class_dropout_probability: 0.1
    expected_latent_artifact: <latent-key>
    decoder_asset:
      name: diffusers_autoencoder_kl
      params:
        source:
          repo_id: stabilityai/sd-vae-ft-ema
          revision: <same-immutable-commit>
```

Builder 必须验证 latent artifact 的 codec/transform identity 与 decoder asset 完全一致。

### 17.3 Diffusers black-box inference

这是未来 `ExternalPipelineSamplingSubject` request 草案，不是当前
checkpoint-backed strict partial sample request：

```yaml
subject:
  kind: external_pipeline
  integration: diffusers
  family: stable-diffusion-1
  source:
    repo_id: stable-diffusion-v1-5/stable-diffusion-v1-5
    revision: <immutable-commit>

sampling:
  prompts_file: ./prompts/flowers-v1.jsonl
  num_inference_steps: 50
  guidance_scale: 7.5
  safety:
    policy: upstream-default
```

`family` 是显式 compatibility selector，不根据 `model_index.json` 猜测并承诺任意
pipeline 都可用。request resolver 不创建 Stochaflow Process/primary provider，
也不允许用 dummy checkpoint 进入当前 checkpoint-bound runtime。

## 18. 分阶段实施

### Stage L0：完成 AFHQ 条件生成前置

依赖：

- class-aware DataBuilder；
- stable class mapping/provenance；
- `ClassConditionalDenoiser` capability；
- class dropout；
- CFG SamplingBuilder；
- ADM-style conditional UNet；
- AMP、gradient accumulation；
- generation Metrics/Evaluation 基础。

验收：

- AFHQ pixel conditional generation 不修改 Process/Sampler root；
- 自定义 external conditional denoiser 通过替换性 contract；
- raw/EMA/CFG sample plan 可重放。

### Stage L1：codec capability 与 reconstruction gate

实现：

- `ImageLatentSpec`；
- encoder/decoder capability；
- first-party adapter contract tests；
- optional Diffusers `AutoencoderKL` adapter；
- immutable source/revision/digest provenance；
- deterministic posterior-mode 与 configured operational-posterior 两套
  reconstruction Evaluation。

验收：

- scaling/shift/mean/std 正反变换 round-trip；
- posterior sample 可按 generator 重放；
- mode encode deterministic；
- shape/range/dtype 失败路径；
- codec test reconstruction report。

### Stage L2：frozen codec training lifecycle

实现：

- latent TrainingBuilder/Strategy；
- codec freeze + `mode="eval"`；
- on-the-fly encode；
- prepared latent clean-state provider；
- prediction type compatibility；
- latent `clip_denoised=false` guard；
- checkpoint strict resume。

验收：

- codec 参数全部 frozen，Trainer 验证 optimizer 与 Plan-selected parameters 精确一致；
- codec 不进入 denoiser EMA；
- dropout/BN buffers 保持 eval；
- uninterrupted 与 resumed tiny run 一致；
- external custom codec capability 通过 LSP test。

### Stage L3：inference auxiliary asset projection

实现：

- sampling checkpoint view 保留 verified inference assets；
- `TrainingPlan.inference_assets` typed projection；
- 新 checkpoint schema 的 `inference_asset_descriptors`，不迁移 v10；
- `InferenceAssetProvider`；
- embedded auxiliary state resolve；
- concrete Builder capability validation；
- 只加载被请求 asset；
- decoded final sample 与可选 decoded trajectory。

验收：

- latent checkpoint 可在独立进程采样；
- 缺 codec、错 asset name、错 state、错 declaration 立即失败；
- distillation teacher 不被 latent sampler 自动加载；
- writer 不把 latent 当 image；
- run manifest 记录 resolved codec。

### Stage L4：Flowers conditional latent UNet vertical slice

实现：

- Flowers class-aware data artifact；
- 两个 profile manifest 与 heldout finalization variants；
- conditional latent ADM UNet recipe；
- balanced labels/CFG；
- codec reconstruction gate；
- generation Evaluation；
- smoke/production config 分离。

验收：

- full-showcase 明确 test contamination；
- heldout-transfer 的 test 不参与 selection；
- KID/FID/class fidelity/nearest-neighbor artifacts 完整；
- 失败样本和 sample IDs 可追溯。

### Stage L5：DiT substitution 与 ablation

实现：

- class-conditional DiT；
- latent geometry/patch validation；
- adaLN-Zero；
- fixed position embedding policy；
- `learned_variance=false` 与 C-channel output contract；
- model-internal class dropout disabled；
- 全部四个 prediction channels 的 CFG policy；
- equal-budget UNet/DiT comparison recipe。

验收：

- 同一 Process/Objective/Sampler/codec 下替换 backbone；
- core runner 无 concrete class/name branch；
- DiT 不是 promotion 前置，只有实证结果才能升级默认 recipe。

### Stage I0：Diffusers complete-pipeline backend

实现：

- optional dependency extra/extension；
- `ExternalPipelineSamplingSubject` 或独立 external-pipeline operation；
- explicit supported family allowlist；
- pinned repo revision；
- deterministic prompt/sample IDs；
- upstream safety/offload policy manifest；
- Evaluation adapter。

验收：

- Diffusers 独占 inference loop；
- 不创建 dummy checkpoint、Stochaflow primary provider 或 Process；
- Stochaflow Sampler 不包装上游 scheduler；
- unsupported pipeline family fail closed；
- offline cached revision 可重放。

### Stage I1：SD 1.x component-native interop

实现：

- SD1 codec/text encoder/tokenizer/UNet adapters；
- text-conditioned narrow capability；
- prompt/negative prompt CFG；
- 明确 scheduler adapter；
- Stochaflow checkpoint；
- Diffusers-format inference export。

验收：

- 与固定 upstream pipeline 做 seeded parity tolerance test；
- prediction type、timestep、latent transform、prompt embeddings 对齐；
- 组件和 Stochaflow lifecycle 只有一个 owner。

### Stage I1b：SD 2.x 独立 compatibility case

实现前冻结：

- OpenCLIP ViT-H text encoder/tokenizer identity；
- checkpoint-specific epsilon/v prediction；
- VAE/latent transform；
- scheduler adapter；
- seeded upstream parity protocol。

SD1 parity 通过不自动使 SD2 checkpoint 兼容；SD2 使用独立 family ID、bundle
descriptor 和测试结果。

### Stage I2：LoRA

实现：

- PEFT adapter injection provider；
- adapter-only parameter selection；
- base assets immutable references；
- adapter checkpoint/resume/export；
- base + adapter bundle；
- SD1 与 SDXL 分别声明 capability。

验收：

- base weights 训练前后 digest 不变；
- optimizer 只包含允许的 adapter parameters；
- adapter-only export 可由固定 base revision 恢复；
- SD1 的通过不自动宣称 SDXL/SD3 兼容。

### Stage I3：SDXL/SD3 decision gates

SDXL gate：

- 第二 text encoder；
- pooled/text sequence condition；
- size/crop condition；
- refiner 是否独立 pipeline；
- LoRA target modules；
- memory/offload。

SD3 gate：

- rectified flow Process/Dynamics/Sampler 已有独立计划和实现；
- MMDiT condition capability；
- multiple text encoders；
- SD3 latent transform；
- license/access policy。

未通过各自 gate 前，只能通过 black-box backend 使用，不能标为 native support。

## 19. 测试矩阵

### 19.1 codec

- encode/decode shape；
- image range；
- exact scaling inverse；
- non-default scaling/shift；
- mean/std transform；
- posterior sample generator replay；
- deterministic mode；
- configured operational posterior reconstruction；
- odd/non-divisible image size；
- fp32/bf16/fp16 policy；
- CPU/CUDA capability matrix；
- non-finite output；
- custom non-Diffusers codec。

### 19.2 managed assets

- explicit freeze；
- eval mode；
- optimizer exclusion；
- EMA exclusion；
- embedded checkpoint；
- typed inference projection/declaration round-trip；
- strict resume；
- immutable reference digest match/mismatch；
- offline missing asset；
- license metadata；
- only requested inference assets loaded。

### 19.3 training

- image-backed batch；
- prepared-latent batch；
- class mapping；
- null token；
- class dropout 0/1/intermediate；
- epsilon/v；
- loss shape/scalar；
- no latent clipping；
- resumed RNG；
- custom conditional denoiser。

### 19.4 sampling

- output image size -> latent shape；
- wrong geometry；
- balanced/empirical labels；
- CFG 1 = conditional baseline；
- batched vs separate cond/uncond parity；
- fixed-variance DiT 对全部四个 prediction channels guidance；
- upstream three-channel parity policy 只能显式启用；
- DDPM/DDIM；
- raw/EMA；
- final decode；
- decoded trajectory sampling；
- latent tensor artifact；
- writer range。

### 19.5 Evaluation

- codec reconstruction profile；
- deterministic/operational codec profile identity 不混用；
- exact reference/sample IDs；
- KID estimator/feature/subset/RNG；
- pinned Clean-FID `clean` FID preprocessing/version；
- empirical vs uniform label prior；
- class classifier reliability/reference baseline 与 gate；
- classifier training lineage、official-test exclusion 与 unknown-lineage limitation；
- intended-class macro accuracy；
- KID within-subset 与 between-seed uncertainty names 不冲突；
- nearest-neighbor/duplicate audit；
- incomplete generation fail closed；
- full-showcase result 不能冒充 heldout result；
- test result 不参与 selection/HPO。

### 19.6 Diffusers interop

- pinned revision；
- external subject 不要求 checkpoint；
- unsupported family；
- SD1 seeded parity；
- scheduler incompatibility；
- missing tokenizer/text encoder；
- prompt/negative prompt；
- upstream safety policy；
- offline cache；
- LoRA base digest；
- adapter-only export。

## 20. 验收标准

### 20.1 架构

- 未新增 `LatentProcess`；
- Process/Sampler root 没有 codec、class、prompt 或 CFG 方法；
- primary model 是 denoiser，codec 是具名 frozen auxiliary；
- Strategy 不构造或 freeze assets；
- SamplingBuilder 拥有 decode；
- Evaluation 复用已验证 inference capability；
- conditional UNet 与 DiT 可替换而无需修改 runner；
- SD1/SDXL/SD3 没有通过一个 mode enum 混合。

### 20.2 可复现性

- dataset、split、sample、codec、weights、transform、prediction type、CFG、
  sampler、seed 和 metric protocol 都有稳定 identity；
- checkpoint 可 strict resume；
- 独立 sampling 可恢复 codec；
- immutable reference 按 digest 校验；
- prepared latents 与 codec identity 严格绑定；
- full-showcase/test contamination 显式。

### 20.3 用户体验

- 用户能列出并复制 Flowers latent recipe；
- smoke config 不需要长训练；
- production config 有明确硬件/时间说明；
- error 指向具体不兼容项；
- stable diffusion 支持页面展示精确 family/level matrix；
- 不需要理解内部 checkpoint mapping 就能采样或评估。

### 20.4 质量

- codec reconstruction gate 先通过；
- Flowers full-showcase 有固定、未挑选的 102-class artifact；
- heldout profile 有完整 distribution/class fidelity/memorization report；
- UNet/DiT 结论来自控制实验；
- 不引用 13 类 FID 表作为完整数据集门槛；
- 不引用 Anthos 作为 held-out quantitative baseline。

## 21. 主要风险与缓解

| 风险 | 后果 | 缓解 |
| --- | --- | --- |
| codec 抹掉细花瓣 | denoiser 再大也无法恢复 | reconstruction gate、比较 codec/压缩率 |
| 把 0.18215 硬编码 | train/sample latent 分布不一致 | transform 完全封装进 adapter |
| codec 只 eval 未 freeze | VAE 进入 optimizer | Builder freeze contract + parameter audit |
| checkpoint 重复 VAE | 大量磁盘占用 | 先 embedded correctness，后 immutable reference |
| sampling 丢 codec | checkpoint 无法独立生成 | inference asset provider |
| prepared latent 漂移 | 数据与 decoder 不匹配 | content-addressed manifest |
| official train 太小 | overfit/低质量 | showcase 与 heldout profile 分离 |
| 全量训练却报告 test | 数据污染 | `uses_official_test_for_training` 强制字段 |
| pretrained VAE 看过 Flowers | from-scratch claim 不成立 | 明确 transfer benchmark 和限制 |
| classwise FID 样本太少 | 噪声排名 | aggregate metrics + macro classifier fidelity |
| CFG 只优化漂亮 grid | 类别/多样性退化 | 预注册 CFG sweep 与全量 metrics |
| DiT 被当硬要求 | 过拟合架构时尚 | 先 UNet vertical slice、等预算 ablation |
| Diffusers 与 core 双重 owner | device/scheduler/state 冲突 | black-box 与 component-native 分离 |
| 镜像 upstream namespace | 维护不可控 | allowlisted family adapter、optional provider |
| “Stable Diffusion supported”过度承诺 | 用户误用 SDXL/SD3 | family + level compatibility matrix |
| model/data license 不清 | 无法安全分发 | artifact provenance 与发布前 license gate |

## 22. 明确不进入首版

- 训练 VAE、VQ-VAE 或 VQGAN；
- VAE + diffusion joint training；
- adversarial autoencoder 和多 optimizer loop；
- 通用 condition/batch schema；
- 通用多模型 YAML DAG；
- 全部 Diffusers model/scheduler/config namespace 镜像；
- arbitrary `.ckpt` 自动转换；
- ControlNet、IP-Adapter、DreamBooth、textual inversion 全套；
- SDXL native training；
- SD3 native Gaussian compatibility；
- 自研 tokenizer、CLIP、T5、LoRA/PEFT 或 safety checker；
- 用一个 universal `StableDiffusionPipeline` 替代 task-specific Builders；
- 因为 image resolution 是 256 就自动选择 latent；
- 因为 model class 是 DiT 就自动选择 flow/diffusion；
- 训练时在线下载浮动 pretrained assets；
- 将 dataset test 用于训练后仍发布 held-out score。

## 23. 调研来源

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
- [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
- [Diffusers Pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview)
- [Diffusers Stable Diffusion pipeline](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
- [Diffusers LoRA training](https://huggingface.co/docs/diffusers/training/lora)
- [Diffusers adapter loading](https://huggingface.co/docs/diffusers/main/using-diffusers/loading_adapters)
- [Scalable Diffusion Models with Transformers / DiT paper](https://arxiv.org/abs/2212.09748)
- [Official DiT implementation](https://github.com/facebookresearch/DiT)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Diffusers scheduler guide](https://huggingface.co/docs/diffusers/using-diffusers/schedulers)
- [Stable Diffusion 2 model card](https://huggingface.co/stabilityai/stable-diffusion-2)
- [SDXL paper](https://arxiv.org/abs/2307.01952)
- [SDXL model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Stable Diffusion 3 paper](https://arxiv.org/abs/2403.03206)
- [Diffusers Stable Diffusion 3](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_3)
- [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- [TFDS Flowers102 split](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)
- [Anthos model card](https://huggingface.co/Glint-Research/Anthos-1)
- [Anthos fixed repository tree](https://huggingface.co/Glint-Research/Anthos-1/tree/56ff6df2849f4c4ad99a5cc5e804da6b4177dfda)
- [Keras Flowers DDPM example](https://keras.io/examples/generative/ddpm/)
- [Keras Flowers DDIM example](https://keras.io/examples/generative/ddim/)
- [KID](https://arxiv.org/abs/1801.01401)
- [Effectively Unbiased FID and Inception Score](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)
- [Clean-FID](https://github.com/GaParmar/clean-fid)
