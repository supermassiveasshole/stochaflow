# Latent Diffusion 支持计划

- 文档性质：开发计划；不属于当前公开 API 或正式用户文档
- 状态：当前产品主线；Phase 1 inference asset projection 已完成，Phase 2–3
  尚未实现且为 P1，Phase 4A–4C 为 P1/P2，开放数据与正式 Evaluation 为 P2
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)；本文拥有 latent
  capability contract，统一排期拥有跨计划执行顺序
- 初始制定日期：2026-07-26
- 本次修订日期：2026-07-30
- 排期前置：Phase 1 已完成；先完成 A0 ADM topology correctness、B1/C1
  Train/Sample authority、Metrics M0–M1 和 A1 learned-range/P2 core，再进入本计划
  Phase 2；这是当前单人实施顺序，不表示 codec 架构依赖 ADM、P2 或 MetricEngine，
  A2 长实验可与 Phase 2 并行
- 当前主线：冻结的预训练图像 codec + conditional latent Gaussian diffusion
- 首个 reference denoiser：DiT-S/2 → DiT-B/2；它是验收实现，不是该计划的
  abstraction boundary
- 并行关联主线：同一 codec/latent contract 上的 Stable Diffusion
  component-native interoperability、text conditioning 与 UNet training；
  由
  [Stable Diffusion Component-Native 支持计划](stable-diffusion-component-native-support-plan.md)
  独立拥有
- 首个开放正式目标：从 The Met Open Access 策展并冻结的 150k–300k
  image/metadata snapshot；精确数量和 taxonomy 由 profiling gate 冻结
- 对照目标：原始分辨率 ImageNet-100 class benchmark；不得使用短边已缩至
  160 pixels 的镜像冒充 256×256 source
- 扩展目标：LHQ quality probe、DomainNet class + domain conditioning，以及
  independent non-DiT denoiser substitution
- correctness/smoke：AFHQ-v2；它不再承担规模和最终质量证明
- 关联计划：
  [默认工作流与推理 Pipeline 支持计划](default-workflow-pipeline-support-plan.md)、
  [Metrics 支持开发计划](metrics-support-plan.md)、
  [训练后 Evaluation 与 Benchmark 支持计划](post-training-evaluation-support-plan.md)、
  [P2 Weighting 与 ADM 拓扑修复计划](p2-weighting-and-adm-topology-refactor-plan.md)

## 1. 本轮结论

本计划冻结以下边界：

1. **Stochaflow 不实现 VAE 网络。**
   首版通过可选 Diffusers adapter 使用独立的
   `diffusers.AutoencoderKL`。
2. **Stochaflow 不负责 VAE 训练 lifecycle。**
   公开预训练 VAE 和外部工程训练的 VAE，都以固定的 Diffusers model
   directory 或 Hugging Face Hub revision 进入 Stochaflow。
3. **“VAE 训练交给 Diffusers”是 workflow ownership 决策，不是对
   Diffusers Trainer API 的承诺。**
   Diffusers 当前提供 `AutoencoderKL`、加载/保存格式和大量训练基础设施，
   但没有一个覆盖 reconstruction、KL、LPIPS、GAN、双优化器、checkpoint
   的稳定通用 VAE Trainer。外部训练工程可以使用 Diffusers model class、
   Accelerate 和自己的训练脚本；该脚本及其恢复语义不属于 Stochaflow。
4. **Stochaflow 必须拥有 codec contract。**
   图像范围、latent geometry、posterior policy、scaling/shift/mean/std、
   precision/upcast 和严格逆变换不能委托给任意上游 Pipeline 隐式处理。
5. **encoder 和 decoder 是同一个不可拆分的 codec asset。**
   不允许从不同 revision 任意拼接 encoder、decoder 或 latent transform。
6. **不新增 `LatentProcess`。**
   Gaussian Process 只处理 tensor state；pixel 或 normalized latent
   的区别属于 task composition。
7. **DiT 是 denoiser backbone，不是新的 Process 或 Sampler family。**
8. **预训练 codec 只在训练配置中声明一次。**
   resolved checkpoint/bundle 保存其精确 identity；sampling、evaluation
   和 resume 不要求用户重新填写 VAE 配置。
9. **prepared latent 是 DataArtifact，VAE 权重不是 DataArtifact。**
   前者是数据物化结果，后者是训练/推理模型资产。
10. **完整 Diffusers Pipeline 互操作是独立轨道。**
    首版只消费独立 `AutoencoderKL`，不把 `DiffusionPipeline` 包装成
    Stochaflow `Sampler`。
11. **Stable Diffusion 没有从计划中移除。**
    Latent Diffusion 先闭合 codec、latent artifact、checkpoint 和 sampling 的
    公共前置能力；
    Stable Diffusion 计划独立实施 1.x-compatible component import、text
    condition assets、UNet full-parameter fine-tuning/training 与 512×512
    sampling parity。
12. **“Stable Diffusion 支持”必须按层级声明。**
    black-box Diffusers Pipeline inference、component-native SD 1.x、
    Stochaflow-owned Stable Diffusion training，以及 SDXL/SD3 不是同一个
    capability，也不能用一个布尔开关或模糊兼容声明代替；这些层级由独立
    Stable Diffusion 计划冻结。

一句话目标：

> 本计划让 Stochaflow 拥有 codec/latent workflow 与 latent diffusion
> training/sampling 的组合、身份和可复现语义；DiT 是首个 reference
> denoiser，而不是公共边界。Diffusers 提供预训练 image codec 的模型实现与
> 标准权重格式。Stable Diffusion 在独立计划中复用这些能力，外部工程继续拥有
> VAE 的训练方法和训练恢复。

## 2. 为什么采用这个边界

### 2.1 当前真正需要证明的是 Latent Diffusion workflow，而不是 VAE research

现阶段的产品和研究目标是：

```text
image data
  -> fixed pretrained codec
  -> diffusion-normalized latent
  -> conditional denoiser training
  -> latent sampling
  -> fixed pretrained codec decode
  -> image evaluation
```

要完成这条链路，框架必须解决：

- codec 如何作为 frozen auxiliary 参与 device/mode/checkpoint；
- pixel batch 如何转换为 normalized latent；
- prepared posterior artifact 如何绑定 codec 和 preprocessing；
- latent checkpoint 如何在独立进程恢复 decoder；
- denoiser、Process、Sampler 和 writer 如何维持清晰边界；
- 训练配置中的 codec 如何只声明一次。

这些问题都不要求 Stochaflow 先具备：

- KL autoencoder 训练；
- LPIPS/perceptual loss；
- discriminator；
- 多优化器交替更新；
- VAE checkpoint resume；
- VAE model hub。

把后者提前纳入会让第一条 Latent Diffusion vertical slice 同时承担两种不同
training-loop family，延迟真正需要验证的框架边界。DiT-S/2 是首个
reference denoiser，因为仓库已有实现且 geometry 清楚。首个 vertical slice 通过
independent fake codec/asset capability test 保护公共边界；independent non-DiT
denoiser substitution 是后续 framework-generalization promotion gate，不阻塞
experimental support。

### 2.2 Diffusers 适合作为 codec provider，不适合作为隐式 framework owner

Diffusers 的 `AutoencoderKL` 已提供：

- `from_pretrained()` 和 `from_single_file()`；
- Diffusers model directory 与 safetensors 权重；
- `encode(...).latent_dist`；
- posterior `sample(generator)` 和 `mode()`；
- `decode(...).sample`；
- slicing、tiling 等模型能力；
- `scaling_factor`、`shift_factor`、`latents_mean`、`latents_std`、
  `force_upcast` 等配置事实。

但脱离完整 Pipeline 后，以下行为仍必须由 Stochaflow adapter 显式拥有：

- 输入是不是 `[-1, 1]`；
- posterior sample 还是 mode；
- normalization 的正向顺序；
- decode 前的严格逆变换；
- `force_upcast` 对 standalone module 的执行；
- output range 和 finite check；
- latent geometry 与具体 denoiser geometry 的兼容验证。

因此正确关系是：

```text
Stochaflow codec capability
        ^
DiffusersAutoencoderKLCodec adapter
        ^
diffusers.AutoencoderKL + pinned weights
```

而不是：

```text
Stochaflow
  -> arbitrary DiffusionPipeline
  -> 猜测其内部 VAE、normalization 和 scheduler
```

### 2.3 “完整 VAE + Latent Diffusion”不等于“同一框架训练所有组件”

本计划中的完整系统是端到端具备：

- image encode；
- stochastic/deterministic posterior policy；
- latent diffusion training；
- latent sampling；
- image decode；
- checkpoint/bundle；
- reconstruction/generation evaluation；
- codec 可替换性。

codec 可以来自：

1. 公开的 pinned pretrained model；
2. 团队独立运行的 VAE training project；
3. 未来其他遵循相同 capability 的 provider。

系统完整性由契约和可重放的组合保证，不由“所有权重必须在 Stochaflow
中训练”保证。

## 3. 术语和 scope

| 名称 | 本计划中的定义 | 不等于 |
| --- | --- | --- |
| image codec | 成对的 image encoder、latent transform 和 image decoder | 任意 feature extractor |
| codec-native latent | `AutoencoderKL.encode()` 的 posterior 所在空间 | diffusion model 直接消费的 state |
| diffusion-normalized latent | 经 codec-owned affine transform 后交给 Process/denoiser 的 tensor | 固定乘 `0.18215` |
| pretrained codec source | pinned Hub snapshot 或本地 Diffusers model directory | DataArtifact |
| external VAE training | Stochaflow 之外产生 codec weights 的训练 workflow | Diffusers 已提供通用 VAE Trainer |
| prepared posterior artifact | 由 image artifact + codec + encode recipe 物化的 posterior moments shards | 临时 read-through cache |
| Latent Diffusion | 在 fixed codec latent 上训练 generative dynamics 的 workflow family | Stable Diffusion 或 DiT 的同义词 |
| DiT | patchified latent/image denoiser backbone | Process、Sampler 或 Pipeline |
| Stable Diffusion-compatible family | frozen VAE、text encoder/tokenizer、conditional UNet、Gaussian schedule 和 CFG 的明确组合 | 任意 latent diffusion、任意 Diffusers Pipeline |
| Diffusers pipeline | 上游拥有 models、scheduler、processors 和推理 loop 的完整组合 | Stochaflow SamplingBuilder |

支持级别明确为：

| 层级 | 能力 | 本计划状态 |
| --- | --- | --- |
| C0 | 独立 Diffusers `AutoencoderKL` 加载、冻结、encode/decode | 首版 |
| C1 | image-backed conditional latent Gaussian training | 首版 |
| C2 | prepared posterior artifact training | 首版 production path |
| C3 | DiT-S/2、DiT-B/2 reference denoiser substitution | 首版 |
| C4 | 外部训练的 Diffusers-format VAE 作为 source | 首版 |
| C5 | independent non-DiT denoiser 复用同一 latent workflow | 后续 framework-generalization gate |
| C6 | 完整 Diffusers Pipeline black-box inference | 独立 Stable Diffusion 计划 |
| C7 | Stable Diffusion 1.x component-native interop/parity | 独立 Stable Diffusion 计划 |
| C8 | Stable Diffusion text-conditioned UNet fine-tuning/training | 独立 Stable Diffusion 计划 |
| C9 | SDXL/SD3 component-native composition | future family-specific gate |
| C10 | Stochaflow-native VAE training | deferred decision gate |
| C11 | joint VAE + denoiser training | 新方法 family，不在本计划 |

## 4. 当前仓库审计

### 4.1 已有正确边界

| 当前能力 | 对 Latent Diffusion 的价值 |
| --- | --- |
| `Process` model-free、task-free | 同一 Gaussian path 可作用于 pixel 或 latent tensor |
| `TrainingBuilder` 是 training composition root | 可组装 denoiser、codec、Process 和 Strategy |
| `TrainingPlan.auxiliary_modules` | codec 可作为具名 frozen auxiliary |
| `ManagedTrainingModule(mode="eval")` | core 可维持 codec eval mode |
| optimizer 只收集 `requires_grad=True` 参数 | frozen codec 可排除在优化器外 |
| EMA 只跟踪 primary model | codec 不进入 denoiser EMA |
| Strategy 解释结构化 `Any` batch | image/posterior batch 不污染 Trainer |
| SamplingBuilder 负责 task sampling | condition、CFG、decode 不进入数值 Sampler |
| checkpoint 保存 `training_assets_state_dict` | embedded codec state 已有存储基础 |
| checkpoint schema 已有 `InferenceAssetDescriptor` | 不需要重新发明 descriptor |

必须继续保留：

- primary model 是 denoiser，不是 codec；
- Strategy 不构造、下载、freeze 或序列化 codec；
- Process 不拥有 codec；
- Sampler 不解释 class、prompt、pixel range 或 decoder；
- DataBuilder 不加载模型权重；
- 不给现有 `standard_denoising` 堆叠 `latent=true`、
  `conditional=true` 等模式开关。

### 4.2 已闭合的 asset projection 与当前断点

Phase 1 已经闭合并验证：

- `TrainingPlan` 产生 `InferenceAssetDescriptor`；
- factory 将 descriptors 与 `training_assets_state_dict` 交给 checkpoint；
- sampling checkpoint view 只投影请求的 descriptor 和 auxiliary state；
- `InferenceAssetProvider` 可从 checkpoint-only embedded state 重建 module；
- `inference_recipe` 与 asset projection 可在 strict resume、断网和非项目 cwd 下
  恢复。

当前剩余断点是 codec-specific composition，而不是 projection wiring：

1. 尚无 image codec capability 与 Diffusers `AutoencoderKL` provider；
2. latent TrainingBuilder 尚未构造、freeze 并声明 codec asset；
3. latent SamplingBuilder 尚未请求 codec、恢复 latent transform 并 decode；
4. diagnostics 仍默认假设 pixel/image state，并可能开启 pixel clipping；
5. 正式长训练尚无 run-level codec asset bundle 去重。

所以本计划不再“设计一个新的通用 asset universe”，而是在已验证的 checkpoint
projection 上接入 concrete codec 与 latent recipe。

### 4.3 当前 DiT 能力

现有 DiT：

- geometry、channel 和 topology 可配置；
- 当前 AFHQ production recipe 是 128×128 pixel、patch 8、3 channels；
- fixed-variance 输出要求 `out_channels == in_channels`。

对 256×256、f8d4 latent：

```text
image: 3 × 256 × 256
latent: 4 × 32 × 32
patch size: 2
tokens: 16 × 16 = 256
```

现有 S topology 可以作为 DiT-S/2 bring-up。DiT-B/2 作为首个 reference
规模目标。官方 DiT 的 learned sigma 会输出 `2C` channels；当前 Stochaflow
只接受 `C` channels，因此首版显式使用 fixed variance。

## 5. Ownership architecture

### 5.1 四条 lifecycle

```text
Lifecycle A: external codec production

image corpus
  -> external Diffusers/Accelerate training project
  -> Diffusers AutoencoderKL directory or Hub revision


Lifecycle B: codec consumption

pinned codec source
  -> Stochaflow source resolver
  -> DiffusersAutoencoderKLCodec
  -> frozen managed auxiliary + inference asset projection


Lifecycle C: data preparation

Image DataArtifact
  + exact codec identity
  + preprocessing/encode/storage recipe
  -> managed posterior-moments DataArtifact


Lifecycle D: latent generation

image batch or posterior-moments batch
  -> normalized clean latent
  -> Process + denoiser + Objective
  -> checkpoint with codec binding
  -> latent Sampler
  -> codec decode
  -> image writer/evaluation
```

四者不能合并成一个 Builder。

### 5.2 责任矩阵

| Concern | Owner |
| --- | --- |
| VAE architecture implementation | Diffusers |
| public pretrained weight hosting/cache | Hugging Face Hub / Diffusers |
| external VAE dataset/loss/optimizer/discriminator | external training project |
| external VAE checkpoint/resume/distributed launch | external training project |
| exported VAE file format | Diffusers `save_pretrained` contract |
| codec source resolution and identity capture | Stochaflow codec provider |
| image range and posterior policy | Stochaflow codec adapter |
| latent affine normalization/inverse | Stochaflow codec adapter |
| codec device/eval/freeze lifecycle | TrainingBuilder + core Trainer |
| image partition/crop/flip/loader | DataBuilder |
| prepared posterior materialization | DataSource + DataArtifactStore |
| Process/Objective/DiT composition | latent TrainingBuilder |
| class dropout and batch interpretation | latent TrainingStrategy |
| CFG/model adapter/decode | latent SamplingBuilder |
| numerical transition | Gaussian Sampler |
| codec reconstruction evaluation | codec evaluation profile |
| generation metrics/artifacts | Evaluation framework |

### 5.3 Dependency direction

Core contracts不能导入 Diffusers：

```text
stochaflow core capability
        ^
stochaflow optional diffusers provider
        ^
diffusers dependency
```

`diffusers` 应作为可选 extra，不进入最小 core dependency。Intel macOS
已经是 deprecated/best-effort target；本 feature 不为其冻结旧版 Diffusers
或 Torch 组合。

## 6. Codec capability

### 6.1 最小公共 API

以下是 responsibility sketch，不是冻结签名：

```python
@dataclass(frozen=True, slots=True)
class ImageLatentSpec:
    image_channels: int
    latent_channels: int
    spatial_downsample_factor: int
    input_value_range: tuple[float, float]
    decoded_value_range: tuple[float, float]


class DiffusionImageEncoder(Protocol):
    @property
    def latent_spec(self) -> ImageLatentSpec: ...

    def encode_for_diffusion(
        self,
        images: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> Tensor: ...


class DiffusionImageDecoder(Protocol):
    @property
    def latent_spec(self) -> ImageLatentSpec: ...

    def decode_from_diffusion(self, latents: Tensor) -> Tensor: ...
```

prepared posterior 物化需要额外的窄 capability，不污染所有 encoder：

```python
@dataclass(frozen=True, slots=True)
class DiagonalGaussianMoments:
    mean: Tensor
    logvar: Tensor


class DiffusionPosteriorEncoder(Protocol):
    def encode_posterior_for_diffusion(
        self,
        images: Tensor,
    ) -> DiagonalGaussianMoments: ...
```

首个 Diffusers adapter 同时实现三个 capability。未来确定性 codec 可以只实现
sample encoder/decoder，不需要伪造 logvar。

### 6.2 不建立万能 representation hierarchy

首版不新增：

- `RepresentationModel`；
- universal `VAE` base class；
- `LatentProcess`；
- arbitrary encoder/decoder registry；
- universal latent batch schema；
- codec-aware Sampler；
- model-family enum。

core 只消费实际 collaboration 需要的窄 capability。Diffusers 类型检查只允许
出现在 optional adapter 内。

### 6.3 adapter 独占的语义

`DiffusersAutoencoderKLCodec` 必须独占：

- expected image channels；
- expected image value range；
- spatial divisibility；
- latent channels 和 downsample factor；
- posterior sample/mode；
- generator 使用；
- scaling factor；
- shift factor；
- latent mean/std；
- normalization 正向和严格逆向；
- decoder output extraction；
- input/output dtype；
- `force_upcast`；
- slicing/tiling 明确 policy；
- finite checks；
- resolved upstream class/config/weights identity。

以下层不得重复这些逻辑：

- DataBuilder；
- TrainingStrategy；
- Process；
- Objective；
- SamplingBuilder；
- EvaluationBuilder。

SamplingBuilder 只调用 `decode_from_diffusion()`；它不读取
`vae.config.scaling_factor`。

### 6.4 normalization 先收敛为 validated affine transform

首版 `AutoencoderKL` provider 不把多个 config 字段散落到训练和采样代码，
而是解析成一个 per-channel affine transform：

```text
z_model  = (z_native - center) * scale
z_native = z_model / scale + center
```

已知 mapping：

- classic SD：`center = 0`，`scale = scaling_factor`；
- shift-based codec：`center = shift_factor`，`scale = scaling_factor`；
- mean/std codec：`center = latents_mean`，
  `scale = scaling_factor / latents_std`。

如果某个 upstream family 同时提供多个存在歧义的字段，adapter 必须按已验证的
family semantics 解析或拒绝，不能猜顺序。

对 diagonal Gaussian posterior moments：

```text
mean_model   = (mean_native - center) * scale
logvar_model = logvar_native + 2 * log(abs(scale))
```

`scale` 必须 finite 且非零；per-channel broadcast shape 必须与 latent channels
完全一致。prepared posterior artifact 保存的是上述 normalized moments。

### 6.5 posterior policy

posterior policy 是 recipe identity：

- DiT training 默认 `posterior_sample`；
- deterministic reconstruction 使用 `posterior_mode`；
- prepared posterior artifact 保存 moments，并在 runtime 通过稳定 generator
  采样；
- 一个训练 recipe 不允许逐 batch 随意切换 sample/mode。

为了避免 adapter method 上出现任意字符串，首版可由构造参数固定 operational
policy；reconstruction 使用独立的 deterministic evaluation view。

### 6.6 precision policy

不能假设 `torch_dtype` 已解决全部 VAE precision 问题：

- `force_upcast` 是配置事实，但 standalone model 不等于完整 Pipeline；
- adapter 负责需要时将 VAE 和输入提升到 fp32；
- decode 后再恢复 writer 需要的 dtype/range；
- fp16-safe checkpoint 可以明确关闭 upcast；
- bf16/fp16/fp32 是 codec compatibility matrix，不由 Trainer precision
  字符串自动推导；
- prepared posterior production 的存储 dtype 与计算 dtype 分开记录。

## 7. Pretrained codec provider

### 7.1 首版 source 形式

支持两个 source：

```yaml
source:
  kind: hub
  repo_id: stabilityai/sd-vae-ft-mse
  revision: <immutable-commit-sha>
  subfolder: null
```

```yaml
source:
  kind: local_diffusers
  path: ./models/my-vae
  expected_weights_sha256: <sha256>
```

首版不支持：

- 未指定或未解析的浮动 `main` 用于 production；
- 任意 Python import path；
- 任意 `.ckpt` 猜 architecture；
- 自动扫描目录并挑选“最像 VAE”的权重；
- encoder/decoder 分别指定 source；
- URL 直接在 forward 中下载；
- credentials 写入 resolved config/checkpoint。

`from_single_file()` 可以作为未来 provider extension，但在配置推断、
conversion parity 和 digest contract 完成前不进入首版。

### 7.2 resolution

source resolver 在 Trainer 启动前完成：

1. 验证 source schema；
2. 对 Hub source 解析 immutable commit；
3. 下载或定位本地 snapshot；
4. 只接受 safetensors 或明确 allowlisted 文件；
5. 计算/验证 config 和 weights digest；
6. 构造 `AutoencoderKL`；
7. 构造 Stochaflow adapter；
8. 运行 cheap structural validation；
9. freeze 全部参数；
10. 将 resolved identity 交给 checkpoint/inference asset projection。

forward、resume 和 sampling 不得偷偷回到浮动 upstream revision。

### 7.3 首个模型选择

首个正式 baseline：

```text
stabilityai/sd-vae-ft-mse
AutoencoderKL
f=8
latent channels=4
256×256 -> 32×32×4
```

理由：

- 官方 DiT PyTorch 实现使用同一 SD VAE family；
- geometry 与当前 DiT-S/2、DiT-B/2 路线直接对应；
- 模型大小适合 4090 上做 online correctness；
- 离线 posterior preparation 后不占用 DiT training VRAM。

对照候选：

| Codec | 用途 | 状态 |
| --- | --- | --- |
| `stabilityai/sd-vae-ft-mse` | f8d4 baseline | 首版 |
| `madebyollin/sdxl-vae-fp16-fix` | fp16-safe/SDXL-style comparison | 第二 codec gate |
| `AutoencoderTiny` | preview/smoke decode | 不作为质量真值 |
| `AutoencoderDC` | 高压缩研究 | 新 latent geometry family |
| `AutoencoderRAE` | high-dimensional semantic latent | 新 representation family |

后四者不能因为 Diffusers 都能加载，就自动继承首版 compatibility。

### 7.4 provider version policy

optional extra 必须使用受控的 Diffusers compatibility range，而不是在每次
安装时追随任意最新版。resolved declaration/run evidence 至少记录：

- Stochaflow adapter name/version；
- Diffusers version；
- resolved upstream `_class_name`；
- config digest；
- weights digest。

升级 Diffusers 时运行固定 codec parity suite，覆盖 encode moments、sample/mode、
normalization、decode、precision 和 save/load。只有 parity 通过后才更新受支持
范围；不能因为 `from_pretrained()` 仍能返回对象就认为语义兼容。

## 8. External VAE training contract

### 8.1 当前 ownership

VAE training 作为 upstream project 运行。它可以使用：

- `diffusers.AutoencoderKL` 作为模型；
- Accelerate 或其他 launcher；
- 自选 reconstruction/KL/LPIPS/GAN loss；
- 自选 optimizer、scheduler、EMA；
- 自己的 checkpoint/resume；
- Hub publish 或本地 export。

Stochaflow 不调用该训练脚本，不解析其 optimizer state，也不承诺恢复其中断运行。

### 8.2 输出契约

外部 training project 必须输出一个可由
`AutoencoderKL.from_pretrained()` 加载的固定目录或 Hub revision，至少包含：

```text
config.json
diffusion_pytorch_model.safetensors
```

导出结果必须明确：

- architecture class；
- input/output channels；
- block topology；
- latent channels；
- spatial compression；
- scaling factor；
- optional shift factor；
- optional latents mean/std；
- `force_upcast`；
- weights digest；
- immutable revision 或本地 directory digest。

外部 project 还负责在代表性训练样本上校准并导出 latent statistics；不能把
Diffusers constructor 的默认 `scaling_factor` 当作新训练 codec 的实测统计。
Stochaflow 可以在 promotion gate 中复核统计和重建，但不会替 upstream training
workflow 猜测或静默改写 model config。

训练专用内容不得成为 inference codec 必需项：

- optimizer；
- scheduler；
- discriminator；
- LPIPS network；
- grad scaler；
- training RNG；
- dataloader state。

### 8.3 外部模型进入 Stochaflow 的 promotion gate

“能被 Diffusers 加载”不是足够条件。进入正式 DiT recipe 前必须通过：

1. config/schema validation；
2. weights digest verification；
3. encode/decode shape；
4. exact latent transform inverse；
5. deterministic posterior-mode reconstruction；
6. operational posterior reconstruction；
7. finite output；
8. expected image range；
9. representative-domain reconstruction report；
10. CPU/CUDA 和声明 dtype policy；
11. prepared posterior reproducibility；
12. decode 与 writer range compatibility。

Stochaflow 只验证消费契约和观察到的质量，不对外部训练 lineage 作通用
provenance 建模。metadata/provenance/capacity framework proposal 仍然 deferred。

### 8.4 对“Diffusers 负责训练”的准确文档表达

允许的表述：

> VAE training is an external workflow that may use Diffusers
> `AutoencoderKL` and Accelerate. Stochaflow consumes the exported,
> immutable Diffusers-format codec.

不允许的表述：

> Diffusers provides the Stochaflow VAE trainer.

除非未来 Diffusers 发布并稳定维护满足本计划完整 lifecycle 的公共 Trainer API，
否则不能把研究脚本当作 dependency contract。

## 9. Training asset 与 checkpoint

### 9.1 TrainingPlan projection

codec 在训练时仍是 managed auxiliary：

```text
TrainingPlan.primary_model = DiT
TrainingPlan.auxiliary_modules["codec"] = frozen codec
TrainingPlan.inference_assets["codec"] = projection to inference
```

计划新增的最小 projection 形态：

```python
@dataclass(frozen=True, slots=True)
class InferenceAssetProjection:
    training_asset_name: str
    declaration: ComponentConfig
    capability_role: str
```

它引用现有 auxiliary module，不复制 module owner。factory 把 projection
转换为当前已有的 `InferenceAssetDescriptor` 并交给 `CheckpointManager`。

首版 role 至少包含：

```text
diffusion_image_encoder_decoder
```

SamplingBuilder 在完整组合处再验证 decoder capability；core 不通过 role
字符串调用 task-specific method。

### 9.2 Phase A：embedded correctness

第一阶段沿用当前 checkpoint schema 的：

```text
persistence = embedded_state
```

需要闭合：

- checkpoint descriptor；
- `training_assets_state_dict["codec"]`；
- sampling checkpoint view；
- `InferenceAssetProvider`；
- declaration -> module construction；
- embedded state load；
- device/eval；
- decoder capability validation。

优点是 correctness 明确、离线采样简单。缺点是每个 checkpoint 可能复制约
数百 MB codec 权重。

Phase A 只用于：

- unit/integration tests；
- smoke run；
- 第一个端到端 latent checkpoint；
- reference-backed persistence 的行为基线。

### 9.3 Phase B：production asset bundle

在 DiT-B/2 多 checkpoint 训练前，必须完成 production persistence decision。
推荐候选是 run-level immutable asset bundle：

```text
<run>/
  assets/
    codec/<asset-digest>/
      config.json
      diffusion_pytorch_model.safetensors
      asset.json
  checkpoints/
    step-00010000.pt
    step-00020000.pt
```

checkpoint 保存：

- asset slot；
- capability role；
- provider declaration；
- exact asset digest；
- bundle-relative locator；
- optional original Hub repo/revision。

每个 checkpoint 不重复 VAE weights；完整 run bundle 可离线复制。
该 bundle 是 model/inference asset persistence，不通过 `DataArtifactStore`，
也不获得 `DataArtifactIdentity`。

在以下问题得到验证前，不把 reference persistence 做成通用公共模型仓库：

- no-replace publication；
- bundle relocation；
- strict digest validation；
- partial copy failure；
- remote Hub fallback 是否允许；
- checkpoint retention 与 asset GC；
- Windows/macOS path semantics；
- external extension provider 的替换性。

### 9.4 sampling 不重复 codec config

训练配置声明 source 一次。resolved checkpoint/bundle 已拥有：

- provider declaration；
- immutable revision/digest；
- latent spec；
- operational posterior policy；
- transform identity；
- decoder role。

因此 sampling overlay 只允许表达：

- sampler numerical params；
- guidance；
- class/condition allocation；
- sample count/seed；
- writer params；
- output/trajectory policy。

不得要求用户再次填写 `repo_id`、scaling factor 或 VAE class。overlay
试图替换 codec 时 fail closed；codec ablation 是新训练/evaluation subject，
不是普通 sampling 参数。

## 10. Data 与 prepared posterior artifact

### 10.1 on-the-fly encode

第一条 correctness path：

```text
ClassLabeledImageDataset batch
  -> frozen codec encode
  -> normalized posterior sample
  -> Gaussian Process
  -> DiT
```

它用于：

- contract tests；
- tiny smoke；
- reconstruction diagnostics；
- 验证 image preprocessing；
- 需要强随机 crop 的实验。

它不是正式大规模训练的默认，因为每个 epoch 重复运行 VAE encoder。

### 10.2 production path：posterior moments artifact

正式固定-codec DiT training 默认物化：

```text
mean
logvar
class/condition
stable sample key
partition
optional fixed view id
```

而不是只保存一次随机 sample。runtime 根据：

```text
experiment seed + sample key + epoch/view policy
```

构造稳定 generator，从 moments 采样。这样保留 stochastic posterior
语义，同时不重复 VAE encode。

首版 moments 明确为 diffusion-normalized diagonal Gaussian parameters；
artifact identity 绑定 exact transform。若未来 codec 使用非 affine 或非
diagonal posterior，新增它自己的 payload/recipe，不扩展成大量 optional 字段。

### 10.3 artifact identity

managed posterior artifact 必须绑定：

- upstream image `DataArtifactIdentity`；
- codec provider；
- codec resolved revision；
- codec config digest；
- codec weights digest；
- input image size/range；
- resize/crop/interpolation/antialias；
- augmentation/view policy；
- posterior representation；
- latent transform；
- latent shape/channels；
- compute dtype；
- storage dtype；
- layout；
- shard schema；
- class mapping/condition schema；
- materializer version。

特别要求：

> `DataArtifactStore` 的 `locator_key` 必须包含 upstream artifact digest、
> codec weights digest 和 encode/storage recipe digest。

否则 locator 在调用 build 前命中旧对象时，更换 VAE 或 preprocessing
可能静默复用错误 latent。

### 10.4 storage

首版采用：

- fixed-shape mmap-friendly shards；
- 每 shard 约 256 MB–1 GB；
- FP16/BF16 storage 由 profile 固定；
- stable sample index；
- 不为每张图创建单独文件；
- 不在 DataLoader worker 中写 cache；
- 不建立 persistent read-through hybrid cache。

对 256×256 f8d4、FP16：

- 单个 latent sample 约 8 KB；
- mean + logvar 约 16 KB/image；
- 500k images 约 8 GB；
- 再预计算 horizontal-flip view 约 16 GB。

实际 artifact size 必须以 materialized inventory 为准，不能把估算写成 guarantee。

### 10.5 augmentation boundary

prepared artifact 不应伪装成支持任意 image augmentation：

- deterministic resize/center crop 可以进入 preparation；
- original + horizontal flip 可以作为两个固定 views；
- arbitrary random crop/color augmentation 需要 online encode；
- fixed multi-view bank 是独立 recipe；
- 不能先 precompute 一个 crop，再在文档中声称保留原 image augmentation。

### 10.6 interruption

最终 posterior artifact 保持 immutable/no-partial-publication。若大数据 preparation
的 staging cleanup 导致昂贵重算，允许 producer-private resumable shard scratch：

- scratch key 必须包含完整 materialization identity；
- 每 shard 有独立 digest；
- final artifact 仍由 `DataArtifactStore` 原子发布；
- partial scratch 不能作为 DataArtifact 被 Builder 消费；
- 不把 resumable scratch 提升为通用 artifact API。

## 11. Latent training composition

### 11.1 recipe

```text
DataBuilder
  -> image batch or posterior-moments batch
TrainingBuilder
  -> denoiser primary
  -> frozen codec auxiliary/decoder asset
  -> Gaussian Process
  -> Objective
  -> concrete latent Strategy
Trainer
  -> automatic single-optimizer loop
```

TrainingBuilder 负责：

- 解析 codec source；
- 构造/freeze adapter；
- 验证 latent spec 和 denoiser geometry；
- 验证 DataBuilder output contract；
- 声明 inference asset；
- 固定 prediction type；
- 固定 concrete condition/dropout policy；
- 固定 `clip_denoised=false`；
- 产生 SamplingRecipe。

TrainingStrategy 只负责：

- 解释 batch；
- 获取 clean normalized latent；
- concrete condition/null condition；
- condition dropout；
- timestep/noise；
- 调用 denoiser/Objective；
- 返回 scalar loss 和 metrics。

Strategy 不负责：

- 构造或下载 codec；
- 冻结参数；
- 选择 optimizer parameters；
- 保存 codec；
- 写 prepared artifacts。

### 11.2 image-backed 与 prepared-backed

不为每个 dataset 创建 Builder。至少存在两种 semantic recipe：

1. image-backed latent recipe；
2. diagonal-posterior-backed latent recipe。

它们可以共享 Process/Objective/denoiser capability，但 batch interpretation 不应通过
一个含大量 optional 字段的万能 schema 隐式猜测。

具体是复用一个 Strategy capability 还是两个 Strategy，由实现时的第二个真实
batch contract 决定；core runner 不增加分支。

### 11.3 prediction

首版冻结：

- Gaussian Process；
- epsilon prediction 作为 parity baseline；
- fixed variance；
- constant/unweighted simple MSE；P2 capability 即使已经在 pixel Gaussian recipe 中
  实现，也不自动成为 latent correctness default；
- `clip_denoised=false`；
- concrete condition dropout 由 Strategy 管理；
- model-internal dropout 关闭；
- CFG 由 SamplingBuilder 管理。

`v` prediction 可以作为受控扩展，但不能因为 pixel AFHQ recipe 可用就自动成为
latent default。

P2 weighting 是 parameterization-dependent 的 Gaussian training policy。只有在
latent epsilon baseline、codec reconstruction 和 decoded evaluation protocol稳定后，
才能以相同 topology/data/budget 做 `constant`/`p2` A/B；pixel AFHQ 的收益不能直接
推广到 latent DiT，更不能把 P2 参数放进 codec 或 Process 配置。

learned variance 后续需要：

- `2C` structured output；
- mean/variance partition contract；
- variance objective；
- Gaussian transition changes；
- sampling compatibility tests。

在这些能力完成前，首个 reference DiT output 保持 `C` channels。

### 11.4 step-based production training

正式 Met/ImageNet-100/DomainNet/Stable Diffusion run 不能只依赖
epoch-oriented UX。进入长训练前
必须确认或补齐：

- `max_train_steps`；
- global-step checkpoint cadence；
- global-step diagnostics cadence；
- global-step evaluation cadence；
- exact resumed batch/epoch semantics；
- local file logger；
- `--no-progress` 可观察性；
- tmux/SSH 断开不影响日志；
- SIGTERM/controlled stop 行为；
- checkpoint completion marker。

这属于严肃训练的 operational prerequisite，不属于 VAE/latent abstraction。

## 12. Dataset 与硬件验证路径

### 12.1 AFHQ-v2

AFHQ-v2 只承担：

- online encode smoke；
- class label/CFG contract；
- reconstruction report；
- checkpoint -> decode；
- resume；
- writer/logging。

它不承担：

- 数据规模证明；
- DiT-B/2 最终质量；
- benchmark promotion；
- pretrained codec 泛化结论。

### 12.2 LHQ quality probe

LHQ 是 90k high-resolution landscape images 的低熵质量候选。它适合：

- 256×256 frozen-codec reconstruction 和 latent throughput profiling；
- unconditional DiT-S/2 quality probe；
- 检查小于 ImageNet-1K 的数据规模能否产生可见、非 toy 的生成结果；
- 验证 filtering quality，而不是只比较 raw sample count。

它不是当前默认公开 showcase。原始集合来自 Unsplash/Flickr，公开镜像的
redistribution 和 ML-use license 必须独立审计；不得仅凭镜像仓库的 license
标签作出正式声明。若 license/source snapshot 不能冻结：

- 它只能作为私有 research profile；
- 不进入 CI、release artifact 或可重新分发的 tutorial；
- 不能成为公开 promotion gate。

参考：
[LHQ paper](https://arxiv.org/abs/2104.06954)、
[Unsplash Dataset](https://unsplash.com/data)。

### 12.3 The Met Open Access curated snapshot

首个开放正式 showcase 候选改为 The Met Open Access。上游提供 public-domain
high-resolution images 和 CC0 metadata/API；Stochaflow 不直接把整个在线集合
视为稳定训练集，而是通过 DataSource 物化一个冻结 snapshot。

profiling gate 必须先统计并决定：

- `isPublicDomain`、`primaryImage` 和最低 native resolution；
- extreme aspect ratio、边框、文档扫描、重复 primary/additional views；
- department、object type、medium、culture、period/date 和 tag 的缺失率；
- long-tail taxonomy、每个 condition bucket 的有效样本数；
- deterministic metadata-derived caption 的覆盖率；
- 下载失败、内容漂移和 immutable source inventory；
- 最终规模是否落在 150k–300k，而不是为了命中名字强凑 200k。

冻结后的 artifact 暂称 `met-open-curated-v1`，但精确名称、数量、taxonomy 和
split 只有在 profiling report 通过评审后才进入 production config。

它服务两条计划内路线：

1. DiT：先使用一个冻结的 coarse condition axis，例如 department/object family；
2. Stable Diffusion：使用确定性 metadata caption；VLM recaption 若进入，必须产生
   不同 materialization identity 和独立质量审计；具体 text contract 由
   [Stable Diffusion 计划](stable-diffusion-component-native-support-plan.md)
   拥有。

参考：
[The Met Open Access](https://www.metmuseum.org/hubs/open-access)、
[Collection API](https://metmuseum.github.io/)。

### 12.4 ImageNet-100 benchmark

ImageNet-100 保留为标准 class-conditional 对照，不再作为唯一或首个开放
showcase。正式 256×256 profile 必须：

- 从受访问条款约束的原始分辨率 ImageNet-1K snapshot 选择固定 synset；
- 冻结 class list、split、source identity 和 access/license decision；
- 使用有足够 native resolution 的原图；
- 训练约 100 classes，而不是把 ImageNet-1K 全部纳入正式 run。

[clane9/imagenet-100](https://huggingface.co/datasets/clane9/imagenet-100)
已有约 126k train images，但较短边已缩至约 160 pixels。它只能用于
128/160 bring-up，不能通过上采样冒充 256×256 formal source。

ImageNet-100 的价值仍是 vanilla class-conditioned DiT benchmark；它不承担
开放数据许可、text conditioning 或 framework 数据策展能力的证明。

### 12.5 DomainNet extension

DomainNet 保留为更大规模和复合 condition 的扩展目标：

- 586,575 total images；
- 345 semantic classes；
- 6 domains；
- class + domain condition；
- domain-balanced/empirical sampling；
- cross-domain evaluation。

候选：
[wltjr1007/DomainNet](https://huggingface.co/datasets/wltjr1007/DomainNet)。

DomainNet 不能直接塞进只理解单 class label 的 recipe。它是验证第二个
condition family 和更大数据吞吐的 decision gate，不取代 Stable Diffusion
text condition 轨道。

### 12.6 Stable Diffusion dataset handoff

共享的 image artifact 可以派生 image-text artifact，但 tokenizer、caption
normalization、VLM recaption、prompt split 和 text evaluation 不属于本计划。
它们由
[Stable Diffusion Component-Native 支持计划](stable-diffusion-component-native-support-plan.md)
独立冻结，不能通过给本计划的 class/domain recipe 添加 nullable prompt 实现。

### 12.7 暂不选择

- Flowers102：退出 active example/recipe 维护，不作为 latent target；
- full ImageNet-1K：首轮成本过高；只允许固定原图子集作为 benchmark；
- DiT-XL/2：不作为 4090/Spark 首个规模；
- 无明确 redistribution/license 的 scraped dataset：不进入正式 showcase。

### 12.8 4090 与 DGX Spark

硬件角色通过实测决定：

- RTX 4090 24 GB：优先作为吞吐 baseline；
- DGX Spark 128 GB unified memory：优先验证更大 batch/model capacity；
- unified memory 容量不等于训练吞吐；
- 在两个设备上各跑固定 1k optimizer steps；
- 比较 images/s、optimizer steps/s、peak memory、data wait、checkpoint time；
- 决定 DiT-B/2 batch、gradient accumulation 和 compile policy；
- 不凭设备名称预先选择 production host。

Stable Diffusion 的 512 UNet hardware profile 由其独立计划定义，但应复用同一
benchmark/result infrastructure。

## 13. Sampling 与 Evaluation

### 13.1 sampling composition

```text
checkpoint
  -> primary model provider
  -> codec inference asset provider
  -> latent SamplingBuilder
  -> class condition + CFG
  -> Gaussian Dynamics
  -> Sampler
  -> normalized latent
  -> codec.decode_from_diffusion
  -> writer-ready image
```

SamplingBuilder 负责：

- latent initial shape；
- condition allocation；
- null condition；
- CFG；
- model adapter；
- Process/Dynamics/Sampler compatibility；
- decoder capability；
- bounded-batch decode；
- output range；
- optional decoded observations。

Sampler 只处理 latent tensor 和 narrow Gaussian Dynamics。

### 13.2 sampling config

checkpoint-backed sampling 的用户输入不出现 VAE declaration：

```yaml
sample:
  sampler:
    name: ddim
    params:
      num_inference_steps: 50
      eta: 0.0
  options:
    weights: ema
    guidance_scale: 4.0
    label_policy: balanced
  num_samples: 100
  batch_size: 25
  seed: 42
  writers:
    - name: image
      params:
        grid_nrow: 5
        denormalize: true
```

实际 SamplingBuilder/recipe 从 checkpoint 固定；用户不需要理解或选择
一个叫 `builder` 的低层字段。

### 13.3 trajectory

数值 trajectory 是 latent space：

- 默认只 decode final state；
- 可选少量 observation steps；
- 不对所有 solver steps 默认 decode；
- raw latent trajectory 使用 tensor writer；
- image writer 只接受 decoded image；
- manifest 区分 solver latent steps 和 decoded artifact steps。

### 13.4 reconstruction gate

DiT 长训练前必须先产生固定 reconstruction report：

- exact reference sample IDs；
- posterior-mode reconstruction；
- operational-posterior reconstruction；
- PSNR；
- SSIM；
- LPIPS；
- reconstruction FID/KID（样本数足够时）；
- non-finite/failed sample count；
- class/domain slice；
- fixed visual panel；
- codec/source/digest/profile identity。

指标阈值在选定 dataset、codec 和 reference implementation 后冻结，不在抽象计划中
虚构通用数字。

### 13.5 generation evaluation

至少区分：

- smoke sample grid；
- fixed-seed diagnostic sample；
- checkpoint selection metric；
- final held-out report；
- uniform-class capability；
- empirical-prior distribution quality。

Evaluation 复用已验证 sampling capability，不重新实现 CFG、latent shape、
decode 或 preprocessing。

## 14. Configuration sketch

以下只表达 ownership，不冻结完整 schema。

### 14.1 image-backed training

```yaml
data:
  name: class_labeled_image
  params:
    source:
      name: imagenet_100
      params:
        revision: <immutable-dataset-revision>
    image:
      size: [256, 256]
      crop: center_square
      interpolation: bicubic
      antialias: true
      output_value_range: [-1.0, 1.0]

model:
  name: dit
  params:
    input_size: 32
    patch_size: 2
    in_channels: 4
    out_channels: 4
    hidden_size: 384
    depth: 12
    num_heads: 6
    num_classes: 100

training:
  name: class_conditional_latent_gaussian
  params:
    prediction_type: epsilon
    class_dropout_probability: 0.1
    codec:
      name: diffusers_autoencoder_kl
      params:
        source:
          kind: hub
          repo_id: stabilityai/sd-vae-ft-mse
          revision: <immutable-commit-sha>
        encoding_policy: posterior_sample
```

codec 是具体 TrainingBuilder 私有 composition，不新增所有 task 都必须填写的
顶层 `codec:`。

### 14.2 prepared-posterior training

```yaml
data:
  name: class_labeled_posterior_moments
  params:
    source:
      name: prepared_imagenet_100_posterior
      params:
        expected_artifact_digest: <artifact-digest>

model:
  name: dit
  params:
    input_size: 32
    patch_size: 2
    in_channels: 4
    out_channels: 4
    hidden_size: 768
    depth: 12
    num_heads: 12
    num_classes: 100

training:
  name: class_conditional_prepared_latent_gaussian
  params:
    prediction_type: epsilon
    class_dropout_probability: 0.1
    decoder:
      name: diffusers_autoencoder_kl
      params:
        source:
          kind: hub
          repo_id: stabilityai/sd-vae-ft-mse
          revision: <same-immutable-commit-sha>
```

Builder 必须验证 posterior artifact 的 codec/transform digest 与 decoder asset
完全一致。最终 resolved checkpoint 保存 declaration；sampling 不再重复它。

### 14.3 local externally trained codec

```yaml
training:
  name: class_conditional_latent_gaussian
  params:
    codec:
      name: diffusers_autoencoder_kl
      params:
        source:
          kind: local_diffusers
          path: ./models/domain-vae
          expected_weights_sha256: <sha256>
        encoding_policy: posterior_sample
```

本地 path 只用于初始解析；resolved identity 和 production bundle 不依赖用户
之后仍从同一 cwd 启动。

## 15. 实施阶段

### Phase 0：修正计划与依赖边界

交付：

- 先完成 Hydra 迁移计划 C0/C1 的 plain Train/Sample authority cutover；不等待
  Hydra H0–H4；
- 随后完成 Metrics 计划 M0–M1，冻结 canonical epoch result 与 monitor contract；
- 本文档冻结；
- public docs 不宣称 VAE training support；
- development docs 区分 pretrained codec、external training、
  native training 和 joint training；
- optional dependency 策略冻结；
- Met/LHQ/ImageNet-100/DomainNet dataset decision record 补齐；
- Stable Diffusion 特有能力由独立计划拥有。

验收：

- repository-wide 搜索无“Diffusers 提供通用 VAE Trainer”；
- Flowers 不再被标为正式规模 target；
- metadata/provenance/capacity proposal 保持 deferred；
- 本 phase 不修改 runtime。

### Phase 1：闭合 inference asset projection（已完成）

状态（2026-07-30）：embedded-state 基础设施已通过 knowledge-distillation reference
extension 的独立 `LogitCalibrator` vertical slice 闭合。该验收覆盖 fresh train、
strict resume、删除 bootstrap、断网、非项目 cwd 的 checkpoint-only sampling；不表示
Diffusers codec、latent workflow、asset bundle 或 latent diagnostic 已实现。

实现：

- `TrainingPlan.inference_assets`；
- projection validation；
- factory -> `CheckpointManager` descriptors；
- sampling view 保留 descriptors/required embedded state；
- `InferenceAssetProvider`；
- slot/capability validation；
- requested-only asset loading；
- embedded asset descriptor 区分 acquisition identity、self-contained reconstruction
  declaration 和 state；reconstruction 不依赖 Hub cache 或原 local directory。

验收：

- 独立 fake asset 证明 extension path；
- descriptor round-trip；
- missing/wrong slot fail closed；
- training-only teacher 不被 sampling 自动加载；
- current pixel recipes descriptor 为空且行为不变；
- 删除原始 provider path 并断网后，fake embedded asset 仍可从 checkpoint 构造。

实现约束：

- Phase 1 只支持通过 model Registry 重建的 `nn.Module`，没有新增 asset registry；
- sampling projection 只保留 descriptor 引用的 embedded state，teacher、Objective、
  optimizer 和其他 training-only auxiliary 不进入 view；
- requested-only 只承诺 module 构造和 state load 延迟，checkpoint 文件仍整体读取；
- diagnostics latent behavior 留到出现真实 latent capability 的后续切片。

### Phase 2：Diffusers codec provider

实现：

- optional Diffusers extra；
- Hub/local source schema；
- pinned revision resolution；
- safetensors/config digest；
- `DiffusersAutoencoderKLCodec`；
- posterior sample/mode；
- latent transform；
- force-upcast/precision；
- freeze/eval；
- reconstruction profile。

验收：

- `stabilityai/sd-vae-ft-mse` 256×256 -> 32×32×4；
- sample generator replay；
- mode deterministic；
- exact transform inverse；
- no duplicated normalization；
- no network access in forward；
- offline snapshot load；
- wrong digest/config/range/geometry fail closed；
- independent fake codec passes public capability tests。

### Phase 3：AFHQ end-to-end correctness

实现：

- AFHQ image-backed latent smoke config；
- DiT-S/2 tiny run；
- local logger + no-progress command；
- embedded codec checkpoint；
- resume；
- independent latent sampling/decode；
- reconstruction report。

验收：

- uninterrupted/resumed tiny run compatibility；
- sampling config 不重复 codec；
- writer 只接收 decoded image；
- pixel clipping 明确关闭；
- AFHQ 文档明确是 correctness showcase。

### Phase 4A：prepared posterior artifact

实现：

- posterior moments payload/source；
- managed artifact producer；
- stable sample key；
- sharded mmap storage；
- original/flip fixed-view policy；
- class mapping；
- locator identity；
- deterministic runtime sampling；
- strict artifact binding。

验收：

- online/offline moments parity tolerance；
- codec/preprocessing/dtype digest mismatch fail closed；
- same-size mutated shard full verification 失败；
- interrupted preparation 不发布 partial artifact；
- Builder 在 Dataset 构造前验证 binding；
- changing codec revision cannot hit old locator。

### Phase 4B：optimizer-step production lifecycle

在任何正式长训练前完成：

- `max_train_steps`；
- optimizer-step checkpoint、log 和 diagnostic cadence；
- mid-epoch resume policy；
- controlled stop/SIGTERM；
- completion marker；
- epoch 与 optimizer-step budget 的明确 precedence。

验收：

- interrupted/resumed optimizer-step sequence 与 documented policy 一致；
- checkpoint cadence 不依赖 dataset epoch 长度；
- `--no-progress` local log 显示 step、throughput、checkpoint 和 stop 状态；
- 本 phase 若证明需要新的 training-loop family，先更新独立架构决策，不通过
  nullable flags 扩大现有 loop。

### Phase 4C：production asset persistence

在第一次正式多 checkpoint 训练前完成：

- embedded duplication measurement；
- run-level asset bundle prototype；
- relocation/offline tests；
- digest validation；
- checkpoint retention/asset retention policy；
- public vs private persistence API decision。

验收：

- 多 checkpoint 不重复 codec weights；
- copied run bundle can sample offline；
- missing/corrupt asset fails before sampling；
- Hub latest revision cannot silently substitute；
- no DataArtifact/model-asset type confusion。

### Phase 5：开放数据 profiling 与 DiT-S/2 bring-up

实现：

- The Met Open Access profiling report；
- frozen curated snapshot/coarse condition mapping；
- LHQ private quality probe 仅在 license gate 允许时运行；
- smoke/production configs；
- fixed diagnostics；
- reconstruction gate；
- uniform/empirical sampling plans；
- initial generation metrics。

验收：

- 1k-step 4090/Spark throughput report；
- no VAE encoder cost in prepared training loop；
- checkpoint/resume/local logging；
- 每个冻结 condition bucket 有固定 sample allocation；
- result manifests bind dataset/codec/artifact/checkpoint。

### Phase 6：开放正式 DiT-B/2

实现：

- DiT-B/2 config；
- frozen `met-open-curated-v1` profile；
- batch/accumulation based on hardware benchmark；
- EMA；
- long-run checkpoint cadence；
- fixed evaluation cadence；
- reproducible sample suite；
- S/2 vs B/2 controlled comparison。

验收：

- training can be paused/resumed across hosts under documented constraints；
- no codec/source fields repeated in sampling；
- final report includes reconstruction ceiling and generation quality；
- no claim extrapolates to ImageNet-1K or general web-scale generation。

### Phase 7：ImageNet-100 与 DomainNet benchmark gates

实现前冻结：

- original-resolution ImageNet-100 source/revision/access/class mapping；
- DomainNet revision/license/class/domain mapping；
- class + domain condition capability；
- domain-balanced sampling；
- prepared artifact size/shard plan；
- domain-specific reconstruction report；
- output evaluation protocol。

验收：

- 新 condition family 不修改 Process/Sampler root；
- 不把 domain 塞进 universal batch schema；
- same codec contract reused without Diffusers-specific core branch；
- ImageNet-100 160-pixel mirror 不进入 256 formal profile；
- second/third real datasets validate DataSource/DataBuilder boundary。

### Phase 8：independent non-DiT denoiser substitution

实现：

- independent extension-owned convolutional or compact UNet-like denoiser；
- 仅依赖现有 Gaussian denoiser capability；
- 复用 image-backed 与 prepared-posterior recipes；
- 复用 codec binding、checkpoint、DDPM/DDIM、decode 和 Evaluation；
- 不通过 concrete-type check、registered name 或 DiT topology 分支接入。

验收：

- core、Process、Sampler 和 codec adapter 无修改；
- Builder 在完整 composition boundary 验证 geometry/prediction compatibility；
- checkpoint/resume/sampling 与 DiT reference path 具有相同 lifecycle guarantee；
- 证明 `Latent Diffusion` 是 framework workflow，而不是 DiT 的别名。

## 16. 测试矩阵

### 16.1 codec contract

- input/output channels；
- divisible/non-divisible geometry；
- image range；
- latent shape；
- scaling only；
- scaling + shift；
- mean/std transform；
- exact inverse；
- posterior sample replay；
- posterior mode deterministic；
- force-upcast；
- fp32/bf16/fp16 policy；
- slicing/tiling explicit policy；
- non-finite input/output；
- custom non-Diffusers implementation。

### 16.2 pretrained source

- Hub pinned commit；
- Hub moving revision rejected/resolved according to profile；
- local directory digest；
- missing config；
- missing safetensors；
- wrong `_class_name`；
- wrong weights digest；
- subfolder；
- offline cache；
- no credentials serialized；
- no network call during forward；
- encoder/decoder cannot use different sources。

### 16.3 managed/inference asset

- freeze；
- eval mode；
- optimizer exclusion；
- EMA exclusion；
- descriptor projection；
- embedded state；
- strict resume；
- sampling view；
- requested-only load；
- wrong role/slot；
- asset state mismatch；
- training-only auxiliary exclusion；
- bundle relocation and corruption（Phase 4C）。

### 16.4 prepared artifact

- identity fields；
- locator key；
- source artifact mismatch；
- codec digest mismatch；
- transform mismatch；
- preprocessing mismatch；
- dtype/layout/shard mismatch；
- class mapping；
- stable sample key；
- deterministic posterior sample；
- shard inventory；
- final-root payload；
- staging cleanup/resume scratch。

### 16.5 training

- image-backed batch；
- moments-backed batch；
- class dropout 0/1/intermediate；
- epsilon prediction；
- no latent clipping；
- loss scalar/finite；
- DiT geometry；
- S/2/B/2；
- resumed RNG；
- max steps/checkpoint cadence；
- independent non-DiT conditional denoiser；
- 不读取 DiT-specific fields 的 capability substitution。

### 16.6 sampling

- checkpoint automatically resolves codec；
- config omits codec；
- latent initial shape；
- CFG 1 baseline；
- balanced/empirical labels；
- raw/EMA；
- DDPM/DDIM；
- final decode；
- bounded decode batch；
- decoded observation selection；
- latent tensor writer；
- image writer range；
- missing codec fail before numerical sampling。

### 16.7 evaluation

- reconstruction mode/operational profiles不混用；
- fixed sample IDs；
- exact codec identity；
- KID/FID protocol；
- uniform/empirical class prior；
- intended-class fidelity；
- incomplete generation fail closed；
- checkpoint selection 与 final report 分离；
- AFHQ smoke 不冒充规模结果；
- ImageNet-100 不冒充 ImageNet-1K。

## 17. 完成标准

### 17.1 architecture

- core 未导入 Diffusers；
- 未新增 `LatentProcess`；
- 未新增 universal VAE/representation base；
- primary model 是 concrete denoiser；
- codec 是 frozen managed auxiliary；
- TrainingPlan 显式投影 inference asset；
- Strategy 不构造/freeze/serialize codec；
- SamplingBuilder 拥有 decode；
- Sampler 不理解 codec/class/image；
- external VAE training 不进入 Stochaflow Trainer；
- independent non-DiT denoiser 可复用同一 codec、Process、Sampler、artifact
  和 checkpoint lifecycle。

### 17.2 reproducibility

- codec Hub revision 解析为 immutable identity；
- config/weights digest 固定；
- posterior policy 固定；
- latent transform 固定；
- prepared artifact 绑定 upstream image + codec + recipe；
- checkpoint 可恢复 decoder；
- sampling 不依赖重复用户配置；
- production run bundle 可离线复制和采样。

### 17.3 UX

- 用户可以选择公开 Hub VAE 或本地 Diffusers VAE；
- 用户不需要理解 encoder/decoder 拼装；
- 用户不需要手写 scaling factor；
- 用户不在 sampling config 重复 VAE；
- error 指向 source、digest、range、geometry 或 capability；
- long run 有本地日志和 no-progress 可观察性；
- external VAE export 有明确接入 checklist。

### 17.4 quality

- codec reconstruction gate 先于 latent denoiser 长训练；
- AFHQ 只报告 correctness；
- Met curated snapshot 有固定开放 protocol；
- ImageNet-100 仅在原始分辨率 source 冻结后作为标准对照；
- DiT-S/2 与 B/2 结论来自控制实验；
- DomainNet 在单独 condition gate 后进入；
- AutoencoderTiny 不作为质量 reference；
- codec 质量限制与 denoiser 质量分开报告。

## 18. Future decision gates

### 18.1 Stochaflow-native VAE training

默认继续 deferred。只有满足以下至少两个真实信号才重新提案：

1. 两个以上外部 VAE training workflow 需要与 Stochaflow 数据/日志/恢复深度集成；
2. external export 多次造成无法通过 contract gate 的语义漂移；
3. alternating multi-optimizer loop 已被另一个非 VAE 方法需要；
4. 用户明确需要在同一实验系统比较 VAE training recipes；
5. Diffusers 不再能表达所需 codec architecture；
6. codec checkpoint 与 downstream latent denoiser 联合 lineage 成为正式产品要求。

若触发，仍需拆成两种 loop family：

- reconstruction + KL/perceptual 的单优化器 automatic loop；
- autoencoder/discriminator 的 alternating multi-optimizer loop。

不能给现有 `TrainingStrategy` 添加 optional discriminator hooks。

### 18.2 external VAE trainer provider

如果 Diffusers 未来发布稳定的公共 VAE Trainer API，单独评估：

- loss contract；
- optimizer count；
- checkpoint schema；
- resume guarantees；
- distributed semantics；
- mixed precision；
- model export；
- API versioning。

只有全部满足后，才允许一个 external training operation 调用它。即便如此，
它仍不自动成为 Stochaflow core TrainingBuilder。

### 18.3 joint VAE + latent denoiser

joint training 是新方法 family：

- codec 不再 frozen；
- prepared posterior artifact 立即失效；
- latent distribution/normalization 会漂移；
- optimizer topology 改变；
- reconstruction、regularization、diffusion loss 共同变化；
- sampling checkpoint 必须绑定同步 codec state。

不能通过：

```yaml
joint_training: true
```

打开。需要独立 research proposal、loop family 和 evaluation protocol。

### 18.4 new codec families

`AutoencoderDC`、`AutoencoderRAE`、VQ、video codec 分别经过：

- latent geometry；
- posterior kind；
- normalization；
- denoiser input channel/spatial or token geometry；
- decode range；
- precision；
- prepared artifact representation；
- quality protocol。

第二个 codec 复用同一 capability 后，再判断是否提升公共 abstraction。

### 18.5 learned variance

P2/ADM 计划 A1 完成后，framework 将具备 `2C` prediction、learned-range variance、
hybrid objective 和 transition tests；latent recipe仍需在自己的 TrainingBuilder /
SamplingBuilder 边界验证 codec latent channels、DiT output layout 与 no-clipping
语义，不能仅因 core capability存在就自动宣称 official DiT learned sigma。

learned variance不阻塞 fixed-variance latent correctness pipeline，也不作为首个
DiT-S/2 config的默认值；通过独立 parity/evaluation gate后再提升。

### 18.6 full Diffusers Pipeline

black-box pipeline、component-native SD 1.x 和 SDXL/SD3 的 ownership/parity
由
[Stable Diffusion Component-Native 支持计划](stable-diffusion-component-native-support-plan.md)
独立冻结。本计划只提供它们复用的 codec、normalized latent、posterior artifact
和 model-asset 前置能力。

### 18.7 text conditioning

本计划先以 concrete class/domain recipe 验证 conditional latent workflow。
CLIP/tokenizer、prompt encoding、cross-attention 和 caption artifacts 属于
Stable Diffusion 计划；不通过给 class-conditioned Builder 添加 nullable prompt
字段进入。

### 18.8 pretrained asset abstraction

首版 codec provider 可以使用 provider-private source schema。只有 teacher、
codec、text encoder 等至少两个真实 provider 展现相同 resolution/pinning/bundle
lifecycle 后，才提炼公共 `PretrainedModuleReference`。

不能为了未来模型仓库预先建立 arbitrary asset registry。

## 19. 风险与缓解

| 风险 | 后果 | 缓解 |
| --- | --- | --- |
| 把 Diffusers model 当完整 Trainer | 外部脚本恢复/损失语义被误承诺 | 文档明确 model provider 与 trainer ownership |
| 硬编码 `0.18215` | train/sample latent 不一致 | adapter 独占 transform |
| standalone VAE 忽略 `force_upcast` | fp16 NaN/颜色异常 | adapter precision contract |
| encoder/decoder revision 不同 | latent 无法正确 decode | 单一 codec asset |
| sampling 重复配置 codec | checkpoint 与 overlay 漂移 | checkpoint-owned asset |
| 每个 checkpoint embedding VAE | 磁盘膨胀 | Phase A 限于 correctness；Phase 4C bundle |
| Hub `main` 漂移 | resume/sample 不可重放 | immutable commit + digest |
| prepared locator 缺 codec digest | 命中旧 latent | 完整 locator/materialization identity |
| 只保存一次 latent sample | posterior 随机性永久冻结 | moments artifact + stable runtime RNG |
| 随机 crop 被 precompute 抹掉 | augmentation claim 不真实 | online/fixed-view recipes 分离 |
| VAE reconstruction 差 | denoiser 无法补救系统性损失 | reconstruction promotion gate |
| AFHQ 太小 | 视觉结果不能代表规模 | Met formal target + LHQ quality probe |
| ImageNet-1K 太早 | 训练成本和调试时间失控 | 仅构造原图 ImageNet-100 对照 |
| reference path 依赖 DiT fields | Latent Diffusion 退化为 DiT feature | fake capability contract 先保护边界；independent non-DiT 作为后续 promotion gate |
| Spark 容量被当吞吐 | 训练计划错误 | 1k-step cross-device benchmark |
| Diffusers 类型泄漏到 core | extension boundary 被锁死 | optional adapter + fake codec LSP test |
| joint training 通过 flag 偷渡 | loop/asset/artifact 语义失真 | 独立 decision gate |

## 20. 明确不进入首版

- Stochaflow-native VAE/VQ-VAE/VQGAN training；
- VAE + denoiser joint training；
- adversarial autoencoder loop；
- LPIPS/discriminator framework abstraction；
- arbitrary model hub；
- universal pretrained model registry；
- arbitrary `.ckpt` conversion；
- encoder/decoder 混搭；
- floating Hub revision production run；
- Stable Diffusion-specific component/text training；它由独立计划拥有，不是
  被整个 roadmap 移除；
- 在首个 latent recipe 中启用 learned variance；
- 本计划中的 512×512 production；Stable Diffusion 计划有独立 512 profile；
- ImageNet-1K production；
- DiT-XL/2；
- universal condition/batch schema；
- universal Dataset/Sampler/DataLoader registry；
- metadata/provenance/capacity framework capability；
- 在 DataLoader worker 中写 latent cache；
- persistent read-through hybrid latent cache。

## 21. 调研与实现参考

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)
- [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [CompVis f8d4 autoencoder config](https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml)
- [Diffusers AutoencoderKL](https://huggingface.co/docs/diffusers/api/models/autoencoderkl)
- [Diffusers training overview](https://huggingface.co/docs/diffusers/training/overview)
- [Diffusers VAE training sample request](https://github.com/huggingface/diffusers/issues/3726)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
- [Official DiT implementation](https://github.com/facebookresearch/DiT)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Fast-DiT](https://github.com/chuanyangjin/fast-DiT)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [The Met Open Access](https://www.metmuseum.org/hubs/open-access)
- [The Met Collection API](https://metmuseum.github.io/)
- [LHQ](https://arxiv.org/abs/2104.06954)
- [ImageNet-100 candidate](https://huggingface.co/datasets/clane9/imagenet-100)
- [DomainNet candidate](https://huggingface.co/datasets/wltjr1007/DomainNet)
- [Stable Diffusion Component-Native 支持计划](stable-diffusion-component-native-support-plan.md)
- [REPA-E](https://github.com/End2End-Diffusion/REPA-E)
