# Hydra 配置组合与 Train/Sample 边界迁移计划

- 文档性质：开发草案；不属于当前公开 API 或正式文档导航
- 状态：分段排期，尚未进入实现；A0 ADM topology correctness 后执行 P0
  C0/C1，H0–H3 为 latent vertical slice 后的 P2，H4 为 Later
- 统一排期：
  [Development Priority Roadmap](development-priority-roadmap.md)；本文拥有配置
  contract，统一排期拥有跨计划执行顺序
- 制定日期：2026-07-29
- 调研基线：Hydra 1.3.4 stable
- 兼容性：breaking；不兼容旧训练 YAML、旧 sample request 或旧 checkpoint
- 首轮维护案例：MNIST 与 AFHQ-v2
- 退出首轮维护：CIFAR-10、Flowers102、MNIST + Flowers102，以及 Physics
  Reconstruction、Knowledge Distillation 两个 extension reference project
- 首版 Hydra 范围：fresh training 的配置 authoring、composition、检查与单次启动
- 不由 Hydra 处理：strict resume、checkpoint sampling、observability resume overlay、
  AFHQ frozen evaluation、capacity trial policy、adaptive HPO
- 关联决策：
  [Sampling Request Config Refactor](sampling-request-config-refactor.md)、
  [正式 Gaussian loss 架构](../framework.md)、
  [Extension 导入边界与激活延迟优化计划](extension-import-boundary-and-activation-latency-plan.md)、
  [自动化模型调优开发计划](automated-model-tuning-plan.md)

实施快照（2026-07-29）：

- C2 的 built-in 子集已提前收束为 `configs/train/mnist.yaml`、
  `configs/sample/mnist-{ddpm,ddim-50}.yaml` 和
  `configs/overlays/mnist-observability.yaml`；CIFAR-10、Flowers102 与 multi-source
  不再作为 maintained runnable built-in examples。
- Gaussian diagnostics 已改为从自己的 `params.sampling.shape` 取得训练期采样形状，
  不再借用顶层 `sampling.shape`。
- 这不代表 C1 已完成：当前 sample profile 仍使用 checkpoint-v10 的
  `sampling:` partial-request envelope，`StochaflowConfig.sampling`、自动 final
  sample 和 checkpoint defaults 仍等待 C1 一次性删除。
- C2 的 AFHQ 目录迁移以及 Physics/KD active-tree 退出仍未完成。

## 1. 本轮结论

本轮先修正 Stochaflow 自身的配置边界，再引入 Hydra。不能用 Hydra config groups 掩盖
一个本来就错误的 schema。

排期拆成三段：

1. C0/C1 作为 pretrained AE/latent diffusion 前的短 P0，只修正 plain
   Train/Sample authority 并一次性 bump checkpoint；
2. C2 是 retained-example cleanup，可在 latent Phase 1–3 旁路进行，但必须在 H2
   parity 前完成；
3. H0–H3 在 AFHQ latent vertical slice 后、The Met/Stable Diffusion 配置扩张前
   完成；H4 不进入当前主线。

因此“先做 C0/C1”不等于“latent 必须等待完整 Hydra 迁移”。这个拆分避免 inference
asset 和 sample/decode 先适配 checkpoint v10，随后又被 C1 重写。

目标用户模型固定为两个独立 workflow：

```text
train config
    -> fresh training
    -> checkpoint(config + state + immutable inference recipe)

checkpoint + sample config
    -> checkpoint-backed inference
    -> sampler / invocation options / writers
```

具体决策如下：

1. **DDPM 和 DDIM 不是两种训练。** 它们是同一 Gaussian denoising checkpoint 的两种
   数值采样方法。仓库不再保留 `ddpm_mnist.yaml` 与 `ddim_mnist.yaml` 两份训练配置。
2. **训练配置与采样配置完全分开。** 训练配置不再包含 `sampling`、sampler、writer、
   final-sample count 或 `run_after_training`。
3. **训练命令只训练。** 删除隐式 post-training sample lifecycle；
   `stochaflow train` 成功结束于 checkpoint，采样由显式 `stochaflow sample` 调用完成。
4. **sample config 是一次完整采样调用的配置，不再是 checkpoint sampling defaults
   上的 partial overlay。** checkpoint 提供模型、Process、extension provenance、
   state 和不可覆盖 inference recipe；sample config 只提供可变的 invocation choice。
5. **训练期 diagnostic sampling 仍属于 train。** 它改变训练过程中的观测成本、频率和
   输出，因此继续放在 `diagnostics`，但不再借用顶层 `sampling.shape`。
6. **Hydra 只组合 fresh train config。** 首版不组合 sample config，也不读取
   checkpoint、resume overlay、evaluation 或 capacity protocol。
7. **首轮只维护 MNIST 与 AFHQ-v2 两个纵向案例。** MNIST 是最小 built-in baseline；
   AFHQ-v2 是真实数据、extension、class conditioning、ADM/DiT 和 evaluation showcase。
8. **两个 extension reference project 退出 active tree。** Physics Reconstruction 和
   Knowledge Distillation 不参与 Hydra parity、CI、公开文档或 scaffold 验收。
9. **配置可读性是一等公民和阻断条件。** Hydra tree 若比一份完整 YAML 更难理解，
   即使去重成功也不接受。
10. **配置不是 Python 对象图。** 不开放通用 `class_name`、Hydra `_target_` 或任意
    dotted import path；`torch.optim.<Class>` 与
    `torch.optim.lr_scheduler.<Class>` 继续作为两个角色受限的 native identifier，
    由 Stochaflow resolver 解释，不经过通用 `importlib`。
11. **Fresh-training 组件选择错误必须早于 artifact I/O。** Hydra composition 完成并
    转换为 typed config 后，在任何下载、认证、物化或 run directory 创建前，完成该
    fresh train invocation 所选 Registry 名称、extension provenance 和 native target
    的无副作用 preflight。
12. **Torch 能力按 framework role 分界，不建立一个“原生 Torch 对象”总入口。**
    optimizer 与 lifecycle-compatible scheduler 保留受限 native provider；model、
    objective、data pipeline 和 execution policy 分别继续由 Registry/Builder/Trainer
    拥有。新增 native family 必须有新的真实复用证据和独立架构决策。
13. **保留简洁的 role-scoped `name`，但不掩盖其 grammar。** 首轮不改成
    `provider: torch`、`class_name` 或 `target`；配置参考、错误信息和 `--check` 必须按
    role 明确显示该字段接受 Registry identity 还是两个受限 native identifier。
14. **P2 research variation 不扩张 production authoring tree。** corrected ADM
    topology 是唯一 ADM production recipe 的组成；`constant`/`p2` A/B 由 benchmark
    protocol基于同一 readable base config产生 frozen resolved configs，不永久增加
    `train-adm-baseline.yaml`/`train-adm-p2.yaml`，也不把训练变化伪装成 sample
    profile。

推荐的最终边界是：

```text
Hydra train frontend
    |
    | Defaults List / config groups / safe fresh overrides
    v
fully resolved plain training mapping
    |
    | Stochaflow typed validation
    v
Registry / DataBuilder / TrainingBuilder / Trainer
    |
    v
resolved_config.yaml + checkpoint + immutable inference recipe

plain sample config + checkpoint
    |
    | Stochaflow sample config validation
    v
checkpoint recipe / SamplingBuilder / Sampler / Writers
```

Hydra 不替代：

- `StochaflowConfig` 的 runtime validation；
- Registry、DataBuilder、TrainingBuilder、SamplingBuilder；
- optimizer/scheduler native-provider resolver；
- checkpoint authority；
- strict resume 的 frozen-state policy；
- sample compatibility 与 recipe contract；
- extension provenance、activation 和 import ordering；
- Stochaflow run directory、logging、manifest 和 checkpoint lifecycle。

## 2. 为什么必须先拆 Train 与 Sample

### 2.1 仓库中的实际重复

当前 built-in image-generation example 有三组 DDPM/DDIM 训练配置：

| Dataset | 两份 train 的真实差异 |
| --- | --- |
| MNIST | experiment name、output dir、最终 sampler 与 trajectory cadence |
| CIFAR-10 | experiment name、output dir、最终 sampler |
| Flowers102 | 上述差异，外加 `num_workers` / `persistent_workers` 漂移 |

MNIST 和 CIFAR-10 在移除 experiment identity 与 `sampling` 后逐字段完全相同。
Flowers102 的训练数学、模型、Process、optimizer、scheduler、EMA 和 diagnostics 同样
一致；loader 差异是完整 YAML 复制后产生的配置漂移，而不是 DDIM 的训练要求。

三组训练配置都使用：

```text
process = discrete_gaussian
training = gaussian_denoising
```

也就是说，文件名中的 `ddpm_*` / `ddim_*` 把“如何从 checkpoint 生成结果”错误地写成了
“如何训练 checkpoint”。这带来三个问题：

- 修一个训练参数需要同步多份文件；
- review 无法区分真实训练差异和 sampler 差异；
- 复制文件会制造与 sampler 无关的 drift。

Hydra 若继续提供：

```text
train/ddpm_mnist
train/ddim_mnist
```

只是在更复杂的目录里保留错误概念。正确入口应该是：

```text
train/mnist
sample/mnist-ddpm
sample/mnist-ddim-50
```

### 2.2 当前 schema 的隐藏耦合

当前 `StochaflowConfig.sampling` 同时承担四种职责：

1. 是否在训练结束后自动采样；
2. checkpoint 中的 mutable sampling defaults；
3. 数值 sampler 和 writer 的一次调用参数；
4. training diagnostics 所借用的 `sample_shape`。

这四种职责没有共同 lifecycle。

尤其是 `build_training_components()` 当前把 `config.sampling.shape` 注入 diagnostics。
`GaussianQualityEngine` 在没有该值时直接失败。因此删除训练 YAML 中的 `sampling`
不能只做文件重命名；必须先解除这项 runtime coupling。

目标规则是：

- post-training execution policy 删除；
- checkpoint 不再保存 mutable sample defaults；
- sampler/writer/count/seed 属于独立 sample invocation；
- diagnostic sample shape 属于对应 diagnostic 的私有训练期参数。

### 2.3 checkpoint 已经拥有大部分推理信息

checkpoint 已经保存或能够保存：

- resolved training config；
- primary model、Process 和 managed assets state；
- EMA state；
- extension provenance；
- `TrainingPlan.inference_recipe` 的 name 与 fixed contract。

sample 文件不应重复：

- data source；
- model declaration；
- Process/schedule；
- TrainingBuilder；
- prediction type；
- class-conditioning topology；
- optimizer、scheduler、EMA policy；
- trainer 或 training diagnostics。

这些值要么由 checkpoint 重建，要么属于 immutable inference recipe。

sample 文件只应表达一次调用真正可变的内容：

- sampler 及其参数；
- recipe 明确允许的 options，例如 weights、guidance、condition 和 trajectory；
- shape（当 recipe 没有固定它时）；
- num samples、batch size 和 seed；
- artifact writers。

## 3. 配置可读性规则

### 3.1 用户入口使用 workflow 语言

首版 Hydra frontend 只支持训练，因此唯一公开 composition selector 使用 `train`，不使用
含义模糊的 `job`：

```bash
stochaflow-launch \
  --config-dir examples/built-in/image-generation/configs \
  train=mnist
```

AFHQ-v2：

```bash
stochaflow-launch \
  --config-dir examples/showcases/afhq-v2/configs \
  train=dit-128 \
  trainer.device=cuda
```

采样不用 Hydra：

```bash
stochaflow sample \
  --checkpoint outputs/.../checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddim-50.yaml
```

文件名表达用户概念：

- `train/mnist.yaml`，不是 `job/ddpm_mnist.yaml`；
- `train/dit-128.yaml`，不是内部 TrainingBuilder 名；
- `sample/mnist-ddim-50.yaml`，因为 DDIM-50 确实是采样语义；
- `data/afhq-v2-128.yaml`，不是 `data_builder/class_labeled_image.yaml`；
- `recipe/dit-b8-128.yaml`，不是一组 Python target path。

### 3.2 一眼可见的操作边界

一个文件只能属于一种配置 authority：

| 目录 | authority | 允许内容 |
| --- | --- | --- |
| `train/` | fresh training | 完整训练入口与 Defaults List |
| `sample/` | checkpoint sampling | 完整 sample invocation config |
| `overlays/` | strict resume observability | 窄 observability overlay |
| `evaluation/` | AFHQ frozen evaluation | evaluation protocol |
| `resources/` | producer/source | source lock 等非运行配置 |

不能因为都是 YAML 就把它们放进同一个 Hydra graph。

### 3.3 Config tree 预算

1. 每个 `train/*` 入口只有一层用户可见 Defaults List。
2. 一个入口通常不超过 6 个 group choice。
3. 不为单个 scalar 建 group。
4. 不把 `model × data × optimizer × scheduler × trainer` 宣称为任意 Cartesian
   product；已耦合的值保留在 cohesive recipe leaf。
5. 不使用 `_target_`、`class_name`、任意 Python import path 或内部 Builder class
   作为 authoring language。仅允许 Stochaflow 明确定义的受限 native identifier
   grammar；它们不是通用导入路径。
6. `_self_` 必须显式出现，且 precedence 由测试锁定。
7. 首版不使用 custom resolver。
8. 普通 interpolation 只用于短距离、显而易见的值复用。
9. list 由最终 owner 完整声明，不依赖隐式 append 或递归 merge。
10. 用户理解一个 train 入口不应打开超过两层文件。
11. `--cfg job --resolve` 必须输出完整 Hydra preview。
12. Stochaflow `--check` 必须进一步执行 typed validation 与 extension metadata
    preflight，并打印纯 resolved training config。

Hydra 自带的 preview 发生在 task callback 前，不等于 Stochaflow validation。
这里 `--cfg job` 中的 `job` 是 Hydra 固定的“输出 task config”参数值，不是
Stochaflow 的 authoring group；用户选择仍然是 `train=<choice>`。

## 4. 目标 Runtime Schema

### 4.1 Training config

`StochaflowConfig` 删除 `sampling`：

```python
@dataclass(slots=True)
class StochaflowConfig:
    experiment: ExperimentConfig
    data: ComponentConfig
    model: ComponentConfig
    training: ComponentConfig
    objective: ComponentConfig | None
    process: ComponentConfig | None
    extensions: ExtensionsConfig
    optimizer: ComponentConfig
    lr_scheduler: LRSchedulerConfig | None
    ema: EMAConfig
    diagnostics: list[ComponentConfig]
    trainer: TrainerConfig
    logging: LoggingConfig
    artifacts: ArtifactConfig
```

训练配置不再接受：

- `sampling.run_after_training`；
- `sampling.sampler`；
- `sampling.options`；
- `sampling.shape`；
- `sampling.num_samples`；
- `sampling.batch_size`；
- `sampling.seed`；
- `sampling.writers`。

`EMAConfig` 同时删除 `use_for_sampling`。它不是 EMA 的训练生命周期，而是推理权重
选择策略；当前 `options.weights=auto` 与 `ema.use_for_sampling` 形成了隐式双重
precedence。训练配置只保留：

```python
EMAConfig(
    enabled: bool,
    decay: float,
    update_after_step: int,
    update_every: int,
)
```

checkpoint 只记录 raw/EMA state 是否存在。sample profile 使用 `weights` 做显式选择；
若某个 recipe 仍允许 `auto`，其唯一 framework 语义固定为“EMA state 存在则用 EMA，
否则用 raw”，不再读取训练配置中的 preference bool。

旧字段作为 unknown field 直接拒绝，不提供 adapter。

### 4.2 Sample invocation config

不再使用“partial request + checkpoint defaults”的模型。新增独立、完整的
`SampleConfig`：

```python
@dataclass(frozen=True, slots=True)
class SampleConfig:
    sampler: ComponentConfig | None
    options: dict[str, object]
    shape: list[int] | None
    num_samples: int
    batch_size: int
    seed: int
    writers: list[ComponentConfig]


@dataclass(frozen=True, slots=True)
class SampleInvocationConfig:
    sample: SampleConfig
    extensions: ExtensionsConfig
```

首版文件保留一个清晰的 `sample` envelope，以便未来可与 optional extension additions
并列，并避免把该文件误送入 training parser：

```yaml
sample:
  sampler:
    name: ddim
    params:
      num_inference_steps: 50
      eta: 0.0
  options:
    weights: ema
    clip_denoised: true
    trajectory:
      enabled: true
      every_steps: 10
  shape: [1, 32, 32]
  num_samples: 36
  batch_size: 36
  seed: 123
  writers:
    - name: tensor
      params: {}
    - name: image
      params:
        grid_nrow: 6
        gif_fps: 3
        denormalize: true
```

可选 additional inference plugin 使用独立顶层字段：

```yaml
extensions:
  plugins:
    - optional-writer-plugin
```

规则固定为：

- `--checkpoint` 与 `--config` 都必填；
- sample config 不继承 checkpoint 中的 mutable sample defaults；
- schema default 只用于真正稳定的 framework default，不用于隐藏 project profile；
- repository-owned profile 显式写出 `weights`、shape、count、batch、seed 和 writers；
- sampler 和 writers 是原子值，不做 deep merge；
- `options` 是 recipe-owned mapping；
- sample config 不能声明 model、data、Process、TrainingBuilder、recipe name 或 fixed
  contract；
- required training extension 来自 checkpoint provenance；
- sample config 只能追加 inference-time plugin；
- recipe fixed contract 与 sample option/sampler collision 时 fail closed；
- sample device 是 CLI/runtime option，未指定时使用 `auto`，不继承 checkpoint 中训练
  所用的 `trainer.device`；
- output dir 继续是 CLI/runtime option，未指定时使用 checkpoint run 下的唯一
  sample invocation 目录。

`SampleRequest`、`ParsedSampleRequest`、`apply_sample_request()` 及其 shallow-overlay
语义删除。该变更需要 checkpoint format bump；不读旧 checkpoint。

### 4.3 Immutable inference recipe

`TrainingPlan.inference_recipe` 保留：

```python
SamplingRecipe(
    name="standard_denoising",
    contract={"prediction_type": "v"},
)
```

它负责：

- checkpoint 使用哪个内部 SamplingBuilder；
- 哪些任务语义不可由 sample config 改写；
- sampler 是否允许由调用选择；
- conditioning/guidance 的 compatibility contract；
- 必要的 model/process adaptation contract。

它不负责：

- num samples；
- batch size；
- seed；
- artifact writer；
- trajectory cadence；
- 一次调用的 class labels 或 guidance value。

`SamplingBuilder` 仍是内部 composition boundary，不出现在 sample YAML。

### 4.4 Training diagnostics

训练期 diagnostics 不属于 `sample` workflow。需要采样的 diagnostic 在自己的 params 中
完整声明：

```yaml
diagnostics:
  - name: diffusion_quality
    params:
      sampling:
        shape: [1, 32, 32]
        sample_num: 32
        batch_size: 32
        seed: 123
      samplers:
        - id: ddim-50
          name: ddim
          params:
            num_inference_steps: 50
            eta: 0.0
```

实施要求：

- `DiagnosticSamplingConfig` 新增 `shape`；
- diagnostic factory 从该 diagnostic 自己的配置取得 shape；
- 删除 `build_diagnostics(..., sample_shape=config.sampling.shape)`；
- 不从 DataBuilder、model class 或 sample profile 猜 shape；
- diagnostic sampler 仍是训练观测 recipe 的一部分；
- diagnostic 输出不等于显式 `stochaflow sample` 输出。

这会在训练 YAML 中保留少量 sampler 信息，但它有真实训练期语义，不会制造两份 train。

### 4.5 删除自动 final sample

删除：

- `SamplingConfig.run_after_training`；
- `EMAConfig.use_for_sampling`；
- `TrainingComponents.use_ema_for_sampling`；
- `train --skip-final-sample`；
- runner 中 selected-best checkpoint 的自动 `run_sampling()`；
- `execution.skip_final_sample`；
- reference generator 与 docs 中相关字段；
- final sample 失败影响训练命令成功状态的隐式耦合。

如果用户需要顺序运行：

```bash
stochaflow train --config ...
stochaflow sample --checkpoint ... --config ...
```

CI、shell、Make、workflow engine 或未来独立 orchestration command 可以显式编排这两个
操作。首版不新增 `train --sample-config`，避免重新把 sample lifecycle塞回 train。

## 5. Workflow Authority 与 Freeze

### 5.1 最终 authority matrix

| Workflow | 唯一 base | 允许变化 | 禁止变化 |
| --- | --- | --- | --- |
| fresh train | 外部 resolved train config | fresh authoring fields、文档化 execution flags | unknown schema、非法 component contract |
| strict resume | checkpoint train config + state | target epoch、device、progress、output root、batch limits、窄 observability overlay | data/model/training/process/objective/optimizer/scheduler/EMA/precision/accumulation/artifact identity |
| checkpoint sample | checkpoint state + fixed recipe +完整 sample config | sample schema 字段、device/output runtime flags、additive plugin | training config、recipe、state topology、required plugin |
| AFHQ evaluation | frozen evaluation protocol + sample outputs/checkpoint | protocol 显式声明的 evaluation invocation | training-style arbitrary merge |

freeze 不通过 OmegaConf readonly node 实现，而是通过“不让错误的配置源进入 workflow”
实现：

- resume 不重新组合当前 project train tree；
- sample 不把 train tree当 base；
- sample config 不修改 checkpoint training config；
- fixed recipe 不进入 Hydra override surface；
- evaluation 继续使用自己的 frozen parser。

### 5.2 三类训练值

#### Persistent train fields

进入 `resolved_config.yaml` 和 checkpoint：

- data/model/training/process/objective；
- optimizer/scheduler/EMA；
- trainer epochs/device/precision/accumulation；
- diagnostics/logging/artifact policy；
- extension selection；
- experiment seed/output root。

#### Runtime-only invocation facts

不进入 component config：

- batch limits；
- deterministic mode；
- artifact verification worker override；
- startup cwd；
- resume lineage。

Hydra frontend 将 fresh-run runtime facts放在 typed conversion 前移除的 `execution`
subtree：

```yaml
execution:
  limit_batches: null
  limit_validation_batches: null
  limit_test_batches: null
  deterministic: false
  artifact_verification_workers: null
```

不存在 `skip_final_sample`。

fresh training 没有 checkpoint provenance 可供比较，因此不提供
`force_extension_version_mismatch`。若未来 fresh invocation 有独立、可验证的
extension lock authority，再由该 authority 的计划定义 mismatch policy。

#### Derived runtime values

- `exp_id`；
- resolved device；
- actual run directory；
- global step；
- artifact bindings；
- selected component identities；
- checkpoint lineage；
- composition audit digest。

这些值不作为普通 Hydra override。

## 6. Hydra 的职责

### 6.1 采用 Hydra，但只做 authoring/composition

Hydra 适合解决：

- Defaults List；
- config groups；
- command-line override grammar；
- interpolation；
- config search path；
- resolved config preview；
- 未来有限 sweep 的 authoring。

Hydra 不负责：

- component construction；
- checkpoint merge；
- strict resume；
- sample config merge；
- extension runtime import；
- artifact validation；
- run identity/logging；
- HPO algorithm、pruner 或 resource scheduler。

composition 结束后必须回到 plain mapping：

```python
resolved = OmegaConf.to_container(
    composed,
    resolve=True,
    throw_on_missing=True,
    enum_to_str=True,
)
config = load_config_dict(training_mapping)
```

`DictConfig`、Defaults List、Hydra runtime nodes 和 resolver 不进入 runtime components 或
checkpoint。

### 6.2 不使用 instantiate

禁止：

- `hydra.utils.instantiate()`；
- `_target_`；
- `class_name`；
- 任意 Python import path 驱动的 component graph；
- 用 Hydra config group 镜像 Stochaflow Registry；
- 用 Hydra search path plugin 绕过 extension metadata preflight。

Stochaflow built-in 与 external component 继续通过同一 Registry/factory/Builder path。

PhysicsNeMo DFSR 示例中的 `class_name` 不构成反例。其用户 YAML 只声明
`arch: dfsr`、`precond: dfsr` 等领域选择；`train.py` 中受信任的 Python recipe
再把一个 `precond` 选择映射为配套的 network 与 loss class，并注入 Dataset 派生尺寸、
condition metadata 和 optimizer parameters。截图展示的是 composition root 生成的内部
构造描述，不是用户直接编写的 Hydra `_target_`，也不是统一的公共配置模型。

Stochaflow 借鉴其中“用户声明语义、composition root 选择实现并注入 runtime dependency”
这一点，不复制 example-local unrestricted class importer。互相耦合的 model wrapper、
loss、condition 和 batch 语义继续由 TrainingBuilder/Strategy 组合，不能把它们拆成任意
class path 的笛卡尔积。

成熟框架对照说明的是“配置能力必须跟随 framework 实际拥有的 lifecycle”，而不是
“谁能接受更多 class path”：

| 系统 | 它允许什么 | 为什么不能机械复制 |
| --- | --- | --- |
| Hydra 1.3 | `_target_` 可以指向 class/callable，并默认递归实例化对象图 | Hydra 提供 construction mechanism，不知道 Stochaflow 的 Registry provenance、Builder compatibility、checkpoint 或 loop contract |
| PhysicsNeMo DFSR example | 用户 YAML 选择 `arch` / `precond`；受信任 Python recipe 再写内部 network/loss `class_name` | 截图中的 class path 是 recipe 的输出，不是要求所有用户直接维护的公共 schema |
| LightningCLI | 可以选择 Torch optimizer/scheduler 和多个 subclass；Lightning 同时拥有 automatic closure、metric monitor 与 manual optimization 等相应 lifecycle | Stochaflow 当前 automatic loop 只有一个 optimizer、一次普通 backward/step 和零参数 scheduler step；暴露相同名字不能凭空获得相同 runtime semantics |
| Stochaflow | Hydra 只组合；Registry/Builder 构造；optimizer/scheduler 使用角色受限 native provider | surface area 由当前 loop 和 composition contract 决定，不能由上游 namespace 大小决定 |

因此，Lightning 能在自己的 automatic loop 中为 LBFGS 提供 closure、为
`ReduceLROnPlateau` 提供 monitor，并不意味着 Stochaflow 应只按基类接受它们。若未来
Stochaflow 新增 closure-required、metric-driven 或 manual/multi-optimizer loop，那是新
execution capability；对应 native target 才能在那条 lifecycle 中开放。

### 6.3 Torch 原生能力声明边界

当前两种合法写法保持不变：

```yaml
optimizer:
  name: torch.optim.AdamW
  params:
    lr: 1.0e-4
    weight_decay: 0.0
    betas: [0.9, 0.999]
    eps: 1.0e-8

lr_scheduler:
  name: torch.optim.lr_scheduler.CosineAnnealingLR
  interval: epoch
  params:
    T_max: 100
```

它们看起来像 Python fully-qualified name，但语义固定为：

```text
NativeOptimizerTarget :=
    "torch.optim." + DirectPublicOptimizerClass

NativeLRSchedulerTarget :=
    "torch.optim.lr_scheduler." + DirectPublicLRSchedulerClass
```

resolver 只能在对应的已导入 namespace 上读取一个直接 public class，分别验证
`torch.optim.Optimizer` 或 `torch.optim.lr_scheduler.LRScheduler` 基类；禁止额外 `.`,
private name、任意 module import 和 Registry 对保留前缀的冒充。它是
namespace-and-contract-restricted provider，不是逐类维护的枚举，也不是通用 importer。

“可写出 target”不等于整个 PyTorch namespace 都被当前 Trainer 支持。实际支持集合是：

```text
direct public class
∩ expected upstream base
∩ constructor 可由 plain config + framework injection 表达
∩ 当前 single-optimizer automatic loop 可驱动
∩ bound step() 可无额外参数调用
```

因此：

- `Adam`、`AdamW`、`SGD`、`StepLR`、`CosineAnnealingLR` 等可以使用；
- `LBFGS` 需要 closure，不属于当前 automatic loop；
- `ReduceLROnPlateau` 需要 metric，不属于当前 scheduler lifecycle；
- `LambdaLR` 需要 callable，`SequentialLR` / `ChainedScheduler` 需要已构造的嵌套
  scheduler graph，不属于 plain-YAML authoring contract；但它们的 target identity 与
  零参数 `step()` 本身可能通过 preflight，必须由 `--check` 的
  `constructor: not_run` 状态明确表示“尚未证明可构造”，不能显示成 supported；
- 首版只注入一个 framework-owned trainable parameter iterable，不开放 YAML parameter
  groups；需要分组学习率或 weight decay 时，应由新的窄 TrainingBuilder/optimization
  plan 显式拥有参数选择，而不是让 config 引用 module path；
- `OneCycleLR` 一类依赖运行总步数的 scheduler 只能接收配置显式给出的普通参数，framework
  不根据 constructor 参数名推导或注入 DataLoader/epoch facts；
- sparse-gradient 和算法特有约束仍由对应 Builder/model 与 PyTorch 在构造或执行边界
  验证，不能仅凭基类声明兼容。

native provider 继续遵守：

- framework 注入 optimizer 的 trainable parameters 和 scheduler 的 optimizer；
- 配置不能覆盖这些 runtime-owned arguments；
- `interval: step | epoch` 是 Stochaflow lifecycle policy，不是 PyTorch constructor
  parameter；
- 其余 `params` 原样传给当前 PyTorch，不复制上游签名、别名和默认值；
- retained MNIST/AFHQ recipes 显式写出会实质改变实验结果的关键参数；其余上游版本与
  默认值由项目 lockfile/environment specification 固定；
- checkpoint 保存 resolved config、具体 class identity 和 state，但不把全部上游默认值
  展开成 Stochaflow schema。

首轮不把 authoring syntax 拆成额外的 `provider: torch` + `name: AdamW`。只有同一
framework role 出现第二个真实、共享相同 lifecycle 的 native provider，且当前 qualified
identifier 已产生歧义时，才重新评估 discriminated provider schema。

`name` 的含义必须由所在 role 明确，不再由一个泛化的 `ComponentConfig` 文档掩盖：

- model、objective、process、training、diagnostic 等 `name` 是对应 Registry identity；
- optimizer/scheduler `name` 可以是 Registry identity，也可以是上述精确 native
  identifier；
- data 的 nested source/provider selection 由该 DataBuilder recipe schema 定义；
- 所有 role 的 `name` 都在 typed validation 或 target preflight 阶段验证非空、类型和
  所属 namespace，不能等到 component construction 才报错。

底层实现可以继续复用小型 value object，但生成配置参考、错误路径和 public docs 必须按
role 展示 accepted grammar。首轮不增加 `class_name`、`target` 或 `provider` 三套同义
字段。

未来新增 native provider 必须同时满足：

1. 上游存在稳定、公开且足够强的 role contract；
2. framework 能明确注入所有 runtime-owned dependency；
3. 当前执行 loop 能驱动其完整 state/step/checkpoint lifecycle；
4. 剩余用户参数能由简单、可审计的 plain values 表达；
5. 替换实现不会静默改变 batch、task、partition、resume 或 distributed 语义；
6. 完整兼容性能在 Builder、Strategy 或其他拥有组合信息的窄边界验证。

据此首轮决策为：

| Torch family | 配置边界 |
| --- | --- |
| Optimizer | 保留受限 native provider |
| LR scheduler | 保留 lifecycle-compatible 子集 |
| `torch.nn` loss | 不开放整个 namespace；出现真实重复后评估窄 Objective adapter |
| 任意 `nn.Module` model | Registry + TrainingBuilder，不开放 class path |
| activation / normalization / initialization | 具体 model 的私有参数或 Python 组合 |
| Dataset / transform / PyTorch sampler / collate / DataLoader | DataSource/DataBuilder recipe |
| precision / gradient clipping / determinism / future `torch.compile` | Trainer execution policy |
| metrics | 未来按 metric state/update/compute lifecycle 单独评估，不开放任意 target |

`logging.torch_logs` 不是 native provider，而是依赖当前 PyTorch 版本的 best-effort
observability passthrough。retained base configs 不保留无意义的 `torch_logs: {}`；只有
实际需要时才由 observability config 声明，并由独立测试约束错误传播和 replacement
语义。

Hydra composition 期间仍不激活 extension 或解析 runtime class。完成 plain mapping 与
typed validation 后，由 Stochaflow trusted activation boundary 执行 target preflight：

```text
load built-ins
→ validate selected extension metadata
→ activate selected extensions
→ resolve Registry names and native targets
→ validate base/envelope contracts
→ only then perform data artifact I/O and create the official run
```

该 preflight 不调用 component constructor、不展开 PyTorch defaults、不通过 signature
推断 runtime 参数，也不声称完成 Builder-level semantic compatibility。`--check` 必须
显示 resolved provider/family/target，并明确列出尚未执行的 constructor、data 和 runtime
checks。

preflight 是 framework-private、按已知 config role 枚举的检查，不是扫描任意字符串的
通用 resolver。它至少覆盖：

- 每个 selected Registry name 已存在，且满足该 Registry 的 declared base contract；
- native target 满足精确 prefix、direct public identifier 和 expected base；
- optimizer/scheduler 的 public `step` contract 没有 framework 无法提供的必需参数；
- extension target 只在对应 plugin 完成 metadata preflight 和受控激活后解析；
- 返回可审计的 resolved identity，不创建 component、不访问数据、不创建 official
  output。

零参数 `step()` 是 automatic loop 已声明的 lifecycle contract，因此可以做窄的静态
callability 检查；禁止的是根据 constructor 参数名猜测 DataLoader 长度、epoch、metric、
model module 或其他 runtime value。实例构造后仍重复验证 bound instance 的 `step()`、
scheduler 所持 optimizer 和实际 state contract。constructor kwargs、parameter group、
callable 参数与完整 Builder compatibility 不属于 `--check` 的成功承诺。

这里明确接受一个有限的失败时机差异：target typo、wrong base、closure-required step 和
metric-required step 必须在 artifact I/O 前失败；plain parameter 的拼写/类型、
callable/nested constructor dependency 和跨组件 runtime compatibility 可能只在真实
construction 时失败。首轮不为消除这一差异维护 PyTorch per-class allowlist，也不以 dummy
parameter/model 实例化来冒充无副作用检查。后续只有在真实用户错误表明这种失败时机不可
接受时，才单独评估 upstream contract descriptor；不能偷偷扩大 `--check` 的承诺。

`--check` 的 target report 使用状态而不是模糊的成功文案，例如：

```yaml
targets:
  optimizer:
    role: optimizer
    provider: torch-native
    identity: torch.optim.AdamW
    identity_resolution: passed
    static_step_contract: passed
    constructor: not_run
  model:
    role: model
    provider: registry
    identity: dit
    identity_resolution: passed
    constructor: not_run
runtime_checks:
  data_artifact: not_run
  builder_compatibility: not_run
```

该报告是诊断输出，不成为新的持久化 schema 或公共 component descriptor。

### 6.4 Optional dependency

首版增加：

```toml
[project.optional-dependencies]
composition = [
  "hydra-core>=1.3.4,<1.4",
]
```

约束：

- core config、programmatic runtime、resume 和 sample 不 import Hydra；
- 只有 native fresh-train launcher 和 composition tooling 需要 extra；
- decorated Hydra app 延迟 import；
- 缺失 extra 时给出明确安装命令；
- 首版固定 `version_base="1.3"`。

## 7. 首轮 Config Layout

### 7.1 MNIST

```text
examples/built-in/image-generation/configs/
  train/
    mnist.yaml
  sample/
    mnist-ddpm.yaml
    mnist-ddim-50.yaml
  data/
    mnist-32.yaml
  recipe/
    mnist-unet-v-cosine.yaml
  observability/
    image-quality.yaml
  overlays/
    mnist-observability.yaml
```

`train/mnist.yaml`：

```yaml
# @package _global_

# MNIST 32x32 Gaussian denoising training.
defaults:
  - /data: mnist-32
  - /recipe: mnist-unet-v-cosine
  - /observability: image-quality
  - _self_

experiment:
  name: mnist
  seed: 42
  output_dir: outputs/mnist

extensions:
  plugins: []
```

这里没有 DDPM/DDIM choice。两种 sample profile 指向同一类 checkpoint。

### 7.2 AFHQ-v2

```text
examples/showcases/afhq-v2/configs/
  train/
    adm-128.yaml
    dit-128.yaml
    smoke-adm-128.yaml
  sample/
    ddim-50-cfg-2.yaml
  data/
    afhq-v2-128.yaml
  recipe/
    adm-128.yaml
    dit-b8-128.yaml
    adm-128-smoke.yaml
  observability/
    class-conditional-quality.yaml
  evaluation/
    ddim-50-cfg-2-kid-fid.yaml
```

ADM 与 DiT production 的以下差异保持为完整、可见的 train recipe：

- model declaration；
- batch size；
- gradient accumulation；
- scheduler warmup/total steps；
- artifact verification policy；
- experiment identity/output；
- diagnostic cost。

它们不能伪装成任意 `model × hardware profile × scheduler` Cartesian product。

`smoke-adm-128` 同时改变 model topology、process steps、loader、optimizer、scheduler、
EMA 和 diagnostics，应继续作为独立 train recipe，而不是 `smoke=true` scalar overlay。

AFHQ sample profile只拥有：

- DDIM-50；
- CFG scale/class selection；
- weights/trajectory；
- count/batch/seed；
- image/tensor writers。

### 7.3 Package 规则

- `train/*` 是 project entry，使用 `# @package _global_`；
- train Defaults List 使用 `/data`、`/recipe`、`/observability` absolute group path；
- `data/*` 使用默认 package 并完整拥有 `data`；
- `recipe/*` 和 `observability/*` 可在确实跨多个 runtime top-level sections 时使用一次
  `_global_`；
- `sample/*` 不进入 Hydra train composition；
- 不使用 `group@some.deep.path`；
- compose 后不得残留 `train`、`recipe` 或 `observability` authoring wrapper key。

Path ownership：

| Leaf | resolved path owner |
| --- | --- |
| `train/*` | experiment、extensions、少量入口可见的耦合 delta |
| `data/*` | 完整 data |
| `recipe/*` | model、process、training、objective、optimizer、scheduler、EMA、trainer、artifacts |
| `observability/*` | diagnostics、logging |

只有 `train/*` 的 `_self_` 可以覆盖前面 leaf 的值，并且 override 必须直接可见。

## 8. Example Scope 收敛

### 8.1 保留

| Example | 角色 | 必须覆盖 |
| --- | --- | --- |
| MNIST | 最小 built-in vertical slice | train、DDPM sample、DDIM sample、resume、observability、short run |
| AFHQ-v2 | 真实规模 showcase 与 extension vertical slice | managed data、ADM、DiT、smoke、class sample、evaluation、capacity、resume |

这两个案例已经覆盖：

- built-in 与 external extension；
- unconditional 与 class-conditional；
- small smoke 与 real data；
- UNet/ADM 与 DiT；
- training、checkpoint、sample、evaluation；
- managed DataArtifact；
- plugin provenance 与 installed-project workflow。

### 8.2 删除 active runnable profiles

删除：

- `ddpm_cifar10.yaml`；
- `ddim_cifar10.yaml`；
- `ddpm_flowers102.yaml`；
- `ddim_flowers102.yaml`；
- `ddpm_mnist_flowers102.yaml`；
- 与这些 profiles 一一绑定的 README 段落、golden、runner/config tests 和质量声明。

这项删除收敛的是维护的 end-to-end examples，不等同于在同一提交中删除所有通用 image
recipe 或 DataSource capability。Core generic capability 若保留，必须由独立 contract
tests 验证，不能再被描述成 maintained runnable showcase。

如果 `torchvision` source 仍公开列出 CIFAR-10/Flowers102，则在实施时必须二选一：

1. 保留低成本 source contract tests，并明确它们只是 source support；
2. 从 source accepted dataset 和公开 reference 中一起删除。

不能删除 example test 后继续宣称同等级 end-to-end support。

### 8.3 删除两个 extension reference project

从 active repository 删除：

```text
examples/extension-projects/physics-reconstruction/
examples/extension-projects/knowledge-distillation/
```

同步删除或改写：

- `tests/test_extension_reference_projects.py` 中对应 project matrix；
- `pyrightconfig.json` execution environments；
- root README 的 reference-project 宣传与命令；
- `docs/configuration/reference-projects.md` 及文档导航；
- Physics sampling-capacity 和 troubleshooting 的 project-specific workflow；
- installed-wheel harness 中的 Physics/KD cases；
- development plans 中仍把 Physics/KD 当作未来强制 gate 的 active 表述。

不因为删除 example 而删除 framework contracts。需要保留的行为改用最小 test fixtures：

- independent custom DataSource/DataBuilder；
- extension metadata preflight/activation；
- custom TrainingBuilder/SamplingBuilder；
- no-artifact synthetic builder contract；
- referenced DataArtifact store contract。

AFHQ-v2 作为唯一真实 extension project 承担 installed-wheel/example acceptance。
test-only fixture 只证明 extension point contract，不演变成第三个维护 example。

历史开发决策可以保留“当时使用 Physics/KD 发现了什么”，但必须标记为历史背景，不能
继续出现在当前验收矩阵和公开支持表中。

## 9. Native Frontend

### 9.1 独立入口

首版增加独立 `stochaflow-launch`，不把 Hydra 隐藏到现有 argparse loader 后：

```bash
stochaflow-launch \
  --config-dir ./examples/built-in/image-generation/configs \
  train=mnist \
  trainer.device=cuda
```

检查：

```bash
stochaflow-launch \
  --check \
  --config-dir ./examples/built-in/image-generation/configs \
  train=mnist
```

低层入口仍可消费 resolved plain YAML：

```bash
stochaflow train --config resolved.yaml
stochaflow train --resume outputs/.../checkpoints/latest.pt
stochaflow sample --checkpoint ... --config sample.yaml
```

`stochaflow train --config` 是否在最终 cutover 后删除，由最后阶段决定；runtime
`load_config_dict()` 不删除。

### 9.2 Public choice

首版唯一 config-group choice：

```text
train=<choice>
```

`data`、`recipe` 和 `observability` 由每个 train entry 的 Defaults List 固定。用户可以
覆盖已声明的普通 fresh field，但不能自由切换内部 group 形成未验证的 Cartesian
product。

允许：

```text
trainer.device=cuda
trainer.num_epochs=50
data.params.loader.batch_size=64
experiment.output_dir=outputs/mnist-test
execution.limit_batches=10
```

拒绝：

- `+` / `++` 添加未知 field；
- `~` 删除 required field；
- command line 更换 `data=` / `recipe=` / `observability=`；
- `hydra.*` controller override；
- user `--config-name`；
- custom resolver；
- `_target_`；
- `-m` / `--multirun`（首版）。

### 9.3 Trusted bootstrap

framework wheel 只提供一个最小 controller bootstrap：

- optional `train` group declaration；
- fixed Hydra run/sweep/output/logging policy；
- no cwd change；
- no Hydra-created official Stochaflow run；
- no automatic extension search path plugin。

项目通过一个显式 `--config-dir` 提供 config root。Project root：

- 不能替换 primary bootstrap；
- 不能声明 `hydra/` group；
- 不能注册 resolver；
- 不能改变 controller policy；
- compose 阶段不导入 extension runtime code。

轻量 shim 在进入 Hydra 前扫描 argv 与 project YAML，拒绝 controller injection。Callback
再次检查最终 `HydraConfig`。

## 10. Runtime Integration

### 10.1 抽出 programmatic invocation

当前 runner 将 argparse、authority resolution、preflight、run creation 和 execution
混在一起。先抽出：

```python
@dataclass(frozen=True, slots=True)
class TrainingInvocation:
    config: StochaflowConfig
    config_source: str
    startup_cwd: Path
    runtime_options: TrainingRuntimeOptions
    composition_audit: ConfigurationCompositionAudit | None


def run_training_invocation(
    invocation: TrainingInvocation,
) -> TrainingResult:
    ...
```

旧 argparse 与 Hydra frontend 都调用它。Hydra app 不复制 runner lifecycle。
无副作用 target preflight 属于 `run_training_invocation()` 的 fresh-training lifecycle，
不是 Hydra adapter 的特殊能力，因此 low-level plain config 与 Hydra composed config
获得相同的失败时机。strict resume、sample 和 evaluation 保持各自的 checkpoint/recipe
authority，不因本轮迁移获得新的通用 target scanner。

### 10.2 Side-effect ordering

固定顺序：

```text
scan controller inputs
-> compose
-> convert to primitive mapping
-> split execution subtree
-> load_config_dict()
-> extension metadata preflight
-> static semantic checks
-> activate selected extension
-> resolve Registry names and restricted native targets
-> validate target namespace / base / lifecycle envelope
-> DataBuilder / artifact resolution
-> create official run directory
-> build training components
-> execute
```

Hydra migration 不应把 extension activation、artifact I/O 或 run directory mutation 提前到
composition 阶段。target preflight 只解析身份与静态 contract，不构造 optimizer、
scheduler、model 或 objective；完整的跨组件兼容性仍由 Builder/Strategy 在拥有 runtime
上下文后验证。

### 10.3 Output ownership

固定 Hydra policy：

```yaml
hydra:
  job:
    chdir: false
  output_subdir: null
  run:
    dir: .
  job_logging:
    version: 1
    disable_existing_loggers: false
    handlers: {}
    root:
      handlers: []
  hydra_logging:
    version: 1
    disable_existing_loggers: false
    handlers: {}
    root:
      handlers: []
```

目标：

- Hydra 不创建 `.hydra/`；
- 不创建第二个 train log；
- 不改变 cwd；
- Stochaflow 唯一拥有 official output root、manifest、metrics、checkpoints 和 samples。

### 10.4 Composition audit

manifest 与 checkpoint metadata 保存：

```yaml
configuration_composition:
  engine:
    name: hydra
    version: 1.3.x
  config_root: project:examples/.../configs
  choices:
    train: mnist
    data: mnist-32
    recipe: mnist-unet-v-cosine
    observability: image-quality
  overrides:
    - train=mnist
    - trainer.device=cuda
  composed_config_sha256: ...
  effective_config_sha256: ...
```

- `composed_config_sha256`：runner-derived `exp_id` 等 mutation 前的 typed training
  config；
- `effective_config_sha256`：实际写入 `resolved_config.yaml` 和 checkpoint 的 config；
- choices 过滤所有 `hydra/*`；
- non-Hydra run 的 composition audit 为 `null`；
- resume 继承原 audit lineage，不重新 compose；
- sampling manifest引用 checkpoint lineage，但不伪造新的 train composition audit。

### 10.5 Sample runtime 不再合并 TrainingConfig

当前 `resolve_sampling_inputs()` 通过复制 checkpoint `StochaflowConfig`、替换其中
`sampling`，生成一个混合对象。目标实现改为并列持有两种 authority：

```python
@dataclass(frozen=True, slots=True)
class ResolvedSamplingInputs:
    checkpoint_config: StochaflowConfig
    sample_config: SampleInvocationConfig
    checkpoint: SamplingCheckpointView
    recipe: SamplingRecipe
    extension_plan: ExtensionActivationPlan
```

runtime 使用 `checkpoint_config` 重建 model 与 Process，使用 `sample_config.sample` 构造
`SamplingBuilderContext` 和 writers。两者不 merge：

- training manifest/selected components 不记录 sampler 或 writers；
- sample manifest 单独记录 resolved sample config、实际 sampler、weights 和 writers；
- sample device 默认 `auto`，不读取 `checkpoint_config.trainer.device`；
- sample seed 来自完整 sample config，不回退到 training experiment seed；
- required extension 来自 checkpoint provenance，additions 来自 sample config。

## 11. 迁移阶段

### Phase C0：冻结当前行为与确认删除范围

目标：建立 breaking migration 前的事实基线。

修改：

- 为 MNIST DDPM/DDIM 去除 identity/sampling 后的等价性加结构测试；
- 记录当前 diagnostics 对 `sampling.shape` 的隐藏依赖；
- 冻结 strict resume、recipe、artifact binding 和 plugin provenance；
- 冻结 MNIST 两份 sample profile 与 AFHQ sample/evaluation profile 的实际值；
- 建立删除清单，确认 public docs、tests、pyright、packaging 和 CI 的 Physics/KD 引用；
- 不为将删除的 CIFAR/Flowers/Physics/KD authoring files建立长期 golden。

完成条件：

- train/sample schema 重构不会误改 model/process/training math；
- retained examples 的真实差异有 typed structural diff；
- 删除范围不触及无关 core contracts。

### Phase C1：拆分 TrainingConfig 与 SampleConfig

目标：在不引入 Hydra 前先得到正确的 plain-YAML boundary。

修改：

- 从 `StochaflowConfig` 删除 `sampling`；
- 新增独立完整 `SampleConfig` parser；
- sample CLI 强制 checkpoint + config；
- 删除 checkpoint mutable sampling defaults 与 partial merge；
- bump checkpoint format，不读旧 checkpoint；
- `TrainingPlan.inference_recipe` 继续固化 internal composition contract；
- 删除 `EMAConfig.use_for_sampling`，令 `weights=auto` 只由 EMA state availability
  决定；
- diagnostic sampling 自己声明 shape；
- 删除 auto final sample 和 `--skip-final-sample`；
- 更新 config reference generator。

完成条件：

- 一份 MNIST train config 可同时用于 DDPM/DDIM sample；
- train config 中没有 sampler/writer；
- sample config 中没有 training/model/data；
- diagnostic sample 仍可运行；
- sample compatibility 仍由 checkpoint recipe fail closed；
- strict resume 不受 sample config 影响。

### Phase C2：收敛 examples

目标：只保留 MNIST 和 AFHQ-v2。本 phase 是高优先级 cleanup，但不是 pretrained
codec 或 AFHQ latent correctness 的功能前置；它可以在 Latent Phase 1–3 旁路进行，
并必须在 H2 parity 前完成。

修改：

- built-in 重命名为 `configs/train/mnist.yaml`；
- 保留 `sample/mnist-ddpm.yaml` 与 `sample/mnist-ddim-50.yaml`；
- AFHQ train/sample/evaluation 配置迁入明确目录；
- 删除 CIFAR/Flowers/multi-source runnable profiles；
- 删除 Physics/KD project directories；
- 以小型 fixtures 补齐必要 extension/core contract coverage；
- 更新 root/public docs、pyright、CI、packaging 和 tests。

完成条件：

- repository-wide 搜索不再把 retired project 当 active support；
- 两个 retained example 均可从 clean environment 执行；
- generic contracts 的测试不依赖 retired projects；
- README 不再列出已删除命令。

### Phase H0：引入 composition kernel

目标：引入 Hydra 但不启动训练。

修改：

- 增加 optional `composition` extra；
- 新增 context-scoped compose adapter；
- 增加 trusted packaged bootstrap；
- 增加 project config-root scanner；
- 实现 primitive conversion、resolver policy、override policy 和 execution schema；
- 实现 composition audit 与 canonical digest；
- 保持 core config/sample/resume 无 Hydra import；
- 增加 MNIST compose-only spike。

完成条件：

- composed typed config 与 C2 plain MNIST train config相等；
- compose 不导入 AFHQ extension target；
- `DictConfig` 不越过 adapter；
- cwd/filesystem 不变化；
- 未安装 Hydra 时低层 train/resume/sample 正常。

### Phase H1：Fresh single-run launcher

目标：提供只支持 fresh training 的 native frontend。

修改：

- 抽 `TrainingInvocation`；
- 新增无 Hydra import shim 与 lazy decorated app；
- 固定 Hydra controller policy；
- 支持 `train=<choice>` 与每个 retained train entry 显式声明的 safe override
  allowlist；
- 增加 `--check`；
- 将 role-aware target preflight 放入共享 `run_training_invocation()`，而不是 Hydra
  callback；
- 在 plain mapping、typed validation 和 controlled extension activation 之后执行无副作用
  component/native target preflight；
- composition/typed/extension/target preflight error 在任何 data artifact I/O 与 official
  run creation 前失败；
- 禁用 resume、sample、controller override 和 multirun。

首版不接受“任意 existing field 都可 override”。例如单独覆盖 batch size 可能破坏
batch/accumulation/scheduler 的 cohesive recipe。未进入 safe allowlist 的字段必须编辑
或新增清晰命名的 recipe leaf；未来若需要 expert override，应以显式不安全入口和单独
审计语义提出，不能成为普通 frontend 默认行为。

完成条件：

- low-level plain config 与 Hydra config构建同一 typed components；
- 无 `.hydra`、重复日志或 cwd change；
- `--cfg job --resolve` 可读；
- `--check` 显示 Registry 或 native-provider 的 resolved identity，并清楚区分
  composition、typed schema、extension metadata、target preflight 与尚未执行的
  constructor、data 和 runtime checks；
- 拼错或越界 target 不下载、不认证、不物化数据，也不创建 output directory。

### Phase H2：MNIST 与 AFHQ parity

本 phase 在 AFHQ latent vertical slice 后启动；此时 plain config 的 codec、latent
recipe、sample authority 以及 corrected ADM topology fields 已稳定，不会在 Hydra
parity 后再次改写。旧 `transformer_depths`/`middle_transformer_depth` 不进入 Hydra
groups。

顺序：

1. MNIST；
2. AFHQ ADM production；
3. AFHQ DiT production；
4. AFHQ ADM smoke。

每个 train 入口验证：

```text
Hydra composed mapping
    -> load_config_dict()
    -> to_dict()
    == checked-in canonical retained config
```

另行验证：

- MNIST DDPM 与 DDIM sample profiles消费同一个 checkpoint family；
- MNIST observability resume overlay 不进入 Hydra；
- AFHQ batch、accumulation、scheduler、verification 不漂移；
- AFHQ ADM parity 使用 A0 后的 canonical skip/attention topology，并拒绝 pre-A0
  config/checkpoint；
- P2 benchmark override 与 production Hydra Defaults List 隔离；
- AFHQ sample config不进入 train Defaults List；
- AFHQ evaluation/capacity authority 不变；
- AFHQ installed-wheel extension activation 正常。

该阶段完成后才删除 retained examples 的旧完整 YAML authoring surface。

### Phase H3：Scaffold、reference 与文档

修改：

- `stochaflow init` 生成 `configs/train/` 与 `configs/sample/`；
- scaffold 只给一个 train entry 和一个 sample profile；
- 默认不展示 parameter-per-file；
- 生成 resolved-config preview tooling；
- 增加 config-tree readability linter；
- 配置参考按 role 描述 `name` 的 accepted grammar，不再把通用
  `ComponentConfig.name` 一律写成 Registry name；
- 更新 public configuration、workflow、migration、extension、troubleshooting、framework
  文档；
- 更新 MNIST 与 AFHQ README；
- 不从 public docs 链接本 development plan。

完成条件：

- 新用户能从文件目录直接判断 train 与 sample；
- sample 文件不出现 Builder；
- train 文件不出现 final sampler；
- generated reference 与 runtime schema一致。

### Phase H4：有限 multirun（可选后续）

首版禁用 Hydra BasicLauncher multirun，因为同一进程依次运行 job 会复用 Registry、
logger、RNG、CUDA 和 extension activation state。

只有满足以下条件才开放：

1. 在任何 job 执行前枚举有限 sweep；
2. 预组合并 typed validate 全部 configs；
3. 验证 extension selection/provenance 静态一致；
4. 在不写文件的隔离 preflight process 中激活共同 extension set，并为每个 job 完成
   Registry/native target 与静态 lifecycle preflight；
5. 任一 preflight 失败时不写 snapshot、不创建 output root、零 job 启动；
6. 全部通过后，先在内存生成每个 job 的 immutable config snapshot；
7. 预留并原子创建独立 output roots，再把对应 snapshot 写入该 job；
8. 每个 job 使用 process-isolated launch。

如果 Hydra stable public seam 不能支持 eager precomposition，继续禁用 `-m`。
Adaptive HPO 仍由独立计划负责。

## 12. 测试计划

### 12.1 Train/sample schema

- training parser 接受无 sampling 的 MNIST/AFHQ configs；
- training parser 拒绝任何 legacy `sampling`；
- sample parser 只接受 `sample` 与 optional `extensions`；
- sample config 拒绝 data/model/process/training/trainer/recipe；
- sample requires checkpoint and config；
- sample sampler/writers 原子解析；
- sample seed 与 repository profiles 中的 weights 必须显式；
- omitted device 解析为 `auto`，不继承 checkpoint trainer device；
- options 与 fixed recipe collision fail closed；
- required plugins只能来自 checkpoint provenance，additional plugins只能追加；
- old checkpoint format拒绝；
- new checkpoint不保存 mutable sampling defaults。

### 12.2 Diagnostics

- Gaussian/class-conditional diagnostic 从自己 params读取 shape；
- 删除顶层 sampling 后 diagnostic仍能运行；
- missing/invalid diagnostic shape 给出 diagnostic-local path error；
- diagnostic sampler不改变 final sample profile；
- disabled diagnostics不要求 shape。

### 12.3 Runner

- train 成功后不自动调用 sampling；
- 不再存在 skip-final-sample option；
- training selected-component manifest 不记录 sampler/writers，只记录 inference recipe；
- strict resume不读取 project train tree；
- resume overlay仍只修改 observability allowlist；
- sample config不进入 training manifest；
- sample manifest记录 checkpoint lineage、独立 invocation config、sampler/writers 与实际
  weights；不把 sample config merge 回 training config。

### 12.4 Hydra composition

- Defaults List order 与 `_self_` precedence；
- train/data/recipe/observability package paths；
- compose 后无 authoring wrapper keys；
- unknown/add/delete/controller override拒绝；
- resolver、`_target_`、`class_name` 与 arbitrary import path 拒绝；
- project root不能覆盖 trusted bootstrap；
- compose不导入 extension；
- `--cfg job --resolve` 与 `--check`；
- no cwd/log/output mutation；
- composed/effective digest；
- no-Hydra low-level runtime。

### 12.5 Native-provider boundary

- `torch.optim.<DirectPublicOptimizer>` 与
  `torch.optim.lr_scheduler.<DirectPublicScheduler>` 仍可由 retained train recipe声明；
- direct namespace、public identifier、expected base 与 reserved-prefix 规则受到测试；
- model/objective/process/training/diagnostic 的 `name` 继续只接受对应 Registry
  identity；各 role 的空 name 与错误 namespace 在 preflight 前后给出明确路径；
- nested/private/missing/wrong-base target 与 `os.system` 一类 arbitrary path 在 preflight
  失败；
- Registry 名不能冒充受限 native namespace；
- framework-owned `params` / `optimizer` 不能由配置覆盖；
- `LBFGS` 和 `ReduceLROnPlateau` 的必需 step 参数在无副作用 target preflight 中被明确
  拒绝；
- `--check` 解析 provider/family/target，但不调用 constructor、不展开上游 defaults；
- `LambdaLR` / nested scheduler target 可以解析，但 `--check` 必须显示
  `constructor: not_run`，不能报告为 runnable 或 supported；其 focused constructor
  test 给出清晰 role/path error；
- target typo 或静态 lifecycle mismatch 在 DataSource、artifact store 和 output
  creation 之前失败；
- low-level plain fresh train 与 Hydra fresh train 复用同一个 target preflight，并产生
  等价的 role/path error；
- resolved config 中除两个受限 grammar 外不出现 `_target_`、`class_name` 或 Python
  import path 驱动的 component declaration；
- base train recipe 不保留空 `logging.torch_logs`，非空 passthrough 的错误传播有 focused
  test。

### 12.6 Retained examples

| Example | 验收 |
| --- | --- |
| MNIST | one train；DDPM sample；DDIM-50 sample；resume；observability；short train |
| AFHQ ADM | production train；sample；resume；evaluation/capacity inputs |
| AFHQ DiT | production train；same sample schema；resume |
| AFHQ smoke | independent readable train recipe；short end-to-end |

Retired example names不得出现在 active example inventory、public quickstart 或 CI matrix。

### 12.7 Contract fixtures

删除 Physics/KD 后仍需覆盖：

- independent custom extension activation；
- independent custom DataSource/DataBuilder；
- referenced artifact producer；
- synthetic/no-artifact DataBuilder；
- custom TrainingBuilder/SamplingBuilder；
- auxiliary managed module；
- writer plugin。

每个 fixture只验证一个 contract，不形成完整 reference project。

## 13. 文档闭环

实施完成必须同步：

- root `README.md`；
- `docs/index.md`；
- `docs/framework.md`；
- `docs/configuration/index.md`；
- `docs/configuration/workflows.md`；
- `docs/configuration/reference.md` 与 generator source；
- `docs/configuration/compatibility-and-migration.md`；
- `docs/configuration/extensions.md`；
- `docs/configuration/troubleshooting.md`；
- MNIST README；
- AFHQ README；
- project scaffold docs；
- active development plans 的 example/gate matrix。

公开文档必须明确：

- train 与 sample 是两种配置；
- DDPM/DDIM 不产生两份 train；
- train 不自动 sample；
- sample 必须显式 checkpoint + config；
- checkpoint拥有 state 与 fixed inference recipe；
- sample config不选择 Builder；
- Hydra只用于 fresh train authoring；
- resume不重新 compose；
- arbitrary `_target_`、`class_name` 与 Python import path 不属于公共 authoring language；
- optimizer/scheduler native identifier 是明确受限的例外，不经过通用 import；
- Registry/Builder/native-provider仍是唯一 construction path；
- 每个 fresh training invocation 所选 component/native target 在 artifact I/O 前完成
  无副作用 preflight；
- `--check` 明确区分 target resolution 与未执行的 constructor/runtime validation；
- Physics/KD 已退出 maintained reference projects；
- 目前只维护 MNIST 与 AFHQ-v2 end-to-end examples。

开发计划不进入 public docs 导航。

## 14. 风险与缓解

### 14.1 去掉 sampling 后 diagnostics 失去 shape

风险：当前 factory 真实依赖 `config.sampling.shape`。

缓解：C1 先把 shape 移到 diagnostic-owned sampling params，并增加 focused tests；禁止从
model/data猜 shape。

### 14.2 删除 checkpoint sampling defaults 后 sample 文件变得不完整

风险：当前 MNIST/AFHQ sample request 依赖 checkpoint default。

缓解：C0 冻结最终 resolved sampling value；C1 将每个 retained profile展开为完整
`SampleConfig`，再删除 merge path。

### 14.3 Train/sample 分开后失去“一条命令出图”

风险：用户训练后还要显式 sample。

缓解：这是有意的 lifecycle boundary。README 提供两条连续命令；未来若需要 orchestration，
增加显式 workflow command，而不是把 sampler重新塞回 train schema。

### 14.4 Hydra 把一份文件拆成更多文件

风险：行数减少但理解成本上升。

缓解：一层 defaults、group 数预算、cohesive recipe、resolved preview、人工 readability
gate。没有第二个真实复用点时不拆。

### 14.5 删除 Physics/KD 导致 extension coverage 下降

风险：完整 extension paths 不再由两个 project覆盖。

缓解：AFHQ承担真实 installed extension；其他 contract用独立最小 fixtures验证。避免用
一个复杂 example同时承担十几个 core contract。

### 14.6 Retired dataset capability 状态含混

风险：删除 runnable config，但 docs仍宣称 CIFAR/Flowers end-to-end support。

缓解：implementation时明确 source-only support或完全删除；reference、tests和README必须
匹配，不保留模糊“可能还能跑”的承诺。

### 14.7 Hydra 改变 freeze 语义

风险：resume/sample被普通 deep merge或 dotted override改写。

缓解：Hydra根本不进入这些 workflows；freeze由 authority base 和 allowlist实现，不依赖
OmegaConf readonly。

### 14.8 两套输出与日志

风险：Hydra和Stochaflow都创建 run。

缓解：禁用 Hydra output/logging/cwd mutation，Stochaflow唯一拥有 official run。

### 14.9 Extension 在 composition 期被导入

风险：SearchPathPlugin或resolver绕过 provenance preflight。

缓解：trusted bootstrap、explicit project root、static scanner、无 runtime extension
config provider、无 custom resolver。

### 14.10 Multirun 共享进程状态

风险：BasicLauncher复用 Registry/logger/RNG/CUDA。

缓解：首版禁用；仅在 eager precomposition + process isolation 后开放。

### 14.11 Native identifier 被误认为通用 import path

风险：`torch.optim.AdamW` 的表面形式与 Python dotted path 相同，用户或后续维护者可能
据此把 resolver 扩大到 `torch.nn.*`、`torchvision.*` 或任意 `module.Class`，绕过
Registry、Builder compatibility、extension provenance 和 trusted preflight。

缓解：语法只承认
`torch.optim.<DirectPublicOptimizer>` 与
`torch.optim.lr_scheduler.<DirectPublicScheduler>` 两个精确 role-scoped prefix；
resolver 不调用 `importlib`，Registry 不得占用这些 prefix。新增第三种 native provider
必须单独通过本计划 6.3 的六项准入条件、contract tests 和新的 architecture decision，
不能通过放宽现有 parser 顺带获得。

另一个风险是 target 虽然通过 namespace/base preflight，却因 callable、嵌套实例或普通
constructor 参数错误在数据物化之后才失败。首轮 preflight 必须覆盖可从 public
`step()` contract 静态判断的 closure/metric lifecycle mismatch；不能静态证明的
constructor 与 Builder-level compatibility 在 `--check` 输出中明确标成未执行，而不是
宣称配置“可运行”。首轮明确接受这个有限差异，不引入 per-class allowlist 或 dummy
construction。

## 15. 明确不采用的替代方案

### 15.1 保留 DDPM/DDIM 两份 train，仅用 Hydra 去重

仍然把 sampler choice写成 training identity，拒绝。

### 15.2 Train config 只保留一个默认 sampler

虽然减少为一份 train，但仍让一个任意 sampler进入 checkpoint training config，并让
用户误以为该 checkpoint“是 DDIM checkpoint”。拒绝。

### 15.3 保留 auto final sample，通过引用 sample profile

这会让 train 的成功语义、资源释放、错误状态和 output lifecycle继续依赖另一个 workflow。
首版拒绝；需要时由显式 orchestration解决。

### 15.4 Sample config 继续作为 partial overlay

没有 checkpoint mutable defaults 后，partial overlay只会依赖隐藏 framework default。
首版使用完整、可读的 invocation config。

### 15.5 用 Hydra compose sample

当前只有 MNIST/AFHQ少量 profile，未证明需要另一套 composition graph。首版 plain schema
更清楚。未来只有 sample profiles出现真实、危险的重复后再评估。

### 15.6 Hydra instantiate

会产生第二套 Registry并绕过 Builder、plugin provenance和 lifecycle，拒绝。

### 15.7 通用 `class_name` 或 Torch namespace 镜像

不新增 PhysicsNeMo 风格的通用 `class_name`、arbitrary dotted-path resolver，也不把
`torch.nn`、`torch.utils.data` 或 `torchvision` namespace 镜像成 Stochaflow 配置 API。
上游基类本身不能表达 model forward、objective reduction、dataset partition、
sampler/resume 或 DataLoader lifecycle compatibility；允许任意构造只会把错误从
composition root 推迟到运行期。

optimizer/scheduler 的两个受限 native identifier 是精确的 role provider exception，
不是该方案的第一阶段。未来 family 必须按 6.3 的准入条件单独决策。

### 15.8 保留 Physics/KD 但标记“无人维护”

文件仍会腐化、占用 CI/文档心智并制造支持承诺。active tree直接删除；有价值的历史结论
留在 development record，contract coverage转为 fixtures。

## 16. Decision Gates

| Gate | 必须满足 | 失败时 |
| --- | --- | --- |
| G0 Boundary fit | 一份 MNIST train可支持两种 sample；diagnostic无顶层 sampling依赖 | 不引入 Hydra |
| G1 Scope fit | 仅 MNIST/AFHQ active；retired docs/tests清理闭环 | 不开始 example composition |
| G2 Composition fit | MNIST typed parity、无 extension import、可读性改善 | Hydra停留在 internal spike |
| G3 Native-provider fit | 仅两个精确 native grammar；无 generic importer；fresh-train selected targets 在 artifact I/O 前完成身份与静态 step contract preflight；constructor 未执行状态可见 | 不发布 Hydra launcher |
| G4 Runtime fit | AFHQ ADM/DiT/smoke parity，run/checkpoint不漂移 | 不切换 public authoring |
| G5 Authoring fit | 用户入口一层 defaults、preview/error UX可接受 | 合并 groups或保留完整 train leaf |
| G6 Extension fit | explicit config root足够；AFHQ installed-wheel正常 | 不实现 external config provider |
| G7 Multirun fit | 全 job eager precompose/target preflight 零写入通过、process isolation、static plugins、atomic outputs | 继续禁用 `-m` |
| G8 Breaking cutover | configs、checkpoint version、docs、scaffold一起迁移 | 不删除旧 authoring surface |

## 17. 最终验收

迁移完成必须同时满足：

1. Repository只有 MNIST 与 AFHQ-v2 两套 maintained end-to-end examples。
2. MNIST只有一份 train config。
3. DDPM和DDIM只出现在 sample profile或training diagnostic中，不出现在 train identity。
4. Train config不含顶层 sampling、sampler或writers。
5. Sample config不含 model/data/process/training或Builder。
6. Train不自动 sample。
7. EMA training config不含 sampling preference；sample weights选择显式且可审计。
8. Checkpoint不保存 mutable sample defaults，但保存 fixed inference recipe。
9. Strict resume frozen assets/state没有 Hydra bypass。
10. Hydra只负责 fresh train authoring。
11. Public config 不接受 Hydra `_target_`、`class_name` 或任意 Python import path。
12. 唯一 native target grammar 是 `torch.optim.<DirectPublicOptimizer>` 与
    `torch.optim.lr_scheduler.<DirectPublicScheduler>`；它们不经过通用 import。grammar
    只是必要条件，closure-required 与 metric-driven `step()` 在 artifact I/O 前拒绝；
    callable/nested constructor requirement 不属于 plain-config contract，且
    `--check` 不宣称已执行 constructor validation。
13. Registry/Builder/受限 native provider仍是唯一 construction path。
14. 每个 fresh training invocation 所选 Registry 名称、extension provenance 与 native
    target 在 artifact I/O 和 official output creation 前完成无副作用 preflight。
15. `--check` 区分“已完成 composition/schema/target resolution”和“未执行
    constructor/data/runtime compatibility”，不做过度成功承诺。
16. Runtime-only facts与resolved train config分离。
17. Stochaflow是唯一 official run/log/checkpoint owner。
18. `resolved_config.yaml` 完整、无 Hydra internals、适合事故审计。
19. AFHQ sample/evaluation/capacity authority保持独立。
20. Physics/KD不再出现在 active CI、public docs或example tree。
21. Generated reference、scaffold、README和公开文档同步。

建议最终验证：

```bash
uv run python tools/generate_config_reference.py
uv run python tools/generate_config_reference.py --check

uv run pytest \
  tests/test_config.py \
  tests/test_experiment_runner.py \
  tests/test_sampling_runtime.py \
  tests/test_checkpoint.py \
  tests/test_extension_plugins.py \
  tests/test_config_reference.py \
  tests/test_data_builder.py \
  tests/test_data_sources.py \
  tests/test_afhq_v2_showcase.py \
  tests/test_afhq_v2_evaluation.py \
  tests/test_project_scaffold.py

uv run ruff check .
uv run pyright
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
uv run pytest
```

Hydra implementation引入后增加独立 composition/launcher tests，不再把 retired example
cases加入 focused command。

## 18. 官方调研来源

- [Hydra Defaults List](https://hydra.cc/docs/advanced/defaults_list/)
- [Hydra packages and `_global_`](https://hydra.cc/docs/advanced/overriding_packages/)
- [Hydra override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Hydra structured config schema](https://hydra.cc/docs/tutorials/structured_config/5_schema/)
- [Hydra Compose API](https://hydra.cc/docs/advanced/compose_api/)
- [Hydra config search path](https://hydra.cc/docs/advanced/search_path/)
- [Hydra plugins](https://hydra.cc/docs/advanced/plugins/overview/)
- [Hydra 1.3 object instantiation and dotpath lookup](https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/)
- [Hydra working directory and output](https://hydra.cc/docs/tutorials/basic/running_your_app/3_working_directory/)
- [Hydra multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [hydra-core PyPI releases](https://pypi.org/project/hydra-core/)
- [Hydra GitHub releases](https://github.com/facebookresearch/hydra/releases)
- [PyTorch optimizer and scheduler contracts](https://docs.pytorch.org/docs/stable/optim.html)
- [PhysicsNeMo DFSR user configuration](https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/flow_reconstruction_diffusion/conf/config_dfsr_train.yaml)
- [PhysicsNeMo DFSR trusted Python composition](https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/flow_reconstruction_diffusion/train.py)
- [PhysicsNeMo model module registry](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/models/modules.html)
- [LightningCLI optimizer and scheduler configuration](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html)
- [Lightning automatic and manual optimization lifecycles](https://lightning.ai/docs/pytorch/stable/common/optimization.html)
