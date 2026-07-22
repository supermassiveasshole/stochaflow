# 自定义代码扩展支持实施计划

- 状态：Stage 5 完成，Stage 6 待实施
- 制定日期：2026-07-17
- 最近修订：2026-07-22
- 目标分支：`feature/custom-code-extension-support`

## 目标

将 Stochaflow 从包含内置实验能力的 Python 包扩展为“核心框架 + 可复用组件 +
用户项目扩展”的训练与采样框架。用户安装 Stochaflow 后，可以：

1. 直接组合内置数据 recipe、模型、训练策略，以及算法需要的 Process 和 Sampler；
2. 只替换任务特有的部分，例如条件模型、数据组织、guidance 或 sampling builder；
3. 注册全新的过程、数值采样算法或训练方法，而无需修改核心分发逻辑；
4. 继续通过统一 CLI 和 YAML 完成训练、恢复与采样。

目标用户流程：

```text
安装 stochaflow
→ stochaflow init my-project
→ cd my-project
→ 在所选 Python 环境中安装项目（例如 pip install -e . 或 uv sync）
→ 编写 extension、data 和 config
→ stochaflow train --config experiments/example/train.yaml
→ stochaflow sample --checkpoint outputs/.../checkpoints/best.pt
```

生成目录是普通、可安装的 `src` 布局 Python 项目。Stochaflow 发布包提供
CLI，extension 项目通过标准 Python packaging 安装到同一环境。pip、uv、
venv、conda、Poetry 或 PDM 都可以；核心不识别 workspace 也不管理环境。

## 架构原则

### Open–Closed Principle

Stochaflow 核心对扩展开放、对任务特例关闭。一个与现有契约兼容的新 DataBuilder、
Process、Sampler、SamplingBuilder、Objective 或 TrainingBuilder，应当只需新增实现、
注册和配置，不应修改核心 runner、CLI 或按名称分支的 dispatch 代码。

具体约束：

- 核心契约只表达生命周期和组件间真正共有的语义，不收纳图像、super-resolution、
  condition、guidance 等任务字段；
- 不为追求“统一”而建立包含大量可选方法和模式枚举的万能基类；差异通过组合、窄
  capability protocol 和边界校验表达；
- 核心不得根据注册名或具体实现类型执行算法分支；组件兼容性由构建它们的 Strategy
  或 Builder 在边界处检查；
- 配置中的 `params` 归具体组件所有。顶层 schema 只负责选择组件，不复制每类组件的
  全部参数；
- 内置便利组件与用户组件使用相同注册和构建路径，不享有隐藏的核心特权；
- 任何必须修改公开扩展边界的施工，先修改本计划，再进入实现。

### 分层边界

数据与生成算法采用相同的分层思路：核心只定义最小的组合边界，任务语义由扩展拥有。

```text
DataBuilder
  └─ directly assembles Dataset / split / transform / Sampler / DataLoader / collate
       └─ returns DataLoaders and arbitrary structured batches

Framework: Registry / config / checkpoint / train and sample lifecycle
  └─ selects registered components without requiring one universal algorithm shape

Training task: TrainingBuilder -> TrainingPlan -> TrainingStrategy
  └─ assembles managed assets, then delegates only step computation to Strategy

Algorithm family: optional Process + family Dynamics + optional compatible Sampler
  └─ defines cohesive mathematics only within that family

Task: SamplingBuilder + model callable + condition / guidance / initial state
  └─ assembles and executes one compatible sampling workflow
```

- **DataBuilder** 是唯一的核心数据扩展入口，拥有完整数据组织逻辑；核心 Trainer 只
  消费组装好的 loader 和原样 structured batch。
- **Process** 的公共根只负责 Registry、device 与 checkpoint 生命周期，不定义万能概率
  API。它是可选的算法资产：Gaussian、flow matching 或 SDE family 可以为 model-free
  probability path 定义内聚契约；没有这类路径的方法不得为了核心 dispatch 虚构 Process。
  Process 不依赖模型 callable、prediction policy 或 Dynamics 类型，也不执行采样循环。
- **GenerativeDynamics** 只是“已组装生成方向”的无行为语义根，不代表不同算法 family
  数学兼容。reverse kernel、vector field、reverse SDE 或 denoiser function 应是各 family
  的窄契约，不设全局 Registry，也不向根类型添加统一求值方法。
- **Sampler** 是完整的数值执行算法，负责时间/噪声离散、solver state、随机增量、
  多步历史和循环。它不解释 condition，也不拥有业务模型。
- 不需要数值求解循环的直接生成变换可以由 SamplingBuilder 执行，不为满足框架形状虚构
  Sampler；统一 `Sampler.sample()` 只约束实际选择了 Sampler 的 workflow。
- **TrainingBuilder** 组装训练依赖并返回 TrainingPlan；核心验证 Plan 并管理
  其资产 lifecycle。**TrainingStrategy** 只定义 batch 到 loss/metrics 的训练计算。
- **SamplingBuilder** 是推理任务组合层，负责模型适配、condition/guidance、
  initialization 和 family 兼容性检查。

这样的划分吸收两类现有实现的优点，而不复制它们的耦合：OpenAI diffusion 将高复用的
Gaussian math 集中在 `GaussianDiffusion`，但也合并了训练 Loss、模型输出解释和采样
循环；Diffusers Scheduler 支持广泛的 DDPM、DDIM、SDE/ODE 与 flow-matching solver，
但一个 Scheduler 通常同时承担部分 Process、推理时间表和 reverse step。Stochaflow
统一扩展生命周期，同时让每个算法 family 自己定义 Process、Dynamics 与 Sampler 数学，
优先保证 family 内复用，而不是制造跨 family 的伪兼容。

首版默认边界：

- 使用本地 `src` 布局项目；
- extension package 通过标准 Python entry point 暴露单一聚合入口；脚手架生成的
  YAML 显式选择该插件，不扫描目录，也不依赖隐式环境全量加载；
- Dataset 单样本与 batching 由 DataBuilder 按需定义；送入 Trainer 的 batch 可以是
  `Tensor`，或由 mapping、tuple、list 和 Tensor 组成的嵌套结构；
- 提供开箱即用的普通图像、多分辨率图像和 paired super-resolution recipe，但这些
  recipe 的参数不是核心数据契约；
- 首版 TrainingStrategy 支持单 primary model、可选 Objective、一个优化器和
  一次反向传播；TrainingBuilder 可组装 teacher 等核心托管的辅助模块；
- 首版训练与采样仍要求一个 primary inference model；Process 则是可选组件，由具体
  TrainingBuilder 或 SamplingBuilder 决定是否需要；
- 核心 Trainer 管理训练循环、反传、调度、日志和 checkpoint；
- 扩展相关 schema 直接使用新格式，不提供 legacy 迁移或兼容别名。

## Stage 1：统一扩展注册与配置入口（已完成）

### 目标

为所有可配置组件建立统一、语义明确的全局扩展入口。

### 已实施

- 新增顶层 `extensions.modules`，在组件构建前按声明顺序导入；
- resolved config 和 checkpoint 保存最终扩展模块列表；
- 新增稳定的 `stochaflow.extensions` 公开入口；
- 保持 Registry 的重复注册、类型校验和异常语义。

Stage 5 会在正式发布前用 packaging entry point 和 `extensions.plugins` 替换这里的
module-path bootstrap。该未发布接口不保留兼容层。

### 逻辑提交

`Add global extension configuration`

## Stage 2：薄 DataBuilder 与任务数据 Recipe（已完成）

### 目标

保留一个非常薄的注册化数据构建入口，删除核心“统一 Dataset、split、sample key、
metadata、sampler、bucket 和 DataLoader”的企图。用户在 Python 中直接组装标准 PyTorch
对象；YAML 只选择一个 recipe/builder 并传递它自己的参数。

原 `DataPipeline`、`DataBundle`、`SplitData`、`DatasetFactory`、`DatasetView` 和通用
split/mixture 系统已由本节的新契约替换，没有保留兼容层。

### 核心契约

```python
@dataclass(frozen=True, slots=True)
class DataBuilderContext:
    params: dict[str, Any]
    seed: int


@dataclass(frozen=True, slots=True)
class DataLoaders:
    train: Iterable[Any]
    validation: Iterable[Any] | None = None
    test: Iterable[Any] | None = None
    steps_per_epoch: int | None = None


class DataBuilder(ABC):
    def __init__(self, context: DataBuilderContext) -> None: ...

    @abstractmethod
    def build(self) -> DataLoaders: ...
```

- `StochaflowConfig.data` 继续是 `ComponentConfig`，但由
  `REGISTRIES.data_builders` 构建；
- `DataBuilderContext` 只提供复制后的 `params` 与实验 seed；
- `DataLoaders` 只表达 Trainer 真正需要的 train/validation/test 角色；
- loader 必须可重复迭代；一次性 generator/iterator 应包装为每次 `__iter__()` 创建新
  iterator 的对象；
- `steps_per_epoch` 只为没有 `len()` 的 iterable/streaming train loader 提供有限 epoch；
  有 `len()` 时可以省略；显式值必须为正；
- builder 每次只描述一次训练运行，不返回 `list[DataBundle]`，核心不理解 fold index 或
  在一次配置中隐式展开多次训练；k-fold 通过显式 fold 参数、多个运行或后续 sweep
  orchestration 表达；
- runner 对 Dataset、Sampler、BatchSampler、collate 和 DataLoader 实现不做类型检查，
  只验证 loader 可迭代和 epoch 可终止；
- reporting 对 `len(loader)`、`loader.dataset`、`batch_size` 等信息只做 best-effort
  introspection，不把这些属性变成扩展契约；
- 核心把 batch 视为 `Any`，递归 device transfer 后原样交给 TrainingStrategy；数据层不
  负责证明 batch 与模型兼容；
- `sampling.shape`、batch size 和 artifact writer 与 data builder 完全独立。

自定义数据的标准写法是直接组装 PyTorch：

```python
@REGISTRIES.data_builders.register("physics")
class PhysicsDataBuilder(DataBuilder):
    def build(self) -> DataLoaders:
        dataset = StreamingPhysicsDataset(...)
        sampler = PhysicsSampler(dataset, ...)
        loader = DataLoader(dataset, sampler=sampler, collate_fn=physics_collate)
        return DataLoaders(train=loader, steps_per_epoch=1000)
```

```yaml
data:
  name: physics
  params:
    path: data/simulation.zarr
    batch_size: 8
```

核心不分别注册通用 Dataset、Sampler、DataLoader、collate 或 split。建立这些全局
Registry 会把 Python 对象拓扑搬进 YAML、制造组合兼容矩阵，并迫使框架进行不可靠的
跨字段校验。

### 内置任务 Recipe

内置能力是自包含、可读、可修改的 DataBuilder，不是万能数据编排框架。

#### `image`

```yaml
data:
  name: image
  params:
    source:
      kind: torchvision
      dataset: CIFAR10
      root: ./data
      download: true
    partition:
      mode: holdout
      validation_size: 5000
    image:
      size: [32, 32]
      channels: 3
      normalize: true
    loader:
      batch_size: 128
      steps_per_epoch: auto
```

负责普通图像的 source、train/validation/test、resize/crop/flip/normalize、collate 和
DataLoader。默认 batch 约定采用 OpenAI 风格：`images, {}`。

#### `super_resolution`

```yaml
data:
  name: super_resolution
  params:
    source:
      kind: image_folder
      path: ./data/images
    image:
      high_resolution: [256, 256]
      low_resolution: [64, 64]
      channels: 3
      normalize: true
    low_resolution:
      kind: bicubic
    loader:
      batch_size: 16
      steps_per_epoch: auto
```

负责对齐 crop、HR target、LR condition、默认 degradation 与 collate。默认 batch 约定为
`high_res, {"low_res": low_res}`。后续配套的 diffusion TrainingStrategy 才解释该约定；
核心 DataBuilder、Trainer、Process 和 Sampler 均不理解 `low_res`。

#### `multi_resolution_image`

保留多源权重、resolution bucket、同 bucket batch、动态像素 budget 和 deterministic
`set_epoch`，但全部是这个高级图像 recipe 的私有实现。bucket、sample key、selection 和
metadata 不进入公共 DataBuilder 契约。

不再提供试图覆盖任意 map-style 数据拓扑的万能 `map` pipeline。普通非图像任务通过很
小的用户 DataBuilder 直接使用 PyTorch Dataset/DataLoader；文档提供最小模板而不建立
新的通用配置层。

### 私有图像 Source Helper

三个内置 recipe 共享普通 Python helper，不建立二级 Registry。Stage 2 只支持 MNIST、
CIFAR10、Flowers102 的 torchvision source、本地 `image_folder`，以及 super-resolution
使用的 `paired_folders`。网络下载只通过 torchvision 的 `download: true` 提供；URL/archive、
Hugging Face Dataset 和 WebDataset 留给自定义 DataBuilder 或后续真实需求。

### 删除与下沉

- 移除 `REGISTRIES.data_pipelines` 和 `REGISTRIES.dataset_factories`，由
  `REGISTRIES.data_builders` 替代，不建立二级公共 Registry；
- 从 `stochaflow.extensions` 移除 `DataBundle`、`SplitData`、`DatasetFactoryContext`、
  `DatasetBuildRequest`、`DatasetView` 等旧契约；
- 删除通用 `DatasetMaterializer`、`DatasetSelection`、split policy、mixture、sample key
  与 batch metadata 公共系统；
- 若内置 recipe 仍需要其中某段算法，将其移入对应 recipe 的私有模块，并以任务术语
  命名；
- runner 直接消费一个 `DataLoaders`，去除 fold-aware 输出目录、resume scope 和
  `DataBundle` 分支；
- resolved config 和 checkpoint 只保存最终 `data: {name, params}`，不序列化 Dataset、
  Sampler 或 DataLoader runtime state。
- checkpoint 格式升级到 v4；训练恢复和 checkpoint-only sampling 直接拒绝 v3。

### 验收条件

- 自定义 physics builder 只注册一个类即可返回自定义 Dataset/Sampler/DataLoader，配置
  不描述它们的内部拓扑；
- bare Tensor、mapping、tuple/list、condition dict 和自定义对象 batch 能原样到达
  TrainingStrategy；
- 可重复 iterable loader 使用 `steps_per_epoch` 工作，缺少有限 epoch 信息或直接返回
  一次性 iterator 时明确失败；
- `image` 可从 torchvision 与本地文件夹训练，split、transform、worker seed 和默认
  batch 约定有测试；
- `super_resolution` 保证在线 bicubic 与 paired folder 的 HR/LR 对齐，并输出
  `high_res, {"low_res": low_res}`；
- `multi_resolution_image` 回归多源权重、bucket 同质性、动态 batch 与 `set_epoch`；
- 未覆盖的 source 或数据组织通过自定义 DataBuilder 接入，不扩充核心配置拓扑；
- 旧 DataPipeline/DatasetFactory 公共符号和配置直接失败，不提供兼容层；
- 数据配置、公开 API、README、扩展手册、配置 reference 和故障排查同步更新；
- 配置 reference 重新生成，`uv run pytest`、`uv run ruff check .` 和 `uv run pyright`
  全部通过。

### 逻辑提交

`Simplify data builder abstraction`

该提交是对已提交但未通过架构验收的 `Refactor modality-neutral data pipeline` 的替代性
重构；不在新契约中保留其公共类型兼容层。

## Stage 3：可注册 Process、Generative Dynamics 与 Sampler（已完成）

### 目标

拆除当前 `GaussianDiffusion` 对 probability process、模型调用、prediction type、reverse
transition 和 sampling loop 的捆绑，并消除 `REGISTRIES.diffusions` 同时选择训练算法与
采样器的语义重载。

Stage 3 建立 Registry、checkpoint、Builder 与完整 `Sampler.sample()` 的框架生命周期，
并为 Gaussian diffusion 建立首个 family-specific Process/Dynamics/Sampler 契约。它不声称
Gaussian、score SDE、probability-flow ODE、sigma-space solver 和 flow matching 共享
同一数学接口；首批只迁移并验证现有 DDPM/DDIM。

### Registry 与配置

新增：

- `REGISTRIES.processes`：model-free probability process；
- `REGISTRIES.samplers`：完整数值采样算法；
- `REGISTRIES.sampling_builders`：任务级推理组合器。

移除 `REGISTRIES.diffusions`，不提供兼容别名。建议配置：

```yaml
process:
  name: discrete_gaussian
  params:
    schedule:
      name: linear_beta
      params:
        num_timesteps: 1000

sampling:
  shape: [3, 64, 64]
  batch_size: 16
  builder:
    name: standard_denoising
    params:
      sampler:
        name: ddim
        params:
          num_inference_steps: 50
          eta: 0.0
  writers:
    - name: tensor
      params: {}
```

顶层 `process` 是可选的概率路径身份：`ComponentConfig | None`。Gaussian diffusion 等
需要 model-free probability path 的 Strategy/Builder 在自己的组合边界要求它；直接生成
变换或其他不需要 Process 的 family 可以省略。`sampling.builder` 描述一次推理任务如何
组合模型、可选 Process、Dynamics 和 Sampler。用户可以覆盖 builder/sampler 的推理参数
而不修改 checkpoint 中已有的 Process，但 Builder 必须验证自己需要的 family capability。

### Process 契约

`Process` 是可注册、可配置、可迁移到 device 且可进入 checkpoint 的最小基类。它不拥有
模型、不读取 structured batch、不编码 condition，也不执行训练或采样循环。

避免要求所有 Process 实现一套臃肿接口。Stage 3 只公开一个内聚的
`DiscreteGaussianDenoisingProcess`，描述离散 Gaussian 路径的 clean/terminal time、terminal
prior、marginal、adjacent posterior 和 state-time validation。第三方只需实现这一个
Process 接口即可复用现有 Gaussian 训练桥接与 DDPM/DDIM；其他数学范式由真实实现需求
驱动新的 Process 类型，不预先拆出 marginal、posterior、prior 等细粒度公共 capability。

该名称刻意包含 `Discrete`：整数 state time 与 adjacent posterior 不是所有 Gaussian
denoising process 的共同能力。连续 Gaussian SDE 或只消费 marginal 的第二种实现出现后，
再由真实消费者驱动窄 capability 拆分；Stage 3 不让它们被迫实现 DDPM 专属 API。

内置 `DiscreteGaussianProcess` 从当前实现中保留：

- beta/alpha/noise schedule 与 timestep 定义；
- `add_noise`、marginal 系数和 posterior math；
- 为 DDPM/DDIM Dynamics 提供 marginal scales 与 posterior 数学量。

它不拥有 model、prediction parameterization、clipping、Loss 或 sample loop。
epsilon、x0、v、score 转换以及 clipping 后 epsilon 重算属于
`GaussianModelDynamics`。

Gaussian noise schedule 是 `DiscreteGaussianProcess` 的私有注册化组合，不扩展为
Flow Matching 的万能 schedule。契约分为三层：

```text
GaussianNoiseSchedule
└── DiscreteVPSchedule                  数学 capability
    └── TabulatedDiscreteVPSchedule     固定 coefficient table 实现
        ├── LinearBetaSchedule
        └── CosineAlphaBarSchedule
```

`DiscreteVPSchedule` 只产生与 `state_times` 同形的 `GaussianScales` 和
`DiscreteVPCoefficients`；Process 负责按样本 rank broadcast。Stage 3 采用“构造期固定
快照”语义：`DiscreteGaussianProcess` 在构造时从 schedule 生成唯一权威 coefficient
table，并将 marginal 与 posterior 所需量作为自己的 buffer 保存；运行期数学不再查询或
缓存一个可被独立修改的 schedule 子模块。第三方 schedule 可以用解析公式产生快照，但
当前契约不支持训练中可变或可学习 schedule。未来 learned schedule 需要新的动态
coefficient capability，不能静默复用该实现。

### Generative Dynamics 边界

“forward process”与“reverse process”不足以覆盖全部算法。DDPM/DDIM 中生成方向表现为
Gaussian denoising prediction 与 reverse transition；在 SDE/ODE 中可能是 reverse SDE
或 probability-flow vector field；flow matching 则直接学习 velocity。
`GenerativeDynamics` 只作为这些对象的无行为语义根，不定义它们共同的数学操作。

首版 Dynamics 是由 Builder/diagnostic 将 Process、模型 callable 和 prediction adapter
组合出的普通对象或 family-specific 抽象能力，不设全局 Registry，也不出现在顶层 YAML。
Process 不提供 Dynamics 工厂方法。不同 family 可按真实实现需要定义：

- `ReverseKernel`：给出离散 `x_t → x_s` 的分布或一步 sample；
- `VectorField`：给出 `dx/dt`；
- `ReverseSDE`：给出 reverse drift 与 diffusion；
- `DenoiserFn`：在 sigma-space 返回 denoised state。

这些名称是设计示例，不是首版公开类型或待实现空壳。Sampler 在调用边界声明并验证自己
消费的 family capability；SamplingBuilder 负责构造兼容组合。核心不维护
`process × sampler` 名称兼容矩阵，也不要求一种 capability 适配其他 family。

Stage 3 的 Gaussian 实现将 `GaussianDenoisingDynamics` 定义为 Sampler 依赖的抽象能力，
并以 `GaussianModelDynamics` 作为普通模型 callable 的具体 adapter。DDPM/DDIM 只依赖
前者；standard builder、diagnostic 或用户 workflow 构造后者，也可以提供自定义
Dynamics 实现或 wrapper 来表达 CFG、condition 与 physics guidance。

### Sampler 契约

Sampler 是一次采样调用内的完整数值执行器，而不是强制统一成 `step()` 的 Scheduler。
它需要支持：

- sampler-owned inference time/sigma grid，与训练 Process schedule 分离；
- 初始化状态、随机数生成器、device/dtype 和可选回调；
- 多次模型求值、multistep history、自适应 step 和随机增量；
- 每次 `sample()` 独立的临时 solver state，不把运行历史写入 checkpoint；
- 返回最终生成状态，并允许 Builder 附加 trajectory 或任务产物。

具体 Sampler 可以提供公开 `step()` 作为便利能力，但它不是所有 solver 的强制基类方法。
首批迁移：

- `DDPMAncestralSampler` 消费 Gaussian reverse-kernel capability；
- `DDIMSampler` 消费 Gaussian deterministic/stochastic transition capability；
- 原有 DDPM/DDIM 数学结果、partial denoising 和 deterministic seed 行为保持一致。

后续新增 Euler–Maruyama、Euler、Heun、LMS、DPM-Solver、UniPC 或 flow-matching solver
时，应在所属算法 family 内新增或复用 Process、Dynamics capability 与 Sampler，不修改
现有 Gaussian 契约和核心 runtime。

### SamplingBuilder 契约

SamplingBuilder 是任务级开放扩展点，负责：

- 选择 checkpoint 中用于推理的模型或 EMA 模型；
- 将任意模型签名包装为 Sampler 可消费的 callable；
- 定义 epsilon/x0/v/score/velocity 等 prediction semantics；
- 注入 condition、classifier-free guidance、physics guidance 或外部控制；
- 构造初始状态，例如纯噪声、从观测 partial noising 或用户 latent；
- 将 Process、model adapter 与 Sampler 组合为一次 sampling run；
- 将运行结果规范化为 writer 可消费的 sampling batch。

Process 和 Sampler 的 API 不增加 `condition` 或 task-specific keyword。无条件 diffusion、
conditional super-resolution、inpainting 和 physics-guided reconstruction 通过不同 Builder
或用户 Builder 扩展，同时复用相同 Process 和 DDIM/DDPM Sampler。

### Checkpoint 与 runtime

- checkpoint 格式升级到 v5，保存 `model_state_dict`、可选 `process_state_dict` 和可选
  `ema_model_state_dict`，直接拒绝 v4；
- 配置存在 Process 时，它是 checkpoint-authoritative 组件，可以包含 buffer 或可学习
  状态；配置省略 Process 时，runtime 不构造、不保存也不恢复占位 Process；
- sampler 和 SamplingBuilder 是可覆盖的推理配置，不保存一次运行的 solver history；
- checkpoint-only sampling 先加载扩展，再构建推理模型、可选 Process、Builder 和 Sampler；
- sampling runtime 不再假设 sampler 返回单一 Tensor，由 Builder 规范化结果后交给 writer；
- `sampling.shape` 继续可选；固定 shape 的 Builder/Sampler 在运行时自行要求；
- diagnostic 不依赖具体 DDPM/DDIM 类，只声明自己需要的 Process/Dynamics capability。

### 已完成的核心实现

- framework-level `Process` 与 `GenerativeDynamics` 根类型已保持无数学行为；Gaussian
  Process 与 Dynamics 契约分别进入 family 模块，公开扩展导出不变；
- Gaussian schedule 已拆分为 `GaussianNoiseSchedule`、`DiscreteVPSchedule` capability
  与 `TabulatedDiscreteVPSchedule` 实现；Process 只消费具名 coefficient 对象，且明确
  拒绝当前不支持的可学习 schedule；
- `DiscreteGaussianProcess` 独立拥有构造期 coefficient snapshot、marginal 和 posterior
  数学，不保留 schedule 子模块，不持有模型、prediction semantics，也不创建 Dynamics；
- `DiscreteGaussianDenoisingProcess` 已形成单一离散 Gaussian 路径接口，训练 factory、
  standard builder 和 diagnostic 不按 Process 注册名或具体实现分支；临时训练桥接自行
  拥有 timestep sampling；
- `DDPMAncestralSampler` 与 `DDIMSampler` 均只实现完整 `sample()` 生命周期，不要求
  universal `step()`，solver 参数和求解区间留在各自构造配置；Builder/diagnostic 根据
  实际 observation 验证 terminal-to-clean 生命周期，partial denoising 仍可由自定义
  Builder 或 Sampler 直接调用；
- `GaussianDenoisingDynamics` 是 sampling 层的抽象 capability，
  `GaussianModelDynamics` 是由 Builder/diagnostic 显式构造的 model adapter；Process 层
  不依赖任一具体 Dynamics；
- `SamplingObserver`、`SamplingObservation` 和 `TrajectoryObserver` 统一 initial、
  accepted step 与 final observation，删除额外 Snapshot 类型和旧的四套
  sample/trajectory 方法；Sampler 直接创建并发送 observation，Observer 决定筛选、
  复制与保存；
- `standard_denoising` builder 负责 raw/EMA 选择、initial prior、model callable、
  Dynamics、Sampler、batching 与 trajectory，核心 runtime 只调用一次 `run()` 并校验
  `SamplingOutput`；
- Gaussian epsilon 训练通过内部 capability bridge 继续工作，optimizer 覆盖 model 与
  Process，EMA 只跟踪 inference model；
- diagnostics、standalone sampling 和训练后 sampling 已迁移到同一 Sampler/observer
  契约；trajectory writer 按 step index 与 coordinate 的声明顺序保存；
- 显式 checkpoint 可搭配只含 `sampling` 与可选 `extensions` 的轻量覆盖文件；完整外部
  配置仍会校验 model 与 Process 兼容性；
- 移除 `REGISTRIES.diffusions`、旧 `stochaflow.diffusion` 路径、sampler-specific CLI
  flags 和旧配置兼容层。
- 测试私有 Flow family 已通过现有 Process/Sampler/SamplingBuilder Registry、checkpoint
  与 sampling runtime 执行，不需要新增 Dynamics Registry、核心名称分支或通用数学方法。

### 已完成的边界收口

Stage 3 在保留既有数学分层的前提下完成以下收口：

- `StochaflowConfig.process`、`SamplingBuilderContext.process` 和 sampling runtime 中的
  Process 已改为可选；`standard_denoising` 在自己的构造边界要求离散 Gaussian capability，
  通用 runtime 不再先验要求每种算法都有 Process；
- 过宽的 Process 契约已重命名为 `DiscreteGaussianDenoisingProcess`，并同步公开导出、
  错误信息和测试，不保留尚未发布的
  旧名称兼容别名；
- schedule 已明确为构造期 coefficient provider，由 `DiscreteGaussianProcess` 持有唯一
  权威的固定 coefficient snapshot，消除“schedule buffer 已变化、posterior cache 未变化”
  的双重状态；
- checkpoint v5 与 resolved config 已支持 `process: null`，仅在存在 Process 时保存和恢复
  `process_state_dict`；现有 Gaussian 训练桥接仍可明确拒绝缺少 Process 的配置，直到
  Stage 4 由 TrainingStrategy 接管该校验；
- 已增加不创建 Process 或 Sampler 的测试私有 direct transform Builder，证明用户无需伪造
  算法角色即可经过 Registry/config/checkpoint-only sampling/runtime；
- 首版继续要求 primary inference model。无模型解析生成不是本 Stage 承诺，未来出现真实
  实现后再把 model provider 可选化。

### 验收条件

- 新 Process、新 Sampler 和新 SamplingBuilder 均可通过“新增类 + 注册 + YAML”接入，
  核心 runner 无名称分支；不需要 Process 的 Builder 也可直接运行；
- `DiscreteGaussianProcess` 不持有模型，DDPM/DDIM Sampler 不解释 batch 或 condition；
- 离散 Gaussian 公共契约不暗示支持连续时间或无需 adjacent posterior 的实现；
- schedule、marginal 与 posterior 从同一固定 coefficient snapshot 读取，不存在可观察的
  缓存失配；
- DDPM ancestral、DDIM、partial denoising、trajectory 和 checkpoint-only sampling 回归通过；
- 同一个 Gaussian Process 可更换 DDPM/DDIM Sampler，同一个 Sampler 可使用用户模型
  adapter；
- 自定义 condition/guidance Builder 可复用内置 Process 与 Sampler；
- capability 不匹配和必需 Process 缺失都在具体 Builder 边界给出包含组件名和缺失能力的
  错误；
- 配置、公开 API、扩展手册、sampling 文档与 reference 同步更新；
- Pytest、Ruff 和 Pyright 全部通过。

### 逻辑提交

`Separate diffusion processes and samplers`

## Stage 4：TrainingBuilder、TrainingPlan 与单一职责 Strategy（已完成）

### Summary

用薄 `TrainingStrategy` 取代 Trainer 中的 `_default_train_step` 和
`train_step_fn`。Strategy 只定义一次训练/评估计算；新增注册化
`TrainingBuilder` 组装 Model、Objective、Process、Strategy 和可选辅助模块，
并将一个已完成依赖注入的 `TrainingPlan` 交给核心。

核心采纳 Plan 中的资产后，统一管理 device、module mode、backward、optimizer、
scheduler、EMA、gradient clipping、日志和 checkpoint。Strategy 可以引用任意已
注入依赖，但不构建、迁移、冻结、选择参数或序列化它们。

Objective 继续是唯一 loss 抽象，不增加重复的 Loss API。首版仍只支持
“一个标量总 loss + 一个 optimizer”的自动优化生命周期。

### 配置与 Registry

`training` 选择 TrainingBuilder，不在 YAML 中建立通用多模型/多 Objective 资产图：

```yaml
training:
  name: gaussian_denoising
  params:
    prediction_type: epsilon

objective:
  name: mse
  params:
    reduction: mean
```

- 新增必填 `training: ComponentConfig` 和 `REGISTRIES.training_builders`；注册类
  必须继承 `TrainingBuilder`。
- `TrainingStrategy` 是 Builder 组装的运行时逻辑契约，没有 Registry 或独立
  YAML 身份。不同 Strategy 的构造依赖不必伪装成统一 context map。
- `StochaflowConfig.objective` 改为 `ComponentConfig | None`；不需要 Objective 的
  Builder 不必伪造 loss。
- 保留 `REGISTRIES.objectives`，将任务绑定的 `ddpm_epsilon` 收缩为可复用的
  `mse` Objective；不新增 `REGISTRIES.losses`。
- 顶层 `model`、`objective` 和 `process` 是 standard Builder 的主资产输入。自定义
  Builder 的 teacher、额外 Objective 等组合配置完全属于它的 `params`。
- `optimizer`、`lr_scheduler`、`ema`、gradient clipping 和 Trainer 配置继续是核心
  通用配置，不进入 Builder/Strategy params。

### 公共契约

```python
@dataclass(frozen=True, slots=True)
class ManagedTrainingModule:
    module: nn.Module
    mode: Literal["follow", "eval"] = "follow"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    strategy: TrainingStrategy
    primary_model: nn.Module
    process: Process | None
    objective: nn.Module | None
    auxiliary_modules: Mapping[str, ManagedTrainingModule]


@dataclass(frozen=True, slots=True)
class TrainingBuilderContext:
    params: dict[str, Any]
    primary_model: nn.Module
    process: Process | None
    objective: nn.Module | None
    model_factory: ModelFactory
    objective_factory: ObjectiveFactory


class TrainingBuilder(ABC):
    @abstractmethod
    def build(self) -> TrainingPlan: ...


class TrainingStrategy(ABC):
    @abstractmethod
    def training_step(self, batch: Any) -> TrainStepOutput: ...

    def evaluation_step(self, batch: Any) -> TrainStepOutput: ...
```

- Builder context 深复制 params。factory 只注入 Builder，因为资产组装是 Builder 的
  职责；Strategy 不访问 Registry 或 factory。
- Strategy 的构造签名不属于公共 schema。具体 Builder 可以直接注入模型、Process、
  Objective、teacher callable 或其他已经组装好的依赖；核心只依赖
  `training_step()` / `evaluation_step()`，不要求所有任务使用统一的 context mapping。
- `primary_model`、`process` 和 `objective` 是具有 sampling/checkpoint 固定身份的
  主资产，不得在 `auxiliary_modules` 重复声明。
- auxiliary name 必须稳定、非空且唯一。首版所有 auxiliary module 都进入
  checkpoint，不增加 external/reference-only 保存策略。
- `mode="follow"` 跟随 train/eval lifecycle；`mode="eval"` 在训练时也保持 eval，
  用于 frozen teacher。是否进入 optimizer 只由 `requires_grad` 决定。
- Strategy 是无持久状态的训练逻辑对象，不继承 `nn.Module`，不提供
  `to/train/eval`、parameter selection、optimizer、factory 或 checkpoint/state API。
- `TrainStepOutput` 包含标量可反传 `loss`、可选 scalar `metrics` 和可选
  `diagnostics`。Trainer 在调用 Strategy 前递归迁移 structured batch。

### 内置 Builder 与 Strategy

`supervised` Builder 使用顶层 Model/Objective 构造通用 Strategy：

- 要求 Objective，只接受 `(inputs, targets)` batch；
- 执行 `predictions = model(inputs)` 和 `loss = objective(predictions, targets)`；
- 不认识 Process、condition、timestep 或 diffusion。

`gaussian_denoising` Builder 接替临时 Gaussian training bridge：

- 要求 `DiscreteGaussianDenoisingProcess` 和 Objective；
- 构造支持 `epsilon`、`x0`、`v` 和 `score` target 的 Strategy；
- Strategy 调用 Objective 计算 prediction/target loss，不复制 MSE；
- 接受 bare Tensor 或 `(images, {})`；非空 condition 明确失败，不静默丢弃
  `low_res`；conditional/SR 由自定义 Builder 组装匹配的 Strategy；
- 删除 `GaussianEpsilonTrainingSystem`、`resolve_train_step_fn()`、
  `ddpm_epsilon_train_step()` 和 factory 中按 objective 名称 dispatch 的分支。

### 蒸馏组合

自定义 `knowledge_distillation` Builder 的私有 params 可声明 teacher model/checkpoint、
蒸馏 Objective 和权重。Builder 通过注入的 factory 构建 teacher 与额外 Objective，
加载并冻结 teacher，然后显式构造：

```python
KnowledgeDistillationStrategy(
    student=primary_model,
    teacher=teacher,
    task_objective=objective,
    distill_objective=distill_objective,
    alpha=alpha,
)
```

Plan 将 student 作为 `primary_model`，teacher 以 `mode="eval"` 进入 auxiliary modules；
teacher 的 `requires_grad=False` 使其不进入 optimizer。Strategy 只执行 teacher
no-grad forward、student forward、两项 Objective 和 total loss/metrics 组合。

```text
TrainingBuilder
  ├─ 构建/加载/冻结 teacher
  ├─ 构建 task 与 distillation Objective
  └─ 返回 TrainingPlan(student, teacher, objectives, strategy)

TrainingStrategy
  └─ L_total = (1 - alpha) * L_task + alpha * L_distill

Trainer/core
  └─ 托管 module mode、device、单 optimizer、backward 与 checkpoint
```

蒸馏不会成为 Strategy 的通用 mode，也不会给通用配置增加 `teacher`、`temperature`、
`feature_layers` 等字段。这些都是具体蒸馏 Builder/Strategy 的私有任务语义。只要多个
损失能够合成为一个标量总 loss，自动训练循环就无需理解蒸馏。

首版覆盖 frozen-teacher online distillation；预先生成 teacher target 的 offline
distillation 则不需要 auxiliary teacher。多 teacher、feature/logit/score distillation
仍沿用同一组合边界：Builder 组装并声明资产，Strategy 解释一次 batch 的 forward 与
loss 数学。若 student 与 teacher 共享同一个 optimizer 和一次 backward，Plan 可以将
两者都声明为可训练资产；需要独立 optimizer、EMA teacher、交替更新或 manual backward
时，变化的是训练生命周期，而不是单步 loss 公式，因此应在出现真实需求后定义新的
training-loop family，不能继续给 Strategy 增加可选控制字段。

### Core runtime 与 lifecycle

- core 先构建顶层 primary model、可选 Process 和可选 Objective，再构建所选
  TrainingBuilder 并只调用一次 `build()`。
- 集中验证 `TrainingPlan`、资产身份/名称/重复、mode policy 和 Strategy 类型；
  core/Trainer 不按 Builder 名称分支。
- core 稳定去重所有 managed module 中 `requires_grad=True` 的参数，并用同一
  tuple 构建 optimizer 与执行 gradient clipping。EMA 仍只跟踪 primary model。
- Trainer 直接调用 `strategy.training_step()` / `evaluation_step()`；移除
  `criterion`、`train_step_fn`、`_default_train_step` 和任意 model-output fallback。
- backward、单 optimizer step、scheduler interval、batch limit 和 epoch lifecycle 保持核心
  职责。多 optimizer、交替更新和 custom backward 不在本 Stage 承诺内。

### Checkpoint v6

- 固定字段保存 `model_state_dict`、可选 `process_state_dict`、可选
  `objective_state_dict`、可选 `ema_model_state_dict`、optimizer、scheduler、EMA、config
  和进度。
- 新增 `training_assets_state_dict: Mapping[str, Mapping[str, Any]]`，按 Plan 中稳定名称
  保存所有 auxiliary module。resume 必须严格匹配当前 Plan 的 auxiliary names。
- 不保存 `strategy_state_dict`；Strategy 契约不允许可变持久状态。
- checkpoint-only sampling 只构建并加载 primary model、可选 Process 与
  SamplingBuilder，忽略 Objective、TrainingBuilder、Strategy 和 training assets。
- v5 直接拒绝，不增加迁移层。

### Diagnostics capability

- Trainer 和 diagnostic event 暴露当前 Plan/Strategy 与 core-managed assets。
- Gaussian diagnostic 通过可选窄 capability 获取 prediction type 和已经适配好模型签名的
  prediction callable；它不得假设 primary model 可直接以 `(state, model_time)` 调用。
  Dynamics 仍由 diagnostic 用该 callable 与 Process 组合。
- 该 callable 只表达 Strategy 已拥有的 Gaussian 模型调用语义，不让 Strategy 构建
  Sampler、运行 diagnostic 或管理 artifacts。需要 condition 的 Strategy 必须显式提供
  diagnostic 可用的上下文；无法提供时不声明 capability，并得到明确不兼容错误。
- step providers 从 `TrainStepOutput.diagnostics` 按需验证字段；通用 Strategy 不被迫
  产生 Gaussian intermediate。
- EvaluationGuard 对全部 managed modules 执行 mode/RNG/EMA 保护，并恢复
  `mode="eval"` 的固定策略。

### Test Plan

- Builder/Plan：错误注册基类、params 深复制、错误返回类型、非法 auxiliary
  name/module/mode、重复资产和不可训练 Plan 明确失败。
- Strategy：非法 step output 失败；公共契约无 lifecycle/state/factory API；自定义
  structured batch 原样到达 Strategy。
- 简单路径：`supervised` 对 `(inputs, targets)` 完成 train/eval；
  `gaussian_denoising + mse` 覆盖四种 prediction target 并与现有 epsilon 数学回归一致。
- OCP：测试私有非 Gaussian Builder/Strategy 只通过注册/config 驱动 Trainer，
  core 无名称分支。
- 蒸馏：测试私有 Builder 显式注入 student/teacher/两个 Objective；teacher 无梯度、
  始终 eval、不进入 optimizer；Strategy 组合总 loss 并记录分项 metrics。
- checkpoint：primary model、可选 Process/Objective、EMA、auxiliary modules、
  optimizer/scheduler/progress 恢复；auxiliary names 不匹配失败；sampling 不构建训练侧
  Builder/Strategy/assets；v5 拒绝。
- diagnostics：内置与真正独立、使用非标准模型签名的测试私有 Gaussian-compatible
  Strategy 均通过 prediction callable 复用 diffusion-quality；非 capability Strategy
  得到明确错误。
- 配置、YAML、公开导出、README、extension/workflow/troubleshooting 和 reference 同步。

实施完成后的日常检查：

```bash
uv run pytest tests/test_training_builder.py tests/test_training_strategy.py \
  tests/test_factory.py tests/test_trainer_reporting.py \
  tests/test_experiment_runner.py tests/test_sampling_runtime.py tests/diagnostics
uv run ruff check .
uv run pyright
```

完整 pytest、build、Sphinx 和额外静态检查留到整版 feature 分支合并验收。

### Assumptions

- Stage 4 不保留 `ddpm_epsilon` Objective 名称、旧训练 bridge、v5 checkpoint 或
  未发布 API 兼容层。
- primary model 仍是顶层必填资产；Objective 和 Process 可选，具体 Builder 验证
  自己必需的资产与 family capability。
- Builder 可构建辅助模块，但不执行训练循环；Strategy 可引用资产，但不管理
  其 lifecycle。
- 首版 auxiliary module 全部 checkpoint 自包含；对外部 teacher reference 只保存路径/
  hash 的容量优化等真实需求再设计。
- Stage 4 只统一自动单 optimizer lifecycle；多 optimizer 需要新的训练 loop family。
- Stage 4 形成单一逻辑提交：`Stage 4: Add extensible training builders`。

## Stage 4.1：成熟依赖原生 Provider 收口

### Summary

配置必须记录运行选择以保证可复现，但 Registry 不应成为 PyTorch 类目录。标准
`torch.optim` optimizer 与 `torch.optim.lr_scheduler` scheduler 通过受限 native-provider
resolver 解析，核心注入运行时依赖后将 `params` 原样传给 PyTorch 构造器。第三方实现只有
在保持同一构造与生命周期契约时才注册为对应 PyTorch 基类；本 Stage 不增加第二套
run-aware scheduler Builder。

### 配置边界

```yaml
optimizer:
  name: torch.optim.AdamW
  params:
    lr: 0.0002
    weight_decay: 0.01

lr_scheduler:
  name: torch.optim.lr_scheduler.CosineAnnealingLR
  interval: epoch
  params:
    T_max: 100
```

- `name` 以 `torch.optim.` 或 `torch.optim.lr_scheduler.` 开头时，只能从对应 allowlisted
  namespace 解析，不能作为任意 Python import path；解析结果分别验证为
  `torch.optim.Optimizer` 子类或当前自动 Trainer 支持的 scheduler contract。
- 两个 native prefix 是保留 namespace；对应 Registry 在注册边界直接拒绝使用这些前缀的
  扩展名称，避免出现“注册成功但永远被 native resolver 遮蔽”的组件。
- 其他 optimizer 名称继续由 `REGISTRIES.optimizers` 解析为
  `torch.optim.Optimizer` 子类；其他 scheduler 名称由 `REGISTRIES.lr_schedulers` 解析为
  `torch.optim.lr_scheduler.LRScheduler` 子类。两类 Registry 都只服务真正的扩展，不复制
  PyTorch namespace。
- optimizer 的 parameter iterable 与 scheduler 的 optimizer 由 core 作为位置参数注入；
  `params` 只包含用户真正覆盖的 PyTorch 构造关键字参数，省略值继续采用当前安装
  PyTorch 的默认值。optimizer 配置拒绝保留键 `params`，scheduler 配置拒绝保留键
  `optimizer`，避免配置覆盖运行时依赖。
- `StochaflowConfig.optimizer` 直接使用已有 `ComponentConfig`；删除字段完全相同、没有独立
  行为的 `OptimizerConfig`。默认 target 为 `torch.optim.Adam`，默认 params 只保留框架
  确实选择的 `lr`，其余采用 PyTorch 默认值。
- `lr_scheduler.interval` 保留为 `step | epoch`，因为 PyTorch 实现不定义 Trainer 应在
  哪个 lifecycle event 调用 `step()`。
- 首版自动生命周期只支持可无参数调用 `step()` 的 optimizer 与 scheduler。需要 closure
  的 optimizer 或 validation metric 的 scheduler 必须等待明确的 lifecycle contract，不能
  根据具体类名特判。
- `lr_scheduler: null` 表示禁用，不需要为“无 scheduler”构造空对象。

### 构造、验证与运行时值

构造路径保持直接且统一：

```python
optimizer = optimizer_class(trainable_parameters, **optimizer_params)
scheduler = scheduler_class(optimizer, **scheduler_params)
```

- `params` 是不透明 mapping；generic factory 不复制、重命名、补全或解释具体 PyTorch
  constructor 字段。PyTorch 构造器是参数合法性的权威来源，factory 只为失败补充配置路径、
  target 和原始异常链。
- 这里的配置 `params` 表示 constructor kwargs，不等于 optimizer 构造器首个 `params`
  iterable，也不从运行时 `.defaults`、`.param_groups` 或 scheduler state 反向生成配置；
  PyTorch 没有跨 optimizer/scheduler 通用的可序列化“参数字段”。
- `inspect.signature()` 可以作为未来 CLI help 的 best-effort 信息源，但不能生成稳定公共
  schema，也不能决定 constructor compatibility。这里的反射只允许用于验证构造后的 bound
  optimizer/scheduler `step` 是否满足当前自动 Trainer 的零参数调用契约：执行
  `inspect.signature(component.step).bind()`；必需 positional/keyword-only 参数导致拒绝，
  默认参数和 `*args` 可以接受零参数。无法取得可靠 signature 时也在构建边界明确拒绝，
  不延迟到训练中途。scheduler 还必须保留 core 注入的同一个 optimizer。
- `T_max`、`total_steps`、`num_training_steps`、`epochs + steps_per_epoch` 等名称不构成通用
  语义协议。首版要求在 `params` 中提供具体值，不根据字段名自动注入 run context，也不接受
  `"auto"` sentinel。
- Stochaflow 内置 warmup-cosine 若继续保留，应实现为真正的 `LRScheduler` 子类并显式接收
  `warmup_steps` 与 `total_steps`，通过和第三方 scheduler 相同的 Registry/构造路径使用；
  它不是 generic factory 的名称分支。
- CLI 改变 epoch 或 batch limit 时不会隐式重写 scheduler kwargs。需要同步改变总步数的运行
  必须修改并传入一份内部一致的完整训练 config；Stage 4.1 不增加通用 CLI override 或训练
  overlay。若未来多个真实用例证明需要运行时引用，再单独设计配置引用或训练组合能力，
  不提前发布 scheduler-only Builder。

### 实现修改

- 删除 Adam、AdamW、CosineAnnealingLR、StepLR、MultiStepLR、ExponentialLR 和 LinearLR
  的逐项 Registry alias；新增一个只解析上述 PyTorch namespace 的通用 resolver。
- 删除重复的 `OptimizerConfig`，配置加载、factory 与公共 utility export 使用
  `ComponentConfig`；`LRSchedulerConfig` 因额外拥有 interval lifecycle 字段而保留。
- generic optimizer/scheduler factory 不按具体 component name 添加 `if/elif`。
- Registry 在注册期只验证 `Optimizer`/`LRScheduler` 子类关系；构建期才通过统一的
  `Class(runtime_dependency, **params)` 调用验证 constructor substitution，并保留原始
  `TypeError` 异常链。不得暗示基类检查已经证明具体 constructor 签名兼容。
- 纯 PyTorch `CosineAnnealingLR` 要求显式合法的 `T_max`；warmup-cosine 要求显式合法的
  `total_steps`。删除 `auto_cosine`、`T_max: auto`、`total_steps: auto` 及其 name-specific
  配置校验。
- optimizer/scheduler checkpoint 继续直接使用 PyTorch `state_dict()`，并在当前未发布格式
  记录 concrete class identity；scheduler 构造时必须保留同一个 optimizer。恢复时严格校验
  runtime/state 存在性与 class identity，先加载 scheduler state，再加载 optimizer state，
  遵守 PyTorch 的恢复 contract。
- 配置 reference 不复制每个 PyTorch class 的完整签名、默认值或版本相关参数；只记录
  allowlisted namespace、core 注入参数、Stochaflow lifecycle 字段并链接上游文档。
- resolved config/checkpoint 保存 target 与用户显式 kwargs，不把当前 PyTorch 默认值展开成
  Stochaflow schema；跨环境复现依赖项目锁定 PyTorch 版本，运行 provenance 可记录版本，
  但不能通过复制上游 defaults 替代依赖锁定。
- 同一 native-provider 原则只在依赖实现与 Stochaflow 行为契约确实一致时复用到其他组件；
  不能仅因为某个 `torch.nn` 类存在，就推断它满足 Objective、Model 或其他框架语义。

### Resume 与 checkpoint 边界

当前尚无需要保留的用户 checkpoint，因此本 Stage 不增加旧格式迁移、兼容层、配置差异
合并或 optimization-state reset 选项。`resume` 保持单一含义：恢复同一训练的完整 PyTorch
optimizer/scheduler state；constructor config 用于重建对象，checkpoint state 用于继续其运行
状态。若用户要更换 optimizer、scheduler 或其超参数，那是以 checkpoint 权重初始化的新训练，
不是 resume，本 Stage 不提前增加 weights-only workflow。

checkpoint 格式直接升级为 v7，加载顺序遵守 PyTorch state contract；不为 v6 草案增加版本
迁移。不同 concrete optimizer/scheduler class 或 scheduler 有无不一致直接拒绝，不能将
state mapping 的可加载性误当作语义兼容。

### Test Plan

- 任意测试私有 `torch.optim.Optimizer` 子类通过 monkeypatched allowlisted namespace 构建，
  证明 resolver 不需要逐项登记；错误类型、未知属性和越界 module path 明确失败。
- PyTorch scheduler 获得 core 注入的同一 optimizer，`step`/`epoch` interval、checkpoint
  save/load 和恢复后的 learning rate 与直接使用 PyTorch 一致。
- 需要 closure 的 optimizer 和需要 positional metric 的 native/registered scheduler 在构建
  边界明确拒绝，不在训练中途才失败。
- extension 注册的 custom optimizer 与 scheduler subclass 继续工作，native 与
  Registry namespace 无歧义。
- constructor params 原样透传且不修改调用方 mapping；未知参数由 PyTorch 报错并保留异常链；
  配置中的保留注入键明确失败。
- optimizer/scheduler bound `step()` 的零参数 bind 覆盖必需参数、默认参数、`*args`、
  keyword-only 参数和无法取得 signature 的实现；错误绑定 optimizer 的 scheduler 明确失败。
- warmup-cosine 使用显式 `total_steps`，CLI epoch/limit 改动不触发隐式重写。
- 配置 reference 不再枚举或复制 Adam/AdamW 与标准 scheduler 的上游参数表。

日常验证：

```bash
uv run pytest tests/test_factory.py tests/test_config.py tests/test_trainer_reporting.py
uv run ruff check .
uv run pyright
```

### 逻辑提交

`Stage 4.1: Reuse native PyTorch optimizers and schedulers`

### 非目标

- 不增加任意 Python class-path import、全局 dependency provider Registry 或自动扫描；
- 不在本 Stage 支持 metric-driven scheduler、多个 scheduler、scheduler chaining 配置图或
  manual optimization；
- 不为只需要显式 PyTorch kwargs 的 scheduler 编写 Stochaflow wrapper；
- 不增加 `LRSchedulerBuilder`、run-context 参数名推断、通用 `auto` sentinel 或配置插值系统。

## Stage 5：Entry-point extension 与多实验项目脚手架（已完成）

### Summary

将扩展激活从 Python module path 收口为标准 Python packaging entry point。
用户在 extension package 内继续通过 decorator 直接注册；Stochaflow 从当前
Python environment 的 `stochaflow.extensions` group 发现并导入聚合模块。

`stochaflow init` 生成的 repo 是普通、可安装、可发布的单 Python
distribution 多实验 research repo。仓库可以同时包含 Stochaflow 实验和任意其他
研究代码；Stochaflow 只管理由其 CLI 启动的工作流，不解释 scope 外的目录或工具。

### 已实施边界

- repo 首版是单一 installable distribution、多实验，不生成多 package workspace；
- 非 Stochaflow 实验可以自由共存，但不进入 Stochaflow 的模板契约或运行时 dispatch；
- extension project 必须与当前 `stochaflow` CLI 位于同一 Python environment；
- pip、uv、conda、Poetry 或 PDM 均可安装该环境，Stochaflow 不管理包管理器；
- extension 通过 `[project.entry-points."stochaflow.extensions"]` 发现，不扫描目录、
  package naming convention 或 namespace package；
- `init` 模板直接生成可运行配置，用户在此基础上修改。

### Entry point 契约

生成项目在 `pyproject.toml` 声明：

```toml
[project.entry-points."stochaflow.extensions"]
my-project = "my_project.stochaflow_ext"
```

`my-project` 与 `my_project` 只表示由用户输入名称派生的 distribution/package 占位符，
不是模板内置的领域或仓库名称。

- entry-point name 是稳定的插件身份；生成模板默认使用 canonical distribution name；
- target 必须是纯 module，不允许 `:callable` 或 extras；activation 按 metadata target
  导入该模块，decorator 完成注册；
- 一个 distribution 可以发布多个具有唯一名称的 Stochaflow entry point；生成模板只
  声明一个聚合入口，该模块可以导入多个 data/model/training/sampling 子模块；
- entry-point 导入阶段只定义类/函数并注册，不读取数据、构建模型、加载
  checkpoint、创建 accelerator context 或启动任务；
- 非 Stochaflow 代码不从该聚合模块导入。

### 配置契约

`ExtensionsConfig` 收口为：

```python
@dataclass
class ExtensionsConfig:
    plugins: list[str] | None = field(default_factory=list)
```

- 省略 `extensions`：不加载第三方插件；
- 显式 `plugins: null`：发现当前环境的全部 `stochaflow.extensions` entry points；
- `plugins: []`：不加载任何第三方插件；
- `plugins: [my-project]`：只加载指定 entry-point names；
- `init` 模板不使用环境自动发现，而是直接写入项目 entry-point name：

  ```yaml
  extensions:
    plugins:
      - my-project
  ```

- resolved config 始终将 `plugins` 保存为确定列表，不保存 `null`；
- 删除 `extensions.modules`，不保留未发布 module bootstrap 的兼容层；
- sampling-only overlay 若提供 `extensions.plugins`，它是覆盖后的完整列表，
  不与 checkpoint list 隐式追加。

### 发现、排序与冲突

Stochaflow 通过 `importlib.metadata.entry_points(group="stochaflow.extensions")` 读取已安装
distribution metadata，不读取未安装源码。entry-point name 按声明值精确匹配；仅
distribution name 使用直接依赖的 `packaging` 做 canonicalization，version 使用其 PEP 440
parser。在加载任何插件前完成预检：

- 按原始 entry-point name、canonical distribution name 和 target 稳定排序；
- 同名 entry point 由多个 distribution 提供时直接失败，不使用后加载覆盖；
- 配置请求的插件缺失时，报告 entry-point name 和当前 Python executable；
- target 不是纯 module、distribution metadata 非法或 import 失败时，报告 name、
  distribution、version、target 和原始异常链；
- 非空显式列表只检查请求 name 的 candidate；未选择插件的重复名或畸形 metadata 不得破坏
  本次运行。`plugins: null` 选择全环境，此时校验完整发现集合；
- Registry 名称重复继续由 `RegistryError` 失败，错误上下文增加当前插件身份；
- 插件 activation 层按完整 provenance 保证进程内幂等，不把 packaging discovery 职责塞入
  RegistryCatalog；一个进程首次激活后 selection 固定，相同 selection 可重复调用，不同
  selection 明确失败，不支持 reload/unload。selection identity 是按 exact entry-point name
  排序后的 provenance tuple；配置中的重复 name 和 metadata 中被选中的同名重复都失败，
  resolved config 保存同一确定顺序。CLI 的一次 invocation 是标准隔离边界。
- 激活使用进程级锁和 `UNACTIVATED → ACTIVATING → ACTIVE | FAILED` 状态机。任一 module
  import、Registry 冲突或 re-entrant activation 失败都会进入终止性 `FAILED`；decorator
  注册不可事务回滚，因此后续 activation 明确要求重启进程。并发调用被串行化，相同
  selection 的幂等只适用于 `ACTIVE`。

配置解析、metadata 预检和代码导入分为明确阶段：

```text
无副作用解析 config
→ discover/resolve entry-point metadata
→ selection 与 checkpoint provenance 预检
→ CLI prompt 或显式 library policy
→ import 聚合模块并注册
→ 构建 Registry 组件
```

`load_config()`/`load_config_dict()` 只解析并验证配置值，不执行插件代码。公开的
`prepare_extension_plugins(config, expected_provenance=...)` 每次都重新执行 discovery 与
expected/current 预检，返回不可变 activation plan；已处于 `ACTIVE` 也不能跳过本次
checkpoint version/identity 检查。公开的 `activate_extension_plugins(plan, policy=...)`
只负责按 plan 导入一次并返回 `ResolvedExtensions`：深拷贝且已 materialize 确定
`extensions.plugins` 的 config、当前 provenance 和本次 acceptance audit。原 config 与调用方
dict 不修改，所有后续组件构建、resolved config、manifest 和 checkpoint 只消费该结果。
library 默认 `ExtensionVersionPolicy.REJECT`，显式 `ALLOW` 的 audit method 为
`library-policy`，且任何 library API 都不读取 stdin；CLI 在相同内部执行边界记录
`prompt` 或 `force-flag`。

`stochaflow.extensions` 公开导出 `ExtensionPluginProvenance`、
`ExtensionVersionMismatch`、`ExtensionPluginError`、`ExtensionIdentityError`、
`ExtensionVersionMismatchError`、`ExtensionVersionPolicy`、`ExtensionActivationPlan`、
`ResolvedExtensions`、`prepare_extension_plugins()` 和 `activate_extension_plugins()`；
它们是 programmatic runtime 的唯一插件入口，调用方不依赖 CLI 私有 prompt helper。

### Resolved config、checkpoint 与版本策略

resolved config 保存决定性插件名列表；checkpoint metadata 另外保存 provenance：

```yaml
extension_plugins:
  - name: my-project
    distribution: my-project
    version: 0.1.0
    target: my_project.stochaflow_ext
```

- 新训练/显式完整 config 按其 `extensions.plugins` 选择当前环境插件；
- checkpoint-only resume/sampling 只加载 checkpoint resolved config 记录的插件，
  忽略环境中后来安装的其他插件；
- v8 的 `metadata.extension_plugins` 始终存在（无插件时为空列表），由一个集中 parser 严格
  校验四个字段、唯一 name、canonical distribution、合法 version 与 pure-module target；
  checkpoint-base 时其 name 集合必须与 resolved `extensions.plugins` 完全一致。显式完整
  config 或显式 extension overlay 只比较 checkpoint/current selection 的 name 交集；
- entry-point name、distribution 或 target 不匹配属于身份错误，始终硬失败，
  不能 force；
- 仅 version 不一致时，CLI 在加载任何插件代码前汇总显示 warning；
- 交互式 TTY 一次询问是否继续，默认为 No；用户拒绝时在任何构建/状态
  恢复前退出；
- 非交互式 stdin 不等待 prompt，默认失败；
- train/sample 提供范围明确的 `--force-extension-version-mismatch`，允许交互或
  非交互命令跳过版本询问；不提供含义模糊的通用 `--force`；
- Python 库 API 不进行交互：默认抛出类型化版本不匹配异常，调用方必须
  显式选择 allow policy；
- 接受版本不一致后，run summary、resolved manifest 和后续 checkpoint metadata
  记录 expected/current version 与接受方式（prompt 或 force flag）；
- editable install 中“代码改变但版本号未变”无法由 entry-point metadata 检测，
  Git commit/wheel hash/lockfile provenance 留到 Stage 8。
- version 使用 PEP 440 `Version` 做等价比较，同时在 provenance/audit 保留 distribution
  metadata 的原始字符串；v8 provenance 的字段类型、重复 name、distribution、version 和
  pure-module target 全部严格验证，缺失或畸形直接失败。

当前没有任何需要保留的用户 checkpoint。Stage 5 直接定义新的 v8 payload 和
`extensions.plugins` 配置，不识别 v7、`extensions.modules` 或任何旧字段，也不提供
迁移、fallback、双写或兼容别名。下述 provenance 与 state contract 校验只约束 Stage 5
之后由新格式产生的 checkpoint，属于恢复正确性，不是 legacy compatibility。

v8 将 checkpoint 值域收窄为 PyTorch `weights_only=True` 默认可加载的 Tensor、primitive
和普通 container，不允许 extension class、任意 pickle object、custom tensor subclass 或
运行前 `add_safe_globals()`。保存时递归校验并报告精确 state path，加载始终显式使用
`weights_only=True`。自定义 module/optimizer/scheduler 的额外状态必须编码为该数据值域。
这样 runtime 可以安全读取 config/provenance、完成版本确认并激活插件，再把已加载 state
注入组件；不会为了读 metadata 提前动态 import 扩展代码。

该修正直接取代 Stage 4 草案的“任意可 pickle extra_state”。没有用户 checkpoint 需要迁移，
也不提供 `safe_globals` 逃生口；sidecar/双文件 envelope 会破坏单 checkpoint 可移植性，因此
不采用。`weights_only=True` 降低动态代码执行风险，但不是对恶意大 Tensor、DoS 或内存耗尽
的完整安全沙箱。

### 用户流程与 CLI

```bash
stochaflow init my-project
cd my-project
```

`init` 只生成文件，不运行 pip/uv、不创建环境且不覆盖非空目录。项目通过
标准 Python packaging 安装到用户选择的环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
stochaflow train --config experiments/example/train.yaml
```

或使用可选 uv 工作流：

```bash
uv sync
uv run stochaflow train --config experiments/example/train.yaml
```

`stochaflow` CLI、extension distribution 及任务依赖必须在同一 Python environment。
独立 pipx/uvx 环境可用于执行 `init`，但不能直接运行另一项目环境内的 extension。

### 配置权威、恢复与覆盖规则

训练与 sampling 使用不同的 checkpoint/config 权威边界。训练 full resume 恢复完整
optimizer/scheduler state，因此 checkpoint config/state 权威；sampling 不恢复训练状态，
允许用本次 sampling config 自由改变样本数量、batch size、shape、Builder、Sampler、solver
参数、trajectory、writer 和 raw/EMA 选择。

config、checkpoint 和 CLI 不作为三份对等配置参与 merge。已经确认的命令先选择唯一 base
config，再应用用户显式请求的局部覆盖：

| 命令 | 显式完整 config | checkpoint | 权威 base config | 后续覆盖 |
| --- | --- | --- | --- | --- |
| 新训练 | 是 | 否 | 外部 config | train CLI flags |
| strict training resume | 否 | 是 | checkpoint 保存的 config/state | 安全 train runtime flags |
| checkpoint-only sampling | 否 | 是 | checkpoint 保存的 config | sample CLI flags |
| 完整 config sampling | 是 | 可选 | 外部 config | sample CLI flags；checkpoint 只提供 state |
| sampling-only overlay | 局部 | 是 | checkpoint 保存的 config | overlay 的 `sampling`/`extensions`，再应用 CLI flags |

- 新训练没有 checkpoint 时必须提供完整 config；
- sampling 的显式完整 config 是整体替换，不与 checkpoint config 做完整 equality 对比；
- sampling-only YAML 是明确声明的局部 overlay，不会被解释成第二份完整配置；
- sampling overlay 的 `extensions: {}` 不表示清空，也不替换 selection；只有 raw overlay
  明确包含 `extensions.plugins` key 时才执行完整替换；
- CLI 只覆盖其文档化字段或 runtime option，不增加 Stage 5 通用 `--set` DSL；
- 配置字段 override 写入 resolved config；`limit-batches`、deterministic、skip-final-sample、
  启动 cwd、lineage 和 version acceptance 等 invocation 事实写入独立 run/sampling manifest
  与 checkpoint metadata，不扩张组件配置 schema。

训练侧已确认的边界是：

- `train --resume CHECKPOINT` 只表示 strict full resume，checkpoint config 与完整训练 state
  共同权威；外部 `--config` 与它互斥；
- 只允许 device、output root、目标 epoch、progress 和 batch limit 等不会重定义
  optimizer/scheduler state 的安全 invocation override；
- 用新 config 加载旧模型权重是独立的 weights-only warm-start（建议未来命名
  `--init-from CHECKPOINT`），不恢复 optimizer、scheduler、epoch/global step；
- 不实现“先加载完整 state、再通用重写 optimizer/scheduler 配置”的 hybrid，因为 PyTorch
  原生 state 没有跨实现统一的配置/运行状态拆分边界。

无论采用何种已确认的恢复入口，都延续“一次 CLI invocation 对应一个输出 run”的生命周期，
而不是重新打开并覆盖旧 run：

- checkpoint-only resume 从保存配置重建实验设置和 state，但创建新的 `exp_id` 与兄弟
  run directory；默认 output root 取前一 resolved run directory 的父目录；
- `--output-dir` 显式覆盖上述 root；
- epoch、global step、模型与 optimizer 等 state 连续，日志与 artifact 写入新 run，避免
  本地日志截断、第三方 logger run-id 复用和旧 artifact 覆盖；
- v8 checkpoint 以 data-only 形式保存 Python、NumPy global RandomState、Torch CPU 和可用
  CUDA RNG state；strict resume 在 selected/inherited-best 全部校验与恢复后才恢复对应 RNG。
  普通 checkpoint load 不修改全局 RNG；sampling 不恢复 checkpoint RNG snapshot，而是按
  `sampling.seed`（为 `null` 时使用 `experiment.seed`）重新初始化 Python、NumPy 与 Torch
  全局 RNG；
- device override 继续允许：CPU 目标忽略保存的 CUDA stream，CUDA 目标在有保存 state 时
  要求 device count 兼容；跨设备或拓扑变化不承诺逐位一致；
- DataBuilder、Dataset、DataLoader iterator/worker、Sampler 及用户私有 generator runtime
  state 不进 checkpoint。自定义随机 loader 必须由 seed/epoch 可重建，并在需要时响应
  duck-typed `set_epoch(epoch)`；
- `latest.pt`/epoch checkpoint 记录的 best identity 必须与 sibling `best.pt` 的 resolved
  config、extension provenance、epoch、metric、monitor 和 mode 一致，防止另一 run 或被
  未来 epoch 覆盖的 mutable best 泄漏错误状态；校验并加载当前资产 topology 后，inherited
  best 在 fit 前以当前 config/provenance 原子物化到新 run，使连续恢复和 sampling 不依赖
  父 run；
- resolved config 只记录最终可重建配置；run manifest 和 checkpoint metadata 记录新 run
  路径以及 `resumed_from` lineage。

因此 resume 表示“恢复训练状态”，不是“原地追加旧输出目录”。这与每次 Hydra job 拥有
独立 output directory 的思路一致，也保持当前 Stochaflow timestamped run 模型。

当 checkpoint 保存的 config 是 base config 时，其插件集合是当前恢复的预期 selection。
显式完整 config 或显式替换了 `extensions.plugins` 的 sampling-only overlay 以当前 config
选择插件，允许增加和删除；但如果复用 checkpoint provenance 中同一 entry-point name，
distribution/target 必须保持 identity，version 差异仍走统一确认。新 checkpoint/manifest
保存本次实际 provenance，历史 expected/current 只进入 acceptance audit。

sampling checkpoint state 是否能加载到显式配置构建的资产，由既有严格 runtime contract 判断：
model/Process/Objective state dict、具名 training assets、optimizer/scheduler/EMA state 和
checkpoint format 必须匹配。核心不通过比较完整 config 推断兼容性；显式改变但仍满足
state contract 的配置是合法用户选择，不满足时由具体加载边界报告 missing/unexpected
state 或资产拓扑错误。

### 插件与 runtime 数据流

Stage 5 用标准 distribution metadata 替换 Stage 1 的 module-path bootstrap，但不把
“项目”引入 runner：

```text
用户选择的包管理器将 extension 安装到 CLI 所在环境
→ load_config() 无副作用解析 extensions.plugins
→ entry-point resolver 预检身份与版本
→ CLI/library policy 接受或拒绝 version mismatch
→ 导入选定聚合模块，decorator 注册组件
→ 构建并运行已注册组件
```

training/sampling input resolver 均携带同一个类型化 extension activation result（materialized
config、actual provenance、acceptance audit）；manifest、checkpoint metadata 和组件构建只从
该结果读取，不在 runner、factory 与 writer 中重复执行 provenance 比较。训练 factory 通过
显式参数接收 checkpoint metadata，不在构造后修改 Trainer。

- `load_config()` / `load_config_dict()` 不增加 project 参数，也不承担插件导入；
- `RegistryCatalog`、runner、Trainer 和 sampling runtime 不知道用户项目；
- checkpoint 保存具体插件列表与 distribution provenance；恢复时要求相应 distribution
  已安装到当前 Python 环境；
- 跨机器运行由 `pyproject.toml`、所选包管理器的 lockfile 或已发布的 extension package
  恢复代码环境，checkpoint 不携带源码。

### 生成项目

```text
my-project/
├── pyproject.toml
├── .gitignore
├── README.md
├── data/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── experiments/
│   └── example/
│       └── train.yaml
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── stochaflow_ext/
│           ├── __init__.py
│           ├── data.py
│           ├── model.py
│           ├── training.py
│           └── sampling.py
└── tests/
    └── test_extensions.py
```

`outputs/` 由第一次运行创建并写入 `.gitignore`，不预生成或提交运行产物。
默认 YAML 使用项目根目录相对值，例如 `data/...` 和 `outputs/...`；模板 README
明确要求从项目根目录运行。包管理器提供切换工作目录的能力时可以使用它，但这不是
Stochaflow 的运行时契约。

首版项目名只接受已经 canonical 的 ASCII slug：

```text
[a-z][a-z0-9]*(?:-[a-z0-9]+)*
```

长度不超过 64；拒绝路径分隔符、`.`/`..`、Python keyword、`stochaflow` 和跨平台保留
文件名。distribution/plugin/目录使用原 slug，Python package 将 `-` 固定替换为 `_`；不
静默改写大小写或其他非法输入。

默认模板是一个数步即可完成的合成回归项目：

- 自定义 DataBuilder 返回小型 structured tensor batches；
- 自定义 Model 注册到 model Registry；
- 自定义 TrainingBuilder 组装薄 TrainingStrategy 与内置 `mse` Objective；
- 自定义 direct-transform SamplingBuilder 复用 checkpoint model，不虚构 Process、
  Dynamics 或 Sampler；
- artifact 使用内置 tensor writer；模板不为了展示 Registry 数量而额外生成 custom writer；
- registry name 使用项目名命名空间，降低不同扩展包的名称冲突；
- `pyproject.toml` 声明 `my-project = "my_project.stochaflow_ext"`；
- 生成的配置是无需补充必填 TODO 的完整可运行配置，直接声明
  `extensions.plugins: [my-project]`；用户以它为起点根据任务修改字段。

该目录树只规定 Stochaflow 示例所需的最小文件，不把整个仓库变成 Stochaflow
workspace。用户可以自行添加其他实验、包内模块、脚本或工具；这些内容既不要求使用
上述目录名，也不会被 `stochaflow` 扫描或驱动。

模板资产作为 `stochaflow.projects` 包资源发布，构建测试必须证明 wheel/sdist 包含所有
文件。资源使用非 dotfile 的 `.tmpl` 名称并通过显式、有序 manifest 映射到 `.gitignore`、
`.gitkeep` 和动态 package 路径，避免 package-data glob 漏掉 dotfile。renderer 只替换少量
预声明 sentinel，不执行模板 Python，也不对含普通花括号的文件使用 `str.format()`。

生成前先校验完整 manifest、目标路径和所有渲染结果，再在目标同级临时目录写入。目标不
存在时以目录 rename 原子发布；平台具备安全 descriptor-relative 文件系统原语时，现有空
真实目录采用明确的非原子 publish 路径：二次确认仍为空，以 exclusive-create 逐文件发布
并记录本次创建项，失败时只回滚本次文件/空目录。不具备这些原语的平台要求删除现有空
目录，由 `init` 创建目标。文件、symlink 或非空目录在任何写入前失败；并发创建任何目标项
都会使 publish 失败而不覆盖。两条路径都不改变 cwd，也不把“支持空目录”和“跨平台原子
替换已有目录”混为一谈。

生成的 `pyproject.toml` 使用标准 PEP 621/setuptools src layout，并默认精确依赖生成它的
Stochaflow 版本；这是 scaffold API 快照，而非跨版本兼容承诺。用户可以自行调整约束或使用
任意包管理器，但模板不生成 uv/Poetry/PDM 专属配置。

### 实现范围与文件

- 新增 `src/stochaflow/projects/`：项目名验证、确定性 renderer 和模板资源；
- 修改 `src/stochaflow/scripts/cli.py`：增加顶层 `init`，不改变 train/sample 的
  项目感知；
- 新增 entry-point discovery、两阶段 activation、插件 provenance 与版本 mismatch policy；
- 配置权威收口仅修改 training/sampling input resolution 和相应 CLI，不增加
  project 数据流；
- 新增训练 `run_manifest.yaml`；sampling manifest 同样记录配置来源、实际插件 provenance、
  version acceptance、checkpoint lineage、启动 cwd 和 runtime-only CLI options；
- 训练、sampling 与 checkpoint metadata 复用同一 provenance/audit 序列化 helper 和稳定
  子结构，避免三套 key 漂移；
- 修改 `pyproject.toml`：确保模板 package data 进入 wheel/sdist；
- 更新配置 reference generator，使顶层 `init` 和 positional NAME 可被校验/渲染；
- 新增 init/CLI 测试，并更新 README、配置入口、扩展、workflow、troubleshooting、
  reference metadata 和本开发决策记录。

### 已完成的验证范围

- config/runtime：上述四种已确认 base-config 路径及最终确认的训练恢复路径、
  resolved config/checkpoint 持久化、训练后采样与 checkpoint-only sampling；
- resume：按 strict-resume/warm-start 边界覆盖命令互斥与 state 恢复范围，另验证
  新 run/旧 state 的 lineage、安全 CLI override 持久化、必需 progress/RNG 字段、全局 RNG
  roundtrip 与 stochastic uninterrupted-vs-resume 等价，以及不兼容资产由 state contract 拒绝；
- scaffold：合法/连字符名称、非法名称、空目录、非空目录、确定性文件内容、合法
  `pyproject.toml`、模板资源完整、symlink/path traversal/隐藏文件拒绝和失败无半成品；
- plugin discovery：显式列表、全量发现、空列表、确定性顺序、同名冲突、缺失 target、
  import failure、配置解析无 import 副作用、相同 selection 进程内幂等、不同 selection
  进程内拒绝、partial import 后终止性 FAILED、并发/re-entrant activation 和 Registry
  冲突上下文；
- version policy：身份变化硬失败；版本变化在交互 TTY 一次确认、非交互默认失败，
  专用 force flag 与 library allow policy 可继续且留下审计记录；
- provenance：entry-point name 精确匹配、distribution canonicalization、external selection
  新增/删除允许、复用同名插件仍校验 identity/version，接受后以当前实际版本作为新基准；
- checkpoint-safe state：v8 全 payload 可由 `weights_only=True` 加载；Tensor/primitive/
  container extra state 通过，custom class、custom tensor subclass、NumPy object 和任意
  pickle global 在保存边界明确失败且不触发插件 import；
- subprocess E2E：在隔离 venv 中用标准 pip 安装当前工作区构建的 Stochaflow wheel，
  通过已安装 console script 执行 `init`，再构建/安装生成项目 wheel 并完成短训练、连续两次
  resume、删除父 run 后的 checkpoint-only sample；全程不使用 `PYTHONPATH`、uv 私有行为
  或 PyPI 上的同版本 Stochaflow，并证明自定义 DataBuilder、Model、
  TrainingBuilder/Strategy 和 SamplingBuilder 均通过 entry point 注册；
- OCP：脚手架不修改 RegistryCatalog、Trainer 或 SamplingBuilder dispatch；生成项目只
  使用公开 extension API；
- public docs/reference：`init` positional 参数和 CLI reference 生成、检查结果确定。

Stage 5 收口执行聚焦测试与常规门禁：

```bash
uv run pytest <stage-5-focused-tests>
uv run python tools/generate_config_reference.py
uv run python tools/generate_config_reference.py --check
uv run ruff check .
uv run pyright
```

聚焦测试覆盖 config/plugin/CLI/checkpoint/scaffold、training resume、sampling runtime、
builder 与 Trainer reporting，并包含“安装 Stochaflow wheel → `init` → 安装 extension wheel
→ train → 连续 resume → 删除父 run → checkpoint-only sample”的隔离环境 E2E。完整 pytest、
build、Sphinx 和其他 CI 静态检查留到整个 feature 分支合并验收，不将其表述为 Stage 5
日常收口结果。

### 独立审查与修复收口

Stage loop 的只读独立审查发现并修复了以下实现边界问题：

- **scaffold TOCTOU 与安全清理：** 不存在目标使用同级 staging directory 发布；成功 rename
  后不再按旧 staging pathname 清理。失败清理绑定到本次创建对象的身份或 descriptor，避免
  并发方复用路径后被递归删除；现有空目录只在平台具备安全 descriptor-relative 原语时支持，
  否则在写入前明确拒绝。
- **strict resume 的 inherited best 身份：** sibling `best.pt` 必须与恢复点的 resolved
  config、extension provenance、best epoch/metric、monitor 和 mode 一致，并先经当前
  TrainingPlan 资产 topology 加载验证；通过后以当前 run 的 config/provenance 原子物化为
  本地 `best.pt`，连续恢复与训练后采样不再依赖父 run。
- **mutable future best 拒绝：** 若历史 epoch checkpoint 旁的可变 `best.pt` 已被未来 epoch
  覆盖，恢复明确失败，不把未来权重误当成当时的 best。
- **EMA device 恢复：** checkpoint 在 CPU 预加载后，恢复的 EMA state 显式迁移到 Trainer
  device，避免后续 update/copy 出现跨设备错误。
- **checkpoint 原子发布：** 普通保存和 inherited-best 本地化共用同目录临时文件与
  `os.replace()`，失败清理临时项，不暴露半写 checkpoint。
- **strict resume 随机流与 progress：** v8 保存 weights-only-safe 的 Python、NumPy、Torch
  CPU/CUDA RNG snapshot，缺失或非法 `epoch`/`global_step` 在修改训练资产前拒绝；全部
  selected/inherited-best state 验证后才恢复适用随机流，并以随机训练不中断/中断续训等价
  回归覆盖。Trainer 在每个 public diagnostic 回调边界隔离三类全局 RNG，使观察性扩展不
  改写训练随机流，也不会因 resume 再次执行 `on_fit_start` 而产生分叉；best/latest
  checkpoint 之后发生的 logger/reporter 回调采用同样隔离，保证保存点就是下一轮的随机
  边界。
- **sampling config/provenance 权威：** 完整外部 sampling config 整体权威，overlay 只替换
  显式 `sampling`/`extensions.plugins`，checkpoint 只提供 state；复用同名插件仍严格校验
  identity/version，本次实际 provenance 成为新 artifact 的基准。strict training resume
  则继续只接受 checkpoint config/state 作为 base。

### 设计取舍

- **采用标准 Python packaging entry point。** extension 身份与 import target 属于
  distribution metadata；Stochaflow 不复制包管理器、环境或源码路径发现。
- **项目创建与环境变更分离。** `init` 只写文件；安装命令由用户和所选包管理器显式执行。
- **模板配置显式选择自己的插件。** entry point 解决可发现性，YAML 决定本次运行激活谁；
  resolved config、checkpoint 和用户看到的配置因此保持同一组件图。
- **环境全量发现是显式 opt-in。** 省略 extensions 默认不导入第三方代码；
  `plugins: null` 可用于通用环境，但生成模板写入确定插件名，避免安装了新 package 后
  悄悄改变运行。
- **版本差异允许知情继续，身份差异不允许。** 专用 force 只跳过 version prompt，不能
  绕过 name/distribution/target 或 state contract。
- **不生成万能 extension 模板。** 一个可运行的最小纵向组合比大量未使用空类更能验证
  API；高级算法 family 继续由文档示例说明。
- **不存在 ProjectManifest。** 项目信息已由 `pyproject.toml` 表达；核心只接收稳定插件
  identity 并执行既有生命周期。
- **sampling 不比较完整 external/checkpoint config。** 二者不是对等候选；显式 config
  选择组件图，checkpoint 只提供 state，兼容性由实际 state/asset contract 验证。训练
  full resume 则以 checkpoint config/state 为权威，不套用 sampling override 语义。

### 已知限制

- `weights_only=True` 防止 checkpoint 反序列化动态导入任意 extension/global，但不能保证
  恶意输入不会造成 DoS、超大内存分配或底层 Tensor decoder 风险；不可信 artifact 仍需
  外层来源校验与资源隔离；
- checkpoint 不保存扩展源码；移动 checkpoint 后必须在 CLI 所在环境安装对应扩展包；
- 依赖锁定由用户选择的包管理器与 lockfile 负责，checkpoint 本身不是 Python 环境快照；
- distribution version 不能检测 editable source 在版本号不变时的代码修改；
- 用户自定义 Builder 的 `params` 是不透明 mapping，核心无法也不应自动识别和
  重写其中的路径；如需不同基准，由该 Builder 显式定义。
- decorator Registry 与 Python module import 都是进程全局状态；首版一个进程只允许固定
  一套插件 selection。需要连续运行不同 selection 的 library 嵌入场景使用独立进程，
  Stage 5 不引入 owner-aware Registry view 或 unload。
- `extensions.plugins` 控制 Stochaflow activation 层导入的 entry point，不是 Python import
  沙箱。受支持的第三方聚合模块只能由 activation 层导入，且不得传递导入其他 distribution
  的注册模块；调用方提前手动导入产生的 Registry 状态不属于 checkpoint provenance 保证。
- strict resume 只能继承仍与所选恢复点一致的 mutable sibling `best.pt`。若该文件已被更晚
  训练覆盖，框架会拒绝恢复，而不会猜测历史 best；任意历史恢复需要 versioned immutable
  best 或在每个 checkpoint 内嵌 best snapshot，留待真实存储需求出现后设计。匹配的 best
  会先本地化到新 run，因此完成恢复后不再依赖父 run。
- 首版只生成一个合成任务，不代表用户仓库必须采用某个科学领域、模型 family 或
  非 Stochaflow 实验布局。

### 逻辑提交

`Stage 5: Add entry-point extension projects`

## Stage 6：大规模 Sampling artifact 容量验收门

### 目标

在 Physics AI 案例前验证当前“SamplingBuilder 先将全部 CPU batch/trajectory 放入
`SamplingOutput`，Writer 随后统一写出”的内存边界。该 Stage 先用真实规模证据决定是否
需要 streaming，不为了理论完整性预建复杂事件系统。

### 评估与决策

- 为高分辨率图像、3D physics field、多 batch 和长 trajectory 建立可重复的峰值内存
  基准，分别记录 accelerator state、CPU output 和 writer 编码开销；
- 明确普通离线 `SamplingOutput` 的推荐规模和失败提示，避免大型运行无界累积后才 OOM；
- 若 Physics AI 目标规模在可接受预算内，保留当前 API，并在文档记录容量边界；
- 若证据表明必须增量输出，再设计最小的 batch lifecycle（例如 begin/write_batch/finish
  或等价 sink），同时保持 Builder 负责任务组装、Writer 负责 artifact I/O；不得让 core
  runtime 解释 field/image/trajectory 语义；
- streaming 决策必须覆盖失败传播、临时文件清理、artifact key 唯一性和最终 manifest，
  不能仅解决 Tensor 内存而破坏 writer contract。

### 验收条件

- 有一份可复现的容量报告和明确结论；
- Stage 7 Physics AI 的预期 shape、sample 数与 trajectory 设置经过该结论验证；
- 如无需新 API，文档明确安全工作集和用户自定义分批 Builder/Writer 的方式；
- 如需要新 API，先更新本计划中的 sampling lifecycle，再实现并完成 tensor/image/custom
  writer 回归，不在 Stage 7 案例中临时修改核心。

### 交付

本 Stage 至少形成容量决策记录；只有实际修改 runtime/writer API 时才形成独立逻辑提交。

## Stage 7：纵向扩展案例

### 目标

用彼此不同的真实任务验证扩展轴，而不是为单一案例修改核心抽象。所有案例使用 Stage 5
生成的标准项目结构，并受 Stage 6 的容量结论约束。

### 7A：Physics AI super-resolution / reconstruction

- 以独立用户项目实现条件模型和 physics 数据处理；
- 复用内置 `DiscreteGaussianProcess` 和 DDPM/DDIM Sampler；
- 使用自定义 TrainingBuilder 组装模型/Objective/Process 和只解释 LR/HR 或
  时序场 batch 的 TrainingStrategy；
- 使用自定义 SamplingBuilder 实现条件输入、partial noising、physics guidance 或所需
  sampling state；
- low-resolution、physics state 和模型签名由 Builder/model callable 拥有；若 physics
  correction 能表达为 Gaussian prediction 或生成方向修正，则实现或包装
  `GaussianDenoisingDynamics`，继续复用 DDPM/DDIM；
- 若 correction 改变 reverse transition、accepted-step 更新或内部子步，则在用户扩展中
  定义匹配的窄 Dynamics 与 Sampler，而不是向内置 DDPM/DDIM 塞入 physics callback；
- 使用自定义 writer 输出场数据和指标；图像预览只是可选 artifact；
- 案例不得向核心 Process/Sampler 添加 physics、PDE、super-resolution 专用参数。

### 7B：知识蒸馏

- 使用自定义 TrainingBuilder 构建教师模型、加载 checkpoint、冻结并将其
  以固定 eval auxiliary module 交给核心托管；
- 提供可复用 Objective 和只定义蒸馏计算的 `KnowledgeDistillationStrategy`；
- Strategy 组合基础与蒸馏 Objective 结果为单一 total loss，并记录分项指标。

### 验收条件

- 普通图像和 paired super-resolution 用户复用 Stage 2 recipe，无需自定义 builder；
- Physics 案例只通过注册扩展与配置组合复用 Process/Sampler，核心无任务分支；
- 更换 DDPM/DDIM 只改变 sampling 配置或 Builder 参数；
- 教师参数无梯度且保持不变，学生正常更新；
- teacher 与额外 Objective 通过 `training_assets_state_dict` 恢复，sampling 不构建它们；
- 两个案例共同证明数据、训练和采样三个扩展轴可以独立替换；
- 大型样本和 trajectory 符合 Stage 6 已验证的输出容量策略。

### 逻辑提交

`Add extension reference projects`

## Stage 8：可复现性与文档收尾

### 实施内容

- resolved config 与 checkpoint metadata 记录已选择的插件 identity、provenance 和所有
  注册组件名，包括显式的 `process: null`；项目和完整依赖锁定由 `pyproject.toml` 与
  用户选择的 lockfile 记录；
- 缺失扩展时报告 entry-point name、distribution、target 和当前 Python executable，
  并建议将对应 extension 安装到 CLI 所在环境；
- 文档覆盖项目创建、自定义 DataBuilder、structured batch、可选 Process、family Dynamics、
  Sampler、SamplingBuilder、模型、Objective、TrainingBuilder/Plan 和 TrainingStrategy；
- 单独提供“复用内置 Process/Sampler 完成新任务”和“不使用 Process 的自定义 family”
  最小教程；
- 记录破坏性变更、checkpoint 可移植性、capability compatibility 和 sampling 容量边界；
- 更新 README、配置参考、故障排查与 API reference；
- 完成构建与质量检查。

### 最终验证

```bash
uv run python tools/generate_config_reference.py
uv run python tools/generate_config_reference.py --check
uv run pytest
uv run ruff check .
uv run pyright
uv build
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Open–Closed 验收矩阵

以下变化必须不修改核心 dispatch：

| 变化 | 只需新增或替换 | 可复用 |
| --- | --- | --- |
| 新数据组织或 split | DataBuilder | Trainer、TrainingStrategy |
| 新模型签名或 condition | model adapter / Strategy / Builder | Process、Sampler |
| 新 probability path | Process | 模型、兼容 Sampler |
| 新数值求解器 | family Sampler | 同 family 的 Process/Dynamics、模型 |
| 新 guidance 或初始化方式 | SamplingBuilder | Process、Sampler、writer |
| 新算法 family | 可选 family Process、Dynamics、Sampler、SamplingBuilder | Registry、config、checkpoint、sampling runtime |
| 无 Process 的直接生成方法 | SamplingBuilder，及需要时的窄 Dynamics | Registry、config、checkpoint、sampling runtime |
| 新训练任务或多目标 | TrainingBuilder + TrainingStrategy / 可选 Objective | DataBuilder、可选 Process、Trainer |
| teacher 或其他训练辅助模块 | TrainingBuilder 返回的具名 managed asset | Trainer、checkpoint、primary model |
| 新 artifact | SamplingArtifactWriter | sampling runtime |

若新增上述能力需要在 runner 中按组件名称添加 `if/elif`、给通用数据 schema 增加任务字段，
或让 Process/Sampler 接收某个任务的 condition 参数，则视为扩展边界失败，先回到设计阶段。

## 首版明确不做

- 自动扫描 `extensions/` 目录；
- 远程插件市场、插件索引或自动下载安装（标准 entry-point discovery 属于 Stage 5）；
- Stochaflow 专用 project manifest、source-root 激活或 train/sample `--project`；
- 全局 Dataset/Sampler/DataLoader/Split/Batching Registry 体系；
- 为所有 Process 强制统一的巨大数学接口；
- 要求所有生成或训练算法为了通用 dispatch 虚构 Process；
- 全局 GenerativeDynamics Registry 或静态 process/sampler 名称兼容矩阵；
- 在 `GenerativeDynamics` 根类型上增加 universal `predict`、`step`、`drift`、`score` 或
  `denoise` 方法；
- 多 optimizer、交替更新或 extension 接管完整 epoch 循环；
- 由核心解释的通用 Objective graph、target adapter 或 condition adapter 配置系统；
- 在 Stage 6 没有容量证据前预建复杂的 streaming event/bus API；
- 将用户扩展源码打包进 checkpoint；
- 自动上传或分发用户项目；
- legacy YAML、Registry 别名或旧 checkpoint 迁移。

## 施工规则

- 严格按 Stage 顺序实施；Stage 3 的可选 Process、离散 Gaussian 命名和 schedule 状态
  语义未验收前，不进入 Stage 4；
- 每个 Stage 独立开发、测试和验收，未通过不进入下一 Stage；
- 每个 Stage 形成一个或文档明确列出的少量逻辑提交，避免跨 Stage 修改；
- Stage 4 的 TrainingBuilder/Plan/Strategy/Objective/checkpoint/diagnostic 边界必须先于
  Stage 4.1；native-provider 收口必须先于 Stage 5 项目模板，使模板不再生成重复的
  PyTorch Registry alias 或过时配置；
- Stage 6 的容量结论是 Stage 7 Physics AI 案例的入口条件；案例不得边做边修改通用
  sampling lifecycle；
- 新 capability 只由真实的第二种实现驱动，不预先添加“可能有用”的通用方法；
- 如实施中发现必须改变公开接口或 OCP 边界，先更新本计划并确认，再继续施工。
