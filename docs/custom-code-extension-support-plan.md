# 自定义代码扩展支持实施计划

- 状态：Stage 4 完成，Stage 5 待实施
- 制定日期：2026-07-17
- 最近修订：2026-07-21
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
→ stochaflow project create my_project
→ 编写并注册 extension
→ 在项目清单中声明 extension 模块
→ 在 YAML 中选择数据 builder、模型、TrainingBuilder、可选 Objective，以及采样所需的
  Process/SamplingBuilder
→ stochaflow train / stochaflow sample
```

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
- 通过项目清单显式加载扩展，不扫描目录；
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
distillation 则不需要 auxiliary teacher。需要联合更新 teacher、独立 teacher optimizer、
EMA teacher 或交替优化的方案具有不同训练 lifecycle，不通过继续给 Strategy 增加可选
控制字段来兼容，而应在出现真实需求后定义新的训练 loop family。

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

## Stage 5：用户项目系统与 CLI 脚手架

### 目标

在数据、训练和采样公共 API 均已存在后，让用户无需操作 `PYTHONPATH` 即可创建和使用
独立 Stochaflow 项目。模板不再引用尚未实现的扩展契约。

### 实施内容

- 新增命令：

  ```bash
  stochaflow project create my_project
  stochaflow train --project /path/to/project --config configs/train.yaml
  stochaflow sample --project /path/to/project --checkpoint checkpoints/best.pt
  ```

- 创建以下项目结构：

  ```text
  my_project/
  ├── stochaflow.project.yaml
  ├── pyproject.toml
  ├── .gitignore
  ├── configs/
  │   └── train.yaml
  ├── src/
  │   └── my_project/
  │       ├── __init__.py
  │       └── extensions/
  │           └── __init__.py
  └── tests/
      └── test_extensions.py
  ```

- 项目清单首版结构：

  ```yaml
  schema_version: 1

  project:
    name: my_project
    source_roots: [src]

  extensions:
    modules: [my_project.extensions]
  ```

- 项目发现顺序：显式 `--project`、配置目录向上、当前工作目录向上；找不到项目时继续
  支持独立配置；
- CLI 加载 `source_roots` 后按清单顺序导入扩展；
- 清单模块与配置中的 `extensions.modules` 按顺序合并并稳定去重；
- 项目名中的连字符转换为 Python 包名下划线；非空目标目录拒绝覆盖；
- 模板按独立小示例展示 DataBuilder、Model、TrainingBuilder/Strategy、可选 Objective、可选 Process、
  Sampler 和 SamplingBuilder 注册，不暗示每个项目必须实现全部角色。

### 验收条件

- CLI 可以生成合法、可安装、可测试的 `src` 项目；
- 自定义 DataBuilder/TrainingBuilder 可以直接被 `train` 使用，自定义 SamplingBuilder
  可以直接被 `sample` 使用；
- checkpoint-only sampling 使用同一项目发现和扩展导入逻辑；
- 临时合成数据完成端到端 CLI 测试，不依赖网络下载；
- 生成模板只使用 Stage 1–4 已稳定的公开入口。

### 逻辑提交

`Add extension project scaffolding`

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

- resolved config 与 checkpoint metadata 记录项目、清单版本、扩展模块和所有已选择的
  注册组件名，包括显式的 `process: null`；
- 缺失扩展时报告模块、项目根目录与 `--project` 修复建议；
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
- Python entry point 插件市场；
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
- Stage 4 的 TrainingBuilder/Plan/Strategy/Objective/checkpoint/diagnostic 边界必须先于 Stage 5 项目模板，
  模板不得引用未来 API；
- Stage 6 的容量结论是 Stage 7 Physics AI 案例的入口条件；案例不得边做边修改通用
  sampling lifecycle；
- 新 capability 只由真实的第二种实现驱动，不预先添加“可能有用”的通用方法；
- 如实施中发现必须改变公开接口或 OCP 边界，先更新本计划并确认，再继续施工。
