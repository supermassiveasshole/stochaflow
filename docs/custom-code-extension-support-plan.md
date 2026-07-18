# 自定义代码扩展支持实施计划

- 状态：Stage 2 完成，Stage 3 待实施
- 制定日期：2026-07-17
- 最近修订：2026-07-19
- 目标分支：`feature/custom-code-extension-support`

## 目标

将 Stochaflow 从包含内置实验能力的 Python 包扩展为“核心框架 + 可复用组件 +
用户项目扩展”的训练与采样框架。用户安装 Stochaflow 后，可以：

1. 直接组合内置数据 recipe、模型、概率过程、训练策略和采样器；
2. 只替换任务特有的部分，例如条件模型、数据组织、guidance 或 sampling builder；
3. 注册全新的过程、数值采样算法或训练方法，而无需修改核心分发逻辑；
4. 继续通过统一 CLI 和 YAML 完成训练、恢复与采样。

目标用户流程：

```text
安装 stochaflow
→ stochaflow project create my_project
→ 编写并注册 extension
→ 在项目清单中声明 extension 模块
→ 在 YAML 中选择或组合数据 builder、模型、Process、Strategy 和 Sampler
→ stochaflow train / stochaflow sample
```

## 架构原则

### Open–Closed Principle

Stochaflow 核心对扩展开放、对任务特例关闭。一个与现有契约兼容的新 DataBuilder、
Process、Sampler、SamplingBuilder、Loss 或 TrainingStrategy，应当只需新增实现、注册和
配置，不应修改核心 runner、CLI 或按名称分支的 dispatch 代码。

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

Process + model callable / prediction adapter
  └─ produces GenerativeDynamics
       └─ consumed by Sampler / numerical solver
            └─ composed by SamplingBuilder for a concrete task
```

- **DataBuilder** 是唯一的核心数据扩展入口，拥有完整数据组织逻辑；核心 Trainer 只
  消费组装好的 loader 和原样 structured batch。
- **Process** 描述 probability path、forward/noising law、marginal、必要的数学转换和
  可构造的生成动力学，不拥有任务模型，也不执行采样循环。
- **GenerativeDynamics** 是 Process 与模型预测组合后的运动规律，例如 reverse kernel、
  vector field、reverse SDE 或 denoiser function。它首先是组件间的窄协议，不设全局
  Registry。
- **Sampler** 是完整的数值执行算法，负责时间/噪声离散、solver state、随机增量、
  多步历史和循环。它不解释 condition，也不拥有业务模型。
- **TrainingStrategy** 和 **SamplingBuilder** 分别是训练与推理的任务组合层，负责解释
  batch、调用模型、条件注入、guidance、prediction semantics 以及兼容性检查。

这样的划分吸收两类现有实现的优点，而不复制它们的耦合：OpenAI diffusion 将高复用的
Gaussian math 集中在 `GaussianDiffusion`，但也合并了训练 Loss、模型输出解释和采样
循环；Diffusers Scheduler 支持广泛的 DDPM、DDIM、SDE/ODE 与 flow-matching solver，
但一个 Scheduler 通常同时承担部分 Process、推理时间表和 reverse step。Stochaflow
将 model-free Process、完整 Sampler 和任务 Builder 分开，使三者可独立扩展和复用。

首版默认边界：

- 使用本地 `src` 布局项目；
- 通过项目清单显式加载扩展，不扫描目录；
- Dataset 单样本与 batching 由 DataBuilder 按需定义；送入 Trainer 的 batch 可以是
  `Tensor`，或由 mapping、tuple、list 和 Tensor 组成的嵌套结构；
- 提供开箱即用的普通图像、多分辨率图像和 paired super-resolution recipe，但这些
  recipe 的参数不是核心数据契约；
- 支持多模型、多 Loss，但首版只有一个优化器和一次反向传播；
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

## Stage 3：可注册 Process、Generative Dynamics 与 Sampler

### 目标

拆除当前 `GaussianDiffusion` 对 probability process、模型调用、prediction type、reverse
transition 和 sampling loop 的捆绑，并消除 `REGISTRIES.diffusions` 同时选择训练算法与
采样器的语义重载。

Stage 3 建立足以覆盖离散 Gaussian diffusion、score SDE、probability-flow ODE、
sigma-space solver 和 flow matching 的组合边界，但首批只迁移并验证现有 DDPM/DDIM。

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

顶层 `process` 是训练与 checkpoint 的概率路径身份；`sampling.builder` 描述一次推理任务
如何组合模型、Process 和 Sampler。用户可以覆盖 builder/sampler 的推理参数而不修改
checkpoint 中的训练 Process，但 Builder 必须验证兼容性。

### Process 契约

`Process` 是可注册、可配置、可迁移到 device 且可进入 checkpoint 的最小基类。它不拥有
模型、不读取 structured batch、不编码 condition，也不执行训练或采样循环。

避免要求所有 Process 实现一套臃肿接口。实际数学能力使用窄协议表达，例如：

- `MarginalProcess`：从 clean state 得到指定 time 的 marginal sample；
- `GaussianPredictionProcess`：在 epsilon、x0、v、score 等参数化之间转换；
- `ReverseKernelProcess`：由模型预测构造离散 reverse transition；
- `SDEProcess`：提供 drift、diffusion 和 reverse-SDE 所需量；
- `ProbabilityPath` / `VectorFieldProcess`：提供连续 path 或 velocity/score 转换。

这些名称是设计方向，不要求 Stage 3 一次冻结全部公开协议。Stage 3 只公开现有内置实现和
Sampler 真正依赖的最小 capability；新增算法时再以实际复用需求扩展协议。

内置 `DiscreteGaussianProcess` 从当前实现中保留：

- beta/alpha/noise schedule 与 timestep 定义；
- `add_noise`、marginal 系数和 posterior math；
- epsilon、x0、v、score 等必要预测转换；
- 从 model output 构造 DDPM/DDIM dynamics 所需的数学量。

它移除 model ownership、`model(xt, t)` 固定调用、Loss 和 sample loop。

### Generative Dynamics 边界

“forward process”与“reverse process”不足以覆盖全部算法。DDPM/DDIM 中生成方向表现为
reverse transition；在 SDE/ODE 中可能是 reverse SDE 或 probability-flow vector field；
flow matching 则直接学习生成方向的 velocity。因此统一概念使用
`GenerativeDynamics`。

首版 Dynamics 是由 Process、模型 callable 和 prediction adapter 组合出的普通对象或
Protocol，不设全局 Registry，也不出现在顶层 YAML。可出现的窄形态包括：

- `ReverseKernel`：给出离散 `x_t → x_s` 的分布或一步 sample；
- `VectorField`：给出 `dx/dt`；
- `ReverseSDE`：给出 reverse drift 与 diffusion；
- `DenoiserFn`：在 sigma-space 返回 denoised state。

Sampler 声明自己消费哪种 capability；SamplingBuilder 负责构造并校验。核心不维护
`process × sampler` 名称兼容矩阵。

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
时，应只新增 Sampler 和必要的窄 capability，不修改现有 Process 与核心 runtime。

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

- checkpoint 格式升级并保存 Process 注册名、配置及 state；
- Process 是训练路径的 checkpoint-authoritative 组件；自定义 Process 可以包含 buffer 或
  可学习状态；
- sampler 和 SamplingBuilder 是可覆盖的推理配置，不保存一次运行的 solver history；
- checkpoint-only sampling 先加载扩展，再构建 Process、推理模型、Builder 和 Sampler；
- sampling runtime 不再假设 sampler 返回单一 Tensor，由 Builder 规范化结果后交给 writer；
- `sampling.shape` 继续可选；固定 shape 的 Builder/Sampler 在运行时自行要求；
- diagnostic 不依赖具体 DDPM/DDIM 类，只声明自己需要的 Process/Dynamics capability。

### 验收条件

- 新 Process、新 Sampler 和新 SamplingBuilder 均可通过“新增类 + 注册 + YAML”接入，
  核心 runner 无名称分支；
- `DiscreteGaussianProcess` 不持有模型，DDPM/DDIM Sampler 不解释 batch 或 condition；
- DDPM ancestral、DDIM、partial denoising、trajectory 和 checkpoint-only sampling 回归通过；
- 同一个 Gaussian Process 可更换 DDPM/DDIM Sampler，同一个 Sampler 可使用用户模型
  adapter；
- 自定义 condition/guidance Builder 可复用内置 Process 与 Sampler；
- capability 不匹配在构建 sampling run 时给出包含组件名和缺失能力的错误；
- 配置、公开 API、扩展手册、sampling 文档与 reference 同步更新；
- Pytest、Ruff 和 Pyright 全部通过。

### 逻辑提交

`Separate diffusion processes and samplers`

## Stage 4：用户项目系统与 CLI 脚手架

### 目标

让用户无需操作 `PYTHONPATH`，即可创建和使用独立 Stochaflow 项目。

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
- 模板展示分别注册 DataBuilder、Process、Sampler、SamplingBuilder、Model、Loss 和
  TrainingStrategy，但不要求项目一次实现所有扩展点。

### 验收条件

- CLI 可以生成合法的 `src` 项目；
- 自定义组件可以直接被 `train` 和 `sample` 使用；
- checkpoint-only sampling 使用同一项目发现逻辑；
- 临时合成数据完成端到端 CLI 测试，不依赖网络下载。

### 逻辑提交

`Add extension project scaffolding`

## Stage 5：Loss 与 TrainingStrategy 扩展 API

### 目标

移除 `diffusion + objective → train_step_fn` 的硬编码，将 batch 解释、模型调用、
Process 使用和 Loss 组合提升为正式训练扩展点。

### 配置接口

```yaml
training:
  strategy:
    name: diffusion_denoising
    params: {}
  losses:
    primary:
      name: epsilon_mse
      params: {}

trainer:
  num_epochs: 30
  device: auto
```

- `training` 描述训练算法与 Loss，`trainer` 描述通用循环；
- 移除顶层 `objective`，旧字段作为未知字段报错；
- 新增 `REGISTRIES.losses` 与 `REGISTRIES.training_strategies`，移除
  `REGISTRIES.objectives`。

### TrainingStrategy 契约

公开策略提供：

- `training_step(batch)` 与 `evaluation_step(batch)`；
- `trainable_parameters()`；
- `to(device)`、`train_mode()` 与 `eval_mode()`；
- `state_dict()` 与 `load_state_dict()`；
- 明确的 `primary_model` / `inference_model` 访问能力，使 EMA、checkpoint 和
  checkpoint-only sampling 不依赖特定策略内部字段。

`TrainStepOutput` 包含可反传总 `loss`、分项 `metrics` 与供 diagnostic 使用的中间结果。
核心 Trainer 继续负责一次反向传播、单 optimizer step、scheduler、EMA、日志、验证、
early stopping 和 checkpoint。

TrainingStrategy 接收 Stage 2 的 structured batch，并自行解释 state、condition、target、
mask 等字段；它通过 Stage 3 的 Process capability 完成 noising、target conversion 或
其他训练数学，不调用 Sampler，也不要求 Process 拥有模型。

### Checkpoint

- checkpoint 加入可选策略 state，不重复保存主模型权重；
- EMA/推理模型身份明确保存；
- sampling 只加载推理所需模型与 Process，不强制实例化训练策略；
- 新格式不承诺读取旧 checkpoint。

### 验收条件

- 内置 Gaussian epsilon denoising 训练结果保持不变；
- 用户可由 YAML 构建自定义 Loss 和 TrainingStrategy；
- 条件模型、多个 Loss、教师模型和自定义 batch 解释不修改核心 Trainer；
- 分项 Loss、checkpoint、恢复训练、EMA 与采样测试通过。

### 逻辑提交

`Add extensible training strategies`

## Stage 6：纵向扩展案例

### 目标

用彼此不同的真实任务验证扩展轴，而不是为单一案例修改核心抽象。

### 6A：Physics AI super-resolution / reconstruction

- 以独立用户项目实现条件模型和 physics 数据处理；
- 复用内置 `DiscreteGaussianProcess` 和 DDPM/DDIM Sampler；
- 使用自定义 TrainingStrategy 解释 LR/HR 或时序场 batch；
- 使用自定义 SamplingBuilder 实现条件输入、partial noising、physics guidance 或所需
  sampling state；
- 使用自定义 writer 输出场数据和指标；图像预览只是可选 artifact；
- 案例不得向核心 Process/Sampler 添加 physics、PDE、super-resolution 专用参数。

### 6B：知识蒸馏

- 提供自定义蒸馏 Loss 与 `KnowledgeDistillationStrategy`；
- 构建教师模型、加载 checkpoint、冻结并保持 eval；
- 组合基础 Loss 和蒸馏 Loss，记录分项指标；
- 验证策略状态与 checkpoint resume。

### 验收条件

- 普通图像和 paired super-resolution 用户复用 Stage 2 recipe，无需自定义 builder；
- Physics 案例只通过注册扩展与配置组合复用 Process/Sampler，核心无任务分支；
- 更换 DDPM/DDIM 只改变 sampling 配置或 Builder 参数；
- 教师参数无梯度且保持不变，学生正常更新；
- 两个案例共同证明数据、训练和采样三个扩展轴可以独立替换；
- 所有示例使用项目脚手架生成的标准结构。

### 逻辑提交

`Add extension reference projects`

## Stage 7：可复现性与文档收尾

### 实施内容

- resolved config 与 checkpoint metadata 记录项目、清单版本、扩展模块和所有已选择的
  注册组件名；
- 缺失扩展时报告模块、项目根目录与 `--project` 修复建议；
- 文档覆盖项目创建、自定义 DataBuilder、structured batch、自定义 Process、
  Dynamics capability、Sampler、SamplingBuilder、模型、Loss 和 TrainingStrategy；
- 单独提供“复用内置 Process/Sampler 完成新任务”的最小教程；
- 记录破坏性变更、checkpoint 可移植性和 capability compatibility 错误；
- 更新 README、配置参考、故障排查与 API reference；
- 完成构建与质量检查。

### 最终验证

```bash
uv run python tools/generate_config_reference.py
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

## Open–Closed 验收矩阵

以下变化必须不修改核心 dispatch：

| 变化 | 只需新增或替换 | 可复用 |
| --- | --- | --- |
| 新数据组织或 split | DataBuilder | Trainer、TrainingStrategy |
| 新模型签名或 condition | model adapter / Strategy / Builder | Process、Sampler |
| 新 probability path | Process | 模型、兼容 Sampler |
| 新数值求解器 | Sampler | 兼容 Process/Dynamics、模型 |
| 新 guidance 或初始化方式 | SamplingBuilder | Process、Sampler、writer |
| 新训练任务或多 Loss | TrainingStrategy / Loss | DataBuilder、Process、Trainer |
| 新 artifact | SamplingArtifactWriter | sampling runtime |

若新增上述能力需要在 runner 中按组件名称添加 `if/elif`、给通用数据 schema 增加任务字段，
或让 Process/Sampler 接收某个任务的 condition 参数，则视为扩展边界失败，先回到设计阶段。

## 首版明确不做

- 自动扫描 `extensions/` 目录；
- Python entry point 插件市场；
- 全局 Dataset/Sampler/DataLoader/Split/Batching Registry 体系；
- 为所有 Process 强制统一的巨大数学接口；
- 全局 GenerativeDynamics Registry 或静态 process/sampler 名称兼容矩阵；
- 多 optimizer、交替更新或 extension 接管完整 epoch 循环；
- 将用户扩展源码打包进 checkpoint；
- 自动上传或分发用户项目；
- legacy YAML、Registry 别名或旧 checkpoint 迁移。

## 施工规则

- 严格按 Stage 顺序实施；Stage 2 必须重新验收后才能进入 Stage 3；
- 每个 Stage 独立开发、测试和验收，未通过不进入下一 Stage；
- 每个 Stage 形成一个或文档明确列出的少量逻辑提交，避免跨 Stage 修改；
- Stage 3 先冻结最小 Process/Sampler/Builder 边界，再迁移 DDPM/DDIM；
- 新 capability 只由真实的第二种实现驱动，不预先添加“可能有用”的通用方法；
- 如实施中发现必须改变公开接口或 OCP 边界，先更新本计划并确认，再继续施工。
