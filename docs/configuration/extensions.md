# 扩展与 Registry

顶层扩展通过标准 Python packaging entry point 发现，再由 `extensions.plugins` 按稳定
插件名选择。用户 project 是普通、可安装的 Python distribution，并且必须与 Stochaflow
CLI 安装在同一 Python environment。Stochaflow 不扫描源码目录、工作区或 package 命名
约定，也不管理 pip、uv、conda、Poetry 或 PDM。

配置中的相对数据/输出路径以及不透明 builder 参数都以进程启动 cwd 为基准。生成项目的
默认配置使用 `data/...` 与 `outputs/...`，因此建议从项目根目录运行。

## RegistryCatalog

| Registry | 扩展契约 | YAML 选择位置 |
| --- | --- | --- |
| `data_builders` | `DataBuilder` 类 | `data.name` |
| `sampling_artifact_writers` | `SamplingArtifactWriter` 类 | `sampling.writers[].name` |
| `models` | `torch.nn.Module` 类 | `model.name` |
| `noise_schedules` | `GaussianNoiseSchedule` 类 | `process.params.schedule` |
| `processes` | `Process` 类 | `process.name` |
| `samplers` | `Sampler` 类 | `sampling.sampler`，或 checkpoint recipe 的 fixed contract |
| `sampling_builders` | `SamplingBuilder` 类 | `TrainingPlan.inference_recipe.name` 固化进 checkpoint；sample request 不直接选择 |
| `training_builders` | `TrainingBuilder` 类 | `training.name` |
| `objectives` | `torch.nn.Module` 类 | `objective.name` |
| `metrics` | `torchmetrics.Metric` 类 | `metrics[].name` |
| `optimizers` | 第三方 `torch.optim.Optimizer` 子类 | `optimizer.name` |
| `lr_schedulers` | Stochaflow 自有或第三方 `LRScheduler` 子类 | `lr_scheduler.name` |
| `loggers` | `ExperimentLogger` 类 | `logging.backends[].name` |
| `diagnostics` | `TrainingDiagnostic` 类 | `diagnostics[].name` |

Dataset、partition、degradation、PyTorch 数据 sampler、collate 和 DataLoader 不拥有
全局 Registry。`IMAGE_DATA_SOURCES` 是内置 image recipes 专用的窄 source-adapter
Registry，不属于 `RegistryCatalog`，也不是通用 Dataset Registry。一个自定义
`DataBuilder` 完整拥有不同的数据组合及其兼容性。生成算法的 `Sampler` 则通过
`REGISTRIES.samplers` 注册，由 checkpoint 选定的 SamplingBuilder 组合。

标准 PyTorch optimizer 和 LR scheduler 不在上述 Registry 中逐项注册。配置可以直接写
受限原生 target：

```yaml
optimizer:
  name: torch.optim.AdamW
  params: {lr: 0.0002, weight_decay: 0.01}

lr_scheduler:
  name: torch.optim.lr_scheduler.CosineAnnealingLR
  interval: epoch
  params: {T_max: 100}
```

原生 resolver 只访问 `torch.optim.<Class>` 和
`torch.optim.lr_scheduler.<Class>`，不是任意 Python class-path importer。核心分别注入
trainable parameters 与 optimizer，再将配置 `params` 原样传给构造器。完整参数和默认值
以 [PyTorch optimizer](https://docs.pytorch.org/docs/stable/optim.html) 与
[LR scheduler](https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
文档为准；Stochaflow 不复制上游 namespace 或签名。
这两个前缀也是保留 namespace，扩展不能用它们作为 Registry 名称。

Registry 仍用于真正的第三方 optimizer/scheduler 子类，以及增加 Stochaflow 专属语义的
自有实现。它们必须保持相同构造协议：optimizer 的第一个位置参数接收 parameters；
scheduler 的第一个位置参数接收 optimizer，且构造后保留同一对象。当前自动 Trainer 还要求
optimizer 与 scheduler 的 bound `step()` 都可以无参数调用。需要 closure 的 optimizer 或
metric 的 scheduler 等到框架拥有明确的 lifecycle 后再接入，不能靠具体类名特判。

## Entry-point 插件发现与激活

extension distribution 在 `pyproject.toml` 声明聚合注册模块：

```toml
[project.entry-points."stochaflow.extensions"]
my-project = "my_project.stochaflow_ext"
```

entry-point name `my-project` 是配置和 checkpoint 使用的稳定插件身份。target 必须是纯
module，不能含 `:callable` 或 extras。聚合模块可以导入**本 distribution 内**多个注册
模块，但导入阶段只应定义类/函数并运行 decorator；不能读取数据、构建模型、加载
checkpoint 或启动任务。

```python
# my_project/stochaflow_ext/__init__.py
from . import data, model, sampling, training  # noqa: F401
```

```python
# my_project/stochaflow_ext/model.py
from stochaflow.extensions import REGISTRIES


@REGISTRIES.models.register("my-project.physics-operator")
class PhysicsOperator(...):
    ...
```

```yaml
extensions:
  plugins:
    - my-project

model:
  name: my-project.physics-operator
  params: {}
```

entry-point name 与 Registry component name 是两层身份：`my-project` 选择一个已安装
distribution 的聚合注册模块，`my-project.physics-operator` 选择该模块注册的一个具体
模型。第三方组件应使用稳定的项目 namespace，避免与其他插件或未来内置名称冲突。

插件激活是 provenance 与确定性加载边界，不是 Python import sandbox。聚合模块在当前
进程中执行普通 Python 代码，因此只应安装并选择可信 distribution。Registry 也不记录
每个 component 的 distribution owner：不要在聚合模块中转导其他 distribution 的注册
模块，也不要在 runtime 激活前手动 import 注册模块。这样产生的注册项可能可见，但不会
自动进入 checkpoint 的 plugin provenance，框架不保证其恢复审计或版本检查。需要组合
多个 distribution 时，应给每个 distribution 声明自己的 entry point，并在
`extensions.plugins` 中显式选择它们。

选择规则是：

- 省略 `extensions` 或写 `plugins: []`：不加载第三方插件；
- 写 `plugins: null`：显式选择当前环境发现的全部插件；
- 写非空列表：只选择精确匹配的 entry-point names；
- resolved config 始终保存经过解析的确定插件列表，不保存 `null`；
- partial sample request 若明确包含 `extensions.plugins`，该列表只追加插件；不能删除
  checkpoint-required selection，也不能写 `null`。仅写 `extensions: {}` 或空列表保留
  checkpoint selection。

`load_config()` 与 `load_config_dict()` 是无副作用解析函数，不导入插件。runtime 先发现
distribution metadata、解析 selection，并在任何插件代码运行前检查 checkpoint 中保存的
插件 identity/version；接受 version policy 后才导入聚合模块并构建 Registry 组件。配置
mapping 和传入的 config 对象不会被原地修改。

插件按 entry-point name、canonical distribution name 和 target 确定性排序。同名 entry
point、缺失插件、非 module target、非法 distribution metadata 或导入失败都会给出包含
插件 provenance 的明确错误。注册名仍必须非空且唯一；错误基类、重复名称和未知名称会
抛出 `RegistryError`。

一次进程首次成功激活后，插件 selection 固定；相同 selection 可重复使用，不同 selection
必须在新进程运行。activation 过程中出现 partial import 或 Registry 冲突时，由于 decorator
注册无法事务回滚，该进程进入失败状态并要求重启。

第三方 runtime 也可以直接使用 `prepare_extension_plugins()` 与
`activate_extension_plugins()`。前者只做 discovery/provenance 预检并返回 immutable plan；
后者才导入代码，默认以 `ExtensionVersionPolicy.REJECT` 拒绝 version mismatch。library API
不读取 stdin；调用方必须显式选择 allow policy。

### Checkpoint provenance 与版本差异

checkpoint 保存 entry-point name、distribution、version 和 target，但不保存或 freeze
extension class/source。恢复时必须在当前 CLI 环境安装相应 distribution：

- strict resume 与 checkpoint-only inference 使用 checkpoint selection；partial sample
  request 可以追加插件，但不能删除 checkpoint provenance 中的 required selection；
- 复用的同名插件仍必须通过 identity/version 检查，新增插件也进入本次 sampling manifest；
- name、distribution 或 target 变化是 identity mismatch，始终失败；
- 仅 version 不一致时，CLI 会在导入插件前集中警告并在交互式终端询问，默认答案为 No；
- 非交互式运行默认失败；`--force-extension-version-mismatch` 可明确接受版本差异，但不会
  绕过 identity 或 checkpoint state compatibility；
- 接受方式与 expected/current version 写入 run/sampling manifest 和后续 checkpoint
  metadata。

editable install 在版本号不变时修改源码无法由 distribution metadata 检测。checkpoint
不是源码或 Python 环境快照；依赖锁定仍由用户选择的包管理器和 lockfile 负责。扩展实现
变化导致的 state shape、资产拓扑或运行逻辑不兼容，会在既有构建/加载边界报错，核心不为
任意第三方实现提供源码兼容保证。

训练 manifest、checkpoint metadata 和 sampling manifest 还保存
`selected_components`，列出最终 typed config 选择的 DataBuilder、model、
TrainingBuilder、Objective、Process、optimizer、scheduler、checkpoint sampling recipe、
writers、
loggers 和 diagnostics。这个摘要保留显式 `null` 与列表顺序，但不会遍历任何 `params`：
具体 sampler、noise schedule、teacher、source 或 condition 仍由其 Builder/Process
私有拥有。它用于快速审计，不冻结 class/source，不参与 Registry dispatch、plugin
ownership 推断或 compatibility 判断；checkpoint 中的完整 config 与 runtime state
仍是恢复权威。

当前 checkpoint v11 使用 `torch.load(..., weights_only=True)`，payload 只允许
Tensor、primitive 和普通 container，且不读取旧格式。扩展组件的 `state_dict`、
extra state 与 `SamplingRecipe.contract` 也必须编码为受支持的数据类型；不能要求
`safe_globals`、保存自定义 class instance 或 custom Tensor subclass。这个约束让 runtime
能在导入扩展前安全读取 config/provenance，但不是针对恶意超大 Tensor 的完整资源沙箱。

### 生成一个扩展项目

```bash
stochaflow init my-project
cd my-project
```

项目名必须是长度不超过 64 的 canonical ASCII slug，例如 `my-project`；Python package
对应为 `my_project`。`init` 只写文件，不创建环境、安装依赖、运行 Git，也不覆盖文件、
symlink 或非空目录。生成结果是单一可安装 distribution、多实验 research repo：

若平台提供安全的 descriptor-relative 文件系统操作，`init` 也可写入已存在的空真实目录；
不具备这些原语的平台应先删除该空目录，由 `init` 创建目标目录。

```text
my-project/
├── pyproject.toml
├── experiments/example/train.yaml
├── src/my_project/stochaflow_ext/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   └── sampling.py
├── data/
├── notebooks/
└── tests/test_extensions.py
```

默认 synthetic regression 示例完整注册 DataBuilder、model、TrainingBuilder/Strategy 和
direct-transform SamplingBuilder，并复用内置 MSE objective 与 tensor writer。它不虚构
Process 或 Sampler。repo 可以自行增加其他实验、研究代码、脚本或工具；这些内容无需采用
Stochaflow 目录结构，也不会被 CLI 扫描或驱动。

生成的 `pyproject.toml` 精确依赖当前 Stochaflow 版本，默认配置显式写入
`extensions.plugins: [my-project]`。使用任意包管理器把它安装到 CLI 所在环境后即可运行：

```bash
python -m pip install -e ".[test]"
stochaflow train --config experiments/example/train.yaml
```

当前 CLI 没有单独的 plugin-list 子命令。需要确认实际 executable、Stochaflow
distribution 版本和当前环境发现的 entry-point names 时，可以直接查询标准 packaging
metadata：

```bash
python -c "import sys; from importlib.metadata import entry_points, version; print(sys.executable); print(version('stochaflow')); print(*(ep.name for ep in entry_points(group='stochaflow.extensions')), sep='\n')"
```

这只列出已安装 entry point，不导入扩展代码或列举其 Registry components。实际 component
registration 在 runtime 激活 YAML 选中的插件后发生。

第三方代码应从 `stochaflow.extensions` 导入稳定契约。`DataBuilder` 是核心运行时数据
组合入口，负责交付完整的 `DataLoaders`。对于复用内置 image recipe 的扩展，
`ImageDataSource`、`IMAGE_DATA_SOURCES`、`DataArtifact`、identity/binding 类型以及公开
image payload 是受限的 source-adapter 契约：source 只读取、处理并物化带 identity 的
artifact，不构造 Dataset、transform、sampler 或 DataLoader。标准
`ClassLabeledImageFolderArtifactPayload` 只是数据载体；内置 `class_labeled_image`
Builder 还要求没有 native validation，并拥有自己的 derived holdout、augmentation、
sampler、loader、resume 与 batch contract。只有 artifact 完整契约和这些 runtime recipe
语义都匹配时才使用 Source-only extension，否则注册自定义 DataBuilder。

自定义 DataSource 必须从 `stochaflow.extensions` 使用 `DataArtifactStore`。managed 与
referenced producer 共享一个 schema-v2 manifest/cache lifecycle，并都返回统一
`DataArtifact`；前者由 cache 拥有内容，后者只拥有索引并引用 cache 外部内容。
extension 只实现领域 build/load callback，不自行计算最终 identity，也不维护 locator、
locking、publication 或 quarantine。完整符号与 callback 契约见
[Extension 公共 API](../api/extensions.md)。

需要查看跨 data/training/sampling 三条扩展轴的完整实现时，参考两个彼此独立、可安装的
[纵向扩展项目](reference-projects.md)。它们分别展示 Physics reconstruction 对离散
Gaussian primitive 的复用与扩展，以及 frozen-teacher distillation 的多资产 checkpoint
组合；两者都只使用这里公开的稳定入口，不依赖 checkout 源码路径。

## 自定义 Process、Sampler 与 SamplingBuilder

生成扩展分成三层，层间共享的是生命周期，不是万能数学接口：

| 层 | 统一内容 | 不统一内容 |
| --- | --- | --- |
| 框架 | Registry、配置、checkpoint、完整 `Sampler.sample()` | 算法数学兼容性 |
| 算法 family | 自己需要的 Process、Dynamics 与兼容 Sampler | 跨 family 的 `predict/step/drift` API |
| 任务 | 模型适配、condition、guidance、initial state、artifact | 核心 runner 分支 |

`Process` 根类型保存可注册、可迁移和可 checkpoint 的状态，但它是可选算法资产；算法
family 只在确有 model-free probability path 时通过 Process 子类定义内聚数学能力。
`GenerativeDynamics`
同样只是无行为语义根，不拥有 Registry 或 YAML 身份。`Sampler` 只统一完整 `sample()`
生命周期，不强制提供单步 `step()`；`SamplingBuilder` 解释具体任务并装配 initial state、
模型 callable、condition/guidance、family Dynamics、Sampler 和 observer。

当前生产实现只有 Gaussian family：

```text
DiscreteGaussianDenoisingProcess
    + GaussianDenoisingDynamics
    + DDPMAncestralSampler / DDIMSampler
```

未来 flow matching 可以定义自己的 probability path、VectorField Dynamics 和 ODE
Sampler；score SDE 或 sigma-space 方法也可以定义自己的窄契约。它们不需要实现 Gaussian
接口，也不会促使核心为 `GenerativeDynamics` 增加 universal `predict`、`drift`、`score`
或 `denoise` 方法。

`GaussianDenoisingDynamics` 是 DDPM/DDIM 依赖的基础抽象能力；respaced learned-range
DDPM 另外使用 target-aware Gaussian Dynamics 的窄能力。内置
`GaussianModelDynamics` 是普通 model callable 的具体 adapter，可按 fixed 或
learned-range variance 解析 `C`/`2C` output。Process 不提供 Dynamics 工厂方法，也不依赖
模型 callable、prediction parameterization、variance、clipping policy 或具体 Dynamics
类型。Builder、diagnostic 或用户推理工作流负责显式构造/包装 Dynamics。

`noise_schedules` Registry 只服务 Gaussian Process，不是 Flow Matching interpolant 的
万能 Schedule Registry。离散 Gaussian Process 依赖 `DiscreteVPSchedule` 数学能力；内置
linear/cosine 实现继承 `TabulatedDiscreteVPSchedule`，第三方实现可以用其他存储或解析
方式提供 `GaussianScales` 与 `DiscreteVPCoefficients`，无需暴露 coefficient table。
Schedule 返回与 `state_times` 同形的系数，样本 rank 的 broadcast 由 Process 负责。
`DiscreteGaussianProcess` 在构造时复制完整 coefficient snapshot，运行时不保留 schedule
子模块；检测到 `Parameter`、需要梯度或非法系数会明确失败。marginal、posterior、device
迁移和 checkpoint 因而始终读取同一份 Process-owned state。

内置 `gaussian_denoising` TrainingBuilder、`standard_denoising` SamplingBuilder、DDPM
和 DDIM 的 fixed/adjacent 路径依赖公开的 `DiscreteGaussianDenoisingProcess`。第三方
实现该 Process 接口即可复用默认路径，无需继承 `DiscreteGaussianProcess`；训练
timestep sampling policy 由 Gaussian TrainingStrategy 拥有。需要 non-adjacent
respacing 时还必须满足 selected-pair coefficient capability；需要 learned-range
variance 时还必须提供 selected-pair lower/upper log-variance bounds。Builder/Sampler 在
完整组合边界分别验证这些窄能力，不把可选方法塞进 Process root。

```python
from stochaflow.extensions import (
    DiscreteGaussianDenoisingProcess,
    GaussianModelDynamics,
    REGISTRIES,
    Sampler,
    SamplerResult,
    SamplingBuilder,
    SamplingOutput,
)


@REGISTRIES.samplers.register("my-project.solver")
class ProjectSolver(Sampler):
    def __init__(self, *, tolerance: float):
        self.tolerance = tolerance

    def sample(self, dynamics, initial_state, *, generator=None, observer=None):
        final_state, steps, diagnostics = solve(
            dynamics,
            initial_state,
            tolerance=self.tolerance,
            generator=generator,
            observer=observer,
        )
        return SamplerResult(final_state, steps, diagnostics)


@REGISTRIES.sampling_builders.register("my-project.physics-sampling")
class PhysicsSamplingBuilder(SamplingBuilder):
    def run(self) -> SamplingOutput:
        process = self.context.process
        if not isinstance(process, DiscreteGaussianDenoisingProcess):
            raise TypeError(
                "my-project.physics-sampling requires discrete Gaussian process"
            )
        dynamics = GaussianModelDynamics(
            process=process,
            predict_fn=build_guided_predict_fn(...),
            prediction_type="epsilon",
            variance_mode="fixed",
            clip_denoised=True,
        )
        # Builder also owns observations, initial state and Sampler execution.
        ...
```

`Sampler.sample()` 不接收 shape、low-res、mask、guidance 或 solver-specific keyword。
shape/initial state 属于 Builder，condition/guidance 通过模型 callable 闭包进入 Dynamics，
solver 参数属于 Sampler 构造配置，trajectory 由 `SamplingObserver` 收集。
Sampler 负责决定 lifecycle 时机、创建完整的 `SamplingObservation` 并直接调用
`observer.observe(observation)`；Observer 只消费事件，决定筛选、复制和记录。
`TrajectoryObserver` 保留 initial、指定间隔和 final observation。Sampler 不提供伪通用
的起止坐标查询；`standard_denoising` 与 image diagnostic 通过真实 observation 验证完整
terminal-to-clean 路径。基于观测或其他中间状态的 partial denoising 应由自定义
SamplingBuilder 构造 initial state，或直接调用 Sampler。

完整 `sample()` 是 framework-level 统一生命周期，不意味着当前支持的算法 family 必须把
可组合数学全部藏进循环。离散 Gaussian family 另外公开：

- selected-pair marginal coefficient snapshot：为 `0 <= s < t <= T` 提供
  `alpha_bar_s`、`alpha_bar_t` 与二者比值；
- `DDPMAncestralSampler.resolve_schedule()`：解析 full adjacent、
  uniform-section respaced 或 explicit public-state schedule；
- `DDPMAncestralSampler.transition()`：从同一 selected-pair snapshot 构造
  `x_t -> x_s` ancestral posterior；learned prediction 可以提供该 pair 的 log variance；
- `DDIMSampler.resolve_schedule()`：解析 configured uniform、explicit 或 partial schedule；
- `DDIMSampler.transition()`：构造 batch-aligned selected-pair `x_t -> x_s` 分布；
- `GaussianTransition.mean/standard_deviation` 与 `sample(generator=...)`：显式抽取 next
  state，零方差时不消费 RNG；
- `normalize_gaussian_prediction()`：把 raw epsilon/x0/v/score 输出归一化为一致的
  `GaussianPrediction`。

这些 API 只属于离散 Gaussian family，不是 `Sampler.step()` 的别名，也不要求未来的 Flow
Matching、SDE 或 sigma-space sampler 实现。transition 不调用模型、不发送 observation、
不应用任务 correction；完整 DDPM/DDIM Sampler 委托这些 primitive 并继续拥有 RNG、loop
和 accepted-step lifecycle。DDPM respacing 不是 DDIM：前者构造 selected-pair ancestral
posterior，后者保留 generalized `eta` update。learned-range DDIM 只消费 prediction half，
明确忽略 variance half。

### 复用 Gaussian family 完成 condition 或 physics guidance

如果任务仍满足 Gaussian denoising 行为契约，condition 不进入 Sampler API。Builder 可以
把 low-resolution、mask、CFG 或 physics state 捕获在模型 callable 中，也可以包装
`GaussianDenoisingDynamics`，在每次 Sampler 查询生成方向时修正 prediction：

```python
from stochaflow.extensions import (
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
)


class PhysicsGuidedDynamics(GaussianDenoisingDynamics):
    def __init__(self, process, model, low_res, physics_state):
        self._process = process
        self._delegate = GaussianModelDynamics(
            process,
            lambda state, time: model(state, time, low_res=low_res),
            prediction_type="epsilon",
            variance_mode="fixed",
            clip_denoised=True,
        )
        self.physics_state = physics_state

    @property
    def process(self):
        return self._process

    def predict(self, state, state_times):
        prediction = self._delegate.predict(state, state_times)
        return apply_physics_guidance(
            prediction,
            state=state,
            state_times=state_times,
            process=self.process,
            physics_state=self.physics_state,
        )
```

`apply_physics_guidance()` 必须返回彼此一致的 clean/epsilon prediction。只要这个行为契约
成立，同一个 Dynamics 可以交给 DDPM 或 DDIM。若 DFSR correction 改变的是 reverse
transition、accepted-step 更新或 solver 内部子步，而不能表达为 Gaussian prediction
修正，则应在项目扩展中提供匹配的窄 Dynamics 和自定义 Sampler；不要给内置 Process、
DDPM/DDIM 或顶层 YAML 增加 physics 专用参数。

exact DFSR 的典型顺序是：guided Dynamics 产生 source-state `GaussianPrediction` 和 physics
correction；项目 Sampler 调用公开 `resolve_schedule()` 与 `transition()`，从标准 DDIM
transition 抽样后再减 correction，最后才发送 accepted observation。correction 的数学方向
由任务 Dynamics 计算，correction 在数值 update 中的位置由项目 Sampler 决定。这样能够
复用内置 DDIM 数学，又不会让 Dynamics 感知 target schedule，或让 Observer 反向修改 solver
state。correction 为零时，项目 Sampler 应在相同 schedule 和 generator 下与内置 DDIM
逐步一致。

### 接入新的算法 family

新的 family 仍复用框架 Registry 和完整 sampling lifecycle，但定义自己的数学契约：

```python
from stochaflow.extensions import (
    GenerativeDynamics,
    Process,
    REGISTRIES,
    Sampler,
    SamplingBuilder,
)


@REGISTRIES.processes.register("my-project.flow-path")
class ProjectFlowPath(Process):
    ...


class ProjectVectorField(GenerativeDynamics):
    def velocity(self, state, time):
        ...


@REGISTRIES.samplers.register("my-project.ode")
class ProjectODESampler(Sampler):
    def sample(self, dynamics, initial_state, *, generator=None, observer=None):
        if not isinstance(dynamics, ProjectVectorField):
            raise TypeError("my-project.ode requires ProjectVectorField")
        ...


@REGISTRIES.sampling_builders.register("my-project.flow-sampling")
class ProjectFlowSamplingBuilder(SamplingBuilder):
    def run(self):
        # Validate ProjectFlowPath, assemble ProjectVectorField, then invoke
        # the registered project ODE sampler and return SamplingOutput.
        ...
```

兼容性检查位于项目 Builder 和 family Sampler 的调用边界。核心不读取 family 名称，不维护
Process/Sampler 组合表，也不注册 Dynamics。

### 不使用 Process 或 Sampler 的生成方法

`SamplingBuilderContext.process` 是可选值。没有 model-free probability path 或 numerical
solver loop 的方法可以直接在 Builder 中调用模型并返回 `SamplingOutput`：

```python
from stochaflow.extensions import (
    REGISTRIES,
    SamplingBatch,
    SamplingBuilder,
    SamplingOutput,
)


@REGISTRIES.sampling_builders.register("my-project.direct-transform")
class DirectTransformBuilder(SamplingBuilder):
    def run(self):
        if self.context.process is not None:
            raise TypeError("my-project.direct-transform does not use a Process")
        model = self.context.model_provider.get("raw")
        samples = run_exact_transform(model, self.context.params)
        return SamplingOutput((SamplingBatch(samples.cpu()),), {"kind": "direct"})
```

对应配置省略 `process` 或写 `process: null`；resolved config 和 checkpoint 不保存
`process_state_dict`。这条路径不需要占位 Process、Dynamics 或 Sampler，也不需要核心增加
按算法名称分支。训练侧是否需要 Process 由所选 TrainingBuilder 在组合边界校验。

## 自定义 TrainingBuilder 与 TrainingStrategy

训练侧的注册入口是 `TrainingBuilder`，而不是 Strategy。Builder 接收核心预先构建的
primary model、可选 Process/Objective 和辅助 factory，返回完成依赖注入的
`TrainingPlan`。`TrainingStrategy` 只是普通训练计算对象：解释 structured batch、调用
注入的模型和 Objective，并返回包含 loss、低成本 step report、diagnostics 和可选
metric channel updates 的 `TrainStepOutput`。
Strategy 没有统一的构造参数 schema；每个 Builder 可以直接向它注入该任务需要的模型、
Objective、Process capability 或 callable。统一的是 step 的输入输出和职责边界，而不是
把所有训练任务压进一个包含大量可选字段的 Strategy context。

Trainer 原生递归迁移 `Tensor`、`Mapping` value、tuple/namedtuple 和 list。其他 leaf
保持原样；包含 Tensor 的领域 dataclass 或自定义容器应实现公开的
`DeviceTransferableBatch.to_device(device)` capability。这个 capability 只负责迁移一个
已构建 batch，不定义字段、collate 或 DataLoader，也不引入额外 registry。

离散 Gaussian 自定义 Strategy 可直接复用 `gaussian_training_target()` 构造 epsilon、x0、
v 或 score target，并用 `compute_objective()` 获得统一的 scalar/per-sample Objective 校验。
batch 解释、condition、timestep sampling、模型调用和 diagnostics 仍由具体 Strategy 拥有；
这两个 helper 不构成新的通用 TrainingStrategy。

```python
from stochaflow.extensions import (
    REGISTRIES,
    TrainStepOutput,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    compute_objective,
)


class PairedRegressionStrategy(TrainingStrategy):
    def __init__(self, model, objective):
        self.model = model
        self.objective = objective

    def training_step(self, batch):
        inputs, targets = batch
        prediction = self.model(inputs)
        loss, _ = compute_objective(self.objective, prediction, targets)
        return TrainStepOutput(loss=loss)


@REGISTRIES.training_builders.register("my-project.paired-regression")
class PairedRegressionBuilder(TrainingBuilder):
    def build(self):
        objective = self.context.objective
        if objective is None:
            raise TypeError("my-project.paired-regression requires objective")
        if self.context.process is not None:
            raise TypeError("my-project.paired-regression does not use a Process")
        return TrainingPlan(
            strategy=PairedRegressionStrategy(
                self.context.primary_model,
                objective,
            ),
            primary_model=self.context.primary_model,
            process=None,
            objective=objective,
        )
```

Strategy 不拥有 `to/train/eval`、optimizer、factory、parameter selection 或 checkpoint
API。Plan 通过稳定名称声明辅助 `nn.Module`，核心统一管理 device、mode、优化和持久化；
EMA 只跟踪 primary model，Process、Objective 和 auxiliary modules 始终保存 raw state。
这个片段是普通 paired regression，不是 diffusion super-resolution。Gaussian 条件超分
还必须由 Strategy 采样 marginal、构造 prediction target 并把 LR condition 传给模型；
完整组合见[条件 Gaussian 超分辨率教程](../tutorials/super-resolution.md)。

### 自定义 Metric 与 Strategy channel

metric 的注册入口是 `REGISTRIES.metrics`，注册类必须继承 `torchmetrics.Metric`。metric
只拥有统计 state 和 `update()`/`compute()`；Strategy 拥有 batch 与模型输出语义，并通过
一个有文档的 channel 把 payload 交给 metric。核心不按 `preds`、`target`、label 或 batch
类型猜参数。

下面的 Strategy 声明一个任务私有的 class prediction channel。`metrics` 仍是低成本
step report；进入跨 batch state 的值必须放在 `metric_updates`：

```python
from stochaflow.extensions import (
    MetricUpdate,
    TrainStepOutput,
    TrainingStrategy,
)


class ClassifierStrategy(TrainingStrategy):
    def __init__(self, model, objective):
        self.model = model
        self.objective = objective

    @property
    def metric_channels(self) -> frozenset[str]:
        return frozenset({"class-eval.prediction_target"})

    def training_step(self, batch):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.objective(logits, targets)
        return TrainStepOutput(
            loss=loss,
            metric_updates={
                "class-eval.prediction_target": MetricUpdate(
                    args=(logits, targets),
                )
            },
            loss_aggregation_weight=targets.numel(),
        )
```

插件的注册模块再给一个 `Metric` 子类稳定命名：

```python
import torch
from torchmetrics import Metric

from stochaflow.extensions import REGISTRIES


@REGISTRIES.metrics.register("class-eval.per-class-recall")
class PerClassRecallMetric(Metric):
    def __init__(self, *, num_classes: int):
        super().__init__(dist_sync_on_step=False, sync_on_compute=True)
        self.num_classes = num_classes
        self.add_state(
            "correct",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, logits, targets):
        predictions = logits.argmax(dim=1)
        self.total += torch.bincount(targets, minlength=self.num_classes)
        matched = targets[predictions == targets]
        self.correct += torch.bincount(
            matched,
            minlength=self.num_classes,
        )

    def compute(self):
        observed = self.total > 0
        values = self.correct.float() / self.total.clamp_min(1)
        return {
            **{
                f"class_{index}": values[index]
                for index in range(self.num_classes)
            },
            "macro": values[observed].mean(),
        }
```

这个例子把每个 state 的 reduction 明确声明为 `sum`，但这不代表 Stochaflow Trainer
支持 distributed execution；当前只承诺单进程训练。实际实现还应在构造、update 和
compute 边界验证 shape、dtype、class 范围，以及 validation 是否覆盖预期类别。完整、
带防御性校验的版本见[按类别验证与自定义 Metric](../tutorials/class-metrics.md)。

配置把 Registry name、channel 和 phase 绑定在一起：

```yaml
extensions:
  plugins: [class-eval]

metrics:
  - id: class_recall
    name: class-eval.per-class-recall
    channel: class-eval.prediction_target
    phases: [validation, test]
    params:
      num_classes: 4

trainer:
  early_stopping:
    enabled: true
    monitor: valid/metrics/class_recall/macro
    mode: max
    missing: error
    patience: 8
    min_delta: 0.001
```

每个 phase 获得独立 metric 实例。结果 key 是
`valid/metrics/class_recall/class_0` 到 `class_3` 以及
`valid/metrics/class_recall/macro`；test 使用 `test/...`，且不能用于 best checkpoint
或 early stopping。`mse`、`mae` 和 `mean` 是仅有的内置 metric 名称，其中 MSE/MAE
固定为单输出 scalar；不要把任意 TorchMetrics class path 写入 `name`。

插件 entry-point name、distribution、version 与 target 会进入 resolved config/run
manifest/checkpoint provenance；metric declaration 与 channel 也保存在完整 resolved
config 中。checkpoint 不保存 Python class/source 或 Metric runtime state，只保存完成
epoch 的 canonical scalar snapshot 与 source metadata。strict resume 因而要求同一插件
identity/version policy 和同一 metric/Strategy 配置，但仍需由项目 lockfile 固定
TorchMetrics 及其他依赖。

### Frozen-teacher 蒸馏

蒸馏不需要 Trainer 的任务分支。自定义 Builder 构建并加载 teacher、关闭其梯度，再用
稳定名称把它交给 Plan；Strategy 只定义 student/teacher forward 和损失组合：

```python
import torch


class DistillationStrategy(TrainingStrategy):
    def __init__(self, student, teacher, task_objective, distill_objective, alpha):
        self.student = student
        self.teacher = teacher
        self.task_objective = task_objective
        self.distill_objective = distill_objective
        self.alpha = alpha

    def training_step(self, batch):
        inputs, targets = batch
        with torch.no_grad():
            teacher_output = self.teacher(inputs)
        student_output = self.student(inputs)
        task_loss = self.task_objective(student_output, targets)
        distill_loss = self.distill_objective(student_output, teacher_output)
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        return TrainStepOutput(
            total_loss,
            metrics={
                "task_loss": task_loss.detach(),
                "distill_loss": distill_loss.detach(),
            },
        )


class DistillationBuilder(TrainingBuilder):
    def build(self):
        if self.context.objective is None:
            raise TypeError("distillation requires a task objective")
        teacher = build_and_load_teacher(self.context)
        teacher.requires_grad_(False)
        distill_objective = build_distillation_objective(self.context)
        strategy = DistillationStrategy(
            self.context.primary_model,
            teacher,
            self.context.objective,
            distill_objective,
            alpha=0.5,
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=self.context.objective,
            auxiliary_modules={
                "teacher": ManagedTrainingModule(teacher, mode="eval"),
                "distill_objective": ManagedTrainingModule(distill_objective),
            },
        )
```

核心会迁移并 checkpoint teacher，保持其 eval mode；`requires_grad=False` 使 teacher 不进入
optimizer。offline distillation 可直接由 DataBuilder 提供 teacher target，因此无需
auxiliary teacher。feature、logit 或 score distillation 也采用同一边界，只需在具体
Strategy 内解释输出并返回单一标量总 loss。多个 teacher 或多个蒸馏 Objective 也不要求
核心新增分支：Builder 用稳定名称声明所有资产，Strategy 在一次 step 内组合它们。

如果 student 与 teacher 共同训练且共享一个 optimizer，Plan 可以将两者都声明为可训练
资产，核心会统一收集 `requires_grad=True` 的参数。Strategy 不选择参数，只负责返回能够
对这些参数反传的总 loss。

当前自动训练生命周期只支持一个 optimizer 和一次 backward。独立 teacher optimizer、
交替更新、EMA teacher 或 manual backward 属于新的训练 loop family，不通过向 Strategy
增加生命周期开关来模拟。

### 复用 Gaussian diagnostic

`diffusion_quality` 不猜测 primary model 的调用签名。希望复用它的 Gaussian Strategy
必须满足稳定的 `GaussianDiagnosticSemantics` 窄能力：

```python
from stochaflow.extensions import GaussianDiagnosticSemantics


class ConditionalGaussianStrategy(TrainingStrategy, GaussianDiagnosticSemantics):
    @property
    def prediction_type(self):
        return "epsilon"

    @property
    def variance_mode(self):
        return "fixed"

    def predict_gaussian_model(self, state, model_time):
        return self.model(
            {"state": state, "time": model_time},
            condition=self.diagnostic_condition,
        )
```

这是结构化 Protocol，不要求 Strategy 显式继承它；示例继承用于类型标注和能力发现。
`prediction_type` 说明 Gaussian parameterization，`variance_mode` 声明 callable 返回
`C` fixed output 还是 `2C` learned-range output，`predict_gaussian_model()` 封装任务自己的
模型签名。diagnostic 用这些语义与 Process 构造 Dynamics，不直接调用 primary model；
learned-range diagnostics 只用 prediction half 做 reconstruction/DDIM，仍按 recipe 验证
完整 raw output layout。

如果条件任务无法为 diagnostic 提供明确的 condition，就不应伪造该 capability；启用
`diffusion_quality` 时会得到清晰的不兼容错误。该能力不让 Strategy 构建 Sampler、运行
diagnostic 或管理 artifact，因此不改变 Strategy 的训练计算职责。

## 自定义 DataBuilder

```python
from torch.utils.data import DataLoader

from stochaflow.extensions import DataBuilder, DataLoaders, REGISTRIES


@REGISTRIES.data_builders.register("my-project.physics-data")
class PhysicsDataBuilder(DataBuilder):
    def build(self) -> DataLoaders:
        params = self.context.params
        train_dataset, validation_dataset = build_physics_datasets(
            params["path"],
            seed=self.context.seed,
        )
        return DataLoaders(
            train=DataLoader(
                train_dataset,
                sampler=PhysicsSampler(train_dataset),
                collate_fn=physics_collate,
            ),
            validation=DataLoader(validation_dataset),
            steps_per_epoch=params.get("steps_per_epoch"),
        )
```

```yaml
extensions:
  plugins: [my-project]
data:
  name: my-project.physics-data
  params:
    path: data/simulation.zarr
    steps_per_epoch: 1000
```

`DataBuilderContext.params` 是调用方配置的深拷贝，`seed` 是实验种子。`build()` 必须
返回一个 `DataLoaders`；所有 loader 必须可重复迭代，不能直接返回一次性的 generator。
train loader 通过自身 `len()` 或显式 `steps_per_epoch` 给出有限 epoch。
validation/test 可以是未知长度的可重复 iterable。

每个 epoch 开始时，Trainer 会对 train loader 的 `sampler` 和 `batch_sampler` 做去重后
的 duck-typed `set_epoch(epoch)` 调用。其他 Dataset 或 DataLoader 属性只用于
best-effort 报告，不属于扩展契约。

## 自定义 sampling artifact writer

```python
from stochaflow.extensions import REGISTRIES, SamplingArtifactWriter


@REGISTRIES.sampling_artifact_writers.register("my-project.netcdf")
class NetCDFWriter(SamplingArtifactWriter):
    def __init__(self, *, variable: str):
        self.variable = variable

    def write(self, context):
        path = context.output_dir / "samples.nc"
        write_netcdf(path, context.batches, variable=self.variable)
        return {"netcdf": path}
```

```yaml
sampling:
  shape: [64, 64, 4]
  writers:
    - name: tensor
      params: {}
    - name: my-project.netcdf
      params: {variable: velocity}
```

writer 返回的 mapping 不能为空；artifact key 在全部 writer 间必须唯一，路径必须已经
存在。任何 writer 失败都会使采样失败。内置 `tensor` 支持任意 rank Tensor；内置
`image` 才校验 NCHW、1/3 通道，并拥有 `grid_nrow`、`gif_fps` 与 `denormalize` 参数。

## 其他构造约定

组件 `params` 通常作为关键字参数传给注册类或受限原生 provider。以下参数由运行时注入：

- DataBuilder：`DataBuilderContext`；
- TrainingBuilder：`TrainingBuilderContext`；
- SamplingBuilder：`SamplingBuilderContext`；
- optimizer / LR scheduler：模型参数或 optimizer；
- logger：`output_dir`、`run_name`；
- diagnostic：`logger`、`output_dir`。需要采样的 diagnostic 在自己的私有配置中声明
  shape，framework 不从独立 sample workflow 或 DataBuilder 猜测。

optimizer 的配置 `params` 不得包含其运行时 `params` iterable；LR scheduler 的配置
`params` 不得包含 `optimizer`。`T_max`、`total_steps` 等具体构造参数必须显式写成确定值，
不会从 epoch、DataLoader 长度或 CLI batch limit 推断，也不接受通用 `auto` sentinel。

`diffusion_quality` 的 provider 仍使用其局部 `diagnostics[].params.modules` 机制。
所有内置名称与构造参数见[生成式组件索引](reference.md#registry-组件索引)。
